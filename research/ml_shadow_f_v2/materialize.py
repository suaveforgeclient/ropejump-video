from __future__ import annotations
import argparse, base64, hashlib, json, os, re, shutil, subprocess, sys, urllib.parse, urllib.request, zipfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from feature_builder import FEATURE_SCHEMA, build_features, build_targets

DE1="JR-20260915-DE1CE4AF"
CADENCES=(10,15,30)
PASS_CONTENT={"PASS_CONTINUOUS","PASS_CROPPED"}
TOKEN_RE=re.compile(r'(?:feedback-media/|"videoToken"\\s*:\\s*")([0-9a-f]{48})',re.I)

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def download(url:str,dst:Path)->None:
    dst.parent.mkdir(parents=True,exist_ok=True)
    req=urllib.request.Request(url,headers={"User-Agent":"ropejump-f-v2"})
    with urllib.request.urlopen(req,timeout=300) as r, dst.open("wb") as w:
        shutil.copyfileobj(r,w,1024*1024)

def run(cmd:list[str])->None:
    print("RUN", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd,check=True)

def trim_video(src:Path,dst:Path,start_ms:float,end_ms:float)->None:
    if end_ms<=start_ms: raise ValueError("invalid_segment")
    dst.parent.mkdir(parents=True,exist_ok=True)
    run([
        "ffmpeg","-hide_banner","-loglevel","error","-y",
        "-ss",f"{start_ms/1000:.6f}","-to",f"{end_ms/1000:.6f}",
        "-i",str(src),"-map","0:v:0","-an",
        "-vf","setpts=PTS-STARTPTS","-c:v","libx264","-preset","ultrafast","-crf","18",
        "-pix_fmt","yuv420p",str(dst)
    ])

def yt_download(url:str,dst:Path)->None:
    dst.parent.mkdir(parents=True,exist_ok=True)
    run([
        "yt-dlp","--no-playlist","--quiet","--no-warnings",
        "-f","bv*[height<=720]+ba/b[height<=720]/best",
        "--merge-output-format","mp4","-o",str(dst),url
    ])
    if not dst.is_file():
        candidates=list(dst.parent.glob(dst.stem+".*"))
        if not candidates: raise FileNotFoundError(dst)
        candidates[0].replace(dst)

def gh_headers()->dict:
    token=os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token: raise RuntimeError("github_token_missing")
    return {"Authorization":f"Bearer {token}","Accept":"application/vnd.github+json","X-GitHub-Api-Version":"2022-11-28"}

def github_json(url:str):
    req=urllib.request.Request(url,headers={"User-Agent":"ropejump-f-v2",**gh_headers()})
    with urllib.request.urlopen(req,timeout=60) as r: return json.load(r)

def github_file_bytes(repo:str,path:str,ref:str)->bytes:
    qpath=urllib.parse.quote(path,safe="/"); qref=urllib.parse.quote(ref,safe="")
    obj=github_json(f"https://api.github.com/repos/{repo}/contents/{qpath}?ref={qref}")
    if obj.get("encoding")=="base64" and obj.get("content"):
        return base64.b64decode(str(obj["content"]).replace("\n",""))
    if obj.get("download_url"):
        req=urllib.request.Request(obj["download_url"],headers={"User-Agent":"ropejump-f-v2",**gh_headers()})
        with urllib.request.urlopen(req,timeout=300) as r: return r.read()
    raise RuntimeError(f"github_file_unreadable:{ref}:{path}")

def evidence_index(video_ids:set[str])->dict[str,str]:
    repo=os.environ.get("GITHUB_REPOSITORY","suaveforge/jumprope")
    p=subprocess.run(["git","ls-remote","--heads","origin","refs/heads/research/train-evidence-*"],check=True,text=True,capture_output=True)
    refs=[]
    for line in p.stdout.splitlines():
        if "refs/heads/" in line: refs.append(line.split("refs/heads/",1)[1].strip())
    found={}
    for ref in refs:
        qref=urllib.parse.quote(ref,safe="")
        try: items=github_json(f"https://api.github.com/repos/{repo}/contents/review-evidence?ref={qref}")
        except Exception: continue
        if not isinstance(items,list): continue
        names={str(x.get("name")) for x in items if x.get("type")=="dir"}
        for vid in sorted(video_ids & names): found.setdefault(vid,ref)
        if len(found)==len(video_ids): break
    missing=sorted(video_ids-set(found))
    if missing: raise RuntimeError(f"evidence_proxy_branch_missing:{missing}")
    return found

def materialize_evidence_proxy(vid:str,x:dict,branch:str,dst:Path)->None:
    source_expected=str(x.get("sha256") or "").lower(); proxy_expected=str(x.get("review_proxy_sha256") or "").lower()
    local_root=str(os.environ.get("F_V2_PROXY_DIR") or "").strip()
    if local_root:
        src=Path(local_root)/f"{vid}.mp4"
        if not src.is_file(): raise RuntimeError(f"local_proxy_missing:{vid}")
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dst)
        actual=sha256(dst)
        if actual!=proxy_expected: raise RuntimeError(f"local_proxy_sha_mismatch:{vid}:{actual}:{proxy_expected}")
        return
    repo=os.environ.get("GITHUB_REPOSITORY","suaveforge/jumprope")
    manifest=json.loads(github_file_bytes(repo,f"review-evidence/{vid}/manifest.json",branch).decode("utf-8"))
    if manifest.get("status")!="READY": raise RuntimeError(f"proxy_not_ready:{vid}:{branch}")
    if str(manifest.get("source_sha256") or "").lower()!=source_expected: raise RuntimeError(f"proxy_source_sha_mismatch:{vid}")
    if str(manifest.get("review_proxy_sha256") or "").lower()!=proxy_expected: raise RuntimeError(f"proxy_manifest_sha_mismatch:{vid}")
    data=github_file_bytes(repo,f"review-evidence/{vid}/review.mp4",branch)
    dst.parent.mkdir(parents=True,exist_ok=True); dst.write_bytes(data)
    actual=sha256(dst)
    if actual!=proxy_expected: raise RuntimeError(f"proxy_file_sha_mismatch:{vid}:{actual}:{proxy_expected}")

def _append_tokens(text:str,found:list[str],seen:set[str])->None:
    for m in TOKEN_RE.finditer(text or ""):
        v=m.group(1).lower()
        if v not in seen: seen.add(v); found.append(v)

def issue_candidates(repo:str,public_id:str)->list[str]:
    found=[]; seen=set(); q=urllib.parse.quote(f"repo:{repo} {public_id}")
    data=github_json(f"https://api.github.com/search/issues?q={q}&per_page=100")
    for item in data.get("items",[]):
        _append_tokens(f"{item.get('title') or ''}\n{item.get('body') or ''}",found,seen)
        num=item.get("number")
        if num:
            try: comments=github_json(f"https://api.github.com/repos/{repo}/issues/{int(num)}/comments?per_page=100")
            except Exception: comments=[]
            if isinstance(comments,list):
                for comment in comments: _append_tokens(str(comment.get("body") or ""),found,seen)
    return found

def materialize_user_original(vid:str,expected:str,dst:Path)->None:
    canonical_root=str(os.environ.get("F_V2_CANONICAL_DIR") or "").strip()
    if canonical_root:
        root=Path(canonical_root)
        matches=sorted(p for p in root.glob(f"{vid}__{expected}.*") if p.is_file())
        if len(matches)!=1:
            raise RuntimeError(f"user_original_canonical_match_error:{vid}:{len(matches)}")
        actual=sha256(matches[0])
        if actual!=expected:
            raise RuntimeError(f"user_original_canonical_sha_mismatch:{vid}:{actual}:{expected}")
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(matches[0],dst)
        copied=sha256(dst)
        if copied!=expected:
            raise RuntimeError(f"user_original_canonical_copy_sha_mismatch:{vid}:{copied}:{expected}")
        return
    repo=os.environ.get("GITHUB_REPOSITORY","suaveforge/jumprope")
    candidates=issue_candidates(repo,vid)
    if not candidates: raise RuntimeError(f"user_original_attachment_token_missing:{vid}")
    for i,token in enumerate(candidates):
        tmp=dst.parent/f".{vid}.{i}.download"
        try: download(f"https://api-jumprope.suaveforge.com/v1/feedback-media/{token}",tmp)
        except Exception:
            tmp.unlink(missing_ok=True); continue
        if sha256(tmp)==expected:
            dst.parent.mkdir(parents=True,exist_ok=True); tmp.replace(dst); return
        tmp.unlink(missing_ok=True)
    raise RuntimeError(f"user_original_sha_match_missing:{vid}:{len(candidates)}")

def ucf_extract(source_ref:str,cache:Path,dst:Path)->None:
    base,frag=urllib.parse.urldefrag(source_ref)
    q=urllib.parse.parse_qs(frag)
    entry=(q.get("entry") or [None])[0]
    if not entry: raise ValueError(f"ucf_entry_missing:{source_ref}")
    archive=cache/"UCF101Fullvideo.zip"
    if not archive.is_file():
        print("Downloading UCF archive once",flush=True)
        download(base,archive)
    with zipfile.ZipFile(archive) as z:
        if entry not in z.namelist(): raise KeyError(f"ucf_entry_not_found:{entry}")
        dst.parent.mkdir(parents=True,exist_ok=True)
        with z.open(entry) as r, dst.open("wb") as w: shutil.copyfileobj(r,w,1024*1024)

def positive_inventory(ground,content,ucf,user):
    out=[]
    for vid,x in ground.items():
        if vid==DE1: continue
        if x.get("jump_rope_type")!="STANDARD_BASIC_SINGLE_UNDER": continue
        if x.get("dataset_eligibility")!="ELIGIBLE_VERIFIED_STANDARD": continue
        role=str(x.get("dataset_role") or "")
        if role not in {"TRAIN","VALIDATION","HOLDOUT"}: continue
        group=str(x.get("dataset_group") or "")
        if not group: raise ValueError(f"group_missing:{vid}")
        if vid.startswith("v_JumpRope_"):
            ann=ucf.get(vid)
            if not ann or ann.get("segment_usage")!="POSITIVE_STANDARD": raise ValueError(f"ucf_annotation_missing:{vid}")
            segments=ann.get("locked_segments") or []
            source_kind="UCF"
        elif str(x.get("source_ref") or "").startswith("USER_ORIGINAL:"):
            ann=user.get(vid)
            if not ann or ann.get("segment_usage")!="POSITIVE_STANDARD": raise ValueError(f"user_positive_annotation_missing:{vid}")
            segments=[s for s in (ann.get("locked_segments") or []) if s.get("segment_usage")=="POSITIVE_STANDARD"]
            source_kind="USER_ORIGINAL"
        else:
            cq=content.get(vid) or {}
            if cq.get("status") not in PASS_CONTENT: raise ValueError(f"external_content_not_pass:{vid}:{cq.get('status')}")
            seg=x.get("segmentation") or {}
            events=[float(v) for v in (x.get("jump_event_ms") or [])]
            if seg.get("required") is False:
                start=x.get("activity_start_ms"); end=x.get("activity_end_ms")
            else:
                start=seg.get("start_ms"); end=seg.get("end_ms")
            if start is None or end is None: raise ValueError(f"external_segment_missing:{vid}")
            segments=[{"segment_id":"seg_001","startMediaTimeMs":float(start),"endMediaTimeMs":float(end),"body_events":[{"visualApexMediaTimeMs":v} for v in events]}]
            source_kind="YOUTUBE_REVIEW_PROXY"
        if not segments: raise ValueError(f"segments_missing:{vid}")
        out.append((vid,x,segments,source_kind))
    return out

def hard_negative_inventory(user,ground):
    out=[]
    for vid,ann in user.items():
        if vid==DE1 or ann.get("segment_usage")!="HARD_NEGATIVE": continue
        segments=[s for s in (ann.get("locked_segments") or []) if s.get("segment_usage")=="HARD_NEGATIVE"]
        if not segments: continue
        x=dict(ground.get(vid) or {})
        x["dataset_role"]="TRAIN"; x["dataset_group"]=f"HARD_NEGATIVE::{vid}"; x["_sample_kind"]="HARD_NEGATIVE"
        out.append((vid,x,segments,"USER_ORIGINAL_HARD_NEGATIVE"))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-root",type=Path,required=True)
    ap.add_argument("--pose-model",type=Path,required=True)
    a=ap.parse_args()
    out=a.output_root.resolve(); cache=out/"cache"; raw=out/"raw"; clips=out/"clips"; pose_root=out/"pose"; data=out/"dataset"
    for d in (cache,raw,clips,pose_root,data): d.mkdir(parents=True,exist_ok=True)

    ground=json.loads((ROOT/"research/viewer/ground-counts.json").read_text())["items"]
    content=json.loads((ROOT/"research/viewer/content-quality.json").read_text()).get("items",{})
    ucf=json.loads((ROOT/"research/viewer/ucf-segment-annotations.json").read_text()).get("items",{})
    user=json.loads((ROOT/"research/viewer/user-original-segment-annotations.json").read_text()).get("items",{})
    registry=json.loads((ROOT/"tests/canonical-video-registry.json").read_text()).get("items",{})

    inventory=positive_inventory(ground,content,ucf,user)+hard_negative_inventory(user,ground)
    external_ids={vid for vid,x,_,kind in inventory if kind=="YOUTUBE_REVIEW_PROXY"}
    proxy_refs=evidence_index(external_ids) if external_ids and not str(os.environ.get("F_V2_PROXY_DIR") or "").strip() else {}
    # Positive USER_ORIGINAL clips must never enter TRAIN. USER_ORIGINAL hard negatives are allowed only as explicit zero-target negatives.
    if any(vid.startswith("JR-") and x.get("dataset_role")=="TRAIN" and x.get("_sample_kind")!="HARD_NEGATIVE" for vid,x,_,_ in inventory):
        raise RuntimeError("user_original_positive_train_forbidden_v2")
    groups={}
    for vid,x,_,_ in inventory:
        groups.setdefault(str(x["dataset_group"]),set()).add(str(x["dataset_role"]))
    leaks={g:sorted(v) for g,v in groups.items() if len(v)>1}
    if leaks: raise RuntimeError(f"group_leakage:{leaks}")

    rows=[]; report={"policy":"ROPEJUMP_F_V2_CLEAN","clips":[],"failures":[]}
    for vid,x,segments,source_kind in inventory:
        src=raw/f"{vid}.source"
        try:
            if source_kind=="UCF":
                ucf_extract(str(x["source_ref"]),cache,src)
                expected=str(x.get("sha256") or "").lower()
                actual=sha256(src)
                if expected and actual!=expected: raise RuntimeError(f"ucf_sha_mismatch:{vid}:{actual}:{expected}")
            elif source_kind=="YOUTUBE_REVIEW_PROXY":
                materialize_evidence_proxy(vid,x,proxy_refs.get(vid,"LOCAL_PROXY"),src)
            else:
                expected=str((registry.get(vid) or {}).get("sha256") or x.get("sha256") or "").lower()
                if not re.fullmatch(r"[0-9a-f]{64}",expected): raise RuntimeError(f"user_original_sha_invalid:{vid}")
                materialize_user_original(vid,expected,src)
            role=str(x["dataset_role"])
            split={"TRAIN":"train","VALIDATION":"val","HOLDOUT":"test"}[role]
            for si,seg in enumerate(segments,1):
                start=float(seg["startMediaTimeMs"]); end=float(seg["endMediaTimeMs"])
                sample_kind=str(x.get("_sample_kind") or "POSITIVE")
                if sample_kind=="HARD_NEGATIVE":
                    events=[]
                else:
                    events_abs=[float(e["visualApexMediaTimeMs"]) for e in (seg.get("body_events") or [])]
                    events=[v-start for v in events_abs if start<=v<=end]
                    if not events: raise RuntimeError(f"positive_segment_no_events:{vid}:{si}")
                clip=clips/f"{vid}__s{si:02d}.mp4"
                trim_video(src,clip,start,end)
                fps_values=(10,) if source_kind=="YOUTUBE_REVIEW_PROXY" else CADENCES
                for fps in fps_values:
                    pose_path=pose_root/f"{vid}__s{si:02d}__{fps}fps.json"
                    run([
                        sys.executable,str(ROOT/"scripts/extract-pose-landmarks.py"),
                        "--video",str(clip),"--model",str(a.pose_model),"--out",str(pose_path),
                        "--analysis-fps",str(fps),"--analysis-phase","0"
                    ])
                    pose=json.loads(pose_path.read_text())
                    features,times=build_features(pose)
                    targets=build_targets(times,events,220.0,.55)
                    sample=data/f"{vid}__s{si:02d}__{fps}fps"; sample.mkdir(parents=True,exist_ok=True)
                    np.save(sample/"features.npy",features); np.save(sample/"targets.npy",targets); np.save(sample/"times_ms.npy",times)
                    row={
                        "policy_id":"ROPEJUMP_F_V2_CLEAN","id":f"{vid}@s{si:02d}@{fps}fps","clip_id":vid,
                        "segment_id":f"s{si:02d}","split":split,"dataset_role":role,"dataset_group":x["dataset_group"],
                        "source_kind":source_kind,"sample_kind":sample_kind,"gt_locked":True,"derived_from_counter":False,
                        "jump_rope_type":x.get("jump_rope_type") if sample_kind=="HARD_NEGATIVE" else "STANDARD_BASIC_SINGLE_UNDER",
                        "event_convention":"no_positive_event" if sample_kind=="HARD_NEGATIVE" else "visual_apex",
                        "expected_count":len(events),"event_times_ms":events,"tolerance_ms":220.0,
                        "analysis_fps":fps,"feature_schema":FEATURE_SCHEMA,
                        "features_npy":str((sample/"features.npy").resolve()),"targets_npy":str((sample/"targets.npy").resolve()),
                        "times_npy":str((sample/"times_ms.npy").resolve()),"materialized_video_sha256":sha256(clip),
                        "negative_reason_codes":seg.get("negative_reason_codes",[]) if sample_kind=="HARD_NEGATIVE" else []
                    }
                    rows.append(row)
            report["clips"].append({"id":vid,"role":role,"group":x["dataset_group"],"sourceKind":source_kind,"sampleKind":str(x.get("_sample_kind") or "POSITIVE"),"segments":len(segments),"evidenceRef":proxy_refs.get(vid)})
        except Exception as e:
            report["failures"].append({"id":vid,"sourceKind":source_kind,"error":repr(e)})
            raise

    train_pos={r["clip_id"] for r in rows if r["split"]=="train" and r["sample_kind"]=="POSITIVE"}
    train_neg={r["clip_id"] for r in rows if r["split"]=="train" and r["sample_kind"]=="HARD_NEGATIVE"}
    val={r["clip_id"] for r in rows if r["split"]=="val"}
    test={r["clip_id"] for r in rows if r["split"]=="test"}
    if not train_pos or not train_neg or not val or len(test)<5: raise RuntimeError(f"split_incomplete:{len(train_pos)}:{len(train_neg)}:{len(val)}:{len(test)}")
    report["summary"]={
        "rows":len(rows),"trainPositiveClips":len(train_pos),"trainHardNegativeClips":len(train_neg),
        "externalProxyClips":len({r["clip_id"] for r in rows if r["source_kind"]=="YOUTUBE_REVIEW_PROXY"}),
        "valClips":len(val),"testClips":len(test),
        "trainEvents":sum(r["expected_count"] for r in rows if r["split"]=="train" and r["sample_kind"]=="POSITIVE"),
        "de1Present":any(r["clip_id"]==DE1 for r in rows)
    }
    if report["summary"]["de1Present"]: raise RuntimeError("DE1_contamination")
    (out/"manifest.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
    (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("FV2_MATERIALIZE="+json.dumps(report["summary"],sort_keys=True))

if __name__=="__main__": main()
