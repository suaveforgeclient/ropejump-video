#!/usr/bin/env python3
import argparse, csv, hashlib, json, re, shutil, ssl, urllib.parse, urllib.request
from pathlib import Path

OFFICIAL_ROOT="https://www.crcv.ucf.edu/THUMOS14/UCF101/UCF101/"
HF_REPO="bitmind/UCF101-Videos"
NAME_RE=re.compile(r"(v_JumpRope_g(\d{2})_c(\d{2})\.avi)")
UA={"User-Agent":"ROPEJUMP-research/3.1 ucf101-jumprope-collector"}

def fetch_bytes(url, timeout=45):
    req=urllib.request.Request(url,headers=UA)
    ctx=ssl._create_unverified_context()
    with urllib.request.urlopen(req,timeout=timeout,context=ctx) as r:
        return r.read()

def list_official():
    raw=fetch_bytes(OFFICIAL_ROOT,30).decode("utf-8","replace")
    found={}
    for name,g,c in NAME_RE.findall(raw):
        found[name]={"filename":name,"group":int(g),"clip":int(c),"sourceKind":"UCF_OFFICIAL_INDIVIDUAL"}
    return sorted(found.values(),key=lambda x:(x["group"],x["clip"]))

def list_hf_mirror():
    # This mirror exposes split CSVs cheaply. Read those instead of listing the
    # entire 13k-file repo. The uploaded AVI objects use literal backslashes in
    # repo filenames, so derive that exact path from clip_path.
    from huggingface_hub import hf_hub_download
    found={}
    for split in ("train","validation","test"):
        try:
            csv_path=hf_hub_download(
                repo_id=HF_REPO,
                filename=f"{split}.csv",
                repo_type="dataset"
            )
        except Exception:
            continue
        with open(csv_path,newline="",encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                if str(row.get("label") or "").strip()!="JumpRope":
                    continue
                clip_name=str(row.get("clip_name") or "").strip()
                if not clip_name:
                    continue
                name=clip_name if clip_name.endswith(".avi") else clip_name+".avi"
                m=NAME_RE.fullmatch(name)
                if not m:
                    continue
                _,g,cc=m.groups()
                clip_path=str(row.get("clip_path") or "").strip().lstrip("/")
                # HF mirror stores these as root-level names containing literal
                # backslashes, matching the repository tree display.
                repo_path=clip_path.replace("/","\\")
                found[name]={
                    "filename":name,
                    "group":int(g),
                    "clip":int(cc),
                    "split":split,
                    "repoPath":repo_path,
                    "sourceKind":"HF_UCF101_MIRROR"
                }
    return sorted(found.values(),key=lambda x:(x["group"],x["clip"]))

def select_diverse(rows, limit):
    # UCF group IDs are source groups. Diversify across groups before taking
    # another clip from the same group to reduce near-duplicate pilot bias.
    by_group={}
    for row in rows:
        by_group.setdefault(row["group"],[]).append(row)
    selected=[]
    round_i=0
    while len(selected)<limit:
        added=False
        for g in sorted(by_group):
            clips=by_group[g]
            if round_i < len(clips):
                selected.append(clips[round_i])
                added=True
                if len(selected)>=limit:
                    break
        if not added:
            break
        round_i+=1
    return selected

def download_hf(repo_path):
    # Normalize only path separators. Do not invent a different dataset object.
    # CSV/UI paths are canonical slash paths such as train/JumpRope/v_...avi.
    from huggingface_hub import hf_hub_download
    clean=str(repo_path or "").strip().replace("\\\\","/").lstrip("/")
    if not clean:
        raise RuntimeError("empty_hf_repo_path")
    errors=[]
    candidates=[clean]
    # Some mirrors expose split CSV paths with a redundant leading folder token.
    # Keep fallbacks deterministic and derived only from the published path.
    parts=[p for p in clean.split("/") if p]
    if len(parts)>=3:
        canonical="/".join(parts[-3:])
        if canonical not in candidates:
            candidates.append(canonical)
    for candidate in candidates:
        try:
            return Path(hf_hub_download(
                repo_id=HF_REPO,
                filename=candidate,
                repo_type="dataset"
            ))
        except Exception as e:
            errors.append(f"hf_hub_download[{candidate}]="+repr(e))
        try:
            encoded="/".join(urllib.parse.quote(part,safe="") for part in candidate.split("/"))
            url=f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{encoded}?download=true"
            data=fetch_bytes(url,90)
            if len(data) < 1024:
                raise RuntimeError(f"download_too_small:{len(data)}")
            cache=Path("/tmp/ucf101-hf")
            cache.mkdir(parents=True,exist_ok=True)
            out=cache/Path(candidate).name
            out.write_bytes(data)
            return out
        except Exception as e:
            errors.append(f"raw_resolve[{candidate}]="+repr(e))
    raise RuntimeError("; ".join(errors))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="out")
    ap.add_argument("--limit",type=int,default=20)
    args=ap.parse_args()

    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    listing_source=""
    listing_error=""
    try:
        rows=list_official()
        listing_source="UCF_OFFICIAL"
    except Exception as e:
        listing_error=repr(e)
        rows=list_hf_mirror()
        listing_source="HF_UCF101_MIRROR"

    selected=select_diverse(rows,args.limit)
    manifest=[]

    for row in selected:
        item={
            **row,
            "dataset":"UCF101",
            "datasetClass":"JumpRope",
            "officialRoot":OFFICIAL_ROOT,
            "mirrorRepo":HF_REPO if row.get("sourceKind")=="HF_UCF101_MIRROR" else "",
            "targetType":"STANDARD_BASIC_SINGLE_UNDER",
            "datasetLabelMeaning":"JumpRope only; subtype is NOT asserted",
            "typeStatus":"NEED_VISUAL_REVIEW",
            "contentStatus":"NEED_VISUAL_REVIEW",
            "countStatus":"NOT_REVIEWED",
            "autoPass":False
        }
        try:
            if row.get("sourceKind")=="HF_UCF101_MIRROR":
                src=download_hf(row["repoPath"])
                dest=out/row["filename"]
                shutil.copyfile(src,dest)
                data=dest.read_bytes()
                item["sourceRef"]=f"hf://datasets/{HF_REPO}/{row['repoPath']}"
            else:
                url=urllib.parse.urljoin(OFFICIAL_ROOT,row["filename"])
                data=fetch_bytes(url,60)
                dest=out/row["filename"]
                dest.write_bytes(data)
                item["sourceRef"]=url
            item["downloadOk"]=True
            item["bytes"]=len(data)
            item["sha256"]=hashlib.sha256(data).hexdigest()
        except Exception as e:
            item["downloadOk"]=False
            item["error"]=repr(e)
        manifest.append(item)

    payload={
        "schemaVersion":2,
        "collector":"UCF101_JUMPROPE_SELECTIVE",
        "officialRoot":OFFICIAL_ROOT,
        "fallbackMirror":HF_REPO,
        "listingSource":listing_source,
        "officialListingError":listing_error,
        "datasetClass":"JumpRope",
        "policy":{
            "fullArchiveDownloadAvoided":True,
            "selection":"group-diverse pilot; one clip per UCF source group before repeats",
            "datasetJumpRopeLabelDoesNotImplyStandard":True,
            "subtypeAutoClassificationForbidden":True,
            "wholeSourceVisualReviewRequired":True,
            "mixedTypeMustNotEnterPureStandardPool":True,
            "countingDeferredUntilPureStandardPass":True
        },
        "availableJumpRopeFiles":len(rows),
        "availableGroups":len(set(x["group"] for x in rows)),
        "selectedCount":len(selected),
        "downloadedCount":sum(1 for x in manifest if x.get("downloadOk")),
        "rows":manifest
    }
    (out/"manifest.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "listingSource":listing_source,
        "officialListingError":listing_error,
        "availableJumpRopeFiles":payload["availableJumpRopeFiles"],
        "availableGroups":payload["availableGroups"],
        "selectedCount":payload["selectedCount"],
        "downloadedCount":payload["downloadedCount"]
    },ensure_ascii=False))

if __name__=="__main__":
    main()
