#!/usr/bin/env python3
import argparse, hashlib, json, re
from pathlib import Path
from remotezip import RemoteZip

REVIEW_DEFAULT="review/ucf101-standard-pilot-20260922.json"
OFFICIAL_DATASET_PAGE="https://www.crcv.ucf.edu/data/UCF101.php"
MATERIALIZATION_ZIP="https://huggingface.co/datasets/bitmind/UCF101Fullvideo/resolve/main/UCF101Fullvideo.zip?download=true"
NAME_RE=re.compile(r"(?:^|/)JumpRope/(v_JumpRope_g(\\d{2})_c(\\d{2})\\.avi)$")

def list_jump_rope(rz):
    rows=[]
    for entry in rz.namelist():
        m=NAME_RE.search(entry)
        if not m:
            continue
        name,g,c=m.groups()
        rows.append({
            "filename":name,
            "group":int(g),
            "clip":int(c),
            "archiveEntry":entry,
            "sourceKind":"UCF101_FULLVIDEO_SELECTIVE_RANGE"
        })
    return sorted(rows,key=lambda x:(x["group"],x["clip"]))

def pass_groups(review):
    out=[]
    for row in review.get("rows",[]):
        if row.get("status")!="PASS_STANDARD":
            continue
        m=re.search(r"_g(\\d{2})_c(\\d{2})$",str(row.get("id") or ""))
        if m:
            out.append(int(m.group(1)))
    return sorted(set(out))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--review",default=REVIEW_DEFAULT)
    ap.add_argument("--out",default="out")
    ap.add_argument("--rank",type=int,default=2,help="1-based clip rank within each already-PASS source group")
    args=ap.parse_args()

    if args.rank < 2:
        raise SystemExit("followup_rank_must_be_at_least_2")

    review=json.loads(Path(args.review).read_text(encoding="utf-8"))
    groups=pass_groups(review)
    if not groups:
        raise SystemExit("no_pass_groups")

    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    manifest=[]

    with RemoteZip(MATERIALIZATION_ZIP) as rz:
        allrows=list_jump_rope(rz)
        by_group={}
        for row in allrows:
            by_group.setdefault(row["group"],[]).append(row)

        selected=[]
        for group in groups:
            clips=by_group.get(group,[])
            idx=args.rank-1
            if idx < len(clips):
                selected.append(clips[idx])

        for row in selected:
            item={
                **row,
                "dataset":"UCF101",
                "datasetClass":"JumpRope",
                "officialDatasetPage":OFFICIAL_DATASET_PAGE,
                "materializationMirror":MATERIALIZATION_ZIP,
                "targetType":"STANDARD_BASIC_SINGLE_UNDER",
                "selectionReason":"NEXT_CLIP_FROM_VISUALLY_PASS_STANDARD_GROUP",
                "groupPassDoesNotImplyClipPass":True,
                "typeStatus":"NEED_VISUAL_REVIEW",
                "contentStatus":"NEED_VISUAL_REVIEW",
                "countStatus":"NOT_REVIEWED",
                "autoPass":False
            }
            try:
                data=rz.read(row["archiveEntry"])
                if len(data)<1024:
                    raise RuntimeError(f"entry_too_small:{len(data)}")
                dest=out/row["filename"]
                dest.write_bytes(data)
                item["sourceRef"]=MATERIALIZATION_ZIP+"#entry="+row["archiveEntry"]
                item["downloadOk"]=True
                item["bytes"]=len(data)
                item["sha256"]=hashlib.sha256(data).hexdigest()
            except Exception as e:
                item["downloadOk"]=False
                item["error"]=repr(e)
            manifest.append(item)

    payload={
        "schemaVersion":1,
        "collector":"UCF101_STANDARD_GROUP_FOLLOWUP",
        "sourceReview":args.review,
        "sourcePassGroups":groups,
        "clipRank":args.rank,
        "policy":{
            "passGroupDoesNotAutoPassSiblingClip":True,
            "nearDuplicateExpansionIsLimited":True,
            "oneFollowupPerPassGroup":True,
            "wholeSourceVisualReviewRequired":True,
            "mixedTypeRejectRequired":True,
            "countingDeferredUntilVisualPass":True
        },
        "selectedCount":len(manifest),
        "downloadedCount":sum(1 for x in manifest if x.get("downloadOk")),
        "rows":manifest
    }
    (out/"manifest.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "sourcePassGroups":groups,
        "selectedCount":payload["selectedCount"],
        "downloadedCount":payload["downloadedCount"]
    },ensure_ascii=False))

if __name__=="__main__":
    main()
