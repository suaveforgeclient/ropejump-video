#!/usr/bin/env python3
import argparse, hashlib, json, re
from pathlib import Path
from remotezip import RemoteZip

OFFICIAL_DATASET_PAGE="https://www.crcv.ucf.edu/data/UCF101.php"
MATERIALIZATION_ZIP="https://huggingface.co/datasets/bitmind/UCF101Fullvideo/resolve/main/UCF101Fullvideo.zip?download=true"
NAME_RE=re.compile(r"(?:^|/)JumpRope/(v_JumpRope_g(\d{2})_c(\d{2})\.avi)$")

def list_rows(rz):
    out=[]
    for entry in rz.namelist():
        m=NAME_RE.search(entry)
        if not m:
            continue
        name,g,c=m.groups()
        out.append({"filename":name,"group":int(g),"clip":int(c),"archiveEntry":entry})
    return sorted(out,key=lambda x:(x["group"],x["clip"]))

def reviewed_group_status(review):
    status={}
    for row in review.get("rows",[]):
        m=re.search(r"_g(\d{2})_c(\d{2})$",str(row.get("id") or ""))
        if not m:
            continue
        g=int(m.group(1)); c=int(m.group(2))
        if c==1:
            status[g]=str(row.get("status") or "")
    return status

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--review",default="review/ucf101-standard-pilot-20260922.json")
    ap.add_argument("--out",default="out")
    ap.add_argument("--rank",type=int,default=2)
    args=ap.parse_args()

    review=json.loads(Path(args.review).read_text(encoding="utf-8"))
    first=reviewed_group_status(review)
    target_groups=sorted(g for g,s in first.items() if s!="PASS_STANDARD")
    if not target_groups:
        raise SystemExit("no_nonpass_groups")

    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    manifest=[]
    with RemoteZip(MATERIALIZATION_ZIP) as rz:
        allrows=list_rows(rz)
        by={}
        for x in allrows:
            by.setdefault(x["group"],[]).append(x)
        for g in target_groups:
            clips=by.get(g,[])
            idx=args.rank-1
            if idx>=len(clips):
                continue
            row=clips[idx]
            item={
                **row,
                "dataset":"UCF101",
                "datasetClass":"JumpRope",
                "officialDatasetPage":OFFICIAL_DATASET_PAGE,
                "materializationMirror":MATERIALIZATION_ZIP,
                "targetType":"STANDARD_BASIC_SINGLE_UNDER",
                "selectionReason":"DIVERSITY_PROBE_NEXT_CLIP_FROM_C01_NONPASS_GROUP",
                "groupPriorStatus":first.get(g),
                "typeStatus":"NEED_VISUAL_REVIEW",
                "contentStatus":"NEED_VISUAL_REVIEW",
                "countStatus":"NOT_REVIEWED",
                "autoPass":False
            }
            try:
                data=rz.read(row["archiveEntry"])
                if len(data)<1024:
                    raise RuntimeError(f"entry_too_small:{len(data)}")
                dest=out/row["filename"]; dest.write_bytes(data)
                item["sourceRef"]=MATERIALIZATION_ZIP+"#entry="+row["archiveEntry"]
                item["downloadOk"]=True
                item["bytes"]=len(data)
                item["sha256"]=hashlib.sha256(data).hexdigest()
            except Exception as e:
                item["downloadOk"]=False; item["error"]=repr(e)
            manifest.append(item)

    payload={
        "schemaVersion":1,
        "collector":"UCF101_STANDARD_DIVERSITY_PROBE",
        "sourceReview":args.review,
        "clipRank":args.rank,
        "targetGroups":target_groups,
        "policy":{
            "priorNonpassDoesNotImplySiblingNonpass":True,
            "programOnlySelectsCandidates":True,
            "noSubtypeAutoPass":True,
            "wholeSourceVisualReviewRequired":True,
            "countingDeferredUntilVisualPass":True
        },
        "selectedCount":len(manifest),
        "downloadedCount":sum(1 for x in manifest if x.get("downloadOk")),
        "rows":manifest
    }
    (out/"manifest.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "targetGroups":target_groups,
        "selectedCount":payload["selectedCount"],
        "downloadedCount":payload["downloadedCount"]
    },ensure_ascii=False))

if __name__=="__main__":
    main()
