#!/usr/bin/env python3
import argparse, hashlib, json, re
from pathlib import Path
from remotezip import RemoteZip

OFFICIAL_DATASET_PAGE="https://www.crcv.ucf.edu/data/UCF101.php"
MATERIALIZATION_ZIP="https://huggingface.co/datasets/bitmind/UCF101Fullvideo/resolve/main/UCF101Fullvideo.zip?download=true"
NAME_RE=re.compile(r"(?:^|/)JumpRope/(v_JumpRope_g(\d{2})_c(\d{2})\.avi)$")

def list_jump_rope(rz):
    rows=[]
    for entry in rz.namelist():
        m=NAME_RE.search(entry)
        if not m:
            continue
        name,g,c=m.groups()
        split=entry.split("/",1)[0] if "/" in entry else ""
        rows.append({
            "filename":name,
            "group":int(g),
            "clip":int(c),
            "split":split,
            "archiveEntry":entry,
            "sourceKind":"UCF101_FULLVIDEO_SELECTIVE_RANGE"
        })
    return sorted(rows,key=lambda x:(x["group"],x["clip"],x["split"]))

def select_diverse(rows, limit):
    by_group={}
    for row in rows:
        by_group.setdefault(row["group"],[]).append(row)
    selected=[]
    round_i=0
    while len(selected)<limit:
        added=False
        for group in sorted(by_group):
            clips=by_group[group]
            if round_i < len(clips):
                selected.append(clips[round_i])
                added=True
                if len(selected)>=limit:
                    break
        if not added:
            break
        round_i+=1
    return selected

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="out")
    ap.add_argument("--limit",type=int,default=20)
    args=ap.parse_args()

    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    manifest=[]

    with RemoteZip(MATERIALIZATION_ZIP) as rz:
        rows=list_jump_rope(rz)
        selected=select_diverse(rows,args.limit)
        for row in selected:
            item={
                **row,
                "dataset":"UCF101",
                "datasetClass":"JumpRope",
                "officialDatasetPage":OFFICIAL_DATASET_PAGE,
                "materializationMirror":MATERIALIZATION_ZIP,
                "targetType":"STANDARD_BASIC_SINGLE_UNDER",
                "datasetLabelMeaning":"JumpRope only; subtype is NOT asserted",
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

    groups=sorted(set(x["group"] for x in rows))
    payload={
        "schemaVersion":3,
        "collector":"UCF101_JUMPROPE_SELECTIVE_RANGE",
        "officialDatasetPage":OFFICIAL_DATASET_PAGE,
        "materializationMirror":MATERIALIZATION_ZIP,
        "datasetClass":"JumpRope",
        "policy":{
            "fullArchiveDownloadAvoided":True,
            "selectiveHttpRangeExtraction":True,
            "selection":"group-diverse pilot; one clip per UCF source group before repeats",
            "datasetJumpRopeLabelDoesNotImplyStandard":True,
            "subtypeAutoClassificationForbidden":True,
            "wholeSourceVisualReviewRequired":True,
            "mixedTypeMustNotEnterPureStandardPool":True,
            "countingDeferredUntilPureStandardPass":True
        },
        "availableJumpRopeFiles":len(rows),
        "availableGroups":len(groups),
        "selectedCount":len(manifest),
        "downloadedCount":sum(1 for x in manifest if x.get("downloadOk")),
        "rows":manifest
    }
    (out/"manifest.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "availableJumpRopeFiles":payload["availableJumpRopeFiles"],
        "availableGroups":payload["availableGroups"],
        "selectedCount":payload["selectedCount"],
        "downloadedCount":payload["downloadedCount"]
    },ensure_ascii=False))

if __name__=="__main__":
    main()
