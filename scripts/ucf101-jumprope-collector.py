#!/usr/bin/env python3
import argparse, hashlib, json, re, ssl, urllib.parse, urllib.request
from pathlib import Path

ROOT="https://www.crcv.ucf.edu/THUMOS14/UCF101/UCF101/"
NAME_RE=re.compile(r"(v_JumpRope_g(\d{2})_c(\d{2})\.avi)")
UA={"User-Agent":"ROPEJUMP-research/3.0 ucf101-jumprope-collector"}

def fetch_bytes(url, timeout=45):
    req=urllib.request.Request(url,headers=UA)
    ctx=ssl._create_unverified_context()
    with urllib.request.urlopen(req,timeout=timeout,context=ctx) as r:
        return r.read()

def list_jump_rope():
    raw=fetch_bytes(ROOT,30).decode("utf-8","replace")
    found={}
    for name,g,c in NAME_RE.findall(raw):
        found[name]={"filename":name,"group":int(g),"clip":int(c)}
    return sorted(found.values(),key=lambda x:(x["group"],x["clip"]))

def select_diverse(rows, limit):
    # UCF101 group IDs correspond to source/video groups used to avoid
    # train/test leakage. For a first visual pilot, prefer one clip per group
    # before taking a second clip from any group so the pilot is not dominated
    # by near-duplicate clips from the same source group.
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

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="out")
    ap.add_argument("--limit",type=int,default=20)
    args=ap.parse_args()

    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    rows=list_jump_rope()
    selected=select_diverse(rows,args.limit)
    manifest=[]

    for row in selected:
        url=urllib.parse.urljoin(ROOT,row["filename"])
        item={
            **row,
            "dataset":"UCF101",
            "datasetClass":"JumpRope",
            "sourceRoot":ROOT,
            "sourceUrl":url,
            "targetType":"STANDARD_BASIC_SINGLE_UNDER",
            "datasetLabelMeaning":"JumpRope only; subtype is NOT asserted",
            "typeStatus":"NEED_VISUAL_REVIEW",
            "contentStatus":"NEED_VISUAL_REVIEW",
            "countStatus":"NOT_REVIEWED",
            "autoPass":False
        }
        try:
            data=fetch_bytes(url,60)
            path=out/row["filename"]
            path.write_bytes(data)
            item["downloadOk"]=True
            item["bytes"]=len(data)
            item["sha256"]=hashlib.sha256(data).hexdigest()
        except Exception as e:
            item["downloadOk"]=False
            item["error"]=repr(e)
        manifest.append(item)

    payload={
        "schemaVersion":1,
        "collector":"UCF101_OFFICIAL_INDIVIDUAL_FILE",
        "root":ROOT,
        "datasetClass":"JumpRope",
        "policy":{
            "fullArchiveDownloadAvoided":True,
            "selection":"group-diverse pilot; one clip per UCF source group before repeats",
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
        "availableJumpRopeFiles":payload["availableJumpRopeFiles"],
        "availableGroups":payload["availableGroups"],
        "selectedCount":payload["selectedCount"],
        "downloadedCount":payload["downloadedCount"]
    },ensure_ascii=False))

if __name__=="__main__":
    main()
