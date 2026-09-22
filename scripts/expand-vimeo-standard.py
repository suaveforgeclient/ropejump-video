#!/usr/bin/env python3
import argparse, json, re, subprocess
from pathlib import Path

STANDARD=re.compile(
    r"(basic\s*(bounce|jump)|single\s*under|single\s*bounce|regular\s*(bounce|jump)|"
    r"basic\s*skipping|two[- ]?foot\s*(bounce|jump)|two\s+feet\s*(bounce|jump)|"
    r"salto\s+b[aá]sico|pular\s+corda.*b[aá]sic|縄跳び.*(基本|前跳び)|줄넘기.*(기본|양발))", re.I)
OTHER=re.compile(
    r"(double\s*under|triple\s*under|crossover|cross\s*over|criss\s*cross|boxer\s*step|"
    r"alternate\s*foot|alternating\s*foot|running\s*step|high\s*knee|side\s*swing|"
    r"freestyle|double\s*dutch|one[- ]?foot|release|mic\s*release|toad\b|frog\b|backward)", re.I)
EXPLAIN=re.compile(r"(tutorial|lesson|how\s+to|instruction|explainer|learn\s+to|coach|teaching|drill)",re.I)

def normalize_entry(src, e):
    title=str(e.get("title") or e.get("fulltitle") or "")
    desc=str(e.get("description") or "")
    text=(title+" "+desc).strip()
    if not STANDARD.search(text) or OTHER.search(text):
        return None
    vid=str(e.get("id") or "").strip()
    url=str(e.get("webpage_url") or e.get("url") or "").strip()
    if url.isdigit():
        url="https://vimeo.com/"+url
    if not url and vid.isdigit():
        url="https://vimeo.com/"+vid
    if not url.startswith("http"):
        return None
    return {
        "targetType":"STANDARD_BASIC_SINGLE_UNDER",
        "sourcePoolId":src["id"],
        "sourceClass":src["class"],
        "authority":src["authority"],
        "sourcePoolUrl":src["url"],
        "platform":"vimeo",
        "id":vid,
        "title":title,
        "description":desc[:1200],
        "videoUrl":url,
        "typeGate":"STANDARD_EXPLICIT_PLATFORM_METADATA",
        "contentRisk":"EXPLANATION_OR_LESSON_RISK" if EXPLAIN.search(text) else "UNKNOWN_CONTENT_STRUCTURE",
        "visualStatus":"UNSCREENED",
        "autoPass":False
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="research/source-pools-standard.json")
    ap.add_argument("--out",default="out-vimeo")
    ap.add_argument("--pilot-limit",type=int,default=20)
    args=ap.parse_args()

    cfg=json.loads(Path(args.config).read_text())
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    rows={}
    sources=[]

    for src in cfg["sources"]:
        if src.get("expander")!="vimeo_flat":
            continue
        cmd=["python","-m","yt_dlp","--flat-playlist","--dump-single-json","--no-warnings",src["url"]]
        p=subprocess.run(cmd,text=True,capture_output=True,timeout=180)
        rec={"sourcePoolId":src["id"],"url":src["url"],"ok":p.returncode==0,"returnCode":p.returncode,"stderrTail":p.stderr[-1800:]}
        if p.returncode!=0:
            sources.append(rec); continue
        try:
            data=json.loads(p.stdout)
        except Exception as e:
            rec.update({"ok":False,"parseError":repr(e),"stdoutTail":p.stdout[-1200:]})
            sources.append(rec); continue
        entries=data.get("entries") or []
        rec["entryCount"]=len(entries)
        sources.append(rec)
        for e in entries:
            if not isinstance(e,dict): continue
            item=normalize_entry(src,e)
            if not item: continue
            key=item["platform"]+":"+item["id"]
            old=rows.get(key)
            if old is None or (item["authority"]=="high" and old.get("authority")!="high"):
                rows[key]=item

    vals=sorted(rows.values(),key=lambda x:(0 if x["authority"]=="high" else 1,x["sourcePoolId"],x["title"],x["id"]))
    pilot=vals[:args.pilot_limit]
    payload={
        "schemaVersion":1,
        "targetType":"STANDARD_BASIC_SINGLE_UNDER",
        "sourceExpansion":sources,
        "candidateCount":len(vals),
        "pilotCount":len(pilot),
        "rows":vals,
        "pilot":pilot
    }
    (out/"vimeo-standard-candidates.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
    (out/"pilot.json").write_text(json.dumps({"rows":pilot},ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"candidateCount":len(vals),"pilotCount":len(pilot),"sources":sources},ensure_ascii=False))

if __name__=="__main__":
    main()
