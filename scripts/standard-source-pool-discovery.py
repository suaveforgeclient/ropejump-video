#!/usr/bin/env python3
import argparse, html, json, re, urllib.parse, urllib.request
from pathlib import Path

STANDARD=re.compile(
    r"(basic\s*(bounce|jump)|single\s*under|single\s*bounce|regular\s*(bounce|jump)|"
    r"basic\s*skipping|two[- ]?foot\s*(bounce|jump)|two\s+feet\s*(bounce|jump)|"
    r"salto\s+b[aá]sico|pular\s+corda.*b[aá]sic|縄跳び.*(基本|前跳び)|줄넘기.*(기본|양발))", re.I)
OTHER=re.compile(
    r"(double\s*under|triple\s*under|crossover|cross\s*over|criss\s*cross|boxer\s*step|"
    r"alternate\s*foot|alternating\s*foot|running\s*step|high\s*knee|side\s*swing|"
    r"freestyle|double\s*dutch|one[- ]?foot|release|mic\s*release|toad\b|frog\b|backward)", re.I)
EXPLAIN=re.compile(
    r"(tutorial|lesson|how\s+to|instruction|explainer|learn\s+to|coach|teaching|drill)", re.I)
DEMO=re.compile(
    r"(demo|demonstration|example|basic\s*bounce|single\s*under|single\s*bounce|basic\s*jump)", re.I)

A_RE=re.compile(r'(?is)<a\b([^>]*?)href=["\']([^"\']+)["\']([^>]*)>(.*?)</a>')
EMBED_RE=re.compile(r'(?is)<(?:iframe|video|source)\b([^>]*?)(?:src|data-src)=["\']([^"\']+)["\']([^>]*)>')
TITLE_RE=re.compile(r'(?is)(?:title|aria-label)=["\']([^"\']+)["\']')
ABS_URL_RE=re.compile(r'https?://[^\s"\'<>\\)]+', re.I)

def fetch(url):
    req=urllib.request.Request(url,headers={"User-Agent":"ROPEJUMP-research/2.3 source-pool"})
    with urllib.request.urlopen(req,timeout=20) as r:
        raw=r.read()
        ctype=r.headers.get("content-type","")
    return raw.decode("utf-8","replace"),ctype

def strip_tags(x):
    x=re.sub(r"(?is)<script.*?</script>|<style.*?</style>"," ",x)
    x=re.sub(r"(?s)<[^>]+>"," ",x)
    x=html.unescape(x)
    return re.sub(r"\s+"," ",x).strip()

def canonical_video(url):
    u=html.unescape(url).strip()
    if u.startswith("//"): u="https:"+u
    if not u.startswith("http"): return None
    p=urllib.parse.urlparse(u)
    host=p.netloc.lower().replace("www.","")
    path=p.path
    if "youtube.com" in host or "youtu.be" in host:
        return ("youtube",u,"LINK_ONLY_KNOWN_ACCESS_RISK")
    if "vimeo.com" in host:
        return ("vimeo",u,"MATERIALIZATION_TEST_REQUIRED")
    if "dailymotion.com" in host or "dai.ly" in host:
        return ("dailymotion",u,"MATERIALIZATION_TEST_REQUIRED")
    if "peertube" in host or "/w/" in path or "/videos/watch/" in path:
        return ("peertube_or_federated",u,"MATERIALIZATION_TEST_REQUIRED")
    if re.search(r"\.(mp4|webm|mov)(?:$|\?)",u,re.I):
        return ("direct_video",u,"DIRECT_MEDIA_TEST_REQUIRED")
    if any(x in host for x in ["instagram.com","tiktok.com","facebook.com"]):
        return ("social_video",u,"LINK_ONLY_UNCERTAIN_ACCESS")
    return None

def classify_label(label):
    label=(label or "").strip()
    standard=bool(STANDARD.search(label))
    other=bool(OTHER.search(label))
    if other and not standard:
        gate="OTHER_TYPE_EXPLICIT"
    elif standard and not other:
        gate="STANDARD_EXPLICIT_LABEL"
    elif standard and other:
        gate="MIXED_TYPE_LABEL"
    else:
        gate="TYPE_UNKNOWN"
    if EXPLAIN.search(label):
        risk="EXPLANATION_OR_LESSON_RISK"
    elif DEMO.search(label):
        risk="DEMO_HINT_NOT_VISUAL_PASS"
    else:
        risk="UNKNOWN_CONTENT_STRUCTURE"
    return gate,risk

def discover(config_path, out_dir, pilot_limit=30):
    cfg=json.loads(Path(config_path).read_text())
    rows={}
    fetches=[]

    def add(src, seed, raw, label, evidence_basis):
        raw=urllib.parse.urljoin(seed,html.unescape(raw).strip())
        cv=canonical_video(raw)
        if not cv: return
        platform,url,access=cv
        gate,risk=classify_label(label)
        key=platform+":"+re.sub(r"[^a-zA-Z0-9]+","_",url)[:220]
        item={
            "key":key,
            "targetType":cfg["targetType"],
            "sourcePoolId":src["id"],
            "sourceClass":src["class"],
            "authority":src["authority"],
            "sourcePoolUrl":seed,
            "platform":platform,
            "videoUrl":url,
            "evidenceBasis":evidence_basis,
            "linkLabel":label,
            "typeGate":gate,
            "contentRisk":risk,
            "accessPolicy":access,
            "visualStatus":"UNSCREENED",
            "autoPass":False
        }
        old=rows.get(key)
        rank={"STANDARD_EXPLICIT_LABEL":4,"MIXED_TYPE_LABEL":3,"OTHER_TYPE_EXPLICIT":2,"TYPE_UNKNOWN":1}
        if old is None or rank[item["typeGate"]]>rank.get(old.get("typeGate"),0):
            rows[key]=item

    for src in cfg["sources"]:
        seed=src["url"]
        try:
            page,ctype=fetch(seed)
            fetches.append({"sourceId":src["id"],"url":seed,"ok":True,"contentType":ctype,"bytes":len(page.encode())})
        except Exception as e:
            fetches.append({"sourceId":src["id"],"url":seed,"ok":False,"error":repr(e)})
            continue

        for pre,raw,post,inner in A_RE.findall(page):
            label=strip_tags(inner)
            if not label:
                m=TITLE_RE.search(pre+" "+post)
                label=html.unescape(m.group(1)).strip() if m else ""
            add(src,seed,raw,label,"anchor_label")

        for pre,raw,post in EMBED_RE.findall(page):
            m=TITLE_RE.search(pre+" "+post)
            label=html.unescape(m.group(1)).strip() if m else ""
            add(src,seed,raw,label,"embed_title" if label else "embed_without_label")

        for raw in set(ABS_URL_RE.findall(page)):
            add(src,seed,raw,"","raw_url_without_label")

    vals=list(rows.values())
    vals.sort(key=lambda x:(
        0 if x["typeGate"]=="STANDARD_EXPLICIT_LABEL" else 1,
        0 if x["authority"]=="high" else 1,
        0 if x["contentRisk"]=="DEMO_HINT_NOT_VISUAL_PASS" else 1,
        x["sourcePoolId"],x["videoUrl"]))

    explicit=[x for x in vals if x["typeGate"]=="STANDARD_EXPLICIT_LABEL" and x["evidenceBasis"] in ("anchor_label","embed_title")]
    accessible=[x for x in explicit if x["accessPolicy"] not in ("LINK_ONLY_KNOWN_ACCESS_RISK","LINK_ONLY_UNCERTAIN_ACCESS")]
    pilot=accessible[:pilot_limit]

    payload={
        "schemaVersion":3,
        "targetType":cfg["targetType"],
        "policy":cfg["policy"],
        "sourceFetches":fetches,
        "totalVideoLinks":len(vals),
        "explicitStandardLabel":len(explicit),
        "accessibleExplicitStandard":len(accessible),
        "visualPilotCount":len(pilot),
        "rows":vals,
        "visualPilot":pilot
    }
    out=Path(out_dir)
    out.mkdir(parents=True,exist_ok=True)
    (out/"source-pool-pilot.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
    (out/"visual-pilot.json").write_text(json.dumps({"rows":pilot},ensure_ascii=False,indent=2)+"\n")
    (out/"summary.json").write_text(json.dumps({
        "sourceFetches":fetches,
        "totalVideoLinks":len(vals),
        "explicitStandardLabel":len(explicit),
        "accessibleExplicitStandard":len(accessible),
        "visualPilotCount":len(pilot)
    },ensure_ascii=False,indent=2)+"\n")
    return payload

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="research/source-pools-standard.json")
    ap.add_argument("--out",default="out")
    ap.add_argument("--pilot-limit",type=int,default=30)
    args=ap.parse_args()
    payload=discover(args.config,args.out,args.pilot_limit)
    print(json.dumps({
        "totalVideoLinks":payload["totalVideoLinks"],
        "explicitStandardLabel":payload["explicitStandardLabel"],
        "accessibleExplicitStandard":payload["accessibleExplicitStandard"],
        "visualPilotCount":payload["visualPilotCount"]
    },ensure_ascii=False))

if __name__=="__main__":
    main()
