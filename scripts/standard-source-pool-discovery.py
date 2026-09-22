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
        if "/review/" in path:
            return ("vimeo",u,"LINK_ONLY_REVIEW_URL")
        if re.fullmatch(r"/\\d+/?", path) or re.search(r"/(?:video/)?\\d+/?$", path):
            return ("vimeo",u,"MATERIALIZATION_TEST_REQUIRED")
        return None
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

def ddg_search(query):
    url="https://html.duckduckgo.com/html/?"+urllib.parse.urlencode({"q":query})
    page,_=fetch(url)
    out=[]
    for href,title_html in DDG_RESULT_RE.findall(page):
        href=html.unescape(href)
        p=urllib.parse.urlparse(href)
        qs=urllib.parse.parse_qs(p.query)
        if "uddg" in qs:
            href=qs["uddg"][0]
        pos=page.find(title_html)
        around=page[pos:pos+2200] if pos>=0 else title_html
        label=strip_tags(title_html+" "+around)[:1400]
        out.append((href,label))
    return out

def discover(config_path, out_dir, pilot_limit=30, constrained_web_search=False):
    cfg=json.loads(Path(config_path).read_text())
    rows={}
    fetches=[]

    def add(src, seed, raw, label, evidence_basis, search_query="", search_alias=""):
        raw=urllib.parse.urljoin(seed,html.unescape(raw).strip())
        cv=canonical_video(raw)
        if not cv: return
        platform,url,access=cv
        gate,risk=classify_label(label)
        if evidence_basis=="constrained_web_search":
            standard=bool(STANDARD.search(label or ""))
            other=bool(OTHER.search(label or ""))
            if standard and not other:
                gate="STANDARD_SEARCH_EVIDENCE"
            elif other and not standard:
                gate="OTHER_TYPE_SEARCH_EVIDENCE"
            else:
                gate="TYPE_UNKNOWN"
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
            "autoPass":False,
            "searchQuery":search_query,
            "searchAlias":search_alias
        }
        old=rows.get(key)
        rank={"STANDARD_EXPLICIT_LABEL":5,"STANDARD_SEARCH_EVIDENCE":4,"MIXED_TYPE_LABEL":3,
              "OTHER_TYPE_EXPLICIT":2,"OTHER_TYPE_SEARCH_EVIDENCE":2,"TYPE_UNKNOWN":1}
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

        for em in EMBED_RE.finditer(page):
            pre,raw,post=em.groups()
            m=TITLE_RE.search(pre+" "+post)
            title=html.unescape(m.group(1)).strip() if m else ""
            label=title
            basis="embed_title" if title else "embed_without_label"

            # For unlabeled/generic embeds, the nearest preceding section heading is
            # usable as metadata evidence only when it is close to the embed.
            heading=""
            last=None
            for hm in HEADING_RE.finditer(page, 0, em.start()):
                last=hm
            if last is not None and em.start()-last.end() <= 2200:
                heading=strip_tags(last.group(1))
            title_gate,_=classify_label(title)
            heading_gate,_=classify_label(heading)
            if title_gate=="TYPE_UNKNOWN" and heading_gate!="TYPE_UNKNOWN":
                label=heading
                basis="nearest_heading"
            add(src,seed,raw,label,basis)

        for raw in set(ABS_URL_RE.findall(page)):
            add(src,seed,raw,"","raw_url_without_label")

    if constrained_web_search:
        search_domains=["vimeo.com","dailymotion.com"]
        phrases=["basic bounce","single bounce","single under","basic jump rope"]
        for src in cfg["sources"]:
            aliases=src.get("searchAliases") or []
            if not aliases:
                continue
            alias=str(aliases[0])
            for domain in search_domains:
                for phrase in phrases:
                    q=f'site:{domain} "{alias}" "{phrase}"'
                    try:
                        found=ddg_search(q)
                    except Exception as e:
                        fetches.append({"sourceId":src["id"],"searchQuery":q,"ok":False,"error":repr(e)})
                        continue
                    fetches.append({"sourceId":src["id"],"searchQuery":q,"ok":True,"results":len(found)})
                    for raw,label in found:
                        add(src,src["url"],raw,label,"constrained_web_search",q,alias)

    vals=list(rows.values())
    vals.sort(key=lambda x:(
        0 if x["typeGate"]=="STANDARD_EXPLICIT_LABEL" else (1 if x["typeGate"]=="STANDARD_SEARCH_EVIDENCE" else 2),
        0 if x["authority"]=="high" else 1,
        0 if x["contentRisk"]=="DEMO_HINT_NOT_VISUAL_PASS" else 1,
        x["sourcePoolId"],x["videoUrl"]))

    explicit=[x for x in vals if x["typeGate"]=="STANDARD_EXPLICIT_LABEL" and x["evidenceBasis"] in ("anchor_label","embed_title","nearest_heading")]
    search_evidence=[x for x in vals if x["typeGate"]=="STANDARD_SEARCH_EVIDENCE" and x["evidenceBasis"]=="constrained_web_search"]
    usable_access=lambda x: x["accessPolicy"] not in (
        "LINK_ONLY_KNOWN_ACCESS_RISK","LINK_ONLY_UNCERTAIN_ACCESS","LINK_ONLY_REVIEW_URL")
    accessible=[x for x in explicit if usable_access(x)]
    accessible_search=[x for x in search_evidence if usable_access(x)]
    pilot=(accessible+accessible_search)[:pilot_limit]

    payload={
        "schemaVersion":3,
        "targetType":cfg["targetType"],
        "policy":cfg["policy"],
        "sourceFetches":fetches,
        "totalVideoLinks":len(vals),
        "explicitStandardLabel":len(explicit),
        "accessibleExplicitStandard":len(accessible),
        "searchEvidenceStandard":len(search_evidence),
        "accessibleSearchEvidence":len(accessible_search),
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
        "searchEvidenceStandard":len(search_evidence),
        "accessibleSearchEvidence":len(accessible_search),
        "visualPilotCount":len(pilot)
    },ensure_ascii=False,indent=2)+"\n")
    return payload

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="research/source-pools-standard.json")
    ap.add_argument("--out",default="out")
    ap.add_argument("--pilot-limit",type=int,default=30)
    ap.add_argument("--constrained-web-search",action="store_true")
    args=ap.parse_args()
    payload=discover(args.config,args.out,args.pilot_limit,args.constrained_web_search)
    print(json.dumps({
        "totalVideoLinks":payload["totalVideoLinks"],
        "explicitStandardLabel":payload["explicitStandardLabel"],
        "accessibleExplicitStandard":payload["accessibleExplicitStandard"],
        "searchEvidenceStandard":payload["searchEvidenceStandard"],
        "accessibleSearchEvidence":payload["accessibleSearchEvidence"],
        "visualPilotCount":payload["visualPilotCount"]
    },ensure_ascii=False))

if __name__=="__main__":
    main()
