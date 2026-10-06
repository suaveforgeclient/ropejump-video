from __future__ import annotations
import argparse,json,math,random
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader

DE1="JR-20260915-DE1CE4AF"
POLICY="ROPEJUMP_F_V2_CLEAN"

class ChannelLayerNorm(nn.Module):
    def __init__(self,c): super().__init__(); self.norm=nn.LayerNorm(c)
    def forward(self,x): return self.norm(x.transpose(1,2)).transpose(1,2)
class CausalConv1d(nn.Module):
    def __init__(self,i,o,k,d=1): super().__init__(); self.pad=(k-1)*d; self.conv=nn.Conv1d(i,o,k,dilation=d)
    def forward(self,x): return self.conv(F.pad(x,(self.pad,0)))
class TBlock(nn.Module):
    def __init__(self,c,d,drop):
        super().__init__(); self.net=nn.Sequential(CausalConv1d(c,c,3,d),nn.ReLU(),ChannelLayerNorm(c),nn.Dropout(drop),CausalConv1d(c,c,3,d),nn.ReLU(),ChannelLayerNorm(c))
    def forward(self,x): return x+self.net(x)
class TCN(nn.Module):
    def __init__(self,f,c,drop):
        super().__init__(); self.stem=nn.Sequential(nn.Conv1d(f,c,1),nn.ReLU(),ChannelLayerNorm(c)); self.blocks=nn.Sequential(*(TBlock(c,d,drop) for d in (1,2,4,8)))
    def forward(self,x): return self.blocks(self.stem(x.transpose(1,2))).transpose(1,2)
class GRUEnc(nn.Module):
    def __init__(self,f,c,drop):
        super().__init__(); self.stem=nn.Sequential(nn.Linear(f,c),nn.ReLU(),nn.LayerNorm(c)); self.gru=nn.GRU(c,c,batch_first=True)
    def forward(self,x): return self.gru(self.stem(x))[0]
class Hybrid(nn.Module):
    def __init__(self,f,c,drop):
        super().__init__(); self.tcn=TCN(f,c,drop); self.gru=nn.GRU(c,c,batch_first=True)
    def forward(self,x): return self.gru(self.tcn(x))[0]
class Model(nn.Module):
    def __init__(self,f,c=64,drop=.1,arch="tcn"):
        super().__init__(); self.arch=arch
        self.enc=TCN(f,c,drop) if arch=="tcn" else GRUEnc(f,c,drop) if arch=="gru" else Hybrid(f,c,drop)
        self.head=nn.Linear(c,1)
    def forward(self,x): return self.head(self.enc(x)).squeeze(-1)

class DS(Dataset):
    def __init__(self,rows): self.rows=rows
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        return np.load(r["features_npy"]).astype(np.float32),np.load(r["targets_npy"]).astype(np.float32),r
def collate(batch):
    n=max(len(x[0]) for x in batch); f=batch[0][0].shape[1]; b=len(batch)
    X=torch.zeros(b,n,f);Y=torch.zeros(b,n);M=torch.zeros(b,n,dtype=torch.bool)
    for i,(x,y,r) in enumerate(batch):
        k=len(x);X[i,:k]=torch.from_numpy(x);Y[i,:k]=torch.from_numpy(y);M[i,:k]=True
    return X,Y,M
def augment(x,mask,strength=.25):
    if strength<=0:return x
    z=x.clone(); noise=torch.randn_like(z); scale=torch.zeros(z.shape[-1])
    scale[:7]=.0025*strength; scale[7:12]=.015*strength; scale[14:]=.01*strength
    z=z+noise*scale
    drop=(torch.rand(z.shape[0],z.shape[1])<.01*strength)&mask
    z[:,:,7:11]=torch.where(drop.unsqueeze(-1),torch.zeros_like(z[:,:,7:11]),z[:,:,7:11])
    z[:,:,12]=torch.where(drop,torch.zeros_like(z[:,:,12]),z[:,:,12])
    return z

def decoder(times,probs,threshold,min_sep):
    active=False;peak_t=None;peak_p=-1.;last=None;out=[]
    for t,p in zip(times,probs):
        if p>=threshold:
            if not active: active=True;peak_t=float(t);peak_p=float(p)
            elif p>peak_p: peak_t=float(t);peak_p=float(p)
        elif active:
            if peak_t is not None and (last is None or peak_t-last>=min_sep): out.append(peak_t);last=peak_t
            active=False;peak_t=None;peak_p=-1.
    if active and peak_t is not None and (last is None or peak_t-last>=min_sep): out.append(peak_t)
    return out

def match(gt,pred,tol):
    gt=sorted(map(float,gt));pred=sorted(map(float,pred));n=len(gt);m=len(pred)
    dp=[[None]*(m+1) for _ in range(n+1)];dp[0][0]=(0,0.,None)
    def better(a,b): return b is None or a[0]>b[0] or (a[0]==b[0] and a[1]<b[1])
    def relax(i,j,v):
        if better(v,dp[i][j]):dp[i][j]=v
    for i in range(n+1):
        for j in range(m+1):
            cur=dp[i][j]
            if cur is None:continue
            tp,err,_=cur
            if i<n:relax(i+1,j,(tp,err,("g",i,j)))
            if j<m:relax(i,j+1,(tp,err,("p",i,j)))
            if i<n and j<m:
                e=abs(gt[i]-pred[j])
                if e<=tol:relax(i+1,j+1,(tp+1,err+e,("m",i,j,e)))
    i=n;j=m;matches=0
    while i or j:
        a=dp[i][j][2] if dp[i][j] else None
        if a is None:break
        if a[0]=="m":matches+=1;i,j=a[1],a[2]
        else:i,j=a[1],a[2]
    fp=m-matches;fn=n-matches
    return {"gt_count":n,"predicted_count":m,"tp":matches,"fp":fp,"fn":fn,"count_exact":n==m,"event_exact":fp==0 and fn==0}

def validate_manifest(rows):
    if not rows:raise RuntimeError("manifest_empty")
    if any(r.get("policy_id")!=POLICY for r in rows):raise RuntimeError("policy_mismatch")
    if any(r.get("clip_id")==DE1 for r in rows):raise RuntimeError("DE1_contamination")
    train=[r for r in rows if r["split"]=="train"]
    for r in train:
        kind=str(r.get("sample_kind") or "POSITIVE")
        if kind=="POSITIVE":
            if r.get("jump_rope_type")!="STANDARD_BASIC_SINGLE_UNDER": raise RuntimeError(f"nonstandard_positive_train:{r.get('id')}")
            if str(r.get("clip_id","")).startswith("JR-"): raise RuntimeError(f"user_original_positive_train_forbidden:{r.get('id')}")
            if int(r.get("expected_count") or 0)<1: raise RuntimeError(f"positive_train_without_events:{r.get('id')}")
        elif kind=="HARD_NEGATIVE":
            if int(r.get("expected_count") or 0)!=0 or (r.get("event_times_ms") or []): raise RuntimeError(f"hard_negative_has_positive_target:{r.get('id')}")
            if r.get("jump_rope_type") not in {"NO_JUMP_HARD_NEGATIVE","NON_STANDARD_NO_VISIBLE_ROPE_BODY_JUMP"}: raise RuntimeError(f"hard_negative_type_invalid:{r.get('id')}")
        else:
            raise RuntimeError(f"sample_kind_invalid:{r.get('id')}:{kind}")
    if not any((r.get("sample_kind") or "POSITIVE")=="HARD_NEGATIVE" for r in train): raise RuntimeError("hard_negative_train_missing")
    groups={}
    for r in rows: groups.setdefault(r["dataset_group"],set()).add(r["split"])
    leak={g:sorted(v) for g,v in groups.items() if len(v)>1}
    if leak:raise RuntimeError(f"group_leakage:{leak}")
    for split in ("train","val","test"):
        if not any(r["split"]==split for r in rows):raise RuntimeError(f"empty_split:{split}")

def train_one(rows,arch,out,epochs=60,seed=119):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);torch.set_num_threads(2)
    train=[r for r in rows if r["split"]=="train"];ds=DS(train);dl=DataLoader(ds,batch_size=8,shuffle=True,collate_fn=collate)
    f=ds[0][0].shape[1];model=Model(f,64,.1,arch);opt=torch.optim.AdamW(model.parameters(),lr=7e-4,weight_decay=1e-4)
    pos=sum(float(np.load(r["targets_npy"]).sum()) for r in train); total=sum(len(np.load(r["targets_npy"])) for r in train);pw=max(1.,min(80.,(total-pos)/max(pos,1e-6)))
    history=[]
    for ep in range(epochs):
        model.train();ls=0.
        for x,y,m in dl:
            x=augment(x,m,.25);opt.zero_grad(set_to_none=True);logits=model(x)
            raw=F.binary_cross_entropy_with_logits(logits,y,reduction="none");w=1+y*(pw-1)
            loss=(raw*w*m).sum()/m.sum().clamp_min(1);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step();ls+=float(loss)
        history.append(ls/max(1,len(dl)))
    state={"model":model.state_dict(),"feature_count":f,"channels":64,"dropout":.1,"architecture":arch,"epochs":epochs,"seed":seed,"random_init":True,"legacy_checkpoint_loaded":False}
    torch.save(state,out)
    return history

def load_model(path):
    s=torch.load(path,map_location="cpu");m=Model(int(s["feature_count"]),int(s["channels"]),float(s["dropout"]),str(s["architecture"]));m.load_state_dict(s["model"]);m.eval();return m

def cache_predictions(rows,model,split):
    out=[]
    with torch.no_grad():
        for r in rows:
            if r["split"]!=split:continue
            x=np.load(r["features_npy"]).astype(np.float32);t=np.load(r["times_npy"]).astype(np.float64)
            p=torch.sigmoid(model(torch.from_numpy(x).unsqueeze(0))[0]).numpy()
            out.append((r,t,p))
    return out
def score(cached,th,sep):
    rr=[]
    for r,t,p in cached:
        pred=decoder(t,p,th,sep);m=match(r["event_times_ms"],pred,float(r["tolerance_ms"]))
        rr.append({"id":r["id"],"clip_id":r["clip_id"],"expected_count":r["expected_count"],**m})
    return {"samples":len(rr),"exact_count_samples":sum(x["count_exact"] for x in rr),"event_exact_samples":sum(x["event_exact"] for x in rr),"total_fp":sum(x["fp"] for x in rr),"total_fn":sum(x["fn"] for x in rr),"results":rr}
def calibrate(cached):
    best=None
    for sep in (100.,120.,145.,160.,180.,200.):
        for th in np.arange(.10,.921,.02):
            s=score(cached,float(th),sep);key=(-s["event_exact_samples"],-s["exact_count_samples"],s["total_fp"]+s["total_fn"],s["total_fp"],abs(float(th)-.5),sep)
            if best is None or key<best[0]:best=(key,float(th),sep,s)
    return {"threshold":best[1],"min_separation_ms":best[2],"metrics":best[3]}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--manifest",type=Path,required=True);ap.add_argument("--output-root",type=Path,required=True);ap.add_argument("--epochs",type=int,default=60);a=ap.parse_args()
    rows=[json.loads(x) for x in a.manifest.read_text().splitlines() if x.strip()];validate_manifest(rows)
    a.output_root.mkdir(parents=True,exist_ok=True);candidates=[]
    for arch in ("tcn","gru","hybrid"):
        ck=a.output_root/f"{arch}.pt";hist=train_one(rows,arch,ck,a.epochs)
        model=load_model(ck);cal=calibrate(cache_predictions(rows,model,"val"));test=score(cache_predictions(rows,model,"test"),cal["threshold"],cal["min_separation_ms"])
        candidates.append({"architecture":arch,"checkpoint":str(ck),"loss_first":hist[0],"loss_last":hist[-1],"calibration":cal,"test":test})
    def val_key(x):
        m=x["calibration"]["metrics"];return (-m["event_exact_samples"],-m["exact_count_samples"],m["total_fp"]+m["total_fn"],m["total_fp"],x["architecture"])
    candidates.sort(key=val_key);winner=candidates[0]
    shutil_src=Path(winner["checkpoint"]);best=a.output_root/"best_model.pt";best.write_bytes(shutil_src.read_bytes())
    summary={
        "policy":POLICY,"random_init_only":True,"legacy_checkpoint_loaded":False,
        "train_rows":sum(r["split"]=="train" for r in rows),"train_clips":len({r["clip_id"] for r in rows if r["split"]=="train"}),
        "train_positive_clips":len({r["clip_id"] for r in rows if r["split"]=="train" and (r.get("sample_kind") or "POSITIVE")=="POSITIVE"}),
        "train_hard_negative_clips":len({r["clip_id"] for r in rows if r["split"]=="train" and r.get("sample_kind")=="HARD_NEGATIVE"}),
        "val_clips":len({r["clip_id"] for r in rows if r["split"]=="val"}),"test_clips":len({r["clip_id"] for r in rows if r["split"]=="test"}),
        "winner":winner,"candidates":candidates,
        "promotion_pass":winner["test"]["total_fp"]==0 and winner["test"]["total_fn"]==0 and winner["test"]["event_exact_samples"]==winner["test"]["samples"]
    }
    (a.output_root/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print("FV2_TRAIN_SUMMARY="+json.dumps({k:summary[k] for k in ("train_rows","train_clips","train_positive_clips","train_hard_negative_clips","val_clips","test_clips","promotion_pass")},sort_keys=True))
    print("FV2_WINNER="+json.dumps({"architecture":winner["architecture"],"calibration":winner["calibration"],"test":{k:v for k,v in winner["test"].items() if k!="results"}},sort_keys=True))

if __name__=="__main__":
    import shutil
    main()
