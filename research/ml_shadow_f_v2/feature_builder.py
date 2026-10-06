from __future__ import annotations
import math
import numpy as np

FEATURE_SCHEMA = "ropejump-shadow-f-v3-motion"
FEATURE_NAMES = [
    "left_ankle_y","right_ankle_y","left_knee_y","right_knee_y",
    "left_hip_y","right_hip_y","body_center_y","left_ankle_vy",
    "right_ankle_vy","body_center_vy","body_center_ay","ankle_delta_y",
    "pose_confidence","dt_s","motion_body_dy","motion_body_confidence",
    "motion_body_energy","camera_dy","camera_confidence","background_energy",
    "mean_luma","contrast","dark_fraction",
]
CONFIDENCE_INDICES=[11,12,23,24,25,26,27,28,29,30,31,32]

def _conf(p):
    if not p: return 0.0
    return float(min(p.get("visibility",1.0),p.get("presence",1.0)))

def pose_confidence(lm):
    if not isinstance(lm,list) or len(lm)<33: return 0.0
    values=sorted(_conf(lm[i]) for i in CONFIDENCE_INDICES)
    return float(sum(values[1:])/len(values[1:])) if len(values)>1 else 0.0

def _finite_motion(motion,key):
    if not isinstance(motion,dict): return 0.0
    v=float(motion.get(key,0.0) or 0.0)
    return v if math.isfinite(v) else 0.0

def build_features(pose):
    rows=pose.get("rows")
    if not isinstance(rows,list) or not rows: raise ValueError("pose_rows_missing")
    out=[]; times=[]; last_coords=None; last_t=None; last_valid=False; last_vy=0.0
    default_dt=1.0/float(pose.get("analysisFps") or pose.get("fps") or 30.0)
    for row in rows:
        t=float(row["mediaTimeMs"])
        if last_t is not None and t<=last_t: raise ValueError("pose_time_not_increasing")
        dt=default_dt if last_t is None else max(.001,min(.25,(t-last_t)/1000.0))
        lm=row.get("landmarks"); valid=isinstance(lm,list) and len(lm)>=33
        conf=pose_confidence(lm) if valid else 0.0
        if valid:
            coords=np.array([lm[27]["y"],lm[28]["y"],lm[25]["y"],lm[26]["y"],lm[23]["y"],lm[24]["y"]],dtype=np.float32)
            if not np.isfinite(coords).all(): raise ValueError("non_finite_landmark")
        elif last_coords is not None:
            coords=last_coords.copy()
        else:
            coords=np.zeros(6,dtype=np.float32)
        body_center=float((coords[4]+coords[5])*.5)
        if last_coords is not None and valid and last_valid:
            left_vy=float((coords[0]-last_coords[0])/dt)
            right_vy=float((coords[1]-last_coords[1])/dt)
            prev_center=float((last_coords[4]+last_coords[5])*.5)
            body_vy=float((body_center-prev_center)/dt)
            body_ay=float((body_vy-last_vy)/dt)
        else:
            left_vy=right_vy=body_vy=body_ay=0.0
        m=row.get("motion")
        out.append([
            *map(float,coords),body_center,left_vy,right_vy,body_vy,body_ay,
            float(coords[0]-coords[1]),conf,dt,
            _finite_motion(m,"bodyDyNorm"),_finite_motion(m,"bodyConfidence"),_finite_motion(m,"bodyEnergy"),
            _finite_motion(m,"cameraDyNorm"),_finite_motion(m,"cameraConfidence"),_finite_motion(m,"backgroundEnergy"),
            _finite_motion(m,"meanLuma"),_finite_motion(m,"contrast"),_finite_motion(m,"darkFraction"),
        ])
        times.append(t); last_coords=coords; last_t=t; last_valid=valid; last_vy=body_vy if valid else 0.0
    features=np.asarray(out,dtype=np.float32); times=np.asarray(times,dtype=np.float64)
    if features.shape[1]!=len(FEATURE_NAMES) or not np.isfinite(features).all(): raise ValueError("invalid_features")
    return features,times

def build_targets(times_ms,event_times,tolerance_ms=220.0,target_width_frames=.55):
    y=np.zeros(len(times_ms),dtype=np.float32)
    if not event_times: return y
    diffs=np.diff(times_ms); diffs=diffs[np.isfinite(diffs)&(diffs>0)]
    frame_ms=float(np.median(diffs)) if len(diffs) else 1000/30
    cutoff=max(frame_ms*target_width_frames,frame_ms*.51); sigma=max(4.0,cutoff*.42)
    for event in event_times:
        d=np.abs(times_ms-float(event)); i=int(np.argmin(d))
        if d[i]>max(tolerance_ms,frame_ms): raise ValueError(f"event_without_near_frame:{event}")
        k=np.exp(-.5*(d/sigma)**2); k[d>cutoff]=0; k[i]=1
        y=np.maximum(y,k.astype(np.float32))
    return y
