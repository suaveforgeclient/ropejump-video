#!/usr/bin/env python3
import argparse
import json
import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def estimate_vertical_shift(prev, curr, width, height, x0f, x1f, y0f, y1f, max_shift, step=2):
    x0 = int(math.floor(width * x0f))
    x1 = int(math.ceil(width * x1f))
    y0 = int(math.floor(height * y0f))
    y1 = int(math.ceil(height * y1f))
    raw_energy = 0.0
    energy_n = 0
    for y in range(y0, y1, step):
        for x in range(x0, x1, step):
            raw_energy += abs(int(curr[y, x]) - int(prev[y, x]))
            energy_n += 1

    scores = []
    for shift in range(-max_shift, max_shift + 1):
        sad = 0.0
        n = 0
        for y in range(y0, y1, step):
            py = y - shift
            if py < y0 or py >= y1:
                continue
            for x in range(x0, x1, step):
                sad += abs(int(curr[y, x]) - int(prev[py, x]))
                n += 1
        scores.append((sad / n if n else 255.0, shift))
    scores.sort(key=lambda item: item[0])
    best_score, best_shift = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else best_score
    confidence = clamp((second_score - best_score) / max(4.0, second_score), 0.0, 1.0)
    energy = raw_energy / energy_n / 255.0 if energy_n else 0.0
    return {"shift": best_shift, "confidence": confidence, "energy": energy}


def motion_from_frame(frame, prev_gray):
    width, height = 96, 144
    tiny = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    b, g, r = cv2.split(tiny)
    gray = ((r.astype(np.uint16) * 54 + g.astype(np.uint16) * 183 + b.astype(np.uint16) * 19) >> 8).astype(np.uint8)
    mean = float(gray.mean())
    variance = float(gray.astype(np.float32).var())
    dark_fraction = float(np.mean(gray < 48))
    lighting = {
        "meanLuma": mean / 255.0,
        "contrast": math.sqrt(max(0.0, variance)) / 255.0,
        "darkFraction": dark_fraction,
    }
    if prev_gray is None:
        return ({
            "bodyDyNorm": 0.0, "bodyConfidence": 0.0, "bodyEnergy": 0.0,
            "cameraDyNorm": 0.0, "cameraConfidence": 0.0, "backgroundEnergy": 0.0,
            **lighting,
        }, gray)

    body = estimate_vertical_shift(prev_gray, gray, width, height, 0.22, 0.78, 0.12, 0.94, 7)
    left = estimate_vertical_shift(prev_gray, gray, width, height, 0.01, 0.18, 0.12, 0.94, 5)
    right = estimate_vertical_shift(prev_gray, gray, width, height, 0.82, 0.99, 0.12, 0.94, 5)
    bg_confidence = max(left["confidence"], right["confidence"])
    camera_shift_px = 0.0
    camera_confidence = 0.0
    if left["confidence"] > 0.18 and right["confidence"] > 0.18 and abs(left["shift"] - right["shift"]) <= 1:
        camera_shift_px = (left["shift"] + right["shift"]) / 2.0
        camera_confidence = min(left["confidence"], right["confidence"])
    elif bg_confidence > 0.35:
        best = left if left["confidence"] >= right["confidence"] else right
        camera_shift_px = best["shift"]
        camera_confidence = best["confidence"] * 0.65

    return ({
        "bodyDyNorm": body["shift"] / height,
        "bodyConfidence": body["confidence"],
        "bodyEnergy": body["energy"],
        "cameraDyNorm": camera_shift_px / height,
        "cameraConfidence": camera_confidence,
        "backgroundEnergy": (left["energy"] + right["energy"]) / 2.0,
        **lighting,
    }, gray)


def landmark_json(point):
    return {
        "x": float(point.x),
        "y": float(point.y),
        "z": float(point.z),
        "visibility": float(getattr(point, "visibility", 1.0)),
        "presence": float(getattr(point, "presence", 1.0)),
    }


def apply_stress(frame, args, index):
    h, w = frame.shape[:2]
    out = frame

    crop = clamp(float(args.stress_crop), 0.0, 0.30)
    if crop > 0:
        x_pad = int(round(w * crop))
        y_pad = int(round(h * crop * 0.55))
        if w - 2 * x_pad >= 64 and h - 2 * y_pad >= 64:
            out = out[y_pad:h-y_pad, x_pad:w-x_pad]
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)

    scale = clamp(float(args.stress_scale), 0.20, 1.0)
    if scale < 0.999:
        sw = max(64, int(round(w * scale)))
        sh = max(64, int(round(h * scale)))
        small = cv2.resize(out, (sw, sh), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    perspective = clamp(float(args.stress_perspective), 0.0, 0.20)
    if perspective > 0:
        dx = float(w) * perspective
        src = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
        dst = np.float32([[dx, 0], [w - 1, 0], [0, h - 1], [w - 1 - dx, h - 1]])
        matrix = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(out, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    base_shift_x = clamp(float(args.stress_shift_x), -0.20, 0.20)
    shake = clamp(float(args.stress_shake), 0.0, 0.08)
    if abs(base_shift_x) > 1e-6 or shake > 0:
        dx = int(round(w * (base_shift_x + math.sin(index * 0.47) * shake)))
        dy = int(round(h * math.cos(index * 0.39) * shake * 0.55))
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        out = cv2.warpAffine(out, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    brightness = clamp(float(args.stress_brightness), 0.15, 1.40)
    if abs(brightness - 1.0) > 1e-6:
        out = np.clip(out.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

    blur = max(0, int(args.stress_blur))
    if blur > 1:
        kernel = blur if blur % 2 == 1 else blur + 1
        kernel = min(kernel, 21)
        out = cv2.GaussianBlur(out, (kernel, kernel), 0)

    occlusion = clamp(float(args.stress_occlusion), 0.0, 0.35)
    if occlusion > 0 and (index % 31) in range(8, 16):
        box_w = max(12, int(round(w * min(0.34, occlusion * 1.7))))
        box_h = max(12, int(round(h * min(0.30, occlusion * 1.35))))
        x0 = max(0, int(w * 0.50 - box_w * 0.50))
        y0 = max(0, int(h * 0.62))
        out = out.copy()
        out[y0:min(h, y0 + box_h), x0:min(w, x0 + box_w)] = 0

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stress-brightness", type=float, default=1.0)
    parser.add_argument("--stress-blur", type=int, default=0)
    parser.add_argument("--stress-scale", type=float, default=1.0)
    parser.add_argument("--stress-crop", type=float, default=0.0)
    parser.add_argument("--stress-shift-x", type=float, default=0.0)
    parser.add_argument("--stress-shake", type=float, default=0.0)
    parser.add_argument("--stress-occlusion", type=float, default=0.0)
    parser.add_argument("--stress-perspective", type=float, default=0.0)
    parser.add_argument("--analysis-fps", type=float, default=0.0)
    parser.add_argument("--analysis-phase", type=float, default=0.0)
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        raise SystemExit(f"video_missing:{video_path}")
    model_path = Path(args.model)
    if not model_path.is_file():
        raise SystemExit(f"model_missing:{model_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit("video_open_failed")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.30,
        min_tracking_confidence=0.30,
        output_segmentation_masks=False,
    )

    rows = []
    prev_gray = None
    last_motion = None
    index = 0
    started = time.perf_counter()
    last_ms = -1
    pose_frames = 0
    missing_frames = 0
    last_analysis_gray_ms = -1000000000
    analysis_fps = max(0.0, float(args.analysis_fps))
    analysis_period_ms = (1000.0 / analysis_fps) if analysis_fps > 0 else 0.0
    analysis_phase = clamp(float(args.analysis_phase), 0.0, 0.999999)
    next_analysis_ms = analysis_period_ms * analysis_phase if analysis_period_ms > 0 else 0.0
    source_frames = 0

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            source_index = source_frames
            source_frames += 1
            media_ms_raw = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if not math.isfinite(media_ms_raw) or media_ms_raw < 0 or (source_index > 0 and media_ms_raw <= last_ms):
                media_ms_raw = source_index * 1000.0 / fps if fps > 0 else source_index * (1000.0 / 30.0)
            media_ms = int(round(media_ms_raw))
            if media_ms <= last_ms:
                media_ms = last_ms + max(1, int(round(1000.0 / fps))) if fps > 0 else last_ms + 33
            last_ms = media_ms

            # Browser parity: requestVideoFrameCallback cannot deliver every decoded frame
            # while synchronous MediaPipe inference is blocking the callback. When a captured
            # session records its observed A cadence, discard source frames BEFORE MediaPipe
            # so the tracker/filter sees the same sparse input sequence rather than a 60fps
            # sequence that is sampled only after inference.
            if analysis_period_ms > 0:
                if media_ms + 0.001 < next_analysis_ms:
                    continue
                while next_analysis_ms <= media_ms:
                    next_analysis_ms += analysis_period_ms

            frame = apply_stress(frame, args, source_index)

            # Match the browser FrameMotionEstimator cadence: optical motion is updated
            # at most once per ~62 ms and the last result is reused between updates.
            analysis_gray = None
            analysis_frame_at = None
            if last_motion is None or media_ms - last_analysis_gray_ms >= 62:
                last_motion, prev_gray = motion_from_frame(frame, prev_gray)
                analysis_gray = prev_gray.reshape(-1).tolist()
                analysis_frame_at = media_ms
                last_analysis_gray_ms = media_ms
            motion = last_motion

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, media_ms)
            landmarks = None
            if result.pose_landmarks:
                landmarks = [landmark_json(point) for point in result.pose_landmarks[0]]
                pose_frames += 1
            else:
                missing_frames += 1
            rows.append({"frame": source_index, "mediaTimeMs": media_ms, "landmarks": landmarks, "motion": motion, "analysisGray": analysis_gray, "analysisWidth": 96 if analysis_gray is not None else 0, "analysisHeight": 144 if analysis_gray is not None else 0, "analysisFrameAt": analysis_frame_at})

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    duration_ms = rows[-1]["mediaTimeMs"] if rows else 0
    payload = {
        "video": str(video_path),
        "fps": analysis_fps if analysis_fps > 0 else fps,
        "inputVideoFps": fps,
        "analysisFps": analysis_fps if analysis_fps > 0 else fps,
        "analysisPhase": analysis_phase if analysis_fps > 0 else 0.0,
        "frameCountHint": frame_count_hint,
        "sourceFrames": source_frames,
        "frames": len(rows),
        "width": width,
        "height": height,
        "durationMs": duration_ms,
        "poseFrames": pose_frames,
        "missingPoseFrames": missing_frames,
        "processingMs": elapsed_ms,
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ["fps", "frames", "durationMs", "poseFrames", "missingPoseFrames", "processingMs"]}))


if __name__ == "__main__":
    main()
