import os
import glob
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

MP_NUM_COLS = [
    "ear", "mar", "frown", "pitch", "yaw", "roll",
    "torso_pitch", "torso_roll", "shoulder_dist",
    "left_wrist_to_face", "right_wrist_to_face",
    "hand_motion", "body_motion", "hand_count"
]

STAT_KEYS = [
    "mean", "std", "min", "max", "range",
    "p10", "p25", "p50", "p75", "p90",
    "diff_mean", "diff_std"
]

def participant_id_from_jsonl(path):
    base = os.path.splitext(os.path.basename(path))[0]
    base = base.replace(".frames", "")
    return base.split("-")[0]

def robust_stats(arr, prefix):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]

    if arr.size == 0:
        return {f"{prefix}_{k}": np.nan for k in STAT_KEYS}

    d = np.diff(arr) if arr.size > 1 else np.array([np.nan])

    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_range": float(np.max(arr) - np.min(arr)),
        f"{prefix}_p10": float(np.quantile(arr, 0.10)),
        f"{prefix}_p25": float(np.quantile(arr, 0.25)),
        f"{prefix}_p50": float(np.quantile(arr, 0.50)),
        f"{prefix}_p75": float(np.quantile(arr, 0.75)),
        f"{prefix}_p90": float(np.quantile(arr, 0.90)),
        f"{prefix}_diff_mean": float(np.nanmean(d)),
        f"{prefix}_diff_std": float(np.nanstd(d)),
    }

def summarize_mp_jsonl_one(jsonl_path, downsample_step=1):
    col_buf = {c: [] for c in MP_NUM_COLS}
    frames_used = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if downsample_step is not None and i % downsample_step != 0:
                continue

            obj = json.loads(line)
            frames_used += 1

            for c in MP_NUM_COLS:
                v = obj.get(c, None)
                col_buf[c].append(np.nan if v is None else v)

    feats = {
        "participant_id": participant_id_from_jsonl(jsonl_path),
        "mp_frames_ds": frames_used
    }

    if frames_used == 0:
        return feats

    for c in MP_NUM_COLS:
        feats.update(robust_stats(col_buf[c], f"mp_{c}"))

    ear = np.asarray(col_buf.get("ear", []), dtype=float)
    ear = ear[~np.isnan(ear)]
    if ear.size > 10:
        thr = np.quantile(ear, 0.10)
        feats["mp_blink_like_rate"] = float(np.mean(ear < thr))

    yaw = np.asarray(col_buf.get("yaw", []), dtype=float)
    yaw = yaw[~np.isnan(yaw)]
    if yaw.size > 10:
        thr = np.quantile(np.abs(yaw), 0.90)
        feats["mp_head_turn_rate"] = float(np.mean(np.abs(yaw) > thr))

    bm = np.asarray(col_buf.get("body_motion", []), dtype=float)
    bm = bm[~np.isnan(bm)]
    if bm.size > 10:
        thr = np.quantile(bm, 0.10)
        feats["mp_stillness_rate"] = float(np.mean(bm < thr))

    return feats

def build_mp_frame_video_features(jsonl_dir, out_csv="mp_frame_video_features.csv", downsample_step=1):
    files = sorted(glob.glob(os.path.join(str(jsonl_dir), "*.jsonl")))
    rows = []

    for p in tqdm(files, desc="MediaPipe JSONL -> video features", unit="video"):
        try:
            rows.append(summarize_mp_jsonl_one(p, downsample_step=downsample_step))
        except Exception as e:
            print(f"\n[WARN] failed: {p} {e}")

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("saved:", out_csv, "shape:", out.shape)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate frame-level MediaPipe JSONL features into video-level CSV features."
    )
    parser.add_argument(
        "--jsonl_dir",
        type=str,
        required=True,
        help="Directory containing per-frame JSONL files."
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="mp_frame_video_features.csv",
        help="Path to the output CSV file."
    )
    parser.add_argument(
        "--downsample_step",
        type=int,
        default=1,
        help="Frame downsampling step. Use 1 to keep all frames."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    jsonl_dir = Path(args.jsonl_dir)
    if not jsonl_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {jsonl_dir}")
    if args.downsample_step <= 0:
        raise ValueError("--downsample_step must be a positive integer.")
    build_mp_frame_video_features(
        jsonl_dir=jsonl_dir,
        out_csv=args.out_csv,
        downsample_step=args.downsample_step
    )


if __name__ == "__main__":
    main()