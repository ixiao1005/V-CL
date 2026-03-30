import os
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import mediapipe as mp

cv2.setUseOptimized(True)
cv2.ocl.setUseOpenCL(True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308, 82, 312]
LEFT_BROW = 336
RIGHT_BROW = 107

POSE_LM = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}

VIDEO_EXTENSIONS = (
    ".mp4", ".avi", ".mov", ".mkv", ".flv",
    ".wmv", ".webm", ".mpg", ".mpeg"
)

mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

def euclidean(p1, p2):
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    return float(np.linalg.norm(p1 - p2))

def calc_ear(pts):
    v1 = euclidean(pts[1], pts[5])
    v2 = euclidean(pts[2], pts[4])
    h = euclidean(pts[0], pts[3])
    return (v1 + v2) / (2.0 * h + 1e-6)

def calc_mar(pts):
    v1 = euclidean(pts[0], pts[1])
    h = euclidean(pts[2], pts[3])
    return v1 / (h + 1e-6)

def calc_frown(face_lm, w, h):
    p1 = np.array([face_lm[LEFT_BROW].x * w, face_lm[LEFT_BROW].y * h], dtype=float)
    p2 = np.array([face_lm[RIGHT_BROW].x * w, face_lm[RIGHT_BROW].y * h], dtype=float)
    return euclidean(p1, p2)

def landmark_to_xy(landmark, w, h):
    return np.array([landmark.x * w, landmark.y * h], dtype=float)

def midpoint(a, b):
    return (a + b) / 2.0

def calc_motion(prev_gray, curr_gray):
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(np.mean(diff))

def jfloat(x):
    if x is None:
        return None
    try:
        if np.isnan(x):
            return None
        return float(x)
    except Exception:
        return None

def calc_head_pose(face_lm, w, h):
    try:
        image_points = np.array([
            [face_lm[33].x * w,  face_lm[33].y * h],
            [face_lm[263].x * w, face_lm[263].y * h],
            [face_lm[1].x * w,   face_lm[1].y * h],
            [face_lm[61].x * w,  face_lm[61].y * h],
            [face_lm[291].x * w, face_lm[291].y * h],
            [face_lm[199].x * w, face_lm[199].y * h],
        ], dtype="double")

        model_points = np.array([
            [-30, -30, 30],
            [30, -30, 30],
            [0, 0, 60],
            [-25, 20, 20],
            [25, 20, 20],
            [0, 50, 0],
        ], dtype="double")

        focal = float(w)
        center = (w / 2.0, h / 2.0)
        camera = np.array([
            [focal, 0, center[0]],
            [0, focal, center[1]],
            [0, 0, 1]
        ], dtype="double")

        ok, rvec, _ = cv2.solvePnP(
            model_points,
            image_points,
            camera,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return np.nan, np.nan, np.nan

        rot_mat, _ = cv2.Rodrigues(rvec)
        r = R.from_matrix(rot_mat)
        pitch, yaw, roll = r.as_euler("xyz", degrees=True)
        return float(pitch), float(yaw), float(roll)

    except Exception:
        return np.nan, np.nan, np.nan

def torso_pitch_deg(left_sh, right_sh, left_hip, right_hip):
    mid_sh = midpoint(left_sh, right_sh)
    mid_hip = midpoint(left_hip, right_hip)
    v = (mid_hip - mid_sh).astype(float)
    v_norm = np.linalg.norm(v) + 1e-8
    v = v / v_norm
    vertical = np.array([0.0, 1.0], dtype=float)
    cosv = float(np.clip(np.dot(v, vertical), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosv)))

def torso_roll_signed_deg(left_sh, right_sh):
    v = (right_sh - left_sh).astype(float)
    dx, dy = float(v[0]), float(v[1])
    ang = np.degrees(np.arctan2(dy, dx))
    return float(ang)

def load_progress(progress_file):
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "done_success": data.get("done_success", []),
                "done_fail": data.get("done_fail", []),
                "errors": data.get("errors", {}),
            }
        except Exception as e:
            print(f"[WARN] Failed to load progress file: {e}")

    return {
        "done_success": [],
        "done_fail": [],
        "errors": {},
    }

def save_progress(progress_file, data):
    try:
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[WARN] Failed to save progress file: {e}")
        return False

def extract_features_to_jsonl(
    video_path,
    jsonl_dir,
    downsample_step=1,
    model_complexity=1,
    max_num_hands=2,
    overwrite_jsonl=False
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "failed", "cannot_open_video"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    basename = os.path.basename(video_path)
    stem = os.path.splitext(basename)[0]
    os.makedirs(jsonl_dir, exist_ok=True)
    jsonl_path = os.path.join(jsonl_dir, f"{stem}.frames.jsonl")

    if (not overwrite_jsonl) and os.path.exists(jsonl_path) and os.path.getsize(jsonl_path) > 0:
        cap.release()
        return jsonl_path, "success", None

    prev_gray = None
    prev_lw = None
    prev_rw = None

    with mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True
    ) as face_mesh, mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False
    ) as pose, mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_num_hands
    ) as hands:
        with open(jsonl_path, "w", encoding="utf-8") as jf, \
             tqdm(total=total_frames, desc=f"Frames {basename}", unit="frame", leave=False) as pbar:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                pbar.update(1)
                if downsample_step > 1 and (frame_idx % downsample_step != 0):
                    frame_idx += 1
                    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    continue
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_res = face_mesh.process(rgb)
                pose_res = pose.process(rgb)
                hands_res = hands.process(rgb)
                face_valid = 0
                ear = mar = frown = pitch = yaw = roll = None
                nose_xy = None
                if face_res.multi_face_landmarks:
                    face_valid = 1
                    flm = face_res.multi_face_landmarks[0].landmark
                    lpts = np.array([[flm[i].x * w, flm[i].y * h] for i in LEFT_EYE], dtype=float)
                    rpts = np.array([[flm[i].x * w, flm[i].y * h] for i in RIGHT_EYE], dtype=float)
                    try:
                        ear = float((calc_ear(lpts) + calc_ear(rpts)) / 2.0)
                    except Exception:
                        ear = None
                    mpts = np.array([[flm[i].x * w, flm[i].y * h] for i in MOUTH], dtype=float)
                    try:
                        mar = float(calc_mar(mpts))
                    except Exception:
                        mar = None
                    try:
                        frown = float(calc_frown(flm, w, h))
                    except Exception:
                        frown = None
                    p_, y_, r_ = calc_head_pose(flm, w, h)
                    pitch = None if np.isnan(p_) else float(p_)
                    yaw = None if np.isnan(y_) else float(y_)
                    roll = None if np.isnan(r_) else float(r_)
                    nose_xy = np.array([flm[1].x * w, flm[1].y * h], dtype=float)

                pose_valid = 0
                torso_pitch = torso_roll = shoulder_dist = None
                left_wrist_to_face = right_wrist_to_face = None
                hand_motion = None
                if pose_res.pose_landmarks:
                    pose_valid = 1
                    plm = pose_res.pose_landmarks.landmark
                    left_sh = landmark_to_xy(plm[POSE_LM["left_shoulder"]], w, h)
                    right_sh = landmark_to_xy(plm[POSE_LM["right_shoulder"]], w, h)
                    left_hip = landmark_to_xy(plm[POSE_LM["left_hip"]], w, h)
                    right_hip = landmark_to_xy(plm[POSE_LM["right_hip"]], w, h)
                    torso_pitch = float(torso_pitch_deg(left_sh, right_sh, left_hip, right_hip))
                    torso_roll = float(torso_roll_signed_deg(left_sh, right_sh))
                    shoulder_dist = float(euclidean(left_sh, right_sh))
                    lw = landmark_to_xy(plm[POSE_LM["left_wrist"]], w, h)
                    rw = landmark_to_xy(plm[POSE_LM["right_wrist"]], w, h)
                    if nose_xy is not None:
                        left_wrist_to_face = float(euclidean(lw, nose_xy))
                        right_wrist_to_face = float(euclidean(rw, nose_xy))
                    spd = 0.0
                    if prev_lw is not None:
                        spd += euclidean(lw, prev_lw)
                    if prev_rw is not None:
                        spd += euclidean(rw, prev_rw)
                    hand_motion = float(spd)
                    prev_lw = lw
                    prev_rw = rw
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                body_motion = None
                if prev_gray is not None:
                    body_motion = calc_motion(prev_gray, gray)
                prev_gray = gray
                hands_valid = 0
                hand_count = 0
                left_hand_pts = []
                right_hand_pts = []
                if hands_res.multi_hand_landmarks:
                    hands_valid = 1
                    hand_count = len(hands_res.multi_hand_landmarks)
                    for idx, hand_lm in enumerate(hands_res.multi_hand_landmarks):
                        pts = [[lm.x * w, lm.y * h, lm.z * w] for lm in hand_lm.landmark]
                        label = None
                        if hands_res.multi_handedness and idx < len(hands_res.multi_handedness):
                            try:
                                label = hands_res.multi_handedness[idx].classification[0].label
                            except Exception:
                                label = None

                        if label == "Left":
                            left_hand_pts = pts
                        elif label == "Right":
                            right_hand_pts = pts
                rec = {
                    "video": basename,
                    "frame": int(frame_idx),
                    "t_sec": (float(frame_idx) / fps) if fps > 0 else None,
                    "w": int(w),
                    "h": int(h),
                    "face_valid": int(face_valid),
                    "pose_valid": int(pose_valid),
                    "hands_valid": int(hands_valid),
                    "ear": jfloat(ear),
                    "mar": jfloat(mar),
                    "frown": jfloat(frown),
                    "pitch": jfloat(pitch),
                    "yaw": jfloat(yaw),
                    "roll": jfloat(roll),
                    "torso_pitch": jfloat(torso_pitch),
                    "torso_roll": jfloat(torso_roll),
                    "shoulder_dist": jfloat(shoulder_dist),
                    "left_wrist_to_face": jfloat(left_wrist_to_face),
                    "right_wrist_to_face": jfloat(right_wrist_to_face),
                    "hand_motion": jfloat(hand_motion),
                    "body_motion": jfloat(body_motion),
                    "hand_count": int(hand_count),
                    "left_hand_keypoints": left_hand_pts,
                    "right_hand_keypoints": right_hand_pts,
                }
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                frame_idx += 1
    cap.release()
    return jsonl_path, "success", None

def process_folder_with_checkpoint(
    video_folder,
    jsonl_dir,
    checkpoint_path="progress.json",
    checkpoint_interval=5,
    resume=True,
    downsample_step=1,
    overwrite_jsonl=False,
    retry_failed=True,
    model_complexity=1,
    max_num_hands=2
):
    os.makedirs(jsonl_dir, exist_ok=True)
    done_success = set()
    done_fail = set()
    errors = {}
    if resume and os.path.exists(checkpoint_path):
        prog = load_progress(checkpoint_path)
        done_success = set(prog.get("done_success", []))
        done_fail = set(prog.get("done_fail", []))
        errors = prog.get("errors", {}) or {}
        print(f"[INFO] Resume from checkpoint: success={len(done_success)} fail={len(done_fail)}")
    skip_set = set(done_success)
    if not retry_failed:
        skip_set |= set(done_fail)
    all_videos = sorted(
        f for f in os.listdir(video_folder)
        if f.lower().endswith(VIDEO_EXTENSIONS)
    )
    remaining = [v for v in all_videos if v not in skip_set]
    print("[INFO] OpenCV optimized:", cv2.useOptimized())
    print("[INFO] OpenCV OpenCL enabled:", cv2.ocl.useOpenCL())
    print("[INFO] Total videos:", len(all_videos))
    print("[INFO] Done success:", len(done_success))
    print("[INFO] Done fail:", len(done_fail))
    print("[INFO] Remaining:", len(remaining))
    if not remaining:
        print("[INFO] Nothing to do.")
        return
    base_done = len(done_success) + len(done_fail)
    for idx, video_name in enumerate(remaining, start=1):
        current_num = base_done + idx
        print(f"\n[{current_num}/{len(all_videos)}] Processing: {video_name}")
        video_path = os.path.join(video_folder, video_name)
        try:
            _, status, err_msg = extract_features_to_jsonl(
                video_path=video_path,
                jsonl_dir=jsonl_dir,
                downsample_step=downsample_step,
                model_complexity=model_complexity,
                max_num_hands=max_num_hands,
                overwrite_jsonl=overwrite_jsonl
            )
        except Exception as e:
            status = "failed"
            err_msg = str(e)
        if status == "success":
            done_success.add(video_name)
            if video_name in done_fail:
                done_fail.remove(video_name)
                errors.pop(video_name, None)
            print("  [OK] success")
        else:
            done_fail.add(video_name)
            errors[video_name] = err_msg or "unknown_error"
            print(f"  [FAIL] {errors[video_name]}")
        should_save = (idx % checkpoint_interval == 0) or (idx == len(remaining))
        if should_save:
            prog_data = {
                "done_success": sorted(done_success),
                "done_fail": sorted(done_fail),
                "errors": errors,
            }
            save_progress(checkpoint_path, prog_data)
            print(
                f"  [INFO] checkpoint saved: "
                f"success={len(done_success)} fail={len(done_fail)} total={len(all_videos)}"
            )
    final_data = {
        "done_success": sorted(done_success),
        "done_fail": sorted(done_fail),
        "errors": errors,
    }
    save_progress(checkpoint_path, final_data)
    print("\n[INFO] Finished.")
    print("[INFO] Progress file:", os.path.abspath(checkpoint_path))
    print("[INFO] JSONL output dir:", os.path.abspath(jsonl_dir))

def build_argparser():
    parser = argparse.ArgumentParser(
        description="Extract per-frame MediaPipe features from a folder of videos into JSONL files."
    )
    parser.add_argument("--video_folder", type=str, required=True, help="Path to input video folder.")
    parser.add_argument("--jsonl_dir", type=str, required=True, help="Path to output JSONL folder.")
    parser.add_argument("--checkpoint_path", type=str, default="progress.json", help="Checkpoint JSON path.")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="Save checkpoint every N videos.")
    parser.add_argument("--downsample_step", type=int, default=1, help="Process every N-th frame.")
    parser.add_argument("--model_complexity", type=int, default=1, choices=[0, 1, 2], help="MediaPipe pose model complexity.")
    parser.add_argument("--max_num_hands", type=int, default=2, help="Maximum number of hands to detect.")
    parser.add_argument("--overwrite_jsonl", action="store_true", help="Overwrite existing JSONL files.")
    parser.add_argument("--no_resume", action="store_true", help="Do not resume from checkpoint.")
    parser.add_argument("--no_retry_failed", action="store_true", help="Do not retry previously failed videos.")
    return parser

def main():
    parser = build_argparser()
    args = parser.parse_args()
    process_folder_with_checkpoint(
        video_folder=args.video_folder,
        jsonl_dir=args.jsonl_dir,
        checkpoint_path=args.checkpoint_path,
        checkpoint_interval=args.checkpoint_interval,
        resume=not args.no_resume,
        downsample_step=args.downsample_step,
        overwrite_jsonl=args.overwrite_jsonl,
        retry_failed=not args.no_retry_failed,
        model_complexity=args.model_complexity,
        max_num_hands=args.max_num_hands,
    )

if __name__ == "__main__":
    main()