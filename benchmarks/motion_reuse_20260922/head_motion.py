"""Extract and compare head-motion trajectories from talking-head videos.

Separates RIGID head motion (brows/eyes/nose-bridge/temples - not moved by speech)
from MOUTH motion (moved by speech), so we can ask two different questions:
  - does head motion change when the SEED changes at fixed audio?
  - does head motion change when the AUDIO changes at fixed seed?

Rigid pose is a 2D similarity transform (translation, rotation, scale) fitted to the
rigid landmark set, expressed relative to each clip's own first frame, and normalised
by inter-ocular distance so clips of different framing stay comparable.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# mediapipe FaceMesh indices
RIGID = [  # not displaced by speech
    33, 133, 362, 263,        # eye corners
    168, 6, 197, 195, 5, 4,   # nose bridge -> tip
    70, 63, 105, 66, 107,     # left brow
    336, 296, 334, 293, 300,  # right brow
    234, 454,                 # temples
    127, 356,                 # upper cheek / ear front
]
MOUTH = [
    61, 291, 0, 17, 13, 14, 78, 308, 82, 312, 87, 317, 37, 267, 40, 270,
]
LEFT_EYE, RIGHT_EYE = 33, 263


def landmarks_for_video(path, max_frames=None):
    import mediapipe as mp

    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(str(path))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out, misses, i = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and i >= max_frames):
            break
        res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            pts = np.array([[p.x * W, p.y * H] for p in lm], dtype=np.float64)
        else:
            pts = out[-1].copy() if out else np.full((478, 2), np.nan)
            misses += 1
        out.append(pts)
        i += 1
    cap.release()
    mesh.close()
    return np.stack(out), misses, (W, H)


def similarity_to_ref(src, dst):
    """Least-squares 2D similarity (scale, rotation, translation) mapping src -> dst."""
    sm, dm = src.mean(0), dst.mean(0)
    s0, d0 = src - sm, dst - dm
    var = (s0 ** 2).sum()
    if var < 1e-9:
        return 1.0, 0.0, np.zeros(2)
    # complex-number closed form
    a = (d0[:, 0] * s0[:, 0] + d0[:, 1] * s0[:, 1]).sum() / var
    b = (d0[:, 1] * s0[:, 0] - d0[:, 0] * s0[:, 1]).sum() / var
    scale = float(np.hypot(a, b))
    theta = float(np.arctan2(b, a))
    t = dm - scale * np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]]) @ sm
    return scale, theta, t


def trajectory(pts):
    """Per-frame rigid pose relative to frame 0, plus mouth shape in the head frame."""
    n = len(pts)
    ref = pts[0][RIGID]
    iod = np.linalg.norm(pts[0][LEFT_EYE] - pts[0][RIGHT_EYE])
    rig = np.zeros((n, 4))       # dx, dy (in IOD units), rotation deg, log scale
    mouth_local = np.zeros((n, len(MOUTH), 2))
    rigid_local = np.zeros((n, len(RIGID), 2))
    for i in range(n):
        cur = pts[i][RIGID]
        s, th, t = similarity_to_ref(ref, cur)
        rig[i] = [t[0] / iod, t[1] / iod, np.degrees(th), np.log(max(s, 1e-6))]
        # express mouth in the current head frame (removes head motion)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        inv = (pts[i][MOUTH] - t) @ R / max(s, 1e-6)
        mouth_local[i] = (inv - ref.mean(0)) / iod
        rigid_local[i] = (pts[i][RIGID] - pts[i][RIGID].mean(0)) / iod
    return {"rigid_pose": rig, "mouth_local": mouth_local, "rigid_local": rigid_local, "iod": float(iod)}


def compare(a, b, label_a, label_b):
    n = min(len(a["rigid_pose"]), len(b["rigid_pose"]))
    ra, rb = a["rigid_pose"][:n], b["rigid_pose"][:n]
    # head-motion trajectory difference, in inter-ocular units / degrees
    d_trans = np.linalg.norm(ra[:, :2] - rb[:, :2], axis=1)
    d_rot = np.abs(ra[:, 2] - rb[:, 2])
    d_scale = np.abs(ra[:, 3] - rb[:, 3])
    # correlation of each pose channel over time
    cors = []
    for c in range(4):
        x, y = ra[:, c], rb[:, c]
        if x.std() < 1e-9 or y.std() < 1e-9:
            cors.append(float("nan"))
        else:
            cors.append(float(np.corrcoef(x, y)[0, 1]))
    ma, mb = a["mouth_local"][:n], b["mouth_local"][:n]
    d_mouth = np.linalg.norm(ma - mb, axis=2).mean(1)
    return {
        "pair": f"{label_a} vs {label_b}",
        "frames": int(n),
        "head_translation_iod_mean": float(d_trans.mean()),
        "head_translation_iod_p95": float(np.percentile(d_trans, 95)),
        "head_rotation_deg_mean": float(d_rot.mean()),
        "head_rotation_deg_p95": float(np.percentile(d_rot, 95)),
        "head_logscale_mean": float(d_scale.mean()),
        "pose_corr_dx": cors[0], "pose_corr_dy": cors[1],
        "pose_corr_rot": cors[2], "pose_corr_scale": cors[3],
        "mouth_shape_iod_mean": float(d_mouth.mean()),
        "mouth_shape_iod_p95": float(np.percentile(d_mouth, 95)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True, help="label=path pairs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    trajs, meta = {}, {}
    for spec in args.videos:
        label, path = spec.split("=", 1)
        if not Path(path).exists():
            print(f"  skip {label}: missing {path}", flush=True)
            continue
        pts, misses, wh = landmarks_for_video(path, args.max_frames)
        trajs[label] = trajectory(pts)
        meta[label] = {"path": path, "frames": int(len(pts)), "detect_misses": int(misses),
                       "size": list(wh), "iod_px": trajs[label]["iod"]}
        rp = trajs[label]["rigid_pose"]
        print(f"  {label:24s} frames={len(pts):4d} misses={misses:3d} "
              f"head_range: dx={np.ptp(rp[:,0]):.3f} dy={np.ptp(rp[:,1]):.3f} "
              f"rot={np.ptp(rp[:,2]):.2f}deg scale={np.ptp(np.exp(rp[:,3])):.3f}", flush=True)

    labels = list(trajs)
    comps = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            comps.append(compare(trajs[labels[i]], trajs[labels[j]], labels[i], labels[j]))

    json.dump({"meta": meta, "comparisons": comps}, open(args.out, "w"), indent=2)
    np.savez(args.out.replace(".json", "_traj.npz"),
             **{f"{k}_pose": v["rigid_pose"] for k, v in trajs.items()})

    print(f"\n{'pair':46s} {'headTrans':>9s} {'headRot°':>8s} {'corr_dy':>8s} {'corr_rot':>8s} {'mouth':>8s}")
    for c in comps:
        print(f"{c['pair']:46s} {c['head_translation_iod_mean']:9.4f} "
              f"{c['head_rotation_deg_mean']:8.3f} {c['pose_corr_dy']:8.3f} "
              f"{c['pose_corr_rot']:8.3f} {c['mouth_shape_iod_mean']:8.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
