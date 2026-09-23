"""Decompose SoulX head motion into audio-driven vs seed-driven components."""
import json, wave
from pathlib import Path
import numpy as np

SP = Path("/tmp/claude-0/-workspace/e3b6c3cf-c47c-4de6-9013-fbab0c27bedf/scratchpad")
CH = ["dx", "dy", "rot_deg", "log_scale"]

z = np.load(SP / "motion_seeds_traj.npz")
seed_keys = sorted(k for k in z.files if k.startswith("s4_seed"))
print("runs (steps=4, same audio 'indian150-a', same reference image):")
for k in seed_keys:
    print(f"  {k}  shape={z[k].shape}")

P = np.stack([z[k] for k in seed_keys])
S, T, C = P.shape
print(f"\nS={S} seeds, T={T} frames, C={C} pose channels\n")

mean_traj = P.mean(0)
resid = P - mean_traj[None]

print(f"{'channel':10s} {'var(shared)':>12s} {'var(resid)':>11s} {'shared frac':>12s} {'mean|resid|':>12s}")
rows = {}
for c in range(C):
    v_shared = float(mean_traj[:, c].var())
    v_resid = float(resid[:, :, c].var())
    frac = v_shared / (v_shared + v_resid) if (v_shared + v_resid) > 0 else float("nan")
    rows[CH[c]] = {"var_shared": v_shared, "var_residual": v_resid,
                   "shared_fraction": frac,
                   "mean_abs_residual": float(np.abs(resid[:, :, c]).mean())}
    print(f"{CH[c]:10s} {v_shared:12.6f} {v_resid:11.6f} {frac*100:11.1f}% {np.abs(resid[:, :, c]).mean():12.5f}")

print("\n--- head motion vs audio envelope ---")
env_corr = {}
try:
    apath = Path("/workspace/SoulX-FlashHead/benchmarks/pro_lite_150x_20260917/audio.wav")
    with wave.open(str(apath), "rb") as w:
        sr, n = w.getframerate(), w.getnframes()
        raw = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float64)
    print(f"  audio: {apath.name} sr={sr} dur={len(raw)/sr:.2f}s")
    hop = sr / 25.0
    env = np.array([np.sqrt((raw[int(i*hop):int((i+1)*hop)]**2).mean() + 1e-12) for i in range(T)])
    env = (env - env.mean()) / (env.std() + 1e-12)
    def best_lag(sig):
        s = (sig - sig.mean()) / (sig.std() + 1e-12)
        cands = []
        for l in range(-8, 9):
            a = env[max(0,l):T+min(0,l)]; b = s[max(0,-l):T+min(0,-l)]
            if len(a) > 10: cands.append((float(np.corrcoef(a,b)[0,1]), l))
        return max(cands, key=lambda p: abs(p[0]))
    for c in range(C):
        r, l = best_lag(np.abs(np.gradient(mean_traj[:, c])))
        env_corr[CH[c]] = {"best_corr": r, "lag_frames": l}
        print(f"  |d{CH[c]}/dt| vs audio envelope: r={r:+.3f} at lag {l:+d} frames")
    r, l = best_lag(np.linalg.norm(np.gradient(mean_traj[:, :2], axis=0), axis=1))
    env_corr["head_speed"] = {"best_corr": r, "lag_frames": l}
    print(f"  head translation SPEED vs audio envelope: r={r:+.3f} at lag {l:+d} frames")
except Exception as e:
    print(f"  envelope correlation failed: {type(e).__name__}: {e}")

print("\n--- seed 50 across a denoise-schedule change (4 steps -> 2 steps) ---")
if "s4_seed50_pose" in z.files and "s2_seed50_pose" in z.files:
    a, b = z["s4_seed50_pose"], z["s2_seed50_pose"]
    n = min(len(a), len(b))
    d = np.linalg.norm(a[:n,:2] - b[:n,:2], axis=1)
    cross = [np.linalg.norm(P[i,:,:2]-P[j,:,:2],axis=1).mean() for i in range(S) for j in range(i+1,S)]
    print(f"  seed50 @4steps vs seed50 @2steps  : mean head-trans diff = {d.mean():.4f} IOD")
    print(f"  typical DIFFERENT-seed pair @4steps: mean head-trans diff = {np.mean(cross):.4f} IOD")
    print(f"  ratio = {d.mean()/np.mean(cross):.2f}x")
    rows["_config_vs_seed"] = {"seed50_4step_vs_2step_iod": float(d.mean()),
                               "typical_cross_seed_iod": float(np.mean(cross)),
                               "ratio": float(d.mean()/np.mean(cross))}

json.dump({"channels": rows, "envelope_correlation": env_corr,
           "seeds_compared": seed_keys, "frames": int(T)},
          open(SP/"motion_decomposition.json","w"), indent=2)
print(f"\nwrote {SP/'motion_decomposition.json'}")
