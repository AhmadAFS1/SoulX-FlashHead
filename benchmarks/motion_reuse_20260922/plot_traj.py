import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
SP = Path("/tmp/claude-0/-workspace/e3b6c3cf-c47c-4de6-9013-fbab0c27bedf/scratchpad")
z = np.load(SP/"motion_seeds_traj.npz")
zr = np.load(SP/"motion_repeat_traj.npz")
t = np.arange(250)/25.0
fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
cols = {"s4_seed0_pose":"#6699cc","s4_seed1_pose":"#88bb77","s4_seed50_pose":"#dd4444","s4_seed51_pose":"#cc9933"}
for k,c in cols.items():
    p = z[k]; lab = k.replace("s4_","").replace("_pose","")
    lw = 2.4 if "50" in lab else 1.3
    ax[0].plot(t, p[:,2], color=c, lw=lw, label=f"{lab} (range {np.ptp(p[:,2]):.1f}°)")
    ax[1].plot(t, p[:,1], color=c, lw=lw, label=lab)
P = np.stack([z[k] for k in cols])
ax[0].plot(t, P.mean(0)[:,2], "k--", lw=2.0, label="mean = audio-driven component")
ax[0].set_ylabel("head roll (deg)"); ax[0].legend(fontsize=8, ncol=3, loc="upper right")
ax[0].set_title("SoulX-FlashHead PRO — same audio + same reference image, seed varied\n"
                "shared curve = audio-driven; spread = seed-driven", fontsize=11)
ax[1].plot(t, P.mean(0)[:,1], "k--", lw=2.0)
ax[1].set_ylabel("head dy (inter-ocular units)")
for k,c,lab in [("ctl_r01_pose","#222222","seed50 ctl r01"),("ctl_r02_pose","#dd4444","seed50 ctl r02 (exact repeat)"),
                ("prealloc_pose","#3388bb","seed50 prealloc arm"),("classic_pose","#bb8833","seed50 classic runtime")]:
    if k in zr.files:
        p = zr[k]; ax[2].plot(np.arange(len(p))/25.0, p[:,2], color=c, lw=2.2 if "r02" in k else 1.4,
                              ls="--" if "r02" in k else "-", label=lab)
ax[2].set_ylabel("head roll (deg)"); ax[2].set_xlabel("time (s)")
ax[2].legend(fontsize=8, loc="upper right")
ax[2].set_title("Seed 50 held fixed, runtime/config varied — r02 overlays r01 exactly; other arms diverge", fontsize=10)
for a in ax: a.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(SP/"motion_trajectories.png", dpi=125)
print("wrote", SP/"motion_trajectories.png")
