import json
import numpy as np
from pathlib import Path
SP = Path("/tmp/claude-0/-workspace/e3b6c3cf-c47c-4de6-9013-fbab0c27bedf/scratchpad")
z  = np.load(SP/"motion_seeds_traj.npz")
zr = np.load(SP/"motion_repeat_traj.npz")

seeds = sorted(k for k in z.files if k.startswith("s4_seed"))
P = np.stack([z[k] for k in seeds])          # (S,T,4)
S, T, _ = P.shape

print("How fast do runs diverge? (mean pairwise head-translation diff, IOD units)\n")
print(f"{'window':>12s} {'cross-SEED':>11s} {'seed50 ctl-vs-prealloc':>23s} {'seed50 r01-vs-r02':>18s}")
rows = []
for a, b in [(0,25),(25,50),(50,75),(75,100),(100,150),(150,200),(200,250)]:
    cross = np.mean([np.linalg.norm(P[i,a:b,:2]-P[j,a:b,:2],axis=1).mean()
                     for i in range(S) for j in range(i+1,S)])
    pre = np.linalg.norm(zr["ctl_r01_pose"][a:b,:2]-zr["prealloc_pose"][a:b,:2],axis=1).mean()
    rep = np.linalg.norm(zr["ctl_r01_pose"][a:b,:2]-zr["ctl_r02_pose"][a:b,:2],axis=1).mean()
    rows.append({"window_s":[a/25,b/25],"cross_seed":float(cross),
                 "ctl_vs_prealloc":float(pre),"exact_repeat":float(rep)})
    print(f"{a/25:5.1f}-{b/25:4.1f}s {cross:11.4f} {pre:23.4f} {rep:18.4f}")

# correlation of the seed ensemble vs time (how much structure is shared)
print("\nShared (audio-driven) fraction of head-roll variance, by window:")
for a,b in [(0,50),(50,100),(100,150),(150,200),(200,250)]:
    mt = P[:,a:b,2].mean(0)
    vs, vr = mt.var(), (P[:,a:b,2]-mt[None]).var()
    print(f"  {a/25:4.1f}-{b/25:4.1f}s  shared={vs/(vs+vr)*100:5.1f}%")

json.dump(rows, open(SP/"divergence_vs_time.json","w"), indent=2)
print(f"\nwrote {SP/'divergence_vs_time.json'}")
