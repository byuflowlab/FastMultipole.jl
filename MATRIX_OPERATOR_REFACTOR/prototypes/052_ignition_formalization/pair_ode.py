#!/usr/bin/env python3
"""Two-particle feedback closure test for gpu40 patient zero + partner."""
import numpy as np
d0 = np.load("gpu40_hist.npz", allow_pickle=True)["rows"]
dt = 3.0864197530864197e-4
rows = [r for r in d0 if "Gp" in r]
print("step  d       |G|      |Gp|    s1_obs  strain_BS  ratio  lam_obs_next")
prev = None
out = []
for r in rows:
    d = np.linalg.norm(r["pos"] - r["posp"])
    nG = np.linalg.norm(r["G"]); nGp = np.linalg.norm(r["Gp"])
    J = r["J"]; s1 = np.linalg.eigvalsh(0.5 * (J + J.T))[-1]
    sBS = nGp / (4 * np.pi * d**3)
    out.append((r["step"], d, nG, nGp, s1, sBS))
o = np.array(out)
for i in range(len(o) - 1):
    st, d, nG, nGp, s1, sBS = o[i]
    ds = o[i + 1][0] - st
    lam = np.log(o[i + 1][2] / nG) / (ds * dt)
    if 900 <= st <= 999 or int(st) % 25 == 0:
        print(f"{int(st):4d} {d:.3e} {nG:.3e} {nGp:.3e} {s1:8.1f} {sBS:8.1f} {s1/sBS:6.2f} {lam:9.1f}")
# ODE: dG/dt = c*k*G^2/(4 pi d^3), c=0.22 (measured), k=s1/sBS at feedback onset
# finite-time blowup: t* - t = 4 pi d^3/(c k G(t))
m = (o[:, 0] >= 988) & (o[:, 0] <= 995)
k = np.median(o[m, 4] / o[m, 5])
print(f"\nmedian k=s1/strain_BS over 988-995: {k:.2f}")
for st in (988, 990, 992):
    i = np.where(o[:, 0] == st)[0][0]
    _, d, nG, nGp, s1, sBS = o[i]
    tstar = 4 * np.pi * d**3 / (0.22 * k * nG) / dt
    print(f"pair-ODE blowup forecast from step {int(st)}: t* = +{tstar:.1f} steps  (observed ignition ~996)")
