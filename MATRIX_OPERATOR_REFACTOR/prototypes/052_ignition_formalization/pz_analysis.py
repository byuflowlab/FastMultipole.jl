#!/usr/bin/env python3
"""Patient-zero quantitative mechanism analysis (task 052 formalization).

Extracts one particle's history from a .vtp series and tests the rVPM
update equations extracted from FLOWVPM source:
  Gamma update (transposed, f=0, g): dG/dt = S - 3 Z G - C*SFS*sig^3/zeta0
     S = J^T G,  Z = g * (G . S)/|G|^2
  sigma update (Euler): sig <- sig * (1 - dt Z)
  CoreSpreading: sig <- sqrt(sig^2 + 2 nu dt)   [applied separately]
Usage: pz_analysis.py label idx partner_idx dt outprefix dir1 [dir2 ...]
       partner_idx = -1 to skip
"""
import sys, glob, re, os
import numpy as np
from vtk import vtkXMLPolyDataReader
from vtk.util.numpy_support import vtk_to_numpy
from scipy.spatial import cKDTree

label, idx, pidx, dt, outpref = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
dirs = sys.argv[6:]
g_rvpm = 0.2
nu = 1.4334181509754028e-5
zeta0 = (2*np.pi)**-1.5

files = {}
for d in dirs:
    for f in glob.glob(os.path.join(d, "*particles.*.vtp")):
        m = re.search(r"\.(\d+)\.vtp$", f)
        if m: files[int(m.group(1))] = f
steps = sorted(files)

rows = []
for s in steps:
    r = vtkXMLPolyDataReader(); r.SetFileName(files[s]); r.Update()
    pd = r.GetOutput()
    n = pd.GetNumberOfPoints()
    if idx >= n: continue
    pts = vtk_to_numpy(pd.GetPoints().GetData())
    A = pd.GetPointData()
    def arr(name): return vtk_to_numpy(A.GetArray(name))
    gam = arr("gamma"); sig = arr("sigma"); vel = arr("velocity")
    J9 = arr("velocity_gradient"); SFS = arr("SFS"); C = arr("C")
    # nearest-neighbor spacing h at patient zero
    tree = cKDTree(pts)
    dnn, _ = tree.query(pts[idx], k=2)
    h = dnn[1]
    row = dict(step=s, pos=pts[idx].copy(), G=gam[idx].copy(), sig=float(sig[idx]),
               u=vel[idx].copy(), J=J9[idx].reshape(3,3).copy(),
               SFS=SFS[idx].copy(), C=C[idx].copy(), h=float(h))
    if pidx >= 0 and pidx < n:
        row["Gp"] = gam[pidx].copy(); row["posp"] = pts[pidx].copy(); row["sigp"] = float(sig[pidx])
    rows.append(row)

np.savez(outpref + "_hist.npz", rows=np.array(rows, dtype=object), allow_pickle=True)

print(f"### {label}: idx={idx} dt={dt:g} steps={rows[0]['step']}..{rows[-1]['step']} n={len(rows)}")
hdr = ("step |G| sig lam_obs lam_par lam_s1 dtZ dlnsig_o dlnsig_p visc_ln h/sig sfs_ratio C1 "
       "cosT cosC magT")
print(hdr)
summ = []
for a, b in zip(rows[:-1], rows[1:]):
    ds = b["step"] - a["step"]
    G, Gn = a["G"], b["G"]
    nG, nGn = np.linalg.norm(G), np.linalg.norm(Gn)
    if nG == 0 or nGn == 0: continue
    T = ds * dt
    lam_obs = np.log(nGn/nG) / T
    J = a["J"]
    # both conventions
    for name, Jm in (("rowmaj", J),):
        pass
    s_parA = G @ (J @ G) / nG**2          # == G.J^T G / |G|^2 (symmetric form)
    Sy = 0.5*(J+J.T)
    s1 = np.linalg.eigvalsh(Sy)[-1]
    Z = g_rvpm * s_parA
    lam_par = (1 - 3*g_rvpm) * s_parA
    dtZ = dt * Z
    dlnsig_o = np.log(b["sig"]/a["sig"]) / ds     # per step
    dlnsig_p = -dtZ                                # per step (linearized)
    visc_ln = nu*dt/a["sig"]**2                    # per step lnsig growth from core spreading
    # SFS forcing magnitude vs stretching magnitude
    S_T = J.T @ G
    sfs_force = a["C"][0] * a["SFS"] * a["sig"]**3 / zeta0
    sfs_ratio = np.linalg.norm(sfs_force) / max(np.linalg.norm(S_T), 1e-300)
    # vector prediction tests (only meaningful for ds small)
    dG = Gn - G
    predT = T * (J.T @ G - 3*Z*G)
    predC = T * (J @ G - 3*Z*G)
    def cs(x, y):
        nx, ny = np.linalg.norm(x), np.linalg.norm(y)
        return x@y/(nx*ny) if nx>0 and ny>0 else np.nan
    cosT, cosC = cs(dG, predT), cs(dG, predC)
    magT = np.linalg.norm(dG)/max(np.linalg.norm(predT),1e-300)
    print(f"{a['step']:4d} {nG:.3e} {a['sig']:.3e} {lam_obs:9.1f} {lam_par:9.1f} {s1:9.1f} "
          f"{dtZ*ds: .3f} {dlnsig_o: .4f} {dlnsig_p: .4f} {visc_ln:.4f} {a['h']/a['sig']:6.2f} "
          f"{sfs_ratio:.2e} {a['C'][0]:.2e} {cosT: .2f} {cosC: .2f} {magT:.2f}")
    summ.append((a["step"], ds, lam_obs, lam_par, s1, dlnsig_o, dlnsig_p, visc_ln, cosT, cosC, magT,
                 a["sig"], nG, a["h"]))

su = np.array(summ)
# invariant sigma*|G|^{g/(1-3g)} = sigma*|G|^0.5
inv = np.array([r["sig"]*np.linalg.norm(r["G"])**0.5 for r in rows])
print(f"invariant sig*|G|^0.5: first={inv[0]:.3e} last={inv[-1]:.3e} "
      f"median={np.median(inv):.3e} cv={np.std(inv)/np.mean(inv):.2f}")
if pidx >= 0 and "Gp" in rows[0]:
    for r in rows[::max(1,len(rows)//10)]:
        d = np.linalg.norm(r["pos"]-r["posp"])
        alig = r["G"]@r["Gp"]/max(np.linalg.norm(r["G"])*np.linalg.norm(r["Gp"]),1e-300)
        nGp = np.linalg.norm(r["Gp"])
        # mutual strain estimate ~ |Gp|/(4 pi d^3)
        mst = nGp/(4*np.pi*d**3)
        print(f"pair step={r['step']} d={d:.3e} align={alig: .2f} |Gp|={nGp:.3e} "
              f"strain_est={mst:9.1f} d/sig={d/r['sig']:.1f}")
# correlation lam_obs vs lam_par and vs s1 (pre-blowup: |lam_obs| < 3000)
m = np.abs(su[:,2]) < 3000
if m.sum() > 3:
    for k, nm in ((3,"lam_par"),(4,"s1")):
        c = np.corrcoef(su[m,2], su[m,k])[0,1]
        sl = np.polyfit(su[m,k], su[m,2], 1)
        print(f"fit lam_obs vs {nm}: corr={c:.3f} slope={sl[0]:.2f} icpt={sl[1]:.1f}")
    c2 = np.corrcoef(su[m,5], su[m,6])[0,1]
    sl2 = np.polyfit(su[m,6], su[m,5], 1)
    print(f"fit dlnsig_obs vs pred: corr={c2:.3f} slope={sl2[0]:.2f} icpt={sl2[1]:.4f}")
    print(f"median cosT={np.nanmedian(su[m,8]):.3f} cosC={np.nanmedian(su[m,9]):.3f} magT={np.nanmedian(su[m,10]):.3f}")
