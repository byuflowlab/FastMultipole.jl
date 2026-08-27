#!/usr/bin/env python3
"""052c review: refit scaling, project plateau-N step cost under the P2 plan."""
import csv, math, statistics as st

D = "/Users/ryan/Dropbox/research/projects/FastMultipole/MATRIX_OPERATOR_REFACTOR/data/052c_phase1"

# N per step from gpu wake health
N = {}
for r in csv.DictReader(open(f"{D}/gpud_wake_health.csv")):
    N[int(r["step"])] = float(r["n_particles"])

rows = list(csv.DictReader(open(f"{D}/stage_d_per_step.csv")))
cats = [c for c in rows[0] if c not in ("step",)]

def getf(r, c):
    v = r.get(c, "")
    try: return float(v)
    except: return None

# power-law fit t = a*N^b over steps where N>0 and t>0.05s (avoid noise floor)
def fit(cat, tmin=0.05):
    xs, ys = [], []
    for r in rows:
        s = int(r["step"]); n = N.get(s, 0); t = getf(r, cat)
        if n and n > 1000 and t and t > tmin:
            xs.append(math.log(n)); ys.append(math.log(t))
    if len(xs) < 20: return None
    mx, my = st.mean(xs), st.mean(ys)
    b = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / sum((x-mx)**2 for x in xs)
    a = math.exp(my - b*mx)
    return a, b, len(xs)

print(f"{'category':28s} {'b (exponent)':>12s} {'t@209k':>8s} {'t@230k':>8s} {'t@260k':>8s}  npts")
proj = {}
for c in cats:
    f = fit(c)
    if not f: continue
    a, b, n = f
    t209, t230, t260 = (a*209e3**b, a*230e3**b, a*260e3**b)
    proj[c] = (t209, t230, t260, b)
    print(f"{c:28s} {b:12.2f} {t209:8.1f} {t230:8.1f} {t260:8.1f}  {n}")

# What does N do late in the run? peak N?
late = sorted(N.items())
print("\nN trajectory (every 50 steps):", [(s, int(n)) for s, n in late if s % 50 == 0])
print("max N observed:", int(max(N.values())), "at step", max(N, key=N.get))

# Budget math for the <=10 s/step target under P2:
# - wake_sfs + particle-portion of pass1 -> GPU FMM (assume FMM cost ~ measured
#   radix FMM: from 024b, ~O(N); take conservative 2-4 s at 230k? unknown -> leave symbolic)
# - remaining direct_rectangular kept for linear terms
# - untouched: io, monitors, solve, body_influence(pass3), shedding, wake_propagation, etc.
untouched = ["io", "monitors", "solve", "body_influence", "shedding",
             "wake_propagation_maintenance", "remaining_aerodynamics",
             "rigid_kinematics", "controls_setup", "unclassified_residual"]
tot230 = 0.0
print("\nprojected s/step at N=230k for categories P2 does NOT restructure:")
for c in untouched:
    if c in proj:
        print(f"  {c:28s} {proj[c][1]:6.2f}  (b={proj[c][3]:.2f})")
        tot230 += proj[c][1]
    else:
        # fall back to last-block mean
        vs = [getf(r, c) for r in rows[-67:] if getf(r, c) is not None]
        if vs:
            m = st.mean(vs)
            print(f"  {c:28s} {m:6.2f}  (last-block mean, no fit)")
            tot230 += m
print(f"  {'SUM (floor before FMM cost)':28s} {tot230:6.2f}")
print("  -> remaining budget for FMM(UJ+SFS) + linear rect terms to hit 10 s/step:",
      round(10 - tot230, 2))

# io+monitors trajectory detail
for c in ("io", "monitors"):
    vs = [(int(r["step"]), getf(r, c)) for r in rows if getf(r, c) is not None]
    blocks = [vs[i*len(vs)//6:(i+1)*len(vs)//6] for i in range(6)]
    print(f"\n{c} block means:", [round(st.mean([v for _, v in b]), 2) for b in blocks if b])
