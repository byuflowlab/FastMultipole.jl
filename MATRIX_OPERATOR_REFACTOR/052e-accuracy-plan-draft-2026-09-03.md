# 052e accuracy-sweep plan draft (2026-09-03) — pre-registered tolerances + comparison-run matrix

> **SUPERSEDED 2026-09-05** by `052e-accuracy-plan-v2-draft-2026-09-05.md` (Ryan ruling: rework around doublet-wake ground-truth verification). Retained for the Stage B run matrix and rationale history.

Archived groundwork for the Stage B portion of
`052e-impl-hybrid-wake-potential-experimental.md` (host regressions remain
BLOCKED until 052b closes; nothing in this file authorizes runs, deployment,
or promotion). All tolerance values
below are PROPOSED and NEED RYAN's ratification before any sweep is run —
pre-registration means locking them BEFORE looking at sweep results.

## 1. Pre-registered tolerance table (PROPOSED — NEEDS RYAN)

Basis for proposals: (a) the 052 campaign locked gate ceilings
(`~/projects/FLOWPanel.jl/data/fm052_campaign_lock/fm052_locked_tolerances.toml`
on orc: CT cycle-mean ceiling 1.8e-3, Gamma M2 max 2.93e-3, RMS 1.5e-3 —
the ceilings trial-1 run 1 passed with ~20x margin), and (b) the 052e doc's
own characterization numbers (trace error ~10%, particle/panel velocity
error ~60% on the small fixture, doc lines 24–25). Hybrid-vs-VTS is a
formulation change, not a backend port, so gates should be looser than the
bit-level 052 GPU gates but tight enough to certify production equivalence
of integrated loads.

| # | Metric | Comparison | Proposed tolerance | Rationale | Status |
|---|---|---|---|---|---|
| T1 | CT (thrust coeff), cycle-mean over final rev | Hybrid vs VTS, matched config | rel ≤ 5e-3 | ~3x the 052 GPU gate ceiling; loads are the promotion currency | NEEDS RYAN |
| T2 | CQ (torque coeff), cycle-mean over final rev | Hybrid vs VTS | rel ≤ 5e-3 | same basis as T1 | NEEDS RYAN |
| T3 | Bound circulation Γ, per-blade M2 (mean-square) over final rev | Hybrid vs VTS | rel ≤ 1e-2 | 052 Gamma gates passed at ~1e-4; order-of-magnitude headroom for formulation change | NEEDS RYAN |
| T4 | Wake trace (gauge-aligned) error | Hybrid vs analytic/VTS trace | ≤ 10% on production mesh (no worse than small fixture) | doc line 24 characterization; gate = "does not degrade at scale" | NEEDS RYAN |
| T5 | Surface velocity U·t at collocation pts, RMS rel | Hybrid vs VTS | ≤ 1e-2 RMS, ≤ 5e-2 max | drives pressure/loads; tighter than the 60% particle/panel figure which the doc marks as characterization-only | NEEDS RYAN |
| T6 | Green residual / gauge defect / Hodge mismatch telemetry | Hybrid absolute | no worse than existing regression targets (doc lines 27–29) | already-locked regression values; reuse, don't invent | NEEDS RYAN |
| P1 | 414-step wall time | Hybrid, production case, CUDA | < 6480 s | doc line 46 (verbatim gate) | LOCKED by doc |
| P2 | Device memory | same run | ≥ 20% reserve at peak | doc line 47 (verbatim gate) | LOCKED by doc |
| P3 | Host-fallback occurrences in CUDA-proof run | log grep | 0 (named-and-prohibited per doc line 44) | route-proof discipline, same as 052 source gates | LOCKED by doc |

Open question for Ryan: whether T1/T2 should be judged against the VTS run
or against the pinned CPU reference family used in 052 (the latter couples
052e to the 052 lock; doc line 17 says matched-config independent runs, so
VTS-vs-Hybrid head-to-head is the default here).

## 2. Comparison-run matrix (doc line 39 axes)

Factor levels (PROPOSED; full factorial is 2×3×3×2×2×2 = 144 runs — NOT
feasible; use the starred baseline + one-factor-at-a-time (OFAT) column):

| Axis | Levels | Baseline (*) | OFAT variants |
|---|---|---|---|
| Formulation | VTS, Hybrid | both (every row is a pair) | — |
| Wake resolution (P_PER_STEP / nwakerows) | coarse / production (12) / fine | production* | coarse, fine |
| Particle core (sigma/overlap) | 2.2 / 2.75* / 3.4 | 2.75* | 2.2, 3.4 |
| Distance (truncation depth) | 3.0R / 4.5R* | 4.5R* | 3.0R |
| Handoff (Das clearance) | 3.4σ* / 2.0σ | 3.4σ* | 2.0σ |
| Rotor count | 1r* / 2r | 1r* | 2r |
| Ground | OGE* / IGE (h/R=1.5) | OGE* | IGE |

Run list (each row = one VTS + one Hybrid run, matched config, short-form
36-step mature-window unless noted):

1. Baseline pair (all starred levels) — also the long-form candidate.
2. Wake-resolution coarse pair.
3. Wake-resolution fine pair.
4. Core 2.2 pair.
5. Core 3.4 pair.
6. Truncation 3.0R pair.
7. Handoff 2.0σ pair.
8. 2r pair (uses 052b two-rotor case; depends on 2r solver policy — blocked
   on the 052b 2r resolution).
9. IGE pair (h/R=1.5, ground panels; depends on 052b IGE acceptance).
10. Perf/memory long run: Hybrid only, 414-step production case, CUDA,
    timers + memory sampling on (gates P1–P3). One run, after rows 1–7 pass.

Total: 9 short pairs (18 runs) + 1 long run. Rows 8–9 are the only ones
gated on 052b closure; rows 1–7 could in principle run on host/GPU once
step 1 regressions are green.

Metrics recorded per pair: T1–T6 table values + step timers; for row 10
additionally Green assembly/factorization time, Hodge diagnostics time,
residual-pass counts, recurring solve time, retained memory (doc lines
41–42).

## 3. Execution order and dependencies

1. 052b closes → step-1 host regressions (julia-test-runner, local or orc
   CPU) → gate: all green.
2. Ryan ratifies §1 tolerances (pre-registration lock).
3. Rows 1–7 short pairs (stage into ONE sbatch multi-stage script per the
   cluster-jobs house rule if run on GPU; host runs can go on idle CPU).
4. Rows 8–9 once their 052b dependencies resolve.
5. Row 10 perf/memory run → P1–P3.
6. CUDA route proof (doc step 4) can piggyback on row 10's log.
7. Promotion ruling package for Ryan (doc step 5).
