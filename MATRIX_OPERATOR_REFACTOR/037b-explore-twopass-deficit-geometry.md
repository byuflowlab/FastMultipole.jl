# 037b Nearfield Reduction II: Two-Pass Deficit Splitting at Co-Designed Geometry

## Status and Entry Gate

**DONE `2026-08-14`.** Verdict: the two-pass co-design hypothesis is
falsified at gate accuracy on all three cases; `037c` is recommended
NO-GO; the `038` entry-gate evidence (multi-scale density as a measured
binding cost) is supplied at headline magnitude by the new rotor-wake
case, including a user-decision item (rotor auto-depth) left explicitly
unshipped. Full record below; awaiting clear-context approval.

Originally staged by user direction `2026-08-13`; precedes `038` (the
adaptive-octree derivation) in the roadmap order by that same direction.

Entry gate: `037a` Done and clear-context approved.
No production default changes without explicit user approval; exploration,
measurement, and pre-registered sweeps do not need per-step approval.

## Motivation

The nearfield is 66-95% of the shipped U/J solve and IS the critical path
(unlike the far-field levers of `037` — see the standing lesson: price
levers against the overlapped critical path, where nearfield reductions
translate ~1:1 until the far-field chain surfaces at roughly 2-5 ms). The
pair count exists because the σ-adequacy gate
`g_min(q)·h_leaf >= rho_t·sigma` forces the direct stencil to reach
`rho_t·sigma ≈ 3.7σ` so that everything beyond it is pure singular math.

Hypothesis under test: **two-pass deficit splitting decouples the stencil
from `rho_t`.** Singular `1/r` math runs everywhere the FMM/stencil
geometry requires (expansion-validity reach only — the scalar path runs
q² = 4-6 one level deeper); a pairwise deficit correction
(`(g(ρ)-1)`-weighted Biot-Savart, smooth, short-ranged) runs only within
`rho_c·sigma` with `rho_c ≈ 2`. Naive scaling `(rho_c/rho_t)^3 ≈ 0.16`
suggests up to ~6x fewer correction pairs plus a deeper admissible tree —
but `032a` Stage D measured partitioned beating two-pass *at matched
geometry*, so the open question is precisely the co-design: two-pass at its
OWN best (ell, q, rho_c, P). This is the same coupled-lever trap that 035
cycle 3 exposed for P/rho_t/stencil; resolve it the same way —
measurement-first.

`TwoPassVortex` already exists as a production kernel surface
(`direct_kernel=:twopass` reaches it from FLOWVPM); the deficit-kernel
theory is in `031a`/`theory/kernel-splitting-nearfield.md`.

## Objective

Determine, by pre-registered measurement, whether two-pass deficit
splitting at co-designed geometry reduces the end-to-end U/J solve by a
reportable margin on the campaign cases and on a realistic rotor wake; ship
it as a default only on explicit user approval; produce an exact
per-approach speedup report; and render the go/no-go evidence for `037c`
(mesh/FFT deficit evaluation).

## Campaign

1. **Realistic rotor-wake case (shared deliverable, consumed by `037c`).**
   Construct a rotor-wake particle field from an actual rotor — e.g. the
   DJI 9443 rotor available in `~/Dropbox/research/projects/FLOWPanel.jl`
   (`examples/dji9443_*.jl`; blade circulation seeding a FLOWVPM wake, or a
   FLOWUnsteady-generated snapshot if one is more direct). Requirements:
   deterministic construction (seeded), n at 1e5 and 1e6, σ distribution
   and aspect ratio documented (real rotor wakes are longer and more
   σ-heterogeneous than the AR=5 helical cylinder of 033), sampled-direct
   velocity references computed once with the 033 erf-oracle machinery and
   sha256-checksummed. This case also produces the multi-scale-density
   evidence (or lack of it) that `038`'s entry gate asks for — record that
   verdict explicitly either way.
2. **Accuracy instrument first.** Extend the 3C error-decomposition oracle
   to the two-pass field: separate truncated-deficit error (pairs beyond
   `rho_c·sigma` whose correction is dropped) from FMM truncation error;
   governing gate remains the conservative sum `< 1e-3` on sampled velocity
   RMS, J diagnostic. Pre-register the gate before any timing sweep.
3. **Pre-registered co-design sweep** (3A pattern; cube + wake + rotor wake
   at the representative n): grid over `rho_c` (≈1.7-2.5), depth
   (adequacy under `rho_c`, incl. one-level-deeper candidates), leaf `q`
   (down to expansion-validity), `P` (literature 4/5/6), both precisions at
   the winner. Same-job anchors at the shipped defaults
   (P5/3.668/partitioned/dense, cubic). All standard gates per row
   (checksummed references, flat 023 counters, zero recurring allocation).
4. **Optimization cycles** (035 rules): ranked levers with a ≥5% expected
   end-to-end bar priced against the overlapped critical path; production
   implementation only with explicit user approval; realized-vs-expected
   recorded per cycle.
5. **Definitive speedup report.** Per case (cube / wake / rotor wake) and
   scale: U/J solve and RK3 medians (15 warmed reps policy), speedup vs the
   shipped-default anchor measured in the same job, per-stage profiles
   showing where the reduction landed, accuracy decomposition per winner,
   and the co-design map (what `rho_c` bought at which depth). Figures per
   the 024a conventions. **Every claimed speedup must be attributable to
   this approach alone** — anchors and winners share everything except the
   split/geometry under test.
6. **037c gate verdict.** Recommend opening `037c` iff the measured
   evidence says mesh evaluation of the deficit beats pair evaluation —
   e.g. the correction-pass pair cost remains a material fraction of the
   solve at the best `rho_c`, or accuracy forces `rho_c` high enough to
   cap the gain. Recommend against iff two-pass captures most of the
   available reduction or the deficit cost is already negligible. Record
   the quantitative basis either way.

## Dependencies and Reading

- `037a` (rectangular-grid/nearfield follow-up measurements), `037` (Done +
  approved), `035` Final Report §3 and cycle-3 records (the
  co-design method and the overlap lesson), `031a` +
  `theory/kernel-splitting-nearfield.md` (deficit kernel, `rho_c` hybrid),
  `032a` Stage D (the matched-geometry two-pass result being superseded),
  `033` (case/reference machinery), `benchmark_035_gpu.jl` harness
  (extend; keep the campaign-file/CSV discipline).
- `~/Dropbox/research/projects/FLOWPanel.jl` — rotor geometry/circulation
  source for the wake case (read its CLAUDE.md/examples before building).

## Verification Gates

- Sampled velocity RMS ≤ 1e-3 (conservative sum form once the decomposition
  instrument exists) against sha256-checksummed references on every
  reported row; J logged as diagnostic.
- 023 counter/allocation contracts unchanged; scalar 028/030 no-regression
  reruns mandatory if FastMultipole `src/` is touched.
- Speedup claims only from same-job anchor/winner pairs; the report states
  the warmup/repeat/median policy and prices every lever against the
  overlapped critical path.

## Work Record

### 2026-08-13 — instruments, rotor case, pre-registration (session 1)

**Built (two file-disjoint lanes, both diffs lead-reviewed, all local
verification re-run by the lead):**

- **Rotor-wake case `rotor`** (`benchmark_033_common.jl`): deterministic
  DJI 9443 prescribed helical wake from the measured FLOWPanel phase-02c
  bound circulation (finest capped mesh `dji121c`, RPM 5400, R=0.119 m,
  B=2), vendored at `data/rotor_wake/dji9443_fixed_bin_circulation.csv`
  with provenance. Per blade: root + one lumped inboard + tip filament
  (trailed sum exactly zero), momentum-theory axial convection
  (T=1.750 N, v_i=4.007 m/s, far-wake doubling), 0.78 contraction,
  14.0 rev age, length 1.202 m, AR 5.05. Per-particle σ (β=2 against
  local filament spacing, core-spreading growth ×3 over the age):
  1e5 σ ∈ [1.73e-4, 3.11e-3] m (18× spread), ∝ 1/n. Seeded
  (`FM033_SEED+104729+n`), exactly n particles, bitwise reproducible.
  Checksummed Float64 sampled-direct references built and manifest-merged
  (additive; cube/wake lines byte-identical):
  n=1e5 `6f8ee709…d6135e`, n=1e6 `95757ef5…81c61b`.
- **Two-pass decomposition oracle** (`benchmark_035_error_decomposition.jl`):
  documented + asserted the equivalence P_shell(rho_c, rho_x) == P(rho_x)
  (the shipped TwoPassVortex field equals the partitioned intermediate at
  the pass-2 truncation radius, independent of rho_c), added
  `component_semantics` column ("truncated_deficit" for twopass), a
  CPU-only `FM035D_EXACT_ONLY` cutoff-curve mode (config file authoritative
  for the (case,n) grid; one exact pass per point for all rhos), and the
  smoke test `test_037b_decomposition_smoke.jl` (87/87 local, includes
  P=4-coverage configs; twopass rows reproduce partitioned cutoff errors
  bitwise at equal rho_t; oracle matches the checksummed wake reference to
  2.09e-16/5.34e-16).
- `rotor_case_stats.jl` + `data/rotor_wake/rotor_case_stats.csv`: σ and
  occupancy stats, rotor vs wake (038 evidence below).
- `cuda_035_submit.sh`: rsync line for `data/rotor_wake` (lead edit).

**Pre-registered accuracy gate (before any timing sweep, 3C policy):**
sampled velocity RMS governs; winners must pass the conservative sum
`(truncated_deficit + fmm)/||R|| < 1e-3` from the decomposition mirror
job; J logged as diagnostic only.

**Instrument data (exact-only truncated-deficit U component vs pass-2
truncation radius rho_x; 512-sample checksummed references; local CPU):**

| rho_x | cube 1e5 | cube 1e6 | wake 1e5 | wake 1e6 | rotor 1e5 | rotor 1e6 |
|---:|---:|---:|---:|---:|---:|---:|
| 3.668 | 5.56e-4 | 5.14e-4 | 1.05e-4 | 3.61e-5 | 4.43e-5 | 4.54e-5 |
| 3.4 | 1.46e-3 | 1.42e-3 | 2.70e-4 | 9.05e-5 | 1.20e-4 | 1.63e-4 |
| 3.2 | 3.09e-3 | 2.95e-3 | 5.19e-4 | 1.72e-4 | 3.00e-4 | 3.55e-4 |
| 3.0 | 5.86e-3 | 5.62e-3 | 9.52e-4 | 3.16e-4 | 5.93e-4 | 7.88e-4 |

(3.668 values reproduce the 035 cycle-3C rows of record exactly.)
Consequences: cube is pinned at rho_t=3.668 (no truncation headroom);
wake admits 3.4 (1e5) and 3.2/3.0 (1e6); rotor admits 3.4 at both n.
The naive "correction within 2σ" reading is refuted outright on accuracy
(6.3e-2 to 9.5e-2 at rho_x=2) — quantitative input to the 037c verdict.

**Admissibility co-design (production gate arithmetic, verified with the
real per-case σ_max and derived boxes; twopass reach = rho_c):** deeper
trees are admissible where 037a's q=3 screen was not — wake 1e5
(ℓ6,q6,ρ_c≤1.95), wake 1e6 (ℓ7,q6,ρ_c≤2.10), cube 1e5 (ℓ5,q12,ρ_c≤1.95),
cube 1e6 (ℓ6,q12,ρ_c≤2.10); rotor σ_max is tiny (3.11e-3/3.11e-4 m), so
adequacy never binds through ℓ7/ℓ8 for either kernel — on the rotor the
two-pass depth-decoupling advantage is void and the comparison is pure
pair-economics.

**Pre-registered H200 grid:** `scripts/fm037b_cases_screen.txt` (44 rows,
F32, dense, profile on; same-job shipped-default anchors per case/scale;
q at/below shipped with deeper-ℓ candidates; rho_t truncation per the
instrument; AABB on with per-scale no-AABB controls; P6 twins only where
P5 FMM error is at risk) and its exact decomposition mirror
`scripts/fm037b_cutoff_configs.txt` (36 unique accuracy configs, both
precisions). Local dryrun 44/44; decomposition parse 36/36. Winners get a
follow-up F64 + RK3 + F32-repeat confirmation grid.

### 2026-08-14 — H200 screen + decomposition results, verdicts (session 1 cont.)

Job history: decomposition mirror job `13170230` COMPLETED (5m58s). First
screen submission `13170229` FAILED in the stage-5 shipped-defaults
refcheck: `cuda_034_refcheck.jl` iterates `FM033_CASES`, which Lane A had
widened with `rotor`, and the rotor n=1e4 reference intentionally did not
exist. Fix (minimal, semantics-preserving): the refcheck case list is now
`FM034_REFCHECK_CASES` (default `cube,wake` — the historical 034 contract
scope; exploration cases are gated per-row in their own sweeps), verified
in host mode locally; the rotor n=1e4 reference was also built and
manifest-merged (additive). Resubmission `13170509` COMPLETED (20m53s):
44/44 rows ok, all preflights green, refcheck PASSED, flat counters and
stable allocation on every row. Data of record:
`data/flowvpm_gpu_campaign/fm037b_screen.csv`,
`fm037b_error_decomposition.csv`, logs `fm035-13170509.out`.

**Accuracy decomposition (conservative sum `truncated_deficit + fmm`,
Float32 ≈ Float64):** cube P5 requires q≥12 and only the q16 deeper trees
pass (sums 9.14e-4/8.85e-4; the q12 deeper trees fail at 1.25–1.27e-3
despite observed totals ~9.1e-4 — the pre-registered gate governs); wake
passes at (ℓ6,q6) ρ_x∈{3.668,3.4,3.2} and (ℓ7,q6) ρ_x∈{3.668,3.4,3.0};
rotor passes at q6 through ℓ8 (FMM component 5.4–7.0e-4, elevated by the
filament concentration) and fails at q4.

**Timing screen (U/J solve median of 15 warmed reps, F32, same-job
anchors; full table in `fm037b_screen.csv`):**

| case/n | anchor (ms) | best gate-passing two-pass | best other |
|---|---:|---|---|
| cube 1e5 | 11.31 | (ℓ5,q16) 81.4 = **0.14x** | — |
| cube 1e6 | 102.3 | (ℓ6,q16) 688.5 = **0.15x** | — |
| wake 1e5 | 7.85 | (ℓ6,q6,ρ_x3.2) 16.35 = **0.48x** | — |
| wake 1e6 | 83.6 | (ℓ7,q6,ρ_x3.0) 112.9 = **0.74x** | — |
| rotor 1e5 | 12.66 | (ℓ7,q6,ρ_x3.4) 8.31 = 1.52x | **partitioned (ℓ6,q6) 6.99 = 1.81x** |
| rotor 1e6 | 238.7 | (ℓ8,q6,ρ_x3.4) 40.3 = 5.92x | **partitioned (ℓ8,q6) 33.3 = 7.16x** |

**Critical-path pricing of the two-pass structure:** the pass-2 deficit
sweep is NOT the binding cost anywhere. Wake 1e6 (ℓ7,q6,ρ_x3.0): pass-2
costs ≈18 ms of the 109.5 ms eval (shell-pair scaling between ρ_x rows);
zeroing it entirely leaves ≈91.5 ms — still slower than the 83.6 ms
anchor, because the ℓ7 far chain (M2L 27.9 + B2M 10.4 vs 4.9 + 3.4 at ℓ6)
and the un-overlapped primary stage dominate. Rotor (ℓ8): pass-2 ≈6.9 ms
on a nearly-empty shell (1.5e7 shell pairs of 3.1e8 candidates — σ is
tiny, so the deficit annulus holds almost nothing), and partitioned at the
SAME geometry needs no deficit pass at all and wins (33.3 vs 41.1 ms).
Direct-pair reductions are real (wake 1e6: 13.4e9 → 1.9e9 primary) but
deeper-tree far-field cost exceeds the saving on the compact/filling
cases. AABB pruning changes two-pass by only ~1–2% at these geometries.

**Verdict A — two-pass co-design hypothesis: FALSIFIED.** At gate
accuracy there is no (ell, q, rho_c, P, ρ_x) where two-pass beats the
shipped partitioned default on cube or wake (0.14x–0.74x), and on the
rotor it is dominated by partitioned at equal geometry. The three
mechanisms: (a) FMM truncation pins the primary stencil (cube q≥12/16,
wake/rotor q6) — the "expansion-validity q²=4–6" premise fails the
conservative-sum gate at P5 and even P6; (b) the depth the smaller ρ_c
unlocks costs more in far-field work than it saves in direct pairs;
(c) where σ is small relative to spacing (the realistic rotor wake), the
deficit shell is nearly empty and pass 2 is pure overhead. No production
default change is proposed from the two-pass side.

**Verdict B (draft) — 037c mesh-deficit evaluation: NO-GO.** The 037c
premise (mesh/FFT evaluation of the smooth deficit) presupposes either a
material pair-evaluated correction cost or a short-ranged correction
(ρ_x ≈ 2). Measured: (i) the exact instrument refutes ρ_x ≈ 2 outright
(truncated-deficit error 6.3e-2–9.5e-2 at ρ_x=2; the admissible floor is
3.2–3.4 on wake/rotor and 3.668 on cube), so the compact-support regime a
mesh stencil needs does not exist at gate accuracy; (ii) even a ZERO-cost
deficit evaluator leaves every two-pass candidate behind the shipped
partitioned baseline (wake 1e6: ≈91.5 vs 83.6 ms; rotor: best-case tie
with partitioned-at-depth, which needs no correction at all). The deficit
pass is not where the time is; the far field of the deeper tree is.
Recommendation: do not open `037c`; close it by pointer and let `038`
follow `037b` directly.

**Incidental major finding (rotor depth mis-selection — USER DECISION
REQUIRED):** the shipped auto-geometry heuristic (occupancy-capped depth,
uniform-field reasoning) picks (ℓ5)/(ℓ6) for the rotor where the measured
optimum is (ℓ6)/(ℓ8): pinning depth gives **1.81x at n=1e5 (12.66 →
6.99 ms)** and **7.16x at n=1e6 (238.7 → 33.3 ms)**, gate-passing
(6.97e-4) with the existing partitioned kernel and no code change. At
n=1e6 the optimum may lie beyond the uniform path's ℓ≤8 cap (nearfield is
still 70% of the ℓ8 eval). Changing the auto-depth rule is a production
default change and is NOT made here; recorded for the user. The adaptive
octree (038) is the structural fix.

**038 multi-scale-density evidence (preview, from `rotor_case_stats.csv`,
n=1e6):** rotor max/mean bodies per occupied cell is 5–7× at every level
(wake ≤1.9×), occupied-cell fraction falls with depth (0.84% at ℓ6, 0.21%
at ℓ7 vs wake's ~3.4–3.6%), and the top-1% densest cells hold ~6% of all
bodies (wake ~1.4–2.4%). The σ_max geometry gate does NOT bind (thin
young tip cores keep σ_max small); the binding multi-scale cost is
density contrast (fat-cell nearfield concentration). **Confirmed as a
measured binding cost by the 13170509 screen:** the uniform-grid
auto-selected geometry costs 7.16x at rotor n=1e6 (238.7 → 33.3 ms at
pinned ℓ8) and 1.81x at n=1e5, with the ℓ≤8 uniform-path cap plausibly
still binding at 1e6 (nearfield remains ~70% of the ℓ8 eval and pairs
concentrate in the top-1% densest cells). The `038` entry-gate evidence
requirement is met: multi-scale density binds, on a realistic rotor wake,
at headline magnitude.

### 2026-08-14 — F64/RK3 confirmation + Final Report (session 1 close)

Confirmation job `13170520` COMPLETED (5m40s, refcheck PASSED at the
fixed cube,wake scope, 15/15 rows ok, gates pass, counters flat, stable
allocation). Data: `fm037b_confirm.csv`, log `fm035-13170520.out`.

## Final Report (definitive for 037b)

Measurement policy: H200 (m13h-1-2), julia 1.11.7; every timing is the
median of 15 warmed reps after 2 warmup solves; stage times are isolated
CUDA-event medians (production overlaps NF with the far chain — the
overlapped U/J wall time decides); every reported row passes sampled
velocity RMS ≤ 1e-3 vs sha256-checksummed references, plus the
pre-registered conservative-sum decomposition gate; J diagnostic; flat
023 counters and stable allocation on every row. Jobs of record:
screen 13170509 (F32, same-job anchors), decomposition mirror 13170230,
confirmation 13170520 (F64/RK3). Anchors reproduce the 035 records
across jobs (wake 1e6 F64: 179.02 here vs 179.00 in 035).

### 1. Two-pass at co-designed geometry vs shipped default (same job)

U/J solve, F32 screen (F64 confirms the ordering; wake 1e6 F64 two-pass
196.4 vs anchor 179.0 = 0.91x):

| case/n | anchor | best gate-passing two-pass | ratio |
|---|---:|---|---:|
| cube 1e5 | 11.31 ms | (ℓ5,q16,ρ_x3.668) 81.4 ms | 0.14x |
| cube 1e6 | 102.3 ms | (ℓ6,q16,ρ_x3.668) 688.5 ms | 0.15x |
| wake 1e5 | 7.85 ms | (ℓ6,q6,ρ_x3.2) 16.35 ms | 0.48x |
| wake 1e6 | 83.6 ms | (ℓ7,q6,ρ_x3.0) 112.9 ms | 0.74x |
| rotor 1e5 | 12.66 ms | (ℓ7,q6,ρ_x3.4) 8.31 ms | 1.52x — but partitioned (ℓ6,q6) = 6.99 ms |
| rotor 1e6 | 238.7 ms | (ℓ8,q6,ρ_x3.4) 40.3 ms | 5.92x — but partitioned (ℓ8,q6) = 33.3 ms |

**No (ell,q,rho_c,P,ρ_x) exists where two-pass wins.** Where it beats
the *anchor* (rotor), plain partitioned at the same pinned geometry beats
it further — the speedup belongs to the geometry, not the split.

### 2. Rotor pinned-depth winners (confirmed both precisions + RK3)

| config | TF | U/J (ms) | RK3 (ms) | u_rms | speedup vs auto anchor |
|---|---|---:|---:|---:|---:|
| rotor 1e5 partitioned (ℓ6,q6) | F32 | **7.01** | 26.16 | 5.41e-4 | 1.76x (12.33 / 37.02 RK3 1.41x) |
| rotor 1e5 partitioned (ℓ6,q6) | F64 | **11.49** | 39.69 | 5.38e-4 | 2.00x (22.94 / 68.68 RK3 1.73x) |
| rotor 1e6 partitioned (ℓ8,q6) | F32 | **33.97** | — | 6.97e-4 | 7.07x (240.2) |
| rotor 1e6 partitioned (ℓ8,q6) | F64 | **60.87** | — | 1.07e-4 | 7.85x (477.9) |

### 3. Where the time went (co-design map, critical-path priced)

- Deeper trees multiply far-field cost faster than they shrink direct
  work on compact/filling domains: cube ℓ4→ℓ5 M2L 4.4→27.6 ms and
  L2B 5.5→36.4 ms; wake ℓ6→ℓ7 far chain 11 → 41.6 ms. The direct-pair
  savings are real (wake 1e6: 13.4e9 → 1.9e9 primary) but never repay
  the far-field growth at gate accuracy.
- The pass-2 deficit sweep is never the binding cost: ≈18 ms of the
  109.5 ms wake-1e6 eval at ρ_x=3.0 (zeroing it leaves ≈91.5 > 83.6
  anchor); ≈6.9 ms on the rotor's nearly-empty shell (1.5e7 shell pairs
  of 3.1e8 candidates — σ small ⇒ the deficit annulus holds ~nothing).
- Accuracy pins the primary stencil: q4/q3 fail conservative-sum at P5
  and (except one wake point) at P6; cube requires q12/q16. The
  truncated-deficit floor is ρ_x ≥ 3.2–3.4 (wake/rotor), 3.668 (cube).
- AABB pruning (037a lever) moves two-pass ≤ 2% at these geometries.

### 4. Optimization-cycle ledger (035 rules)

No two-pass-side lever reaches the ≥5% bar — every candidate is a
regression; no production cycle run. One lever above the bar emerged:
the **rotor auto-depth rule** (§ incidental finding; 43–86% U/J
reduction on the rotor, zero effect on cube/wake which auto-select
correctly). It is a production default change and therefore NOT
implemented — awaiting explicit user decision; the adaptive octree
(038) is the structural fix that supersedes a heuristic patch.

### 5. Figures and data of record

`data/figures/fig12_037b_deficit_curves.{tex,pdf}` (+`fig12_037b_deficit_curves/`)
— exact truncated-deficit error vs ρ_x, all cases/scales, with the gate
and the refuted ρ_x≈2 premise annotated.
`data/figures/fig13_037b_codesign_screen.{tex,pdf}` (+ dir) — per-candidate
speedup vs same-job anchor, gate-pass distinguished.
Tables regenerated by `scripts/figures_037b_prepare.jl`; analysis by
`scripts/analyze_037b_screen.jl`. CSVs:
`fm037b_screen.csv`, `fm037b_confirm.csv`, `fm037b_error_decomposition.csv`,
`fm037b_deficit_curves.csv` (+ 037b rotor reference rows in the 033
manifest). Case files: `fm037b_cases_screen.txt`, `fm037b_cases_confirm.txt`,
`fm037b_cutoff_configs.txt`.

### 6. Verdicts

- **037c: NO-GO** (quantitative basis in Verdict B above: the compact
  ρ_x≈2 regime fails accuracy by 60–100x, and a zero-cost deficit
  evaluator still loses to the shipped baseline everywhere measured).
  Recommend closing `037c` by pointer; `038` follows `037b` directly.
- **038 entry gate: MET.** Multi-scale density measured as a binding
  cost on a realistic rotor wake (7.16x from depth alone at n=1e6; ℓ≤8
  uniform cap plausibly still binding; occupancy contrast tables in
  `rotor_case_stats.csv`).
- **No production default changed.** User-decision item: rotor
  auto-depth (interim heuristic patch vs waiting for 038).
