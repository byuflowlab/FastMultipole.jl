# Review of 052d-host-profile-2026-08-28.md (independent, adversarial)

**Reviewer run:** 2026-08-28, same M2 machine, 4 threads, Julia 1.12.5,
`JULIA_DEPOT_PATH=/private/tmp/flowpanel-052b-depot:...`, FLOWPanel project.
Rerun script + log:
`/private/tmp/claude-502/-Users-ryan-Dropbox-research-projects-FastMultipole/83be01c6-0db6-48f9-890d-6bdca3c16bbc/scratchpad/review_rerun.{jl,log}`.
Original materials audited: `profile_fmm.jl`/`profile.log`, `addendum_fmm.jl`/`addendum.log`,
`diag_tree.jl`/`diag.log`, `extract_snapshot.jl`/`extract.log`, `profile_degenerate.log`,
`snapshot472/SNAPSHOT_INDEX.md` (in `…/1a9c539a-…/scratchpad/`).

## Verdicts

### Claim 1 — 44.0 s/call, near-field direct = 99.8% — **CONFIRMED (with one fidelity caveat)**

- **Rerun:** production config (p=4, θ=0.4, leaf=50, shrink=true, hessian=false):
  **43.93 s** vs reported 44.00–44.14 s (<0.5% off). Interaction count independently
  rebuilt from trees + lists: **686,891,512** — exact match with the report's 6.87e8.
- The stage breakdown is a *manual replication* of the `fmm!` pipeline, not
  instrumentation of the actual call — but its sum (44.09 s) matches the real
  `fmm!` wall time (44.0–44.1 s) to 0.1%, so the partition is trustworthy. Stage
  sum audited: 0.006+0.016+0.009+0.001+44.032+0.010+0.003+0.010+0.002 = 44.089. ✓
  Warmup properly excluded (separate JIT call, 3 clean reps). Counts are measured
  from the actual direct list, not inferred. ✓
- Degenerate-run cross-check verified: 6.89e9 interactions / 437.4 s = 15.75M/s,
  same rate as the production-config near field (15.6M/s) — near-field time is
  linear in interaction count as claimed.
- **Caveat (new finding):** production `_panel_fmm_evaluate!`
  (`FLOWPanel_gpu_influence.jl:856`) passes `hessian=grad`, and for VPM particle
  targets `grad = velocity_gradient = true` (J is needed for stretching;
  see `:719` and the `:915` call site). The profile ran `hessian=false`.
  **Rerun with `hessian=true`: 62.43 s** (×1.42). So the production leg's host
  cost at this snapshot is ~62 s at 4 M2 threads, not 44 s. This does not change
  the diagnosis (near field still >99.8%) but changes the absolute numbers and
  the cross-machine consistency arithmetic (see Claim 3).

### Claim 2 — root cause: core-size radius inflation floors the source tree + widens MAC failures — **CONFIRMED**

- **Mechanism verified in source, both halves:**
  - `radius_inflation(::Type{VortexRing}, …)` at
    `FLOWPanel_elements_fmm.jl:1150` implements the gradient-aware Gaussian
    fixed point `z ← ln((1+2z)/tol)`; I recomputed it independently:
    z\* = 17.393, Δr/rc = √(2z\*) = **5.8980** → Δr = 5.898e-3 at rc = 1e-3,
    tol = 1e-6 — exactly the value in the `@warn`
    (`FLOWPanel_abstractbody.jl:1202`, fires in every log). The
    `Union{ConstantSource, VortexRing}` kernel routes to the VortexRing rule
    (`:1164`), so the profiled body does get the 5.9e-3 inflation. ✓
  - Subdivision guard verified verbatim at `FastMultipole/src/tree.jl:521`:
    `exceeds(...) && (target || child_radius >= max_body_radius)` — source-only
    stop. 36,752/76 leaves = 483.6 ✓. Independent confirmation that the floor
    (not `leaf_size`) binds: the addendum stage run at leaf=20 still shows
    125 branches / 76 leaves. ✓
  - MAC-widening half verified: `shrink_radius_source` (`tree.jl:1631-1652`)
    sets branch radius to max(distance + body_radius), so the +5.9e-3 per-panel
    radius inflates every source branch radius, directly widening MAC failure. ✓
- **A/B rerun:** `FMM_RADIUS_TOL[]=Inf` at (θ=0.4, leaf=50): **8.66 s**
  (reported 8.7), interactions **133,991,820** (exact match with 1.34e8),
  source tree 1223 branches / 928 leaves (exact match), relRMS **3.079e-4**
  (exact match). Production-config accuracy rerun: **1.995e-5** (exact match).
- One imprecision in the report's framing, not affecting the conclusion: the
  A/B flips *both* levers at once (tree floor and MAC widening), so it proves
  the inflation causes the 5× near-field excess but does not apportion it
  between (a) and (b). The 76→928 leaf change and the direct-pair granularity
  change travel together.

### Claim 3 — production consistency (43.5M inter/s at 32 threads) — **CONFIRMED-WITH-CAVEATS**

- Arithmetic audited: 6.869e8/15.8 = 43.47M/s; /32 = 1.36M/s/thread; M2
  hessian-false rate 15.6M/s/4 = 3.90M/s/thread; ratio 2.87 ≈ "~3×". ✓ as stated.
- **Caveat 1 (material):** the "~3× slower per thread, plausible for older
  server cores" explanation is partly an artifact of comparing unlike kernels.
  Production computes the hessian; with `hessian=true` the M2 rate is
  6.869e8/62.4 s = 11.0M/s → 2.75M/s/thread, and the per-thread gap to the
  A100 node shrinks to ~2.0× — more plausible, and it means the report's M2
  throughput numbers (15.6M/s, and every "would take X s" host projection)
  understate production cost by ~1.42×.
- **Caveat 2:** 15.8 s is the *median over 104 steps* of a run whose wake
  (and hence target count and interaction count) grows with step, while
  6.87e8 is the count at step 472 specifically. The division mixes the two;
  it is a plausibility check, not a measurement. It survives at the
  factor-of-~1.5 level, which is all the report needs.

### Claim 4 — no host config reaches the 0.6 s gate at production accuracy — **CONFIRMED (strengthened by the hessian finding)**

- (θ=0.5, leaf=20) = 15.305 s @ 5.947e-5 verified in `addendum.log`; the sweep
  grid in the report matches `profile.log`/`addendum.log` line for line.
- Extrapolation arithmetic audited: 15.3×4/64 = 0.96 s ✓; the 2.4e8-interaction
  figure for (0.5,20) is inferred from the constant 15.6M/s rate (15.3 s ×
  15.6M/s = 2.39e8 — consistent, though never directly counted);
  2.4e8/(2×43.5M/s) = 2.76 s ≈ "~2.8 s" ✓.
- The conclusion is *conservative in the right direction*: (i) linear 4→64
  thread scaling is optimistic for a memory-heavy kernel; (ii) the A100 node's
  own measured throughput gives 2.8 s ≫ 0.6 s; (iii) my hessian finding adds
  another ×1.42 the report didn't count (naive-linear 64T becomes ~1.36 s).
  Every unmodeled effect pushes further from the gate. Verdict robust.
- Weak spot in framing only: "production accuracy" (~5e-5) is inherited from
  what the current config happens to deliver, not from a stated requirement.
  If the actual requirement were 3e-4, the tol=Inf point (8.7 s → ~0.5–0.8 s
  naive-linear @64T, before the hessian factor) would graze the gate. The
  report acknowledges this implicitly but never states whose requirement 5e-5 is.

### Claim 5 — near-field-only device offload: ~0.26 s (prod tree) / ~0.05 s (fixed tree) — **CONFIRMED-WITH-CAVEATS (projection, not measurement)**

- Provenance audited: 36,752 × 241,986 = 8.8935e9 dense pairs ✓; 8.89e9/3.3 s
  = 2.695e9 pairs/s ✓; 6.869e8/2.7e9 = 0.254 s ✓; 1.340e8/2.7e9 = 0.050 s ✓.
  The dense GPU path computes U+J, so the rate is for the full production
  kernel — the right basis.
- **Caveats:** the 2.7e9 pairs/s was achieved on one massive regular launch.
  The near field is 39,228 ragged blocks (~17.5k pair-interactions each;
  average target leaf ~33 bodies), and the radius-fixed tree's is 71,791
  blocks of ~1.9k each — launch overhead, tail effects, and occupancy will cut
  the achieved rate, plausibly by several ×, especially for the 0.05 s figure.
  Per-step H2D/D2H of lists/buffers is also uncounted. Even at 5× derating
  (~1.3 s / ~0.25 s) the strategy beats the 3.3 s dense path and the 15.8 s
  host path, so the *direction* stands; the specific 0.26/0.05 numbers should
  be read as lower bounds.

## Arithmetic audit summary

All checked, all correct: stage sum (44.089 vs 44.0–44.1); dense pair count
8.8935e9; 92.3% culled; 15.6M/s and 3.9M/s/thread; Δr/rc = 5.8980 (independent
fixed-point recomputation); 483.6 panels/leaf; 43.47M/s and 1.36M/s/thread;
0.96 s and 2.76 s extrapolations; 2.695e9 pairs/s; 0.254 s and 0.050 s
projections. No arithmetic errors found.

## Reproduction fidelity

- Geometry is real step-472 production data (extract.log: 241,986 particles,
  18,380 nodes, 36,752 type-5 cells; bboxes consistent between .vtp/.vtu and
  the rebuilt body).
- The degenerate-controlpoints pitfall was real (`diag.log` shows all-zero
  `get_position`, 1-leaf source tree) but the **final scripts do call
  `calc_normals!`/`calc_controlpoints!`** (`profile_fmm.jl:33-34`,
  `addendum_fmm.jl:19`), and my rerun's control point 1 is nonzero with the
  same 125/76 tree — final numbers are from the corrected body. ✓
- Remaining fidelity gaps: (i) `hessian=false` vs production `hessian=grad=true`
  — quantified above at ×1.42; (ii) random strengths vs solved strengths
  (affects the accuracy column only; report flags it); (iii) the two
  trailing-wake strip bodies omitted (few panels, justified);
  (iv) **documentation trap:** `SNAPSHOT_INDEX.md`'s "validated route" snippet
  omits `calc_normals!`/`calc_controlpoints!` and therefore reproduces the
  *degenerate* body — it should be amended.

## Weakest assumptions, ranked

1. **hessian=false profiling** of a hessian=true production call (×1.42 on all
   host near-field numbers; strengthens claims 1/4, softens the specific
   per-thread story in claim 3).
2. **Device-rate transfer** from one dense launch to block-sparse near-field
   work (claim 5's 0.26/0.05 s are optimistic lower bounds).
3. **Accuracy target never specified** — the "production accuracy regime" is
   circular (defined as what the current config delivers).
4. **Median-vs-snapshot mixing** in the 43.5M/s consistency check.
5. Single-rep sweep timings (fine — the two configs run in both scripts agree
   to <1%, and near-field time tracks counts).

## Things the report should have measured but didn't

- `hessian=true` timing (done here: 62.4 s production config).
- Host thread-scaling (1→4T) to ground the linear 64-thread extrapolations.
- Direct interaction count at the recommended (θ=0.5, leaf=20) config
  (inferred from the throughput constant, never counted).
- Compact-support regularization A/B (Δr = 1e-3): the report proposes it as
  lever (i) but only ever measured tol∈{1e-6, Inf}; the intermediate point
  that would make the "manage the accuracy cost" story concrete is missing.

## Overall recommendation

The report's diagnosis survives adversarial re-derivation and re-measurement:
the radius-inflation mechanism is exactly what the code does, every rerun
number matches within 1% (several exactly, for counts), and the arithmetic is
clean. **Near-field-only device offload is the right strategy** — the near
field is >99.8% of the call and is the only stage that scales with the
problem; trees/lists/expansions cost <0.1 s on host. A full dual-tree device
port would buy at most that same <0.1 s and is not justified by these data.
Two amendments before acting on the numbers: (1) re-baseline host and device
projections with `hessian=true` (production ~62 s-equivalent at 4 M2 threads,
not 44 s; device near-field estimate unchanged in basis but the host-side
margin claims shift by ×1.42); (2) treat 0.26/0.05 s as lower bounds pending a
measured block-sparse device rate, and measure the compact-support (Δr=rc)
intermediate point before shipping any radius-policy change. The proposed
host-side levers (radius policy revisit; θ=0.5–0.6/leaf=20 defaults) are
well-supported.
