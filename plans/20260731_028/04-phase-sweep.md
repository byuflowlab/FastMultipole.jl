# 028 Phase A.3 — Full measurement campaign (heavy cluster run)

One or two H200 jobs; each n=1e6 case constructs in ~20-30 s (flat measurement) so
the sweep is hours — this is a heavy run; use the pilot's per-case timing to size
`--time`. If cases risk OOM/failure cascades, use the per-case one-process-per-job
+ resume idioms from `cuda_024b_run.sh`.

## Core matrix (prune with pilot evidence; log anything dropped)

- n = 1e6 (plus 2e5 or 3.16e5 rows for the winning configs, to tie into the 027
  record and expose the scaling exponent),
- expansion_order = 3 (P=4), LH off (one LH-on row if budget allows),
- policies: hier12 (presumptive verdict config), hier3, flat-oracle at ell=4,
- ell ∈ {4, 5, 6} hierarchical (pilot may narrow),
- K ∈ {256, 1740} — passed explicitly (K-default trap),
- strategies: dense + precomputed_y (concat only if cheap),
- precision: Float64 and Float32 (first hierarchical F32 data anywhere; watch the
  023d F32 construction anomaly — construction-only, report it separately).

## Per case (all implemented in the harness, see 02-phase-harness.md)

Three boundaries; per-stage medians; `profile_stages` telemetry; per-level M2L +
class histograms; host-alloc bytes/step; persistent bytes; counters asserted;
sampled-direct errors + F32 admissibility. Convection sanity: after k Euler steps
the re-evaluated field must still pass the error check (fresh on-device 512-sample
`direct!` reference, or reuse the 024b reference only for step 0). Also measure the
stale-tree reuse policy (skip refresh k steps) if time permits — it is the only
cheaper refresh policy that exists.

## Progress

- [x] Sweep submitted (job ids: **12997508**, 2026-07-31, `--time=06:00:00`)
- [x] Fetched into `MOR/data/feasibility_1m_10ms/` (7 row-CSVs + class companions, 27 rows total)
- [x] Failure ledger reviewed — **empty**: all 27 rows `fit=true`, no OOM, both test gates exit 0
- Notes:
  - **Matrix pruned from pilot 12996475 evidence** (details in `03-phase-pilot.md`). The core
    matrix as written above is a full cross product = 96 hierarchical cases ≈ 4 h at the
    measured ~2.5 min/case. Pilot showed ell=5 is the integer optimum at n=1e6 (ell=4/6
    extrapolate to ~430/~220 ms vs the measured 127 ms), so **ell is bracketed with one
    confirming case per side rather than swept across every other axis**. Implemented as
    tiers in `cuda_028_run.sh`; 25 cases total.
  - **Dropped, and why** (per the "log anything dropped" requirement):
    - ell=4 and ell=6 at every (K, policy, strategy, precision) combination — replaced by
      2 confirmation cases at hier12/dense/F64/K=256. Justification is the measured
      1.370 G routes/s leaf-M2L rate plus ~10.9x route growth per level; if Tier A
      contradicts the extrapolation, the ell axis must be reopened.
    - ell=6/7 were *not* dropped for memory reasons — peak was only 1.65 GB of 140 GB.
  - Tiers: **A** ell bracket {4,6} n=1e6 (2 cases) · **B** main matrix ell=5 n=1e6,
    policy{hier12,hier3} × strat{dense,precomputed_y} × K{256,1740} × TF{F64,F32}
    (16 cases, first hierarchical Float32 data anywhere) · **C** scaling tie-in
    n∈{2e5,3.16e5} ell=5 + n=2e5 ell=4 for the 027 record (3 cases) · **D** LH-on
    (1 case) · **E** stale-tree accuracy at STALE=5 (1 case) · flat baseline F64+F32
    (2 cases).
  - Each of the 7 invocations writes its own timestamped CSV (`STAMP` is evaluated per
    process), so the fetch collects 7 row-CSVs + hierarchical `.classes.csv` companions.
  - `precomputed_y` at n=1e6 may OOM or fall back (it did on the flat path in 024b, 3.28 s);
    the harness's per-case try/catch records it as `fit=false` with a note rather than
    killing the job.
