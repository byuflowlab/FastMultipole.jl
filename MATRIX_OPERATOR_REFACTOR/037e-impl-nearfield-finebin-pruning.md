# 037e Nearfield Direct-List Pruning (Fine Bins / Exact Geometry)

## Status and Entry Gate

**Staged by user request 2026-08-14; not started.**

Entry gate: `037b` Done and Approved (met 2026-08-14). Does not gate `038`
and may proceed in parallel with it. Coordinated with `037f` (one effort,
adjacent surfaces); neither blocks the other.

## Motivation

The shipped partitioned nearfield covers the regularized ball `r <= rho_t
sigma` with a cell stencil, which over-covers geometrically: corner cells
and quantization slop put the candidate:accepted pair ratio well above 1
(sphere/enclosing-cube volume alone is ~0.52). `037a` validated the remedy
on the two-pass correction list — exact source-directed AABB-gap predicates
plus fine-bin traversal removed 56–65% of candidates with bit-identical
errors and cut that solve 12.9% at `n=1e6`. This row points the same
mechanism at the production partitioned direct list, where the nearfield is
the measured dominant cost at `n=1e6`.

## Scope

- Apply exact source-directed AABB-gap predicates (the `037a` production
  rule: exact AABB minimum-gap test, not center-distance + half-diagonal)
  and/or a fine-bin/sub-Morton candidate index to the partitioned direct
  nearfield route generation and/or kernel traversal. Choose the granularity
  (route-level pruning vs bin-level traversal vs both) by profiling; record
  the decision basis.
- The regularized/singular split within the direct list is unchanged; only
  which candidate pairs are visited changes. Pairs inside `rho_t sigma`
  must be visited exactly once (exact-once coverage is a hard invariant —
  brute-force verify on cube, wake, and rotor samples).
- Preserve: capacity contract / zero recurring allocation, transfer
  counters, graph capture, device residency, the general directed
  source/target nearfield path, and CPU-path behavior (this is a GPU lever;
  the host path may adopt it only if free).
- Off by default behind an option/`Val` flag with the shipped path retained
  as control, mirroring the `037a` AABB-lever pattern.

## Benchmark and Test Plan

- Pre-register (in this file, before submission) an H200 screen: cube, AR-5
  wake, and rotor wake at `n = 1e5` and `1e6`, Float32 and Float64, warmed
  U/J medians with same-job baseline anchors, candidate/accepted pair
  counts, per-stage timings, allocations, counters.
- Accuracy: sampled velocity RMS `<= 1e-3` on every row against the
  checksummed references (`033`/`037b` sets); errors must be bit-identical
  to the control where the visited accepted-pair set is unchanged; Jacobian
  RMS logged as diagnostic.
- Tests: host/device parity of the predicate, exact-once brute-force checks,
  `P=4` coverage, graph replay, counter stability, zero recurring
  allocation, out-of-box/recenter behavior. Local runs `<= 4` threads.
- Promotion gate (user approval required for any default change): `>= 5%`
  faster end-to-end U/J on a material wake or rotor case, no `> 3%`
  regression on any other measured case. Otherwise ship opt-in or record as
  falsified with the measured ceiling.

## Theory/Artifact Dependencies

`031a` (partitioned nearfield contract), `025` (routing invariants,
untouched), `037a` work record (AABB predicate + fine-bin mechanism and its
measured behavior on the correction list).

## Pre-Registered H200 Screen (registered 2026-08-14, before submission)

Mechanism under test (decision basis recorded here): on the partitioned path
every direct pair contributes to the field (singular outside the ball), so no
candidate pair can be dropped outright; the prunable quantities are (E1)
expensive-branch candidacy and warp divergence in the mixed
`_nearfield_pair_bucket` bucket, and (E2, sigma-directed only) cell pairs
re-routable to leaf M2L. E1 is implemented (`CUDA_NEARFIELD_PAIR_AABB`,
off by default): per 32-lane target block, an exact point-vs-source-cell
AABB reachability vote (the 037a predicate with per-source-cell
`sigma_max`) converts provably all-singular blocks to the exact singular
inner loop, FP-identical to the split branch's own singular outcome —
the accepted regularized-pair set is unchanged, so errors are the control's.
E2 is implement-only-if: the E0 scoping run must show `e2_share >= 0.10`
(body pairs in pure-singular cell pairs at M2L-admissible offsets) on at
least one case; otherwise its ceiling is recorded and E2 is not built.

- Case grid: `scripts/fm037e_cases_screen.txt` — 24 rows; cube (`ell=4`
  at `1e5`, `ell=5` at `1e6`, `q=12`), AR-5 wake (`ell=5/6`, `q=6`), rotor
  (`ell=5/6`, `q=6`); `n = 1e5, 1e6`; Float32 and Float64;
  `kernel=partitioned`, `expansion_order=4` (literature P5),
  `rho_t=3.668`, `strategy=dense`, `profile=1`. Each `*_paabb` row has a
  same-job `*_anchor` control at identical geometry (the 037b anchor
  convention).
- Driver: `scripts/benchmark_035_gpu.jl` (new `pair_aabb` key; Ref set
  before cache construction per the graph-bake contract), warmed U/J
  medians (`FM035_REPS=15`, warmup 2), checksummed `033` references,
  per-stage timings, allocation and transfer-counter columns, and the new
  telemetry columns `nf_mixed_pairs` / `pair_aabb_tested` /
  `pair_aabb_skipped` recorded on every split-kernel row.
- E0 scoping: `scripts/fm037e_scoping.jl` full grid on a cluster CPU node
  (`scripts/cpu_037e_scoping_run.sh`) — bucket-resolved cell/body-pair
  census, E1 skip ceilings at pair/32-lane/16-lane granularity, per-cell
  `sigma_max` spread, and the E2 re-route ceiling (`e2_share`, floor
  stencil `|o|^2 > 3`).
- Preflight on the same job: `test/cuda_radix_nearfield_binning_test.jl`
  (which now contains the 037e testsets: bitwise/twin-run identity at
  P4/P8 x F32/F64 x classsplit/classsplit_ballot, gradient-only branch,
  graph replay with the flag baked, 023 counters flat, step-allocation
  equality, telemetry monotonicity `skipped <= tested == mixed`).
- Gates (promotion; any default change additionally requires explicit user
  approval): velocity RMS `<= 1e-3` on every row; `>= 5%` faster
  end-to-end U/J on a material wake or rotor case; no `> 3%` regression on
  any other measured case. Expected-identity check: `u_rel_rms` of each
  `*_paabb` row must equal its anchor to accumulation-order tolerance.

## Work Record

**Done 2026-08-14; E1 falsified as a default candidate (ships opt-in); the
E2 sigma-directed ceiling measured at ~60% of rotor direct pairs and routed
to row `038` (whose scope already includes the per-cell sigma geometry
gate). No default changed.** H200 job `13170768` (stage e; screen CSV of
record `data/flowvpm_gpu_campaign/fm037e_screen.csv`, scoping census
`fm037e_scoping_13170768.csv`, analyzer
`scripts/analyze_037ef_screen.jl`).

Implemented (all committed on `matrix-ops`):

- `CUDA_NEARFIELD_PAIR_AABB` (off by default): warp-level exact
  target-point/source-cell AABB reachability vote in the mixed
  `_nearfield_pair_bucket` bucket, on both the dedicated bucket kernel
  (`_cuda_direct_pairs_mixed_aabb_kernel!`) and the ballot-queue kernel;
  provably all-singular 32-lane blocks run the exact singular inner loop
  FP-identically to the split branch's own singular outcome. Shared scalar
  predicate `_nearfield_point_aabb_reach` (host/device/scoping single
  source). Diag slots 11/12 + `cuda_nearfield_pair_aabb_stats` telemetry
  (never in the production APPLY launch). Driver key `pair_aabb` + CSV
  telemetry columns.
- E0 scoping census `scripts/fm037e_scoping.jl` (+ CPU sbatch wrapper);
  full-n run on the job. Caveat: the census builds its cache from raw
  bounds without FLOWVPM's 10% padding, so cube/wake `n=1e5` failed the
  adequacy gate inside the census (production geometry passes); the four
  surviving configs carry the decision data.
- Tests (all green on H200, first attempt): CUDA nearfield-binning suite
  grew 307 -> 358 (bitwise/twin-run identity at P4/P8 x F32/F64 x
  classsplit/classsplit_ballot, gradient-only branch, graph replay with the
  flag baked, 023 counters flat, step-allocation equality, telemetry
  monotonicity), host predicate testset 8362 assertions (runs without
  CUDA).

E0 census (full n, decision basis):

| case | n | mixed bp share | E1 blk32 skip (of mixed bp) | E2 share (all bp) |
|---|---:|---:|---:|---:|
| cube | 1e6 | 0.99 | 0.023 | 0.0 |
| wake | 1e6 | 1.00 | 0.003 | 0.0 |
| rotor | 1e5 | 0.41 | 0.876 | **0.587** |
| rotor | 1e6 | 0.37 | 0.855 | **0.627** |

Uniform-sigma cases have no sigma-directed slack (the shipped stencil is
gate-minimal for the global `sigma_max`); the rotor's ~18x per-cell
`sigma_max` spread creates both ceilings.

Screen verdict (12 candidate rows vs same-job anchors, warmed U/J
medians): accuracy identical to anchors on every row (accepted-pair set
unchanged, as designed; all `<= 1e-3`); performance FAILS the promotion
gate — best material row `+2.4%` (rotor `1e6` F64), worst regression
`-17.3%` (wake `1e5` F32); F32 rows regress broadly (predicate/blocked-
traversal overhead exceeds the removed divergence). Counters flat and
per-step allocation flag-independent on all rows.

E2 disposition: the pre-registered `>= 10%` build criterion fired (rotor),
but E2 is not built here. Decision basis: (i) it requires the leaf-M2L
route/class extension plus a new M2L-admissibility derivation (rotor F32
`n=1e6` headroom is only `~3e-4` and close-range M2L error lands exactly
there; F64 rotor has 10x headroom); (ii) row `038`'s adaptive octree
already scopes the per-cell sigma geometry gate that makes this mechanism
native rather than bolted onto the rigid uniform grid. The measured
ceiling (11.9B of 19.0B rotor `1e6` direct body pairs re-routable) is the
`038` entry-gate evidence.

Observation for future screens: stage-e vs stage-f same-config anchors
differ by 20-45% across the two H200 jobs/nodes (e.g. wake `1e6` F64
258.8 vs 178.8 ms) — cross-job comparisons are invalid; the same-job
anchor convention is load-bearing.
