# 055 Impl: Scatter/Gather Efficiency Program

## Status and entry gate

**Staged `2026-08-20` (user direction). Not started.**

Entry gate: `053` and `054` complete and approved (leverage from `054`'s
retuned K* changes the M2L/nearfield balance this row optimizes against).
Individual levers from this row may be pulled forward by a `052`
escape-hatch verdict with user approval.

## Motivation

The 2026-08-20 efficiency-gap analysis: the FMM's effective efficiency is
**~10× below the brute-force kernel's 38%-of-peak**, and the losses are NOT
in P2P (already partitioned + fp32 + target-owned CSR) — they live in M2L
scatter/gather, small memory-bound GEMMs, the ~50 µs/window launch floors
(`027`), the ~0.87 ms per-GPU control floor (`028`/`029`), and refresh.
`028` established that leaf M2L is **load-bound, not atomic-bound**, so the
effort goes to the read/load side, not contention.

## Objective

A portfolio of individually A/B'd scatter/gather and launch-floor levers,
each promoted or rejected on its own measured merit, moving the FMM stages
toward the roofline that `056` will account.

## Method

Each lever: implement behind a flag → same-job A/B on the standard cases +
018 operating point → promote (with user approval for default changes) or
reject with the recorded number. Leaf-size retune after any promoted
kernel cheapening (the `054` rule).

### Lever 1 — exclusive-ownership (gather) accumulation

For M2L-accumulate and L2B: replace scattered atomic writes with
target-owned gather accumulation (the `041e` pattern applied to the
expansion stages).

### Lever 2 — read-side densification of ragged M2L gathers

Beyond the `023d` precomputed-y tables: class-contiguous/coalesced source
layouts so ragged gathers become dense reads (`028`: load-bound — this is
the side that pays).

### Lever 3 — whole-pass fusion + graph consolidation

Deeper fusion across windows and graph consolidation to shave the
~50 µs/window launch floors and the ~0.87 ms per-GPU control floor
(directly funds the multi-GPU `<= 1 ms` goal, which needs that floor
shaved).

### Lever 4 — persistent mega-kernel spike (bounded)

Untried; explicitly **time-boxed** — kill it if it fights graph capture or
the cached-window machinery. A spike report either way.

## Gates and verdict

- Per lever: same-job A/B numbers, accuracy at the 1e-3 gate, P=4 both
  precisions, contracts (exact-once, counters, zero-alloc, graph capture)
  unbroken; promotion only with the `037f`-style gate + user approval for
  defaults.
- Verdict: the promoted set, the rejected set with reasons, and the
  measured end-to-end delta feeding `056`.

## Artifacts

- Source changes + tests; `scripts/fm055_*` A/B drivers;
  `data/scatter_gather_efficiency/` per-lever CSVs + `report.md`.

## Verification

- Job IDs on all A/Bs; per-lever isolation (one flag at a time) before the
  combined measurement; critical-path pricing (not isolated stage sums).

## Recorded context (2026-08-20 staging)

**Efficiency-gap analysis (2026-08-20 discussion):** FMM effective
efficiency ~10× below brute force (041k: 3.3e11 pairs/s ≈ 38% FP32 FMA
peak); loss is in M2L scatter/gather, small memory-bound GEMMs, ~50
µs/window launch floors (`027`), ~0.87 ms per-GPU control floor
(`028`/`029`), refresh — NOT in P2P (already partitioned
singular/regularized + `037f` fp32 + `041e` target-owned CSR). Cost model
T(K) ≈ αKN + βN/K + floors; GPU optimum K=256 (`027`); cheapening α lets
the autotuner raise K*. Scatter/gather tricks list from the discussion:
port 041k opt levers into the partitioned U-list kernel (`054`);
exclusive-ownership gather (M2L accumulate + L2B); coalesced reads via
class-contiguous layouts (023/025 grouped-GEMM; **023d precomputed-y is the
fastest resident M2L**); deeper whole-pass fusion + graphs; persistent
mega-kernel (untried); leaf-size retune after each kernel cheapening.

**028 finding:** leaf M2L is load-bound, not atomic-bound → direct effort
to the load side.

**Multi-GPU stake:** the recorded 8-GPU feasibility bound (perfect 8-way
compute ~0.58 ms + ~0.87 ms control floor + ~0.2 ms comm ≈ 1.0–1.2 ms)
means the `<= 1 ms` multi-GPU goal requires shaving the per-GPU launch
floor — Lever 3 is the single-GPU end of that work.

**FMM step anchors (041a fig15, unitcube GPU best-uniform):** 1.31 ms @1e4,
7.40 @1e5, 92.3 @1e6.
