# 041e Implementation: Target-Owned Fused U-Neighbor Kernel

## Status and entry gate

**DONE `2026-08-18` — verdict REGIME-ONLY (see Result section): shape 2
`:fused_packed` ships off-by-default behind `CUDA_NEARFIELD_SHAPE` with a
`400_000`-body regime selector; shapes 1/4 killed by measurement; clear-
context approval pending.** Reviewed `2026-08-17` (two-agent review). The
Review Amendment section below rebalances the mechanism hypotheses to match
the measured record and is binding on execution; where it conflicts with the
original prose, the amendment governs.

Entry gate: `035`, `037e`, `037f`, and `041` complete and approved. This row
is independent of the unapproved 041c/041d theory artifacts: their NO-GO
results motivate prioritization but are not mathematical dependencies of the
kernel. This row blocks `042` so the adaptive-octree milestone reviews the
measured outcome.

Production changes are limited to the CUDA resident direct-nearfield path and
its cache/list metadata. The host fallback, public API, far-field operators,
adaptive U/V/W/X ownership, and FLOWVPM are unchanged. The new path must be
off by default until it passes every gate below and the user explicitly
approves a default change.

## Objective

Attempt a structural redesign of the shipped partitioned U/J nearfield:
assign each target leaf to one CUDA work owner, traverse that leaf's complete
U-source adjacency inside one fused kernel, reuse target and source data
across neighbor cells, accumulate all contributions before one final target
write, and preserve the exact shipped `gaussianerf` mathematics.

The hypothesis is **not** that atomics alone are expensive. Existing no-store
evidence bounds that lever at 0.13%. The possible gain is the combined removal
of repeated per-U-edge target setup, ragged `n_t mod 32` route tails, repeated
target loads, and per-edge accumulator retirement, plus shared-memory reuse of
source tiles across several target bodies. The mixed `PartitionedVortex`
bucket remains the primary target because it is 82--90% of nearfield kernel
time and the current kernel reaches only 39--60% of the estimated H200 vector
operation ceiling.

This row does not implement a new expansion, prune interactions, alter the
regularization cutoff, weaken U/J output, or revisit 041c/041d.

## Review Amendment (`2026-08-17`, two-agent review)

The 035 bound-ness analysis (jobs 13157887/13157931, recorded in 041b §1.1)
classifies this kernel as **compute-bound at 39–60% of the H200 vector-op
ceiling with DRAM at only 2–12% of peak**. The shipped organization is
warp-per-U-edge with lanes mapped one-to-one to target bodies, each lane
serially looping all sources (`_cuda_direct_pairs_*` kernels,
`src/translate_batched_cuda.jl:1875/2159/2256/2431`). Four consequences
rebalance this row's hypotheses:

1. **Two primary hypotheses — stated as hypotheses, not established
   mechanisms:**
   - *(a) Target-lane repacking.* Fusing all source neighbors does **not**
     by itself recover ragged-tail underfill: under any lane-per-target
     mapping, `(32*ceil(n_t/32) - n_t) * sum(n_s)` lane-slots are lost
     whether the sources arrive as one fused stream or as separate edges (a
     lane-utilization ceiling; masked lanes still occupy lockstep issue
     slots). Recovery requires *repacking* — the shape-2 persistent-CTA
     multi-leaf packing (priced against its divergent per-lane source
     streams) or the new shape 4 below. Headroom concentrates in
     low-occupancy leaves: wake adaptive `K_max=64` mean leaf population is
     17.1 (`data/fm041a_leafpop.csv`), i.e. ~47% single-wave underfill.
   - *(b) Mixed-predicate coherence* in the mixed `PartitionedVortex`
     bucket (82–90% of nearfield kernel time), where the per-body-pair
     `rho <= rho_t` predicate makes warps pay both paths when lanes
     disagree. Neither hypothesis is pre-proven: the ballot/queue mechanism
     was a **measured loss everywhere** and sub-Morton ordering raised
     homogeneity 0.826→0.887 for only a 2–4% gain (032a Stage-C record,
     `032a-impl-split-nearfield-comparison.md:303-338`) — both within the
     lanes=targets organization. Stage A must bound each hypothesis's
     ceiling separately, without double counting between them.
2. **Shape 4 (new, first-class): dynamic lanes-per-target source-splitting.**
   Lanes traverse sources for one or a few fixed targets, with a
   12-component warp-shuffle U/J reduction at the end; measured crossover by
   `n_t`, total source count, and precision. Its coherence experiment:
   per-target source ordering keyed on `r/sigma_s` makes predicate outcomes
   monotone along the lane axis — qualitatively unavailable to any
   target-lane ordering — but it is an *experiment*, and must beat the
   current sub-Morton coherence, hardware predication, the negative ballot
   result, and its own ordering cost. The canonical Stage-B comparison is
   `lanes=targets, fixed source` (current) versus `lanes=sources, fixed
   target` (new), each organization given its best practical ordering.
3. **Demoted to bounded controls (one A/B each, not design pillars):** the
   single final write (028 measured 0.13% for removing *all* stores) and
   shared-memory source tiling / target-load reuse (DRAM is at 2–12% of
   peak, and warp-identical source loads are already cache/broadcast
   friendly). No L2/Morton block-assignment work.
4. **Ceiling honesty:** the 39–60%-of-ceiling residual must not be presented
   as fully decomposable into these levers — it also contains dependency
   chains, rsqrt/special-function throughput, issue limits, register
   effects, and op-count-model uncertainty. Stage A reports bounded
   attribution, not exact accounting.

CSR feasibility note: the uniform hierarchical direct-pair generation
enumerates target-major within each window (`g = c*kn + k`,
`src/translate_batched_cuda.jl:6553-6560`), so a no-scatter fast path is
plausible there once cross-window concatenation is verified. It is **not**
established for the adaptive DTR-produced U list, and the `:classsplit`
bucket compaction uses atomic claims that destroy target-major order, so
per-target class subranges may still need the priced counts/prefix/scatter
path. "Near-zero refresh" may be claimed only for the verified fast path.

## Current baseline and required control

Compare against the current shipped winner at identical geometry and in the
same job:

- adaptive and uniform resident paths as selected by the 041a evidence;
- `PartitionedVortex` with the user-approved 037f `:fp32` g/h mode where it
  applies (`:shipped` control also retained);
- current `:classsplit` / `:classsplit_ballot` winner chosen by existing
  policy, including graph capture and overlap;
- production `near_radius2=5`, P=4 primary, with P=8 and both Float32 and
  Float64 as contract checks.

Never compare against a pre-037f or non-adaptive stale baseline.

## Design contract

### 1. Target-owned U adjacency

Represent the ordered U list as a device-resident target CSR:

```text
target_leaf_offsets[1:n_target_leaves+1]
u_source_leaf[1:n_u]
u_edge_class[1:n_u]          # singular / regularized / mixed when enabled
```

Reuse an already target-major list without a scatter when production ordering
proves it. Otherwise build counts, prefix offsets, and a stable or
order-insensitive scatter into construction-sized persistent buffers during
the existing refresh. Price the scan/scatter on the complete critical path.
No recurring allocation, host round trip, or per-step capacity growth is
permitted.

Every target leaf appears under exactly one work owner. Its CSR interval is
the exact set of ordered U edges previously launched independently. The
existing `r2 > 0` self exclusion remains unchanged.

### 2. Fused kernel shape

Prototype at least these bounded shapes and select by measurement:

1. one CTA per target leaf, processing target bodies in warp-sized waves;
2. a persistent CTA queue for sparse/ragged target leaves, while retaining
   exclusive target ownership;
3. source-cell tiles loaded cooperatively into shared memory and reused by
   all active target warps before advancing the CSR iterator (bounded
   control only, per the Review Amendment);
4. dynamic lanes-per-target source-splitting: lanes traverse sources for one
   or a few fixed targets with a final 12-component warp-shuffle U/J
   reduction, with a measured crossover by `n_t`, total source count, and
   precision, and optionally per-target `r/sigma_s`-ordered sources for
   monotone predicate runs (Review Amendment item 2). A per-lane
   multi-target register-tiling factor (e.g. 2 targets ~ 24 F32 accumulator
   registers) is a knob on shape 1, measured against occupancy.

Each lane accumulates the same 3 U plus 9 J values required by the shipped
vortex path. Warp/block reductions and final stores must be deterministic in
ownership, though bitwise equality is not required where accumulation order
changes. Register pressure, shared-memory occupancy, active-warps/SM, target
wave count, and source-tile reuse are mandatory measurements — instrumented
via compiler register/shared-memory reports, PTX inspection, occupancy
arithmetic, nsys timing, and explicit operation/lane-slot models, because
NCU hardware counters are blocked on unprivileged H200 jobs
(`ERR_NVGPUCTRPERM`, 041b §1.1); privileged profiling may be used only if
separately authorized. Do not select a
shape from isolated throughput alone; evaluate its full resident lifecycle.

The kernel may write a target body once only because no other CTA owns that
target leaf. If implementation constraints require an atomic final write,
record why; it does not invalidate the experiment, but it removes one
secondary benefit.

### 3. Branch organization

Preserve all existing pair decisions and exact g/h implementations. Measure:

- a single fused traversal with the current per-pair predicate;
- target-local CSR subranges for pure singular, pure regularized, and mixed
  U edges, with branch-free loops for the pure ranges;
- the existing ballot organization inside only the mixed range when it is
  profitable.

Classification metadata must be produced by the existing exact cell/AABB
rules. No pair may be omitted merely because it is singular outside the
regularization ball: the singular Biot--Savart interaction is still required.
Do not resurrect 037e's failed point/AABB predicate as an unpriced inner-loop
test.

### 4. Exact-once and multi-system behavior

Prove that grouping ordered edges by target is a permutation of the current U
list. Cross-check explicit ordered body-pair IDs against the existing
standalone and production painters on self, face, edge, corner, outer-shell,
unequal adaptive-level, empty-child, boundary, coincident, static-source, and
static-target cases. Assert zero omissions and duplicates.

Support distinct source and target systems, coincident source/target systems,
gradient-disabled output, and all currently supported vortex policies. Any
unsupported combination must select the shipped fallback automatically, not
throw during a resident step.

## Gated execution

### Stage A -- read-only census and roofline

Before kernel work, extract from existing 041/041a/035 records and a
deterministic list replay — all local and read-only (<= 4 threads), using
`data/fm041a_leafpop.csv` / `data/fm041a_gpu_widen.csv`, the standalone
census builders (`scripts/adaptive_octree_verify.jl`), and the existing H200
homogeneity telemetry (`cuda_nearfield_homogeneity`,
`src/translate_batched_cuda.jl:4852`; `cuda_nearfield_pair_aabb_stats`,
`:4960`) where records exist:

- exact target-leaf U degree and target occupancy distributions;
- exact lane-slot underfill arithmetic per the Review Amendment formula;
- source bodies visited per target leaf and source-tile reuse;
- fraction of target leaves requiring 1, 2, or more target waves;
- pure/mixed edge runs per target CSR, and body-pair predicate coherence
  from deterministic, checksummed *sampling* of actual snapshots under both
  lane mappings (exact full replay only on census-scale cases — no
  million-particle predicate sweep);
- current repeated target loads and per-edge accumulator retirements.

The output must bound five components separately, de-double-counted:
(1) underfill recoverable by repacking/source-splitting, (2) mixed-predicate
divergence under the lanes=targets and lanes=sources mappings, (3) per-edge
setup/retirement, (4) source/target load reuse, and (5) irreducible pair
arithmetic — as bounded attribution of the 39–60% ceiling residual, not
exact accounting (Review Amendment item 4), explicitly excluding the
already-falsified atomic-only benefit.

Record a CONTINUE/KILL decision. A kill is allowed only if the optimistic
de-double-counted sum of components (1)+(2)+(3) cannot reach 10% nearfield
and 5% complete-solve improvement on a material case. Otherwise implement
Stage B.

### Stage B -- bounded CUDA prototype

Implement the target CSR and the three bounded kernel shapes behind an
internal runtime policy. Pre-register the H200 microbenchmark matrix before
running it. Use observed occupancy/degree quantiles rather than arbitrary
synthetic sizes, and include fragmented/mixed target adjacency.

Kill shapes that cannot beat the current isolated nearfield kernel by at
least 10% at equal work and accuracy. Retain the best one only for Stage C.

### Stage C -- full resident A/B

Run same-job warmed H200 comparisons on deterministic cube, wake, and
multiscale/rotor cases at `n in {1e5, 1e6}`, both precisions, P in `{4,8}` for
contract coverage, the 041a adaptive `K_max` winners, and the corresponding
best supported uniform geometry. Include the sigma-heterogeneous case.

Report CUDA-event nearfield time, graph-overlapped lifecycle time, full step
time, refresh/list construction, memory, allocations/transfers/counters, and
sampled-direct U/J accuracy. Attribute gains among target reuse, source-tile
reuse, ragged-lane reduction, branch organization, and final-write changes.

## Acceptance and promotion gates

Before marking 041e Done:

1. exact-once CSR permutation and both painters pass all adversarial cases;
2. velocity relative RMS remains `<= 1e-3`; U/J deltas versus the shipped
   control remain within the established accumulation-order tolerances;
3. at least 10% nearfield critical-path reduction and 5% complete overlapped
   solve reduction on a material `n=1e6` case — the **rotor** is the primary
   promotion case (92–96% nearfield share, and the regime that survives a
   future uniform-sigma VIC default); cube/wake serve as regression and
   regime-boundary controls;
4. no supported case regresses by more than 3% under the automatic policy;
5. refresh is allocation-free, device-resident, capacity-bounded, and graph
   capture compatible, with existing transfer counters flat;
6. the selector falls back to the shipped kernel for degrees, occupancies,
   policies, or hardware regimes outside the measured win envelope;
7. all comparisons use the current 037f-enabled same-job baseline.

If only a measurable occupancy/degree regime wins, ship an off-by-default
regime selector and record **REGIME-ONLY**. If the gates fail, record whether
the cause is insufficient reuse, register pressure, shared-memory occupancy,
ragged target ownership, CSR refresh, mixed-branch divergence, or overlap
with work already removed by 037f. A general default change requires explicit
user approval after the evidence is reviewed.

## Stage A record (`2026-08-18`)

Read-only census complete: `scripts/fm041e_target_owned_censusA.jl`,
`data/target_owned_nearfield/censusA_*.csv` (+ report, manifest, checksums).
Standalone exact replay (sticky-demotion U lists, `rho_t=4.789`, `q=5`) on
cube/wake/sigma_multiscale proxies at `n=1e5` and the real DJI-9443 rotor at
`n=1e5` (K=64/256) and `n=1e6` (K=64); 200 seeded sampled leaves per case
for the both-mapping predicate replay; brackets per the registered op model.

Component bounds (optimistic, de-double-counted via a direct shape-4 ops
simulation): rotor `n=1e6` K=64 — combined (1)+(2)+(3) = **32.7%** of
nearfield pair-op cost (c1 underfill 36.1%, c2 divergence 7.7%, c3 per-edge
overhead 1.2%); rotor `n=1e5` K=64 34.1%; K=256 10.6%; wake 55.9%; cube
35.7%; sigma_multiscale 50.6%. c4 (load reuse) confirmed negligible as a
bytes-bound control. Notable negative: per-target `r/sigma_s`-sorted source
coherence barely beats the shipped mapping on the rotor (divS ≈ divT
0.12–0.16) because per-target source lists are short (median Σn_s ≈ 221 ⇒
~7 warps ⇒ boundary-warp fraction ~0.15) — shape 4's value is underfill
repacking, not predicate coherence; the coherence experiment stays but with
tempered expectations.

**Decision: CONTINUE to Stage B** (the optimistic sum clears the 10%
nearfield bar on every material case; rotor nearfield share 92–96% makes
the 5% complete-solve bar follow).

## Result (`2026-08-18`): **REGIME-ONLY**

Stages B and C ran as three same-job H200 rounds (jobs 13193465 / 13193488 /
13193492, plus the Stage C crossover round 13193508; node m13h-1-1). All
artifacts under `data/target_owned_nearfield/` (`stageB_bench.csv` rows keyed
by job id) and `test/cuda_radix_fused_nearfield_test.jl` (387/387 pass,
including CSR permutation, adversarial cases, uniform-cache fallback parity,
`:lut`/invalid-shape guards, and the 023 counter contract).

**Stage B.** Target-major U CSR on the adaptive path (deterministic
(target-slot, index) key sort at occupancy epochs, V-scratch reuse, zero
allocation) plus three kernel shapes behind the construction/graph-baked
`CUDA_NEARFIELD_SHAPE` Ref (default `:pairs`, silent shipped fallback):
shape 1 `:fused_cta`, shape 4 `:fused_srclanes`, shape 2 `:fused_packed`
(thread per target body over the dense leaf-major order — zero lane
underfill by construction). Kill rule applied: shapes 1 and 4 cannot beat
the shipped kernel by 10% anywhere material (best: `:fused_cta` −7.1% rotor
1e6 F32; `:fused_srclanes` loses everywhere — its coherence experiment
confirmed Stage A's tempered prediction). **`:fused_packed` survives**:
isolated nearfield rotor 1e6 −23.8% (F32) / −23.2% (F64); cube 1e6 −26.6% /
−47.3%; wake 1e6 F64 −9.8%.

**Stage C.** Full-lifecycle A/B and win envelope: cube 1e6 lifecycle
**−10.9% (F32) / −27.2% (F64)** — passes both promotion bars; wake 1e6 F64
−8.4% complete (near just under 10%); rotor 1e6 complete only −2.0/−2.6%
despite the −24% kernel win, because the adaptive-path rotor nearfield is
only ~10% of lifecycle (2.65/26.8 ms; the task file's 82–90%/92–96% shares
are the UNIFORM partitioned configuration — the adaptive sticky-demotion
path converts most of that direct work into M2T/S2L, so the premise does
not transfer). Crossover (n=316,228): cube already wins (−24/−48% near);
wake F32 regresses (+16% near); rotor neutral (−5/−7% near, lifecycle
noise). All n=1e5 rows regress (worst wake F32 +54% near). Accuracy:
sampled-direct U deltas vs the shipped control are zero to accumulation
tolerance on every row (387/387 parity); counters flat; zero per-step
allocation; graph-capture compatible.

**Selector (gate 6).** `CUDA_NEARFIELD_FUSED_MIN_BODIES = 400_000`
(construction/graph-baked like the shape Ref): below it the shipped
organization runs even when a fused shape is selected, which zeroes every
measured regression (conservatively forgoing the small cube-316k win).
Under the automatic policy no supported case regresses >3% (gate 4 ✓).

**Attribution** (rotor 1e6 F32 ladder): fused traversal + one retirement
per target (`:fused_cta`) ≈ −6%; dense body repacking (`:fused_packed`)
≈ a further −19% — Stage A's optimistic 32.7% ceiling partially realized
(≈24%), the gap being leaf-boundary warp divergence and own-leaf edge-loop
divergence. The mixed-predicate lever (c2) does not exist on the adaptive
path (regularized-everywhere under sticky demotion has no per-pair branch);
the final-write lever was excluded up front (028: 0.13%).

**Gate walk:** (1) exact-once ✓; (2) accuracy ✓ (deltas ~0; the rotor-1e5
bench rows sit at 1.43e-3 in BOTH control and fused — untuned bench
geometry, not a fused regression); (3) ≥10% near + ≥5% complete on a
material 1e6 case: **cube passes both; the rotor passes only the nearfield
bar** → REGIME-ONLY, not GO; (4) ✓ via selector; (5) ✓; (6) ✓; (7) ✓
(same-job, 037f `:fp32` default on both sides). Defaults unchanged
(`:pairs`); enabling `:fused_packed` on large-n adaptive configurations is
recommended to the `042` review as an opt-in.

**Known limitations recorded:** fused shapes engage on adaptive caches only
(uniform caches take the verified silent fallback; a uniform-path CSR fast
path remains future work); `:lut` g/h mode unsupported (throws with
guidance); the `sigma_multiscale` log-uniform 1e6 bench proxy exceeds
first the DTR frontier then (with vfac=500) the U capacity for EVERY shape
including the shipped control — a pathological-case capacity limitation of
the bench proxy, not of the fused path; the rotor covers the
sigma-heterogeneous 1e6 slot.

## Clear-context approval (`2026-08-18`)

**APPROVED (REGIME-ONLY verdict upheld).** Independently re-verified: the
CSR build is a correct deterministic (target-slot, emission-index) permutation
of the slot-mapped U list (disjoint per-thread boundary fill covers empty
leaves; the `n_u + 1` prefill covers slots above the last occupied target;
epoch check blocks stale CSR use; capacity asserts are loud at refresh, zero
recurring allocation); all three fused kernels preserve the shipped
`_direct_pair_ug/_ugh` math, `i == j` self-exclusion, and `r2 > 0` guard, with
atomic final writes justified by concurrent far-stream M2T; the `:fused_srclanes`
shuffle reduction operates on warp-uniform loop bounds (FULL_MASK safe); the
dispatch falls back silently on non-adaptive caches, unarmed CSR, and
`n < 400_000` (`:lut` throws loudly with guidance — documented, parity with the
existing unbinned `:lut` guard). Every headline number matches
`stageB_bench.csv` to the digit (rotor 1e6 near 2.6498→2.0186 F32 /
4.1926→3.2206 F64; cube 1e6 lifecycle 11.65→10.383 / 21.685→15.786; crossover
cube −24/−48% near, wake F32 +16%, rotor near −5/−7% with lifecycle noise; all
n=1e5 rows regress, worst wake F32 +54.0%; accuracy columns identical
control-vs-fused per row; counters flat; sigma_multiscale 1e6 fails capacity
asserts for the shipped control too — bench-proxy pathology as recorded). Test
file covers CSR permutation on real lists, synthetic empty-slot boundary fill,
sigma-demotion unequal-level structure, uniform-cache fallback, guards,
counters, P=4/P=8, both precisions, and default restoration. Gate walk and
selector default confirmed; defaults unchanged (`:pairs`). Minor non-blocking
notes: the gate-1 adversarial geometries are covered via real-list permutation
checks rather than literal per-geometry enumeration, and the offsets kernel's
per-thread empty-gap fill is O(gap) serial (epoch-only, leaf-capacity-bounded).

## Required artifacts

- production CUDA/cache changes under the existing `*_batched_cuda.jl` and
  core-container placement rules;
- focused host-free CUDA tests plus exact-once/graph/allocation/counter tests;
- `scripts/fm041e_target_owned_*.jl` benchmark and analysis drivers;
- `data/target_owned_nearfield/` manifest, census, calibration, raw compact
  measurements, selector table, report, and checksums;
- this task's Result section with GO, REGIME-ONLY, or NO-GO.

Do not commit particle-scale U lists or body-pair traces.
