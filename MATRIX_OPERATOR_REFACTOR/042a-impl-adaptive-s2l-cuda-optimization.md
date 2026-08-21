# 042a Implementation: Adaptive CUDA S2L Optimization

## Status and Entry Gate

**Proposed follow-on item; not started.**

Entry gate: `042` complete and clear-context approved.

## Motivation

The `041a` widened H200 stage breakdown identifies adaptive X-list
source-to-local (S2L) as the largest adaptive-only stage on the uniformly
sparse wake: about **22.2 ms** at `n=1e6`, plus 7.2 ms for its M2T dual. This
explains most of the adaptive/deep-uniform lifecycle gap. The shipped CUDA S2L
kernel is block-per-X-pair and thread-per-source-body, with atomic coefficient
accumulation. It preserves correctness but repeatedly initializes harmonics,
loads target metadata, and retires the same target local expansion across
independent X entries.

This row optimizes the execution shape only. It must not change the adaptive
tree, X list, S2L mathematics, Lamb–Helmholtz convention, expansion order, or
accuracy budget.

## Objective

Build and measure a target-owned/batched CUDA S2L path that groups the existing
X-list entries by target local expansion, reuses source/target data, reduces
atomic retirement, and specializes the common `P=4` workload without creating
new operator tables or a new interaction policy.

## Required Work

1. **Baseline census and profile.** Before optimizing, record X-entry counts,
   source populations, entries per target, coefficient count, achieved
   occupancy, register/shared-memory use, atomic transactions, and S2L time for
   the `041a` unitcube, wake, and multiscale100 cases at `n=1e5` and `n=1e6`.
   The `n=1e6` F64 wake row is the primary target; include the shipped F32
   configuration and `P=8` as robustness points.
2. **Target-major X representation.** Build a capacity-sized target CSR or an
   equivalent deterministic grouping from the exact shipped X list on
   occupancy epochs. Reuse existing sort/scan scratch where safe; otherwise
   size buffers at construction. Ordinary recurring steps must allocate
   nothing and transfer no routes or expansion data through the host.
3. **Kernel variants.** Measure at least:
   - shipped block-per-X-pair S2L baseline;
   - one target-owned CTA/warp design that traverses all X sources for a target
     local expansion and retires each coefficient once where ownership permits;
   - one bounded `P=4` specialization (fixed coefficient shape/unrolled
     harmonic recurrence or an equally explicit alternative).
   Multi-CTA splitting is allowed for high-degree targets, but its reduction
   and capacity costs must be included. Do not fuse S2L with M2L/L2L unless a
   profile first demonstrates that the added coupling is necessary.
4. **Supported physics.** Preserve `Point{Source}` scalar operation and
   `Point{Vortex}` Lamb–Helmholtz operation, including
   `P_chi=P_phi+1`, Float32/Float64, and `P=4/8`. Preserve the documented
   LH+hessian guard unless that separate missing operator surface is explicitly
   staged by the user.
5. **Selection and fallback.** Keep the optimized shape selectable and off by
   default through measurement. If promoted, select it only in its measured
   regime and retain automatic fallback to the shipped kernel for unsupported
   orders, body types, small workloads, or unfavorable X-list shapes.

## Constraints

- Exact X-list endpoints and exact-once U/V/W/X coverage are unchanged.
- No new M2L/S2L operator tables and no change to `025`, `038`, or `040`
  mathematics.
- New core types or persistent buffer fields belong in `src/containers.jl`;
  CUDA execution stays in `src/translate_batched_cuda.jl` and, if grouping is
  built with the adaptive tree, `src/tree_batched_cuda.jl`.
- Preserve graph capture, occupancy-epoch invalidation, fixed-capacity overflow
  throws, `recenter!`, route/operator transfer counters, and warmed zero-device-
  allocation behavior.
- GPU benchmarks run on one H200 through the standing cluster workflow; local
  verification uses at most four threads.

## Verification and Promotion Gates

- Re-run the complete `test/cuda_radix_adaptive_test.jl` surface and add direct
  baseline-vs-optimized S2L tests for Float32/Float64, `P=4/8`, scalar and LH
  vortex sources, empty/tiny/skewed X groups, epoch rebuilds, and capacity
  failures.
- Baseline and optimized outputs must satisfy the existing sampled-direct
  velocity RMS gate `<=1e-3`. Float64 optimized-vs-baseline expansion/output
  differences must remain within the established adaptive CUDA accumulation-
  order tolerance; Float32 must not consume a material fraction of the physical
  error budget.
- Same-job H200 A/B, warm median of at least five after warm-up, reporting both
  S2L-stage and complete recurring-step time. Include graph-captured and
  serialized stage measurements so overlap cannot be mistaken for kernel gain.
- **Promotion requires both** at least **25% lower S2L time** and at least
  **3% lower complete adaptive recurring-step time** on the `n=1e6` F64 wake,
  with no more than 3% complete-step regression on unitcube, multiscale100,
  F32, `P=8`, or `n=1e5` after selector fallback. Otherwise retain the shipped
  default and record a NO-GO or regime-only verdict.

## Deliverables

- Production implementation and focused CUDA tests.
- Pre-registered H200 benchmark/submit/fetch scripts under
  `MATRIX_OPERATOR_REFACTOR/scripts/`.
- Raw profiles and A/B CSVs plus a short verdict under
  `MATRIX_OPERATOR_REFACTOR/data/adaptive_s2l_optimization/`.
- Completion notes stating the selected kernel shape, selector threshold,
  accuracy results, stage/full-step speedup, memory/capacity delta, and whether
  the adaptive-vs-wake gap materially closed.

## Acceptance

The correctness, lifecycle-contract, and measurement gates above pass; the
task records a promotion/regime-only/no-go verdict; the row is marked Done and
receives independent clear-context approval before `043` starts.
