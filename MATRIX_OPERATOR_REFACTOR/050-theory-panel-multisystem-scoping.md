# 050 Theory: Panel/Multi-System GPU Scoping (decision row)

## Status and entry gate

**Staged `2026-08-20` (user direction). Not started.**

Entry gate: `046` and `049` complete and approved (the unified branches
exist and the `049` per-pass budget table says which constraint binds).
Decision/scoping row: `theory/`, `scripts/`, `data/` artifacts only — no
production implementation (that is `051`).

## Motivation

User direction: resolve multi-system GPU generalization vs individual
system-on-system evaluations — "check if system-on-system looks
significantly easier before doing it." FLOWPanel's per-step influence
structure already runs **three separate fmm! passes with targets ≠ sources**
(`FLOWPanel_simulate.jl:673-712`), while the GPU radix path has a v1
restriction requiring `targets === sources`. This row prices the options
with recorded facts plus the `049` budget table, and names the `051`
implementation shape.

## Objective

A priced verdict among options A/B/C below, plus a scoping of the panel GPU
kernel itself and where flowpanel-20260817's FmmPlan/NearfieldInfluenceCache
layers fit, ending in a named implementation shape for `051`.

## Method

### Stage 1 — price the options

- **(A) Lift the radix v1 `targets===sources` restriction**
  (`_assert_radix_targets_are_sources`,
  `translate_batched_resident.jl:1735-1743`; `target_bodies` is literally
  aliased to `source_bodies` at `translate_batched_cuda.jl:5447,:5526`) to
  support distinct target sets incl. panel centers/probes. Enumerate every
  surface the alias touches (downward pass, L2B scatter, nearfield target
  indexing, counters, capacity) and estimate the diff.
- **(B) Individual system-on-system GPU evaluations** mapped onto
  FLOWPanel's existing 3-pass structure — **a-priori favorite: least
  invasive; the CPU path already runs separate passes** with different
  kerneloffsets/derivative switches per pass. Measure/estimate the per-pass
  overhead (separate trees/caches, refresh, launch floors) at the 018
  operating point; choose A only if B's measured pass overhead is material.
- **(C) Hybrid:** particles on GPU, panel passes on 64-thread CPU. Bounded
  by the ~36 s body-pass floor → **cannot reach 30 rev/h**; state what it
  CAN reach (useful as the minimal-risk fallback and as `051`'s fallback
  envelope).

### Stage 2 — scope the panel GPU kernel

Scope the panel-side `direct!` GPU port: FLOWPanel's `direct!` overload at
`FLOWPanel_abstractbody.jl:1260`; element types = constant source/doublet
tris + vortex rings/sheets/filaments. Note the radix homogeneity
requirement (one shared `body_type`, `strength_dims`, `direct_kernel`
across systems, `translate_batched_resident.jl:2255-2263`) vs FLOWPanel's
heterogeneous tuples — this constrains option A's multi-system form.

### Stage 3 — FmmPlan / NearfieldInfluenceCache fit

Decide where flowpanel-20260817's FmmPlan and NearfieldInfluenceCache
(dense nearfield as packed BLAS matvecs) sit relative to RadixFMMCache in
the chosen shape — reuse, wrap, or bypass.

## Gates and verdict

- Each option priced with the `049` budget table and the recorded facts (no
  hand-waving on the binding constraint).
- Verdict names the `051` implementation shape (A, B, C, or a stated
  combination), the panel-kernel scope, and the FmmPlan/cache disposition.

## Artifacts

- `theory/` or `data/panel_multisystem_scoping/` — the pricing analysis and
  verdict `report.md`; any small measurement scripts under `scripts/`.

## Verification

- Any measured pass-overhead numbers carry job IDs and same-job baselines;
  the verdict's arithmetic is checkable against the `049` budget table.

## Recorded context (2026-08-20 staging)

**Multi-system on the radix path (what exists):** multi-SOURCE-system is
real — `RadixFMMCache.n_systems` (`containers.jl:2334`),
`source_buffers::NTuple`, all systems concatenated into ONE flat device
matrix with `body_system_ids`/`body_indices` pack+scatter
(`translate_batched_cuda.jl:944-962`, `:5717`); homogeneity required: one
shared `body_type`, `strength_dims`, `direct_kernel` across systems
(`translate_batched_resident.jl:2255-2263`). **BUT v1 restriction:
`_assert_radix_targets_are_sources` requires `t === s` elementwise
(`translate_batched_resident.jl:1735-1743`); `target_bodies` is literally
aliased to `source_bodies`** (`translate_batched_cuda.jl:5447,:5526`). No
target-only/probe support on the GPU path; CPU legacy octree DOES support
distinct source/target trees (`fmm.jl:258`); solvers also assert
targets===sources (`solve.jl:549,:1090`). `041b` strategic-target was NO-GO
but studied reduced targets within one set, not external panel/probe
targets — a different question.

**FLOWPanel FMM-compatibility layer (CPU, complete):**
`source_system_to_buffer!` `FLOWPanel_abstractbody.jl:1096`, `direct!`
`:1260`, `body_to_multipole!` per element type
(`FLOWPanel_nonliftingbody.jl:224-234`, `FLOWPanel_liftingbody.jl:705,797,906`),
PanelWake trio (`FLOWPanel_wake.jl:326,525,562,565`), filaments
(`:2763,2825,2841`). Multi-system fmm! is real: `FLOWPanel_fmm.jl:60`
`influence!` over heterogeneous tuples → `fmm!(targets::Tuple,
sources::Tuple)` at `:88` (plan-reusing at :114). Per step
(`FLOWPanel_simulate.jl:673-712`): **three separate fmm! passes** —
wake→(bodies+particles), panel solve, bodies→targets (different
kerneloffsets/derivative switches per pass) + separate `Estr_fmm!` call
(`FLOWPanel_wake.jl:2052`) reusing wake trees. Targets ≠ sources in these
passes.

**GPU support in FLOWPanel: none** (only a `GPUArray` kwarg on the dense
linear solve, `FLOWPanel_liftingbody.jl:379-412`; no CUDA dep).

**018 cost facts for pricing:** 170–230 s/step on 64 cores; split = wake
influence 64.2% / body 25.3% / solve 9.3%; body pass floor ~36 s
(kerneloffset-radius-bound); ~75% of step = `Estr_fmm!`; 36,752 panels
(45_185_ct4 mesh), ~181k particles at maturity (342k on the 6R arm);
NT=36 steps/rev; target ≤3.3 s/step.
