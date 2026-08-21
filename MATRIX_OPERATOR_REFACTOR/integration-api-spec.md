# Integration API Specification (Task 031)

Status: **signed off by user 2026-08-04** (decisions recorded in §9 and in
`031-integration-api-design.md`). Author: task 031.
Scope: design only — no `src/` changes. This document specifies the
generalizable device-resident system interface that task `032` implements and
that FLOWVPM (task `034`) consumes first. File:line references are to
FastMultipole branch `matrix-ops` and FLOWVPM branch `gpu-full` as of
2026-08-04.

## 1. Goal and design tenets

An external code keeps its particle state on the GPU and drives the resident
FMM lifecycle every time step with:

- **no per-step host/device body transfer** (`body_uploads = 0`,
  `expansion_host_copies = 0` — the task 023 counter contract);
- **no per-step allocation** once the cache is constructed
  (capacity-sized buffers, valid-prefix step counts);
- **full output**: scalar potential, gradient, and the **9-component
  hessian** (user decision 2026-08-04 — no 6-component symmetric variant;
  see §6);
- support for **vector strengths and the Lamb-Helmholtz channel**
  (vortex methods) and per-body **extra states** (e.g. smoothing radius σ);
- a documented **transfer-based fallback** for consumers whose scale does not
  justify device residency (§8).

Tenet: the interface is *general-consumer-first*. FLOWVPM appears only as the
worked example (§7); nothing in the contract may assume the 46×N FLOWVPM
layout.

## 2. Interface surface (consumer-facing)

A consumer defines a system type `S` and implements:

| function | role | when called |
|---|---|---|
| `get_n_bodies(sys)` | live body count `n` (may vary per step, `1 ≤ n ≤ max_n_bodies`) | every step |
| `data_per_body(sys)` | packed rows per body (§3) | construction |
| `strength_dims(sys)` | 1 (scalar) or 3 (vector strength) | construction |
| `get_position(sys, i)` | `SVector{3}`; host-side construction/recenter bounds derivation only | construction; explicit host `recenter!` |
| `has_vector_potential(sys)` | selects the Lamb-Helmholtz lifecycle (χ channel) | construction |
| `residency(sys)` | `HostResident()` (default) or `DeviceResident()` | construction + each step |
| `body_type(sys)` | **new (032)**: B2M kernel selection, e.g. `Point{Vortex}`, `Point{Source}` | construction |
| `direct_kernel(sys)` | **new (032)**: nearfield kernel functor (§5) | construction |
| `source_to_buffer!(buf, sys, sort_index)` | pack bodies into the framework buffer (device method for `DeviceResident`) | every step |
| `buffer_to_target!(sys, buf, switch, sort_index)` | consume results from the framework buffer (device method for `DeviceResident`) | every step |

Construction and stepping (unchanged surface, task 023):

```julia
cache = RadixFMMCache(system; expansion_order, ell, max_n_bodies, bounds,
                      device=true, options...)
fmm!(system, cache; scalar_potential, gradient, hessian, lamb_helmholtz)
```

Ownership: the packed body matrix, output buffer, per-system scatter/staging
buffers, expansion buffers, routes, and operator tables are framework-owned
and capacity-sized at construction (`src/containers.jl:1682-1781`). The
consumer owns its own state arrays; the framework never retains references to
them between calls.

Allocation contract for this surface: the consumer-owned trait/accessor
methods (`get_n_bodies` through `direct_kernel`) must be allocation-free.
`source_to_buffer!` and `buffer_to_target!` are consumer-implemented kernels or
bulk operations that write into framework-owned persistent buffers and must be
steady-state allocation-free. `RadixFMMCache` and the framework allocate those
buffers and all scratch at construction; `fmm!` only mutates their valid
prefixes. The explicit `recenter!` exception is specified below and is also
zero-allocation after construction.

### Per-step call order (device-resident, one `fmm!(sys, cache)` call)

1. `source_to_buffer!(device_buf, sys, sort_index)` — consumer fills the
   framework's **persistent** per-system device buffer (§4). `sort_index` is
   the identity on this path (`translate_batched_cuda.jl:896-899`).
2. In-place device tree/route refresh (`update_cuda_radix_state!`,
   `translate_batched_cuda.jl:4165`).
3. Resident lifecycle B2M→M2M→M2L→L2L→L2B + nearfield direct.
4. Scatter to the per-system output buffer (switch-relative rows), then
   `buffer_to_target!(sys, device_out, switch, sort_index)`.

**Delivery semantics** (resolves the overwrite-vs-accumulate ambiguity): the
framework always delivers the **total influence of this evaluation** in the
output buffer (it zeroes its accumulators each step). Whether the consumer
overwrites or accumulates into its own state inside `buffer_to_target!` is the
consumer's choice; FLOWVPM accumulates (§7), the 028 harness overwrites. The
032 docs must state this explicitly.

## 3. Body packing and storage layout

Packed body matrix (framework-owned, device): `TF × data_per_body × max_n`,
column per body, rows:

| rows | content |
|---|---|
| 1:3 | position |
| 4 | body radius (finite-size/regularization radius; MAC/error use) |
| 5 : 4+`strength_dims` | strength (scalar q, or vector Γ) |
| 4+`strength_dims`+1 : `data_per_body` | consumer extra states (e.g. σ), opaque to the far field, visible to the nearfield kernel |

This is the existing host-buffer convention (`src/compatibility.jl:18-27`)
promoted to the resident path. **Change from shipped v1**: the resident pack
kernel currently truncates to 5 rows, keeps only scalar row 5, and hard-zeros
radius row 4 (`_cuda_pack_radix_body_kernel!`,
`translate_batched_cuda.jl:922-936`); 032 generalizes this. **Profile-triggered
layout decision (roadmap review 2026-08-05, superseding the mandatory two-layout
benchmark)**: `032` first implements one packed matrix carrying all
`data_per_body` rows in sorted order. The 5-row core matrix plus a separate
capacity-sized side buffer for extra states is implemented only if a profile
or bandwidth model predicts at least a 5% end-to-end U/J-solve improvement;
it becomes the default only if measurement confirms that gain. The
consumer-facing contract is identical either way (rows addressed by the §2
accessor conventions).

Output buffer (framework-owned): `TF × 13 × max_n` — row 1 scalar potential,
rows 2:4 gradient, rows 5:13 hessian (column-major 3×3, matching the legacy
`set_hessian!` order, `src/compatibility.jl:713-737`). The per-system scatter
buffer stays switch-relative (`scalar_potential_index`, `gradient_range`,
`hessian_range` of `DerivativesSwitch`, `src/derivativesswitch.jl:72-98`), so
consumers that skip potential or hessian pay no bandwidth for them.

Precision: `TF = options.precision`. Float32 admissible per the 024/028
accuracy rules (default tables in `translate_batched_resident.jl:852-870`).

## 4. Capacity and allocation contract

Restated from task 023 (`src/containers.jl:1739-1747`) and generalized:

- Fixed for cache lifetime: normalized internal box, `ell`, `expansion_order`,
  `max_n_bodies`, precision/strategies, LH flag, `n_systems`, and all derived
  capacities. The physical-to-normalized coordinate map (`x_min`, `L`) is
  fixed between explicit `recenter!` calls, not immutable for the full cache
  lifetime.
- Step-varying: positions, strengths, extra states, and the live count
  `get_n_bodies` (valid-prefix tracking via `RadixStepCounts`,
  `containers.jl:1668-1674`). A consumer with known maximum particle count
  pre-sizes once with `max_n_bodies` and never reallocates; adding/removing
  particles is just the consumer updating its arrays and `get_n_bodies`.
- Bodies leaving the domain box throw `ArgumentError` at the next step
  (`translate_batched_resident.jl:664`) — no silent geometry rebuild.
- **032 fix required**: the device-resident source refresh allocates a fresh
  `CuArray` every step (`translate_batched_cuda.jl:4005`) instead of reusing a
  persistent per-system device buffer as the host-resident path does
  (`:3902-3908`). The generalized contract is: one persistent device buffer
  per system, allocated at construction, passed to `source_to_buffer!` each
  step. Steady state is then zero-allocation on both residency paths.

**Domain-box policy (user decision 2026-08-04):** `032` ships
`recenter!(cache, systems; bounds=nothing, padding=0.05)` with this contract:

- The consumer calls it explicitly between evaluations, before the next
  `fmm!`, when the physical box should change. `fmm!` never recenters
  implicitly; an out-of-box body still throws and leaves the cache usable.
- `bounds=(x_min, L)` is the deterministic fast path and is recommended for a
  device-native consumer that already tracks its domain. With `bounds=nothing`,
  the framework derives the union bounds of all live bodies: host systems use
  `get_position`; device systems first fill their persistent canonical source
  buffers and use a device reduction with construction-sized scratch, copying
  only the six extrema scalars to the host. No body array is transferred.
- For derived bounds, `padding` is a nonnegative fraction of the tight cube's
  side added on each face: `x_min = lo - padding*L_tight` and
  `L = (1 + 2padding)*L_tight`. The default 5% per face gives moving bodies
  headroom. Caller-supplied `bounds` are final and are not padded. Empty
  systems, non-finite bounds, nonpositive `L`, negative padding, a changed
  system count, or a live count above capacity throw `ArgumentError` without
  mutating the cache.
- The framework owns and mutates the coordinate map and reduction scratch;
  the consumer retains ownership of its state. The operation performs no
  allocation after cache construction. It does not run B2M or deliver output;
  the following `fmm!` performs the ordinary pack/tree/route refresh.

The helper reuses every allocation and refreshes only the geometry-dependent
state. The preferred implementation runs the lifecycle in
**normalized (unit-cube) internal coordinates**: positions are mapped
`x' = (x - x_min)/L` at pack time, the per-body radius and σ rows are scaled
by `1/L` alongside (the regularized kernel depends only on `r/σ`, so it is
scale-invariant), and outputs are rescaled per derivative order at the
scatter/finalize stage — potential `×1/L`, gradient `×1/L²`, hessian `×1/L³`
(one multiply per row). With box size folded out this way, the
distance-scaled operator tables are box-size-invariant and `recenter!` never
rebuilds them — it only restamps `x_min`/`L` and invalidates the prior
step-count prefixes until the next refresh. **Strength scaling is not
used**: rescaling strengths (e.g. `Γ' = Γ/L²`) can make exactly one output
order pass through unscaled but the remaining orders still need per-channel
factors, so strengths stay physical and all scaling lives in the coordinate
map and the output factors. `032` must verify the normalized-coordinate route
against the current absolute-coordinate lifecycle for accuracy parity; if it
proves invasive, the fallback `recenter!` re-derives the geometry-dependent
tables while still reusing all allocations.

## 5. Far-field and nearfield kernels (vector strength, LH, hessian)

- **B2M**: selected by `body_type(sys)` (new trait). 032 ships device B2M for
  `Point{Source}` (current scalar behavior, default) and `Point{Vortex}`
  (vector Γ, writes both φ and χ channels). The resident buffers, M2M/M2L/L2L
  and L2B are already χ-capable end-to-end
  (`translate_batched_cuda.jl:39-43, 1434-2378`); only B2M writes are missing
  (`:1180`). The χ channel is carried at `P+1` per the approved `008h` rule —
  operator sizing already honors this via `basis_dof_chi`.
- **L2B**: extend the output kernels (`_cuda_l2b_output_kernel!`
  `:1618-1647`, host mirror `translate_batched_resident.jl:437`) to emit the
  9 hessian rows for both channels (second derivatives of the local
  expansion; LH velocity-gradient terms included).
- **Nearfield direct**: the shipped kernels hard-code the singular scalar
  kernel (`u += q/4πr`, `translate_batched_cuda.jl:1106-1612`). The
  generalized contract: `direct_kernel(sys)` returns an **isbits functor**
  called per (target, source) pair as
  `direct_kernel(dx, dy, dz, r2, source_column_view) -> (u, g₁..g₃, h₁..h₉)`
  with access to the source's packed extra-state rows (σ). 032 ships two
  built-ins: `SingularSource()` (current behavior) and `RegularizedVortex`
  implementing the FLOWVPM Biot–Savart with `g_dgdr` for the **`gaussianerf`
  regularization only** (user decision 2026-08-05: the phase supports only
  the FLOWVPM default kernel, the sole `CoreSpreading`-compatible one;
  `winckelmans` dropped). The device-safe erf is borrowed from FLOWVPM's
  `custom_erf` FDLIBM rational-polynomial port
  (`../FLOWVPM.jl/src/FLOWVPM_gpu_erf.jl`) and the fused U+J per-pair math
  from `../FLOWVPM.jl/ext/FLOWVPMCUDAExt.jl:172-234`; FastMultipole's copies
  must be self-contained.

  **Amendment (user direction 2026-08-05, revised after numerical review and
  extended the same day):** the vortex nearfield has **three candidate
  strategies**, decided by measurement: (i) the regularized-everywhere
  `RegularizedVortex` functor above; (ii) a **partitioned replacement** kernel
  that keeps the singular FMM far field, evaluates pairs inside the cutoff once
  with cancellation-safe regularized U/J formulas, and evaluates remaining
  direct pairs with the singular kernel; and (iii) a **two-pass additive
  correction** that leaves the FMM entirely unmodified (singular far field *and*
  singular direct) and adds a second pass carrying only the regularization
  deficit over the cutoff shell. Row `031a` derives all three — the stable
  small-`ρ` series, cutoff bounds, exact-once geometry contract
  (§§1-6), and the two-pass operator with its conditioning and cost analysis
  (§6.1); `032a` implements them and selects the measured default. No runtime
  `erfc` is used by any candidate.

  Candidate (iii) touches no `025` routing invariant, because its correction
  rides on a complete singular evaluation and never asks which route owned a
  pair — but its subtraction lands in the target accumulator across two kernels,
  amplifying rounding by `~ρ⁻³`. It therefore requires either **Float64
  accumulation of the singular direct term and its correction** (the far field
  may stay FP16-WMMA/Float32), or the **`ρ_c = 2` hybrid** that evaluates
  `ρ ≤ ρ_c` pairs with the stable regularized form inside pass 1. Two-pass and
  partitioning have opposite depth trends (`031a` §6.1), so `032a` must measure
  both at fixed adequate geometry rather than at each strategy's own optimum.

  **Near-set adequacy applies to both strategies (031a review correction,
  2026-08-05).** The FMM far field is singular under *either* kernel, so the
  direct geometry must cover every pair with `r/σ_src ≤ ρ_t` in both cases —
  this is a correctness condition on the vortex FMM coupling, not a
  partitioning-specific constraint. The binding quantity is the *minimum AABB
  gap* the stencil leaves to M2L, not its outer radius, so the condition is
  `g_min · h_leaf > ρ_t · σ_max` with
  `g_min = min_{o∉D} sqrt(Σ_q max(0,|o_q|−1)²)` — `√5` for the shipped
  `|o|²≤12` stencil, `1` for the classic — equivalently the depth ceiling
  `2^ℓ < g_min · L_box / (ρ_t σ_max)` (see
  `theory/kernel-splitting-nearfield.md` §5.1-§5.2). At `n=1e6, ℓ=5, β=2,
  ε=1e-3` the shipped leaf stencil is **not adequate**; the cheap remedy is
  enlarging the deepest-level near set from 179 to 389 offset classes.
  **`032` acceptance item**: the `RegularizedVortex` baseline must evaluate
  that geometric test at its configured geometry — cache box, leaf `h`, and a
  max-reduction over the packed σ row, all already on device — and either
  enlarge the deepest-level near set to `{o : gap(o) ≤ ρ_t σ_max}` or reject
  the configuration naming the measured ratio. It may not silently run on an
  inadequate stencil. `032a` repeats the assertion against its production route
  construction.

  The uniform form `n/8^ℓ > (ρ_t β / g_min)³` (79 bodies/cell at `β=2,
  ε=1e-3`) is the **design-time** sizing law only. Its body count is per
  *occupied* cell, so `n/8^ℓ` substitutes for it only when the field fills its
  box: on the `033` wake cylinder — 3.14% fill — it under-reports the
  admissible depth by one to two levels (§5.2), and it cannot see a σ grown by
  `CoreSpreading`. Do not implement the runtime assertion in that form.
  The SFS (`ζ`/`Estr`) kernel derivation is deferred to a later row. Consumer-supplied functors are allowed if isbits
  and GPU-compilable; they compile into the pair kernels via type
  specialization, so each distinct kernel is one extra kernel instantiation,
  not a runtime branch. **Measurement requirement (user direction
  2026-08-04)**: `032` benchmarks the functor-dispatched `SingularSource`
  against the current hard-coded kernel on the `028` workload; if the
  abstraction costs measurably, that is reported to the user before
  proceeding.

## 6. Hessian: always 9 components

For a scalar potential φ, `H = ∇∇φ` is symmetric and 6 components suffice.
For the vector-potential/LH channel the consumer-visible quantity is the
gradient of the *evaluated field* `u = ∇φ + ∇×ψ`; `∂ᵢ(∇×ψ)ⱼ` is **not
symmetric** in general (its antisymmetric part carries the local vorticity),
and FLOWVPM's stretching term needs the full `J = ∇u`. Decision (user,
2026-08-04): a single 9-component layout everywhere; no 6-component variant,
no layout branch on `has_vector_potential`. Memory cost at 1e6 bodies,
Float32: 9 rows × 4 B × 1e6 = 36 MB — negligible against the 2.00 GB
persistent footprint measured in 028.

## 7. Worked example: FLOWVPM

FLOWVPM state is one dense 46×N matrix (column per particle), `Matrix` or
`CuArray` (`FLOWVPM_particlefield.jl:282-295`). Mapping:

| FLOWVPM rows | quantity | interface side |
|---|---|---|
| `X` 1:3 | position | packed rows 1:3 |
| `GAMMA` 4:6 | Γ | packed rows 5:7 (`strength_dims = 3`) |
| σ-derived `ρ_σ·σ` | regularization radius | packed row 4 (computed in `source_to_buffer!`, cf. `FLOWVPM_fmm.jl:62-71`) |
| `SIGMA` 7 | σ | packed row 8 (extra state, read by `RegularizedVortex` nearfield) |
| `U` 10:12 | velocity | **accumulated** from output gradient rows |
| `J` 16:24 | velocity gradient | **accumulated** from output hessian rows |

Notes:

- FLOWVPM's "velocity" is the FMM **gradient** channel and its "J" is the FMM
  **hessian** channel (`FLOWVPM_fmm.jl:170-176`) — the 13-row output covers
  exactly its needs; `scalar_potential=false` skips row 1.
- `buffer_to_target!` accumulates (`.+=`) because FLOWVPM's own
  `_reset_particles` zeroes U/J at the top of each UJ evaluation; delivery
  semantics per §2 make this correct.
- One RK3 time step performs **three** UJ evaluations, each preceded by a
  reset and each moving particles (substeps) — so each evaluation is a full
  `fmm!(sys, cache)` call including tree/route refresh. The refresh is cheap
  (028: full in-place re-sort/re-tree inside the 9.6 ms verdict budget), but
  035 should measure the 3×-per-step multiplier explicitly.
- `solve_ρ_over_σ` (Roots.jl bisection per body, `FLOWVPM_fmm.jl:26-48`) runs
  inside `source_to_buffer!` today. On device this must be branch-light: with
  autotuning off it is the constant `default_rho_over_sigma·σ` (no solve);
  the autotuned path needs a device-safe closed-form or lookup — flagged to
  034, not an interface requirement.
- Depth tuning in `034`/`035` is capped by near-set adequacy: with the `θ=0.5`
  stencil and `ε=1e-3`, `2^ℓ < √5·L_box/(ρ_t σ_max)` unless the deepest-level
  near set is enlarged. Both `033` cases use overlap `β=2` against the local
  mean spacing, but only the cube fills its box, so the ceilings differ: cube
  `ℓ≤1/2/3/4` and wake `ℓ≤2/3/5/6` at `n=1e3/1e4/1e5/1e6`
  (`theory/kernel-splitting-nearfield.md` §5.2,
  `data/kernel_splitting/case_adequacy.csv`). The wake's AR=5 cylinder fills
  3.14% of its bounding cube; that residual penalty is a staged `035` lever.
- The legacy dynamic-P error tolerance (`PowerRelativeGradient`) has no
  resident-path equivalent; accuracy is set by `expansion_order` + stencil
  geometry (task 025/028 machinery). 034/035 tune those against the fixed
  phase tolerance: sampled relative gradient (velocity) RMS error ≤ 1e-3
  (user decision 2026-08-05; the earlier "match FLOWVPM default parameter
  accuracy" gate is superseded).

## 8. Resident vs transfer-based coupling

Measured evidence (task 028, H200, n=1e6, P=4):

- Fully host-resident lifecycle: 159.4 ms vs 91.4 ms device-resident at
  Phase A — ≈1.7× (028 §Phase A); the gap is per-step body H2D/D2H plus
  host-side packing/allocation.
- After optimization the device-resident verdict cost is 9.591 ms with
  `body_uploads = 0`; the "including transfers" boundary added ≈2 ms at
  Phase A scale.

Guideline for consumers: estimated per-step transfer time ≈
`n·(data_per_body + output_rows)·sizeof(TF) / PCIe_BW` (≈25 GB/s effective).
At n=1e6 Float32 with 8-in/13-out rows that is ≈3.4 ms round trip —
comparable to the entire optimized step, so **device residency is required at
n ≳ 1e5-1e6 whenever the step budget is single-digit ms**. At n ≲ 1e4, or
step budgets ≫100 ms, the host-resident path (default `residency`, one H2D of
the packed prefix + one D2H of results, `translate_batched_cuda.jl:3992-4008`)
is simpler and adequate: the consumer implements only the host hooks it
already has for the legacy path. Both modes are first-class; the choice is
per-system via the `residency` trait.

## 9. Sign-off decisions (user, 2026-08-04)

- (a) **Domain box policy**: `recenter!` ships in `032`, preferably via
  normalized unit-cube internal coordinates with per-derivative-order output
  scaling; strengths stay physical (full rationale in §4).
- (b) **Nearfield kernel mechanism**: the isbits-functor trait, with a
  mandatory functor-vs-hard-coded benchmark; report to the user if the
  abstraction is slow (§5).
- (c) **Packed-row policy**: implement the all-row canonical layout first;
  add the 5-row core + side table only under the profile/model trigger and
  measured-gain rule in §3.
- (d) **032 scope**: approved as specified — §§2-6 plus the decisions above;
  deferring `targets !== sources` and multi-GPU. (`recenter!` moved from the
  deferral list into scope by decision (a).)

## 10. Gap-analysis summary (shipped v1 → required)

| # | gap | evidence | 032 item |
|---|---|---|---|
| 1 | pack kernel keeps scalar strength only, zeroes radius, truncates extras | `translate_batched_cuda.jl:922-936` | generalized `data_per_body`-row packing |
| 2 | B2M never writes χ (LH structurally zero) | `:1180`; host `translate_batched_resident.jl:164-172` | `Point{Vortex}` B2M, φ+χ |
| 3 | no hessian output; `fmm!` throws | `src/fmm.jl:876-878`; buffer `TF×4×n` `:3926` | 13-row output; L2B + direct hessian |
| 4 | `targets === sources` restriction | `translate_batched_resident.jl:654-662` | keep (document); FLOWVPM unaffected |
| 5 | device-resident refresh allocates per step | `translate_batched_cuda.jl:4005` | persistent per-system device buffer |
| 6 | overwrite-vs-accumulate unspecified | `fm028_device_system.jl:72-79` | delivery semantics (§2) in docs |
| 7 | deprecated hooks with swapped arg order | `src/compatibility.jl:53,69` | remove or hard-error in 032 |
| 8 | nearfield kernel hard-coded singular scalar | `translate_batched_cuda.jl:1106-1612` | `032`: `direct_kernel` trait + regularized-everywhere `RegularizedVortex` (gaussianerf); `032a`: partitioned replacement, two-pass additive correction (`031a` §6.1), and default selection |
| 8a | leaf near set inadequate for the smoothing cutoff — applies to **both** nearfield strategies | `theory/kernel-splitting-nearfield.md` §5.1-§5.2; shipped `\|o\|²≤12` gives `g_min h = 2.236h` against `ρ_t σ = 3.065h` at `n=1e6, ℓ=5, β=2` | `032`: assert `g_min·h_leaf > ρ_t·σ_max` (equivalently `2^ℓ < g_min L_box/(ρ_t σ_max)`) from live geometry, then enlarge the deepest-level near set (179 → 389 classes) or reject; **not** the box-filling `n/8^ℓ` form, which mis-ranks the `033` wake by one to two levels; `032a`: verify against production routes |
| 9 | device-system worked example lives in a benchmark script | `scripts/fm028_device_system.jl` | promote pattern to `src/` + docs |
