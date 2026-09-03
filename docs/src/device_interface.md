# Connecting an External Code to the Resident GPU Lifecycle

This page documents the device-resident FMM interface (tasks 023–032): how an
external code that keeps its particle state on the GPU drives the recurring
radix-grid FMM lifecycle with **zero per-step host/device body transfer** and
**zero per-step allocation**. It restates the contracts of the signed-off
integration API specification
(`MATRIX_OPERATOR_REFACTOR/integration-api-spec.md`); a complete runnable
program implementing everything described here is
[`examples/device_resident_system.jl`](https://github.com/byuflowlab/FastMultipole.jl/blob/main/examples/device_resident_system.jl).

!!! note "Task 032 staging"
    Most of this page describes interface surface that exists on the
    `matrix-ops` branch today. The explicit [`recenter!`](@ref recenter-section)
    helper is specified but lands in task 032 stage 3; it is marked as such
    below.

## Overview: the resident radix path

The resident path is an opt-in alternative to the legacy octree `fmm!`. Instead
of rebuilding an adaptive octree each call, it fixes a domain box and a radix
grid depth at construction time, allocates every buffer and operator table once
at capacity, and then treats each subsequent call as a pure refresh-and-execute
step:

```julia
using FastMultipole
using KernelAbstractions, CUDA   # loads FastMultipoleKAExt, which registers the device lifecycle

cache = RadixFMMCache(system; expansion_order=4, ell=4,
                      max_n_bodies=n_max, bounds=(x_min, box_size),
                      device=true)

for step in 1:n_steps
    fmm!(system, cache; scalar_potential=false, gradient=true, hessian=true)
    my_time_integrator!(system)   # moves particles on the device
end
```

Each `fmm!(system, cache)` call performs, in order (device-resident case):

1. `source_to_buffer!(device_buffer, system, sort_index)` — *you* fill the
   framework's persistent per-system device buffer. `sort_index` is the
   identity on this path.
2. In-place device tree/route refresh (re-sort, re-count, re-route — no
   allocation).
3. The resident lifecycle: B2M → M2M → M2L → L2L → L2B, plus the nearfield
   direct stage.
4. Scatter into the per-system output buffer, then
   `buffer_to_target!(system, device_output, switch, sort_index)` — *you*
   consume the results.

The counter contract from task 023 makes "resident" checkable: after the first
call, `cache.state.counters.body_uploads`, `influence_downloads`,
`route_uploads`, and `operator_uploads` stay flat across steps, and
`expansion_host_copies == 0` always.

## The consumer surface

A consumer defines a system type `S` and implements the following functions.
"Owner" states who allocates the memory a function touches.

| function | role | when called | owner / allocation |
|---|---|---|---|
| `get_n_bodies(sys)` | live body count `n` (may vary per step, `1 ≤ n ≤ max_n_bodies`) | every step | consumer; must be allocation-free |
| `data_per_body(sys)` | packed rows per body (see layout below) | construction | consumer; allocation-free |
| `strength_dims(sys)` | 1 (scalar) or 3 (vector strength) | construction | consumer; allocation-free |
| `get_position(sys, i)` | `SVector{3}` position of body `i`; used only for host-side construction/recenter bounds derivation | construction; explicit host `recenter!` | consumer; allocation-free |
| `has_vector_potential(sys)` | selects the Lamb-Helmholtz lifecycle (χ channel) | construction | consumer; allocation-free |
| `residency(sys)` | `HostResident()` (default) or `DeviceResident()` | construction + each step | consumer; allocation-free |
| `body_type(sys)` | B2M kernel selection, e.g. `Point{Source}` (default), `Point{Vortex}` | construction | consumer; allocation-free |
| `direct_kernel(sys)` | nearfield kernel functor (see below) | construction | consumer; allocation-free |
| `source_to_buffer!(buf, sys, sort_index)` | pack bodies into the framework buffer (device method for `DeviceResident`) | every step | writes into a **framework-owned** persistent buffer; must be steady-state allocation-free |
| `buffer_to_target!(sys, buf, switch, sort_index)` | consume results from the framework buffer (device method for `DeviceResident`) | every step | reads from a **framework-owned** persistent buffer; must be steady-state allocation-free |

Construction and stepping:

```julia
cache = RadixFMMCache(system; expansion_order, ell, max_n_bodies, bounds,
                      device=true, options...)
fmm!(system, cache; scalar_potential, gradient, hessian)
```

**Ownership.** The packed body matrix, output buffer, per-system
scatter/staging buffers, expansion buffers, routes, and operator tables are
framework-owned and capacity-sized at construction. The consumer owns its own
state arrays; the framework never retains references to them between calls.

**Trait consistency.** All source systems sharing one `RadixFMMCache` must
report the same `body_type`, the same `strength_dims`, and equal
`direct_kernel` values; violations throw `ArgumentError` at construction.
`Point{Vortex}` requires `has_vector_potential(system) == true` (the
Lamb-Helmholtz χ channel), also checked at construction.

### Delivery semantics

The framework always delivers the **total influence of this evaluation** in
the output buffer: it zeroes its accumulators each step, so the buffer handed
to `buffer_to_target!` contains the complete potential/gradient/hessian of the
current call, never a running sum across calls. Whether the consumer
*overwrites* or *accumulates* into its own state inside `buffer_to_target!` is
the consumer's choice:

- a time stepper that consumes velocity directly typically **overwrites**
  (the worked example below does);
- a code that zeroes its own accumulators at the top of each evaluation and
  sums several contributions **accumulates** (`.+=`) — FLOWVPM does this.

Both are correct; the framework side is identical either way.

## Packed body layout and output layout

**Input.** The packed body matrix is framework-owned, `TF × data_per_body ×
max_n_bodies`, one column per body. `source_to_buffer!` must write columns in
this layout:

| rows | content |
|---|---|
| `1:3` | position |
| `4` | body radius (finite-size/regularization radius; used by MAC/error logic) |
| `5 : 4 + strength_dims` | strength (scalar `q`, or vector `Γ`) |
| `4 + strength_dims + 1 : data_per_body` | consumer extra states (e.g. smoothing radius σ) — opaque to the far field, visible to the nearfield kernel |

This is the same column convention as the legacy host path
(`source_system_to_buffer!`), promoted to the resident path. The precision
`TF` is `cache.state.options.precision` (auto-selected from `expansion_order`
by the measured task 024/028 rules unless you pass `options` explicitly).

**Output.** The framework's canonical output buffer is `TF × 13 ×
max_n_bodies`:

| rows | content |
|---|---|
| `1` | scalar potential |
| `2:4` | gradient |
| `5:13` | hessian (column-major 3×3, matching the legacy `set_hessian!` order) |

The per-system scatter buffer actually handed to `buffer_to_target!` is
**switch-relative**: it carries only the channels the `DerivativesSwitch`
requested, so consumers that skip the potential or the hessian pay no
bandwidth for them. Always address it through the switch accessors rather than
hard-coded rows:

```julia
function FastMultipole.buffer_to_target!(sys::MySystem, out, switch, sort_index)
    spi = FastMultipole.scalar_potential_index(switch)  # 0 when not requested
    spi > 0 && (sys.potential .= vec(view(out, spi, :)))
    gr = FastMultipole.gradient_range(switch)           # empty when not requested
    isempty(gr) || (sys.velocity .= view(out, gr, :))
    hr = FastMultipole.hessian_range(switch)            # 9 rows when requested
    isempty(hr) || (sys.jacobian .= view(out, hr, :))
    return sys
end
```

The hessian is **always 9 components** when requested — there is no
6-component symmetric variant. For a scalar potential the 3×3 block happens to
be symmetric; for the Lamb-Helmholtz channel the consumer-visible quantity is
$\nabla u$ with $u = \nabla\phi + \nabla\times\psi$, whose antisymmetric part
carries the local vorticity, so all 9 entries are meaningful. Hessian output
requires a cache built with `RadixFMMCache(...; hessian=true)`; requesting
`fmm!(...; hessian=true)` against a 4-row cache throws.

## Capacity and allocation contract

Fixed for the cache's lifetime: the domain box (between explicit `recenter!`
calls), `ell`, `expansion_order`, `max_n_bodies`, precision and operator
strategies, the Lamb-Helmholtz flag, the number of systems, and all derived
capacities.

Step-varying: positions, strengths, extra states, and the live count returned
by `get_n_bodies`. All step-varying state is tracked as a valid prefix of the
capacity-sized buffers, so:

- **Pre-size once.** A consumer with a known maximum particle count constructs
  with `max_n_bodies = n_max` and never reallocates. Adding or removing
  particles is just the consumer updating its own arrays and the value
  returned by `get_n_bodies` — any `1 ≤ n ≤ max_n_bodies` is valid at any
  step.
- **Zero per-step allocation.** After construction, `fmm!` mutates only the
  valid prefixes of framework-owned buffers. Consumer hooks must likewise be
  steady-state allocation-free (broadcasts into existing device arrays,
  kernels, `copyto!` — no fresh `CuArray`s per step).
- **Out-of-box bodies throw.** A body outside the cache's fixed box raises
  `ArgumentError` at the next step. There is no silent geometry rebuild; the
  cache remains usable afterwards.

### [`recenter!` — moving the domain box](@id recenter-section)

*The following is the specified contract; the helper lands in task 032
stage 3.*

```julia
recenter!(cache, systems; bounds=nothing, padding=0.05)
```

- The consumer calls it **explicitly** between evaluations, before the next
  `fmm!`, when the physical box should change. `fmm!` never recenters
  implicitly.
- `bounds = (x_min, L)` is the deterministic fast path, recommended for a
  device-native consumer that already tracks its own domain. With
  `bounds = nothing` the framework derives union bounds of all live bodies —
  host systems via `get_position`, device systems via an on-device reduction
  (only six extrema scalars cross to the host; no body array is transferred).
- For derived bounds, `padding` is a nonnegative fraction of the tight cube's
  side added on each face: $x_{\min} = \mathrm{lo} - p\,L_{\text{tight}}$ and
  $L = (1 + 2p)\,L_{\text{tight}}$. Caller-supplied `bounds` are final and not
  padded.
- The operation performs **no allocation** after cache construction, does not
  run B2M, and delivers no output; the following `fmm!` performs the ordinary
  pack/tree/route refresh. Invalid inputs (empty systems, non-finite bounds,
  nonpositive `L`, negative padding, changed system count, live count above
  capacity) throw `ArgumentError` without mutating the cache.

## The nearfield `direct_kernel` functor surface

The nearfield direct stage dispatches on an **isbits functor** returned by the
`direct_kernel(system)` trait. It is stamped into the cache options at
construction, so the pair kernels specialize on it at compile time — one
kernel instantiation per functor type, no runtime branch in the pair loop.

Shipped kernels:

- `SingularSource()` — singular scalar $q/(4\pi r)$ kernel; the default for
  `Point{Source}`.
- `SingularVortex()` — singular Biot–Savart kernel; the default for
  `Point{Vortex}`.
- `PartitionedVortex(; sigma_row, rho_t=4.252)` — **the recommended default
  for σ-carrying vortex systems** (task 032a Checkpoint D, 2026-08-07):
  cancellation-safe regularized `gaussianerf` U/J inside the smoothing cutoff
  `r/σ_src ≤ rho_t`, exact singular Biot–Savart beyond it. Measured
  1.16–1.77x faster step-level than `RegularizedVortex` on H200 at identical
  1e-3-gate accuracy; on device it runs through the distance-binned pair
  stream (class-split compaction + within-cell sub-Morton ordering).
- `RegularizedVortex(; sigma_row, rho_t=4.789)` — regularized-everywhere
  Biot–Savart with the FLOWVPM default `gaussianerf` regularization (the only
  regularization supported in the Integration Phase; the evaluation is
  erf-free on device). The divergence-proof fallback.
- `TwoPassVortex(; sigma_row, rho_t=4.252, rho_c=2.0)` — supported
  alternative: unmodified singular FMM plus an additive deficit sweep over
  the `(rho_c, rho_t]` shell (hierarchical-policy device caches or host).

### The `sigma_row` convention

`RegularizedVortex` reads the raw smoothing radius σ from packed extra-state
row `sigma_row` of each **source** body — **never** from radius row 4. Row 4
carries the MAC/error radius (e.g. FLOWVPM's inflated $\rho_\sigma \sigma$),
which controls where the far field is trusted; the nearfield kernel needs the
physical σ, which therefore travels as a consumer extra-state row
(`sigma_row ≥ 5`, enforced at construction). Sources with `σ ≤ 0` (e.g.
zero-padded columns) fall back to the singular kernel.

### The near-set adequacy gate

The FMM far field is **singular** regardless of the nearfield kernel, so the
direct near set must contain every pair inside the smoothing cutoff
$r/\sigma_{\text{src}} \le \rho_t$ — otherwise regularization error is
silently introduced where M2L took over. The binding quantity is the minimum
axis-aligned gap the leaf stencil leaves to M2L, so each evaluation with a
regularized kernel asserts, from the live geometry (cache box, leaf size, and
an on-device max-reduction over the packed σ row):

$$g_{\min} \cdot h_{\text{leaf}} > \rho_t \cdot \sigma_{\max}$$

where $g_{\min} = \min_{o \notin D} \sqrt{\sum_q \max(0, |o_q| - 1)^2}$ over
offsets outside the direct set — $\sqrt{5}$ for the shipped $|o|^2 \le 12$
stencil, $1$ for the classic FMM stencil. Equivalently, the depth ceiling

$$2^{\ell} < \frac{g_{\min}\, L_{\text{box}}}{\rho_t\, \sigma_{\max}}.$$

If the configured geometry fails this test, the evaluation **throws**, naming
the measured ratio and the admissible depth — it never silently runs on an
inadequate stencil. The remedy is a smaller `ell` (or, in a later task,
enlarging the deepest-level near set). `rho_t = 4.789` is the conservative
cutoff at which the regularized and singular kernels agree to the phase
tolerance $\varepsilon = 10^{-3}$.

### Custom kernels

Consumer-supplied functors are allowed if isbits and (for `device=true`
caches) GPU-compilable. Subtype `AbstractDirectKernel` and implement, with
`dx, dy, dz = target - source` and `r2 = dx^2 + dy^2 + dz^2 > 0`
(self/coincident pairs are skipped by the caller):

- `FastMultipole._direct_pair_ug(kernel, dx, dy, dz, r2, source_bodies, j)`
  returning `(u, gx, gy, gz)`;
- `FastMultipole._direct_pair_ugh(kernel, dx, dy, dz, r2, source_bodies, j)`
  returning `(u, gx, gy, gz, h1, ..., h9)` (hessian in column-major 3×3
  order);
- `FastMultipole._emits_potential(kernel)::Bool` — whether `u` is meaningful
  (output row 1 written).

`source_bodies[:, j]` is the packed source column
(`[x, y, z, radius, strength..., extras...]`), giving the kernel access to
per-source extra states such as σ. (This flat-argument form is the shipped
signature; it lets the same code compile as a CUDA device function without
constructing a view per pair.)

## Lamb-Helmholtz and vortex consumers

`Point{Vortex}` sources require the Lamb-Helmholtz χ channel:
`has_vector_potential(system)` must return `true` (or pass
`lamb_helmholtz=true` at construction); a `Point{Vortex}` cache with the
channel off throws at construction.

For a vortex consumer the output channels read as:

- **gradient slot = velocity** $u = \nabla\phi + \nabla\times\psi$;
- **hessian slot = the 9-component velocity gradient** $J = \nabla u$ (needed
  in full, e.g. for vortex stretching — see the hessian discussion above).

As a concrete mapping, FLOWVPM (one dense 46×N state matrix, column per
particle) connects as:

| FLOWVPM rows | quantity | interface side |
|---|---|---|
| `X` 1:3 | position | packed rows 1:3 |
| `GAMMA` 4:6 | Γ | packed rows 5:7 (`strength_dims = 3`) |
| σ-derived `ρ_σ·σ` | regularization radius | packed row 4 (computed in `source_to_buffer!`) |
| `SIGMA` 7 | σ | packed row 8 (extra state, read by the `RegularizedVortex` nearfield) |
| `U` 10:12 | velocity | **accumulated** from output gradient rows |
| `J` 16:24 | velocity gradient | **accumulated** from output hessian rows |

FLOWVPM's `buffer_to_target!` accumulates (`.+=`) because its own reset zeroes
`U`/`J` at the top of each evaluation — correct under the delivery semantics
above. `scalar_potential=false` skips output row 1 entirely.

## Resident vs transfer-based coupling: which do you need?

Both modes are first-class; the choice is per-system via the `residency`
trait. `HostResident()` (the default) keeps the consumer's arrays on the host
and performs one H2D upload of the packed prefix plus one D2H download of the
results per step — the consumer implements only the host hooks it already has
for the legacy path. `DeviceResident()` eliminates those transfers entirely.

Measured evidence (task 028, H200, $n = 10^6$, literature $P = 4$):

- Fully host-resident lifecycle: **159.4 ms** vs **91.4 ms** device-resident
  at Phase A — a factor of ≈ **1.7×**, entirely per-step body H2D/D2H plus
  host-side packing/allocation.
- After optimization the device-resident verdict cost is **9.591 ms** with
  `body_uploads = 0`; the "including transfers" boundary added ≈ 2 ms at
  Phase A scale.

Rule of thumb for your own scale — estimated per-step transfer time:

$$t_{\text{xfer}} \approx \frac{n \cdot (\texttt{data\_per\_body} + \text{output rows}) \cdot \mathrm{sizeof}(TF)}{25\ \text{GB/s}}$$

(≈ 25 GB/s effective PCIe). At $n = 10^6$ Float32 with 8 input rows and 13
output rows that is ≈ 3.4 ms round trip — comparable to the entire optimized
step. Therefore:

- **Device residency is required at $n \gtrsim 10^5$–$10^6$ whenever the step
  budget is single-digit milliseconds.**
- **At $n \lesssim 10^4$, or step budgets ≫ 100 ms, the host-resident path is
  simpler and adequate.**

## Restrictions (v1)

- `target_systems === source_systems` — the radix path evaluates a system's
  influence on itself; distinct target sets throw `ArgumentError`.
- Body count `≤ max_n_bodies`; positions inside the cache's fixed box (see
  the capacity contract above).
- Hessian output requires `RadixFMMCache(...; hessian=true)`.
- The `device=true` cache requires a registered device backend: load
  `KernelAbstractions` together with a GPU package (CUDA, Metal) so the
  `FastMultipoleKAExt` extension registers it. `radix_device_backend_available()`
  is a convenient graceful-skip gate for scripts and tests, and
  `radix_device_status()` explains the state.

## Worked example

See [`examples/device_resident_system.jl`](https://github.com/byuflowlab/FastMultipole.jl/blob/main/examples/device_resident_system.jl)
for a complete device-resident scalar system: all trait overloads, the device
`source_to_buffer!`/`buffer_to_target!` methods, capacity-sized cache
construction, and a three-step convection loop that asserts the zero-transfer
counter contract.
