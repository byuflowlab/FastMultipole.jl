# Running on a GPU

FastMultipole's FMM runs on a GPU through a
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
extension, so the same kernels run on any backend KernelAbstractions supports:
NVIDIA through CUDA.jl, Apple through Metal.jl, AMD through AMDGPU.jl and Intel
through oneAPI.jl. Only CUDA and Metal have been tested; the other backends
should work but may need modification (Metal, for example, required Float32-only
constants and closure-free kernels). The extension is a package extension: it
loads automatically when `KernelAbstractions` is loaded next to `FastMultipole`,
and the GPU package supplies the backend.

```julia
using FastMultipole, KernelAbstractions
using CUDA          # or: using Metal
```

Nothing else changes in how systems are declared; two traits and one keyword
move the work to the device.

## What runs on the device

The device path is the *resident radix lifecycle*: a fixed domain box, a radix
grid of leaf cells, and a [`RadixFMMCache`](@ref) built once at a capacity
and reused every step, with the bodies, the expansions and the results all
living on the device. It is the path for a time-stepping code that keeps its
particles on the GPU (FLOWVPM is the reference consumer): after construction a
step moves no per-body data between host and device and, after one warm-up call
per output layout, allocates nothing.

It is not the tree-based `fmm!(system)` of the [Quick Start](quickstart.md)
made faster; that path stays on the host. Which bodies the device path
accepts, compared with the host FMM:

| | host `fmm!` | device `RadixFMMCache` |
|---|---|---|
| `Point{Source}` | yes | yes |
| `Point{Dipole}` | yes | yes |
| `Point{Vortex}` | yes | yes (Lamb-Helmholtz channel required) |
| `Point{SourceVortex}` | yes | yes (Lamb-Helmholtz channel required) |
| `Filament{Source}`, `Filament{Dipole}`, `Filament{Vortex}` | yes | yes (vortex: Lamb-Helmholtz channel required) |
| `Panel{3,Source}`, `Panel{3,Dipole}`, `Panel{3,SourceDipole}`, `Panel{3,Vortex}` (planar triangles) | yes | yes (vortex sheet: Lamb-Helmholtz channel required) |
| nearfield kernels | any user `direct!` | `SingularSource`, `SingularDipole`, `SingularVortex`, `SingularSourceVortex`, `RegularizedVortex`, `PartitionedVortex`, `TwoPassVortex`, `SourceFilamentKernel`, `DipoleFilamentKernel`, `VortexFilamentKernel`, `SourcePanelKernel`, `DipolePanelKernel`, `SourceDipolePanelKernel`, `VortexSheetPanelKernel` |
| outputs | potential, gradient, hessian | potential, gradient, hessian (with the Lamb-Helmholtz channel on, the potential is the monopole term only) |

Every source system sharing one cache must report the same body type.

## Declaring a device-resident system

A system opts in with two traits beyond the usual ones
([Compatibility Functions](reference.md)):

```julia
FastMultipole.residency(::MySystem) = FastMultipole.DeviceResident()   # default: HostResident()
FastMultipole.device_backend(::MySystem) = CUDABackend()                # or Metal.MetalBackend()
```

and provides the two per-step hooks on device arrays. `source_to_buffer!`
fills the framework's packed source buffer, `data_per_body × n` columns of
`[x, y, z, radius, strength..., extra states...]`; `buffer_to_target!` reads
the output buffer back into the system's own arrays, addressing rows through
the derivative-switch accessors rather than fixed indices:

```julia
using GPUArraysCore: AnyGPUMatrix      # one signature for CUDA and Metal, views included

function FastMultipole.source_to_buffer!(buf::AnyGPUMatrix, sys::MySystem, sort_index)
    buf[1:3, :] .= sys.positions
    buf[4, :]   .= sys.radii
    buf[5:7, :] .= sys.strengths           # rows 5:4+strength_dims
    buf[8, :]   .= sys.sigma               # extra state, named by the kernel's sigma_row
    return buf
end

function FastMultipole.buffer_to_target!(sys::MySystem, out::AnyGPUMatrix, switch, sort_index)
    g = FastMultipole.gradient_range(switch)
    isempty(g) || (sys.velocity .= view(out, g, :))
    return sys
end
```

The framework delivers the total influence of the evaluation each step;
whether the consumer overwrites or accumulates is its own choice.

## Building the cache and stepping

```julia
options = CUDARadixLifecycleOptions(; precision = Float32,
                                      m2l_strategy = ConcatenatedFixedZM2L())
cache = RadixFMMCache(sys; expansion_order = 4, ell = 3, max_n_bodies = n,
                      bounds = (x_min, box_size), device = true, options)
for step in 1:nsteps
    fmm!(sys, cache; scalar_potential = false, gradient = true)
    advance!(sys)                      # on the device; keep bodies inside the box
end
```

`max_n_bodies` is the capacity: the live count may vary below it from step to
step without any reallocation. `bounds` fixes the domain box for the cache's
lifetime; a body that leaves it makes the next step throw, by contract, and
`recenter!` moves the box. Two option fields need stating on the device:

* `precision`: the default is Float64 from expansion order 4 up; Metal has no
  Float64, and Float32 is the usual choice on an H200 too.
* `m2l_strategy`: the extension builds the `ConcatenatedFixedZM2L` and
  `DenseTranslationM2L` plans; the options constructor's host default is
  neither. (Leaving `options` out entirely lets the cache pick a device-capable
  default for both.)

A regularized vortex kernel also sets the core-size rule the cache enforces at
construction: `rho_t * sigma_max` must fit inside the gap the direct stencil
leaves to the first far-field cell, or the near field would be truncated.

`examples/device_resident_system.jl` is the complete program: a vortex-particle
system on Metal or CUDA, three steps of convection on the device, and the
transfer counters checked flat across the steps.

## Testing on a device

`Pkg.test()` runs the host suites everywhere. `FASTMULTIPOLE_GPU_TESTS=0|1`
overrides GPU detection (hosted CI sets `0`). On Apple, the bundled
`test/metal_env` project supplies Metal. On NVIDIA, the device suites run only
when `FASTMULTIPOLE_GPU_TEST_PROJECT` points to a CUDA-enabled Julia project;
otherwise they are skipped even when a GPU is detected.
The suites compare every stage of the device lifecycle, and the whole of it,
against the host implementation of the same lifecycle, for both body types;
From the repository root, run them directly with
`FASTMULTIPOLE_GPU_TEST_PROJECT=/path/to/cuda/project bash test/metal_env/run_suites.sh`.
The script prints one line per suite. On a cluster, build that environment on
the login node first; compute nodes rarely have network access.

## Restrictions

* One body type and one `strength_dims` per cache.
* One GPU per cache, one queue (stream) per lifecycle: a wake lives on a single device and the stages run back to back with no transfer/compute overlap. Multi-GPU memory pooling and stream overlap are planned (needed for AD through the vortex particle method, which multiplies device memory) but not available.
* The domain box and `ell` are fixed; the capacity `max_n_bodies` is fixed.
* No third derivatives. Panels are planar triangles (`Panel{3,TK}`); quadrilaterals only as extra sources with host-side expansions (see [Device Interface](device_interface.md)). The vortex sheet nearfield is a Dunavant quadrature (`VortexSheetPanelKernel(; order)`: 7 points at the default order 2, 13 at order 3), approximate within about one panel size.
* An element (filament or panel) must fit its cell: its packed radius in row 4 has to
  be small against the leaf cell, as the expansion about the cell center is
  only valid outside the element.
* Metal: Float32 only.
