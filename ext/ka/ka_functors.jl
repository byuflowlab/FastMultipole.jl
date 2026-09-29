#------- precision-parameterized device functors (KA-only) -------#
#
# WHY THIS EXISTS. `PartitionedVortex`, `RegularizedVortex` and `TwoPassVortex`
# store their cutoffs as HARD `Float64` fields -- `rho_t::Float64` /
# `rho_c::Float64`, force-coerced by the inner constructors
# (src/containers.jl). The direct-pair kernels take the
# functor BY VALUE for compile-time specialization, so those fields cross into
# device code as doubles.
#
# On CUDA that is invisible: H100/H200 have native FP64 units, the field loads
# and the comparison simply execute in double. Metal has no Float64 at all, so
# the same IR fails to compile outright:
#
#   InvalidIRError: ... gpu_ka_direct_pairs_warp_kernel!(::PartitionedVortex,
#   ...) resulted in invalid LLVM IR / Reason: unsupported use of double value
#
# The per-pair ARITHMETIC is already precision-generic and needs no change:
# `_direct_pair_ug`/`_ugh` convert the cutoff with `T(...)`
# (src/resident/resident_pair_kernels.jl), `_gaussianerf_g_h` converts
# every constant with `T(...)`, and Float32 series coefficients already exist
# (`_gausserf_series_g(z::Float32)`). The double is materialized by the
# FIELD LOAD, which happens before any of that -- so no use-site conversion can
# remove it. The type has to arrive on the device already carrying `TF`.
#
# These mirrors do exactly that and nothing else. The precision follows what the
# particle field passes in (`options.precision`, i.e. FLOWVPM's
# `RadixFMMSettings.precision`), so a Float64 field on a Float64-capable backend
# still runs in Float64 -- this is not a downcast, it is the removal of a
# HARDCODED Float64 from a type that should always have been parameterized.
#
# The host types in `src/containers.jl` stay as they are; the host path keeps
# building and consuming the stock `PartitionedVortex`. Conversion happens on
# the host, in `_ka_device_direct_kernel`, right before each device launch.
#
# The duplicated code is the ~10-line functor WRAPPER only. All real math --
# `_gaussianerf_g_h`, `_vortex_pair_ug`, `_vortex_pair_ugh` -- is called
# straight out of FastMultipole, so the numerics cannot drift from the host
# oracle: same functions, same order, only the cutoff's storage type differs.

"""
    KAPartitionedVortex{TF}(sigma_row, rho_t, inv_sigma_row)

Device mirror of `PartitionedVortex` with the cutoff stored as `TF` instead of
a hardcoded `Float64`. `inv_sigma_row > 0` is the source-body row holding
1/sigma (0: divide by the `sigma_row` value instead). See the block comment
above.
"""
struct KAPartitionedVortex{TF} <: FastMultipole.AbstractDirectKernel
    sigma_row::Int
    rho_t::TF
    inv_sigma_row::Int
end

"""
    KARegularizedVortex{TF}(sigma_row, rho_t, inv_sigma_row)

Device mirror of `RegularizedVortex`; fields as for [`KAPartitionedVortex`](@ref).
"""
struct KARegularizedVortex{TF} <: FastMultipole.AbstractDirectKernel
    sigma_row::Int
    rho_t::TF
    inv_sigma_row::Int
end

const KARegularizedFunctor{TF} =
    Union{KAPartitionedVortex{TF},KARegularizedVortex{TF}}

# trait parity with the host functors (`_emits_potential` in src/containers.jl)
FastMultipole._emits_potential(::KARegularizedFunctor) = false

# regularization cutoff, mirroring the host `_direct_pair_ug` exactly: the
# partitioned kernel goes singular beyond rho_t, the regularized kernel
# regularizes every pair (TwoPassVortex is refused at cache build).
@inline _ka_pass1_cutoff(k::KAPartitionedVortex) = k.rho_t
@inline _ka_pass1_cutoff(k::KARegularizedVortex{TF}) where TF = typemax(TF)

"""
    _ka_device_direct_kernel(kernel, ::Type{TF}, inv_sigma_row=0) -> device functor

Host-side conversion, called before each device launch. Singular kernels carry
no float fields and are returned unchanged; the regularized family is rebuilt
with `TF` cutoffs and the reciprocal-sigma row `inv_sigma_row` (0: none).
"""
_ka_device_direct_kernel(k::FastMultipole.AbstractDirectKernel, ::Type{TF},
    inv_sigma_row::Integer=0) where TF = k
_ka_device_direct_kernel(k::FastMultipole.PartitionedVortex, ::Type{TF},
    inv_sigma_row::Integer=0) where TF =
    KAPartitionedVortex{TF}(k.sigma_row, TF(k.rho_t), Int(inv_sigma_row))
_ka_device_direct_kernel(k::FastMultipole.RegularizedVortex, ::Type{TF},
    inv_sigma_row::Integer=0) where TF =
    KARegularizedVortex{TF}(k.sigma_row, TF(k.rho_t), Int(inv_sigma_row))

"""
    _ka_kernel_sigma_row(kernel) -> Int

`sigma_row` for the regularized family, 0 for every kernel that carries no
sigma. Host-side only; decides whether the reciprocal-sigma row is allocated.
"""
_ka_kernel_sigma_row(::FastMultipole.AbstractDirectKernel) = 0
_ka_kernel_sigma_row(k::FastMultipole.PartitionedVortex) = k.sigma_row
_ka_kernel_sigma_row(k::FastMultipole.RegularizedVortex) = k.sigma_row

# rho = |r|/sigma for the regularized family, plus the guard value the caller
# tests for `> 0`. With the reciprocal-sigma row wired (`inv_sigma_row > 0`) this
# is one load and one MULTIPLY; without it, a load and a divide. The
# branch is on a struct field, so it is uniform across every thread in the launch
# and there is exactly one compiled variant per functor type either way.
#
# The guard is exact, not approximate: the pack kernel stores 0 in the reciprocal
# row for every body with sigma <= 0, so `inv_sigma > 0` selects the same bodies
# `sigma > 0` did. `rho` differs from the divide by at most one Float32 rounding.
@inline function _ka_rho_and_guard(kernel, source_bodies, j, r2::T, invr::T) where T
    isr = kernel.inv_sigma_row
    if isr > 0
        @inbounds invsig = source_bodies[isr, j]
        return r2 * invr * invsig, invsig
    else
        @inbounds sigma = source_bodies[kernel.sigma_row, j]
        return sigma > zero(T) ? r2 * invr / sigma : zero(T), sigma
    end
end

# Mirror of the host `_direct_pair_ug(::PartitionedVortex, ...)`
# (src/resident/resident_pair_kernels.jl): the same statements, except that the
# cutoff is already `TF` and rho may come from the reciprocal-sigma row. `_gaussianerf_g_h` and `_vortex_pair_ug` are FastMultipole's own.
@inline function FastMultipole._direct_pair_ug(kernel::KARegularizedFunctor,
        dx, dy, dz, r2, invr, source_bodies, j)
    T = typeof(r2)
    @inbounds gsx = source_bodies[5, j]
    @inbounds gsy = source_bodies[6, j]
    @inbounds gsz = source_bodies[7, j]
    rho, guard = _ka_rho_and_guard(kernel, source_bodies, j, r2, invr)
    g = one(T)
    if guard > zero(T)
        if rho <= T(_ka_pass1_cutoff(kernel))
            g, _ = FastMultipole._gaussianerf_g_h(rho)
        end
    end
    return FastMultipole._vortex_pair_ug(dx, dy, dz, invr, gsx, gsy, gsz, g)
end

# all-pairs extra-source hooks for the mirror kernels (oversize particles): the
# host definitions in src/radix_extra_systems.jl, on the typed functor
@inline function FastMultipole._extra_pair_ug(kernel::KARegularizedFunctor, tx, ty, tz, source_buffer, j)
    T = typeof(tx)
    @inbounds dx = tx - source_buffer[1, j]
    @inbounds dy = ty - source_buffer[2, j]
    @inbounds dz = tz - source_buffer[3, j]
    r2 = dx * dx + dy * dy + dz * dz
    r2 > zero(T) || return zero(T), zero(T), zero(T), zero(T)
    return FastMultipole._direct_pair_ug(kernel, dx, dy, dz, r2, inv(sqrt(r2)), source_buffer, j)
end
@inline function FastMultipole._extra_pair_ugh(kernel::KARegularizedFunctor, tx, ty, tz, source_buffer, j)
    T = typeof(tx)
    @inbounds dx = tx - source_buffer[1, j]
    @inbounds dy = ty - source_buffer[2, j]
    @inbounds dz = tz - source_buffer[3, j]
    r2 = dx * dx + dy * dy + dz * dz
    z = zero(T)
    r2 > z || return z, z, z, z, z, z, z, z, z, z, z, z, z
    return FastMultipole._direct_pair_ugh(kernel, dx, dy, dz, r2, inv(sqrt(r2)), source_buffer, j)
end
FastMultipole._extra_pair_has_hessian(::KARegularizedFunctor) = true

@inline function FastMultipole._direct_pair_ugh(kernel::KARegularizedFunctor,
        dx, dy, dz, r2, invr, source_bodies, j)
    T = typeof(r2)
    @inbounds gsx = source_bodies[5, j]
    @inbounds gsy = source_bodies[6, j]
    @inbounds gsz = source_bodies[7, j]
    rho, guard = _ka_rho_and_guard(kernel, source_bodies, j, r2, invr)
    g = one(T)
    h = -T(3)
    if guard > zero(T)
        if rho <= T(_ka_pass1_cutoff(kernel))
            g, h = FastMultipole._gaussianerf_g_h(rho)
        end
    end
    return FastMultipole._vortex_pair_ugh(dx, dy, dz, r2, invr, gsx, gsy, gsz, g, h)
end



