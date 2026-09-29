#------- RECTANGULAR (SOURCE-SET -> DISTINCT TARGET-SET) DIRECT EVALUATION -------#
#
# Brute-force cross-pass evaluation outside the FMM: every column of a source
# matrix influences every column of a distinct target matrix. The pair math is
# supplied by a consumer kernel type (see `AbstractRectangularKernel`); this file
# owns only the argument checks and the loop over targets. The device loop is
# installed by the KernelAbstractions extension and calls the same `rect_pair`.

"""
    AbstractRectangularKernel

Supertype for kernels accepted by [`direct_rectangular!`](@ref). A kernel must
be an isbits type (it is passed into device kernels) and defines

- [`rect_source_rows`](@ref)`(kernel)`: the number of packed rows per source column;
- [`rect_pair`](@ref)`(kernel, target, sources, q, Val(gradient), Val(scalar_potential))`:
  the influence of source column `q` at `target`.

Optionally it overloads [`rect_has_potential`](@ref) (default `false`) and
[`rect_check_sources`](@ref) (default: no check). When `rect_pair` is
GPU-compilable, `direct_rectangular!` also runs on device arrays once a device
backend extension is loaded; no device-specific method is needed.
"""
abstract type AbstractRectangularKernel end

"Return the required number of packed source rows for a rectangular kernel."
function rect_source_rows end

"""
    rect_pair(kernel, target::SVector{3,T}, sources, q, ::Val{GRAD}, ::Val{POT}) -> (u, J, phi)

Influence of source column `q` of `sources` at `target`: velocity
`u::SVector{3,T}`, velocity gradient `J::SMatrix{3,3,T,9}` with
`J[i, j] = ∂u_i/∂x_j` (may be zero when `GRAD` is false) and scalar potential
`phi::T` (may be zero when `POT` is false). Must not allocate; inlined into the
host and device target loops.
"""
function rect_pair end

"Whether a rectangular kernel can return the scalar potential. Defaults to `false`."
rect_has_potential(::AbstractRectangularKernel) = false

"""
    rect_check_sources(kernel, sources; scalar_potential=false)

Optional validation of the packed source columns before evaluation; throw an
`ArgumentError` for a malformed layout. Defaults to no check.
"""
rect_check_sources(::AbstractRectangularKernel, sources; scalar_potential::Bool=false) = nothing

"Return the required output rows for rectangular velocity, gradient, and potential output."
rect_output_rows(gradient::Bool, scalar_potential::Bool=false) =
    (gradient ? 12 : 3) + scalar_potential
rect_potential_row(gradient::Bool) = gradient ? 13 : 4

function _rect_check_args(out, targets, kernel, sources, gradient::Bool,
        scalar_potential::Bool=false)
    size(targets, 1) >= 3 || throw(ArgumentError(
        "targets must have at least 3 rows (position); got $(size(targets, 1))"))
    size(sources, 1) >= rect_source_rows(kernel) || throw(ArgumentError(
        "$(typeof(kernel)) sources require $(rect_source_rows(kernel)) rows; " *
        "got $(size(sources, 1))"))
    size(out, 2) == size(targets, 2) || throw(ArgumentError(
        "out has $(size(out, 2)) columns but targets has $(size(targets, 2))"))
    size(out, 1) >= rect_output_rows(gradient, scalar_potential) ||
        throw(ArgumentError(
        "out must have at least $(rect_output_rows(gradient, scalar_potential)) " *
        "rows for gradient=$gradient, scalar_potential=$scalar_potential; " *
        "got $(size(out, 1))"))
    scalar_potential && !rect_has_potential(kernel) &&
        throw(ArgumentError("scalar potential is not available for $(typeof(kernel))"))
    rect_check_sources(kernel, sources; scalar_potential)
    return nothing
end

# The host method has an AbstractMatrix signature, so a GPU-array call made
# before a device backend extension installs its method would land here and
# die inside Threads.@threads scalar indexing with an opaque error. Fail with
# the actual fix instead.
_rect_root_parent(a) = (p = parent(a); p === a ? a : _rect_root_parent(p))

function _rect_assert_host(out, targets, sources)
    (_rect_root_parent(out) isa Array && _rect_root_parent(targets) isa Array &&
        _rect_root_parent(sources) isa Array) || throw(ArgumentError(
        "direct_rectangular! host method called with non-host arrays " *
        "($(typeof(out))); for GPU arrays load a device backend extension " *
        "first (and pass out/targets/sources all on the same side)"))
    return nothing
end

# sum over all sources at target column i, then accumulate into out; shared by
# the host loop below and the device kernel in the KernelAbstractions extension
@inline function _rect_target!(out, targets, kernel, sources, i, n_sources,
        ::Val{GRAD}, ::Val{POT}) where {GRAD,POT}
    T = eltype(out)
    @inbounds begin
        target = SVector{3,T}(targets[1, i], targets[2, i], targets[3, i])
        u = zero(SVector{3,T})
        g = zero(SMatrix{3,3,T,9})
        p = zero(T)
        for q in 1:n_sources
            uq, gq, pq = rect_pair(kernel, target, sources, q, Val(GRAD), Val(POT))
            u += uq
            GRAD && (g += gq)
            POT && (p += pq)
        end
        out[1, i] += u[1]; out[2, i] += u[2]; out[3, i] += u[3]
        if GRAD
            for j in 1:3, k in 1:3
                out[3 + (j-1)*3 + k, i] += g[k, j]
            end
        end
        POT && (out[rect_potential_row(GRAD), i] += p)
    end
    return nothing
end

"""
    direct_rectangular!(out, targets, kernel, sources;
        gradient=false, scalar_potential=false)

Brute-force rectangular direct evaluation: every source column of `sources`
influences every target column of `targets` (rows 1:3 = position; extra rows
are ignored), accumulating (+=) velocity into `out[1:3, :]` and, when
`gradient=true`, the velocity gradient into `out[4:12, :]` (order
`out[3 + (j-1)*3 + i] = ∂u_i/∂x_j`). Zero `out` first for a fresh evaluation.
`scalar_potential=true` adds the potential in row 4 (`gradient=false`) or row 13
(`gradient=true`) for kernels with [`rect_has_potential`](@ref). The source row
layout is set by `kernel`; see [`AbstractRectangularKernel`](@ref). The host
method is threaded over targets; device-array methods are installed by a device
backend extension.
"""
function direct_rectangular!(out::AbstractMatrix{T}, targets::AbstractMatrix{T},
        kernel::AbstractRectangularKernel, sources::AbstractMatrix{T};
        gradient::Bool=false, scalar_potential::Bool=false) where T
    _rect_check_args(out, targets, kernel, sources, gradient, scalar_potential)
    _rect_assert_host(out, targets, sources)
    _rect_host!(out, targets, kernel, sources, Val(gradient), Val(scalar_potential))
    return out
end

function _rect_host!(out, targets, kernel, sources, grad::Val, pot::Val)
    n_sources = size(sources, 2)
    Threads.@threads for i in 1:size(targets, 2)
        _rect_target!(out, targets, kernel, sources, i, n_sources, grad, pot)
    end
    return nothing
end
