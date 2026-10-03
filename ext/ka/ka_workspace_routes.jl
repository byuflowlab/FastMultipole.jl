#------- RESIDENT OPERATOR WORKSPACE -------#

"""
    ka_radix_cache_workspace(backend, TF, basis_info, ell, h0, max_cells, max_nodes,
                             route_capacity, accepted_offsets, invariant;
                             ell_axes, first_level, m2l_strategy, stage_batch)

Build a device-resident [`ResidentOperatorWorkspace`](@ref) on any KA backend.

This is a thin forward to `FastMultipole._radix_cache_workspace`, which is already
backend-generic: every allocation in it goes through `similar(exemplar.phi, ...)`,
`_array_like_vector(exemplar.phi, ...)` or `DegreeMajorMaps(TF, P, exemplar.phi)`,
so handing it a KA-array exemplar returns a KA-resident workspace.

The KA path builds the `ConcatenatedFixedZM2L` plan with
`MaterializedYRotationM2L`, which matches the host bit for bit; the
dense and factored strategies stay host-only.

If this ever needs a keyword the generic builder does not already take, fix the
genericity in `src/translate_batched.jl` rather than branching here.
"""
function ka_radix_cache_workspace(backend, ::Type{TF},
        basis_info::FastMultipole.OperatorBasisInfo{B,LH}, ell::Integer, h0::TF,
        max_cells::Integer, max_nodes::Integer, route_capacity::Integer,
        accepted_offsets::Vector{SVector{3,Int}},
        invariant::FastMultipole.OperatorInvariantCache;
        ell_axes::SVector{3,Int}=SVector(Int(ell), Int(ell), Int(ell)),
        first_level::Integer=0,
        m2l_strategy::FastMultipole.ConcatenatedFixedZM2L=FastMultipole.ConcatenatedFixedZM2L(),
        stage_batch::Integer=1 << 14) where {TF,B,LH}
    exemplar = _ka_flat_buffer(backend, TF, basis_info, 1)
    function build(chunk)
        return FastMultipole._radix_cache_workspace(TF, basis_info, exemplar, Int(ell), h0,
            Int(max_cells), Int(max_nodes), Int(route_capacity), accepted_offsets, invariant,
            FastMultipole.ConcatenatedFixedZM2L(chunk), FastMultipole.MaterializedYRotationM2L();
            ell_axes=ell_axes, first_level=Int(first_level), stage_batch=Int(stage_batch),
            # the device M2L always passes the epoch window cache's classes, so
            # the plan's own class stream only needs one chunk
            m2l_route_class_capacity=chunk,
            # the device concat M2L reads per-class tables and reuses its slabs
            # (ka_resident_m2l_concat_apply!), so the plan skips that scratch
            slim_m2l=true)
    end
    m2l_strategy.chunk == 0 || return build(m2l_strategy.chunk)
    # automatic: sized from free device memory
    return build(_ka_auto_m2l_chunk(backend, TF, basis_info))
end

# Automatic concat-M2L chunk (`ConcatenatedFixedZM2L(0)`): the largest power of
# two up to the backend cap whose scratch fits a tenth of the free device
# memory, when the backend's package reports it. Cap: 2^17 on GPUs, where large
# chunks are much faster (H200 UJ at 1M bodies: 2.27 s at 2^14, 1.24 s at 2^17),
# but 2^15 on Metal, whose measured best it was (2^16 was slower). The scratch
# estimate, 18 x ndof values per column, is a little above the measured slim
# plan (3.3 KB per column at P = 6, Float32).
const _KA_MIN_M2L_CHUNK = 1 << 12

function _ka_auto_m2l_chunk(backend, ::Type{TF}, basis_info) where TF
    cap = nameof(typeof(backend)) === :MetalBackend ? 1 << 15 : 1 << 17
    free = FastMultipole._device_free_bytes(backend)
    free === nothing && return cap
    orders = basis_info.orders
    ndof = max(FastMultipole.degree_major_dof(orders.P_phi),
               FastMultipole.degree_major_dof(orders.P_active))
    fit = (free ÷ 10) ÷ (18 * ndof * sizeof(TF))
    chunk = cap
    while chunk > _KA_MIN_M2L_CHUNK && chunk > fit
        chunk >>= 1
    end
    return chunk
end
