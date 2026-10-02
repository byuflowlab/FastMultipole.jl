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
    return FastMultipole._radix_cache_workspace(TF, basis_info, exemplar, Int(ell), h0,
        Int(max_cells), Int(max_nodes), Int(route_capacity), accepted_offsets, invariant,
        m2l_strategy, FastMultipole.MaterializedYRotationM2L();
        ell_axes=ell_axes, first_level=Int(first_level), stage_batch=Int(stage_batch),
        # the device M2L always passes the epoch window cache's classes, so the
        # plan's own class stream only needs one chunk
        m2l_route_class_capacity=m2l_strategy.chunk)
end
