# Local gate for `ka_lifecycle_body!` -- the whole uniform radix lifecycle
# (nearfield -> B2M -> M2M -> M2L -> L2L -> L2B), run end to end on a KA
# backend and compared against FastMultipole's OWN host resident lifecycle.
#
# Why this file exists: the session-3 note recorded `ka_lifecycle_body!` as
# "reachable only on a CUDA-resident state", and the acceptance gate was
# therefore aimed straight at H200. That is not true. `host_radix_state`
# (src/translate_batched_resident.jl) builds a complete
# `DeviceResidentRadixState` with no CUDA anywhere -- host body matrix, host
# tree routes, flat coefficient buffers and a real `ResidentOperatorWorkspace`
# -- and `run_host_radix_lifecycle!` runs every stage of it on the CPU. The
# state type carries no array type in its parameters, so the same state is
# constructible on any KA backend. That makes the whole lifecycle body gateable
# locally, against an exact CPU oracle, before any HPC round trip.
#
# The oracle is `run_host_radix_lifecycle!` over host `Array`s: B2M, then the
# operator pipeline M2M -> M2L -> L2L -> L2B, with `_add_host_direct_pairs!`
# folded into L2B. `ka_lifecycle_body!` runs nearfield first (clearing output)
# and folds direct into that stage instead; the stage ORDER differs but the sum
# does not, because output accumulates and starts zeroed on both sides.
#
# What this deliberately does NOT cover, so no one reads it as the acceptance
# gate:
#   1. `ka_hierarchical_m2l!`. `host_radix_state` stores a flat
#      `RadixInteractionList`, so `ka_launch_m2l!` takes its flat branch here.
#      The hierarchical branch reuses `_cuda_hier_generate_window!` for window
#      generation, which is CUDA-only by construction, so it stays an H200 gate.
#   2. The cache-level call. `_radix_cache_device_step!` is a stub that throws
#      `CUDARadixUnavailable` until translate_batched_cuda.jl redefines it, so
#      a full `UJ_fmm` through a `RadixFMMCache` cannot run off CUDA.
#   3. Timing. Metal-vs-CPU numbers say nothing about KA-vs-native on one GPU.
#
# What it DOES cover is every stage FLOWVPM's uniform per-step lifecycle runs,
# plus the sequencing between them -- which is where a silent wrong answer of
# the kind that bit the M2L branch would show up.
#
# Body types: `Point{Vortex}` with Lamb-Helmholtz on (what FLOWVPM runs), and
# `Point{Source}` with and without the channel (the library default body, the
# gravitational test system), each against the same host lifecycle.
include("ka_backend.jl")
using FastMultipole, Random, Test
using FastMultipole.StaticArrays

const FM = FastMultipole

# the repo's own vortex system + generator (correct traits, body_type Point{Vortex})
include(joinpath(@__DIR__, "..", "vortex.jl"))
include(joinpath(@__DIR__, "..", "gravitational.jl"))   # Point{Source} system

if !dev_functional()
    println("$(DEV_NAME) not functional; skipping")
    exit(0)
end
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext !== nothing || error("FastMultipoleKAExt did not load")

relerr(a, b) = (d = maximum(abs.(Array(a) .- Array(b))); s = maximum(abs.(Array(b)));
                s == 0 ? d : d / s)

#------- host state -> device state -------#
#
# `DeviceRadixGrid` and `DeviceResidentRadixState` are both fully generic in
# their array types, so the mirror is a field-by-field upload. The `host_*`
# fields of the state stay host `Array`s by contract (that is what they are
# for); everything else moves. `invariant_cache` stays host too: the KA stage
# drivers read only `basis_info.orders` off it, which is host scalars.

to_dev(x::AbstractArray) = devarray(x)
to_dev(x) = x

function dev_grid(g::FM.DeviceRadixGrid)
    return FM.DeviceRadixGrid(
        g.x_min, g.h0, g.ell, g.n_bodies, g.n_cells,
        to_dev(g.perm), to_dev(g.invperm), to_dev(g.cell_keys), to_dev(g.cell_ranges),
        to_dev(g.body_system), to_dev(g.body_index), to_dev(g.cell_centers),
        to_dev(g.node_levels), to_dev(g.node_keys), to_dev(g.node_coords),
        to_dev(g.node_centers), to_dev(g.parent_index), to_dev(g.child_ranges),
        to_dev(g.leaf_to_node))
end

# Rebuild the flat coefficient buffer on device rather than uploading it: the
# buffer carries its `basis_info` and the chi slab is 0x0 when LH is off, and
# `_ka_flat_buffer` is the ext's own allocator for exactly this shape.
function dev_flat(backend, buf::FM.FlatCoefficientBuffer{TF}) where TF
    b = ext._ka_flat_buffer(backend, TF, buf.basis_info, size(buf.phi, 2))
    copyto!(b.phi, buf.phi)
    size(b.chi, 2) > 0 && copyto!(b.chi, buf.chi)
    return b
end

function dev_state(hs::FM.DeviceResidentRadixState{TF,B,LH}, backend) where {TF,B,LH}
    g = dev_grid(hs.grid)
    src = to_dev(hs.source_bodies)
    mult = dev_flat(backend, hs.multipoles)
    locs = dev_flat(backend, hs.locals)
    # The workspace must be built through the SAME constructor `host_radix_state`
    # uses, off the concrete grid and list -- it is backend-generic, deciding
    # every array type from the `exemplar` buffer, so a device exemplar puts the
    # whole workspace on the backend. The cache-shaped `ka_radix_cache_workspace`
    # is NOT interchangeable here: it builds stage groups from capacity plus an
    # accepted-offset set rather than from this grid's node table, so its
    # m2m/l2l groups do not address the same nodes and M2M silently fills
    # nothing.
    ws = FM.ResidentOperatorWorkspace(TF, hs.invariant_cache.basis_info, mult,
        hs.grid, hs.interaction_list,
        hs.host_m2m_parent_routes, hs.host_m2m_child_routes,
        hs.host_l2l_parent_routes, hs.host_l2l_child_routes,
        hs.host_node_levels, hs.host_node_centers,
        hs.host_route_targets, hs.host_route_sources;
        m2l_strategy=FM.ConcatenatedFixedZM2L(),
        operator=hs.options.operator)
    return FM.DeviceResidentRadixState{TF,B,LH}(
        g, hs.interaction_list, src, src,
        to_dev(hs.body_perm), to_dev(hs.body_system_ids), to_dev(hs.body_indices),
        hs.host_body_perm, hs.host_body_system_ids, hs.host_body_indices,
        hs.host_cell_centers, hs.host_m2m_parent_routes, hs.host_m2m_child_routes,
        hs.host_l2l_parent_routes, hs.host_l2l_child_routes, hs.host_node_levels,
        hs.host_node_centers, hs.host_route_targets, hs.host_route_sources,
        to_dev(hs.cell_centers), to_dev(hs.cell_ranges),
        to_dev(hs.m2m_parent_routes), to_dev(hs.m2m_child_routes),
        to_dev(hs.l2l_parent_routes), to_dev(hs.l2l_child_routes),
        mult, locs,
        to_dev(hs.route_levels), to_dev(hs.route_offsets),
        to_dev(hs.route_targets), to_dev(hs.route_sources),
        to_dev(hs.direct_targets), to_dev(hs.direct_sources), to_dev(hs.output),
        hs.invariant_cache, ws, FM.CUDARadixTransferCounters(), hs.options, hs.counts,
    )
end

#------- cases -------#

# (P, ell, n) -- Float32 throughout, the precision Metal supports and the one
# the CUDA arm runs for FLOWVPM.
# A unit run takes the short case list; FM_FULL_SWEEP=1 takes the full one.
# The sweep is a robustness study, not a check: it belongs in a debugging pass
# (debug/run_full_sweeps.sh), not in every run.
const CASES_FULL = [
    (2, 3,   64),
    (4, 3,  256),
    (4, 4, 1024),
    (6, 3,  256),
    (8, 4, 2048),
]
const CASES_SHORT = [
    (4, 3, 256),
    (8, 4, 512),   # a second expansion order and a deeper tree: the
                   # operator tables are built per order
]
const CASES = haskey(ENV, "FM_FULL_SWEEP") ? CASES_FULL : CASES_SHORT

const TOL = 2e-4  # Float32 whole-lifecycle accumulation; per-stage suites run 1e-4/1e-5

npass = Ref(0); nfail = Ref(0)

for (ci, (P, ell, n)) in pairs(CASES)
    t_case = time()
    TF = Float32
    # Float32 all the way down: `generate_vortex` is Float64, which makes
    # `RadixGrid` Float64 and the grid upload throw on Metal even though
    # options.precision is Float32.
    Random.seed!(4400 + ci)
    system = VortexParticles(rand(TF, 3, n), (randn(TF, 3, n) ./ TF(n)),
        zeros(TF, n); potential=zeros(TF, 13, n),
        gradient_stretching=zeros(TF, 6, n))

    grid = FM.RadixGrid(system, ell)
    list = FM.build_radix_interaction_list(FM.LazyMaterializedBatches(1),
        FM.ParentNeighborM2L(), grid)
    opts = FM.CUDARadixLifecycleOptions(; precision=TF,
        m2l_strategy=FM.ConcatenatedFixedZM2L(), body_type=FM.Point{FM.Vortex})

    hs = FM.host_radix_state(system, grid, list, P, Val(true); options=opts)

    local ds
    try
        ds = dev_state(hs, DEV_BACKEND)
    catch err
        nfail[] += 1
        println("case $ci (P=$P, ell=$ell, n=$n): device state build FAILED: $err")
        continue
    end

    FM.run_host_radix_lifecycle!(hs)
    try
        ext.ka_lifecycle_body!(ds)
    catch err
        nfail[] += 1
        println("case $ci (P=$P, ell=$ell, n=$n): ka_lifecycle_body! THREW: $err")
        continue
    end

    e_mul = relerr(ds.multipoles.phi, hs.multipoles.phi)
    e_loc = relerr(ds.locals.phi, hs.locals.phi)
    e_out = relerr(ds.output, hs.output)
    ok = e_mul < TOL && e_loc < TOL && e_out < TOL
    ok ? (npass[] += 1) : (nfail[] += 1)
    println("[$(round(Int, time() - t_case))s] case $ci (P=$P, ell=$ell, n=$n): ", ok ? "PASS" : "FAIL",
        "  multipoles=", e_mul, "  locals=", e_loc, "  output=", e_out)
end

println("\nka_lifecycle_body! vs run_host_radix_lifecycle! (Point{Vortex}): $(npass[])/$(length(CASES)) pass")
nfail[] == 0 || error("lifecycle gate failed (vortex)")

# --- Point{Source}: the gravitational system, with and without Lamb-Helmholtz ---
npass[] = 0; nfail[] = 0; ncase = 0
for (ci, (P, ell, n)) in pairs(CASES), lh in (false, true)
    global ncase += 1
    t_case = time()
    TF = Float32
    Random.seed!(4500 + ci)
    bodies = rand(TF, 8, n)
    bodies[4, :] ./= TF(n^(1 / 3) * 2); bodies[4, :] .*= TF(0.1)
    bodies[5, :] ./= TF(n)
    system = Gravitational(bodies)
    grid = FM.RadixGrid(system, ell)
    list = FM.build_radix_interaction_list(FM.LazyMaterializedBatches(1),
        FM.ParentNeighborM2L(), grid)
    opts = FM.CUDARadixLifecycleOptions(; precision=TF,
        m2l_strategy=FM.ConcatenatedFixedZM2L(), body_type=FM.Point{FM.Source})
    hs = FM.host_radix_state(system, grid, list, P, Val(lh); options=opts)
    local ds
    try
        ds = dev_state(hs, DEV_BACKEND)
    catch err
        nfail[] += 1
        println("source case $ci LH=$lh (P=$P, ell=$ell, n=$n): device state build FAILED: $err")
        continue
    end
    FM.run_host_radix_lifecycle!(hs)
    try
        ext.ka_lifecycle_body!(ds)
    catch err
        nfail[] += 1
        println("source case $ci LH=$lh (P=$P, ell=$ell, n=$n): ka_lifecycle_body! THREW: $err")
        continue
    end
    e_mul = relerr(ds.multipoles.phi, hs.multipoles.phi)
    e_loc = relerr(ds.locals.phi, hs.locals.phi)
    e_out = relerr(ds.output, hs.output)
    ok = e_mul < TOL && e_loc < TOL && e_out < TOL
    ok ? (npass[] += 1) : (nfail[] += 1)
    println("[$(round(Int, time() - t_case))s] source case $ci LH=$lh (P=$P, ell=$ell, n=$n): ", ok ? "PASS" : "FAIL",
        "  multipoles=", e_mul, "  locals=", e_loc, "  output=", e_out)
end
println("\nka_lifecycle_body! vs run_host_radix_lifecycle! (Point{Source}): $(npass[])/$ncase pass")
nfail[] == 0 || error("lifecycle gate failed (source)")

# --- Point{Dipole} and Point{SourceVortex}: packed-matrix systems ---
struct PackedPoints{TF,BT}
    data::Matrix{TF}
end
Base.eltype(::PackedPoints{TF}) where TF = TF
FM.get_n_bodies(s::PackedPoints) = size(s.data, 2)
FM.data_per_body(s::PackedPoints) = size(s.data, 1)
FM.strength_dims(s::PackedPoints) = size(s.data, 1) - 4
FM.get_position(s::PackedPoints{TF}, i) where TF = SVector{3,TF}(s.data[1, i], s.data[2, i], s.data[3, i])
FM.body_type(::PackedPoints{TF,BT}) where {TF,BT} = BT
FM.has_vector_potential(::PackedPoints{TF,BT}) where {TF,BT} = BT <: FM.Point{FM.SourceVortex}
FM.source_system_to_buffer!(buffer, i_buffer, s::PackedPoints, i_body) =
    (buffer[1:size(s.data, 1), i_buffer] .= view(s.data, :, i_body))
npass[] = 0; nfail[] = 0; ncase = 0
for (ci, (P, ell, n)) in pairs(CASES),
        (label, BT, dpb, lh) in (("dipole", FM.Point{FM.Dipole}, 7, false),
                                 ("dipole LH", FM.Point{FM.Dipole}, 7, true),
                                 ("source-vortex", FM.Point{FM.SourceVortex}, 8, true))
    global ncase += 1
    t_case = time()
    TF = Float32
    Random.seed!(4600 + ci)
    data = rand(TF, dpb, n); data[4, :] .= TF(1e-3); data[5:end, :] .= (data[5:end, :] .- TF(0.5)) ./ TF(n)
    system = PackedPoints{TF,BT}(data)
    grid = FM.RadixGrid(system, ell)
    list = FM.build_radix_interaction_list(FM.LazyMaterializedBatches(1),
        FM.ParentNeighborM2L(), grid)
    opts = FM.CUDARadixLifecycleOptions(; precision=TF,
        m2l_strategy=FM.ConcatenatedFixedZM2L(), body_type=BT)
    hs = FM.host_radix_state(system, grid, list, P, Val(lh); options=opts)
    local ds
    try
        ds = dev_state(hs, DEV_BACKEND)
    catch err
        nfail[] += 1
        println("$label case $ci (P=$P, ell=$ell, n=$n): device state build FAILED: $err")
        continue
    end
    FM.run_host_radix_lifecycle!(hs)
    try
        ext.ka_lifecycle_body!(ds)
    catch err
        nfail[] += 1
        println("$label case $ci (P=$P, ell=$ell, n=$n): ka_lifecycle_body! THREW: $err")
        continue
    end
    e_mul = relerr(ds.multipoles.phi, hs.multipoles.phi)
    e_loc = relerr(ds.locals.phi, hs.locals.phi)
    e_out = relerr(ds.output, hs.output)
    ok = e_mul < TOL && e_loc < TOL && e_out < TOL
    ok ? (npass[] += 1) : (nfail[] += 1)
    println("[$(round(Int, time() - t_case))s] $label case $ci (P=$P, ell=$ell, n=$n): ", ok ? "PASS" : "FAIL",
        "  multipoles=", e_mul, "  locals=", e_loc, "  output=", e_out)
end
println("\nka_lifecycle_body! vs run_host_radix_lifecycle! (Point{Dipole}, Point{SourceVortex}): $(npass[])/$ncase pass")
nfail[] == 0 || error("$(nfail[]) case(s) failed")

# --- straight filaments: packed systems with vertices ---
struct PackedFilaments{TF,BT}
    data::Matrix{TF}
end
Base.eltype(::PackedFilaments{TF}) where TF = TF
FM.get_n_bodies(s::PackedFilaments) = size(s.data, 2)
FM.data_per_body(s::PackedFilaments) = size(s.data, 1)
FM.strength_dims(::PackedFilaments{TF,BT}) where {TF,BT} = FM.element_strength_dims(BT)
FM.get_position(s::PackedFilaments{TF}, i) where TF = SVector{3,TF}(s.data[1, i], s.data[2, i], s.data[3, i])
FM.body_type(::PackedFilaments{TF,BT}) where {TF,BT} = BT
FM.has_vector_potential(::PackedFilaments{TF,BT}) where {TF,BT} = BT <: FM.Filament{FM.Vortex}
FM.source_system_to_buffer!(buffer, i_buffer, s::PackedFilaments, i_body) =
    (buffer[1:size(s.data, 1), i_buffer] .= view(s.data, :, i_body))
npass[] = 0; nfail[] = 0; ncase = 0
for (ci, (P, ell, n)) in pairs(CASES),
        (label, BT, sd, lh) in (("source filament", FM.Filament{FM.Source}, 1, false),
                               ("dipole filament", FM.Filament{FM.Dipole}, 3, false),
                               ("vortex filament", FM.Filament{FM.Vortex}, 3, true))
    global ncase += 1
    t_case = time()
    TF = Float32
    Random.seed!(4700 + ci)
    mids = rand(TF, 3, n); dirs = randn(TF, 3, n); dirs ./= sqrt.(sum(dirs .^ 2; dims = 1))
    len = TF(0.004)
    data = zeros(TF, 4 + sd + 6, n)
    data[1:3, :] .= mids; data[4, :] .= len / 2
    data[5:4+sd, :] .= (rand(TF, sd, n) .- TF(0.5)) ./ TF(n)
    BT <: FM.Filament{FM.Vortex} && (data[5:7, :] .= dirs .* ((rand(TF, 1, n) .- TF(0.5)) ./ TF(n)))
    data[5+sd:7+sd, :] .= mids .- dirs .* (len / 2); data[8+sd:10+sd, :] .= mids .+ dirs .* (len / 2)
    system = PackedFilaments{TF,BT}(data)
    grid = FM.RadixGrid(system, ell)
    list = FM.build_radix_interaction_list(FM.LazyMaterializedBatches(1),
        FM.ParentNeighborM2L(), grid)
    opts = FM.CUDARadixLifecycleOptions(; precision=TF,
        m2l_strategy=FM.ConcatenatedFixedZM2L(), body_type=BT)
    hs = FM.host_radix_state(system, grid, list, P, Val(lh); options=opts)
    local ds
    try
        ds = dev_state(hs, DEV_BACKEND)
    catch err
        nfail[] += 1
        println("$label case $ci (P=$P, ell=$ell, n=$n): device state build FAILED: $err")
        continue
    end
    FM.run_host_radix_lifecycle!(hs)
    try
        ext.ka_lifecycle_body!(ds)
    catch err
        nfail[] += 1
        println("$label case $ci (P=$P, ell=$ell, n=$n): ka_lifecycle_body! THREW: $err")
        continue
    end
    e_mul = relerr(ds.multipoles.phi, hs.multipoles.phi)
    e_loc = relerr(ds.locals.phi, hs.locals.phi)
    e_out = relerr(ds.output, hs.output)
    ok = e_mul < TOL && e_loc < TOL && e_out < TOL
    ok ? (npass[] += 1) : (nfail[] += 1)
    println("[$(round(Int, time() - t_case))s] $label case $ci (P=$P, ell=$ell, n=$n): ", ok ? "PASS" : "FAIL",
        "  multipoles=", e_mul, "  locals=", e_loc, "  output=", e_out)
end
println("\nka_lifecycle_body! vs run_host_radix_lifecycle! (filaments): $(npass[])/$ncase pass")
nfail[] == 0 || error("$(nfail[]) filament case(s) failed")
