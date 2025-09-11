using Pkg
this_dir = @__DIR__
Pkg.activate(normpath(this_dir,".."))
include("../test/gravitational.jl")
# using BenchmarkTools
using Random
using WriteVTK
# using BSON

function generate_gravitational(seed, n_bodies; radius_factor=0.1)
    Random.seed!(123)
    bodies = rand(8,n_bodies)
    # bodies[1:3,3] .=  0.811770914672987, 0.15526131946379113, 0.30656077208169424
    # bodies[1:3,3] .=   0.7427186184997012, 0.2351893322824516, 0.3380666354208596
    bodies[4,:] ./= (n_bodies^(1/3)*2)
    bodies[4,:] .*= radius_factor
    system = Gravitational(bodies)
end

function bm_fmm_1024()
    expansion_order, leaf_size, multipole_acceptance = 5, 100, 0.4
    n_bodies = 1024
    system = generate_gravitational(123, n_bodies)
    # tree = fmm.Tree(system; expansion_order, leaf_size, shrink_recenter=true)
    fmm.fmm!(system; expansion_order=expansion_order, leaf_size=leaf_size, multipole_acceptance=multipole_acceptance, nearfield=true, farfield=true, unsort_bodies=true, shrink_recenter=false)
    return nothing
end

function bm_fmm_4096()
    expansion_order, leaf_size, multipole_acceptance = 5, 100, 0.4
    n_bodies = 4096
    system = generate_gravitational(123, n_bodies)
    # tree = fmm.Tree(system; expansion_order, leaf_size, shrink_recenter=true)
    fmm.fmm!(system; expansion_order=expansion_order, leaf_size=leaf_size, multipole_acceptance=multipole_acceptance, nearfield=true, farfield=true, unsort_bodies=true, shrink_recenter=false)
    return nothing
end

function bm_fmm_16384()
    expansion_order, leaf_size, multipole_acceptance = 5, 100, 0.4
    n_bodies = 16384
    system = generate_gravitational(123, n_bodies)
    # tree = fmm.Tree(system; expansion_order, leaf_size, shrink_recenter=true)
    fmm.fmm!(system; expansion_order=expansion_order, leaf_size=leaf_size, multipole_acceptance=multipole_acceptance, nearfield=true, farfield=true, unsort_bodies=true, shrink_recenter=false)
    return nothing
end

function bm_fmm_65536()
    expansion_order, leaf_size, multipole_acceptance = 5, 100, 0.4
    n_bodies = 65536
    system = generate_gravitational(123, n_bodies)
    # tree = fmm.Tree(system; expansion_order, leaf_size, shrink_recenter=true)
    fmm.fmm!(system; expansion_order=expansion_order, leaf_size=leaf_size, multipole_acceptance=multipole_acceptance, nearfield=true, farfield=true, unsort_bodies=true, shrink_recenter=false)
    return nothing
end

function bm_fmm_262144()
    expansion_order, leaf_size, multipole_acceptance = 5, 100, 0.4
    n_bodies = 262144
    system = generate_gravitational(123, n_bodies)
    # tree = fmm.Tree(system; expansion_order, leaf_size, shrink_recenter=true)
    fmm.fmm!(system; expansion_order=expansion_order, leaf_size=leaf_size, multipole_acceptance=multipole_acceptance, nearfield=true, farfield=true, unsort_bodies=true, shrink_recenter=false)
    return nothing
end

function bm_fmm_1048576()
    expansion_order, leaf_size, multipole_acceptance = 5, 100, 0.4
    n_bodies = 1048576
    system = generate_gravitational(123, n_bodies)
    # tree = fmm.Tree(system; expansion_order, leaf_size, shrink_recenter=true)
    fmm.fmm!(system; expansion_order=expansion_order, leaf_size=leaf_size, multipole_acceptance=multipole_acceptance, nearfield=true, farfield=true, unsort_bodies=true, shrink_recenter=false)
    return nothing
end

function bm_fmm_4194304()
    expansion_order, leaf_size, multipole_acceptance = 5, 100, 0.4
    n_bodies = 4194304
    system = generate_gravitational(123, n_bodies)
    # tree = fmm.Tree(system; expansion_order, leaf_size, shrink_recenter=true)
    fmm.fmm!(system; expansion_order=expansion_order, leaf_size=leaf_size, multipole_acceptance=multipole_acceptance, nearfield=true, farfield=true, unsort_bodies=true, shrink_recenter=false)
    return nothing
end

function bm_fmm()
    expansion_order, leaf_size, multipole_acceptance = 5, 100, 0.4
    n_bodies = 10_000
    system = generate_gravitational(123, n_bodies)
    # tree = fmm.Tree(system; expansion_order, leaf_size, shrink_recenter=true)
    fmm.fmm!(system; expansion_order=expansion_order, leaf_size=leaf_size, multipole_acceptance=multipole_acceptance, nearfield=true, farfield=true, unsort_bodies=true, shrink_recenter=false)
    return nothing
end

function bm_fmm_system(system)
    expansion_order, leaf_size, multipole_acceptance = 5, 100, 0.4
    n_bodies = 10_000
    # tree = fmm.Tree(system; expansion_order, leaf_size, shrink_recenter=true)
    fmm.fmm!(system; expansion_order=expansion_order, leaf_size=leaf_size, multipole_acceptance=multipole_acceptance, nearfield=true, farfield=true, unsort_bodies=true, shrink_recenter=false)
    return nothing
end

function bm_direct()
    n_bodies = 5000
    system2 = generate_gravitational(123, n_bodies)
    fmm.direct!(system2, 1:n_bodies, system2, 1:n_bodies)
    return sum(system2.potential)
end

function bm_fmm_accuracy(expansion_order, leaf_size, multipole_acceptance, n_bodies, shrink_recenter)
    system = generate_gravitational(123, n_bodies)
    # system = (generate_gravitational(123, n_bodies),)
    println("Create tree")
    @time tree = fmm.Tree(system; expansion_order, leaf_size, shrink_recenter=shrink_recenter)
    println("Run fmm")
    @time fmm.fmm!(tree, system; multipole_acceptance=multipole_acceptance, reset_tree=true, nearfield=true, farfield=true, unsort_bodies=true)
    # println("Run fmm again")
    # @time fmm.fmm!(tree, system; multipole_acceptance=multipole_acceptance, reset_tree=true, nearfield=true, farfield=true, unsort_bodies=true)
    println("BEGIN DIRECT")
    system2 = generate_gravitational(123, n_bodies)
    @time fmm.direct!(system2, 1:n_bodies, system2, 1:n_bodies)
    phi = system.potential[1,:]
    # phi = system[1].potential[1,:]
    phi2 = system2.potential[1,:]
    return maximum(abs.(phi2 - phi)), system, tree, system2
end

function bm_fmm_accuracy_dual_tree(expansion_order, leaf_size, multipole_acceptance, n_bodies, shrink_recenter)
    system = (generate_gravitational(123, n_bodies),)
    println("Create trees")
    @time source_tree = fmm.Tree(system; expansion_order, leaf_size, shrink_recenter=shrink_recenter)
    @time target_tree = fmm.Tree(system; expansion_order, leaf_size, shrink_recenter=shrink_recenter)
    println("Run fmm")
    @time fmm.fmm!(target_tree, system, source_tree, system; multipole_acceptance=multipole_acceptance, nearfield=true, farfield=true)
    println("BEGIN DIRECT")
    system2 = generate_gravitational(123, n_bodies)
    @time fmm.direct!(system2, 1:n_bodies, system2, 1:n_bodies)
    phi = system[1].potential[1,:]
    phi2 = system2.potential[1,:]
    return maximum(abs.(phi2 - phi)), system, source_tree, target_tree, system2
end

function bm_fmm_accuracy_dual_tree_wrapped(expansion_order, leaf_size, multipole_acceptance, n_bodies, shrink_recenter)
    println("Create systems")
    source_system = generate_gravitational(123, n_bodies)
    target_system = fmm.SortWrapper(source_system)

    println("Create trees")
    @time source_tree = fmm.Tree(source_system; expansion_order, leaf_size, shrink_recenter=shrink_recenter)
    @time target_tree = fmm.Tree(target_system; expansion_order, leaf_size, shrink_recenter=shrink_recenter)

    println("Run fmm")
    @time fmm.fmm!(target_tree, target_system, source_tree, source_system; multipole_acceptance=multipole_acceptance, nearfield=true, farfield=true, unsort_source_bodies=true, unsort_target_bodies=true)

    println("BEGIN DIRECT")
    system2 = generate_gravitational(123, n_bodies)
    @time fmm.direct!(system2, 1:n_bodies, system2, 1:n_bodies)
    phi = target_system.system.potential[1,:]
    phi2 = system2.potential[1,:]

    return maximum(abs.(phi2 - phi)), source_system, source_tree, target_system, target_tree, system2
end

function bm_fmm_accuracy_dual_tree_wrapped_multiple(expansion_order, leaf_size, multipole_acceptance, n_bodies, shrink_recenter)
    println("Create systems")
    source_system1 = generate_gravitational(123, n_bodies)
    source_system2 = generate_gravitational(456, n_bodies)
    target_system1 = generate_gravitational(789, n_bodies)
    target_system2 = fmm.SortWrapper(source_system1)

    println("Create trees")
    @time source_tree = fmm.Tree((source_system1, source_system2); expansion_order, leaf_size, shrink_recenter=shrink_recenter)
    @time target_tree = fmm.Tree((target_system1, target_system2); expansion_order, leaf_size, shrink_recenter=shrink_recenter)

    println("Run fmm")
    @time fmm.fmm!(target_tree, (target_system1, target_system2), source_tree, (source_system1, source_system2); multipole_acceptance=multipole_acceptance, nearfield=true, farfield=true, unsort_source_bodies=true, unsort_target_bodies=true)

    println("BEGIN DIRECT")
    source_system_direct = (generate_gravitational(123, n_bodies), generate_gravitational(456, n_bodies))
    target_system_direct = (generate_gravitational(789, n_bodies), source_system_direct[1])
    @time fmm.direct!(target_system_direct, source_system_direct)
    phi = target_system2.system.potential[1,:]
    phi2 = target_system_direct[2].potential[1,:]

    return maximum(abs.(phi2 - phi)), source_system, source_tree, target_system, target_tree, system2
end

function bm_fmm_accuracy_dual_tree_wrapped_multiple_nested(expansion_order, leaf_size, multipole_acceptance, n_bodies, shrink_recenter)
    println("Create systems")
    source_system1 = fmm.SortWrapper(generate_gravitational(123, n_bodies))
    source_system2 = generate_gravitational(456, n_bodies)
    target_system1 = generate_gravitational(789, n_bodies)
    target_system2 = fmm.SortWrapper(source_system1)

    println("Create trees")
    @time source_tree = fmm.Tree((source_system1, source_system2); expansion_order, leaf_size, shrink_recenter=shrink_recenter)
    @time target_tree = fmm.Tree((target_system1, target_system2); expansion_order, leaf_size, shrink_recenter=shrink_recenter)

    println("Run fmm")
    @time fmm.fmm!(target_tree, (target_system1, target_system2), source_tree, (source_system1, source_system2); multipole_acceptance=multipole_acceptance, nearfield=true, farfield=true, unsort_source_bodies=true, unsort_target_bodies=true)

    println("BEGIN DIRECT")
    source_system_direct = (generate_gravitational(123, n_bodies), generate_gravitational(456, n_bodies))
    target_system_direct = (generate_gravitational(789, n_bodies), source_system_direct[1])
    @time fmm.direct!(target_system_direct, source_system_direct)
    phi = target_system2.system.system.potential[1,:]
    phi2 = target_system_direct[2].potential[1,:]

    return maximum(abs.(phi2 - phi)), source_system, source_tree, target_system, target_tree, system2
end

function bm_fmm_accuracy_dual_tree_wrapped_multiple_nested_api(expansion_order, leaf_size, multipole_acceptance, n_bodies, shrink_recenter)
    println("Create systems")
    source_system1 = fmm.SortWrapper(generate_gravitational(123, n_bodies))
    source_system2 = generate_gravitational(456, n_bodies)
    target_system1 = generate_gravitational(789, n_bodies)
    target_system2 = fmm.SortWrapper(source_system1)

    println("Run fmm")
    @time fmm.fmm!((target_system1, target_system2), (source_system1, source_system2); expansion_order, leaf_size_source=leaf_size, leaf_size_target=leaf_size, multipole_acceptance=multipole_acceptance, nearfield=true, farfield=true, unsort_source_bodies=true, unsort_target_bodies=true)

    println("BEGIN DIRECT")
    source_system_direct = (generate_gravitational(123, n_bodies), generate_gravitational(456, n_bodies))
    target_system_direct = (generate_gravitational(789, n_bodies), source_system_direct[1])
    @time fmm.direct!(target_system_direct, source_system_direct)
    phi = target_system2.system.system.potential[1,:]
    phi2 = target_system_direct[2].potential[1,:]

    return maximum(abs.(phi2 - phi)), source_system, source_tree, target_system, target_tree, system2
end

function bm_fmm_accuracy_dual_tree_wrapped_multiple_nested_api_twice(expansion_order, leaf_size, multipole_acceptance, n_bodies, shrink_recenter)
    println("Create systems")
    source_system1 = fmm.SortWrapper(generate_gravitational(123, n_bodies))
    source_system2 = generate_gravitational(456, n_bodies)
    target_system1 = generate_gravitational(789, n_bodies)
    target_system2 = fmm.SortWrapper(source_system1)

    println("Run fmm")
    @time fmm.fmm!((target_system1, target_system2), (source_system1, source_system2); expansion_order, leaf_size_source=leaf_size, leaf_size_target=leaf_size, multipole_acceptance=multipole_acceptance, nearfield=true, farfield=true, unsort_source_bodies=true, unsort_target_bodies=true)
    @time fmm.fmm!((target_system1, target_system2), (source_system1, source_system2); expansion_order, leaf_size_source=leaf_size, leaf_size_target=leaf_size, multipole_acceptance=multipole_acceptance, nearfield=true, farfield=true, unsort_source_bodies=true, unsort_target_bodies=true)

    println("BEGIN DIRECT")
    source_system_direct = (generate_gravitational(123, n_bodies), generate_gravitational(456, n_bodies))
    target_system_direct = (generate_gravitational(789, n_bodies), source_system_direct[1])
    @time fmm.direct!(target_system_direct, source_system_direct)
    @time fmm.direct!(target_system_direct, source_system_direct)
    phi = target_system2.system.system.potential[1,:]
    phi2 = target_system_direct[2].potential[1,:]

    return maximum(abs.(phi2 - phi)), source_system, source_tree, target_system, target_tree, system2
end

function visualize_tree(name, system, tree; probe_indices=[])
    #####
    ##### branches
    #####
    branches = tree.branches
    n_branches = length(branches)
    branch_locations = Array{Float64,4}(undef,3,n_branches,1,1)
    branch_radii = Array{Float64,3}(undef,n_branches,1,1)
    for (i,branch) in enumerate(branches)
        branch_locations[:,i,1,1] .= branch.center
        branch_radii[i,1,1] = branch.radius
    end
    vtk_grid(name*"_branches", branch_locations) do vtk
        vtk["radius"] = branch_radii
    end

    #####
    ##### bodies
    #####
    n_bodies = fmm.get_n_bodies(system)
    body_locations = Array{Float64,4}(undef,3,n_bodies,1,1)
    body_radii = Array{Float64,3}(undef,n_bodies,1,1)
    scalar_strength = Array{Float64,3}(undef,n_bodies,1,1)
    vector_strength = Array{Float64,4}(undef,3,n_bodies,1,1)
    for i in 1:n_bodies
        body_locations[:,i,1,1] .= system[i,fmm.POSITION]
        body_radii[i,1,1] = system[i,fmm.RADIUS]
        scalar_strength[i,1,1] = system[i,fmm.SCALAR_STRENGTH]
        vector_strength[:,i,1,1] .= system[i,fmm.VECTOR_STRENGTH]
    end
    vtk_grid(name*"_bodies", body_locations) do vtk
        vtk["radius"] = body_radii
        vtk["scalar strength"] = scalar_strength
        vtk["vector strength"] = vector_strength
    end

    #####
    ##### probes
    #####
    n_probes = length(probe_indices)
    if n_probes > 0
        body_locations = Array{Float64,4}(undef,3,n_probes,1,1)
        body_radii = Array{Float64,3}(undef,n_probes,1,1)
        scalar_strength = Array{Float64,3}(undef,n_probes,1,1)
        vector_strength = Array{Float64,4}(undef,3,n_probes,1,1)
        for (i_probe,i_body) in enumerate(probe_indices)
            body_locations[:,i_probe,1,1] .= system[i_body,fmm.POSITION]
            body_radii[i_probe,1,1] = system[i_body,fmm.RADIUS]
            scalar_strength[i_probe,1,1] = system[i_body,fmm.SCALAR_STRENGTH]
            vector_strength[:,i_probe,1,1] .= system[i_body,fmm.VECTOR_STRENGTH]
        end
        vtk_grid(name*"_probes", body_locations) do vtk
            vtk["radius"] = body_radii
            vtk["scalar strength"] = scalar_strength
            vtk["vector strength"] = vector_strength
            vtk["index"] = probe_indices
        end
    end

    return nothing
end
# println("generate bodies")
# @time generate_gravitational(123, 5000)
# @time generate_gravitational(123, 5000)


# shrink_recenter = false
# farfield=nearfield=true
expansion_order, leaf_size, multipole_acceptance = 10, 500, 0.4
n_bodies = 10_000

shrink_recenter = true
# println("create system...")
# sys = generate_gravitational(123, n_bodies)
# println("create tree...")
# tree = fmm.Tree(sys; leaf_size=leaf_size, expansion_order=expansion_order, n_divisions=n_divisions)
# println("done.")
# err, system, tree, system2 = bm_fmm_accuracy(expansion_order, leaf_size, multipole_acceptance, n_bodies, shrink_recenter)
# @show err

println("===== nthreads: $(Threads.nthreads()) =====")
# err, sys, tree, sys2 = bm_fmm_accuracy(expansion_order, leaf_size, multipole_acceptance, n_bodies, shrink_recenter)
# @show err

n_bodies = 10_000
system = generate_gravitational(123, n_bodies)
optimized_args, cache, target_tree, source_tree, m2l_list, direct_list, derivatives_switches, error_success = fmm!(system)
# bm_fmm_system(system)
# @time bm_fmm_system(system)





# why is it spending so much time precompiling? Apparently because I am creating a new system in the benchmark function
# using BenchmarkTools
# system = generate_gravitational(123, n_bodies)
# @btime bm_fmm_system($system)

# println("===== nthreads: $(Threads.nthreads()) =====")
# err, sys, tree, sys2 = bm_fmm_accuracy(expansion_order, leaf_size, multipole_acceptance, n_bodies, shrink_recenter)
# @show err
# err_ns, sys_ns, tree_ns, sys2_ns = bm_fmm_accuracy(expansion_order, leaf_size, multipole_acceptance, n_bodies, false)
# @show err_ns

# err, source_system, source_tree, target_system, target_tree, system2 = bm_fmm_accuracy_dual_tree_wrapped(expansion_order, leaf_size, multipole_acceptance, n_bodies, shrink_recenter)
# @show err
# err, source_system, source_tree, target_system, target_tree, system2 = bm_fmm_accuracy_dual_tree_wrapped_multiple(expansion_order, leaf_size, multipole_acceptance, n_bodies, shrink_recenter)
# @show err
# err, source_system, source_tree, target_system, target_tree, system2 = bm_fmm_accuracy_dual_tree_wrapped_multiple_nested(expansion_order, leaf_size, multipole_acceptance, n_bodies, shrink_recenter)
# @show err
# err, source_system, source_tree, target_system, target_tree, system2 = bm_fmm_accuracy_dual_tree_wrapped_multiple_nested_api(expansion_order, leaf_size, multipole_acceptance, n_bodies, shrink_recenter)
# @show err
# err, source_system, source_tree, target_system, target_tree, system2 = bm_fmm_accuracy_dual_tree_wrapped_multiple_nested_api_twice(expansion_order, leaf_size, multipole_acceptance, n_bodies, shrink_recenter)
# @show err

# bm_fmm_accuracy_dual_tree(expansion_order, leaf_size, multipole_acceptance, n_bodies, shrink_recenter)

# println("===== begin benchmark: $(Threads.nthreads()) threads =====")
# ts = zeros(7)
# println("n = 1024")
# bm_fmm_1024()
# ts[1] = @elapsed bm_fmm_1024()
# @show ts[1]
# println("n = 4096")
# bm_fmm_4096()
# ts[2] = @elapsed bm_fmm_4096()
# @show ts[2]
# println("n = 16384")
# bm_fmm_16384()
# ts[3] = @elapsed bm_fmm_16384()
# @show ts[3]
# println("n = 65536")
# bm_fmm_65536()
# ts[4] = @elapsed bm_fmm_65536()
# @show ts[4]
# println("n = 262144")
# bm_fmm_262144()
# ts[5] = @elapsed bm_fmm_262144()
# @show ts[5]
# println("n = 1048576")
# bm_fmm_1048576()
# ts[6] = @elapsed bm_fmm_1048576()
# @show ts[6]
# println("n = 4194304")
# bm_fmm_4194304()
# ts[7] = @elapsed bm_fmm_4194304()
# @show ts[7]

# n_bodies = [4^i for i in 5:11]

# BSON.@save "benchmark_$(Threads.nthreads()).bson" ts n_bodies

# # using BenchmarkTools

# #####
# ##### translate multipoles
# #####
# tm_st = []
# tm_mt = []
# mt_tm_fun(this_index) = fmm.translate_multipoles_multi_thread!(tree.branches, expansion_order, this_index)
# st_tm_fun(this_index) = fmm.translate_multipoles_single_thread!(tree.branches, expansion_order, this_index)
# for i in 1:6
#     levels_index = tree.levels_index[i]
#     this_index = [levels_index]
#     mt_tm_fun(this_index)
#     st_tm_fun(this_index)
#     t_mt = @elapsed mt_tm_fun(this_index)
#     t_st = @elapsed st_tm_fun(this_index)
#     # t_mt = @belapsed mt_tm_fun($this_index)
#     # t_st = @belapsed st_tm_fun($this_index)
#     push!(tm_mt, t_mt)
#     push!(tm_st, t_st)
# end
# tm_speedup = tm_st ./ tm_mt
# tm_summary = hcat([length(this_index) for this_index in tree.levels_index[1:6]], tm_st, tm_mt, tm_speedup)
# println("n m2m translations | 1 thread, workstation | 72 threads, workstation | speedup")
# println("--- | --- | --- | ---")
# println(round.(tm_summary, digits=5))

# #####
# ##### b2m
# #####
# b2m_st = []
# b2m_mt = []
# mt_b2m_fun(this_index) = fmm.body_2_multipole_multi_thread!(tree.branches, sys, expansion_order, this_index)
# st_b2m_fun(this_index) = fmm.body_2_multipole_single_thread!(tree.branches, sys, expansion_order, this_index)
# for i in [1, 10, 100, 1000, 10000]
#     this_index = tree.leaf_index[1:i]
#     mt_b2m_fun(this_index)
#     st_b2m_fun(this_index)
#     t_mt = @elapsed mt_b2m_fun(this_index)
#     t_st = @elapsed st_b2m_fun(this_index)
#     # t_mt = @belapsed mt_b2m_fun($this_index)
#     # t_st = @belapsed st_b2m_fun($this_index)
#     push!(b2m_mt,t_mt)
#     push!(b2m_st,t_st)
# end
# b2m_speedup = b2m_st ./ b2m_mt
# b2m_summary = hcat([1, 10, 100, 1000, 10000], b2m_st, b2m_mt, b2m_speedup)
# println("n leaves (b2m) | 1 thread, workstation | 72 threads, workstation | speedup")
# println("--- | --- | --- | ---")
# println(round.(b2m_summary, digits=5))

#####
##### m2l
#####
# m2l_list, direct_list = fmm.build_interaction_lists(tree.branches, multipole_acceptance, farfield, nearfield)
# m2l_st = []
# m2l_mt = []
# mt_m2l_fun(this_index) = fmm.horizontal_pass_multi_thread!(tree.branches, tree.branches, this_index, expansion_order)
# st_m2l_fun(this_index) = fmm.horizontal_pass_single_thread!(tree.branches, tree.branches, this_index, expansion_order)
# for i in [1, 10, 100, 1000, 10000, 100000]
#     this_index = m2l_list[1:i]
#     mt_m2l_fun(this_index)
#     st_m2l_fun(this_index)
#     t_mt = @elapsed mt_m2l_fun(this_index)
#     t_st = @elapsed st_m2l_fun(this_index)
#     # t_mt = @belapsed mt_m2l_fun($this_index)
#     # t_st = @belapsed st_m2l_fun($this_index)
#     push!(m2l_mt, t_mt)
#     push!(m2l_st, t_st)
# end
# m2l_speedup = m2l_st ./ m2l_mt
# m2l_summary = hcat([1, 10, 100, 1000, 10000, 100000], m2l_st, m2l_mt, m2l_speedup)
# println("n m2l transformations | 1 thread | $(Threads.nthreads()) threads | speedup")
# println("--- | --- | --- | ---")
# println(round.(m2l_summary, digits=5))

# #####
# ##### direct
# #####
# direct_mt = []
# direct_st = []
# mt_direct_fun(this_index) = fmm.nearfield_multi_thread!(sys, tree.branches, sys, tree.branches, tree.cost_parameters, this_index)
# st_direct_fun(this_index) = fmm.nearfield_single_thread!(sys, tree.branches, sys, tree.branches, this_index)
# for i in [1, 10, 100, 1000, 10000, 100000, 1000000]
#     println("i passes: $i")
#     this_index = direct_list[1:i]
#     mt_direct_fun(this_index)
#     st_direct_fun(this_index)
#     t_mt = @elapsed mt_direct_fun(this_index)
#     t_st = @elapsed st_direct_fun(this_index)
#     # t_mt = @belapsed mt_direct_fun($this_index)
#     # t_st = @belapsed st_direct_fun($this_index)
#     push!(direct_mt, t_mt)
#     push!(direct_st, t_st)
# end
# direct_speedup = direct_st ./ direct_mt
# direct_summary = hcat([1, 10, 100, 1000, 10000, 100000, 1000000], direct_st, direct_mt, direct_speedup)
# println("n leaves | 1 thread | $(Threads.nthreads()) threads | speedup")
# println("--- | --- | --- | ---")
# println(round.(direct_summary, digits=5))

# #####
# ##### translate locals
# #####
# tl_mt = []
# tl_st = []
# mt_tl_fun(this_index) = fmm.translate_multipoles_multi_thread!(tree.branches, expansion_order, this_index)
# st_tl_fun(this_index) = fmm.translate_multipoles_single_thread!(tree.branches, expansion_order, this_index)
# for i in 1:6
#     levels_index = tree.levels_index[i]
#     this_index = [levels_index]
#     mt_tl_fun(this_index)
#     st_tl_fun(this_index)
#     t_mt = @elapsed mt_tl_fun(this_index)
#     t_st = @elapsed st_tl_fun(this_index)
#     # t_mt = @belapsed mt_tl_fun($this_index)
#     # t_st = @belapsed st_tl_fun($this_index)
#     push!(tl_mt, t_mt)
#     push!(tl_st, t_st)
# end
# tl_speedup = tl_st ./ tl_mt
# tl_summary = hcat([length(tree.levels_index[i]) for i in 1:6], tl_st, tl_mt, tl_speedup)
# println("n l2l translations | 1 thread, workstation | 72 threads, workstation | speedup")
# println("--- | --- | --- | ---")
# println(round.(tl_summary, digits=5))

# #####
# ##### l2b
# #####
# l2b_mt = []
# l2b_st = []
# mt_l2b_fun(this_index) = fmm.local_2_body_multi_thread!(tree.branches, sys, expansion_order, this_index)
# st_l2b_fun(this_index) = fmm.local_2_body_single_thread!(tree.branches, sys, expansion_order, this_index)
# for i in [1, 10, 100, 1000, 10000]
#     println("i leaves: $i")
#     this_index = tree.leaf_index[1:i]
#     mt_l2b_fun(this_index)
#     st_l2b_fun(this_index)
#     t_mt = @elapsed mt_l2b_fun(this_index)
#     t_st = @elapsed st_l2b_fun(this_index)
#     # t_mt = @belapsed mt_l2b_fun($this_index)
#     # t_st = @belapsed st_l2b_fun($this_index)
#     push!(l2b_mt, t_mt)
#     push!(l2b_st, t_st)
# end
# l2b_speedup = l2b_st ./ l2b_mt
# l2b_summary = hcat([1, 10, 100, 1000, 10000], l2b_st, l2b_mt, l2b_speedup)
# println("n leaves | 1 thread, workstation | 72 threads, workstation | speedup")
# println("--- | --- | --- | ---")
# println(round.(l2b_summary, digits=5))

# sys = generate_gravitational(123, 500000)
# tree = fmm.Tree(sys; expansion_order=expansion_order, leaf_size=leaf_size)
# m2l_list, direct_list = fmm.build_interaction_lists(tree.branches, multipole_acceptance, true, true)
# fmm.horizontal_pass_multi_thread!(tree.branches, m2l_list, expansion_order)
# t = @elapsed fmm.horizontal_pass_multi_thread!(tree.branches, m2l_list, expansion_order)
# t_per_op = t / length(m2l_list)
# @show t_per_op

# fmm.horizontal_pass_single_thread!(tree.branches, m2l_list, expansion_order)
# t = @elapsed fmm.horizontal_pass_single_thread!(tree.branches, m2l_list, expansion_order)
# t_per_op = t / length(m2l_list)
# @show t_per_op

# fmm.fmm!(sys)





# nfp, nfe = fmm.get_nearfield_parameters(sys)
# params, errors, nearfield_params, nearfield_errors = fmm.estimate_tau(sys; expansion_orders = 1:9:20, epsilon=0.1, cost_file_read=false, cost_file_write=true)

# note: old_bodies[tree.index_list[1]] = systems[1].bodies
# println("Run FMM:")
# @time bm_fmm()
# @time bm_fmm()

# println("Run Direct:")
# @time bm_direct()
# @time bm_direct()
# @btime fmm.fmm!($tree, $systems, $options; unsort_bodies=true)
# println("Calculating accuracy:")
# expansion_order, leaf_size, multipole_acceptance = 8, 300, 0.3
# n_bodies = 100000
# shrink_recenter, n_divisions = true, 10
# sys = generate_gravitational(123,n_bodies)
# function bmtree()# let sys=sys, expansion_order=expansion_order, leaf_size=leaf_size, n_divisions=n_divisions, shrink_recenter=shrink_recenter
#         return fmm.Tree(sys; expansion_order, leaf_size, n_divisions=n_divisions, shrink_recenter=shrink_recenter)
#     # end
# end
# @time fmm.Tree(sys; expansion_order=expansion_order, leaf_size=leaf_size, n_divisions=n_divisions, shrink_recenter=shrink_recenter)
# fmm.unsort!(sys, tree)
# @time bmtree()
# fmm.unsort!(sys, tree)
# @time bmtree()
# @time bmtree()
# @time bmtree()

# println("done")

# sys_noshrinking = generate_gravitational(123,n_bodies)
# tree_noshrinking = fmm.Tree(sys_noshrinking; expansion_order, leaf_size, n_divisions=5, shrink_recenter=false)

# println("done")
# run_bm_accuracy() = bm_fmm_accuracy(expansion_order, leaf_size, multipole_acceptance, n_bodies, shrink_recenter)
# accuracy, system, tree, system2 = run_bm_accuracy()
# accuracy, system, tree, system2 = run_bm_accuracy()
# println("single tree accuracy: $accuracy")
# run_bm_accuracy_dual_tree() = bm_fmm_accuracy_dual_tree(expansion_order, leaf_size, multipole_acceptance, n_bodies, shrink_recenter)
# accuracy, system, tree, system2 = run_bm_accuracy_dual_tree()
# accuracy, system, tree, system2 = run_bm_accuracy_dual_tree()
# println("dual tree accuracy: $accuracy")

# visualize tree
# visualize_tree("test_fmm", system, tree; probe_indices=[11])
# visualize_tree("test_direct", system2, tree)
# visualize_tree("test_shrinking", sys, tree)
# visualize_tree("test_noshrinking", sys_noshrinking, tree_noshrinking)

