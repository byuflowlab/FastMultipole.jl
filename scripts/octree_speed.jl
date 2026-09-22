using Pkg
this_dir = @__DIR__
Pkg.activate(normpath(this_dir,".."))
include("../test/gravitational.jl")
# using BenchmarkTools
using Random
using WriteVTK
# using BSON

function generate_gravitational(seed, n_bodies; radius_factor=0.1, strength_factor=1/n_bodies)
    Random.seed!(123)
    bodies = rand(8,n_bodies)
    # bodies[1:3,3] .=  0.811770914672987, 0.15526131946379113, 0.30656077208169424
    # bodies[1:3,3] .=   0.7427186184997012, 0.2351893322824516, 0.3380666354208596
    bodies[4,:] ./= (n_bodies^(1/3)*2)
    bodies[4,:] .*= radius_factor
    bodies[5,:] .*= strength_factor
    system = Gravitational(bodies)
end

expansion_order, leaf_size, multipole_acceptance = 10, 500, 0.4
n_bodies = 1_000_000
leaf_size = 100_000
multipole_acceptance = 0.5
nearfield=false

system = generate_gravitational(123, n_bodies)

@time _, _, tt, st, _ = fmm!(system; expansion_order=3, leaf_size, multipole_acceptance, gradient=true, scalar_potential=true, hessian=false, shrink_recenter=false, nearfield)
@time fmm!(system; expansion_order=3, leaf_size, multipole_acceptance, gradient=false, scalar_potential=true, hessian=false, shrink_recenter=false, nearfield)
@time fmm!(system; expansion_order=3, leaf_size, multipole_acceptance, gradient=false, scalar_potential=true, hessian=false, shrink_recenter=false, nearfield)
potential_fmm = system.potential[1,:]

system.potential .= 0.0
# direct!(system; scalar_potential=true, gradient=false, hessian=false)
# potential_direct = system.potential[1,:]

# max_err = maximum(abs.(potential_fmm - potential_direct))
# @show max_err
# @time fmm!(system; expansion_order=7, leaf_size=50, multipole_acceptance=0.5, gradient=false, scalar_potential=true, hessian=false)
# @time fmm!(system; expansion_order=7, leaf_size=50, multipole_acceptance=0.5, gradient=false, scalar_potential=true, hessian=false)



println("done.")




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

