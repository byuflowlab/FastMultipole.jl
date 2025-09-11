using FastMultipole

include("../test/gravitational.jl")
include("simple_gravitational.jl")

for expansion_order in [] #[1,2,16,17,18,19,20]
    println("\n\n#=== P=$expansion_order ===#")
    multipole_acceptance = 0.4
    n_bodies = 1_000_000
    leaf_size = 50
    sys = generate_gravitational(123, n_bodies; radius_factor=0)
    tree = Tree(sys; expansion_order, leaf_size, shrink_recenter=false)

    # upward pass
    FastMultipole.upward_pass_singlethread!(tree.branches, sys, tree.expansion_order)

    # horizontal pass, existing functions
    farfield, nearfield, self_induced = true, true, true
    m2l_list, direct_target_bodies, direct_source_bodies = FastMultipole.build_interaction_lists(tree.branches, tree.branches, tree.leaf_index, multipole_acceptance, farfield, nearfield, self_induced)

    #@time FastMultipole.horizontal_pass_singlethread!(tree.branches, tree.branches, m2l_list, tree.expansion_order)
    #@time FastMultipole.horizontal_pass_singlethread!(tree.branches, tree.branches, m2l_list, tree.expansion_order)
    @time FastMultipole.horizontal_pass_singlethread!(tree.branches, tree.branches, m2l_list, tree.expansion_order)

end

#--- VS Code Profiler
# @profview FastMultipole.horizontal_pass_singlethread!(tree.branches, tree.branches, m2l_list, tree.expansion_order)

#--- PProf Profiler
# using Profile
# using PProf
# @profile FastMultipole.horizontal_pass_singlethread!(tree.branches, tree.branches, m2l_list, tree.expansion_order)
# Profile.clear()
# @profile FastMultipole.horizontal_pass_singlethread!(tree.branches, tree.branches, m2l_list, tree.expansion_order)
# pprof()

using PythonPlot

ps_old = collect(1:20)
old_FastMultipole = [1.0, 1.7, 3.1, 5.8, 13, 23, 37, 57, 84, 119, 310, 420, 563, 737, 945, 1200, 1502, 1861, 2275, 2761]
ps_new = collect(1:20)
new_FastMultipole = [0.77, 1.5, 2.6, 4.4, 7.4, 11, 16, 24, 33, 45, 59, 77, 100, 127, 157, 195, 241, 294, 355, 427]

fig = figure("benchmark_20240914")
fig.clear()
fig.add_subplot(xlabel="expansion order", ylabel="time, seconds")
ax = fig.get_axes()[0]
cmap = get_cmap("RdBu",7)
ax.plot(ps_old, old_FastMultipole, color=cmap(0.15), "-x")
ax.plot(ps_new, new_FastMultipole, color=cmap(0.85), "->")
ax.legend(["old FastMultipole", "new expansions"])
ax.set_yscale("log")
ax.set_xscale("log")

#p3_line_x = [5.0,8]
#p3_line_y = [5.0^3, 8^3] .- 100.0
#p4_line_x = [5.0,8]
#p4_line_y = [5.0^4, 8^4] .- 1000.0
#
#ax.plot(p4_line_x, p4_line_y, "--", color=cmap(0.15))
#ax.plot(p3_line_x, p3_line_y, "--", color=cmap(0.85))

#asymptotic behavior:
get_slope(ps, ts; n=3) = (log(ts[end]) - log(ts[end-n])) / (log(ps[end]) - log(ps[end-n]))
slope_old = get_slope(ps_old, old_FastMultipole)
slope_new = get_slope(ps_new, new_FastMultipole)
@show slope_old, slope_new
