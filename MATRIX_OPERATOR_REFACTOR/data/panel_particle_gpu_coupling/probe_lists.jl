# Probe: why is the CPU fmm arm bit-identical to exact direct in
# fm051_attribution_debug.jl? Count m2l/direct pairs for the same body.
import FLOWPanel as pnl
import FastMultipole as fm
using StaticArrays

const FPROOT = "/Users/ryan/Dropbox/research/projects/FLOWPanel.jl"
include(joinpath(FPROOT, "test", "test_helpers.jl"))

nspan = 200
body = make_dirichlet_diamond_body(; nspan=nspan, thick=0.06, das=0.15)
body.needs_velocity_gradient[] = false
for j in axes(body.strength, 2), i in axes(body.strength, 1)
    body.strength[i, j] = sin(0.7 * i + 1.3 * j) + 0.1
end
body.core_size_panel = 1e-10
body.core_size_targets = 1e-3
pnl._set_core_sizes!((body,), :core_size_targets)

ds = fm.DerivativesSwitch([false], [true], [false], (body,))
cache = fm.Cache((body,), (body,), ds)
leaf = fm.to_vector(16, 1)

for ilm in (fm.SelfTuningTargetStop(), fm.SelfTuning(), fm.Barba())
    tt = fm.Tree((body,), fm.TargetTree(), ds, Float64;
        buffers=cache.target_buffers, small_buffers=cache.target_small_buffers,
        expansion_order=4, leaf_size=leaf, shrink=true, recenter=false,
        interaction_list_method=ilm)
    st = fm.Tree((body,), fm.SourceTree(), ds, Float64;
        buffers=cache.source_buffers, small_buffers=cache.source_small_buffers,
        expansion_order=4, leaf_size=leaf, shrink=true, recenter=false,
        interaction_list_method=ilm)
    println("method $(nameof(typeof(ilm))): n_branches tgt=$(length(tt.branches)) src=$(length(st.branches)) leaves=$(length(st.leaf_index))")
    for mac in (0.4, 0.5, 0.9)
        m2l, dl = fm.build_interaction_lists(tt.branches, st.branches, leaf, mac,
            true, true, true, ilm)
        println("  mac=$(mac): m2l=$(length(m2l))  direct=$(length(dl))")
    end
end
