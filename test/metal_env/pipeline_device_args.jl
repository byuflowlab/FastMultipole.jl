# Builds the KA device cache from a host RadixFMMCache.
#
# STAGES 1, 2, 5 AND 6 HAVE LANDED. `LH` is no longer read off the host cache type,
# it is `ka_validate_radix_arguments`' own trait resolution; `x_min`, `h0`,
# `ell_axes` and `box_extent` are no longer read off the host cache's fields,
# they are `ka_radix_geometry`'s own derivation. Both are asserted against the
# host cache below, and those assertions ARE the stages' accuracy checks.
#
# NOTHING is read off the host cache's fields any more. `device_build_args` --
# the copy of ka_device_cache_correctness.jl:105-109 that re-derived the policy
# tables and `max_level_nodes` from the host cache -- is GONE, along with the
# five capacities. What remains below is the CALLER side: the arguments
# FLOWVPM computes and hands `RadixFMMCache`, which are inputs to the front end,
# not part of it.
#
# The host cache is still built, and every stage's output is asserted against
# it. That is now purely a gate, not a source of arguments.

# The arguments FLOWVPM computes and hands `RadixFMMCache` -- CALLER inputs to
# the front end, not part of it -- reproduced from the same settings the bench
# gave `_build_radix_fmm_cache` (FLOWVPM_fmm_radix.jl:521-546, :560-568).
# `bounds`, `ell` and `near_radius2` are one derivation: the center snap and
# the auto-geometry rule feed each other, so they cannot be split.
function production_caller_args(pf, st, ::Type{TF}) where TF
    b = st.bounds === nothing ?
        V._radix_derive_bounds(pf, st.padding; rectangular=st.rectangular) :
        st.bounds
    kernel = V._radix_direct_kernel(st)
    L_geo = b[2] isa Real ? Float64(b[2]) : Float64(maximum(b[2]))
    ell, q = st.ell === nothing ?
        V._radix_auto_geometry(L_geo, Float64(V._radix_sigma_max(pf)), pf.np,
            st.near_radius2, V._radix_primary_reach(kernel), st.accuracy_margin) :
        (st.ell, st.near_radius2)
    if st.bounds === nothing && st.rectangular
        b = V._radix_center_snapped_bounds(b, ell)
    end
    bounds = (SVector{3,TF}(b[1]),
        b[2] isa Real ? TF(b[2]) : SVector{3,TF}(b[2]))
    # device=true, so the window-class default is FLOWVPM's device value
    K = st.window_classes === nothing ? 256 : st.window_classes
    return (; bounds, ell, near_radius2=q, window_classes=K,
        level_radii2=st.level_radii2)
end

function build_ka_cache(ext, hcache, sys_d, c)
    # STAGE 1: argument + trait validation. `LH` used to be read straight off
    # the host cache type (`typeof(hcache).parameters[2]`).
    v = ext.ka_validate_radix_arguments(DEV_BACKEND, (sys_d,);
        expansion_order=hcache.expansion_order, options=hcache.options,
        hessian=hcache.hessian, max_n_bodies=hcache.max_n_bodies)
    v.LH == typeof(hcache).parameters[2] || error(
        "stage 1 resolved lamb_helmholtz=$(v.LH); the host cache says " *
        "$(typeof(hcache).parameters[2])")

    # STAGE 2: root geometry. `x_min`, `h0`, `ell_axes` and `box_extent` used
    # to be read off the host cache's fields.
    g = ext.ka_radix_geometry(v.sources, v.TF, c.ell; bounds=c.bounds)
    for (name, got, want) in (("x_min", g.x_min, hcache.x_min),
                              ("h0", g.h0, hcache.h0),
                              ("ell_axes", g.ell_axes, hcache.ell_axes),
                              ("box_extent", g.box_extent, hcache.box_extent))
        got == want || error("stage 2 $name = $got; the host cache says $want")
    end

    # STAGE 5: stencil policy, hierarchical tables, options finalization. The
    # policy and the offset sets used to come off the host cache; the tables,
    # their metadata and `max_level_nodes` came from `device_build_args`.
    sp = ext.ka_radix_stencil_policy(v, c.ell, g.h0, g.ell_axes;
        near_radius2=c.near_radius2, window_classes=c.window_classes,
        level_radii2=c.level_radii2)
    sp.hierarchical || error("expected a hierarchical policy; got $(typeof(sp.stencil_policy))")
    for (name, got, want) in (("policy", sp.stencil_policy, hcache.policy),
                              ("accepted", sp.accepted, hcache.accepted_offsets),
                              ("rejected", sp.rejected, hcache.rejected_offsets),
                              ("options", sp.options, hcache.options))
        got == want || error("stage 5 $name disagrees with the host cache")
    end

    # STAGE 6: capacity sizing. All five used to be read off the host cache --
    # the numbers that fix every persistent device allocation.
    cap = ext.ka_radix_capacities(v, c.ell, g.ell_axes, sp)
    for (name, got, want) in (("max_cells", cap.max_cells, hcache.max_cells),
                              ("max_nodes", cap.max_nodes, hcache.max_nodes),
                              ("route_capacity", cap.route_capacity, hcache.route_capacity),
                              ("direct_capacity", cap.direct_capacity, hcache.direct_capacity))
        got == want || error("stage 6 $name = $got; the host cache says $want")
    end

    return ext.ka_radix_cache_device_build(DEV_BACKEND, (sys_d,),
        v.P, c.ell, g.x_min, g.h0,
        v.maxn, sp.options, sp.stencil_policy,
        sp.accepted, sp.rejected,
        cap.max_cells, cap.max_nodes, cap.route_capacity,
        cap.direct_capacity, sp.basis_info, Val(v.LH);
        hierarchical_tables=sp.hierarchical_tables,
        class_level=sp.class_level, class_offset=sp.class_offset,
        hierarchical_level_class_of=sp.hierarchical_level_class_of,
        hierarchical_level_radii2=sp.hierarchical_level_radii2,
        max_level_nodes=cap.max_level_nodes, hessian=v.hessian,
        ell_axes=g.ell_axes, box_extent=g.box_extent)
end
