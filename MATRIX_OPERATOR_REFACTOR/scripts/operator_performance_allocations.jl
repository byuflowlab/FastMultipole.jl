# Reproducible CPU allocation and retained-storage measurements for item 019.
#
# Run from the repository root with:
#   julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/operator_performance_allocations.jl
#
# The script writes CSV to stdout. Allocation measurements are taken after one warmup
# call. Retained-storage rows report array payload bytes (not Julia object headers), so
# their values are independent of the host allocator.
using FastMultipole
using Random

const FM = FastMultipole

function body_matrix(::Type{TF}, n::Integer; seed::Integer=19019) where TF
    rng = MersenneTwister(seed)
    return vcat(
        TF.(rand(rng, 3, n)),
        zeros(TF, 1, n),
        reshape(TF.(randn(rng, n)), 1, n),
    )
end

csv_bool(::Val{LH}) where LH = string(LH)

function allocation_rows(::Type{TF}, lh::Val{LH}; n=2_000, ell=3, P=4) where {TF,LH}
    bodies = body_matrix(TF, n; seed=19019 + (LH ? 1 : 0))
    grid = RadixGrid(bodies, ell)
    list = build_radix_interaction_list(
        LazyMaterializedBatches(32), ParentNeighborM2L(), grid,
    )
    options = CUDARadixLifecycleOptions(
        ; precision=TF, m2l_strategy=ConcatenatedFixedZM2L(),
    )
    state = host_radix_state(bodies, grid, list, P, lh; options)
    ws = state.scratch

    FM._launch_host_b2m!(state)
    FM._launch_resident_m2m!(state)
    FM._launch_resident_m2l!(state)
    FM._launch_resident_l2l!(state)

    m2m_bytes = @allocated FM._launch_resident_m2m!(state)
    m2l_bytes = @allocated FM._launch_resident_m2l!(state)
    l2l_bytes = @allocated FM._launch_resident_l2l!(state)

    plan = ws.m2l_concat
    nchunks = max(cld(plan.nroutes, plan.chunk), 1)
    nm2m = max(length(ws.m2m_groups), 1)
    nl2l = max(length(ws.l2l_groups), 1)
    rows = [
        ("allocation", "concat_m2l", string(TF), csv_bool(lh), P, n, ell,
            nchunks, m2l_bytes, m2l_bytes / nchunks, "warmed host launch; unit=chunk"),
        ("allocation", "resident_m2m", string(TF), csv_bool(lh), P, n, ell,
            nm2m, m2m_bytes, m2m_bytes / nm2m, "warmed host launch; unit=group"),
        ("allocation", "resident_l2l", string(TF), csv_bool(lh), P, n, ell,
            nl2l, l2l_bytes, l2l_bytes / nl2l, "warmed host launch; unit=group"),
    ]

    return rows
end

function geometry_rows(::Type{TF}; n=2_000, ell=3, P=4) where TF
    bodies = body_matrix(TF, n; seed=19019)
    grid = RadixGrid(bodies, ell)
    list = build_radix_interaction_list(
        LazyMaterializedBatches(32), ConstantPAnalyticStencil(P, TF(1e-4)), grid,
    )
    state = host_radix_state(bodies, grid, list, P; options=CUDARadixLifecycleOptions(
        ; precision=TF, m2l_strategy=ConcatenatedFixedZM2L(),
    ))
    plan = state.scratch.m2l_concat
    old_geometry = 4 * plan.nroutes * sizeof(TF)
    class_geometry = sizeof(plan.phis) + sizeof(plan.thetas) + sizeof(plan.rs) +
        sizeof(plan.invrs) + sizeof(plan.route_class)
    return [
        ("storage", "geometry_before_analytic", string(TF), "false", P, n, ell,
            plan.nroutes, old_geometry, old_geometry / max(plan.nroutes, 1),
            "constant-P; 4 per-route floating-point vectors; analytical payload"),
        ("storage", "geometry_current_measured", string(TF), "false", P, n, ell,
            plan.nroutes, class_geometry, class_geometry / max(plan.nroutes, 1),
            "constant-P; class r/theta/phi/invr plus Int32 route_class; measured payload"),
    ]
end

function cache_rows()
    rows = Tuple[]
    for P in (4, 8, 20)
        cache = OperatorInvariantCache(Float64, P, Val(false))
        swap_bytes = sizeof(cache.S_pos) + sizeof(cache.S_neg)
        ymode_bytes = sum(sizeof, (
            cache.y_mult_U, cache.y_mult_V, cache.y_loc_U, cache.y_loc_V,
        ))
        push!(rows, ("storage", "cache_S_pos_S_neg", "Float64", "false", P,
            0, 0, 1, swap_bytes, Float64(swap_bytes), "measured vector payload"))
        push!(rows, ("storage", "cache_factored_y_modes", "Float64", "false", P,
            0, 0, 1, ymode_bytes, Float64(ymode_bytes), "measured vector payload"))
    end
    return rows
end

println("category,metric,precision,lamb_helmholtz,P,n,ell,count,total_bytes,bytes_per_count,notes")
rows = Tuple[]
for TF in (Float64, Float32), lh in (Val(false), Val(true))
    append!(rows, allocation_rows(TF, lh))
end
for TF in (Float64, Float32)
    append!(rows, geometry_rows(TF))
end
append!(rows, cache_rows())
for row in rows
    category, metric, precision, lh, P, n, ell, count, total, per, notes = row
    println(join((category, metric, precision, lh, P, n, ell, count, total,
        string(round(per; digits=3)), notes), ','))
end
