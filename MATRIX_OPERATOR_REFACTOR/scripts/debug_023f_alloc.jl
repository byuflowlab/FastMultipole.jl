# Task 023f debug: locate the steady-state device allocation in the dense M2L
# launch (12 test failures at cuda_radix_integration_test.jl:869, sizes 9.6-48.9 KB
# scaling with precision and route count). Reproduces the exact failing configs
# and measures CUDA.@allocated stage by stage.

using FastMultipole
using FastMultipole.StaticArrays

include(joinpath(@__DIR__, "..", "..", "test", "gravitational.jl"))

FastMultipole.load_cuda_radix_lifecycle!() ||
    error("CUDA radix lifecycle failed to load: $(FastMultipole.cuda_radix_status())")
using CUDA
using LinearAlgebra

const seed = 20260714
const bounds = (SVector(-0.1, -0.1, -0.1), 1.2)

function diagnose(P, TF, LH)
    println("\n=== P=$P TF=$TF LH=$LH")
    nf = 800
    opts = FastMultipole.CUDARadixLifecycleOptions(; precision=TF,
        operator=MaterializedYRotationM2L(),
        m2l_strategy=DenseTranslationM2L())
    sys = generate_gravitational(seed + 71, nf)
    cache = RadixFMMCache(sys; expansion_order=P, ell=3, bounds=bounds,
        lamb_helmholtz=LH, device=true, options=opts)
    fmm!(sys, cache; scalar_potential=!LH, gradient=true)
    state = cache.state
    ws = state.scratch
    plan = ws.m2l_concat
    n_routes = state.counts.n_routes
    println("n_routes=$n_routes D=$(plan.ndof) ndof_phi=$(plan.ndof_phi) " *
        "nclasses=$(plan.nclasses) W=$(plan.width) chunk=$(plan.whole_pass[].chunk)")

    # full launch, three repeats (is the alloc steady?)
    FastMultipole._launch_resident_m2l!(state); CUDA.synchronize()
    for i in 1:3
        a = CUDA.@allocated FastMultipole._launch_resident_m2l!(state)
        println("full _launch_resident_m2l!  rep$i: $a bytes")
    end

    # dense entry minus generic dispatch
    a = CUDA.@allocated FastMultipole._launch_resident_m2l_dense!(state)
    println("_launch_resident_m2l_dense!: $a bytes")

    # stage-by-stage replication of _launch_resident_m2l_dense_whole!
    wp = plan.whole_pass[]
    W = wp.chunk
    starts = plan.class_starts
    ndof_phi = plan.ndof_phi
    lh = Val(LH)

    a_fill = CUDA.@allocated begin
        fill!(state.locals.phi, zero(TF))
        LH && fill!(state.locals.chi, zero(TF))
    end
    println("fill locals: $a_fill bytes")

    a_gather = 0; a_gemm = 0; a_scatter = 0; ngemm = 0
    kcur = 1
    for c0 in 1:W:n_routes
        n = min(W, n_routes - c0 + 1)
        chi_hi = c0 + n - 1
        a_gather += CUDA.@allocated FastMultipole._cuda_dense_gather!(
            FastMultipole._matrix_col_view(plan.src_slab, n), state.multipoles,
            ws.phi_flat_idx, ws.chi_flat_idx, view(state.route_sources, c0:chi_hi),
            ndof_phi, lh)
        while starts[kcur + 1] <= c0
            kcur += 1
        end
        k = kcur
        while k <= plan.nclasses && starts[k] <= chi_hi
            lo = max(starts[k], c0); hi = min(starts[k + 1] - 1, chi_hi)
            if hi >= lo
                a_gemm += CUDA.@allocated FastMultipole._cuda_dense_class_gemm!(
                    plan.dst_slab, plan.operators, k, plan.src_slab,
                    lo - c0 + 1, hi - c0 + 1, wp.alpha, wp.beta)
                ngemm += 1
            end
            k += 1
        end
        a_scatter += CUDA.@allocated FastMultipole._cuda_dense_scatter_add!(
            state.locals, FastMultipole._matrix_col_view(plan.dst_slab, n),
            ws.phi_flat_idx, ws.chi_flat_idx, view(state.route_targets, c0:chi_hi),
            ndof_phi, lh)
    end
    println("gather: $a_gather  gemm ($ngemm calls): $a_gemm  scatter: $a_scatter")

    # single gemm shapes: what does one mul! allocate, and per class?
    if a_gemm > 0
        per = Int[]
        kcur = 1
        for k in 1:plan.nclasses
            lo = starts[k]; hi = starts[k + 1] - 1
            hi >= lo || continue
            hi = min(hi, n_routes); lo > n_routes && continue
            b = CUDA.@allocated FastMultipole._cuda_dense_class_gemm!(
                plan.dst_slab, plan.operators, k, plan.src_slab, lo, hi,
                wp.alpha, wp.beta)
            push!(per, b)
        end
        nz = count(!=(0), per)
        println("per-class gemm allocs: nonzero=$nz total=$(sum(per)) " *
            "first-few=$(per[1:min(end, 8)])")
        # contiguous full-width control: plain CuMatrix operands
        A = plan.operators[findfirst(k -> starts[k + 1] > starts[k], 1:plan.nclasses)]
        c = CUDA.@allocated mul!(plan.dst_slab, A, plan.src_slab)
        println("control mul!(full slab, CuMatrix, full slab): $c bytes")
        v = CUDA.@allocated mul!((@view plan.dst_slab[:, 1:5]), A,
            (@view plan.src_slab[:, 1:5]))
        println("control mul!(view 1:5): $v bytes")
    end

    # per-class reference driver
    saved = FastMultipole.DENSE_CUDA_WHOLE_PASS[]
    try
        FastMultipole.DENSE_CUDA_WHOLE_PASS[] = false
        FastMultipole._launch_resident_m2l!(state); CUDA.synchronize()
        a = CUDA.@allocated FastMultipole._launch_resident_m2l!(state)
        println("per-class driver: $a bytes")
    finally
        FastMultipole.DENSE_CUDA_WHOLE_PASS[] = saved
    end
end

diagnose(4, Float64, false)
diagnose(4, Float64, true)
diagnose(4, Float32, false)
diagnose(8, Float64, false)
diagnose(12, Float64, false)
println("\nDONE")
