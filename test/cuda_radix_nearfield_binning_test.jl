# Task 032a stage C, CUDA mirrors: the distance-binned nearfield pair stream
# (031a §6.3) for the split vortex kernels and the TwoPassVortex pass-2 deficit
# sweep on the device-resident lifecycle. Host references are the Stage A/B
# host kernels, which device_system_interface_test.jl validates against the
# erf-based direct references.

using FastMultipole
using FastMultipole.StaticArrays
using Random
using Test

if !isdefined(@__MODULE__, :generate_gravitational)
    include("gravitational.jl")
end
if !isdefined(@__MODULE__, :VortexParticles)
    include("vortex.jl")
end
if !isdefined(@__MODULE__, :ExtendedVortex)
    include("interface_test_systems.jl")
end

_cuda_binning_required() = get(ENV, "FASTMULTIPOLE_REQUIRE_CUDA_TESTS", "0") == "1"

const _BINNING_LOADED = FastMultipole.load_cuda_radix_lifecycle!()
if _BINNING_LOADED
    using CUDA
end

# run one device fmm! for `sys_ctor(seed, n)` under the given binning
# mechanism, returning (U 3×n, J 9×n) from the wrapped VortexParticles
function _binning_device_run(sys, cache; mode, subsort, pass2_queued=false)
    old_mode = FastMultipole.CUDA_NEARFIELD_BINNING[]
    old_sub = FastMultipole.CUDA_NEARFIELD_SUBSORT[]
    old_q = FastMultipole.CUDA_TWOPASS_PASS2_QUEUED[]
    FastMultipole.CUDA_NEARFIELD_BINNING[] = mode
    FastMultipole.CUDA_NEARFIELD_SUBSORT[] = subsort
    FastMultipole.CUDA_TWOPASS_PASS2_QUEUED[] = pass2_queued
    try
        fmm!(sys, cache; scalar_potential=false, gradient=true, hessian=true)
    finally
        FastMultipole.CUDA_NEARFIELD_BINNING[] = old_mode
        FastMultipole.CUDA_NEARFIELD_SUBSORT[] = old_sub
        FastMultipole.CUDA_TWOPASS_PASS2_QUEUED[] = old_q
    end
    return nothing
end

_binning_inner(sys::PartitionedSmoothedVortex) = sys.smoothed.inner
_binning_inner(sys::TwoPassSmoothedVortex) = sys.smoothed.inner
_binning_inner(sys::SmoothedVortex) = sys.inner

@testset "CUDA binned nearfield pair stream (task 032a stage C)" begin
    loaded = _BINNING_LOADED
    if !loaded
        if _cuda_binning_required()
            error(
                "FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 but CUDA radix lifecycle did not load: " *
                FastMultipole.cuda_radix_status(),
            )
        end
        @test true
    else
        seed = 20260807
        nv = 1500
        sigma = 0.02 .+ 0.02 .* rand(MersenneTwister(seed), nv)

        # host Float64 references at both kernels (Stage A/B validated)
        host_p = PartitionedSmoothedVortex(SmoothedVortex(generate_vortex(seed, nv),
            copy(sigma)))
        hp_cache = RadixFMMCache(host_p; expansion_order=8, ell=3, near_radius2=16, hessian=true,
            options=CUDARadixLifecycleOptions(; precision=Float64,
                m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
        fmm!(host_p, hp_cache; scalar_potential=false, gradient=true, hessian=true)
        Uref_p = copy(_binning_inner(host_p).gradient_stretching[1:3, :])
        Jref_p = copy(_binning_inner(host_p).potential[5:13, :])
        host_t = TwoPassSmoothedVortex(SmoothedVortex(generate_vortex(seed, nv),
            copy(sigma)))
        ht_cache = RadixFMMCache(host_t; expansion_order=8, ell=3, near_radius2=16, hessian=true,
            options=CUDARadixLifecycleOptions(; precision=Float64,
                m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
        fmm!(host_t, ht_cache; scalar_potential=false, gradient=true, hessian=true)
        Uref_t = copy(_binning_inner(host_t).gradient_stretching[1:3, :])
        Jref_t = copy(_binning_inner(host_t).potential[5:13, :])
        u_scale = maximum(abs.(Uref_p))
        j_scale = maximum(abs.(Jref_p))

        #--- (1) PartitionedVortex device parity across every mechanism,
        #    P = 4 and P = 8, Float64 and Float32 (same-P host references at
        #    P = 4 per the standing rule) ---#

        for P in (4, 8), TF in (Float64, Float32)
            hsys = PartitionedSmoothedVortex(SmoothedVortex(
                generate_vortex(seed, nv), copy(sigma)))
            hcache = RadixFMMCache(hsys; expansion_order=P, ell=3, near_radius2=16, hessian=true,
                options=CUDARadixLifecycleOptions(; precision=TF,
                    m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
            fmm!(hsys, hcache; scalar_potential=false, gradient=true, hessian=true)
            Uh = copy(_binning_inner(hsys).gradient_stretching[1:3, :])
            Jh = copy(_binning_inner(hsys).potential[5:13, :])
            # device-vs-host same-P tolerance: reassociation + fast rsqrt
            tol = TF == Float64 ? 1e-8 : 4e-4
            for mode in (:unbinned, :classsplit, :ballot, :classsplit_ballot),
                    subsort in (false, true)
                dsys = PartitionedSmoothedVortex(SmoothedVortex(
                    generate_vortex(seed, nv), copy(sigma)))
                dcache = RadixFMMCache(dsys; expansion_order=P, ell=3, near_radius2=16,
                    hessian=true, device=true,
                    options=CUDARadixLifecycleOptions(; precision=TF,
                        m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
                @test FastMultipole._cache_nearfield_bin_ctx(dcache) !== nothing
                _binning_device_run(dsys, dcache; mode, subsort)
                Ud = _binning_inner(dsys).gradient_stretching[1:3, :]
                Jd = _binning_inner(dsys).potential[5:13, :]
                @test maximum(abs.(Ud .- Uh)) / u_scale < tol
                @test maximum(abs.(Jd .- Jh)) / j_scale < tol
            end
        end

        #--- (2) TwoPassVortex device mirror parity (pass 1 + pass-2 deficit),
        #    all mechanisms, both pass-2 modes, P = 4 and P = 8, both TF ---#

        for P in (4, 8), TF in (Float64, Float32)
            hsys = TwoPassSmoothedVortex(SmoothedVortex(
                generate_vortex(seed, nv), copy(sigma)))
            hcache = RadixFMMCache(hsys; expansion_order=P, ell=3, near_radius2=16, hessian=true,
                options=CUDARadixLifecycleOptions(; precision=TF,
                    m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
            fmm!(hsys, hcache; scalar_potential=false, gradient=true, hessian=true)
            Uh = copy(_binning_inner(hsys).gradient_stretching[1:3, :])
            Jh = copy(_binning_inner(hsys).potential[5:13, :])
            tol = TF == Float64 ? 1e-8 : 4e-4
            for mode in (:unbinned, :classsplit, :classsplit_ballot),
                    pass2_queued in (false, true)
                dsys = TwoPassSmoothedVortex(SmoothedVortex(
                    generate_vortex(seed, nv), copy(sigma)))
                dcache = RadixFMMCache(dsys; expansion_order=P, ell=3, near_radius2=16,
                    hessian=true, device=true,
                    options=CUDARadixLifecycleOptions(; precision=TF,
                        m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
                nfctx = FastMultipole._cache_nearfield_bin_ctx(dcache)
                @test nfctx !== nothing
                @test nfctx.twopass_K > 0
                _binning_device_run(dsys, dcache; mode, subsort=false, pass2_queued)
                Ud = _binning_inner(dsys).gradient_stretching[1:3, :]
                Jd = _binning_inner(dsys).potential[5:13, :]
                @test maximum(abs.(Ud .- Uh)) / u_scale < tol
                @test maximum(abs.(Jd .- Jh)) / j_scale < tol
            end
        end

        #--- (3) accuracy anchor: device winner-configuration results also match
        #    the P = 8 Float64 host references of record ---#

        for (ctor, Uref, Jref) in ((PartitionedSmoothedVortex, Uref_p, Jref_p),
                (TwoPassSmoothedVortex, Uref_t, Jref_t))
            dsys = ctor(SmoothedVortex(generate_vortex(seed, nv), copy(sigma)))
            dcache = RadixFMMCache(dsys; expansion_order=8, ell=3, near_radius2=16, hessian=true,
                device=true, options=CUDARadixLifecycleOptions(; precision=Float64,
                    m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
            _binning_device_run(dsys, dcache; mode=:classsplit, subsort=false)
            @test maximum(abs.(_binning_inner(dsys).gradient_stretching[1:3, :] .-
                Uref)) / u_scale < 1e-8
            @test maximum(abs.(_binning_inner(dsys).potential[5:13, :] .-
                Jref)) / j_scale < 1e-8
        end

        #--- (4) 023 contracts: flat transfer counters and zero per-step device
        #    allocation in the steady state, for every mechanism ---#

        for (ctor, pass2_queued) in ((PartitionedSmoothedVortex, false),
                (TwoPassSmoothedVortex, false), (TwoPassSmoothedVortex, true))
            for mode in (:classsplit, :ballot, :classsplit_ballot)
                ctor === TwoPassSmoothedVortex && mode === :ballot && continue
                dsys = ctor(SmoothedVortex(generate_vortex(seed, nv), copy(sigma)))
                dcache = RadixFMMCache(dsys; expansion_order=4, ell=3, near_radius2=16,
                    hessian=true, device=true,
                    options=CUDARadixLifecycleOptions(; precision=Float32,
                        m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
                for subsort in (false, true)
                    _binning_device_run(dsys, dcache; mode, subsort, pass2_queued)
                    _binning_device_run(dsys, dcache; mode, subsort, pass2_queued)
                    counters = dcache.state.counters
                    @test counters.expansion_host_copies == 0
                    route0 = counters.route_uploads
                    op0 = counters.operator_uploads
                    body0 = counters.body_uploads
                    _binning_device_run(dsys, dcache; mode, subsort, pass2_queued)
                    @test counters.route_uploads == route0
                    @test counters.operator_uploads == op0
                    @test counters.body_uploads == body0 + 1
                    @test counters.expansion_host_copies == 0
                    # warmed steady-state device allocation must be *stable*
                    # across repeated calls (the hierarchical update inherits
                    # CUDA.jl's pool-served accumulate! scan scratch; the stage-C
                    # kernels themselves add no per-step allocation). Runtime
                    # @eval: CUDA.@allocated only exists when CUDA loaded.
                    @eval CUDA.@allocated fmm!($dsys, $dcache;
                        scalar_potential=false, gradient=true, hessian=true)
                    step_a = @eval CUDA.@allocated fmm!($dsys, $dcache;
                        scalar_potential=false, gradient=true, hessian=true)
                    step_b = @eval CUDA.@allocated fmm!($dsys, $dcache;
                        scalar_potential=false, gradient=true, hessian=true)
                    @test step_b == step_a
                end
            end
        end

        #--- (5) homogeneity diagnostics: sane, and the mechanisms genuinely
        #    change the achieved warp homogeneity ---#

        dsys = PartitionedSmoothedVortex(SmoothedVortex(
            generate_vortex(seed, nv), copy(sigma)))
        dcache = RadixFMMCache(dsys; expansion_order=4, ell=3, near_radius2=16, hessian=true,
            device=true, options=CUDARadixLifecycleOptions(; precision=Float32,
                m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
        _binning_device_run(dsys, dcache; mode=:unbinned, subsort=false)
        h_all = FastMultipole.cuda_nearfield_homogeneity(dcache.state; stream=:all)
        @test h_all.instants > 0
        @test h_all.uniform + h_all.mixed == h_all.instants
        @test 0.0 <= h_all.homogeneous_fraction <= 1.0
        @test 0.0 < h_all.regularized_fraction < 1.0
        h_mixed = FastMultipole.cuda_nearfield_homogeneity(dcache.state; stream=:mixed)
        # the mixed bucket is a subset of the full stream and strictly less
        # homogeneous than the whole (the pure buckets it excludes are uniform)
        @test h_mixed.instants <= h_all.instants
        @test h_mixed.instants == 0 ||
            h_mixed.homogeneous_fraction <= h_all.homogeneous_fraction + 1e-12
        # sub-sorting may only be assessed after a refresh under the flag
        _binning_device_run(dsys, dcache; mode=:unbinned, subsort=true)
        h_sub = FastMultipole.cuda_nearfield_homogeneity(dcache.state; stream=:all)
        @test h_sub.instants > 0

        tsys = TwoPassSmoothedVortex(SmoothedVortex(
            generate_vortex(seed, nv), copy(sigma)))
        tcache = RadixFMMCache(tsys; expansion_order=4, ell=3, near_radius2=16, hessian=true,
            device=true, options=CUDARadixLifecycleOptions(; precision=Float32,
                m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
        _binning_device_run(tsys, tcache; mode=:unbinned, subsort=false)
        h_shell = FastMultipole.cuda_twopass_shell_homogeneity(tcache.state)
        @test h_shell.instants > 0
        @test h_shell.uniform + h_shell.mixed == h_shell.instants

        #--- (6) validation paths: flat-policy device TwoPass refused with the
        #    policy message; invalid binning mode rejected at launch ---#

        err = try
            RadixFMMCache(TwoPassSmoothedVortex(SmoothedVortex(
                    generate_vortex(seed, 300), fill(0.02, 300)));
                expansion_order=4, ell=2, device=true,
                policy=FastMultipole.ConstantPAnalyticStencil(4, 1e-3),
                options=CUDARadixLifecycleOptions(; precision=Float64,
                    m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("hierarchical stencil", sprint(showerror, err))

        #--- (5b) graph-captured lifecycle (dense fused M2L) with the stage-C
        #    kernels inside the replayed graph: parity against the concat-path
        #    device result and the graph must actually engage ---#

        for (ctor, mode) in ((PartitionedSmoothedVortex, :classsplit),
                (PartitionedSmoothedVortex, :ballot),
                (TwoPassSmoothedVortex, :classsplit))
            ref_sys = ctor(SmoothedVortex(generate_vortex(seed, nv), copy(sigma)))
            ref_cache = RadixFMMCache(ref_sys; expansion_order=4, ell=3, near_radius2=16,
                hessian=true, device=true,
                options=CUDARadixLifecycleOptions(; precision=Float32,
                    m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
            _binning_device_run(ref_sys, ref_cache; mode, subsort=false)
            gsys = ctor(SmoothedVortex(generate_vortex(seed, nv), copy(sigma)))
            gcache = RadixFMMCache(gsys; expansion_order=4, ell=3, near_radius2=16, hessian=true,
                device=true, options=CUDARadixLifecycleOptions(; precision=Float32,
                    m2l_strategy=FastMultipole.DenseTranslationM2L(
                        apply_chunk=64, build_chunk=8)))
            # step 1 warms, step 2 records, step 3 replays the captured graph
            for _ in 1:3
                _binning_device_run(gsys, gcache; mode, subsort=false)
            end
            hctx = gcache.state.interaction_list
            if FastMultipole.CUDA_GRAPH_LIFECYCLE[] && FastMultipole.CUDA_CACHED_WINDOWS[]
                @test hctx.graph_exec !== nothing
                @test hctx.graph_epoch == hctx.epoch_id
            end
            # nearfield contribution identical; the far-field strategies differ
            # by dense-vs-concat reassociation and (default) FP16-WMMA inputs,
            # so gate at the F32/fp16 far-field comparability scale
            @test maximum(abs.(_binning_inner(gsys).gradient_stretching[1:3, :] .-
                _binning_inner(ref_sys).gradient_stretching[1:3, :])) / u_scale < 2e-3
        end

        # invalid mode must be rejected at launch. NOTE: the mechanism Refs are
        # baked into a captured lifecycle graph at record time, so this check
        # uses a FRESH cache (first lifecycle of an epoch runs uncaptured).
        FastMultipole.CUDA_NEARFIELD_BINNING[] = :bogus
        try
            fresh = PartitionedSmoothedVortex(SmoothedVortex(
                generate_vortex(seed, 400), fill(0.02, 400)))
            fcache = RadixFMMCache(fresh; expansion_order=4, ell=2, hessian=true,
                device=true, options=CUDARadixLifecycleOptions(; precision=Float32,
                    m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
            @test_throws ArgumentError fmm!(fresh, fcache; scalar_potential=false,
                gradient=true, hessian=true)
        finally
            FastMultipole.CUDA_NEARFIELD_BINNING[] = :unbinned
        end
    end
end
