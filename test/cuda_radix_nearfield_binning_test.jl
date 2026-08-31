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

# run `f()` with the binning-mechanism Refs set as requested, restoring the
# previous values afterwards. Construction-locked settings (BINNING, GH_MODE,
# PASS2_QUEUED, TARGET_AABB_PRUNE, PAIR_AABB; 047 lock contract) must hold
# their run values while a cache is BUILT, so cache construction goes through
# this window too. `subsort=nothing` leaves the (runtime) subsort Ref alone.
function _with_binning_refs(f; mode, subsort=nothing, pass2_queued=false,
        pass2_aabb=false, pair_aabb=false, gh_mode=:shipped)
    old_mode = FastMultipole.CUDA_NEARFIELD_BINNING[]
    old_sub = FastMultipole.CUDA_NEARFIELD_SUBSORT[]
    old_q = FastMultipole.CUDA_TWOPASS_PASS2_QUEUED[]
    old_aabb = FastMultipole.CUDA_TWOPASS_TARGET_AABB_PRUNE[]
    old_paabb = FastMultipole.CUDA_NEARFIELD_PAIR_AABB[]
    old_gh = FastMultipole.CUDA_NEARFIELD_GH_MODE[]
    FastMultipole.CUDA_NEARFIELD_BINNING[] = mode
    subsort === nothing || (FastMultipole.CUDA_NEARFIELD_SUBSORT[] = subsort)
    FastMultipole.CUDA_TWOPASS_PASS2_QUEUED[] = pass2_queued
    FastMultipole.CUDA_TWOPASS_TARGET_AABB_PRUNE[] = pass2_aabb
    FastMultipole.CUDA_NEARFIELD_PAIR_AABB[] = pair_aabb
    FastMultipole.CUDA_NEARFIELD_GH_MODE[] = gh_mode
    try
        return f()
    finally
        FastMultipole.CUDA_NEARFIELD_BINNING[] = old_mode
        FastMultipole.CUDA_NEARFIELD_SUBSORT[] = old_sub
        FastMultipole.CUDA_TWOPASS_PASS2_QUEUED[] = old_q
        FastMultipole.CUDA_TWOPASS_TARGET_AABB_PRUNE[] = old_aabb
        FastMultipole.CUDA_NEARFIELD_PAIR_AABB[] = old_paabb
        FastMultipole.CUDA_NEARFIELD_GH_MODE[] = old_gh
    end
end

# run one device fmm! under the given binning mechanism. `cache` may be a
# 0-arg builder closure, in which case the cache is CONSTRUCTED inside the
# settings window (required for any run whose construction-locked settings
# differ from the shipped defaults). Returns the (built) cache.
function _binning_device_run(sys, cache; mode, subsort, pass2_queued=false,
        pass2_aabb=false, pair_aabb=false, gh_mode=:shipped)
    _with_binning_refs(; mode, subsort, pass2_queued, pass2_aabb, pair_aabb,
            gh_mode) do
        c = cache isa Function ? cache() : cache
        fmm!(sys, c; scalar_potential=false, gradient=true, hessian=true)
        c
    end
end

#--- 037e host unit tests: pair-AABB reachability predicate (no CUDA needed;
#    the device kernels share this exact scalar function) ---#

@testset "037e pair-AABB predicate (host)" begin
    rng = MersenneTwister(20260814)
    for T in (Float64, Float32)
        h = T(0.125)
        rho = T(3.668)
        # exact point-to-AABB gap agreement with an independent clamp form
        for _ in 1:300
            slo = SVector{3,T}(rand(rng, T, 3) .- T(0.5))
            p = SVector{3,T}(T(4) .* rand(rng, T, 3) .- T(2))
            smax = T(0.05) * rand(rng, T)
            qv = clamp.(p, slo, slo .+ h)
            d2 = (p[1] - qv[1])^2 + (p[2] - qv[2])^2 + (p[3] - qv[3])^2
            expect = smax > zero(T) && d2 <= (rho * smax)^2
            @test FastMultipole._nearfield_point_aabb_reach(p[1], p[2], p[3],
                slo[1], slo[2], slo[3], h, rho, smax) == expect
        end
        # nonpositive sigma is never reachable, even inside the box
        @test !FastMultipole._nearfield_point_aabb_reach(T(0.5) * h, T(0.5) * h,
            T(0.5) * h, zero(T), zero(T), zero(T), h, rho, zero(T))
        # consistency with the cell-level classification: a bucket-1 (pure
        # singular) pair means NO point of the target cell can reach the
        # source-cell AABB within rho*sigma_max — the E1 fast path may only
        # ever skip what the bucket predicate already proves singular
        for _ in 1:300
            o = SVector{3,Int}(rand(rng, -3:3), rand(rng, -3:3), rand(rng, -3:3))
            smax_s = T(0.1) * rand(rng, T)
            smin_s = smax_s * rand(rng, T)
            b = FastMultipole._nearfield_pair_bucket(o[1], o[2], o[3], h, rho,
                smax_s, smin_s)
            b == Int32(1) || continue
            for _ in 1:20
                p = SVector{3,T}((T(o[1]) + rand(rng, T)) * h,
                    (T(o[2]) + rand(rng, T)) * h,
                    (T(o[3]) + rand(rng, T)) * h)
                @test !FastMultipole._nearfield_point_aabb_reach(p[1], p[2],
                    p[3], zero(T), zero(T), zero(T), h, rho, smax_s)
            end
        end
    end
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
        # shipped stage-C defaults (Checkpoint C/D approvals): asserted before
        # any test mutates the Refs
        @test FastMultipole.CUDA_NEARFIELD_BINNING[] === :classsplit
        @test FastMultipole.CUDA_NEARFIELD_SUBSORT[]
        @test !FastMultipole.CUDA_TWOPASS_PASS2_QUEUED[]
        # 037e: the mixed-bucket pair-AABB fast path ships OFF (control =
        # shipped classsplit stream until the H200 measurement)
        @test !FastMultipole.CUDA_NEARFIELD_PAIR_AABB[]

        seed = 20260807
        nv = 1500
        sigma = 0.02 .+ 0.02 .* rand(MersenneTwister(seed), nv)

        # host Float64 references at both kernels (Stage A/B validated).
        # The HOST nearfield also consults CUDA_NEARFIELD_GH_MODE, whose
        # ambient default flipped to :fp32 (037f, user-approved 2026-08-14);
        # references must be produced at the :shipped control mode to match
        # the pinned device runs, else U/J shift ~1.8e-7/3e-7 relative and
        # the 1e-8 parity contract breaks (cf. 1b3c65a7 for the interface
        # parity test).
        Uref_p, Jref_p, Uref_t, Jref_t = _with_binning_refs(;
                mode=:classsplit, gh_mode=:shipped) do
            host_p = PartitionedSmoothedVortex(SmoothedVortex(generate_vortex(seed, nv),
                copy(sigma)))
            hp_cache = RadixFMMCache(host_p; expansion_order=8, ell=3, near_radius2=16, hessian=true,
                options=CUDARadixLifecycleOptions(; precision=Float64,
                    m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
            fmm!(host_p, hp_cache; scalar_potential=false, gradient=true, hessian=true)
            host_t = TwoPassSmoothedVortex(SmoothedVortex(generate_vortex(seed, nv),
                copy(sigma)))
            ht_cache = RadixFMMCache(host_t; expansion_order=8, ell=3, near_radius2=16, hessian=true,
                options=CUDARadixLifecycleOptions(; precision=Float64,
                    m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
            fmm!(host_t, ht_cache; scalar_potential=false, gradient=true, hessian=true)
            (copy(_binning_inner(host_p).gradient_stretching[1:3, :]),
             copy(_binning_inner(host_p).potential[5:13, :]),
             copy(_binning_inner(host_t).gradient_stretching[1:3, :]),
             copy(_binning_inner(host_t).potential[5:13, :]))
        end
        u_scale = maximum(abs.(Uref_p))
        j_scale = maximum(abs.(Jref_p))

        #--- (1) PartitionedVortex device parity across every mechanism,
        #    P = 4 and P = 8, Float64 and Float32 (same-P host references at
        #    P = 4 per the standing rule) ---#

        for P in (4, 8), TF in (Float64, Float32)
            # host reference at the :shipped g/h control mode (see the
            # testset-preamble note; ambient default is :fp32)
            Uh, Jh = _with_binning_refs(; mode=:classsplit, gh_mode=:shipped) do
                hsys = PartitionedSmoothedVortex(SmoothedVortex(
                    generate_vortex(seed, nv), copy(sigma)))
                hcache = RadixFMMCache(hsys; expansion_order=P, ell=3, near_radius2=16, hessian=true,
                    options=CUDARadixLifecycleOptions(; precision=TF,
                        m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
                fmm!(hsys, hcache; scalar_potential=false, gradient=true, hessian=true)
                (copy(_binning_inner(hsys).gradient_stretching[1:3, :]),
                 copy(_binning_inner(hsys).potential[5:13, :]))
            end
            # device-vs-host same-P tolerance: reassociation + fast rsqrt
            tol = TF == Float64 ? 1e-8 : 4e-4
            for mode in (:unbinned, :classsplit, :ballot, :classsplit_ballot),
                    subsort in (false, true)
                dsys = PartitionedSmoothedVortex(SmoothedVortex(
                    generate_vortex(seed, nv), copy(sigma)))
                dcache = _binning_device_run(dsys,
                    () -> RadixFMMCache(dsys; expansion_order=P, ell=3, near_radius2=16,
                        hessian=true, device=true,
                        options=CUDARadixLifecycleOptions(; precision=TF,
                            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()));
                    mode, subsort)
                @test FastMultipole._cache_nearfield_bin_ctx(dcache) !== nothing
                Ud = _binning_inner(dsys).gradient_stretching[1:3, :]
                Jd = _binning_inner(dsys).potential[5:13, :]
                @test maximum(abs.(Ud .- Uh)) / u_scale < tol
                @test maximum(abs.(Jd .- Jh)) / j_scale < tol
            end
        end

        #--- (2) TwoPassVortex device mirror parity (pass 1 + pass-2 deficit),
        #    all mechanisms, both pass-2 modes, P = 4 and P = 8, both TF ---#

        for P in (4, 8), TF in (Float64, Float32)
            # host reference at the :shipped g/h control mode (see the
            # testset-preamble note; ambient default is :fp32)
            Uh, Jh = _with_binning_refs(; mode=:classsplit, gh_mode=:shipped) do
                hsys = TwoPassSmoothedVortex(SmoothedVortex(
                    generate_vortex(seed, nv), copy(sigma)))
                hcache = RadixFMMCache(hsys; expansion_order=P, ell=3, near_radius2=16, hessian=true,
                    options=CUDARadixLifecycleOptions(; precision=TF,
                        m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
                fmm!(hsys, hcache; scalar_potential=false, gradient=true, hessian=true)
                (copy(_binning_inner(hsys).gradient_stretching[1:3, :]),
                 copy(_binning_inner(hsys).potential[5:13, :]))
            end
            tol = TF == Float64 ? 1e-8 : 4e-4
            for mode in (:unbinned, :classsplit, :classsplit_ballot),
                    pass2_queued in (false, true)
                dsys = TwoPassSmoothedVortex(SmoothedVortex(
                    generate_vortex(seed, nv), copy(sigma)))
                dcache = _binning_device_run(dsys,
                    () -> RadixFMMCache(dsys; expansion_order=P, ell=3, near_radius2=16,
                        hessian=true, device=true,
                        options=CUDARadixLifecycleOptions(; precision=TF,
                            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()));
                    mode, subsort=false, pass2_queued)
                nfctx = FastMultipole._cache_nearfield_bin_ctx(dcache)
                @test nfctx !== nothing
                @test nfctx.twopass_K > 0
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
            _binning_device_run(dsys,
                () -> RadixFMMCache(dsys; expansion_order=8, ell=3, near_radius2=16, hessian=true,
                    device=true, options=CUDARadixLifecycleOptions(; precision=Float64,
                        m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()));
                mode=:classsplit, subsort=false)
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
                dcache = _with_binning_refs(
                    () -> RadixFMMCache(dsys; expansion_order=4, ell=3, near_radius2=16,
                        hessian=true, device=true,
                        options=CUDARadixLifecycleOptions(; precision=Float32,
                            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()));
                    mode, pass2_queued)
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
                    _with_binning_refs(; mode, subsort, pass2_queued) do
                        @eval CUDA.@allocated fmm!($dsys, $dcache;
                            scalar_potential=false, gradient=true, hessian=true)
                    end
                    step_a = _with_binning_refs(; mode, subsort, pass2_queued) do
                        @eval CUDA.@allocated fmm!($dsys, $dcache;
                            scalar_potential=false, gradient=true, hessian=true)
                    end
                    step_b = _with_binning_refs(; mode, subsort, pass2_queued) do
                        @eval CUDA.@allocated fmm!($dsys, $dcache;
                            scalar_potential=false, gradient=true, hessian=true)
                    end
                    @test step_b == step_a
                end
            end
        end

        #--- (5) homogeneity diagnostics: sane, and the mechanisms genuinely
        #    change the achieved warp homogeneity ---#

        dsys = PartitionedSmoothedVortex(SmoothedVortex(
            generate_vortex(seed, nv), copy(sigma)))
        dcache = _binning_device_run(dsys,
            () -> RadixFMMCache(dsys; expansion_order=4, ell=3, near_radius2=16, hessian=true,
                device=true, options=CUDARadixLifecycleOptions(; precision=Float32,
                    m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()));
            mode=:unbinned, subsort=false)
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
        tcache = _binning_device_run(tsys,
            () -> RadixFMMCache(tsys; expansion_order=4, ell=3, near_radius2=16, hessian=true,
                device=true, options=CUDARadixLifecycleOptions(; precision=Float32,
                    m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()));
            mode=:unbinned, subsort=false)
        h_shell = FastMultipole.cuda_twopass_shell_homogeneity(tcache.state)
        @test h_shell.instants > 0
        @test h_shell.uniform + h_shell.mixed == h_shell.instants
        @test h_shell.candidate_pairs >= h_shell.shell_pairs > 0
        # TARGET_AABB_PRUNE is construction-locked: the pruned run gets its
        # own cache built under the flag (same seed → identical tree, so the
        # shell-pair comparison below stays valid)
        tsys2 = TwoPassSmoothedVortex(SmoothedVortex(
            generate_vortex(seed, nv), copy(sigma)))
        tcache2 = _binning_device_run(tsys2,
            () -> RadixFMMCache(tsys2; expansion_order=4, ell=3, near_radius2=16, hessian=true,
                device=true, options=CUDARadixLifecycleOptions(; precision=Float32,
                    m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()));
            mode=:unbinned, subsort=true, pass2_aabb=true)
        FastMultipole.CUDA_TWOPASS_TARGET_AABB_PRUNE[] = true
        h_pruned = try
            FastMultipole.cuda_twopass_shell_homogeneity(tcache2.state)
        finally
            FastMultipole.CUDA_TWOPASS_TARGET_AABB_PRUNE[] = false
        end
        @test h_pruned.shell_pairs == h_shell.shell_pairs
        @test h_pruned.candidate_pairs <= h_shell.candidate_pairs

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
            ref_cache = _binning_device_run(ref_sys,
                () -> RadixFMMCache(ref_sys; expansion_order=4, ell=3, near_radius2=16,
                    hessian=true, device=true,
                    options=CUDARadixLifecycleOptions(; precision=Float32,
                        m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()));
                mode, subsort=false)
            gsys = ctor(SmoothedVortex(generate_vortex(seed, nv), copy(sigma)))
            gcache = _with_binning_refs(
                () -> RadixFMMCache(gsys; expansion_order=4, ell=3, near_radius2=16, hessian=true,
                    device=true, options=CUDARadixLifecycleOptions(; precision=Float32,
                        m2l_strategy=FastMultipole.DenseTranslationM2L(
                            apply_chunk=64, build_chunk=8)));
                mode)
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

        #--- (7) 037e mixed-bucket pair-AABB fast path: flag-on output must be
        #    the flag-off result — bitwise when the configuration is atomically
        #    deterministic (probed by twin flag-off runs; the pair stream is
        #    compacted by atomic claim, so cross-run atomic accumulation order
        #    is not guaranteed), else at accumulation-order tolerance — plus
        #    telemetry monotonicity on both traversal mechanisms ---#

        _mk_psys() = PartitionedSmoothedVortex(SmoothedVortex(
            generate_vortex(seed, nv), copy(sigma)))
        function _mk_pcache(sys, P, TF; hess=true,
                strategy=FastMultipole.ConcatenatedFixedZM2L(), pair_aabb=false,
                mode=:classsplit, gh_mode=:shipped, pass2_queued=false)
            # every construction-locked Ref is read in the lifecycle body
            # (graph-bake / 047 lock contract): the cache must be CONSTRUCTED
            # under the exact settings its runs will use
            _with_binning_refs(; mode, pair_aabb, gh_mode, pass2_queued) do
                RadixFMMCache(sys; expansion_order=P, ell=3,
                    near_radius2=16, hessian=hess, device=true,
                    options=CUDARadixLifecycleOptions(; precision=TF,
                        m2l_strategy=strategy))
            end
        end
        for P in (4, 8), TF in (Float64, Float32),
                mode in (:classsplit, :classsplit_ballot)
            UJs = map(1:2) do _
                s = _mk_psys()
                c = _mk_pcache(s, P, TF; mode)
                _binning_device_run(s, c; mode, subsort=false)
                (copy(_binning_inner(s).gradient_stretching[1:3, :]),
                 copy(_binning_inner(s).potential[5:13, :]))
            end
            deterministic = UJs[1] == UJs[2]
            s_on = _mk_psys()
            c_on = _mk_pcache(s_on, P, TF; mode, pair_aabb=true)
            _binning_device_run(s_on, c_on; mode, subsort=false, pair_aabb=true)
            Uon = _binning_inner(s_on).gradient_stretching[1:3, :]
            Jon = _binning_inner(s_on).potential[5:13, :]
            if deterministic
                @test Uon == UJs[1][1]
                @test Jon == UJs[1][2]
            else
                tol = TF == Float64 ? 1e-13 : 2e-6
                @test maximum(abs.(Uon .- UJs[1][1])) / u_scale < tol
                @test maximum(abs.(Jon .- UJs[1][2])) / j_scale < tol
            end
            # telemetry replay: every mixed pair tested exactly once, skips
            # are a subset (skipped <= tested == mixed bucket count)
            st = FastMultipole.cuda_nearfield_pair_aabb_stats(c_on.state)
            @test 0 <= st.skipped <= st.tested
            @test st.tested == st.mixed_pairs
            @test 0.0 <= st.skipped_fraction <= 1.0
        end

        #--- (7b) gradient-only (HS=false) branch of the fast path ---#

        s_off = _mk_psys()
        c_off = _mk_pcache(s_off, 4, Float32; hess=false)
        _with_binning_refs(; mode=:classsplit) do
            FastMultipole.fmm!(s_off, c_off; scalar_potential=false,
                gradient=true, hessian=false)
        end
        s_on = _mk_psys()
        c_on = _mk_pcache(s_on, 4, Float32; hess=false, pair_aabb=true)
        _with_binning_refs(; mode=:classsplit, pair_aabb=true) do
            FastMultipole.fmm!(s_on, c_on; scalar_potential=false,
                gradient=true, hessian=false)
        end
        @test maximum(abs.(_binning_inner(s_on).gradient_stretching[1:3, :] .-
            _binning_inner(s_off).gradient_stretching[1:3, :])) / u_scale < 2e-6

        #--- (7c) 023 contracts + graph replay with the flag ON: fresh cache
        #    constructed under the flag, dense fused M2L so the captured graph
        #    carries the AABB traversal; counters flat, steady-state device
        #    allocation stable, parity against the flag-off dense result ---#

        ds_ref = _mk_psys()
        dc_ref = _mk_pcache(ds_ref, 4, Float32;
            strategy=FastMultipole.DenseTranslationM2L(apply_chunk=64,
                build_chunk=8))
        for _ in 1:3
            _binning_device_run(ds_ref, dc_ref; mode=:classsplit, subsort=false)
        end
        Uref_dense = copy(_binning_inner(ds_ref).gradient_stretching[1:3, :])

        ds = _mk_psys()
        dc = _mk_pcache(ds, 4, Float32; pair_aabb=true,
            strategy=FastMultipole.DenseTranslationM2L(apply_chunk=64,
                build_chunk=8))
        FastMultipole.CUDA_NEARFIELD_PAIR_AABB[] = true
        try
            # step 1 warms, step 2 records, step 3 replays the captured graph
            for _ in 1:3
                _binning_device_run(ds, dc; mode=:classsplit, subsort=false,
                    pair_aabb=true)
            end
            hctx7 = dc.state.interaction_list
            if FastMultipole.CUDA_GRAPH_LIFECYCLE[] && FastMultipole.CUDA_CACHED_WINDOWS[]
                @test hctx7.graph_exec !== nothing
                @test hctx7.graph_epoch == hctx7.epoch_id
            end
            counters7 = dc.state.counters
            @test counters7.expansion_host_copies == 0
            route0 = counters7.route_uploads
            op0 = counters7.operator_uploads
            body0 = counters7.body_uploads
            _binning_device_run(ds, dc; mode=:classsplit, subsort=false,
                pair_aabb=true)
            @test counters7.route_uploads == route0
            @test counters7.operator_uploads == op0
            @test counters7.body_uploads == body0 + 1
            @test counters7.expansion_host_copies == 0
            _with_binning_refs(; mode=:classsplit, pair_aabb=true) do
                @eval CUDA.@allocated fmm!($ds, $dc;
                    scalar_potential=false, gradient=true, hessian=true)
            end
            step_a = _with_binning_refs(; mode=:classsplit, pair_aabb=true) do
                @eval CUDA.@allocated fmm!($ds, $dc;
                    scalar_potential=false, gradient=true, hessian=true)
            end
            step_b = _with_binning_refs(; mode=:classsplit, pair_aabb=true) do
                @eval CUDA.@allocated fmm!($ds, $dc;
                    scalar_potential=false, gradient=true, hessian=true)
            end
            @test step_b == step_a
        finally
            FastMultipole.CUDA_NEARFIELD_PAIR_AABB[] = false
        end
        @test maximum(abs.(_binning_inner(ds).gradient_stretching[1:3, :] .-
            Uref_dense)) / u_scale < 1e-5

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
            FastMultipole.CUDA_NEARFIELD_BINNING[] = :classsplit
        end
    end
end

@testset "CUDA cheapened g/h modes (task 037f)" begin
    if !_BINNING_LOADED
        if _cuda_binning_required()
            error(
                "FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 but CUDA radix lifecycle did not load: " *
                FastMultipole.cuda_radix_status(),
            )
        end
        @test true
    else
        # production default asserted before any test mutates the Ref
        # (:fp32 for Float64 configs, user-approved 2026-08-14; bitwise
        # no-op on Float32 configs; :shipped retained as control/opt-out)
        @test FastMultipole.CUDA_NEARFIELD_GH_MODE[] === :fp32

        seed = 20260814
        nv = 1500
        sigma = 0.02 .+ 0.02 .* rand(MersenneTwister(seed), nv)
        fresh_psys() = PartitionedSmoothedVortex(SmoothedVortex(
            generate_vortex(seed, nv), copy(sigma)))
        fresh_tsys() = TwoPassSmoothedVortex(SmoothedVortex(
            generate_vortex(seed, nv), copy(sigma)))
        fresh_cache(sys, P, TF) = RadixFMMCache(sys; expansion_order=P, ell=3,
            near_radius2=16, hessian=true, device=true,
            options=CUDARadixLifecycleOptions(; precision=TF,
                m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))

        #--- (1) per-mode device parity vs the device :shipped result at the
        #    budgeted per-mode tolerances, P = 4/8, F64/F32, both binning
        #    mechanisms (bucket + ballot kernels); the fp32 modes are skipped
        #    on Float32 configurations (documented no-ops) ---#

        for P in (4, 8), TF in (Float64, Float32)
            ref_sys = fresh_psys()
            _binning_device_run(ref_sys, () -> fresh_cache(ref_sys, P, TF);
                mode=:classsplit, subsort=false, gh_mode=:shipped)
            U0 = copy(_binning_inner(ref_sys).gradient_stretching[1:3, :])
            J0 = copy(_binning_inner(ref_sys).potential[5:13, :])
            u_scale = maximum(abs.(U0))
            j_scale = maximum(abs.(J0))
            # mode tolerances: pointwise deltas vs shipped (fm037f_budget.csv)
            # times a generous local-sum amplification, floored at the F32
            # run-to-run reassociation scale for F32 configurations
            base_tol = TF == Float64 ? 1e-8 : 4e-4
            gh_modes = TF == Float64 ?
                ((:reduced, 1e-4), (:fp32, 1e-5), (:reduced_fp32, 1e-4),
                    (:lut, 1e-4)) :
                ((:reduced, 5e-4), (:lut, 5e-4))
            for (gh_mode, mode_tol) in gh_modes, bmode in (:classsplit, :ballot)
                dsys = fresh_psys()
                dcache = _binning_device_run(dsys,
                    () -> fresh_cache(dsys, P, TF);
                    mode=bmode, subsort=false, gh_mode)
                nfctx = FastMultipole._cache_nearfield_bin_ctx(dcache)
                @test nfctx !== nothing
                @test size(nfctx.gh_lut) == (2, FastMultipole._NF_GH_LUT_N)
                Ud = _binning_inner(dsys).gradient_stretching[1:3, :]
                Jd = _binning_inner(dsys).potential[5:13, :]
                tol = max(mode_tol, base_tol)
                @test maximum(abs.(Ud .- U0)) / u_scale < tol
                @test maximum(abs.(Jd .- J0)) / j_scale < tol
            end
        end

        #--- (2) TwoPassVortex pass 1 inherits the mode; the pass-2 deficit
        #    stays shipped (scope decision, budget note SS5) ---#

        for TF in (Float64, Float32)
            ref_sys = fresh_tsys()
            _binning_device_run(ref_sys, () -> fresh_cache(ref_sys, 4, TF);
                mode=:classsplit, subsort=false, gh_mode=:shipped)
            U0 = copy(_binning_inner(ref_sys).gradient_stretching[1:3, :])
            u_scale = maximum(abs.(U0))
            dsys = fresh_tsys()
            _binning_device_run(dsys, () -> fresh_cache(dsys, 4, TF);
                mode=:classsplit, subsort=false, gh_mode=:reduced)
            Ud = _binning_inner(dsys).gradient_stretching[1:3, :]
            @test maximum(abs.(Ud .- U0)) / u_scale <
                max(1e-4, TF == Float64 ? 1e-8 : 4e-4)
        end

        #--- (3) 023 contracts under non-default modes: flat route/operator
        #    counters and stable steady-state allocation (:lut exercises the
        #    construction-only table upload) ---#

        for gh_mode in (:reduced, :lut)
            dsys = fresh_psys()
            dcache = _with_binning_refs(() -> fresh_cache(dsys, 4, Float32);
                mode=:classsplit, gh_mode)
            for _ in 1:2
                _binning_device_run(dsys, dcache; mode=:classsplit,
                    subsort=false, gh_mode)
            end
            counters = dcache.state.counters
            @test counters.expansion_host_copies == 0
            route0 = counters.route_uploads
            op0 = counters.operator_uploads
            body0 = counters.body_uploads
            _binning_device_run(dsys, dcache; mode=:classsplit, subsort=false,
                gh_mode)
            @test counters.route_uploads == route0
            @test counters.operator_uploads == op0
            @test counters.body_uploads == body0 + 1
            old_gh = FastMultipole.CUDA_NEARFIELD_GH_MODE[]
            FastMultipole.CUDA_NEARFIELD_GH_MODE[] = gh_mode
            try
                @eval CUDA.@allocated fmm!($dsys, $dcache;
                    scalar_potential=false, gradient=true, hessian=true)
                step_a = @eval CUDA.@allocated fmm!($dsys, $dcache;
                    scalar_potential=false, gradient=true, hessian=true)
                step_b = @eval CUDA.@allocated fmm!($dsys, $dcache;
                    scalar_potential=false, gradient=true, hessian=true)
                @test step_b == step_a
            finally
                FastMultipole.CUDA_NEARFIELD_GH_MODE[] = old_gh
            end
        end

        #--- (4) graph-captured lifecycle with a non-default mode inside the
        #    replayed graph (mirrors the stage-C (5b) pattern; the Ref is
        #    baked at record time, so the mode is set for all three runs) ---#

        for gh_mode in (:reduced, :lut)
            ref_sys = fresh_psys()
            ref_cache = _binning_device_run(ref_sys,
                () -> fresh_cache(ref_sys, 4, Float32);
                mode=:classsplit, subsort=false, gh_mode)
            gsys = fresh_psys()
            gcache = _with_binning_refs(
                () -> RadixFMMCache(gsys; expansion_order=4, ell=3, near_radius2=16,
                    hessian=true, device=true,
                    options=CUDARadixLifecycleOptions(; precision=Float32,
                        m2l_strategy=FastMultipole.DenseTranslationM2L(
                            apply_chunk=64, build_chunk=8)));
                mode=:classsplit, gh_mode)
            for _ in 1:3
                _binning_device_run(gsys, gcache; mode=:classsplit,
                    subsort=false, gh_mode)
            end
            hctx = gcache.state.interaction_list
            if FastMultipole.CUDA_GRAPH_LIFECYCLE[] && FastMultipole.CUDA_CACHED_WINDOWS[]
                @test hctx.graph_exec !== nothing
                @test hctx.graph_epoch == hctx.epoch_id
            end
            u_scale = maximum(abs.(_binning_inner(ref_sys).gradient_stretching[1:3, :]))
            @test maximum(abs.(_binning_inner(gsys).gradient_stretching[1:3, :] .-
                _binning_inner(ref_sys).gradient_stretching[1:3, :])) / u_scale < 2e-3
        end

        #--- (5) validation paths: invalid mode rejected at launch; :lut on a
        #    regularized kernel without the bin context (flat path) refused ---#

        FastMultipole.CUDA_NEARFIELD_GH_MODE[] = :bogus
        try
            fresh = fresh_psys()
            fcache = RadixFMMCache(fresh; expansion_order=4, ell=3, near_radius2=16,
                hessian=true, device=true,
                options=CUDARadixLifecycleOptions(; precision=Float32,
                    m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
            @test_throws ArgumentError fmm!(fresh, fcache; scalar_potential=false,
                gradient=true, hessian=true)
        finally
            FastMultipole.CUDA_NEARFIELD_GH_MODE[] = :shipped
        end
        # the cache must be BUILT under :lut (047 lock contract) so fmm!
        # reaches the flat-path launch validation rather than the lock error
        rsys = SmoothedVortex(generate_vortex(seed, 400), fill(0.02, 400))
        FastMultipole.CUDA_NEARFIELD_GH_MODE[] = :lut
        try
            rcache = RadixFMMCache(rsys; expansion_order=4, ell=2, hessian=true,
                device=true, policy=FastMultipole.ConstantPAnalyticStencil(4, 1e-3),
                options=CUDARadixLifecycleOptions(; precision=Float32,
                    m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
            @test_throws ArgumentError fmm!(rsys, rcache; scalar_potential=false,
                gradient=true, hessian=true)
        finally
            FastMultipole.CUDA_NEARFIELD_GH_MODE[] = :shipped
        end
    end
end
