using FastMultipole
using FastMultipole.StaticArrays
using Random
using Test

if !isdefined(@__MODULE__, :generate_gravitational)
    include("gravitational.jl")
end

_cuda_integration_required() = get(ENV, "FASTMULTIPOLE_REQUIRE_CUDA_TESTS", "0") == "1"

# Task 027 made HierarchicalRigidStencil the default policy. This file is the FLAT
# device-path regression suite: it asserts the flat route-generation order, flat
# per-class plan partitions, and the flat four-strategy device behavior, all of
# which task 027 must leave unchanged. Passing stencil_epsilon pins the flat
# classifier so this suite keeps testing what it was written to test; hierarchical
# device coverage lives in test/cuda_radix_hierarchical_test.jl.

@testset "CUDA radix fmm! integration (task 023)" begin
    loaded = FastMultipole.load_cuda_radix_lifecycle!()
    if !loaded
        if _cuda_integration_required()
            error(
                "FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 but CUDA radix lifecycle did not load: " *
                FastMultipole.cuda_radix_status(),
            )
        end
        # no functional CUDA runtime: the device cache path must fail loudly
        sys = generate_gravitational(1, 50)
        @test_throws Exception RadixFMMCache(sys; stencil_epsilon=1e-4, expansion_order=4, ell=2, device=true)
    else
        @eval using CUDA
        seed = 20260714
        n = 2000
        P = 6
        ell = 3
        bounds = (SVector(-0.1, -0.1, -0.1), 1.2)

        # DenseTranslationM2L is supported on the recurring cache (task 023f); the
        # recurring-cache success coverage is exercised in the dense block below.
        # Here only the one-shot cuda_radix_state builders keep rejecting it.

        #--- device route generation parity vs the host in-place builder ---#

        sys = generate_gravitational(seed, n)
        host_cache = RadixFMMCache(sys; stencil_epsilon=1e-4, expansion_order=P, ell=ell, bounds=bounds)
        device_cache = RadixFMMCache(sys; stencil_epsilon=1e-4, expansion_order=P, ell=ell, bounds=bounds,
            device=true)
        hs = host_cache.state
        ds = device_cache.state
        @test ds.counts.n_cells == hs.counts.n_cells
        @test ds.counts.n_nodes == hs.counts.n_nodes
        @test ds.counts.n_routes == hs.counts.n_routes
        @test ds.counts.n_direct == hs.counts.n_direct
        nr = hs.counts.n_routes
        nd = hs.counts.n_direct
        @test Array(ds.route_targets)[1:nr] == hs.route_targets[1:nr]
        @test Array(ds.route_sources)[1:nr] == hs.route_sources[1:nr]
        @test Array(ds.route_levels)[1:nr] == hs.route_levels[1:nr]
        @test Array(ds.route_offsets)[:, 1:nr] == hs.route_offsets[:, 1:nr]
        @test Array(ds.direct_targets)[1:nd] == hs.direct_targets[1:nd]
        @test Array(ds.direct_sources)[1:nd] == hs.direct_sources[1:nd]
        host_class = hs.scratch.m2l_concat.route_class
        device_class = Array(ds.scratch.m2l_concat.route_class)
        @test device_class[1:nr] == host_class[1:nr]

        #--- GPU fmm! vs direct! ---#

        ref = generate_gravitational(seed, n)
        FastMultipole.direct!(ref; scalar_potential=true, gradient=true)
        fmm!(sys, device_cache; scalar_potential=true, gradient=true)
        @test maximum(abs.(sys.potential[1, :] .- ref.potential[1, :])) < 1e-4
        @test maximum(abs.(sys.potential[5:7, :] .- ref.potential[5:7, :])) < 1e-2

        # host cache and device cache agree tightly (same algorithm, same grid)
        host_sys = generate_gravitational(seed, n)
        fmm!(host_sys, host_cache; scalar_potential=true, gradient=true)
        @test maximum(abs.(sys.potential[1, :] .- host_sys.potential[1, :])) < 1e-9
        @test maximum(abs.(sys.potential[5:7, :] .- host_sys.potential[5:7, :])) < 1e-8

        #--- stepping loop: transfer-counter contract ---#

        counters = device_cache.state.counters
        base_route_uploads = counters.route_uploads
        base_operator_uploads = counters.operator_uploads
        base_body_uploads = counters.body_uploads
        base_metadata_downloads = counters.metadata_downloads
        # persistent-state contract: the state wrapper and its device arrays are
        # built once at construction and refreshed in place every step
        captured_state = device_cache.state
        captured_arrays = (
            captured_state.grid.perm, captured_state.grid.cell_keys,
            captured_state.grid.node_centers, captured_state.route_targets,
            captured_state.direct_targets, captured_state.m2m_parent_routes,
            captured_state.output, captured_state.host_body_perm,
        )
        rng = MersenneTwister(seed)
        nsteps = 5
        for step in 1:nsteps
            for i in eachindex(sys.bodies)
                b = sys.bodies[i]
                pos = clamp.(b.position .+ 0.02 .* (rand(rng, SVector{3,Float64}) .- 0.5),
                    -0.05, 1.05)
                sys.bodies[i] = Body(pos, b.radius, b.strength)
            end
            fmm!(sys, device_cache; scalar_potential=true, gradient=true)
            @test device_cache.state === captured_state
            @test all(current === original for (current, original) in zip((
                captured_state.grid.perm, captured_state.grid.cell_keys,
                captured_state.grid.node_centers, captured_state.route_targets,
                captured_state.direct_targets, captured_state.m2m_parent_routes,
                captured_state.output, captured_state.host_body_perm,
            ), captured_arrays))
            @test device_cache.state.counters === counters
            @test counters.route_uploads == base_route_uploads
            @test counters.operator_uploads == base_operator_uploads
            @test counters.body_uploads == base_body_uploads + step
            @test counters.metadata_downloads == base_metadata_downloads + 3 * step
            @test counters.expansion_host_copies == 0

            ref_step = Gravitational(copy(sys.bodies), zeros(16, length(sys.bodies)))
            FastMultipole.direct!(ref_step; scalar_potential=true, gradient=true)
            @test maximum(abs.(sys.potential[1, :] .- ref_step.potential[1, :])) < 1e-4
        end

        #--- fixed-box contract: a body leaving the box throws, then recovers ---#

        good_body = sys.bodies[1]
        sys.bodies[1] = Body(SVector(5.0, 0.5, 0.5), good_body.radius, good_body.strength)
        @test_throws ArgumentError fmm!(sys, device_cache;
            scalar_potential=true, gradient=true)
        sys.bodies[1] = good_body
        fmm!(sys, device_cache; scalar_potential=true, gradient=true)
        ref_recover = Gravitational(copy(sys.bodies), zeros(16, length(sys.bodies)))
        FastMultipole.direct!(ref_recover; scalar_potential=true, gradient=true)
        @test maximum(abs.(sys.potential[1, :] .- ref_recover.potential[1, :])) < 1e-4

        #--- per-step timing: update+lifecycle stays near the bare lifecycle ---#

        t_step = @elapsed fmm!(sys, device_cache; scalar_potential=true, gradient=true)
        t_lifecycle = @elapsed begin
            run_cuda_radix_lifecycle!(device_cache.state)
            CUDA.synchronize()
        end
        @test t_step < 50 * t_lifecycle + 1.0   # loose sanity; hard numbers in 023 benchmarks

        #--- factored resident M2L on the device lifecycle (task 023b) ---#

        mk_factored_opts(TF) = CUDARadixLifecycleOptions(; precision=TF,
            operator=FactoredRotationM2L(),
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L())

        # the one-shot state builders keep rejecting the factored operator
        grid_1shot = FastMultipole.RadixGrid(sys, 3)
        list_1shot = FastMultipole.build_radix_interaction_list(
            FastMultipole.LazyMaterializedBatches(1), FastMultipole.ParentNeighborM2L(),
            grid_1shot)
        @test_throws ArgumentError FastMultipole.cuda_radix_state(
            sys, grid_1shot, list_1shot, 4; options=mk_factored_opts(Float64))

        for P in (4, 8, 12), (TF, LH) in ((Float64, false), (Float64, true),
                                      (Float32, false), (Float32, true))
            nf = 800
            fseed = seed + 23
            factored_opts = mk_factored_opts(TF)
            dev_sys = generate_gravitational(fseed, nf)
            dev_fcache = RadixFMMCache(dev_sys; stencil_epsilon=1e-4, expansion_order=P, ell=3, bounds=bounds,
                lamb_helmholtz=LH, device=true, options=factored_opts)
            host_sys = generate_gravitational(fseed, nf)
            host_fcache = RadixFMMCache(host_sys; stencil_epsilon=1e-4, expansion_order=P, ell=3, bounds=bounds,
                lamb_helmholtz=LH, options=factored_opts)
            dplan = dev_fcache.state.scratch.m2l_concat
            hplan = host_fcache.state.scratch.m2l_concat
            @test dplan isa FastMultipole.ResidentM2LFactoredPlan
            @test dplan.route_class isa CUDA.AnyCuArray
            @test isempty(dplan.groups)
            @test length(hplan.groups) == length(dplan.host_class_counts)

            # Compact operator tables are the complete CUDA per-class geometry.
            # Reconstruct every class independently from its Cartesian stencil
            # offset and rebuild every fixed-m z column from scratch.
            cell_width = 2 * dev_fcache.h0 / (1 << dev_fcache.ell)
            zhost = Array(dplan.z_flat)
            zfresh = Vector{TF}(undef, size(zhost, 1))
            geom_tol = TF === Float64 ? 5e-15 : 5f-6
            for (k, offset) in enumerate(dev_fcache.accepted_offsets)
                delta = SVector{3,TF}(offset) * TF(cell_width)
                r, theta, phi = FastMultipole.cartesian_to_spherical(delta)
                @test isapprox(dplan.class_r[k], r; rtol=geom_tol, atol=geom_tol)
                @test isapprox(dplan.class_theta[k], theta; rtol=geom_tol, atol=geom_tol)
                @test isapprox(dplan.class_phi[k], phi; rtol=geom_tol, atol=geom_tol)
                FastMultipole.m2l_z_blocks!(zfresh, r,
                    dev_fcache.state.invariant_cache.basis_info.orders.P_active)
                @test isapprox(view(zhost, :, k), zfresh; rtol=geom_tol, atol=geom_tol)
            end

            # class-contiguity invariant behind the per-class column ranges, and
            # histogram/starts consistency with the emitted route classes
            nrf = dev_fcache.state.counts.n_routes
            dclass = Array(dplan.route_class)[1:nrf]
            @test issorted(dclass)
            @test dplan.class_starts[end] == nrf
            counts_from_class = zeros(Int, length(dplan.host_class_counts))
            for c in dclass
                counts_from_class[c] += 1
            end
            @test counts_from_class == Int.(dplan.host_class_counts)

            # device-vs-host route/direct parity; at P = 4 the stencil rejects
            # more offsets than max_cells, exercising the chunked direct-flag
            # generation (order must stay elementwise identical to the host)
            if TF === Float64
                hfs = host_fcache.state
                dfs = dev_fcache.state
                @test dfs.counts.n_routes == hfs.counts.n_routes
                @test dfs.counts.n_direct == hfs.counts.n_direct
                @test Array(dfs.route_targets)[1:nrf] == hfs.route_targets[1:nrf]
                @test Array(dfs.route_sources)[1:nrf] == hfs.route_sources[1:nrf]
                @test dclass == hplan.route_class[1:nrf]
                ndir = hfs.counts.n_direct
                @test Array(dfs.direct_targets)[1:ndir] == hfs.direct_targets[1:ndir]
                @test Array(dfs.direct_sources)[1:ndir] == hfs.direct_sources[1:ndir]
            end

            # Independent per-class CUDA oracle vs the 023a host implementation
            # at the M2L output-buffer boundary, before L2L/L2B can mask errors.
            FastMultipole._launch_host_b2m!(host_fcache.state)
            FastMultipole._launch_host_m2m!(host_fcache.state)
            FastMultipole._launch_host_m2l!(host_fcache.state)
            FastMultipole._launch_cuda_b2m!(dev_fcache.state)
            FastMultipole._launch_cuda_resident_m2m!(dev_fcache.state)
            saved_wp_stage = FastMultipole.FACTORED_CUDA_WHOLE_PASS[]
            try
                FastMultipole.FACTORED_CUDA_WHOLE_PASS[] = false
                FastMultipole._launch_cuda_resident_m2l!(dev_fcache.state)
                CUDA.synchronize()
            finally
                FastMultipole.FACTORED_CUDA_WHOLE_PASS[] = saved_wp_stage
            end
            m2l_rtol = TF === Float64 ? 1e-9 : 5f-3
            m2l_atol = TF === Float64 ? 1e-10 : 5f-4
            nnodes = host_fcache.state.counts.n_nodes
            @test Array(dev_fcache.state.locals.phi)[:, 1:nnodes] ≈
                host_fcache.state.locals.phi[:, 1:nnodes] rtol=m2l_rtol atol=m2l_atol
            if LH
                @test Array(dev_fcache.state.locals.chi)[:, 1:nnodes] ≈
                    host_fcache.state.locals.chi[:, 1:nnodes] rtol=m2l_rtol atol=m2l_atol
            end

            fmm!(dev_sys, dev_fcache; scalar_potential=!LH, gradient=true)
            fmm!(host_sys, host_fcache; scalar_potential=!LH, gradient=true)

            # GPU-factored vs host-factored (023a oracle): same operators, same grid
            p_tol = TF === Float64 ? 1e-9 : 1f-3
            g_tol = TF === Float64 ? 1e-8 : 1f-2
            !LH && @test maximum(abs.(dev_sys.potential[1, :] .-
                host_sys.potential[1, :])) < p_tol
            @test maximum(abs.(dev_sys.potential[5:7, :] .-
                host_sys.potential[5:7, :])) < g_tol

            # absolute accuracy vs direct! at the stencil design order
            if P == 8
                ref_f = generate_gravitational(fseed, nf)
                FastMultipole.direct!(ref_f; scalar_potential=true, gradient=true)
                direct_p_tol = TF === Float64 ? 1e-4 : 1f-3
                direct_g_tol = TF === Float64 ? 1e-2 : 1f-1
                !LH && @test maximum(abs.(dev_sys.potential[1, :] .-
                    ref_f.potential[1, :])) < direct_p_tol
                @test maximum(abs.(dev_sys.potential[5:7, :] .-
                    ref_f.potential[5:7, :])) < direct_g_tol
            end

            # whole-pass (production default) vs per-class reference parity: same
            # per-column arithmetic, differing only in launch shape / accumulation
            # order at the atomic scatter
            @test dplan.whole_pass[] !== nothing
            whole_pot = copy(dev_sys.potential)
            saved_wp = FastMultipole.FACTORED_CUDA_WHOLE_PASS[]
            try
                FastMultipole.FACTORED_CUDA_WHOLE_PASS[] = false
                fmm!(dev_sys, dev_fcache; scalar_potential=!LH, gradient=true)
            finally
                FastMultipole.FACTORED_CUDA_WHOLE_PASS[] = saved_wp
            end
            wp_tol = TF === Float64 ? 1e-10 : 1f-4
            !LH && @test maximum(abs.(dev_sys.potential[1, :] .- whole_pot[1, :])) < wp_tol
            @test maximum(abs.(dev_sys.potential[5:7, :] .- whole_pot[5:7, :])) < 10 * wp_tol
            fmm!(dev_sys, dev_fcache; scalar_potential=!LH, gradient=true)

            # steady-state device allocation: the factored M2L stage runs entirely
            # on preallocated plan/workspace storage (runtime @eval: CUDA.@allocated
            # is a macro and CUDA only loads inside this branch)
            FastMultipole._launch_resident_m2l!(dev_fcache.state)
            dev_alloc = @eval CUDA.@allocated FastMultipole._launch_resident_m2l!($(dev_fcache).state)
            @test dev_alloc == 0
        end

        #--- factored stepping loop: transfer-counter contract + refresh ---#

        fstep_sys = generate_gravitational(seed + 31, 1500)
        fstep_host_sys = generate_gravitational(seed + 31, 1500)
        fstep_cache = RadixFMMCache(fstep_sys; stencil_epsilon=1e-4, expansion_order=6, ell=3, bounds=bounds,
            device=true, options=mk_factored_opts(Float64))
        fstep_host_cache = RadixFMMCache(fstep_host_sys; stencil_epsilon=1e-4, expansion_order=6, ell=3,
            bounds=bounds, options=mk_factored_opts(Float64))
        fcounters = fstep_cache.state.counters
        fbase_route = fcounters.route_uploads
        fbase_operator = fcounters.operator_uploads
        fbase_body = fcounters.body_uploads
        fbase_metadata = fcounters.metadata_downloads
        fcaptured_state = fstep_cache.state
        fplan = fstep_cache.state.scratch.m2l_concat
        fcaptured_arrays = (fplan.route_class, fplan.class_counts,
            fcaptured_state.route_targets, fcaptured_state.output)
        frng = MersenneTwister(seed + 31)
        for step in 1:3
            for i in eachindex(fstep_sys.bodies)
                b = fstep_sys.bodies[i]
                pos = clamp.(b.position .+ 0.02 .* (rand(frng, SVector{3,Float64}) .- 0.5),
                    -0.05, 1.05)
                fstep_sys.bodies[i] = Body(pos, b.radius, b.strength)
                hb = fstep_host_sys.bodies[i]
                fstep_host_sys.bodies[i] = Body(pos, hb.radius, hb.strength)
            end
            fmm!(fstep_sys, fstep_cache; scalar_potential=true, gradient=true)
            fmm!(fstep_host_sys, fstep_host_cache; scalar_potential=true, gradient=true)
            @test fstep_cache.state === fcaptured_state
            @test all(current === original for (current, original) in zip((
                fplan.route_class, fplan.class_counts,
                fcaptured_state.route_targets, fcaptured_state.output,
            ), fcaptured_arrays))
            @test fcounters.route_uploads == fbase_route
            @test fcounters.operator_uploads == fbase_operator
            @test fcounters.body_uploads == fbase_body + step
            @test fcounters.metadata_downloads == fbase_metadata + 3 * step
            @test fcounters.expansion_host_copies == 0
            nr_step = fstep_cache.state.counts.n_routes
            @test fplan.class_starts[end] == nr_step
            @test sum(Int, fplan.host_class_counts) == nr_step
            hstep = fstep_host_cache.state
            hstep_plan = hstep.scratch.m2l_concat
            @test Array(fplan.route_class)[1:nr_step] == hstep_plan.route_class[1:nr_step]
            @test Array(fstep_cache.state.route_sources)[1:nr_step] ==
                hstep.route_sources[1:nr_step]
            @test Array(fstep_cache.state.route_targets)[1:nr_step] ==
                hstep.route_targets[1:nr_step]

            ref_step = Gravitational(copy(fstep_sys.bodies), zeros(16, length(fstep_sys.bodies)))
            FastMultipole.direct!(ref_step; scalar_potential=true, gradient=true)
            @test maximum(abs.(fstep_sys.potential[1, :] .- ref_step.potential[1, :])) < 1e-4
        end

        #--- factored empty-route degenerate case: all-nearfield cluster ---#

        cl_sys = generate_gravitational(seed + 37, 60)
        for i in eachindex(cl_sys.bodies)
            b = cl_sys.bodies[i]
            cl_sys.bodies[i] = Body(SVector(0.5, 0.5, 0.5) .+ 0.01 .* (b.position .- 0.5),
                b.radius, b.strength)
        end
        cl_cache = RadixFMMCache(cl_sys; stencil_epsilon=1e-4, expansion_order=4, ell=3, bounds=bounds,
            device=true, options=mk_factored_opts(Float64))
        fmm!(cl_sys, cl_cache; scalar_potential=true, gradient=true)
        cl_plan = cl_cache.state.scratch.m2l_concat
        @test cl_cache.state.counts.n_routes == cl_plan.class_starts[end]
        cl_ref = Gravitational(copy(cl_sys.bodies), zeros(16, length(cl_sys.bodies)))
        FastMultipole.direct!(cl_ref; scalar_potential=true, gradient=true)
        # a tight cluster resolves (almost) entirely through direct pairs
        @test maximum(abs.(cl_sys.potential[1, :] .- cl_ref.potential[1, :])) < 1e-6

        #--- DEBUG[]-on factored device run (016b guard, device mirror) ---#

        saved_debug = FastMultipole.DEBUG[]
        try
            FastMultipole.DEBUG[] = true
            dbg_sys = generate_gravitational(seed + 41, 400)
            dbg_cache = RadixFMMCache(dbg_sys; stencil_epsilon=1e-4, expansion_order=4, ell=3, bounds=bounds,
                device=true, options=mk_factored_opts(Float64))
            fmm!(dbg_sys, dbg_cache; scalar_potential=true, gradient=true)
            dbg_ref = Gravitational(copy(dbg_sys.bodies), zeros(16, length(dbg_sys.bodies)))
            FastMultipole.direct!(dbg_ref; scalar_potential=true, gradient=true)
            @test maximum(abs.(dbg_sys.potential[1, :] .- dbg_ref.potential[1, :])) < 1e-2
        finally
            FastMultipole.DEBUG[] = saved_debug
        end

        #--- precomputed-y resident M2L on the device lifecycle (task 023d) ---#

        mk_prey_opts(TF) = CUDARadixLifecycleOptions(; precision=TF,
            operator=FactoredRotationM2L(),
            m2l_strategy=FastMultipole.PrecomputedFactoredYM2L())
        mk_concat_opts(TF) = CUDARadixLifecycleOptions(; precision=TF,
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L())

        # options must keep pairing the strategy with the factored operator, and
        # the one-shot state builders keep rejecting the factored operator family
        @test_throws ArgumentError CUDARadixLifecycleOptions(;
            m2l_strategy=FastMultipole.PrecomputedFactoredYM2L())
        @test_throws ArgumentError FastMultipole.cuda_radix_state(
            sys, grid_1shot, list_1shot, 4; options=mk_prey_opts(Float64))

        for P in (4, 8, 12), (TF, LH) in ((Float64, false), (Float64, true),
                                      (Float32, false), (Float32, true))
            nf = 800
            fseed = seed + 51
            prey_opts = mk_prey_opts(TF)
            dev_sys = generate_gravitational(fseed, nf)
            dev_pcache = RadixFMMCache(dev_sys; stencil_epsilon=1e-4, expansion_order=P, ell=3, bounds=bounds,
                lamb_helmholtz=LH, device=true, options=prey_opts)
            host_sys = generate_gravitational(fseed, nf)
            host_pcache = RadixFMMCache(host_sys; stencil_epsilon=1e-4, expansion_order=P, ell=3, bounds=bounds,
                lamb_helmholtz=LH, options=prey_opts)
            dplan = dev_pcache.state.scratch.m2l_concat
            hplan = host_pcache.state.scratch.m2l_concat
            @test dplan isa FastMultipole.ResidentM2LPrecomputedYPlan
            @test dplan.route_class isa CUDA.AnyCuArray
            # compact device plan: flat tables only, host oracle keeps the nested
            # per-angle/per-offset storage
            @test isempty(dplan.y_mult) && isempty(dplan.z_phi)
            @test !isempty(hplan.y_mult) && !isempty(hplan.z_phi)
            @test dplan.whole_pass[] !== nothing
            @test dplan.offset_to_angle == hplan.offset_to_angle
            @test dplan.angle_keys == hplan.angle_keys
            @test dplan.angle_thetas == hplan.angle_thetas

            # Flat operator tables are the complete device geometry. Reconstruct
            # every angle key and z column from the Cartesian stencil offsets, and
            # compare every flat y block against the host plan's nested matrices.
            P_active = dev_pcache.state.invariant_cache.basis_info.orders.P_active
            cell_width = 2 * dev_pcache.h0 / (1 << dev_pcache.ell)
            zhost = Array(dplan.z_flat)
            zfresh = Vector{TF}(undef, size(zhost, 1))
            geom_tol = TF === Float64 ? 5e-15 : 5f-6
            for (k, offset) in enumerate(dev_pcache.accepted_offsets)
                a = dplan.offset_to_angle[k]
                @test dplan.angle_keys[a] ==
                    FastMultipole._precomputed_y_angle_key(offset)
                delta = SVector{3,TF}(offset) * TF(cell_width)
                r, theta, phi = FastMultipole.cartesian_to_spherical(delta)
                @test isapprox(dplan.angle_thetas[a], theta; rtol=geom_tol, atol=geom_tol)
                @test isapprox(dplan.offset_phis[k], phi; rtol=geom_tol, atol=geom_tol)
                @test isapprox((dplan.offset_rs::Vector{TF})[k], r;
                    rtol=geom_tol, atol=geom_tol)
                FastMultipole.m2l_z_blocks!(zfresh, r, P_active)
                @test isapprox(view(zhost, :, k), zfresh; rtol=geom_tol, atol=geom_tol)
            end
            yfm = Array(dplan.y_flat_mult)
            yfl = Array(dplan.y_flat_loc)
            for a in eachindex(dplan.angle_thetas), n in 0:P_active
                d = 2 * n + 1
                off = FastMultipole.ymode_offset(n)
                # the flat block is the column-major image of M_n(theta) itself
                seg = reshape(view(yfm, off .+ (1:(d * d)), a), d, d)
                @test seg == hplan.y_mult[a][n + 1]
                segl = reshape(view(yfl, off .+ (1:(d * d)), a), d, d)
                @test segl == hplan.y_loc[a][n + 1]
            end

            # route-class parity with the host cache, histogram/prefix consistency
            nrp = dev_pcache.state.counts.n_routes
            dclass = Array(dplan.route_class)[1:nrp]
            @test issorted(dclass)
            @test dplan.offset_starts[end] - 1 == nrp
            @test sum(dplan.offset_counts) == nrp
            counts_from_class = zeros(Int, length(dplan.offset_counts))
            for c in dclass
                counts_from_class[c] += 1
            end
            @test counts_from_class == dplan.offset_counts
            if TF === Float64
                @test nrp == host_pcache.state.counts.n_routes
                @test dclass == hplan.route_class[1:nrp]
                @test dplan.offset_counts == hplan.offset_counts
                @test dplan.angle_counts == hplan.angle_counts
            end

            # M2L output-buffer parity vs the approved 023c host implementation,
            # before L2L/L2B can mask errors: per-class reference first, then the
            # whole-pass production dispatch
            FastMultipole._launch_host_b2m!(host_pcache.state)
            FastMultipole._launch_host_m2m!(host_pcache.state)
            FastMultipole._launch_host_m2l!(host_pcache.state)
            FastMultipole._launch_cuda_b2m!(dev_pcache.state)
            FastMultipole._launch_cuda_resident_m2m!(dev_pcache.state)
            m2l_rtol = TF === Float64 ? 1e-9 : 5f-3
            m2l_atol = TF === Float64 ? 1e-10 : 5f-4
            nnodes = host_pcache.state.counts.n_nodes
            saved_wp_stage = FastMultipole.PRECOMPUTED_CUDA_WHOLE_PASS[]
            try
                FastMultipole.PRECOMPUTED_CUDA_WHOLE_PASS[] = false
                FastMultipole._launch_cuda_resident_m2l!(dev_pcache.state)
                CUDA.synchronize()
            finally
                FastMultipole.PRECOMPUTED_CUDA_WHOLE_PASS[] = saved_wp_stage
            end
            @test Array(dev_pcache.state.locals.phi)[:, 1:nnodes] ≈
                host_pcache.state.locals.phi[:, 1:nnodes] rtol=m2l_rtol atol=m2l_atol
            if LH
                @test Array(dev_pcache.state.locals.chi)[:, 1:nnodes] ≈
                    host_pcache.state.locals.chi[:, 1:nnodes] rtol=m2l_rtol atol=m2l_atol
            end
            FastMultipole._launch_cuda_resident_m2l!(dev_pcache.state)
            CUDA.synchronize()
            @test Array(dev_pcache.state.locals.phi)[:, 1:nnodes] ≈
                host_pcache.state.locals.phi[:, 1:nnodes] rtol=m2l_rtol atol=m2l_atol
            if LH
                @test Array(dev_pcache.state.locals.chi)[:, 1:nnodes] ≈
                    host_pcache.state.locals.chi[:, 1:nnodes] rtol=m2l_rtol atol=m2l_atol
            end

            # end-to-end: GPU-precomputed vs host-precomputed (023c oracle)
            fmm!(dev_sys, dev_pcache; scalar_potential=!LH, gradient=true)
            fmm!(host_sys, host_pcache; scalar_potential=!LH, gradient=true)
            p_tol = TF === Float64 ? 1e-9 : 1f-3
            g_tol = TF === Float64 ? 1e-8 : 1f-2
            !LH && @test maximum(abs.(dev_sys.potential[1, :] .-
                host_sys.potential[1, :])) < p_tol
            @test maximum(abs.(dev_sys.potential[5:7, :] .-
                host_sys.potential[5:7, :])) < g_tol

            # cross-strategy parity on device: precomputed-y vs the CUDA concat
            # path and the CUDA per-degree factored path (same grid, same order)
            if TF === Float64
                concat_sys = generate_gravitational(fseed, nf)
                concat_cache = RadixFMMCache(concat_sys; stencil_epsilon=1e-4, expansion_order=P, ell=3,
                    bounds=bounds, lamb_helmholtz=LH, device=true,
                    options=mk_concat_opts(TF))
                fmm!(concat_sys, concat_cache; scalar_potential=!LH, gradient=true)
                !LH && @test maximum(abs.(dev_sys.potential[1, :] .-
                    concat_sys.potential[1, :])) < p_tol
                @test maximum(abs.(dev_sys.potential[5:7, :] .-
                    concat_sys.potential[5:7, :])) < g_tol
                fac_sys = generate_gravitational(fseed, nf)
                fac_cache = RadixFMMCache(fac_sys; stencil_epsilon=1e-4, expansion_order=P, ell=3,
                    bounds=bounds, lamb_helmholtz=LH, device=true,
                    options=mk_factored_opts(TF))
                fmm!(fac_sys, fac_cache; scalar_potential=!LH, gradient=true)
                !LH && @test maximum(abs.(dev_sys.potential[1, :] .-
                    fac_sys.potential[1, :])) < p_tol
                @test maximum(abs.(dev_sys.potential[5:7, :] .-
                    fac_sys.potential[5:7, :])) < g_tol
            end

            # absolute accuracy vs direct! at the stencil design order
            if P == 8
                ref_p = generate_gravitational(fseed, nf)
                FastMultipole.direct!(ref_p; scalar_potential=true, gradient=true)
                direct_p_tol = TF === Float64 ? 1e-4 : 1f-3
                direct_g_tol = TF === Float64 ? 1e-2 : 1f-1
                !LH && @test maximum(abs.(dev_sys.potential[1, :] .-
                    ref_p.potential[1, :])) < direct_p_tol
                @test maximum(abs.(dev_sys.potential[5:7, :] .-
                    ref_p.potential[5:7, :])) < direct_g_tol
            end

            # whole-pass (production default) vs per-class reference, end to end
            whole_pot = copy(dev_sys.potential)
            saved_wp = FastMultipole.PRECOMPUTED_CUDA_WHOLE_PASS[]
            try
                FastMultipole.PRECOMPUTED_CUDA_WHOLE_PASS[] = false
                fmm!(dev_sys, dev_pcache; scalar_potential=!LH, gradient=true)
            finally
                FastMultipole.PRECOMPUTED_CUDA_WHOLE_PASS[] = saved_wp
            end
            wp_tol = TF === Float64 ? 1e-10 : 1f-4
            !LH && @test maximum(abs.(dev_sys.potential[1, :] .- whole_pot[1, :])) < wp_tol
            @test maximum(abs.(dev_sys.potential[5:7, :] .- whole_pot[5:7, :])) < 10 * wp_tol
            fmm!(dev_sys, dev_pcache; scalar_potential=!LH, gradient=true)

            # steady-state device allocation: the precomputed-y M2L stage runs
            # entirely on preallocated plan/bundle storage
            FastMultipole._launch_resident_m2l!(dev_pcache.state)
            dev_alloc = @eval CUDA.@allocated FastMultipole._launch_resident_m2l!($(dev_pcache).state)
            @test dev_alloc == 0
        end

        #--- precomputed-y tiny-chunk stress: multi-chunk whole pass and per-class
        #    sub-chunking both reduce to the same per-column arithmetic ---#

        saved_chunk = FastMultipole.PRECOMPUTED_CUDA_CHUNK[]
        try
            FastMultipole.PRECOMPUTED_CUDA_CHUNK[] = 7
            tc_sys = generate_gravitational(seed + 53, 400)
            tc_cache = RadixFMMCache(tc_sys; stencil_epsilon=1e-4, expansion_order=4, ell=3, bounds=bounds,
                device=true, options=mk_prey_opts(Float64))
            @test (tc_cache.state.scratch.m2l_concat.whole_pass[]::NamedTuple).chunk == 7
            fmm!(tc_sys, tc_cache; scalar_potential=true, gradient=true)
            tc_pot = copy(tc_sys.potential)
            saved_tc_wp = FastMultipole.PRECOMPUTED_CUDA_WHOLE_PASS[]
            try
                FastMultipole.PRECOMPUTED_CUDA_WHOLE_PASS[] = false
                fmm!(tc_sys, tc_cache; scalar_potential=true, gradient=true)
            finally
                FastMultipole.PRECOMPUTED_CUDA_WHOLE_PASS[] = saved_tc_wp
            end
            @test maximum(abs.(tc_sys.potential[1, :] .- tc_pot[1, :])) < 1e-10
            tc_host_sys = generate_gravitational(seed + 53, 400)
            tc_host_cache = RadixFMMCache(tc_host_sys; stencil_epsilon=1e-4, expansion_order=4, ell=3,
                bounds=bounds, options=mk_prey_opts(Float64))
            fmm!(tc_host_sys, tc_host_cache; scalar_potential=true, gradient=true)
            @test maximum(abs.(tc_sys.potential[1, :] .- tc_host_sys.potential[1, :])) < 1e-9
        finally
            FastMultipole.PRECOMPUTED_CUDA_CHUNK[] = saved_chunk
        end

        #--- precomputed-y stepping loop: transfer-counter contract + refresh ---#

        pstep_sys = generate_gravitational(seed + 57, 1500)
        pstep_host_sys = generate_gravitational(seed + 57, 1500)
        pstep_cache = RadixFMMCache(pstep_sys; stencil_epsilon=1e-4, expansion_order=6, ell=3, bounds=bounds,
            device=true, options=mk_prey_opts(Float64))
        pstep_host_cache = RadixFMMCache(pstep_host_sys; stencil_epsilon=1e-4, expansion_order=6, ell=3,
            bounds=bounds, options=mk_prey_opts(Float64))
        pcounters = pstep_cache.state.counters
        pbase_route = pcounters.route_uploads
        pbase_operator = pcounters.operator_uploads
        pbase_body = pcounters.body_uploads
        pbase_metadata = pcounters.metadata_downloads
        pcaptured_state = pstep_cache.state
        pplan = pstep_cache.state.scratch.m2l_concat
        pcaptured_arrays = (pplan.route_class, pplan.class_counts, pplan.y_flat_mult,
            pplan.z_flat, pcaptured_state.route_targets, pcaptured_state.output)
        prng = MersenneTwister(seed + 57)
        for step in 1:3
            for i in eachindex(pstep_sys.bodies)
                b = pstep_sys.bodies[i]
                pos = clamp.(b.position .+ 0.02 .* (rand(prng, SVector{3,Float64}) .- 0.5),
                    -0.05, 1.05)
                pstep_sys.bodies[i] = Body(pos, b.radius, b.strength)
                hb = pstep_host_sys.bodies[i]
                pstep_host_sys.bodies[i] = Body(pos, hb.radius, hb.strength)
            end
            fmm!(pstep_sys, pstep_cache; scalar_potential=true, gradient=true)
            fmm!(pstep_host_sys, pstep_host_cache; scalar_potential=true, gradient=true)
            @test pstep_cache.state === pcaptured_state
            @test all(current === original for (current, original) in zip((
                pplan.route_class, pplan.class_counts, pplan.y_flat_mult,
                pplan.z_flat, pcaptured_state.route_targets, pcaptured_state.output,
            ), pcaptured_arrays))
            @test pcounters.route_uploads == pbase_route
            @test pcounters.operator_uploads == pbase_operator
            @test pcounters.body_uploads == pbase_body + step
            @test pcounters.metadata_downloads == pbase_metadata + 3 * step
            @test pcounters.expansion_host_copies == 0
            nr_step = pstep_cache.state.counts.n_routes
            @test pplan.offset_starts[end] - 1 == nr_step
            @test sum(pplan.offset_counts) == nr_step
            hstep = pstep_host_cache.state
            hstep_plan = hstep.scratch.m2l_concat
            @test Array(pplan.route_class)[1:nr_step] == hstep_plan.route_class[1:nr_step]
            @test pplan.offset_counts == hstep_plan.offset_counts
            @test pplan.angle_counts == hstep_plan.angle_counts
            @test Array(pstep_cache.state.route_sources)[1:nr_step] ==
                hstep.route_sources[1:nr_step]
            @test Array(pstep_cache.state.route_targets)[1:nr_step] ==
                hstep.route_targets[1:nr_step]

            ref_step = Gravitational(copy(pstep_sys.bodies), zeros(16, length(pstep_sys.bodies)))
            FastMultipole.direct!(ref_step; scalar_potential=true, gradient=true)
            @test maximum(abs.(pstep_sys.potential[1, :] .- ref_step.potential[1, :])) < 1e-4
            @test maximum(abs.(pstep_sys.potential[1, :] .-
                pstep_host_sys.potential[1, :])) < 1e-9
        end

        #--- precomputed-y empty-route degenerate case: all-nearfield cluster ---#

        pcl_sys = generate_gravitational(seed + 61, 60)
        for i in eachindex(pcl_sys.bodies)
            b = pcl_sys.bodies[i]
            pcl_sys.bodies[i] = Body(SVector(0.5, 0.5, 0.5) .+ 0.01 .* (b.position .- 0.5),
                b.radius, b.strength)
        end
        pcl_cache = RadixFMMCache(pcl_sys; stencil_epsilon=1e-4, expansion_order=4, ell=3, bounds=bounds,
            device=true, options=mk_prey_opts(Float64))
        fmm!(pcl_sys, pcl_cache; scalar_potential=true, gradient=true)
        pcl_plan = pcl_cache.state.scratch.m2l_concat
        @test pcl_cache.state.counts.n_routes == pcl_plan.offset_starts[end] - 1
        pcl_ref = Gravitational(copy(pcl_sys.bodies), zeros(16, length(pcl_sys.bodies)))
        FastMultipole.direct!(pcl_ref; scalar_potential=true, gradient=true)
        @test maximum(abs.(pcl_sys.potential[1, :] .- pcl_ref.potential[1, :])) < 1e-6

        #--- DEBUG[]-on precomputed-y device run (016b guard, device mirror) ---#

        saved_pdebug = FastMultipole.DEBUG[]
        try
            FastMultipole.DEBUG[] = true
            pdbg_sys = generate_gravitational(seed + 63, 400)
            pdbg_cache = RadixFMMCache(pdbg_sys; stencil_epsilon=1e-4, expansion_order=4, ell=3, bounds=bounds,
                device=true, options=mk_prey_opts(Float64))
            fmm!(pdbg_sys, pdbg_cache; scalar_potential=true, gradient=true)
            pdbg_ref = Gravitational(copy(pdbg_sys.bodies), zeros(16, length(pdbg_sys.bodies)))
            FastMultipole.direct!(pdbg_ref; scalar_potential=true, gradient=true)
            @test maximum(abs.(pdbg_sys.potential[1, :] .- pdbg_ref.potential[1, :])) < 1e-2
        finally
            FastMultipole.DEBUG[] = saved_pdebug
        end

        #--- dense-translation resident M2L on the device lifecycle (task 023f) ---#

        mk_dense_opts(TF; kw...) = CUDARadixLifecycleOptions(; precision=TF,
            operator=MaterializedYRotationM2L(),
            m2l_strategy=DenseTranslationM2L(; kw...))

        # options keep enforcing the materialized-y pairing, and the one-shot state
        # builders keep rejecting the dense strategy
        @test_throws ArgumentError CUDARadixLifecycleOptions(
            operator=FactoredRotationM2L(), m2l_strategy=DenseTranslationM2L())
        @test_throws ArgumentError FastMultipole.cuda_radix_state(
            sys, grid_1shot, list_1shot, 4; options=mk_dense_opts(Float64))

        for P in (4, 8, 12), (TF, LH) in ((Float64, false), (Float64, true),
                                      (Float32, false), (Float32, true))
            nf = 800
            fseed = seed + 71
            dense_opts = mk_dense_opts(TF)
            if TF === Float32 && P == 12
                for dev in (false, true)
                    rejected = try
                        RadixFMMCache(generate_gravitational(fseed, nf); stencil_epsilon=1e-4,
                            expansion_order=P, ell=3, bounds=bounds,
                            lamb_helmholtz=LH, device=dev, options=dense_opts)
                        nothing
                    catch err
                        err
                    end
                    @test rejected isa ArgumentError
                    rejected_msg = sprint(showerror, rejected)
                    @test occursin("DenseTranslationM2L", rejected_msg)
                    @test occursin("non-finite", rejected_msg)
                    @test occursin("precision=Float32", rejected_msg)
                    @test occursin("P=12", rejected_msg)
                    @test occursin("Lamb-Helmholtz=$(LH)", rejected_msg)
                    @test occursin("PrecomputedFactoredYM2L", rejected_msg)
                end
                continue
            end
            dev_sys = generate_gravitational(fseed, nf)
            dev_dcache = RadixFMMCache(dev_sys; stencil_epsilon=1e-4, expansion_order=P, ell=3, bounds=bounds,
                lamb_helmholtz=LH, device=true, options=dense_opts)
            host_sys = generate_gravitational(fseed, nf)
            host_dcache = RadixFMMCache(host_sys; stencil_epsilon=1e-4, expansion_order=P, ell=3, bounds=bounds,
                lamb_helmholtz=LH, options=dense_opts)
            dplan = dev_dcache.state.scratch.m2l_concat
            hplan = host_dcache.state.scratch.m2l_concat
            @test dplan isa FastMultipole.ResidentM2LDenseCUDAPlan
            @test hplan isa FastMultipole.ResidentM2LDensePlan
            @test dplan.route_class isa CUDA.AnyCuArray
            @test dplan.operators isa CUDA.AnyCuArray && ndims(dplan.operators) == 3
            @test size(dplan.operators) == (dplan.ndof, dplan.ndof, dplan.nclasses)
            @test dplan.class_counts isa CUDA.AnyCuArray
            @test dplan.src_slab isa CUDA.AnyCuArray && dplan.dst_slab isa CUDA.AnyCuArray
            @test dplan.whole_pass[] !== nothing
            @test dplan.nclasses == length(dev_dcache.accepted_offsets)
            @test dplan.class_capacities == hplan.class_capacities
            @test dplan.estimated_peak_bytes > dplan.persistent_bytes

            # Every packed device operator slice is a bit-exact copy of the host
            # oracle matrix (the CUDA plan uploads the host-built operators verbatim).
            dops_packed = Array(dplan.operators)
            dops = [dops_packed[:, :, k] for k in 1:size(dops_packed, 3)]
            @test all(o -> all(isfinite, o), dops)
            D = dplan.ndof
            wsref = FastMultipole.DenseM2LBuilderWorkspace(TF,
                dev_dcache.state.invariant_cache.basis_info,
                dev_dcache.state.invariant_cache, D)
            Kref = Matrix{TF}(undef, D, D)
            cell_width = 2 * dev_dcache.h0 / (1 << dev_dcache.ell)
            for (k, offset) in enumerate(dev_dcache.accepted_offsets)
                delta = SVector{3,TF}(offset) * TF(cell_width)
                r, theta, phi = FastMultipole.cartesian_to_spherical(delta)
                FastMultipole.build_dense_m2l_operator!(Kref, r, theta, phi,
                    dev_dcache.state.invariant_cache, wsref, Val(LH))
                @test all(isequal.(dops[k], Kref))
                @test all(isequal.(dops[k], hplan.operators[k]))
            end

            # class-contiguity invariant + histogram/prefix consistency
            nrf = dev_dcache.state.counts.n_routes
            dclass = Array(dplan.route_class)[1:nrf]
            @test issorted(dclass)
            @test dplan.class_starts[end] - 1 == nrf
            counts_from_class = zeros(Int, dplan.nclasses)
            for c in dclass
                counts_from_class[c] += 1
            end
            @test counts_from_class == Int.(dplan.host_class_counts)
            if TF === Float64
                @test nrf == host_dcache.state.counts.n_routes
                @test dclass == hplan.route_class[1:nrf]
                @test counts_from_class == hplan.class_counts
            end

            # M2L output-buffer parity vs the approved 023e host dense strategy,
            # before L2L/L2B can mask errors: per-class reference then whole-pass
            FastMultipole._launch_host_b2m!(host_dcache.state)
            FastMultipole._launch_host_m2m!(host_dcache.state)
            FastMultipole._launch_host_m2l!(host_dcache.state)
            FastMultipole._launch_cuda_b2m!(dev_dcache.state)
            FastMultipole._launch_cuda_resident_m2m!(dev_dcache.state)
            m2l_rtol = TF === Float64 ? 1e-9 : 5f-3
            m2l_atol = TF === Float64 ? 1e-10 : 5f-4
            nnodes = host_dcache.state.counts.n_nodes
            # force the GEMM drivers here (fused is the production default): per-class
            # reference, then whole-pass, then the fused default itself
            saved_wp_stage = FastMultipole.DENSE_CUDA_WHOLE_PASS[]
            saved_fused_stage = FastMultipole.DENSE_CUDA_FUSED[]
            try
                FastMultipole.DENSE_CUDA_FUSED[] = false
                FastMultipole.DENSE_CUDA_WHOLE_PASS[] = false
                FastMultipole._launch_cuda_resident_m2l!(dev_dcache.state)
                CUDA.synchronize()
                @test Array(dev_dcache.state.locals.phi)[:, 1:nnodes] ≈
                    host_dcache.state.locals.phi[:, 1:nnodes] rtol=m2l_rtol atol=m2l_atol
                LH && @test Array(dev_dcache.state.locals.chi)[:, 1:nnodes] ≈
                    host_dcache.state.locals.chi[:, 1:nnodes] rtol=m2l_rtol atol=m2l_atol
                FastMultipole.DENSE_CUDA_WHOLE_PASS[] = true
                FastMultipole._launch_cuda_resident_m2l!(dev_dcache.state)   # whole-pass
                CUDA.synchronize()
                @test Array(dev_dcache.state.locals.phi)[:, 1:nnodes] ≈
                    host_dcache.state.locals.phi[:, 1:nnodes] rtol=m2l_rtol atol=m2l_atol
                LH && @test Array(dev_dcache.state.locals.chi)[:, 1:nnodes] ≈
                    host_dcache.state.locals.chi[:, 1:nnodes] rtol=m2l_rtol atol=m2l_atol
            finally
                FastMultipole.DENSE_CUDA_WHOLE_PASS[] = saved_wp_stage
                FastMultipole.DENSE_CUDA_FUSED[] = saved_fused_stage
            end
            FastMultipole._launch_cuda_resident_m2l!(dev_dcache.state)   # production default (fused)
            CUDA.synchronize()
            @test Array(dev_dcache.state.locals.phi)[:, 1:nnodes] ≈
                host_dcache.state.locals.phi[:, 1:nnodes] rtol=m2l_rtol atol=m2l_atol
            LH && @test Array(dev_dcache.state.locals.chi)[:, 1:nnodes] ≈
                host_dcache.state.locals.chi[:, 1:nnodes] rtol=m2l_rtol atol=m2l_atol

            # end-to-end: GPU-dense vs host-dense (023e oracle) and vs the CUDA
            # concat / factored / precomputed-y resident paths
            fmm!(dev_sys, dev_dcache; scalar_potential=!LH, gradient=true)
            fmm!(host_sys, host_dcache; scalar_potential=!LH, gradient=true)
            p_tol = TF === Float64 ? 1e-9 : 1f-3
            g_tol = TF === Float64 ? 1e-8 : 1f-2
            !LH && @test maximum(abs.(dev_sys.potential[1, :] .-
                host_sys.potential[1, :])) < p_tol
            @test maximum(abs.(dev_sys.potential[5:7, :] .-
                host_sys.potential[5:7, :])) < g_tol
            if TF === Float64
                xconcat_sys = generate_gravitational(fseed, nf)
                xconcat_cache = RadixFMMCache(xconcat_sys; stencil_epsilon=1e-4, expansion_order=P, ell=3,
                    bounds=bounds, lamb_helmholtz=LH, device=true,
                    options=mk_concat_opts(TF))
                fmm!(xconcat_sys, xconcat_cache; scalar_potential=!LH, gradient=true)
                !LH && @test maximum(abs.(dev_sys.potential[1, :] .-
                    xconcat_sys.potential[1, :])) < p_tol
                @test maximum(abs.(dev_sys.potential[5:7, :] .-
                    xconcat_sys.potential[5:7, :])) < g_tol
                xprey_sys = generate_gravitational(fseed, nf)
                xprey_cache = RadixFMMCache(xprey_sys; stencil_epsilon=1e-4, expansion_order=P, ell=3,
                    bounds=bounds, lamb_helmholtz=LH, device=true,
                    options=mk_prey_opts(TF))
                fmm!(xprey_sys, xprey_cache; scalar_potential=!LH, gradient=true)
                !LH && @test maximum(abs.(dev_sys.potential[1, :] .-
                    xprey_sys.potential[1, :])) < p_tol
                @test maximum(abs.(dev_sys.potential[5:7, :] .-
                    xprey_sys.potential[5:7, :])) < g_tol
            end

            # absolute accuracy vs direct! at the stencil design order
            if P == 8
                ref_d = generate_gravitational(fseed, nf)
                FastMultipole.direct!(ref_d; scalar_potential=true, gradient=true)
                direct_p_tol = TF === Float64 ? 1e-4 : 1f-3
                direct_g_tol = TF === Float64 ? 1e-2 : 1f-1
                !LH && @test maximum(abs.(dev_sys.potential[1, :] .-
                    ref_d.potential[1, :])) < direct_p_tol
                @test maximum(abs.(dev_sys.potential[5:7, :] .-
                    ref_d.potential[5:7, :])) < direct_g_tol
            end

            # production default (fused per-route kernel) vs the per-class and
            # whole-pass GEMM drivers, end to end: same per-column arithmetic
            # through the packed operators, differing only in launch shape /
            # atomic accumulation order; the GEMM drivers must also stay
            # allocation-free at steady state
            whole_pot = copy(dev_sys.potential)   # production default (fused) result
            wp_tol = TF === Float64 ? 1e-10 : 1f-4
            saved_wp = FastMultipole.DENSE_CUDA_WHOLE_PASS[]
            saved_fused = FastMultipole.DENSE_CUDA_FUSED[]
            # DENSE_CUDA_WHOLE_PASS is construction-locked (047 contract: a
            # late flip on an existing cache errors loudly), so each GEMM
            # driver gets its own cache built under its setting;
            # DENSE_CUDA_FUSED is a runtime setting and flips in place
            try
                FastMultipole.DENSE_CUDA_FUSED[] = false
                FastMultipole.DENSE_CUDA_WHOLE_PASS[] = false
                pc_sys = generate_gravitational(fseed, nf)
                pc_cache = RadixFMMCache(pc_sys; stencil_epsilon=1e-4, expansion_order=P, ell=3,
                    bounds=bounds, lamb_helmholtz=LH, device=true, options=dense_opts)
                fmm!(pc_sys, pc_cache; scalar_potential=!LH, gradient=true)
                !LH && @test maximum(abs.(pc_sys.potential[1, :] .-
                    whole_pot[1, :])) < wp_tol
                @test maximum(abs.(pc_sys.potential[5:7, :] .-
                    whole_pot[5:7, :])) < 10 * wp_tol
                FastMultipole.DENSE_CUDA_WHOLE_PASS[] = true
                wp_sys = generate_gravitational(fseed, nf)
                wp_cache = RadixFMMCache(wp_sys; stencil_epsilon=1e-4, expansion_order=P, ell=3,
                    bounds=bounds, lamb_helmholtz=LH, device=true, options=dense_opts)
                fmm!(wp_sys, wp_cache; scalar_potential=!LH, gradient=true)
                !LH && @test maximum(abs.(wp_sys.potential[1, :] .-
                    whole_pot[1, :])) < wp_tol
                @test maximum(abs.(wp_sys.potential[5:7, :] .-
                    whole_pot[5:7, :])) < 10 * wp_tol
                FastMultipole._launch_resident_m2l!(wp_cache.state)
                gemm_alloc = @eval CUDA.@allocated FastMultipole._launch_resident_m2l!(
                    $(wp_cache).state)
                @test gemm_alloc == 0
            finally
                FastMultipole.DENSE_CUDA_WHOLE_PASS[] = saved_wp
                FastMultipole.DENSE_CUDA_FUSED[] = saved_fused
            end
            fmm!(dev_sys, dev_dcache; scalar_potential=!LH, gradient=true)

            # steady-state device allocation on the production default (fused):
            # the dense M2L stage runs entirely on preallocated plan storage
            FastMultipole._launch_resident_m2l!(dev_dcache.state)
            dev_alloc = @eval CUDA.@allocated FastMultipole._launch_resident_m2l!($(dev_dcache).state)
            @test dev_alloc == 0
        end

        #--- dense tiny-chunk stress: multi-chunk whole pass and per-class
        #    sub-chunking both reduce to the same per-column arithmetic ---#

        saved_dchunk = FastMultipole.DENSE_CUDA_CHUNK[]
        try
            FastMultipole.DENSE_CUDA_CHUNK[] = 5
            dtc_sys = generate_gravitational(seed + 73, 400)
            dtc_cache = RadixFMMCache(dtc_sys; stencil_epsilon=1e-4, expansion_order=4, ell=3, bounds=bounds,
                device=true, options=mk_dense_opts(Float64))
            @test (dtc_cache.state.scratch.m2l_concat.whole_pass[]::NamedTuple).chunk == 5
            # chunking only exists on the GEMM drivers; force fused off so the
            # multi-chunk whole pass and per-class sub-chunking are exercised
            saved_dtc_wp = FastMultipole.DENSE_CUDA_WHOLE_PASS[]
            saved_dtc_fused = FastMultipole.DENSE_CUDA_FUSED[]
            local dtc_pot, dtc2_sys
            try
                FastMultipole.DENSE_CUDA_FUSED[] = false
                fmm!(dtc_sys, dtc_cache; scalar_potential=true, gradient=true)
                dtc_pot = copy(dtc_sys.potential)
                # DENSE_CUDA_WHOLE_PASS is construction-locked: the per-class
                # sub-chunking run needs its own cache built under wp=false
                FastMultipole.DENSE_CUDA_WHOLE_PASS[] = false
                dtc2_sys = generate_gravitational(seed + 73, 400)
                dtc2_cache = RadixFMMCache(dtc2_sys; stencil_epsilon=1e-4, expansion_order=4,
                    ell=3, bounds=bounds, device=true, options=mk_dense_opts(Float64))
                fmm!(dtc2_sys, dtc2_cache; scalar_potential=true, gradient=true)
            finally
                FastMultipole.DENSE_CUDA_WHOLE_PASS[] = saved_dtc_wp
                FastMultipole.DENSE_CUDA_FUSED[] = saved_dtc_fused
            end
            @test maximum(abs.(dtc2_sys.potential[1, :] .- dtc_pot[1, :])) < 1e-10
            dtc_host_sys = generate_gravitational(seed + 73, 400)
            dtc_host_cache = RadixFMMCache(dtc_host_sys; stencil_epsilon=1e-4, expansion_order=4, ell=3,
                bounds=bounds, options=mk_dense_opts(Float64))
            fmm!(dtc_host_sys, dtc_host_cache; scalar_potential=true, gradient=true)
            @test maximum(abs.(dtc_sys.potential[1, :] .- dtc_host_sys.potential[1, :])) < 1e-9
        finally
            FastMultipole.DENSE_CUDA_CHUNK[] = saved_dchunk
        end

        #--- dense stepping loop: transfer-counter contract + refresh ---#

        dstep_sys = generate_gravitational(seed + 77, 1500)
        dstep_host_sys = generate_gravitational(seed + 77, 1500)
        dstep_cache = RadixFMMCache(dstep_sys; stencil_epsilon=1e-4, expansion_order=6, ell=3, bounds=bounds,
            device=true, options=mk_dense_opts(Float64))
        dstep_host_cache = RadixFMMCache(dstep_host_sys; stencil_epsilon=1e-4, expansion_order=6, ell=3,
            bounds=bounds, options=mk_dense_opts(Float64))
        dcounters = dstep_cache.state.counters
        dbase_route = dcounters.route_uploads
        dbase_operator = dcounters.operator_uploads
        dbase_body = dcounters.body_uploads
        dbase_metadata = dcounters.metadata_downloads
        dcaptured_state = dstep_cache.state
        ddplan = dstep_cache.state.scratch.m2l_concat
        dcaptured_arrays = (ddplan.route_class, ddplan.class_counts, ddplan.operators,
            ddplan.src_slab, dcaptured_state.route_targets, dcaptured_state.output)
        drng = MersenneTwister(seed + 77)
        for step in 1:3
            for i in eachindex(dstep_sys.bodies)
                b = dstep_sys.bodies[i]
                pos = clamp.(b.position .+ 0.02 .* (rand(drng, SVector{3,Float64}) .- 0.5),
                    -0.05, 1.05)
                dstep_sys.bodies[i] = Body(pos, b.radius, b.strength)
                hb = dstep_host_sys.bodies[i]
                dstep_host_sys.bodies[i] = Body(pos, hb.radius, hb.strength)
            end
            fmm!(dstep_sys, dstep_cache; scalar_potential=true, gradient=true)
            fmm!(dstep_host_sys, dstep_host_cache; scalar_potential=true, gradient=true)
            @test dstep_cache.state === dcaptured_state
            @test all(current === original for (current, original) in zip((
                ddplan.route_class, ddplan.class_counts, ddplan.operators,
                ddplan.src_slab, dcaptured_state.route_targets, dcaptured_state.output,
            ), dcaptured_arrays))
            @test dcounters.route_uploads == dbase_route
            @test dcounters.operator_uploads == dbase_operator
            @test dcounters.body_uploads == dbase_body + step
            @test dcounters.metadata_downloads == dbase_metadata + 3 * step
            @test dcounters.expansion_host_copies == 0
            nr_step = dstep_cache.state.counts.n_routes
            @test ddplan.class_starts[end] - 1 == nr_step
            @test sum(Int, ddplan.host_class_counts) == nr_step
            hstep = dstep_host_cache.state
            hstep_plan = hstep.scratch.m2l_concat
            @test Array(ddplan.route_class)[1:nr_step] == hstep_plan.route_class[1:nr_step]
            @test Int.(ddplan.host_class_counts) == hstep_plan.class_counts
            @test Array(dstep_cache.state.route_sources)[1:nr_step] ==
                hstep.route_sources[1:nr_step]
            ref_step = Gravitational(copy(dstep_sys.bodies), zeros(16, length(dstep_sys.bodies)))
            FastMultipole.direct!(ref_step; scalar_potential=true, gradient=true)
            @test maximum(abs.(dstep_sys.potential[1, :] .- ref_step.potential[1, :])) < 1e-4
            @test maximum(abs.(dstep_sys.potential[1, :] .-
                dstep_host_sys.potential[1, :])) < 1e-9
        end

        #--- dense empty-route degenerate case: all-nearfield cluster ---#

        dcl_sys = generate_gravitational(seed + 79, 60)
        for i in eachindex(dcl_sys.bodies)
            b = dcl_sys.bodies[i]
            dcl_sys.bodies[i] = Body(SVector(0.5, 0.5, 0.5) .+ 0.01 .* (b.position .- 0.5),
                b.radius, b.strength)
        end
        dcl_cache = RadixFMMCache(dcl_sys; stencil_epsilon=1e-4, expansion_order=4, ell=3, bounds=bounds,
            device=true, options=mk_dense_opts(Float64))
        fmm!(dcl_sys, dcl_cache; scalar_potential=true, gradient=true)
        dcl_plan = dcl_cache.state.scratch.m2l_concat
        @test dcl_cache.state.counts.n_routes == dcl_plan.class_starts[end] - 1
        dcl_ref = Gravitational(copy(dcl_sys.bodies), zeros(16, length(dcl_sys.bodies)))
        FastMultipole.direct!(dcl_ref; scalar_potential=true, gradient=true)
        @test maximum(abs.(dcl_sys.potential[1, :] .- dcl_ref.potential[1, :])) < 1e-6

        #--- dense device-memory gates: persistent payload and free-memory headroom ---#

        # a 1-byte persistent cap rejects before any large device allocation
        low_persist = try
            RadixFMMCache(generate_gravitational(seed + 81, 200); stencil_epsilon=1e-4, expansion_order=4,
                ell=3, bounds=bounds, device=true,
                options=mk_dense_opts(Float64; max_persistent_bytes=1))
            nothing
        catch err
            err
        end
        @test low_persist isa ArgumentError
        @test occursin("max_persistent_bytes", sprint(showerror, low_persist))
        @test occursin("dense operators", sprint(showerror, low_persist))

        # an absurd headroom (all of device memory) rejects on the free-memory gate
        big_headroom = try
            RadixFMMCache(generate_gravitational(seed + 82, 200); stencil_epsilon=1e-4, expansion_order=4,
                ell=3, bounds=bounds, device=true,
                options=mk_dense_opts(Float64;
                    cuda_headroom_bytes=Int(CUDA.total_memory())))
            nothing
        catch err
            err
        end
        @test big_headroom isa ArgumentError
        @test occursin("free-memory", sprint(showerror, big_headroom))
        @test occursin("headroom", sprint(showerror, big_headroom))
    end
end
