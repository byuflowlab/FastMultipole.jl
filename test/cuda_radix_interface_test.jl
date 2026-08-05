# Task 032 stage 1, CUDA mirrors: generalized packed layout, Point{Vortex}
# device B2M (φ + χ), and the 13-row hessian output on the device-resident
# lifecycle, validated against the host-resident results (which
# device_system_interface_test.jl validates against direct references).

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

_cuda_interface_required() = get(ENV, "FASTMULTIPOLE_REQUIRE_CUDA_TESTS", "0") == "1"

@testset "CUDA device-system interface (task 032)" begin
    loaded = FastMultipole.load_cuda_radix_lifecycle!()
    if !loaded
        if _cuda_interface_required()
            error(
                "FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 but CUDA radix lifecycle did not load: " *
                FastMultipole.cuda_radix_status(),
            )
        end
        sys = generate_vortex(1, 50)
        @test_throws Exception RadixFMMCache(sys; expansion_order=4, ell=2,
            hessian=true, device=true)
    else
        @eval using CUDA
        seed = 20260805
        opts64 = CUDARadixLifecycleOptions(; precision=Float64,
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L())

        #--- (1) scalar hessian: device vs host parity + P=4 coverage ---#

        for P in (4, 8)
            host_sys = generate_gravitational(seed, 2000)
            dev_sys = generate_gravitational(seed, 2000)
            hc = RadixFMMCache(host_sys; expansion_order=P, ell=3, hessian=true,
                options=opts64)
            dc = RadixFMMCache(dev_sys; expansion_order=P, ell=3, hessian=true,
                options=opts64, device=true)
            fmm!(host_sys, hc; scalar_potential=true, gradient=true, hessian=true)
            fmm!(dev_sys, dc; scalar_potential=true, gradient=true, hessian=true)
            # same geometry/order: differences are pair-summation reassociation only
            @test maximum(abs.(dev_sys.potential[1, :] .- host_sys.potential[1, :])) < 1e-10
            @test maximum(abs.(dev_sys.potential[5:7, :] .- host_sys.potential[5:7, :])) < 1e-9
            @test maximum(abs.(dev_sys.potential[8:16, :] .- host_sys.potential[8:16, :])) < 1e-7
        end

        #--- (2) vortex + Lamb-Helmholtz + hessian: device vs host parity ---#

        for (TF, gtol, htol) in ((Float64, 1e-9, 1e-7), (Float32, 2f-4, 2f-1))
            opts = CUDARadixLifecycleOptions(; precision=TF,
                m2l_strategy=FastMultipole.ConcatenatedFixedZM2L())
            for P in (4, 8)
                host_sys = generate_vortex(seed, 1500)
                dev_sys = generate_vortex(seed, 1500)
                hc = RadixFMMCache(host_sys; expansion_order=P, ell=3, hessian=true,
                    options=opts)
                dc = RadixFMMCache(dev_sys; expansion_order=P, ell=3, hessian=true,
                    options=opts, device=true)
                @test dc.state.options.body_type === Point{Vortex}
                fmm!(host_sys, hc; scalar_potential=false, gradient=true, hessian=true)
                fmm!(dev_sys, dc; scalar_potential=false, gradient=true, hessian=true)
                @test maximum(abs.(dev_sys.gradient_stretching[1:3, :] .-
                    host_sys.gradient_stretching[1:3, :])) < gtol
                @test maximum(abs.(dev_sys.potential[5:13, :] .-
                    host_sys.potential[5:13, :])) < htol
                @test any(!iszero, Array(dc.state.multipoles.chi))
            end
        end

        #--- (3) packed-layout round trip on device (data_per_body = 9) ---#

        ext = ExtendedVortex(generate_vortex(seed, 400; radius_factor=0.1))
        ext_host = ExtendedVortex(generate_vortex(seed, 400; radius_factor=0.1))
        ec = RadixFMMCache(ext; expansion_order=8, ell=3, hessian=true,
            options=opts64, device=true)
        ehc = RadixFMMCache(ext_host; expansion_order=8, ell=3, hessian=true,
            options=opts64)
        fmm!(ext, ec; scalar_potential=false, gradient=true, hessian=true)
        fmm!(ext_host, ehc; scalar_potential=false, gradient=true, hessian=true)
        @test maximum(abs.(ext.inner.gradient_stretching[1:3, :] .-
            ext_host.inner.gradient_stretching[1:3, :])) < 1e-9
        st = ec.state
        @test size(st.source_bodies, 1) == 9
        packed = Array(st.source_bodies)
        perm = Array(st.body_perm)
        idx = Array(st.body_indices)
        for sorted_i in 1:st.counts.n_bodies
            ibody = idx[perm[sorted_i]]
            @test packed[4, sorted_i] == ext.inner.bodies[ibody].sigma
            @test packed[8, sorted_i] == 10.0 + ibody
            @test packed[9, sorted_i] == -Float64(ibody)
        end

        #--- (4) counter contract across recurring steps ---#

        counters = ec.state.counters
        @test counters.expansion_host_copies == 0
        route_uploads0 = counters.route_uploads
        operator_uploads0 = counters.operator_uploads
        body_uploads0 = counters.body_uploads
        fmm!(ext, ec; scalar_potential=false, gradient=true, hessian=true)
        @test counters.expansion_host_copies == 0
        @test counters.route_uploads == route_uploads0
        @test counters.operator_uploads == operator_uploads0
        # one host-resident source system: exactly one upload per step
        @test counters.body_uploads == body_uploads0 + 1

        #--- (5) hessian=false device cache keeps the 4-row output ---#

        plain = generate_gravitational(seed, 500)
        pc = RadixFMMCache(plain; expansion_order=4, ell=2, options=opts64,
            device=true)
        @test size(pc.state.output, 1) == 4
        @test_throws ArgumentError fmm!(plain, pc; hessian=true)
    end
end
