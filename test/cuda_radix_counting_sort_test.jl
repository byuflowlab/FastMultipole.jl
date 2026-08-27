using FastMultipole
using FastMultipole.StaticArrays
using Test

include("gravitational.jl")

FastMultipole.load_cuda_radix_lifecycle!() ||
    error("CUDA radix lifecycle failed to load: $(FastMultipole.cuda_radix_status())")
using CUDA

const CSFM = FastMultipole

@testset "bounded Morton counting sort (task 028 stage 6)" begin
    n = 4000
    ell = 5
    q = 6
    bounds = (SVector(-0.1, -0.1, -0.1), 1.2)
    opts = CUDARadixLifecycleOptions(; precision=Float32,
        m2l_strategy=DenseTranslationM2L())
    epsq = rigid_stencil_epsilon(3, 0.6, ell, q; TF=Float32)
    policy = HierarchicalRigidStencil(3, epsq; near_radius2=q,
        window_classes=length(RigidHierarchicalTables(q).push_offsets))

    old = CSFM.RADIX_CUDA_COUNTING_SORT[]
    try
        CSFM.RADIX_CUDA_COUNTING_SORT[] = false
        a = generate_gravitational(28060, n)
        ca = RadixFMMCache(a; expansion_order=3, ell, bounds, policy,
            options=opts, device=true)
        fmm!(a, ca; scalar_potential=true, gradient=true)

        CSFM.RADIX_CUDA_COUNTING_SORT[] = true
        b = generate_gravitational(28060, n)
        cb = RadixFMMCache(b; expansion_order=3, ell, bounds, policy,
            options=opts, device=true)
        fmm!(b, cb; scalar_potential=true, gradient=true)

        grid = cb.state.grid
        ctx = cb.device_ctx
        p = Array(view(grid.perm, 1:n))
        ip = Array(view(grid.invperm, 1:n))
        keys = Array(view(ctx.sorted_keys, 1:n))
        @test sort(p) == collect(1:n)
        @test all(ip[p[i]] == i for i in 1:n)
        @test issorted(keys)
        @test length(ctx.counting_histogram) == 1 << (3ell)
        @test Array(view(grid.cell_keys, 1:grid.n_cells)) ==
            Array(view(ca.state.grid.cell_keys, 1:ca.state.grid.n_cells))
        @test Array(view(grid.cell_ranges, :, 1:grid.n_cells)) ==
            Array(view(ca.state.grid.cell_ranges, :, 1:ca.state.grid.n_cells))
        @test b.potential[1, :] ≈ a.potential[1, :] rtol=2e-4 atol=2e-5
        @test b.potential[5:7, :] ≈ a.potential[5:7, :] rtol=2e-3 atol=2e-4

        ids = (objectid(ctx.counting_histogram), objectid(ctx.counting_prefix),
            objectid(ctx.counting_cursor), objectid(grid.perm),
            objectid(grid.invperm))
        for _ in 1:2
            fmm!(b, cb; scalar_potential=true, gradient=true)
            p = Array(view(grid.perm, 1:n))
            ip = Array(view(grid.invperm, 1:n))
            @test sort(p) == collect(1:n)
            @test all(ip[p[i]] == i for i in 1:n)
            @test issorted(Array(view(ctx.sorted_keys, 1:n)))
        end
        @test ids == (objectid(ctx.counting_histogram),
            objectid(ctx.counting_prefix), objectid(ctx.counting_cursor),
            objectid(grid.perm), objectid(grid.invperm))

        # `ca` was built with the knob off, so its histogram spans no key domain.
        # 047/048 lock contract: enabling the knob after construction is a loud
        # error (pre-lock this silently fell back to the comparison sort);
        # running `ca` requires restoring the value it was built with.
        @test length(ca.device_ctx.counting_histogram) == 1
        @test !CSFM._cuda_counting_sort_ready(ca.device_ctx, ell)
        a.potential .= 0
        b.potential .= 0
        @test_throws r"construction-locked" fmm!(a, ca;
            scalar_potential=true, gradient=true)
        CSFM.RADIX_CUDA_COUNTING_SORT[] = false
        fmm!(a, ca; scalar_potential=true, gradient=true)   # comparison sort
        CSFM.RADIX_CUDA_COUNTING_SORT[] = true
        fmm!(b, cb; scalar_potential=true, gradient=true)   # counting sort
        @test sort(Array(view(ca.state.grid.perm, 1:n))) == collect(1:n)
        @test issorted(Array(view(ca.device_ctx.sorted_keys, 1:n)))
        @test a.potential[5:7, :] ≈ b.potential[5:7, :] rtol=2e-3 atol=2e-4
    finally
        CSFM.RADIX_CUDA_COUNTING_SORT[] = old
    end
end
