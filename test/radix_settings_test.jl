# task 047: consolidated radix settings surface + construction-lock contract.
# Host-side regression tests of the mechanism; the device wiring
# (verify at _radix_cache_device_step!) is exercised by the cluster jobs.

using Test
using FastMultipole
using FastMultipole: RADIX_SETTING_SPECS, CUDA_NEARFIELD_GH_MODE

if !isdefined(Main, :Gravitational)
    include("gravitational.jl")
end

@testset "radix settings surface (047)" begin

    @testset "registry + accessors" begin
        # every spec has a lock class and a doc
        for (name, spec) in RADIX_SETTING_SPECS
            @test spec.lock in (:construction, :runtime)
            @test !isempty(spec.doc)
        end
        # eagerly-defined settings are readable
        @test radix_setting(:CUDA_NEARFIELD_GH_MODE) isa Symbol
        @test radix_setting(:FACTORED_Y_GEMM_MIN_COLS) isa Integer
        @test radix_setting_lock(:CUDA_NEARFIELD_GH_MODE) === :construction
        @test radix_setting_lock(:FACTORED_Y_GEMM_MIN_COLS) === :runtime
        # radix_settings() reports defined settings only, as a NamedTuple
        settings = radix_settings()
        @test settings isa NamedTuple
        @test haskey(settings, :CUDA_NEARFIELD_GH_MODE)
        # unknown names throw
        @test_throws ArgumentError radix_setting(:NOT_A_SETTING)
        @test_throws ArgumentError set_radix_setting!(:NOT_A_SETTING, 1)
        @test_throws ArgumentError radix_setting_lock(:NOT_A_SETTING)
    end

    @testset "validation" begin
        old = radix_setting(:CUDA_NEARFIELD_GH_MODE)
        try
            # valid enum value round-trips
            @test set_radix_setting!(:CUDA_NEARFIELD_GH_MODE, :shipped) === :shipped
            @test radix_setting(:CUDA_NEARFIELD_GH_MODE) === :shipped
            # invalid enum + wrong types rejected without writing
            @test_throws ArgumentError set_radix_setting!(:CUDA_NEARFIELD_GH_MODE, :bogus)
            @test_throws ArgumentError set_radix_setting!(:CUDA_NEARFIELD_GH_MODE, 3)
            @test radix_setting(:CUDA_NEARFIELD_GH_MODE) === :shipped
            @test_throws ArgumentError set_radix_setting!(:FACTORED_Y_GEMM_MIN_COLS, -1)
            @test_throws ArgumentError set_radix_setting!(:FACTORED_Y_GEMM_MIN_COLS, 1.5)
            @test_throws ArgumentError set_radix_setting!(:FACTORED_Y_GEMM_MIN_COLS, true)
        finally
            set_radix_setting!(:CUDA_NEARFIELD_GH_MODE, old)
        end
        if !FastMultipole.cuda_radix_available() &&
                !isdefined(FastMultipole, :CUDA_NEARFIELD_BINNING)
            # CUDA-only settings are named but unreachable before lifecycle load
            @test_throws ArgumentError radix_setting(:CUDA_NEARFIELD_BINNING)
            @test_throws ArgumentError set_radix_setting!(:CUDA_NEARFIELD_BINNING, :unbinned)
        end
    end

    @testset "atomic batch + CUDA thread validation" begin
        old_gh = radix_setting(:CUDA_NEARFIELD_GH_MODE)
        old_cols = radix_setting(:FACTORED_Y_GEMM_MIN_COLS)
        @test_throws ArgumentError set_radix_settings!((;
            CUDA_NEARFIELD_GH_MODE=:shipped,
            FACTORED_Y_GEMM_MIN_COLS=-1,
        ))
        @test radix_setting(:CUDA_NEARFIELD_GH_MODE) === old_gh
        @test radix_setting(:FACTORED_Y_GEMM_MIN_COLS) == old_cols
        thread_validator = RADIX_SETTING_SPECS[:DENSE_CUDA_TILED_THREADS].validate
        @test thread_validator(128) === nothing
        @test_throws ArgumentError thread_validator(true)
        @test_throws ArgumentError thread_validator(Int32(64))
        @test_throws ArgumentError thread_validator(33)
        @test_throws ArgumentError thread_validator(2048)
        if isdefined(FastMultipole, :DENSE_CUDA_TILED_THREADS)
            old_threads = radix_setting(:DENSE_CUDA_TILED_THREADS)
            try
                @test set_radix_setting!(:DENSE_CUDA_TILED_THREADS, 128) == 128
            finally
                set_radix_setting!(:DENSE_CUDA_TILED_THREADS, old_threads)
            end
        end
    end

    @testset "construction-lock snapshot + drift detection" begin
        snapshot = snapshot_locked_radix_settings()
        @test snapshot isa Vector{Pair{Symbol,Any}}
        # only construction-locked settings are snapshotted
        @test all(FastMultipole.radix_setting_lock(name) === :construction
                  for (name, _) in snapshot)
        @test any(name === :CUDA_NEARFIELD_GH_MODE for (name, _) in snapshot)
        # unchanged snapshot verifies clean; nothing (legacy caches) verifies clean
        @test verify_locked_radix_settings(snapshot) === nothing
        @test verify_locked_radix_settings(nothing) === nothing
        # a post-snapshot flip of a locked setting errors loudly...
        old = CUDA_NEARFIELD_GH_MODE[]
        try
            CUDA_NEARFIELD_GH_MODE[] = old === :shipped ? :fp32 : :shipped
            err = try
                verify_locked_radix_settings(snapshot)
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("construction-locked", err.msg)
            @test occursin("CUDA_NEARFIELD_GH_MODE", err.msg)
        finally
            CUDA_NEARFIELD_GH_MODE[] = old
        end
        # ...and restoring the value clears the error
        @test verify_locked_radix_settings(snapshot) === nothing
    end

    @testset "cache carries the snapshot (host construction, P=4)" begin
        n = 200
        bodies = rand(8, n)
        bodies[1:3, :] .= bodies[1:3, :] .* 0.5 .+ 0.25
        sys = Gravitational(bodies)
        cache = RadixFMMCache(sys; expansion_order=4, ell=2)
        @test cache.locked_settings isa Vector{Pair{Symbol,Any}}
        @test verify_locked_radix_settings(cache.locked_settings) === nothing

        switches = (DerivativesSwitch(false, true, false, sys),)
        source_tagged = Tree((sys,), SourceTree(), switches; leaf_size=SVector{1}(32))
        target_tagged = Tree((sys,), TargetTree(), switches; leaf_size=SVector{1}(32))
        @test source_tagged isa Tree
        @test target_tagged isa Tree
        @test Tree(sys, SourceTree(), switches[1]; leaf_size=SVector{1}(32)) isa Tree
        @test Tree(sys, TargetTree(), switches[1]; leaf_size=SVector{1}(32)) isa Tree
        @test FastMultipole._device_nearfield(HostNearfield()) === false
        @test FastMultipole._device_nearfield(DeviceNearfield()) === true
    end

end
