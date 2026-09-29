# Device regression gates for defects fixed in the KernelAbstractions extension.
# Each block would fail on the unfixed extension; the CPU-backend counterparts
# live in test/ka_cpu_regression_test.jl.
#
#   1. all-direct demotion on a device cache (sigma too large for any stencil)
#      runs end to end and matches the regularized direct sum
#   2. the device RegularizedVortex regularizes pairs beyond rho_t, as the host
#      functor does, instead of going singular there
#   3. direct_rectangular! reuses its compiled kernel
#   4. ka_launch_nearfield! resolves workgroup=0 and refuses a workgroup that
#      is not a whole number of lane teams
#   5. ka_launch_l2b! resolves workgroup=0 instead of launching nothing
#   6. checked keys raise ArgumentError for NaN/huge positions on the device
#   7. ka_extra_tree_finish! passes its workgroup to the near sweep
include("ka_backend.jl")
using FastMultipole, Random, Test, Printf, LinearAlgebra
using FastMultipole.StaticArrays
const FM = FastMultipole
include(joinpath(@__DIR__, "..", "vortex.jl"))
include(joinpath(@__DIR__, "..", "gravitational.jl"))
include(joinpath(@__DIR__, "..", "interface_test_systems.jl"))
include(joinpath(@__DIR__, "ka_extra_tree_correctness_systems.jl"))

if !dev_functional()
    println("$(DEV_NAME) not functional; skipping")
    exit(0)
end
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext !== nothing || error("FastMultipoleKAExt did not load")

const DTF = DEV_NAME == "Metal" ? Float32 : Float64
npass = Ref(0); nfail = Ref(0)
check(ok, msg) = (ok ? (npass[] += 1) : (nfail[] += 1); println(ok ? "  PASS  $msg" : "  FAIL  $msg"))
relerr(a, b) = (d = maximum(abs.(Array(a) .- Array(b))); s = maximum(abs.(Array(b)));
                s == 0 ? d : d / s)
function guarded(f, name)
    try
        f()
    catch err
        nfail[] += 1
        println("  FAIL  $name threw ", sprint(showerror, err)[1:min(end, 600)])
    end
end

FM.device_backend(::SmoothedVortex) = DEV_BACKEND
FM.device_backend(::VortexParticles) = DEV_BACKEND

vortex_opts(TF) = FM.RadixLifecycleOptions(; precision=TF,
    m2l_strategy=FM.ConcatenatedFixedZM2L(), body_type=FM.Point{FM.Vortex})

println("1. all-direct demotion on a device cache")
guarded("demotion") do
    n = 400
    Random.seed!(31)
    pos = rand(3, n); str = randn(3, n) ./ n
    mk(TF) = SmoothedVortex(VortexParticles(TF.(pos), TF.(str), zeros(TF, n);
        potential=zeros(TF, 13, n), gradient_stretching=zeros(TF, 6, n)), fill(TF(0.2), n))
    sys = mk(DTF)
    cache = RadixFMMCache(sys; expansion_order=4, ell=3, window_classes=64,
        hessian=true, device=true, options=vortex_opts(DTF))
    fmm!(sys, cache; scalar_potential=false, gradient=true, hessian=true)
    check(cache.ell == 2 && isempty(cache.accepted_offsets),
        "cache demoted to the zero-M2L geometry (ell=$(cache.ell))")
    U, J = _interface_regularized_direct(mk(Float64))
    eu = relerr(sys.inner.gradient_stretching[1:3, :], U)
    ej = relerr(sys.inner.potential[5:13, :], J)
    # all pairs direct: bounded by the erf-free g/h evaluation, as on the host
    check(eu < 5e-4 && ej < 5e-4, @sprintf("demoted device result vs regularized direct sum (U %.2e, J %.2e)", eu, ej))
    # a second step on the demoted cache (epoch fast path)
    fill!(sys.inner.gradient_stretching, 0); fill!(sys.inner.potential, 0)
    fmm!(sys, cache; scalar_potential=false, gradient=true, hessian=true)
    check(relerr(sys.inner.gradient_stretching[1:3, :], U) < 5e-4, "second step on the demoted cache")
end

println("2. device RegularizedVortex beyond rho_t")
guarded("regularized") do
    rk = RegularizedVortex(; sigma_row=8)
    # the old device behavior: singular beyond the regularized kernel's rho_t
    pk = PartitionedVortex(; sigma_row=8, rho_t=rk.rho_t)
    m = 8; sigma = 0.01
    rhos = range(rk.rho_t + 0.05, rk.rho_t + 0.3; length=m)
    Random.seed!(32)
    src = zeros(8, 2m)
    for k in 1:m
        u = normalize(randn(3))
        c = [1000sigma * k, 0.0, 0.0]
        src[1:3, k] .= c; src[1:3, m + k] .= c .+ u .* (rhos[k] * sigma)
    end
    src[5:7, :] .= randn(3, 2m); src[8, :] .= sigma
    src = DTF.(src)
    # host functors at the device precision: the erf-free g/h approximant is
    # precision-specific, so a Float64 reference would measure it, not the cutoff
    function host_sum(kern)
        out = zeros(13, 2m)
        for i in 1:2m, j in 1:2m
            i == j && continue
            d = src[1:3, i] .- src[1:3, j]; r2 = dot(d, d)
            out[:, i] .+= FM._direct_pair_ugh(kern, d[1], d[2], d[3], r2, inv(sqrt(r2)), src, j)
        end
        return out
    end
    ref = host_sum(rk); old = host_sum(pk)
    out = devarray(zeros(DTF, 13, 2m))
    dk = ext._ka_device_direct_kernel(rk, DTF, 0)
    kern = ext._cached_kernel(ext.ka_direct_all_pairs_kernel!, DEV_BACKEND, 64)
    kern(dk, out, devarray(src), 2m, DTF, Val(true); ndrange=64)
    KernelAbstractions.synchronize(DEV_BACKEND)
    rows = 2:13
    e_dev = relerr(Array(out)[rows, :], ref[rows, :])
    e_old = relerr(old[rows, :], ref[rows, :])
    check(e_dev < e_old / 10,
        @sprintf("device vs host RegularizedVortex %.2e, partitioned difference %.2e", e_dev, e_old))
end

println("3-5, 7. launch contracts on a device state")
guarded("launch contracts") do
    n = 512
    Random.seed!(33)
    sys = VortexParticles(DTF.(rand(3, n)), DTF.(randn(3, n) ./ n), fill(DTF(0.01), n);
        potential=zeros(DTF, 13, n), gradient_stretching=zeros(DTF, 6, n))
    cache = RadixFMMCache(sys; expansion_order=4, ell=3, window_classes=64,
        hessian=true, device=true, options=vortex_opts(DTF))
    fmm!(sys, cache)
    state = cache.state

    # 4. nearfield
    ext.ka_launch_nearfield!(state; clear=true); KernelAbstractions.synchronize(DEV_BACKEND)
    nf_ref = Array(state.output)
    for wg in (0, 128)
        ext.ka_launch_nearfield!(state; workgroup=wg, clear=true)
        KernelAbstractions.synchronize(DEV_BACKEND)
        check(relerr(Array(state.output), nf_ref) < 1e-5, "nearfield workgroup=$wg matches the default")
    end
    threw = try
        ext.ka_launch_nearfield!(state; workgroup=96, clear=true); false
    catch err
        err isa ArgumentError
    end
    check(threw, "nearfield workgroup=96 (not a multiple of 64 lanes) is refused")

    # 5. L2B
    fill!(state.output, 0); ext.ka_launch_l2b!(state; workgroup=64)
    KernelAbstractions.synchronize(DEV_BACKEND)
    l2b_ref = Array(state.output)
    fill!(state.output, 0); ext.ka_launch_l2b!(state; workgroup=0)
    KernelAbstractions.synchronize(DEV_BACKEND)
    check(maximum(abs, l2b_ref) > 0 && relerr(Array(state.output), l2b_ref) < 1e-6,
        "L2B workgroup=0 evaluates the locals")

    # 7. extra tree near sweep honors workgroup
    Random.seed!(34); ns = 60
    r1 = [SVector{3,DTF}(rand(3)) for _ in 1:ns]
    ex = Segs(r1, [r1[i] + SVector{3,DTF}(0.01 .* randn(3)) for i in 1:ns],
              DTF.(randn(ns) ./ ns), fill(DTF(0.005), ns))
    prepared = ext.ka_extra_tree_prepare(state, ex)
    fill!(state.output, 0); ext.ka_extra_tree_finish!(state, prepared; workgroup=128)
    KernelAbstractions.synchronize(DEV_BACKEND)
    ex_ref = Array(state.output)
    fill!(state.output, 0); ext.ka_extra_tree_finish!(state, prepared; workgroup=32)
    KernelAbstractions.synchronize(DEV_BACKEND)
    check(haskey(ext._KERNEL_CACHE, (ext.ka_extra_tree_near_kernel!, typeof(DEV_BACKEND), 32)) &&
        relerr(Array(state.output), ex_ref) < 1e-5,
        "extra-tree finish runs the near sweep at the requested workgroup")
end

println("3. direct_rectangular! kernel reuse")
struct RegRectSource <: FM.AbstractRectangularKernel end
FM.rect_source_rows(::RegRectSource) = 4
FM.rect_has_potential(::RegRectSource) = true
@inline function FM.rect_pair(::RegRectSource, target::SVector{3,T}, sources, q,
        ::Val{GRAD}, ::Val{POT}) where {T,GRAD,POT}
    @inbounds d = target - SVector{3,T}(sources[1, q], sources[2, q], sources[3, q])
    r2 = d[1]*d[1] + d[2]*d[2] + d[3]*d[3]
    iszero(r2) && return zero(SVector{3,T}), zero(SMatrix{3,3,T,9}), zero(T)
    @inbounds c = sources[4, q] / (4 * T(pi) * r2 * sqrt(r2))
    return c * d, zero(SMatrix{3,3,T,9}), zero(T)
end
guarded("direct_rectangular") do
    tgt = devarray(rand(DTF, 3, 50) .+ DTF(2)); src = devarray(rand(DTF, 4, 40))
    key = (ext.ka_rect_kernel!, typeof(DEV_BACKEND), 64)
    delete!(ext._KERNEL_CACHE, key)
    out = devarray(zeros(DTF, 3, 50))
    FM.direct_rectangular!(out, tgt, RegRectSource(), src)
    k1 = get(ext._KERNEL_CACHE, key, nothing)
    FM.direct_rectangular!(out, tgt, RegRectSource(), src)
    check(k1 !== nothing && ext._KERNEL_CACHE[key] === k1, "direct_rectangular! kernel is cached")
end

println("6. checked keys with NaN/huge positions")
guarded("checked keys") do
    x_min = SVector{3,DTF}(0, 0, 0); box = SVector{3,DTF}(1, 1, 1)
    for bad in (DTF(NaN), floatmax(DTF), -floatmax(DTF))
        pos = rand(DTF, 3, 16); pos[1, 7] = bad
        threw = try
            ext.ka_radix_keys_checked!(devarray(zeros(UInt64, 16)), devarray(zeros(Int32, 1)),
                zeros(Int32, 1), devarray(pos), x_min, box, DTF(0.5), 3)
            false
        catch err
            err isa ArgumentError
        end
        check(threw, "position $bad raises ArgumentError")
    end
end

@printf("\n%d passed, %d failed\n", npass[], nfail[])
nfail[] == 0 || error("$(nfail[]) check(s) failed")
println("gate passed: KA extension regressions")
