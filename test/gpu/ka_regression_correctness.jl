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
#   7. ka_extra_tree_finish! passes its workgroup to the near sweep, and
#      keeps 128 for the near sweep when none is named
#   8. rectangular box: a body on a short axis' upper face, all leaf cells
#      filled, matches the host cache
#   9. host-resident target metadata rows are refilled every step
#  10. a contract-only extra source functor (not an AbstractDirectKernel)
#      carried by the tree matches the same source summed all-pairs
#  11. concurrent _device_row_extrema callers do not share scratch
include("ka_backend.jl")
using FastMultipole, Random, Test, Printf, LinearAlgebra
using FastMultipole.StaticArrays
const FM = FastMultipole
include(joinpath(@__DIR__, "..", "helpers", "vortex.jl"))
include(joinpath(@__DIR__, "..", "helpers", "gravitational.jl"))
include(joinpath(@__DIR__, "..", "helpers", "interface_test_systems.jl"))
include(joinpath(@__DIR__, "..", "helpers", "extra_tree_test_systems.jl"))

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
    # one thread per target body: any group size, and the same sum order
    for wg in (0, 96, 128)
        ext.ka_launch_nearfield!(state; workgroup=wg, clear=true)
        KernelAbstractions.synchronize(DEV_BACKEND)
        check(Array(state.output) == nf_ref, "nearfield workgroup=$wg matches the default bit for bit")
    end

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
    check(haskey(ext._KERNEL_CACHE, (ext.ka_extra_tree_near_bodies_kernel!, typeof(DEV_BACKEND), 32)) &&
        relerr(Array(state.output), ex_ref) < 1e-5,
        "extra-tree finish runs the near sweep at the requested workgroup")
    key128 = (ext.ka_extra_tree_near_bodies_kernel!, typeof(DEV_BACKEND), 128)
    delete!(ext._KERNEL_CACHE, key128)
    fill!(state.output, 0); ext.ka_extra_tree_finish!(state, prepared)
    KernelAbstractions.synchronize(DEV_BACKEND)
    check(haskey(ext._KERNEL_CACHE, key128) && relerr(Array(state.output), ex_ref) < 1e-5,
        "extra-tree finish runs the near sweep at 128 by default")
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

# a gravitational system carrying one controlled metadata row, recording the
# metadata row it sees in buffer_to_target_system!
struct RegMetaGrav{TF}
    inner::Gravitational{TF}
    meta::Vector{TF}
    seen::Vector{TF}
end
RegMetaGrav(inner::Gravitational{TF}) where TF = (n = FM.get_n_bodies(inner);
    RegMetaGrav{TF}(inner, zeros(TF, n), fill(TF(NaN), n)))
FM.device_backend(::RegMetaGrav) = DEV_BACKEND
FM.get_n_bodies(s::RegMetaGrav) = FM.get_n_bodies(s.inner)
FM.get_position(s::RegMetaGrav, i) = FM.get_position(s.inner, i)
FM.data_per_body(s::RegMetaGrav) = FM.data_per_body(s.inner)
FM.strength_dims(s::RegMetaGrav) = FM.strength_dims(s.inner)
FM.has_vector_potential(::RegMetaGrav) = false
FM.source_system_to_buffer!(buffer, i_buffer, s::RegMetaGrav, i_body) =
    FM.source_system_to_buffer!(buffer, i_buffer, s.inner, i_body)
FM.body_to_multipole!(s::RegMetaGrav, args...) =
    FM.body_to_multipole!(FM.Point{FM.Source}, s, args...; scale_strength=-1.0)
FM.direct!(tb, ti, sw::FM.DerivativesSwitch, s::RegMetaGrav, sb, si) =
    FM.direct!(tb, ti, sw, s.inner, sb, si)
FM.metadata_per_body(::RegMetaGrav) = 1
FM.metadata_to_buffer!(buffer, switch, i_buffer, s::RegMetaGrav, i_body) =
    (buffer[FM.metadata_index(switch, 1), i_buffer] = s.meta[i_body])
function FM.buffer_to_target_system!(s::RegMetaGrav, i_target, switch, buffer, i_buffer)
    s.seen[i_target] = buffer[FM.metadata_index(switch, 1), i_buffer]
    return FM.buffer_to_target_system!(s.inner, i_target, switch, buffer, i_buffer)
end
grav_opts(TF) = FM.RadixLifecycleOptions(; precision=TF,
    m2l_strategy=FM.ConcatenatedFixedZM2L(), body_type=FM.Point{FM.Source})

println("8. rectangular box, body on a short axis' upper face")
guarded("rectangular box") do
    # bounds (8,2,2) at ell=3: unit leaf cells, ell_axes (3,1,1), all 32 leaf
    # cells filled, plus one body exactly on the y = 2 face
    Random.seed!(35)
    cells = collect(Iterators.product(0:7, 0:1, 0:1))
    nper = 4
    n = length(cells) * nper + 1
    bodies = rand(8, n)
    for (k, (i, j, l)) in enumerate(cells), b in 1:nper
        bodies[1:3, (k - 1) * nper + b] .= (i, j, l) .+ 0.1 .+ 0.8 .* rand(3)
    end
    bodies[1:3, n] .= (3.5, 2.0, 0.5)
    bodies[4, :] .*= 0.01
    bodies[5, :] ./= n
    bounds(TF) = (SVector{3,TF}(0, 0, 0), SVector{3,TF}(8, 2, 2))
    sysh = RegMetaGrav(Gravitational(copy(bodies)))
    sysd = RegMetaGrav(Gravitational(DTF.(bodies)))
    host = RadixFMMCache(sysh; expansion_order=4, ell=3, window_classes=64,
        bounds=bounds(Float64), options=grav_opts(Float64))
    dev = RadixFMMCache(sysd; expansion_order=4, ell=3, window_classes=64,
        bounds=bounds(DTF), device=true, options=grav_opts(DTF))
    fmm!(sysh, host); fmm!(sysd, dev)
    check(dev.ell_axes == SVector(3, 1, 1) && dev.state.counts.n_cells == length(cells),
        "device grid holds exactly the 32 leaf cells (n_cells $(dev.state.counts.n_cells))")
    e = relerr(sysd.inner.potential[1:7, :], sysh.inner.potential[1:7, :])
    check(e < 1e-4, @sprintf("device vs host rectangular cache (%.2e)", e))
end

println("9. metadata rows on a device cache")
guarded("metadata") do
    sys = RegMetaGrav(Gravitational(DTF.(rand(8, 300) .* [1, 1, 1, 0.01, 1/300, 1, 1, 1])))
    cache = RadixFMMCache(sys; expansion_order=4, ell=3, window_classes=64,
        device=true, options=grav_opts(DTF))
    ok = true
    for call in 1:3
        sys.meta .= DTF.(1:300) .+ 10call
        fill!(sys.seen, NaN)
        fmm!(sys, cache)
        ok &= sys.seen == sys.meta
    end
    check(ok, "every step sees its current metadata")
end

println("10. contract-only extra source functor carried by the tree")
# not an AbstractDirectKernel, and no `_emits_potential`
struct ContractKernel end
@inline function FM._extra_pair_ug(::ContractKernel, tx, ty, tz, buf, j)
    T = typeof(tx)
    @inbounds begin
        dx = tx - buf[1, j]; dy = ty - buf[2, j]; dz = tz - buf[3, j]
    end
    r2 = dx * dx + dy * dy + dz * dz
    r2 > zero(T) || return zero(T), zero(T), zero(T), zero(T)
    return FM._direct_pair_ug(FM.SingularVortex(), dx, dy, dz, r2, inv(sqrt(r2)), buf, j)
end
struct ContractPV{TF}
    pv::PV{TF}
end
FM.get_n_bodies(p::ContractPV) = FM.get_n_bodies(p.pv)
FM.data_per_body(::ContractPV) = 8
FM.strength_dims(::ContractPV) = 3
FM.has_vector_potential(::ContractPV) = true
FM.body_type(::ContractPV) = FM.Point{FM.Vortex}
FM.get_position(p::ContractPV, i) = FM.get_position(p.pv, i)
FM.source_system_to_buffer!(b, ib, p::ContractPV, i) = FM.source_system_to_buffer!(b, ib, p.pv, i)
FM.direct_kernel(::ContractPV) = ContractKernel()
guarded("contract functor") do
    n = 600; ns = 80
    Random.seed!(36)
    pos = DTF.(rand(3, n)); str = DTF.(randn(3, n) ./ n)
    ex = ContractPV(PV([SVector{3,DTF}(rand(3)) for _ in 1:ns],
        [SVector{3,DTF}(randn(3) ./ ns) for _ in 1:ns]))
    function velocity(kw)
        sys = VortexParticles(copy(pos), copy(str), fill(DTF(0.01), n);
            potential=zeros(DTF, 13, n), gradient_stretching=zeros(DTF, 6, n))
        cache = RadixFMMCache(sys; expansion_order=4, ell=3, window_classes=64,
            hessian=true, device=true, options=vortex_opts(DTF))
        sw = FM.DerivativesSwitch(false, true, false, (sys,))
        ext.ka_radix_cache_device_step!(cache, (sys,), sw; kw...)
        st = cache.state; out = Array(st.output); inv = Array(st.grid.invperm)
        return reduce(hcat, [out[2:4, inv[i]] for i in 1:Int(st.counts.n_bodies)])
    end
    base = velocity((;))
    tree = velocity((; extra_tree_sources=(ex,))) .- base
    direct = velocity((; extra_sources=(ex,))) .- base
    e = maximum(abs.(tree .- direct)) / maximum(abs, direct)
    check(e < 5e-3, @sprintf("tree-carried contract functor vs all-pairs (%.2e)", e))
end

println("11. concurrent row extrema")
guarded("row extrema") do
    Random.seed!(37)
    Ah = DTF.(randn(4, 5000) .* [1, 10, 100, 1000])
    A = devarray(Ah)
    rows = repeat(1:4, 4)
    # discriminating only with several threads (julia --threads=N); on one
    # thread the tasks do not interleave inside the call
    tasks = [Threads.@spawn FM._device_row_extrema(A, r, 5000) for r in rows]
    got = fetch.(tasks)
    want = [(minimum(Ah[r, :]), maximum(Ah[r, :])) for r in rows]
    check(got == want, "16 concurrent callers each get their own row's extrema")
end

@printf("\n%d passed, %d failed\n", npass[], nfail[])
nfail[] == 0 || error("$(nfail[]) check(s) failed")
println("gate passed: KA extension regressions")
