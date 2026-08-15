#=
Adaptive octree host resident lifecycle (task 040).

Verifies the end-to-end host lifecycle over the task-039 adaptive octree
(theory/adaptive-radix-octree.md §2.6, §4):
  - accuracy gates: velocity RMS <= 1e-3 against full direct references on the
    cube, wake-like filament, and multi-scale (100x contrast) cases, at P = 4
    and P = 8, Float64 and Float32, scalar (Gravitational) and Lamb-Helmholtz
    vortex (VortexParticles) — with W/X lists exercised (asserted nonempty on
    the multi-level cases);
  - uniform-limit lifecycle parity: at matched HierarchicalRigidStencil policy
    and forced single-depth leaves, the adaptive lifecycle reproduces the
    production uniform hierarchical path to machine precision;
  - the theory §4 exact M2L-composition oracles for the new operators:
    M2T == (dense M2L to a point-target local) evaluated at its center, and
    S2L == (point P2M -> dense M2L), per channel (phi + chi at P_active =
    P + 1 per 008h), both precisions, P = 4 and P = 8 — including the
    038-mandated Lamb-Helmholtz vortex S2L numerical parity, and the
    scalar M2T hessian;
  - regularized-nearfield cross-parity (RegularizedVortex through the adaptive
    U list + per-cell sigma gate vs the validated uniform path);
  - zero-allocation contract: typed per-step lifecycle refresh allocates 0
    bytes; the typed lifecycle run allocates only a small constant (dynamic
    Val() dispatch at the shared nearfield mode barrier, same idiom as the
    uniform host path); warm fmm! stays under the repo step gate; the 023
    host counter contract (all transfer counters zero);
  - task-040 construction guards (TwoPassVortex/PartitionedVortex, ungated
    regularized kernels, hessian + Lamb-Helmholtz).
=#

using Test
using Random
using Statistics
using LinearAlgebra
using FastMultipole
using FastMultipole.StaticArrays

if !isdefined(@__MODULE__, :Gravitational)
    include("gravitational.jl")
end
if !isdefined(@__MODULE__, :VortexParticles)
    include("vortex.jl")
end
if !isdefined(@__MODULE__, :SmoothedVortex)
    include("interface_test_systems.jl")
end

const ALT_FM = FastMultipole

#--- distributions (5 x n: x, y, z, radius, strength) ---#

function _alt_cube(n; seed=40101)
    rng = MersenneTwister(seed)
    b = rand(rng, 5, n)
    b[4, :] .*= 1e-3
    b[5, :] ./= n
    return b
end

"""Clustered multi-scale field (embedded dense cluster, 100x contrast)."""
function _alt_multiscale(n; contrast=100.0, seed=40102)
    rng = MersenneTwister(seed)
    frac = 0.35
    nc = round(Int, frac * n)
    nb = n - nc
    Rc = (3 * nc / (4pi * contrast * nb))^(1 / 3)
    b = rand(rng, 5, n)
    b[4, :] .*= 1e-3
    b[5, :] ./= n
    b[1:3, 1:nb] .= rand(rng, 3, nb)
    ctr = (0.6, 0.4, 0.55)
    k = 0
    while k < nc
        p = 2 .* (rand(rng, 3) .- 0.5)
        if sum(abs2, p) <= 1
            k += 1
            b[1:3, nb + k] .= ctr .+ Rc .* p
        end
    end
    return b
end

"""Wake-like field: thin helical filament plus diffuse haze."""
function _alt_filament(n; seed=40103)
    rng = MersenneTwister(seed)
    nf = round(Int, 0.6n)
    b = rand(rng, 5, n)
    b[4, :] .*= 1e-3
    b[5, :] ./= n
    core = 0.004
    for p in 1:nf
        t = 4pi * (p - 1) / nf
        c = (0.5 + 0.35cos(t), 0.5 + 0.35sin(t), 0.15 + 0.7t / (4pi))
        b[1:3, p] .= c .+ core .* randn(rng, 3)
    end
    b[1:3, nf+1:end] .= rand(rng, 3, n - nf)
    return b
end

_alt_rel_rms(a, b) = sqrt(mean(abs2, a .- b)) / sqrt(mean(abs2, b))

#--- accuracy gates: scalar (Gravitational), all three cases ---#

@testset "adaptive lifecycle accuracy (scalar, velocity RMS <= 1e-3)" begin
    n = 2000
    cases = (("cube", _alt_cube(n)), ("filament", _alt_filament(n)),
             ("multiscale", _alt_multiscale(n)))
    for (name, b) in cases
        ref = Gravitational(copy(b))
        FastMultipole.direct!(ref; scalar_potential=true, gradient=true)
        gref = ref.potential[5:7, :]
        pref = ref.potential[1, :]
        for P in (4, 8), TF in (Float64, Float32)
            sys = Gravitational(copy(b))
            pol = AdaptiveTreePolicy(K_max=16, ell_max=6, near_radius2=5)
            opts = ALT_FM.CUDARadixLifecycleOptions(precision=TF,
                m2l_strategy=ALT_FM.ConcatenatedFixedZM2L())
            cache = ALT_FM.RadixFMMCache(sys; expansion_order=P, ell=3,
                adaptive=pol, options=opts)
            if name != "cube"
                # multi-level tree: the W/X paths must actually be exercised
                @test cache.adaptive_lists.n_w > 0
                @test cache.adaptive_lists.n_x > 0
            end
            fmm!(sys, cache; scalar_potential=true, gradient=true)
            @test _alt_rel_rms(sys.potential[5:7, :], gref) <= 1e-3
            @test maximum(abs.(sys.potential[1, :] .- pref)) /
                maximum(abs.(pref)) <= 1e-3
        end
    end
end

#--- accuracy gates: Lamb-Helmholtz vortex ---#

@testset "adaptive lifecycle accuracy (LH vortex, velocity RMS <= 1e-3)" begin
    n = 2000
    rng = MersenneTwister(40110)
    str = randn(rng, 3, n) ./ n
    for (name, b) in (("cube", _alt_cube(n; seed=40111)),
                      ("multiscale", _alt_multiscale(n; seed=40112)))
        pos = b[1:3, :]
        ref = VortexParticles(copy(pos), copy(str))
        FastMultipole.direct!(ref; scalar_potential=false, gradient=true)
        gref = ref.gradient_stretching[1:3, :]
        for P in (4, 8), TF in (Float64, Float32)
            sys = VortexParticles(copy(pos), copy(str))
            pol = AdaptiveTreePolicy(K_max=16, ell_max=6, near_radius2=5)
            opts = ALT_FM.CUDARadixLifecycleOptions(precision=TF,
                m2l_strategy=ALT_FM.ConcatenatedFixedZM2L(),
                body_type=ALT_FM.Point{ALT_FM.Vortex})
            cache = ALT_FM.RadixFMMCache(sys; expansion_order=P, ell=3,
                adaptive=pol, options=opts, lamb_helmholtz=true)
            name == "multiscale" && @test cache.adaptive_lists.n_w > 0
            fmm!(sys, cache; scalar_potential=false, gradient=true)
            @test _alt_rel_rms(sys.gradient_stretching[1:3, :], gref) <= 1e-3
        end
    end
end

#--- uniform-limit lifecycle parity (machine precision at matched policy) ---#

@testset "adaptive uniform-limit lifecycle parity" begin
    for ell in (2, 3), P in (4, 8)
        G = 1 << ell
        rng = MersenneTwister(40120 + ell)
        nb = G^3
        b = zeros(5, nb)
        Δ = 1.0 / G
        i = 0
        for z in 0:G-1, y in 0:G-1, x in 0:G-1
            i += 1
            b[1:3, i] .= (Δ * (x + 0.5 + 0.4 * (rand(rng) - 0.5)),
                          Δ * (y + 0.5 + 0.4 * (rand(rng) - 0.5)),
                          Δ * (z + 0.5 + 0.4 * (rand(rng) - 0.5)))
            b[4, i] = 1e-4
            b[5, i] = rand(rng) / nb
        end
        x_min = SVector(0.0, 0.0, 0.0)
        q = 5
        policy = ALT_FM.HierarchicalRigidStencil(P,
            ALT_FM.rigid_stencil_epsilon(P, 0.5, ell, q); near_radius2=q)
        sys_u = Gravitational(copy(b))
        cache_u = ALT_FM.RadixFMMCache(sys_u; expansion_order=P, ell=ell,
            bounds=(x_min, 1.0), policy=policy)
        fmm!(sys_u, cache_u; scalar_potential=true, gradient=true)
        sys_a = Gravitational(copy(b))
        pol = AdaptiveTreePolicy(K_max=1, ell_max=ell, near_radius2=q)
        cache_a = ALT_FM.RadixFMMCache(sys_a; expansion_order=P, ell=ell,
            bounds=(x_min, 1.0), policy=policy, adaptive=pol)
        @test cache_a.adaptive_lists.n_w == 0
        @test cache_a.adaptive_lists.n_x == 0
        fmm!(sys_a, cache_a; scalar_potential=true, gradient=true)
        scale = maximum(abs.(sys_u.potential[5:7, :]))
        @test maximum(abs.(sys_a.potential[5:7, :] .- sys_u.potential[5:7, :])) <=
            1e-12 * scale
        @test maximum(abs.(sys_a.potential[1, :] .- sys_u.potential[1, :])) <=
            1e-12 * maximum(abs.(sys_u.potential[1, :]))
    end
end

#--- M2T / S2L exact M2L-composition oracles (theory §4) ---#

# Dense-oracle M2L: local expansion at `c_tgt` from multipole buffer column 1
# about `c_src`, through the validated MaterializedYRotationM2L operator batch.
function _alt_m2l_local(mp, c_src, c_tgt, basis, ::Type{TF}, lhv::Val{LH}) where {TF,LH}
    cache = ALT_FM.OperatorInvariantCache(TF, basis)
    scratch = ALT_FM.M2LOperatorScratch(TF, basis, 1)
    targets = ALT_FM.FlatCoefficientBuffer(TF, basis, 1)
    sources = ALT_FM.FlatCoefficientBuffer(TF, basis, 1)
    sources.phi .= mp.phi
    LH && (sources.chi .= mp.chi)
    d = SVector{3,TF}(c_tgt - c_src)
    r, theta, phi = ALT_FM.cartesian_to_spherical(d)
    ALT_FM.m2l_operator_batch!(ALT_FM.MaterializedYRotationM2L(), targets,
        sources, [TF(phi)], [TF(theta)], [TF(r)], cache, scratch, lhv)
    return targets
end

@testset "M2T oracle: M2L composition ($(TF), P=$(P))" for TF in (Float64, Float32),
        P in (4, 8)
    rtol = TF === Float64 ? 1e-10 : 2e-4
    basis = ALT_FM.OperatorBasisInfo(ALT_FM.CompressedComplexBasis(), P, Val(false))
    basis_lh = ALT_FM.OperatorBasisInfo(ALT_FM.CompressedComplexBasis(), P, Val(true))
    rng = MersenneTwister(40130 + P)
    c_src = SVector{3,TF}(0.1, 0.2, 0.3)
    ns = 8
    src = zeros(TF, 7, ns)
    for k in 1:ns
        src[1:3, k] .= c_src .+ TF(0.02) .* (rand(rng, SVector{3,TF}) .- TF(0.5))
        src[5:7, k] .= randn(rng, TF, 3)
    end
    cell_ranges = reshape([1, ns], 2, 1)
    cc = reshape(collect(c_src), 3, 1)
    nH = ALT_FM.harmonic_index(P + 2, P + 2)
    H = Array{TF,3}(undef, 2, 1, nH)
    targets_x = (SVector{3,TF}(0.8, 0.7, 0.9), SVector{3,TF}(0.55, 0.15, 0.72))

    # scalar channel (+ hessian)
    mp = ALT_FM._host_flat_buffer(TF, basis, 1)
    ALT_FM._host_b2m_kernel!(ALT_FM.phi_slab(mp), src, cell_ranges, cc, [1], P, 1)
    for x_t in targets_x
        d = x_t - c_src
        r, th, ph = ALT_FM.cartesian_to_spherical(d)
        ALT_FM.irregular_harmonics!(H, r, th, ph, P + 2)
        got = ALT_FM._resident_multipole_eval_flat_hessian(ALT_FM.phi_slab(mp),
            ALT_FM.chi_slab(mp), 1, H, P, P, Val(false))
        loc = _alt_m2l_local(mp, c_src, x_t, basis, TF, Val(false))
        want = ALT_FM._resident_local_eval_flat_hessian(ALT_FM.phi_slab(loc),
            ALT_FM.chi_slab(loc), 1, zero(TF), zero(TF), zero(TF), P, P, Val(false))
        scale_g = maximum(abs, want[2:4])
        scale_h = maximum(abs, want[5:13])
        @test isapprox(got[1], want[1]; rtol=rtol)
        for j in 2:4
            @test abs(got[j] - want[j]) <= rtol * scale_g
        end
        for j in 5:13
            @test abs(got[j] - want[j]) <= 10 * rtol * scale_h
        end
    end

    # Lamb-Helmholtz vortex channel (phi + chi at P_active = P + 1)
    mp_lh = ALT_FM._host_flat_buffer(TF, basis_lh, 1)
    ALT_FM._host_b2m_vortex_kernel!(ALT_FM.phi_slab(mp_lh), ALT_FM.chi_slab(mp_lh),
        src, cell_ranges, cc, [1], P, P + 1, 1)
    for x_t in targets_x
        d = x_t - c_src
        r, th, ph = ALT_FM.cartesian_to_spherical(d)
        ALT_FM.irregular_harmonics!(H, r, th, ph, P + 2)
        got = ALT_FM._resident_multipole_eval_flat(ALT_FM.phi_slab(mp_lh),
            ALT_FM.chi_slab(mp_lh), 1, H, P, P + 1, Val(true))
        loc = _alt_m2l_local(mp_lh, c_src, x_t, basis_lh, TF, Val(true))
        want = ALT_FM._resident_local_eval_flat(ALT_FM.phi_slab(loc),
            ALT_FM.chi_slab(loc), 1, zero(TF), zero(TF), zero(TF), P, P + 1, Val(true))
        scale_g = maximum(abs, (want[2], want[3], want[4]))
        for j in 2:4
            @test abs(got[j] - want[j]) <= rtol * scale_g
        end
    end
end

@testset "S2L oracle: P2M-M2L composition ($(TF), P=$(P))" for TF in (Float64, Float32),
        P in (4, 8)
    rtol = TF === Float64 ? 1e-10 : 2e-4
    basis = ALT_FM.OperatorBasisInfo(ALT_FM.CompressedComplexBasis(), P, Val(false))
    basis_lh = ALT_FM.OperatorBasisInfo(ALT_FM.CompressedComplexBasis(), P, Val(true))
    rng = MersenneTwister(40140 + P)
    nH = ALT_FM.harmonic_index(P + 2, P + 2)
    H = Array{TF,3}(undef, 2, 1, nH)
    c_tgt = SVector{3,TF}(0.85, 0.75, 0.95)
    for trial in 1:2
        x_s = SVector{3,TF}(0.1, 0.2, 0.3) .+ TF(0.15) .* rand(rng, SVector{3,TF})
        src = zeros(TF, 7, 1)
        src[1:3, 1] .= x_s
        src[5:7, 1] .= randn(rng, TF, 3)
        cell_ranges = reshape([1, 1], 2, 1)
        cc3 = reshape(collect(x_s), 3, 1)
        ctr3 = reshape(collect(c_tgt), 3, 1)
        eval_pts = (SVector{3,TF}(0.02, -0.015, 0.01), SVector{3,TF}(-0.01, 0.02, -0.02))

        # scalar: point P2M (degree 0) -> M2L vs S2L kernel
        mp = ALT_FM._host_flat_buffer(TF, basis, 1)
        ALT_FM._host_b2m_kernel!(ALT_FM.phi_slab(mp), src, cell_ranges, cc3, [1], P, 1)
        loc_comp = _alt_m2l_local(mp, x_s, c_tgt, basis, TF, Val(false))
        loc_s2l = ALT_FM._host_flat_buffer(TF, basis, 1)
        fill!(ALT_FM.phi_slab(loc_s2l), zero(TF))
        ALT_FM._host_s2l_pairs_kernel!(ALT_FM.phi_slab(loc_s2l), src, [1], [1],
            ctr3, [1], [1], 1, H, P)
        # coefficient-level parity
        scale_c = maximum(abs, ALT_FM.phi_slab(loc_comp))
        @test maximum(abs, ALT_FM.phi_slab(loc_s2l) .- ALT_FM.phi_slab(loc_comp)) <=
            rtol * scale_c
        # evaluated parity
        for dx in eval_pts
            a = ALT_FM._resident_local_eval_flat(ALT_FM.phi_slab(loc_s2l),
                ALT_FM.chi_slab(loc_s2l), 1, dx[1], dx[2], dx[3], P, P, Val(false))
            w = ALT_FM._resident_local_eval_flat(ALT_FM.phi_slab(loc_comp),
                ALT_FM.chi_slab(loc_comp), 1, dx[1], dx[2], dx[3], P, P, Val(false))
            for j in 1:4
                @test abs(a[j] - w[j]) <= rtol * max(abs(w[1]), abs(w[2]), abs(w[3]), abs(w[4]))
            end
        end

        # Lamb-Helmholtz vortex: point P2M (degree <= 1) -> M2L vs vortex S2L
        # (the 038-mandated numerical parity, both channels)
        mp_lh = ALT_FM._host_flat_buffer(TF, basis_lh, 1)
        ALT_FM._host_b2m_vortex_kernel!(ALT_FM.phi_slab(mp_lh),
            ALT_FM.chi_slab(mp_lh), src, cell_ranges, cc3, [1], P, P + 1, 1)
        loc_comp_lh = _alt_m2l_local(mp_lh, x_s, c_tgt, basis_lh, TF, Val(true))
        loc_s2l_lh = ALT_FM._host_flat_buffer(TF, basis_lh, 1)
        fill!(ALT_FM.phi_slab(loc_s2l_lh), zero(TF))
        fill!(ALT_FM.chi_slab(loc_s2l_lh), zero(TF))
        ALT_FM._host_s2l_vortex_pairs_kernel!(ALT_FM.phi_slab(loc_s2l_lh),
            ALT_FM.chi_slab(loc_s2l_lh), src, [1], [1], ctr3, [1], [1], 1, H,
            P, P + 1)
        # phi rows: machine-exact parity
        scale_phi = maximum(abs, ALT_FM.phi_slab(loc_comp_lh))
        @test maximum(abs, ALT_FM.phi_slab(loc_s2l_lh) .- ALT_FM.phi_slab(loc_comp_lh)) <=
            rtol * scale_phi
        # chi rows through degree P: machine-exact parity. The chi TOP row
        # (P_active = P + 1, the 008h neighbor row) is representation-
        # dependent: the M2L composition's top row carries its own truncated
        # Lamb-Helmholtz row-up mixing, while S2L projects it exactly —
        # measured equally accurate physically (dev probe 2026-08-14; both
        # evaluate the analytic Biot-Savart field to the same truncation
        # order), so the top row is compared through the evaluated field
        # below, not coefficient-wise.
        scale_chi = maximum(abs, ALT_FM.chi_slab(loc_comp_lh))
        chi_rows_leq_P = [ALT_FM.flat_basis_index(n, m, ri)
            for n in 0:P for m in 0:n for ri in 1:2]
        @test maximum(abs, ALT_FM.chi_slab(loc_s2l_lh)[chi_rows_leq_P, 1] .-
            ALT_FM.chi_slab(loc_comp_lh)[chi_rows_leq_P, 1]) <= rtol * scale_chi
        # evaluated velocity parity (both channels active; tolerance covers the
        # top-row truncation-tail difference, which is O((|dx|/r)^{P+1}))
        eval_rtol = TF === Float64 ? 1e-5 : 2e-3
        for dx in eval_pts
            a = ALT_FM._resident_local_eval_flat(ALT_FM.phi_slab(loc_s2l_lh),
                ALT_FM.chi_slab(loc_s2l_lh), 1, dx[1], dx[2], dx[3], P, P + 1, Val(true))
            w = ALT_FM._resident_local_eval_flat(ALT_FM.phi_slab(loc_comp_lh),
                ALT_FM.chi_slab(loc_comp_lh), 1, dx[1], dx[2], dx[3], P, P + 1, Val(true))
            scale_g = max(abs(w[2]), abs(w[3]), abs(w[4]))
            for j in 2:4
                @test abs(a[j] - w[j]) <= eval_rtol * scale_g
            end
        end
        # analytic anchor: the S2L local evaluates the exact point-vortex
        # Biot-Savart velocity to truncation (Float64 only; independent of the
        # M2L composition, so a shared convention error cannot hide here)
        if TF === Float64
            Gs = SVector{3,TF}(src[5, 1], src[6, 1], src[7, 1])
            for dx in eval_pts
                x_t = c_tgt + dx
                dv = x_t - x_s
                vref = cross(Gs, dv) ./ (TF(4pi) * norm(dv)^3)
                a = ALT_FM._resident_local_eval_flat(ALT_FM.phi_slab(loc_s2l_lh),
                    ALT_FM.chi_slab(loc_s2l_lh), 1, dx[1], dx[2], dx[3], P, P + 1,
                    Val(true))
                va = SVector(a[2], a[3], a[4])
                @test norm(va .- vref) <= 1e-4 * norm(vref)
            end
        end
    end
end

#--- regularized nearfield through the adaptive U list + per-cell gate ---#

@testset "adaptive RegularizedVortex vs erf-based regularized direct" begin
    nv = 500
    seed = 40150
    base = generate_vortex(seed, nv)
    sigma = 0.01 .+ 0.01 .* rand(MersenneTwister(seed), nv)
    ssys = SmoothedVortex(base, sigma)
    U_ref, _ = _interface_regularized_direct(SmoothedVortex(base, sigma))
    dk = ALT_FM.RegularizedVortex(; sigma_row=8)
    pol = AdaptiveTreePolicy(K_max=16, ell_max=6, near_radius2=5,
        rho_t=dk.rho_t, sigma_row=8)
    cache = ALT_FM.RadixFMMCache(ssys; expansion_order=8, ell=3, adaptive=pol)
    @test cache.state.options.direct_kernel == dk
    @test cache.adaptive_tree.sigma_armed
    fmm!(ssys, cache; scalar_potential=false, gradient=true)
    u_scale = maximum(abs.(U_ref))
    @test maximum(abs.(base.gradient_stretching[1:3, :] .- U_ref)) / u_scale < 1e-3
end

#--- zero-allocation + counters + array identity ---#

_alt_typed_refresh_alloc(al, st, tr, li, bufs) =
    @allocated ALT_FM._refresh_adaptive_lifecycle_typed!(al, st, tr, li, bufs)
_alt_typed_run_alloc(al, st, tr, li) =
    @allocated ALT_FM._run_adaptive_host_lifecycle_typed!(al, st, tr, li)
_alt_fmm_alloc(sys, cache) =
    @allocated fmm!(sys, cache; scalar_potential=true, gradient=true)

@testset "adaptive lifecycle zero-allocation + counter contract" begin
    n = 1500
    b = _alt_multiscale(n; seed=40160)
    sys = Gravitational(copy(b))
    pol = AdaptiveTreePolicy(K_max=16, ell_max=6, near_radius2=5)
    cache = ALT_FM.RadixFMMCache(sys; expansion_order=4, ell=3, adaptive=pol)
    fmm!(sys, cache; scalar_potential=true, gradient=true)
    fmm!(sys, cache; scalar_potential=true, gradient=true)
    al = cache.adaptive_state
    st = al.state
    tree = cache.adaptive_tree
    lists = cache.adaptive_lists
    # typed per-step refresh is exactly allocation-free
    @test _alt_typed_refresh_alloc(al, st, tree, lists, cache.source_buffers) == 0
    # the typed lifecycle run carries only the small constant Val() dispatch of
    # the shared nearfield-mode barrier (same idiom as the uniform host path)
    @test _alt_typed_run_alloc(al, st, tree, lists) <= 4096
    # whole warm step under the repo step gate
    @test _alt_fmm_alloc(sys, cache) < 512_000
    # 023 host counter contract: no transfers ever
    c = st.counters
    @test c.body_uploads == 0
    @test c.influence_downloads == 0
    @test c.expansion_host_copies == 0
    @test c.route_uploads == 0
    @test c.operator_uploads == 0
    @test c.metadata_downloads == 0
    # array identity across steps (capacity contract: no reallocation)
    out_id = objectid(st.output)
    mp_id = objectid(st.multipoles.phi)
    fmm!(sys, cache; scalar_potential=true, gradient=true)
    @test objectid(cache.adaptive_state.state.output) == out_id
    @test objectid(cache.adaptive_state.state.multipoles.phi) == mp_id
end

#--- task-040 construction guards ---#

@testset "adaptive lifecycle guards" begin
    seed = 40170
    base = generate_vortex(seed, 300)
    sigma = fill(0.004, 300)
    pol = AdaptiveTreePolicy(K_max=16, ell_max=6, near_radius2=5)
    dk = ALT_FM.RegularizedVortex(; sigma_row=8)
    pol_gate = AdaptiveTreePolicy(K_max=16, ell_max=6, near_radius2=5,
        rho_t=dk.rho_t, sigma_row=8)
    # regularized kernel without the per-cell gate armed: refused
    @test_throws ArgumentError ALT_FM.RadixFMMCache(
        SmoothedVortex(base, sigma); expansion_order=4, ell=3, adaptive=pol)
    # hessian + LH + adaptive: refused (W-list M2T LH hessian deferral)
    @test_throws ArgumentError ALT_FM.RadixFMMCache(
        SmoothedVortex(base, sigma); expansion_order=4, ell=3,
        adaptive=pol_gate, hessian=true)
    # TwoPassVortex + adaptive: refused (uniform-lattice deficit sweep)
    opts_tp = ALT_FM.CUDARadixLifecycleOptions(precision=Float64,
        m2l_strategy=ALT_FM.ConcatenatedFixedZM2L(),
        body_type=ALT_FM.Point{ALT_FM.Vortex},
        direct_kernel=ALT_FM.TwoPassVortex(; sigma_row=8))
    @test_throws ArgumentError ALT_FM.RadixFMMCache(
        SmoothedVortex(base, sigma); expansion_order=4, ell=3,
        options=opts_tp, lamb_helmholtz=true,
        adaptive=AdaptiveTreePolicy(K_max=16, ell_max=6, near_radius2=5,
            rho_t=4.789, sigma_row=8))
end
