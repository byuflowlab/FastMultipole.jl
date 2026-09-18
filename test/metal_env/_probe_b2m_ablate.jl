# B2M attribution. The WG sweep (_probe_b2m_cost.jl) ruled out the barrier
# hypothesis: time FALLS from WG=16 to WG=128 while barrier count RISES, and
# WG=64 ties WG=128. So the cost is per-body-x-(n,m) work, not the reduction.
#
# What that work is: the (n,m) loop is OUTSIDE the body loop, so for EACH of
# the 63 (n,m) pairs every body redoes
#   _resident_harmonic_setup   sqrt + acos + atan + 2x sincos   (loop-invariant!)
#   3x _resident_regular_harmonic_coeff   a full O(n^2) Legendre recurrence
#                                          with a divide in the inner loop
# i.e. ~1.0M transcendental prologues and ~3.0M recurrences per B2M call.
#
# Ablate ONE op at a time ([[feedback-ablations-must-isolate-one-variable]]),
# replacing rather than deleting, and check the parts against the whole:
#   :full     the shipping kernel
#   :algsetup setup's transcendentals replaced by their exact algebraic
#             identities (cos(theta)=dz/rho etc). NOT an ablation -- a
#             candidate optimisation; accuracy is scored.
#   :norecur  setup kept, the three q lookups replaced by setup values.
#             Wrong answers by construction; isolates the recurrence.
#   :noboth   both. The floor: memory traffic + reduction + barriers only.

include("ka_backend.jl")
include("pipeline_field.jl")

using FastMultipole, Printf, Statistics
const FM = FastMultipole
const V = FLOWVPM
import KernelAbstractions as KA

dev_functional() || (println("$(DEV_NAME) not functional; skipping"); exit(0))
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext === nothing && error("the KA extension is not loaded")

const STEP = 36
const P = 5
const CALLS = 15
const WG = 128

# ---- probe-local copies of the per-body math, with the two ablation switches.
@inline function p_setup(dx, dy, dz, ::Val{ALG}) where {ALG}
    TF = typeof(dx)
    rho = sqrt(dx * dx + dy * dy + dz * dz)
    rho == zero(TF) && return (rho, zero(TF), zero(TF), zero(TF), zero(TF))
    if ALG
        x = clamp(dz / rho, -one(TF), one(TF))
        y = sqrt(max(zero(TF), one(TF) - x * x))
        rxy = sqrt(dx * dx + dy * dy)
        if rxy == zero(TF)
            return (rho, x, y, zero(TF), one(TF))
        end
        return (rho, x, y, -dy / rxy, dx / rxy)
    else
        theta = acos(clamp(dz / rho, -one(TF), one(TF)))
        phi = atan(dy, dx)
        y, x = sincos(theta)
        iei_imag, iei_real = sincos(phi + convert(TF, pi / 2))
        return (rho, x, y, iei_real, iei_imag)
    end
end

@inline function p_q(setup, n, m, ::Val{REC}) where {REC}
    TF = typeof(setup[1])
    REC || return (setup[2] * setup[4], setup[3] * setup[5])
    return FM._resident_vortex_q(setup, n, m)
end

@inline function p_phi_contrib(mdx, mdy, mdz, vx, vy, vz, n, m, va::Val, vr::Val)
    TF = typeof(mdx)
    setup = p_setup(mdx, mdy, mdz, va)
    qmm1_re, qmm1_im = p_q(setup, n, m - 1, vr)
    qm_re, qm_im = p_q(setup, n, m, vr)
    qmp1_re, qmp1_im = p_q(setup, n, m + 1, vr)
    nmmp1_2 = TF(n - m + 1) * TF(0.5); npmp1_2 = TF(n + m + 1) * TF(0.5)
    _1_np1 = inv(TF(n + 1)); _1_m = isodd(m) ? -one(TF) : one(TF)
    re = _1_m * ((-vx * qmm1_re + vy * qmm1_im) * nmmp1_2 +
                 (vx * qmp1_re + vy * qmp1_im) * npmp1_2 - vz * m * qm_im) * _1_np1
    im = _1_m * ((vx * qmm1_im + vy * qmm1_re) * nmmp1_2 +
                 (-vx * qmp1_im + vy * qmp1_re) * npmp1_2 - vz * m * qm_re) * _1_np1
    return re, im
end

@inline function p_chi_contrib(mdx, mdy, mdz, vx, vy, vz, n, m, va::Val, vr::Val)
    TF = typeof(mdx)
    setup = p_setup(mdx, mdy, mdz, va)
    qmm1_re, qmm1_im = p_q(setup, n - 1, m - 1, vr)
    qm_re, qm_im = p_q(setup, n - 1, m, vr)
    qmp1_re, qmp1_im = p_q(setup, n - 1, m + 1, vr)
    _1_over_n = inv(TF(n)); _1_m = isodd(m) ? -one(TF) : one(TF)
    re = -_1_m * _1_over_n * (TF(0.5) * (-vy * qmm1_re - vx * qmm1_im +
        vy * qmp1_re - vx * qmp1_im) - vz * qm_re)
    im = -_1_m * _1_over_n * (TF(0.5) * (vy * qmm1_im - vx * qmm1_re -
        vy * qmp1_im - vx * qmp1_re) + vz * qm_im)
    return re, im
end

@kernel function p_b2m_kernel!(phi, chi, @Const(source_bodies), @Const(cell_centers),
        @Const(cell_ranges), @Const(leaf_to_node), P_phi, P_chi, ncell,
        ::Type{TF}, ::Val{W}, va::Val, vr::Val) where {TF,W}
    i_cell = @index(Group); tid = @index(Local)
    shre = @localmem TF (W,); shim = @localmem TF (W,)
    @inbounds begin
        first = cell_ranges[1, i_cell]; count = cell_ranges[2, i_cell]
        cx = cell_centers[1, i_cell]; cy = cell_centers[2, i_cell]
        cz = cell_centers[3, i_cell]
        node = leaf_to_node[i_cell]
        for n in 0:P_phi, m in 0:n
            acc_re = zero(TF); acc_im = zero(TF)
            k = first + tid - 1
            while k <= first + count - 1
                re_, im_ = p_phi_contrib(cx - source_bodies[1, k],
                    cy - source_bodies[2, k], cz - source_bodies[3, k],
                    source_bodies[5, k], source_bodies[6, k], source_bodies[7, k],
                    n, m, va, vr)
                acc_re += re_; acc_im += im_
                k += W
            end
            shre[tid] = acc_re; shim[tid] = acc_im
            @synchronize()
            s = W >> 1
            while s >= 1
                if tid <= s
                    shre[tid] += shre[tid + s]; shim[tid] += shim[tid + s]
                end
                @synchronize()
                s >>= 1
            end
            if tid == 1
                row = FM.flat_basis_index(n, m, 1)
                phi[row, node] = shre[1]; phi[row + 1, node] = shim[1]
            end
            @synchronize()
        end
        for n in 1:P_chi, m in 0:n
            acc_re = zero(TF); acc_im = zero(TF)
            k = first + tid - 1
            while k <= first + count - 1
                re_, im_ = p_chi_contrib(cx - source_bodies[1, k],
                    cy - source_bodies[2, k], cz - source_bodies[3, k],
                    source_bodies[5, k], source_bodies[6, k], source_bodies[7, k],
                    n, m, va, vr)
                acc_re += re_; acc_im += im_
                k += W
            end
            shre[tid] = acc_re; shim[tid] = acc_im
            @synchronize()
            s = W >> 1
            while s >= 1
                if tid <= s
                    shre[tid] += shre[tid + s]; shim[tid] += shim[tid + s]
                end
                @synchronize()
                s >>= 1
            end
            if tid == 1
                row = FM.flat_basis_index(n, m, 1)
                chi[row, node] = shre[1]; chi[row + 1, node] = shim[1]
            end
            @synchronize()
        end
    end
end

host = load_wake(STEP; TF=Float64, P=P)
NP = V.get_np(host)
dev = V.ParticleField(host.maxparticles, Float32; arraytype=devmatrix, np=host.np,
    fmm=V.FMM(; p=P+1, autotune_p=false, autotune_ncrit=false,
                autotune_reg_error=false, default_rho_over_sigma=1.0))
dev.particles .= devarray(Float32.(Array(host.particles)))
V.radix_fmm_settings!(dev; m2l_strategy=:concat)
V.UJ_fmm(dev)

st = V._radix_fmm_couplings[dev]
state = st.cache.state
orders = state.invariant_cache.basis_info.orders
backend = KA.get_backend(state.multipoles.phi)
ncell = state.counts.n_cells
TFv = eltype(state.multipoles.phi)

@printf("=== B2M ablation ===  np=%d  n_cells=%d  WG=%d  P_phi=%d P_chi=%d\n\n",
        NP, ncell, WG, orders.P_phi, orders.P_active)

function run_variant(alg::Bool, rec::Bool)
    k = p_b2m_kernel!(backend, WG)
    f() = begin
        fill!(state.multipoles.phi, zero(TFv)); fill!(state.multipoles.chi, zero(TFv))
        k(state.multipoles.phi, state.multipoles.chi, state.source_bodies,
          state.cell_centers, state.cell_ranges, state.grid.leaf_to_node,
          orders.P_phi, orders.P_active, ncell, TFv, Val(WG), Val(alg), Val(rec);
          ndrange=ncell*WG)
        KA.synchronize(backend)
    end
    f()
    ts = Float64[]
    for _ in 1:CALLS
        t0 = time_ns(); f(); push!(ts, (time_ns()-t0)/1e3)
    end
    (median(ts), Array(state.multipoles.phi), Array(state.multipoles.chi))
end

# reference: the shipping launcher
ext.ka_launch_b2m!(state; workgroup=WG); KA.synchronize(backend)
ref_phi = Array(state.multipoles.phi); ref_chi = Array(state.multipoles.chi)
tship = begin
    ts = Float64[]
    for _ in 1:CALLS
        t0=time_ns(); ext.ka_launch_b2m!(state; workgroup=WG); KA.synchronize(backend)
        push!(ts,(time_ns()-t0)/1e3)
    end
    median(ts)
end

relerr(a,b) = (d = sqrt(sum(abs2, a .- b)); n = sqrt(sum(abs2, b)); n == 0 ? d : d/n)

@printf("  %-12s %10s %8s %12s\n", "variant", "us", "vs full", "relerr(phi/chi)")
@printf("  %-12s %10.1f %8s %12s\n", "shipping", tship, "-", "-")
results = Dict{Symbol,Float64}()
for (name, alg, rec) in (("full", false, true), ("algsetup", true, true),
                         ("norecur", false, false), ("noboth", true, false))
    t, ph, ch = run_variant(alg, rec)
    results[Symbol(name)] = t
    e = @sprintf("%.2e / %.2e", relerr(ph, ref_phi), relerr(ch, ref_chi))
    @printf("  %-12s %10.1f %8.2fx %12s\n", name, t, results[:full]/t, e)
    flush(stdout)
end
println()
@printf("attribution of full=%.1f us:\n", results[:full])
@printf("  transcendental prologue : %8.1f us (%4.1f%%)\n",
        results[:full]-results[:algsetup], 100*(results[:full]-results[:algsetup])/results[:full])
@printf("  Legendre recurrence     : %8.1f us (%4.1f%%)\n",
        results[:full]-results[:norecur], 100*(results[:full]-results[:norecur])/results[:full])
@printf("  floor (traffic+reduce)  : %8.1f us (%4.1f%%)\n",
        results[:noboth], 100*results[:noboth]/results[:full])
@printf("  parts sum vs whole      : %8.1f us vs %.1f us\n",
        (results[:full]-results[:algsetup])+(results[:full]-results[:norecur])+results[:noboth],
        results[:full])
