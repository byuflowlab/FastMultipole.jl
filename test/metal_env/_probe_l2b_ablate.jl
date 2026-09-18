# Does L2B carry the same redundancy as B2M? `_resident_local_eval_flat_hessian`
# (the branch production takes -- output has >=13 rows) DOES hoist the
# transcendental prologue, but still runs a full O(n^2)
# `_resident_regular_harmonic_coeff` recurrence from scratch at each of its FOUR
# call sites, i.e. ~2x per (n,m) per body. Ablate the recurrence exactly as
# _probe_b2m_ablate.jl does.
include("ka_backend.jl")
include("pipeline_field.jl")
using FastMultipole, Printf, Statistics
const FM = FastMultipole
const V = FLOWVPM
import KernelAbstractions as KA
dev_functional() || (println("skipping"); exit(0))
const ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
const P = 5
const CALLS = 15
const WG = 64

@inline p_coeff(::Val{true}, setup, n, m) = FM._resident_regular_harmonic_coeff(setup, n, m)
@inline p_coeff(::Val{false}, setup, n, m) = (setup[2] * setup[4], setup[3] * setup[5])

@inline function p_local_eval_h(ph, ch, node, dx, dy, dz,
        P_phi, P_active, lhv::Val{LH}, ::Val{REC}) where {LH,REC}
    TF = eltype(ph)
    c = inv(TF(4) * TF(pi))
    u = zero(TF)
    vx = zero(TF); vy = zero(TF); vz = zero(TF)
    # one transcendental prologue per body, shared by both passes below
    setup = FM._resident_harmonic_setup(dx, dy, dz)
    @inbounds for n in 0:P_active
        rre, rim = p_coeff(Val(REC), setup, n, 0)
        if n <= P_phi && (!LH || n == 0)
            u += rre * FM._resident_flat_phi_re(ph, node, P_phi, n, 0) -
                 rim * FM._resident_flat_phi_im(ph, node, P_phi, n, 0)
        end
        vxr, vxi, vyr, vyi, vzr, vzi =
            FM._resident_gradient_coeff(ph, ch, node, P_phi, P_active, n, 0, lhv)
        vx += vxr * rre - vxi * rim
        vy += vyr * rre - vyi * rim
        vz += vzr * rre - vzi * rim
        for m in 1:n
            rre, rim = p_coeff(Val(REC), setup, n, m)
            if n <= P_phi && !LH
                u += 2 * (rre * FM._resident_flat_phi_re(ph, node, P_phi, n, m) -
                          rim * FM._resident_flat_phi_im(ph, node, P_phi, n, m))
            end
            vxr, vxi, vyr, vyi, vzr, vzi =
                FM._resident_gradient_coeff(ph, ch, node, P_phi, P_active, n, m, lhv)
            vx += 2 * (vxr * rre - vxi * rim)
            vy += 2 * (vyr * rre - vyi * rim)
            vz += 2 * (vzr * rre - vzi * rim)
        end
    end

    hxx = zero(TF); hxy = zero(TF); hxz = zero(TF)
    hyx = zero(TF); hyy = zero(TF); hyz = zero(TF)
    hzx = zero(TF); hzy = zero(TF); hzz = zero(TF)
    @inbounds for n in 0:(P_active - 1)
        rre, rim = p_coeff(Val(REC), setup, n, 0)
        # gradient coefficients at (n+1, 0) and (n+1, 1)
        g0x_r, g0x_i, g0y_r, g0y_i, g0z_r, g0z_i =
            FM._resident_gradient_coeff(ph, ch, node, P_phi, P_active, n + 1, 0, lhv)
        g1x_r, g1x_i, g1y_r, g1y_i, g1z_r, g1z_i =
            FM._resident_gradient_coeff(ph, ch, node, P_phi, P_active, n + 1, 1, lhv)
        hxx += -g1x_i * rre
        hyx += -g1x_r * rre
        hzx += -g0x_r * rre + g0x_i * rim
        hxy += -g1y_i * rre
        hyy += -g1y_r * rre
        hzy += -g0y_r * rre + g0y_i * rim
        hxz += -g1z_i * rre
        hyz += -g1z_r * rre
        hzz += -g0z_r * rre + g0z_i * rim
        for m in 1:n
            rre, rim = p_coeff(Val(REC), setup, n, m)
            amx_r, amx_i, amy_r, amy_i, amz_r, amz_i =
                FM._resident_gradient_coeff(ph, ch, node, P_phi, P_active, n + 1, m - 1, lhv)
            bmx_r, bmx_i, bmy_r, bmy_i, bmz_r, bmz_i =
                FM._resident_gradient_coeff(ph, ch, node, P_phi, P_active, n + 1, m, lhv)
            cmx_r, cmx_i, cmy_r, cmy_i, cmz_r, cmz_i =
                FM._resident_gradient_coeff(ph, ch, node, P_phi, P_active, n + 1, m + 1, lhv)
            # x column: ∂x, ∂y, ∂z of vx
            tr = -(amx_i + cmx_i) * TF(0.5); ti = (amx_r + cmx_r) * TF(0.5)
            hxx += 2 * (tr * rre - ti * rim)
            tr = (amx_r - cmx_r) * TF(0.5); ti = (amx_i - cmx_i) * TF(0.5)
            hyx += 2 * (tr * rre - ti * rim)
            hzx += 2 * (-bmx_r * rre + bmx_i * rim)
            # y column
            tr = -(amy_i + cmy_i) * TF(0.5); ti = (amy_r + cmy_r) * TF(0.5)
            hxy += 2 * (tr * rre - ti * rim)
            tr = (amy_r - cmy_r) * TF(0.5); ti = (amy_i - cmy_i) * TF(0.5)
            hyy += 2 * (tr * rre - ti * rim)
            hzy += 2 * (-bmy_r * rre + bmy_i * rim)
            # z column
            tr = -(amz_i + cmz_i) * TF(0.5); ti = (amz_r + cmz_r) * TF(0.5)
            hxz += 2 * (tr * rre - ti * rim)
            tr = (amz_r - cmz_r) * TF(0.5); ti = (amz_i - cmz_i) * TF(0.5)
            hyz += 2 * (tr * rre - ti * rim)
            hzz += 2 * (-bmz_r * rre + bmz_i * rim)
        end
    end
    return u * c, vx * c, vy * c, vz * c,
        hxx * c, hxy * c, hxz * c, hyx * c, hyy * c, hyz * c, hzx * c, hzy * c, hzz * c
end
@kernel function p_l2b_h_kernel!(output, @Const(source_bodies), @Const(cell_centers),
        @Const(cell_ranges), @Const(leaf_to_node), @Const(local_phi), @Const(local_chi),
        P_phi, P_active, ::Val{LHV}, ncell, ::Val{W}, vr::Val) where {LHV,W}
    cell = @index(Group); tid = @index(Local)
    @inbounds begin
        node = leaf_to_node[cell]
        first = cell_ranges[1, cell]
        last = first + cell_ranges[2, cell] - 1
        cx = cell_centers[1, cell]; cy = cell_centers[2, cell]; cz = cell_centers[3, cell]
        i = first + tid - 1
        while i <= last
            vals = p_local_eval_h(local_phi, local_chi, node,
                source_bodies[1, i] - cx, source_bodies[2, i] - cy,
                source_bodies[3, i] - cz, P_phi, P_active, Val(LHV), vr)
            Base.Cartesian.@nexprs 13 r -> (output[r, i] += vals[r])
            i += W
        end
    end
end

host = load_wake(36; TF=Float64, P=P)
dev = V.ParticleField(host.maxparticles, Float32; arraytype=devmatrix, np=host.np,
    fmm=V.FMM(; p=P+1, autotune_p=false, autotune_ncrit=false,
                autotune_reg_error=false, default_rho_over_sigma=1.0))
dev.particles .= devarray(Float32.(Array(host.particles)))
V.radix_fmm_settings!(dev; m2l_strategy=:concat)
V.UJ_fmm(dev)
state = V._radix_fmm_couplings[dev].cache.state
orders = state.invariant_cache.basis_info.orders
backend = KA.get_backend(state.output)
ncell = state.counts.n_cells
LH = state isa FastMultipole.DeviceResidentRadixState{<:Any,<:Any,true}

@printf("=== L2B (hessian branch) recurrence ablation ===\n")
@printf("np=%d n_cells=%d WG=%d LH=%s output rows=%d\n\n",
        V.get_np(dev), ncell, WG, LH, size(state.output,1))

function timed(f)
    f(); KA.synchronize(backend)
    ts = Float64[]
    for _ in 1:CALLS
        t0=time_ns(); f(); KA.synchronize(backend); push!(ts,(time_ns()-t0)/1e3)
    end
    median(ts)
end

k = p_l2b_h_kernel!(backend, WG)
variant(rec) = timed(() -> k(state.output, state.source_bodies, state.cell_centers,
    state.cell_ranges, state.grid.leaf_to_node, state.locals.phi, state.locals.chi,
    orders.P_phi, orders.P_active, Val(LH), ncell, Val(WG), Val(rec); ndrange=ncell*WG))

t_ship = timed(() -> ext.ka_launch_l2b!(state; workgroup=WG))
t_full = variant(true)
t_norec = variant(false)
@printf("  %-12s %10s\n", "variant", "us")
@printf("  %-12s %10.1f\n", "shipping", t_ship)
@printf("  %-12s %10.1f\n", "full(copy)", t_full)
@printf("  %-12s %10.1f   (wrong by construction)\n", "norecur", t_norec)
@printf("\n  Legendre recurrence : %.1f us = %.1f%% of L2B\n",
        t_full - t_norec, 100*(t_full - t_norec)/t_full)
