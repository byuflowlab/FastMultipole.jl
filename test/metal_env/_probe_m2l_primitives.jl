# Where does M2L's per-route cost go, on PRODUCTION geometry?
#
# Session 26 split the apply on the synthetic ell=4/n=4096 cache and found no
# hotspot (~17 passes, ~6 GB/s). Since then the concat window cache landed and
# steady state takes `ka_hierarchical_m2l_cached_concat!`: ONE apply over the
# whole route stream, LH=true (FLOWVPM is Lamb-Helmholtz), so the pass chain is
# the longer of the two branches in `ka_resident_m2l_concat_apply!`.
#
# This is an instrumented mirror of that driver on the real-wake cache: same
# calls, same order, a synchronize + timer per primitive. The syncs cost, so the
# primitive column sums to more than the un-instrumented apply -- read SHARES.
# Bytes per primitive are counted from the slabs each op reads+writes, giving a
# per-op effective bandwidth to separate bandwidth-bound passes from launch-
# bound ones.

include("ka_backend.jl")
include("pipeline_field.jl")

using FastMultipole, Printf, Statistics, LinearAlgebra, KernelAbstractions
const KA = KernelAbstractions
const FM = FastMultipole
const V = FLOWVPM
const ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)

dev_functional() || (println("$(DEV_NAME) not functional; skipping"); exit(0))
ext === nothing && error("the KA extension is not loaded")

const TF = Float32
const STEP = 36
const P = 5
const NP = parse(Int, get(ENV, "PROF_NP", "8192"))
const NTRIAL = parse(Int, get(ENV, "PROF_TRIALS", "20"))

function device_field(np)
    host = load_wake(STEP; np, TF, P)
    d = V.ParticleField(host.maxparticles, TF; arraytype=devmatrix, np=host.np,
        fmm=V.FMM(; p=P + 1, autotune_p=false, autotune_ncrit=false,
                    autotune_reg_error=false, default_rho_over_sigma=1.0))
    d.particles .= devarray(host.particles)
    V.radix_fmm_settings!(d; m2l_strategy=:concat)
    return d
end

# ---- instrumented mirror of ka_resident_m2l_concat_apply! -------------------
# Returns Dict(label => (ms, bytes)) accumulated over the chunk loop.

bytes(a) = length(a) * sizeof(eltype(a))

function instrumented_apply!(dest, src, ws, route_sources, route_targets, nroutes,
                             classes, acc, backend)
    plan = ws.m2l_concat
    LH = size(dest.chi, 1) > 0
    TFx = eltype(dest.phi)
    function step!(label, nbytes, f)
        t = time(); f(); KA.synchronize(backend)
        ms = (time() - t) * 1e3
        old = get(acc, label, (0.0, 0))
        acc[label] = (old[1] + ms, old[2] + nbytes)
        return nothing
    end

    for c0 in 1:plan.chunk:nroutes
        cols = c0:min(c0 + plan.chunk - 1, nroutes)
        n = length(cols)
        cls = @view classes[cols]
        phis = @view plan.col_phi[1:n]
        thetas = @view plan.col_theta[1:n]
        invr_col = @view plan.col_invr[1:n]
        step!("gather phi/theta/invr (3x)", 6 * bytes(phis) + 3 * bytes(cls), () -> begin
            ka_g = ext.ka_gather_values!
            ka_g(phis, plan.phis, cls); ka_g(thetas, plan.thetas, cls)
            ka_g(invr_col, plan.invrs, cls)
        end)
        invr_row = transpose(invr_col)
        rs_col = @view plan.col_r[1:n]
        if LH
            step!("gather r", 2 * bytes(rs_col) + bytes(cls),
                  () -> ext.ka_gather_values!(rs_col, plan.rs, cls))
        end
        rs_row = transpose(rs_col)

        src_cols = @view route_sources[cols]
        tgt_cols = @view route_targets[cols]
        aphi = @view plan.aphi[:, 1:n]; yphi = @view plan.yphi[:, 1:n]
        zphi = @view plan.zphi[:, 1:n]; rphi = @view plan.rphi[:, 1:n]
        ops_phi = plan.ops_phi
        ndof_phi = size(plan.aphi, 1)
        Gphi = @view ops_phi.G[:, 1:n]; G2phi = @view ops_phi.G2[:, 1:n]
        Cphi = @view ops_phi.Cy[:, 1:n]; Sphi = @view ops_phi.Sy[:, 1:n]
        sphi = @view ops_phi.scale[:, 1:n]
        thetas_row = transpose(thetas)

        step!("phi cos/sin(nu*theta)", 2 * bytes(Cphi) + 2 * bytes(Sphi), () -> begin
            Cphi .= cos.(ops_phi.nu .* thetas_row); Sphi .= sin.(ops_phi.nu .* thetas_row)
        end)
        step!("phi scale invr^rexp", bytes(sphi) + bytes(invr_col),
              () -> (sphi .= invr_row .^ plan.rexp_phi))
        step!("phi gather_rotate_z (src)", 2 * bytes(aphi) + bytes(phis),
              () -> ext.ka_gather_rotate_z!(aphi, src.phi, ws.phi_flat_idx, src_cols,
                        ws.maps_phi.row_m, ws.maps_phi.row_ssign, ws.maps_phi.row_pair,
                        phis, one(TFx)))
        step!("phi stacked_y (src)", bytes(aphi) + bytes(yphi) + bytes(Cphi) + bytes(Sphi),
              () -> ext.ka_stacked_y_dense!(yphi, aphi, ops_phi.yU_mult, ops_phi.yV_mult,
                        Cphi, Sphi, Gphi, G2phi, ndof_phi))
        step!("phi y *= scale", 2 * bytes(yphi) + bytes(sphi), () -> (yphi .*= sphi))
        step!("phi mul!(zD)", bytes(yphi) + bytes(zphi), () -> mul!(zphi, ops_phi.zD, yphi))
        step!("phi z *= scale", 2 * bytes(zphi) + bytes(sphi), () -> (zphi .*= sphi))
        ret_phi = zphi

        if LH
            ops_chi = plan.ops_chi
            ndof_chi = size(plan.achi, 1)
            Gchi = @view ops_chi.G[:, 1:n]; G2chi = @view ops_chi.G2[:, 1:n]
            Cchi = @view ops_chi.Cy[:, 1:n]; Schi = @view ops_chi.Sy[:, 1:n]
            schi = @view ops_chi.scale[:, 1:n]
            achi = @view plan.achi[:, 1:n]; ychi = @view plan.ychi[:, 1:n]
            zchi = @view plan.zchi[:, 1:n]; rchi = @view plan.rchi[:, 1:n]
            cphi = @view plan.cphi[:, 1:n]; cchi = @view plan.cchi[:, 1:n]
            lhgp = @view plan.lhgp[:, 1:n]; lhgu = @view plan.lhgu[:, 1:n]

            step!("chi cos/sin(nu*theta)", 2 * bytes(Cchi) + 2 * bytes(Schi), () -> begin
                Cchi .= cos.(ops_chi.nu .* thetas_row); Schi .= sin.(ops_chi.nu .* thetas_row)
            end)
            step!("chi scale invr^rexp", bytes(schi) + bytes(invr_col),
                  () -> (schi .= invr_row .^ plan.rexp_chi))
            step!("chi gather_rotate_z (src)", 2 * bytes(achi) + bytes(phis),
                  () -> ext.ka_gather_rotate_z!(achi, src.chi, ws.chi_flat_idx, src_cols,
                            ws.maps_chi.row_m, ws.maps_chi.row_ssign, ws.maps_chi.row_pair,
                            phis, one(TFx)))
            step!("chi stacked_y (src)", bytes(achi) + bytes(ychi) + bytes(Cchi) + bytes(Schi),
                  () -> ext.ka_stacked_y_dense!(ychi, achi, ops_chi.yU_mult, ops_chi.yV_mult,
                            Cchi, Schi, Gchi, G2chi, ndof_chi))
            step!("chi y *= scale", 2 * bytes(ychi) + bytes(schi), () -> (ychi .*= schi))
            step!("chi mul!(zD)", bytes(ychi) + bytes(zchi), () -> mul!(zchi, ops_chi.zD, ychi))
            step!("chi z *= scale", 2 * bytes(zchi) + bytes(schi), () -> (zchi .*= schi))
            step!("LH gather_rows (2x)", 2 * (bytes(lhgp) + bytes(lhgu)), () -> begin
                ext.ka_gather_rows!(lhgp, zchi, ws.maps_phi.row_pair)
                ext.ka_gather_rows!(lhgu, zchi, ws.maps_chi.row_up)
            end)
            step!("LH combine cphi/cchi", 2 * (bytes(cphi) + bytes(cchi)), () -> begin
                cphi .= zphi .+ (plan.lh_arow_unit .* rs_row) .* lhgp
                cchi .= zchi .+ (plan.lh_brow_unit .* rs_row) .* lhgu
            end)
            step!("chi stacked_y (loc)", bytes(cchi) + bytes(rchi) + bytes(Cchi) + bytes(Schi),
                  () -> ext.ka_stacked_y_dense!(rchi, cchi, ops_chi.yU_loc, ops_chi.yV_loc,
                            Cchi, Schi, Gchi, G2chi, ndof_chi))
            step!("chi rotate_z_scatter (dst)", 3 * bytes(rchi),
                  () -> ext.ka_rotate_z_scatter_accumulate!(dest.chi, rchi, ws.chi_flat_idx,
                            tgt_cols, ws.maps_chi.row_m, ws.maps_chi.row_ssign,
                            ws.maps_chi.row_pair, phis))
            ret_phi = cphi
        end

        step!("phi stacked_y (loc)", bytes(ret_phi) + bytes(rphi) + bytes(Cphi) + bytes(Sphi),
              () -> ext.ka_stacked_y_dense!(rphi, ret_phi, ops_phi.yU_loc, ops_phi.yV_loc,
                        Cphi, Sphi, Gphi, G2phi, ndof_phi))
        step!("phi rotate_z_scatter (dst)", 3 * bytes(rphi),
              () -> ext.ka_rotate_z_scatter_accumulate!(dest.phi, rphi, ws.phi_flat_idx,
                        tgt_cols, ws.maps_phi.row_m, ws.maps_phi.row_ssign,
                        ws.maps_phi.row_pair, phis))
    end
    return dest
end

med_ms(f, n, backend) = begin
    ts = Float64[]
    for _ in 1:n
        t = time(); f(); KA.synchronize(backend); push!(ts, (time() - t) * 1e3)
    end
    median(ts)
end

println("=== M2L primitive breakdown, np=$NP, P=$P, $NTRIAL trials ==="); flush(stdout)

d = device_field(NP)
V.UJ_fmm(d); V.UJ_fmm(d)
cache = V._radix_fmm_coupling!(d).cache
state = cache.state
hctx = state.interaction_list
ws = state.scratch
backend = KA.get_backend(state.output)
plan = hctx.apply_plan

nroutes = hctx.total_routes
@printf("levels %d:%d  K=%d  noffsets=%d  routes=%d  win_valid=%s  chunk=%d  LH=%s\n",
        hctx.first_m2l_level, hctx.ell, hctx.window_classes, hctx.noffsets,
        nroutes, hctx.win_valid, plan.chunk, size(state.locals.chi, 1) > 0)

t_step = med_ms(() -> V.UJ_fmm(d), NTRIAL, backend)
t_m2l  = med_ms(() -> ext.ka_launch_m2l!(state, ws), NTRIAL, backend)
@printf("step %8.2f ms    M2L %8.2f ms (%4.1f%% of step)    %6.3f us/route\n\n",
        t_step, t_m2l, 100 * t_m2l / t_step, 1e3 * t_m2l / nroutes)
flush(stdout)

# instrumented: accumulate over NTRIAL applies of the cached route stream
acc = Dict{String,Tuple{Float64,Int}}()
srcs = view(hctx.win_sources, 1:nroutes)
tgts = view(hctx.win_targets, 1:nroutes)
cls  = view(hctx.win_class, 1:nroutes)
instrumented_apply!(state.locals, state.multipoles, ws, srcs, tgts, nroutes, cls,
                    Dict{String,Tuple{Float64,Int}}(), backend)   # warm
for _ in 1:NTRIAL
    instrumented_apply!(state.locals, state.multipoles, ws, srcs, tgts, nroutes,
                        cls, acc, backend)
end

tot_ms = sum(v[1] for v in values(acc)) / NTRIAL
tot_gb = sum(v[2] for v in values(acc)) / NTRIAL / 2^30
@printf("%-30s %9s %7s %10s %9s\n", "primitive", "ms", "share", "GiB/apply", "GB/s")
for (label, (ms_sum, b_sum)) in sort(collect(acc); by=kv -> -kv[2][1])
    ms = ms_sum / NTRIAL; gib = b_sum / NTRIAL / 2^30
    @printf("%-30s %9.3f %6.1f%% %10.4f %9.1f\n", label, ms, 100 * ms / tot_ms, gib,
            gib * 2^30 / (ms * 1e-3) / 1e9)
end
@printf("%-30s %9.3f %6.1f%% %10.4f %9.1f\n", "TOTAL (instrumented)", tot_ms, 100.0,
        tot_gb, tot_gb * 2^30 / (tot_ms * 1e-3) / 1e9)
@printf("%-30s %9.3f          (instrumentation overhead %.1f%%)\n",
        "un-instrumented M2L", t_m2l, 100 * (tot_ms - t_m2l) / t_m2l)
flush(stdout)
