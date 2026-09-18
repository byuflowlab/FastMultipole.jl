# Where does the M2L constant live at ell=4 / n=4096?
#
# Session 15 left M2L at ~355 of a 407 ms lifecycle with the per-window sync
# already retired (12 windows, ~20 ms), so the remainder is the concat apply
# itself. This splits that apply into its primitives.
#
# Three levels of breakdown:
#   1. m2l total -> window generation vs. concat apply, summed over windows.
#   2. per window: nroutes, chunk count, apply ms.
#   3. per primitive inside `ka_resident_m2l_concat_apply!`, via an
#      instrumented mirror of that driver (same calls, same order, a
#      synchronize + timer around each). The syncs cost, so the primitive
#      column sums to more than the un-instrumented apply; read the SHARES,
#      not the absolute ms.
include("ka_backend.jl")
using FastMultipole, Random, Printf, LinearAlgebra
using FastMultipole.StaticArrays
using KernelAbstractions
const FM = FastMultipole
include(joinpath(@__DIR__, "..", "vortex.jl"))

dev_functional() || (println("$(DEV_NAME) not functional; skipping"); exit(0))
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext !== nothing || error("FastMultipoleKAExt did not load")

make_system(seed, n, TF) = (Random.seed!(seed);
    VortexParticles(rand(TF, 3, n), (randn(TF, 3, n) ./ TF(n)), zeros(TF, n);
        potential=zeros(TF, 13, n), gradient_stretching=zeros(TF, 6, n)))

function device_build_args(hcache)
    sp = hcache.policy; ell = hcache.ell
    tables, level_class_of, level_radii2, root_level, first_m2l_level =
        FM._hierarchical_scheduled_tables(sp, ell, hcache.ell_axes)
    class_level, class_offset, _ =
        FM._hierarchical_class_metadata(tables, ell, first_m2l_level)
    max_level_nodes = ell >= 2 ? maximum(
        (FM._radix_level_node_capacity(L, hcache.ell_axes, ell, hcache.max_cells)
         for L in first_m2l_level:ell); init=0) : 0
    return (; tables, level_class_of, level_radii2, root_level, first_m2l_level,
        class_level, class_offset, max_level_nodes)
end

const P, ell, n, wc, TF = 4, 4, 4096, 256, Float32
sys_h = make_system(6101, n, TF); sys_d = make_system(6101, n, TF)
opts = FM.CUDARadixLifecycleOptions(; precision=TF,
    m2l_strategy=FM.ConcatenatedFixedZM2L(), body_type=FM.Point{FM.Vortex})
hcache = RadixFMMCache(sys_h; expansion_order=P, ell=ell, window_classes=wc, options=opts)
fmm!(sys_h, hcache)
a = device_build_args(hcache); LH = typeof(hcache).parameters[2]
dcache = ext.ka_radix_cache_device_build(DEV_BACKEND, (sys_d,),
    hcache.expansion_order, ell, hcache.x_min, hcache.h0, hcache.max_n_bodies,
    hcache.options, hcache.policy, hcache.accepted_offsets,
    hcache.rejected_offsets, hcache.max_cells, hcache.max_nodes,
    hcache.route_capacity, hcache.direct_capacity,
    hcache.state.multipoles.basis_info, Val(LH);
    hierarchical_tables=a.tables, class_level=a.class_level,
    class_offset=a.class_offset, hierarchical_level_class_of=a.level_class_of,
    hierarchical_level_radii2=a.level_radii2,
    max_level_nodes=a.max_level_nodes, hessian=hcache.hessian,
    ell_axes=hcache.ell_axes, box_extent=hcache.box_extent,
    root_level=a.root_level, first_m2l_level=a.first_m2l_level)
switches = FM.DerivativesSwitch(FM.to_vector(false,1), FM.to_vector(true,1),
    FM.to_vector(false,1), (sys_d,))

st = dcache.state; hctx = st.interaction_list; ws = st.scratch
plan = hctx.apply_plan
B = DEV_BACKEND
sync() = KernelAbstractions.synchronize(B)

for _ in 1:3; ext.ka_radix_cache_device_step!(dcache, (sys_d,), switches); sync(); end

t(f, k=8) = (for _ in 1:2; f(); sync(); end;
    minimum(begin t0=time_ns(); f(); sync(); (time_ns()-t0)/1e6 end for _ in 1:k))

println("device = $DEV_NAME  P=$P ell=$ell n=$n wc=$wc  LH=$LH")
@printf("plan.chunk = %d  ndof_phi = %d  zD = %s  noffsets = %d  K = %d\n",
    plan.chunk, size(plan.aphi,1), string(size(plan.ops_phi.zD)),
    hctx.noffsets, hctx.window_classes)

m2l_total = t(() -> ext.ka_launch_m2l!(st, ws))
@printf("\nm2l total %8.2f ms\n", m2l_total)

# ---- 2. per-window generate vs apply ----------------------------------------
route_class = plan.route_class
windows = NamedTuple[]
for L in hctx.first_m2l_level:hctx.ell
    class_base = (L - hctx.first_m2l_level) * hctx.noffsets
    for fo in 1:hctx.window_classes:hctx.noffsets
        lo = min(fo + hctx.window_classes - 1, hctx.noffsets)
        push!(windows, (; L, fo, lo, class_base))
    end
end
@printf("\n%-3s %-3s %8s %8s %8s %8s\n", "L", "win", "nroutes", "chunks", "gen_ms", "apply_ms")
gen_sum = 0.0; app_sum = 0.0
for (i, w) in enumerate(windows)
    nr = ext.ka_hier_generate_window!(st, hctx, route_class, w.L, w.fo, w.lo, w.class_base)
    gms = t(() -> ext.ka_hier_generate_window!(st, hctx, route_class, w.L, w.fo, w.lo, w.class_base))
    nr == 0 && (@printf("%-3d %-3d %8d %8d %8.2f %8s\n", w.L, i, 0, 0, gms, "-");
                global gen_sum += gms; continue)
    st.counts.n_routes = nr
    ams = t(() -> ext.ka_resident_m2l_concat_apply!(st.locals, st.multipoles, ws,
        st.route_sources, st.route_targets, nr))
    nch = length(1:plan.chunk:nr)
    @printf("%-3d %-3d %8d %8d %8.2f %8.2f\n", w.L, i, nr, nch, gms, ams)
    global gen_sum += gms; global app_sum += ams
end
@printf("%-3s %-3s %8s %8s %8.2f %8.2f\n", "", "sum", "", "", gen_sum, app_sum)

# ---- 3. per-primitive inside the apply, on the biggest window ---------------
best = nothing; bestn = 0
for (i, w) in enumerate(windows)
    nr = ext.ka_hier_generate_window!(st, hctx, route_class, w.L, w.fo, w.lo, w.class_base)
    nr > bestn && (global bestn = nr; global best = w)
end
best === nothing && exit(0)
nr = ext.ka_hier_generate_window!(st, hctx, route_class, best.L, best.fo, best.lo, best.class_base)
st.counts.n_routes = nr
@printf("\nper-primitive on window L=%d nroutes=%d (syncs included; read shares)\n", best.L, nr)

const ACC = Dict{String,Float64}()
macro p(name, ex)
    quote
        sync(); local t0 = time_ns(); $(esc(ex)); sync()
        ACC[$name] = get(ACC, $name, 0.0) + (time_ns() - t0)/1e6
    end
end

function profiled_apply!(dest, src, ws, route_sources, route_targets, nroutes::Int)
    plan = ws.m2l_concat
    LH = size(dest.chi, 1) > 0
    TF = eltype(dest.phi)
    for c0 in 1:plan.chunk:nroutes
        cols = c0:min(c0 + plan.chunk - 1, nroutes); nn = length(cols)
        cls = @view plan.route_class[cols]
        phis = @view plan.col_phi[1:nn]; thetas = @view plan.col_theta[1:nn]
        invr_col = @view plan.col_invr[1:nn]
        @p "gather phi/theta/invr" begin
            ext.ka_gather_values!(phis, plan.phis, cls)
            ext.ka_gather_values!(thetas, plan.thetas, cls)
            ext.ka_gather_values!(invr_col, plan.invrs, cls)
        end
        invr_row = transpose(invr_col)
        if LH
            rs_col = @view plan.col_r[1:nn]
            @p "gather rs" ext.ka_gather_values!(rs_col, plan.rs, cls)
            rs_row = transpose(rs_col)
        end
        src_cols = @view route_sources[cols]; tgt_cols = @view route_targets[cols]
        aphi = @view plan.aphi[:, 1:nn]; yphi = @view plan.yphi[:, 1:nn]
        zphi = @view plan.zphi[:, 1:nn]; rphi = @view plan.rphi[:, 1:nn]
        ops_phi = plan.ops_phi; ndof_phi = size(plan.aphi, 1)
        Gphi = @view ops_phi.G[:, 1:nn]; G2phi = @view ops_phi.G2[:, 1:nn]
        Cphi = @view ops_phi.Cy[:, 1:nn]; Sphi = @view ops_phi.Sy[:, 1:nn]
        sphi = @view ops_phi.scale[:, 1:nn]
        thetas_row = transpose(thetas)
        @p "phi trig+scale" begin
            Cphi .= cos.(ops_phi.nu .* thetas_row)
            Sphi .= sin.(ops_phi.nu .* thetas_row)
            sphi .= invr_row .^ plan.rexp_phi
        end
        @p "phi rotate_z gather" ext.ka_gather_rotate_z!(aphi, src.phi, ws.phi_flat_idx, src_cols,
            ws.maps_phi.row_m, ws.maps_phi.row_ssign, ws.maps_phi.row_pair, phis, one(TF))
        @p "phi stacked_y (src)" ext.ka_stacked_y_dense!(yphi, aphi, ops_phi.yU_mult, ops_phi.yV_mult,
            Cphi, Sphi, Gphi, G2phi, ndof_phi)
        @p "phi scale mul" (yphi .*= sphi)
        @p "phi mul!(zD)" mul!(zphi, ops_phi.zD, yphi)
        @p "phi scale mul" (zphi .*= sphi)
        ret_phi = zphi
        if LH
            ops_chi = plan.ops_chi; ndof_chi = size(plan.achi, 1)
            Gchi = @view ops_chi.G[:, 1:nn]; G2chi = @view ops_chi.G2[:, 1:nn]
            Cchi = @view ops_chi.Cy[:, 1:nn]; Schi = @view ops_chi.Sy[:, 1:nn]
            schi = @view ops_chi.scale[:, 1:nn]
            achi = @view plan.achi[:, 1:nn]; ychi = @view plan.ychi[:, 1:nn]
            zchi = @view plan.zchi[:, 1:nn]; rchi = @view plan.rchi[:, 1:nn]
            cphi = @view plan.cphi[:, 1:nn]; cchi = @view plan.cchi[:, 1:nn]
            lhgp = @view plan.lhgp[:, 1:nn]; lhgu = @view plan.lhgu[:, 1:nn]
            @p "chi trig+scale" begin
                Cchi .= cos.(ops_chi.nu .* thetas_row)
                Schi .= sin.(ops_chi.nu .* thetas_row)
                schi .= invr_row .^ plan.rexp_chi
            end
            @p "chi rotate_z gather" ext.ka_gather_rotate_z!(achi, src.chi, ws.chi_flat_idx, src_cols,
                ws.maps_chi.row_m, ws.maps_chi.row_ssign, ws.maps_chi.row_pair, phis, one(TF))
            @p "chi stacked_y (src)" ext.ka_stacked_y_dense!(ychi, achi, ops_chi.yU_mult, ops_chi.yV_mult,
                Cchi, Schi, Gchi, G2chi, ndof_chi)
            @p "chi scale mul" (ychi .*= schi)
            @p "chi mul!(zD)" mul!(zchi, ops_chi.zD, ychi)
            @p "chi scale mul" (zchi .*= schi)
            @p "lh gather_rows + combine" begin
                ext.ka_gather_rows!(lhgp, zchi, ws.maps_phi.row_pair)
                ext.ka_gather_rows!(lhgu, zchi, ws.maps_chi.row_up)
                cphi .= zphi .+ (plan.lh_arow_unit .* rs_row) .* lhgp
                cchi .= zchi .+ (plan.lh_brow_unit .* rs_row) .* lhgu
            end
            @p "chi stacked_y (loc)" ext.ka_stacked_y_dense!(rchi, cchi, ops_chi.yU_loc, ops_chi.yV_loc,
                Cchi, Schi, Gchi, G2chi, ndof_chi)
            @p "chi scatter" ext.ka_rotate_z_scatter_accumulate!(dest.chi, rchi, ws.chi_flat_idx, tgt_cols,
                ws.maps_chi.row_m, ws.maps_chi.row_ssign, ws.maps_chi.row_pair, phis)
            ret_phi = cphi
        end
        @p "phi stacked_y (loc)" ext.ka_stacked_y_dense!(rphi, ret_phi, ops_phi.yU_loc, ops_phi.yV_loc,
            Cphi, Sphi, Gphi, G2phi, ndof_phi)
        @p "phi scatter" ext.ka_rotate_z_scatter_accumulate!(dest.phi, rphi, ws.phi_flat_idx, tgt_cols,
            ws.maps_phi.row_m, ws.maps_phi.row_ssign, ws.maps_phi.row_pair, phis)
    end
    return dest
end

profiled_apply!(st.locals, st.multipoles, ws, st.route_sources, st.route_targets, nr)
empty!(ACC)
const REP = 3
for _ in 1:REP
    profiled_apply!(st.locals, st.multipoles, ws, st.route_sources, st.route_targets, nr)
end
tot = sum(values(ACC))
for (k, v) in sort(collect(ACC); by = x -> -x[2])
    @printf("  %-26s %8.2f ms  %5.1f%%\n", k, v/REP, 100v/tot)
end
@printf("  %-26s %8.2f ms\n", "TOTAL (instrumented)", tot/REP)
