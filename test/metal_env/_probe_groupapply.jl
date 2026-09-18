include("ka_backend.jl"); include("pipeline_field.jl")
using FastMultipole, Printf
using FastMultipole.StaticArrays
using LinearAlgebra: mul!
const FM = FastMultipole; const V = FLOWVPM
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
include(joinpath(@__DIR__, "pipeline_device_args.jl"))

function measured(f)
    f(); GC.gc(); st = Base.gc_num(); t0 = time_ns()
    v = f(); t1 = time_ns(); d = Base.GC_Diff(Base.gc_num(), st)
    (v, (t1-t0)/1e9, Base.gc_alloc_count(d), d.allocd)
end
rep(l,t,a,b) = @printf("  %-34s %8.4f s %12d allocs %10.2f MiB\n", l,t,a,b/2^20)

np = 512
pf_h = load_wake(36; np=np); pf_d = load_wake(36; np=np)
st = V.RadixFMMSettings(; precision=Float32, window_classes=256, m2l_strategy=:concat)
hcache = V._build_radix_fmm_cache(pf_h, st)
fmm!(pf_h, hcache)
dcache = build_ka_cache(ext, hcache, pf_d, hcache.ell)
state = dcache.state
ws = state.scratch
flush(stdout)

println("m2m_groups: ", length(ws.m2m_groups), "  l2l_groups: ", length(ws.l2l_groups))
println("group counts m2m: ", [g.count[] for g in ws.m2m_groups])
println("group counts l2l: ", [g.count[] for g in ws.l2l_groups])
flush(stdout)

# whole loops
(_,t,a,b) = measured(() -> (for g in ws.m2m_groups; ext.ka_resident_stage_group_apply!(state.multipoles, state.multipoles, g, ws, :m2m); end)); rep("m2m loop total",t,a,b)
(_,t,a,b) = measured(() -> (for g in ws.l2l_groups; ext.ka_resident_stage_group_apply!(state.locals, state.locals, g, ws, :l2l); end)); rep("l2l loop total",t,a,b)
flush(stdout)

# per-group m2m
for (i,g) in enumerate(ws.m2m_groups)
    (_,t,a,b) = measured(() -> ext.ka_resident_stage_group_apply!(state.multipoles, state.multipoles, g, ws, :m2m))
    rep("m2m group $i (n=$(g.count[]))",t,a,b)
end
flush(stdout)

# line-by-line on the biggest m2m group
g = argmax(x->x.count[], ws.m2m_groups)
println("\nline split, m2m group n=", g.count[])
dest = state.multipoles; src = state.multipoles; group = g; kind = :m2m
n = group.count[]; mult = true
ystk = ws.ystk_phi; Ur = ystk.mult_Ur; Vs = ystk.mult_Vs
ndof_phi = size(ws.aphi,1)
(_,t,a,b) = measured(function()
    FM._vector_prefix_view(group.source_idx, n); FM._vector_prefix_view(group.target_idx, n)
    FM._vector_prefix_view(group.phis, n); FM._vector_prefix_view(group.thetas, n)
    for A in (ws.aphi, ws.yphi, ws.zphi, ws.rphi, ws.cphi, ystk.Cy, ystk.Sy, ystk.G, ystk.G2)
        FM._matrix_col_view(A, n)
    end
    nothing
end); rep("views",t,a,b)
source_idx = FM._vector_prefix_view(group.source_idx, n)
target_idx = FM._vector_prefix_view(group.target_idx, n)
group_phis = FM._vector_prefix_view(group.phis, n)
group_thetas = FM._vector_prefix_view(group.thetas, n)
aphi = FM._matrix_col_view(ws.aphi,n); yphi = FM._matrix_col_view(ws.yphi,n)
zphi = FM._matrix_col_view(ws.zphi,n); rphi = FM._matrix_col_view(ws.rphi,n)
cphi = FM._matrix_col_view(ws.cphi,n)
C = FM._matrix_col_view(ystk.Cy,n); S = FM._matrix_col_view(ystk.Sy,n)
G = FM._matrix_col_view(ystk.G,n); G2 = FM._matrix_col_view(ystk.G2,n)
thetas_row = transpose(group_thetas)
(_,t,a,b) = measured(() -> (C .= cos.(ystk.nu .* thetas_row))); rep("C .= cos",t,a,b)
(_,t,a,b) = measured(() -> (S .= sin.(ystk.nu .* thetas_row))); rep("S .= sin",t,a,b)
(_,t,a,b) = measured(() -> ext.ka_gather_rotate_z!(aphi, src.phi, ws.phi_flat_idx, source_idx,
    ws.maps_phi.row_m, ws.maps_phi.row_ssign, ws.maps_phi.row_pair, group_phis, one(eltype(aphi)))); rep("ka_gather_rotate_z!",t,a,b)
(_,t,a,b) = measured(() -> ext.ka_stacked_y_dense!(yphi, aphi, Ur, Vs, C, S, G, G2, ndof_phi)); rep("ka_stacked_y_dense!",t,a,b)
(_,t,a,b) = measured(() -> mul!(zphi, group.phi_dense, yphi)); rep("mul!",t,a,b)
(_,t,a,b) = measured(() -> ext.ka_rotate_z_scatter_accumulate!(dest.phi, rphi, ws.phi_flat_idx, target_idx,
    ws.maps_phi.row_m, ws.maps_phi.row_ssign, ws.maps_phi.row_pair, group_phis)); rep("ka_rotate_z_scatter_acc!",t,a,b)
println("has_lh = ", size(dest.chi,1) > 0)
flush(stdout)
