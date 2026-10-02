# Gate for the production KA M2M stage driver (`ka_resident_stage_group_apply!`,
# what `ka_lifecycle_body!` runs) over a real `ka_radix_cache_workspace`, against
# the host stage driver `_resident_stage_group_apply!` (src/translate_batched.jl)
# over the equivalent host workspace: same group, same random edges, same
# multipoles, so the comparison isolates the backend and nothing else.
include("ka_backend.jl")
using FastMultipole, Random, Test, LinearAlgebra
using StaticArrays: SVector

const FM = FastMultipole

if !dev_functional()
    println("$(DEV_NAME) not functional; skipping")
    exit(0)
end
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext !== nothing || error("FastMultipoleKAExt did not load")

relerr(a, b) = (d = maximum(abs.(a .- b)); s = maximum(abs.(b)); s == 0 ? d : d / s)

# A representative well-separated V-list offset set (2 <= max|component| <= 3).
function offset_set()
    offs = SVector{3,Int}[]
    for i in -3:3, j in -3:3, k in -3:3
        m = maximum(abs.((i, j, k)))
        2 <= m <= 3 && push!(offs, SVector(i, j, k))
    end
    return offs
end

const CASES = [
    # (P, lamb_helmholtz, nbatch, ell)
    (4, false,  64, 3),
    (4,  true,  64, 3),
    (8, false, 256, 3),
    (8,  true, 256, 3),
    (12, true, 512, 4),
]

npass = Ref(0); nfail = Ref(0)
for (case_i, (P, lh, nbatch, ell)) in pairs(CASES)
    Random.seed!(9100 + case_i)   # per case: Metal draws from the task-local RNG on every launch
    TF = Float32
    basis_info = FM.OperatorBasisInfo(FM.CompressedComplexBasis(), P, Val(lh))
    invariant = FM.OperatorInvariantCache(TF, basis_info)
    h0 = TF(1); max_cells = nbatch; max_nodes = 2 * nbatch
    route_capacity = 4 * nbatch; offs = offset_set()

    ws_dev = ext.ka_radix_cache_workspace(DEV_BACKEND, TF, basis_info, ell, h0,
        max_cells, max_nodes, route_capacity, offs, invariant)
    host_ex = FM.FlatCoefficientBuffer{TF,Matrix{TF},FM.CompressedComplexBasis,lh}(
        zeros(TF, basis_info.basis_dof_phi, 1),
        lh ? zeros(TF, basis_info.basis_dof_chi, 1) : zeros(TF, 0, 0), basis_info)
    ws_host = FM._radix_cache_workspace(TF, basis_info, host_ex, ell, h0, max_cells,
        max_nodes, route_capacity, offs, invariant,
        FM.ConcatenatedFixedZM2L(), FM.MaterializedYRotationM2L())

    isempty(ws_host.m2m_groups) && (println("  SKIP case $case_i: no m2m groups"); continue)

    # --- the workspace itself must build identically (pure host math) ---
    ok_ws = true
    for f in (:phi_flat_idx, :chi_flat_idx)
        ok_ws &= Array(getfield(ws_dev, f)) == getfield(ws_host, f)
    end
    ok_ws || (nfail[] += 1; println("  FAIL case $case_i: workspace flat-index mismatch"); continue)

    # --- host stage driver vs the production KA stage driver ---
    gi = min(1, length(ws_host.m2m_groups))
    gh = ws_host.m2m_groups[gi]; gd = ws_dev.m2m_groups[gi]
    m = min(length(gh.phis), nbatch)
    m == 0 && (println("  SKIP case $case_i: empty group"); continue)

    src_idx = rand(1:max_nodes, m); tgt_idx = rand(1:max_nodes, m)
    phis = rand(TF, m) .* TF(2π); thetas = rand(TF, m) .* TF(π)
    for (g, dev) in ((gh, false), (gd, true))
        cp(dst, v) = dev ? copyto!(dst, 1, v, 1, length(v)) : (dst[1:length(v)] .= v)
        cp(g.source_idx, src_idx); cp(g.target_idx, tgt_idx)
        cp(g.phis, phis); cp(g.thetas, thetas)
        g.count[] = m
    end

    dof_phi = basis_info.basis_dof_phi; dof_chi = lh ? basis_info.basis_dof_chi : 0
    mp = rand(TF, dof_phi, max_nodes); mc = lh ? rand(TF, dof_chi, max_nodes) : zeros(TF, 0, 0)

    host_buf = FM.FlatCoefficientBuffer{TF,Matrix{TF},FM.CompressedComplexBasis,lh}(
        copy(mp), copy(mc), basis_info)
    dphi = devarray(copy(mp)); dchi = devarray(copy(mc))
    dev_buf = FM.FlatCoefficientBuffer{TF,typeof(dphi),FM.CompressedComplexBasis,lh}(
        dphi, dchi, basis_info)

    local res_h, res_d
    try
        res_h = FM._resident_stage_group_apply!(host_buf, host_buf, gh, ws_host, :m2m)
        res_d = ext.ka_resident_stage_group_apply!(dev_buf, dev_buf, gd, ws_dev, :m2m)
        KernelAbstractions.synchronize(DEV_BACKEND)
    catch e
        nfail[] += 1
        println("  FAIL case $case_i (P=$P lh=$lh nbatch=$nbatch): driver threw")
        println("        ", sprint(showerror, e)[1:min(end, 300)])
        continue
    end

    ep = relerr(Array(res_d.phi), res_h.phi)
    ec = lh ? relerr(Array(res_d.chi), res_h.chi) : 0.0
    tol = 1e-4
    if ep <= tol && ec <= tol
        npass[] += 1
        println("  PASS  P=$P lh=$lh nbatch=$nbatch m=$m  relerr phi=$(round(ep,sigdigits=3)) chi=$(round(ec,sigdigits=3))")
    else
        nfail[] += 1
        println("  FAIL  P=$P lh=$lh nbatch=$nbatch m=$m  relerr phi=$ep chi=$ec (tol $tol)")
    end
end

println("\nProduction KA stage driver over $(DEV_NAME): $(npass[]) passed, $(nfail[]) failed")
nfail[] == 0 || error("production driver gate failed")
println("✓✓✓ Production KA stage driver gate passed on $(DEV_NAME) ✓✓✓")
