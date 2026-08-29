# Gate for step 5: KA U-list direct nearfield (`:pairs` shape).
#
# Oracle is a host loop over the SAME pair list and the SAME per-pair math
# (`_direct_pair_ugh` / `_direct_pair_ug`, src/translate_batched_resident.jl),
# so this tests the traversal, the striding and the atomic accumulation, not
# the physics.
#
# Gated against the CPU, deliberately NOT against the CUDA kernel: the KA
# kernel uses a plain `inv(sqrt(r2))` where CUDA uses `_cuda_fast_rsqrt`, which
# makes CUDA the less accurate side. Comparing the two would measure CUDA's
# fast-math error, not this port's correctness.
include("ka_backend.jl")
include("ka_list_luts.jl")
using FastMultipole, Random
const FM = FastMultipole
if !dev_functional(); println("$(DEV_NAME) not functional; skipping"); exit(0); end
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext !== nothing || error("FastMultipoleKAExt did not load")

relerr(a,b) = (s=maximum(abs.(b)); d=maximum(abs.(a .- b)); s==0 ? d : d/s)

function host_nearfield(kern, sb, cell_ranges, dtgt, dsrc, npairs, nrow, nbody, ::Type{T}) where T
    out = zeros(T, nrow, nbody); hs = nrow >= 13; ghv = Val(:shipped)
    ep = FM._emits_potential(kern)
    for p in 1:npairs
        tc = dtgt[p]; sc = dsrc[p]
        tf = cell_ranges[1,tc]; tl = tf + cell_ranges[2,tc] - 1
        sf = cell_ranges[1,sc]; sl = sf + cell_ranges[2,sc] - 1
        for i in tf:tl
            xi,yi,zi = sb[1,i], sb[2,i], sb[3,i]
            for j in sf:sl
                i == j && continue
                dx = xi-sb[1,j]; dy = yi-sb[2,j]; dz = zi-sb[3,j]
                r2 = dx*dx+dy*dy+dz*dz
                r2 > zero(r2) || continue
                invr = inv(sqrt(r2))
                if hs
                    v = FM._direct_pair_ugh(kern, dx,dy,dz,r2,invr, sb, j, ghv)
                    ep && (out[1,i] += v[1])
                    for r in 2:13; out[r,i] += v[r]; end
                else
                    du,dgx,dgy,dgz = FM._direct_pair_ug(kern, dx,dy,dz,r2,invr, sb, j, ghv)
                    ep && (out[1,i] += du)
                    out[2,i]+=dgx; out[3,i]+=dgy; out[4,i]+=dgz
                end
            end
        end
    end
    return out
end

const CASES = [
    # n, K_max, ell_max, P, q, rows
    ( 200,  8, 5, 4, 3, 13),
    ( 400,  8, 5, 4, 3, 13),
    ( 400,  4, 5, 4, 1,  4),
    (2000, 16, 6, 4, 3, 13),
    (2000,  4, 6, 2, 3, 13),
]
npass=Ref(0); nfail=Ref(0)
for (ci,(n,K_max,ell_max,P,q,rows)) in pairs(CASES)
    Random.seed!(7700+ci)      # per case: Metal draws from the task-local RNG per launch
    TF=Float32
    positions=rand(TF,3,n); dpb=8
    sb=rand(TF,dpb,n); sb[1:3,:].=positions
    sb[4,:] .= TF(0.05)                 # smoothing radius: keep the vortex kernel well-posed
    nl=max(2*n÷K_max,16); leaf_capacity=10*nl+256; node_capacity=100*nl+256
    # The DTR pair frontier ALIASES the tree-build frontier scratch on the
    # adaptive context, so it has to be sized here, not on the lists context --
    # and it grows with node PAIRS, so 16*leaf_capacity (linear in n) overflows.
    actx=ext.ka_allocate_adaptive_context(DEV_BACKEND,TF,n;
        leaf_capacity, frontier_capacity=64*node_capacity, node_capacity)
    devb=devarray(sb)
    build=ext.ka_build_adaptive_tree!(actx,devarray(positions),ell_max,K_max,true,(TF(0),TF(0),TF(0)),TF(1))
    reach=1<<ell_max
    lut, lcls, noff = build_luts(reach,q,ell_max)
    cap=max(4096, 4*build.n_nodes^2)
    lctx=ext.ka_allocate_lists_context(actx, devarray(lut), devarray(lcls);
        u_capacity=cap, v_capacity=cap, wx_capacity=cap, lut_reach=reach,
        noffsets=noff, first_m2l_level=0, ell_max=ell_max,
        # the DTR frontier grows with NODE PAIRS, not with leaves: sizing it off
        # `leaf_capacity` (linear in n) overflows well before the U list does.
        leaf_capacity=leaf_capacity, maxn=n)
    lists=ext.ka_refresh_adaptive_lists!(lctx,actx,build;
        near_radius2=q, ell_max=ell_max, rho_t=0.0f0, sigma_armed=false)
    opts=FM.CUDARadixLifecycleOptions(;precision=TF, body_type=FM.Point{FM.Vortex})
    state=ext.ka_radix_state(actx,build,devb,P,Val(true); options=opts, lists=lists, output_rows=rows)

    try
        ext.ka_launch_nearfield!(state); KernelAbstractions.synchronize(DEV_BACKEND)
    catch e
        nfail[]+=1; println("  FAIL case $ci: kernel threw: ", sprint(showerror,e)[1:min(end,220)]); continue
    end

    np = state.counts.n_direct
    hout = host_nearfield(opts.direct_kernel, Array(state.source_bodies),
        Array(state.cell_ranges), Array(state.direct_targets), Array(state.direct_sources),
        np, rows, size(state.output,2), TF)
    e = relerr(Array(state.output), hout)
    tol = 1e-5
    if np == 0
        println("  SKIP case $ci: no direct pairs")
    elseif e <= tol
        npass[]+=1; println("  PASS  n=$n P=$P q=$q rows=$rows pairs=$np  relerr=$(round(e,sigdigits=3))")
    else
        nfail[]+=1; println("  FAIL  n=$n P=$P q=$q rows=$rows pairs=$np  relerr=$e (tol $tol)")
    end
end
println("\nKA nearfield on $(DEV_NAME): $(npass[]) passed, $(nfail[]) failed")
nfail[]==0 || error("nearfield gate failed")
println("✓✓✓ Step 5 (nearfield) gate passed on $(DEV_NAME) ✓✓✓")
