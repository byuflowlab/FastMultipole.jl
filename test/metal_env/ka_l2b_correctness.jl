# Gate for step 4: KA local-to-body evaluation (L2B), both the 4-row and the
# 13-row hessian variant (FLOWVPM runs hessian=true and takes the latter).
#
# Oracle is FastMultipole's OWN host L2B (`_host_l2b_kernel!` /
# `_host_l2b_hessian_kernel!`, src/translate_batched_resident.jl), not a
# reimplementation. Each body's value is computed independently of the thread
# mapping (no reduction), so this should be bit-exact, not merely close --
# hence the tolerance is exact equality with a documented float fallback.
include("ka_backend.jl")
using FastMultipole, Random
const FM = FastMultipole
if !dev_functional(); println("$(DEV_NAME) not functional; skipping"); exit(0); end
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext !== nothing || error("FastMultipoleKAExt did not load")

relerr(a,b) = (s=maximum(abs.(b)); d=maximum(abs.(a .- b)); s==0 ? d : d/s)

const CASES = [
    # n, K_max, ell_max, P, lh, rows
    ( 400,  8, 5, 4, true,  4),
    ( 400,  8, 5, 4, true, 13),
    (2000, 16, 6, 4, true, 13),
    (2000,  4, 6, 2, false,13),
    (5000, 32, 6, 6, true, 13),
    (   1,  4, 4, 2, true, 13),
]
npass=Ref(0); nfail=Ref(0)
for (ci,(n,K_max,ell_max,P,lh,rows)) in pairs(CASES)
    Random.seed!(6100+ci)          # per case: Metal draws from the task-local RNG per launch
    TF=Float32
    positions=rand(TF,3,n); sb=rand(TF,8,n); sb[1:3,:].=positions
    nl=max(2*n÷K_max,16)
    actx=ext.ka_allocate_adaptive_context(DEV_BACKEND,TF,n;
        leaf_capacity=10*nl+256, frontier_capacity=16*(10*nl+256), node_capacity=100*nl+256)
    build=ext.ka_build_adaptive_tree!(actx,devarray(positions),ell_max,K_max,true,(TF(0),TF(0),TF(0)),TF(1))
    opts=FM.CUDARadixLifecycleOptions(;precision=TF, body_type=FM.Point{FM.Vortex})
    state=ext.ka_radix_state(actx,build,devarray(sb),P,Val(lh); options=opts, output_rows=rows)

    # populate locals with a well-scaled pseudo-expansion (the evaluation, not
    # the expansion, is what is under test here)
    lp=rand(TF,size(state.locals.phi)...) .* TF(0.01)
    lc=lh ? rand(TF,size(state.locals.chi)...) .* TF(0.01) : zeros(TF,size(state.locals.chi)...)
    copyto!(state.locals.phi,lp); lh && copyto!(state.locals.chi,lc)

    try
        ext.ka_launch_l2b!(state); KernelAbstractions.synchronize(DEV_BACKEND)
    catch e
        nfail[]+=1; println("  FAIL case $ci: kernel threw: ", sprint(showerror,e)[1:min(end,200)]); continue
    end

    ncell=build.n_leaves; o=state.invariant_cache.basis_info.orders
    hout=zeros(TF,rows,size(state.output,2))
    hargs=(hout, Array(state.source_bodies), Array(state.cell_ranges),
           Array(state.cell_centers), Array(actx.grid.leaf_to_node), lp, lc,
           o.P_phi, o.P_active, Val(lh), ncell)
    rows >= 13 ? FM._host_l2b_hessian_kernel!(hargs...) : FM._host_l2b_kernel!(hargs...)

    got=Array(state.output); e=relerr(got,hout); exact = got == hout
    if exact || e <= 1e-6
        npass[]+=1
        println("  PASS  n=$n P=$P lh=$lh rows=$rows cells=$ncell  ", exact ? "bit-exact" : "relerr=$(round(e,sigdigits=3))")
    else
        nfail[]+=1; println("  FAIL  n=$n P=$P lh=$lh rows=$rows  relerr=$e")
    end
end
println("\nKA L2B on $(DEV_NAME): $(npass[]) passed, $(nfail[]) failed")
nfail[]==0 || error("L2B gate failed")
println("✓✓✓ Step 4 (L2B) gate passed on $(DEV_NAME) ✓✓✓")
