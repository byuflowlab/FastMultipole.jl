# Gate for step 6a: KA X-list source-to-local (S2L), vortex channel.
#
# Oracle is FastMultipole's OWN host S2L (`_host_s2l_vortex_pairs_kernel!`,
# src/translate_batched.jl), not a reimplementation. The host form walks node
# body ranges (`node_lo`/`node_hi`) while the device form goes through
# `leaf_slot_of` -> `cell_ranges`; those are equivalent because X sources are
# always leaves.
#
# The device kernel accumulates atomically across X pairs that share a target,
# so the summation order differs from the host's sequential walk -- expect a
# small order-dependent difference at Float32, not bit-exactness.
include("ka_backend.jl")
include("ka_list_luts.jl")
using FastMultipole, Random
const FM = FastMultipole
if !dev_functional(); println("$(DEV_NAME) not functional; skipping"); exit(0); end
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext !== nothing || error("FastMultipoleKAExt did not load")
relerr(a,b)=(s=maximum(abs.(b)); d=maximum(abs.(a .- b)); s==0 ? d : d/s)

const CASES = [
    # n, K_max, ell_max, P, q
    ( 200,  4, 5, 4, 3),
    ( 400,  4, 5, 4, 3),
    ( 400,  8, 5, 2, 3),
    (1000,  4, 6, 4, 3),
    (2000,  4, 6, 4, 3),
]
npass=Ref(0); nfail=Ref(0); nskip=Ref(0)
for (ci,(n,K_max,ell_max,P,q)) in pairs(CASES)
    Random.seed!(8800+ci)   # per case: Metal draws from the task-local RNG per launch
    TF=Float32
    positions=rand(TF,3,n); dpb=8; sb=rand(TF,dpb,n); sb[1:3,:].=positions
    nl=max(2*n÷K_max,16); leaf_capacity=10*nl+256; node_capacity=100*nl+256
    actx=ext.ka_allocate_adaptive_context(DEV_BACKEND,TF,n;
        leaf_capacity, frontier_capacity=64*node_capacity, node_capacity)
    devb=devarray(sb)
    build=ext.ka_build_adaptive_tree!(actx,devarray(positions),ell_max,K_max,true,(TF(0),TF(0),TF(0)),TF(1))
    reach=1<<ell_max; lut,lcls,noff = build_luts(reach,q,ell_max)
    cap=max(4096, 4*build.n_nodes^2)
    lctx=ext.ka_allocate_lists_context(actx, devarray(lut), devarray(lcls);
        u_capacity=cap, v_capacity=cap, wx_capacity=cap, lut_reach=reach,
        noffsets=noff, first_m2l_level=0, ell_max=ell_max,
        leaf_capacity=leaf_capacity, maxn=n)
    lists=ext.ka_refresh_adaptive_lists!(lctx,actx,build;
        near_radius2=q, ell_max=ell_max, rho_t=0.0f0, sigma_armed=false)
    if lists.n_x == 0
        nskip[]+=1; println("  SKIP case $ci: no X pairs"); continue
    end
    opts=FM.CUDARadixLifecycleOptions(;precision=TF, body_type=FM.Point{FM.Vortex})
    state=ext.ka_radix_state(actx,build,devb,P,Val(true); options=opts, lists=lists, output_rows=13)
    o=state.invariant_cache.basis_info.orders
    fill!(state.locals.phi, zero(TF)); fill!(state.locals.chi, zero(TF))
    H = ext.ka_allocate_harmonics_scratch(DEV_BACKEND, TF, o.P_phi)
    try
        ext.ka_launch_adaptive_s2l!(state, lists, actx, H)
        KernelAbstractions.synchronize(DEV_BACKEND)
    catch e
        nfail[]+=1; println("  FAIL case $ci: kernel threw: ", sprint(showerror,e)[1:min(end,2500)]); continue
    end
    # host oracle
    nH2 = FM.harmonic_index(o.P_phi+2, o.P_phi+2)
    Hh = zeros(TF, 2, 1, nH2)
    hlp = zeros(TF, size(state.locals.phi)...); hlc = zeros(TF, size(state.locals.chi)...)
    FM._host_s2l_vortex_pairs_kernel!(hlp, hlc, Array(state.source_bodies),
        Array(actx.bufs.node_lo), Array(actx.bufs.node_hi), Array(state.grid.node_centers),
        Array(lctx.bufs.x_targets), Array(lctx.bufs.x_sources), lists.n_x, Hh,
        o.P_phi, o.P_active)
    ep=relerr(Array(state.locals.phi), hlp); ec=relerr(Array(state.locals.chi), hlc)
    tol=1e-4
    if ep<=tol && ec<=tol
        npass[]+=1; println("  PASS  n=$n P=$P n_x=$(lists.n_x)  relerr phi=$(round(ep,sigdigits=3)) chi=$(round(ec,sigdigits=3))")
    else
        nfail[]+=1; println("  FAIL  n=$n P=$P n_x=$(lists.n_x)  relerr phi=$ep chi=$ec (tol $tol)")
    end
end
println("\nKA S2L on $(DEV_NAME): $(npass[]) passed, $(nfail[]) failed, $(nskip[]) skipped")
nfail[]==0 || error("S2L gate failed")
npass[] > 0 || error("S2L gate vacuous: every case had n_x == 0")
println("✓✓✓ Step 6a (S2L) gate passed on $(DEV_NAME) ✓✓✓")
