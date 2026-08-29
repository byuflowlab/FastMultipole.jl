# Gate for step 6b: KA W-list multipole-to-target (M2T), 13-row hessian.
#
# Oracle is FastMultipole's OWN host M2T (`_host_m2t_pairs_kernel!`,
# src/translate_batched.jl), not a reimplementation. The host form walks node
# body ranges (`node_lo`/`node_hi`) while the device form goes through
# `leaf_slot_of` -> `cell_ranges`; those are equivalent because W targets are
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
    Random.seed!(9300+ci)   # per case: Metal draws from the task-local RNG per launch
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
    if lists.n_w == 0
        nskip[]+=1; println("  SKIP case $ci: no W pairs"); continue
    end
    opts=FM.CUDARadixLifecycleOptions(;precision=TF, body_type=FM.Point{FM.Vortex})
    state=ext.ka_radix_state(actx,build,devb,P,Val(true); options=opts, lists=lists, output_rows=13)
    o=state.invariant_cache.basis_info.orders
    # populate multipoles with a well-scaled pseudo-expansion; the evaluation,
    # not the expansion, is what is under test
    mp = rand(TF, size(state.multipoles.phi)...) .* TF(0.01)
    mc = rand(TF, size(state.multipoles.chi)...) .* TF(0.01)
    copyto!(state.multipoles.phi, mp); copyto!(state.multipoles.chi, mc)
    fill!(state.output, zero(TF))
    H = ext.ka_allocate_harmonics_scratch(DEV_BACKEND, TF, o.P_phi)
    try
        ext.ka_launch_adaptive_m2t!(state, lists, actx, H)
        KernelAbstractions.synchronize(DEV_BACKEND)
    catch e
        nfail[]+=1; println("  FAIL case $ci: kernel threw: ", sprint(showerror,e)[1:min(end,2500)]); continue
    end
    # host oracle
    nH2 = FM.harmonic_index(o.P_phi+2, o.P_phi+2)
    Hh = zeros(TF, 2, 1, nH2)
    hout = zeros(TF, size(state.output)...)
    FM._host_m2t_pairs_kernel!(hout, Array(state.source_bodies),
        Array(actx.bufs.node_lo), Array(actx.bufs.node_hi), Array(state.grid.node_centers),
        Array(lctx.bufs.w_targets), Array(lctx.bufs.w_sources), lists.n_w,
        mp, mc, Hh, o.P_phi, o.P_active, Val(true), Val(true))
    e=relerr(Array(state.output), hout)
    tol=1e-4
    if e<=tol
        npass[]+=1; println("  PASS  n=$n P=$P n_w=$(lists.n_w)  relerr=$(round(e,sigdigits=3))")
    else
        nfail[]+=1; println("  FAIL  n=$n P=$P n_w=$(lists.n_w)  relerr=$e (tol $tol)")
    end
end
println("\nKA M2T on $(DEV_NAME): $(npass[]) passed, $(nfail[]) failed, $(nskip[]) skipped")
nfail[]==0 || error("M2T gate failed")
npass[] > 0 || error("M2T gate vacuous: every case had n_w == 0")
println("✓✓✓ Step 6b (M2T) gate passed on $(DEV_NAME) ✓✓✓")
