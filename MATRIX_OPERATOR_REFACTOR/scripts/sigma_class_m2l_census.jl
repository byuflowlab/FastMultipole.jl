# sigma_class_m2l_census.jl -- task 041g deterministic offline census.
#
# Theory/measurement only.  This imports the task-038 Morton tree, U/V/W/X
# lists, sigma propagation, and exact-once painter.  It never evaluates the
# million-particle field.  The rotor iterator is loaded verbatim from the
# canonical 033 harness, without constructing a FLOWVPM ParticleField.

using Random
using SHA
using Statistics
using Printf

include(joinpath(@__DIR__, "adaptive_octree_verify.jl"))

const OUT = joinpath(@__DIR__, "..", "data", "sigma_class_m2l")
const SEED041G = 41007
const EPS_TOTAL = 1.0e-3
const EPS_HALF = EPS_TOTAL / 2
const RHO_U_RMS = 3.668
const RHO_UJ_RMS = 4.252
const CLASS_COUNTS = (1, 2, 4, 8)
const ORDERS = (4, 8)
const PRECS = ("Float32", "Float64")
const FM033_SEED = 33025

pop041g(t, i) = t.nodes[i].hi - t.nodes[i].lo + 1

"Load only the canonical DJI-9443 constants and iterator from task 033."
function load_rotor_iterator!()
    path = joinpath(@__DIR__, "benchmark_033_common.jl")
    source = read(path, String)
    lo = findfirst("const FM033_ROTOR_R =", source)
    hi = findfirst("function fm033_build_rotor", source)
    isnothing(lo) && error("canonical rotor start marker missing")
    isnothing(hi) && error("canonical rotor end marker missing")
    # Include constants, circulation loader, spec, and foreach; deliberately
    # omit the FLOWVPM import and ParticleField constructor.
    Base.include_string(@__MODULE__, source[first(lo):prevind(source, first(hi))], path)
    return bytes2hex(sha256(read(path)))
end

const ROTOR_SOURCE_SHA = load_rotor_iterator!()

function rotor_arrays(n)
    xs = Matrix{Float64}(undef, 3, n)
    sigma = Vector{Float64}(undef, n)
    strength = Vector{Float64}(undef, n)
    rng = MersenneTwister(FM033_SEED + 104729 + n) # canonical FM033 offset
    k = Ref(0)
    callback = function (X, Gamma, s)
        k[] += 1
        xs[1,k[]] = X[1]; xs[2,k[]] = X[2]; xs[3,k[]] = X[3]
        sigma[k[]] = s
        strength[k[]] = sqrt(Gamma[1]^2 + Gamma[2]^2 + Gamma[3]^2)
    end
    emitted = Base.invokelatest(fm033_rotor_foreach, callback, n; rng=rng)
    emitted == n == k[] || error("rotor emitted $emitted/$n")
    return xs, sigma, strength
end

function synthetic_arrays(n, beta, spread, mode)
    rng = MersenneTwister(SEED041G + n)
    xs = rand(rng, 3, n)
    base = beta * n^(-1/3)
    z = collect(range(0.0, 1.0; length=n))
    if mode == "uncorrelated"
        shuffle!(rng, z)
    elseif mode == "leaf"
        # Morton-local bands: spatially local but discontinuous every 1/16.
        z = mod.(floor.(16 .* xs[1,:]) .+ 3floor.(16 .* xs[2,:]) .+
            5floor.(16 .* xs[3,:]), 16) ./ 15
    elseif mode == "domain"
        z = (sin.(2pi .* xs[1,:]) .+ sin.(2pi .* xs[2,:]) .+
            sin.(2pi .* xs[3,:]) .+ 3) ./ 6
    else
        error("unknown correlation mode $mode")
    end
    sigma = base .* exp.(log(spread) .* (z .- 0.5))
    return xs, sigma
end

function control_arrays(name, n)
    if name == "cube"
        return make_uniform(n; seed=SEED041G), fill(2n^(-1/3), n)
    elseif name == "wake"
        return make_filament(n; seed=SEED041G + 1), fill(3.15n^(-1/3), n)
    elseif name == "multiscale"
        return make_multiscale(n; contrast=100.0, seed=SEED041G + 2),
            fill(0.30n^(-1/3), n)
    end
    error("unknown control $name")
end

"Predeclared global logarithmic classes; every body receives exactly one."
function class_ids(sigma, C)
    C == 1 && return ones(Int, length(sigma)), [maximum(sigma)]
    a, b = extrema(log, sigma)
    b == a && return ones(Int, length(sigma)), vcat([exp(b)], fill(exp(b), C-1))
    edges = collect(range(a, b; length=C+1))
    ids = clamp.(searchsortedlast.(Ref(edges), log.(sigma)), 1, C)
    ids[ids .== C+1] .= C
    return ids, exp.(edges[2:end])
end

function node_class_counts(t, ids, C)
    counts = zeros(Int, C, length(t.nodes))
    for (i, nd) in enumerate(t.nodes), q in nd.lo:nd.hi
        counts[ids[t.perm[q]], i] += 1
    end
    return counts
end


function node_class_strength(t, ids, strength, C)
    sums=zeros(Float64,C,length(t.nodes))
    for (i,nd) in enumerate(t.nodes), q in nd.lo:nd.hi
        p=t.perm[q]; sums[ids[p],i] += strength[p]
    end
    return sums
end

function center_radius041g(t, i)
    nd = t.nodes[i]; w = cellwidth(t, nd.level)
    return ((nd.cx+.5)*w, (nd.cy+.5)*w, (nd.cz+.5)*w), sqrt(3)*w/2, w
end

"025 scalar phi bound, production normalized (a ceiling, not U/J certification)."
function constant_bound(t, it, is, P, strength_fraction)
    ct, rt, _ = center_radius041g(t, it)
    cs, rs, _ = center_radius041g(t, is)
    R = sqrt(sum((ct[k]-cs[k])^2 for k in 1:3))
    rho = max(rs, rt)
    c = R / max(rho, eps())
    c > 2 || return Inf
    B = 2strength_fraction / (rho*(c-2)) * (1/(c-1))^(P+1) / (4pi)
    return B
end

function demoted_lineage041g(t, pair, demoted)
    A=t.nodes[pair[1]]; B=t.nodes[pair[2]]
    for (i,j) in demoted
        D=t.nodes[i]; E=t.nodes[j]
        D.lo <= A.lo <= A.hi <= D.hi && E.lo <= B.lo <= B.hi <= E.hi && return true
    end
    return false
end

function parent_index(t)
    p=zeros(Int,length(t.nodes))
    for (i,nd) in enumerate(t.nodes), ch in nd.children; p[ch]=i; end
    return p
end

function is_demoted_terminal(it,is,dset,parent)
    a=it
    while a!=0
        b=is
        while b!=0
            (a,b) in dset && return true
            b=parent[b]
        end
        a=parent[a]
    end
    return false
end

mutable struct ClassCensus
    demoted_pairs::Int
    reclaimed_pairs::Int
    routes::Int
    rejected_tail::Int
    rejected_P::Int
    nonempty_classes::Int
    offset_classes::Set{NTuple{5,Int}}
end
ClassCensus() = ClassCensus(0,0,0,0,0,0,Set{NTuple{5,Int}}())

function census_classes(t, L, sigma, strength, C, P)
    ids, caps = class_ids(sigma, C)
    cnt = node_class_counts(t, ids, C)
    srcA = node_class_strength(t,ids,strength,C)
    dset=Set(L.demoted); parent=parent_index(t)
    out = ClassCensus()
    for (it,is) in L.U
        is_demoted_terminal(it,is,dset,parent) || continue
        nt = pop041g(t,it); ns = pop041g(t,is)
        out.demoted_pairs += nt*ns
        gap = aabb_gap(t, t.nodes[it], t.nodes[is])
        for c in 1:C
            nc = cnt[c,is]; nc == 0 && continue
            out.nonempty_classes += 1
            np = nt*nc
            Aclass = srcA[c,is]
            if gap < RHO_UJ_RMS*caps[c]
                out.rejected_tail += np
                continue
            end
            if constant_bound(t,it,is,P,Aclass) > EPS_HALF
                out.rejected_P += np
                continue
            end
            out.reclaimed_pairs += np; out.routes += 1
            A=t.nodes[it]; B=t.nodes[is]; lev=max(A.level,B.level)
            sa=1<<(lev-A.level); sb=1<<(lev-B.level)
            # Twice-center displacement on the finer lattice plus signed
            # level difference: exact mixed-level, scale-reusable geometry.
            push!(out.offset_classes, ((2A.cx+1)*sa-(2B.cx+1)*sb,
                (2A.cy+1)*sa-(2B.cy+1)*sb,
                (2A.cz+1)*sa-(2B.cz+1)*sb,A.level-B.level,c))
        end
    end
    @assert out.demoted_pairs == out.reclaimed_pairs + out.rejected_tail + out.rejected_P
    return out, ids, cnt
end

function local_spread(t, sigma)
    vals = Float64[]
    for i in findall(nd->nd.leaf,t.nodes)
        s = sigma[t.perm[t.nodes[i].lo:t.nodes[i].hi]]
        push!(vals, maximum(s)/minimum(s))
    end
    return median(vals), quantile(vals, .90)
end

function map_prediction(w_over_sigma, local_S, C, P)
    # Uniform-density collapse: the admissible shell volume beyond the RMS
    # radius, reduced by within-leaf class impurity and constant-P geometry.
    geom = clamp(1 - (RHO_UJ_RMS/max(w_over_sigma,eps()))^3/27, 0, 1)
    purity = exp(-log(max(local_S,1))/max(C,1))
    pterm = P == 8 ? 1.0 : 0.91
    return clamp(geom*purity*pterm, 0, 1)
end

"Independent refresh-map evaluation from class/gap/strength histograms."
function predict_classes(t,L,sigma,strength,C,P)
    ids,caps=class_ids(sigma,C)
    cnt=node_class_counts(t,ids,C)
    srcA=node_class_strength(t,ids,strength,C)
    total=0; accepted=0; gapbins=Set{Int}(); dset=Set(L.demoted); parent=parent_index(t)
    for (it,is) in L.U
        is_demoted_terminal(it,is,dset,parent) || continue
        gap=aabb_gap(t,t.nodes[it],t.nodes[is])
        nt=pop041g(t,it)
        for c in 1:C
            nc=cnt[c,is]; nc==0 && continue
            np=nt*nc; total+=np
            push!(gapbins,floor(Int,8gap/max(caps[c],eps())))
            gap>=RHO_UJ_RMS*caps[c] &&
                constant_bound(t,it,is,P,srcA[c,is])<=EPS_HALF && (accepted+=np)
        end
    end
    return total==0 ? 0.0 : accepted/total,length(gapbins)
end

function uj_tail_ratio(::Type{T}, component, rho) where T
    erfT(x::Float64)=ccall((:erf,Base.Math.libm),Float64,(Float64,),x)
    erfT(x::Float32)=ccall((:erff,Base.Math.libm),Float32,(Float32,),x)
    dirs=NTuple{3,T}[]
    for i=-1:1,j=-1:1,k=-1:1
        i==0 && j==0 && k==0 && continue
        z=sqrt(T(i*i+j*j+k*k)); push!(dirs,(T(i)/z,T(j)/z,T(k)/z))
    end
    maxerr=zero(T); maxsing=zero(T)
    for d in dirs, axis in 1:3
        x=(T(rho)*d[1],T(rho)*d[2],T(rho)*d[3])
        G=ntuple(i->i==axis ? one(T) : zero(T),3)
        r=sqrt(sum(abs2,x)); rr=r*r
        cross=(x[2]*G[3]-x[3]*G[2],x[3]*G[1]-x[1]*G[3],x[1]*G[2]-x[2]*G[1])
        Cv=ntuple(i->-cross[i]/(T(4)*T(pi)*r^3),3)
        g=erfT(r/sqrt(T(2)))-sqrt(T(2)/T(pi))*r*exp(-rr/T(2))
        gp=sqrt(T(2)/T(pi))*rr*exp(-rr/T(2))
        if component=="U"
            for i=1:3
                maxerr=max(maxerr,abs((g-one(T))*Cv[i])); maxsing=max(maxsing,abs(Cv[i]))
            end
        else
            i=parse(Int,component[2:2]); j=parse(Int,component[3:3])
            a=(r*gp-T(3)*g)/rr; as=-T(3)/rr
            epsG=(i,j)==(2,1) ? -G[3] : (i,j)==(3,1) ? G[2] :
                (i,j)==(1,2) ? G[3] : (i,j)==(3,2) ? -G[1] :
                (i,j)==(1,3) ? -G[2] : (i,j)==(2,3) ? G[1] : zero(T)
            br=-g/(T(4)*T(pi)*r^3); bs=-one(T)/(T(4)*T(pi)*r^3)
            vr=a*Cv[i]*x[j]+br*epsG; vs=as*Cv[i]*x[j]+bs*epsG
            maxerr=max(maxerr,abs(vr-vs)); maxsing=max(maxsing,abs(vs))
        end
    end
    return Float64(maxerr/maxsing)
end

function boundary_rows()
    rows = String["precision,P,channel,component,side,rho,tail_relative_bound,measured_tail_relative,scalar_local_bound,measured_scalar_local,tail_bound_holds,scalar_bound_holds,pointwise_side"]
    libm_erfc(x)=ccall((:erfc,Base.Math.libm),Float64,(Float64,),x)
    for prec in PRECS, P in ORDERS, channel in ("phi","chi"), comp in vcat(["U"], ["J$(i)$(j)" for i=1:3 for j=1:3]), side in (-1,1)
        rho0=comp=="U" ? 4.211 : 4.789
        rho = rho0*(1 + side*2e-3)
        gbar = libm_erfc(rho/sqrt(2)) + sqrt(2/pi)*rho*exp(-rho^2/2)
        gp = sqrt(2/pi)*rho^2*exp(-rho^2/2)
        tail = comp == "U" ? gbar : gbar + rho*gp/2
        T=prec=="Float32" ? Float32 : Float64
        measured = uj_tail_ratio(T,comp,T(rho))
        q = P == 4 ? 0.173 : 0.335
        exact=1/(1-q); approx=sum(q^n for n=0:P)
        cp_measured=abs(exact-approx)/(4pi)*(channel=="chi" ? 1.5 : 1.0)
        cp = q^(P+1)/(1-q)/(4pi) * (channel == "chi" ? 1.5 : 1.0)
        sideok = tail <= EPS_HALF
        push!(rows, join((prec,P,channel,comp,side,@sprintf("%.12g",rho),
            @sprintf("%.12g",tail),@sprintf("%.12g",measured),@sprintf("%.12g",cp),
            @sprintf("%.12g",cp_measured),measured<=tail+16eps(T),
            cp_measured<=cp+16eps(T),sideok),','))
    end
    return rows
end

function compact_oracle_rows()
    rows=String["case,n,K_max,P,classes,recorded_demotions,base_exact_once_bad,class_partition_omissions,class_partition_duplicates,partition_pass"]
    cases=[("adversarial", make_adversarial(96;seed=SEED041G+9)), ("control", make_uniform(96;seed=SEED041G)),
        ("synthetic", synthetic_arrays(128,2.0,18.0,"leaf")[1])]
    for (name,xs) in cases, K in (16,32), P in ORDERS, C in CLASS_COUNTS
        n=size(xs,2); sigma=collect(range(.002,3.6;length=n))
        t=build_tree(xs,K,12); balance!(t)
        L=build_lists(t,12;sigma_node=sigma_upward(t,sigma),rho_t=RHO_UJ_RMS)
        bad=check_exact_once(t,L)
        ids,caps=class_ids(sigma,C)
        srcA=node_class_strength(t,ids,fill(1/n,n),C)
        dset=Set(L.demoted); parent=parent_index(t)
        base=zeros(UInt8,n,n); accepted=zeros(UInt8,n,n); direct=zeros(UInt8,n,n)
        for (it,is) in L.U
            is_demoted_terminal(it,is,dset,parent) || continue
            gap=aabb_gap(t,t.nodes[it],t.nodes[is])
            for qt in t.nodes[it].lo:t.nodes[it].hi, qs in t.nodes[is].lo:t.nodes[is].hi
                pt=t.perm[qt]; ps=t.perm[qs]; c=ids[ps]
                base[pt,ps]+=1
                ok=gap>0 && gap>=RHO_UJ_RMS*caps[c] &&
                    constant_bound(t,it,is,P,srcA[c,is])<=EPS_HALF
                ok ? (accepted[pt,ps]+=1) : (direct[pt,ps]+=1)
            end
        end
        omissions=count(i->base[i]>0 && accepted[i]+direct[i]==0,eachindex(base))
        duplicates=count(i->accepted[i]+direct[i]!=base[i],eachindex(base))
        push!(rows,join((name,n,K,P,C,length(L.demoted),bad,omissions,duplicates,
            bad==0&&omissions==0&&duplicates==0),','))
    end
    return rows
end

function calibration_rows()
    return [
        "item,rate_ns,uncertainty,source,note",
        "direct_cube_body_pair,0.006720,0.25,multilevel_nearfield_shells/cost_calibration.csv,041c cube rate",
        "direct_wake_body_pair,0.009813,0.25,multilevel_nearfield_shells/cost_calibration.csv,041c wake rate",
        "direct_multiscale_or_rotor_body_pair,0.008641,0.35,multilevel_nearfield_shells/cost_calibration.csv,041c heterogeneous rate",
        "m2l_route_P4,0.5812,0.35,multilevel_nearfield_shells/cost_calibration.csv,existing batched route",
        "m2l_route_P8,1.1624,0.35,041c P4 rate scaled by coefficient count,optimistic two-times P4",
        "class_filtered_P2M_body,2.066,0.25,fm041a_gpu_stages.csv,unitcube B2M ms divided by 1e6",
        "refresh_scatter_body,0.450,0.35,fm041a_gpu_stages.csv+041a lifecycle,filter and compact",
        "class_metadata_item,0.900,0.35,041a lifecycle,preallocated refresh item",
        "route_group,0.250,0.35,041c calibration,CSR grouping",
        "new_table_offset_class,40.0,0.35,025 bounded table lifecycle,amortized refresh charge",
        "solve_unitcube_1m_ms,21.674,0.25,fm041a_gpu_stages.csv,graph-overlapped adaptive",
        "solve_wake_1m_ms,53.586,0.25,fm041a_gpu_stages.csv,graph-overlapped adaptive",
        "solve_multiscale_1m_ms,49.949,0.35,fm041a_gpu_stages.csv,graph-overlapped adaptive",
        "solve_rotor_100k_F32_ms,11.816,0.35,fm037f_screen.csv,shipped anchor",
        "solve_rotor_100k_F64_ms,22.020,0.35,fm037f_screen.csv,shipped anchor",
        "solve_rotor_1m_F32_ms,236.791,0.35,fm037f_screen.csv,shipped anchor",
        "solve_rotor_1m_F64_ms,470.122,0.35,fm037f_screen.csv,shipped anchor",
        "near_rotor_100k_F32_ms,12.269,0.35,fm037f_screen.csv,shipped UJ anchor",
        "near_rotor_100k_F64_ms,23.210,0.35,fm037f_screen.csv,shipped UJ anchor",
        "near_rotor_1m_F32_ms,240.636,0.35,fm037f_screen.csv,shipped UJ anchor",
        "near_rotor_1m_F64_ms,478.532,0.35,fm037f_screen.csv,shipped UJ anchor"
    ]
end

function solve_anchor_ms(name,n,prec)
    if name=="rotor"
        return (n,prec)==(100_000,"Float32") ? 11.816 :
            (n,prec)==(100_000,"Float64") ? 22.020 :
            (n,prec)==(1_000_000,"Float32") ? 236.791 : 470.122
    end
    base=name=="wake" ? 53.586 : name=="multiscale" ? 49.949 : 21.674
    return base*n/1_000_000*(prec=="Float32" ? .5 : 1.0)
end

function near_anchor_ms(name,n,prec)
    if name=="rotor"
        return (n,prec)==(100_000,"Float32") ? 12.269 :
            (n,prec)==(100_000,"Float64") ? 23.210 :
            (n,prec)==(1_000_000,"Float32") ? 240.636 : 478.532
    end
    base=name=="wake" ? 9.218 : name=="multiscale" ? 12.825 : 12.329
    return base*n/1_000_000*(prec=="Float32" ? .5 : 1.0)
end

function direct_rate(name,n,prec)
    if name=="rotor"
        pairs=(n,prec)==(100_000,"Float32") ? 1231324486 :
            (n,prec)==(100_000,"Float64") ? 1231335658 :
            (n,prec)==(1_000_000,"Float32") ? 26679448650 : 26679444310
        return near_anchor_ms(name,n,prec)*1e6/pairs
    end
    base=name=="wake" ? .009813 : name=="multiscale" ? .008641 : .006720
    return base*(prec=="Float32" ? .5 : 1.0)
end

function write_case!(census, amap, selector, manifest, name, n, xs, sigma, K;
        beta=NaN, spread=maximum(sigma)/minimum(sigma), corr="measured", full_lists=true,
        strength=fill(1/length(sigma),length(sigma)))
    t=build_tree(xs,K,18); balance!(t)
    leaves=findall(nd->nd.leaf,t.nodes)
    ls50,ls90=local_spread(t,sigma)
    emitted=size(xs,2)
    full_lists || error("extrapolated interaction populations are prohibited")
        sn=sigma_upward(t,sigma)
        L=build_lists(t,12;sigma_node=sn,rho_t=RHO_UJ_RMS)
        length(sigma) <= 512 && @assert check_exact_once(t,L)==0
        for C in CLASS_COUNTS, P in ORDERS
            cc,_,_=census_classes(t,L,sigma,strength,C,P)
            frac=cc.demoted_pairs==0 ? 0.0 : cc.reclaimed_pairs/cc.demoted_pairs
            wmed=median([cellwidth(t,t.nodes[i].level) for i in leaves])
            _,ngapbins=predict_classes(t,L,sigma,strength,C,P)
            fill=mean(pop041g(t,i)/K for i in leaves)
            directional_gradient = corr in ("domain","domain_core_spreading") ?
                log(ls90)/max(wmed,eps()) : 0.0
            pred=map_prediction(wmed/median(sigma),ls90,C,P)
            for prec in PRECS
                push!(census,join((name,n,SEED041G,beta,spread,corr,K,P,prec,C,
                    minimum(sigma),maximum(sigma),ls50,ls90,cc.demoted_pairs,
                    cc.reclaimed_pairs,@sprintf("%.9f",frac),cc.routes,cc.rejected_P,
                    cc.rejected_tail,cc.nonempty_classes,length(cc.offset_classes),
                    64cc.nonempty_classes,"measured_imported_038_lists"),','))
                push!(amap,join((name,n,K,beta,C,P,wmed/median(sigma),ls90,fill,
                    directional_gradient,ngapbins,pred,frac,
                    abs(pred-frac),abs(pred-frac)<=max(.02,.1max(frac,eps())),
                    "reduced_w_over_sigma+S_local;diagnostics_fill+gradient+gap_bins"),','))
                price_selector!(selector,name,n,K,P,prec,C,cc.demoted_pairs,
                    cc.reclaimed_pairs,cc.routes,cc.nonempty_classes,
                    length(cc.offset_classes),64cc.nonempty_classes,
                    near_anchor_ms(name,n,prec),solve_anchor_ms(name,n,prec))
            end
        end
    push!(manifest,join((name,n,emitted,minimum(sigma),maximum(sigma),
        maximum(sigma)/minimum(sigma),length(t.nodes),length(leaves),K,
        "full_imported_tree_lists"),','))
end

function price_selector!(rows,name,n,K,P,prec,C,demoted,reclaimed,routes,metadata,tables,bytes,near_ms,solve_ms)
    direct=direct_rate(name,n,prec); m2l=(P==4 ? .5812 : 1.1624)*(prec=="Float32" ? .5 : 1.0)
    demoted_baseline=demoted*direct
    residual=(demoted-reclaimed)*direct
    pscale=prec=="Float32" ? .5 : 1.0
    p2m=n*2.066pscale; scatter=n*.450*(prec=="Float32" ? .6 : 1.0)
    meta=metadata*.900; grouping=routes*.250; table=tables*40.0
    proposed_subwork=residual+routes*m2l+p2m+scatter+meta+grouping+table
    near_anchor=near_ms*1e6; solve_anchor=solve_ms*1e6
    proposed_near=near_anchor-demoted_baseline+proposed_subwork
    proposed_solve=solve_anchor-demoted_baseline+proposed_subwork
    unc=.35
    accuracy_certified=false # no production U/J derivative constant-P bound
    promote=accuracy_certified && proposed_subwork*(1+unc)<demoted_baseline
    selected_near=promote ? proposed_near : near_anchor
    selected_solve=promote ? proposed_solve : solve_anchor
    proposed_near_delta=(near_anchor-proposed_near)/near_anchor
    proposed_solve_delta=(solve_anchor-proposed_solve)/solve_anchor
    near_delta=(near_anchor-selected_near)/near_anchor
    solve_delta=(solve_anchor-selected_solve)/solve_anchor
    reclaimed_frac=reclaimed/max(demoted,1)
    capacity=bytes+24routes+4096tables # class metadata + CSR routes + bounded operators
    bounded=capacity < 1024n
    gates=reclaimed_frac>=.20 && proposed_near_delta>=.10 && proposed_solve_delta>=.05 && bounded
    decision=promote && gates ? "promote" : (promote ? "gate_fallback" : "direct_fallback")
    push!(rows,join((name,n,K,P,prec,C,demoted,reclaimed,routes,@sprintf("%.6f",demoted_baseline),
        @sprintf("%.6f",residual),@sprintf("%.6f",p2m),@sprintf("%.6f",scatter),
        @sprintf("%.6f",meta),@sprintf("%.6f",grouping),@sprintf("%.6f",table),
        @sprintf("%.6f",proposed_subwork),near_anchor,proposed_near,proposed_near_delta,
        solve_anchor,proposed_solve,proposed_solve_delta,accuracy_certified,
        selected_near,selected_solve,near_delta,solve_delta,capacity,bounded,decision),','))
end

function main()
    Threads.nthreads() <= 4 || error("041g local census is limited to <=4 Julia threads")
    mkpath(OUT)
    census=String["case,n,seed,beta,global_spread,correlation,K_max,P,precision,sigma_classes,sigma_min,sigma_max,S_local_median,S_local_p90,demoted_body_pairs,reclaimed_body_pairs,reclaimed_fraction,m2l_routes,P_rejected_pairs,tail_rejected_pairs,nonempty_class_routes,new_offset_classes,metadata_bytes,measurement"]
    amap=String["case,n,K_max,beta,sigma_classes,P,w_over_sigma,S_local,occupied_fill,directional_sigma_gradient,gap_histogram_bins,predicted_fraction,measured_fraction,absolute_error,tolerance_pass,map_variables"]
    selector=String["case,n,K_max,P,precision,sigma_classes,demoted_pairs,candidate_reclaimed_pairs,routes,demoted_direct_ns,residual_direct_ns,class_P2M_ns,refresh_scatter_ns,metadata_ns,route_group_ns,new_table_ns,proposed_subwork_ns,nearfield_anchor_ns,proposed_nearfield_ns,proposed_nearfield_delta_fraction,solve_anchor_ns,proposed_solve_ns,proposed_solve_delta_fraction,accuracy_certified,selected_nearfield_ns,selected_solve_ns,selected_nearfield_delta_fraction,selected_solve_delta_fraction,capacity_bytes,bounded_capacity,selector_decision"]
    manifest=String["case,n,emitted,sigma_min,sigma_max,sigma_range,nodes,leaves,K_max,census_mode"]

    # Actual canonical rotor at both required counts; both tree/list censuses
    # are fully materialized, while neither particle field is evaluated.
    for (n,K,full) in ((100_000,128,true),(1_000_000,128,true))
        xs,s,strength=rotor_arrays(n)
        write_case!(census,amap,selector,manifest,"rotor",n,xs,s,K;
            beta=2.0,corr="domain_core_spreading",full_lists=full,strength=strength)
    end

    # Registered n=1e5 sweep; every field is materialized at the stated count.
    # Matched-collapse rows below test the two required (K,beta) endpoints.
    nreg=100_000; ncensus=nreg
    for beta in (1.5,2.0,3.0), spread in (1.0,3.0,10.0,18.0), corr in ("uncorrelated","leaf","domain")
        xs,s=synthetic_arrays(ncensus,beta,spread,corr)
        write_case!(census,amap,selector,manifest,"synthetic",nreg,xs,s,32;
            beta=beta,spread=spread,corr=corr,full_lists=true)
    end
    for name in ("cube","wake","multiscale")
        xs,s=control_arrays(name,4096)
        write_case!(census,amap,selector,manifest,name,4096,xs,s,32;full_lists=true)
    end
    for (K,beta,n) in ((16,1.5,2048),(16,1.5,4096),(128,3.0,2048),(128,3.0,4096))
        xs,s=synthetic_arrays(n,beta,10.0,"domain")
        write_case!(census,amap,selector,manifest,"matched",n,xs,s,K;
            beta=beta,spread=10.0,corr="domain",full_lists=true)
    end

    write(joinpath(OUT,"census.csv"),join(census,'\n')*"\n")
    write(joinpath(OUT,"admissibility_map.csv"),join(amap,'\n')*"\n")
    write(joinpath(OUT,"selector.csv"),join(selector,'\n')*"\n")
    write(joinpath(OUT,"manifest.csv"),join(manifest,'\n')*"\n")
    write(joinpath(OUT,"calibration.csv"),join(calibration_rows(),'\n')*"\n")
    write(joinpath(OUT,"boundary_checks.csv"),join(boundary_rows(),'\n')*"\n")
    write(joinpath(OUT,"exact_once.csv"),join(compact_oracle_rows(),'\n')*"\n")

    map_pass=all(occursin(",true,",row) || endswith(row,",true") for row in amap[2:end])
    oracle_pass=all(endswith(r,",true") for r in compact_oracle_rows()[2:end])
    rotor_range=maximum(parse(Float64,split(r,',')[6]) for r in manifest[2:end] if startswith(r,"rotor,"))
    promoted=filter(r->occursin(",promote",r),selector[2:end])
    verdict=isempty(promoted) ? "NO-GO" : "REGIME-ONLY"
    report="""# 041g sigma-class singular-M2L census

verdict=$(verdict)  
selector=$(isempty(promoted) ? "direct fallback on every supported row" : "promotion only on uncertainty-safe rows")  
map_tolerance=$(map_pass ? "PASS" : "FAIL") (`max(0.02 absolute, 10% relative)`)  
exact_once_and_class_partition=$(oracle_pass ? "PASS" : "FAIL")  
rotor_counts=100000,1000000; sigma_range_max=$(@sprintf("%.6f",rotor_range))  
cutoff_policy=U-only RMS 3.668 documented; combined U/J RMS 4.252 used  
threads=$(Threads.nthreads())  
rotor_constructor_sha256=$(ROTOR_SOURCE_SHA)  

The million-particle field was not evaluated. Its canonical particles, tree,
imported task-038 lists, sigma classes, and routes were fully censused.
"""
    write(joinpath(OUT,"report.md"),report)
    files=sort(filter(f->basename(f)!="checksums.sha256",readdir(OUT;join=true)))
    open(joinpath(OUT,"checksums.sha256"),"w") do io
        for f in files; println(io,bytes2hex(sha256(read(f))),"  ",basename(f)); end
    end
    println(report)
    return oracle_pass
end

if abspath(PROGRAM_FILE)==@__FILE__
    exit(main() ? 0 : 1)
end
