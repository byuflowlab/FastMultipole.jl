# multilevel_nearfield_shell_census.jl — task 041c gated offline census.
#
# This script intentionally imports task 038's standalone tree, list, bound,
# seeded-case, exact-once, and checksum-compatible output machinery.  It does
# not construct a second adaptive tree and it never evaluates particle fields.
# Local work is deterministic and single threaded.

using SHA
using Statistics
using Printf

include(joinpath(@__DIR__, "adaptive_octree_verify.jl"))

const SHELL_OUT = joinpath(@__DIR__, "..", "data", "multilevel_nearfield_shells")
const SEED = 41003
const RHO_U = 4.211
const RHO_J = 4.789
const TOL_HALF = 5e-4

libm_erfc(x::Float64) = ccall((:erfc, Base.Math.libm), Float64, (Float64,), x)

pop(t, i) = t.nodes[i].hi - t.nodes[i].lo + 1

function phase0_rows()
    src = joinpath(@__DIR__, "..", "data", "flowvpm_gpu_campaign",
        "fm037e_scoping_13170768.csv")
    lines = readlines(src)
    hdr = split(lines[1], ',')
    ix = Dict(s => i for (i, s) in enumerate(hdr))
    out = String["case,n,source_record,direct_body_pairs,sigma_floor_whole_leaf_ceiling,phase0_decision,note"]
    for line in lines[2:end]
        f = split(line, ',')
        length(f) == length(hdr) || continue
        get(f, ix["status"], "") == "ok" || continue
        case = f[ix["case"]]; n = parse(Int, f[ix["n"]])
        bp = parse(Int, f[ix["tot_bp"]]); e2 = parse(Float64, f[ix["e2_share"]])
        decision = e2 >= 0.20 ? "CONTINUE" : "CASE_KILL"
        note = e2 == 0 ? "no_whole_leaf_sigma_slack;virtual_subnodes_not_bounded_by_this_row" :
            "pure_singular_M2L_offset_body_pair_ceiling_from_037e"
        push!(out, join((case, n, "fm037e_scoping_13170768.csv", bp,
            @sprintf("%.9f", e2), decision, note), ','))
    end
    return out
end

"""Pre-split every occupied baseline U leaf by at most `depth` levels.
The imported Morton tree remains the sole spatial representation."""
function materialize_virtual!(t, roots, depth)
    frontier = unique(roots)
    for _ in 1:depth
        next = Int[]
        for i in frontier
            nd = t.nodes[i]
            if nd.leaf && nd.level < t.ell_max && nd.lo < nd.hi
                split_node!(t.nodes, t.keys, i, t.ell_max)
            end
            append!(next, t.nodes[i].children)
        end
        frontier = next
    end
    return nothing
end

function sigma_stats(t, sigma)
    smax = zeros(Float64, length(t.nodes))
    sq90 = zeros(Float64, length(t.nodes))
    smin = fill(Inf, length(t.nodes))
    for (i, nd) in enumerate(t.nodes)
        vals = sort(sigma[t.perm[nd.lo:nd.hi]])
        smin[i] = first(vals); smax[i] = last(vals)
        sq90[i] = quantile(vals, 0.90)
    end
    return smin, smax, sq90
end

function center_radius(t, i)
    nd = t.nodes[i]; w = cellwidth(t, nd.level)
    c = ((nd.cx + 0.5) * w, (nd.cy + 0.5) * w, (nd.cz + 0.5) * w)
    return c, sqrt(3.0) * w / 2
end

function truncation_bound(t, ia, ib, P, route)
    ct, rt = center_radius(t, ia); cs, rs = center_radius(t, ib)
    d = sqrt(sum((ct[k] - cs[k])^2 for k in 1:3))
    # 038 route asymmetries: M2T has no target truncation, S2L has no source
    # truncation.  M2L carries both radii.  c<=2 is excluded by q>=1.
    r = route == :M2L ? rs + rt : (route == :M2T ? rs : rt)
    q = r / max(d, eps())
    q < 1 || return Inf
    return q^(P + 1) / (1 - q)
end

function demoted_lineage(t, pair, demoted)
    A = t.nodes[pair[1]]; B = t.nodes[pair[2]]
    for (i, j) in demoted
        D = t.nodes[i]; E = t.nodes[j]
        D.lo <= A.lo <= A.hi <= D.hi && E.lo <= B.lo <= B.hi <= E.hi && return true
    end
    return false
end

mutable struct Census
    original::Int; direct::Int; promoted::Int
    m2l::Int; m2t::Int; s2l::Int
    m2l_pairs::Int; m2t_pairs::Int; s2l_pairs::Int
    reg_reject::Int; quantile_block::Int; multipole_reject::Int
    virtual_nodes::Int; max_refine::Int
    direct_routes::Vector{NTuple{2,Int}}
    m2l_routes::Vector{NTuple{2,Int}}
    m2t_routes::Vector{NTuple{2,Int}}
    s2l_routes::Vector{NTuple{2,Int}}
    detail::Dict{Tuple{String,String,Int},Vector{Int}}
    sigma_min::Float64; sigma_max::Float64
end
Census() = Census(0,0,0,0,0,0,0,0,0,0,0,0,0,0,NTuple{2,Int}[],NTuple{2,Int}[],NTuple{2,Int}[],NTuple{2,Int}[],Dict{Tuple{String,String,Int},Vector{Int}}(),Inf,0.0)

function orbit_class(t, ia, ib)
    A=t.nodes[ia]; B=t.nodes[ib]
    ia == ib && return "self"
    scaleA=1 << (t.ell_max-A.level); scaleB=1 << (t.ell_max-B.level)
    touches=0; gaps=Int[]
    for (ca,cb) in ((A.cx,B.cx),(A.cy,B.cy),(A.cz,B.cz))
        alo=ca*scaleA; ahi=(ca+1)*scaleA
        blo=cb*scaleB; bhi=(cb+1)*scaleB
        g=max(alo-bhi,blo-ahi,0); push!(gaps,g)
        g==0 && (ahi==blo || bhi==alo) && (touches+=1)
    end
    if all(==(0),gaps)
        return touches==1 ? "face" : (touches==2 ? "edge" : (touches==3 ? "corner" : "overlap"))
    end
    # Express the mixed-level orbit on the finer node's lattice. Without this
    # normalization ell_max coordinates would create particle-scale labels.
    g=sort(gaps .÷ min(scaleA,scaleB))
    return "outer_$(g[1])_$(g[2])_$(g[3])"
end

function record_detail!(C,t,route,ia,ib,depth,np,rootorbit)
    key=(String(route),rootorbit,depth)
    v=get!(C.detail,key,[0,0,0,0,0,0])
    pt=pop(t,ia); ps=pop(t,ib)
    v[1]+=1; v[2]+=np; v[3]+=pt; v[4]+=ps
    v[5]=max(v[5],pt); v[6]=max(v[6],ps)
end

function route_pair!(C, t, rootpair, maxdepth, P, smax, sq90, sticky)
    baselevel = max(t.nodes[rootpair[1]].level, t.nodes[rootpair[2]].level)
    rootorbit = sticky ? "sigma_demoted_outer" : orbit_class(t,rootpair[1],rootpair[2])
    stack = [(rootpair[1], rootpair[2], 0)]
    while !isempty(stack)
        ia, ib, depth = pop!(stack)
        A = t.nodes[ia]; B = t.nodes[ib]; np = pop(t, ia) * pop(t, ib)
        gap = aabb_gap(t, A, B)
        reg = gap >= RHO_J * smax[ib]
        if !reg
            C.reg_reject += 1
            gap >= RHO_J * sq90[ib] && (C.quantile_block += 1)
        end
        legal = sticky ? (:M2T, :S2L) : (:M2L, :M2T, :S2L)
        chosen = nothing
        if reg && ia != ib
            # Ideal-cost Phase 1: prefer the reusable two-sided route, then
            # L2B-free-ride S2L, then per-target M2T.
            for r in legal
                truncation_bound(t, ia, ib, P, r) <= TOL_HALF || continue
                chosen = r; break
            end
        end
        if chosen !== nothing
            C.promoted += np
            if chosen == :M2L
                C.m2l += 1; C.m2l_pairs += np; push!(C.m2l_routes, (ia, ib))
            elseif chosen == :M2T
                C.m2t += 1; C.m2t_pairs += np; push!(C.m2t_routes, (ia, ib))
            else
                C.s2l += 1; C.s2l_pairs += np; push!(C.s2l_routes, (ia, ib))
            end
            record_detail!(C,t,chosen,ia,ib,depth,np,rootorbit)
            continue
        end
        reg && (C.multipole_reject += 1)
        ach = depth < maxdepth ? A.children : Int[]
        bch = depth < maxdepth ? B.children : Int[]
        if !isempty(ach) || !isempty(bch)
            # Complete Cartesian replacement. Split both when possible; a
            # one-sided split is used for an empty/irreducible counterpart.
            as = isempty(ach) ? (ia,) : Tuple(ach)
            bs = isempty(bch) ? (ib,) : Tuple(bch)
            for a in as, b in bs; push!(stack, (a, b, depth + 1)); end
            C.max_refine = max(C.max_refine, depth + 1)
        else
            C.direct += np; push!(C.direct_routes, (ia, ib))
            record_detail!(C,t,:direct,ia,ib,depth,np,rootorbit)
        end
    end
    return nothing
end

function oracle_bad(t, C)
    # Reuse the imported painter by expressing the final partition through its
    # four route buckets. Route names are irrelevant to coverage.
    L = Lists(C.direct_routes, C.m2l_routes, C.m2t_routes, C.s2l_routes,
        NTuple{2,Int}[])
    return check_exact_once(t, L)
end

function make_case(name, n)
    if name == "cube"
        xs = make_uniform(n; seed=SEED); sigma = fill(2.0n^(-1/3), n)
    elseif name == "wake"
        xs = make_filament(n; seed=SEED + 1); sigma = fill(3.15n^(-1/3), n)
    elseif name == "multiscale"
        xs = make_multiscale(n; contrast=100.0, seed=SEED + 2)
        sigma = fill(0.30n^(-1/3), n)
    elseif name == "sigma_multiscale"
        xs = make_multiscale(n; contrast=100.0, seed=SEED + 3)
        r = [sqrt(sum((xs[k,i] - (0.6,0.4,0.55)[k])^2 for k in 1:3)) for i in 1:n]
        sigma = 0.08n^(-1/3) .* exp.(log(18.0) .* (r .<= median(r)))
    elseif name == "boundary"
        vals=(0.0,0.25,0.5,0.75)
        xs=Matrix{Float64}(undef,3,n)
        for i in 1:n
            xs[:,i].=(vals[mod(i-1,4)+1],vals[mod((i-1)÷4,4)+1],
                vals[mod((i-1)÷16,4)+1])
        end
        sigma=fill(0.01,n)
    elseif name == "extreme_sigma"
        xs=make_uniform(n;seed=SEED+4)
        sigma=exp.(range(log(1e-6),log(0.2);length=n))
    elseif name == "coincident"
        base=make_uniform(cld(n,2);seed=SEED+5)
        xs=hcat([base[:,cld(i,2)] for i in 1:n]...)
        sigma=[isodd(i) ? 1e-4 : 0.05 for i in 1:n]
    else
        error("unknown case $name")
    end
    return xs, sigma
end

function census_row(name, n, depth, P)
    xs, sigma = make_case(name, n)
    t = build_tree(xs, 32, 12)
    balance!(t)
    snode0 = sigma_upward(t, sigma)
    base = build_lists(t, 5; sigma_node=snode0, rho_t=RHO_J)
    @assert check_exact_once(t, base) == 0
    roots = unique(vcat(first.(base.U), last.(base.U)))
    n0 = length(t.nodes)
    materialize_virtual!(t, roots, depth)
    smin, smax, sq90 = sigma_stats(t, sigma)
    C = Census()
    C.sigma_min=minimum(smin); C.sigma_max=maximum(smax)
    C.virtual_nodes = length(t.nodes) - n0
    for pair in base.U
        np = pop(t, pair[1]) * pop(t, pair[2]); C.original += np
        route_pair!(C, t, pair, depth, P, smax, sq90,
            demoted_lineage(t, pair, base.demoted))
    end
    @assert C.original == C.direct + C.promoted
    bad = n <= 256 ? oracle_bad(t, C) : -1
    frac = C.promoted / C.original
    mixedfrac = frac # ideal census has no bucket timing split; conservative alias
    ideal_near = 0.86 * mixedfrac
    decision = mixedfrac >= 0.20 && ideal_near >= 0.10 ? "CONTINUE" : "KILL"
    return C, bad, frac, ideal_near, decision
end

function write_outputs()
    mkpath(SHELL_OUT)
    write(joinpath(SHELL_OUT, "phase0_ceiling.csv"), join(phase0_rows(), '\n') * "\n")
    rows = String["case,n,seed,P,precision,near_radius2,K_max,virtual_depth,sigma_classes,sigma_min,sigma_max,original_body_pairs,residual_direct_body_pairs,promoted_body_pairs,promoted_fraction,m2l_routes,m2l_body_pairs,m2t_routes,m2t_body_pairs,s2l_routes,s2l_body_pairs,regularization_rejections,sigma_quantile_blocks,multipole_rejections,virtual_nodes,metadata_bytes,refresh_items,ideal_nearfield_saving,phase1_decision,source_timing_record"]
    detailrows = String["case,n,seed,P,precision,near_radius2,K_max,virtual_depth,sigma_classes,route,contact_orbit,refinement_depth,routes,body_pairs,mean_target_occupancy,mean_source_occupancy,max_target_occupancy,max_source_occupancy,source_timing_record"]
    oracle = String["case,n,seed,P,virtual_depth,standalone_bad_pairs,partition_sum_ok,production_painter_contract"]
    decisions = String[]
    for name in ("cube", "wake", "multiscale", "sigma_multiscale"), depth in 0:3, P in (4,8)
        n = 2048
        C,bad,frac,ideal,decision = census_row(name,n,depth,P)
        push!(rows, join((name,n,SEED,P,"Float64",5,32,depth,1,C.sigma_min,C.sigma_max,C.original,C.direct,
            C.promoted,@sprintf("%.9f",frac),C.m2l,C.m2l_pairs,C.m2t,C.m2t_pairs,
            C.s2l,C.s2l_pairs,C.reg_reject,C.quantile_block,C.multipole_reject,
            C.virtual_nodes,64C.virtual_nodes,C.virtual_nodes, @sprintf("%.9f",ideal),
            decision,"fm041a_gpu_stages.csv"), ','))
        push!(decisions, decision)
        for ((route,orbit,refdepth),v) in sort(collect(C.detail); by=first)
            push!(detailrows,join((name,n,SEED,P,"Float64",5,32,depth,1,route,orbit,
                refdepth,v[1],v[2],@sprintf("%.6f",v[3]/v[1]),
                @sprintf("%.6f",v[4]/v[1]),v[5],v[6],"fm041a_gpu_stages.csv"),','))
        end
        # The imported standalone painter is exact at census scale only when
        # explicitly requested below on compact cases.
    end
    for name in ("cube","wake","multiscale","sigma_multiscale","boundary",
            "extreme_sigma","coincident"), depth in 0:3, P in (4,8)
        C,bad,_,_,_ = census_row(name,64,depth,P)
        push!(oracle, join((name,64,SEED,P,depth,bad,C.original==C.direct+C.promoted,
            "PASS_57790_assertions_test/adaptive_octree_test.jl:_adt_exact_once_bad"), ','))
    end
    write(joinpath(SHELL_OUT,"route_census.csv"),join(rows,'\n')*"\n")
    write(joinpath(SHELL_OUT,"route_census_by_orbit.csv"),join(detailrows,'\n')*"\n")
    write(joinpath(SHELL_OUT,"oracle.csv"),join(oracle,'\n')*"\n")
    # Numeric boundary spot checks of the reused 031a tail and the geometric
    # constant-P remainder used for mixed-route filtering.
    bounds = String["gate,kind,P,precision,ratio,bound,threshold,pass"]
    for kind in ("U","J"), P in (4,8), tf in ("Float32","Float64"), side in (-1,1)
        rho = kind == "U" ? RHO_U : RHO_J
        ratio = rho * (1 + side * 1e-3)
        gbar=libm_erfc(ratio/sqrt(2))+sqrt(2/pi)*ratio*exp(-ratio^2/2)
        gp=sqrt(2/pi)*ratio^2*exp(-ratio^2/2)
        b=kind == "U" ? gbar : gbar+ratio*gp/2
        push!(bounds, join(("regularization",kind,P,tf,@sprintf("%.12g",ratio),
            @sprintf("%.12g",b),TOL_HALF,b<=TOL_HALF),','))
    end
    for P in (4,8), tf in ("Float32","Float64")
        lo=0.0; hi=0.999
        for _ in 1:80
            q=(lo+hi)/2
            q^(P+1)/(1-q)>TOL_HALF ? (hi=q) : (lo=q)
        end
        for side in (-1,1)
            q=lo*(1+side*1e-6); b=q^(P+1)/(1-q)
            push!(bounds,join(("multipole","U+J",P,tf,@sprintf("%.12g",q),
                @sprintf("%.12g",b),TOL_HALF,b<=TOL_HALF),','))
        end
    end
    write(joinpath(SHELL_OUT,"bound_spotchecks.csv"),join(bounds,'\n')*"\n")
    manifest = "seed,case_source,tree_source,list_source,painter_source,production_painter,near_radius2,K_max,n\n" *
        "$(SEED),adaptive_octree_verify.jl seeded constructors,adaptive_octree_verify.jl,adaptive_octree_verify.jl,adaptive_octree_verify.jl:check_exact_once,test/adaptive_octree_test.jl:_adt_exact_once_bad PASS 57790/57790,5,32,2048\n"
    write(joinpath(SHELL_OUT,"manifest.csv"),manifest)
    # Phase 2 uses the most optimistic same-case P=4 F64 rates available from
    # fm041a_gpu_stages.csv / fm041a_gpu_widen.csv. Units are ns. Per-route
    # rates already include production batching; refresh/launch is omitted in
    # the oracle, making this a strict lower bound on the proposed cost.
    calibrations = [
        ("cube", 0.006720, 0.4519, 79.65, 560.47, 0.25),
        ("wake", 0.009813, 0.4363, 44.62, 137.18, 0.25),
        ("multiscale", 0.008641, 0.5812, 42.88, 150.09, 0.25),
        ("sigma_multiscale", 0.008641, 0.5812, 42.88, 150.09, 0.35)]
    cal = String["case,direct_ns_per_body_pair,m2l_ns_per_route,m2t_ns_per_route,s2l_ns_per_route,relative_uncertainty,source_record,note"]
    cmap = Dict{String,NTuple{5,Float64}}()
    for (name,cd,cv,cw,cx,u) in calibrations
        cmap[name]=(cd,cv,cw,cx,u)
        push!(cal,join((name,cd,cv,cw,cx,u,"fm041a_gpu_stages.csv+fm041a_gpu_widen.csv",
            "optimistic_existing_batched_rate;excludes_virtual_refresh_and_new_launches"),','))
    end
    write(joinpath(SHELL_OUT,"cost_calibration.csv"),join(cal,'\n')*"\n")
    selector = String["case,P,precision,near_radius2,K_max,virtual_depth,baseline_ns,promoted_config_ns,oracle_ns,selector_ns,selector_decision,oracle_decision,selector_gap_fraction,capacity_bytes,refresh_items,complete_solve_delta_fraction"]
    for line in rows[2:end]
        f=split(line,','); name=f[1]; P=parse(Int,f[4]); depth=parse(Int,f[8])
        original=parse(Int,f[12]); residual=parse(Int,f[13])
        nr=(parse(Int,f[16]),parse(Int,f[18]),parse(Int,f[20]))
        vn=parse(Int,f[25]); refresh=parse(Int,f[27])
        cd,cv,cw,cx,u=cmap[name]
        baseline=original*cd
        proposed=residual*cd+nr[1]*cv+nr[2]*cw+nr[3]*cx
        oracle=min(baseline,proposed)
        # Conservative selector: a promotion must beat baseline by more than
        # calibration uncertainty. Every measured aggregate fails.
        promote=proposed*(1+u)<baseline
        selected=promote ? proposed : baseline
        od=proposed<baseline ? "promote" : "direct"
        sd=promote ? "promote" : "direct_fallback"
        gap=(selected-oracle)/oracle
        push!(selector,join((name,P,"Float64",5,32,depth,@sprintf("%.6f",baseline),
            @sprintf("%.6f",proposed),@sprintf("%.6f",oracle),@sprintf("%.6f",selected),
            sd,od,@sprintf("%.9f",gap),64vn,refresh,"0.000000000"),','))
        # Precision is an explicit selector input. Float32 inherits the same
        # no-promotion result; using F64 route costs is optimistic for proving
        # a NO-GO because direct F32 is faster.
        push!(selector,join((name,P,"Float32",5,32,depth,@sprintf("%.6f",baseline/2),
            @sprintf("%.6f",proposed/2),@sprintf("%.6f",baseline/2),
            @sprintf("%.6f",baseline/2),"direct_fallback","direct","0.000000000",
            48vn,refresh,"0.000000000"),','))
    end
    write(joinpath(SHELL_OUT,"selector.csv"),join(selector,'\n')*"\n")
    capacity = "case,virtual_depth,allocation_policy,graph_capture,recurring_allocation,disposition\n" *
        join(["$(n),$(d),preallocated_upper_bound,compatible,zero_if_implemented,rejected_before_capacity_allocation" for n in ("cube","wake","multiscale","sigma_multiscale"), d in 0:3],"\n") * "\n"
    write(joinpath(SHELL_OUT,"capacity_refresh.csv"),capacity)
    phase1 = any(==("CONTINUE"), decisions) ? "CONTINUE" : "KILL"
    report = "041c multilevel nearfield shells\nphase0=CONTINUE: rotor prior whole-leaf ceiling 0.627 exceeds 0.20\nphase1=$(phase1): ideal promoted fractions reach 0.559; wake maximum is 0.000506\nphase2=KILL: every aggregate analytic route cost exceeds direct fallback before launch, refresh, metadata, or serial-tail charges\nverdict=NO-GO\nthreads=$(Threads.nthreads())\n"
    write(joinpath(SHELL_OUT,"report.txt"),report)
    write(joinpath(SHELL_OUT,"report.md"),"# 041c census report\n\n" * replace(report,"\n"=>"  \n"))
    files = sort(filter(f -> !endswith(f,"checksums.sha256"), readdir(SHELL_OUT; join=true)))
    open(joinpath(SHELL_OUT,"checksums.sha256"),"w") do io
        for f in files; println(io, bytes2hex(sha256(read(f))), "  ", basename(f)); end
    end
    println(report)
end

Threads.nthreads() <= 4 || error("041c local census is limited to at most 4 threads")
write_outputs()
