#!/usr/bin/env julia

# Task 041b: deterministic arithmetic sizing and offline numerical-rank/QDEIM
# ceiling probe.  No production package code or external package is loaded.

using LinearAlgebra
using Random
using Printf
using Statistics

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUTDIR = joinpath(ROOT, "data", "strategic_target_feasibility")
mkpath(OUTDIR)

parsecsv(path) = begin
    lines = readlines(path)
    names = split(first(lines), ',')
    [Dict(names .=> split(line, ','; keepempty=true)) for line in lines[2:end]
     if !isempty(strip(line))]
end

number(row, key, default=NaN) = tryparse(Float64, get(row, key, "")) === nothing ?
    default : parse(Float64, row[key])

function stage0!()
    rows = NamedTuple[]
    screen = parsecsv(joinpath(ROOT, "data", "flowvpm_gpu_campaign", "fm037e_screen.csv"))
    for r in screen
        occursin("anchor", get(r, "label", "")) || continue
        get(r, "status", "") == "ok" || continue
        n = Int(number(r, "n")); nc = number(r, "n_cells")
        q = number(r, "q"); rho = number(r, "rho_t")
        nt = n / nc
        # Equal-cell cutoff relation used by the approved stencil:
        # rho*sigma/h ~= sqrt(q).  h/(0.55sigma) is the measured 037d
        # p=4 sample count per coordinate.  The J-bearing estimate doubles
        # resolution per coordinate to expose the derivative penalty.
        k_u = ceil(Int, rho / (0.55 * sqrt(q)))
        r_u = k_u^3; r_j = (2k_u)^3
        push!(rows, (; source="037e", case=get(r,"case",""), n,
            tf=get(r,"tf",""), structure="uniform", param=get(r,"ell",""),
            n_target_mean=nt, n_target_max=NaN,
            direct_sources=number(r,"direct_body_pairs") / n,
            samples_u=r_u, samples_uj=r_j,
            u_survives=r_u < 0.5nt, uj_survives=r_j < 0.5nt))
    end

    adaptive = parsecsv(joinpath(ROOT, "data", "fm041_cuda_cost.csv"))
    # One lowest-step row per case/n/tf/structure/param; strategy changes only
    # the far-field operator and must not duplicate occupancy geometry.
    best = Dict{Tuple,Dict{SubString{String},SubString{String}}}()
    for r in adaptive
        get(r,"status","") == "ok" || continue
        get(r,"structure","") == "adaptive" || continue
        key = (get(r,"case",""), get(r,"n",""), get(r,"tf",""), get(r,"param",""))
        if !haskey(best,key) || number(r,"t_step_ms") < number(best[key],"t_step_ms")
            best[key] = r
        end
    end
    for r in values(best)
        n = Int(number(r,"n")); leaves = number(r,"n_leaves")
        nt = n / leaves
        # Adaptive CSV has no per-leaf sigma/h histogram.  Bracket with the
        # two approved uniform stencil resolutions q=6 and q=12 and retain the
        # optimistic lower sample count; this can only make the screen kinder.
        rho = 3.668; k_u = ceil(Int, rho / (0.55sqrt(12.0)))
        r_u = k_u^3; r_j = (2k_u)^3
        push!(rows, (; source="041", case=get(r,"case",""), n,
            tf=get(r,"tf",""), structure="adaptive", param=get(r,"param",""),
            n_target_mean=nt, n_target_max=number(r,"popmax"),
            direct_sources=number(r,"u_pairs") / n,
            samples_u=r_u, samples_uj=r_j,
            u_survives=r_u < 0.5nt, uj_survives=r_j < 0.5nt))
    end

    sort!(rows; by=x->(x.source,x.case,x.n,x.tf,x.param))
    open(joinpath(OUTDIR,"stage0_sizing.csv"),"w") do io
        println(io,"source,case,n,tf,structure,param,n_target_mean,n_target_max,direct_sources,samples_u,samples_uj,u_survives,uj_survives")
        for x in rows
            @printf(io,"%s,%s,%d,%s,%s,%s,%.6f,%.6f,%.6f,%d,%d,%s,%s\n",
                x.source,x.case,x.n,x.tf,x.structure,x.param,x.n_target_mean,
                x.n_target_max,x.direct_sources,x.samples_u,x.samples_uj,
                x.u_survives,x.uj_survives)
        end
    end
    rows
end

# Gaussian-erf regularized Biot-Savart coefficient.  Constants do not affect
# numerical rank but are retained for physical scaling.
function erf_local(x)
    # Abramowitz-Stegun 7.1.26; max error ~1.5e-7, ample for a rank ceiling
    # whose registered tolerances are 2.5e-4 and 1e-3.
    s = sign(x); z = abs(x); t = 1 / (1 + 0.3275911z)
    p = (((((1.061405429t - 1.453152027)t) + 1.421413741)t -
          0.284496736)t + 0.254829592)t
    s * (1 - p * exp(-z*z))
end

function scalar_g(r2, sigma)
    r2 == 0 && return 0.0
    r = sqrt(r2); a = r / sigma
    (erf_local(a) - 2a / sqrt(pi) * exp(-a*a)) / (4pi*r^3)
end

function velocity(x, y, gamma, sigma)
    r = x - y
    cross(gamma, r) * scalar_g(dot(r,r), sigma)
end

function uj_column(x, y, gamma, sigma; want_j=true)
    u = velocity(x,y,gamma,sigma)
    want_j || return u
    d = 2e-5 * max(sigma, 1.0)
    J = zeros(3,3)
    for j in 1:3
        e = zeros(3); e[j] = d
        J[:,j] .= (velocity(x+e,y,gamma,sigma)-velocity(x-e,y,gamma,sigma))/(2d)
    end
    vcat(u, vec(J))
end

function halton(i, b)
    f=1.0; x=0.0
    while i > 0
        f /= b; x += f*(i % b); i ÷= b
    end
    x
end
points(n, shift, start=1) = [([halton(i,2),halton(i,3),halton(i,5)] .- 0.5) .+ shift for i in start:start+n-1]

function snapshot_local(case, n, seed)
    rng=MersenneTwister(seed); raw=zeros(3,n)
    if case=="unitcube"
        raw .= rand(rng,3,n)
    elseif case=="wake"
        for i in 1:n
            r=0.5sqrt(rand(rng)); th=2pi*rand(rng)
            raw[:,i] .= (r*cos(th),r*sin(th),5rand(rng))
        end
        raw[1:2,:] .+= 0.5; raw[3,:] ./= 5
    else
        raw .= rand(rng,3,n)
        nc=round(Int,0.35n); radius=0.08
        for i in n-nc+1:n
            raw[:,i] .= [0.6,0.4,0.55] .+ radius .* randn(rng,3)
        end
    end
    # Normalize intra-cell coordinates, as the displacement-conditioned
    # operator sees them. Eight bins preserve the fm041 construction seeds.
    [mod.(8 .* raw[:,i],1) .- 0.5 for i in 1:n]
end

mixed_points(case,n,shift,seed,start) = vcat(
    [p .+ shift for p in snapshot_local(case,n÷2,seed)],
    points(n-n÷2,shift,start))

function operator_matrix(targets, sources, sigmas; want_j=true)
    no = want_j ? 12 : 3
    A = zeros(no*length(targets), 3length(sources))
    for (ti,x) in pairs(targets), (si,y) in pairs(sources), k in 1:3
        gamma=zeros(3); gamma[k]=1
        A[(ti-1)*no+1:ti*no, (si-1)*3+k] .= uj_column(x,y,gamma,sigmas[si]; want_j)
    end
    A
end

qdeim(U, r) = qr(transpose(U[:,1:r]), ColumnNorm()).p[1:r]

relerr(A,B) = norm(A-B)/max(norm(A),eps())

function rotor_blocks()
    rows = parsecsv(joinpath(OUTDIR,"rotor_n1000000_cells.csv"))
    blocks = Dict{String,NamedTuple}()
    for class in ("self","face","edge","corner","shell")
        tr = filter(r->get(r,"class","")==class && get(r,"set","")=="target", rows)
        sr = filter(r->get(r,"class","")==class && get(r,"set","")=="source", rows)
        xyz(r) = [number(r,"x"),number(r,"y"),number(r,"z")]
        targets = xyz.(tr)
        sources = xyz.(sr)
        sigmas = [number(r,"sigma_h") for r in sr]
        offset = [number(first(sr),"offset_x"),number(first(sr),"offset_y"),
                  number(first(sr),"offset_z")]
        blocks[class] = (; targets,sources,sigmas,offset,
            target_cell_count=Int(number(first(tr),"target_cell_count")),
            source_cell_count=Int(number(first(sr),"source_cell_count")))
    end
    blocks
end

function component_scales(A, no)
    nt = size(A,1) ÷ no
    [max(norm(A[c:no:no*nt,:]) / sqrt(nt*size(A,2)), eps()) for c in 1:no]
end

scale_rows(A, scales) = A ./ repeat(scales, size(A,1) ÷ length(scales))

function group_error(A, B, no, component_range)
    rows = reduce(vcat, (collect((t-1)*no .+ component_range)
                         for t in 1:(size(A,1)÷no)))
    relerr(A[rows,:],B[rows,:])
end

function monomial_exponents(degree)
    [(i,j,k) for i in 0:degree for j in 0:degree-i for k in 0:degree-i-j]
end

function vandermonde(xs, exps)
    V=zeros(length(xs),length(exps))
    for (p,x) in pairs(xs), (j,(a,b,c)) in pairs(exps)
        V[p,j]=x[1]^a*x[2]^b*x[3]^c
    end
    V
end

function polynomial_screen(H, candidates, targets, no)
    ns=length(candidates); nt=length(targets)
    best=(points=0,u=Inf,j=no==3 ? NaN : Inf,passes=false)
    for degree in 1:12
        exps=monomial_exponents(degree); r=length(exps)
        r <= ns || break
        Vs=vandermonde(candidates[1:r],exps)
        Vt=vandermonde(targets,exps)
        recon=zeros(no*nt,size(H,2)); truth=H[ns*no+1:end,:]
        for c in 1:no
            sample=H[c:no:r*no,:]
            recon[c:no:end,:] .= Vt*(Vs\sample)
        end
        uerr=group_error(truth,recon,no,1:3)
        jerr=no==3 ? NaN : group_error(truth,recon,no,4:12)
        if uerr < best.u || (uerr <= 2.5e-4 && no==12 && jerr < best.j)
            best=(points=r,u=uerr,j=jerr,passes=false)
        end
        if uerr <= 2.5e-4 && (no==3 || jerr <= 1.0e-3)
            return (points=r,u=uerr,j=jerr,passes=true)
        end
    end
    merge(best,(method="total_degree",))
end

chebval(n,x) = cos(n*acos(clamp(2x,-1.0,1.0)))
cheb_points(k) = [[x,y,z] for x in reverse(cos.(range(0,pi;length=k))./2)
                            for y in reverse(cos.(range(0,pi;length=k))./2)
                            for z in reverse(cos.(range(0,pi;length=k))./2)]

function tensor_chebyshev_screen(held_sources, held_sigmas, targets, no)
    best=(points=0,u=Inf,j=no==3 ? NaN : Inf,passes=false,method="tensor_chebyshev")
    for k in 2:6
        nodes=cheb_points(k); exps=[(i,j,l) for i in 0:k-1 for j in 0:k-1 for l in 0:k-1]
        V(xs) = [chebval(a,x[1])*chebval(b,x[2])*chebval(c,x[3])
                 for x in xs, (a,b,c) in exps]
        H=operator_matrix(vcat(nodes,targets),held_sources,held_sigmas;want_j=no==12)
        ns=length(nodes); nt=length(targets); recon=zeros(no*nt,size(H,2))
        truth=H[ns*no+1:end,:]; Vs=V(nodes); Vt=V(targets)
        for c in 1:no
            recon[c:no:end,:] .= Vt*(Vs\H[c:no:ns*no,:])
        end
        uerr=group_error(truth,recon,no,1:3)
        jerr=no==3 ? NaN : group_error(truth,recon,no,4:12)
        best=(points=k^3,u=uerr,j=jerr,passes=false,method="tensor_chebyshev")
        if uerr <= 2.5e-4 && (no==3 || jerr <= 1.0e-3)
            return merge(best,(passes=true,))
        end
    end
    best
end

function rbf_screen(H, candidates, targets, no)
    ns=length(candidates); truth=H[ns*no+1:end,:]
    best=(points=0,u=Inf,j=no==3 ? NaN : Inf,passes=false,method="gaussian_rbf")
    for r in (8,14,20,28,40,56,84,120,165,216), ell in (0.15,0.3,0.6,1.2)
        centers=candidates[1:r]
        Kcc=[exp(-sum(abs2,x-y)/(2ell^2)) for x in centers,y in centers]
        Ktc=[exp(-sum(abs2,x-y)/(2ell^2)) for x in targets,y in centers]
        Kcc[diagind(Kcc)] .+= 1e-11*opnorm(Kcc,Inf)+eps()
        recon=zeros(no*length(targets),size(H,2))
        for c in 1:no
            recon[c:no:end,:] .= Ktc*(Kcc\H[c:no:r*no,:])
        end
        uerr=group_error(truth,recon,no,1:3)
        jerr=no==3 ? NaN : group_error(truth,recon,no,4:12)
        if uerr < best.u || (uerr <= 2.5e-4 && no==12 && jerr < best.j)
            best=(points=r,u=uerr,j=jerr,passes=false,method="gaussian_rbf")
        end
        if uerr <= 2.5e-4 && (no==3 || jerr <= 1.0e-3)
            return (points=r,u=uerr,j=jerr,passes=true,method="gaussian_rbf")
        end
    end
    best
end

function probe!()
    blocks = rotor_blocks()
    rankrows = NamedTuple[]
    verdictrows = NamedTuple[]
    # The actual selected fat-cell snapshot is the only Stage-0 U/J survivor.
    # Per-bin scale factors span its recorded global 18x sigma range while
    # preserving property-conditioned bases (each factor is screened alone).
    for name in ("self","face","edge","corner","shell"),
            sigma_bin in (1.0,6.0,18.0), output in (:u,:uj)
        b=blocks[name]
        candidates=points(216,[0.,0.,0.],701)
        targets=b.targets
        train_actual=b.sources[1:32]
        held_actual=b.sources[33:48]
        train_sources=vcat(train_actual,points(16,b.offset,901))
        held_sources=vcat(held_actual,points(8,b.offset,1201))
        medsigma=median(b.sigmas)
        train_sigmas=sigma_bin .* vcat(b.sigmas[1:32],fill(medsigma,16))
        held_sigmas=sigma_bin .* vcat(b.sigmas[33:48],fill(medsigma,8))
        alltargets=vcat(candidates,targets)
        A=operator_matrix(alltargets,train_sources,train_sigmas; want_j=output==:uj)
        H=operator_matrix(alltargets,held_sources,held_sigmas; want_j=output==:uj)
        no=output==:u ? 3 : 12
        scales=component_scales(A,no)
        Aw=scale_rows(A,scales); Hw=scale_rows(H,scales)
        F=svd(Aw); total=sum(abs2,F.S)
        nsamplerows=length(candidates)*no
        sample_rows=1:nsamplerows
        validation_rows=nsamplerows+1:size(Aw,1)
        best_u=Inf; best_j=output==:u ? NaN : Inf
        chosen=0; selected_points=0; passed=false
        for r in 1:length(F.S)
            tail=sqrt(sum(abs2,F.S[r+1:end])/total)
            push!(rankrows,(; case="rotor",class=name,sigma_bin,output=String(output),rank=r,
                            svd_relative_rms=tail,singular_value=F.S[r]))
            idx=qdeim(F.U[sample_rows,:],r)
            coeff=F.U[idx,1:r] \ Hw[idx,:]
            recon=F.U[validation_rows,1:r]*coeff
            truth=Hw[validation_rows,:]
            uerr=group_error(truth,recon,no,1:3)
            jerr=output==:u ? NaN : group_error(truth,recon,no,4:12)
            if uerr < best_u || (output==:uj && uerr <= 2.5e-4 && jerr < best_j)
                best_u=uerr; best_j=jerr; chosen=r
                selected_points=length(unique(cld.(idx,no)))
            end
            if uerr <= 2.5e-4 && (output==:u || jerr <= 1.0e-3)
                best_u=uerr; best_j=jerr; chosen=r
                selected_points=length(unique(cld.(idx,no))); passed=true
                break
            end
        end
        point_lb=cld(chosen,no)
        practical=(points=0,u=NaN,j=NaN,passes=false,method="not_run")
        if passed
            total=merge(polynomial_screen(H,candidates,targets,no),(method="total_degree",))
            tensor=tensor_chebyshev_screen(held_sources,held_sigmas,targets,no)
            rbf=rbf_screen(H,candidates,targets,no)
            options=(total,tensor,rbf)
            passing=filter(x->x.passes,options)
            practical=isempty(passing) ? argmin(x->x.u,options) :
                                         argmin(x->x.points,passing)
        end
        rank_ratio=passed && practical.passes ? practical.points/max(chosen,1) : Inf
        practical_gate=practical.passes && rank_ratio <= 1.25
        push!(verdictrows,(; case="rotor",class=name,sigma_bin,output=String(output),rank=chosen,
            point_lower_bound=point_lb,selected_scalar_sample_points=selected_points,
            velocity_holdout_error=best_u,j_holdout_error=best_j,passes=passed,
            practical_points=practical.points,practical_velocity_error=practical.u,
            practical_j_error=practical.j,practical_passes=practical.passes,
            practical_method=practical.method,
            practical_to_qdeim_ratio=rank_ratio,practical_gate,
            target_cell_count=b.target_cell_count,source_cell_count=b.source_cell_count))
    end
    open(joinpath(OUTDIR,"rank_curves.csv"),"w") do io
        println(io,"case,class,sigma_bin,output,rank,svd_relative_rms,singular_value")
        for x in rankrows
            @printf(io,"%s,%s,%.3f,%s,%d,%.9e,%.9e\n",x.case,x.class,x.sigma_bin,x.output,x.rank,x.svd_relative_rms,x.singular_value)
        end
    end
    open(joinpath(OUTDIR,"qdeim_summary.csv"),"w") do io
        println(io,"case,class,sigma_bin,output,rank,point_lower_bound,selected_scalar_sample_points,velocity_holdout_error,j_holdout_error,passes,practical_points,practical_velocity_error,practical_j_error,practical_passes,practical_method,practical_to_qdeim_ratio,practical_gate,target_cell_count,source_cell_count")
        for x in verdictrows
            @printf(io,"%s,%s,%.3f,%s,%d,%d,%d,%.9e,%.9e,%s,%d,%.9e,%.9e,%s,%s,%.9e,%s,%d,%d\n",
                x.case,x.class,x.sigma_bin,x.output,x.rank,x.point_lower_bound,
                x.selected_scalar_sample_points,x.velocity_holdout_error,x.j_holdout_error,
                x.passes,x.practical_points,x.practical_velocity_error,x.practical_j_error,
                x.practical_passes,x.practical_method,x.practical_to_qdeim_ratio,x.practical_gate,
                x.target_cell_count,x.source_cell_count)
        end
    end
    open(joinpath(OUTDIR,"metadata.csv"),"w") do io
        println(io,"key,value")
        println(io,"task,041b")
        println(io,"construction_source,actual fm033 DJI-9443 rotor n=1000000 ell=6")
        println(io,"snapshot_seed,1137754")
        println(io,"candidate_halton_start,701")
        println(io,"training_halton_start,901")
        println(io,"heldout_halton_start,1201")
        println(io,"candidate_sample_count,216")
        println(io,"validation_target_count,64")
        println(io,"training_source_count,48")
        println(io,"heldout_source_count,24")
        println(io,"strength_directions,3")
        println(io,"sigma_bins,1;6;18 times actual per-cell sigma/h")
        println(io,"velocity_tolerance,2.5e-4")
        println(io,"uj_screen_tolerance,1e-3")
    end
    verdictrows
end

stage = stage0!()
verdict = probe!()
println("wrote $(joinpath(OUTDIR,"stage0_sizing.csv"))")
println("wrote $(joinpath(OUTDIR,"rank_curves.csv"))")
println("wrote $(joinpath(OUTDIR,"qdeim_summary.csv"))")
println("Stage-0 U survivors: ", count(x->x.u_survives,stage), "/", length(stage))
println("Stage-0 U/J survivors: ", count(x->x.uj_survives,stage), "/", length(stage))
println("QDEIM held-out passes: ", count(x->x.passes,verdict), "/", length(verdict))
println("Practical rank gates: ", count(x->x.practical_gate,verdict), "/", length(verdict))
