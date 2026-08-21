#!/usr/bin/env julia

# Task 041b correction: extract compact, deterministic cell blocks from the
# actual 1M-particle DJI-9443 rotor snapshot.  Run with the FLOWVPM project;
# the resulting CSV is consumed by the stdlib-only rank probe.

include(joinpath(@__DIR__, "benchmark_033_common.jl"))
using Printf

const N = 1_000_000
const ELL = 6
const NGRID = 1 << ELL
const OUT = joinpath(@__DIR__, "..", "data", "strategic_target_feasibility",
                     "rotor_n1000000_cells.csv")

X = Matrix{Float64}(undef, 3, N)
sigma = Vector{Float64}(undef, N)
rng = MersenneTwister(FM033_SEED + FM033_ROTOR_SEED_OFFSET + N)
i = Ref(0)
fm033_rotor_foreach(N; rng) do x, _, s
    i[] += 1
    X[:, i[]] .= x
    sigma[i[]] = s
end
@assert i[] == N

lo = vec(minimum(X; dims=2)); hi = vec(maximum(X; dims=2))
side = maximum(hi - lo); h = side / NGRID
cells = Dict{NTuple{3,Int},Vector{Int}}()
for p in 1:N
    key = ntuple(d -> clamp(floor(Int, (X[d,p]-lo[d])/h), 0, NGRID-1), 3)
    push!(get!(cells, key, Int[]), p)
end

face = [(s,0,0) for s in (-1,1)]
append!(face, [(0,s,0) for s in (-1,1)], [(0,0,s) for s in (-1,1)])
edge = NTuple{3,Int}[]
corner = NTuple{3,Int}[]
shell = NTuple{3,Int}[]
for a in 1:3, b in a+1:3, sa in (-1,1), sb in (-1,1)
    v = [0,0,0]; v[a]=sa; v[b]=sb; push!(edge, Tuple(v))
end
for a in (-1,1), b in (-1,1), c in (-1,1)
    push!(corner, (a,b,c))
end
for zeroaxis in 1:3, twoaxis in 1:3
    zeroaxis == twoaxis && continue
    oneaxis = only(setdiff(1:3, (zeroaxis,twoaxis)))
    for s2 in (-1,1), s1 in (-1,1)
        v=[0,0,0]; v[twoaxis]=2s2; v[oneaxis]=s1; push!(shell,Tuple(v))
    end
end
classes = Dict("self"=>[(0,0,0)], "face"=>face, "edge"=>edge,
               "corner"=>corner, "shell"=>shell)

function best_pair(offsets)
    best = nothing; bestscore = (-1,-1,-1)
    for (tk, ti) in cells, off in offsets
        sk = (tk[1]+off[1], tk[2]+off[2], tk[3]+off[3])
        si = get(cells, sk, nothing)
        si === nothing && continue
        length(ti) >= 64 && length(si) >= 24 || continue
        score = (min(length(ti),length(si)), length(ti), length(si))
        if score > bestscore
            bestscore = score; best = (tk,sk,off,ti,si)
        end
    end
    best === nothing && error("no sufficiently occupied cell pair")
    best
end

sample_indices(v, n; reverse_order=false) = begin
    pos = round.(Int, range(1, length(v); length=n))
    reverse_order ? reverse(v[pos]) : v[pos]
end

mkpath(dirname(OUT))
open(OUT, "w") do io
    println(io, "case,class,set,index,x,y,z,sigma_h,offset_x,offset_y,offset_z,target_cell_count,source_cell_count")
    for name in ("self","face","edge","corner","shell")
        tk, sk, off, ti, si = best_pair(classes[name])
        targets = sample_indices(ti, 64)
        sources = sample_indices(si, 48; reverse_order=name=="self")
        for (set, ids) in (("target",targets),("source",sources))
            for (j,p) in enumerate(ids)
                # Coordinates relative to the target-cell center.  Source rows
                # therefore retain their exact integer cell displacement.
                q = (X[:,p] .- lo) ./ h .- collect(tk) .- 0.5
                @printf(io,"rotor,%s,%s,%d,%.17g,%.17g,%.17g,%.17g,%d,%d,%d,%d,%d\n",
                    name,set,j,q[1],q[2],q[3],sigma[p]/h,off...,
                    length(ti),length(si))
            end
        end
    end
end
println("wrote $OUT")
