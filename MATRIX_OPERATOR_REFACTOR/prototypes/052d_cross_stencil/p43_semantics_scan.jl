# P4.3 — semantics scan: which dense-recipe variant reproduces production?
#
# p42: the production-vs-dense delta at oracle targets (dev≈hfmm to <2e-5) is
# collective — no single column explains it. Here we recompute the dense
# reference under systematic variants of the wake-arm / kernel semantics and
# score each against the dumped deviceU over the 400 nearest-TE particles.
# The variant that collapses relL2(dev, dense_variant) identifies what the
# production routes actually compute.
#
# Run: JULIA_NUM_THREADS=4 DUMPDIR=<dir> \
#      julia --project=../../../../FLOWPanel.jl p43_semantics_scan.jl

include(joinpath(@__DIR__, "CrossStencil.jl"))
using .CrossStencil
import FastMultipole
const FM = FastMultipole
using StaticArrays, Random, Statistics, Printf, LinearAlgebra
using Base.Threads

const DUMPDIR = get(ENV, "DUMPDIR", "")
isempty(DUMPDIR) && error("set DUMPDIR")
const REG = 4

readmat(f, r) = collect(reshape(reinterpret(Float64, read(f)), r, :))

@inline function colverts(E, j)
    (SVector(E[3, j], E[4, j], E[5, j]), SVector(E[6, j], E[7, j], E[8, j]),
     SVector(E[9, j], E[10, j], E[11, j]), SVector(E[12, j], E[13, j], E[14, j]))
end

"expand_columns with tweakable arm semantics"
function expand_columns_v(srcmat, wakemat; da_scale=1.0, mu_scale=1.0,
        wake_tag=3, arm_core=nothing, drop_arms=false, core_scale_all=1.0)
    ns = size(srcmat, 2)
    nshed = drop_arms ? 0 : count(k -> wakemat[1, k] > 0, 1:ns)
    E = zeros(17, ns + 2 * nshed)
    E[:, 1:ns] .= srcmat
    E[17, 1:ns] .*= core_scale_all
    col = ns
    drop_arms && return E
    for k in 1:ns
        idx1 = Int(wakemat[1, k])
        idx1 > 0 || continue
        idx2 = Int(wakemat[5, k])
        tag = Int(srcmat[1, k])
        vs = (SVector(srcmat[3, k], srcmat[4, k], srcmat[5, k]),
              SVector(srcmat[6, k], srcmat[7, k], srcmat[8, k]),
              SVector(srcmat[9, k], srcmat[10, k], srcmat[11, k]))
        w1 = vs[idx1]; w2 = vs[idx2]
        v1w = w1 + da_scale * SVector(wakemat[2, k], wakemat[3, k], wakemat[4, k])
        v2w = w2 + da_scale * SVector(wakemat[6, k], wakemat[7, k], wakemat[8, k])
        mu = mu_scale * ((tag == 2 || tag == 3) ? srcmat[15, k] : srcmat[16, k])
        koff = (arm_core === nothing ? srcmat[17, k] : arm_core) * core_scale_all
        for (a, b, c) in ((w1, w2, v1w), (v1w, w2, v2w))
            col += 1
            E[1, col] = wake_tag; E[2, col] = 3
            E[3:5, col] .= a; E[6:8, col] .= b; E[9:11, col] .= c
            E[12:14, col] .= c
            E[15, col] = mu; E[16, col] = 0.0; E[17, col] = koff
        end
    end
    return E
end

function dense_at(E, targets)
    U = zeros(3, length(targets))
    @threads for s in eachindex(targets)
        t = targets[s]
        u = zero(SVector{3,Float64})
        @inbounds for j in 1:size(E, 2)
            tag = Int(E[1, j]); nv = Int(E[2, j])
            (1 <= tag <= 5 && nv >= 3) || continue
            v1, v2, v3, v4 = colverts(E, j)
            uq, _ = FM._rect_panel_pair(FM.RectangularPanelInfluence(), t,
                tag, nv, v1, v2, v3, v4, E[15, j], E[16, j], E[17, j],
                Val(false), Val(REG))
            u += uq
        end
        U[:, s] .= u
    end
    return U
end

relL2(a, b) = sqrt(sum(abs2, a .- b)) / max(sqrt(sum(abs2, b)), eps())

for np in (3544, 12776)
    pre = joinpath(DUMPDIR, "dump_np$np")
    isfile(pre * "_meta.txt") || continue
    positions = readmat(pre * "_positions_3xN_f64.bin", 3)
    deviceU = readmat(pre * "_deviceU_3xN_f64.bin", 3)
    hostU = readmat(pre * "_hostU_3xN_f64.bin", 3)
    srcmat = readmat(pre * "_body1_srcmat_17xS_f64.bin", 17)
    cent = readmat(pre * "_body1_cent_3xS_f64.bin", 3)
    wakemat = readmat(pre * "_body1_wakemat_8xS_f64.bin", 8)
    ns = size(srcmat, 2)
    particles = [SVector{3,Float64}(positions[:, i]) for i in 1:np]
    shedc = [SVector{3,Float64}(cent[:, k]) for k in 1:ns if wakemat[1, k] > 0]
    dTE = [minimum(c -> norm(particles[i] - c), shedc) for i in 1:np]
    pool = sortperm(dTE)[1:min(400, np)]
    tgts = [particles[i] for i in pool]
    Ud = deviceU[:, pool]; Uh = hostU[:, pool]

    variants = [
        ("baseline",            Dict()),
        ("Da x0.5",             Dict(:da_scale => 0.5)),
        ("Da x2",               Dict(:da_scale => 2.0)),
        ("mu x0.5",             Dict(:mu_scale => 0.5)),
        ("mu x2",               Dict(:mu_scale => 2.0)),
        ("mu negated",          Dict(:mu_scale => -1.0)),
        ("arms dropped",        Dict(:drop_arms => true)),
        ("arms as doublet tag2", Dict(:wake_tag => 2)),
        ("arm core x10",        Dict(:arm_core => 0.01)),
        ("arm core x50",        Dict(:arm_core => 0.05)),
        ("all core x10",        Dict(:core_scale_all => 10.0)),
        ("all core x50",        Dict(:core_scale_all => 50.0)),
    ]
    println("\n================ np=$np (pool=400 nearest-TE particles) ================")
    @printf("%-22s | %12s %12s\n", "variant", "dev_vs_dns", "hfmm_vs_dns")
    for (nm, kw) in variants
        E = expand_columns_v(srcmat, wakemat; (Symbol(k) => v for (k, v) in kw)...)
        U = dense_at(E, tgts)
        @printf("%-22s | %12.3e %12.3e\n", nm, relL2(Ud, U), relL2(Uh, U))
        flush(stdout)
    end
end
println("DONE")
