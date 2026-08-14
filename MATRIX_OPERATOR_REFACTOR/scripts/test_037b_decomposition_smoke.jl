# Task 037b: local CPU-only smoke test for the error-decomposition oracle
# (benchmark_035_error_decomposition.jl) in its exact-only cutoff-curve mode.
#
# What it verifies (all without CUDA, wake n=1000, <= 4 threads, ~2 min):
#   1. End-to-end exact-only run against the checksummed 033 wake n=1000
#      reference (the script itself asserts the independent-oracle match to
#      5e-12; the 037a smoke precedent measured 2.09e-16 / 5.34e-16).
#   2. Exact-field identities: the prefix-sum partitioned reconstruction
#      P(rho) matches an independent per-cutoff brute-force evaluation at
#      three rho values, P(rho -> 0) == S, and P(rho -> large) -> R.
#   3. The truncated-deficit error ||P(rho) - R|| / ||R|| is strictly
#      decreasing in rho (both function-level and in the emitted CSV), and
#      twopass rows reproduce the partitioned cutoff column exactly at equal
#      rho_t (the P_shell(rho_c, rho_x) == P(rho_x) equivalence).
#   4. Config parsing + FLOWVPM kernel resolution for kernel=twopass rows
#      (rho_c/rho_t threading, dryrun-style, no GPU), including P=4
#      (expansion_order=3) coverage per the standing project rule.
#
# Run (repo root = the FastMultipole tree containing MATRIX_OPERATOR_REFACTOR):
#   julia --project=<env with FLOWVPM + this FastMultipole dev'd> --threads=4 \
#       MATRIX_OPERATOR_REFACTOR/scripts/test_037b_decomposition_smoke.jl

using Test
using Base.Threads

nthreads() <= 4 || error("project rule: run local jobs with <= 4 threads")

const SMOKE_REPO = abspath(joinpath(@__DIR__, "..", ".."))
const SMOKE_DIR = mktempdir()
const SMOKE_OUT = joinpath(SMOKE_DIR, "fm037b_smoke_decomposition.csv")
const SMOKE_CONFIG = joinpath(SMOKE_DIR, "fm037b_smoke_configs.txt")

const SMOKE_RHOS = (2.0, 2.5, 3.0, 3.4, 3.668)
open(SMOKE_CONFIG, "w") do io
    println(io, "# 037b smoke grid: partitioned vs twopass(rho_c=1.7) cutoff curve")
    for rho in SMOKE_RHOS
        tag = replace(string(rho), "." => "")
        println(io, "label=sp_r$tag case=wake n=1000 expansion_order=4 " *
            "ell=4 q=6 kernel=partitioned rho_t=$rho")
        println(io, "label=st_r$tag case=wake n=1000 expansion_order=4 " *
            "ell=4 q=6 kernel=twopass rho_t=$rho rho_c=1.7")
    end
    # P=4 coverage rows (standing rule: all new tests cover P=4)
    println(io, "label=sp4_r3668 case=wake n=1000 expansion_order=3 ell=4 " *
        "q=6 kernel=partitioned rho_t=3.668")
    println(io, "label=st4_r34 case=wake n=1000 expansion_order=3 ell=4 " *
        "q=6 kernel=twopass rho_t=3.4 rho_c=1.7")
end

ENV["FM035_FMDIR"] = SMOKE_REPO
ENV["FM035D_EXACT_ONLY"] = "1"
ENV["FM035D_CONFIG_FILE"] = SMOKE_CONFIG
ENV["FM035D_OUT"] = SMOKE_OUT

# End-to-end exact-only run; also loads the oracle's functions and the 033
# harness (vpm, fm033_build, reference IO) into Main. The script itself
# hard-asserts the independent R oracle against the checksummed reference.
include(joinpath(SMOKE_REPO, "MATRIX_OPERATOR_REFACTOR", "scripts",
    "benchmark_035_error_decomposition.jl"))

# Independent per-cutoff brute force: same pair primitives, but a direct
# regularized/singular branch per pair — no binning, no prefix sum.
function smoke_partitioned_bruteforce(pfield, indices, cutoff)
    A = pfield.particles
    n = vpm.get_np(pfield)
    S = length(indices)
    U = zeros(3, S); J = zeros(9, S)
    Aconst = sqrt(2 / pi)
    for k in 1:S
        i = indices[k]
        xi = Float64(A[vpm.X_INDEX[1], i])
        yi = Float64(A[vpm.X_INDEX[2], i])
        zi = Float64(A[vpm.X_INDEX[3], i])
        for j in 1:n
            j == i && continue
            dx = xi - Float64(A[vpm.X_INDEX[1], j])
            dy = yi - Float64(A[vpm.X_INDEX[2], j])
            dz = zi - Float64(A[vpm.X_INDEX[3], j])
            r2 = dx*dx + dy*dy + dz*dz
            r2 == 0 && continue
            r = sqrt(r2); invr = inv(r)
            sigma = Float64(A[vpm.SIGMA_INDEX, j])
            gx = Float64(A[vpm.GAMMA_INDEX[1], j])
            gy = Float64(A[vpm.GAMMA_INDEX[2], j])
            gz = Float64(A[vpm.GAMMA_INDEX[3], j])
            rho = r / sigma
            cr3 = inv(4pi) * invr^3
            cx = (dz*gy - dy*gz)*cr3
            cy = (dx*gz - dz*gx)*cr3
            cz = (dy*gx - dx*gy)*cr3
            if rho <= cutoff
                e = exp(-rho*rho/2)
                g = fm035d_ref_erf(rho/sqrt(2.0)) - Aconst*rho*e
                gp = Aconst*rho*rho*e
                h = rho*gp - 3g
                fm035d_addpair!(U, J, k, dx, dy, dz, r2, cr3,
                    cx, cy, cz, gx, gy, gz, g, h)
            else
                fm035d_addpair!(U, J, k, dx, dy, dz, r2, cr3,
                    cx, cy, cz, gx, gy, gz, 1.0, -3.0)
            end
        end
    end
    return U, J
end

smoke_rel(A, B) = sqrt(norm2(A .- B) / max(norm2(B), eps(Float64)))

@testset "037b decomposition smoke" begin

    cpu = fm033_build("wake", 1000)
    ref = fm033_read_reference(
        fm033_reference_path(FM035D_REFDIR, "wake", 1000), "wake", 1000, 1000)
    rhos = [1e-9, 2.5, 3.2, 20.0]
    ex = fm035d_exact_fields(cpu, ref.indices, rhos)

    @testset "exact-field identities (R, S, prefix-sum P)" begin
        # no pair has rho <= 1e-9, so P(1e-9) is exactly the singular field
        @test ex.partitioned[1e-9].U == ex.US
        @test ex.partitioned[1e-9].J == ex.JS
        # beyond rho=20 the deficit is ~exp(-200): P(20) -> R.  Residual is
        # pure accumulation-order noise (P is built as S + binned deltas, R
        # is summed directly; the J channel's +-3/r^2 split cancels, leaving
        # ~2.5e-13 relative fp noise), so the tolerance is fp-noise-sized,
        # far below any physical deficit.
        @test smoke_rel(ex.partitioned[20.0].U, ex.UR) < 1e-11
        @test smoke_rel(ex.partitioned[20.0].J, ex.JR) < 1e-11
        # independent brute-force reconstruction at three cutoffs
        for cutoff in (1e-9, 2.5, 3.2)
            Ub, Jb = smoke_partitioned_bruteforce(cpu, ref.indices, cutoff)
            @test smoke_rel(ex.partitioned[cutoff].U, Ub) < 1e-12
            @test smoke_rel(ex.partitioned[cutoff].J, Jb) < 1e-12
        end
    end

    @testset "monotone truncated-deficit error" begin
        rhos2 = collect(2.0:0.25:4.5)
        ex2 = fm035d_exact_fields(cpu, ref.indices, rhos2)
        errs = [smoke_rel(ex2.partitioned[r].U, ex2.UR) for r in rhos2]
        @test all(errs[i+1] < errs[i] for i in 1:length(errs)-1)
        @test errs[end] > 1e-13   # still far above fp noise, so the
                                  # comparison above is meaningful
    end

    @testset "twopass config parsing + kernel resolution (no GPU)" begin
        cfg = fm035d_parse_config("label=t1 case=wake n=1000 " *
            "expansion_order=4 ell=6 q=4 kernel=twopass rho_t=3.4 rho_c=1.7")
        @test cfg.kernel === :twopass
        @test cfg.rho == 3.4          # rho_t: the P-lookup key
        @test cfg.rho_c == 1.7
        @test cfg.order == 4 && cfg.ell == 6 && cfg.q == 4
        s = vpm.RadixFMMSettings(; expansion_order=cfg.order, ell=cfg.ell,
            near_radius2=cfg.q, direct_kernel=cfg.kernel, rho_t=cfg.rho,
            rho_c=cfg.rho_c)
        k = vpm._radix_direct_kernel(s)
        @test k isa FM.TwoPassVortex
        @test k.rho_t == 3.4 && k.rho_c == 1.7

        # P=4 coverage: expansion_order=3 rows parse and resolve
        cfg4 = fm035d_parse_config("label=t2 case=wake n=1000 " *
            "expansion_order=3 ell=4 q=6 kernel=twopass rho_t=3.668 rho_c=2.0")
        @test cfg4.order == 3         # literature P = 4
        s4 = vpm.RadixFMMSettings(; expansion_order=cfg4.order, ell=cfg4.ell,
            near_radius2=cfg4.q, direct_kernel=cfg4.kernel, rho_t=cfg4.rho,
            rho_c=cfg4.rho_c)
        k4 = vpm._radix_direct_kernel(s4)
        @test k4 isa FM.TwoPassVortex && k4.rho_c == 2.0
        cfg4p = fm035d_parse_config("label=t3 case=wake n=1000 " *
            "expansion_order=3 ell=4 q=6 rho_t=3.668")
        @test cfg4p.kernel === :partitioned && cfg4p.rho_c === nothing
        @test vpm._radix_direct_kernel(vpm.RadixFMMSettings(;
            expansion_order=cfg4p.order, direct_kernel=cfg4p.kernel,
            rho_t=cfg4p.rho)) isa FM.PartitionedVortex

        # guard rails: rho_c is twopass-only, and must sit below rho_t
        @test_throws ErrorException vpm._radix_direct_kernel(
            vpm.RadixFMMSettings(; direct_kernel=:partitioned, rho_c=1.7))
        @test_throws ArgumentError vpm._radix_direct_kernel(
            vpm.RadixFMMSettings(; direct_kernel=:twopass, rho_t=3.4,
                rho_c=3.7))
    end

    @testset "exact-only CSV output" begin
        lines = readlines(SMOKE_OUT)
        header = split(lines[1], ',')
        col = Dict(name => i for (i, name) in enumerate(header))
        @test haskey(col, "component_semantics")
        rows = [split(l, ',') for l in lines[2:end]]
        @test length(rows) == 2 * length(SMOKE_RHOS) + 2
        part = Dict{Float64,Any}(); twop = Dict{Float64,Any}()
        for r in rows
            kernel = r[col["kernel"]]
            sem = r[col["component_semantics"]]
            @test sem == (kernel == "twopass" ? "truncated_deficit" : "cutoff")
            @test r[col["tf"]] == "Float64"
            @test isempty(r[col["u_fmm_rel"]])       # no GPU field ran
            @test isempty(r[col["u_total_rel"]])
            kernel == "twopass" && @test r[col["rho_c"]] == "1.7"
            if r[col["expansion_order"]] == "4"
                d = kernel == "twopass" ? twop : part
                d[parse(Float64, r[col["rho_t"]])] = r[col["u_cutoff_rel"]]
            end
        end
        # P_shell(rho_c, rho_t) == P(rho_t): identical cutoff column
        for rho in SMOKE_RHOS
            @test twop[rho] == part[rho]
        end
        # strictly decreasing truncated-deficit error along the sweep
        vals = [parse(Float64, part[rho]) for rho in SMOKE_RHOS]
        @test all(vals[i+1] < vals[i] for i in 1:length(vals)-1)
        # P=4 rows made it into the CSV
        @test any(r[col["expansion_order"]] == "3" for r in rows)
    end
end

println("037b decomposition smoke test passed; CSV at $SMOKE_OUT")
