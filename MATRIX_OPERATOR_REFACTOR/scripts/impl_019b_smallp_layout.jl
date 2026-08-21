# =============================================================================
# 019b Small-P / Tiny-Batch Fallback + Padded-vs-Ragged Chi Layout (exploratory)
# =============================================================================
#
# Exploratory CPU benchmark for task 019b. Produces the evidence for the two
# REQUIRED USER DECISIONS (no policy is implemented here; no src/ changes):
#
#   (A) Small-P / tiny-batch fallback: where does each dense operator form beat
#       the legacy production recurrence (`multipole_to_local!`), across
#       (P, batch)? Three dense forms are measured:
#         * materialized : per-column MaterializedYRotationM2L (task 013/014)
#         * factored     : per-column FactoredRotationM2L (task 013c/014)
#         * concat_host  : the tuned 019 whole-slab dense GEMM concat M2L,
#           measured at the M2L-stage level on real host radix states
#           (`_launch_host_m2l!` with ConcatenatedFixedZM2L) against a per-route
#           legacy-recurrence loop on the SAME state — the exact shape a
#           recurrence fallback would take.
#
#   (B) Padded-uniform vs ragged chi layout (Val(true) only): simulate the
#       dominant dense GEMM chain of the concat path (stacked-y GEMMs + dense z
#       GEMM + elementwise rotations/LH mix) in three layout variants —
#       ragged (phi at P, chi at P+1; current default), padded (both at P+1,
#       separate slabs), padded_merged (both at P+1, one [phi chi] slab, halving
#       GEMM launches — padding's only plausible win) — plus a storage table.
#
# -----------------------------------------------------------------------------
# HOW TO RUN  (from the repository root)
# -----------------------------------------------------------------------------
# Like 008c/015, BLAS threading MUST be pinned at process start via env var
# (runtime BLAS.set_num_threads() is unreliable for OpenBLAS). Run TWICE:
#
#   Single-thread BLAS:
#     OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
#       julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/impl_019b_smallp_layout.jl
#
#   Multi-thread BLAS (all cores; Linux: $(nproc)):
#     OPENBLAS_NUM_THREADS=$(sysctl -n hw.ncpu) OMP_NUM_THREADS=$(sysctl -n hw.ncpu) \
#       julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/impl_019b_smallp_layout.jl
#
# Sweep overrides: P_LIST, BATCH_LIST, SAMPLES, LAYOUT_P_LIST, LAYOUT_BATCH_LIST,
# STAGE_P_LIST env vars (comma-separated ints).
#
# Output (machine-tagged): MATRIX_OPERATOR_REFACTOR/data/smallp_fallback_layout/<host>/
#   env.md                          environment + BLAS metadata
#   crossover_isolated_blas<N>.csv  isolated per-column operators vs recurrence
#   crossover_stage_blas<N>.csv     state-level concat M2L stage vs per-route recurrence
#   layout_lh_blas<N>.csv           ragged / padded / padded_merged LH chain timing
#   layout_storage.csv              per-column storage (BLAS-independent)
#
# Self-contained: no plotting / BenchmarkTools deps.
# =============================================================================

using FastMultipole
using FastMultipole.LinearAlgebra
using FastMultipole.LinearAlgebra: BLAS, mul!
using FastMultipole.StaticArrays
using Printf
using Dates
using Random

const FM = FastMultipole

# -----------------------------------------------------------------------------
# Timing helper (identical policy to 008c/015): calibrate inner eval count, then
# min-per-call over `samples` repeats. `setup` runs untimed before each sample.
# -----------------------------------------------------------------------------
function timeit(f; setup=nothing, samples::Int=SAMPLES)
    setup === nothing || setup()
    f()  # warmup / compile
    evals = 1
    while true
        setup === nothing || setup()
        dt = @elapsed for _ in 1:evals; f(); end
        dt > 1e-4 && break
        evals *= 10
        evals > 10^8 && break
    end
    best = Inf
    for _ in 1:samples
        setup === nothing || setup()
        dt = @elapsed for _ in 1:evals; f(); end
        best = min(best, dt / evals)
    end
    return best
end

# ---- parameters (overridable via ENV) ---------------------------------------
_parse_int_list(s) = parse.(Int, split(s, ","))
const P_LIST      = haskey(ENV, "P_LIST")      ? _parse_int_list(ENV["P_LIST"])      : [1, 2, 3, 4, 5, 6, 8, 12]
const BATCH_LIST  = haskey(ENV, "BATCH_LIST")  ? _parse_int_list(ENV["BATCH_LIST"])  : [1, 2, 4, 8, 16, 32, 64, 256, 1024, 4096]
const SAMPLES     = haskey(ENV, "SAMPLES")     ? parse(Int, ENV["SAMPLES"])          : 30
const SEED        = 190219
const RECURRENCE_BATCH_CAP = haskey(ENV, "RECURRENCE_BATCH_CAP") ? parse(Int, ENV["RECURRENCE_BATCH_CAP"]) : 2048
# state-level stage sweep
const STAGE_P_LIST = haskey(ENV, "STAGE_P_LIST") ? _parse_int_list(ENV["STAGE_P_LIST"]) : [1, 2, 3, 4, 6, 8]
# layout comparison sweep
const LAYOUT_P_LIST     = haskey(ENV, "LAYOUT_P_LIST")     ? _parse_int_list(ENV["LAYOUT_P_LIST"])     : [2, 4, 8, 12, 20]
const LAYOUT_BATCH_LIST = haskey(ENV, "LAYOUT_BATCH_LIST") ? _parse_int_list(ENV["LAYOUT_BATCH_LIST"]) : [64, 4096]

# ---- output location (machine-tagged) ---------------------------------------
const HOST = gethostname()
const OUTDIR = normpath(joinpath(@__DIR__, "..", "data", "smallp_fallback_layout", HOST))
mkpath(OUTDIR)

function progress(msg)
    println("[", Dates.format(Dates.now(), "HH:MM:SS"), "] ", msg)
    flush(stdout)
end

# -----------------------------------------------------------------------------
# Environment metadata + weak-BLAS detection (mirrors 008c/015).
# -----------------------------------------------------------------------------
function blas_description()
    libs = String[]
    try
        cfg = BLAS.get_config()
        for lib in cfg.loaded_libs
            push!(libs, basename(String(lib.libname)))
        end
    catch err
        push!(libs, "unknown ($(err))")
    end
    return libs
end

git_head() = try strip(read(`git rev-parse HEAD`, String)) catch; "unknown" end
git_dirty() = try !isempty(strip(read(`git status --porcelain`, String))) catch; false end

function blas_is_optimized(libs)
    known = ("openblas", "mkl", "blis", "accelerate", "veclib", "armpl")
    return any(l -> any(k -> occursin(k, lowercase(l)), known), libs)
end

function write_env(io)
    libs = blas_description()
    optimized = blas_is_optimized(libs)
    println(io, "# 019b small-P fallback + chi layout exploratory benchmarks -- environment")
    println(io)
    println(io, "- date: ", Dates.now())
    println(io, "- hostname: ", HOST)
    println(io, "- julia: ", VERSION)
    cpu = Sys.cpu_info()
    println(io, "- cpu_model: ", isempty(cpu) ? "unknown" : cpu[1].model)
    println(io, "- physical/logical cores: ", Sys.CPU_THREADS)
    println(io, "- Threads.nthreads(): ", Threads.nthreads())
    println(io, "- BLAS.get_num_threads(): ", BLAS.get_num_threads())
    println(io, "- BLAS libs: ", join(libs, ", "))
    println(io, "- BLAS optimized?: ", optimized)
    if !optimized
        println(io)
        println(io, "> WARNING: no tuned BLAS detected. Dense GEMM timings are a LOWER bound.")
    end
    println(io, "- git HEAD: ", git_head())
    println(io, "- git dirty: ", git_dirty())
    println(io, "- P_LIST: ", P_LIST)
    println(io, "- BATCH_LIST: ", BATCH_LIST)
    println(io, "- SAMPLES: ", SAMPLES)
    println(io, "- RECURRENCE_BATCH_CAP: ", RECURRENCE_BATCH_CAP)
    println(io, "- STAGE_P_LIST: ", STAGE_P_LIST)
    println(io, "- LAYOUT_P_LIST: ", LAYOUT_P_LIST)
    println(io, "- LAYOUT_BATCH_LIST: ", LAYOUT_BATCH_LIST)
    return optimized
end

function ensure_globals!(Pmax)
    FM.update_Hs_π2!(FM.Hs_π2, Pmax)
    FM.update_ζs_mag!(FM.ζs_mag, Pmax)
    FM.update_ηs_mag!(FM.ηs_mag, Pmax)
    FM.update_M̃!(FM.M̃, Pmax)
    FM.update_L̃!(FM.L̃, Pmax)
    return nothing
end

# A representative non-axis offset (general direction; nonzero phi and theta).
const REF_OFFSET = SVector{3}(1.5, -2.0, 2.5)

# Physical random source in a FlatCoefficientBuffer (m = 0 rows have zero
# imaginary part, as every real expansion does).
function fill_physical_source!(buf::FlatCoefficientBuffer, ::Val{LH}) where {LH}
    TF = eltype(buf.phi)
    orders = buf.basis_info.orders
    fill!(buf.phi, zero(TF))
    LH && fill!(buf.chi, zero(TF))
    for j in axes(buf.phi, 2)
        for n in 0:orders.P_phi, m in 0:n
            buf.phi[FM.flat_basis_index(n, m, 1), j] = randn(TF)
            m == 0 || (buf.phi[FM.flat_basis_index(n, m, 2), j] = randn(TF))
        end
        if LH
            for n in 0:orders.P_active, m in 0:n
                buf.chi[FM.flat_basis_index(n, m, 1), j] = randn(TF)
                m == 0 || (buf.chi[FM.flat_basis_index(n, m, 2), j] = randn(TF))
            end
        end
    end
    return buf
end

# Production recurrence reference closure for one fixed pair (015 pattern).
function make_production_ref(P, TF, ::Val{LH}, Δx) where {LH}
    box = SVector{3}(zero(TF), zero(TF), zero(TF))
    src_branch = FM.Branch(2:2, 0, 1:0, 0, 1, SVector{3}(zero(TF), zero(TF), zero(TF)), zero(TF), box)
    tgt_branch = FM.Branch(2:2, 0, 1:0, 0, 1, SVector{3}(Δx), zero(TF), box)
    Hs = TF[1.0]; FM.update_Hs_π2!(Hs, P)
    Ts = zeros(TF, FM.length_Ts(P))
    eimϕs = zeros(TF, 2, P + 1)
    w1 = FM.initialize_expansion(P, TF)
    w2 = FM.initialize_expansion(P, TF)
    w3 = FM.initialize_expansion(P, TF)
    ζ = zeros(TF, FM.length_ζs(P)); FM.update_ζs_mag!(ζ, 0, P)
    η = zeros(TF, FM.length_ηs(P)); FM.update_ηs_mag!(η, 0, P)
    src = FM.initialize_expansion(P, TF)
    for n in 0:P, m in 0:n
        i = FM.harmonic_index(n, m)
        src[1, 1, i] = randn(TF); src[2, 1, i] = m == 0 ? zero(TF) : randn(TF)
        if LH
            src[1, 2, i] = randn(TF); src[2, 2, i] = m == 0 ? zero(TF) : randn(TF)
        end
    end
    local_exp = FM.initialize_expansion(P, TF)
    lh = Val(LH)
    return function ()
        FM.multipole_to_local!(local_exp, tgt_branch, src, src_branch,
            w1, w2, w3, Ts, eimϕs, ζ, η, Hs, FM.M̃, FM.L̃, P, lh)
        return nothing
    end
end

# -----------------------------------------------------------------------------
# Correctness gate (015 pattern): the timed per-column operator must reproduce
# production `multipole_to_local!`. Aborts on mismatch.
# -----------------------------------------------------------------------------
function sanity_check(P, TF)
    lh = Val(false)
    cache = OperatorInvariantCache(TF, P, lh)
    scratch = M2LOperatorScratch(TF, cache.basis_info, 1)
    sources = FlatCoefficientBuffer(TF, cache.basis_info, 1)
    fill_physical_source!(sources, lh)
    src = FM.initialize_expansion(P, TF)
    FM._pack_flat_column!(src, sources, 1, P, P, lh)
    r, θ, ϕ = FM.cartesian_to_spherical(REF_OFFSET)
    targets = FlatCoefficientBuffer(TF, cache.basis_info, 1)
    FM.m2l_operator_batch!(MaterializedYRotationM2L(), targets, sources, [ϕ], [θ], [r], cache, scratch, lh)

    box = SVector{3}(zero(TF), zero(TF), zero(TF))
    src_branch = FM.Branch(2:2, 0, 1:0, 0, 1, SVector{3}(zero(TF), zero(TF), zero(TF)), zero(TF), box)
    tgt_branch = FM.Branch(2:2, 0, 1:0, 0, 1, SVector{3}(REF_OFFSET), zero(TF), box)
    Hs = TF[1.0]; FM.update_Hs_π2!(Hs, P)
    Ts = zeros(TF, FM.length_Ts(P)); eimϕs = zeros(TF, 2, P + 1)
    w1 = FM.initialize_expansion(P, TF); w2 = FM.initialize_expansion(P, TF); w3 = FM.initialize_expansion(P, TF)
    ζ = zeros(TF, FM.length_ζs(P)); FM.update_ζs_mag!(ζ, 0, P)
    η = zeros(TF, FM.length_ηs(P)); FM.update_ηs_mag!(η, 0, P)
    ref = FM.initialize_expansion(P, TF)
    FM.multipole_to_local!(ref, tgt_branch, src, src_branch, w1, w2, w3, Ts, eimϕs, ζ, η, Hs, FM.M̃, FM.L̃, P, lh)

    maxerr = 0.0
    for n in 0:P, m in 0:n, ri in 1:2
        i = FM.harmonic_index(n, m)
        maxerr = max(maxerr, abs(targets.phi[FM.flat_basis_index(n, m, ri), 1] - ref[ri, 1, i]))
    end
    if maxerr > 1e-6
        error("019b sanity check FAILED at P=$P: max |operator - production| = $maxerr (> 1e-6). Aborting.")
    end
    progress("sanity check OK at P=$P (max abs diff vs production = $(@sprintf("%.2e", maxerr)))")
    return nothing
end

# -----------------------------------------------------------------------------
# (A1) Isolated crossover sweep: per-column operators vs production recurrence,
# focused on small P and tiny batch. Float64 (production precision).
# -----------------------------------------------------------------------------
function bench_isolated(io, blas_threads)
    println(io, "variant,form,precision,blas_threads,P,lamb_helmholtz,batch,measured_batch,scaled,seconds,seconds_per_expansion")
    T = Float64
    variants = (("materialized", MaterializedYRotationM2L()), ("factored", FactoredRotationM2L()))
    for P in P_LIST
        for LHbool in (false, true)
            lh = Val(LHbool)
            progress("isolated crossover: P=$P lamb_helmholtz=$LHbool")
            cache = OperatorInvariantCache(T, P, lh)
            r, θ, ϕ = FM.cartesian_to_spherical(SVector{3}(T.(REF_OFFSET)))

            for B in BATCH_LIST
                scratch = M2LOperatorScratch(T, cache.basis_info, B)
                sources = FlatCoefficientBuffer(T, cache.basis_info, B)
                fill_physical_source!(sources, lh)
                targets = FlatCoefficientBuffer(T, cache.basis_info, B)
                phis = fill(T(ϕ), B); thetas = fill(T(θ), B); rs = fill(T(r), B)
                for (vname, vop) in variants
                    tt = timeit(() -> FM.m2l_operator_batch!(vop, targets, sources, phis, thetas, rs, cache, scratch, lh);
                                setup=() -> (fill!(targets.phi, zero(T)); LHbool && fill!(targets.chi, zero(T))))
                    @printf(io, "%s,batched_operator,%s,%d,%d,%s,%d,%d,false,%.6e,%.6e\n",
                            vname, string(T), blas_threads, P, string(LHbool), B, B, tt, tt / B)
                end
                flush(io)
            end

            ref = make_production_ref(P, T, lh, SVector{3}(REF_OFFSET))
            for B in BATCH_LIST
                Brec = min(B, RECURRENCE_BATCH_CAP)
                scaled = Brec != B
                tr = timeit(() -> (for _ in 1:Brec; ref(); end))
                tr_pe = tr / Brec
                @printf(io, "production_recurrence,recurrence,Float64,%d,%d,%s,%d,%d,%s,%.6e,%.6e\n",
                        blas_threads, P, string(LHbool), B, Brec, string(scaled), tr_pe * B, tr_pe)
            end
            flush(io)
        end
    end
end

# -----------------------------------------------------------------------------
# (A2) State-level M2L stage crossover on real host radix states.
#
#   concat     : `_launch_host_m2l!` with ConcatenatedFixedZM2L (the tuned 019
#                whole-slab dense GEMM path, generic host fallbacks).
#   recurrence : per-route legacy `multipole_to_local!` loop over the same state
#                (pack column -> legacy M2L at the run order -> accumulate), with
#                all work buffers preallocated — the shape a fallback would take.
#                For Val(true) the legacy call runs at P_active = P + 1 so the chi
#                channel keeps the 008h order rule.
#
# A parity gate compares the two locals before timing each configuration.
# -----------------------------------------------------------------------------
function make_body_matrix(n; seed=SEED + 5)
    rng = MersenneTwister(seed)
    return vcat(rand(rng, 3, n), zeros(1, n), reshape(randn(rng, n), 1, n))
end

# Preallocated per-route legacy recurrence M2L over a host radix state.
struct RouteRecurrenceScratch{TF}
    P_run::Int
    src::Array{TF,3}
    dst::Array{TF,3}
    w1::Array{TF,3}
    w2::Array{TF,3}
    w3::Array{TF,3}
    Ts::Vector{TF}
    eimϕs::Matrix{TF}
    ζ::Vector{TF}
    η::Vector{TF}
    Hs::Vector{TF}
end

function RouteRecurrenceScratch(::Type{TF}, P_run) where TF
    Hs = TF[1.0]; FM.update_Hs_π2!(Hs, P_run)
    ζ = zeros(TF, FM.length_ζs(P_run)); FM.update_ζs_mag!(ζ, 0, P_run)
    η = zeros(TF, FM.length_ηs(P_run)); FM.update_ηs_mag!(η, 0, P_run)
    return RouteRecurrenceScratch{TF}(P_run,
        FM.initialize_expansion(P_run, TF), FM.initialize_expansion(P_run, TF),
        FM.initialize_expansion(P_run, TF), FM.initialize_expansion(P_run, TF),
        FM.initialize_expansion(P_run, TF),
        zeros(TF, FM.length_Ts(P_run)), zeros(TF, 2, P_run + 1), ζ, η, Hs)
end

function route_recurrence_m2l!(state, rs::RouteRecurrenceScratch{TF}, ::Val{LH}) where {TF,LH}
    orders = state.invariant_cache.basis_info.orders
    P_phi = orders.P_phi
    P_active = orders.P_active
    P_run = rs.P_run
    lh = Val(LH)
    box = SVector{3}(zero(TF), zero(TF), zero(TF))
    fill!(state.locals.phi, zero(TF))
    LH && fill!(state.locals.chi, zero(TF))
    @inbounds for route_i in eachindex(state.route_targets)
        target = state.route_targets[route_i]
        source = state.route_sources[route_i]
        FM._pack_flat_column!(rs.src, state.multipoles, source, P_phi, P_active, lh)
        fill!(rs.dst, zero(TF))
        src_center = SVector{3,TF}(state.grid.node_centers[1, source],
                                   state.grid.node_centers[2, source],
                                   state.grid.node_centers[3, source])
        tgt_center = SVector{3,TF}(state.grid.node_centers[1, target],
                                   state.grid.node_centers[2, target],
                                   state.grid.node_centers[3, target])
        src_branch = FM.Branch(2:2, 0, 1:0, 0, 1, src_center, zero(TF), box)
        tgt_branch = FM.Branch(2:2, 0, 1:0, 0, 1, tgt_center, zero(TF), box)
        FM.multipole_to_local!(rs.dst, tgt_branch, rs.src, src_branch,
            rs.w1, rs.w2, rs.w3, rs.Ts, rs.eimϕs, rs.ζ, rs.η, rs.Hs, FM.M̃, FM.L̃, P_run, lh)
        FM._unpack_flat_column_accumulate!(state.locals, rs.dst, target, P_phi, P_active, lh)
    end
    return state
end

function bench_stage(io, blas_threads)
    println(io, "config,n,ell,policy,P,lamb_helmholtz,routes,list_seconds,form,seconds,seconds_per_route")
    T = Float64
    # The tiny/small configs are the decision-relevant regime (few routes; where a
    # fallback could win) and get the full P list. The medium config (~12M routes at
    # ell=4) only anchors the known large-batch dense advantage, so it is trimmed to
    # two P values and few samples — each of its recurrence calls is tens of seconds.
    configs = (
        (; name="tiny_parent", n=256, ell=2, policy=:parent, P_list=STAGE_P_LIST, samples=SAMPLES),
        (; name="small_constp", n=2048, ell=3, policy=:constp, P_list=STAGE_P_LIST, samples=SAMPLES),
        (; name="medium_constp", n=20_000, ell=4, policy=:constp, P_list=intersect(STAGE_P_LIST, [2, 4]), samples=2),
    )
    for cfg in configs
        bodies = make_body_matrix(cfg.n)
        grid = RadixGrid(bodies, cfg.ell)
        for P in cfg.P_list
            # conservative-stencil tolerance loosens as P shrinks so small-P
            # configurations still produce a nonempty M2L route set.
            policy = cfg.policy === :parent ? ParentNeighborM2L() :
                ConstantPAnalyticStencil(P, P >= 8 ? 1e-8 : (P >= 4 ? 1e-4 : 1e-2))
            list_seconds = @elapsed list = build_radix_interaction_list(
                LazyMaterializedBatches(32), policy, grid)
            routes = sum((length(b.targets) for b in list.m2l_batches); init=0)
            routes == 0 && (progress("stage: $(cfg.name) P=$P has 0 routes; skipped"); continue)
            for LHbool in (false, true)
                lh = Val(LHbool)
                progress("stage crossover: $(cfg.name) P=$P lamb_helmholtz=$LHbool routes=$routes")
                state = host_radix_state(bodies, grid, list, P, lh;
                    options=CUDARadixLifecycleOptions(; m2l_strategy=ConcatenatedFixedZM2L()))
                FM._launch_host_b2m!(state)
                FM._launch_host_m2m!(state)

                orders = state.invariant_cache.basis_info.orders
                rscratch = RouteRecurrenceScratch(T, orders.P_active)

                # parity gate: concat vs per-route recurrence on identical inputs
                fill!(state.locals.phi, zero(T)); LHbool && fill!(state.locals.chi, zero(T))
                FM._launch_host_m2l!(state)
                concat_phi = copy(state.locals.phi)
                route_recurrence_m2l!(state, rscratch, lh)
                scale = max(maximum(abs, concat_phi), one(T))
                maxerr = maximum(abs.(state.locals.phi .- concat_phi)) / scale
                if maxerr > 1e-8
                    error("019b stage parity FAILED: $(cfg.name) P=$P LH=$LHbool rel err $maxerr")
                end

                t_concat = timeit(() -> FM._launch_host_m2l!(state);
                    setup=() -> (fill!(state.locals.phi, zero(T)); LHbool && fill!(state.locals.chi, zero(T))),
                    samples=cfg.samples)
                t_recur = timeit(() -> route_recurrence_m2l!(state, rscratch, lh); samples=cfg.samples)
                for (form, tt) in (("concat_host", t_concat), ("route_recurrence", t_recur))
                    @printf(io, "%s,%d,%d,%s,%d,%s,%d,%.6e,%s,%.6e,%.6e\n",
                            cfg.name, cfg.n, cfg.ell, string(cfg.policy), P, string(LHbool),
                            routes, list_seconds, form, tt, tt / routes)
                end
                flush(io)
            end
        end
    end
end

# -----------------------------------------------------------------------------
# (B) Padded-vs-ragged chi layout: simulate the dominant per-chunk dense chain of
# the concat M2L (forward stacked-y, dense z GEMM with separable r scaling,
# LH row mix, return stacked-y, plus the elementwise Z_phi rotations) over
# degree-major slabs, in three layout variants. Uses the production dense
# operator builders (`_ymode_stacked_dense`, `_m2l_dense_factorial_matrix`,
# `_degree_row_nus`, `_stacked_y_dense!`) so relative costs are representative.
# -----------------------------------------------------------------------------
struct LayoutChannelSim{TF}
    ndof::Int
    Ur_mult::Matrix{TF}
    Vs_mult::Matrix{TF}
    Ur_loc::Matrix{TF}
    Vs_loc::Matrix{TF}
    zD::Matrix{TF}
    C::Matrix{TF}
    S::Matrix{TF}
    G::Matrix{TF}
    G2::Matrix{TF}
    a::Matrix{TF}   # input slab
    y::Matrix{TF}   # post-forward-y slab
    z::Matrix{TF}   # post-z slab
    o::Matrix{TF}   # output slab
end

function LayoutChannelSim(::Type{TF}, cache, P::Int, width::Int, rng) where TF
    exemplar = zeros(TF, 1, 1)
    Ur_mult, Vs_mult = FM._ymode_stacked_dense(exemplar, TF, cache.y_mult_U, cache.y_mult_V, P)
    Ur_loc, Vs_loc = FM._ymode_stacked_dense(exemplar, TF, cache.y_loc_U, cache.y_loc_V, P)
    zD = Matrix(FM._m2l_dense_factorial_matrix(exemplar, TF, P))
    ndof = FM.degree_major_dof(P)
    nu = Vector(FM._degree_row_nus(exemplar, TF, P))
    thetas = TF.(0.4 .+ 0.2 .* rand(rng, width))
    C = [cos(nu[i] * thetas[j]) for i in 1:ndof, j in 1:width]
    S = [sin(nu[i] * thetas[j]) for i in 1:ndof, j in 1:width]
    return LayoutChannelSim{TF}(ndof, Matrix(Ur_mult), Matrix(Vs_mult), Matrix(Ur_loc), Matrix(Vs_loc),
        zD, C, S,
        zeros(TF, 2 * ndof, width), zeros(TF, 2 * ndof, width),
        randn(rng, TF, ndof, width), zeros(TF, ndof, width), zeros(TF, ndof, width), zeros(TF, ndof, width))
end

# One channel pass through the dense chain (matches the concat stage order:
# forward y(mult), dense z with elementwise separable scaling, return y(loc)).
function run_channel!(sim::LayoutChannelSim)
    FM._stacked_y_dense!(sim.y, sim.a, sim.Ur_mult, sim.Vs_mult, sim.C, sim.S, sim.G, sim.G2, sim.ndof)
    mul!(sim.z, sim.zD, sim.y)
    sim.z .*= sim.C   # stand-in for the elementwise r^-(n+1/2) pre/post scaling
    FM._stacked_y_dense!(sim.o, sim.z, sim.Ur_loc, sim.Vs_loc, sim.C, sim.S, sim.G, sim.G2, sim.ndof)
    return sim
end

# LH row mix stand-in: phi rows get an elementwise chi contribution (same row
# count in every layout; measures the traffic, not the exact coupling stencil).
function lh_mix!(phi_slab, chi_slab, nrows)
    p = @view phi_slab[1:nrows, :]
    c = @view chi_slab[1:nrows, :]
    p .+= 0.125 .* c
    return phi_slab
end

function bench_layout(io, blas_threads)
    println(io, "layout,precision,blas_threads,P_phi,P_active,batch,ndof_phi,ndof_chi,gemms_per_stagepass,seconds,seconds_per_expansion")
    T = Float64
    rng = MersenneTwister(SEED + 11)
    for P in LAYOUT_P_LIST
        cache = OperatorInvariantCache(T, P, Val(true))
        P_active = cache.basis_info.orders.P_active
        for B in LAYOUT_BATCH_LIST
            progress("layout: P=$P (P_active=$P_active) batch=$B")
            # ragged: phi chain at P, chi chain at P_active, separate slabs
            sim_phi = LayoutChannelSim(T, cache, P, B, rng)
            sim_chi = LayoutChannelSim(T, cache, P_active, B, rng)
            t_ragged = timeit() do
                run_channel!(sim_phi)
                run_channel!(sim_chi)
                lh_mix!(sim_phi.o, sim_chi.o, sim_phi.ndof)
            end
            @printf(io, "ragged,%s,%d,%d,%d,%d,%d,%d,10,%.6e,%.6e\n",
                    string(T), blas_threads, P, P_active, B,
                    FM.degree_major_dof(P), FM.degree_major_dof(P_active), t_ragged, t_ragged / B)

            # padded: both channels at P_active, separate slabs
            simp_phi = LayoutChannelSim(T, cache, P_active, B, rng)
            simp_chi = LayoutChannelSim(T, cache, P_active, B, rng)
            t_padded = timeit() do
                run_channel!(simp_phi)
                run_channel!(simp_chi)
                lh_mix!(simp_phi.o, simp_chi.o, FM.degree_major_dof(P))
            end
            @printf(io, "padded,%s,%d,%d,%d,%d,%d,%d,10,%.6e,%.6e\n",
                    string(T), blas_threads, P, P_active, B,
                    FM.degree_major_dof(P_active), FM.degree_major_dof(P_active), t_padded, t_padded / B)

            # padded_merged: both channels at P_active in ONE slab of width 2B
            # (channel-merged GEMMs; half the launches for the same math)
            sim_m = LayoutChannelSim(T, cache, P_active, 2 * B, rng)
            ndofa = FM.degree_major_dof(P)
            t_merged = timeit() do
                run_channel!(sim_m)
                phi_half = @view sim_m.o[:, 1:B]
                chi_half = @view sim_m.o[:, (B + 1):(2 * B)]
                lh_mix!(phi_half, chi_half, ndofa)
            end
            @printf(io, "padded_merged,%s,%d,%d,%d,%d,%d,%d,5,%.6e,%.6e\n",
                    string(T), blas_threads, P, P_active, B,
                    FM.degree_major_dof(P_active), FM.degree_major_dof(P_active), t_merged, t_merged / B)
            flush(io)
        end
    end
end

# -----------------------------------------------------------------------------
# (B2) Storage table (BLAS-independent): per-column coefficient rows and bytes
# for both buffer families, ragged vs padded, Val(true). Val(false) has one
# channel at P in every layout (no question there).
# -----------------------------------------------------------------------------
function bench_storage(io)
    println(io, "buffer,P_phi,P_active,rows_phi_ragged,rows_chi,rows_phi_padded,ragged_bytes_per_col_f64,padded_bytes_per_col_f64,padded_overhead_frac")
    for P in sort(unique(vcat(LAYOUT_P_LIST, STAGE_P_LIST, P_LIST)))
        info = OperatorBasisInfo(CompressedComplexBasis(), P, Val(true))
        Pa = info.orders.P_active
        # flat compressed-complex buffer (task 017)
        rp = info.basis_dof_phi; rc = info.basis_dof_chi; rpad = info.basis_dof_active
        rag = 8 * (rp + rc); pad = 8 * (rpad + rc)
        @printf(io, "FlatCoefficientBuffer,%d,%d,%d,%d,%d,%d,%d,%.4f\n",
                P, Pa, rp, rc, rpad, rag, pad, pad / rag - 1)
        # degree-major real buffer (task 022)
        dp = FM.degree_major_dof(P); dc = FM.degree_major_dof(Pa)
        drag = 8 * (dp + dc); dpad = 8 * (2 * dc)
        @printf(io, "DegreeMajorRealBuffer,%d,%d,%d,%d,%d,%d,%d,%.4f\n",
                P, Pa, dp, dc, dc, drag, dpad, dpad / drag - 1)
    end
end

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
function main()
    Random.seed!(SEED)
    Pmax = maximum(vcat(P_LIST, STAGE_P_LIST, LAYOUT_P_LIST)) + 1
    ensure_globals!(Pmax)

    progress("Writing 019b benchmarks to: $OUTDIR")
    open(joinpath(OUTDIR, "env.md"), "w") do io
        write_env(io)
    end
    progress("Wrote environment metadata")

    sanity_check(minimum(P_LIST), Float64)
    sanity_check(maximum(P_LIST), Float64)

    blas_threads = BLAS.get_num_threads()

    fname = "crossover_isolated_blas$(blas_threads).csv"
    progress("[1/4] isolated crossover sweep (BLAS threads = $blas_threads) ...")
    open(joinpath(OUTDIR, fname), "w") do io
        bench_isolated(io, blas_threads)
    end
    progress("[1/4] wrote $fname")

    sname = "crossover_stage_blas$(blas_threads).csv"
    progress("[2/4] state-level M2L stage crossover ...")
    open(joinpath(OUTDIR, sname), "w") do io
        bench_stage(io, blas_threads)
    end
    progress("[2/4] wrote $sname")

    lname = "layout_lh_blas$(blas_threads).csv"
    progress("[3/4] padded-vs-ragged LH layout chain ...")
    open(joinpath(OUTDIR, lname), "w") do io
        bench_layout(io, blas_threads)
    end
    progress("[3/4] wrote $lname")

    progress("[4/4] storage table ...")
    open(joinpath(OUTDIR, "layout_storage.csv"), "w") do io
        bench_storage(io)
    end
    progress("[4/4] wrote layout_storage.csv")

    progress("Done. Results in: $OUTDIR")
end

main()
