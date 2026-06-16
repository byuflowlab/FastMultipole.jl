# =============================================================================
# 008c Implementation Performance Baseline -- GPU  (USER-RUN)
# =============================================================================
#
# Companion to impl_baseline_cpu.jl. This script measures the dense / batched-
# GEMM operator forms on a GPU so the dense-vs-recurrence operator-form decision
# (task 008c) can be confirmed in the regime the Matrix Operator Refactor is
# actually betting on: large batched small-matrix GEMM where a GPU amortizes
# kernel-launch + host/device transfer.
#
# It deliberately mirrors the CPU dense prototypes in impl_baseline_cpu.jl:
#   * same two GEMM-relevant stages
#       - M2L z-translation : block-diagonal over m, block size (P+1-m)
#       - axis-swap (y rot) : block-diagonal over n, block size (2n+1)
#   * same batch sweep
#   * re/im modeled as 2 columns per expansion
# so the CSV columns line up with dense_vs_loop.csv and CPU<->GPU numbers are
# directly comparable.
#
# Two GPU variants are measured per (stage, batch):
#   - device_resident : inputs already on the GPU (steady-state FMM: coefficient
#                        buffers live on device). Pure kernel time.
#   - with_transfer   : host->device upload + compute + device->host download,
#                        to expose the launch/transfer break-even.
#
# The GPU has NO compiled-recurrence baseline here: the production recurrences
# are scalar CPU code. The relevant GPU question is "does batched GEMM scale",
# and the per-expansion time is compared against the CPU recurrence numbers from
# impl_baseline_cpu.jl (dense_vs_loop.csv, form=recurrence).
#
# -----------------------------------------------------------------------------
# REQUIREMENTS
# -----------------------------------------------------------------------------
#   * An NVIDIA GPU + CUDA. Add CUDA to whatever environment you run this from:
#       julia --project=. -e 'import Pkg; Pkg.add("CUDA")'
#     (CUDA is intentionally NOT a FastMultipole dependency; this is a throwaway
#      benchmark.)
#
# -----------------------------------------------------------------------------
# HOW TO RUN  (on the GPU machine, from the repository root)
# -----------------------------------------------------------------------------
#       julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/impl_baseline_gpu.jl
#
#   Override the sweep (P and precision default to a full sweep):
#       P_DENSE_LIST="2,3,4,5,6,7,10,14,20" BATCH_LIST="64,512,4096,32768" \
#       PREC_LIST="Float64,Float32" \
#         julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/impl_baseline_gpu.jl
#
# Output (machine-tagged, alongside the CPU results):
#   MATRIX_OPERATOR_REFACTOR/data/impl_performance_baseline/<hostname>/
#       env_gpu.md
#       dense_gpu.csv  (columns: stage,form,precision,P,batch,seconds,seconds_per_expansion)
#
# On a machine WITHOUT a functional GPU the script prints a clear skip message
# and exits 0 (it is safe to leave in CI / run on the CPU dev box).
# =============================================================================

using LinearAlgebra
using Statistics
using Printf
using Dates

# ---- parameters (overridable via ENV) ---------------------------------------
_parse_int_list(s) = parse.(Int, split(s, ","))
function _parse_prec_list(s)
    map(split(s, ",")) do tok
        t = lowercase(strip(tok))
        t in ("float64", "f64", "double") ? Float64 :
        t in ("float32", "f32", "single") ? Float32 :
        error("unknown precision '$tok' (use Float64 / Float32)")
    end
end
const BATCH_LIST = haskey(ENV, "BATCH_LIST") ? _parse_int_list(ENV["BATCH_LIST"]) : [64, 512, 4096, 32768]
const P_DENSE_LIST = haskey(ENV, "P_DENSE_LIST") ? _parse_int_list(ENV["P_DENSE_LIST"]) : [2, 3, 4, 5, 6, 7, 10, 14, 20]
const SAMPLES    = haskey(ENV, "SAMPLES")    ? parse(Int, ENV["SAMPLES"])         : 50
# Sweep both precisions by default: FMM works in Float64, but Float32 may be
# much faster on GPU -- we want to quantify that speedup.
const PREC_LIST  = haskey(ENV, "PREC_LIST")  ? _parse_prec_list(ENV["PREC_LIST"]) : [Float64, Float32]

const HOST   = gethostname()
const OUTDIR = normpath(joinpath(@__DIR__, "..", "data", "impl_performance_baseline", HOST))

# block sizes (identical to impl_baseline_cpu.jl)
mblocks(P) = [P + 1 - m for m in 0:P]   # M2L z-translation
nblocks(P) = [2n + 1 for n in 0:P]      # axis-swap (y rotation)

# -----------------------------------------------------------------------------
# Try to load CUDA; skip cleanly if unavailable.
# -----------------------------------------------------------------------------
const CUDA_OK = try
    @eval using CUDA
    CUDA.functional()
catch err
    @info "CUDA not available -- skipping GPU baseline." exception = err
    false
end

if !CUDA_OK
    println("""
    ============================================================================
    impl_baseline_gpu.jl: no functional CUDA GPU detected on host '$(HOST)'.
    Nothing was measured. Run this script on a CUDA-capable machine.
    (CPU dense/recurrence numbers live in dense_vs_loop_blas<N>.csv from
     impl_baseline_cpu.jl and are the comparison target.)
    ============================================================================
    """)
    exit(0)
end

mkpath(OUTDIR)

# GPU timing helper: synchronize device, minimum over samples.
function gpu_timeit(f; setup=nothing, samples::Int=SAMPLES)
    setup === nothing || setup()
    CUDA.@sync f()  # warmup / compile
    best = Inf
    for _ in 1:samples
        setup === nothing || setup()
        dt = CUDA.@elapsed f()
        best = min(best, dt)
    end
    return best
end

# Build per-block device matrices and a closure that runs all the GEMMs.
# Returns (run!, upload!, download!) so we can time kernel-only and with-transfer.
function make_gpu_apply(::Type{T}, blocks, cols) where {T}
    hA = [randn(T, b, b) for b in blocks]
    hX = [randn(T, b, cols) for b in blocks]
    dA = [CuArray(a) for a in hA]
    dX = [CuArray(x) for x in hX]
    dY = [CUDA.zeros(T, b, cols) for b in blocks]
    run! = function ()
        @inbounds for k in eachindex(dA)
            mul!(dY[k], dA[k], dX[k])
        end
        return nothing
    end
    # transfer variant: re-upload X, compute, download Y
    upload!   = () -> (for k in eachindex(dX); copyto!(dX[k], hX[k]); end)
    download! = function ()
        @inbounds for k in eachindex(dY)
            copyto!(hX[k], dY[k])   # reuse hX as a same-size host sink
        end
        return nothing
    end
    run_with_transfer! = function ()
        upload!()
        run!()
        download!()
        return nothing
    end
    return run!, run_with_transfer!
end

function write_env_gpu(io)
    println(io, "# 008c GPU baseline -- environment")
    println(io)
    println(io, "- date: ", Dates.now())
    println(io, "- hostname: ", HOST)
    println(io, "- julia: ", VERSION)
    try
        dev = CUDA.device()
        println(io, "- gpu: ", CUDA.name(dev))
        println(io, "- cuda runtime: ", CUDA.runtime_version())
        println(io, "- total mem (GiB): ", round(CUDA.totalmem(dev) / 2^30; digits=2))
    catch err
        println(io, "- gpu: unknown (", err, ")")
    end
    println(io, "- P_DENSE_LIST: ", P_DENSE_LIST)
    println(io, "- BATCH_LIST: ", BATCH_LIST)
    println(io, "- PREC_LIST: ", PREC_LIST)
    println(io, "- SAMPLES: ", SAMPLES)
    println(io)
    println(io, "Compare seconds_per_expansion against dense_vs_loop_blas<N>.csv (CPU),")
    println(io, "form=recurrence and form=dense, same P and batch. The production")
    println(io, "recurrence is Float64; the precision=Float32 GPU rows quantify the")
    println(io, "speedup from dropping to single precision.")
end

function main()
    println("Writing GPU baseline to: ", OUTDIR)
    open(joinpath(OUTDIR, "env_gpu.md"), "w") do io
        write_env_gpu(io)
    end

    open(joinpath(OUTDIR, "dense_gpu.csv"), "w") do io
        println(io, "stage,form,precision,P,batch,seconds,seconds_per_expansion")
        for P in P_DENSE_LIST
            for T in PREC_LIST
                for B in BATCH_LIST
                    cols = 2 * B  # re + im lanes

                    for (stage, blocks) in (("m2l_z_translation", mblocks(P)),
                                            ("axis_swap", nblocks(P)))
                        run!, run_xfer! = make_gpu_apply(T, blocks, cols)

                        t = gpu_timeit(run!)
                        @printf(io, "%s,device_resident,%s,%d,%d,%.6e,%.6e\n",
                                stage, T, P, B, t, t / B)

                        t = gpu_timeit(run_xfer!)
                        @printf(io, "%s,with_transfer,%s,%d,%d,%.6e,%.6e\n",
                                stage, T, P, B, t, t / B)

                        flush(io)
                    end
                end
            end
        end
    end
    println("Done. Results in: ", OUTDIR)
end

main()
