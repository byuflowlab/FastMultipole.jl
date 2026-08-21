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
#   * same GEMM-relevant stage shapes
#       - M2M/M2L/L2L z-translation : block-diagonal over m, block size (P+1-m)
#       - axis-swap (y rot) : block-diagonal over n, block size (2n+1)
#   * same batch sweep
#   * re/im modeled as 2 columns per expansion
# so CPU<->GPU seconds_per_expansion numbers are directly comparable.
#
# Two GPU data-residency variants are measured per (stage, launch strategy, batch):
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
#       dense_gpu.csv  (columns include launch_strategy and transfer_variant)
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

function progress(msg)
    println("[", Dates.format(Dates.now(), "HH:MM:SS"), "] ", msg)
    flush(stdout)
end

# block sizes (identical to impl_baseline_cpu.jl)
mblocks(P) = [P + 1 - m for m in 0:P]   # M2L z-translation
nblocks(P) = [2n + 1 for n in 0:P]      # axis-swap (y rotation)
block_offsets(blocks) = cumsum(vcat(1, blocks[1:end-1]))

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

function fused_block_kernel!(Y, A, X, offsets, sizes, nblocks, nrows, cols)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    total = nrows * cols
    while idx <= total
        row = (idx - 1) % nrows + 1
        col = (idx - 1) ÷ nrows + 1

        k = 1
        @inbounds while k < nblocks && row >= offsets[k + 1]
            k += 1
        end
        @inbounds begin
            offset = offsets[k]
            b = sizes[k]
            local_i = row - offset + 1
            s = zero(eltype(Y))
            for h in 1:b
                s += A[row, h] * X[offset + h - 1, col]
            end
            Y[row, col] = s
        end
        idx += stride
    end
    return nothing
end

"""
Single-launch packed-block GPU prototype. It intentionally uses a simple custom
kernel rather than cuBLAS so we can measure the launch-fusion side of the design
space for the small block-diagonal operators.
"""
function make_gpu_fused_apply(::Type{T}, blocks, cols) where {T}
    offsets = block_offsets(blocks)
    nrows = sum(blocks)
    maxb = maximum(blocks)
    hA = zeros(T, nrows, maxb)
    @inbounds for k in eachindex(blocks)
        b = blocks[k]
        r0 = offsets[k]
        hA[r0:(r0 + b - 1), 1:b] .= randn(T, b, b)
    end
    hX = randn(T, nrows, cols)
    hY = zeros(T, nrows, cols)
    dA = CuArray(hA)
    dX = CuArray(hX)
    dY = CUDA.zeros(T, nrows, cols)
    dOffsets = CuArray(Int32.(offsets))
    dSizes = CuArray(Int32.(blocks))

    run! = function ()
        total = nrows * cols
        threads = 256
        blocks_grid = min(cld(total, threads), 65535)
        @cuda threads=threads blocks=blocks_grid fused_block_kernel!(
            dY, dA, dX, dOffsets, dSizes, Int32(length(blocks)), Int32(nrows), Int32(cols)
        )
        return nothing
    end
    run_with_transfer! = function ()
        copyto!(dX, hX)
        run!()
        copyto!(hY, dY)
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
    progress("Writing GPU baseline to: $OUTDIR")
    progress("Sweep parameters: P_DENSE_LIST=$P_DENSE_LIST BATCH_LIST=$BATCH_LIST PREC_LIST=$PREC_LIST SAMPLES=$SAMPLES")
    try
        dev = CUDA.device()
        progress("CUDA device: $(CUDA.name(dev)) runtime=$(CUDA.runtime_version()) total_mem_GiB=$(round(CUDA.totalmem(dev) / 2^30; digits=2))")
    catch err
        progress("CUDA device metadata unavailable: $err")
    end

    open(joinpath(OUTDIR, "env_gpu.md"), "w") do io
        write_env_gpu(io)
    end
    progress("Wrote GPU environment metadata: $(joinpath(OUTDIR, "env_gpu.md"))")

    open(joinpath(OUTDIR, "dense_gpu.csv"), "w") do io
        println(io, "stage,form,launch_strategy,transfer_variant,precision,P,batch,seconds,seconds_per_expansion")
        for P in P_DENSE_LIST
            progress("gpu dense sweep: P=$P")
            for T in PREC_LIST
                progress("  precision=$(string(T))")
                for B in BATCH_LIST
                    cols = 2 * B  # re + im lanes
                    progress("    batch=$B cols=$cols")

                    for (stage, blocks) in (("m2m_z_translation", mblocks(P)),
                                            ("m2l_z_translation", mblocks(P)),
                                            ("l2l_z_translation", mblocks(P)),
                                            ("axis_swap", nblocks(P)))
                        progress("      preparing stage=$stage")
                        run!, run_xfer! = make_gpu_apply(T, blocks, cols)

                        progress("      timing stage=$stage launch=per_block_launch transfer=device_resident")
                        t = gpu_timeit(run!)
                        @printf(io, "%s,dense,per_block_launch,device_resident,%s,%d,%d,%.6e,%.6e\n",
                                stage, string(T), P, B, t, t / B)

                        progress("      timing stage=$stage launch=per_block_launch transfer=with_transfer")
                        t = gpu_timeit(run_xfer!)
                        @printf(io, "%s,dense,per_block_launch,with_transfer,%s,%d,%d,%.6e,%.6e\n",
                                stage, string(T), P, B, t, t / B)

                        run!, run_xfer! = make_gpu_fused_apply(T, blocks, cols)

                        progress("      timing stage=$stage launch=fused_kernel transfer=device_resident")
                        t = gpu_timeit(run!)
                        @printf(io, "%s,dense,fused_kernel,device_resident,%s,%d,%d,%.6e,%.6e\n",
                                stage, string(T), P, B, t, t / B)

                        progress("      timing stage=$stage launch=fused_kernel transfer=with_transfer")
                        t = gpu_timeit(run_xfer!)
                        @printf(io, "%s,dense,fused_kernel,with_transfer,%s,%d,%d,%.6e,%.6e\n",
                                stage, string(T), P, B, t, t / B)

                        flush(io)
                    end
                end
            end
        end
    end
    progress("Wrote dense_gpu.csv")
    progress("Done. Results in: $OUTDIR")
end

main()
