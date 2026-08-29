# Correctness gate for `ka_collect_positions!` / `ka_extract_source_positions_kernel!`
# (ext/FastMultipoleKAExt.jl), the KA port of `_radix_cache_collect_positions!`
# (src/translate_batched_cuda.jl:6612) and `_cuda_extract_source_positions_kernel!`
# (:90) -- the first stage of `update_cuda_radix_state!`, which gathers every
# system's xyz rows into one concatenated position array and records each body's
# (system, within-system index) attribution.
#
# The reference is the definition itself, written out on the host: the stage has
# no host oracle in the package because on the CPU path the positions are read
# straight out of the systems. Multi-system offsets and a ragged final system are
# what actually break here, so the cases vary the system count and sizes.
include("ka_backend.jl")
using Test

using FastMultipole

const ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)

println("Starting KA source-position extraction test on $DEV_NAME...")
if !dev_functional()
    println("$DEV_NAME not functional; skipping")
    exit(0)
end

function run_case(TF, dpb, sizes)
    n = sum(sizes)
    host_bufs = [rand(TF, dpb, nb) for nb in sizes]
    dev_bufs = Tuple(devarray(b) for b in host_bufs)

    # capacity-sized, as in the cache: the tail past `n` must stay untouched
    cap = n + 7
    positions = devarray(fill(TF(-1), 3, cap))
    body_system = devarray(fill(Int32(-1), cap))
    body_index = devarray(fill(Int32(-1), cap))

    total = ext.ka_collect_positions!(positions, body_system, body_index, dev_bufs)
    total == n || error("returned total=$total, expected $n")

    ref_pos = fill(TF(-1), 3, cap)
    ref_sys = fill(Int32(-1), cap)
    ref_idx = fill(Int32(-1), cap)
    offset = 0
    for (isys, b) in enumerate(host_bufs)
        for i in axes(b, 2)
            ref_pos[:, offset + i] .= b[1:3, i]
            ref_sys[offset + i] = isys
            ref_idx[offset + i] = i
        end
        offset += size(b, 2)
    end

    Array(positions)   == ref_pos || error("positions mismatch (sizes=$sizes)")
    Array(body_system) == ref_sys || error("body_system mismatch (sizes=$sizes)")
    Array(body_index)  == ref_idx || error("body_index mismatch (sizes=$sizes)")
    return n
end

for (dpb, sizes) in ((8, (1000,)), (8, (500, 300)), (12, (7, 1024, 33)),
                     (8, (1,)), (8, (2048, 1)))
    n = run_case(Float32, dpb, sizes)
    println("✓ dpb=$dpb sizes=$sizes: $n bodies extracted and attributed correctly")
end

println("\n✓✓✓ KA source-position gate passed on $DEV_NAME ✓✓✓")
