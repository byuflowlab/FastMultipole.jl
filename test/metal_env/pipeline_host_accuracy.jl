# Why is the HOST radix FMM less accurate than the KA device port, and why does
# its error grow with np while the device's stays flat?
#
# Hypothesis: the accumulation pattern in the nearfield, not the FMM.
# `_host_direct_pairs_functor_kernel!` (translate_batched_resident.jl:453) adds
# straight into `output[r, i] += ...` inside the inner j loop, so each target's
# accumulator takes one rounding step per SOURCE BODY -- a sequential Float32
# chain whose length grows with bodies-per-cell, i.e. with np at the fixed
# ell=3 the geometry rule picks here. The KA kernel accumulates a whole source
# cell into registers and does ONE atomic add per (pair, target), so its chain
# length is set by the pair count, which the geometry fixes independently of np.
#
# Test: recompute the same host nearfield over the same pair list with BLOCKED
# accumulation (registers per pair, one add-back), and with a Float64
# accumulator, and score all three against a Float64 all-pairs reference. If the
# hypothesis holds, blocked-Float32 tracks the device and plain-Float32 is the
# outlier that grows.

include("ka_backend.jl")
include("pipeline_field.jl")

using FastMultipole, Printf
const FM = FastMultipole
const V = FLOWVPM

const TF = Float32
const STEP = 36
const P = 5
const SIZES = (512, 2048, 8192)


relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))

# the host kernel's own pattern, but accumulating per (pair, target) in
# registers -- the KA kernel's pattern, on the CPU
function blocked_nearfield!(out, kern, sb, cr, dt, ds, npairs, ::Type{ACC}) where ACC
    ghv = Val(:shipped)
    @inbounds for p in 1:npairs
        tc = dt[p]; sc = ds[p]
        tf = cr[1, tc]; tl = tf + cr[2, tc] - 1
        sf = cr[1, sc]; sl = sf + cr[2, sc] - 1
        for i in tf:tl
            xi, yi, zi = sb[1, i], sb[2, i], sb[3, i]
            a2 = zero(ACC); a3 = zero(ACC); a4 = zero(ACC)
            for j in sf:sl
                i == j && continue
                dx = xi - sb[1, j]; dy = yi - sb[2, j]; dz = zi - sb[3, j]
                r2 = dx * dx + dy * dy + dz * dz
                r2 == zero(r2) && continue
                invr = inv(sqrt(r2))
                _, gx, gy, gz = FM._direct_pair_ug(kern, dx, dy, dz, r2, invr, sb, j, ghv)
                a2 += ACC(gx); a3 += ACC(gy); a4 += ACC(gz)
            end
            out[2, i] += a2; out[3, i] += a3; out[4, i] += a4
        end
    end
    return out
end

println("=== host nearfield accumulation: sequential vs blocked ===")
println("(reference = the SAME pair list and math in Float64; velocity rows only)")
flush(stdout)

for np in SIZES
    host = load_wake(STEP; np, TF, P)
    V.UJ_fmm_gpu!(host; reset=true)
    cache = V._radix_fmm_coupling!(host).cache
    st = cache.state
    n = st.counts.n_bodies
    nd = st.counts.n_direct
    sb = Array(st.source_bodies)
    cr = Array(st.cell_ranges)
    dt = Array(st.direct_targets)
    ds = Array(st.direct_sources)
    kern = st.options.direct_kernel

    # (1) the shipped host kernel, sequential Float32 accumulation
    o_seq = zeros(TF, size(st.output, 1), n)
    FM._host_direct_pairs_functor_kernel!(kern, o_seq, sb, cr, dt, ds, nd,
        Val(size(st.output, 1) >= 13))
    # (2) blocked Float32, (3) blocked Float64 -- same pair list, same math
    o_b32 = zeros(TF, size(st.output, 1), n)
    blocked_nearfield!(o_b32, kern, sb, cr, dt, ds, nd, Float32)
    o_b64 = zeros(Float64, size(st.output, 1), n)
    blocked_nearfield!(o_b64, kern, sb, cr, dt, ds, nd, Float64)

    @printf("np=%-6d n_direct=%-8d bodies/cell~%.0f\n", np, nd, n / st.grid.n_cells)
    @printf("   nearfield vs blocked-F64:  shipped seq-F32 %.3e   blocked-F32 %.3e\n",
            relerr(o_seq[2:4, :], o_b64[2:4, :]),
            relerr(o_b32[2:4, :], o_b64[2:4, :]))
    flush(stdout)
end
