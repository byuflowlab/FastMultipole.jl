#------- device row selection for the oversize mask (moved from FLOWVPM, 2026-10-03) -------#
#
# `radix_rows_top`: a histogram of the row over [lo, hi] finds the bin holding the
# K-th largest value, a compaction collects every column at or above that bin's
# lower edge. `radix_rows_above`: one compaction. Both sort the result, so the list
# does not depend on the compaction's arrival order.

@kernel function _row_hist_kernel!(hist, @Const(P), row, np, lo, inv_w, nb)
    i = @index(Global)
    @inbounds if i <= np
        b = clamp(floor(Int32, (P[row, i] - lo) * inv_w) + Int32(1), Int32(1), Int32(nb))
        KernelAbstractions.@atomic hist[b] += Int32(1)
    end
end
@kernel function _row_collect_kernel!(idx, counter, @Const(P), row, np, thr, cap)
    i = @index(Global)
    @inbounds if i <= np && P[row, i] >= thr
        k = KernelAbstractions.@atomic counter[1] += Int32(1)
        k <= cap && (idx[k] = Int32(i))
    end
end
# every column in histogram bin `b` or above, binned exactly as
# `_row_hist_kernel!` bins it, so the count is the histogram's
@kernel function _row_collect_bins_kernel!(idx, counter, @Const(P), row, np, lo, inv_w, nb, b, cap)
    i = @index(Global)
    @inbounds if i <= np &&
            clamp(floor(Int32, (P[row, i] - lo) * inv_w) + Int32(1), Int32(1), Int32(nb)) >= b
        k = KernelAbstractions.@atomic counter[1] += Int32(1)
        k <= cap && (idx[k] = Int32(i))
    end
end

function FastMultipole.radix_rows_top(P::AnyGPUMatrix, row::Int, np::Int, K::Int)
    backend = KA.get_backend(P)
    T = eltype(P); nb = 2048
    lo, hi = FastMultipole._device_row_extrema(P, row, np)
    hi > lo * (1 + 1e-6) || return Int[]                 # all equal: nothing to mask
    hist = KA.zeros(backend, Int32, nb)
    inv_w = T(nb) / (T(hi) - T(lo))
    _row_hist_kernel!(backend, 256)(hist, P, row, np, T(lo), inv_w, nb; ndrange=np)
    KA.synchronize(backend)
    h = Array(hist)
    # walk the bins from the top until K columns are covered
    acc = 0; b = nb
    while b > 1 && acc + h[b] < K
        acc += h[b]; b -= 1
    end
    b == nb && acc + h[b] <= 1 && return Int[]           # only the maximum itself: skip
    cap = acc + h[b]
    idx = KA.zeros(backend, Int32, cap); counter = KA.zeros(backend, Int32, 1)
    _row_collect_bins_kernel!(backend, 256)(idx, counter, P, row, np, T(lo), inv_w, nb, Int32(b), Int32(cap); ndrange=np)
    KA.synchronize(backend)
    n = min(Int(Array(counter)[1]), cap)
    return sort!(Int.(Array(view(idx, 1:n))))
end

function FastMultipole.radix_rows_above(P::AnyGPUMatrix, row::Int, np::Int, thr, cap::Int)
    backend = KA.get_backend(P)
    T = eltype(P)
    idx = KA.zeros(backend, Int32, cap); counter = KA.zeros(backend, Int32, 1)
    _row_collect_kernel!(backend, 256)(idx, counter, P, row, np, nextfloat(T(thr)), Int32(cap); ndrange=np)
    KA.synchronize(backend)
    n = Int(Array(counter)[1])
    if n > cap
        # which `cap` of them the compaction kept depends on arrival order:
        # take the largest `cap` on the host instead, as the Matrix method does
        sig = Array(view(P, row, 1:np))
        lim = nextfloat(T(thr))                 # the device collect's comparison
        above = findall(>=(lim), sig)
        return sort!(above[partialsortperm(view(sig, above), 1:cap; rev=true)])
    end
    return sort!(Int.(Array(view(idx, 1:n))))
end
