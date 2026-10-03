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

#------- the masked bodies' slots on the device (moved from FLOWVPM, 2026-10-03) -------#

# sorted slot of each masked body: one thread per slot, a binary search of the
# masked global indices
@kernel function _masked_slots_kernel!(mslot, @Const(perm), @Const(sysid), @Const(bidx),
        @Const(psorted), @Const(korder), K, n)
    s = @index(Global)
    @inbounds if s <= n
        g = perm[s]
        if sysid[g] == 1
            p = bidx[g]
            lo = 1; hi = K
            while lo < hi
                mid = (lo + hi) >>> 1
                if psorted[mid] < p
                    lo = mid + 1
                else
                    hi = mid
                end
            end
            if K >= 1 && psorted[lo] == p
                mslot[korder[lo]] = Int32(s)
            end
        end
    end
end

# the masked grid (FastMultipole.radix_masked_grid) on the device, with the slots
# searched there
function FastMultipole.radix_masked_grid_device(m, nf, backend::KA.Backend)
    up(A) = (d = KA.allocate(backend, eltype(A), size(A)...); copyto!(d, A); d)
    mslot = KA.zeros(backend, Int32, m.K)
    psorted = up(collect(m.psorted)); korder = up(m.korder)
    _masked_slots_kernel!(backend, 256)(mslot, nf.body_perm, nf.body_system_ids, nf.body_indices,
        psorted, korder, m.K, nf.n_bodies; ndrange = cld(nf.n_bodies, 256) * 256)
    return (; K = m.K, mx = up(m.mx), ms = up(m.ms), mslot, offsets = up(m.offsets),
              o = m.origin, h = m.h, dims = m.dims)
end
