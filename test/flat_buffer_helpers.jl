# Shared helpers for the native flat coefficient buffer tests (Matrix Operator
# Refactor, task 017). Convert between the legacy [2,2,nh,batch] arrays the parity
# references use and the native FlatCoefficientBuffer the operators now consume.
#
# φ is packed/unpacked through P_phi and χ through P_active (Val(true)), matching
# the ragged buffer's physical content. `from_flat_buffer!` zeroes the legacy frame
# first, so φ degrees above P_phi read back as exact zero (the ragged buffer has no
# such rows) — this is what the Val(true) "no nonphysical φ output" check verifies.

flat_hi(n, m) = FastMultipole.harmonic_index(n, m)
flat_fbi(n, m, r) = FastMultipole.flat_basis_index(n, m, r)

function to_flat_buffer(legacy4d, basis_info)
    TF = eltype(legacy4d)
    nbatch = size(legacy4d, 4)
    buf = FlatCoefficientBuffer(TF, basis_info, nbatch)
    LH = basis_info.channel_count == 2
    P_phi = basis_info.orders.P_phi
    P_active = basis_info.orders.P_active
    for j in 1:nbatch
        for n in 0:P_phi, m in 0:n
            i = flat_hi(n, m); fr = flat_fbi(n, m, 1)
            buf.phi[fr, j] = legacy4d[1, 1, i, j]
            buf.phi[fr + 1, j] = legacy4d[2, 1, i, j]
        end
        if LH
            for n in 0:P_active, m in 0:n
                i = flat_hi(n, m); fr = flat_fbi(n, m, 1)
                buf.chi[fr, j] = legacy4d[1, 2, i, j]
                buf.chi[fr + 1, j] = legacy4d[2, 2, i, j]
            end
        end
    end
    return buf
end

function from_flat_buffer!(legacy4d, buf)
    fill!(legacy4d, zero(eltype(legacy4d)))
    basis_info = buf.basis_info
    LH = basis_info.channel_count == 2
    P_phi = basis_info.orders.P_phi
    P_active = basis_info.orders.P_active
    for j in 1:size(legacy4d, 4)
        for n in 0:P_phi, m in 0:n
            i = flat_hi(n, m); fr = flat_fbi(n, m, 1)
            legacy4d[1, 1, i, j] = buf.phi[fr, j]
            legacy4d[2, 1, i, j] = buf.phi[fr + 1, j]
        end
        if LH
            for n in 0:P_active, m in 0:n
                i = flat_hi(n, m); fr = flat_fbi(n, m, 1)
                legacy4d[1, 2, i, j] = buf.chi[fr, j]
                legacy4d[2, 2, i, j] = buf.chi[fr + 1, j]
            end
        end
    end
    return legacy4d
end
