# Shared offset/class LUT builder for the KA interaction-list suites.
#
# NOTE: `ka_radix_lists_wiring_correctness.jl` still carries its own identical
# copy of `build_luts`. That suite is green and gated, so it was left untouched
# rather than refactored; fold it onto this file the next time it is edited.
function build_luts(reach::Int, q::Int, ell_max::Int)
    side = 2 * reach + 1
    offset_lut = zeros(Int32, side, side, side)
    k = 0
    for oz in -reach:reach, oy in -reach:reach, ox in -reach:reach
        if ox * ox + oy * oy + oz * oz > q
            k += 1
            offset_lut[ox + reach + 1, oy + reach + 1, oz + reach + 1] = Int32(k)
        end
    end
    return offset_lut, ones(Int32, 8, max(k, 1), ell_max + 1), k
end
