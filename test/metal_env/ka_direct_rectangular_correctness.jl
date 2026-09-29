# Gate: direct_rectangular! on device arrays (ext ka_rect_kernel!) against the
# threaded host method, velocity + gradient + potential, for a kernel type
# defined in this file: the device method must reach a consumer's own
# `rect_pair` without any device-specific code on the consumer side.
include("ka_backend.jl")
using FastMultipole, Random, StaticArrays
const FM = FastMultipole
if !dev_functional(); println("$(DEV_NAME) not functional; skipping"); exit(0); end
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext !== nothing || error("FastMultipoleKAExt did not load")

# singular point source, rows x y z q
struct DevRectSource <: FM.AbstractRectangularKernel end
FM.rect_source_rows(::DevRectSource) = 4
FM.rect_has_potential(::DevRectSource) = true
@inline function FM.rect_pair(::DevRectSource, target::SVector{3,T}, sources, q,
        ::Val{GRAD}, ::Val{POT}) where {T,GRAD,POT}
    @inbounds d = target - SVector{3,T}(sources[1, q], sources[2, q], sources[3, q])
    @inbounds s = sources[4, q]
    r2 = d[1]*d[1] + d[2]*d[2] + d[3]*d[3]
    iszero(r2) && return zero(SVector{3,T}), zero(SMatrix{3,3,T,9}), zero(T)
    r = sqrt(r2)
    c = s / (4 * T(pi) * r2 * r)
    g = GRAD ? c * (SMatrix{3,3,T,9}(1, 0, 0, 0, 1, 0, 0, 0, 1) - 3 * d * transpose(d) / r2) :
        zero(SMatrix{3,3,T,9})
    return c * d, g, POT ? s / (4 * T(pi) * r) : zero(T)
end

relerr(a, b) = (s = maximum(abs.(b)); d = maximum(abs.(a .- b)); s == 0 ? d : d / s)
npass = Ref(0); nfail = Ref(0)
TF = Float32
Random.seed!(9100)

let n_tgt = 300, n_src = 200
    tgt = rand(TF, 3, n_tgt) .+ TF(1.2); src = vcat(rand(TF, 3, n_src), rand(TF, 1, n_src) .- TF(0.5))
    for grad in (false, true), pot in (false, true)
        rows = FM.rect_output_rows(grad, pot)
        ref = zeros(TF, rows, n_tgt)
        FM.direct_rectangular!(ref, tgt, DevRectSource(), src; gradient = grad, scalar_potential = pot)
        out = devarray(zeros(TF, rows, n_tgt))
        try
            FM.direct_rectangular!(out, devarray(tgt), DevRectSource(), devarray(src); gradient = grad, scalar_potential = pot)
        catch e
            nfail[] += 1; println("  FAIL grad=$grad pot=$pot: threw ", sprint(showerror, e)[1:min(end, 3000)]); continue
        end
        e = relerr(Array(out), ref)
        ok = e <= 1e-5
        ok ? (npass[] += 1) : (nfail[] += 1)
        println("  ", ok ? "PASS" : "FAIL", "  consumer kernel grad=$grad pot=$pot relerr=$(round(e, sigdigits = 3))")
    end
    # mixed host/device arguments are refused
    try
        FM.direct_rectangular!(devarray(zeros(TF, 3, n_tgt)), tgt, DevRectSource(), devarray(src))
        nfail[] += 1; println("  FAIL mixed host/device arguments were accepted")
    catch e
        ok = e isa ArgumentError
        ok ? (npass[] += 1) : (nfail[] += 1)
        println("  ", ok ? "PASS" : "FAIL", "  mixed host/device arguments throw ", typeof(e))
    end
end
println("\nKA direct_rectangular! on $(DEV_NAME): $(npass[]) passed, $(nfail[]) failed")
nfail[] == 0 || error("direct_rectangular! device gate failed")
println("✓✓✓ direct_rectangular! device gate passed on $(DEV_NAME) ✓✓✓")
