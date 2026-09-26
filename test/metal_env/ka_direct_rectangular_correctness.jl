# Gate: direct_rectangular! on device arrays (ext ka_rect_points_kernel! /
# ka_rect_panels_kernel!) against the threaded host method, both functors,
# velocity + gradient (+ potential for panels), all four filament
# regularization codes on the panel functor.
include("ka_backend.jl")
using FastMultipole, Random, StaticArrays
const FM = FastMultipole
if !dev_functional(); println("$(DEV_NAME) not functional; skipping"); exit(0); end
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext !== nothing || error("FastMultipoleKAExt did not load")

relerr(a, b) = (s = maximum(abs.(b)); d = maximum(abs.(a .- b)); s == 0 ? d : d / s)
npass = Ref(0); nfail = Ref(0)
TF = Float32
Random.seed!(9100)

# --- point functor (RectangularGaussianErfVortex): rows x y z Gx Gy Gz sigma ---
let n_tgt = 300, n_src = 200
    tgt = rand(TF, 3, n_tgt); src = rand(TF, 7, n_src); src[4:6, :] .-= TF(0.5); src[7, :] .= TF(0.05)
    for grad in (false, true)
        rows = grad ? 12 : 3
        ref = zeros(TF, rows, n_tgt); FM.direct_rectangular!(ref, tgt, FM.RectangularGaussianErfVortex(), src; gradient = grad)
        out = devarray(zeros(TF, rows, n_tgt))
        try
            FM.direct_rectangular!(out, devarray(tgt), FM.RectangularGaussianErfVortex(), devarray(src); gradient = grad)
        catch e
            nfail[] += 1; println("  FAIL points grad=$grad: threw ", sprint(showerror, e)[1:min(end, 300)]); continue
        end
        e = relerr(Array(out), ref)
        ok = e <= 5e-6
        ok ? (npass[] += 1) : (nfail[] += 1)
        println("  ", ok ? "PASS" : "FAIL", "  points grad=$grad relerr=$(round(e, sigdigits = 3))")
    end
end

# --- panel functor: rows tag nv v1(3) v2(3) v3(3) v4(3) s1 s2 core ---
function pack_panel!(A, q, tag, verts, s1, s2, core)
    A[1, q] = tag; A[2, q] = length(verts)
    for (iv, v) in enumerate(verts); A[3 + 3(iv - 1):5 + 3(iv - 1), q] .= v; end
    A[15, q] = s1; A[16, q] = s2; A[17, q] = core
    return A
end
let n_tgt = 200, n_src = 120
    src = zeros(TF, 17, n_src)
    for q in 1:n_src
        c = rand(TF, 3); tag = rand((1, 2, 3, 4, 5))
        # tag 3 with nv = 2 (open filament) has no potential: keep every source a triangle
        verts = (c, c .+ TF(0.02) .* rand(TF, 3), c .+ TF(0.02) .* rand(TF, 3))
        pack_panel!(src, q, tag, verts, rand(TF) - TF(0.5), rand(TF) - TF(0.5), TF(0.01))
    end
    tgt = rand(TF, 3, n_tgt) .+ TF(0.3)
    # every family in Float32; on backends with Float64 (CUDA) LineGauss again in Float64
    has_f64 = DEV_NAME != "Metal"
    for reg in 1:4, grad in (false, true), pot in (false, true), T2 in (TF, Float64)
        T2 == Float64 && !(has_f64 && reg == 4) && continue
        rows = FM.rect_output_rows(grad, pot)
        kern = FM.RectangularPanelInfluence(Int32(reg))
        ref = zeros(T2, rows, n_tgt)
        FM.direct_rectangular!(ref, T2.(tgt), kern, T2.(src); gradient = grad, scalar_potential = pot)
        out = devarray(zeros(T2, rows, n_tgt))
        try
            FM.direct_rectangular!(out, devarray(T2.(tgt)), kern, devarray(T2.(src)); gradient = grad, scalar_potential = pot)
        catch e
            nfail[] += 1; println("  FAIL panels reg=$reg grad=$grad pot=$pot: threw ", sprint(showerror, e)[1:min(end, 3000)]); continue
        end
        e = relerr(Array(out), ref)
        ok = e <= 5e-5
        ok ? (npass[] += 1) : (nfail[] += 1)
        println("  ", ok ? "PASS" : "FAIL", "  panels reg=$reg $T2 grad=$grad pot=$pot relerr=$(round(e, sigdigits = 3))")
    end
end
println("\nKA direct_rectangular! on $(DEV_NAME): $(npass[]) passed, $(nfail[]) failed")
nfail[] == 0 || error("direct_rectangular! device gate failed")
println("✓✓✓ direct_rectangular! device gate passed on $(DEV_NAME) ✓✓✓")
