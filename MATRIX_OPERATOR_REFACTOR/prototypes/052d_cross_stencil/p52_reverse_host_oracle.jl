# P5.2 runner — see p52_lib.jl for the fixture + host pipeline.
include(joinpath(@__DIR__, "p52_lib.jl"))

# ---- verdict: P-convergence sweep ----
# Expansion error must decay ~exponentially with P at fixed geometry; a table
# or scaling bug plateaus. PASS on strong decay + small terminal error.
maxrel = Float64[]
meanrel = Float64[]
for P in (4, 6, 8, 10, 12)
    U_tot = far_field(P) .+ U_near
    rel = [norm(U_tot[i] - U_ref[i]) / norm(U_ref[i]) for i in 1:nt]
    push!(maxrel, maximum(rel))
    push!(meanrel, sum(rel) / nt)
    @printf("  P=%2d  relU: max %.3e  mean %.3e  (q=%d, ell=%d)\n",
        P, maxrel[end], meanrel[end], Q, ELL)
end
check("relU decays with P (P=12 < P=4 / 100)", maxrel[end] < maxrel[1] / 100)
# q=3 admits |o|=2 M2L pairs (box gap = one cell width), so worst-case
# convergence is slow; observed 2026-08-31: max 1.14e-3, mean 2.1e-5 at P=12
# with clean ~4x/2-orders geometric decay (a table/scaling bug plateaus
# instead). Production runs q=12 where separations are far wider.
check("terminal accuracy (P=12 max relU < 2e-3)", maxrel[end] < 2e-3)
check("terminal accuracy (P=12 mean relU < 1e-4)", meanrel[end] < 1e-4)

@printf("\nP5.2 reverse host oracle: %d PASS, %d FAIL\n", npass, nfail)
exit(nfail == 0 ? 0 : 1)
