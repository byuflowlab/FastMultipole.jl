# Compare the saved end states of leapfrog_bench.jl arms against the CPU one.
using Serialization, Printf, Statistics
dir = ARGS[1]
ref = deserialize(joinpath(dir, "final_cpu_Float64.jls"))
relmax(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))
rell2(a, b) = sqrt(sum(abs2, a .- b) / sum(abs2, b))
println("reference: CPU Float64, np=$(ref.n), $(ref.NSTEPS) steps")
@printf("%-22s %12s %12s %12s %12s\n", "arm", "X max/R", "X L2", "U max", "U L2")
Rr = 0.7906
arms = Dict{String,Any}()
for f in sort(filter(x -> startswith(x, "final_") && x != "final_cpu_Float64.jls", readdir(dir)))
    a = deserialize(joinpath(dir, f)); arms[f] = a
    @printf("%-22s %12.3e %12.3e %12.3e %12.3e\n", replace(f, "final_" => "", ".jls" => ""),
            maximum(abs.(a.X .- ref.X)) / Rr, rell2(a.X, ref.X), relmax(a.U, ref.U), rell2(a.U, ref.U))
end
for TF in ("Float32", "Float64")
    k, nt = "final_ka_$TF.jls", "final_native_$TF.jls"
    if haskey(arms, k) && haskey(arms, nt)
        a, b = arms[k], arms[nt]
        @printf("KA vs native %-8s  X max/R %.3e   U max %.3e  U L2 %.3e\n", TF,
                maximum(abs.(a.X .- b.X)) / Rr, relmax(a.U, b.U), rell2(a.U, b.U))
    end
end
