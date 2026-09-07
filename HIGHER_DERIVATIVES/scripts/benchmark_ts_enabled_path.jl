#=##############################################################################
Enabled-path performance record for the third-derivative feature
(HIGHER_DERIVATIVES/REVIEW_PLAN.md, Test and Acceptance Plan): record
enabled-path time, allocations, scratch bytes per worker, and output-buffer
bytes, WITHOUT imposing a speed threshold (no correct baseline exists yet).

Usage (from the feature checkout):
    julia --project=test --threads=1 benchmark_ts_enabled_path.jl [n_bodies]

Run `Pkg.develop(path=".")` in the test env first if FastMultipole is not
already dev'ed there. Default n_bodies = 30_000.
=###############################################################################

using FastMultipole
using FastMultipole.StaticArrays
using FastMultipole.LinearAlgebra
using Random
using Statistics

repo = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(repo, "test", "gravitational.jl"))

n_bodies = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 30_000
expansion_order = 5
TF = Float64

system = generate_gravitational(2026, n_bodies)
kwargs = (; scalar_potential=false, gradient=true, hessian=false,
    leaf_size=50, expansion_order, multipole_acceptance=0.4)

run_off() = (system.potential .= 0; fmm!(system; kwargs...))
run_on() = (system.potential .= 0; fmm!(system; kwargs..., third_derivative=true))

# warm-up
run_off(); run_on()

n_rep = 5
t_off = [(@elapsed run_off()) for _ in 1:n_rep]
t_on = [(@elapsed run_on()) for _ in 1:n_rep]
a_off = @allocated run_off()
a_on = @allocated run_on()

# scratch bytes per worker: the TS-aware L2B scratch vs the ordinary one
scratch_on = Base.summarysize(FastMultipole.initialize_gradient_n_m(expansion_order, TF; third_derivative=true))
scratch_off = Base.summarysize(FastMultipole.initialize_gradient_n_m(expansion_order, TF; third_derivative=false))

# output-buffer bytes: 18 extra target-buffer rows per body when TS is on
switch_on = DerivativesSwitch(false, true, false; third_derivative=true)
switch_off = DerivativesSwitch(false, true, false)
rows_on = FastMultipole.target_buffer_rows(switch_on)
rows_off = FastMultipole.target_buffer_rows(switch_off)

println("n_bodies = $n_bodies, expansion_order = $expansion_order, $(Threads.nthreads()) thread(s)")
println("fmm! median time      off: $(round(median(t_off); digits=4)) s   on: $(round(median(t_on); digits=4)) s   (n_rep=$n_rep)")
println("fmm! allocations      off: $a_off bytes   on: $a_on bytes")
println("L2B scratch/worker    off: $scratch_off bytes   on: $scratch_on bytes")
println("target buffer rows    off: $rows_off   on: $rows_on   (+$(rows_on - rows_off) rows = $((rows_on - rows_off) * n_bodies * sizeof(TF)) output bytes at n=$n_bodies)")
