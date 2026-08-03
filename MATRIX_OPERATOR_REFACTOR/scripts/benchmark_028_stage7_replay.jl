# Task 028 Stage 7: linear sampled-field replay and q=5 -> q=6 error attribution.
# Benchmark-only: production public API and normal lifecycle behavior are unchanged.

using FastMultipole
using FastMultipole.StaticArrays
using CUDA
using LinearAlgebra
using Statistics
using Dates
using Printf
using SHA

const FM = FastMultipole
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(REPO, "test", "gravitational.jl"))
include(joinpath(@__DIR__, "benchmark_024b_common.jl"))
include(joinpath(@__DIR__, "fm028_device_system.jl"))

FM.load_cuda_radix_lifecycle!() || error(FM.cuda_radix_status())
FM.DENSE_CUDA_TILED_THREADS[] = 64
FM.DENSE_CUDA_TILED_MAX_BLOCKS[] = 65536
FM.RADIX_CUDA_COUNTING_SORT[] = true
FM.CUDA_OVERLAP_NEARFIELD[] = false

const N = parse(Int, get(ENV, "FM028_N", "1000000"))
const P = parse(Int, get(ENV, "FM028_P", "3"))
const ELL = parse(Int, get(ENV, "FM028_ELL", "5"))
const TF = get(ENV, "FM028_TF", "Float32") == "Float64" ? Float64 : Float32
const SEED = 24025
const SAMPLE_SEED = 24026
const BOX_MIN = SVector(-0.01, -0.01, -0.01)
const BOX_SIZE = 1.02
const STAMP = Dates.format(now(), "yyyymmdd-HHMMSS")
const OUT = get(ENV, "FM028_REPLAY_OUT", joinpath(@__DIR__, "..", "data",
    "feasibility_1m_10ms", "stage7_replay_$(gethostname())_$(STAMP).csv"))

function source_manifest()
    ctx = SHA.SHA256_CTX()
    for f in sort(filter(f -> endswith(f, ".jl"), readdir(joinpath(REPO, "src"))))
        SHA.update!(ctx, codeunits(f))
        SHA.update!(ctx, read(joinpath(REPO, "src", f)))
    end
    return bytes2hex(SHA.digest!(ctx))[1:16]
end

function sample_field(sys, indices)
    p, g = fm028_sampled_output(sys, indices)
    return vcat(reshape(Float64.(p), 1, :), Float64.(g))
end

function finalize_sample!(sys, state, indices)
    switches = (FM.DerivativesSwitch(true, true, false, sys),)
    FM.finalize_cuda_radix_output!(state, (sys,); derivatives_switches=switches)
    CUDA.synchronize()
    return sample_field(sys, indices)
end

function replay_fields(q, bodies, indices)
    sys = FM028DeviceSystem{TF}(bodies)
    stencil_eps = rigid_stencil_epsilon(P, TF(BOX_SIZE / 2), ELL, q; TF)
    policy = HierarchicalRigidStencil(P, stencil_eps; near_radius2=q,
        window_classes=length(RigidHierarchicalTables(q).push_offsets))
    opts = CUDARadixLifecycleOptions(; precision=TF,
        operator=MaterializedYRotationM2L(), m2l_strategy=DenseTranslationM2L())
    cache = RadixFMMCache(sys; expansion_order=P, ell=ELL, max_n_bodies=N,
        bounds=(BOX_MIN, BOX_SIZE), device=true, options=opts, policy)
    fmm!(sys, cache; scalar_potential=true, gradient=true)
    normal = sample_field(sys, indices)
    state = cache.state
    FM.update_cuda_radix_state!(cache, (sys,))
    FM._launch_cuda_b2m!(state)
    FM._launch_cuda_resident_m2m!(state)

    fields = Dict{Tuple{Int,NTuple{3,Int}},Matrix{Float64}}()
    FM._launch_cuda_nearfield_kernel!(state)
    fields[(0, (0, 0, 0))] = finalize_sample!(sys, state, indices)

    hctx = state.interaction_list
    level_class_of = Array(hctx.d_class_of)
    orbits = sort!(unique(FM._rigid_orbit_key(o) for o in hctx.tables.push_offsets))
    for L in 2:ELL, orbit in orbits
        # Skip orbit-level combinations absent from the complete phase mask.
        ks = findall(o -> FM._rigid_orbit_key(o) == orbit, hctx.tables.push_offsets)
        any(any(!iszero, @view level_class_of[:, k, L + 1]) for k in ks) || continue
        FM._launch_cuda_hierarchical_m2l_replay!(state; levels=L:L, orbit)
        FM._launch_cuda_resident_l2l!(state)
        fill!(state.output, zero(TF))
        FM._launch_cuda_resident_l2b_only!(state, nothing)
        fields[(L, orbit)] = finalize_sample!(sys, state, indices)
    end
    reconstructed = reduce(+, values(fields))
    denom = max(norm(normal), eps(Float64))
    reconstruction_rel = norm(reconstructed - normal) / denom
    tol = TF === Float32 ? 2e-5 : 2e-11
    reconstruction_rel <= tol || error(
        "q=$q replay reconstruction $reconstruction_rel exceeds tolerance $tol")
    return normal, fields, reconstruction_rel
end

bodies = fm028_body_matrix(SEED, N)
indices = fm024b_expected_reference_indices(N, SAMPLE_SEED)
dpos = CUDA.CuArray(bodies[1:3, :])
dstr = CUDA.CuArray(bodies[5, :])
reference = Float64.(Array(fm028_direct_sample_reference(dpos, dstr, indices)))
CUDA.unsafe_free!(dpos)
CUDA.unsafe_free!(dstr)

normal5, fields5, reconstruction5 = replay_fields(5, bodies, indices)
GC.gc(); CUDA.reclaim()
normal6, fields6, reconstruction6 = replay_fields(6, bodies, indices)
err5 = vec(normal5[2:4, :] - reference[2:4, :])
refgrad = vec(reference[2:4, :])
refnorm = norm(refgrad)
total_rel = norm(err5) / refnorm

keys_all = sort!(collect(union(Set(keys(fields5)), Set(keys(fields6))));
    by=k -> (k[1], k[2]))
rows = NamedTuple[]
z = zeros(Float64, size(normal5))
for key in keys_all
    f5 = get(fields5, key, z)
    f6 = get(fields6, key, z)
    delta = vec(f6[2:4, :] - f5[2:4, :])
    contribution_rel_rms = norm(delta) / refnorm
    correlation = norm(delta) == 0 || norm(err5) == 0 ? 0.0 :
        dot(delta, err5) / (norm(delta) * norm(err5))
    counterfactual = norm(err5 + delta) / refnorm
    push!(rows, (; manifest=source_manifest(), job=get(ENV, "SLURM_JOB_ID", ""),
        precision=string(TF), n=N, ell=ELL, group_level=key[1],
        orbit=join(key[2], ':'), q5_total_gradient_rel_rms=total_rel,
        correction_gradient_rel_rms=contribution_rel_rms,
        correction_correlation=correlation,
        counterfactual_gradient_rel_rms=counterfactual,
        q5_reconstruction_rel=reconstruction5,
        q6_reconstruction_rel=reconstruction6))
end

mkpath(dirname(OUT))
open(OUT, "w") do io
    println(io, join(string.(keys(first(rows))), ','))
    for row in rows
        println(io, join(string.(values(row)), ','))
    end
end
@printf("q5 total gradient relative RMS %.6e; reconstruction q5 %.3e q6 %.3e\n",
    total_rel, reconstruction5, reconstruction6)
println("wrote ", OUT)
