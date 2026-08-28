# CPU vs Metal (KA) speed comparison for the single-group resident M2M path
# (ka_resident_stage_group_apply!, wired via FastMultipole.ka_m2m_operator_batch!).
# Not a correctness check (see ka_m2m_correctness.jl for that) -- this only times the
# CPU-backend branch (plain FastMultipole.m2m_operator_batch! delegate) against the
# Metal-backend branch (the real KA kernel chain) across a range of batch sizes
# spanning what an upward-pass tree level's node count might realistically look like.
using Metal, KernelAbstractions, FastMultipole, Random
using FastMultipole: FlatCoefficientBuffer, M2MOperatorScratch, OperatorInvariantCache,
                     MaterializedYRotationM2M

function to_metal(buf::FlatCoefficientBuffer{TF,A,B,LH}) where {TF,A,B,LH}
    phi_dev = Metal.MtlArray(buf.phi)
    chi_dev = Metal.MtlArray(buf.chi)
    return FlatCoefficientBuffer{TF,typeof(phi_dev),B,LH}(phi_dev, chi_dev, buf.basis_info)
end

if !Metal.functional()
    println("Metal not functional; skipping")
    exit(0)
end

function bench_once(P, nbatch, LHbool; nreps=10)
    TF = Float32
    lh = Val(LHbool)
    Random.seed!(1)
    phis = rand(TF, nbatch)
    thetas = acos.(2 .* rand(TF, nbatch) .- 1)
    r = TF(0.5) + TF(0.5) * rand(TF)
    rs = fill(r, nbatch)

    cache = OperatorInvariantCache(TF, P, lh)
    scratch = M2MOperatorScratch(TF, cache.basis_info, nbatch)
    sources = FlatCoefficientBuffer(TF, cache.basis_info, nbatch)
    for j in 1:nbatch, i in 1:size(sources.phi, 1)
        sources.phi[i, j] = randn(TF)
    end
    if LHbool
        for j in 1:nbatch, i in 1:size(sources.chi, 1)
            sources.chi[i, j] = randn(TF)
        end
    end
    op = MaterializedYRotationM2M()

    # CPU (plain resident-dispatch delegate to m2m_operator_batch!)
    targets_cpu = FlatCoefficientBuffer(TF, cache.basis_info, nbatch)
    FastMultipole.ka_m2m_operator_batch!(op, targets_cpu, sources, phis, thetas, rs, cache, scratch, lh)  # warmup/JIT
    t_cpu = @elapsed for _ in 1:nreps
        FastMultipole.ka_m2m_operator_batch!(op, targets_cpu, sources, phis, thetas, rs, cache, scratch, lh)
    end
    t_cpu /= nreps

    # Metal
    sources_metal = to_metal(sources)
    targets_metal = to_metal(FlatCoefficientBuffer(TF, cache.basis_info, nbatch))
    FastMultipole.ka_m2m_operator_batch!(op, targets_metal, sources_metal, phis, thetas, rs, cache, scratch, lh)  # warmup/compile
    Metal.synchronize()
    t_metal = @elapsed begin
        for _ in 1:nreps
            FastMultipole.ka_m2m_operator_batch!(op, targets_metal, sources_metal, phis, thetas, rs, cache, scratch, lh)
        end
        Metal.synchronize()
    end
    t_metal /= nreps

    return t_cpu, t_metal
end

println("Starting CPU vs KA/Metal M2M speed comparison...")
println(rpad("P", 5), rpad("LH", 7), rpad("nbatch", 9), rpad("CPU (ms)", 14), rpad("Metal (ms)", 14), "speedup (CPU/Metal)")
for P in (4, 8)
    for LHbool in (false, true)
        for nbatch in (10, 50, 200, 1000, 5000, 20000)
            t_cpu, t_metal = bench_once(P, nbatch, LHbool; nreps = nbatch > 2000 ? 5 : 20)
            speedup = t_cpu / t_metal
            println(rpad(P, 5), rpad(LHbool, 7), rpad(nbatch, 9),
                    rpad(round(t_cpu * 1e3, digits=4), 14), rpad(round(t_metal * 1e3, digits=4), 14),
                    round(speedup, digits=3))
        end
    end
end
