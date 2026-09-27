# The consumer near-field pass hook: `fmm!(...; nearfield_pass = f)` calls
# f(cache) once the resident bodies' standard outputs are complete, and
# `radix_nearfield(cache)` hands it the sorted data. Host only.
@testset "radix near-field pass hook" begin
    Random.seed!(11)
    n = 300
    sys = VortexParticles(rand(3, n), randn(3, n) ./ n, fill(0.02, n))
    cache = RadixFMMCache(sys; expansion_order=4, ell=2, hessian=true)
    seen = Ref{Any}(nothing)
    function pass(c)
        nf = radix_nearfield(c)
        seen[] = (; n_direct=nf.n_direct, n_bodies=nf.n_bodies, n_cells=nf.n_cells,
                    u=copy(view(nf.output, 2:4, 1:nf.n_bodies)), perm=copy(nf.host_body_perm[1:nf.n_bodies]),
                    sys=copy(nf.host_body_system_ids[1:nf.n_bodies]), idx=copy(nf.host_body_indices[1:nf.n_bodies]),
                    rows=size(nf.source_bodies, 1), device=nf.device)
        return nothing
    end
    fmm!(sys, cache; gradient=true, hessian=true, nearfield_pass=pass)
    s = seen[]
    @test s !== nothing && s.n_bodies == n && s.n_direct > 0 && s.n_cells > 0 && !s.device
    @test s.rows >= 7 && all(s.sys .== 1) && sort(s.perm) == 1:n
    # the pass saw the same induced velocity the standard delivery hands the
    # system: sorted column k is global body perm[k]
    delivered = zeros(3, n)
    for k in 1:n
        delivered[:, s.perm[k]] .= s.u[:, k]
    end
    v = hcat([sys.gradient_stretching[1:3, i] for i in 1:n]...)
    @test maximum(abs.(delivered .- v)) <= 1e-12 * maximum(abs.(v))
    # the pass needs the direct pairs, which only the self-inducing call builds
    other = VortexParticles(rand(3, 20), randn(3, 20) ./ 20, fill(0.02, 20))
    @test_throws ArgumentError fmm!(sys, other, cache; nearfield_pass=pass)
end
