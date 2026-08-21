#=
Parity tests for the explicit z-rotation operators (Matrix Operator Refactor,
task 010) in src/rotate_batched.jl. They must reproduce the production
rotate_z! / back_rotate_z! behavior (src/rotate.jl) bit-for-bit.
=#

using FastMultipole
using Random
using Test
using FastMultipole: harmonic_index, initialize_expansion, rotate_z!, back_rotate_z!,
    update_eimϕs!, z_rotation_diagonals!, apply_z_rotation!,
    update_Ts!, update_Hs_π2!, update_ζs_mag!, update_ηs_mag!, length_Ts,
    rotate_multipole_y!, back_rotate_multipole_y!, rotate_local_y!, back_rotate_local_y!,
    _rotate_multipole_y!, _rotate_local_y!,
    update_S_blocks!, build_Ts_from_S!, length_S_block, length_Ss, S_block_offset, S_index,
    rotate_multipole_y_op!, back_rotate_multipole_y_op!,
    rotate_local_y_op!, back_rotate_local_y_op!,
    multipole_y_swap_pos90!, multipole_y_swap_neg90!,
    local_y_swap_pos90!, local_y_swap_neg90!,
    FactoredRotationStageStats, apply_z_rotation_batch!, z_theta_batch_diagonals!,
    multipole_factored_source_alignment_batch!, local_factored_source_alignment_batch!,
    multipole_factored_return_alignment_batch!, local_factored_return_alignment_batch!,
    _factored_y_batch!, update_factored_y_modes!, ymode_offset, length_ymodes

const Z_ROT_ORDERS = (0, 1, 3, 6, 9)
const Z_ROT_ANGLES = (0.0, 0.25, -1.125, π/3, 2.4)

ncomplex(P) = ((P + 1) * (P + 2)) >> 1

function random_expansion(P, TF, ::Val{LH}) where LH
    w = initialize_expansion(P, TF)
    nh = ncomplex(P)
    for i in 1:nh
        w[1,1,i] = randn(TF)
        w[2,1,i] = randn(TF)
        if LH
            w[1,2,i] = randn(TF)
            w[2,2,i] = randn(TF)
        end
    end
    return w
end

#=
Fixed ±90 degree y-swap primitive tests (Matrix Operator Refactor, task 013b).
These intentionally validate only the split primitive stages, not the assembled
Z_phi -> S -> Z_theta -> S_inv alignment chain.
=#

@testset "fixed y-swap primitives (013b)" begin

    @test !(:multipole_y_swap_pos90! in names(FastMultipole))
    @test !(:multipole_y_swap_neg90! in names(FastMultipole))
    @test !(:local_y_swap_pos90! in names(FastMultipole))
    @test !(:local_y_swap_neg90! in names(FastMultipole))
    @test FastMultipole.multipole_y_swap_pos90! === multipole_y_swap_pos90!
    @test FastMultipole.multipole_y_swap_neg90! === multipole_y_swap_neg90!
    @test FastMultipole.local_y_swap_pos90! === local_y_swap_pos90!
    @test FastMultipole.local_y_swap_neg90! === local_y_swap_neg90!

    Random.seed!(20131)

    for TF in (Float64, Float32)
        atol_rot = TF == Float64 ? 1e-11 : 1f-3

        for LHbool in (false, true)
            lamb_helmholtz = Val(LHbool)

            for P in Z_ROT_ORDERS
                basis_info = OperatorBasisInfo(P, lamb_helmholtz)
                P_active = basis_info.orders.P_active
                cache = OperatorInvariantCache(TF, basis_info)

                @test eltype(cache.T_y_pos90) === TF
                @test eltype(cache.T_y_neg90) === TF
                @test length(cache.T_y_pos90) == length_Ts(P_active)
                @test length(cache.T_y_neg90) == length_Ts(P_active)

                rebuilt_pos = zeros(TF, length_Ts(P_active))
                rebuilt_neg = zeros(TF, length_Ts(P_active))
                trig = Vector{TF}(undef, 2 * max(P_active, 1))
                build_Ts_from_S!(rebuilt_pos, cache.S_pos, cache.S_neg, TF(pi / 2), P_active, trig)
                build_Ts_from_S!(rebuilt_neg, cache.S_pos, cache.S_neg, TF(-pi / 2), P_active, trig)
                @test cache.T_y_pos90 == rebuilt_pos
                @test cache.T_y_neg90 == rebuilt_neg

                source = random_expansion(P_active, TF, lamb_helmholtz)
                if LHbool
                    active_degree = harmonic_index(P_active, 0):harmonic_index(P_active, P_active)
                    @test P_active == P + 1
                    @test any(!=(zero(TF)), @view source[:, 2, active_degree])
                end

                for (θ, T_y, mp_fun!, local_fun!) in (
                    (TF(pi / 2), cache.T_y_pos90, multipole_y_swap_pos90!, local_y_swap_pos90!),
                    (TF(-pi / 2), cache.T_y_neg90, multipole_y_swap_neg90!, local_y_swap_neg90!),
                )
                    ref_mp = initialize_expansion(P_active, TF)
                    Ts_scratch = zeros(TF, length_Ts(P_active))
                    rotate_multipole_y!(ref_mp, source, Ts_scratch, cache.Hs_pi2, cache.zeta_mag, θ, P_active, lamb_helmholtz)

                    op_mp = initialize_expansion(P_active, TF)
                    mp_fun!(op_mp, source, T_y, cache.zeta_mag, P_active, lamb_helmholtz)
                    @test isapprox(op_mp, ref_mp; atol=atol_rot)

                    ref_mp_back = initialize_expansion(P_active, TF)
                    copyto!(Ts_scratch, T_y)
                    back_rotate_multipole_y!(ref_mp_back, source, Ts_scratch, cache.zeta_mag, P_active, lamb_helmholtz)
                    @test isapprox(op_mp, ref_mp_back; atol=atol_rot)

                    preload = random_expansion(P_active, TF, lamb_helmholtz)
                    op_mp_pre = initialize_expansion(P_active, TF); copyto!(op_mp_pre, preload)
                    mp_fun!(op_mp_pre, source, T_y, cache.zeta_mag, P_active, lamb_helmholtz)
                    @test op_mp_pre == op_mp

                    ref_local = initialize_expansion(P_active, TF)
                    rotate_local_y!(ref_local, source, Ts_scratch, cache.Hs_pi2, cache.eta_mag, θ, P_active, lamb_helmholtz)

                    op_local = initialize_expansion(P_active, TF)
                    local_fun!(op_local, source, T_y, cache.Hs_pi2, cache.eta_mag, P_active, lamb_helmholtz)
                    @test isapprox(op_local, ref_local; atol=atol_rot)

                    ref_local_back = initialize_expansion(P_active, TF)
                    copyto!(Ts_scratch, T_y)
                    back_rotate_local_y!(ref_local_back, source, Ts_scratch, cache.Hs_pi2, cache.eta_mag, P_active, lamb_helmholtz)
                    @test isapprox(op_local, ref_local_back; atol=atol_rot)

                    op_local_pre = initialize_expansion(P_active, TF); copyto!(op_local_pre, preload)
                    local_fun!(op_local_pre, source, T_y, cache.Hs_pi2, cache.eta_mag, P_active, lamb_helmholtz)
                    @test op_local_pre == op_local

                    if !LHbool
                        @test all(==(zero(TF)), @view op_mp[:, 2, :])
                        @test all(==(zero(TF)), @view op_mp_pre[:, 2, :])
                        @test all(==(zero(TF)), @view op_local[:, 2, :])
                        @test all(==(zero(TF)), @view op_local_pre[:, 2, :])
                    else
                        @test any(!=(zero(TF)), @view op_mp[:, 2, active_degree])
                        @test any(!=(zero(TF)), @view op_mp_pre[:, 2, active_degree])
                        @test any(!=(zero(TF)), @view op_local[:, 2, active_degree])
                        @test any(!=(zero(TF)), @view op_local_pre[:, 2, active_degree])
                    end
                end
            end
        end
    end

end

const FACTORED_ALIGNMENT_VECTORS = (
    (0.0, 0.0, 1.75),
    (0.0, 0.0, -2.25),
    (1.7, 0.8, 2.4),
    (-2.3, -0.9, 1.6),
)

function batch_from_expansions(expansions)
    proto = first(expansions)
    batch = zeros(eltype(proto), size(proto, 1), size(proto, 2), size(proto, 3), length(expansions))
    for j in eachindex(expansions)
        batch[:,:,:,j] .= expansions[j]
    end
    return batch
end

# Physical expansion: like random_expansion but with the structurally-real m=0
# coefficients carrying zero imaginary part (the factored y stage operates on the
# 2n+1 real dofs per degree, which excludes the unphysical im(m=0) slot).
function physical_expansion(P, TF, lh::Val{LH}) where LH
    w = random_expansion(P, TF, lh)
    for n in 0:P
        i = harmonic_index(n, 0)
        w[2,1,i] = zero(TF)
        LH && (w[2,2,i] = zero(TF))
    end
    return w
end

@testset "factored rotation alignment (013c)" begin

    @test !(:multipole_factored_source_alignment_batch! in names(FastMultipole))
    @test !(:local_factored_source_alignment_batch! in names(FastMultipole))
    @test !(:multipole_factored_return_alignment_batch! in names(FastMultipole))
    @test !(:local_factored_return_alignment_batch! in names(FastMultipole))
    @test FastMultipole.multipole_factored_source_alignment_batch! === multipole_factored_source_alignment_batch!
    @test FastMultipole.local_factored_source_alignment_batch! === local_factored_source_alignment_batch!
    @test FastMultipole.multipole_factored_return_alignment_batch! === multipole_factored_return_alignment_batch!
    @test FastMultipole.local_factored_return_alignment_batch! === local_factored_return_alignment_batch!

    # Anti-collapse source guard: the runtime y-stage apply (_factored_y_batch!) must
    # NOT rebuild any per-angle operator. It may only read the precomputed fixed modes
    # and a cheap e^{iνθ} diagonal. The earlier collapsed attempts hid a per-θ Σ_ν
    # contraction (build_Ts_from_S!-style) inside the apply; assert it is absent from
    # the apply body specifically (update_Ts! is legitimately used only by the one-time
    # cache builder update_factored_y_modes!, not the hot path).
    rotate_src = read(joinpath(@__DIR__, "..", "src", "rotate_batched.jl"), String)
    apply_body = split(split(rotate_src, "function _factored_y_batch!")[2], "\nend\n")[1]
    @test !occursin("build_Ts_from_S!", apply_body)
    @test !occursin("update_Ts!", apply_body)
    @test !occursin("update_factored_y_modes!", apply_body)

    Random.seed!(20132)

    # ---- structural anti-collapse: a θ-INDEPENDENT fixed matrix decomposition exists ----
    # PROTECTED ANTI-COLLAPSE INVARIANT (016b, watch item 3). This is the load-bearing
    # guard that twice caught the factored path regressing into materialized
    # `Σ_ν S·trig(νθ)` per-entry arithmetic (013c). It must NOT be weakened to a
    # string/name check (those are the separate, weaker checks above): it numerically
    # reconstructs the production y-operator from the SAME constant cached U_n/V_n modes
    # across multiple θ, which a per-θ rebuild cannot satisfy. Any future refactor
    # (e.g. the deferred "shared-plain-swap + dressing" variant noted in 013c) must keep
    # this test intact.
    #
    # The genuine factorization is Y_n(θ) = real(U_n · diag(e^{iνθ}) · V_n) with U_n, V_n
    # FIXED (cache constants). Reconstruct the production y-operator from the cached modes
    # for several θ using the SAME U_n, V_n. A per-θ Σ_ν per-entry rebuild has no such
    # stored angle-independent matrices and cannot pass this with constant U/V.
    let TF = Float64, lh = Val(false), P = 6
        cache = OperatorInvariantCache(TF, OperatorBasisInfo(P, lh))
        for n in 1:P
            d = 2n + 1
            off = ymode_offset(n)
            U = reshape(cache.y_mult_U[off+1:off+d*d], d, d)
            V = reshape(cache.y_mult_V[off+1:off+d*d], d, d)
            for θ in (0.31, -1.4, 2.7)
                # production operator on the 2n+1 real dofs at this θ
                Ts = zeros(TF, length_Ts(n)); update_Ts!(Ts, cache.Hs_pi2, TF(θ), n)
                Yprod = zeros(TF, d, d)
                src = initialize_expansion(n, TF); out = initialize_expansion(n, TF)
                for k in 1:d
                    src .= zero(TF)
                    if k == 1
                        src[1,1,harmonic_index(n,0)] = 1
                    else
                        m = k >> 1; ri = iseven(k) ? 1 : 2
                        src[ri,1,harmonic_index(n,m)] = 1
                    end
                    _rotate_multipole_y!(out, src, Ts, cache.zeta_mag, n, lh)
                    Yprod[1,k] = out[1,1,harmonic_index(n,0)]
                    for mm in 1:n
                        Yprod[2mm,k]   = out[1,1,harmonic_index(n,mm)]
                        Yprod[2mm+1,k] = out[2,1,harmonic_index(n,mm)]
                    end
                end
                # Mrec = real(U · diag(e^{iνθ}) · V) using the SAME fixed U,V for every θ
                phases = ComplexF64[cis((νidx-n-1)*θ) for νidx in 1:d]
                Mrec = zeros(Float64, d, d)
                for c in 1:d, r in 1:d
                    acc = zero(ComplexF64)
                    for νidx in 1:d
                        acc += U[r,νidx] * phases[νidx] * V[νidx,c]
                    end
                    Mrec[r,c] = real(acc)
                end
                @test isapprox(Mrec, Yprod; atol=1e-10)
            end
        end
    end

    for TF in (Float64, Float32)
        atol_rot = TF == Float64 ? 1e-10 : 3f-3

        for LHbool in (false, true)
            lamb_helmholtz = Val(LHbool)

            for P in Z_ROT_ORDERS
                basis_info = OperatorBasisInfo(P, lamb_helmholtz)
                P_active = basis_info.orders.P_active
                cache = OperatorInvariantCache(TF, basis_info)
                scratch = OperatorScratch(TF, basis_info)
                transforms = [FastMultipole.cartesian_to_spherical(TF.(Δx)) for Δx in FACTORED_ALIGNMENT_VECTORS]
                θs = TF[t[2] for t in transforms]
                ϕs = TF[t[3] for t in transforms]
                nbatch = length(θs)

                sources = [physical_expansion(P_active, TF, lamb_helmholtz) for _ in 1:nbatch]
                preloads = [random_expansion(P_active, TF, lamb_helmholtz) for _ in 1:nbatch]
                source_batch = batch_from_expansions(sources)
                preload_batch = batch_from_expansions(preloads)
                tmp_batch = similar(source_batch)
                gbuf = scratch.y_mode_buf

                ref_mp_batch = similar(source_batch)
                ref_local_batch = similar(source_batch)
                for j in 1:nbatch
                    ref_z = initialize_expansion(P_active, TF)
                    rotate_z!(ref_z, sources[j], scratch.eimphis, ϕs[j], P_active, lamb_helmholtz)
                    rotate_multipole_y!((@view ref_mp_batch[:,:,:,j]), ref_z, scratch.Ts, cache.Hs_pi2, cache.zeta_mag, θs[j], P_active, lamb_helmholtz)
                    rotate_local_y!((@view ref_local_batch[:,:,:,j]), ref_z, scratch.Ts, cache.Hs_pi2, cache.eta_mag, θs[j], P_active, lamb_helmholtz)
                end

                out_mp_batch = similar(source_batch)
                stats = FactoredRotationStageStats()
                multipole_factored_source_alignment_batch!(
                    out_mp_batch, source_batch, tmp_batch, gbuf, ϕs, θs,
                    cache.y_mult_U, cache.y_mult_V, P_active, lamb_helmholtz; stats,
                )
                @test isapprox(out_mp_batch, ref_mp_batch; atol=atol_rot)
                @test stats.z_phi_calls == 1
                @test stats.z_theta_calls == 1
                @test stats.fixed_swap_calls == 2

                out_local_batch = similar(source_batch)
                stats = FactoredRotationStageStats()
                local_factored_source_alignment_batch!(
                    out_local_batch, source_batch, tmp_batch, gbuf, ϕs, θs,
                    cache.y_loc_U, cache.y_loc_V, P_active, lamb_helmholtz; stats,
                )
                @test isapprox(out_local_batch, ref_local_batch; atol=atol_rot)
                @test stats.z_phi_calls == 1
                @test stats.z_theta_calls == 1
                @test stats.fixed_swap_calls == 2

                ref_return_batch = copy(preload_batch)
                ref_local_return_batch = copy(preload_batch)
                for j in 1:nbatch
                    ref_y_back = initialize_expansion(P_active, TF)
                    update_Ts!(scratch.Ts, cache.Hs_pi2, θs[j], P_active)
                    back_rotate_multipole_y!(ref_y_back, (@view out_mp_batch[:,:,:,j]), scratch.Ts, cache.zeta_mag, P_active, lamb_helmholtz)
                    z_rotation_diagonals!(scratch.z_cos, scratch.z_sin, ϕs[j], P_active)
                    apply_z_rotation!((@view ref_return_batch[:,:,:,j]), ref_y_back, scratch.z_cos, scratch.z_sin, P_active, lamb_helmholtz, Val(:accumulate))

                    ref_local_y_back = initialize_expansion(P_active, TF)
                    update_Ts!(scratch.Ts, cache.Hs_pi2, θs[j], P_active)
                    back_rotate_local_y!(ref_local_y_back, (@view out_local_batch[:,:,:,j]), scratch.Ts, cache.Hs_pi2, cache.eta_mag, P_active, lamb_helmholtz)
                    z_rotation_diagonals!(scratch.z_cos, scratch.z_sin, ϕs[j], P_active)
                    apply_z_rotation!((@view ref_local_return_batch[:,:,:,j]), ref_local_y_back, scratch.z_cos, scratch.z_sin, P_active, lamb_helmholtz, Val(:accumulate))
                end

                op_return_batch = copy(preload_batch)
                stats = FactoredRotationStageStats()
                multipole_factored_return_alignment_batch!(
                    op_return_batch, out_mp_batch, tmp_batch, gbuf, ϕs, θs,
                    cache.y_mult_U, cache.y_mult_V, P_active, lamb_helmholtz; stats,
                )
                @test isapprox(op_return_batch, ref_return_batch; atol=atol_rot)
                @test stats.z_phi_calls == 1
                @test stats.z_theta_calls == 1
                @test stats.fixed_swap_calls == 2

                op_local_return_batch = copy(preload_batch)
                stats = FactoredRotationStageStats()
                local_factored_return_alignment_batch!(
                    op_local_return_batch, out_local_batch, tmp_batch, gbuf, ϕs, θs,
                    cache.y_loc_U, cache.y_loc_V, P_active, lamb_helmholtz; stats,
                )
                @test isapprox(op_local_return_batch, ref_local_return_batch; atol=atol_rot)
                @test stats.z_phi_calls == 1
                @test stats.z_theta_calls == 1
                @test stats.fixed_swap_calls == 2

                if !LHbool
                    @test all(==(zero(TF)), @view out_mp_batch[:, 2, :, :])
                    @test all(==(zero(TF)), @view out_local_batch[:, 2, :, :])
                    @test op_return_batch[:, 2, :, :] == preload_batch[:, 2, :, :]
                    @test op_local_return_batch[:, 2, :, :] == preload_batch[:, 2, :, :]
                end
            end
        end
    end

end

alloc_checked_rotate_multipole_y_op!(out, source, Ts, S_pos, S_neg, ζs_mag, θ, P, lamb_helmholtz, trig) =
    rotate_multipole_y_op!(out, source, Ts, S_pos, S_neg, ζs_mag, θ, P, lamb_helmholtz, trig)

alloc_checked_back_rotate_multipole_y_op!(out, source, Ts, S_pos, S_neg, ζs_mag, θ, P, lamb_helmholtz, trig) =
    back_rotate_multipole_y_op!(out, source, Ts, S_pos, S_neg, ζs_mag, θ, P, lamb_helmholtz, trig)

alloc_checked_rotate_local_y_op!(out, source, Ts, Hs_π2, S_pos, S_neg, ηs_mag, θ, P, lamb_helmholtz, trig) =
    rotate_local_y_op!(out, source, Ts, Hs_π2, S_pos, S_neg, ηs_mag, θ, P, lamb_helmholtz, trig)

alloc_checked_back_rotate_local_y_op!(out, source, Ts, Hs_π2, S_pos, S_neg, ηs_mag, θ, P, lamb_helmholtz, trig) =
    back_rotate_local_y_op!(out, source, Ts, Hs_π2, S_pos, S_neg, ηs_mag, θ, P, lamb_helmholtz, trig)

allocated_rotate_multipole_y_op!(out, source, Ts, S_pos, S_neg, ζs_mag, θ, P, lamb_helmholtz, trig) =
    @allocated rotate_multipole_y_op!(out, source, Ts, S_pos, S_neg, ζs_mag, θ, P, lamb_helmholtz, trig)

allocated_back_rotate_multipole_y_op!(out, source, Ts, S_pos, S_neg, ζs_mag, θ, P, lamb_helmholtz, trig) =
    @allocated back_rotate_multipole_y_op!(out, source, Ts, S_pos, S_neg, ζs_mag, θ, P, lamb_helmholtz, trig)

allocated_rotate_local_y_op!(out, source, Ts, Hs_π2, S_pos, S_neg, ηs_mag, θ, P, lamb_helmholtz, trig) =
    @allocated rotate_local_y_op!(out, source, Ts, Hs_π2, S_pos, S_neg, ηs_mag, θ, P, lamb_helmholtz, trig)

allocated_back_rotate_local_y_op!(out, source, Ts, Hs_π2, S_pos, S_neg, ηs_mag, θ, P, lamb_helmholtz, trig) =
    @allocated back_rotate_local_y_op!(out, source, Ts, Hs_π2, S_pos, S_neg, ηs_mag, θ, P, lamb_helmholtz, trig)

@testset "z-rotation operators (batched)" begin

    @test !(:z_rotation_diagonals! in names(FastMultipole))
    @test !(:apply_z_rotation! in names(FastMultipole))
    @test FastMultipole.z_rotation_diagonals! === z_rotation_diagonals!
    @test FastMultipole.apply_z_rotation! === apply_z_rotation!

    Random.seed!(2010)

    for TF in (Float64, Float32)
        atol_fwd = TF == Float64 ? 1e-13 : 1f-5
        atol_rt  = TF == Float64 ? 1e-11 : 1f-4

        for LHbool in (false, true)
            lamb_helmholtz = Val(LHbool)

            for P in Z_ROT_ORDERS, ϕ in Z_ROT_ANGLES
                ϕt = TF(ϕ)
                nh = ncomplex(P)

                source = random_expansion(P, TF, lamb_helmholtz)

                C = zeros(TF, nh)
                S = zeros(TF, nh)
                z_rotation_diagonals!(C, S, ϕt, P)

                #--- diagonals correctness: C[i]=cos(mϕ), S[i]=sin(mϕ) ---#
                i = 1
                for n in 0:P, m in 0:n
                    @test isapprox(C[i], cos(m*ϕt); atol=atol_fwd)
                    @test isapprox(S[i], sin(m*ϕt); atol=atol_fwd)
                    @test i == harmonic_index(n, m)
                    i += 1
                end

                #--- forward parity vs rotate_z! (overwrite) ---#
                ref = initialize_expansion(P, TF)
                eimϕs = zeros(TF, 2, P + 1)
                rotate_z!(ref, source, eimϕs, ϕt, P, lamb_helmholtz)

                op = initialize_expansion(P, TF)
                apply_z_rotation!(op, source, C, S, P, lamb_helmholtz, Val(:overwrite))

                for i in 1:nh
                    @test isapprox(op[1,1,i], ref[1,1,i]; atol=atol_fwd)
                    @test isapprox(op[2,1,i], ref[2,1,i]; atol=atol_fwd)
                    if LHbool
                        @test isapprox(op[1,2,i], ref[1,2,i]; atol=atol_fwd)
                        @test isapprox(op[2,2,i], ref[2,2,i]; atol=atol_fwd)
                    end
                end

                #--- inverse parity vs back_rotate_z! (accumulate onto preload) ---#
                preload = random_expansion(P, TF, lamb_helmholtz)

                ref_back = initialize_expansion(P, TF)
                copyto!(ref_back, preload)
                # back_rotate_z! relies on eimϕs computed by the forward rotate_z!
                back_rotate_z!(ref_back, ref, eimϕs, P, lamb_helmholtz)

                op_back = initialize_expansion(P, TF)
                copyto!(op_back, preload)
                apply_z_rotation!(op_back, op, C, S, P, lamb_helmholtz, Val(:accumulate))

                for i in 1:nh
                    @test isapprox(op_back[1,1,i], ref_back[1,1,i]; atol=atol_fwd)
                    @test isapprox(op_back[2,1,i], ref_back[2,1,i]; atol=atol_fwd)
                    if LHbool
                        @test isapprox(op_back[1,2,i], ref_back[1,2,i]; atol=atol_fwd)
                        @test isapprox(op_back[2,2,i], ref_back[2,2,i]; atol=atol_fwd)
                    end
                end

                #--- round-trip identity: forward then inverse onto zeroed dest ---#
                rt = initialize_expansion(P, TF)  # zeroed
                apply_z_rotation!(rt, op, C, S, P, lamb_helmholtz, Val(:accumulate))
                for i in 1:nh
                    @test isapprox(rt[1,1,i], source[1,1,i]; atol=atol_rt)
                    @test isapprox(rt[2,1,i], source[2,1,i]; atol=atol_rt)
                    if LHbool
                        @test isapprox(rt[1,2,i], source[1,2,i]; atol=atol_rt)
                        @test isapprox(rt[2,2,i], source[2,2,i]; atol=atol_rt)
                    end
                end

                #--- m = 0 exact identity (forward) and exact pass-through (accumulate) ---#
                for n in 0:P
                    i0 = harmonic_index(n, 0)
                    @test op[1,1,i0] == source[1,1,i0]
                    @test op[2,1,i0] == source[2,1,i0]
                    if LHbool
                        @test op[1,2,i0] == source[1,2,i0]
                        @test op[2,2,i0] == source[2,2,i0]
                    end
                    # accumulate of forward result onto preload adds source back unchanged
                    @test op_back[1,1,i0] == preload[1,1,i0] + source[1,1,i0]
                    @test op_back[2,1,i0] == preload[2,1,i0] + source[2,1,i0]
                end
            end
        end
    end

end

#=
Parity tests for the invariant axis-swap / y-rotation operators (Matrix Operator
Refactor, task 013) in src/rotate_batched.jl. The cached S blocks
(update_S_blocks!) plus build_Ts_from_S! must reconstruct the production Wigner Ts
(update_Ts!) and, fed through the reused production apply kernels, reproduce
rotate_multipole_y! / rotate_local_y! and their back variants. The parity target is
current production behavior (including the extra π z-axis convention); matched to
~1e-12 (Float64) / ~1e-4 (Float32) since the cached contraction reassociates the
production arithmetic.
=#

const Y_ROT_ORDERS = (0, 1, 3, 6, 9)
const Y_ROT_ANGLES = (0.0, π, π/7, -2π/5, 1.3, -0.4)  # axis-aligned θ=0, θ=π, off-axis ±

function explicit_length_Ss(P)
    total = 0
    for n in 0:P
        total += length_S_block(n)
    end
    return total
end

@testset "axis-swap y-rotation operators (batched)" begin

    @test !(:update_S_blocks! in names(FastMultipole))
    @test !(:build_Ts_from_S! in names(FastMultipole))
    @test !(:rotate_multipole_y_op! in names(FastMultipole))
    @test !(:back_rotate_multipole_y_op! in names(FastMultipole))
    @test !(:rotate_local_y_op! in names(FastMultipole))
    @test !(:back_rotate_local_y_op! in names(FastMultipole))
    @test FastMultipole.update_S_blocks! === update_S_blocks!
    @test FastMultipole.build_Ts_from_S! === build_Ts_from_S!

    Random.seed!(2013)

    @testset "axis-swap index layout" begin
        for P in Y_ROT_ORDERS
            @test length_Ss(P) == explicit_length_Ss(P)
            @test S_block_offset(0) == 0
            for n in 0:P
                @test S_block_offset(n + 1) - S_block_offset(n) == length_S_block(n)
            end

            seen = falses(length_Ss(P))
            for n in 0:P, m in 0:n, mp in 0:m, ν in 0:n
                idx = S_index(n, mp, m, ν)
                @test 1 <= idx <= length_Ss(P)
                @test !seen[idx]
                seen[idx] = true
            end
            @test all(seen)
        end
    end

    for TF in (Float64, Float32)
        atol_ts  = TF == Float64 ? 1e-12 : 1f-4
        atol_rot = TF == Float64 ? 1e-11 : 1f-3

        # angle-independent invariants (built once per TF, sized to the max order)
        Pmax = maximum(Y_ROT_ORDERS)
        Hs_π2 = ones(TF, 1); update_Hs_π2!(Hs_π2, Pmax)
        ζs_mag = ones(TF, 1); update_ζs_mag!(ζs_mag, Pmax)
        ηs_mag = ones(TF, 1); update_ηs_mag!(ηs_mag, Pmax)
        S_pos = zeros(TF, length_Ss(Pmax))
        S_neg = zeros(TF, length_Ss(Pmax))
        update_S_blocks!(S_pos, S_neg, Hs_π2, Pmax)

        for LHbool in (false, true)
            lamb_helmholtz = Val(LHbool)

            for P in Y_ROT_ORDERS, θ in Y_ROT_ANGLES
                θt = TF(θ)
                nh = ncomplex(P)

                #--- Ts reconstruction parity: build_Ts_from_S! vs update_Ts! ---#
                Ts_ref = zeros(TF, length_Ts(P))
                update_Ts!(Ts_ref, Hs_π2, θt, P)

                Ts_op = zeros(TF, length_Ts(P))
                build_Ts_from_S!(Ts_op, S_pos, S_neg, θt, P)

                @test Ts_op[1] == one(TF)               # n=0 monopole exact
                @test isapprox(Ts_op, Ts_ref; atol=atol_ts)
                # In Float64 the reconstruction is BIT-EXACT vs update_Ts!: the only
                # reassociation is multiplication by get_scalar ∈ {0, ±1}, which is
                # exact under IEEE. (Float32 stores S at reduced precision, so it stays
                # within atol_ts.) Locks the operator's exactness against regressions.
                if TF === Float64
                    @test Ts_op == Ts_ref
                end

                #--- allocation-free scratch form is identical to convenience form ---#
                Ts_trig = zeros(TF, length_Ts(P))
                trig = Vector{TF}(undef, 2 * max(P, 1))
                build_Ts_from_S!(Ts_trig, S_pos, S_neg, θt, P, trig)
                @test Ts_trig == Ts_op
                if P > 0
                    build_Ts_from_S!(Ts_trig, S_pos, S_neg, θt, P, trig)  # warm up
                    @test (@allocated build_Ts_from_S!(Ts_trig, S_pos, S_neg, θt, P, trig)) == 0
                    @test_throws ArgumentError build_Ts_from_S!(Ts_trig, S_pos, S_neg, θt, P, trig[1:end-1])
                end

                source = random_expansion(P, TF, lamb_helmholtz)

                #--- multipole forward parity vs rotate_multipole_y! (ζ table) ---#
                ref_mp = initialize_expansion(P, TF)
                Ts_scratch = zeros(TF, length_Ts(P))
                rotate_multipole_y!(ref_mp, source, Ts_scratch, Hs_π2, ζs_mag, θt, P, lamb_helmholtz)

                op_mp = initialize_expansion(P, TF)
                rotate_multipole_y_op!(op_mp, source, Ts_op, S_pos, S_neg, ζs_mag, θt, P, lamb_helmholtz)
                @test isapprox(op_mp, ref_mp; atol=atol_rot)

                op_mp_scratch = initialize_expansion(P, TF)
                alloc_checked_rotate_multipole_y_op!(op_mp_scratch, source, Ts_trig, S_pos, S_neg, ζs_mag, θt, P, lamb_helmholtz, trig)
                @test op_mp_scratch == op_mp
                alloc_checked_rotate_multipole_y_op!(op_mp_scratch, source, Ts_trig, S_pos, S_neg, ζs_mag, θt, P, lamb_helmholtz, trig)  # warm up
                if TF === Float64
                    allocated_rotate_multipole_y_op!(op_mp_scratch, source, Ts_trig, S_pos, S_neg, ζs_mag, θt, P, lamb_helmholtz, trig)
                    @test allocated_rotate_multipole_y_op!(op_mp_scratch, source, Ts_trig, S_pos, S_neg, ζs_mag, θt, P, lamb_helmholtz, trig) == 0
                end

                #--- local forward parity vs rotate_local_y! (η table), shared Ts ---#
                ref_local = initialize_expansion(P, TF)
                rotate_local_y!(ref_local, source, Ts_scratch, Hs_π2, ηs_mag, θt, P, lamb_helmholtz)

                op_local = initialize_expansion(P, TF)
                rotate_local_y_op!(op_local, source, Ts_op, Hs_π2, S_pos, S_neg, ηs_mag, θt, P, lamb_helmholtz)
                @test isapprox(op_local, ref_local; atol=atol_rot)

                op_local_scratch = initialize_expansion(P, TF)
                alloc_checked_rotate_local_y_op!(op_local_scratch, source, Ts_trig, Hs_π2, S_pos, S_neg, ηs_mag, θt, P, lamb_helmholtz, trig)
                @test op_local_scratch == op_local
                alloc_checked_rotate_local_y_op!(op_local_scratch, source, Ts_trig, Hs_π2, S_pos, S_neg, ηs_mag, θt, P, lamb_helmholtz, trig)  # warm up
                if TF === Float64
                    allocated_rotate_local_y_op!(op_local_scratch, source, Ts_trig, Hs_π2, S_pos, S_neg, ηs_mag, θt, P, lamb_helmholtz, trig)
                    @test allocated_rotate_local_y_op!(op_local_scratch, source, Ts_trig, Hs_π2, S_pos, S_neg, ηs_mag, θt, P, lamb_helmholtz, trig) == 0
                end

                # Reuse the SAME reconstructed Ts_op for the local apply: proves Ts is
                # shared and only the sign table (ζ vs η) differs between paths.
                op_local_shared = initialize_expansion(P, TF)
                _rotate_local_y!(op_local_shared, source, Ts_op, Hs_π2, ηs_mag, P, lamb_helmholtz)
                @test isapprox(op_local_shared, ref_local; atol=atol_rot)

                #--- inactive-channel reset for Val(false): component 2 is exactly zero ---#
                if !LHbool
                    @test all(==(zero(TF)), @view op_mp[:, 2, :])
                    @test all(==(zero(TF)), @view op_local[:, 2, :])
                end

                #--- back-rotation reset semantics (resets, not accumulates) ---#
                # multipole: result must be independent of the destination preload
                preload = random_expansion(P, TF, lamb_helmholtz)
                back_pre = initialize_expansion(P, TF); copyto!(back_pre, preload)
                back_rotate_multipole_y_op!(back_pre, op_mp, Ts_op, S_pos, S_neg, ζs_mag, θt, P, lamb_helmholtz)
                back_zero = initialize_expansion(P, TF)
                back_rotate_multipole_y_op!(back_zero, op_mp, Ts_op, S_pos, S_neg, ζs_mag, θt, P, lamb_helmholtz)
                @test back_pre == back_zero  # reset: preload has no effect

                back_scratch = initialize_expansion(P, TF)
                alloc_checked_back_rotate_multipole_y_op!(back_scratch, op_mp, Ts_trig, S_pos, S_neg, ζs_mag, θt, P, lamb_helmholtz, trig)
                @test back_scratch == back_zero
                alloc_checked_back_rotate_multipole_y_op!(back_scratch, op_mp, Ts_trig, S_pos, S_neg, ζs_mag, θt, P, lamb_helmholtz, trig)  # warm up
                if TF === Float64
                    allocated_back_rotate_multipole_y_op!(back_scratch, op_mp, Ts_trig, S_pos, S_neg, ζs_mag, θt, P, lamb_helmholtz, trig)
                    @test allocated_back_rotate_multipole_y_op!(back_scratch, op_mp, Ts_trig, S_pos, S_neg, ζs_mag, θt, P, lamb_helmholtz, trig) == 0
                end

                # and parity vs production back_rotate_multipole_y!
                ref_back = initialize_expansion(P, TF)
                back_rotate_multipole_y!(ref_back, op_mp, Ts_ref, ζs_mag, P, lamb_helmholtz)
                @test isapprox(back_zero, ref_back; atol=atol_rot)

                # local back-rotation parity + reset
                lback_pre = initialize_expansion(P, TF); copyto!(lback_pre, preload)
                back_rotate_local_y_op!(lback_pre, op_local, Ts_op, Hs_π2, S_pos, S_neg, ηs_mag, θt, P, lamb_helmholtz)
                lback_zero = initialize_expansion(P, TF)
                back_rotate_local_y_op!(lback_zero, op_local, Ts_op, Hs_π2, S_pos, S_neg, ηs_mag, θt, P, lamb_helmholtz)
                @test lback_pre == lback_zero

                lback_scratch = initialize_expansion(P, TF)
                alloc_checked_back_rotate_local_y_op!(lback_scratch, op_local, Ts_trig, Hs_π2, S_pos, S_neg, ηs_mag, θt, P, lamb_helmholtz, trig)
                @test lback_scratch == lback_zero
                alloc_checked_back_rotate_local_y_op!(lback_scratch, op_local, Ts_trig, Hs_π2, S_pos, S_neg, ηs_mag, θt, P, lamb_helmholtz, trig)  # warm up
                if TF === Float64
                    allocated_back_rotate_local_y_op!(lback_scratch, op_local, Ts_trig, Hs_π2, S_pos, S_neg, ηs_mag, θt, P, lamb_helmholtz, trig)
                    @test allocated_back_rotate_local_y_op!(lback_scratch, op_local, Ts_trig, Hs_π2, S_pos, S_neg, ηs_mag, θt, P, lamb_helmholtz, trig) == 0
                end

                ref_lback = initialize_expansion(P, TF)
                back_rotate_local_y!(ref_lback, op_local, Ts_ref, Hs_π2, ηs_mag, P, lamb_helmholtz)
                @test isapprox(lback_zero, ref_lback; atol=atol_rot)
            end
        end
    end

end
