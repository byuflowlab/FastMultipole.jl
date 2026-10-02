#=
Parity tests for the explicit z-rotation operators in src/rotate_batched.jl.
They must reproduce the production
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
    update_S_blocks!, build_Ts_from_S!, length_Ss, S_block_offset,
    rotate_multipole_y_op!, back_rotate_local_y_op!,
    update_factored_y_modes!, ymode_offset, length_ymodes

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

allocated_rotate_multipole_y_op!(out, source, Ts, S_pos, S_neg, ζs_mag, θ, P, lamb_helmholtz, trig) =
    @allocated rotate_multipole_y_op!(out, source, Ts, S_pos, S_neg, ζs_mag, θ, P, lamb_helmholtz, trig)

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
                ch = 1:(LHbool ? 2 : 1)   # active channels

                #--- diagonals correctness: C[i]=cos(mϕ), S[i]=sin(mϕ) ---#
                nm = [(n, m) for n in 0:P for m in 0:n]
                @test maximum(abs, C[1:length(nm)] .- [cos(m*ϕt) for (n, m) in nm]) <= atol_fwd
                @test maximum(abs, S[1:length(nm)] .- [sin(m*ϕt) for (n, m) in nm]) <= atol_fwd
                @test [harmonic_index(n, m) for (n, m) in nm] == 1:length(nm)

                #--- forward parity vs rotate_z! (overwrite) ---#
                ref = initialize_expansion(P, TF)
                eimϕs = zeros(TF, 2, P + 1)
                rotate_z!(ref, source, eimϕs, ϕt, P, lamb_helmholtz)

                op = initialize_expansion(P, TF)
                apply_z_rotation!(op, source, C, S, P, lamb_helmholtz, Val(:overwrite))

                @test maximum(abs, op[1:2, ch, 1:nh] .- ref[1:2, ch, 1:nh]) <= atol_fwd

                #--- inverse parity vs back_rotate_z! (accumulate onto preload) ---#
                preload = random_expansion(P, TF, lamb_helmholtz)

                ref_back = initialize_expansion(P, TF)
                copyto!(ref_back, preload)
                # back_rotate_z! relies on eimϕs computed by the forward rotate_z!
                back_rotate_z!(ref_back, ref, eimϕs, P, lamb_helmholtz)

                op_back = initialize_expansion(P, TF)
                copyto!(op_back, preload)
                apply_z_rotation!(op_back, op, C, S, P, lamb_helmholtz, Val(:accumulate))

                @test maximum(abs, op_back[1:2, ch, 1:nh] .- ref_back[1:2, ch, 1:nh]) <= atol_fwd

                #--- round-trip identity: forward then inverse onto zeroed dest ---#
                rt = initialize_expansion(P, TF)  # zeroed
                apply_z_rotation!(rt, op, C, S, P, lamb_helmholtz, Val(:accumulate))
                @test maximum(abs, rt[1:2, ch, 1:nh] .- source[1:2, ch, 1:nh]) <= atol_rt

                #--- m = 0 exact identity (forward) and exact pass-through (accumulate) ---#
                i0 = [harmonic_index(n, 0) for n in 0:P]
                @test op[1:2, ch, i0] == source[1:2, ch, i0]
                # accumulate of forward result onto preload adds source back unchanged
                @test op_back[1:2, 1, i0] == preload[1:2, 1, i0] .+ source[1:2, 1, i0]
            end
        end
    end

end

#=
Parity tests for the invariant axis-swap / y-rotation operators in
src/rotate_batched.jl. The cached S blocks
(update_S_blocks!) plus build_Ts_from_S! must reconstruct the production Wigner Ts
(update_Ts!) and, fed through the reused production apply kernels, reproduce
rotate_multipole_y! / rotate_local_y! and their back variants. The parity target is
current production behavior (including the extra π z-axis convention); matched to
~1e-12 (Float64) / ~1e-4 (Float32) since the cached contraction reassociates the
production arithmetic.
=#

const Y_ROT_ORDERS = (0, 1, 3, 6, 9)
const Y_ROT_ANGLES = (0.0, π, π/7, -2π/5, 1.3, -0.4)  # axis-aligned θ=0, θ=π, off-axis ±

@testset "axis-swap y-rotation operators (batched)" begin

    @test !(:update_S_blocks! in names(FastMultipole))
    @test !(:build_Ts_from_S! in names(FastMultipole))
    @test !(:rotate_multipole_y_op! in names(FastMultipole))
    @test !(:back_rotate_local_y_op! in names(FastMultipole))
    @test FastMultipole.update_S_blocks! === update_S_blocks!
    @test FastMultipole.build_Ts_from_S! === build_Ts_from_S!

    Random.seed!(2013)

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
                trig = Vector{TF}(undef, 2 * max(P, 1))
                build_Ts_from_S!(Ts_op, S_pos, S_neg, θt, P, trig)

                @test Ts_op[1] == one(TF)               # n=0 monopole exact
                @test isapprox(Ts_op, Ts_ref; atol=atol_ts)
                # In Float64 the reconstruction is BIT-EXACT vs update_Ts!: the only
                # reassociation is multiplication by get_scalar ∈ {0, ±1}, which is
                # exact under IEEE. (Float32 stores S at reduced precision, so it stays
                # within atol_ts.) Locks the operator's exactness against regressions.
                if TF === Float64
                    @test Ts_op == Ts_ref
                end

                #--- allocation-free with caller scratch; short scratch rejected ---#
                Ts_trig = zeros(TF, length_Ts(P))
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
                rotate_multipole_y_op!(op_mp, source, Ts_trig, S_pos, S_neg, ζs_mag, θt, P, lamb_helmholtz, trig)
                @test isapprox(op_mp, ref_mp; atol=atol_rot)
                if TF === Float64
                    allocated_rotate_multipole_y_op!(op_mp, source, Ts_trig, S_pos, S_neg, ζs_mag, θt, P, lamb_helmholtz, trig)
                    @test allocated_rotate_multipole_y_op!(op_mp, source, Ts_trig, S_pos, S_neg, ζs_mag, θt, P, lamb_helmholtz, trig) == 0
                end

                #--- the shared Ts drives the local (η table) kernel too ---#
                ref_local = initialize_expansion(P, TF)
                rotate_local_y!(ref_local, source, Ts_scratch, Hs_π2, ηs_mag, θt, P, lamb_helmholtz)
                op_local = initialize_expansion(P, TF)
                _rotate_local_y!(op_local, source, Ts_op, Hs_π2, ηs_mag, P, lamb_helmholtz)
                @test isapprox(op_local, ref_local; atol=atol_rot)

                #--- inactive-channel reset for Val(false): component 2 is exactly zero ---#
                if !LHbool
                    @test all(==(zero(TF)), @view op_mp[:, 2, :])
                end

                #--- local back-rotation parity + reset semantics (resets, not accumulates) ---#
                preload = random_expansion(P, TF, lamb_helmholtz)
                lback_pre = initialize_expansion(P, TF); copyto!(lback_pre, preload)
                back_rotate_local_y_op!(lback_pre, op_local, Ts_trig, Hs_π2, S_pos, S_neg, ηs_mag, θt, P, lamb_helmholtz, trig)
                lback_zero = initialize_expansion(P, TF)
                back_rotate_local_y_op!(lback_zero, op_local, Ts_trig, Hs_π2, S_pos, S_neg, ηs_mag, θt, P, lamb_helmholtz, trig)
                @test lback_pre == lback_zero  # reset: preload has no effect

                lback_scratch = initialize_expansion(P, TF)
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
