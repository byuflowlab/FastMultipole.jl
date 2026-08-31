# Host-side tests for the 052d cross pass (CPU-only; wired into runtests.jl,
# also runnable standalone):
#   julia --project=. test/cross_stencil_test.jl

using Test
using FastMultipole
using FastMultipole: _cross_box_gap, _cross_demoted, _cross_b2m_arms,
    cross_m2m_operators,
    _cross_dummy_branch, multipole_to_multipole!, initialize_expansion,
    update_Hs_π2!, update_ζs_mag!, Hs_π2, ζs_mag, length_Ts,
    cross_m2l_operators, cross_m2l_class_slots, cross_m2l_level_scales,
    cross_row_degrees, _CROSS_M2L_L_REF, multipole_to_local!,
    update_ηs_mag!, update_M̃!, update_L̃!, ηs_mag, M̃, L̃,
    cross_l2l_operators, local_to_local!, evaluate_local, DerivativesSwitch,
    initialize_harmonics, initialize_gradient_n_m, _resident_local_eval_flat,
    flat_basis_index
using FastMultipole.StaticArrays
using LinearAlgebra: norm
import Random

@testset "CrossStencilTables" begin
    q = 12
    ell_x = 5
    h0 = 1.0

    @testset "far/demoted masks partition the push set" begin
        for Rg in (0.0, 0.006, 0.05, 0.4)
            ct = CrossStencilTables(q, ell_x, h0, Rg)
            t = ct.tables
            K = length(t.push_offsets)
            for L in 0:ell_x, k in 1:K, phase in 1:8
                base = t.class_of[phase, k]
                far = ct.level_class_far[phase, k, L + 1]
                dem = ct.level_class_demoted[phase, k, L + 1]
                if base == 0
                    @test far == 0 && dem == 0
                else
                    # exactly one of the two masks carries the offset id
                    @test (far == Int32(k)) ⊻ (dem == Int32(k))
                end
            end
        end
    end

    @testset "guard rule: strict-< box-gap threshold" begin
        ct = CrossStencilTables(q, ell_x, h0, 0.05)
        t = ct.tables
        for L in 0:ell_x, k in 1:length(t.push_offsets)
            w_L = 2 * h0 / (1 << L)
            o = t.push_offsets[k]
            # independent gap computation: nearest points of the two boxes
            gap = sqrt(sum(a -> (max(abs(o[a]) - 1, 0) * w_L)^2, 1:3))
            @test _cross_box_gap(o, w_L) ≈ gap atol=1e-14
            phase = findfirst(p -> t.class_of[p, k] != 0, 1:8)
            phase === nothing && continue
            expect_demoted = gap < 0.05
            @test (ct.level_class_demoted[phase, k, L + 1] != 0) == expect_demoted
        end
        # boundary: gap exactly at R_guard stays in M2L (strict <)
        o = SVector(2, 0, 0)                    # gap = w_L * 1
        @test !_cross_demoted(o, 0.25, 0.25)    # gap == Rg -> NOT demoted
        @test _cross_demoted(o, 0.25, 0.25 + 1e-9)
    end

    @testset "demotion grows monotonically with depth" begin
        ct = CrossStencilTables(q, ell_x, h0, 0.05)
        census = cross_demotion_census(ct)
        for L in 0:ell_x - 1
            @test census[L + 1] <= census[L + 2]
        end
        for L in 0:ell_x - 1, k in 1:length(ct.tables.push_offsets), phase in 1:8
            if ct.level_class_demoted[phase, k, L + 1] != 0
                @test ct.level_class_demoted[phase, k, L + 2] != 0
            end
        end
    end

    @testset "degenerate guards" begin
        ct0 = CrossStencilTables(q, ell_x, h0, 0.0)
        @test all(==(0), cross_demotion_census(ct0))
        # touching/overlapping boxes have zero gap and always demote for Rg > 0
        cthuge = CrossStencilTables(q, ell_x, h0, 1e6)
        t = cthuge.tables
        for L in 0:ell_x, k in 1:length(t.push_offsets), phase in 1:8
            @test cthuge.level_class_far[phase, k, L + 1] == 0
        end
    end

    @testset "dense M2M operators match the host M2M" begin
        Random.seed!(52)
        for (P, h0t) in ((4, 0.75), (6, 1.3))
            H = ((P + 1) * (P + 2)) >> 1
            D = 2 * H
            ops = cross_m2m_operators(P, h0t, 3)
            @test size(ops) == (D, D, 8, 3)
            update_Hs_π2!(Hs_π2, P)
            update_ζs_mag!(ζs_mag, P)
            w1 = initialize_expansion(P); w2 = initialize_expansion(P)
            Ts = zeros(length_Ts(P)); eimϕs = zeros(2, P + 1)
            pb = _cross_dummy_branch(SVector(0.0, 0.0, 0.0))
            for Lc in 1:3, phase in 0:7
                wc = 2 * h0t / (1 << Lc)
                u = SVector(phase & 1, (phase >> 1) & 1, (phase >> 2) & 1)
                cb = _cross_dummy_branch(SVector{3,Float64}((u .- 0.5) .* wc))
                ce = initialize_expansion(P)
                ce[:, 1, :] .= randn(2, H)   # phi channel only
                pe = initialize_expansion(P)
                multipole_to_multipole!(pe, pb, ce, cb, w1, w2, Ts, eimϕs,
                    ζs_mag, Hs_π2, P, Val(false))
                vec = [ce[2 - (c & 1), 1, (c + 1) >> 1] for c in 1:D]
                got = ops[:, :, phase + 1, Lc] * vec
                want = [pe[2 - (c & 1), 1, (c + 1) >> 1] for c in 1:D]
                @test isapprox(got, want; rtol = 1e-12, atol = 1e-14)
            end
        end
    end

    @testset "dense M2L operators match the host M2L (ref level + scaling)" begin
        Random.seed!(52)
        P = 6
        h0t = 1.3
        ct = CrossStencilTables(q, ell_x, h0t, 0.05)
        H = ((P + 1) * (P + 2)) >> 1
        D = 2 * H
        ops, class_slot = cross_m2l_operators(P, h0t, ct)
        cs2, n_slots = cross_m2l_class_slots(ct)
        @test cs2 == class_slot
        @test size(ops) == (D, D, n_slots)

        # every far-admitted (phase, k, L) class has a slot
        for L in 2:ell_x, k in 1:length(ct.tables.push_offsets), phase in 1:8
            if ct.level_class_far[phase, k, L + 1] != 0
                @test class_slot[k] != 0
            end
        end

        # degree map and level-scale tables
        row_n = cross_row_degrees(P)
        @test length(row_n) == D
        @test row_n[1] == 0 && row_n[2] == 0 && row_n[end] == P
        scale2, pow2lvl = cross_m2l_level_scales(P, ell_x)
        for L in 0:ell_x, r in 1:D
            @test scale2[r, L + 1] == 2.0^((L - _CROSS_M2L_L_REF) * row_n[r])
        end

        # sampled classes: reference-level matvec + exact level rescaling vs the
        # production host M2L, at every route level 2..ell_x
        update_Hs_π2!(Hs_π2, P); update_ζs_mag!(ζs_mag, P)
        update_ηs_mag!(ηs_mag, P); update_M̃!(M̃, P); update_L̃!(L̃, P)
        w1 = initialize_expansion(P); w2 = initialize_expansion(P)
        w3 = initialize_expansion(P)
        Ts = zeros(length_Ts(P)); eimϕs = zeros(2, P + 1)
        sb = _cross_dummy_branch(SVector(0.0, 0.0, 0.0))
        active = findall(!=(Int32(0)), class_slot)
        sample = active[Random.shuffle(1:length(active))[1:12]]
        for k in sample, L in 2:ell_x
            o = ct.tables.push_offsets[k]
            w_L = 2 * h0t / (1 << L)
            se = initialize_expansion(P)
            se[:, 1, :] .= randn(2, H)   # phi channel only
            te = initialize_expansion(P)
            tb = _cross_dummy_branch(SVector{3,Float64}(o) * w_L)
            multipole_to_local!(te, tb, se, sb, w1, w2, w3, Ts, eimϕs,
                ζs_mag, ηs_mag, Hs_π2, M̃, L̃, P, Val(false), nothing)
            want = [te[2 - (c & 1), 1, (c + 1) >> 1] for c in 1:D]
            vec = [se[2 - (c & 1), 1, (c + 1) >> 1] for c in 1:D]
            slot = class_slot[k]
            got = (ops[:, :, slot] * (vec .* scale2[:, L + 1])) .*
                scale2[:, L + 1] .* pow2lvl[L + 1]
            @test isapprox(got, want; rtol = 1e-11, atol = 1e-13 * norm(want))
        end
    end

    @testset "dense L2L operators match the host L2L" begin
        Random.seed!(52)
        for (P, h0t) in ((4, 0.75), (6, 1.3))
            H = ((P + 1) * (P + 2)) >> 1
            D = 2 * H
            ops = cross_l2l_operators(P, h0t, 3)
            @test size(ops) == (D, D, 8, 3)
            update_Hs_π2!(Hs_π2, P)
            update_ηs_mag!(ηs_mag, P)
            w1 = initialize_expansion(P); w2 = initialize_expansion(P)
            Ts = zeros(length_Ts(P)); eimϕs = zeros(2, P + 1)
            pb = _cross_dummy_branch(SVector(0.0, 0.0, 0.0))
            for Lc in 1:3, phase in 0:7
                wc = 2 * h0t / (1 << Lc)
                u = SVector(phase & 1, (phase >> 1) & 1, (phase >> 2) & 1)
                cb = _cross_dummy_branch(SVector{3,Float64}((u .- 0.5) .* wc))
                se = initialize_expansion(P)
                se[:, 1, :] .= randn(2, H)   # phi channel only
                te = initialize_expansion(P)
                local_to_local!(te, cb, se, pb, w1, w2, Ts, eimϕs,
                    ηs_mag, Hs_π2, P, Val(false))
                vec = [se[2 - (c & 1), 1, (c + 1) >> 1] for c in 1:D]
                got = ops[:, :, phase + 1, Lc] * vec
                want = [te[2 - (c & 1), 1, (c + 1) >> 1] for c in 1:D]
                @test isapprox(got, want; rtol = 1e-12, atol = 1e-14)
            end
        end
    end

    @testset "flat local eval ≡ classic evaluate_local (L2B reuse)" begin
        # the device L2B reuses _resident_local_eval_flat on classic-phi locals;
        # its flat layout is flat_basis_index = the classic row basis — prove
        # value equivalence (potential + gradient, LH off) on random locals
        Random.seed!(52)
        P = 6
        H = ((P + 1) * (P + 2)) >> 1
        D = 2 * H
        @test all(flat_basis_index(n, m, reim) ==
                2 * (FastMultipole.harmonic_index(n, m) - 1) + reim
            for n in 0:P for m in 0:n for reim in 1:2)
        harmonics = initialize_harmonics(P)
        gradient_n_m = initialize_gradient_n_m(P)
        ds = DerivativesSwitch(true, true, false)
        for trial in 1:8
            le = initialize_expansion(P)
            le[:, 1, :] .= randn(2, H)
            flat = zeros(D, 1)
            for c in 1:D
                flat[c, 1] = le[2 - (c & 1), 1, (c + 1) >> 1]
            end
            Δx = SVector{3}(randn(3) * 0.3)
            u_ref, g_ref, _ = evaluate_local(Δx, harmonics, gradient_n_m, le,
                P, Val(false), ds)
            u, gx, gy, gz = _resident_local_eval_flat(flat, flat, 1,
                Δx[1], Δx[2], Δx[3], P, P, Val(false))
            @test isapprox(u, u_ref; rtol = 1e-12, atol = 1e-14)
            @test isapprox(SVector(gx, gy, gz), g_ref; rtol = 1e-12, atol = 1e-14)
        end
    end

    @testset "B2M tag arms (incl. tag-3 vortex ring, Step-4 review fix)" begin
        s1, s2 = 1.25, -0.75
        # (do_source, do_dipole, s_source, s_dipole) per the host shims:
        # 1 Source(s1); 2 Dipole(s1); 3 Dipole(s1) — closed ring == dipole
        # panel (pure-VortexRing body_to_multipole! overload); 4/5 SourceDipole
        @test _cross_b2m_arms(1, s1, s2) == (true, false, s1, s2)
        @test _cross_b2m_arms(2, s1, s2) == (false, true, s1, s1)
        @test _cross_b2m_arms(3, s1, s2) == (false, true, s1, s1)
        @test _cross_b2m_arms(4, s1, s2) == (true, true, s1, s2)
        @test _cross_b2m_arms(5, s1, s2) == (true, true, s1, s2)
    end

    @testset "demotion census changes with h0 (union-box rebuild required)" begin
        # A grown root box halves nothing for free: cell widths scale with h0,
        # so the guard predicate (physical R_guard vs w_L-scaled box gap) flips
        # offsets between the far and demoted masks. This is why an
        # out-of-box containment failure must rebuild the FULL producer state
        # (stencil tables + masks + operators), never patch x_min/h0 in place.
        # pick R_guard from the tables themselves: just above the smallest
        # leaf-level box gap in the push set at h0 = 1, so the closest offsets
        # demote there — while doubling h0 doubles every box gap, promoting
        # them all back to far
        ct0 = CrossStencilTables(q, ell_x, 1.0, 0.0)
        w_leaf = 2 * 1.0 / (1 << ell_x)
        gmin = minimum(_cross_box_gap(o, w_leaf)
            for o in ct0.tables.push_offsets)
        Rg = 1.5 * gmin
        ct_small = CrossStencilTables(q, ell_x, 1.0, Rg)
        ct_big = CrossStencilTables(q, ell_x, 2.0, Rg)
        @test cross_demotion_census(ct_small)[ell_x + 1] > 0
        @test all(==(0), cross_demotion_census(ct_big))
        @test cross_demotion_census(ct_small) != cross_demotion_census(ct_big)
        # and the far/demoted masks themselves classify differently
        @test ct_small.level_class_far != ct_big.level_class_far
        @test ct_small.level_class_demoted != ct_big.level_class_demoted
    end

    @testset "near class admits the whole shell, unmasked" begin
        ct = CrossStencilTables(q, ell_x, h0, 0.006)
        Kn = length(ct.tables.near_offsets)
        @test size(ct.near_class) == (8, Kn, ell_x + 1)
        @test all(!=(Int32(0)), ct.near_class)
        for k in 1:Kn, L in 0:ell_x, phase in 1:8
            @test ct.near_class[phase, k, L + 1] == Int32(k)
        end
    end
end
