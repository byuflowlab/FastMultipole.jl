@testset "operator cache support types" begin
    @testset "operator orders" begin
        for P in (0, 1, 6)
            orders = OperatorOrders(P, Val(false))
            @test orders.P_phi == P
            @test orders.P_chi == P
            @test orders.P_active == P

            orders_lh = OperatorOrders(P, Val(true))
            @test orders_lh.P_phi == P
            @test orders_lh.P_chi == P + 1
            @test orders_lh.P_active == P + 1
        end

        @test_throws ArgumentError OperatorOrders(-1, Val(false))
        @test_throws ArgumentError OperatorOrders(-1, Val(true))
    end

    @testset "compressed complex basis sizing" begin
        for P in (0, 1, 6)
            info = OperatorBasisInfo(CompressedComplexBasis(), P, Val(false))
            expected_phi = 2 * ((info.orders.P_phi + 1) * (info.orders.P_phi + 2) ÷ 2)
            @test info.channel_count == 1
            @test info.basis_dof_phi == expected_phi
            @test info.basis_dof_chi == expected_phi
            @test info.basis_dof_active == expected_phi

            info_lh = OperatorBasisInfo(CompressedComplexBasis(), P, Val(true))
            expected_phi_lh = 2 * ((info_lh.orders.P_phi + 1) * (info_lh.orders.P_phi + 2) ÷ 2)
            expected_active_lh = 2 * ((info_lh.orders.P_active + 1) * (info_lh.orders.P_active + 2) ÷ 2)
            @test info_lh.channel_count == 2
            @test info_lh.basis_dof_phi == expected_phi_lh
            @test info_lh.basis_dof_chi == expected_active_lh
            @test info_lh.basis_dof_active == expected_active_lh
        end
    end

    @testset "cache and scratch construction" begin
        expected_invariants(P, TF) = begin
            Hs_pi2 = ones(TF, 1)
            zeta_mag = ones(TF, 1)
            eta_mag = ones(TF, 1)
            M_tilde = ones(TF, 1)
            L_tilde = ones(TF, 1)

            FastMultipole.update_Hs_π2!(Hs_pi2, P)
            FastMultipole.update_ζs_mag!(zeta_mag, P)
            FastMultipole.update_ηs_mag!(eta_mag, P)
            FastMultipole.update_M̃!(M_tilde, P)
            FastMultipole.update_L̃!(L_tilde, P)

            return Hs_pi2, zeta_mag, eta_mag, M_tilde, L_tilde
        end

        for TF in (Float32, Float64), lamb_helmholtz in (Val(false), Val(true))
            P = 3
            basis_info = OperatorBasisInfo(P, lamb_helmholtz)
            P_active = basis_info.orders.P_active

            cache = OperatorInvariantCache(TF, basis_info)
            @test cache.basis_info === basis_info
            @test eltype(cache.Hs_pi2) === TF
            @test eltype(cache.zeta_mag) === TF
            @test eltype(cache.eta_mag) === TF
            @test eltype(cache.M_tilde) === TF
            @test eltype(cache.L_tilde) === TF
            expected_Hs_pi2, expected_zeta_mag, expected_eta_mag, expected_M_tilde, expected_L_tilde =
                expected_invariants(P_active, TF)
            @test cache.Hs_pi2 == expected_Hs_pi2
            @test cache.zeta_mag == expected_zeta_mag
            @test cache.eta_mag == expected_eta_mag
            @test cache.M_tilde == expected_M_tilde
            @test cache.L_tilde == expected_L_tilde

            scratch = OperatorScratch(TF, basis_info)
            expected_expansion = initialize_expansion(P_active, TF)
            @test scratch.basis_info === basis_info
            @test size(scratch.weights_tmp_1) == size(expected_expansion)
            @test size(scratch.weights_tmp_2) == size(expected_expansion)
            @test size(scratch.weights_tmp_3) == size(expected_expansion)
            @test eltype(scratch.weights_tmp_1) === TF
            @test eltype(scratch.weights_tmp_2) === TF
            @test eltype(scratch.weights_tmp_3) === TF
            @test eltype(scratch.Ts) === TF
            @test eltype(scratch.eimphis) === TF
            @test length(scratch.Ts) == FastMultipole.length_Ts(P_active)
            @test size(scratch.eimphis) == (2, P_active + 1)

            threaded = ThreadedOperatorScratch(TF, basis_info)
            @test length(threaded.scratch) == Threads.nthreads()
            @test all(s -> s.basis_info === basis_info, threaded.scratch)
            @test all(s -> s.Ts !== scratch.Ts, threaded.scratch)
            @test all(s -> s.eimphis !== scratch.eimphis, threaded.scratch)
            @test all(s -> s.weights_tmp_1 !== scratch.weights_tmp_1, threaded.scratch)
            for i in 1:length(threaded.scratch), j in i+1:length(threaded.scratch)
                @test threaded.scratch[i].Ts !== threaded.scratch[j].Ts
                @test threaded.scratch[i].eimphis !== threaded.scratch[j].eimphis
                @test threaded.scratch[i].weights_tmp_1 !== threaded.scratch[j].weights_tmp_1
                @test threaded.scratch[i].weights_tmp_2 !== threaded.scratch[j].weights_tmp_2
                @test threaded.scratch[i].weights_tmp_3 !== threaded.scratch[j].weights_tmp_3
            end
        end
    end
end

@testset "operator cache construction is side-effect-free" begin
    function filled_expansion(P, TF)
        expansion = initialize_expansion(P, TF)
        for i in eachindex(expansion)
            expansion[i] = TF(sin(i) + cos(2i) / 10)
        end
        return expansion
    end

    function translation_workspace(P, TF)
        Hs_pi2 = ones(TF, 1)
        FastMultipole.update_Hs_π2!(Hs_pi2, P)
        zeta_mag = ones(TF, 1)
        FastMultipole.update_ζs_mag!(zeta_mag, P)
        eta_mag = ones(TF, 1)
        FastMultipole.update_ηs_mag!(eta_mag, P)
        M_tilde = ones(TF, 1)
        FastMultipole.update_M̃!(M_tilde, P)
        L_tilde = ones(TF, 1)
        FastMultipole.update_L̃!(L_tilde, P)

        return (
            initialize_expansion(P, TF),
            initialize_expansion(P, TF),
            initialize_expansion(P, TF),
            zeros(TF, FastMultipole.length_Ts(P)),
            zeros(TF, 2, P + 1),
            zeta_mag,
            eta_mag,
            Hs_pi2,
            M_tilde,
            L_tilde,
        )
    end

    function legacy_outputs(P, TF)
        lamb_helmholtz = Val(false)
        box = SVector{3,TF}(0, 0, 0)
        source_branch = Branch(1:1, 0, 1:0, 0, 1, SVector{3,TF}(0.1, -0.2, 0.3), zero(TF), box)
        m2m_branch = Branch(1:1, 0, 1:0, 0, 1, SVector{3,TF}(0.6, 0.1, -0.1), zero(TF), box)
        m2l_branch = Branch(1:1, 0, 1:0, 0, 1, SVector{3,TF}(-0.4, 0.5, 0.7), zero(TF), box)
        l2l_branch = Branch(1:1, 0, 1:0, 0, 1, SVector{3,TF}(0.2, 0.3, 0.9), zero(TF), box)

        source = filled_expansion(P, TF)

        m2m_target = initialize_expansion(P, TF)
        weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimphis, zeta_mag, eta_mag, Hs_pi2, M_tilde, L_tilde =
            translation_workspace(P, TF)
        FastMultipole.multipole_to_multipole!(
            m2m_target, m2m_branch, source, source_branch,
            weights_tmp_1, weights_tmp_2, Ts, eimphis, zeta_mag, Hs_pi2, P, lamb_helmholtz,
        )

        m2l_target = initialize_expansion(P, TF)
        weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimphis, zeta_mag, eta_mag, Hs_pi2, M_tilde, L_tilde =
            translation_workspace(P, TF)
        FastMultipole.multipole_to_local!(
            m2l_target, m2l_branch, source, source_branch,
            weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimphis, zeta_mag, eta_mag,
            Hs_pi2, M_tilde, L_tilde, P, lamb_helmholtz,
        )

        l2l_target = initialize_expansion(P, TF)
        weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimphis, zeta_mag, eta_mag, Hs_pi2, M_tilde, L_tilde =
            translation_workspace(P, TF)
        FastMultipole.local_to_local!(
            l2l_target, l2l_branch, source, source_branch,
            weights_tmp_1, weights_tmp_2, Ts, eimphis, eta_mag, Hs_pi2, P, lamb_helmholtz,
        )

        return m2m_target, m2l_target, l2l_target
    end

    P = 4
    TF = Float64
    before = legacy_outputs(P, TF)

    global_lengths = (
        length(FastMultipole.Hs_π2),
        length(FastMultipole.ζs_mag),
        length(FastMultipole.ηs_mag),
        length(FastMultipole.M̃),
        length(FastMultipole.L̃),
    )

    basis_info = OperatorBasisInfo(P, Val(false))
    cache = OperatorInvariantCache(TF, basis_info)
    scratch = OperatorScratch(TF, basis_info)
    threaded = ThreadedOperatorScratch(TF, basis_info)

    after = legacy_outputs(P, TF)

    @test before == after
    @test length(cache.Hs_pi2) == FastMultipole.length_Hs(P)
    @test length(scratch.Ts) == FastMultipole.length_Ts(P)
    @test !isempty(threaded.scratch)
    @test global_lengths == (
        length(FastMultipole.Hs_π2),
        length(FastMultipole.ζs_mag),
        length(FastMultipole.ηs_mag),
        length(FastMultipole.M̃),
        length(FastMultipole.L̃),
    )
end

@testset "cache and scratch drive translations identically to legacy workspace" begin
    # Run the three translation kernels with a caller-supplied workspace so the
    # legacy global-array workspace and the new cache/scratch instances can be
    # exercised through the exact same code path and compared bit-for-bit. This
    # proves the new types are correct, correctly sized, drop-in workspaces, not
    # just that they have the right shapes.
    function run_translations(P, TF, lamb_helmholtz, make_workspace)
        box = SVector{3,TF}(0, 0, 0)
        source_branch = Branch(1:1, 0, 1:0, 0, 1, SVector{3,TF}(0.1, -0.2, 0.3), zero(TF), box)
        m2m_branch = Branch(1:1, 0, 1:0, 0, 1, SVector{3,TF}(0.6, 0.1, -0.1), zero(TF), box)
        m2l_branch = Branch(1:1, 0, 1:0, 0, 1, SVector{3,TF}(-0.4, 0.5, 0.7), zero(TF), box)
        l2l_branch = Branch(1:1, 0, 1:0, 0, 1, SVector{3,TF}(0.2, 0.3, 0.9), zero(TF), box)

        source = initialize_expansion(P, TF)
        for i in eachindex(source)
            source[i] = TF(sin(i) + cos(2i) / 10)
        end

        # fresh workspace per call, matching how the legacy reference is driven
        m2m_target = initialize_expansion(P, TF)
        w1, w2, w3, Ts, eimphis, zeta_mag, eta_mag, Hs_pi2, M_tilde, L_tilde = make_workspace()
        FastMultipole.multipole_to_multipole!(
            m2m_target, m2m_branch, source, source_branch,
            w1, w2, Ts, eimphis, zeta_mag, Hs_pi2, P, lamb_helmholtz,
        )

        m2l_target = initialize_expansion(P, TF)
        w1, w2, w3, Ts, eimphis, zeta_mag, eta_mag, Hs_pi2, M_tilde, L_tilde = make_workspace()
        FastMultipole.multipole_to_local!(
            m2l_target, m2l_branch, source, source_branch,
            w1, w2, w3, Ts, eimphis, zeta_mag, eta_mag,
            Hs_pi2, M_tilde, L_tilde, P, lamb_helmholtz,
        )

        l2l_target = initialize_expansion(P, TF)
        w1, w2, w3, Ts, eimphis, zeta_mag, eta_mag, Hs_pi2, M_tilde, L_tilde = make_workspace()
        FastMultipole.local_to_local!(
            l2l_target, l2l_branch, source, source_branch,
            w1, w2, Ts, eimphis, eta_mag, Hs_pi2, P, lamb_helmholtz,
        )

        return m2m_target, m2l_target, l2l_target
    end

    # legacy workspace built directly from the module globals at order P
    function legacy_workspace(P, TF)
        Hs_pi2 = ones(TF, 1); FastMultipole.update_Hs_π2!(Hs_pi2, P)
        zeta_mag = ones(TF, 1); FastMultipole.update_ζs_mag!(zeta_mag, P)
        eta_mag = ones(TF, 1); FastMultipole.update_ηs_mag!(eta_mag, P)
        M_tilde = ones(TF, 1); FastMultipole.update_M̃!(M_tilde, P)
        L_tilde = ones(TF, 1); FastMultipole.update_L̃!(L_tilde, P)
        return (
            initialize_expansion(P, TF),
            initialize_expansion(P, TF),
            initialize_expansion(P, TF),
            zeros(TF, FastMultipole.length_Ts(P)),
            zeros(TF, 2, P + 1),
            zeta_mag, eta_mag, Hs_pi2, M_tilde, L_tilde,
        )
    end

    P = 4
    for TF in (Float32, Float64), lamb_helmholtz in (Val(false), Val(true))
        # cache/scratch are padded to P_active (= P+1 for the LH χ channel); the
        # P-order translation must touch only the leading P entries and so must
        # reproduce the legacy P-sized result exactly.
        basis_info = OperatorBasisInfo(P, lamb_helmholtz)
        cache = OperatorInvariantCache(TF, basis_info)
        cache_workspace = function ()
            scratch = OperatorScratch(TF, basis_info)
            return (
                scratch.weights_tmp_1, scratch.weights_tmp_2, scratch.weights_tmp_3,
                scratch.Ts, scratch.eimphis,
                cache.zeta_mag, cache.eta_mag, cache.Hs_pi2, cache.M_tilde, cache.L_tilde,
            )
        end

        legacy = run_translations(P, TF, lamb_helmholtz, () -> legacy_workspace(P, TF))
        cached = run_translations(P, TF, lamb_helmholtz, cache_workspace)

        @test cached == legacy
    end
end
