function evaluate_local!(system, i_system, tree::Tree, harmonics, gradient_n_m, expansion_order, lamb_helmholtz, derivatives_switches)

    # loop over leaf branches
    for i_branch in tree.leaf_index
        evaluate_local!(system, i_system, tree, i_branch, harmonics, gradient_n_m, expansion_order, lamb_helmholtz, derivatives_switches)
    end
end

function evaluate_local!(system, i_system, tree::Tree, i_branch, harmonics, gradient_n_m, expansion_order, lamb_helmholtz, derivatives_switches)
    branch = tree.branches[i_branch]
    local_expansion = view(tree.expansions, :, :, :, i_branch)
    evaluate_local!(system, branch.bodies_index[i_system], harmonics, gradient_n_m, local_expansion, branch.center, expansion_order, lamb_helmholtz, derivatives_switches[i_system])
end

# function evaluate_local!(systems::Tuple, branch::Branch, harmonics, gradient_n_m, expansion_order, lamb_helmholtz, derivatives_switches)
#     for (system, bodies_index, derivatives_switch) in zip(systems, branch.bodies_index, derivatives_switches)
#         evaluate_local!(system, bodies_index, harmonics, gradient_n_m, branch.local_expansion, branch.center, expansion_order, lamb_helmholtz, derivatives_switch)
#     end
# end

# function evaluate_local!(systems, branch::Branch{TF,<:Any}, expansion_order, lamb_helmholtz, derivatives_switches) where TF
#     harmonics = branch.harmonics
#     gradient_n_m = initialize_gradient_n_m(expansion_order, TF)
#     for (i, system) in enumerate(systems)
#         evaluate_local!(system, branch.bodies_index[i], harmonics, gradient_n_m, branch.local_expansion, branch.center, expansion_order, lamb_helmholtz, derivatives_switches[i])
#     end
# end

# function evaluate_local!(system, branch::SingleBranch, harmonics, gradient_n_m, expansion_order, lamb_helmholtz, derivatives_switch)
#     evaluate_local!(system, branch.bodies_index, harmonics, gradient_n_m, branch.local_expansion, branch.center, expansion_order, lamb_helmholtz, derivatives_switch)
# end

function evaluate_local!(system, bodies_index, harmonics, gradient_n_m, local_expansion, expansion_center, expansion_order, lamb_helmholtz, derivatives_switch::DerivativesSwitch{PS,GS,HS,NO,NM,TS}) where {PS,GS,HS,NO,NM,TS}
    for i_body in bodies_index
        pos = get_position(system, i_body)
        result = evaluate_local(pos - expansion_center, harmonics, gradient_n_m, local_expansion, expansion_order, lamb_helmholtz, derivatives_switch)
        scalar_potential, gradient, hessian = result[1], result[2], result[3]

        PS && set_scalar_potential!(system, derivatives_switch, i_body, scalar_potential)

        GS && set_gradient!(system, derivatives_switch, i_body, gradient)

        HS && set_hessian!(system, derivatives_switch, i_body, hessian)
        TS && set_third_derivative!(system, derivatives_switch, i_body, result[4])
    end
end

function evaluate_local(Δx, harmonics, gradient_n_m, local_expansion, expansion_order, ::Val{LH}, ::DerivativesSwitch{PS,GS,HS,NO,NM,TS}) where {LH,PS,GS,HS,NO,NM,TS}
    # convert to spherical coordinates
    r, θ, ϕ = cartesian_to_spherical(Δx)

    # expansion basis is the regular solid harmonics
    regular_harmonics!(harmonics, r, θ, ϕ, expansion_order)

    #--- declare/reset variables ---#

    # scalar potential
    u = zero(eltype(local_expansion))

    # vector field
    vx, vy, vz = zero(eltype(local_expansion)), zero(eltype(local_expansion)), zero(eltype(local_expansion))
    (HS || TS) && ( gradient_n_m .= zero(eltype(local_expansion)) )

    # vector gradient
    vxx, vxy, vxz = zero(eltype(local_expansion)), zero(eltype(local_expansion)), zero(eltype(local_expansion))
    vyx, vyy, vyz = zero(eltype(local_expansion)), zero(eltype(local_expansion)), zero(eltype(local_expansion))
    vzx, vzy, vzz = zero(eltype(local_expansion)), zero(eltype(local_expansion)), zero(eltype(local_expansion))

    # index
    i_n_m = 0

    #------- n = 0, m = 0 -------#

    n = 0

    # update index
    i_n_m += 1

    # get regular harmonic
    Rnm_real, Rnm_imag = harmonics[1,1,i_n_m], harmonics[2,1,i_n_m]

    # scalar potential
    if PS # && !LH # scalar potential becomes non-sensical to preserve vector field when using Lamb-Helmholtz
        ϕ_n_m_real = local_expansion[1,1,i_n_m]
        ϕ_n_m_imag = local_expansion[2,1,i_n_m]
        u += Rnm_real * ϕ_n_m_real - Rnm_imag * ϕ_n_m_imag
    end

    # vector
    if GS || HS || TS
        #=
        vx = -Im[ϕ_{1}^{1}]
        vy = -Re[ϕ_{1}^{1}]
        vz = -ϕ_{1}^0
        =#

        # expansion coefficients
        vx_n_m_real = zero(eltype(local_expansion))
        vx_n_m_imag = zero(eltype(local_expansion))
        vy_n_m_real = zero(eltype(local_expansion))
        vy_n_m_imag = zero(eltype(local_expansion))
        vz_n_m_real = zero(eltype(local_expansion))
        vz_n_m_imag = zero(eltype(local_expansion))

        # due to ϕ
        ϕ_np1_m_real = local_expansion[1,1,i_n_m+n+1]
        ϕ_np1_m_imag = local_expansion[2,1,i_n_m+n+1]
        ϕ_np1_mp1_real = local_expansion[1,1,i_n_m+n+2]
        ϕ_np1_mp1_imag = local_expansion[2,1,i_n_m+n+2]

        vx_n_m_real -= ϕ_np1_mp1_imag
        vy_n_m_real -= ϕ_np1_mp1_real
        vz_n_m_real -= ϕ_np1_m_real
        vz_n_m_imag -= ϕ_np1_m_imag

        if GS
            # evaluate expansion
            vx += vx_n_m_real * Rnm_real
            vy += vy_n_m_real * Rnm_real
            vz += vz_n_m_real * Rnm_real - vz_n_m_imag * Rnm_imag
        end

        if HS || TS
            # store components for computing vector gradient
            gradient_n_m[1,1,i_n_m] = vx_n_m_real # x component
            gradient_n_m[1,2,i_n_m] = vy_n_m_real # y component
            gradient_n_m[1,3,i_n_m] = vz_n_m_real # z component
            gradient_n_m[2,3,i_n_m] = vz_n_m_imag # z component
        end

    end

    #------- n > 0 -------#

    for n in 1:expansion_order

        #--- m = 0 ---#

        # update index
        i_n_m += 1

        # get regular harmonic
        Rnm_real, Rnm_imag = harmonics[1,1,i_n_m], harmonics[2,1,i_n_m]

        # scalar potential
        if PS && !LH
            ϕ_n_m_real = local_expansion[1,1,i_n_m]
            ϕ_n_m_imag = local_expansion[2,1,i_n_m]
            u += Rnm_real * ϕ_n_m_real - Rnm_imag * ϕ_n_m_imag
        end

        # vector
        if GS || HS || TS
            #=
            vx = (Re[χ_1^1] - Im[ϕ_2^1]) R_n^m
            vy = -(Re[ϕ_2^1] + Im[χ_1^1]) R_n^m
            vz = -ϕ_2^0 R_n^m
            =#

            # expansion coefficients
            vx_n_m_real = zero(eltype(local_expansion))
            vx_n_m_imag = zero(eltype(local_expansion))
            vy_n_m_real = zero(eltype(local_expansion))
            vy_n_m_imag = zero(eltype(local_expansion))
            vz_n_m_real = zero(eltype(local_expansion))
            vz_n_m_imag = zero(eltype(local_expansion))

            # due to ϕ
            if n < expansion_order
                ϕ_np1_m_real = local_expansion[1,1,i_n_m+n+1]
                ϕ_np1_m_imag = local_expansion[2,1,i_n_m+n+1]
                ϕ_np1_mp1_real = local_expansion[1,1,i_n_m+n+2]
                ϕ_np1_mp1_imag = local_expansion[2,1,i_n_m+n+2]

                vx_n_m_real -= ϕ_np1_mp1_imag
                vy_n_m_real -= ϕ_np1_mp1_real
                vz_n_m_real -= ϕ_np1_m_real
                vz_n_m_imag -= ϕ_np1_m_imag
            end


            # due to χ
            if LH
                χ_n_mp1_real = local_expansion[1,2,i_n_m+1]
                χ_n_mp1_imag = local_expansion[2,2,i_n_m+1]

                vx_n_m_real += n * χ_n_mp1_real
                vy_n_m_real -= n * χ_n_mp1_imag
            end

            if GS
                # evaluate expansion
                vx += vx_n_m_real * Rnm_real
                vy += vy_n_m_real * Rnm_real
                vz += vz_n_m_real * Rnm_real - vz_n_m_imag * Rnm_imag
            end

            if HS || TS
                # store components for computing vector gradient
                gradient_n_m[1,1,i_n_m] = vx_n_m_real # x component
                gradient_n_m[1,2,i_n_m] = vy_n_m_real # y component
                gradient_n_m[1,3,i_n_m] = vz_n_m_real # z component
                gradient_n_m[2,3,i_n_m] = vz_n_m_imag # z component
            end

        end

        #--- m > 0 ---#

        for m in 1:n

            # update index
            i_n_m += 1

            # get regular harmonic
            Rnm_real, Rnm_imag = harmonics[1,1,i_n_m], harmonics[2,1,i_n_m]

            # scalar potential
            if PS && !LH
                ϕ_n_m_real = local_expansion[1,1,i_n_m]
                ϕ_n_m_imag = local_expansion[2,1,i_n_m]
                u += 2 * (Rnm_real * ϕ_n_m_real - Rnm_imag * ϕ_n_m_imag)
            end

            # vector
            if GS || HS || TS
                #=
                vx = (im * (ϕ_{n+1}^{m-1} + ϕ_{n+1}^{m+1}) + (n-m) χ_n^{m+1} - (n+m) χ_n^{m-1}) / 2
                vy = (ϕ_{n+1}^{m-1} - ϕ_{n+1}^{m+1} + im * (n-m) * χ_n^{m+1} + im * (n+m) χ_n^{m-1}) / 2
                vz = -ϕ_{n+1}^m - im * m * χ_n^m
                =#

                # expansion coefficients
                vx_n_m_real = zero(eltype(local_expansion))
                vx_n_m_imag = zero(eltype(local_expansion))
                vy_n_m_real = zero(eltype(local_expansion))
                vy_n_m_imag = zero(eltype(local_expansion))
                vz_n_m_real = zero(eltype(local_expansion))
                vz_n_m_imag = zero(eltype(local_expansion))

                if n < expansion_order
                    ϕ_np1_mm1_real = local_expansion[1,1,i_n_m+n]
                    ϕ_np1_mm1_imag = local_expansion[2,1,i_n_m+n]
                    ϕ_np1_m_real = local_expansion[1,1,i_n_m+n+1]
                    ϕ_np1_m_imag = local_expansion[2,1,i_n_m+n+1]
                    ϕ_np1_mp1_real = local_expansion[1,1,i_n_m+n+2]
                    ϕ_np1_mp1_imag = local_expansion[2,1,i_n_m+n+2]

                    vx_n_m_real -= (ϕ_np1_mm1_imag + ϕ_np1_mp1_imag) * 0.5
                    vx_n_m_imag += (ϕ_np1_mm1_real + ϕ_np1_mp1_real) * 0.5
                    vy_n_m_real += (ϕ_np1_mm1_real - ϕ_np1_mp1_real) * 0.5
                    vy_n_m_imag += (ϕ_np1_mm1_imag - ϕ_np1_mp1_imag) * 0.5
                    vz_n_m_real -= ϕ_np1_m_real
                    vz_n_m_imag -= ϕ_np1_m_imag
                end

                if LH
                    # extract expansion coefficients
                    χ_n_mm1_real = local_expansion[1,2,i_n_m-1]
                    χ_n_mm1_imag = local_expansion[2,2,i_n_m-1]

                    # form vector coefficients
                    vx_n_m_real -= (n+m) * χ_n_mm1_real * 0.5
                    vx_n_m_imag -= (n+m) * χ_n_mm1_imag * 0.5
                    vy_n_m_real -= (n+m) * χ_n_mm1_imag * 0.5
                    vy_n_m_imag += (n+m) * χ_n_mm1_real * 0.5
                    if m < n
                        χ_n_mp1_real = local_expansion[1,2,i_n_m+1]
                        χ_n_mp1_imag = local_expansion[2,2,i_n_m+1]

                        vx_n_m_real += (n-m) * χ_n_mp1_real * 0.5
                        vx_n_m_imag += (n-m) * χ_n_mp1_imag * 0.5
                        vy_n_m_real -= (n-m) * χ_n_mp1_imag * 0.5
                        vy_n_m_imag += (n-m) * χ_n_mp1_real * 0.5
                    end

                    # extract expansion coefficients
                    χ_n_m_real = local_expansion[1,2,i_n_m]
                    χ_n_m_imag = local_expansion[2,2,i_n_m]

                    # form vector coefficients
                    vz_n_m_real += m * χ_n_m_imag
                    vz_n_m_imag -= m * χ_n_m_real
                end

                # evaluate expansion
                vx += 2 * (vx_n_m_real * Rnm_real - vx_n_m_imag * Rnm_imag)
                vy += 2 * (vy_n_m_real * Rnm_real - vy_n_m_imag * Rnm_imag)
                vz += 2 * (vz_n_m_real * Rnm_real - vz_n_m_imag * Rnm_imag)

                if HS || TS
                    # store components for computing vector gradient
                    gradient_n_m[1,1,i_n_m] = vx_n_m_real # x component
                    gradient_n_m[2,1,i_n_m] = vx_n_m_imag # x component
                    gradient_n_m[1,2,i_n_m] = vy_n_m_real # y component
                    gradient_n_m[2,2,i_n_m] = vy_n_m_imag # y component
                    gradient_n_m[1,3,i_n_m] = vz_n_m_real # z component
                    gradient_n_m[2,3,i_n_m] = vz_n_m_imag # z component
                end

            end
        end
    end

    if HS || TS

        # index
        i_n_m = 0

        for n in 0:expansion_order-1

            #--- m = 0 ---#

            # update index
            i_n_m += 1

            # get regular harmonic
            Rnm_real, Rnm_imag = harmonics[1,1,i_n_m], harmonics[2,1,i_n_m]

            vg_xx_real = -gradient_n_m[2,1,i_n_m+n+2]
            vxx += vg_xx_real * Rnm_real

            vg_yx_real = -gradient_n_m[1,1,i_n_m+n+2]
            vyx += vg_yx_real * Rnm_real

            vg_zx_real = -gradient_n_m[1,1,i_n_m+n+1]
            vg_zx_imag = -gradient_n_m[2,1,i_n_m+n+1]
            vzx += vg_zx_real * Rnm_real - vg_zx_imag * Rnm_imag

            vg_xy_real = -gradient_n_m[2,2,i_n_m+n+2]
            vxy += vg_xy_real * Rnm_real

            vg_yy_real = -gradient_n_m[1,2,i_n_m+n+2]
            vyy += vg_yy_real * Rnm_real

            vg_zy_real = -gradient_n_m[1,2,i_n_m+n+1]
            vg_zy_imag = -gradient_n_m[2,2,i_n_m+n+1]
            vzy += vg_zy_real * Rnm_real - vg_zy_imag * Rnm_imag

            vg_xz_real = -gradient_n_m[2,3,i_n_m+n+2]
            vxz += vg_xz_real * Rnm_real

            vg_yz_real = -gradient_n_m[1,3,i_n_m+n+2]
            vyz += vg_yz_real * Rnm_real

            vg_zz_real = -gradient_n_m[1,3,i_n_m+n+1]
            vg_zz_imag = -gradient_n_m[2,3,i_n_m+n+1]
            vzz += vg_zz_real * Rnm_real - vg_zz_imag * Rnm_imag

            #--- m > 0 ---#

            for m in 1:n

                # update index
                i_n_m += 1

                # get regular harmonic
                Rnm_real, Rnm_imag = harmonics[1,1,i_n_m], harmonics[2,1,i_n_m]

                vg_xx_real = -(gradient_n_m[2,1,i_n_m+n] + gradient_n_m[2,1,i_n_m+n+2]) * 0.5
                vg_xx_imag = (gradient_n_m[1,1,i_n_m+n] + gradient_n_m[1,1,i_n_m+n+2]) * 0.5
                vxx += 2 * (vg_xx_real * Rnm_real - vg_xx_imag * Rnm_imag)

                vg_yx_real = (gradient_n_m[1,1,i_n_m+n] - gradient_n_m[1,1,i_n_m+n+2]) * 0.5
                vg_yx_imag = (gradient_n_m[2,1,i_n_m+n] - gradient_n_m[2,1,i_n_m+n+2]) * 0.5
                vyx += 2 * (vg_yx_real * Rnm_real - vg_yx_imag * Rnm_imag)

                vg_zx_real = -gradient_n_m[1,1,i_n_m+n+1]
                vg_zx_imag = -gradient_n_m[2,1,i_n_m+n+1]
                vzx += 2 * (vg_zx_real * Rnm_real - vg_zx_imag * Rnm_imag)

                vg_xy_real = -(gradient_n_m[2,2,i_n_m+n] + gradient_n_m[2,2,i_n_m+n+2]) * 0.5
                vg_xy_imag = (gradient_n_m[1,2,i_n_m+n] + gradient_n_m[1,2,i_n_m+n+2]) * 0.5
                vxy += 2 * (vg_xy_real * Rnm_real - vg_xy_imag * Rnm_imag)

                vg_yy_real = (gradient_n_m[1,2,i_n_m+n] - gradient_n_m[1,2,i_n_m+n+2]) * 0.5
                vg_yy_imag = (gradient_n_m[2,2,i_n_m+n] - gradient_n_m[2,2,i_n_m+n+2]) * 0.5
                vyy += 2 * (vg_yy_real * Rnm_real - vg_yy_imag * Rnm_imag)

                vg_zy_real = -gradient_n_m[1,2,i_n_m+n+1]
                vg_zy_imag = -gradient_n_m[2,2,i_n_m+n+1]
                vzy += 2 * (vg_zy_real * Rnm_real - vg_zy_imag * Rnm_imag)

                vg_xz_real = -(gradient_n_m[2,3,i_n_m+n] + gradient_n_m[2,3,i_n_m+n+2]) * 0.5
                vg_xz_imag = (gradient_n_m[1,3,i_n_m+n] + gradient_n_m[1,3,i_n_m+n+2]) * 0.5
                vxz += 2 * (vg_xz_real * Rnm_real - vg_xz_imag * Rnm_imag)

                vg_yz_real = (gradient_n_m[1,3,i_n_m+n] - gradient_n_m[1,3,i_n_m+n+2]) * 0.5
                vg_yz_imag = (gradient_n_m[2,3,i_n_m+n] - gradient_n_m[2,3,i_n_m+n+2]) * 0.5
                vyz += 2 * (vg_yz_real * Rnm_real - vg_yz_imag * Rnm_imag)

                vg_zz_real = -gradient_n_m[1,3,i_n_m+n+1]
                vg_zz_imag = -gradient_n_m[2,3,i_n_m+n+1]
                vzz += 2 * (vg_zz_real * Rnm_real - vg_zz_imag * Rnm_imag)

            end
        end
    end

    base = (u * ONE_OVER_4π, SVector{3}(vx,vy,vz) * ONE_OVER_4π,
        SMatrix{3,3,eltype(local_expansion),9}(vxx, vxy, vxz, vyx, vyy, vyz, vzx, vzy, vzz) * ONE_OVER_4π)
    if TS
        third = _third_derivative_from_gradient_coefficients!(gradient_n_m, harmonics,
            Int(expansion_order), Val(LH)) * ONE_OVER_4π
        return (base..., ThirdDerivativeTensor(third))
    end
    return base
end

function evaluate_local(Δx, harmonics, gradient_n_m,
                        local_expansion::FlatCoefficientBuffer{TF,A,RealSolidHarmonicBasis,LH},
                        expansion_order, lamb_helmholtz::Val{LH},
                        derivatives_switch::DerivativesSwitch{PS,GS,HS,NO,NM,TS}) where {TF,A,LH,PS,GS,HS,NO,NM,TS}
    return evaluate_local(Δx, harmonics, gradient_n_m, local_expansion, 1,
        expansion_order, lamb_helmholtz, derivatives_switch)
end

function evaluate_local(Δx, harmonics, gradient_n_m,
                        local_expansion::FlatCoefficientBuffer{TF,A,RealSolidHarmonicBasis,LH},
                        column::Integer, expansion_order, ::Val{LH},
                        ::DerivativesSwitch{PS,GS,HS,NO,NM,TS}) where {TF,A,LH,PS,GS,HS,NO,NM,TS}
    info = local_expansion.basis_info
    P_phi = info.orders.P_phi
    P_active = info.orders.P_active
    Int(expansion_order) == P_phi ||
        throw(ArgumentError("real-basis evaluation expects expansion_order == P_phi"))

    r, θ, ϕ = cartesian_to_spherical(Δx)
    regular_harmonics!(harmonics, r, θ, ϕ, P_active)

    ph = phi_slab(local_expansion)
    ch = chi_slab(local_expansion)
    u = zero(TF)
    vx = zero(TF); vy = zero(TF); vz = zero(TF)
    vxx = zero(TF); vxy = zero(TF); vxz = zero(TF)
    vyx = zero(TF); vyy = zero(TF); vyz = zero(TF)
    vzx = zero(TF); vzy = zero(TF); vzz = zero(TF)

    if PS && !LH
        u = _real_scalar_contract(harmonics, ph, column, P_phi)
    end

    if GS || HS || TS
        fill!(gradient_n_m, zero(eltype(gradient_n_m)))
        _real_gradient_coefficients!(gradient_n_m, ph, ch, column, P_phi, P_active, Val(LH))
        vx = _complex_scalar_contract(harmonics, view(gradient_n_m, :, 1, :), P_active)
        vy = _complex_scalar_contract(harmonics, view(gradient_n_m, :, 2, :), P_active)
        vz = _complex_scalar_contract(harmonics, view(gradient_n_m, :, 3, :), P_active)
    end

    if HS || TS
        vxx, vyx, vzx = _complex_gradient_contract(harmonics, view(gradient_n_m, :, 1, :), P_active)
        vxy, vyy, vzy = _complex_gradient_contract(harmonics, view(gradient_n_m, :, 2, :), P_active)
        vxz, vyz, vzz = _complex_gradient_contract(harmonics, view(gradient_n_m, :, 3, :), P_active)
    end

    base = (u * ONE_OVER_4π,
        SVector{3}(vx, vy, vz) * ONE_OVER_4π,
        SMatrix{3,3,TF,9}(vxx, vxy, vxz, vyx, vyy, vyz, vzx, vzy, vzz) * ONE_OVER_4π)
    if TS
        third = _third_derivative_from_gradient_coefficients!(gradient_n_m, harmonics,
            P_active, Val(LH)) * ONE_OVER_4π
        return (base..., ThirdDerivativeTensor(third))
    end
    return base
end

@inline function _real_coeff_re(slab, j, P, n, m)
    (n < 0 || n > P || m < 0 || m > n) && return zero(eltype(slab))
    return m == 0 ? slab[real_basis_index(n, 0), j] : slab[real_basis_index(n, m, Val(:cos)), j]
end

@inline function _real_coeff_im(slab, j, P, n, m)
    (n < 0 || n > P || m <= 0 || m > n) && return zero(eltype(slab))
    return slab[real_basis_index(n, m, Val(:sin)), j]
end

function _real_scalar_contract(harmonics, slab, j, P)
    acc = zero(eltype(slab))
    @inbounds for n in 0:P
        i = harmonic_index(n, 0)
        acc += harmonics[1, 1, i] * slab[real_basis_index(n, 0), j]
        for m in 1:n
            i = harmonic_index(n, m)
            a = slab[real_basis_index(n, m, Val(:cos)), j]
            b = slab[real_basis_index(n, m, Val(:sin)), j]
            acc += 2 * (harmonics[1, 1, i] * a - harmonics[2, 1, i] * b)
        end
    end
    return acc
end

function _complex_scalar_contract(harmonics, coeffs, P)
    acc = zero(eltype(coeffs))
    @inbounds for n in 0:P
        i = harmonic_index(n, 0)
        acc += harmonics[1, 1, i] * coeffs[1, i] - harmonics[2, 1, i] * coeffs[2, i]
        for m in 1:n
            i = harmonic_index(n, m)
            acc += 2 * (harmonics[1, 1, i] * coeffs[1, i] - harmonics[2, 1, i] * coeffs[2, i])
        end
    end
    return acc
end

function _real_gradient_coefficients!(g, ph, ch, j, P_phi, P_active, ::Val{LH}) where LH
    @inbounds for n in 0:P_active
        i0 = harmonic_index(n, 0)
        phi1c = _real_coeff_re(ph, j, P_phi, n + 1, 1)
        phi1s = _real_coeff_im(ph, j, P_phi, n + 1, 1)
        phi0c = _real_coeff_re(ph, j, P_phi, n + 1, 0)

        g[1, 1, i0] = -phi1s
        g[1, 2, i0] = -phi1c
        g[1, 3, i0] = -phi0c
        if LH
            g[1, 1, i0] += n * _real_coeff_re(ch, j, P_active, n, 1)
            g[1, 2, i0] -= n * _real_coeff_im(ch, j, P_active, n, 1)
        end

        for m in 1:n
            i = harmonic_index(n, m)
            amm1 = _real_coeff_re(ph, j, P_phi, n + 1, m - 1)
            bmm1 = _real_coeff_im(ph, j, P_phi, n + 1, m - 1)
            amp1 = _real_coeff_re(ph, j, P_phi, n + 1, m + 1)
            bmp1 = _real_coeff_im(ph, j, P_phi, n + 1, m + 1)
            am = _real_coeff_re(ph, j, P_phi, n + 1, m)
            bm = _real_coeff_im(ph, j, P_phi, n + 1, m)

            g[1, 1, i] = -(bmm1 + bmp1) * 0.5
            g[2, 1, i] = (amm1 + amp1) * 0.5
            g[1, 2, i] = (amm1 - amp1) * 0.5
            g[2, 2, i] = (bmm1 - bmp1) * 0.5
            g[1, 3, i] = -am
            g[2, 3, i] = -bm

            if LH
                cmm1 = _real_coeff_re(ch, j, P_active, n, m - 1)
                dmm1 = _real_coeff_im(ch, j, P_active, n, m - 1)
                cmp1 = _real_coeff_re(ch, j, P_active, n, m + 1)
                dmp1 = _real_coeff_im(ch, j, P_active, n, m + 1)
                cm = _real_coeff_re(ch, j, P_active, n, m)
                dm = _real_coeff_im(ch, j, P_active, n, m)

                g[1, 1, i] += ((n - m) * cmp1 - (n + m) * cmm1) * 0.5
                g[2, 1, i] += ((n - m) * dmp1 - (n + m) * dmm1) * 0.5
                g[1, 2, i] -= ((n - m) * dmp1 + (n + m) * dmm1) * 0.5
                g[2, 2, i] += ((n - m) * cmp1 + (n + m) * cmm1) * 0.5
                g[1, 3, i] += m * dm
                g[2, 3, i] -= m * cm
            end
        end
    end
    return g
end

@inline function _complex_coeff_re(coeffs, P, n, m)
    (n < 0 || n > P || m < 0 || m > n) && return zero(eltype(coeffs))
    return coeffs[1, harmonic_index(n, m)]
end

@inline function _complex_coeff_im(coeffs, P, n, m)
    (n < 0 || n > P || m < 0 || m > n) && return zero(eltype(coeffs))
    return coeffs[2, harmonic_index(n, m)]
end

function _complex_gradient_contract(harmonics, coeffs, P)
    gx = zero(eltype(coeffs)); gy = zero(eltype(coeffs)); gz = zero(eltype(coeffs))
    @inbounds for n in 0:(P - 1)
        i0 = harmonic_index(n, 0)
        Rr = harmonics[1, 1, i0]
        gx += -_complex_coeff_im(coeffs, P, n + 1, 1) * Rr
        gy += -_complex_coeff_re(coeffs, P, n + 1, 1) * Rr
        ar = -_complex_coeff_re(coeffs, P, n + 1, 0)
        ai = -_complex_coeff_im(coeffs, P, n + 1, 0)
        gz += ar * Rr - ai * harmonics[2, 1, i0]

        for m in 1:n
            i = harmonic_index(n, m)
            Rr = harmonics[1, 1, i]
            Ri = harmonics[2, 1, i]
            xr = -(_complex_coeff_im(coeffs, P, n + 1, m - 1) +
                   _complex_coeff_im(coeffs, P, n + 1, m + 1)) * 0.5
            xi = (_complex_coeff_re(coeffs, P, n + 1, m - 1) +
                  _complex_coeff_re(coeffs, P, n + 1, m + 1)) * 0.5
            yr = (_complex_coeff_re(coeffs, P, n + 1, m - 1) -
                  _complex_coeff_re(coeffs, P, n + 1, m + 1)) * 0.5
            yi = (_complex_coeff_im(coeffs, P, n + 1, m - 1) -
                  _complex_coeff_im(coeffs, P, n + 1, m + 1)) * 0.5
            zr = -_complex_coeff_re(coeffs, P, n + 1, m)
            zi = -_complex_coeff_im(coeffs, P, n + 1, m)
            gx += 2 * (xr * Rr - xi * Ri)
            gy += 2 * (yr * Rr - yi * Ri)
            gz += 2 * (zr * Rr - zi * Ri)
        end
    end
    return gx, gy, gz
end

# Apply the Cartesian solid-harmonic differentiation recurrence to every source
# component. Destination components are grouped as (source component, x/y/z).
function _differentiate_complex_coefficients!(dest, dest_offset, src, src_offset, ncomponents, P)
    @views fill!(dest[:, dest_offset + 1:dest_offset + 3ncomponents, :], zero(eltype(dest)))
    @inbounds for component in 1:ncomponents
        s = src_offset + component
        dx = dest_offset + 3(component - 1) + 1
        dy = dx + 1
        dz = dx + 2
        for n in 0:(P - 1)
            i0 = harmonic_index(n, 0)
            ip10 = harmonic_index(n + 1, 0)
            ip11 = harmonic_index(n + 1, 1)
            dest[1, dx, i0] = -src[2, s, ip11]
            dest[1, dy, i0] = -src[1, s, ip11]
            dest[1, dz, i0] = -src[1, s, ip10]
            dest[2, dz, i0] = -src[2, s, ip10]
            for m in 1:n
                i = harmonic_index(n, m)
                im1 = harmonic_index(n + 1, m - 1)
                ip1 = harmonic_index(n + 1, m + 1)
                im = harmonic_index(n + 1, m)
                dest[1, dx, i] = -(src[2, s, im1] + src[2, s, ip1]) * 0.5
                dest[2, dx, i] =  (src[1, s, im1] + src[1, s, ip1]) * 0.5
                dest[1, dy, i] =  (src[1, s, im1] - src[1, s, ip1]) * 0.5
                dest[2, dy, i] =  (src[2, s, im1] - src[2, s, ip1]) * 0.5
                dest[1, dz, i] = -src[1, s, im]
                dest[2, dz, i] = -src[2, s, im]
            end
        end
    end
    return dest
end

function _third_derivative_from_gradient_coefficients!(scratch, harmonics, P, ::Val{LH}) where LH
    _differentiate_complex_coefficients!(scratch, 3, scratch, 0, 3, P)
    axx, axy, axz = _complex_gradient_contract(harmonics, view(scratch, :, 4, :), P - 1)
    _, ayy, ayz = _complex_gradient_contract(harmonics, view(scratch, :, 5, :), P - 1)
    _, _, azz = _complex_gradient_contract(harmonics, view(scratch, :, 6, :), P - 1)
    if !LH
        _, byy, byz = _complex_gradient_contract(harmonics, view(scratch, :, 8, :), P - 1)
        _, cyz, czz = _complex_gradient_contract(harmonics, view(scratch, :, 12, :), P - 1)
        return SVector{18}(axx, axy, axz, ayy, ayz, azz,
            axy, ayy, ayz, byy, byz, cyz,
            axz, ayz, azz, byz, cyz, czz)
    end
    bxx, bxy, bxz = _complex_gradient_contract(harmonics, view(scratch, :, 7, :), P - 1)
    _, byy, byz = _complex_gradient_contract(harmonics, view(scratch, :, 8, :), P - 1)
    _, _, bzz = _complex_gradient_contract(harmonics, view(scratch, :, 9, :), P - 1)
    cxx, cxy, cxz = _complex_gradient_contract(harmonics, view(scratch, :, 10, :), P - 1)
    _, cyy, cyz = _complex_gradient_contract(harmonics, view(scratch, :, 11, :), P - 1)
    _, _, czz = _complex_gradient_contract(harmonics, view(scratch, :, 12, :), P - 1)
    return SVector{18}(axx, axy, axz, ayy, ayz, azz,
        bxx, bxy, bxz, byy, byz, bzz,
        cxx, cxy, cxz, cyy, cyz, czz)
end
