function evaluate_multipole!(systems, target_branch::Branch, source_branch, harmonics, expansion_order, lamb_helmholtz, derivatives_switches)
    for i in eachindex(systems)
        evaluate_multipole!(systems[i], target_branch.bodies_index[i], harmonics, source_branch.multipole_expansion, source_branch.center, expansion_order, lamb_helmholtz, derivatives_switches[i])
    end
end

function evaluate_multipole!(system, target_branch, source_branch, expansion_order, lamb_helmholtz, derivatives_switch)
    harmonics = source_branch.harmonics
    evaluate_multipole!(system, target_branch, source_branch, harmonics, expansion_order, lamb_helmholtz, derivatives_switch)
end

function evaluate_multipole!(system, bodies_index, harmonics, multipole_expansion, expansion_center, expansion_order, lamb_helmholtz, derivatives_switch::DerivativesSwitch{PS,GS,HS}) where {PS,GS,HS}
    for i_body in bodies_index
        scalar_potential, gradient, gradient = evaluate_multipole(FastMultipole.get_position(system, i_body) - expansion_center, harmonics, multipole_expansion, expansion_order, lamb_helmholtz, derivatives_switch)
        
        PS && FastMultipole.set_scalar_potential!(system, i_body, scalar_potential)
        GS && FastMultipole.set_gradient!(system, i_body, gradient)
        HS && FastMultipole.set_hessian!(system, i_body, gradient)
    end
end

function evaluate_multipole(x_target, source_center, multipole_expansion, derivatives_switch, expansion_order, lamb_helmholtz=Val(false))
    Δx = x_target - source_center
    harmonics = initialize_harmonics(expansion_order+2)
    return evaluate_multipole(Δx, harmonics, multipole_expansion, expansion_order, lamb_helmholtz, derivatives_switch)
end

function check_S(r,θ,ϕ,n,m)
    if m > n
        return 0.0
    else
        return (-1)^m * im^abs(m) * r^(-n-1) * Plm(cos(θ),n,abs(m)) * exp(im*m*ϕ) * factorial(n-abs(m))
    end
end

function evaluate_multipole(Δx, harmonics, multipole_expansion, expansion_order, ::Val{LH}, ::DerivativesSwitch{PS,GS,HS}) where {LH,PS,GS,HS}
    # convert to spherical coordinates
    r, θ, ϕ = FastMultipole.cartesian_to_spherical(Δx)

    # expansion basis is the irregular solid harmonics
    FastMultipole.irregular_harmonics!(harmonics, r, θ, ϕ, expansion_order+2)

    #--- declare/reset variables ---#

    # scalar potential
    u = zero(eltype(multipole_expansion))

    # gradient
    vx, vy, vz = zero(eltype(multipole_expansion)), zero(eltype(multipole_expansion)), zero(eltype(multipole_expansion))

    # vector gradient
    vxx, vxy, vxz = zero(eltype(multipole_expansion)), zero(eltype(multipole_expansion)), zero(eltype(multipole_expansion))
    vyx, vyy, vyz = zero(eltype(multipole_expansion)), zero(eltype(multipole_expansion)), zero(eltype(multipole_expansion))
    vzx, vzy, vzz = zero(eltype(multipole_expansion)), zero(eltype(multipole_expansion)), zero(eltype(multipole_expansion))

    # index
    i_n_m = 0

    _1_m = -1.0

    for n in 0:expansion_order
        for m in 0:n

            # update index
            i_n_m += 1

            # update (-1)^m
            _1_m = -_1_m

            # update multiplier (due to ignoring -m)
            scalar = m > 0 ? 2.0 : 1.0

            # get multipole coefficients
            ϕ_n_m_real = multipole_expansion[1,1,i_n_m]
            ϕ_n_m_imag = multipole_expansion[2,1,i_n_m]

            if LH
                χ_n_m_real = multipole_expansion[1,2,i_n_m]
                χ_n_m_imag = multipole_expansion[2,2,i_n_m]
            end

            # expansion basis
            S_n_m_real, S_n_m_imag = harmonics[1,1,i_n_m], harmonics[2,1,i_n_m]

            # scalar potential
            if PS # && !LH
                u += scalar * (S_n_m_real * ϕ_n_m_real - S_n_m_imag * ϕ_n_m_imag)
            end

            # gradient
            if GS

                # due to ϕ
                if n < expansion_order
                    S_np1_mp1_real, S_np1_mp1_imag = harmonics[1,1,i_n_m+n+2], harmonics[2,1,i_n_m+n+2]
                    S_np1_m_real, S_np1_m_imag = harmonics[1,1,i_n_m+n+1], harmonics[2,1,i_n_m+n+1]
                    S_np1_mm1_real, S_np1_mm1_imag = harmonics[1,1,i_n_m+n], harmonics[2,1,i_n_m+n]
                    if m == 0; S_np1_mm1_real, S_np1_mm1_imag = -S_np1_mp1_real, S_np1_mp1_imag; end
                else
                    S_np1_mp1_real, S_np1_mp1_imag = zero(eltype(harmonics)), zero(eltype(harmonics))
                    S_np1_m_real, S_np1_m_imag = zero(eltype(harmonics)), zero(eltype(harmonics))
                    S_np1_mm1_real, S_np1_mm1_imag = zero(eltype(harmonics)), zero(eltype(harmonics))
                end

                vx += scalar * -0.5 * (ϕ_n_m_real * (S_np1_mp1_imag + S_np1_mm1_imag) + ϕ_n_m_imag * (S_np1_mp1_real + S_np1_mm1_real))
                vy += scalar * 0.5 * (ϕ_n_m_real * (S_np1_mp1_real - S_np1_mm1_real) - ϕ_n_m_imag * (S_np1_mp1_imag - S_np1_mm1_imag))
                vz += scalar * (-ϕ_n_m_real * S_np1_m_real + ϕ_n_m_imag * S_np1_m_imag)

                # due to χ
                if LH && n>0

                    S_n_mp1_real, S_n_mp1_imag = m < n ? (harmonics[1,1,i_n_m+1], harmonics[2,1,i_n_m+1]) : (zero(eltype(harmonics)), zero(eltype(harmonics)))
                    S_n_mm1_real, S_n_mm1_imag = harmonics[1,1,i_n_m-1], harmonics[2,1,i_n_m-1]
                    if m == 0; S_n_mm1_real, S_n_mm1_imag = -S_n_mp1_real, S_n_mp1_imag; end

                    vx += scalar * ( χ_n_m_real * 0.5 * (-(n+m)*S_n_mm1_real + (n-m)*S_n_mp1_real) -
                                    χ_n_m_imag * 0.5 * (-(n+m)*S_n_mm1_imag + (n-m)*S_n_mp1_imag) )
                    vy += scalar * ( χ_n_m_real * 0.5 * ((n+m)*S_n_mm1_imag + (n-m)*S_n_mp1_imag) +
                                    χ_n_m_imag * 0.5 * ((n+m)*S_n_mm1_real + (n-m)*S_n_mp1_real) )
                    vz += scalar * m * (χ_n_m_real * S_n_m_imag + χ_n_m_imag * S_n_m_real)

                end

            end

            # vector gradient
            if HS

                # due to ϕ
                S_np2_mp2_real, S_np2_mp2_imag = harmonics[1,1,i_n_m+n+n+5], harmonics[2,1,i_n_m+n+n+5]
                S_np2_mp1_real, S_np2_mp1_imag = harmonics[1,1,i_n_m+n+n+4], harmonics[2,1,i_n_m+n+n+4]
                S_np2_m_real, S_np2_m_imag = harmonics[1,1,i_n_m+n+n+3], harmonics[2,1,i_n_m+n+n+3]
                S_np2_mm1_real, S_np2_mm1_imag = harmonics[1,1,i_n_m+n+n+2], harmonics[2,1,i_n_m+n+n+2]
                S_np2_mm2_real, S_np2_mm2_imag = harmonics[1,1,i_n_m+n+n+1], harmonics[2,1,i_n_m+n+n+1]
                if m == 0
                    S_np2_mm1_real, S_np2_mm1_imag = -S_np2_mp1_real, S_np2_mp1_imag
                    S_np2_mm2_real, S_np2_mm2_imag = S_np2_mp2_real, -S_np2_mp2_imag
                end
                if m == 1
                    S_np2_mm2_real, S_np2_mm2_imag = -S_np2_m_real, S_np2_m_imag
                end

                vxx += scalar * 0.25 * (-ϕ_n_m_real * (S_np2_mp2_real + 2.0*S_np2_m_real + S_np2_mm2_real) +
                                        ϕ_n_m_imag * (S_np2_mp2_imag + 2.0*S_np2_m_imag + S_np2_mm2_imag) )
                this_vxy = scalar * -0.25 * (ϕ_n_m_real * (S_np2_mp2_imag - S_np2_mm2_imag) +
                                             ϕ_n_m_imag * (S_np2_mp2_real - S_np2_mm2_real) )
                vxy += this_vxy
                vyx += this_vxy
                this_vxz = scalar * 0.5 * (ϕ_n_m_real * (S_np2_mp1_imag + S_np2_mm1_imag) +
                                           ϕ_n_m_imag * (S_np2_mp1_real + S_np2_mm1_real) )
                vxz += this_vxz
                vzx += this_vxz
                vyy += scalar * 0.25 * (ϕ_n_m_real * (S_np2_mp2_real - 2.0*S_np2_m_real + S_np2_mm2_real) +
                                        -ϕ_n_m_imag * (S_np2_mp2_imag - 2.0*S_np2_m_imag + S_np2_mm2_imag) )
                this_vyz = scalar * 0.5 * (-ϕ_n_m_real * (S_np2_mp1_real - S_np2_mm1_real) +
                                           ϕ_n_m_imag * (S_np2_mp1_imag - S_np2_mm1_imag) )
                vyz += this_vyz
                vzy += this_vyz
                vzz += scalar * (ϕ_n_m_real * S_np2_m_real - ϕ_n_m_imag * S_np2_m_imag)

                # due to χ
                if LH && n>0

                    throw("vector gradient with lamb_helmholtz=true is not working")

                    S_np1_mp2_real, S_np1_mp2_imag = harmonics[1,1,i_n_m+n+3], harmonics[2,1,i_n_m+n+3]
                    S_np1_mp1_real, S_np1_mp1_imag = harmonics[1,1,i_n_m+n+2], harmonics[2,1,i_n_m+n+2]
                    S_np1_m_real, S_np1_m_imag = harmonics[1,1,i_n_m+n+1], harmonics[2,1,i_n_m+n+1]
                    S_np1_mm1_real, S_np1_mm1_imag = harmonics[1,1,i_n_m+n], harmonics[2,1,i_n_m+n]
                    S_np1_mm2_real, S_np1_mm2_imag = harmonics[1,1,i_n_m+n-1], harmonics[2,1,i_n_m+n-1]
                    if m+2 > n+1
                        S_np1_mp2_real, S_np1_mp2_imag = 0.0, 0.0
                    end
                    if m == 0
                        S_np1_mm1_real, S_np1_mm1_imag = -S_np1_mp1_real, S_np1_mp1_imag
                        S_np1_mm2_real, S_np1_mm2_imag = S_np1_mp2_real, -S_np1_mp2_imag
                    end
                    if m == 1
                        S_np1_mm2_real, S_np1_mm2_imag = -S_np1_m_real, S_np1_m_imag
                    end

                    # S_np1_mp2_check = check_S(r,θ,ϕ,n+1,m+2)
                    # S_np1_mp1_check = check_S(r,θ,ϕ,n+1,m+1)
                    # S_np1_m_check = check_S(r,θ,ϕ,n+1,m)
                    # S_np1_mm1_check = check_S(r,θ,ϕ,n+1,m-1)
                    # S_np1_mm2_check = check_S(r,θ,ϕ,n+1,m-2)

                    # @assert isapprox(real(S_np1_mp2_check), S_np1_mp2_real;atol=1e-12)
                    # @assert isapprox(imag(S_np1_mp2_check), S_np1_mp2_imag;atol=1e-12)
                    # @assert isapprox(real(S_np1_mp1_check), S_np1_mp1_real;atol=1e-12)
                    # @assert isapprox(imag(S_np1_mp1_check), S_np1_mp1_imag;atol=1e-12)
                    # @assert isapprox(real(S_np1_m_check), S_np1_m_real;atol=1e-12)
                    # @assert isapprox(imag(S_np1_m_check), S_np1_m_imag;atol=1e-12)
                    # @assert isapprox(real(S_np1_mm1_check), S_np1_mm1_real;atol=1e-12)
                    # @assert isapprox(imag(S_np1_mm1_check), S_np1_mm1_imag;atol=1e-12)
                    # @assert isapprox(real(S_np1_mm2_check), S_np1_mm2_real;atol=1e-12)
                    # @assert isapprox(imag(S_np1_mm2_check), S_np1_mm2_imag;atol=1e-12)

                    vxx += scalar * 0.25 * (ϕ_n_m_real * ((n+m)*S_np1_mm2_imag + 2*m*S_np1_m_imag - (n-m)*S_np1_mp2_imag) +
                                            ϕ_n_m_imag * ((n+m)*S_np1_mm2_real + 2*m*S_np1_m_real - (n-m)*S_np1_mp2_real) )
                    vxy += scalar * 0.25 * (ϕ_n_m_real * ((n+m)*S_np1_mm2_real - 2*n*S_np1_m_real + (n-m)*S_np1_mp2_real) -
                                            ϕ_n_m_imag * ((n+m)*S_np1_mm2_imag - 2*n*S_np1_m_imag + (n-m)*S_np1_mp2_imag) )
                    vxz += scalar * 0.5 * (ϕ_n_m_real * ((n+m) * S_np1_mm1_real - (n-m) * S_np1_mp1_real) -
                                           ϕ_n_m_imag * ((n+m) * S_np1_mm1_imag - (n-m) * S_np1_mp1_imag) )
                    vyx += scalar * 0.25 * (ϕ_n_m_real * ((n+m) * S_np1_mm2_real + n * S_np1_m_real + (n-m) * S_np1_mp2_real) -
                                            ϕ_n_m_imag * ((n+m) * S_np1_mm2_imag + n * S_np1_m_imag + (n-m) * S_np1_mp2_imag) )
                    vyy += scalar * 0.25 * (ϕ_n_m_real * (-(n+m) * S_np1_mm2_imag + m * S_np1_m_imag + (n-m) * S_np1_mp2_imag) +
                                            ϕ_n_m_imag * (-(n+m) * S_np1_mm2_real + m * S_np1_m_real + (n-m) * S_np1_mp2_real) )
                    vyz += scalar * 0.5 * (ϕ_n_m_real * (-(n+m) * S_np1_mm1_imag - (n-m) * S_np1_mp1_imag) +
                                           ϕ_n_m_imag * (-(n+m) * S_np1_mm1_real - (n-m) * S_np1_mp1_real) )
                    vzx += scalar * 0.5 * (ϕ_n_m_real * m * (S_np1_mp1_real + S_np1_mm1_real) -
                                           ϕ_n_m_imag * m * (S_np1_mp1_imag + S_np1_mm1_imag) )
                    vzy += scalar * 0.5 * (ϕ_n_m_real * -m * (S_np1_mp1_imag - S_np1_mm1_imag) +
                                           ϕ_n_m_imag * -m * (S_np1_mp1_real - S_np1_mm1_real) )
                    vzz += scalar * m * (-ϕ_n_m_real * S_np1_m_imag - ϕ_n_m_imag * S_np1_m_real)

                end

            end
        end # m
    end # n

    return -u * FastMultipole.ONE_OVER_4π, SVector{3}(vx,vy,vz) * FastMultipole.ONE_OVER_4π, SMatrix{3,3,eltype(multipole_expansion),9}(vxx, vxy, vxz, vyx, vyy, vyz, vzx, vzy, vzz) * FastMultipole.ONE_OVER_4π
end

