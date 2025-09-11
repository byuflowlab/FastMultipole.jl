function spherical_to_cartesian(ρ,θ,ϕ)
    z = ρ * cos(θ)
    x = ρ * sin(θ) * cos(ϕ)
    y = ρ * sin(θ) * sin(ϕ)
    return x, y, z
end

function build_system(center, ρs, θs, ϕs, radii, qs)
    source_bodies = zeros(8,length(ρs))
    for (i,(ρ,θ,ϕ,r,q)) in enumerate(zip(ρs, θs, ϕs, radii, qs))
        x, y, z = spherical_to_cartesian(ρ,θ,ϕ)
        source_bodies[1:3,i] .= center + SVector{3}(x,y,z)
        source_bodies[4,i] = r
        source_bodies[5,i] = q
    end
    system = Gravitational(source_bodies)
    return system
end

function build_system_range(center, ρ_range, θ_range, ϕ_range, r_range, A)
    ρs = Float64[]
    θs = Float64[]
    ϕs = Float64[]
    rs = Float64[]
    qs = Float64[]
    q_density = A / (4/3*pi*ρ_range[end]^3)
    if length(ρ_range) > 1
        for iρ in 1:length(ρ_range)-1
            ρ = (ρ_range[iρ] + ρ_range[iρ+1])/2
            dV = 4/3*pi*(ρ_range[iρ+1]^3 - ρ_range[iρ]^3)
            total_q = dV * q_density
            q = total_q / (length(θ_range) * length(ϕ_range))
            total_circ = 0.0
            for θ in θ_range
                total_circ += 2*pi*ρ*sin(θ)
            end
            dq = q / total_circ
            for θ in θ_range
                circ = 2*pi*ρ*sin(θ)
                this_q = circ * dq / length(ϕ_range)
                for ϕ in ϕ_range
                    r = r_range[1] + rand() * (r_range[2]-r_range[1])
                    #q = q_range[1] + rand() * (q_range[2]-q_range[1])
                    push!(ρs, ρ)
                    push!(θs, θ)
                    push!(ϕs, ϕ)
                    push!(rs, r)
                    push!(qs, this_q)
                end

            end
        end
    else
        ρ = ρ_range[end]
        q = A / (length(θ_range) * length(ϕ_range))
        for θ in θ_range
            for ϕ in ϕ_range
                r = r_range[1] + rand() * (r_range[2]-r_range[1])
                #q = q_range[1] + rand() * (q_range[2]-q_range[1])
                push!(ρs, ρ)
                push!(θs, θ)
                push!(ϕs, ϕ)
                push!(rs, r)
                push!(qs, q)
            end
        end
    end
    return build_system(center, ρs, θs, ϕs, rs, qs)
end

function build_system_cartesian(center, xs, ys, zs, radii, qs)
    source_bodies = zeros(8,length(xs))
    for (i,(x,y,z,r,q)) in enumerate(zip(xs, ys, zs, radii, qs))
        source_bodies[1:3,i] .= center + SVector{3}(x,y,z)
        source_bodies[4,i] = r
        source_bodies[5,i] = q
    end
    system = Gravitational(source_bodies)
    return system
end

function build_system_cartesian_range(center, x_range, y_range, z_range, r_range, q_range)
    xs = Float64[]
    ys = Float64[]
    zs = Float64[]
    rs = Float64[]
    qs = Float64[]
    for x in x_range
        for y in y_range
            for z in z_range
                push!(xs, x)
                push!(ys, y)
                push!(zs, z)
                r = r_range[1] + rand() * (r_range[2]-r_range[1])
                q = q_range[1] + rand() * (q_range[2]-q_range[1])
                push!(rs, r)
                push!(qs, q)
            end
        end
    end
    return build_system_cartesian(center, xs, ys, zs, rs, qs)
end

function generate_branch(center, ρ_range, θ_range, ϕ_range, r_range, A, expansion_order, shrink)
    # build system
    system = build_system_range(center, ρ_range, θ_range, ϕ_range, r_range, A)

    # build branch
    branch = Branch(1:get_n_bodies(system), 0, 1:0, 0, 1, center, sqrt(3.0), sqrt(3.0), SVector{6}(-1.0,1.0,-1.0,1.0,-1.0,1.0), SVector{3}(1.0,1.0,1.0), expansion_order)

    # shrink branch
    branches = [branch]
    shrink && FastMultipole.shrink_recenter!(branches, [1:1], system)
    branch = branches[1]

    return branch, system
end

function generate_branch_cartesian(center, x_range, y_range, z_range, r_range, q_range, expansion_order, shrink)
    # build system
    system = build_system_cartesian_range(center, x_range, y_range, z_range, r_range, q_range)

    # build branch
    branch = Branch(1:get_n_bodies(system), 0, 1:0, 0, 1, center, sqrt(3.0), sqrt(3.0), SVector{6}(-1.0,1.0,-1.0,1.0,-1.0,1.0), SVector{3}(1.0,1.0,1.0), expansion_order)

    # shrink branch
    branches = [branch]
    shrink && FastMultipole.shrink_recenter!(branches, [1:1], system)
    branch = branches[1]

    return branch, system
end

function evaluate_multipole(xt, branch::Branch, expansion_order, lamb_helmholtz::Bool)
    Δx = xt - branch.source_center
    ϕ_m2b, v_m2b, g_m2b = evaluate_multipole(Δx, branch.harmonics, branch.multipole_expansion, Val(expansion_order), Val(lamb_helmholtz), DerivativesSwitch(true,true,false))
    return ϕ_m2b, v_m2b
end

function evaluate_direct(xt, system::Gravitational, multipole_branch::SingleBranch)
    u = 0.0
    for i in multipole_branch.bodies_index
        body = system.bodies[i]
        xs = body.position
        q = body.strength
        u += q / norm(xt-xs)
    end
    return u / 4 / pi
end

function multipole_to_local!(local_branch, multipole_branch, expansion_order, lamb_helmholtz)
    # reset expansion
    local_branch.local_expansion .= zero(eltype(local_branch.local_expansion))

    # preallocate containers
    Hs_π2 = [1.0]
    FastMultipole.update_Hs_π2!(Hs_π2, Val(expansion_order))
    Ts = zeros(FastMultipole.length_Ts(expansion_order))
    eimϕs = zeros(2, expansion_order+2)
    weights_tmp_1 = initialize_expansion(expansion_order, eltype(Ts))
    weights_tmp_2 = initialize_expansion(expansion_order, eltype(Ts))

    # normalization
    ζs_mag = zeros(FastMultipole.length_ζs(expansion_order))
    FastMultipole.update_ζs_mag!(ζs_mag, 0, expansion_order)
    ηs_mag = zeros(FastMultipole.length_ηs(expansion_order))
    FastMultipole.update_ηs_mag!(ηs_mag, 0, expansion_order)

    # local coefficients
    FastMultipole.multipole_to_local!(local_branch, multipole_branch, weights_tmp_1, weights_tmp_2, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, expansion_order, lamb_helmholtz)

    return nothing
end

function evaluate_local(xt, branch::Branch, expansion_order, lamb_helmholtz)
    Δx = xt - branch.target_center
    velocity_n_m = zeros(2,3,size(branch.multipole_expansion,3))
    u, velocity, gradient = FastMultipole.evaluate_local(Δx, branch.harmonics, velocity_n_m, branch.local_expansion, Val(expansion_order), Val(lamb_helmholtz), DerivativesSwitch())
    return u, velocity
end

function relative_error(u_true, u_test)
    return (u_test .- u_true) ./ u_true
end

function absolute_error(u_true::Vector{Float64}, u_test::Vector{Float64})
    return u_test .- u_true
end

function reset!(system::Gravitational)
    system.potential .= 0.0
end

function reset!(system::VortexParticles)
    system.potential .= 0.0
    system.velocity_stretching .= 0.0
end

function absolute_error(direct, mp)
    delta = similar(direct)
    for i in eachindex(direct)
        delta[i] = direct[i] - mp[i]
    end
    err = [norm(v) for v in delta]
    return err
end

function multipole_local_error(local_branch::SingleBranch, local_system, multipole_branch::SingleBranch, multipole_system, expansion_order::Int; body_type=Point{Source})
    # reset bodies
    reset!(local_system)

    # get direct potential
    # potential_direct = [evaluate_direct(local_system[i,Position()], multipole_system, multipole_branch) for i in local_branch.bodies_index]
    FastMultipole._direct!(local_system, local_branch.bodies_index, DerivativesSwitch(true,true,false), multipole_system, multipole_branch.bodies_index)
    potential_direct = [local_system[i,FastMultipole.ScalarPotential()] for i in local_branch.bodies_index]
    velocity_direct = [SVector{3}(local_system[i,FastMultipole.Velocity()]) for i in local_branch.bodies_index]
    if body_type == Point{Vortex}
        reset!(local_system)
        multipole_branch.multipole_expansion .= 0.0
        FastMultipole.body_to_multipole!(multipole_branch, multipole_system, multipole_branch.harmonics, Val(20))
        potential_direct = [evaluate_multipole(local_system[i,Position()], multipole_branch, 20, true)[1] for i in local_branch.bodies_index]
    end

    # multipole expansion
    # multipole_branch.multipole_expansion .= 0.0
    # FastMultipole.body_to_multipole!(multipole_branch, multipole_system, multipole_branch.harmonics, Val(expansion_order))

    # multipole error
    lamb_helmholtz = body_type == Point{Vortex}
    potential_mp = [evaluate_multipole(local_system[i,Position()], multipole_branch, expansion_order, lamb_helmholtz)[1] for i in local_branch.bodies_index]
    velocity_mp = [evaluate_multipole(local_system[i,Position()], multipole_branch, expansion_order, lamb_helmholtz)[2] for i in local_branch.bodies_index]

    # check contributions of ϕ vs χ
    ϕ = deepcopy(multipole_branch.multipole_expansion[:,1,:])
    multipole_branch.multipole_expansion[:,1,:] .= 0.0
    velocity_mp_χ = [evaluate_multipole(local_system[i,Position()], multipole_branch, expansion_order, lamb_helmholtz)[2] for i in local_branch.bodies_index]
    multipole_branch.multipole_expansion[:,1,:] .= ϕ
    χ = deepcopy(multipole_branch.multipole_expansion[:,2,:])
    multipole_branch.multipole_expansion[:,2,:] .= 0.0
    velocity_mp_ϕ = [evaluate_multipole(local_system[i,Position()], multipole_branch, expansion_order, lamb_helmholtz)[2] for i in local_branch.bodies_index]
    multipole_branch.multipole_expansion[:,2,:] .= χ
    mags_ϕ = [norm(this_v) for this_v in velocity_mp_ϕ]
    mags_χ = [norm(this_v) for this_v in velocity_mp_χ]
    mags_mp = [norm(this_v) for this_v in velocity_mp]

    ϕ_contributions = mags_ϕ ./ mags_mp
    χ_contributions = mags_χ ./ mags_mp

    ϕmean = mean(ϕ_contributions)
    ϕstd = std(ϕ_contributions)
    χmean = mean(χ_contributions)
    χstd = std(χ_contributions)

    # @show ϕmean, ϕstd, χmean, χstd

    error_mp = absolute_error(potential_direct, potential_mp)
    error_mp_velocity = absolute_error(velocity_direct, velocity_mp)

    # local expansion
    if body_type == Point{Source} || body_type == Point{Vortex}
        local_branch.local_expansion .= 0.0
        x_l = local_branch.target_center
        for i in multipole_branch.bodies_index
            Δx = multipole_system[i,Position()] - x_l
            body_to_local_point!(body_type, local_branch.local_expansion, local_branch.harmonics, Δx, multipole_system[i, Strength()], Val(20))
        end

        # local error
        lamb_helmholtz = body_type == Point{Vortex} ? true : false
        potential_l = [evaluate_local(local_system[i,Position()], local_branch, expansion_order, lamb_helmholtz)[1] for i in local_branch.bodies_index]
        velocity_l = [evaluate_local(local_system[i,Position()], local_branch, expansion_order, lamb_helmholtz)[2] for i in local_branch.bodies_index]
        error_l = absolute_error(potential_direct, potential_l)
        error_l_velocity = absolute_error(velocity_direct, velocity_l)

        #=
        # check relative contribution
        ϕ = deepcopy(local_branch.local_expansion[:,1,:])
        χ = deepcopy(local_branch.local_expansion[:,2,:])
        local_branch.local_expansion[:,1,:] .= 0.0
        velocity_l_χ = [evaluate_local(local_system[i,Position()], local_branch, expansion_order, lamb_helmholtz)[2] for i in local_branch.bodies_index]
        local_branch.local_expansion[:,1,:] .= ϕ
        local_branch.local_expansion[:,2,:] .= 0.0
        velocity_l_ϕ = [evaluate_local(local_system[i,Position()], local_branch, expansion_order, lamb_helmholtz)[2] for i in local_branch.bodies_index]
        local_branch.local_expansion[:,2,:] .= χ

        mags_ϕ = [norm(this_v) for this_v in velocity_l_ϕ]
        mags_χ = [norm(this_v) for this_v in velocity_l_χ]
        mags_l = [norm(this_v) for this_v in velocity_l]

        i = findfirst((x)->x==maximum(abs.(error_l_velocity)), abs.(error_l_velocity))
        # println("\nChecking relative local contribution:")
        # @show norm(mags_ϕ[i] / mags_χ[i])
        =#
    end

    # overall error
    ϕχ20 = deepcopy(local_branch.local_expansion)
    local_branch.local_expansion .= 0.0
    multipole_to_local!(local_branch, multipole_branch, expansion_order, Val(lamb_helmholtz))
    potential_o = [evaluate_local(local_system[i,Position()], local_branch, expansion_order, lamb_helmholtz)[1] for i in local_branch.bodies_index]
    velocity_o = [evaluate_local(local_system[i,Position()], local_branch, expansion_order, lamb_helmholtz)[2] for i in local_branch.bodies_index]
    χ = deepcopy(local_branch.local_expansion[:,2,:])
    local_branch.local_expansion[:,2,:] .= 0.0
    velocity_o_ϕ = [evaluate_local(local_system[i,Position()], local_branch, expansion_order, lamb_helmholtz)[2] for i in local_branch.bodies_index]
    local_branch.local_expansion[:,2,:] .= χ
    ϕ = deepcopy(local_branch.local_expansion[:,1,:])
    local_branch.local_expansion[:,1,:] .= 0.0
    velocity_o_χ = [evaluate_local(local_system[i,Position()], local_branch, expansion_order, lamb_helmholtz)[2] for i in local_branch.bodies_index]
    local_branch.local_expansion[:,1,:] .= ϕ

    # println("\nChecking local contributors")
    mags_ϕ = [norm(this_v) for this_v in velocity_o_ϕ]
    mags_χ = [norm(this_v) for this_v in velocity_o_χ]
    mags_o = [norm(this_v) for this_v in velocity_o]
    ratio_ϕ = mags_ϕ ./ mags_o
    # @show mean(abs.(ratio_ϕ))
    # @show std(abs.(ratio_ϕ))
    ratio_χ = mags_χ ./ mags_o
    # @show mean(abs.(ratio_χ))
    # @show std(abs.(ratio_χ))

    error_o = absolute_error(potential_direct, potential_o)
    error_o_velocity = absolute_error(velocity_direct, velocity_o)


    if body_type !== Point{Source} && body_type !== Point{Vortex}
        error_l = abs.(error_o) .- abs.(error_mp)
        dV_mp = [velocity_direct[i] - velocity_mp[i] for i in eachindex(velocity_mp)]
        dV_o = [velocity_direct[i] - velocity_o[i] for i in eachindex(velocity_o)]
        dV = [dV_o[i] - dV_mp[i] for i in eachindex(dV_mp)]
        error_l_velocity = [norm(V) for V in dV]
        # error_l_velocity = abs.(error_o_velocity) .- abs.(error_mp_velocity)
    end

    local_branch.local_expansion .= ϕχ20

    return error_mp, error_l, error_o, error_mp_velocity, error_l_velocity, error_o_velocity
end

function mymax(vect)
    #=
    mp, mn = 0.0, 0.0
    for v in vect
        mp = max(v,mp)
        mn = min(v,mn)
    end
    if mp > -mn
        return mp
    else
        return mn
    end
    =#
    return maximum(abs.(vect))
end

"""
Note that \$\\Delta C\$ should be provided in the evaluation vector r⃗ rotated frame, and the source box is approximated as a sphere.
"""
function ϕχ_integrals(ΔC, R, nmax, s=1.0; method=1)

    # lengths
    target_radius = s * 0.5
    dx = s / 100

    # rotation details
    # sθ, cθ = sincos(ẑθ)
    # sϕ, cϕ = sincos(ẑϕ)
    # Ĉz = SVector{3}(sθ*cϕ, sθ*sϕ, cθ)
    # sθ, cθ = sincos(ẑθ+π*0.5)
    # Ĉx = SVector{3}(sθ*cϕ, sθ*sϕ, cθ)
    # Ĉy = cross(Ĉz, Ĉx)

    # Ĉx = SVector{3,eltype(ΔC)}(R[1,1],R[2,1],R[3,1])
    # Ĉy = SVector{3,eltype(ΔC)}(R[1,2],R[2,2],R[3,2])
    # Ĉz = SVector{3,eltype(ΔC)}(R[1,3],R[2,3],R[3,3])

    # if FastMultipole.DEBUG[]
    #     @show Ĉx Ĉy Ĉz
    # end

    Ĉz = ΔC / norm(ΔC)
    Ĉx = SVector{3}(1.0,0,0) - Ĉz * Ĉz[1]
    Ĉx /= norm(Ĉx)
    Ĉy = cross(Ĉz, Ĉx)

    @assert isapprox(dot(Ĉx,Ĉy), 0.0; atol=1e-12)
    @assert isapprox(dot(Ĉy,Ĉz), 0.0; atol=1e-12)
    @assert isapprox(dot(Ĉx,Ĉz), 0.0; atol=1e-12)

    # integration bounds
    x⃗ = -target_radius * Ĉx + dx * 0.5 * Ĉx
    δx⃗ = Ĉx * dx
    y⃗0 = -target_radius * Ĉy + dx * 0.5 * Ĉy
    δy⃗ = Ĉy * dx
    z⃗0 = -target_radius * Ĉz + dx * 0.5 * Ĉz
    δz⃗ = Ĉz * dx

    # preallocate integrals
    res = zeros(Complex{Float64},7,nmax)
    Φₙ = 0.0 + 0im
    Φₙ0 = 0.0 + 0im
    Φₙ1 = 0.0 + 0im
    Φₙ2 = 0.0 + 0im
    Χₙ0 = 0.0 + 0im
    Χₙ1 = 0.0 + 0im
    Χₙ2 = 0.0 + 0im

    # perform integration
    for ix in 1:100
        y⃗ = y⃗0
        for iy in 1:100
            z⃗ = z⃗0
            for iz in 1:100
                ρ⃗ = ΔC + x⃗ + y⃗ + z⃗
                if norm(ρ⃗ - ΔC) <= target_radius # || true
                    ρ, θ, ϕ = FastMultipole.cartesian_to_spherical(ρ⃗)
                    ρnp1 = ρ * ρ
                    ρnp2 = ρ * ρnp1
                    eimϕ = exp(-im*ϕ)
                    my_eimϕ = abs(real(eimϕ)) + im * abs(imag(eimϕ))
                    e2imϕ = eimϕ * eimϕ
                    my_e2imϕ = abs(real(e2imϕ)) + im * abs(imag(e2imϕ))
                    sθ, cθ = sincos(θ)

                    # legendre polynomials, n=1
                    P_nm1_0 = 1.0
                    P_n_0 = cθ

                    P_nm1_1 = 0.0
                    P_n_1 = -sθ

                    P_nm1_2 = 0.0
                    P_n_2 = 0.0
                    P_np1_2 = 3 * sθ * sθ

                    # update integrals
                    for n in 1:nmax

                        # next legendre polynomials
                        P_np1_0 = ( (2*n+1) * cθ * P_n_0 - n * P_nm1_0 ) / (n+1)
                        P_np1_1 = ( (2*n+1) * cθ * P_n_1 - (n+1) * P_nm1_1 ) / n
                        P_np2_2 = ( (2*n+3) * cθ * P_np1_2 - (n+3) * P_n_2 ) / n

                        #=
                        @assert isapprox(Plm(cθ,n,1), P_n_1; atol=1e-12)
                        @assert isapprox(Plm(cθ,n,0), P_n_0; atol=1e-12)
                        if n > 1
                            @assert isapprox(Plm(cθ,n,2), P_n_2; atol=1e-12)
                        end
                        @assert isapprox(Plm(cθ,n+1,0), P_np1_0; atol=1e-12)
                        @assert isapprox(Plm(cθ,n+1,1), P_np1_1; atol=1e-12)
                        @assert isapprox(Plm(cθ,n+1,2), P_np1_2; atol=1e-12)
                        =#

                        # update integrals
                        if method == 1
                            res[1,n] += im / ρnp1 * P_n_1 * eimϕ
                            res[2,n] += (n+1) * 0.5 / ρnp1 * P_n_0
                            res[3,n] += 1.0 / (n*ρnp1) * P_n_1 * eimϕ
                            res[4,n] += 1.0 / (2*n*ρnp1) * P_n_2 * e2imϕ
                            res[5,n] += n * 0.5 / ρnp2 * P_np1_0
                            res[6,n] += n / ((n+1)*ρnp2) * P_np1_1 * eimϕ
                            res[7,n] += 0.5 / ((n+1)*ρnp2) * P_np1_2 * e2imϕ

                        elseif method == 2
                            res[1,n] += im / ρnp1 * abs(P_n_1) * (real(my_eimϕ) - im*imag(my_eimϕ))
                            res[2,n] += (n+1) * 0.5 / ρnp1 * abs(P_n_0)
                            res[3,n] += 1.0 / (n*ρnp1) * abs(P_n_1) * my_eimϕ
                            res[4,n] += 1.0 / (2*n*ρnp1) * abs(P_n_2) * my_e2imϕ
                            res[5,n] += n * 0.5 / ρnp2 * abs(P_np1_0)
                            res[6,n] += n / ((n+1)*ρnp2) * abs(P_np1_1) * my_eimϕ
                            res[7,n] += 0.5 / ((n+1)*ρnp2) * abs(P_np1_2) * my_e2imϕ
                        end

                        # recurse
                        ρnp1 *= ρ
                        ρnp2 *= ρ

                        # Pn0
                        P_nm1_0 = P_n_0
                        P_n_0 = P_np1_0

                        # Pn1
                        P_nm1_1 = P_n_1
                        P_n_1 = P_np1_1

                        # Pn2
                        P_nm1_2 = P_n_2
                        P_n_2 = P_np1_2
                        P_np1_2 = P_np2_2

                    end
                end
                z⃗ += δz⃗
            end
            y⃗ += δy⃗
        end
        x⃗ += δx⃗
    end

    # volume
    V = 4/3*pi*target_radius*target_radius*target_radius

    res .*= dx * dx * dx / V
    #res[6,:] .*= im

    #=
    # ϕ error
    Φₙ  *= dx * dx * dx / V
    Φₙ0 *= dx * dx * dx / V
    Φₙ1 *= dx * dx * dx / V
    Φₙ2 *= dx * dx * dx / V

    # χ error
    Χₙ0 *= dx * dx * dx / V
    Χₙ1 *= dx * dx * dx / V * im
    Χₙ2 *= dx * dx * dx / V
    =#

    #return Φₙ, Φₙ0, Φₙ1, Φₙ2, Χₙ0, Χₙ1, Χₙ2
    return res
end

function get_ϕχ_error(ΔC, target_radius, ω⃗, n, v̂, r, R; method=2)

    ωx, ωy, ωz = ω⃗
    vx, vy, vz = v̂

    Φₙ, Φₙ0, Φₙ1, Φₙ2, Χₙ0, Χₙ1, Χₙ2 = ϕχ_integrals(ΔC, R, n, target_radius * 2; method)[:,end]

    if FastMultipole.DEBUG[]
        Φₙs, Φₙ0s, Φₙ1s, Φₙ2s, Χₙ0s, Χₙ1s, Χₙ2s = ϕχ_integrals(ΔC, R, n, 1.0; method)[:,end]
        Φₙz, Φₙ0z, Φₙ1z, Φₙ2z, Χₙ0z, Χₙ1z, Χₙ2z = ϕχ_integrals(ΔC, R, n, 3.0; method)[:,end]
        s = target_radius * 2
        @show Φₙs, Φₙz
        @show Φₙ0s, Φₙ0z
        @show Φₙ1s, Φₙ1z
        @show Φₙ2s, Φₙ2z
        @show Χₙ0s, Χₙ0z
        @show Χₙ1s, Χₙ1z
        @show Χₙ2s, Χₙ2z
    end

    #=
    @assert isapprox(Inϕ, Φₙ; atol=1e-6) "Inϕ = $Inϕ, Φₙ=$Φₙ"
    @assert isapprox(Inϕ0, Φₙ0; atol=1e-6)
    @assert isapprox(Inϕ1, Φₙ1; atol=1e-6)
    @assert isapprox(Inϕ2, Φₙ2; atol=1e-6)
    @assert isapprox(Inχ0, Χₙ0; atol=1e-6)
    @assert isapprox(Inχ1, Χₙ1; atol=1e-6) "Inχ1 = $Inχ1, Χₙ1 = $Χₙ1"
    @assert isapprox(Inχ2, Χₙ2; atol=1e-6)
    =#

    val = vy * (ωx * (real(Φₙ2) + real(Φₙ0)) + ωy * (imag(Φₙ0) - imag(Φₙ2)) + ωz * real(Φₙ1))
    val += vx * (ωx * (imag(Φₙ2) + imag(Φₙ0)) + ωy * (real(Φₙ2) - real(Φₙ0)) + ωz * imag(Φₙ1))
    val += vz * (ωx * real(Φₙ) - ωy * imag(Φₙ))
    val *= r^(n-1)

    valϕ = val
    val += (vx + im*vy) * ( (ωy+im*ωx)*Χₙ0 - (ωy-im*ωx)*Χₙ2 + ωz*Χₙ1 ) * r^n
    valχ = val - valϕ

    if method == 1
        ωx = abs(ωx)
        ωy = abs(ωy)
        ωz = abs(ωz)
        vx = abs(vx)
        vy = abs(vy)
        Φₙ = abs(Φₙ)
        Φₙ0 = abs(Φₙ0)
        Φₙ1 = abs(Φₙ1)
        Φₙ2 = abs(Φₙ2)
        Χₙ0 = abs(Χₙ0)
        Χₙ1 = abs(Χₙ1)
        Χₙ2 = abs(Χₙ2)
        val = vy * (ωx * (Φₙ2 + Φₙ0) + ωy * (Φₙ0 + Φₙ2) + ωz * Φₙ1)
        val += vx * (ωx * (Φₙ2 + Φₙ0) + ωy * (Φₙ2 + Φₙ0) + ωz * Φₙ1)
        val += vz * (ωx * Φₙ + ωy * Φₙ)
        val *= r^(n-1)

        valϕ = val
        val += (vx + vy) * ( (ωy+ωx)*Χₙ0 + (ωy+ωx)*Χₙ2 + ωz*Χₙ1 ) * r^n
        valχ = val - valϕ

    elseif method == 2
        ωx = abs(ωx)
        ωy = abs(ωy)
        ωz = abs(ωz)

        # due to ϕ
        val = (ωy * (real(Φₙ2) + real(Φₙ0)) + ωx * (imag(Φₙ2) + imag(Φₙ0)) + ωz * imag(Φₙ1) )# v̂x
        val += (ωx * (real(Φₙ2) + real(Φₙ0)) + ωy * (imag(Φₙ2) + imag(Φₙ0)) + ωz * real(Φₙ1) )# v̂y
        val += (ωx * real(Φₙ) + ωy * imag(Φₙ) )# v̂z
        val *= r^(n-1)

        # due to χ
        # += here before vvv
        valχ = (ωy * (Χₙ0 + Χₙ2) + im * ωx * real(Χₙ0 + Χₙ2) + ωx * imag(Χₙ0 + Χₙ2) + ωz * Χₙ1 )# v̂x
        valχ += (ωx * (Χₙ0 + Χₙ2) + im * ωy * real(Χₙ0 + Χₙ2) + ωy * imag(Χₙ0 + Χₙ2) + im * ωz * real(Χₙ1) + ωz * imag(Χₙ1) )# v̂y

        Χx = abs(Χₙ0+Χₙ2)
        Χy = Χx
        Χz = abs(Χₙ1)
        valχ = abs(ωx) * Χx + abs(ωy) * Χy + abs(ωz) * Χz

        if FastMultipole.DEBUG[]
            println("Working (slow) method (rotated frame):")
            @show Χx Χy Χz
        end


        valχ *= r^n

        # combined
        val += valχ

    end

    return val
end

function local_error_ϕχ(local_branch, multipole_branch, expansion_order)
    # extract fields
    source_center = multipole_branch.source_center
    target_center = local_branch.target_center
    source_box = multipole_branch.source_box
    target_box = local_branch.target_box

    # calculate vector strength of multipole branch
    ωx, ωy, ωz = FastMultipole.vortex_from_multipole(multipole_branch.multipole_expansion)
    ω⃗ = SVector{3}(ωx, ωy, ωz)

    # rotate coordinate system
    r⃗ = FastMultipole.closest_corner(source_center, target_center, target_box)

    r = norm(r⃗)
    r̂ = r⃗ / r
    R = FastMultipole.rotate(r̂)
    r⃗ = R * r⃗
    #=
    ẑ = SVector{3,eltype(r⃗)}(R[1,3],R[2,3],R[3,3])
    _, ẑθ, ẑϕ = FastMultipole.cartesian_to_spherical(ẑ)
    if ẑϕ < 0.0
        ẑϕ += 2π
    end

    @assert 0.0 <= ẑθ <= π
    @assert 0.0 <= ẑϕ <= 2*π "ϕ = $ẑϕ"

    # ẑθ = mod(ẑθ, FastMultipole.π2)
    # ẑϕ = mod(ẑϕ, FastMultipole.π2)

    if FastMultipole.DEBUG[]
        @show ẑθ ẑϕ
        this_x = R * SVector{3}(1.0,0,0)
        this_y = R * SVector{3}(0,1.0,0)
        this_z = R * SVector{3}(0,0,1.0)
        @show this_x this_y this_z
    end
    =#

    # length scale
    iszero(source_box) && (return zero(r))
    s = (source_box[1] + source_box[2] + source_box[3]) * 0.6666666666666666

    # rotate vortex strength into evaluation frame
    ω⃗ = R * SVector{3,eltype(source_box)}(ωx, ωy, ωz)

    # estimate induced velocity by approximating a point vortex
    # at the expansion center
    vx = ω⃗[2] * r⃗[3] - ω⃗[3] * r⃗[2]
    vy = ω⃗[3] * r⃗[1] - ω⃗[1] * r⃗[3]
    vz = ω⃗[1] * r⃗[2] - ω⃗[2] * r⃗[1]
    vinv = 1/sqrt(vx*vx + vy*vy + vz*vz)
    v̂x = vx*vinv
    v̂y = vy*vinv
    v̂z = vz*vinv
    v̂ = SVector{3,typeof(v̂x)}(v̂x, v̂y, v̂z)

    # ϕχ velocity
    εϕχ = 0.0 + 0im
    for n in expansion_order+1:expansion_order+1
        εϕχ += get_ϕχ_error(R * (source_center-target_center), mean(source_box), ω⃗, n, v̂, r, R; method=2)
    end

    return norm(εϕχ/4/pi)
end

function test_error(local_branch, local_system, multipole_branch, multipole_system, expansion_order::Int, i_m2l)
    # experimental error
    if typeof(multipole_system) <: Gravitational
        body_type = Point{Source}
    elseif typeof(multipole_system) <: VortexParticles
        body_type = Point{Vortex}
    end
    e_mp, e_l, e_o, ev_mp, ev_l, ev_o = multipole_local_error(local_branch, local_system, multipole_branch, multipole_system, expansion_order; body_type)

    # unequal spheres
    ub_mp_us = multipole_error(local_branch, multipole_branch, expansion_order, UnequalSpheres())
    ub_l_us = local_error(local_branch, multipole_branch, expansion_order, UnequalSpheres())

    # uniform unequal spheres
    ub_mp_uus = multipole_error(local_branch, multipole_branch, expansion_order, UniformUnequalSpheres())
    ub_l_uus = local_error(local_branch, multipole_branch, expansion_order, UniformUnequalSpheres())

    # unequal boxes
    ub_mp_ub = multipole_error(local_branch, multipole_branch, expansion_order, UnequalBoxes())
    ub_l_ub = local_error(local_branch, multipole_branch, expansion_order, UnequalBoxes())

    # uniform unequal boxes
    ub_mp_uub = multipole_error(local_branch, multipole_branch, expansion_order, UniformUnequalBoxes())
    ub_l_uub = local_error(local_branch, multipole_branch, expansion_order, UniformUnequalBoxes())

    # uniform cubes
    e_mp_uc = multipole_error(local_branch, multipole_branch, expansion_order, UniformCubes())
    e_l_uc = local_error(local_branch, multipole_branch, expansion_order, UniformCubes())

    # uniform cubes velocity
    ev_mp_uc = multipole_error(local_branch, multipole_branch, expansion_order, UniformCubesVelocity())
    ev_l_uc = local_error(local_branch, multipole_branch, expansion_order, UniformCubesVelocity())

    # LambHelmholtz χ potential
    e_mp_lh = multipole_error(local_branch, multipole_branch, expansion_order, FastMultipole.LambHelmholtzΧVelocity())
    e_l_lh = local_error(local_branch, multipole_branch, expansion_order, FastMultipole.LambHelmholtzΧVelocity())
    # e_l_lh = local_error_ϕχ(local_branch, multipole_branch, expansion_order)

    mp_ratio = maximum(abs.(ev_mp)) / e_mp_lh
    l_ratio = maximum(abs.(ev_l)) / e_l_lh

    # if l_ratio > 10
    #     println("\ni_m2l: $i_m2l")
    #     @show maximum(abs.(ev_l)) / e_l_lh maximum(abs.(ev_mp)) / e_mp_lh
    #     println()
    # end

    return mymax(e_mp), mymax(e_l), mymax(e_o), mymax(ev_mp), mymax(ev_l), mymax(ev_o), ub_mp_us, ub_l_us, ub_mp_uus, ub_l_uus, ub_mp_ub, ub_l_ub, ub_mp_uub, ub_l_uub, e_mp_uc, e_l_uc, ev_mp_uc, ev_l_uc, e_mp_lh, e_l_lh
end

function closest_corner(target_branch, source_branch)
    x0 = source_branch.source_center
    dx = Inf

    # check midpoints
    lx, ly, lz = target_branch.target_box
    points = [
                 target_branch.target_center + SVector{3}(lx,ly,lz),
                 target_branch.target_center + SVector{3}(-lx,ly,lz),
                 target_branch.target_center + SVector{3}(lx,-ly,lz),
                 target_branch.target_center + SVector{3}(lx,ly,-lz),
                 target_branch.target_center + SVector{3}(-lx,-ly,lz),
                 target_branch.target_center + SVector{3}(-lx,ly,-lz),
                 target_branch.target_center + SVector{3}(lx,-ly,-lz),
                 target_branch.target_center + SVector{3}(-lx,-ly,-lz)
                ]
    xt = SVector{3}(0.0,0.0,0.0)
    for pt in points
        if norm(pt - x0) < dx
            dx = norm(pt-x0)
            xt = pt
        end
    end

    return xt
end

function get_points(target_branch)
    lx, ly, lz = target_branch.target_box
    points = [
                 target_branch.target_center + SVector{3}(lx,0.0,0.0),
                 target_branch.target_center + SVector{3}(0.0,ly,0.0),
                 target_branch.target_center + SVector{3}(0.0,0.0,lz),
                 target_branch.target_center - SVector{3}(lx,0.0,0.0),
                 target_branch.target_center - SVector{3}(0.0,ly,0.0),
                 target_branch.target_center - SVector{3}(0.0,0.0,lz),
                 target_branch.target_center + SVector{3}(lx,ly,lz),
                 target_branch.target_center + SVector{3}(-lx,ly,lz),
                 target_branch.target_center + SVector{3}(lx,-ly,lz),
                 target_branch.target_center + SVector{3}(lx,ly,-lz),
                 target_branch.target_center + SVector{3}(-lx,-ly,lz),
                 target_branch.target_center + SVector{3}(-lx,ly,-lz),
                 target_branch.target_center + SVector{3}(lx,-ly,-lz),
                 target_branch.target_center + SVector{3}(-lx,-ly,-lz),
                 target_branch.target_center + SVector{3}(lx,0.0,0.0)*0.5,
                 target_branch.target_center + SVector{3}(0.0,ly,0.0)*0.5,
                 target_branch.target_center + SVector{3}(0.0,0.0,lz)*0.5,
                 target_branch.target_center - SVector{3}(lx,0.0,0.0)*0.5,
                 target_branch.target_center - SVector{3}(0.0,ly,0.0)*0.5,
                 target_branch.target_center - SVector{3}(0.0,0.0,lz)*0.5,
                 target_branch.target_center + SVector{3}(lx,ly,lz)*0.5,
                 target_branch.target_center + SVector{3}(-lx,ly,lz)*0.5,
                 target_branch.target_center + SVector{3}(lx,-ly,lz)*0.5,
                 target_branch.target_center + SVector{3}(lx,ly,-lz)*0.5,
                 target_branch.target_center + SVector{3}(-lx,-ly,lz)*0.5,
                 target_branch.target_center + SVector{3}(-lx,ly,-lz)*0.5,
                 target_branch.target_center + SVector{3}(lx,-ly,-lz)*0.5,
                 target_branch.target_center + SVector{3}(-lx,-ly,-lz)*0.5
                ]
    return points
end

function closest_point(target_branch, source_branch)
    x0 = source_branch.source_center
    dx = Inf

    # check midpoints
    lx, ly, lz = target_branch.target_box
    points = [
                 target_branch.target_center + SVector{3}(lx,0.0,0.0),
                 target_branch.target_center + SVector{3}(0.0,ly,0.0),
                 target_branch.target_center + SVector{3}(0.0,0.0,lz),
                 target_branch.target_center - SVector{3}(lx,0.0,0.0),
                 target_branch.target_center - SVector{3}(0.0,ly,0.0),
                 target_branch.target_center - SVector{3}(0.0,0.0,lz),
                 target_branch.target_center + SVector{3}(lx,ly,lz),
                 target_branch.target_center + SVector{3}(-lx,ly,lz),
                 target_branch.target_center + SVector{3}(lx,-ly,lz),
                 target_branch.target_center + SVector{3}(lx,ly,-lz),
                 target_branch.target_center + SVector{3}(-lx,-ly,lz),
                 target_branch.target_center + SVector{3}(-lx,ly,-lz),
                 target_branch.target_center + SVector{3}(lx,-ly,-lz),
                 target_branch.target_center + SVector{3}(-lx,-ly,-lz)
                ]
    xt = SVector{3}(0.0,0.0,0.0)
    for pt in points
        if norm(pt - x0) < dx
            dx = norm(pt-x0)
            xt = pt
        end
    end

    return xt
end

function multipole_preintegration(rvec, n_max, s, nx)
    dx = s/nx
    x = y = z = -0.5*s + dx*0.5
    R2 = s*s*0.25 # squared radius of largest enclosed sphere
    r = norm(rvec)
    r̂x, r̂y, r̂z = rvec / r # unit vector pointing to target
    rinv = 1/r

    res = zeros(n_max) # preallocate result
    dV = dx * dx * dx # differential volume
    for ix in 1:nx
        y = -0.5*s + dx*0.5
        for iy in 1:nx
            z = -0.5*s + dx*0.5
            for iz in 1:nx
                ρ2 = x*x + y*y + z*z
                if ρ2 > R2 # only integrate outside the sphere
                    ρ = sqrt(ρ2)
                    cθ = (r̂x * x + r̂y * y + r̂z * z) / ρ # cosine theta
                    ρn = ρ # ρ^n
                    Pnm1 = 1.0 # Legendre polynomial of degree n-1
                    Pn = cθ # Legendre polynomial of degree n
                    #ρr = ρ * rinv
                    #ρr_n = ρr

                    # update integral for each n
                    for n in 1:n_max
                        # res[n] += ρr_n * Pn * rinv # actual integral for ρ/r
                        res[n] += ρn * Pn

                        # next Legendre polynomial
                        Pnp1 = ((2n+1) * cθ * Pn - n * Pnm1) / (n+1)

                        # recurse
                        Pnm1 = Pn
                        Pn = Pnp1
                        ρn *= ρ
                        #ρr_n *= ρr
                    end
                end
                z += dx # increment z
            end
            y += dx # increment y
        end
        x += dx # increment x
    end

    # finish integration
    res .*= dV

    return res
end

function multipole_full_integration(rvec, n_max, s, nx)
    dx = s/nx
    x = y = z = -0.5*s + dx*0.5
    R2 = s*s*0.25 # squared radius of largest enclosed sphere
    r = norm(rvec)
    r̂x, r̂y, r̂z = rvec / r # unit vector pointing to target
    rinv = 1/r

    res = zeros(n_max) # preallocate result
    dV = dx * dx * dx # differential volume
    for ix in 1:nx
        y = -0.5*s + dx*0.5
        for iy in 1:nx
            z = -0.5*s + dx*0.5
            for iz in 1:nx
                ρ2 = x*x + y*y + z*z
                if ρ2 > R2 # only integrate outside the sphere
                    ρ = sqrt(ρ2)
                    cθ = (r̂x * x + r̂y * y + r̂z * z) / ρ # cosine theta
                    ρn = ρ # ρ^n
                    Pnm1 = 1.0 # Legendre polynomial of degree n-1
                    Pn = cθ # Legendre polynomial of degree n
                    ρr = ρ * rinv
                    ρr_n = ρr

                    # update integral for each n
                    for n in 1:n_max
                        res[n] += ρr_n * Pn * rinv # actual integral for ρ/r
                        # res[n] += ρn * Pn

                        # next Legendre polynomial
                        Pnp1 = ((2n+1) * cθ * Pn - n * Pnm1) / (n+1)

                        # recurse
                        Pnm1 = Pn
                        Pn = Pnp1
                        ρn *= ρ
                        ρr_n *= ρr
                    end
                end
                z += dx # increment z
            end
            y += dx # increment y
        end
        x += dx # increment x
    end

    # finish integration
    res .*= dV

    return res
end

function multipole_preintegration(n_max, s, nx, nθ, nϕ)
    res = zeros(n_max, nθ, nϕ)
    for (iθ,θ) in enumerate(range(0, stop=pi, length=nθ))
    # for (icθ,cθ) in enumerate(range(1.0, stop=-1.0, length=nθ))
        for (iϕ,ϕ) in enumerate(range(0, stop=pi, length=nϕ))
            # sθ = sqrt(1-cθ*cθ)
            sθ, cθ = sincos(θ)
            sϕ, cϕ = sincos(ϕ)
            r = SVector{3}(sθ*cϕ, sθ*sϕ, cθ)
            # res[:,icθ,iϕ] .= multipole_preintegration(r, n_max, s, nx)
            res[:,iθ,iϕ] .= multipole_preintegration(r, n_max, s, nx)
        end
    end
    return res
end

function get_iΔθ(Δθr,nθ)
    pi05 = pi*0.5
    dΔθ = pi05/nθ
    Δθ = 0.0
    for i in 1:nθ
        if Δθ >= Δθr
            Δθ-Δθr > Δθr-Δθ+dΔθ && (return i-1)
            return i
        end
        Δθ += dΔθ
    end
    pi05-Δθr > Δθr-pi05+dΔθ && (return nθ-1)
    return nθ
end

function local_preintegration(dtheta,n; R=1.0)
    dtheta < 1e-6 && (dtheta += 1e-6)
    drho = R*sin(dtheta)
	nx = 30
	dx = drho*2/nx
	x0 = -drho + dx*0.5
	y0 = -drho + dx*0.5
	z0 = R - drho + dx*0.5
	cx, cy, cz = 0.0, 0.0, R
	val = 0.0
	for x in x0:dx:x0+2*drho
		for y in y0:dx:y0+2*drho
			for z in z0:dx:z0+2*drho
				if (x^2+y^2+(z-cz)^2) <= drho*drho
                    rho = sqrt(x*x+y*y+z*z)
					val += rho^(-n-1)
				end
			end
		end
	end
	val *= dx*dx*dx
	return val
end

function local_preintegration(n_max, θmax, nθ)
    res = zeros(n_max, nθ)
    for (iθ,Δθ) in enumerate(range(0, stop=pi/2, length=nθ))
        for n in 1:n_max
            res[n,iθ] = local_preintegration(Δθ,n)
        end
    end
    return res
end

function local_error_r(x_t, l_branch, m_branch, system, x0, y0, z0, dx, dy, dz, nx, ny, nz, lx, ly, lz, A, P; method="loop", debug=false)
    integral_local = 0.0

    if method == "numerical"
        rvec = x_t - l_branch.target_center
        r = norm(x_t - l_branch.target_center)
        for z in range(z0, step=dz, length=nz)
            for y in range(y0, step=dy, length=ny)
                for x in range(x0, step=dx, length=nx)
                    rho_vec = SVector{3}(x,y,z) + m_branch.source_center - l_branch.target_center
                    rho = norm(rho_vec)
                    cos_gamma = dot(rvec, rho_vec) / (r * rho)
                    for n in P+1:P+2
                        rho_inv = 1/rho
                        Pn = Plm(cos_gamma, n, 0)
                        integral_local += (r * rho_inv)^n * rho_inv * Pn
                    end
                end
            end
        end
        integral_local *= A * dx*dy*dz / (8*lx*ly*lz)

    elseif method == "sphere"
        rvec = x_t - l_branch.target_center
        r = norm(x_t - l_branch.target_center)
        cosines = Float64[]
        s_over_2 = mean([lx,ly,lz])
        # radius2 = (x0*x0 + y0*y0 + z0*z0) / 3 # 3 makes this the largest encompassed sphere
        # radius2 = 3/(4pi)^(0.666666666) * s2

        for z in range(z0, step=dz, length=nz)
            for y in range(y0, step=dy, length=ny)
                for x in range(x0, step=dx, length=nx)
                    rhop = SVector{3}(x,y,z)
                    if rhop'*rhop < s_over_2 * s_over_2 # radius2 # contained in the largest encompassed sphere
                        rho_vec = rhop + m_branch.source_center - l_branch.target_center
                        rho = norm(rho_vec)
                        cos_gamma = dot(rvec, rho_vec) / (r * rho)
                        push!(cosines, cos_gamma)
                        for n in P+1:P+2
                            rho_inv = 1/rho
                            integral_local += (r * rho_inv)^n * rho_inv * Plm(cos_gamma, n, 0)
                        end
                    end
                end
            end
        end
        integral_local *= A * dx*dy*dz / (8*lx*ly*lz)

    elseif method == "separate_radial_sphere"
        rvec = x_t - l_branch.target_center
        r = norm(x_t - l_branch.target_center)
        cosines = Float64[]
        s_over_2 = mean([lx,ly,lz])
        # radius2 = (x0*x0 + y0*y0 + z0*z0) / 3 # 3 makes this the largest encompassed sphere
        # radius2 = 3/(4pi)^(0.666666666) * s2

        for z in range(z0, step=dz, length=nz)
            for y in range(y0, step=dy, length=ny)
                for x in range(x0, step=dx, length=nx)
                    rhop = SVector{3}(x,y,z)
                    if rhop'*rhop < s_over_2 * s_over_2 # radius2 # contained in the largest encompassed sphere
                        rho_vec = rhop + m_branch.source_center - l_branch.target_center
                        rho = norm(rho_vec)
                        cos_gamma = dot(rvec, rho_vec) / (r * rho)
                        push!(cosines, cos_gamma)
                        for n in P+1:P+1
                            rho_inv = 1/rho
                            integral_local += (rho_inv)^(n+1) * Plm(cos_gamma, n, 0)
                        end
                    end
                end
            end
        end
        integral_local *= dx*dy*dz / (8*lx*ly*lz)
        integral_local *= A * r^(P+1)

    elseif method == "separate_angular_sphere"
        rvec = x_t - l_branch.target_center
        r = norm(x_t - l_branch.target_center)
        cosines = Float64[]
        s2 = x0*x0 + y0*y0 + z0*z0
        radius2 = (x0*x0 + y0*y0 + z0*z0) / 3 # 3 makes this the largest encompassed sphere
        # radius2 = 3/(4pi)^(0.666666666) * s2
        this_cos = dot(rvec, m_branch.source_center-l_branch.target_center) / (r*norm(m_branch.source_center-l_branch.target_center))
        this_P = Plm(this_cos, P+1,0)
        for z in range(z0, step=dz, length=nz)
            for y in range(y0, step=dy, length=ny)
                for x in range(x0, step=dx, length=nx)
                    rhop = SVector{3}(x,y,z)
                    if rhop'*rhop < radius2 # contained in the largest encompassed sphere
                        rho_vec = rhop + m_branch.source_center - l_branch.target_center
                        rho = norm(rho_vec)
                        cos_gamma = dot(rvec, rho_vec) / (r * rho)
                        push!(cosines, cos_gamma)
                        for n in P+1:P+2
                            this_integral = 0.0
                            rho_inv = 1/rho
                            integral_local += (r * rho_inv)^n * rho_inv #* this_P#Plm(cos_gamma, n, 0)
                            this_integral += (r * rho_inv)^n * rho_inv #* this_P#Plm(cos_gamma, n, 0)
                        end
                    end
                end
            end
        end

        integral_local *= A * dx*dy*dz / (8*lx*ly*lz) * this_P

    elseif method == "precomputed"
        rvec = x_t - l_branch.target_center
        r = norm(x_t - l_branch.target_center)
        cosines = Float64[]
        # s2 = x0*x0 + y0*y0 + z0*z0
        # radius2 = (x0*x0 + y0*y0 + z0*z0) / 3 # 3 makes this the largest encompassed sphere
        radius2 = mean(l_branch.target_box)^2
        this_cos = dot(rvec, m_branch.source_center-l_branch.target_center) / (r*norm(m_branch.source_center-l_branch.target_center))
        this_P = Plm(this_cos, P+1,0)
        R = norm(m_branch.source_center - l_branch.target_center)
        Δθ = asin(sqrt(radius2)/R)
        iΔθ = get_iΔθ(Δθ, size(FastMultipole.LOCAL_INTEGRALS,2))
        Rinv = 1/R
        R_np2 = R
        rn = r
        for n in 1:P
            R_np2 *= Rinv
            rn *= r
        end

        for n in P+1:P+2
            pint = FastMultipole.LOCAL_INTEGRALS[n,5,5] * R_np2
            integral_local += pint * rn
            R_np2 *= Rinv
            rn *= r
        end
        integral_local *= A / (8*lx*ly*lz) * this_P

    elseif method == "separate_angular"
        rvec = x_t - l_branch.target_center
        r = norm(x_t - l_branch.target_center)
        cosines = Float64[]
        for z in range(z0, step=dz, length=nz)
            for y in range(y0, step=dy, length=ny)
                for x in range(x0, step=dx, length=nx)
                    rho_vec = SVector{3}(x,y,z) + m_branch.source_center - l_branch.target_center
                    rho = norm(rho_vec)
                    cos_gamma = dot(rvec, rho_vec) / (r * rho)
                    push!(cosines, cos_gamma)
                    for n in P+1:20
                        this_cos = dot(rvec, m_branch.source_center-l_branch.target_center) / (r*norm(m_branch.source_center-l_branch.target_center))
                        this_P = Plm(this_cos, n,0)
                        rho_inv = 1/rho
                        integral_local += (r * rho_inv)^n * rho_inv * this_P
                    end
                end
            end
        end
        integral_local *= A * dx*dy*dz / (8*lx*ly*lz)

    elseif method == "separate_angular"
        rvec = x_t - l_branch.target_center
        r = norm(x_t - l_branch.target_center)
        cosines = Float64[]
        for z in range(z0, step=dz, length=nz)
            for y in range(y0, step=dy, length=ny)
                for x in range(x0, step=dx, length=nx)
                    rho_vec = SVector{3}(x,y,z) + m_branch.source_center - l_branch.target_center
                    rho = norm(rho_vec)
                    cos_gamma = dot(rvec, rho_vec) / (r * rho)
                    push!(cosines, cos_gamma)
                    for n in P+1:20
                        this_cos = dot(rvec, m_branch.source_center-l_branch.target_center) / (r*norm(m_branch.source_center-l_branch.target_center))
                        this_P = Plm(this_cos, n,0)
                        rho_inv = 1/rho
                        integral_local += (r * rho_inv)^n * rho_inv * this_P
                    end
                end
            end
        end
        integral_local *= A * dx*dy*dz / (8*lx*ly*lz)

    elseif method == "loop"
        rvec = x_t - l_branch.target_center
        r = norm(rvec)
        err = 0.0
        for i_body in m_branch.bodies_index
            x_s = system[i_body,FastMultipole.Position()]
            q = system[i_body,FastMultipole.Strength()]
            rho_vec = x_s - l_branch.target_center
            rho = norm(rho_vec)
            rho_inv = 1/rho
            cos_gamma = dot(rvec, rho_vec) / (r*rho)
            for n in P+1:20
                if typeof(system) <: VortexParticles
                    err += dot(q,rvec)/(r*r) * n * (r*rho_inv)^n * rho_inv * Plm(cos_gamma, n, 0)
                else
                    err += q * (r*rho_inv)^n * rho_inv * Plm(cos_gamma, n, 0)
                end
            end
        end

        integral_local = err

    elseif method == "loop2"

        rvec = x_t - l_branch.target_center
        r, θt, ϕt = FastMultipole.cartesian_to_spherical(rvec)
        cos_theta_t = cos(θt)
        err = 0.0 + 0im
        for i_body in m_branch.bodies_index
            x_s = system[i_body,FastMultipole.Position()]
            q = system[i_body,FastMultipole.Strength()]
            rho_vec = x_s - l_branch.target_center
            rho, θs, ϕs = FastMultipole.cartesian_to_spherical(rho_vec)
            rho_inv = 1/rho
            cos_theta_s = cos(θs)
            for n in P+1:20
                for m in -n:n
                    if typeof(system) <: VortexParticles
                        q = dot(q,rvec)/(r*r) * n
                    end
                    err += q * Float64(factorial(big(n-abs(m)))/factorial(big(n+abs(m)))) * (r*rho_inv)^n * rho_inv * Plm(cos_theta_s, n, abs(m)) * Plm(cos_theta_t, n, abs(m)) * exp(im*m*(ϕs-ϕt))
                end
            end
        end

        integral_local = err

    end

    return integral_local
end

function get_iθ(θr,nθ)
    Δθ = pi/nθ
    θ = 0.0
    for i in 1:nθ
        if θ >= θr
            θ-θr > θr-θ+Δθ && (return i-1)
            return i
        end
        θ += Δθ
    end
    pi-θr > θr-pi+Δθ && (return nθ-1)
    return nθ
end

function get_iϕ(ϕr,nϕ)
    Δϕ = pi/nϕ
    # ϕr < 0.0 && (ϕr = -ϕr)
    ϕr = abs(ϕr)
    ϕ = 0.0
    for i in 1:nϕ
        if ϕ >= ϕr
            ϕ-ϕr > ϕr-ϕ+Δϕ && (return i-1)
            return i
        end
        ϕ += Δϕ
    end
    pi-ϕr > ϕr-pi+Δϕ && (return nϕ-1)
    return nϕ
end

function integrate_multipole(x_t, m_branch, n)
    s = mean(m_branch.target_box) * 2
    dx = s / 200
    x0 = m_branch.source_center[1] - s*0.5 + dx * 0.5
    y0 = m_branch.source_center[2] - s*0.5 + dx * 0.5
    z0 = m_branch.source_center[3] - s*0.5 + dx * 0.5
    r_vec = x_t - m_branch.source_center
    r = norm(r_vec)
    r_inv = 1/r
    integral = 0.0
    for x in range(x0, step=dx, length=200)
        for y in range(y0, step=dx, length=200)
            for z in range(z0, step=dx, length=200)
                rho_vec = SVector{3}(x,y,z) - m_branch.source_center
                rho = norm(rho_vec)
                cos_gamma = dot(rho_vec, r_vec) / (rho*r)
                integral += rho^n * r_inv^(n+1) * Plm(cos_gamma,n,0)
            end
        end
    end
    Q = m_branch.multipole_expansion[1,1,1]
    integral *= dx*dx*dx * Q/(s*s*s)
    return integral
end

function multipole_error_r(rvec, r, r_inv, m_branch, system, x0, y0, z0, dx, dy, dz, nx, ny, nz, lx, ly, lz, A, P; method="loop", debug=false)

    integral = 0.0
    if method == "loop_LH"
        err = 0.0
        for i_body in m_branch.bodies_index
            x_s = system[i_body,FastMultipole.Position()]
            ω⃗ = system[i_body,FastMultipole.Strength()]
            ρ⃗ = x_s - m_branch.source_center
            ρ = norm(ρ⃗)
            ρ̂ = ρ⃗ ./ ρ
            cosγ = dot(rvec, ρ̂) .* r_inv
            ẑ = rvec .* r_inv
            x̂ = SVector{3}(1.0,0,0) - dot(SVector{3}(1.0,0,0), ẑ) .* ẑ
            if iszero(x̂)
                x̂ = SVector{3}(0,1.0,0) - dot(SVector{3}(0,1.0,0), ẑ) .* ẑ
            end
            x̂ = x̂ ./ norm(x̂)
            ŷ = cross(ẑ, x̂)
            ωx = dot(ω⃗, x̂)
            ωy = dot(ω⃗, ŷ)
            sinϕ = dot(ρ̂,ŷ)
            cosϕ = dot(ρ̂,x̂)
            for n in P+1:20
                if typeof(system) <: VortexParticles
                    err += ρ^n / (n+1) * Plm(cosγ,n,0) * (ωy * cosϕ - ωx * sinϕ)
                else
                    throw("loop_LH not defined for $(typeof(system))")
                end
            end
        end
        integral = err
    elseif method == "loop"
        err = 0.0
        for i_body in m_branch.bodies_index
            x_s = system[i_body,FastMultipole.Position()]
            q = system[i_body,FastMultipole.Strength()]
            rho_vec = x_s - m_branch.source_center
            rho = norm(rho_vec)
            cos_gamma = dot(rvec, rho_vec) / (r*rho)
            for n in P+1:20
                if typeof(system) <: VortexParticles
                    err += (n+1) * dot(q,rvec) * r_inv * r_inv * (rho*r_inv)^n * r_inv * Plm(cos_gamma, n, 0)
                else
                    err += q * (rho*r_inv)^n * r_inv * Plm(cos_gamma, n, 0)
                end
            end
        end
        integral = err

    elseif method == "numerical"
        V = 0.0
        for z in range(z0, step=dz, length=nz)
            for y in range(y0, step=dy, length=ny)
                for x in range(x0, step=dx, length=nx)
                    rho = sqrt(x^2+y^2+z^2)
                    cos_gamma = dot(rvec, SVector{3}(x,y,z)) / (r * rho)
                    for n in P+1:20
                        integral += (rho * r_inv)^n * r_inv * Plm(cos_gamma, n, 0)
                    end
                end
            end
        end
        integral *= A * dx*dy*dz / (8*lx*ly*lz)

    elseif method == "numerical_corners"

        s = mean(m_branch.target_box)*2
        V = 0.0
        for z in range(z0, step=dz, length=nz)
            for y in range(y0, step=dy, length=ny)
                for x in range(x0, step=dx, length=nx)
                    rho = sqrt(x^2+y^2+z^2)
                    if rho >= s*0.5
                        cos_gamma = dot(rvec, SVector{3}(x,y,z)) / (r * rho)
                        for n in P+1:100
                            integral += (rho * r_inv)^n * r_inv * Plm(cos_gamma, n, 0)
                        end
                        V += dx*dy*dz
                    end
                end
            end
        end
        integral *= A * dx*dy*dz / (8*lx*ly*lz)

    elseif method == "numerical_squared"

        for z in range(z0, step=dz, length=nz)
            for y in range(y0, step=dy, length=ny)
                for x in range(x0, step=dx, length=nx)
                    rho = sqrt(x^2+y^2+z^2)
                    cos_gamma = dot(rvec, SVector{3}(x,y,z)) / (r * rho)
                    for n in P+1:100
                        integral += (rho * r_inv)^(2*n) * r_inv * r_inv * Plm(cos_gamma, n, 0)^2
                    end
                end
            end
        end
        integral *= A * A * dx*dy*dz / (8*lx*ly*lz)

    elseif method == "remove_legendre"
        s = mean(m_branch.target_box) * 2
        for n in P+1:100
            #=
            t1 = (s*0.5)^(n+3) / (n+3)
            t2 = f_gpt(r,s,n)
            integral += 1/r^(n+1)*sqrt(2/(2*n+1)) * (t1+t2)
            =#
            t1 = 1/r^(n+1)
            t2 = 0.0
            nint = 5000
            dθ = pi/nint
            for theta in range(dθ/2, step=dθ, length=nint)
                t2 += dθ * Plm(cos(theta),n,0) * sin(theta)
            end
            # t2 = sqrt(2/(2*n+1))
            t3 = 0.0
            rstop = s*0.5
            dρ = rstop/50
            for rho in range(dρ*0.5, stop=rstop-dρ*0.5, length=50)
                t3 += rho^n * dρ
            end
            rstart = s*0.5
            rstop = s*sqrt(2)*0.5
            dρ = (rstart-rstop)/50
            for rho in range(dρ*0.5+rstart, stop=rstop-dρ*0.5, length=50)
                t3 += rho^n * dρ * (-2+1.5*s/rho)
            end
            integral += t1*t2*t3
        end
        integral *= 2*pi*A/(8*lx*ly*lz)

    elseif method == "precomputed"
        s = mean(m_branch.target_box) * 2
        integral = 0.0

        # get polar/azimuth angles for accessing error database
        _, θr, ϕr = FastMultipole.cartesian_to_spherical(rvec)
        _, nθ, nϕ = size(FastMultipole.MULTIPOLE_INTEGRALS)
        iθ = get_iθ(θr,nθ)
        iϕ = get_iϕ(ϕr,nϕ)

        # get s^(P+4)/r^(P+1)
        s_rinv = s / r
        scalar = s * s * s_rinv
        for n in 1:P
            scalar *= s_rinv
        end

        n_max, ndiv_x = P+1, 200
        isodd(n_max) && (n_max += 1)
        # mpint = multipole_preintegration(rvec, n_max, s, ndiv_x)
        # mpint_full = multipole_full_integration(rvec, n_max, s, nx)

        # compute error
        test_integral = 0.0
        for n in P+1:n_max
            integral += FastMultipole.MULTIPOLE_INTEGRALS[n,iθ,iϕ] * scalar
            scalar *= s_rinv
            # test_integral += integrate_multipole(rvec, m_branch, n)
        end

        # integral *= A/(8*lx*ly*lz)
        integral *= A/(s*s*s)
        # println()

        iszero(m_branch.source_box) && (integral = 0.0)

    end
    return integral
end

function rectangle_error(l_branch, l_system, m_branch, m_system, P; m_method="loop", l_method="loop")
    # multipole error (integrate over source)
    nx = ny = nz = 10
    lx, ly, lz = m_branch.target_box
    dx = 2*lx/nx
    dy = 2*ly/ny
    dz = 2*lz/nz
    x0 = -lx+dx*0.5
    y0 = -ly+dy*0.5
    z0 = -lz+dz*0.5

    x_t = closest_point(l_branch, m_branch)
    rvec = x_t - m_branch.source_center
    r = norm(rvec)
    r_inv = 1/r
    A = m_branch.multipole_expansion[1,1,1]

    integral = 0.0
    pts = get_points(l_branch) # absolute coordinates of all corners and centers
    i_pt = 0
    for (i,pt) in enumerate(pts)
        r_t2 = pt - m_branch.source_center
        r_t2n = norm(r_t2)
        r_t2inv = 1/r_t2n
        this_integral = multipole_error_r(r_t2, r_t2n, r_t2inv, m_branch, m_system, x0, y0, z0, dx, dy, dz, nx, ny, nz, lx, ly, lz, A, P; method=m_method)
        integral = max(integral, abs(this_integral))
        if integral == abs(this_integral)
            i_pt = i
        end
    end

    # local error (integrate over source)
    # x_t = closest_corner(l_branch, m_branch)
    # @assert x_t in pts "x_t not found in pts"
    max_local = 0.0
    this_pt = pts[1]
    for x_t2 in pts
        integral_local = local_error_r(x_t2, l_branch, m_branch, m_system, x0, y0, z0, dx, dy, dz, nx, ny, nz, lx, ly, lz, A, P; method=l_method)
        max_local = max(abs(integral_local), max_local)
        if max_local == abs(integral_local)
            this_pt = x_t2
        end
    end

    m_err = integral
    l_err = max_local

    return m_err/4/pi, l_err/4/pi
end

function test_error_from_m2l_list(system; expansion_order=5, multipole_acceptance=0.5, leaf_size=30, shrink_recenter=true, r=true)
    tree = Tree(system; expansion_order=20, leaf_size, shrink_recenter)
    m2l_list, direct_list = build_interaction_lists(tree.branches, tree.branches, tree.leaf_index, multipole_acceptance, true, true, true, UnequalSpheres(), expansion_order, Val(true))

    errs_mp, errs_l, errs_o = Float64[], Float64[], Float64[]
    errs_mp_v, errs_l_v, errs_o_v = Float64[], Float64[], Float64[]
    ubs_mp_us, ubs_l_us = Float64[], Float64[]
    ubs_mp_uus, ubs_l_uus = Float64[], Float64[]
    ubs_mp_ub, ubs_l_ub = Float64[], Float64[]
    ubs_mp_uub, ubs_l_uub = Float64[], Float64[]
    errs_mp_uc, errs_l_uc = Float64[], Float64[]
    errs_mp_uc_v, errs_l_uc_v = Float64[], Float64[]
    errs_mp_r, errs_l_r = Float64[], Float64[]
    errs_mp_lh = Float64[]
    errs_l_lh = Float64[]
    i_m2l = 1
    print_every = Int(ceil(length(m2l_list) / 10))
    @show length(m2l_list)
    for (i_target, i_source, P) in m2l_list

        if i_m2l == 28
            FastMultipole.DEBUG[] = true
        end
        if i_m2l % print_every == 0
            println("Progress: $(i_m2l/length(m2l_list)*100) %")
        end
        local_branch = tree.branches[i_target]
        multipole_branch = tree.branches[i_source]
        if i_m2l == 9 || true
            e_mp, e_l, e_o, e_mp_v, e_l_v, e_o_v, ub_mp_us, ub_l_us, ub_mp_uus, ub_l_uus, ub_mp_ub, ub_l_ub, ub_mp_uub, ub_l_uub, e_mp_uc, e_l_uc, ev_mp_uc, ev_l_uc, e_mp_lh, e_l_lh = test_error(local_branch, system, multipole_branch, system, P, i_m2l)
        else
            e_mp, e_l, e_o, e_mp_v, e_l_v, e_o_v, ub_mp_us, ub_l_us, ub_mp_uus, ub_l_uus, ub_mp_ub, ub_l_ub, ub_mp_uub, ub_l_uub, e_mp_uc, e_l_uc, ev_mp_uc, ev_l_uc, e_mp_lh, e_l_lh = 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
        end
        # settings
        if r
            e_mp_r, e_l_r = rectangle_error(local_branch, system, multipole_branch, system, P; m_method="loop", l_method="loop")
        else
            e_mp_r = e_l_r = 0.0
        end

        if i_m2l == 28
            FastMultipole.DEBUG[] = false
        end
        i_m2l += 1
        push!(errs_mp, e_mp)
        push!(errs_l, e_l)
        push!(errs_o, e_o)
        push!(errs_mp_v, e_mp_v)
        push!(errs_l_v, e_l_v)
        push!(errs_o_v, e_o_v)
        push!(ubs_mp_us, ub_mp_us)
        push!(ubs_l_us, ub_l_us)
        push!(ubs_mp_uus, ub_mp_uus)
        push!(ubs_l_uus, ub_l_uus)
        push!(ubs_mp_ub, ub_mp_ub)
        push!(ubs_l_ub, ub_l_ub)
        push!(ubs_mp_uub, ub_mp_uub)
        push!(ubs_l_uub, ub_l_uub)
        push!(errs_mp_uc, e_mp_uc)
        push!(errs_l_uc, e_l_uc)
        push!(errs_mp_uc_v, ev_mp_uc)
        push!(errs_l_uc_v, ev_l_uc)
        push!(errs_mp_r, e_mp_r)
        push!(errs_l_r, e_l_r)
        push!(errs_mp_lh, e_mp_lh)
        push!(errs_l_lh, e_l_lh)
    end

    # source_xs = [system[i,FastMultipole.Position()] for i in 923:966]
    # source_Γs = [system[i,FastMultipole.Strength()] for i in 923:966]
    # target_x = system[336,FastMultipole.Position()]

    # @show source_xs source_Γs target_x
    # @show target_x

    return errs_mp, errs_l, errs_o, errs_mp_v, errs_l_v, errs_o_v, ubs_mp_us, ubs_l_us, ubs_mp_uus, ubs_l_uus, ubs_mp_ub, ubs_l_ub, ubs_mp_uub, ubs_l_uub, errs_mp_uc, errs_l_uc, errs_mp_uc_v, errs_l_uc_v, errs_mp_lh, errs_l_lh, errs_mp_r, errs_l_r
end



#--- run tests ---#

n_max, nθ, nϕ = 20, 80, 10
nx = 100
s = 1.0
# const MULTIPOLE_INTEGRALS = multipole_preintegration(n_max, s, nx, nθ, nϕ)
θ_max, nθ = pi/2, 50
# const LOCAL_INTEGRALS = local_preintegration(n_max, θ_max, nθ)

#function expected_legendre(n_max, cos0, dcos

# result
#=
n_bodies = 10000
Random.seed!(123)
bodies = rand(7,n_bodies)
bodies[4,:] .= 0.0
# bodies[5,:] .-= 0.5 # coulombic interaction (positive and negative charges)
system = Gravitational(bodies)

# vortex system
# system = generate_vortex(123, n_bodies)

expansion_order, multipole_acceptance, leaf_size = 4, 0.5, 100
# expansion_order, multipole_acceptance, leaf_size = 4, 0.5, 260

leaf_size, multipole_acceptance = 25,0.5
FastMultipole.direct!(system)
velocity_direct = deepcopy(system.velocity_stretching)
reset!(system)
FastMultipole.fmm!(system; expansion_order=10, multipole_acceptance, leaf_size, lamb_helmholtz=true)
velocity_fmm = deepcopy(system.velocity_stretching)
println("\nP=10:")
@show mean(abs.(velocity_direct[1:3,:] .- velocity_fmm[1:3,:]))
reset!(system)
FastMultipole.fmm!(system; expansion_order=5, multipole_acceptance, leaf_size, lamb_helmholtz=true)
velocity_fmm = deepcopy(system.velocity_stretching)
println("\nP=5:")
@show mean(abs.(velocity_direct[1:3,:] .- velocity_fmm[1:3,:]))
reset!(system)
FastMultipole.fmm!(system; expansion_order=1, multipole_acceptance, leaf_size, lamb_helmholtz=true)
velocity_fmm = deepcopy(system.velocity_stretching)
println("\nP=1:")
@show mean(abs.(velocity_direct[1:3,:] .- velocity_fmm[1:3,:]))
=#

# results
means_local = Float64[]
means_multipole = Float64[]
Ps = 1:10
#  for expansion_order in Ps
#      errs_mp, errs_l, errs_o, errs_mp_v, errs_l_v, errs_o_v, ubs_mp_us, ubs_l_us, ubs_mp_uus, ubs_l_uus, ubs_mp_ub, ubs_l_ub, ubs_mp_uub, ubs_l_uub, errs_mp_uc, errs_l_uc, errs_mp_uc_v, errs_l_uc_v, errs_mp_r, errs_l_r = test_error_from_m2l_list(system; expansion_order, multipole_acceptance, leaf_size, shrink_recenter=true) # leaf_size=260
#      push!(means_local, output_stuff(errs_l_v ./ errs_l_uc_v)[1])
#      push!(means_multipole, output_stuff(errs_mp_v ./ errs_mp_uc_v)[1])
#  end


n_bodies = 2000
# n_bodies = 10000
Random.seed!(123)
bodies = rand(7,n_bodies)
bodies[4,:] .= 0.0
bodies[5:7,:] .*= 2
bodies[5:7,:] .-= 0.5
# bodies[5,:] .-= 0.5 # coulombic interaction (positive and negative charges)
# system = Gravitational(bodies)

# vortex system
system = generate_vortex(123, n_bodies)

# for expansion_order in [1,4,10]
expansion_order = 4

    # multipole_acceptance, leaf_size = 0.5, 100
    multipole_acceptance, leaf_size = 0.5, 26

    errs_mp, errs_l, errs_o, errs_mp_v, errs_l_v, errs_o_v, ubs_mp_us, ubs_l_us, ubs_mp_uus, ubs_l_uus, ubs_mp_ub, ubs_l_ub, ubs_mp_uub, ubs_l_uub, errs_mp_uc, errs_l_uc, errs_mp_uc_v, errs_l_uc_v, errs_mp_lh, errs_l_lh, errs_mp_r, errs_l_r = test_error_from_m2l_list(system; expansion_order, multipole_acceptance, leaf_size, shrink_recenter=true, r=false) # leaf_size=260

    #------- get data for paper -------#

    # checking multipole plus local error equals total error (close...)
    v_mp_l = (abs.(errs_mp) .+ abs.(errs_l)) ./ errs_o
    output_stuff(v_mp_l; verbose=true)

    # remove zeros
    this_zero = 1e-16
    threshold = 1e-12
    for i in eachindex(errs_mp)
        if abs(errs_mp_v[i]) < threshold
            errs_mp_v[i] = this_zero
        end
        if abs(errs_l_v[i]) < threshold
            errs_l_v[i] = this_zero
        end
        if abs(errs_o_v[i]) < threshold
            errs_o_v[i] = this_zero
        end
        if abs(errs_mp[i]) < threshold
            errs_mp[i] = this_zero
        end
        if abs(errs_l[i]) < threshold
            errs_l[i] = this_zero
        end
        if abs(errs_o[i]) < threshold
            errs_o[i] = this_zero
        end
        if abs(errs_mp_uc[i]) < threshold
            errs_mp_uc[i] = errs_mp[i]
        end
        if abs(errs_mp_uc_v[i]) < threshold
            errs_mp_uc_v[i] = errs_mp_v[i]
        end
        if abs(ubs_mp_ub[i]) < threshold
            ubs_mp_ub[i] = errs_mp[i]
        end
        if abs(errs_mp_lh[i]) < threshold
            errs_mp_lh[i] = errs_mp_v[i]
        end
        if abs(errs_l_uc[i]) < threshold
            errs_l_uc[i] = errs_l[i]
        end
        if abs(errs_l_uc_v[i]) < threshold
            errs_l_uc_v[i] = errs_l_v[i]
        end
        if abs(ubs_l_ub[i]) < threshold
            ubs_l_ub[i] = errs_l[i]
        end
        if abs(errs_l_lh[i]) < threshold
            errs_l_lh[i] = errs_l_v[i]
        end
    end

    # log scale comparison
    v_mp_uc = log10.(abs.(errs_mp ./ errs_mp_uc))#)
    v_mp_uc_v = log10.(abs.(errs_mp_v ./ errs_mp_uc_v))#)
    v_mp_ub = log10.(abs.(errs_mp ./ ubs_mp_ub))#)
    v_mp_lh = log10.(abs.(errs_mp_v ./ errs_mp_lh))#)
    v_l_lh = log10.(abs.(errs_l_v ./ errs_l_lh))#)
    v_o_lh = log10.(abs.(errs_o_v) ./ (abs.(errs_l_lh) .+ abs.(errs_mp_lh)))
    # v_o_lh = log10.((abs.(errs_l_v) .+ abs.(errs_mp_v)) ./ (abs.(errs_l_lh) .+ abs.(errs_mp_lh)))

    @show minimum(abs.(errs_l_v ./ errs_l_lh))

    v_l_uc = log10.(abs.(errs_l ./ errs_l_uc))#)
    v_l_uc_v = log10.(abs.(errs_l_v ./ errs_l_uc_v))#)
    v_l_ub = log10.(abs.(errs_l ./ ubs_l_ub))#)

    # kde partitioning
    npoints = 2^7
    if typeof(system) <: Gravitational
        # multipole error
        u_mp_uc = kde(v_mp_uc; npoints)
        u_mp_uc_v = kde(v_mp_uc_v; npoints)
        u_mp_ub = kde(v_mp_ub; npoints)

        # local error
        u_l_uc = kde(v_l_uc; npoints)
        u_l_uc_v = kde(v_l_uc_v; npoints)
        u_l_ub = kde(v_l_ub; npoints)
    else
        # lamb-helmholtz error
        u_mp_lh = kde(v_mp_lh; npoints)
        u_l_lh = kde(v_l_lh; npoints)
        u_o_lh = kde(v_o_lh; npoints)
    end

    # make preliminary plots
    fig = figure("multipole error")
    fig.clear()
    fig.add_subplot(111, xlabel="relative prediction error", ylabel="density")
    ax = fig.get_axes()[0]
    if typeof(system) <: Gravitational
        ax.plot(u_mp_uc.x, u_mp_uc.density)
        ax.plot(u_mp_uc_v.x, u_mp_uc_v.density)
        ax.plot(u_mp_ub.x, u_mp_ub.density)
    else
        ax.plot(u_mp_lh.x, u_mp_lh.density)
    end
    ax.set_xticks([-3.0,-2.0,-1.0,0.0,1.0], labels=["0.001",".01", "0.1", "1", "10"])
    ax.set_xlim([-4.0,1.5])
    if typeof(system) <: Gravitational
        legend(["potential", "velocity magnitude", "setting "*L"P_n=1"])
        savefig("figs/compare_multipole_error_n$(n_bodies)_p$expansion_order.png")
    else
        legend(["χ velocity magnitude"])
        savefig("figs/compare_multipole_error_n$(n_bodies)_p$(expansion_order)_lh.png")
    end

    fig2 = figure("local error")
    fig2.clear()
    fig2.add_subplot(111, xlabel="relative prediction error", ylabel="density")
    ax = fig2.get_axes()[0]
    if typeof(system) <: Gravitational
        ax.plot(u_l_uc.x, u_l_uc.density)
        ax.plot(u_l_uc_v.x, u_l_uc_v.density)
        ax.plot(u_l_ub.x, u_l_ub.density)
        ax.set_xticks([-3.0,-2.0,-1.0,0.0,1.0], labels=["0.001",".01", "0.1", "1", "10"])
        ax.set_xlim([-4.0,1.5])
        legend(["potential", "velocity magnitude", "setting "*L"P_n=1"])
        savefig("figs/compare_local_error_n$(n_bodies)_p$expansion_order.png")
    else
        ax.plot(u_l_lh.x, u_l_lh.density)
        ax.set_xticks([-3.0,-2.0,-1.0,0.0,1.0], labels=["0.001",".01", "0.1", "1", "10"])
        ax.set_xlim([-4.0,1.5])
        legend(["χ velocity magnitude"])
        savefig("figs/compare_local_error_n$(n_bodies)_p$(expansion_order)_lh.png")
    end

    fig3 = figure("total error")
    fig3.clear()
    fig3.add_subplot(111, xlabel="relative prediction error", ylabel="density")
    ax = fig3.get_axes()[0]
    if typeof(system) <: Gravitational
        a = nothing
    else
        ax.plot(u_o_lh.x, u_o_lh.density)
        ax.set_xticks([-3.0,-2.0,-1.0,0.0,1.0], labels=["0.001",".01", "0.1", "1", "10"])
        ax.set_xlim([-4.0,1.5])
        savefig("figs/compare_total_error_n$(n_bodies)_p$(expansion_order)_lh.png")
    end

    if typeof(system) <: Gravitational
        # save data
        multipole_comparison=zeros(npoints,6)
        multipole_comparison[:,1] .= u_mp_uc.x
        multipole_comparison[:,2] .= u_mp_uc.density
        multipole_comparison[:,3] .= u_mp_uc_v.x
        multipole_comparison[:,4] .= u_mp_uc_v.density
        multipole_comparison[:,5] .= u_mp_ub.x
        multipole_comparison[:,6] .= u_mp_ub.density
        multipole_header = Matrix{String}(undef,1,6)
        for i in 1:6
            multipole_header[i] = ["potential x", "potential density", "velocity magnitude x", "velocity magnitude density", "Pn=0 x", "Pn=0 density"][i]
        end
        writedlm("data/compare_multipole_error_n$(n_bodies)_p$expansion_order.png", vcat(multipole_header,multipole_comparison), ','; header=true)
    end

    if typeof(system) <: Gravitational
        local_comparison=zeros(npoints,6)
        local_comparison[:,1] .= u_l_uc.x
        local_comparison[:,2] .= u_l_uc.density
        local_comparison[:,3] .= u_l_uc_v.x
        local_comparison[:,4] .= u_l_uc_v.density
        local_comparison[:,5] .= u_l_ub.x
        local_comparison[:,6] .= u_l_ub.density
        local_header = Matrix{String}(undef,1,6)
        for i in 1:6
            local_header[i] = ["potential x", "potential density", "velocity magnitude x", "velocity magnitude density", "Pn=0 x", "Pn=0 density"][i]
        end
        writedlm("data/compare_local_error_n$(n_bodies)_p$expansion_order.png", vcat(local_header,local_comparison), ','; header=true)
    end

# end
#-----------------------------------#

# fig = figure("error with expansion order 3")
# fig.clear()
# fig.add_subplot(111,xlabel="expansion order", ylabel="mean potential error")
# ax = fig.get_axes()[0]
# ax.plot(Ps, means_multipole)
# # ax.plot(Ps, means_local)
# ax.legend(["multipole error", "local error"])


#=
@testset "error: multipole" begin

#--- test multipole error ---#

v_mp = errs_mp ./ errs_mp_r
me_mp, ma_mp, mi_mp, st_mp = output_stuff(v_mp)
v_mp2 = errs_mp ./ errs_mp_uc
me_mp2, ma_mp2, mi_mp2, st_mp2 = output_stuff(v_mp2)

@test isapprox(me_mp, me_mp2; rtol=0.5)
@test isapprox(ma_mp, ma_mp2; rtol=0.5)
@test isapprox(mi_mp, mi_mp2; rtol=0.5)
@test isapprox(st_mp, st_mp2; rtol=0.5)

end

@testset "error: local" begin

#--- test local error ---#

v_l = errs_l ./ errs_l_r
me_l, ma_l, mi_l, st_l = output_stuff(v_l)
v_l2 = errs_l ./ errs_l_uc
me_l2, ma_l2, mi_l2, st_l2 = output_stuff(v_l2)

@test isapprox(me_l, me_l2; rtol=0.5)
@test isapprox(ma_l, ma_l2; rtol=0.5)
@test isapprox(mi_l, mi_l2; rtol=0.5)
@test isapprox(st_l, st_l2; rtol=0.5)

end
=#

#=
# explore multipole preintegration
rvec = SVector{3}(5*sqrt(3),0,0)
n_max = 10
s = 1.0
nx = 100
res = multipole_preintegration(rvec, n_max, s, nx)

# explore scaling as we change s
res2 = multipole_preintegration(rvec, n_max, s*3, nx)

# try different polar angles
new_rvec = SVector{3}(5.0,0,5.0)
new_rvec = new_rvec ./ norm(new_rvec) * 5 * sqrt(3)
res3 = multipole_preintegration(new_rvec, n_max, s, nx)

# try different azimuthal angles
new_rvec = SVector{3}(5.0,5.0,0.0)
new_rvec = new_rvec ./ norm(new_rvec) * 5 * sqrt(3)
res4 = multipole_preintegration(new_rvec, n_max, s, nx)

# check convergence
# new_nx = 1000
# res5 = multipole_preintegration(rvec, n_max, s, new_nx)
# @show res5 .- res

# new_nx = 2000
# res6 = multipole_preintegration(rvec, n_max, s, new_nx)
# @show res6 ./ res

function vary_polar_angle(res, itheta, iϕ)
    fig = figure("vary polar angle")
    fig.clear()
    fig.add_subplot(111,xlabel="n",ylabel="integral")
    ax = fig.get_axes()[0]
    iθ = 16
    for iθ in itheta
        ax.plot(1:size(res,1), abs.(res[:,iθ,iϕ]))
    end
    dθ = pi / (size(res,2)-1)
    ax.legend([L"\theta="*"$((round(i*dθ*180/pi, digits=1)))"*L"^\circ" for i in 0:size(res,2)-1 if i+1 in itheta])
    ax.set_ylim([1e-7,1e-1])
    ax.set_yscale("log")

    fig2 = figure("vary polar angle 2")
    fig2.clear()
    fig2.add_subplot(111,xlabel="theta",ylabel="integral")
    ax2 = fig2.get_axes()[0]
    for n in 1:8#size(res,1)
        ax2.plot(range(0,stop=pi,length=size(res,2)), res[n,:,iϕ])# ./ maximum(abs.(res[n,:,iϕ])))
    end
    ax2.legend(["n=$i" for i in 1:size(res,1)])
    # ax2.set_yscale("log")

    return fig, fig2
end

itheta = collect(1:10)
iϕ = 5
fig = vary_polar_angle(FastMultipole.MULTIPOLE_INTEGRALS, itheta, iϕ)

function vary_azimuth(res, iθ, iphi)
    fig = figure("vary azimuth")
    fig.clear()
    fig.add_subplot(111,xlabel="n",ylabel="integral")
    ax = fig.get_axes()[0]
    for iϕ in iphi
        ax.scatter(1:size(res,1), abs.(res[:,iθ,iϕ]))
    end
    dϕ = 2*pi / (size(res,3)-1)
    ax.legend([L"\theta="*"$((round(i*dϕ*180/pi, digits=1)))"*L"^\circ" for i in 0:size(res,3)-1 if i+1 in iphi])
    ax.set_ylim([1e-7,1e-1])
    ax.set_yscale("log")
    return fig
end

iphi = collect(1:size(FastMultipole.MULTIPOLE_INTEGRALS,3))
iθ = 4
fig_az = vary_azimuth(FastMultipole.MULTIPOLE_INTEGRALS, iθ, iphi)


# x axis adjacent
#R = SVector{3}(2.0,0,0)
#r_multipole, r_local = 0.5, 0.5
#nx_source_bodies, nx_target_bodies = 10, 10
#expansion_order = 3
#system, direct, local_potential = test_rms_error(R, r_multipole, r_local, nx_source_bodies, nx_target_bodies, expansion_order)

# z axis adjacent
R = SVector{3}(2.0,0,0)
r_multipole, r_local = 0.5, 0.5
nx_source_bodies, nx_target_bodies = 10, 10
expansion_order = 5
system, direct, multipole_potential, local_potential, e_mp, e_l, e_o = test_rms_error(R, r_multipole, r_local, nx_source_bodies, nx_target_bodies, expansion_order)
e_o_rms_rel = sqrt(Base.sum((e_o) .^ 2)/1000)
s_error_rel = sqrt(solvason_error(expansion_order))
s_error_abs = direct.potential[1,:] .* s_error_rel
s_diff = abs.(e_o) ./ abs.(s_error_abs[1001:2000])
@show maximum(s_diff) mean(s_diff) median(s_diff) minimum(s_diff)

s = sqrt(1.5^2+0.5^2*2)

r_err = r_averaged_error(expansion_order, s)

=#
