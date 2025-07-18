using Random
using LegendrePolynomials
using Statistics

include("vortex.jl")
include("evaluate_multipole.jl")
include("bodytolocal.jl")

function flatten_derivatives!(jacobian)
    # vector field
    gradient = zeros(3)
    gradient[1] = jacobian[2,3] - jacobian[3,2]
    gradient[2] = jacobian[3,1] - jacobian[1,3]
    gradient[3] = jacobian[1,2] - jacobian[2,1]

    return gradient
end

function Snm(r,θ,ϕ,n,m)
    if abs(m) > n
        return 0.0 + 0im
    end
    t1 = (1.0im)^(-abs(m)) * factorial(big(n-abs(m))) / r^(n+1)
    t2 = Plm(cos(θ),n,abs(m)) * exp(im*m*ϕ)
    return Float64(t1*t2)
end

function Rnm(ρ,θ,ϕ,n,m)
    if abs(m) > n
        return 0.0 + 0im
    end
    t1 = (-1)^n * (1.0*im)^abs(m) / Float64(factorial(big(n+abs(m))))
    t2 = ρ^n
    t3 = Plm(cos(θ),n,abs(m))
    t4 = exp(im*m*ϕ)
    return t1 * t2 * t3 * t4
end

function rotate(r̂)
    ẑ = r̂
    x̂ = SVector{3}(1.0,0,0) - dot(SVector{3}(1.0,0,0), r̂) * r̂
    if iszero(x̂)
        x̂ = SVector{3}(0.0,1.0,0.0) - dot(SVector{3}(0.0,1,0), r̂) * r̂
    end
    x̂ /= norm(x̂)
    ŷ = cross(ẑ, x̂)
    R = vcat(x̂', ŷ', ẑ')
    return x̂, ŷ, ẑ, R
end

function preintegrate_multipole_χ(r̂, n_max; s=1.0)
    # rotate frame
    _, _, _, R = rotate(r̂)

    # preallocate results
    ITns = zeros(Complex{Float64},2,n_max)

    # perform integration
    nx = 100
    dx = dy = dz = s / nx
    x0 = -s*0.5 + dx * 0.5
    y0 = -s*0.5 + dy * 0.5
    z0 = -s*0.5 + dz * 0.5
    for x in range(x0, step=dx, length=nx)
        for y in range(y0, step=dy, length=nx)
            for z in range(z0, step=dz, length=nx)
                ρ, θ, ϕ = FastMultipole.cartesian_to_spherical(R * SVector{3}(x,y,z))
                sθ, cθ = sincos(θ)
                e2imϕ = exp(-2*im*ϕ)

                # Legendre polynomial recursions
                P_nm1_0 = 1.0

                # n=1
                ITns[1,1] += -im * abs(P_nm1_0)
                ITns[2,1] += -abs(P_nm1_0)

                # recurse
                P_nm1_0 = cθ
                P_nm2_0 = 1.0

                # n=2
                ITns[1,2] += -1.5 * ρ * im * abs(P_nm1_0)
                ITns[2,2] += -1.5 * ρ * abs(P_nm1_0)

                # next polynomial of order 0
                n = 2
                P_n_0 = ( (2*n-1) * cθ * P_nm1_0 - (n-1) * P_nm2_0 ) / n

                # recurse (n=3)
                P_nm2_0 = P_nm1_0
                P_nm1_0 = P_n_0

                # next polynomial of order 1 (n=2)
                P_nm1_1 = -sθ
                P_n_1 = (2*n-1) * cθ * P_nm1_1 / (n-1)

                # recurse (n=3)
                P_nm1_1 = P_n_1

                # next polynomial of order 2 (n=3)
                P_nm2_2 = 0.0
                P_nm1_2 = 3 * sθ * sθ

                # n>2
                ρnm1 = ρ * ρ
                for n in 3:n_max
                    ITns[1,n] += ρnm1 * im * 0.5 * (-(n+1) * abs(P_nm1_0) + 1/n * abs(P_nm1_2) * e2imϕ)
                    ITns[2,n] += ρnm1 * 0.5 * (-(n+1) * abs(P_nm1_0) - 1/n * abs(P_nm1_2) * e2imϕ)

                    #--- recurse ---#

                    # ρ^(n-1)
                    ρnm1 *= ρ

                    # order 0
                    P_n_0 = ( (2*n-1) * cθ * P_nm1_0 - (n-1) * P_nm2_0 ) / n
                    P_nm2_0 = P_nm1_0
                    P_nm1_0 = P_n_0

                    # order 2
                    P_n_2 = ( (2*n-1) * cθ * P_nm1_2 - (n+1) * P_nm2_2 ) / (n-2)
                    P_nm2_2 = P_nm1_2
                    P_nm1_2 = P_n_2
                end
            end
        end
    end
    dV = dx * dy * dz
    V = dV * nx * nx * nx
    dV_V = dV / V
    ITns .*= dV_V

    return ITns
end

function preintegrate_multipole_χ(n_max, nθ, nϕ; s=1.0, θ_max=π, ϕ_max=2*π)
    ITns = zeros(Float64,2,2,n_max,nθ,nϕ)
    container = zeros(Complex{Float64},2,n_max)
    for (iϕ,ϕ) in enumerate(range(0.0, ϕ_max, nϕ))
        @show ϕ
        for (iθ,θ) in enumerate(range(0.0, θ_max, nθ))
            @show θ
            sθ, cθ = sincos(θ)
            sϕ, cϕ = sincos(ϕ)
            r̂ = SVector{3}(sθ * cϕ, sθ * sϕ, cθ)
            container .= preintegrate_multipole_χ(r̂, n_max; s)
            for n in 1:n_max
                for i in 1:2 # z component is always zero
                    ITns[1,i,n,iθ,iϕ] = real(container[i,n])
                    ITns[2,i,n,iθ,iϕ] = imag(container[i,n])
                end
            end
        end
    end
    return ITns
end

const ε_Nθ_χ = 80
const ε_Δθ_χ = π / ε_Nθ_χ
const ε_Nϕ_χ = 20
const ε_Δϕ_χ = 2π / ε_Nϕ_χ

function get_iθ_χ(θr)
    θ = 0.0
    for i in 1:ε_Nθ_χ
        if θ >= θr
            θ-θr > θr-θ+ε_Δθ_χ && (return i-1)
            return i
        end
        θ += ε_Δθ_χ
    end
    π-θr > θr-π+ε_Δθ_χ && (return ε_Nθ_χ-1)
    return ε_Nθ_χ
end

function get_iϕ_χ(ϕr)
    if ϕr < 0
        ϕr += FastMultipole.π2
    end
    ϕ = 0.0
    for i in 1:ε_Nϕ_χ
        if ϕ >= ϕr
            ϕ-ϕr > ϕr-ϕ+ε_Δϕ_χ && (return i-1)
            return i
        end
        ϕ += ε_Δϕ_χ
    end
    FastMultipole.π2-ϕr > ϕr-FastMultipole.π2+ε_Δϕ_χ && (return ε_Nϕ_χ-1)
    return ε_Nϕ_χ
end

# IΧ2 = preintegrate_multipole_χ(30, ε_Nθ_χ, ε_Nϕ_χ)
# FastMultipole.write_file("src/multipole_integrals_chi.csv", IΧ2)
# const IΧ = FastMultipole.read_file("src/multipole_integrals_chi.csv")

function multipole_check(x_target, n_sources; seed=123, expansion_order=5, method="final χ", χ_method = "loop", v_method="loop")
    # source system
    # source_system = generate_vortex(seed, n_sources)

    # Random.seed!(123)
    # position = zeros(3,n_sources)
    # position .= 0.7,0.4,0.5
    # position .= 0.5,0.5,0.5
    position = rand(3,n_sources) .* 2
    strength = rand(3,n_sources)
    # strength[1,1] = 1.0 # a single vortex of unit strength aligned with the x axis
                        # means the vector field should be in the -y direction
    strength = rand(3,n_sources)
    source_system = VortexParticles(position, strength)

    # source branch
    bodies_index, n_branches, branch_index, i_parent, i_leaf_index = 1:n_sources, 0, 1:0, 0, 1
    center = SVector{3}(0.5,0.5,0.5) * 2
    radius = 0.5 * sqrt(3) * 2
    box = SVector{3}(0.5,0.5,0.5) * 2

    branch = Branch(bodies_index, n_branches, branch_index, i_parent, i_leaf_index, center, radius, box, expansion_order+20)

    # get multipole coefficients
    FastMultipole.body_to_multipole!(branch, source_system, branch.harmonics, Val(expansion_order+20))
    # branch.multipole_expansion[:,1,:] .= 0.0 # we are testing just χ
    # branch.multipole_expansion[:,2,:] .= 0.0 # we are testing just ϕ

    # evaluate expansion
    ϕ, v, _ = evaluate_multipole(x_target, source_center, branch.multipole_expansion, DerivativesSwitch(true,true,false), Val(expansion_order), Val(true))
    ϕtrue, vtrue, _ = evaluate_multipole(x_target, source_center, branch.multipole_expansion, DerivativesSwitch(true,true,false), Val(expansion_order+20), Val(true))

    # check vector field
    v_analytic = SVector{3}(0.0,0,0)
    for i_body in branch.bodies_index
        dx = x_target-source_system[i_body,FastMultipole.Position()]
        q = source_system[i_body,FastMultipole.Strength()]
        r = norm(dx)
        gamma_over_R = q / r / (4*pi)
        jacobian = zeros(3,3)
        gamma_over_R /= r^2
        for j_potential in 1:3
            for i_r in 1:3
                jacobian[i_r,j_potential] -= gamma_over_R[j_potential] * dx[i_r]
            end
        end
        v_analytic += flatten_derivatives!(jacobian)
    end

    # rotate frame
    r⃗ = x_target - source_center
    r = norm(r⃗)
    r̂ = r⃗ / r
    x̂, ŷ, ẑ, R = rotate(r̂)
    @show R

    @assert isapprox(dot(x̂,ŷ),0.0; atol=1e-12)
    @assert isapprox(dot(x̂,ẑ),0.0; atol=1e-12)
    @assert isapprox(dot(ŷ,ẑ),0.0; atol=1e-12)

    ωx, ωy, ωz = FastMultipole.vortex_from_multipole(branch.multipole_expansion)
    ω⃗ = SVector{3}(ωx, ωy, ωz)
    ωxr = dot(ω⃗, x̂)
    ωyr = dot(ω⃗, ŷ)
    ωzr = dot(ω⃗, ẑ)
    ω⃗r = SVector{3}(ωxr, ωyr, ωzr)

    v̂x = dot(vtrue, x̂) / norm(vtrue)
    v̂y = dot(vtrue, ŷ) / norm(vtrue)
    v̂z = dot(vtrue, ẑ) / norm(vtrue)

    # rotated system
    rotated_position = similar(position)
    rotated_strength = similar(strength)
    for i in 1:size(rotated_position,2)
        rotated_position[:,i] = source_center + R * (position[:,i] - source_center)
        rotated_strength[:,i] = R * strength[:,i]
    end
    rotated_system = VortexParticles(rotated_position, rotated_strength)

    rotated_branch = Branch(bodies_index, n_branches, branch_index, i_parent, i_leaf_index, center, radius, box, expansion_order+20)

    # get multipole coefficients
    FastMultipole.body_to_multipole!(rotated_branch, rotated_system, branch.harmonics, Val(expansion_order+20))

    # evaluate expansion
    x_target_rotated = source_center + R * (x_target - source_center)
    ϕtruer, vtruer, _ = evaluate_multipole(x_target_rotated, source_center, rotated_branch.multipole_expansion, DerivativesSwitch(true,true,false), Val(expansion_order+20), Val(true))

    println("\nVerifying rotated χ")
    @show R * vtrue - vtruer

    # calculate v using single sum over n
    vmag = 0.0
    vmag_nov = 0.0
    if method == "simplest ϕ"
        v̂ = v / norm(v)
        inm = 3
        r = norm(x_target - source_center)

        for n in 1:expansion_order
            ϕn1 = branch.multipole_expansion[1,1,inm] + branch.multipole_expansion[2,1,inm]*im
            vmag += ϕn1 * (n+1) / r^(n+2)
            inm += n+1
        end
        vmag *= (-v̂[2] + v̂[1]*im) / 4 / pi

    elseif method == "2 terms ϕ"

        v̂ = v / norm(v)
        vmag = 0.0
        inm = 3
        r = norm(x_target - source_center)

        for n in 1:expansion_order
            ϕn0 = branch.multipole_expansion[1,1,inm-1] + branch.multipole_expansion[2,1,inm-1]*im
            ϕn1 = branch.multipole_expansion[1,1,inm] + branch.multipole_expansion[2,1,inm]*im
            vmag += ϕn0 * (n+1) / r^(n+2) * -v̂[3]
            vmag += ϕn1 * (n+1) / r^(n+2) * (-v̂[2] + v̂[1]*im)
            inm += n+1
        end
        vmag /= 4 * pi

    elseif method == "3 terms ϕ"

        v̂ = v / norm(v)
        vmag = 0.0
        inm = 3
        r = norm(x_target - source_center)

        for n in 1:expansion_order
            ϕn0 = branch.multipole_expansion[1,1,inm-1] + branch.multipole_expansion[2,1,inm-1]*im
            ϕn1 = branch.multipole_expansion[1,1,inm] + branch.multipole_expansion[2,1,inm]*im
            ϕn_1 = -conj(ϕn1)
            vmag += ϕn0 * (n+1) / r^(n+2) * -v̂[3]
            vmag += ϕn1 * (n+1) / r^(n+2) * 0.5 * (-v̂[2] + v̂[1]*im)
            vmag += ϕn_1 * (n+1) / r^(n+2) * 0.5 * (v̂[2] + v̂[1]*im)
            inm += n+1
        end
        vmag /= 4 * pi

    elseif method == "rotated ϕ"

        in0 = 2
        for n in 1:expansion_order
            ϕn0 = 0.0 + 0im
            ϕn1 = 0.0 + 0im
            for i in rotated_branch.bodies_index
                ρ⃗ = rotated_system[i,Position()] - rotated_branch.center
                ρ, θ, ϕ = FastMultipole.cartesian_to_spherical(ρ⃗)
                this_ωxr, this_ωyr, this_ωzr = rotated_system[i,Strength()]

                # ϕn0
                Φₙ0 = ρ^n * Plm(cos(θ),n,1) * exp(im*ϕ)
                ϕn0 += Float64(1/factorial(big(n+1))) * (this_ωyr * real(Φₙ0) - this_ωxr * imag(Φₙ0))

                # ϕn1
                this_Plm = n >= 2 ? Plm(cos(θ),n,2) : 0.0
                Φₙ1x = n*0.5 * ρ^n * Plm(cos(θ),n,0) + 0.5 * ρ^n * this_Plm * exp(-2*im*ϕ) / (n+1)
                Φₙ1y = -im*n*0.5 * ρ^n * Plm(cos(θ),n,0) + im*0.5 * ρ^n * this_Plm * exp(-2*im*ϕ) / (n+1)
                this_Plm = n >= 1 ? Plm(cos(θ),n,1) : 0.0
                Φₙ1z = ρ^n * this_Plm * exp(-im*ϕ) / (n+1)
                ϕn1 += Float64(1/factorial(big(n+1))) * (this_ωxr * Φₙ1x + this_ωyr * Φₙ1y + this_ωzr * Φₙ1z)

            end

            ϕn0_test = rotated_branch.multipole_expansion[1,1,in0] + im * rotated_branch.multipole_expansion[2,1,in0]
            ϕn1_test = rotated_branch.multipole_expansion[1,1,in0+1] + im * rotated_branch.multipole_expansion[2,1,in0+1]
            println()
            @show n ϕn0 / ϕn0_test
            @show ϕn1 / ϕn1_test

            in0 += n+1
        end

    elseif method == "final ϕ"

        vmag = 0.0
        v̂ = v / norm(v)
        r⃗ = x_target - source_center
        r, θ, ϕ = FastMultipole.cartesian_to_spherical(r⃗)
        inm = 1
        in0 = 1
        for n in 0:expansion_order
            # m=0
            ϕn0 = branch.multipole_expansion[1,1,in0] + branch.multipole_expansion[2,1,in0]*im

            Sn1m = factorial(n+1) / r^(n+2)
            vmag += ϕn0 * -v̂[3]*Sn1m

            # m=1
            ϕn1 = (branch.multipole_expansion[1,1,in0+1] + branch.multipole_expansion[2,1,in0+1]*im) * (n>0)
            vmag += ϕn1 * -((v̂[2] - im*v̂[1]) * factorial(n+1) / r^(n+2))

            # recurse indices
            in0 += n+1

        end

        vmag /= 4*pi

    elseif method == "final χ"

        vmag = 0.0
        vmag_nov = 0.0
        v̂ = vtruer / norm(vtruer) # all quantities in rotated frame
        this_r⃗ = R * (x_target - source_center)
        r, θ, ϕ = FastMultipole.cartesian_to_spherical(this_r⃗)
        @show θ, ϕ
        in1 = 3
        for n in 1:expansion_order+20
            # m=1
            χn1 = (rotated_branch.multipole_expansion[1,2,in1] + rotated_branch.multipole_expansion[2,2,in1]*im)
            Sn1 = Float64(factorial(big(n))) / r^(n+1)
            Sn1_check = Snm(r,θ,ϕ,n,0)
            @assert isapprox(Sn1,Sn1_check;rtol=1e-10) "Sn1 = $(Sn1), Sn1_check=$Sn1_check"
            vmag += -χn1 * (v̂[1] + im*v̂[2]) * (n+1) * Float64(factorial(big(n))) / r^(n+1)
            vmag_nov += -abs(χn1) * (n+1) * Float64(factorial(big(n))) / r^(n+1)

            # recurse indices
            in1 += n+1
        end

        vmag /= 4*pi
        vmag_nov /= 4*pi

    end

    @show vmag vmag_nov norm(v) norm(vtrue) norm(vtruer)

    # return vmag_nov / norm(vtrue)

    # calculate error
    εϕ, εχ = 0.0, 0.0

    # check error using known χ
    in1 = (((expansion_order+1) * (expansion_order + 2)) >> 1) + 2
    for n in expansion_order+1:expansion_order+2
        εχ += Complex{Float64}((branch.multipole_expansion[1,2,in1] + branch.multipole_expansion[2,2,in1]*im) * factorial(big(n+1)) / r^(n+1))
        @assert in1 == ((n*(n+1))>>1) + 2
        in1 += n+1
    end
    εχ *= -(v̂x + im*v̂y) / 4 / pi

    println("\nChecking error estimate with known χ")
    # @show εχ
    # @show norm(εχ) / norm(v-vtrue)

    ω⃗_check = SVector{3}(0.0,0,0)
    for i_source in branch.bodies_index
        ω⃗_check += source_system[i_source, FastMultipole.Strength()]
    end
    # println("\nEstimating ω")
    # @show sum(abs.(ω⃗ - ω⃗_check))
    # @show sum(abs.(ω⃗r - R * ω⃗_check))
    # @show ω⃗r

    εχ2 = 0.0
    εχ_loop = 0.0

    println("\nEstimating χ")
    # if χ_method == "preintegration"
    #     ITns = preintegrate_multipole_χ(r̂, n_max; s=1.0)
    # end

    for n in expansion_order+1:expansion_order+2 # +2
        in1 = (((n + 1) * n) >> 1) + 2

        if χ_method == "integration"
            nx = 100
            dx = source_box[1] * 2 / nx
            dy = source_box[2] * 2 / nx
            dz = source_box[3] * 2 / nx
            x0 = -source_box[1] + dx * 0.5
            y0 = -source_box[2] + dy * 0.5
            z0 = -source_box[3] + dz * 0.5
            ITnx = ITny = ITnz = 0.0 + 0im
            for x in range(x0, step=dx, length=nx)
                for y in range(y0, step=dy, length=nx)
                    for z in range(z0, step=dz, length=nx)
                        ρ, θ, ϕ = FastMultipole.cartesian_to_spherical(R * SVector{3}(x,y,z))
                        cθ = cos(θ)
                        # ITnx += ρ^(n-1) * im * 0.5 * (-(n+1) * Plm(cos(θ),n-1,0) + 1/n * Plm(cos(θ),n-1,2) * exp(-2*im*ϕ))
                        # ITny += ρ^(n-1) * 0.5 * (-(n+1) * Plm(cos(θ),n-1,0) - 1/n * Plm(cos(θ),n-1,2) * exp(-2*im*ϕ))
                        # ITnz += ρ^(n-1) * (1 + 1/n) * im * Plm(cos(θ),n-1,1) * exp(-im*ϕ)
                        ITnx += ρ^(n-1) * im * 0.5 * (-(n+1) * abs(Plm(cos(θ),n-1,0)) + 1/n * abs(Plm(cos(θ),n-1,2)) ) * -im #* exp(-2*im*ϕ))
                        ITny += ρ^(n-1) * 0.5 * (-(n+1) * abs(Plm(cos(θ),n-1,0)) - 1/n * abs(Plm(cos(θ),n-1,2)) )# * exp(-2*im*ϕ))
                        ITnz += ρ^(n-1) * (1 + 1/n) * im * abs(Plm(cos(θ),n-1,1)) # * exp(-im*ϕ)
                    end
                end
            end
            dV = dx * dy * dz
            V = dV * nx * nx * nx
            ITnx *= dV / V # / Float64(factorial(big(n+1)))
            ITny *= dV / V # / Float64(factorial(big(n+1)))
            ITnz *= dV / V # / Float64(factorial(big(n+1)))

            println("\nMultipole Integration:")
            # @show n, ITnx, ITny, ITnz ωxr ωyr ωzr

            # χn1 = Complex{Float64}(ωxr * ITnx + ωyr * ITny + ωzr * ITnz)
            χn1 = Complex{Float64}(ωxr * ITnx + ωyr * ITny + ωzr * ITnz)

            in1 = 3
            for n in 2:expansion_order+1
                # recurse indices
                in1 += n
            end


            χn1_true = rotated_branch.multipole_expansion[1,2,in1] + rotated_branch.multipole_expansion[2,2,in1]*im

            @show χn1
            @show χn1_true
            # @show χn1+ωzr*ITnz
            println()


        elseif χ_method == "preintegration"

            # s = mean(source_box)*2
            # χn1 = Complex{Float64}(ωxr * ITns[1,n] + ωyr * ITns[2,n] + ωzr * ITns[3,n])

            # get iθ, iϕ
            _, θr, ϕr = FastMultipole.cartesian_to_spherical(r⃗)
            iθ = get_iθ_χ(θr)
            iϕ = get_iϕ_χ(ϕr)
            println("\nVerifying lookup table indices:")
            @show (iθ-1) * ε_Δθ_χ, θr, iθ
            @show (iϕ-1) * ε_Δϕ_χ, ϕr, iϕ

            # estimate coefficient
            χn1_real = (ωxr * IΧ[1,1,n,iθ,iϕ] + ωyr * IΧ[1,2,n,iθ,iϕ]) # + ωzr * IΧ[1,3,n,iθ,iϕ])
            χn1_imag = (ωxr * IΧ[2,1,n,iθ,iϕ] + ωyr * IΧ[2,2,n,iθ,iϕ]) # + ωzr * IΧ[2,3,n,iθ,iϕ])
            χn1 = χn1_real + im * χn1_imag

            @show n IΧ[1,1,n,iθ,iϕ] + im*IΧ[2,1,n,iθ,iϕ] IΧ[1,2,n,iθ,iϕ] + im*IΧ[2,2,n,iθ,iϕ]# , IΧ[1,3,n,iθ,iϕ] + im*IΧ[2,3,n,iθ,iϕ]

            println()
        end
        # elseif χ_method == "loop"

            χn1_loop = 0.0
            for i_body in branch.bodies_index
                ρ, θ, ϕ = FastMultipole.cartesian_to_spherical(R * (source_system[i_body,Position()] - source_center))
                this_ω⃗ = R * source_system[i_body, Strength()]
                χn1_loop += this_ω⃗[1] * ρ^(n-1) * im * 0.5 * (-(n+1) * Plm(cos(θ),n-1,0) + 1/n * Plm(cos(θ),n-1,2) * exp(-2*im*ϕ))
                χn1_loop += this_ω⃗[2] * ρ^(n-1) * 0.5 * (-(n+1) * Plm(cos(θ),n-1,0) - 1/n * Plm(cos(θ),n-1,2) * exp(-2*im*ϕ))
                χn1_loop += this_ω⃗[3] * ρ^(n-1) * (1 + 1/n) * im * Plm(cos(θ),n-1,1) * exp(-im*ϕ)
            end
            # χn1 /= factorial(big(n+1))
            # χn1 = Complex{Float64}(χn1)

        # end

        # check value
        χn1_check = rotated_branch.multipole_expansion[1,2,in1] + im * rotated_branch.multipole_expansion[2,2,in1]

        # @show χn1 / Float64(factorial(big(n+1))) χn1_check χn1 / Float64(factorial(big(n+1))) / χn1_check
        s = χ_method == "preintegration" ? mean(source_box) * 2 : 1.0
        @show χn1 / Float64(factorial(big(n+1))) / χn1_check * s^(n-1)
        @show χn1_loop / Float64(factorial(big(n+1))) / χn1_check

        @show s^(n-1) / r^(n+1) χn1
        εχ2 += χn1 / r^(n+1) * s^(n-1)
        εχ_loop += χn1_loop / r^(n+1)
        # εχ2 += χn1 * Float64(factorial(big(n+1))) / r^(n+1)
        in1 += n+1
    end
    εχ2 *= -(v̂x + im*v̂y) / 4 / pi
    εχ_loop *= -(v̂x + im*v̂y) / 4 / pi

    println("\nChecking error estimate with estimated χ")
    @show norm(εχ2) / norm(v-vtrue)
    @show norm(εχ_loop) / norm(v-vtrue)

    # test FastMultipole
    εχ_fmm = FastMultipole.multipole_error(r⃗, branch, expansion_order, FastMultipole.LambHelmholtzΧVector())

    @show εχ_fmm / norm(v-vtrue)

    return ϕ, v, v_analytic, εϕ, εχ, branch, source_system
end

function Snm(ρ,θ,ϕ,n,m)
    return (1.0*im)^(-abs(m)) * Float64(factorial(big(n-abs(m)))) / ρ^(n+1) * Plm(cos(θ),n,abs(m)) * exp(im*m*ϕ)
end

function body_to_local_check(target_branch, Δx, ω⃗, P)
    harmonics = target_branch.harmonics
    ρ, θ, ϕ = FastMultipole.cartesian_to_spherical(Δx)
    ωx, ωy, ωz = ω⃗

    thisϕ = deepcopy(target_branch.local_expansion)
    thisϕ .= 0.0
    i = 1

    println("\nDEBUGGING\n")
    @show ρ θ ϕ
    for n in 0:P
        @show n
        for m in 0:n
            # ϕnm
            Snm_a = abs(-m-1) <= n ? Snm(ρ,θ,ϕ,n,-m-1) : 0.0 + 0im
            Snm_b = abs(-m+1) <= n ? Snm(ρ,θ,ϕ,n,-m+1) : 0.0 + 0im
            Snm_c = Snm(ρ,θ,ϕ,n,-m)
            ϕnm = -(-1)^n/n * (0.5 * (ωx+im*ωy) * (n-m) * Snm_a - 0.5*(ωx-im*ωy) * (n+m) * Snm_b - im * ωz * m * Snm_c)
            thisϕ[1,1,i] = real(ϕnm)
            thisϕ[2,1,i] = imag(ϕnm)

            # χnm
            Snm_a = abs(-m+1) <= n+1 ? Snm(ρ,θ,ϕ,n+1,-m+1) : 0.0 + 0im
            Snm_b = abs(-m-1) <= n+1 ? Snm(ρ,θ,ϕ,n+1,-m-1) : 0.0 + 0im
            Snm_c = abs(-m) <= n+1 ? Snm(ρ,θ,ϕ,n+1,-m) : 0.0 + 0im
            if m == 1 && n<4
                @show Snm_a Snm_b Snm_c
            end
            χnm = (-1)^(n+1)/(n+1) * (0.5*(ωy+im*ωx)*Snm_a - 0.5*(ωy-im*ωx)*Snm_b - ωz*Snm_c)
            thisϕ[1,2,i] = real(χnm)
            thisϕ[2,2,i] = imag(χnm)

            # recurse index
            i += 1
        end
    end

    return thisϕ
end

function complexify(arr::Matrix)
    return arr[1,:] .+ im .* arr[2,:]
end

function estimate_ϕn0(source_center, source_box, ω⃗, target_center, n)

    ωx, ωy, ωz = ω⃗
    s = mean(source_box) * 2
    dx = s / 100
    x0 = source_center[1] - s*0.5 + dx*0.5
    y0 = source_center[2] - s*0.5 + dx*0.5
    z0 = source_center[3] - s*0.5 + dx*0.5
    Inϕ = 0.0 + 0im
    for x in range(x0, step=dx, length=100)
        for y in range(y0, step=dx, length=100)
            for z in range(z0, step=dx, length=100)
                ρ⃗ = SVector{3}(x,y,z) - target_center
                ρ, θ, ϕ = FastMultipole.cartesian_to_spherical(ρ⃗)
                Inϕ += im / ρ^(n+1) * Plm(cos(θ),n,1) * exp(-im*ϕ)
            end
        end
    end
    Inϕ *= dx * dx * dx / (s*s*s)
    # println("\nϕn0 integrals:")
    # @show n Inϕ
    return ωx * (-1)^n * Float64(factorial(big(n-1))) * real(Inϕ) - ωy * (-1)^n * Float64(factorial(big(n-1))) * imag(Inϕ)
end

function estimate_ϕn1(source_center, source_box, ω⃗, target_center, n)

    ωx, ωy, ωz = ω⃗
    s = mean(source_box) * 2
    dx = s / 100
    x0 = source_center[1] - s*0.5 + dx*0.5
    y0 = source_center[2] - s*0.5 + dx*0.5
    z0 = source_center[3] - s*0.5 + dx*0.5
    Inϕ0 = 0.0 + 0im
    Inϕ1 = 0.0 + 0im
    Inϕ2 = 0.0 + 0im
    for x in range(x0, step=dx, length=100)
        for y in range(y0, step=dx, length=100)
            for z in range(z0, step=dx, length=100)
                ρ⃗ = SVector{3}(x,y,z) - target_center
                ρ, θ, ϕ = FastMultipole.cartesian_to_spherical(ρ⃗)
                Inϕ0 += (n+1) * 0.5 / ρ^(n+1) * Plm(cos(θ),n,0)
                Inϕ1 += 1.0 / (n*ρ^(n+1)) * Plm(cos(θ),n,1) * exp(-im*ϕ)
                Plm2 = n > 1 ? Plm(cos(θ),n,2) : 0.0
                Inϕ2 += 1.0 / (2*n*ρ^(n+1)) * Plm2 * exp(-2*im*ϕ)
            end
        end
    end
    # println("\nϕn1 integrals:")
    # @show n Inϕ0 Inϕ1 Inϕ2
    Inϕ0 *= dx * dx * dx / (s*s*s)
    Inϕ1 *= dx * dx * dx / (s*s*s)
    Inϕ2 *= dx * dx * dx / (s*s*s)

    return (-1)^(n+1) * Float64(factorial(big(n-1))) * ( (ωx+im*ωy)*Inϕ2 + (ωx-im*ωy)*Inϕ0 + ωz*Inϕ1 )
end

function get_ϕ_error(source_center, source_box, ω⃗, target_center, n, v̂, r; sphere=false)

    ωx, ωy, ωz = ω⃗
    vx, vy, vz = v̂
    s = mean(source_box) * 2
    dx = s / 100
    x0 = source_center[1] - s*0.5 + dx*0.5
    y0 = source_center[2] - s*0.5 + dx*0.5
    z0 = source_center[3] - s*0.5 + dx*0.5
    Inϕ = 0.0 + 0im
    Inϕ0 = 0.0 + 0im
    Inϕ1 = 0.0 + 0im
    Inϕ2 = 0.0 + 0im
    println("ϕerror:")
    for x in range(x0, step=dx, length=100)
        for y in range(y0, step=dx, length=100)
            for z in range(z0, step=dx, length=100)
                if !sphere || norm(SVector{3}(x,y,z)-source_center) <= s*0.5
                    ρ⃗ = SVector{3}(x,y,z) - target_center
                    ρ, θ, ϕ = FastMultipole.cartesian_to_spherical(ρ⃗)
                    Inϕ += im / ρ^(n+1) * abs(Plm(cos(θ),n,1)) * exp(-im*ϕ)
                    Inϕ0 += (n+1) * 0.5 / ρ^(n+1) * abs(Plm(cos(θ),n,0))
                    Inϕ1 += 1.0 / (n*ρ^(n+1)) * abs(Plm(cos(θ),n,1)) * exp(-im*ϕ)
                    Plm2 = n > 1 ? abs(Plm(cos(θ),n,2)) : 0.0
                    Inϕ2 += 1.0 / (2*n*ρ^(n+1)) * Plm2 * exp(-2*im*ϕ)
                end
            end
        end
    end

    V = sphere ? 4/3*pi*(s*0.5)^3 : s * s * s
    Inϕ *= dx * dx * dx / V
    Inϕ0 *= dx * dx * dx / V
    Inϕ1 *= dx * dx * dx / V
    Inϕ2 *= dx * dx * dx / V

    val = vy * (ωx * (real(Inϕ2) + real(Inϕ0)) + ωy * (imag(Inϕ0) - imag(Inϕ2)) + ωz * real(Inϕ1))
    val += vx * (ωx * (imag(Inϕ2) + imag(Inϕ0)) + ωy * (real(Inϕ2) - real(Inϕ0)) + ωz * imag(Inϕ1))
    val += vz * (ωx * real(Inϕ) - ωy * imag(Inϕ))
    val *= r^(n-1)

    return val
end

function estimate_χn1(source_center, source_box, ω⃗, target_center, n)

    ωx, ωy, ωz = ω⃗
    s = mean(source_box) * 2
    dx = s / 100
    x0 = source_center[1] - s*0.5 + dx*0.5
    y0 = source_center[2] - s*0.5 + dx*0.5
    z0 = source_center[3] - s*0.5 + dx*0.5
    Inχ0 = 0.0 + 0im
    Inχ1 = 0.0 + 0im
    Inχ2 = 0.0 + 0im
    for x in range(x0, step=dx, length=100)
        for y in range(y0, step=dx, length=100)
            for z in range(z0, step=dx, length=100)
                ρ⃗ = SVector{3}(x,y,z) - target_center
                ρ, θ, ϕ = FastMultipole.cartesian_to_spherical(ρ⃗)
                Inχ0 += n * 0.5 / ρ^(n+2) * Plm(cos(θ),n+1,0)
                Inχ1 += n / ((n+1)*ρ^(n+2)) * Plm(cos(θ),n+1,1) * exp(-im*ϕ)
                Inχ2 += 0.5 / ((n+1)*ρ^(n+2)) * Plm(cos(θ),n+1,2) * exp(-2*im*ϕ)
            end
        end
    end
    # println("\nχn1 integrals:")
    # @show n Inχ0 Inχ1 Inχ2
    Inχ0 *= dx * dx * dx / (s*s*s)
    Inχ1 *= dx * dx * dx / (s*s*s) * im
    Inχ2 *= dx * dx * dx / (s*s*s)

    return (-1)^(n+1) * Float64(factorial(big(n-1))) * ( (ωy+im*ωx)*Inχ0 + (ωy-im*ωx)*Inχ2 + ωz*Inχ1 )
end

function get_χ_error(source_center, source_box, ω⃗, target_center, n, v̂, r; sphere=false)

    ωx, ωy, ωz = ω⃗
    vx, vy, vz = v̂
    s = mean(source_box) * 2
    dx = s / 100
    x0 = source_center[1] - s*0.5 + dx*0.5
    y0 = source_center[2] - s*0.5 + dx*0.5
    z0 = source_center[3] - s*0.5 + dx*0.5
    Inχ0 = 0.0 + 0im
    Inχ1 = 0.0 + 0im
    Inχ2 = 0.0 + 0im
    for x in range(x0, step=dx, length=100)
        for y in range(y0, step=dx, length=100)
            for z in range(z0, step=dx, length=100)
                if !sphere || norm(SVector{3}(x,y,z)-source_center) <= s*0.5
                    ρ⃗ = SVector{3}(x,y,z) - target_center
                    ρ, θ, ϕ = FastMultipole.cartesian_to_spherical(ρ⃗)
                    Inχ0 += n * 0.5 / ρ^(n+2) * abs(Plm(cos(θ),n+1,0))
                    Inχ1 += n / ((n+1)*ρ^(n+2)) * abs(Plm(cos(θ),n+1,1)) * exp(-im*ϕ)
                    Inχ2 += 0.5 / ((n+1)*ρ^(n+2)) * abs(Plm(cos(θ),n+1,2)) * exp(-2*im*ϕ)
                end
            end
        end
    end

    # println("\nχn1 integrals:")
    V = sphere ? 4/3*π*(s*0.5)^3 : s * s * s
    Inχ0 *= dx * dx * dx / V
    Inχ1 *= dx * dx * dx / V * im
    Inχ2 *= dx * dx * dx / V

    val = (vx + im*vy) * ( (ωy+im*ωx)*Inχ0 - (ωy-im*ωx)*Inχ2 + ωz*Inχ1 ) * r^n

    return val
end

function get_ϕχ_error(source_center, source_box, ω⃗, target_center, n, v̂, r; sphere=false)

    ωx, ωy, ωz = ω⃗
    vx, vy, vz = v̂
    s = mean(source_box) * 2
    dx = s / 100
    x0 = source_center[1] - s*0.5 + dx*0.5
    y0 = source_center[2] - s*0.5 + dx*0.5
    z0 = source_center[3] - s*0.5 + dx*0.5
    Inϕ = 0.0 + 0im
    Inϕ0 = 0.0 + 0im
    Inϕ1 = 0.0 + 0im
    Inϕ2 = 0.0 + 0im
    Inχ0 = 0.0 + 0im
    Inχ1 = 0.0 + 0im
    Inχ2 = 0.0 + 0im
    println("ϕχerror:")
    for x in range(x0, step=dx, length=100)
        for y in range(y0, step=dx, length=100)
            for z in range(z0, step=dx, length=100)
                if !sphere || norm(SVector{3}(x,y,z)-source_center) <= s*0.5
                    ρ⃗ = SVector{3}(x,y,z) - target_center
                    ρ, θ, ϕ = FastMultipole.cartesian_to_spherical(ρ⃗)
                    Inϕ += im / ρ^(n+1) * abs(Plm(cos(θ),n,1)) * exp(-im*ϕ)
                    Inϕ0 += (n+1) * 0.5 / ρ^(n+1) * abs(Plm(cos(θ),n,0))
                    Inϕ1 += 1.0 / (n*ρ^(n+1)) * abs(Plm(cos(θ),n,1)) * exp(-im*ϕ)
                    Plm2 = n > 1 ? abs(Plm(cos(θ),n,2)) : 0.0
                    Inϕ2 += 1.0 / (2*n*ρ^(n+1)) * Plm2 * exp(-2*im*ϕ)
                    Inχ0 += n * 0.5 / ρ^(n+2) * abs(Plm(cos(θ),n+1,0))
                    Inχ1 += n / ((n+1)*ρ^(n+2)) * abs(Plm(cos(θ),n+1,1)) * exp(-im*ϕ)
                    Inχ2 += 0.5 / ((n+1)*ρ^(n+2)) * abs(Plm(cos(θ),n+1,2)) * exp(-2*im*ϕ)
                end
            end
        end
    end

    # volume
    V = sphere ? 4/3*pi*(s*0.5)^3 : s * s * s

    # ϕ error
    Inϕ *= dx * dx * dx / V
    Inϕ0 *= dx * dx * dx / V
    Inϕ1 *= dx * dx * dx / V
    Inϕ2 *= dx * dx * dx / V

    val = vy * (ωx * (real(Inϕ2) + real(Inϕ0)) + ωy * (imag(Inϕ0) - imag(Inϕ2)) + ωz * real(Inϕ1))
    val += vx * (ωx * (imag(Inϕ2) + imag(Inϕ0)) + ωy * (real(Inϕ2) - real(Inϕ0)) + ωz * imag(Inϕ1))
    val += vz * (ωx * real(Inϕ) - ωy * imag(Inϕ))
    val *= r^(n-1)

    # χ error
    Inχ0 *= dx * dx * dx / V
    Inχ1 *= dx * dx * dx / V * im
    Inχ2 *= dx * dx * dx / V

    val += (vx + im*vy) * ( (ωy+im*ωx)*Inχ0 - (ωy-im*ωx)*Inχ2 + ωz*Inχ1 ) * r^n

    return val
end

function local_check(x_target, dx_local, n_sources; seed=123, expansion_order=5, methodϕ="loop", methodχ = "loop", v_method="loop")

    Random.seed!(seed)
    position = rand(3,n_sources) .* 2
    # position[:,1] .= [1.0,1.0,1.0]
    strength = rand(3,n_sources)
    # strength[:,1] .= [1.0,0,0]
    source_system = VortexParticles(position, strength)

    return local_check(x_target, dx_local, source_system; seed, expansion_order, methodϕ, methodχ, v_method)
end

function local_check(x_target, dx_local, source_system::VortexParticles; seed=123, expansion_order=5, methodϕ="loop", methodχ = "loop", v_method="loop")

    # source branch
    bodies_index, n_branches, branch_index, i_parent, i_leaf_index = 1:n_sources, 0, 1:0, 0, 1
    center = SVector{3}(0.5,0.5,0.5) * 2
    radius = 0.5 * sqrt(3) * 2
    box = SVector{3}(0.5,0.5,0.5) * 2

    source_branch = Branch(bodies_index, n_branches, branch_index, i_parent, i_leaf_index, center, radius, box, expansion_order+20)

    # multipole expansion
    FastMultipole.body_to_multipole!(source_branch, source_system, source_branch.harmonics, Val(expansion_order+20))
    _, v_mp, _ = evaluate_multipole(x_target, center, source_branch.multipole_expansion, DerivativesSwitch(false,true,false), Val(expansion_order+20), Val(true))

    # target system
    target_system = FastMultipole.ProbeSystem([x_target]; scalar_potential=true, vector=true)

    # target branch
    target_center = x_target - dx_local
    target_radius = norm(dx_local)
    target_box = SVector{3}(abs(dx_local[1]), abs(dx_local[2]), abs(dx_local[3]))
    target_branch = Branch(1:1, n_branches, branch_index, i_parent, i_leaf_index, target_center, target_center, target_radius, target_radius, target_box, target_box, expansion_order+20)

    # check local coefficients
    weights_tmp_1 =
    Ts = zeros(FastMultipole.length_Ts(expansion_order+10))
    eimϕs = zeros(Float64, 2, expansion_order+11)
    weights_tmp_1 = initialize_expansion(expansion_order+10, Float64)
    weights_tmp_2 = initialize_expansion(expansion_order+10, Float64)
    gradient_n_m = zeros(2,3,size(target_branch.local_expansion,3))
    FastMultipole.multipole_to_local!(target_branch, source_branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, FastMultipole.ζs_mag, FastMultipole.ηs_mag, FastMultipole.Hs_π2, FastMultipole.M̃, FastMultipole.L̃, expansion_order+10, Val(true))
    _, v_l_check, _ = FastMultipole.evaluate_local(x_target-target_center, target_branch.harmonics, gradient_n_m, target_branch.local_expansion, Val(expansion_order), Val(true), DerivativesSwitch(false,true,false))

    local_expansion_check = deepcopy(target_branch.local_expansion)
    target_branch.local_expansion .= 0.0

    # get local coefficients
    for i in source_branch.bodies_index
        body_to_local_point!(Point{Vortex}, target_branch.local_expansion, target_branch.harmonics, source_system[i,Position()] - target_branch.center, source_system[i,Strength()], Val(expansion_order+10))
    end

    # branch.multipole_expansion[:,1,:] .= 0.0 # we are testing just χ
    # branch.multipole_expansion[:,2,:] .= 0.0 # we are testing just ϕ

    # evaluate expansion
    _, v, _ = FastMultipole.evaluate_local(x_target-target_center, target_branch.harmonics, gradient_n_m, target_branch.local_expansion, Val(expansion_order), Val(true), DerivativesSwitch(false,true,false))
    _, vtrue, _ = FastMultipole.evaluate_local(x_target-target_center, target_branch.harmonics, gradient_n_m, target_branch.local_expansion, Val(expansion_order+20), Val(true), DerivativesSwitch(false,true,false))

    this_L = deepcopy(target_branch.local_expansion)
    target_branch.local_expansion[:,2,:] .= 0.0
    _, v_ϕonly, _ = FastMultipole.evaluate_local(x_target-target_center, target_branch.harmonics, gradient_n_m, target_branch.local_expansion, Val(expansion_order), Val(true), DerivativesSwitch(true,true,false))
    _, vtrue_ϕonly, _ = FastMultipole.evaluate_local(x_target-target_center, target_branch.harmonics, gradient_n_m, target_branch.local_expansion, Val(expansion_order+20), Val(true), DerivativesSwitch(true,true,false))
    target_branch.local_expansion .= this_L
    target_branch.local_expansion[:,1,:] .= 0.0
    _, v_χonly, _ = FastMultipole.evaluate_local(x_target-target_center, target_branch.harmonics, gradient_n_m, target_branch.local_expansion, Val(expansion_order), Val(true), DerivativesSwitch(true,true,false))
    _, vtrue_χonly, _ = FastMultipole.evaluate_local(x_target-target_center, target_branch.harmonics, gradient_n_m, target_branch.local_expansion, Val(expansion_order+20), Val(true), DerivativesSwitch(true,true,false))
    target_branch.local_expansion .= this_L

    # check vector field
    v_analytic = SVector{3}(0.0,0,0)
    for i_body in source_branch.bodies_index
        dx = x_target-source_system[i_body,FastMultipole.Position()]
        q = source_system[i_body,FastMultipole.Strength()]
        this_r = norm(dx)
        gamma_over_R = q / this_r / (4*pi)
        jacobian = zeros(3,3)
        gamma_over_R /= this_r^2
        for j_potential in 1:3
            for i_r in 1:3
                jacobian[i_r,j_potential] -= gamma_over_R[j_potential] * dx[i_r]
            end
        end
        v_analytic += flatten_derivatives!(jacobian)
    end

    @assert isapprox(vtrue, v_analytic; atol=1e-12)

    # rotate frame
    r⃗ = x_target - target_center
    @show r⃗
    r = norm(r⃗)
    r̂ = r⃗ / r
    x̂, ŷ, ẑ, R = rotate(r̂)
    r⃗r = R * r⃗

    @assert isapprox(dot(x̂,ŷ),0.0; atol=1e-12)
    @assert isapprox(dot(x̂,ẑ),0.0; atol=1e-12)
    @assert isapprox(dot(ŷ,ẑ),0.0; atol=1e-12)
    @assert isapprox(R * r⃗, SVector{3}(0,0,r); atol=1e-12)

    ωx, ωy, ωz = FastMultipole.vortex_from_multipole(source_branch.multipole_expansion)
    ω⃗ = SVector{3}(ωx, ωy, ωz)
    ωxr = dot(ω⃗, x̂)
    ωyr = dot(ω⃗, ŷ)
    ωzr = dot(ω⃗, ẑ)
    ω⃗r = SVector{3}(ωxr, ωyr, ωzr)

    v̂x = dot(vtrue, x̂) / norm(vtrue)
    v̂y = dot(vtrue, ŷ) / norm(vtrue)
    v̂z = dot(vtrue, ẑ) / norm(vtrue)

    #=
    # rotated system
    rotated_position = similar(position)
    rotated_strength = similar(strength)
    for i in 1:size(rotated_position,2)
        rotated_position[:,i] = target_center + R * (position[:,i] - target_center)
        rotated_strength[:,i] = R * strength[:,i]
    end
    rotated_system = VortexParticles(rotated_position, rotated_strength)

    rotated_source_branch = Branch(bodies_index, n_branches, branch_index, i_parent, i_leaf_index, source_center, target_center, source_radius, target_radius, source_box, target_box, expansion_order+20)

    # get multipole coefficients
    FastMultipole.body_to_multipole!(rotated_branch, rotated_system, branch.harmonics, Val(expansion_order+20))

    # evaluate expansion
    x_target_rotated = source_center + R * (x_target - source_center)
    ϕtruer, vtruer, _ = evaluate_multipole(x_target_rotated, source_center, rotated_branch.multipole_expansion, DerivativesSwitch(true,true,false), Val(expansion_order+20), Val(true))

    println("\nVerifying rotated χ")
    @show R * vtrue - vtruer
    =#

    # calculate v using single sum over n
    if methodϕ == "loop"
        mag∇ϕ = 0.0
        i = 2
        for n in 1:expansion_order
            ϕn0 = target_branch.local_expansion[1,1,i] + target_branch.local_expansion[2,1,i] * im
            ϕn1 = target_branch.local_expansion[1,1,i+1] + target_branch.local_expansion[2,1,i] * im
            mag∇ϕ += (real(ϕn1) * v̂y + imag(ϕn1) * v̂x + ϕn0 * v̂z) * r^(n-1) * (-1)^(n) / Float64(factorial(big(n-1)))
            i += n+1
        end

        mag∇ϕ /= 4π

    elseif methodϕ == "loop2"
        mag∇ϕ = 0.0
        for i in source_branch.bodies_index
            this_ωx, this_ωy, this_ωz = R * source_system[i,Strength()]
            ρ⃗ = R * (source_system[i,Position()] - target_center)
            ρ, θs, ϕs = FastMultipole.cartesian_to_spherical(ρ⃗)
            for n in 1:expansion_order+10
                this_I = im / ρ^(n+1) * Plm(cos(θ),n,1) * exp(-im*ϕ)
                ϕn0 = this_ωx * real(this_I) - this_ωy * imag(this_I)
                mag∇ϕ += ϕn0 * r^(n-1)
            end
        end
    end

    @show mag∇ϕ norm(vtrue_ϕonly)

    if methodχ == "loop"

        mag∇χ = 0.0
        i = 2
        for n in 1:expansion_order
            χn1 = target_branch.local_expansion[1,2,i+1] + target_branch.local_expansion[2,2,i+1] * im
            mag∇χ += χn1 * (v̂x + im*v̂y) * (-1)^n * r^n / Float64(factorial(big(n-1)))
            i += n + 1
        end

        mag∇χ /= 4π
    end

    @show mag∇χ norm(vtrue_χonly)

    #--- check estimates of local coefficients ---#

    #=
    # ϕn0
    ϕn0s = zeros(Complex{Float64},expansion_order)
    for n in 1:expansion_order
        ϕn0s[n] = estimate_ϕn0(source_center, source_box, ω⃗, target_center, n)
    end

    # check
    ϕn0s_check = similar(ϕn0s)
    i = 2
    for n in 1:expansion_order
        ϕn0s_check[n] = target_branch.local_expansion[1,1,i] + im * target_branch.local_expansion[2,1,i]
        i += n+1
    end

    @show ϕn0s ϕn0s_check

    # ϕn1
    ϕn1s = zeros(Complex{Float64}, expansion_order)
    for n in 1:expansion_order
        ϕn1s[n] = estimate_ϕn1(source_center, source_box, ω⃗, target_center, n)
    end

    # check
    ϕn1s_check = similar(ϕn1s)
    i = 2
    for n in 1:expansion_order
        ϕn1s_check[n] = target_branch.local_expansion[1,1,i+1] + im * target_branch.local_expansion[2,1,i+1]
        i += n+1
    end

    @show ϕn1s ϕn1s_check

    # χn1
    χn1s = zeros(Complex{Float64}, expansion_order)
    for n in 1:expansion_order
        χn1s[n] = estimate_χn1(source_center, source_box, ω⃗, target_center, n)
    end

    # check
    χn1s_check = similar(χn1s)
    i = 2
    for n in 1:expansion_order
        χn1s_check[n] = target_branch.local_expansion[1,2,i+1] + im * target_branch.local_expansion[2,2,i+1]
        i += n+1
    end

    @show χn1s χn1s_check
    =#

    #--- error estimate ---#

    v̂ = vtrue / norm(vtrue)

    println("\n==== error estimates ====\n")

    # ϕ vector
    εϕ = 0.0 + 0im
    for n in expansion_order+1:expansion_order+1
        εϕ += get_ϕ_error(source_center, source_box, ω⃗, target_center, n, v̂, r; sphere=false)
    end

    @show εϕ/4/pi norm(vtrue_ϕonly-v_ϕonly)

    # χ vector
    εχ = 0.0 + 0im
    for n in expansion_order+1:expansion_order+1
        εχ += get_χ_error(source_center, source_box, ω⃗, target_center, n, v̂, r; sphere=false)
    end

    @show norm(εχ/4/pi) norm(vtrue_χonly-v_χonly)

    # ϕχ vector
    εϕχ = 0.0 + 0im
    for n in expansion_order+1:expansion_order+1
        εϕχ += get_ϕχ_error(source_center, source_box, ω⃗, target_center, n, v̂, r; sphere=false)
    end

    @show norm(εϕχ/4/pi) norm(vtrue-v)

    #--- spherical source approximation ---#

    println("\n==== spherical source approximation ====\n")

    # ϕ vector
    εϕ = 0.0 + 0im
    sphere = true
    for n in expansion_order+1:expansion_order+1
        εϕ += get_ϕ_error(source_center, source_box, ω⃗, target_center, n, v̂, r; sphere)
    end

    @show εϕ/4/pi norm(vtrue_ϕonly-v_ϕonly)

    # χ vector
    εχ = 0.0 + 0im
    for n in expansion_order+1:expansion_order+1
        εχ += get_χ_error(source_center, source_box, ω⃗, target_center, n, v̂, r; sphere)
    end

    @show norm(εχ/4/pi) norm(vtrue_χonly-v_χonly)

    # ϕχ vector
    εϕχ = 0.0 + 0im
    for n in expansion_order+1:expansion_order+1
        εϕχ += get_ϕχ_error(source_center, source_box, ω⃗, target_center, n, v̂, r; sphere=true)
    end

    @show norm(εϕχ/4/pi) norm(vtrue-v)

end

# x_target = SVector{3}(0.0,1.0,1.0) + SVector{3}(0.5,0.5,0.5)
x_target = -10*SVector{3}(1.0,0.8,1.2) + SVector{3}(0.5,0.5,0.5)
# x_target *= 2
expansion_order = 5
n_sources = 30
seed = 123

# ϕ, v, v_analytic, εϕ, εχ, branch, source_system = multipole_check(x_target, n_sources; seed, expansion_order, χ_method="integration")

vmags_nov = Float64[]
for r in range(sqrt(3)*1.1,stop=20,length=10)
    for θ in range(0, stop=π, length=10)
        for ϕ in range(0, stop=2π, length=20)
            sθ, cθ = sincos(θ)
            sϕ, cϕ = sincos(ϕ)
            x_target = SVector{3}(sθ*cϕ, sθ*sϕ, cθ) * r + SVector{3}(1.0,1,1)
            vmag_nov = multipole_check(x_target, n_sources; seed, expansion_order, method="final χ", χ_method="integration")
            push!(vmags_nov, vmag_nov)
        end
    end
end

# println()

# errs = Float64[]
# for θ in range(0, stop=pi/2, length=10)
#     for ϕ in range(0, stop=pi, length=10)
#         x_target = SVector{3}(sin(θ)*cos(ϕ),sin(θ)*sin(ϕ),cos(θ))
#         x_target = x_target / norm(x_target) * 4.0
#         for seed in 1:1000
#             ϕ, v, v_analytic, εϕ, εχ = multipole_check(x_target, n_sources; seed, expansion_order)
#             push!(errs,norm(v-v_analytic)/(εϕ+εχ))
#         end
#     end
# end
# @show err10

n_sources = 30
local_center = SVector{3}(4.0,-5.0,7.0)
dx_local = SVector{3}(0.0, 0.0, 0.4)
x_target = local_center + dx_local
seed = 123
expansion_order = 5
# local_check(x_target, dx_local, n_sources; seed, expansion_order, methodϕ="loop", methodχ="loop")

