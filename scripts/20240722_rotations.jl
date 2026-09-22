using LegendrePolynomials, WriteVTK, Random

# rotations of the spherical harmonics

#--- containers ---#
struct SphericalHarmonics{TF}
    p::Int
    weights::Vector{Vector{Complex{TF}}}
end

function SphericalHarmonics(seed::Int, p::Int, TF=Float64)
    Random.seed!(seed)
    weights = Vector{Vector{Complex{TF}}}(undef, p+1)
    for i in 0:p
        weights[i+1] = rand(Complex{TF}, 2*i+1)
    end
    return SphericalHarmonics(p, weights)
end

function Ylm(n,m,θ,ϕ)
    return Plm(cos(θ),n,abs(m)) * exp(im*m*ϕ) * (-1)^m * sqrt((2*n+1)/(4*pi)) * Float64(sqrt(factorial(big((n-abs(m)))) / factorial(big(n+abs(m)))))
end

function evaluate(harmonics::SphericalHarmonics{TF}; nθ, nϕ) where TF
    xs = zeros(3,nθ,nϕ,1)
    us_real = zeros(nθ,nϕ,1)
    us_imag = zeros(nθ,nϕ,1)
    for (iϕ,ϕ) in enumerate(range(0,2*pi,nϕ+1)[1:end-1])
        for (iθ,θ) in enumerate(range(0,pi,nθ))
            xs[:,iθ,iϕ,1] .= sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ)
            u = evaluate(harmonics, θ, ϕ)
            us_real[iθ,iϕ,1] = real(u)
            us_imag[iθ,iϕ,1] = imag(u)
        end
    end
    return xs, us_real, us_imag
end

function evaluate(harmonics::SphericalHarmonics{TF}, θ, ϕ) where TF
    res = zero(Complex{TF})
    for h in harmonics.weights
        n = (length(h)-1) >> 1
        for m in -n:n
            ind = m + n + 1
            res += Ylm(n,m,θ,ϕ) * h[ind]
        end
    end
    return res
end

function save_vtk(name, harmonics; nθ=10,nϕ=20)
    xs, us_real, us_imag = evaluate(harmonics; nθ, nϕ)
    vtk_grid(name, xs) do vtk
        vtk["real"] = us_real
        vtk["imag"] = us_imag
    end
end

#--- z axis rotations ---#
function rotate_z(harmonics, α)
    transformed_harmonics = deepcopy(harmonics)
    for h in transformed_harmonics.weights
        p = (length(h)-1) >> 1
        for m in -p:p
            ind = m + p + 1
            h[ind] *= exp(im*m*α)
        end
    end
    return transformed_harmonics
end

#--- y axis rotations ---#
# function bnm(n,m)
#     @assert abs(m) <= n
#     return sqrt((n-m-1)*(n-m) / (2*n-1) * (2*n+1)) * (-1)^(m<0)
# end
#
# function anm(n,m)
#     @assert n >= abs(m)
#     return sqrt((n+1+m)*(n+1-m) / (2*n+1) / (2*n+3))
# end
#
# function build_Hs(p, β, TF=Float64)
#     Hs = [zeros(TF,2*n+1,2*n+1) for n in 0:p]
#
#     # initiate recursion
#     j, m = 1, 0
#     for n in 0:p
#         H = Hs[n+1]
#         for mp in 0:n
#             i = mp+1
#             H[i,j] = (-1)^mp * sqrt(factorial(n-abs(mp)) / factorial(n+abs(mp))) * Plm(cos(β),n,abs(mp))
#         end
#     end
#
#     # initial value
#
#     # perform recursion
#     for n in 0:p
#         H = Hs[n+1]
#         for m in 1:n
#             j = m+1
#             for mp in 0:n
#                 i = mp+1
#
#                 H[i,j] = 1/bnm(n,m) * (0.5 * (bnm(n,-mp-1) * (1-cos(β)) * H[i+1,j-1] - bnm(n,mp-1) * (1 + cos(β)) * H[i-1,j-1]) - anm(n-1,mp) * sin(β) * H[i,j-1])
#             end
#         end
#     end
#
#     return Hs
# end

function build_d(d_nm1, β)
    nm1 = (size(d_nm1,1) - 1) >> 1
    n = nm1 + 1
    d = zeros(2*n+1,2*n+1)

    for m in -n:n
        mp = -n
        x1 = -n < m-1 < n ? sin(β/2)^2 * sqrt((n+m)*(n+m-1)/(n-mp)/(n-mp-1)) * d_nm1[nm1+mp+2,nm1+m] : 0.0
        x2 = -n < m < n ? 2 * sin(β/2) * cos(β/2) * sqrt((n+m)*(n-m)/(n-mp)/(n-mp-1)) * d_nm1[nm1+mp+2,nm1+m+1] : 0.0
        x3 = -n < m+1 < n ? cos(β/2)^2 * sqrt((n-m)*(n-m-1)/(n-mp)/(n-mp-1)) * d_nm1[nm1+mp+2,nm1+m+2] : 0.0
        d[n+mp+1,n+m+1] = x1 + x2 + x3

        for mp in -n+1:n-1
            x1 = -n < m-1 < n ? sin(β/2) * cos(β/2) * sqrt((n+m)*(n+m-1)/(n+mp)/(n-mp)) * d_nm1[nm1+mp+1,nm1+m] : 0.0
            x2 = -n < m < n ? (cos(β/2)^2 - sin(β/2)^2) * sqrt((n-m)*(n+m)/(n-mp)/(n+mp)) * d_nm1[nm1+mp+1,nm1+m+1] : 0.0
            x3 = -n < m+1 < n ? sin(β/2) * cos(β/2) * sqrt((n-m)*(n-m+1)/(n-mp)/(n+mp)) * d_nm1[nm1+mp+1,nm1+m+2] : 0.0
            d[n+mp+1,n+m+1] = x1 + x2 - x3
        end

        mp = n
        x1 = -n < m-1 < n ? cos(β/2)^2 * sqrt((n+m)*(n+m-1)/(n+mp)/(n+mp-1)) * d_nm1[nm1+mp,nm1+m] : 0.0
        x2 = -n < m < n ? 2 * sin(β/2) * cos(β/2) * sqrt((n+m)*(n-m)/(n+mp)/(n+mp-1)) * d_nm1[nm1+mp,nm1+m+1] : 0.0
        x3 = -n < m+1 < n ? sin(β/2)^2 * sqrt((n-m)*(n-m-1)/(n+mp)/(n+mp-1)) * d_nm1[nm1+mp,nm1+m+2] : 0.0
        d[n+mp+1,n+m+1] = x1 - x2 + x3
    end

    return d
end

function access(H, n, mp, m)
    i = n + mp + 1
    j = n + m + 1
    # @assert !iszero(H[i,j]) "attempted access of null mp=$mp, m=$m"
    return H[i,j]
end

function set!(H, val, n, mp, m)
    i = n + mp + 1
    j = n + m + 1
    # @assert iszero(H[i,j])
    H[i,j] = val
end

function a(n,m)
    if n < abs(m)
        return 0.0
    else
        # @assert n >= abs(m)
        return sqrt((n+1+m)*(n+1-m)/(2*n+1)/(2*n+3))
    end
end

function sgn(x)
    return x >= 0 ? 1 : -1
end

function b(n,m)
    if n < abs(m)
        return 0
    else
        # @assert n >= abs(m)
        return sgn(m) * sqrt((n-m-1)*(n-m)/(2*n-1)/(2*n+1))
    end
end

function c(n,m)
    return 1/2*(-1)^m*sgn(m) * sqrt((n-m)*(n+m+1))
end

function d(n,m)
    # @assert -n-1 <= m <= n
    return sgn(m)/2 * sqrt((n-m)*(n+m+1))
end

function build_H(H_nm1, β, TF=Complex{Float64})
    nm1 = (size(H_nm1,1) - 1) >> 1
    n = nm1 + 1
    H = zeros(TF,2*n+1,2*n+1)
    H_np1 = zeros(2*(n+1)+1)

    # populate H_n_0m
    mp = 0
    for m in 0:n
        set!(H, (-1)^m * Float64(sqrt(factorial(big(n-abs(m)))/factorial(big(n+abs(m))))) * Plm(cos(β),n,abs(m)), n, mp, m)
    end

    # compute H_np1_0m
    for m in 0:n+1
        H_np1[n+1+m+1] = (-1)^m * Float64(sqrt(factorial(big(n+1-abs(m)))/factorial(big(n+1+abs(m))))) * Plm(cos(β),n+1,abs(m))
    end

    # compute H_n_1m
    for m in 1:n
        set!(H, (b(n+1,-m-1) * (1-cos(β)) / 2 * H_np1[n+1+m+1+1] - b(n+1,m-1) * (1+cos(β))/2 * H_np1[n+1+m-1+1] - a(n,m)*sin(β)*H_np1[n+1+m+1]) / b(n+1,0), n, 1, m)
    end

    # compute H_n_mp+1_m
    for mp in 1:n-1
        for m in mp+1:n-1
            val = d(n,mp-1) * access(H,n,mp-1,m) - d(n,m-1)*access(H,n,mp,m-1) + d(n,m)*access(H,n,mp,m+1)
            set!(H, val/d(n,mp), n, mp+1, m)
        end
        m = n
        val = d(n,mp-1) * access(H,n,mp-1,m) - d(n,m-1)*access(H,n,mp,m-1)
        set!(H, val/d(n,mp), n, mp+1, m)
    end

    # compute H_n_mp-1_m
    for mp in 0:-1:-n+1
        for m in -mp+1:n-1
            val = d(n,mp) * access(H,n,mp+1,m) + d(n,m-1)*access(H,n,mp,m-1) - d(n,m)*access(H,n,mp,m+1)
            set!(H, val/d(n,mp-1), n, mp-1, m)
        end
        m = n
        val = d(n,mp) * access(H,n,mp+1,m) + d(n,m-1) * access(H,n,mp,m-1)
        set!(H,val/d(n,mp-1),n,mp-1,m)
    end

    # leverage symmetry
    for m in 1:n
        for mp in -m:m-1
            set!(H, access(H,n,mp,m), n, m, mp)
        end
    end

    for m in -n:n
        for mp in -m+1:n
            set!(H, access(H,n,mp,m), n, -m, -mp)
        end
    end

    return H
end

function build_Hs(p, β, TF=Complex{Float64})
    H0 = TF[1.0;;]
    Hs = [H0]
    for n in 1:p
        push!(Hs, build_H(Hs[end], β))
    end
    return Hs
end

const p_const = 10
const Hs_pi2 = build_Hs(p_const, pi/2)

function build_Hs_flip(p, β)
    Hs = build_Hs(p, pi/2)
    for n in 0:p
        Hpi2 = Hs_pi2[n+1]
        H = Hs[n+1]
        for m in -n:n
            for mp in -n:n
                val = 0.0
                for ν in 1:n
                    val += real(access(Hpi2, n, mp, ν)) * real(access(Hpi2, n, m, ν)) * cos(ν*β + pi/2*(mp+m))
                end
                val *= 2
                val += real(access(Hpi2, n, mp, 0)) * real(access(Hpi2, n, m, 0)) * cos(pi/2*(mp+m))
                set!(H, val, n, mp, m)
            end
        end
    end
    return Hs
end

function build_ds(p, β, TF=Float64)
    d0 = TF[1.0;;]
    ds = [d0]
    for n in 1:p
        push!(ds, build_d(ds[end], β))
    end
    return ds
end

function H2Ry(Hs)
    Rys = deepcopy(Hs)
    for Ry in Rys
        n = (size(Ry,1)-1) >> 1
        for mp in -n:n
            for m in -n:n
                set!(Ry, access(Ry,n,mp,m) * (-1)^mp, n, mp, m)
            end
        end
    end
    return Rys
end

function rotate_y(harmonics, β)
    transformed_harmonics = deepcopy(harmonics)
    Hs = build_Hs(harmonics.p, β)
    Hs = H2Ry(Hs)
    # Hs = H2d(Hs)
    for (H,h) in zip(Hs,transformed_harmonics.weights)
        hp = H * h
        h .= hp
    end
    return transformed_harmonics
end

function build_Ts(p, α, β, γ)
    Ts = build_Hs(p, β)
    for T in Ts
        n = (size(T,1)-1) >> 1
        for m in -n:n
            for mp in -n:n
                set!(T, exp(-im*mp*γ) * access(T,n,mp,m) * exp(im*m*α), n, mp, m)
            end
        end
    end
    return Ts
end

const DEBUG = Vector{Bool}(undef,1)
DEBUG[] = false

function build_Ts_flip(p, α, β, γ)
    Ts = build_Hs_flip(p, β)
    for T in Ts
        n = (size(T,1)-1) >> 1
        for m in -n:n
            for mp in -n:n
                set!(T, exp(-im*mp*γ) * access(T,n,mp,m) * exp(im*m*α), n, mp, m)
            end
        end
    end
    return Ts
end

function rotate(harmonics, α, β, γ)
    @assert 0 <= β <= π

    p = harmonics.p
    Ts = build_Ts(p, α, β, γ)
    rotated_harmonics = deepcopy(harmonics)

    # rotate
    for (h, T) in zip(rotated_harmonics.weights, Ts)
        rotated_h = T * h
        h .= rotated_h
    end

    return rotated_harmonics
end

function inverse_rotate(harmonics, α, β, γ)
    # ensure 0 < β < π
    @assert 0 <= β <= π

    p = harmonics.p
    Ts = build_Ts(p, α, β, γ)
    rotated_harmonics = deepcopy(harmonics)

    # rotate
    for (h, T) in zip(rotated_harmonics.weights, Ts)
        rotated_h = T' * h
        h .= rotated_h
    end

    return rotated_harmonics
end
function rotate_flip(harmonics, α, β, γ)
    p = harmonics.p
    Ts = build_Ts_flip(p, α, β, γ)
    rotated_harmonics = deepcopy(harmonics)

    # rotate
    for (h, T) in zip(rotated_harmonics.weights, Ts)
        rotated_h = T * h
        h .= rotated_h
    end

    return rotated_harmonics
end

function rotate_euler_flip(harmonics, αe, βe, γe)
    return rotate_flip(harmonics, αe, βe, π - γe)
end

function rotate_euler(harmonics, αe, βe, γe)
    return rotate(harmonics, αe, βe, π - γe)
end

function inverse_rotate_euler(harmonics, αe, βe, γe)
    return inverse_rotate(harmonics, αe, βe, π - γe)
end

function ε(m)
    if m >= 0
        return (-1)^m
    else
        return 1
    end
end

function H2d(H::Matrix)
    n = (size(H,1)-1) >> 1
    d = deepcopy(H)
    for m in -n:n
        for mp in -n:n
            val = access(d,n,mp,m) * ε(mp) * ε(-m)
            set!(d,val,n,mp,m)
        end
    end
    return d
end

function H2d(Hs::Vector)
    ds = similar(Hs)
    for i in eachindex(ds)
        ds[i] = H2d(Hs[i])
    end
    return ds
end

function test_H(n, β)
    H = zeros(2*n+1,2*n+1)
    for mp in 0:n
        val = ε(mp) * Float64(sqrt(factorial(big(2*n)) / factorial(big(n-mp)) / factorial(big(n+mp)))) * cos(β/2)^(n+mp) * sin(β/2)^(n-mp)
        set!(H, val, n, mp, n)
    end
    return H
end

function testall()
    #--- test z axis rotations
    # original harmonics
    p = 3
    seed = 123
    harmonics = SphericalHarmonics(seed, p)

    # rotate
    Δϕ = 3*pi/8
    rotated_harmonics = rotate_z(harmonics, Δϕ)

    # test
    θ, ϕ = pi/7, pi/5*9
    v_unrotated = evaluate(harmonics, θ, ϕ)
    v_rotated = evaluate(rotated_harmonics, θ, ϕ-Δϕ)

    @assert isapprox(v_unrotated, v_rotated)
    # it works!

    # test
    p = 4
    β = 0.0
    ds = build_ds(p, β)
    Hs = build_Hs(p, β)
    ds2 = H2d(Hs)
    Htest = test_H(1, β)

    #--- test y axis rotations
    # original harmonics
    p = 1
    seed = 123
    harmonics = SphericalHarmonics(seed, p)
    Δϕ_z = pi/6
    harmonics_z = rotate_z(harmonics, Δϕ_z)

    save_vtk("unrotated.vts", harmonics)
    save_vtk("rotated_z.vts", harmonics_z)

    # rotate
    Δθ = pi/2
    rotated_harmonics = rotate_y(harmonics, Δθ)

    save_vtk("rotated_y.vts", rotated_harmonics)

    # test
    ϕ = 0.0
    θs = range(0,pi,9)
    vs_unrotated = zeros(Complex{Float64}, length(θs))
    for i in eachindex(vs_unrotated)
        vs_unrotated[i] = evaluate(harmonics, θs[i] + Δθ, ϕ)
    end

    vs_rotated = zeros(Complex{Float64}, length(θs))
    for i in eachindex(vs_rotated)
        vs_rotated[i] = evaluate(rotated_harmonics, θs[i], ϕ)
    end

    for i in 1:length(θs)>>1+1
        @assert isapprox(vs_unrotated[i], vs_rotated[i]; atol=1e-12)
    end

    #--- test y axis rotations: general ---#
    rotated_harmonics_general = rotate_euler(harmonics, 0.0, Δθ, 0.0)
    save_vtk("rotated_y_general.vts", rotated_harmonics_general)

    for i in 0:p
        @assert isapprox(rotated_harmonics_general.weights[i+1], rotated_harmonics.weights[i+1]; atol=1e-12)
    end

    #--- test z axis rotations: general ---#
    rotated_harmonics_z_general = rotate_euler(harmonics, 0.0, 0.0, Δϕ_z)

    save_vtk("rotated_z_general.vts", rotated_harmonics_z_general)

    for i in 0:p
        @assert isapprox(rotated_harmonics_z_general.weights[i+1], harmonics_z.weights[i+1]; atol=1e-12)
    end

    rotated_harmonics_z_general_2 = rotate_euler(harmonics, Δϕ_z, 0.0, 0.0)

    for i in 0:p
        @assert isapprox(rotated_harmonics_z_general_2.weights[i+1], harmonics_z.weights[i+1]; atol=1e-12)
    end

    # coordinate flips
    rotated_harmonics_general_flip = rotate_euler_flip(harmonics, 0.0, Δθ, 0.0)
    for i in 0:p
        @assert isapprox(rotated_harmonics_general_flip.weights[i+1], rotated_harmonics.weights[i+1]; atol=1e-12)
    end

    rotated_harmonics_z_general_flip = rotate_euler_flip(harmonics, 0.0, 0.0, Δϕ_z)
    for i in 0:p
        @assert isapprox(rotated_harmonics_z_general_flip.weights[i+1], harmonics_z.weights[i+1]; atol=1e-12)
    end

    rotated_harmonics_z_general_flip_2 = rotate_euler_flip(harmonics, Δϕ_z, 0.0, 0.0)
    for i in 0:p
        @assert isapprox(rotated_harmonics_z_general_flip_2.weights[i+1], harmonics_z.weights[i+1]; atol=1e-12)
    end

    #--- test inverse rotation ---#

    α, β, γ = 3*pi/8, pi/7, pi/9

    # first rotation
    rotated_1 = rotate(harmonics, α, β, γ)

    # inverse rotation
    rotated_2 = inverse_rotate(rotated_1, α, β, γ)

    # another inverse rotation
    rotated_3 = rotate(rotated_1, γ, β, α)

    for i in 0:p
        @assert isapprox(rotated_2.weights[i+1], harmonics.weights[i+1]; atol=1e-12)
        @assert isapprox(rotated_3.weights[i+1], harmonics.weights[i+1]; atol=1e-12)
    end

    # check how to obtain inverse rotation
    Ts_forward = build_Ts_flip(p, α, β, γ)
    Ts_backward = build_Ts_flip(p, γ, β, α)
    for i in 0:p
        @assert isapprox(Ts_forward[i+1]', Ts_backward[i+1]; atol=1e-12)
    end
end

#testall()

Ts = build_Ts_flip(7, 0.0, 3*pi/7, 0.0)
println("done.")
