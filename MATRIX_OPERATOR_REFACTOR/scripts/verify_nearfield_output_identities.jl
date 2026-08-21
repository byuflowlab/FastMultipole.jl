#!/usr/bin/env julia

# Task 041b: production-independent verification of the reduced-output
# identities against analytic vortex-pair Jacobians.  Base/stdlib only.

using LinearAlgebra
using Random
using Printf

Random.seed!(0x041b)

# A&S 7.1.26, matching the standalone rank probe.  Its approximation error
# affects the radial coefficient, not the analytic zero-trace cancellation.
function erf_local(x::T) where T
    s = sign(x); z = abs(x); t = inv(one(T) + T(0.3275911) * z)
    p = (((((T(1.061405429)*t - T(1.453152027))*t + T(1.421413741))*t -
          T(0.284496736))*t + T(0.254829592))*t)
    s * (one(T) - p * exp(-z*z))
end

function radial_coefficients(r2::T, sigma::T, kernel::Symbol) where T
    c = inv(T(4) * T(pi))
    r = sqrt(r2)
    if kernel === :singular
        g = c / r^3
        dgdr2 = -T(3) * c / (T(2) * r^5)
    elseif kernel === :gaussianerf
        a = r / sigma
        n = erf_local(a) - T(2) * a / sqrt(T(pi)) * exp(-a*a)
        dnda = T(4) * a*a / sqrt(T(pi)) * exp(-a*a)
        g = c * n / r^3
        dgdr2 = c * (dnda / (sigma*r^3) - T(3)*n/r^4) / (T(2)*r)
    else
        error("unknown kernel $kernel")
    end
    return g, dgdr2
end

function pair_jacobian(x::Vector{T}, y::Vector{T}, gamma::Vector{T}, sigma::T,
                       kernel::Symbol) where T
    r = x - y
    r2 = dot(r, r)
    iszero(r2) && return zeros(T, 3, 3)
    g, dgdr2 = radial_coefficients(r2, sigma, kernel)
    w = cross(gamma, r)
    # d[g(r^2) gamma x r]/dx_j
    C = T[0 -gamma[3] gamma[2]; gamma[3] 0 -gamma[1]; -gamma[2] gamma[1] 0]
    T(2) * dgdr2 .* (w * transpose(r)) + g .* C
end

function pair_contraction(x::Vector{T}, y::Vector{T}, gamma_source::Vector{T},
                          sigma::T, kernel::Symbol, v::Vector{T},
                          transposed::Bool) where T
    r=x-y; r2=dot(r,r)
    iszero(r2) && return zeros(T,3)
    g,dgdr2=radial_coefficients(r2,sigma,kernel)
    w=cross(gamma_source,r)
    transposed ? T(2)*dgdr2.*w.*dot(r,v) .+ g.*cross(gamma_source,v) :
                 T(2)*dgdr2.*r.*dot(w,v) .- g.*cross(gamma_source,v)
end

op(J, v, transposed) = transposed ? J * v : transpose(J) * v

function field_jacobians(x, gamma, sigma, kernel)
    [sum((pair_jacobian(x[p], x[q], gamma[q], sigma[q], kernel)
          for q in eachindex(x)); init=zeros(eltype(first(x)), 3, 3))
     for p in eachindex(x)]
end

zeta_weight(xp, xq, sigmaq) =
    exp(-dot(xp-xq, xp-xq) / (2sigmaq^2)) / sigmaq^3

function sfs_direct(x, gamma, sigma, J, transposed, active)
    T = eltype(first(x)); E = [fill(T(NaN), 3) for _ in eachindex(x)]
    for p in eachindex(x)
        active[p] || continue
        E[p] = sum((zeta_weight(x[p], x[q], sigma[q]) .*
                    (op(J[p], gamma[q], transposed) -
                     op(J[q], gamma[q], transposed)) for q in eachindex(x));
                   init=zeros(T, 3))
    end
    E
end

function sfs_factored(x, gamma, sigma, J, transposed, active)
    T = eltype(first(x)); E = [fill(T(NaN), 3) for _ in eachindex(x)]
    for p in eachindex(x)
        active[p] || continue
        omega = sum((zeta_weight(x[p], x[q], sigma[q]) .* gamma[q]
                     for q in eachindex(x)); init=zeros(T, 3))
        Q = sum((zeta_weight(x[p], x[q], sigma[q]) .*
                 op(J[q], gamma[q], transposed) for q in eachindex(x));
                init=zeros(T, 3))
        E[p] = op(J[p], omega, transposed) - Q
    end
    E
end

function dynamic_values(stretch_test, stretch_domain, E_test, E_domain, gamma,
                        sigma; alpha=1.5, relax=0.63, old_num=0.17, old_den=0.91)
    Mstretch = stretch_test - stretch_domain
    ME = E_test - E_domain
    numerator = (3alpha - 2) * dot(Mstretch, gamma)
    denominator = dot(ME, gamma) * sigma^3
    numerator = relax*numerator + (1-relax)*old_num
    denominator = relax*denominator + (1-relax)*old_den
    return numerator, denominator, numerator/denominator
end

function trial(::Type{T}, transposed::Bool) where T
    n = 19
    # Unequal sigma, a realistic clustered/elongated snapshot, one coincident
    # self pair per particle, and one static target that remains untouched.
    x = [T[randn(T)/5, randn(T)/8, T(2)*(p-1)/(n-1)] for p in 1:n]
    gamma = [randn(T, 3) for _ in 1:n]
    sigma = [T(0.18) * (T(1) + T(2)*(p-1)/(n-1)) for p in 1:n]
    active = trues(n); active[4] = false

    Jreg = field_jacobians(x, gamma, sigma, :gaussianerf)
    Jsing = field_jacobians(x, gamma, sigma, :singular)
    trace_reg = maximum(abs(tr(Jreg[p])) for p in eachindex(x))
    trace_sing = maximum(abs(tr(Jsing[p])) for p in eachindex(x))

    # A distant-source sum exercises the analytic far-field evaluation limit.
    farx = [T[0,0,0], T[7,3,-5], T[-6,4,8]]
    farg = [randn(T,3) for _ in farx]
    fars = fill(T(0.2), length(farx))
    trace_far = maximum(abs(tr(J)) for J in field_jacobians(farx, farg, fars, :singular))

    direct = sfs_direct(x, gamma, sigma, Jreg, transposed, active)
    factored = sfs_factored(x, gamma, sigma, Jreg, transposed, active)
    ids = findall(active)
    rel = norm(reduce(vcat, direct[ids]) - reduce(vcat, factored[ids])) /
          max(norm(reduce(vcat, direct[ids])), eps(T))
    @assert all(isnan, direct[4]) && all(isnan, factored[4])

    contraction = maximum(norm(op(Jreg[p], gamma[p], transposed) -
        sum((pair_contraction(x[p],x[q],gamma[q],sigma[q],:gaussianerf,
                              gamma[p],transposed) for q in eachindex(x));
            init=zeros(T,3))) for p in ids)

    # Two filter widths, then the exact pseudo-three-level numerator,
    # denominator, and coefficient reduction used by FLOWVPM.
    sigma_test = T(1.5) .* sigma
    Jtest = field_jacobians(x, gamma, sigma_test, :gaussianerf)
    Ed = direct
    Ef = factored
    Etest_d = sfs_direct(x, gamma, sigma_test, Jtest, transposed, active)
    Etest_f = sfs_factored(x, gamma, sigma_test, Jtest, transposed, active)
    dynerr = zero(T)
    for p in ids
        sd = op(Jreg[p], gamma[p], transposed)
        st = op(Jtest[p], gamma[p], transposed)
        vd = dynamic_values(st, sd, Etest_d[p], Ed[p], gamma[p], sigma[p])
        vf = dynamic_values(st, sd, Etest_f[p], Ef[p], gamma[p], sigma[p])
        delta = maximum(abs.(collect(vd) .- collect(vf)))
        dynscale = max(maximum(abs.(collect(vd))), eps(T))
        dynerr = max(dynerr, delta / dynscale)
    end
    return rel, contraction, trace_sing, trace_reg, trace_far, dynerr
end

println("type,transposed,sfs_relative_error,contraction_absolute_error,trace_singular,trace_gaussianerf,trace_farfield,dynamic_value_error")
for T in (Float32, Float64, BigFloat), transposed in (false, true)
    vals = trial(T, transposed)
    @printf("%s,%s,%s\n", T, transposed, join((@sprintf("%.9e", Float64(v)) for v in vals), ','))
    scale_tol = T == Float32 ? 5e-5 : T == Float64 ? 2e-13 : big"1e-55"
    @assert vals[1] <= scale_tol
    @assert vals[2] <= scale_tol
    @assert maximum(vals[3:6]) <= scale_tol
end
