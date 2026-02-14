struct SourcePoints{TF}
    x::Vector{SVector{3,TF}}
    strength::Vector{TF}
    potential::Vector{TF}
    gradient::Vector{SVector{3,TF}}
    hessian::Vector{SMatrix{3,3,TF,9}}
end

function SourcePoints(x, strength::Vector{TF};
        potential = zeros(length(strength)),
        gradient = zeros(SVector{3,TF},size(x,2)),
        hessian = zeros(SMatrix{3,3,TF,9},size(x,2))
    ) where TF

    return SourcePoints(x, strength, potential, gradient, hessian)
end

Base.eltype(::SourcePoints{TF}) where TF = TF

function FastMultipole.source_system_to_buffer!(buffer, i_buffer, system::SourcePoints, i_body)
    buffer[1:3,i_buffer] .= system.x[i_body]
    buffer[5,i_buffer] = system.strength[i_body]
end

FastMultipole.data_per_body(::SourcePoints) = 5

function FastMultipole.reset!(system::SourcePoints{TF}) where TF
    system.potential .= zero(TF)
    system.gradient .= zero(eltype(system.gradient))
    system.hessian .= zero(eltype(system.hessian))
end

FastMultipole.strength_dims(::SourcePoints) = 1

FastMultipole.get_n_bodies(system::SourcePoints) = length(system.strength)

FastMultipole.get_position(system::SourcePoints,i) = system.x[i]

struct SourceFilaments{TF}
    x::Matrix{SVector{3,TF}}
    strength::Vector{TF}
    potential::Vector{TF}
    gradient::Vector{SVector{3,TF}}
    hessian::Vector{SMatrix{3,3,TF,9}}
end

function SourceFilaments(x, strength::Vector{TF};
        potential = zeros(size(x,2)),
        gradient = zeros(SVector{3,TF},size(x,2)),
        hessian = zeros(SMatrix{3,3,TF,9},size(x,2))
    ) where TF

    return SourceFilaments(x, strength, potential, gradient, hessian)
end

Base.eltype(::SourceFilaments{TF}) where TF = TF

function FastMultipole.source_system_to_buffer!(buffer, i_buffer, system::SourceFilaments, i_body)
    buffer[1:3,i_buffer] .= (system.x[1,i_body] + system.x[2,i_body]) * 0.5
    buffer[4,i_buffer] = norm(system.x[2,i_body] - system.x[1,i_body])
    buffer[5,i_buffer] = system.strength[i_body]
    buffer[6:8,i_buffer] .= system.x[1,i_body]
    buffer[9:11,i_buffer] .= system.x[2,i_body]
end

FastMultipole.data_per_body(::SourceFilaments) = 11

function FastMultipole.reset!(system::SourceFilaments{TF}) where TF
    system.potential .= zero(TF)
    system.gradient .= zero(eltype(system.gradient))
    system.hessian .= zero(eltype(system.hessian))
end

FastMultipole.strength_dims(::SourceFilaments) = 1

FastMultipole.get_n_bodies(system::SourceFilaments) = length(system.strength)

FastMultipole.get_position(system::SourceFilaments,i) = (system.x[1,i]+system.x[2,i]) * 0.5

struct SourcePanels{TF}
    x::Vector{SVector{3,SVector{3,TF}}} # vector of groups of 3 vertices
    strength::Vector{TF}
    potential::Vector{TF}
    gradient::Vector{SVector{3,TF}}
    hessian::Vector{SMatrix{3,3,TF,9}}
end

function SourcePanels(x, strength::Vector{TF};
        potential = zeros(length(x)),
        gradient = zeros(SVector{3,TF},size(x,2)),
        hessian = zeros(SMatrix{3,3,TF,9},size(x,2))
    ) where TF

    return SourcePanels(x, strength, potential, gradient, hessian)
end

function FastMultipole.source_system_to_buffer!(buffer, i_buffer, system::SourcePanels, i_body)
    buffer[1:3,i_buffer] .= (system.x[i_body][1] + system.x[i_body][2] + system.x[i_body][3]) * 0.33333333333333333
    # buffer[4,i_buffer] =
    buffer[5,i_buffer] = system.strength[i_body]
    buffer[6:8,i_buffer] .= system.x[i_body][1]
    buffer[9:11,i_buffer] .= system.x[i_body][2]
    buffer[12:14,i_buffer] .= system.x[i_body][3]
end

FastMultipole.data_per_body(::SourcePanels) = 16

function FastMultipole.reset!(system::SourcePanels{TF}) where TF
    system.potential .= zero(TF)
    system.gradient .= zero(eltype(system.gradient))
    system.hessian .= zero(eltype(system.hessian))
end

FastMultipole.strength_dims(::SourcePanels) = 1

FastMultipole.get_n_bodies(system::SourcePanels) = length(system.strength)

FastMultipole.get_position(system::SourcePanels,i) = (system.x[i][1]+system.x[i][2] + system.x[i][3]) * 0.33333333333333333

Base.eltype(::SourcePanels{TF}) where TF = TF

struct DipolePoints{TF}
    x::Vector{SVector{3,TF}}
    strength::Vector{SVector{3,TF}}
    potential::Vector{TF}
    gradient::Vector{SVector{3,TF}}
    hessian::Vector{SMatrix{3,3,TF,9}}
end

function DipolePoints(x, strength::Vector{SVector{3,TF}};
        potential = zeros(length(strength)),
        gradient = zeros(SVector{3,TF},size(x,2)),
        hessian = zeros(SMatrix{3,3,TF,9},size(x,2))
    ) where TF

    return DipolePoints(x, strength, potential, gradient, hessian)
end

function FastMultipole.source_system_to_buffer!(buffer, i_buffer, system::DipolePoints, i_body)
    buffer[1:3,i_body] .= system.x[i_body]
    buffer[5:7,i_body] .= system.strength[i_body]
end

function FastMultipole.data_per_body(system::DipolePoints)
    return 7
end

function Base.eltype(system::DipolePoints{TF}) where TF
    return TF
end

function FastMultipole.reset!(system::DipolePoints)
    system.potential .= zero(eltype(system))
    system.gradient .= zero(eltype(system.gradient))
    system.hessian .= zero(eltype(system.hessian))
end

function FastMultipole.get_position(system::DipolePoints, i)
    return system.x[i]
end

function FastMultipole.get_n_bodies(system::DipolePoints)
    return length(system.x)
end

function FastMultipole.strength_dims(system::DipolePoints)
    return 3
end

struct DipoleFilaments{TF}
    x::Matrix{SVector{3,TF}}
    strength::Vector{SVector{3,TF}}
    potential::Vector{TF}
    gradient::Vector{SVector{3,TF}}
    hessian::Vector{SMatrix{3,3,TF,9}}
end

function DipoleFilaments(x, strength::Vector{SVector{3,TF}};
        potential = zeros(size(x,2)),
        gradient = zeros(SVector{3,TF},size(x,2)),
        hessian = zeros(SMatrix{3,3,TF,9},size(x,2))
    ) where TF

    return DipoleFilaments(x, strength, potential, gradient, hessian)
end

function FastMultipole.source_system_to_buffer!(buffer, i_buffer, system::DipoleFilaments, i_body)
    buffer[1:3,i_body] .= (system.x[1,i_body] + system.x[2,i_body]) * 0.5
    buffer[4,i_body] = norm(system.x[2,i_body] - system.x[1,i_body])
    buffer[5:7,i_body] .= system.strength[i_body]
    buffer[8:10,i_body] .= system.x[1,i_body]
    buffer[11:13,i_body] .= system.x[2,i_body]
end

function FastMultipole.data_per_body(system::DipoleFilaments)
    return 13
end

function Base.eltype(system::DipoleFilaments{TF}) where TF
    return TF
end

function FastMultipole.reset!(system::DipoleFilaments)
    system.potential .= zero(eltype(system))
    system.gradient .= zero(eltype(system.gradient))
    system.hessian .= zero(eltype(system.hessian))
end

function FastMultipole.get_position(system::DipoleFilaments, i)
    return (system.x[1,i] + system.x[2,i]) * 0.5
end

function FastMultipole.get_n_bodies(system::DipoleFilaments)
    return length(system.strength)
end

function FastMultipole.strength_dims(system::DipoleFilaments)
    return 3
end

struct DipolePanels{TF}
    x::Vector{SVector{3,SVector{3,TF}}} # vector of groups of 3 vertices
    strength::Vector{SVector{1,TF}}
    potential::Vector{TF}
    gradient::Vector{SVector{3,TF}}
    hessian::Vector{SMatrix{3,3,TF,9}}
end

function DipolePanels(x, strength::Vector{SVector{1,TF}};
        potential = zeros(length(x)),
        gradient = zeros(SVector{3,TF},size(x,2)),
        hessian = zeros(SMatrix{3,3,TF,9},size(x,2))
    ) where TF

    return DipolePanels(x, strength, potential, gradient, hessian)
end

function FastMultipole.source_system_to_buffer!(buffer, i_buffer, system::DipolePanels, i_body)
    buffer[1:3,i_body] .= (system.x[i_body][1] + system.x[i_body][2] + system.x[i_body][3]) * 0.33333333333333333
    # buffer[4,i_body] =
    buffer[5,i_body] = system.strength[i_body][1]
    buffer[6:8,i_body] .= system.x[i_body][1]
    buffer[9:11,i_body] .= system.x[i_body][2]
    buffer[12:14,i_body] .= system.x[i_body][3]
end

function FastMultipole.data_per_body(system::DipolePanels)
    return 14
end

function Base.eltype(::DipolePanels{TF}) where TF
    return TF
end

function FastMultipole.reset!(system::DipolePanels)
    system.potential .= zero(eltype(system))
    system.gradient .= zero(eltype(system.gradient))
    system.hessian .= zero(eltype(system.hessian))
end

function FastMultipole.get_position(system::DipolePanels, i)
    return (system.x[i][1] + system.x[i][2] + system.x[i][3]) * 0.3333333333333333333
end

function FastMultipole.get_n_bodies(system::DipolePanels)
    return length(system.strength)
end

function FastMultipole.strength_dims(system::DipolePanels)
    return 1
end

struct Vortons{TF}
    position::Vector{SVector{3,TF}}
    strength::Vector{SVector{3,TF}}
    potential::Vector{TF}
    gradient::Vector{SVector{3,TF}}
    hessian::Vector{SMatrix{3,3,TF,9}}
end

function Vortons(position::Vector{SVector{3,TF}}, strength;
        potential = zeros(length(position)),
        gradient = zeros(SVector{3,TF},length(position)),
        hessian = zeros(SMatrix{3,3,TF,9},length(position))
    ) where TF

    return Vortons(position, strength, potential, gradient, hessian)
end

function FastMultipole.source_system_to_buffer!(buffer, i_buffer, system::Vortons, i_body)
    buffer[1:3,i_buffer] .= system.position[i_body]
    buffer[5:7,i_buffer] .= system.strength[i_body]
end

function FastMultipole.strength_dims(system::Vortons)
    return 3
end

FastMultipole.get_n_bodies(system::Vortons) = length(system.position)

FastMultipole.data_per_body(::Vortons) = 7

Base.eltype(::Vortons{TF}) where TF = TF

FastMultipole.get_position(system::Vortons, i) = system.position[i]

struct VortexPanels{TF}
    x::Vector{SVector{3,SVector{3,TF}}} # vector of groups of 3 vertices
    strength::Vector{SVector{3,TF}}
    potential::Vector{TF}
    gradient::Vector{SVector{3,TF}}
    hessian::Vector{SMatrix{3,3,TF,9}}
end

function VortexPanels(x, strength::Vector{SVector{3,TF}};
        potential = zeros(length(x)),
        gradient = zeros(SVector{3,TF},size(x,2)),
        hessian = zeros(SMatrix{3,3,TF,9},size(x,2))
    ) where TF

    return VortexPanels(x, strength, potential, gradient, hessian)
end

function FastMultipole.source_system_to_buffer!(buffer, i_buffer, system::VortexPanels, i_body)
    buffer[1:3,i_buffer] .= (system.x[i_body][1] + system.x[i_body][2] + system.x[i_body][3]) * 0.33333333333333333
    # buffer[4,i_buffer] =
    buffer[5:7,i_buffer] .= system.strength[i_body]
    buffer[8:10,i_buffer] .= system.x[i_body][1]
    buffer[11:13,i_buffer] .= system.x[i_body][2]
    buffer[14:16,i_buffer] .= system.x[i_body][3]
end

function FastMultipole.strength_dims(system::VortexPanels)
    return 3
end

FastMultipole.get_n_bodies(system::VortexPanels) = length(system.strength)

FastMultipole.data_per_body(::VortexPanels) = 16

Base.eltype(::VortexPanels{TF}) where TF = TF

FastMultipole.get_position(system::VortexPanels, i) = (system.x[i][1] + system.x[i][2] + system.x[i][3]) * 0.33333333333333333

function FastMultipole.reset!(system::VortexPanels)
    system.potential .= zero(eltype(system.potential))
    system.gradient .= zero(eltype(system.gradient))
    system.hessian .= zero(eltype(system.hessian))
end

function Branch(bodies_index::UnitRange, n_branches, branch_index, i_parent, i_leaf_index, center, radius, box)
    return FastMultipole.Branch(SVector{1}(length(bodies_index)), SVector{1}([bodies_index]), n_branches, branch_index, i_parent, i_leaf_index, center, radius, box)
end

@testset "body-to-multipole: point source" begin

x = SVector{3}(0.1,0.2,-0.3)
xs = x + SVector{3}(-0.2,0.07,-0.1)
bodies = zeros(5,1)
bodies[:,1] .= [xs[1], xs[2], xs[3], 0.0, 0.7]
masses = Gravitational(bodies)
buffer = FastMultipole.system_to_buffer(masses)
expansion_order = 10
box = SVector{3}(0.0,0.0,0.0)
branch = Branch(1:1, 0, 1:0, 0, 1, x, 0.0, box)
harmonics = initialize_harmonics(expansion_order)
multipole_expansion = initialize_expansion(expansion_order)

body_to_multipole!(Point{Source}, masses, multipole_expansion, buffer, branch.center, 1:1, harmonics, expansion_order)

Δx = SVector{3}(bodies[1:3]) - branch.center
ρ, θ, ϕ = FastMultipole.cartesian_to_spherical(Δx...)

# check against known values
expansion_order_check = 10
for n in 0:expansion_order_check
    for m in -n:-1
        i = FastMultipole.harmonic_index(n,-m)
        Mnm = (multipole_expansion[1,1,i] - im*multipole_expansion[2,1,i]) * (-1)^m
        local Rnm = (-1)^n * im^abs(m) * ρ^n * Plm(cos(θ),n,abs(m)) * exp(im*m*ϕ) / factorial(n+abs(m))
        Mnm_check = (-1)^(n+m) * bodies[5,1] * conj(Rnm)
        @test isapprox(Mnm, -Mnm_check; atol=1e-12)
    end
    for m in 0:n
        i = FastMultipole.harmonic_index(n,m)
        Mnm = multipole_expansion[1,1,i] + im*multipole_expansion[2,1,i]
        local Rnm = (-1)^n * im^abs(m) * ρ^n * Plm(cos(θ),n,abs(m)) * exp(im*m*ϕ) / factorial(n+abs(m))
        Mnm_check = (-1)^(n+m) * bodies[5,1] * conj(Rnm)
        @test isapprox(Mnm, -Mnm_check; atol=1e-12)
    end
end

#--- evaluate multipole expansion ---#

# evaluate the multipole expansion
x_target = SVector{3}(2.3,-4.1, 0.4)
r = x_target - xs
rnorm = sqrt(r'*r)
ϕ_analytic = masses.bodies[1].strength/rnorm/4/pi
v_analytic = r/rnorm^3/4/pi * masses.bodies[1].strength
rx, ry, rz = r
g_analytic = -masses.bodies[1].strength .* [
   3*rx^2/rnorm^5 - 1/rnorm^3 3*rx*ry/rnorm^5 3*rx*rz/rnorm^5;
   3*rx*ry/rnorm^5 3*ry^2/rnorm^5 - 1/rnorm^3 3*ry*rz/rnorm^5;
   3*rx*rz/rnorm^5 3*ry*rz/rnorm^5 3*rz^2/rnorm^5 - 1/rnorm^3
  ] ./ (4*pi)
ϕ_m2b, v_m2b, g_m2b = evaluate_multipole(x_target, branch.center, multipole_expansion, DerivativesSwitch(), expansion_order)

@test isapprox(ϕ_m2b, ϕ_analytic; atol=1e-12)
@test isapprox(v_m2b, v_analytic; atol=1e-12)
@test isapprox(g_m2b, g_analytic; atol=1e-12)

#--- evaluate local expansion ---#

target_center = x_target + SVector{3}(0.05, -0.1, -0.2)
box = SVector{3}(0.0,0.0,0.0)
target_branch = Branch(1:1, 0, 1:0, 0, 1, target_center, 0.0, box)
target_expansion = initialize_expansion(expansion_order)

# preallocate containers
Hs_π2 = [1.0]
FastMultipole.update_Hs_π2!(Hs_π2, expansion_order)
Ts = zeros(FastMultipole.length_Ts(expansion_order))
eimϕs = zeros(2, expansion_order+2)
weights_tmp_1 = initialize_expansion(expansion_order, eltype(Ts))
weights_tmp_2 = initialize_expansion(expansion_order, eltype(Ts))
weights_tmp_3 = initialize_expansion(expansion_order, eltype(Ts))

# normalization
ζs_mag = zeros(FastMultipole.length_ζs(expansion_order))
FastMultipole.update_ζs_mag!(ζs_mag, 0, expansion_order)
ηs_mag = zeros(FastMultipole.length_ηs(expansion_order))
FastMultipole.update_ηs_mag!(ηs_mag, 0, expansion_order)

lamb_helmholtz = Val(false)
FastMultipole.multipole_to_local!(target_expansion, target_branch, multipole_expansion, branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, FastMultipole.M̃, FastMultipole.L̃, expansion_order, lamb_helmholtz)

gradient_n_m = FastMultipole.initialize_gradient_n_m(expansion_order)
ϕ_l2b, v_l2b, g_l2b = FastMultipole.evaluate_local(x_target - target_center, harmonics, gradient_n_m, target_expansion, expansion_order, lamb_helmholtz, DerivativesSwitch())

@test isapprox(v_l2b, v_analytic; atol=1e-12)

# compute local coefficients directly
target_branch_2 = deepcopy(target_branch)
target_expansion_2 = initialize_expansion(expansion_order)
body_to_local_point!(Point{Source}, target_expansion_2, harmonics, masses.bodies[1].position - target_branch_2.center, masses.bodies[1].strength, expansion_order)

gradient_n_m .= 0.0
ϕ_l2b_2, v_l2b_2, g_l2b_2 = FastMultipole.evaluate_local(x_target - target_center, harmonics, gradient_n_m, target_expansion_2, expansion_order, lamb_helmholtz, DerivativesSwitch())

@test isapprox(v_l2b_2, v_analytic; atol=1e-12)

end

@testset "body-to-multipole: point dipole" begin

xs = SVector{3}(0.5,-0.3,0.45)
xt = SVector{3}(7.3,8.4,-7.2)
q = SVector{3}(0.7,0.3,1.5)
center = SVector{3}(0.0,0,0)
radius = sqrt(xs'*xs)
expansion_order = 10

multipole_check = ComplexF64[0.0 + 0.0im, 0.15 - 0.35im, 1.5 + 0.0im, -0.15 - 0.35im, -0.11 + 0.015000000000000006im, -0.15750000000000003 - 0.5325im, 0.5449999999999999 - 0.0im, 0.15750000000000003 - 0.5325im, -0.11 - 0.015000000000000006im, 0.010124999999999997 + 0.012624999999999997im, -0.07949999999999999 + 0.06299999999999999im, -0.08268750000000004 - 0.1730625im, -0.03412500000000001 + 0.0im, 0.08268750000000004 - 0.1730625im, -0.07949999999999999 - 0.06299999999999999im, -0.010124999999999997 + 0.012624999999999997im, 0.0005458333333333332 - 0.001475im, 0.010743749999999996 + 0.005368749999999996im, -0.02065416666666666 + 0.024781249999999987im, -0.009421875000000001 - 0.013340624999999993im, -0.042231250000000005 - 0.0im, 0.009421875000000001 - 0.013340624999999993im, -0.02065416666666666 - 0.024781249999999987im, -0.010743749999999996 + 0.005368749999999996im, 0.0005458333333333332 + 0.001475im, -0.00011265625000000009 + 2.1197916666666686e-5im, -5.937500000000571e-6 - 0.0010387499999999997im, 0.0034603124999999964 + 0.0008761458333333326im, -0.0020656249999999985 + 0.0034068749999999984im, 0.0012980078125 + 0.003358815104166667im, -0.007125234374999999 + 0.0im, -0.0012980078125 + 0.003358815104166667im, -0.0020656249999999985 - 0.0034068749999999984im, -0.0034603124999999964 + 0.0008761458333333326im, -5.937500000000571e-6 + 0.0010387499999999997im, 0.00011265625000000009 + 2.1197916666666686e-5im, 4.43958333333334e-6 + 4.996875000000008e-6im, -6.189843750000005e-5 + 3.3367187500000044e-5im, -6.285625000000003e-5 - 0.00028651875000000024im, 0.0004916250000000004 + 4.906249999999985e-5im, 9.071093750000005e-5 - 5.777343750000411e-6im, 0.000438952734375 + 0.0008640425781249999im, -0.00022750911458333293 - 0.0im, -0.000438952734375 + 0.0008640425781249999im, 9.071093750000005e-5 + 5.777343750000411e-6im, -0.0004916250000000004 + 4.906249999999985e-5im, -6.285625000000003e-5 + 0.00028651875000000024im, 6.189843750000005e-5 + 3.3367187500000044e-5im, 4.43958333333334e-6 - 4.996875000000008e-6im, 9.721354166666659e-8 - 3.099045138888889e-7im, 3.27072916666667e-6 + 2.1196875000000036e-6im, -1.4690065104166683e-5 + 1.2224457465277795e-5im, -1.511773437500001e-5 - 3.978656250000003e-5im, 2.0853691406250014e-5 - 4.499459635416686e-6im, 4.616617187500002e-5 - 5.9581875000000086e-5im, 4.0636233723958294e-5 + 7.035868381076384e-5im, 7.688800130208335e-5 + 0.0im, -4.0636233723958294e-5 + 7.035868381076384e-5im, 4.616617187500002e-5 + 5.9581875000000086e-5im, -2.0853691406250014e-5 - 4.499459635416686e-6im, -1.511773437500001e-5 + 3.978656250000003e-5im, 1.4690065104166683e-5 + 1.2224457465277795e-5im, 3.27072916666667e-6 - 2.1196875000000036e-6im, -9.721354166666659e-8 - 3.099045138888889e-7im, -1.315116567460317e-8 + 3.168898809523811e-9im, 1.1865513392857145e-8 - 1.8215606398809538e-7im, 9.526511656746043e-7 + 3.8884542410714344e-7im, -1.8955683593750022e-6 + 2.1069563802083367e-6im, -1.4576134982638887e-6 - 2.419467447916667e-6im, -3.5215048828125026e-6 - 9.739384765624999e-7im, 4.896028862847224e-6 - 7.63043522135417e-6im, -1.7391971261160826e-7 - 1.1484881184895846e-6im, 1.1709973229786705e-5 - 0.0im, 1.7391971261160826e-7 - 1.1484881184895846e-6im, 4.896028862847224e-6 + 7.63043522135417e-6im, 3.5215048828125026e-6 - 9.739384765624999e-7im, -1.4576134982638887e-6 + 2.419467447916667e-6im, 1.8955683593750022e-6 + 2.1069563802083367e-6im, 9.526511656746043e-7 - 3.8884542410714344e-7im, -1.1865513392857145e-8 - 1.8215606398809538e-7im, -1.315116567460317e-8 - 3.168898809523811e-9im, 3.4561244419642996e-10 + 3.515570746527791e-10im, -6.654608444940471e-9 + 3.2228794642857208e-9im, -5.190910993303557e-9 - 4.683708844866071e-8im, 1.495010230654763e-7 + 3.7817410714285645e-8im, -1.248901960100446e-7 + 1.8695223563058024e-7im, -1.6097639973958294e-8 + 9.095527343750019e-8im, -6.312749023437503e-7 - 6.452887369791655e-8im, 1.4630857049851168e-7 - 3.104571568080355e-7im, -3.759955413818363e-7 - 7.266106824602406e-7im, 5.803834975469673e-7 + 0.0im, 3.759955413818363e-7 - 7.266106824602406e-7im, 1.4630857049851168e-7 + 3.104571568080355e-7im, 6.312749023437503e-7 - 6.452887369791655e-8im, -1.6097639973958294e-8 - 9.095527343750019e-8im, 1.248901960100446e-7 + 1.8695223563058024e-7im, 1.495010230654763e-7 - 3.7817410714285645e-8im, 5.190910993303557e-9 - 4.683708844866071e-8im, -6.654608444940471e-9 - 3.2228794642857208e-9im, -3.4561244419642996e-10 + 3.515570746527791e-10im, 4.0052668926366666e-12 - 1.5459630249669243e-11im, 2.1771519252232125e-10 + 1.4871343057622366e-10im, -1.5317197920111357e-9 + 1.0822130249669325e-9im, -1.7221313476562567e-9 - 6.886385904947929e-9im, 1.3362126658606163e-8 + 1.6077458844866028e-9im, 9.24869210379457e-11 + 4.75633120582216e-9im, 1.103551204943783e-8 + 2.974370155939978e-8im, -4.036778982979905e-8 - 1.532068064856001e-10im, -1.8915326273277332e-8 + 2.3491573520236574e-8im, -3.4658169097900355e-8 - 6.169071084885364e-8im, -2.388126997205949e-8 - 0.0im, 3.4658169097900355e-8 - 6.169071084885364e-8im, -1.8915326273277332e-8 - 2.3491573520236574e-8im, 4.036778982979905e-8 - 1.532068064856001e-10im, 1.103551204943783e-8 - 2.974370155939978e-8im, -9.24869210379457e-11 + 4.75633120582216e-9im, 1.3362126658606163e-8 - 1.6077458844866028e-9im, 1.7221313476562567e-9 - 6.886385904947929e-9im, -1.5317197920111357e-9 - 1.0822130249669325e-9im, -2.1771519252232125e-10 + 1.4871343057622366e-10im, 4.0052668926366666e-12 + 1.5459630249669243e-11im]
multipole_check_expansion = initialize_expansion(expansion_order)
vector_to_expansion!(multipole_check_expansion, multipole_check, 1, expansion_order)

# define system
system = DipolePoints([xs], [q])
buffer = FastMultipole.system_to_buffer(system)

# create branch
bodies_index, n_branches, branch_index, i_parent, i_leaf_index = 1:1, 0, 1:0, 0, 0
box = SVector{3}(0.0,0.0,0.0)
branch = Branch(bodies_index, n_branches, branch_index, i_parent, i_leaf_index, center, radius, box)
multipole_expansion = initialize_expansion(expansion_order)
harmonics = initialize_harmonics(expansion_order)
body_to_multipole!(Point{Dipole}, system, multipole_expansion, buffer, branch.center, 1:1, harmonics, expansion_order)

test_expansion!(multipole_expansion, -multipole_check_expansion, 1, expansion_order)

# local branch
target_center = xt + SVector{3}(0.1, 0.2, -0.3)
target_branch = Branch(bodies_index, n_branches, branch_index, i_parent, i_leaf_index, target_center, radius, box)
target_expansion = initialize_expansion(expansion_order)

# preallocate containers
Hs_π2 = [1.0]
FastMultipole.update_Hs_π2!(Hs_π2, expansion_order)
Ts = zeros(FastMultipole.length_Ts(expansion_order))
eimϕs = zeros(2, expansion_order+2)
weights_tmp_1 = initialize_expansion(expansion_order, eltype(Ts))
weights_tmp_2 = initialize_expansion(expansion_order, eltype(Ts))
weights_tmp_3 = initialize_expansion(expansion_order, eltype(Ts))

# normalization
ζs_mag = zeros(FastMultipole.length_ζs(expansion_order))
FastMultipole.update_ζs_mag!(ζs_mag, 0, expansion_order)
ηs_mag = zeros(FastMultipole.length_ηs(expansion_order))
FastMultipole.update_ηs_mag!(ηs_mag, 0, expansion_order)

lamb_helmholtz = Val(false)
FastMultipole.multipole_to_local!(target_expansion, target_branch, multipole_expansion, branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, FastMultipole.M̃, FastMultipole.L̃, expansion_order, lamb_helmholtz)

gradient_n_m = zeros(2,3,size(harmonics,3))
ϕ_l2b, v_l2b, g_l2b = FastMultipole.evaluate_local(xt - target_center, harmonics, gradient_n_m, target_expansion, expansion_order, lamb_helmholtz, DerivativesSwitch())

# analytic
q = FastMultipole.get_strength(buffer,system,1)
x_point = FastMultipole.get_position(system,1)
Δx = xt - x_point
Δx̂ = Δx / norm(Δx)
r = norm(Δx)
ϕ_point = -1/4/pi/r^2 * dot(q,Δx̂)

@test isapprox(ϕ_point, ϕ_l2b; atol=1e-12)

end

@testset "body-to-multipole: point vortex" begin

# check expansion
expansion_order = 10
multipole_weights_1 = ComplexF64[0.0 + 0.0im, -0.0025 - 0.005im, 0.0 + 0.0im, 0.0025 - 0.005im, 3.333333333333335e-5 + 2.500000000000001e-5im, -4.9999999999999996e-5 - 9.999999999999999e-5im, 0.0 + 0.0im, 4.9999999999999996e-5 - 9.999999999999999e-5im, 3.333333333333335e-5 - 2.500000000000001e-5im, -1.7187500000000013e-7 - 3.125000000000006e-8im, 7.500000000000004e-7 + 5.625000000000003e-7im, -4.843749999999998e-7 - 9.687499999999996e-7im, 0.0 + 0.0im, 4.843749999999998e-7 - 9.687499999999996e-7im, 7.500000000000004e-7 - 5.625000000000003e-7im, 1.7187500000000013e-7 - 3.125000000000006e-8im, 5.000000000000007e-10 - 1.458333333333335e-10im, -4.125000000000003e-9 - 7.500000000000011e-10im, 8.16666666666667e-9 + 6.125000000000003e-9im, -2.6249999999999974e-9 - 5.2499999999999966e-9im, 0.0 + 0.0im, 2.6249999999999974e-9 - 5.2499999999999966e-9im, 8.16666666666667e-9 - 6.125000000000003e-9im, 4.125000000000003e-9 - 7.500000000000011e-10im, 5.000000000000007e-10 + 1.458333333333335e-10im, -8.897569444444466e-13 + 8.246527777777796e-13im, 1.2500000000000005e-11 - 3.6458333333333345e-12im, -4.798177083333337e-11 - 8.72395833333334e-12im, 5.4166666666666686e-11 + 4.062500000000001e-11im, -5.772569444444427e-12 - 1.1545138888888862e-11im, 0.0 + 0.0im, 5.772569444444427e-12 - 1.1545138888888862e-11im, 5.4166666666666686e-11 - 4.062500000000001e-11im, 4.798177083333337e-11 - 8.72395833333334e-12im, 1.2500000000000005e-11 + 3.6458333333333345e-12im, 8.897569444444466e-13 + 8.246527777777796e-13im, 8.184523809523819e-16 - 2.1763392857142933e-15im, -2.2879464285714336e-14 + 2.1205357142857183e-14im, 1.5178571428571467e-13 - 4.427083333333343e-14im, -3.4988839285714296e-13 - 6.36160714285715e-14im, 2.235863095238098e-13 + 1.676897321428574e-13im, 2.834821428571441e-14 + 5.669642857142871e-14im, 0.0 + 0.0im, -2.834821428571441e-14 + 5.669642857142871e-14im, 2.235863095238098e-13 - 1.676897321428574e-13im, 3.4988839285714296e-13 - 6.36160714285715e-14im, 1.5178571428571467e-13 + 4.427083333333343e-14im, 2.2879464285714336e-14 + 2.1205357142857183e-14im, 8.184523809523819e-16 + 2.1763392857142933e-15im, 3.933376736111114e-19 + 3.7706163194444516e-18im, 2.1484375000000026e-17 - 5.71289062500002e-17im, -2.8639051649305604e-16 + 2.6543511284722267e-16im, 1.1718750000000027e-15 - 3.417968750000007e-16im, -1.7254638671875033e-15 - 3.1372070312500076e-16im, 4.316406250000017e-16 + 3.2373046875000143e-16im, 3.2781304253472126e-16 + 6.556260850694421e-16im, 0.0 + 0.0im, -3.2781304253472126e-16 + 6.556260850694421e-16im, 4.316406250000017e-16 - 3.2373046875000143e-16im, 1.7254638671875033e-15 - 3.1372070312500076e-16im, 1.1718750000000027e-15 + 3.417968750000007e-16im, 2.8639051649305604e-16 + 2.6543511284722267e-16im, 2.1484375000000026e-17 + 5.71289062500002e-17im, -3.933376736111114e-19 + 3.7706163194444516e-18im, -2.8935185185185183e-21 - 4.538346009700186e-21im, 1.0489004629629646e-20 + 1.0054976851851883e-19im, 2.750909391534395e-19 - 7.3149181547619315e-19im, -2.298538773148151e-18 + 2.130353009259262e-18im, 6.322337962962972e-18 - 1.844015239197533e-18im, -5.621744791666671e-18 - 1.0221354166666681e-18im, -1.1800733024691392e-18 - 8.85054976851854e-19im, 1.5913835152116374e-18 + 3.1827670304232737e-18im, 0.0 + 0.0im, -1.5913835152116374e-18 + 3.1827670304232737e-18im, -1.1800733024691392e-18 + 8.85054976851854e-19im, 5.621744791666671e-18 - 1.0221354166666681e-18im, 6.322337962962972e-18 + 1.844015239197533e-18im, 2.298538773148151e-18 + 2.130353009259262e-18im, 2.750909391534395e-19 + 7.3149181547619315e-19im, -1.0489004629629646e-20 + 1.0054976851851883e-19im, -2.8935185185185183e-21 + 4.538346009700186e-21im, 5.808027963789718e-24 + 3.4780350942460546e-24im, -7.812499999999994e-23 - 1.2253534226190493e-22im, 1.3668484157986129e-22 + 1.3102891710069483e-21im, 2.2712053571428575e-21 - 6.039341517857155e-21im, -1.311199854290675e-20 + 1.2152584015376984e-20im, 2.445312500000003e-20 - 7.132161458333344e-21im, -8.998074001736076e-21 - 1.6360134548611065e-21im, -1.3312872023809537e-20 - 9.984654017857154e-21im, 4.541180323040666e-21 + 9.082360646081333e-21im, 0.0 + 0.0im, -4.541180323040666e-21 + 9.082360646081333e-21im, -1.3312872023809537e-20 + 9.984654017857154e-21im, 8.998074001736076e-21 - 1.6360134548611065e-21im, 2.445312500000003e-20 + 7.132161458333344e-21im, 1.311199854290675e-20 + 1.2152584015376984e-20im, 2.2712053571428575e-21 + 6.039341517857155e-21im, -1.3668484157986129e-22 + 1.3102891710069483e-21im, -7.812499999999994e-23 + 1.2253534226190493e-22im, -5.808027963789718e-24 + 3.4780350942460546e-24im, -7.6232782938512e-27 - 5.798193054052366e-28im, 1.5840076264880992e-25 + 9.485550257034661e-26im, -1.0324600168350196e-24 - 1.619364371643026e-24im, 1.1531945430871238e-24 + 1.105476148200761e-23im, 1.3465518043154768e-23 - 3.580603661475252e-23im, -5.548703412472947e-23 + 5.1427007237554135e-23im, 6.463350018037516e-23 - 1.8851437552609444e-23im, 1.9525437127976443e-23 + 3.5500794778138974e-24im, -5.77788323149651e-23 - 4.333412423622383e-23im, 5.356561569940499e-24 + 1.0713123139881006e-23im, 0.0 + 0.0im, -5.356561569940499e-24 + 1.0713123139881006e-23im, -5.77788323149651e-23 + 4.333412423622383e-23im, -1.9525437127976443e-23 + 3.5500794778138974e-24im, 6.463350018037516e-23 + 1.8851437552609444e-23im, 5.548703412472947e-23 + 5.1427007237554135e-23im, 1.3465518043154768e-23 + 3.580603661475252e-23im, -1.1531945430871238e-24 + 1.105476148200761e-23im, -1.0324600168350196e-24 + 1.619364371643026e-24im, -1.5840076264880992e-25 + 9.485550257034661e-26im, -7.6232782938512e-27 + 5.798193054052366e-28im]
multipole_weights_2 = ComplexF64[0.0 + 0.0im, 0.0 + 0.0im, 1.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, -0.005000000000000002 + 0.002500000000000001im, 0.015 + 0.0im, 0.005000000000000002 + 0.002500000000000001im, 0.0 + 0.0im, 0.0 + 0.0im, 1.2500000000000012e-5 - 1.6666666666666674e-5im, -0.00010000000000000006 + 5.000000000000003e-5im, 0.00010833333333333327 + 0.0im, 0.00010000000000000006 + 5.000000000000003e-5im, 1.2500000000000012e-5 + 1.6666666666666674e-5im, 0.0 + 0.0im, 0.0 + 0.0im, -1.0416666666666685e-8 + 5.7291666666666745e-8im, 2.812500000000002e-7 - 3.7500000000000017e-7im, -9.687500000000003e-7 + 4.843750000000001e-7im, 1.8749999999999924e-7 + 0.0im, 9.687500000000003e-7 + 4.843750000000001e-7im, 2.812500000000002e-7 + 3.7500000000000017e-7im, 1.0416666666666685e-8 + 5.7291666666666745e-8im, 0.0 + 0.0im, 0.0 + 0.0im, -3.645833333333341e-11 - 1.250000000000003e-10im, -2.500000000000002e-10 + 1.3750000000000007e-9im, 3.0625000000000018e-9 - 4.083333333333336e-9im, -5.2500000000000015e-9 + 2.6250000000000008e-9im, -3.7187500000000064e-9 + 0.0im, 5.2500000000000015e-9 + 2.6250000000000008e-9im, 3.0625000000000018e-9 + 4.083333333333336e-9im, 2.500000000000002e-10 + 1.3750000000000007e-9im, -3.645833333333341e-11 + 1.250000000000003e-10im, 0.0 + 0.0im, 0.0 + 0.0im, 1.6493055555555618e-13 + 1.7795138888888932e-13im, -9.114583333333346e-13 - 3.125000000000006e-12im, -2.907986111111119e-12 + 1.599392361111115e-11im, 2.0312500000000004e-11 - 2.7083333333333334e-11im, -1.1545138888888888e-11 + 5.772569444444444e-12im, -4.046875000000002e-11 + 0.0im, 1.1545138888888888e-11 + 5.772569444444444e-12im, 2.0312500000000004e-11 + 2.7083333333333334e-11im, 2.907986111111119e-12 + 1.599392361111115e-11im, -9.114583333333346e-13 + 3.125000000000006e-12im, -1.6493055555555618e-13 + 1.7795138888888932e-13im, 0.0 + 0.0im, 0.0 + 0.0im, -3.627232142857149e-16 - 1.3640873015873046e-16im, 4.241071428571445e-15 + 4.575892857142869e-15im, -1.1067708333333349e-14 - 3.7946428571428636e-14im, -2.12053571428572e-14 + 1.1662946428571454e-13im, 8.384486607142875e-14 - 1.1179315476190498e-13im, 5.6696428571428594e-14 - 2.8348214285714297e-14im, -2.146701388888889e-13 + 0.0im, -5.6696428571428594e-14 - 2.8348214285714297e-14im, 8.384486607142875e-14 + 1.1179315476190498e-13im, 2.12053571428572e-14 + 1.1662946428571454e-13im, -1.1067708333333349e-14 + 3.7946428571428636e-14im, -4.241071428571445e-15 + 4.575892857142869e-15im, -3.627232142857149e-16 + 1.3640873015873046e-16im, 0.0 + 0.0im, 0.0 + 0.0im, 5.386594742063502e-19 - 5.619109623015824e-20im, -9.521484375000029e-18 - 3.58072916666668e-18im, 5.308702256944467e-17 + 5.727810329861126e-17im, -8.544921875000009e-17 - 2.929687500000004e-16im, -1.045735677083335e-16 + 5.751546223958339e-16im, 1.618652343750001e-16 - 2.1582031250000004e-16im, 6.556260850694454e-16 - 3.278130425347227e-16im, -6.278366815476184e-16 + 0.0im, -6.556260850694454e-16 - 3.278130425347227e-16im, 1.618652343750001e-16 + 2.1582031250000004e-16im, 1.045735677083335e-16 + 5.751546223958339e-16im, -8.544921875000009e-17 + 2.929687500000004e-16im, -5.308702256944467e-17 + 5.727810329861126e-17im, -9.521484375000029e-18 + 3.58072916666668e-18im, -5.386594742063502e-19 - 5.619109623015824e-20im, 0.0 + 0.0im, 0.0 + 0.0im, -5.672932512125255e-22 + 3.6168981481481694e-22im, 1.4364252645502657e-20 - 1.4984292328042185e-21im, -1.2191530257936541e-19 - 4.584848985890667e-20im, 4.2607060185185297e-19 + 4.597077546296302e-19im, -4.610038097993824e-19 - 1.5805844907407403e-18im, -3.4071180555555596e-19 + 1.873914930555557e-18im, -4.425274884259278e-19 + 5.900366512345702e-19im, 3.1827670304232833e-18 - 1.5913835152116416e-18im, -2.716053688547128e-19 + 0.0im, -3.1827670304232833e-18 - 1.5913835152116416e-18im, -4.425274884259278e-19 - 5.900366512345702e-19im, 3.4071180555555596e-19 + 1.873914930555557e-18im, -4.610038097993824e-19 + 1.5805844907407403e-18im, -4.2607060185185297e-19 + 4.597077546296302e-19im, -1.2191530257936541e-19 + 4.584848985890667e-20im, -1.4364252645502657e-20 - 1.4984292328042185e-21im, -5.672932512125255e-22 - 3.6168981481481694e-22im, 0.0 + 0.0im, 0.0 + 0.0im, 3.8644834380511364e-25 - 6.453364404210751e-25im, -1.531691778273814e-23 + 9.765625000000024e-24im, 1.8718416728670722e-22 - 1.9526405939980042e-23im, -1.0065569196428603e-21 - 3.785342261904774e-22im, 2.4305168030754038e-21 + 2.6223997085813528e-21im, -1.783040364583333e-21 - 6.113281250000001e-21im, -5.453378182870358e-22 + 2.9993580005786955e-21im, -4.992327008928588e-21 + 6.656436011904783e-21im, 9.082360646081339e-21 - 4.5411803230406695e-21im, 7.142101469494068e-21 + 0.0im, -9.082360646081339e-21 - 4.5411803230406695e-21im, -4.992327008928588e-21 - 6.656436011904783e-21im, 5.453378182870358e-22 + 2.9993580005786955e-21im, -1.783040364583333e-21 + 6.113281250000001e-21im, -2.4305168030754038e-21 + 2.6223997085813528e-21im, -1.0065569196428603e-21 + 3.785342261904774e-22im, -1.8718416728670722e-22 - 1.9526405939980042e-23im, -1.531691778273814e-23 - 9.765625000000024e-24im, -3.8644834380511364e-25 - 6.453364404210751e-25im, 0.0 + 0.0im]

multipole_check_expansion = initialize_expansion(expansion_order)
vector_to_expansion!(multipole_check_expansion, multipole_weights_1, 1, expansion_order)
vector_to_expansion!(multipole_check_expansion, multipole_weights_2, 2, expansion_order)

# vortex particle
# xs = SVector{3}(0.0,0,0)
# q = SVector{3}(0,0,1.0)
xs = SVector{3}(1.0,1.0,1.0)
q = SVector{3}(1.0,0,0)

system = Vortons([xs], [q])
buffer = FastMultipole.system_to_buffer(system)

# create branch
bodies_index, n_branches, branch_index, i_parent, i_leaf_index = 1:1, 0, 1:0, 0, 0
center = xs + SVector{3}(0.01, 0.02, -0.03)
radius = 0.0
box = SVector{3}(0.0,0.0,0.0)
branch = Branch(bodies_index, n_branches, branch_index, i_parent, i_leaf_index, center, radius, box)
multipole_expansion = initialize_expansion(expansion_order)
harmonics = initialize_harmonics(expansion_order)

body_to_multipole!(Point{Vortex}, system, multipole_expansion, buffer, branch.center, 1:1, harmonics, expansion_order)

# test_expansion!(branch.multipole_expansion, multipole_check_expansion, 1, expansion_order; throwme=true)
# test_expansion!(branch.multipole_expansion, multipole_check_expansion, 2, expansion_order)

# xt = SVector{3}(6.8, 0.0, 0.0)
# target_center = xt + SVector{3}(0.1,-0.2,-0.3)
xt = SVector{3}(1.0,1.0,7.4)
target_center = SVector{3}(1.0,1.0,7.0)

target_branch = Branch(bodies_index, n_branches, branch_index, i_parent, i_leaf_index, target_center, radius, box)
target_expansion = initialize_expansion(expansion_order)

# preallocate containers
Hs_π2 = [1.0]
FastMultipole.update_Hs_π2!(Hs_π2, expansion_order+1)
Ts = zeros(FastMultipole.length_Ts(expansion_order+1))
eimϕs = zeros(2, expansion_order+2)
weights_tmp_1 = initialize_expansion(expansion_order+1, eltype(Ts))
weights_tmp_2 = initialize_expansion(expansion_order+1, eltype(Ts))
weights_tmp_3 = initialize_expansion(expansion_order+1, eltype(Ts))

# normalization
ζs_mag = zeros(FastMultipole.length_ζs(expansion_order+1))
FastMultipole.update_ζs_mag!(ζs_mag, 0, expansion_order+1)
ηs_mag = zeros(FastMultipole.length_ηs(expansion_order+1))
FastMultipole.update_ηs_mag!(ηs_mag, 0, expansion_order+1)

# perform transformation
lamb_helmholtz = Val(true)
FastMultipole.multipole_to_local!(target_expansion, target_branch, multipole_expansion, branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, FastMultipole.M̃, FastMultipole.L̃, expansion_order, lamb_helmholtz)

# evaluate at target
gradient_n_m = FastMultipole.initialize_gradient_n_m(expansion_order)
ϕ_l2b, v_l2b, g_l2b = FastMultipole.evaluate_local(xt - target_center, harmonics, gradient_n_m, target_expansion, expansion_order, lamb_helmholtz, DerivativesSwitch())

# evaluate multipole at target
ϕ_m2b, v_m2b, g_m2b = evaluate_multipole(xt, branch.center, multipole_expansion, DerivativesSwitch(true,true,false), expansion_order, lamb_helmholtz)

@test isapprox(v_m2b, v_l2b; atol=1e-12)
# @test isapprox(g_m2b, g_l2b; atol=1e-12) # this doesn't work for some reason

# generate local expansion directly
target_branch_2 = deepcopy(target_branch)
target_expansion_2 = initialize_expansion(expansion_order)
body_to_local_point!(Point{Vortex}, target_expansion_2, harmonics, FastMultipole.get_position(system,1) - target_branch_2.center, FastMultipole.get_strength(buffer,system,1), expansion_order)
gradient_n_m .= 0.0
ϕ_l2b_2, v_l2b_2, g_l2b_2 = FastMultipole.evaluate_local(xt - target_center, harmonics, gradient_n_m, target_expansion_2, expansion_order, lamb_helmholtz, DerivativesSwitch())

@test isapprox(v_l2b_2, v_l2b; atol=1e-12)
@test isapprox(g_l2b_2, g_l2b; atol=1e-11)

# analytic result
dx = xt-xs
r = norm(dx)
gamma_over_R = q / r / (4*pi)

function flatten_derivatives!(jacobian)
    # vector
    gradient = zeros(3)
    gradient[1] = jacobian[2,3] - jacobian[3,2]
    gradient[2] = jacobian[3,1] - jacobian[1,3]
    gradient[3] = jacobian[1,2] - jacobian[2,1]

    return gradient
end

jacobian = zeros(3,3)
gamma_over_R /= r^2
for j_potential in 1:3
    for i_r in 1:3
        jacobian[i_r,j_potential] -= gamma_over_R[j_potential] * dx[i_r]
    end
end
v_analytic = flatten_derivatives!(jacobian)

@test isapprox(v_analytic, v_l2b; atol=1e-12)

end

# auxilliary functions

function Rnm(x,n,m)
    ρ, θ, ϕ = FastMultipole.cartesian_to_spherical(x)
    return Complex{Float64}(exp(im*m*ϕ) * Plm(cos(θ),n,abs(m)) * (-1)^n * (1.0im)^abs(m) * ρ^n/ factorial(big(n+abs(m))))
end

function calculate_p_quad!(harmonics, x0, xu, expansion_order)
    ts = range(0,1,length=1000)
    xs = [x0+xu*t for t in ts]
    i = 1
    for n in 0:expansion_order
        for m in 0:n
            rnms = [Rnm(x,n,m) for x in xs]
            pnm = trapz(ts,rnms)
            harmonics[1,2,i] = real(pnm)
            harmonics[2,2,i] = imag(pnm)
            i += 1
        end
    end
end

function test_pq()
    expansion_order = 4
    x0 = [-0.1, -0.2001500000000001, -0.08]
    xu = [0.0, 0.00030000000000002247, 0.0]
    x2_r = x0 + xu
    harmonics_check = initialize_harmonics(expansion_order)
    ρ, θ, ϕ = FastMultipole.cartesian_to_spherical(x2_r)
    FastMultipole.regular_harmonics!(harmonics_check, ρ,θ,ϕ, expansion_order)

    harmonics = initialize_harmonics(expansion_order)
    ξ0_real, ξ0_imag, η0_real, η0_imag, z0 = FastMultipole.xyz_to_ξηz(x0)
    ξu_real, ξu_imag, ηu_real, ηu_imag, zu = FastMultipole.xyz_to_ξηz(xu)
    FastMultipole.calculate_q!(harmonics, ξ0_real+ξu_real, ξ0_imag+ξu_imag, η0_real+ηu_real, η0_imag+ηu_imag, z0+zu, expansion_order)

    test_expansion!(harmonics, harmonics_check, 1, expansion_order)

    FastMultipole.calculate_pj!(harmonics, ξ0_real, ξ0_imag, η0_real, η0_imag, z0, expansion_order)
    calculate_p_quad!(harmonics_check, x0, xu, expansion_order)
    test_expansion!(harmonics, harmonics_check, 2, expansion_order)
end

@testset "body-to-multipole: auxilliary functions" begin

test_pq()

end

@testset "body-to-multipole: source filament" begin

# source filament

x1 = SVector{3,Float64}(0.0,0.2999,0.0)
x2 = x1 + SVector{3,Float64}(0.0,0.0003,0.0)
x = zeros(SVector{3,Float64},2,1)
q = 1.0
x[1,1] = x1
x[2,1] = x2
system = SourceFilaments(x, [q])
buffer = FastMultipole.system_to_buffer(system)

xt = SVector{3}(0.0, 0.3, 8.5)

function source_filament(x1, x2, xt, q)
    # get altitude
    t = dot(xt-x1,(x2-x1) / norm(x2-x1))
    xa = x1 + t * (x2-x1) / norm(x2-x1)
    a = norm(xt-xa)

    # get L1 and L2
    L1 = -t
    L2 = dot(x2-xt,(x2-x1) / norm(x2-x1))

    # evaluate integral
    ϕ = q * (atan(L2/sqrt(a^2+L2^2)) - atan(L1/sqrt(a^2+L1^2))) / 4 / pi
    return ϕ
end

ϕ_check = source_filament(x1, x2, xt, q)

expansion_order = 10
bodies_index, n_branches, branch_index, i_parent, i_leaf_index = 1:1, 0, 1:0, 0, 0
center = (x1+x2)/2 + SVector{3}(0.1,0.2,0.08)
radius = 0.0
box = SVector{3}(0.0,0.0,0.0)

branch = Branch(bodies_index, n_branches, branch_index, i_parent, i_leaf_index, center, radius, box)
harmonics = initialize_harmonics(expansion_order)
multipole_expansion = initialize_expansion(expansion_order)
body_to_multipole!(Filament{Source}, system, multipole_expansion, buffer, branch.center, 1:1, harmonics, expansion_order)
ϕ_m2b, v_m2b, g_m2b = evaluate_multipole(xt, branch.center, multipole_expansion, DerivativesSwitch(), expansion_order)

@test isapprox(ϕ_check, ϕ_m2b; atol=1e-12)

end

@testset "body-to-multipole: dipole filament" begin

# source filament

x1 = SVector{3,Float64}(0.0,0.29,0.0)
x2 = x1 + SVector{3,Float64}(0.0,0.03,0.0)
x = zeros(SVector{3,Float64},2,1)
q = SVector{3}(0,1.0,0)
x[1,1] = x1
x[2,1] = x2
system = DipoleFilaments(x, [q])
buffer = FastMultipole.system_to_buffer(system)

xt = SVector{3}(4.0, 0.3, 0.0)

function source_filament(x1, x2, xt, q)
    # get altitude
    t = dot(xt-x1,(x2-x1) / norm(x2-x1))
    xa = x1 + t * (x2-x1) / norm(x2-x1)
    a = norm(xt-xa)

    # get L1 and L2
    L1 = -t
    L2 = dot(x2-xt,(x2-x1) / norm(x2-x1))

    # evaluate integral
    ϕ = q * (atan(L2/sqrt(a^2+L2^2)) - atan(L1/sqrt(a^2+L1^2))) / 4 / pi
    return ϕ
end

function dipole_filament(x1,x2,xt,q)
    grad = ForwardDiff.gradient((x)->source_filament(x1,x2,x,1.0), xt)
    dipole_potential = -dot(grad,q)
    return dipole_potential
end

function get_s_manual(p, q, P)
    s = zero(p)
    i = 2
    for n in 1:P
        i_nm1_m = i - n
        for m in 0:n
            if m+1 < n
                p_nm1_mp1 = p[i_nm1_m+1]
            else
                p_nm1_mp1 = 0.0im
            end
            if m < n
                p_nm1_m = p[i_nm1_m]
            else
                p_nm1_m = 0.0im
            end
            if n > 1
                if -1 < m-1
                    p_nm1_mm1 = p[i_nm1_m-1]
                elseif m-1 == -1
                    p_nm1_mm1 = -(-1)^m * conj(p[i_nm1_m+1])
                else
                    throw()
                end
            else
                if m-1==0
                    p_nm1_mm1 = p[1]
                else
                    p_nm1_mm1 = 0.0im
                end
            end
            snm = q[1]/2 * im * (p_nm1_mp1 + p_nm1_mm1) + q[2]/2 * (p_nm1_mp1 - p_nm1_mm1) - q[3] * p_nm1_m
            s[i] = snm
            i += 1
            i_nm1_m += 1
        end
    end
    return s
end

ϕ_check = dipole_filament(x1, x2, xt, q*norm(x2-x1))

expansion_order = 10
bodies_index, n_branches, branch_index, i_parent, i_leaf_index = 1:1, 0, 1:0, 0, 0
center = (x1+x2)/2 + SVector{3}(0.1,0.2,0.08)
radius = 0.0
box = SVector{3}(0.0,0.0,0.0)

branch = Branch(bodies_index, n_branches, branch_index, i_parent, i_leaf_index, center, radius, box)
harmonics = initialize_harmonics(expansion_order)
multipole_expansion = initialize_expansion(expansion_order)
body_to_multipole!(Filament{Dipole}, system, multipole_expansion, buffer, branch.center, 1:1, harmonics, expansion_order)

s = harmonics[1,1,:] .+ im .* harmonics[2,1,:]
p = harmonics[1,2,:] .+ im .* harmonics[2,2,:]
s_man = get_s_manual(p, q, expansion_order)

for i in eachindex(s)
    @test isapprox(s[i], s_man[i]; atol=1e-12)
end

ϕ_m2b, v_m2b, g_m2b = evaluate_multipole(xt, branch.center, multipole_expansion, DerivativesSwitch(), expansion_order)

# point dipole
x_point = (x1+x2)/2
Δx = xt - x_point
Δx̂ = Δx / norm(Δx)
r = norm(Δx)
ϕ_point = 1/4/pi/r^2 * dot(q*norm(x2-x1),Δx̂)

cθ = dot(Δx, q)/norm(Δx)/norm(q)
V_point = norm(q) * norm(x2-x1)/4/pi/r^2 * cθ

@test isapprox(1 - ϕ_m2b/ϕ_point, 0.0; atol=1e-4)

end

@testset "body-to-multipole: vortex filament" begin

x1 = SVector{3,Float64}(0.0,0.29,0.0)
x2 = x1 + SVector{3,Float64}(0.0,0.03,0.0)
x = zeros(SVector{3,Float64},2,1)
q = SVector{3}(0,1.0,0)
x[1,1] = x1
x[2,1] = x2
system = VortexFilaments(x, [q]; core_size=fill(1e-12, size(x,2)),)
buffer = FastMultipole.system_to_buffer(system)

xt = SVector{3}(4.0, 0.3, 0.0)

q_mag = norm(q) * sign(dot(q, x2-x1))
v_check = vortex_filament(x1, x2, xt, q_mag)

expansion_order = 10
bodies_index, n_branches, branch_index, i_parent, i_leaf_index = 1:1, 0, 1:0, 0, 0
center = (x1+x2)/2 + SVector{3}(0.1,0.2,0.08)
radius = 0.0
box = SVector{3}(0.0,0.0,0.0)

branch = Branch(bodies_index, n_branches, branch_index, i_parent, i_leaf_index, center, radius, box)
harmonics = initialize_harmonics(expansion_order)
multipole_expansion = initialize_expansion(expansion_order)
body_to_multipole!(Filament{Vortex}, system, multipole_expansion, buffer, branch.center, 1:1, harmonics, expansion_order)

ϕnm_filament = multipole_expansion[1,1,:] .+ multipole_expansion[2,1,:] .* im
χnm_filament = multipole_expansion[1,2,:] .+ multipole_expansion[2,2,:] .* im

# evaluate local expansion
target_center = xt + SVector{3}(0.0001, -0.0002, 0.003)

target_branch = Branch(bodies_index, n_branches, branch_index, i_parent, i_leaf_index, target_center, radius, box)
target_expansion = initialize_expansion(expansion_order)

# preallocate containers
Hs_π2 = [1.0]
FastMultipole.update_Hs_π2!(Hs_π2, expansion_order)
Ts = zeros(FastMultipole.length_Ts(expansion_order))
eimϕs = zeros(2, expansion_order+2)
weights_tmp_1 = initialize_expansion(expansion_order, eltype(Ts))
weights_tmp_2 = initialize_expansion(expansion_order, eltype(Ts))
weights_tmp_3 = initialize_expansion(expansion_order, eltype(Ts))

# normalization
ζs_mag = zeros(FastMultipole.length_ζs(expansion_order))
FastMultipole.update_ζs_mag!(ζs_mag, 0, expansion_order)
ηs_mag = zeros(FastMultipole.length_ηs(expansion_order))
FastMultipole.update_ηs_mag!(ηs_mag, 0, expansion_order)

# perform transformation
lamb_helmholtz = Val(true)
FastMultipole.multipole_to_local!(target_expansion, target_branch, multipole_expansion, branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, FastMultipole.M̃, FastMultipole.L̃, expansion_order, lamb_helmholtz)

gradient_n_m = FastMultipole.initialize_gradient_n_m(expansion_order)
ϕ_l2b, v_l2b, g_l2b = FastMultipole.evaluate_local(xt - target_center, harmonics, gradient_n_m, target_expansion, expansion_order, Val(true), DerivativesSwitch())

@test isapprox(v_check, v_l2b; atol=1e-12)

end

@testset "body-to-multipole: source panel" begin

# source panel
x1 = SVector{3,Float64}(0.0,0.29,0.0)
x2 = x1 + SVector{3,Float64}(0.0,0.03,0.0) * 10
x3 = x2 + SVector{3,Float64}(-0.05,0,0) * 10
vertices = SVector(x1,x2,x3)
x = [vertices]
q = 1.0
system = SourcePanels(x, [q])
buffer = FastMultipole.system_to_buffer(system)

xt = SVector{3}(0.2, 0.3, 5.7)

normal = FastMultipole.get_normal(buffer, system, 1)
strength = FastMultipole.get_strength(buffer, system, 1)
centroid = FastMultipole.get_position(buffer, 1)
ϕ_check, v_check, g_check = induced(xt, vertices, normal, strength, centroid, Panel{Source}, DerivativesSwitch(true,true,true))

expansion_order = 10
bodies_index, n_branches, branch_index, i_parent, i_leaf_index = 1:1, 0, 1:0, 0, 0
center = centroid + SVector{3}(0.1,0.2,0.08)
radius = 0.0
box = SVector{3}(0.0,0.0,0.0)

branch = Branch(bodies_index, n_branches, branch_index, i_parent, i_leaf_index, center, radius, box)
multipole_expansion = initialize_expansion(expansion_order)
harmonics = initialize_harmonics(expansion_order)
body_to_multipole!(Panel{Source}, system, multipole_expansion, buffer, branch.center, 1:1, harmonics, expansion_order)

ϕnm_panel = multipole_expansion[1,1,:] .+ multipole_expansion[2,1,:] .* im
χnm_panel = multipole_expansion[1,2,:] .+ multipole_expansion[2,2,:] .* im

# equivalent point source
x_point = centroid
area = norm(cross(x2-x1,x3-x1))/2
q_point = q * area

system_point = SourcePoints([x_point], [q_point])
branch_point = Branch(bodies_index, n_branches, branch_index, i_parent, i_leaf_index, center, radius, box)
expansion_point = initialize_expansion(expansion_order)
harmonics_point = initialize_harmonics(expansion_order)
body_to_multipole!(Point{Source}, system_point, expansion_point, buffer, branch_point.center, 1:1, harmonics_point, expansion_order)
#ϕ_point, v_point, g_point = evaluate_multipole(xt, branch_point.center, branch_point.multipole_expansion, DerivativesSwitch(true,true,true), expansion_order)

ϕnm_point = expansion_point[1,1,:] .+ expansion_point[2,1,:] .* im
χnm_point = expansion_point[1,2,:] .+ expansion_point[2,2,:] .* im

# evaluate local expansion
target_center = xt + SVector{3}(0.0001, -0.0002, 0.003)
target_branch = Branch(bodies_index, n_branches, branch_index, i_parent, i_leaf_index, target_center, radius, box)
target_expansion = initialize_expansion(expansion_order)
target_branch_point = Branch(bodies_index, n_branches, branch_index, i_parent, i_leaf_index, target_center, radius, box)
target_expansion_point = initialize_expansion(expansion_order)

# preallocate containers
Hs_π2 = [1.0]
FastMultipole.update_Hs_π2!(Hs_π2, expansion_order)
Ts = zeros(FastMultipole.length_Ts(expansion_order))
eimϕs = zeros(2, expansion_order+2)
weights_tmp_1 = initialize_expansion(expansion_order, eltype(Ts))
weights_tmp_2 = initialize_expansion(expansion_order, eltype(Ts))
weights_tmp_3 = initialize_expansion(expansion_order, eltype(Ts))

# normalization
ζs_mag = zeros(FastMultipole.length_ζs(expansion_order))
FastMultipole.update_ζs_mag!(ζs_mag, 0, expansion_order)
ηs_mag = zeros(FastMultipole.length_ηs(expansion_order))
FastMultipole.update_ηs_mag!(ηs_mag, 0, expansion_order)

# perform transformation
lamb_helmholtz = Val(false)
FastMultipole.multipole_to_local!(target_expansion, target_branch, multipole_expansion, branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, FastMultipole.M̃, FastMultipole.L̃, expansion_order, lamb_helmholtz)
FastMultipole.multipole_to_local!(target_expansion_point, target_branch_point, expansion_point, branch_point, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, FastMultipole.M̃, FastMultipole.L̃, expansion_order, lamb_helmholtz)

gradient_n_m = FastMultipole.initialize_gradient_n_m(expansion_order)
ϕ_l2b, v_l2b, g_l2b = FastMultipole.evaluate_local(xt - target_center, harmonics, gradient_n_m, target_expansion, expansion_order, lamb_helmholtz, DerivativesSwitch())

# repeat for point
Hs_π2 = [1.0]
FastMultipole.update_Hs_π2!(Hs_π2, expansion_order)
Ts = zeros(FastMultipole.length_Ts(expansion_order))
eimϕs = zeros(2, expansion_order+2)
weights_tmp_1 = initialize_expansion(expansion_order, eltype(Ts))
weights_tmp_2 = initialize_expansion(expansion_order, eltype(Ts))
weights_tmp_3 = initialize_expansion(expansion_order, eltype(Ts))
ζs_mag = zeros(FastMultipole.length_ζs(expansion_order))
FastMultipole.update_ζs_mag!(ζs_mag, 0, expansion_order)
ηs_mag = zeros(FastMultipole.length_ηs(expansion_order))
FastMultipole.update_ηs_mag!(ηs_mag, 0, expansion_order)

ϕ_point, v_point, g_point = FastMultipole.evaluate_local(xt - target_center, harmonics, gradient_n_m, target_expansion_point, expansion_order, lamb_helmholtz, DerivativesSwitch())

@test isapprox(v_check, v_l2b; atol=1e-12)

end

@testset "body-to-multipole: dipole panel" begin

# source panel
x1 = SVector{3,Float64}(0.0,0.29,0.0)
x2 = x1 + SVector{3,Float64}(0.0,0.03,0.0) * 10
x3 = x2 + SVector{3,Float64}(-0.05,0,0) * 10
vertices = SVector(x1,x2,x3)
x = [vertices]
q = SVector{1}(1.0)
system = DipolePanels(x, [q])
buffer = FastMultipole.system_to_buffer(system)

xt = SVector{3}(0.2, 0.3, 5.7)

normal = FastMultipole.get_normal(buffer,system,1)
strength = FastMultipole.get_strength(buffer,system,1)
centroid = FastMultipole.get_position(buffer,1)
ϕ_check, v_check, g_check = induced(xt, vertices, normal, SVector{1}(norm(strength)), centroid, Panel{Dipole}, DerivativesSwitch(true,true,true))

expansion_order = 10
bodies_index, n_branches, branch_index, i_parent, i_leaf_index = 1:1, 0, 1:0, 0, 0
center = centroid + SVector{3}(0.1,0.2,0.08)
radius = 0.0
box = SVector{3}(0.0,0.0,0.0)

branch = Branch(bodies_index, n_branches, branch_index, i_parent, i_leaf_index, center, radius, box)
multipole_expansion = initialize_expansion(expansion_order)
harmonics = initialize_harmonics(expansion_order)
body_to_multipole!(Panel{Dipole}, system, multipole_expansion, buffer, branch.center, 1:1, harmonics, expansion_order)

ϕnm_panel = multipole_expansion[1,1,:] .+ multipole_expansion[2,1,:] .* im
χnm_panel = multipole_expansion[1,2,:] .+ multipole_expansion[2,2,:] .* im

# equivalent point dipole
x_point = centroid
area = norm(cross(x2-x1,x3-x1))/2
q_point = q[1] * area * normal

system_point = DipolePoints([x_point], [q_point])
buffer_point = FastMultipole.system_to_buffer(system_point)

branch_point = Branch(bodies_index, n_branches, branch_index, i_parent, i_leaf_index, center, radius, box)
expansion_point = initialize_expansion(expansion_order)
harmonics_point = initialize_harmonics(expansion_order)
body_to_multipole!(Point{Dipole}, system_point, expansion_point, buffer_point, branch_point.center, 1:1, harmonics_point, expansion_order)

ϕnm_point = expansion_point[1,1,:] .+ expansion_point[2,1,:] .* im
χnm_point = expansion_point[1,2,:] .+ expansion_point[2,2,:] .* im

# evaluate local expansion
target_center = xt + SVector{3}(0.0001, -0.0002, 0.003)
box = SVector{3}(0.0,0.0,0.0)
target_branch = Branch(bodies_index, n_branches, branch_index, i_parent, i_leaf_index, target_center, radius, box)
target_expansion = initialize_expansion(expansion_order)
target_branch_point = Branch(bodies_index, n_branches, branch_index, i_parent, i_leaf_index, target_center, radius, box)
target_expansion_point = initialize_expansion(expansion_order)

# preallocate containers
Hs_π2 = [1.0]
FastMultipole.update_Hs_π2!(Hs_π2, expansion_order)
Ts = zeros(FastMultipole.length_Ts(expansion_order))
eimϕs = zeros(2, expansion_order+2)
weights_tmp_1 = initialize_expansion(expansion_order, eltype(Ts))
weights_tmp_2 = initialize_expansion(expansion_order, eltype(Ts))
weights_tmp_3 = initialize_expansion(expansion_order, eltype(Ts))

# normalization
ζs_mag = zeros(FastMultipole.length_ζs(expansion_order))
FastMultipole.update_ζs_mag!(ζs_mag, 0, expansion_order)
ηs_mag = zeros(FastMultipole.length_ηs(expansion_order))
FastMultipole.update_ηs_mag!(ηs_mag, 0, expansion_order)

# perform transformation
lamb_helmholtz = Val(false)
FastMultipole.multipole_to_local!(target_expansion, target_branch, multipole_expansion, branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, FastMultipole.M̃, FastMultipole.L̃, expansion_order, lamb_helmholtz)
FastMultipole.multipole_to_local!(target_expansion_point, target_branch_point, expansion_point, branch_point, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, FastMultipole.M̃, FastMultipole.L̃, expansion_order, lamb_helmholtz)

gradient_n_m = FastMultipole.initialize_gradient_n_m(expansion_order)
ϕ_l2b, v_l2b, g_l2b = FastMultipole.evaluate_local(xt - target_center, harmonics, gradient_n_m, target_expansion, expansion_order, lamb_helmholtz, DerivativesSwitch())

# repeat for point
Hs_π2 = [1.0]
FastMultipole.update_Hs_π2!(Hs_π2, expansion_order)
Ts = zeros(FastMultipole.length_Ts(expansion_order))
eimϕs = zeros(2, expansion_order+2)
weights_tmp_1 = initialize_expansion(expansion_order, eltype(Ts))
weights_tmp_2 = initialize_expansion(expansion_order, eltype(Ts))
weights_tmp_3 = initialize_expansion(expansion_order, eltype(Ts))
ζs_mag = zeros(FastMultipole.length_ζs(expansion_order))
FastMultipole.update_ζs_mag!(ζs_mag, 0, expansion_order)
ηs_mag = zeros(FastMultipole.length_ηs(expansion_order))
FastMultipole.update_ηs_mag!(ηs_mag, 0, expansion_order)

ϕ_point, v_point, g_point = FastMultipole.evaluate_local(xt - target_center, harmonics, gradient_n_m, target_expansion_point, expansion_order, lamb_helmholtz, DerivativesSwitch())

@test isapprox(v_check, v_l2b; atol=1e-12)

end

@testset "body-to-multipole: vortex panel" begin

# vortex panel
x1 = SVector{3,Float64}(0.0,0.29,0.0)
x2 = x1 + SVector{3,Float64}(0.0,0.03,0.0) * 10
x3 = x2 + SVector{3,Float64}(-0.05,0,0) * 10
vertices = SVector(x1,x2,x3)
x = [vertices]
q = SVector{3}(0,1.0,0.0)
system = VortexPanels(x, [q])
buffer = FastMultipole.system_to_buffer(system)

xt = SVector{3}(0.2, 0.3, 5.7)

normal = FastMultipole.get_normal(buffer,system,1)
strength = FastMultipole.get_strength(buffer,system,1)
centroid = FastMultipole.get_position(buffer,1)
#ϕ_check, v_check, g_check = induced(xt, vertices, normal, SVector{1}(norm(strength)), centroid, Panel{Vortex}, DerivativesSwitch(true,true,true))

nodes = zeros(3,3)
nodes[:,1] .= x1
nodes[:,2] .= x2
nodes[:,3] .= x3
panel = [1,2,3]
gammat = q[1]
gammao = q[2]
targets = zeros(3,1)
targets[:,1] .= xt
out = zeros(3,1)
U_constant_vortexsheet(nodes, panel,
    gammat, gammao,
    targets, out;
    dot_with=nothing,
    cutoff=1e-14, offset=1e-8
)
v_check = SVector(out[:,1]...)

expansion_order = 10
bodies_index, n_branches, branch_index, i_parent, i_leaf_index = 1:1, 0, 1:0, 0, 0
center = centroid + SVector{3}(0.1,0.2,0.08)
radius = 0.0
box = SVector{3}(0.0,0.0,0.0)

branch = Branch(bodies_index, n_branches, branch_index, i_parent, i_leaf_index, center, radius, box)
harmonics = initialize_harmonics(expansion_order)
multipole_expansion = initialize_expansion(expansion_order)
body_to_multipole!(Panel{Vortex}, system, multipole_expansion, buffer, branch.center, 1:1, harmonics, expansion_order)

ϕnm_panel = multipole_expansion[1,1,:] .+ multipole_expansion[2,1,:] .* im
χnm_panel = multipole_expansion[1,2,:] .+ multipole_expansion[2,2,:] .* im

# equivalent point vortex
x_point = centroid
area = norm(cross(x2-x1,x3-x1))/2
q_point = q * area

system_point = Vortons([x_point], [q_point])
buffer_point = FastMultipole.system_to_buffer(system_point)

branch_point = Branch(bodies_index, n_branches, branch_index, i_parent, i_leaf_index, center, radius, box)
harmonics_point = initialize_harmonics(expansion_order)
expansion_point = initialize_expansion(expansion_order)
body_to_multipole!(Point{Vortex}, system_point, expansion_point, buffer_point, branch_point.center, 1:1, harmonics_point, expansion_order)

ϕnm_point = expansion_point[1,1,:] .+ expansion_point[2,1,:] .* im
χnm_point = expansion_point[1,2,:] .+ expansion_point[2,2,:] .* im

# evaluate local expansion
target_center = xt + SVector{3}(0.0001, -0.0002, 0.003)
target_branch = Branch(bodies_index, n_branches, branch_index, i_parent, i_leaf_index, target_center, radius, box)
target_expansion = initialize_expansion(expansion_order)
target_branch_point = Branch(bodies_index, n_branches, branch_index, i_parent, i_leaf_index, target_center, radius, box)
target_expansion_point = initialize_expansion(expansion_order)

# preallocate containers
Hs_π2 = [1.0]
FastMultipole.update_Hs_π2!(Hs_π2, expansion_order)
Ts = zeros(FastMultipole.length_Ts(expansion_order))
eimϕs = zeros(2, expansion_order+2)
weights_tmp_1 = initialize_expansion(expansion_order, eltype(Ts))
weights_tmp_2 = initialize_expansion(expansion_order, eltype(Ts))
weights_tmp_3 = initialize_expansion(expansion_order, eltype(Ts))

# normalization
ζs_mag = zeros(FastMultipole.length_ζs(expansion_order))
FastMultipole.update_ζs_mag!(ζs_mag, 0, expansion_order)
ηs_mag = zeros(FastMultipole.length_ηs(expansion_order))
FastMultipole.update_ηs_mag!(ηs_mag, 0, expansion_order)

# perform transformation
lamb_helmholtz = Val(true)
FastMultipole.multipole_to_local!(target_expansion, target_branch, multipole_expansion, branch, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, FastMultipole.M̃, FastMultipole.L̃, expansion_order, lamb_helmholtz)
FastMultipole.multipole_to_local!(target_expansion_point, target_branch_point, expansion_point, branch_point, weights_tmp_1, weights_tmp_2, weights_tmp_3, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, FastMultipole.M̃, FastMultipole.L̃, expansion_order, lamb_helmholtz)

gradient_n_m = FastMultipole.initialize_gradient_n_m(expansion_order)
ϕ_l2b, v_l2b, g_l2b = FastMultipole.evaluate_local(xt - target_center, harmonics, gradient_n_m, target_expansion, expansion_order, lamb_helmholtz, DerivativesSwitch())

# repeat for point
Hs_π2 = [1.0]
FastMultipole.update_Hs_π2!(Hs_π2, expansion_order)
Ts = zeros(FastMultipole.length_Ts(expansion_order))
eimϕs = zeros(2, expansion_order+2)
weights_tmp_1 = initialize_expansion(expansion_order, eltype(Ts))
weights_tmp_2 = initialize_expansion(expansion_order, eltype(Ts))
weights_tmp_3 = initialize_expansion(expansion_order, eltype(Ts))
ζs_mag = zeros(FastMultipole.length_ζs(expansion_order))
FastMultipole.update_ζs_mag!(ζs_mag, 0, expansion_order)
ηs_mag = zeros(FastMultipole.length_ηs(expansion_order))
FastMultipole.update_ηs_mag!(ηs_mag, 0, expansion_order)

ϕ_point, v_point, g_point = FastMultipole.evaluate_local(xt - target_center, harmonics, gradient_n_m, target_expansion_point, expansion_order, lamb_helmholtz, DerivativesSwitch())

@test isapprox(v_check, v_l2b; atol=1e-12)

end

