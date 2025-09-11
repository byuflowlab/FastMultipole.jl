# using Roots

struct VortexFilaments{TF}
    x::Matrix{SVector{3,TF}}
    strength::Vector{SVector{3,TF}}
    core_size::Vector{TF}
    error_tolerance::Vector{TF}
    potential::Vector{TF}
    gradient::Vector{SVector{3,TF}}
    hessian::Vector{SMatrix{3,3,TF,9}}
end

# function VortexFilaments(filaments::VortexFilaments)
#     core_size = fill(1e-3, length(filaments.potential))
#     error_tolerance = fill(1e-4, length(filaments.potential))
#     return VortexFilaments(filaments.x, filaments.strength, core_size, error_tolerance, filaments.potential, filaments.gradient, filaments.gradient)
# end

# function viz(fname, vortex_filaments::VortexFilaments)
#     # create points
#     pts = zeros(3, length(vortex_filaments.strength)+1)
#     for i in 1:length(vortex_filaments.strength)
#         pts[:,i] .= vortex_filaments.x[1,i]
#     end
#     pts[:,end] .= vortex_filaments.x[2,end]

#     # create lines
#     lines = [MeshCell(PolyData.Lines(), (i, i + 1)) for i in 1:length(vortex_filaments.strength)]

#     vtk_grid(fname, pts, lines) do vtk
#         vtk["strength"] = vortex_filaments.strength
#         vtk["vector"] = vortex_filaments.gradient
#     end
# end

function VortexFilaments(x, strength::Vector{SVector{3,TF}};
        core_size = fill(1e-2, size(x,2)),
        error_tolerance = fill(1e-4, size(x,2)),
        potential = zeros(size(x,2)),
        gradient = zeros(SVector{3,TF},size(x,2)),
        hessian = zeros(SMatrix{3,3,TF,9},size(x,2))
    ) where TF

    return VortexFilaments(x, strength, core_size, error_tolerance, potential, gradient, hessian)
end

function generate_inline_filaments(n_bodies; strength, noise=false, noise_strength=0.1)
    Random.seed!(123)
    pts = zeros(SVector{3,Float64}, 2, n_bodies)
    strength_vec = zeros(SVector{3,Float64}, size(pts,2))
    x1 = SVector{3}(0.0, 0.0, 0.0)
    i_filament = 1

    dx = 5.0 / n_bodies
    for x in range(0, stop=5.0, length=n_bodies+1)
        if x > eps()
            x2 = SVector{3}(x,0.0,0.0)
            if noise
                x2 += SVector{3}(1.0-2*rand(), 1.0-2*rand(), 1.0-2*rand()) * dx * noise_strength
            end
            pts[1,i_filament] = x1
            pts[2,i_filament] = x2

            # determine strength
            Γ = x2 - x1
            Γ *= strength / norm(Γ)
            strength_vec[i_filament] = Γ

            # recurse
            x1 = x2
            i_filament += 1
        end
    end
    # create filaments
    core_size = fill(1e-2, size(x,2))
    error_tolerance = fill(1e-4, size(x,2))
    potential = zeros(length(strength_vec))
    gradient = zeros(SVector{3,Float64}, length(strength_vec))
    hessian = zeros(SMatrix{3,3,Float64,9}, length(strength_vec))

    return VortexFilaments(pts, strength_vec, core_size, error_tolerance, potential, gradient, hessian)
end

function refine_filaments(filaments::VortexFilaments, max_length)
    # determine how many we need
    n_filaments = 0
    for i in 1:length(filaments.strength)
        len = norm(filaments.x[2,i] - filaments.x[1,i])
        n_here = Int(div(len, max_length)) + 1
        n_filaments += n_here
    end

    # preallocate
    refined_filaments = zeros(SVector{3,Float64}, 2, n_filaments)
    strength_vec = zeros(SVector{3,Float64}, n_filaments)

    # create
    i_filament = 1
    for i in 1:length(filaments.strength)
        old_strength = filaments.strength[i]
        dx_vec = filaments.x[2,i] - filaments.x[1,i]
        len = norm(dx_vec)
        n_here = Int(div(len, max_length)) + 1

        x1 = filaments.x[1,i]
        dx = len / n_here
        dx_vec *= dx / len

        for j in 1:n_here
            x2 = x1 + dx_vec
            refined_filaments[1,i_filament] = x1
            refined_filaments[2,i_filament] = x2
            strength_vec[i_filament] = old_strength

            # recurse
            x1 = x2
            i_filament += 1
        end
    end

    @assert i_filament == n_filaments + 1

    # create filaments
    core_size = fill(1e-2, size(refined_filaments,2))
    error_tolerance = fill(1e-4, size(refined_filaments,2))
    potential = zeros(length(strength_vec))
    gradient = zeros(SVector{3,Float64}, length(strength_vec))
    hessian = zeros(SMatrix{3,3,Float64,9}, length(strength_vec))

    return VortexFilaments(refined_filaments, strength_vec, core_size, error_tolerance, potential, gradient, hessian)
end

function total_length(filaments::VortexFilaments)
    l = 0.0
    for i in 1:length(filaments.strength)
        x1 = filaments.x[1,i]
        x2 = filaments.x[2,i]
        this_l = norm(x2 - x1)
        l += this_l
    end

    return l
end

function generate_vortex_filaments(ntheta; nrings=2, r=1.0, dz=1.0, strength=1e-2)
    pts = zeros(SVector{3,Float64}, 2, (ntheta-1)*nrings)
    strength_vec = zeros(SVector{3,Float64}, size(pts,2))

    # create rings
    i_filament = 1
    for z in range(0, stop=dz, length=nrings)
        x1 = SVector{3}(1.0, 0.0, z)
        for theta in range(0,stop=2*pi, length=ntheta)
            if theta > 10*eps()
                # filament
                x2 = SVector{3}(cos(theta)*r, sin(theta)*r, z)
                pts[1,i_filament] = x1
                pts[2,i_filament] = x2

                # determine strength
                Γ = x2 - x1
                Γ = Γ * strength / norm(Γ)
                strength_vec[i_filament] = Γ

                # recurse
                x1 = x2
                i_filament += 1
            end
        end
    end

    # create filaments
    core_size = fill(1e-2, size(x,2))
    error_tolerance = fill(1e-4, size(x,2))
    potential = zeros(length(strength_vec))
    gradient = zeros(SVector{3,Float64}, length(strength_vec))
    hessian = zeros(SMatrix{3,3,Float64,9}, length(strength_vec))

    return VortexFilaments(pts, strength_vec, core_size, error_tolerance, potential, gradient, hessian)
end

function generate_filament_field(n_filaments, length_scale, seed=123; strength_scale=1/n_filaments)
    Random.seed!(seed)
    centers = rand(SVector{3,Float64}, n_filaments)
    pts = zeros(SVector{3,Float64}, 2, n_filaments)
    strength_vec= zeros(SVector{3,Float64}, n_filaments)
    for (i,center) in enumerate(centers)
        dx = (rand(SVector{3,Float64}) * 2 .- 1.0) * 0.5 * length_scale
        pts[1,i] = center - dx
        pts[2,i] = center + dx
        Γ = dx / norm(dx) * rand() * strength_scale
        strength_vec[i] = Γ
    end

    # create filaments
    core_size = fill(1e-2, size(pts,2))
    error_tolerance = fill(1e-4, size(pts,2))
    potential = zeros(length(strength_vec))
    gradient = zeros(SVector{3,Float64}, length(strength_vec))
    hessian = zeros(SMatrix{3,3,Float64,9}, length(strength_vec))

    return VortexFilaments(pts, strength_vec, core_size, error_tolerance, potential, gradient, hessian)
end

function minimum_distance(x1, x2, P)

    # direction vector of the segment
    Δx = x2 - x1

    # vector from x1 to P
    v = P - x1

    # scalar projection of v onto Δx
    dot_vd = dot(v, Δx)
    x2 = dot(Δx, Δx)
    t = dot_vd / x2

    # clamp t to ensure the closest point is within the segment
    t = clamp(t, zero(t), one(t))

    # get distance
    return norm(P - (x1 + t * Δx))
end

function vortex_filament_gauss_compressed(x1,x2,xt,q,core_size)

    # intermediate values
    r1 = xt - x1
    r2 = xt - x2
    nr1 = norm(r1)
    nr2 = norm(r2)
    rcross = cross(r1, r2)
    rdot = dot(r1, r2)

    # (mostly) singular case
    denom = nr1 * nr2 + rdot

    # check if we're at the midpoint
    if abs(denom) < eps(max(nr1, nr2)) # at the midpoint, so return zero
        return zero(SVector{3,eltype(x1)})
    end

    denom += 10*eps(max(nr1, nr2)) * (abs(denom) < 5 * eps(max(nr1, nr2)))
    nr1 += 10*eps(max(nr1, nr2)) * (abs(nr1) < 5 * eps(max(nr1, nr2)))
    nr2 += 10*eps(max(nr1, nr2)) * (abs(nr2) < 5 * eps(max(nr1, nr2)))
    Vhat = rcross / denom * (1/nr1 + 1/nr2)

    # regularize
    d = minimum_distance(x1, x2, xt)
    Vhat *= (1 - exp(-d*d*d / (core_size*core_size*core_size)))
    # @show (1 - exp(-d*d*d / (core_size*core_size*core_size)))

    if true in isnan.(Vhat)
        @show Vhat
        @show d
        @show core_size
        @show denom
        @show nr1
        @show nr2
    end
    return Vhat * q
end

function mysign(x)
    if abs(x) > zero(x)
        return sign(x)
    end

    return one(x)
end

function get_δ_try1(distance, core_size)
    δ = distance < core_size ? -distance + core_size : zero(distance)
    return δ
end

function get_δ(distance, core_size)
    δ = distance < core_size ? (distance-core_size) * (distance-core_size) : zero(distance)
    return δ
end

function vortex_filament_finite_core_2(x1,x2,xt,q,core_size)
    # intermediate values
    r1 = xt - x1
    r2 = xt - x2

    nr1 = norm(r1)
    nr2 = norm(r2)

    num = cross(r1, r2)
    denom = nr1 * nr2 + dot(r1, r2)

    # core size comes into play here
    distance_1 = norm((x1+x2)*0.5 - xt)
    δ1 = get_δ(distance_1, core_size)

    distance_2 = norm(x1 - xt)
    δ2 = get_δ(distance_2, core_size)

    distance_3 = norm(x2 - xt)
    δ3 = get_δ(distance_3, core_size)

    # desingularized terms
    f1 = num/(denom + δ1)
    f2 = 1/(nr1+δ2)
    f3 = 1/(nr2+δ3)

    # evaluate vector field
    V = (f1*(f2+f3))/(4*pi) * q

    return V
end

function vortex_filament_finite_core(x1,x2,xt,q,core_size)

    # intermediate values
    r1 = xt - x1
    r2 = xt - x2
    nr1 = norm(r1)
    nr2 = norm(r2)
    nr1nr2 = nr1*nr2
    rcross = cross(r1, r2)
    rdot = dot(r1, r2)

    if abs(rdot + nr1nr2) < eps(max(nr1, nr2)) # at the midpoint, so return zero
        return zero(typeof(r1))
    end

    r1s, r2s, εs = nr1*nr1, nr2*nr2, core_size*core_size
    f1 = rcross/(r1s*r2s - rdot*rdot + εs*(r1s + r2s - 2*nr1nr2))
    f2 = (r1s - rdot)/sqrt(r1s + εs) + (r2s - rdot)/sqrt(r2s + εs)
    Vhat = (f1*f2)/(4*pi)

    return Vhat * q

end

function vortex_filament(x1,x2,xt,q)
    r1 = xt - x1
    r2 = xt - x2

    nr1 = norm(r1)
    nr2 = norm(r2)

    num = cross(r1, r2)
    denom = nr1 * nr2 + dot(r1, r2)
    if abs(denom) < eps(max(nr1,nr2))
        return zero(SVector{3,eltype(x1)})
    end

    f1 = num/denom
    f2 = (1/nr1 + 1/nr2)

    # evaluate vector field
    V = (f1*f2)/(4*pi) * q

    return V
end

function reset!(filaments::VortexFilaments)
    filaments.potential .= zero(eltype(filaments.potential))
    for i in eachindex(filaments.potential)
        filaments.gradient[i] = SVector{3,Float64}(0.0,0,0)
        filaments.hessian[i] = zero(SMatrix{3,3,Float64,9})
    end
end

#------- compatibility functions -------#

function FastMultipole.source_system_to_buffer!(buffer, i_buffer, system::VortexFilaments, i_body)
    buffer[1:3,i_buffer] .= (system.x[1,i_body]  + system.x[2,i_body]) * 0.5
    
    # get regularization radius
    Γ = system.strength[i_body]
    # Γmag = norm(Γ)
    core_size = system.core_size[i_body]
    # if !iszero(Γmag)
    #     ε = system.error_tolerance[i_body]
    #     ρ = norm(system.x[2,i_body] - system.x[1,i_body]) * 0.5
    #     d_upper = sqrt(Γmag / (4 * π * ε))
    #     d = Roots.find_zero(d -> exp(-d*d*d / (core_size * core_size * core_size)) - 4 * π * d * d / Γmag * ε, (zero(d_upper), d_upper), Roots.Brent())
    #     buffer[4,i_buffer] = ρ + d
    # else
    #     buffer[4,i_buffer] = 0.0
    # end
    buffer[4,i_buffer] = system.core_size[i_body] + 0.5 * norm(system.x[2,i_body] - system.x[1,i_body])

    # remainding quantities
    buffer[5:7,i_buffer] .= Γ
    buffer[8:10,i_buffer] .= system.x[1,i_body]
    buffer[11:13,i_buffer] .= system.x[2,i_body]
    buffer[14,i_buffer] = core_size
end

function FastMultipole.strength_dims(system::VortexFilaments)
    return 3
end

FastMultipole.get_n_bodies(system::VortexFilaments) = length(system.strength)

FastMultipole.data_per_body(::VortexFilaments) = 14

Base.eltype(::VortexFilaments{TF}) where TF = TF

FastMultipole.get_position(system::VortexFilaments, i) = (system.x[1,i] + system.x[2,i]) * 0.5

function FastMultipole.reset!(system::VortexFilaments)
    system.potential .= zero(eltype(system.potential))
    system.gradient .= zero(eltype(system.gradient))
    system.hessian .= zero(eltype(system.hessian))
end

function FastMultipole.has_vector_potential(system::VortexFilaments)
    return true
end

function FastMultipole.direct!(target_system, target_index, derivatives_switch::DerivativesSwitch{PS,GS,HS}, source_system::VortexFilaments, source_buffer, source_index) where {PS,GS,HS}
    for i_source in source_index
        x1 = FastMultipole.get_vertex(source_buffer, source_system, i_source, 1)
        x2 = FastMultipole.get_vertex(source_buffer, source_system, i_source, 2)
        q = FastMultipole.get_strength(source_buffer, source_system, i_source)
        core_size = source_buffer[14, i_source]

        for i_target in target_index
            xt = FastMultipole.get_position(target_system, i_target)

            if GS
                # determine sign of q
                q_mag = norm(q) * sign(dot(q, x2-x1))

                # calculate vector field
                # v = vortex_filament(x1,x2,xt,q_mag)
                v = vortex_filament_finite_core_2(x1,x2,xt,q_mag,core_size)
                # v = vortex_filament_gauss_compressed(x1,x2,xt,q_mag,core_size)
                FastMultipole.set_gradient!(target_system, i_target, v)
            end
        end
    end
end

FastMultipole.body_to_multipole!(system::VortexFilaments, args...) = FastMultipole.body_to_multipole!(Filament{Vortex}, system, args...)

function FastMultipole.buffer_to_target_system!(target_system::VortexFilaments, i_target, derivatives_switch, target_buffer, i_buffer)

    # extract from buffer
    gradient = FastMultipole.get_gradient(target_buffer, i_buffer)

    # load into system
    target_system.gradient[i_target] += gradient

end

function viz_filament(fname, vortex_filaments::VortexFilaments)
    # create points
    pts = zeros(SVector{3,eltype(vortex_filaments)}, length(vortex_filaments.strength) * 2)
    ic = 1
    for i in 1:length(vortex_filaments.strength)
        pts[ic] = vortex_filaments.x[1,i]
        ic += 1
        pts[ic] = vortex_filaments.x[2,i]
        ic += 1
    end

    # create lines
    lines = [MeshCell(PolyData.Lines(), (2*i-1, 2*i)) for i in 1:length(vortex_filaments.strength)]

    # save as VTK
    vtk_grid(fname, pts, lines) do vtk
        vtk["strength"] = vortex_filaments.strength
        vtk["gradient"] = vortex_filaments.gradient
    end
end

#------- test -------#
#=
filaments = generate_vortex_filaments(100; nrings=50, r=1.0, dz=10.0, strength=1e-2)

direct!(filaments)
v_direct = deepcopy(filaments.gradient)

reset!(filaments)
optargs, target_tree, source_tree, m2l_list, direct_list, derivatives_switches, error_success = fmm!(filaments; leaf_size_source=10, multipole_acceptance=0.5, expansion_order=10)

v_fmm = deepcopy(filaments.gradient)

@show maximum(norm.(v_fmm - v_direct))
=#

#------- inline filaments -------#

#=
n_bodies=100
filaments = generate_inline_filaments(n_bodies; strength=1e-1, noise=true, noise_strength=0.01)

l = total_length(filaments)
max_length = l / n_bodies / 4
refined_filaments = refine_filaments(filaments, max_length)

direct!(filaments)
v_direct = deepcopy(filaments.gradient)

reset!(filaments)
direct!(filaments, refined_filaments)
v_direct_r = deepcopy(filaments.force)

@show maximum(norm.(v_direct - v_direct_r))

expansion_order = 15

reset!(filaments)
optargs, target_tree, source_tree, m2l_list, direct_list, derivatives_switches, error_success = fmm!(filaments; leaf_size_source=1, multipole_acceptance=0.4, expansion_order, ε_abs=nothing)

v_fmm = deepcopy(filaments.force)

@show maximum(norm.(v_fmm - v_direct))

# try refined filaments
reset!(filaments)
optargs, target_tree, source_tree, m2l_list, direct_list, derivatives_switches, error_success = fmm!(filaments, refined_filaments; leaf_size_source=1, multipole_acceptance=0.4, expansion_order, ε_abs=nothing)

v_fmm_r = deepcopy(filaments.force)

@show maximum(norm.(v_fmm_r - v_direct))
=#

#=
#------- rotor -------#

filaments = generate_rotor()
l = total_length(filaments)
n = size(filaments.x, 2)
# refined_filaments = refine_filaments(filaments, l / n / 20)

direct!(filaments)
v_direct = deepcopy(filaments.force)

# reset!(filaments)
# direct!(filaments, refined_filaments)
# v_direct_r = deepcopy(filaments.force)

# @show maximum(norm.(v_direct - v_direct_r))

reset!(filaments)
optargs, target_tree, source_tree, m2l_list, direct_list, derivatives_switches, error_success = fmm!(filaments; leaf_size_source=1, multipole_acceptance=0.4, expansion_order=20, ε_abs=nothing)

v_fmm = deepcopy(filaments.force)

@show maximum(norm.(v_fmm - v_direct))
=#

#=
#------- random filament field -------#

n_filaments = 200000
length_scale = 1.0 / n_filaments^(1/3)
filaments = generate_filament_field(n_filaments, length_scale; strength_scale=1/n_filaments)

# direct!(filaments)
optargs, target_tree, source_tree, m2l_list, direct_list, derivatives_switches, error_success = fmm!(filaments; leaf_size_source=20, multipole_acceptance=0.4, expansion_order=20, ε_abs=nothing)
v_direct = deepcopy(filaments.force)

reset!(filaments)
optargs, target_tree, source_tree, m2l_list, direct_list, derivatives_switches, error_success = fmm!(filaments; leaf_size_source=20, multipole_acceptance=0.4, expansion_order=10, ε_abs=nothing)

v_fmm = deepcopy(filaments.force)

@show maximum(norm.(v_fmm - v_direct))
=#

#=
#------- test vector functions -------#
function vector_comparison(ys;
    x1 = SVector{3}(0.0, 0.0, 0),
    x2 = SVector{3}(0.1, 0, 0),
    q = (x2 - x1) * 1.7308e-2,
    core_size = 1e-6)

    q_mag = norm(q)*sign(dot(q,x2-x1))

    vs_fc = zeros(length(ys))
    vs_fc2 = zeros(length(ys))
    vs_check = zeros(length(ys))

    for (i,y) in enumerate(ys)

        xt = SVector{3}(0.05,y,0.0)
        v_fc = vortex_filament_finite_core(x1,x2,xt,q_mag,core_size)
        v_fc2 = vortex_filament_finite_core_2(x1,x2,xt,q_mag,core_size)
        v_check = vortex_filament(x1,x2,xt,q_mag)

        vs_fc[i] = v_fc[3]
        vs_fc2[i] = v_fc2[3]
        vs_check[i] = v_check[3]
    end

    return vs_fc, vs_fc2, vs_check
end

ys = range(0.0, stop=1.0e-1, length=100000)
vs_fc, vs_fc2, vs_check = vector_comparison(ys; core_size=1e-2)

fig = figure("filament")
fig.clear()
fig.add_subplot(111, xlabel="distance", ylabel="vector magnitude")
ax = fig.get_axes()[0]
ax.plot(ys, vs_fc)
ax.plot(ys, vs_fc2, "--")
ax.plot(ys, vs_check, ":")
ax.legend(["finite core model 1", "finite core model 2", "infinite core"])
# ax.set_ylim([0.0, 300.0])
ax.set_yscale("log")
=#
