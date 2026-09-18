# Test systems shared by ka_extra_tree_correctness.jl and its debug script.
#--- a filament system packed exactly as a vortex-filament source is ---#
struct Segs{TF}
    r1::Vector{SVector{3,TF}}
    r2::Vector{SVector{3,TF}}
    gamma::Vector{TF}
    core::Vector{TF}
end
FM.get_n_bodies(s::Segs) = length(s.gamma)
FM.data_per_body(::Segs) = 15
FM.strength_dims(::Segs) = 3
FM.has_vector_potential(::Segs) = true
FM.body_type(::Segs) = FM.Filament{FM.Vortex}
FM.get_position(s::Segs, i) = (s.r1[i] + s.r2[i]) / 2
function FM.source_system_to_buffer!(b, ib, s::Segs, i)
    r1 = s.r1[i]; r2 = s.r2[i]; d = r2 - r1; len = norm(d)
    b[1:3, ib] .= (r1 + r2) / 2
    b[4, ib] = len / 2 + s.core[i]
    b[5:7, ib] .= iszero(len) ? zero(d) : s.gamma[i] .* d ./ len
    b[8:10, ib] .= r1
    b[11:13, ib] .= r2
    b[14, ib] = s.core[i]
    b[15, ib] = s.gamma[i]
end
struct SegKernel <: FM.AbstractDirectKernel end
FM.direct_kernel(::Segs) = SegKernel()
FM._emits_potential(::SegKernel) = false
@inline function FM._extra_pair_ug(::SegKernel, tx, ty, tz, buf, j)
    T = typeof(tx)
    @inbounds begin
        g = T(buf[15, j]); cs = T(buf[14, j])
        ax = tx - T(buf[8, j]); ay = ty - T(buf[9, j]); az = tz - T(buf[10, j])
        bx = tx - T(buf[11, j]); by = ty - T(buf[12, j]); bz = tz - T(buf[13, j])
    end
    r1n = sqrt(ax*ax + ay*ay + az*az); r2n = sqrt(bx*bx + by*by + bz*bz)
    nx = ay*bz - az*by; ny = az*bx - ax*bz; nz = ax*by - ay*bx
    rdot = ax*bx + ay*by + az*bz
    cs2 = cs*cs; r1s = r1n*r1n; r2s = r2n*r2n
    den = nx*nx + ny*ny + nz*nz + cs2*(r1n-r2n)*(r1n-r2n)
    den <= eps(T)*r1s*r2s && return (zero(T), zero(T), zero(T), zero(T))
    f2 = (r1s - rdot)/sqrt(r1s + cs2) + (r2s - rdot)/sqrt(r2s + cs2)
    k = g * (one(T)/(T(4)*T(pi))) * f2 / den
    return (zero(T), k*nx, k*ny, k*nz)
end

#--- a point system, to check the slab layout against the trusted kernel ---#
struct PV{TF}
    x::Vector{SVector{3,TF}}
    g::Vector{SVector{3,TF}}
end
FM.get_n_bodies(p::PV) = length(p.g)
FM.data_per_body(::PV) = 8
FM.strength_dims(::PV) = 3
FM.has_vector_potential(::PV) = true
FM.body_type(::PV) = FM.Point{FM.Vortex}
FM.get_position(p::PV, i) = p.x[i]
FM.source_system_to_buffer!(b, ib, p::PV, i) =
    (b[1:3, ib] .= p.x[i]; b[4, ib] = 0; b[5:7, ib] .= p.g[i]; b[8, ib] = 0)

