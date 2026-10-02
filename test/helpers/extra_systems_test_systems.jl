# Systems shared by the extra target/source checks: the host checks in
# extra_systems_test.jl and the device-vs-host checks in
# gpu/ka_extra_systems_correctness.jl. `segments` is a pure source (a straight
# vortex filament with the extra-source contract), `probes` a pure target.

relerr(a, b) = (d = maximum(abs.(Array(a) .- Array(b))); s = maximum(abs.(Array(b)));
                s == 0 ? d : d / s)

# every generator draws in Float64 and casts, so the Float32 systems and their
# Float64 references share positions (the Float32/Float64 random streams differ)
make_system(seed, n, TF) = (Random.seed!(seed);
    VortexParticles(TF.(rand(3, n)), TF.(randn(3, n) ./ n), zeros(TF, n);
        potential=zeros(TF, 13, n), gradient_stretching=zeros(TF, 6, n)))

#------- a straight vortex segment as an extra source -------#

struct TestSegments{TF}
    r1::Vector{SVector{3,TF}}
    r2::Vector{SVector{3,TF}}
    gamma::Vector{TF}
end
function make_segments(seed, ns, TF)
    Random.seed!(seed)
    r1 = [SVector{3,TF}(rand(3)) for _ in 1:ns]
    r2 = [r1[i] + SVector{3,TF}(0.05 .* randn(3)) for i in 1:ns]
    TestSegments(r1, r2, TF.(randn(ns) ./ ns))
end
FM.get_n_bodies(s::TestSegments) = length(s.gamma)
FM.data_per_body(::TestSegments) = 12
FM.strength_dims(::TestSegments) = 1
FM.has_vector_potential(::TestSegments) = true
FM.get_position(s::TestSegments, i) = (s.r1[i] + s.r2[i]) / 2
function FM.source_system_to_buffer!(buffer, ib, s::TestSegments, i)
    c = (s.r1[i] + s.r2[i]) / 2
    buffer[1:3, ib] .= c
    buffer[4, ib] = norm(s.r2[i] - s.r1[i]) / 2
    buffer[5, ib] = s.gamma[i]
    buffer[6:8, ib] .= s.r1[i]
    buffer[9:11, ib] .= s.r2[i]
    buffer[12, ib] = 0
end
struct SegmentKernel <: FM.AbstractDirectKernel end
FM.direct_kernel(::TestSegments) = SegmentKernel()

# singular straight segment, Biot-Savart
@inline function FM._extra_pair_ug(::SegmentKernel, tx, ty, tz, buf, j)
    T = typeof(tx)
    @inbounds begin
        g = buf[5, j]
        ax = tx - buf[6, j]; ay = ty - buf[7, j]; az = tz - buf[8, j]
        bx = tx - buf[9, j]; by = ty - buf[10, j]; bz = tz - buf[11, j]
    end
    cx = ay * bz - az * by; cy = az * bx - ax * bz; cz = ax * by - ay * bx
    c2 = cx * cx + cy * cy + cz * cz
    na = sqrt(ax * ax + ay * ay + az * az); nb = sqrt(bx * bx + by * by + bz * bz)
    if c2 <= zero(T) || na <= zero(T) || nb <= zero(T)
        return (zero(T), zero(T), zero(T), zero(T))
    end
    r0x = ax - bx; r0y = ay - by; r0z = az - bz
    s = (r0x * ax + r0y * ay + r0z * az) / na - (r0x * bx + r0y * by + r0z * bz) / nb
    k = g * s / (T(4) * T(pi) * c2)
    return (zero(T), k * cx, k * cy, k * cz)
end

# Float64 reference loop through the same functor
function segments_on_points(segs::TestSegments, pts::AbstractMatrix)
    buf = FM._radix_extra_source_buffer(Float64, segs)
    out = zeros(Float64, 4, size(pts, 2))
    FM._host_targets_from_extra_source!(out, SegmentKernel(), Float64.(pts), size(pts, 2),
        buf, Val(false))
    return out[2:4, :]
end

particle_positions(sys) = reduce(hcat, [Vector(b.position) for b in sys.bodies])
probe_velocity(p) = reduce(hcat, [Vector(g) for g in p.gradient])

function make_probes(seed, nprobe, TF)
    Random.seed!(seed)
    p = FM.ProbeSystem(nprobe, TF)
    for i in 1:nprobe
        p.position[i] = SVector{3,TF}(rand(3))
    end
    p
end
