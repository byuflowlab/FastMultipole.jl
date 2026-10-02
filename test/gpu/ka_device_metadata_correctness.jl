# Metadata rows of a DeviceResident target on a device radix cache: delivered
# through metadata_to_device_buffer!, rejected before any work without it, and
# droppable with metadata=0. Outputs must not depend on the metadata rows.
include("ka_backend.jl")
using FastMultipole, Random, Test
using FastMultipole.StaticArrays
const FM = FastMultipole

if !dev_functional()
    println("$(DEV_NAME) not functional; skipping")
    exit(0)
end

# device-resident point sources; `meta_seen` records the metadata rows the
# framework hands buffer_to_target!
struct DevMetaPoints{M}
    data::M          # x, y, z, radius, strength
    gradient::M
    meta::M
    meta_seen::M
end
struct DevNoHookPoints{M}
    inner::DevMetaPoints{M}
end
function dev_points(seed, n)
    Random.seed!(seed)
    data = vcat(rand(Float32, 3, n), fill(0.001f0, 1, n), randn(Float32, 1, n) ./ n)
    DevMetaPoints(devarray(data), devarray(zeros(Float32, 3, n)),
        devarray(rand(Float32, 2, n)), devarray(zeros(Float32, 2, n)))
end
for T in (:DevMetaPoints, :DevNoHookPoints)
    @eval begin
        pts(s::$T) = s isa DevNoHookPoints ? s.inner : s
        FM.residency(::$T) = FM.DeviceResident()
        FM.device_backend(::$T) = DEV_BACKEND
        FM.get_n_bodies(s::$T) = size(pts(s).data, 2)
        FM.data_per_body(::$T) = 5
        FM.strength_dims(::$T) = 1
        FM.has_vector_potential(::$T) = false
        FM.body_type(::$T) = FM.Point{FM.Source}
        FM.metadata_per_body(::$T) = 2
        Base.eltype(::$T) = Float32
        function FM.source_to_buffer!(buffer, s::$T, sort_index)
            view(buffer, 1:5, :) .= pts(s).data
            return buffer
        end
        function FM.buffer_to_target!(s::$T, buffer, switch, sort_index)
            pts(s).gradient .= view(buffer, FM.gradient_range(switch), :)
            isempty(FM.metadata_range(switch)) ||
                (pts(s).meta_seen .= view(buffer, FM.metadata_range(switch), :))
            return s
        end
    end
end
FM.metadata_to_device_buffer!(buffer, switch, s::DevMetaPoints) =
    (view(buffer, FM.metadata_range(switch), :) .= s.meta; buffer)

n = 4000
cache_of(s) = RadixFMMCache(s; expansion_order=3, ell=3, device=true,
    bounds=(SVector(-0.01, -0.01, -0.01), 1.02))

npass = 0; nfail = 0
check(ok, msg) = (ok ? (global npass += 1; println("  PASS  ", msg)) :
                       (global nfail += 1; println("  FAIL  ", msg)))

sys = dev_points(1, n)
fmm!(sys, cache_of(sys); gradient=true)
check(Array(sys.meta_seen) == Array(sys.meta), "metadata rows delivered through the hook")
g_meta = Array(sys.gradient)
check(all(isfinite, g_meta) && any(!iszero, g_meta), "outputs finite and nonzero")

sys.gradient .= 0
fmm!(sys, cache_of(sys); gradient=true, metadata=0)
# device runs are not bit-reproducible (unstable counting sort), so compare
# at Float32 round-off
relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))
e = relerr(Array(sys.gradient), g_meta)
check(e < 1e-5, "metadata=0 gives the same outputs (relerr $e)")

nohook = DevNoHookPoints(dev_points(1, n))
err = try
    fmm!(nohook, cache_of(nohook); gradient=true); nothing
catch e
    e
end
check(err isa ArgumentError && occursin("metadata=0", err.msg),
    "no hook: ArgumentError that names metadata=0")
fmm!(nohook, cache_of(nohook); gradient=true, metadata=0)
e = relerr(Array(nohook.inner.gradient), g_meta)
check(e < 1e-5, "no hook, metadata=0: same outputs (relerr $e)")

println("\nDevice metadata over $(DEV_NAME): $npass passed, $nfail failed")
nfail == 0 || error("device metadata gate failed")
println("device metadata gate passed")
