# The REAL call: FLOWVPM's `UJ_fmm` on a Metal-backed ParticleField, reaching
# KA through `fmm!` dispatch -- not through `ext.ka_fmm!`.
#
# Everything here is FLOWVPM's own entry point. The only device-specific line
# is the `arraytype`, which is what makes `residency(pfield)` DeviceResident
# and sends the coupling down the device path (FLOWVPM_fmm_radix.jl:51).
#
# `m2l_strategy=:concat`: FLOWVPM defaults to `:dense`, which has no KA plan
# (the CUDA dense lifecycle is ~1500 lines resting on CUBLAS batched GEMM).
# `:concat` is an equally supported production value.
#
# The device field is built host-side and copied, following
# FLOWVPM/test/runtests_gpu.jl:20-25 -- `add_particle` scalar-writes, which is
# not something to do to device memory.

include("ka_backend.jl")
include("pipeline_field.jl")

using FastMultipole, Printf
const FM = FastMultipole
const V = FLOWVPM

dev_functional() || (println("$(DEV_NAME) not functional; skipping"); exit(0))
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext === nothing && error("the KA extension is not loaded")

println("=== real UJ_fmm through fmm! dispatch ===")
println("backend registered: ", FM.radix_device_backend_available(),
        "  name=", repr(FM.radix_device_backend_name()))
flush(stdout)

const TF = Float32
const STEP = 36
const P = 5

function device_copy_of(host)
    d = V.ParticleField(host.maxparticles, TF; arraytype=devmatrix, np=host.np,
        fmm=V.FMM(; p=P + 1, autotune_p=false, autotune_ncrit=false,
                    autotune_reg_error=false, default_rho_over_sigma=1.0))
    d.particles .= devarray(host.particles)
    return d
end

const STRATEGY = Symbol(get(ENV, "UJ_M2L", "concat"))
println("m2l_strategy = ", STRATEGY); flush(stdout)

for np in (512, 2048, 8192)
    host = load_wake(STEP; np, TF, P)
    V.radix_fmm_settings!(host; m2l_strategy=STRATEGY)
    V.UJ_fmm(host)
    U_ref = Array(host.particles)[V.U_INDEX, 1:host.np]

    fresh = load_wake(STEP; np, TF, P)
    dev = device_copy_of(fresh)
    V.radix_fmm_settings!(dev; m2l_strategy=STRATEGY)

    t = @elapsed V.UJ_fmm(dev)
    U_dev = Array(dev.particles)[V.U_INDEX, 1:dev.np]

    relerr = maximum(abs.(U_dev .- U_ref)) / maximum(abs.(U_ref))
    @printf("np=%-6d  %.4f s   relerr(U) = %.3e\n", np, t, relerr)
    flush(stdout)
end
