# Isolated per-route gate for the production KA M2L driver
# `ka_resident_m2l_concat_apply!` (the KA counterpart of the host
# `_launch_resident_m2l_concat!`, the ConcatenatedFixedZM2L plan every device cache runs). A single-chunk
# ResidentM2LConcatPlan is built below with one geometry class per
# (source i -> target i) route, so arbitrary (r, theta, phi) are exercised without a
# tree. The reference is the host per-column pipeline `m2l_operator_batch!`
# (MaterializedYRotationM2L; production use: building DenseTranslationM2L class
# matrices), an independent factorization of the same M2L math.
include("ka_backend.jl")
using KernelAbstractions, FastMultipole, StaticArrays, Random
using FastMultipole: FlatCoefficientBuffer, M2LOperatorScratch, OperatorInvariantCache,
                     MaterializedYRotationM2L

const FM = FastMultipole
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)

# One ResidentM2LConcatPlan + the M2L-only slice of a ResidentOperatorWorkspace on
# `exemplar`'s backend: `nbatch` independent (i -> i) routes, one class per route.
# `offsets` only sizes the class count; the per-class (r, theta, phi) tables are
# overwritten with the test angles right after construction.
function build_m2l_concat_plan_and_workspace(exemplar, ::Type{TF}, invariant_cache,
        phis_host::Vector{TF}, thetas_host::Vector{TF}, rs_host::Vector{TF},
        ::Val{LH}) where {TF,LH}
    basis_info = invariant_cache.basis_info
    B = typeof(basis_info.basis)
    nbatch = length(phis_host)
    P_phi = basis_info.orders.P_phi
    P_active = basis_info.orders.P_active

    offsets = [SVector{3,Int}(i, 0, 0) for i in 1:nbatch]
    plan = FM.ResidentM2LConcatPlan(TF, basis_info, exemplar,
        FM.ConcatenatedFixedZM2L(nbatch), invariant_cache, offsets, one(TF), nbatch)
    copyto!(plan.phis, phis_host)
    copyto!(plan.thetas, thetas_host)
    copyto!(plan.rs, rs_host)
    copyto!(plan.invrs, inv.(rs_host))
    copyto!(plan.route_class, Int32.(1:nbatch))

    phi_flat_idx = FM._array_like_vector(exemplar, Int, FM._degree_major_to_flat_indices(P_phi))
    chi_flat_idx = LH ?
        FM._array_like_vector(exemplar, Int, FM._degree_major_to_flat_indices(P_active)) :
        FM._array_like_vector(exemplar, Int, Int[])
    maps_phi = FM.DegreeMajorMaps(TF, P_phi, exemplar)
    maps_chi = LH ? FM.DegreeMajorMaps(TF, P_active, exemplar) : maps_phi

    empty_sm() = similar(exemplar, TF, 0, 0)
    ws = FM.ResidentOperatorWorkspace{TF,B,LH}(
        basis_info, phi_flat_idx, chi_flat_idx, maps_phi, maps_chi,
        nothing, nothing, nothing, nothing, nothing,
        empty_sm(), empty_sm(), empty_sm(), empty_sm(),
        empty_sm(), empty_sm(), empty_sm(), empty_sm(),
        empty_sm(), empty_sm(),
        nothing, nothing, nothing, nothing, nothing, plan,
        nothing, nothing,
    )
    return plan, ws
end

# `FlatCoefficientBuffer.phi .= devarray(host)` silently no-ops (broadcasting a
# host-array destination from a device-array source does not perform the transfer);
# building the buffer directly over device arrays via the struct's default inner
# constructor is the correct way to get a Metal-backed FlatCoefficientBuffer.
function to_metal(buf::FlatCoefficientBuffer{TF,A,B,LH}) where {TF,A,B,LH}
    phi_dev = devarray(buf.phi)
    chi_dev = devarray(buf.chi)
    return FlatCoefficientBuffer{TF,typeof(phi_dev),B,LH}(phi_dev, chi_dev, buf.basis_info)
end

function to_cpu(buf::FlatCoefficientBuffer{TF,A,B,LH}) where {TF,A,B,LH}
    return FlatCoefficientBuffer{TF,Matrix{TF},B,LH}(Array(buf.phi), Array(buf.chi), buf.basis_info)
end

println("Starting KA M2L correctness test...")
if !dev_functional()
    println("$(DEV_NAME) not functional; skipping")
    exit(0)
end

seed = 42
for P in (4, 8)
    for LHbool in (true, false)
        for nbatch in (5, 10)
            Random.seed!(seed)
            TF = Float32
            lh = Val(LHbool)
            phis = rand(TF, nbatch)
            thetas = acos.(2 .* rand(TF, nbatch) .- 1)  # uniform on [0, π]
            rs = TF(1.5) .+ TF(1.0) .* rand(TF, nbatch)  # independent per-route separation (M2L, unlike M2M, has no shared-radius constraint)

            cache = OperatorInvariantCache(TF, P, lh)
            scratch = M2LOperatorScratch(TF, cache.basis_info, nbatch)

            sources = FlatCoefficientBuffer(TF, cache.basis_info, nbatch)
            targets_cpu = FlatCoefficientBuffer(TF, cache.basis_info, nbatch)

            for j in 1:nbatch
                for i in 1:size(sources.phi, 1)
                    sources.phi[i, j] = randn(TF)
                end
                if LHbool
                    for i in 1:size(sources.chi, 1)
                        sources.chi[i, j] = randn(TF)
                    end
                end
            end
            # Zero the physically-unused m=0-imaginary "padding" slots of the flat
            # compressed-complex layout (present in the storage but never part of the
            # degree-major dof set `phi_flat_idx`/`chi_flat_idx` address): real multipole
            # data from body_to_multipole!/M2M never populates these, and leaving them
            # random makes the resident path (which structurally never reads them) diverge
            # from the per-column path (which does, folding the garbage into physically
            # meaningful outputs through the rotation math).
            P_phi = cache.basis_info.orders.P_phi
            phi_idx = FastMultipole._degree_major_to_flat_indices(P_phi)
            phi_mask = falses(size(sources.phi, 1)); phi_mask[phi_idx] .= true
            sources.phi[.!phi_mask, :] .= 0
            if LHbool
                P_active = cache.basis_info.orders.P_active
                chi_idx = FastMultipole._degree_major_to_flat_indices(P_active)
                chi_mask = falses(size(sources.chi, 1)); chi_mask[chi_idx] .= true
                sources.chi[.!chi_mask, :] .= 0
            end

            op = MaterializedYRotationM2L()

            # CPU reference (non-resident, per-column materialized-rotation path)
            FastMultipole.m2l_operator_batch!(op, targets_cpu, sources,
                                              phis, thetas, rs,
                                              cache, scratch, lh)

            # Device path: the production KA concat driver over a one-chunk plan
            sources_metal = to_metal(sources)
            targets_metal = to_metal(FlatCoefficientBuffer(TF, cache.basis_info, nbatch))

            plan, ws = build_m2l_concat_plan_and_workspace(targets_metal.phi, TF, cache,
                phis, thetas, rs, lh)
            idx = FM._array_like_vector(targets_metal.phi, Int, collect(1:nbatch))
            ext.ka_resident_m2l_concat_apply!(targets_metal, sources_metal, ws, idx, idx, nbatch)
            KernelAbstractions.synchronize(DEV_BACKEND)

            targets_metal_cpu = to_cpu(targets_metal)

            phi_err = maximum(abs.(targets_cpu.phi .- targets_metal_cpu.phi))
            phi_relerr = phi_err / maximum(abs.(targets_cpu.phi))
            chi_err = 0.0
            chi_relerr = 0.0
            if LHbool
                chi_err = maximum(abs.(targets_cpu.chi .- targets_metal_cpu.chi))
                chi_relerr = chi_err / maximum(abs.(targets_cpu.chi))
            end

            # Absolute Float32 tolerance is too tight for chi (LH row-mix adds an extra
            # gather + GEMM pass over phi's error budget); gate on relative error instead,
            # matching the relative-error convention used to report this kernel chain's
            # Float32 accuracy on Metal hardware.
            rtol = 1e-4  # matches this repo's scalar-potential Float32 tolerance convention
            if phi_relerr >= rtol
                error("P=$P, LHbool=$LHbool, nbatch=$nbatch: phi_err=$phi_err, phi_relerr=$phi_relerr >= $rtol")
            end
            if LHbool && chi_relerr >= rtol
                error("P=$P, LHbool=$LHbool, nbatch=$nbatch: chi_err=$chi_err, chi_relerr=$chi_relerr >= $rtol")
            end

            println("✓ P=$P, LHbool=$LHbool, nbatch=$nbatch: phi_err=$phi_err (relerr=$phi_relerr), chi_err=$chi_err (relerr=$chi_relerr)")
        end
    end
end

println("\n✓✓✓ All KA M2L correctness tests passed on $(DEV_NAME)! ✓✓✓")
