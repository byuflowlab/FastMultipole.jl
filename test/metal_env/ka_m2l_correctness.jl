# Isolated M2L correctness check for the ext/FastMultipoleKAExt.jl KA M2L building
# blocks (ka_gather_values!, ka_resident_m2l_concat_apply!, reusing M2M's
# ka_gather_rotate_z!/ka_stacked_y_dense!/ka_gather_rows!/ka_rotate_z_scatter_accumulate!),
# the KA-ported form of the real production resident M2L path
# (_launch_resident_m2l_concat!, src/translate_batched.jl:3828, the ConcatenatedFixedZM2L
# strategy -- confirmed the real GPU path via RadixFMMCache, see session notes). Dispatch
# entry point is FastMultipole.ka_m2l_operator_batch!, which builds a single-chunk
# ResidentM2LConcatPlan on the fly, one geometry class per (source i -> target i) route
# (no tree required -- isolated per-route check, per the ka-migration plan's step-3 scope)
# and compares against the CPU non-resident m2l_operator_batch! (MaterializedYRotationM2L).
include("ka_backend.jl")
using KernelAbstractions, FastMultipole, StaticArrays, Random
using FastMultipole: FlatCoefficientBuffer, M2LOperatorScratch, OperatorInvariantCache,
                     MaterializedYRotationM2L

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

            # Device (Metal) path via the real ka_m2l_operator_batch! dispatch
            sources_metal = to_metal(sources)
            targets_metal = to_metal(FlatCoefficientBuffer(TF, cache.basis_info, nbatch))

            FastMultipole.ka_m2l_operator_batch!(op, targets_metal, sources_metal,
                                                 phis, thetas, rs,
                                                 cache, scratch, lh)

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
            # matching the "relerr" convention already used to report this kernel chain's
            # Metal-hardware accuracy in ext/FastMultipoleKAExt.jl's header.
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
