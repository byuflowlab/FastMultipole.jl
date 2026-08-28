# Isolated L2L correctness check for ext/FastMultipoleKAExt.jl's ka_l2l_operator_batch!.
# L2L reaches the exact same ka_resident_stage_group_apply! group-apply as M2M
# (src/translate_batched.jl's _resident_stage_group_apply!, reached in production via
# _launch_resident_l2l!), selected by kind=:l2l -- no new KA kernel needed. This test
# mirrors ka_m2m_correctness.jl but compares against the CPU non-resident
# l2l_operator_batch! (MaterializedYRotationL2L) reference instead.
using Metal, KernelAbstractions, FastMultipole, StaticArrays, Random
using FastMultipole: FlatCoefficientBuffer, L2LOperatorScratch, OperatorInvariantCache,
                     MaterializedYRotationL2L

# `FlatCoefficientBuffer.phi .= Metal.MtlArray(host)` silently no-ops (broadcasting a
# host-array destination from a device-array source does not perform the transfer);
# building the buffer directly over device arrays via the struct's default inner
# constructor is the correct way to get a Metal-backed FlatCoefficientBuffer.
function to_metal(buf::FlatCoefficientBuffer{TF,A,B,LH}) where {TF,A,B,LH}
    phi_dev = Metal.MtlArray(buf.phi)
    chi_dev = Metal.MtlArray(buf.chi)
    return FlatCoefficientBuffer{TF,typeof(phi_dev),B,LH}(phi_dev, chi_dev, buf.basis_info)
end

function to_cpu(buf::FlatCoefficientBuffer{TF,A,B,LH}) where {TF,A,B,LH}
    return FlatCoefficientBuffer{TF,Matrix{TF},B,LH}(Array(buf.phi), Array(buf.chi), buf.basis_info)
end

println("Starting KA L2L correctness test...")
if !Metal.functional()
    println("Metal not functional; skipping")
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
            r = TF(0.5) + TF(0.5) * rand(TF)            # resident L2L groups share one radius
            rs = fill(r, nbatch)

            cache = OperatorInvariantCache(TF, P, lh)
            scratch = L2LOperatorScratch(TF, cache.basis_info, nbatch)

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
            # compressed-complex layout (see ka_m2m_correctness.jl for the full
            # explanation) -- real local-expansion data never populates these.
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

            op = MaterializedYRotationL2L()

            # CPU reference (non-resident, per-column materialized-rotation path)
            FastMultipole.l2l_operator_batch!(op, targets_cpu, sources,
                                              phis, thetas, rs,
                                              cache, scratch, lh)

            # Device (Metal) path via the real ka_l2l_operator_batch! dispatch
            sources_metal = to_metal(sources)
            targets_metal = to_metal(FlatCoefficientBuffer(TF, cache.basis_info, nbatch))

            FastMultipole.ka_l2l_operator_batch!(op, targets_metal, sources_metal,
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

println("\n✓✓✓ All KA L2L correctness tests passed on Metal! ✓✓✓")
