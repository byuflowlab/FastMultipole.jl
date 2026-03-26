using Pkg
this_dir = @__DIR__
Pkg.activate(normpath(this_dir,".."))
using StaticArrays
using LinearAlgebra
using PythonPlot
include("../test/vortex.jl")
include("../test/gravitational.jl")

using Random

function verify_vortex_symmetry(n_bodies=10_000; expansion_order=10)
    println("--- Vortex Symmetry Validation ---")
    println("Generating $(n_bodies) point vortices...")

    # 1. Generate system
    system = generate_vortex(123, n_bodies; strength_scale=1/n_bodies, radius_factor=0.0, strength_mean=1.0, strength_std=0.2)

    # 2. Direct momentum
    println("Computing direct interaction (exact)...")
    reset!(system)
    fmm.direct!(system)
    
    # Extract total velocity (induced linear momentum)
    P_direct = sum(system.gradient_stretching[1:3, :], dims=2)[:, 1]
    println("P_direct: ", P_direct)

    # 3. FMM momentum
    println("Computing FMM interaction (Lamb-Helmholtz)...")
    reset!(system)
    fmm.fmm!((system,); expansion_order, leaf_size=50, multipole_acceptance=0.9, nearfield=true, farfield=true)
    
    P_fmm = sum(system.gradient_stretching[1:3, :], dims=2)[:, 1]
    println("P_fmm:    ", P_fmm)
    
    # 4. Error comparison
    err = P_direct .- P_fmm
    println("Absolute differences:")
    println("  x: ", err[1])
    println("  y: ", err[2])
    println("  z: ", err[3])
    
    rel_err = norm(err) / norm(P_direct)
    println("Relative L2 Error: ", rel_err)
    
    return P_direct, P_fmm
end

function vortex_to_gravitational(vortex_system)
    n_bodies = fmm.get_n_bodies(vortex_system)
    
    bodies_x = zeros(5, n_bodies)
    bodies_y = zeros(5, n_bodies)
    bodies_z = zeros(5, n_bodies)
    
    for i in 1:n_bodies
        v = vortex_system.bodies[i]
        
        bodies_x[1:3, i] .= v.position
        bodies_y[1:3, i] .= v.position
        bodies_z[1:3, i] .= v.position
        
        bodies_x[4, i] = v.sigma
        bodies_y[4, i] = v.sigma
        bodies_z[4, i] = v.sigma
        
        bodies_x[5, i] = v.strength[1]
        bodies_y[5, i] = v.strength[2]
        bodies_z[5, i] = v.strength[3]
    end
    
    return Gravitational(bodies_x), Gravitational(bodies_y), Gravitational(bodies_z)
end

function verify_scalar_vs_lh(n_bodies=1_000; expansion_order=10)
    println("\n--- Scalar Potentials vs. Lamb-Helmholtz Validation ---")
    println("Generating $(n_bodies) point vortices...")

    # 1. Generate system
    vortex_system = generate_vortex(123, n_bodies; strength_scale=1/n_bodies, radius_factor=0.0, strength_mean=1.0, strength_std=0.2)
    sys_x, sys_y, sys_z = vortex_to_gravitational(vortex_system)

    # 2. Compute Scalar Direct Interactions
    println("Computing Scalar direct interactions...")
    reset!(sys_x)
    reset!(sys_y)
    reset!(sys_z)
    fmm.direct!(sys_x)
    fmm.direct!(sys_y)
    fmm.direct!(sys_z)

    # 3. Derive Equivalent Vector Field
    println("Deriving Equivalent Vector Field...")
    v_recombined = zeros(3, n_bodies)
    for i in 1:n_bodies
        g_x = sys_x.potential[5:7, i]  # [dAx/dx, dAx/dy, dAx/dz]
        g_y = sys_y.potential[5:7, i]  # [dAy/dx, dAy/dy, dAy/dz]
        g_z = sys_z.potential[5:7, i]  # [dAz/dx, dAz/dy, dAz/dz]
        
        v_recombined[1, i] = g_z[2] - g_y[3]
        v_recombined[2, i] = g_x[3] - g_z[1]
        v_recombined[3, i] = g_y[1] - g_x[2]
    end
    P_scalar_direct = sum(v_recombined, dims=2)[:, 1]
    println("P_scalar_direct: ", P_scalar_direct)

    # 4. Verify Induced Vector Field
    println("Computing Vortex direct interactions (exact)...")
    reset!(vortex_system)
    fmm.direct!(vortex_system)
    
    v_vortex = vortex_system.gradient_stretching[1:3, :]
    P_vortex_direct = sum(v_vortex, dims=2)[:, 1]
    println("P_vortex_direct: ", P_vortex_direct)

    err_field = maximum(abs.(v_recombined .- v_vortex))
    println("Max error in recombined velocity field vs exact direct: ", err_field)

    # 5. Compare Momentum Calculations
    println("Computing Scalar FMM interactions...")
    reset!(sys_x)
    reset!(sys_y)
    reset!(sys_z)
    
    fmm_options = (; interaction_list_method=FastMultipole.Barba(), expansion_order, leaf_size=50, multipole_acceptance=0.9, nearfield=true, farfield=true)
    fmm.fmm!((sys_x,); fmm_options...)
    fmm.fmm!((sys_y,); fmm_options...)
    fmm.fmm!((sys_z,); fmm_options...)

    v_recombined_fmm = zeros(3, n_bodies)
    for i in 1:n_bodies
        g_x = sys_x.potential[5:7, i]
        g_y = sys_y.potential[5:7, i]
        g_z = sys_z.potential[5:7, i]
        
        v_recombined_fmm[1, i] = g_z[2] - g_y[3]
        v_recombined_fmm[2, i] = g_x[3] - g_z[1]
        v_recombined_fmm[3, i] = g_y[1] - g_x[2]
    end
    P_scalar_fmm = sum(v_recombined_fmm, dims=2)[:, 1]
    println("P_scalar_fmm:    ", P_scalar_fmm)
    
    println("Computing Vortex FMM interaction (Lamb-Helmholtz)...")
    reset!(vortex_system)
    fmm.fmm!((vortex_system,); fmm_options...)
    P_vortex_fmm = sum(vortex_system.gradient_stretching[1:3, :], dims=2)[:, 1]
    println("P_vortex_fmm:    ", P_vortex_fmm)

    println("Absolute differences (Scalar FMM vs Lamb-Helmholtz FMM):")
    err_fmm = P_scalar_fmm .- P_vortex_fmm
    println("  x: ", err_fmm[1])
    println("  y: ", err_fmm[2])
    println("  z: ", err_fmm[3])
    
    P_diff = norm(err_fmm) / norm(P_vortex_fmm)
    println("Relative L2 Error between FMM implementations: ", P_diff)
end

function shift_system(system::VortexParticles, dv::AbstractVector)
    n_bodies = fmm.get_n_bodies(system)
    shifted_bodies = [Vorton(system.bodies[i].position .+ dv, system.bodies[i].strength, system.bodies[i].sigma) for i in 1:n_bodies]
    return VortexParticles(shifted_bodies, zero(system.potential), zero(system.gradient_stretching))
end

function shift_system(system::Gravitational, dv::AbstractVector)
    n_bodies = fmm.get_n_bodies(system)
    shifted_bodies = [Body(system.bodies[i].position .+ dv, system.bodies[i].radius, system.bodies[i].strength) for i in 1:n_bodies]
    return Gravitational(shifted_bodies, zero(system.potential))
end

function verify_vortex_symmetry_separated(n_bodies=10_000; expansion_order=10, dv=[2.0, 0.0, 0.0])
    println("\n--- Vortex Symmetry Validation (Separated Systems) ---")
    println("Generating $(n_bodies) point vortices for source...")
    
    source_system = generate_vortex(123, n_bodies; strength_scale=1/n_bodies, radius_factor=0.0, strength_mean=1.0, strength_std=0.2)
    target_system = generate_vortex(456, n_bodies; shift_position=dv, strength_scale=1/n_bodies, radius_factor=0.0, strength_mean=1.0, strength_std=0.2)
    
    println("Computing direct interaction (exact)...")
    reset!(target_system)
    fmm.direct!(target_system, source_system)
    
    P_direct = sum(target_system.gradient_stretching[1:3, :], dims=2)[:, 1]
    println("P_direct: ", P_direct)
    
    println("Computing FMM interaction (Lamb-Helmholtz)...")
    reset!(target_system)
    fmm.fmm!((target_system,), (source_system,); expansion_order, leaf_size_source=50, multipole_acceptance=0.9, nearfield=true, farfield=true)
    
    P_fmm = sum(target_system.gradient_stretching[1:3, :], dims=2)[:, 1]
    println("P_fmm:    ", P_fmm)
    
    err = P_direct .- P_fmm
    println("Absolute differences:")
    println("  x: ", err[1])
    println("  y: ", err[2])
    println("  z: ", err[3])
    
    rel_err = norm(err) / norm(P_direct)
    println("Relative L2 Error: ", rel_err)
    
    return P_direct, P_fmm
end

function verify_scalar_vs_lh_separated(n_bodies=1_000; expansion_order=10, dv=[2.0, 0.0, 0.0])
    println("\n--- Scalar Potentials vs. Lamb-Helmholtz Validation (Separated Systems) ---")
    println("Generating $(n_bodies) point vortices for source...")
    
    vortex_source = generate_vortex(123, n_bodies; strength_scale=1/n_bodies, radius_factor=0.0, strength_mean=1.0, strength_std=0.2)
    vortex_target = generate_vortex(456, shift_position=dv, n_bodies; strength_scale=1/n_bodies, radius_factor=0.0, strength_mean=1.0, strength_std=0.2)
    
    sys_x_src, sys_y_src, sys_z_src = vortex_to_gravitational(vortex_source)
    sys_x_tgt, sys_y_tgt, sys_z_tgt = vortex_to_gravitational(vortex_target)
    
    println("Computing Scalar direct interactions...")
    reset!(sys_x_tgt)
    reset!(sys_y_tgt)
    reset!(sys_z_tgt)
    fmm.direct!(sys_x_tgt, sys_x_src)
    fmm.direct!(sys_y_tgt, sys_y_src)
    fmm.direct!(sys_z_tgt, sys_z_src)
    
    println("Deriving Equivalent Vector Field...")
    v_recombined = zeros(3, n_bodies)
    for i in 1:n_bodies
        g_x = sys_x_tgt.potential[5:7, i]
        g_y = sys_y_tgt.potential[5:7, i]
        g_z = sys_z_tgt.potential[5:7, i]
        
        v_recombined[1, i] = g_z[2] - g_y[3]
        v_recombined[2, i] = g_x[3] - g_z[1]
        v_recombined[3, i] = g_y[1] - g_x[2]
    end
    P_scalar_direct = sum(v_recombined, dims=2)[:, 1]
    println("P_scalar_direct: ", P_scalar_direct)
    
    println("Computing Vortex direct interactions (exact)...")
    reset!(vortex_target)
    fmm.direct!(vortex_target, vortex_source)
    
    v_vortex = vortex_target.gradient_stretching[1:3, :]
    P_vortex_direct = sum(v_vortex, dims=2)[:, 1]
    println("P_vortex_direct: ", P_vortex_direct)
    
    err_field = maximum(abs.(v_recombined .- v_vortex))
    println("Max error in recombined velocity field vs exact direct: ", err_field)
    
    println("Computing Scalar FMM interactions...")
    reset!(sys_x_tgt)
    reset!(sys_y_tgt)
    reset!(sys_z_tgt)
    
    fmm_options = (; interaction_list_method=FastMultipole.Barba(), expansion_order, leaf_size_source=50, multipole_acceptance=0.9, nearfield=true, farfield=true)
    optimized_args, cache, target_tree, source_tree, m2l_list, direct_list, derivatives_switches, error_success = fmm.fmm!((sys_x_tgt,), (sys_x_src,); fmm_options...)
    # @assert length(m2l_list) > 0 "M2L list should not be empty for non-trivial interactions"
    fmm.fmm!((sys_y_tgt,), (sys_y_src,); fmm_options...)
    fmm.fmm!((sys_z_tgt,), (sys_z_src,); fmm_options...)
    
    v_recombined_fmm = zeros(3, n_bodies)
    for i in 1:n_bodies
        g_x = sys_x_tgt.potential[5:7, i]
        g_y = sys_y_tgt.potential[5:7, i]
        g_z = sys_z_tgt.potential[5:7, i]
        
        v_recombined_fmm[1, i] = g_z[2] - g_y[3]
        v_recombined_fmm[2, i] = g_x[3] - g_z[1]
        v_recombined_fmm[3, i] = g_y[1] - g_x[2]
    end
    P_scalar_fmm = sum(v_recombined_fmm, dims=2)[:, 1]
    println("P_scalar_fmm:    ", P_scalar_fmm)
    
    println("Computing Vortex FMM interaction (Lamb-Helmholtz)...")
    reset!(vortex_target)
    fmm.fmm!((vortex_target,), (vortex_source,); fmm_options...)
    P_vortex_fmm = sum(vortex_target.gradient_stretching[1:3, :], dims=2)[:, 1]
    println("P_vortex_fmm:    ", P_vortex_fmm)
    
    println("Absolute differences (Scalar FMM vs Lamb-Helmholtz FMM):")
    err_fmm = P_scalar_fmm .- P_vortex_fmm
    println("  x: ", err_fmm[1])
    println("  y: ", err_fmm[2])
    println("  z: ", err_fmm[3])
    
    P_diff = norm(err_fmm) / norm(P_vortex_fmm)
    println("Relative L2 Error between FMM implementations: ", P_diff)

    err_fmm_scalar = P_scalar_fmm .- P_vortex_direct
    err_fmm_vortex = P_vortex_fmm .- P_vortex_direct
    println()
    println("Relative L2 Error of Scalar FMM vs Direct: ", norm(err_fmm_scalar) / norm(P_vortex_direct))
    println("Relative L2 Error of Vortex FMM vs Direct: ", norm(err_fmm_vortex) / norm(P_vortex_direct))

    return norm(err_fmm_scalar) / norm(P_vortex_direct), norm(err_fmm_vortex) / norm(P_vortex_direct)
end

# verify_vortex_symmetry(10_000; expansion_order=10)
# verify_scalar_vs_lh(10_000; expansion_order=10)
# verify_vortex_symmetry_separated(2; expansion_order=3)

function run_symmetry_tests(;
    Ps = [2, 3, 4, 8, 16],
    N_bodies = [2, 10, 50]
)

    n_P = length(Ps)
    n_bodies = length(N_bodies)

    errs_scalar, errs_vortex = zeros(n_P, n_bodies), zeros(n_P, n_bodies)

    for (i, P) in enumerate(Ps)
        for (j, N) in enumerate(N_bodies)
            println("\n=== Testing P = $P, N = $N ===")
            err_scalar, err_vortex = verify_scalar_vs_lh_separated(N; expansion_order=P)
            errs_scalar[i, j] = err_scalar
            errs_vortex[i, j] = err_vortex
        end
    end

    return errs_scalar, errs_vortex
end

function plot_symmetry_errors(errs_scalar, errs_vortex; Ps = [2, 3, 4, 8, 16], N_bodies = [2, 10, 100, 1_000, 10_000])

    cmap = PythonPlot.get_cmap("RdBu", 16)
    cmap_colors = [cmap(i) for i in [1, 10, 13]]
    fig, (ax1, ax2) = subplots(1, 2, figsize=(10, 4), num="errors")
    fig.clear()
    fig, (ax1, ax2) = subplots(1, 2, figsize=(10, 4), num="errors")
    
    for (j, N) in enumerate(N_bodies)
        ax1.semilogy(Ps, errs_scalar[:, j], marker="o", label="N = $N", color=cmap_colors[j])
        ax2.semilogy(Ps, errs_vortex[:, j], marker="o", label="N = $N", color=cmap_colors[j])
    end
    
    ax1.set_xlabel("Expansion Order (P)")
    ax1.set_ylabel("Relative L2 Error")
    ax1.set_title("Scalar Separated Field Error")
    ax1.legend()
    ax1.spines["top"].set_visible(false)
    ax1.spines["right"].set_visible(false)
    
    ax2.set_xlabel("Expansion Order (P)")
    ax2.set_ylabel("Relative L2 Error")
    ax2.set_title("Vortex Separated Field Error")
    ax2.legend()
    ax2.spines["top"].set_visible(false)
    ax2.spines["right"].set_visible(false)
    
    fig.tight_layout()

    fig2, ax3 = subplots(1, 1, figsize=(5, 4), num="error ratio")
    fig2.clear()
    fig2, ax3 = subplots(1, 1, figsize=(5, 4), num="error ratio")
    
    for (j, N) in enumerate(N_bodies)
        ax3.semilogy(Ps, errs_vortex[:, j] ./ errs_scalar[:, j], marker="o", label="N bodies = $N", color=cmap_colors[j])
    end
    ax3.set_xlabel("Expansion Order (P)")
    ax3.set_ylabel("Vortex Error / Scalar Error")
    ax3.spines["top"].set_visible(false)
    ax3.spines["right"].set_visible(false)
    # ax3.set_title("Error Ratio")
    ax3.legend()
    fig2.savefig("error_ratio.png", dpi=300)
    fig2.savefig("error_ratio.pdf")
    # ax3.grid(true)

    # show()
end

Ps = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]
N_bodies = [2, 10, 50]
# errs_scalar, errs_vortex = run_symmetry_tests(; Ps, N_bodies)
plot_symmetry_errors(errs_scalar, errs_vortex; Ps, N_bodies)