# Gate for step 3: KA body-to-multipole (B2M), vortex channel.
#
# Oracle is a straightforward host loop over FastMultipole's OWN per-body
# contribution functions (`_resident_vortex_phi_contrib` / `_chi_contrib`,
# src/translate_batched_resident.jl) -- the same `@inline` code the device
# kernel calls, so this tests the reduction and indexing, not the physics.
#
# The host loop sums sequentially while the kernel tree-reduces, so the two
# differ by summation order alone; at Float32 that shows up as a relative
# error growing with cell occupancy. Anything NOT reduction-order-shaped is a
# bug, per the plan's gate.
include("ka_backend.jl")
using FastMultipole, Random, Test
const FM = FastMultipole

if !dev_functional()
    println("$(DEV_NAME) not functional; skipping"); exit(0)
end
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext !== nothing || error("FastMultipoleKAExt did not load")

function host_b2m_vortex(source_bodies, cell_centers, cell_ranges, leaf_to_node,
                         P_phi, P_chi, ncell, dof_phi, dof_chi, nnode, ::Type{TF}) where TF
    phi = zeros(TF, dof_phi, nnode); chi = zeros(TF, dof_chi, nnode)
    for i_cell in 1:ncell
        first = cell_ranges[1, i_cell]; count = cell_ranges[2, i_cell]
        cx, cy, cz = cell_centers[1, i_cell], cell_centers[2, i_cell], cell_centers[3, i_cell]
        node = leaf_to_node[i_cell]
        for n in 0:P_phi, m in 0:n
            ar = zero(TF); ai = zero(TF)
            for k in first:(first + count - 1)
                re, im = FM._resident_vortex_phi_contrib(cx - source_bodies[1,k],
                    cy - source_bodies[2,k], cz - source_bodies[3,k],
                    source_bodies[5,k], source_bodies[6,k], source_bodies[7,k], n, m)
                ar += re; ai += im
            end
            row = FM.flat_basis_index(n, m, 1)
            phi[row, node] = ar; phi[row+1, node] = ai
        end
        for n in 1:P_chi, m in 0:n
            ar = zero(TF); ai = zero(TF)
            for k in first:(first + count - 1)
                re, im = FM._resident_vortex_chi_contrib(cx - source_bodies[1,k],
                    cy - source_bodies[2,k], cz - source_bodies[3,k],
                    source_bodies[5,k], source_bodies[6,k], source_bodies[7,k], n, m)
                ar += re; ai += im
            end
            row = FM.flat_basis_index(n, m, 1)
            chi[row, node] = ar; chi[row+1, node] = ai
        end
    end
    return phi, chi
end

relerr(a,b) = (s = maximum(abs.(b)); s == 0 ? maximum(abs.(a .- b)) : maximum(abs.(a .- b))/s)

const CASES = [
    #  n,  K_max, ell_max, P, balance
    (  50,  4, 4, 2, true),
    ( 400,  8, 5, 4, true),
    (2000, 16, 6, 4, true),
    (2000,  4, 6, 2, false),
    (5000, 32, 6, 6, true),
    (   1,  4, 4, 2, true),
]

npass = Ref(0); nfail = Ref(0)
for (case_i, (n, K_max, ell_max, P, balance)) in pairs(CASES)
    Random.seed!(5300 + case_i)   # per case: Metal draws from the task-local RNG per launch
    TF = Float32
    positions = rand(TF, 3, n)
    dpb = 8
    source_buffer = rand(TF, dpb, n); source_buffer[1:3, :] .= positions

    nl = max(2 * n ÷ K_max, 16)
    actx = ext.ka_allocate_adaptive_context(DEV_BACKEND, TF, n;
        leaf_capacity=10*nl+256, frontier_capacity=16*(10*nl+256), node_capacity=100*nl+256)
    build = ext.ka_build_adaptive_tree!(actx, devarray(positions), ell_max, K_max,
        balance, (TF(0), TF(0), TF(0)), TF(1))
    options = FM.CUDARadixLifecycleOptions(; precision=TF, body_type=FM.Point{FM.Vortex})
    state = ext.ka_radix_state(actx, build, devarray(source_buffer), P, Val(true); options)

    try
        ext.ka_launch_b2m!(state)
        KernelAbstractions.synchronize(DEV_BACKEND)
    catch e
        nfail[] += 1
        println("  FAIL case $case_i: kernel threw: ", sprint(showerror, e)[1:min(end,300)])
        continue
    end

    ncell = build.n_leaves; nnode = build.n_nodes
    orders = state.invariant_cache.basis_info.orders
    hphi, hchi = host_b2m_vortex(Array(state.source_bodies), Array(state.cell_centers),
        Array(state.cell_ranges), Array(actx.grid.leaf_to_node),
        orders.P_phi, orders.P_active, ncell,
        size(state.multipoles.phi,1), size(state.multipoles.chi,1),
        size(state.multipoles.phi,2), TF)

    ep = relerr(Array(state.multipoles.phi), hphi)
    ec = relerr(Array(state.multipoles.chi), hchi)
    occ = maximum(Array(state.cell_ranges)[2, 1:ncell])
    tol = 1e-5
    if ep <= tol && ec <= tol
        npass[] += 1
        println("  PASS  n=$n P=$P cells=$ncell maxocc=$occ  relerr phi=$(round(ep,sigdigits=3)) chi=$(round(ec,sigdigits=3))")
    else
        nfail[] += 1
        println("  FAIL  n=$n P=$P cells=$ncell maxocc=$occ  relerr phi=$ep chi=$ec (tol $tol)")
    end
end
println("\nKA B2M (vortex) on $(DEV_NAME): $(npass[]) passed, $(nfail[]) failed")
nfail[] == 0 || error("B2M gate failed")
println("✓✓✓ Step 3 (B2M vortex) gate passed on $(DEV_NAME) ✓✓✓")
