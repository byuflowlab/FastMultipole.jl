# Metal FMM vs Metal DIRECT, on the same device and the same body array.
#
# Every previous Metal benchmark compared the KA FMM against a CPU arm, which
# conflates the port with the hardware. The question "is the FMM actually
# buying anything" is only answerable against direct summation on the SAME
# device, and that arm did not exist: `direct_gpu!` is exported
# (src/FastMultipole.jl:201) but has no definition anywhere in the repo.
#
# Rather than write a new kernel, this drives the PRODUCTION pair kernel
# `ka_direct_pairs_functor_kernel!` with an all-pairs list: targets are split
# into `chunk`-body cells so there is real parallelism (one workgroup per
# chunk), and every chunk is paired against one cell holding all n sources.
# So the per-pair physics is byte-identical to the nearfield stage -- this
# measures the O(n^2) work, not a second implementation of the kernel.
#
# ell per case follows FLOWVPM's occupancy cap, ell = floor(log2(np)/3)
# (FLOWVPM/src/FLOWVPM_fmm_radix.jl:490). That is the UPPER bound the auto
# rule starts from; _radix_auto_geometry then walks ell down until the
# sigma-adequacy inequality passes, so a real wake often runs shallower.
include("ka_backend.jl")
using FastMultipole, Random, Printf
using LinearAlgebra
using FastMultipole.StaticArrays
using KernelAbstractions
const KA = KernelAbstractions

const FM = FastMultipole
include(joinpath(@__DIR__, "..", "vortex.jl"))

if !dev_functional()
    println("$(DEV_NAME) not functional; skipping")
    exit(0)
end
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext !== nothing || error("FastMultipoleKAExt did not load")

make_system(seed, n, TF) = (Random.seed!(seed);
    VortexParticles(rand(TF, 3, n), (randn(TF, 3, n) ./ TF(n)), zeros(TF, n);
        potential=zeros(TF, 13, n), gradient_stretching=zeros(TF, 6, n)))

function device_build_args(hcache)
    sp = hcache.policy
    ell = hcache.ell
    tables, level_class_of, level_radii2, root_level, first_m2l_level =
        FM._hierarchical_scheduled_tables(sp, ell, hcache.ell_axes)
    class_level, class_offset, _ =
        FM._hierarchical_class_metadata(tables, ell, first_m2l_level)
    max_level_nodes = ell >= 2 ? maximum(
        (FM._radix_level_node_capacity(L, hcache.ell_axes, ell, hcache.max_cells)
         for L in first_m2l_level:ell); init=0) : 0
    return (; tables, level_class_of, level_radii2, root_level, first_m2l_level,
        class_level, class_offset, max_level_nodes)
end

function timeit(f, trials, warmup)
    for _ in 1:warmup; f(); end
    ts = Float64[]
    for _ in 1:trials
        t0 = time_ns(); f(); push!(ts, (time_ns() - t0) / 1e6)
    end
    sort!(ts)
    return (min=ts[1], med=ts[(length(ts)+1) ÷ 2])
end

# All-pairs launcher over the state's own body array. `chunk` target bodies per
# workgroup; one source cell spanning all n bodies. The kernel's own `i != j`
# guard handles self-interaction, exactly as in the nearfield stage.
function make_direct_arm(state, chunk::Int, workgroup::Int)
    TF = eltype(state.output)
    n = size(state.source_bodies, 2)
    ng = cld(n, chunk)
    CR = typeof(state.cell_ranges)
    IT = eltype(state.cell_ranges)
    h_cr = zeros(IT, 2, ng + 1)
    for g in 1:ng
        first_b = (g - 1) * chunk + 1
        h_cr[1, g] = first_b
        h_cr[2, g] = min(chunk, n - first_b + 1)
    end
    h_cr[1, ng+1] = 1; h_cr[2, ng+1] = n          # the all-source cell
    cr = CR(undef, 2, ng + 1); copyto!(cr, h_cr)

    DT = typeof(state.direct_targets)
    JT = eltype(state.direct_targets)
    tg = DT(undef, ng); copyto!(tg, JT.(1:ng))
    sr = DT(undef, ng); copyto!(sr, fill(JT(ng + 1), ng))

    hs = size(state.output, 1) >= 13
    backend = KA.get_backend(state.output)
    kern = ext._cached_kernel(ext.ka_direct_pairs_functor_kernel!, backend, workgroup)
    return function ()
        fill!(state.output, zero(TF))
        kern(state.options.direct_kernel, state.output, state.source_bodies,
             cr, tg, sr, ng, TF, Val(hs), Val(workgroup),
             state.source_bodies, Val(false); ndrange=ng * workgroup)
        return nothing
    end
end

# (P, ell, n, window_classes, trials)
const CASES = [
    (4, 2,  4_096, 256, 10),
    (4, 4,  4_096, 256, 10),
    (4, 4, 16_384, 256,  5),
]

println("threads = $(Threads.nthreads()), device = $DEV_NAME\n")
@printf("%-8s %4s %4s | %10s %10s | %10s %10s | %8s | %9s\n",
    "n", "P", "ell", "fmm_min", "fmm_med", "direct_min", "direct_med",
    "x_direct", "relerr")

for (ci, (P, ell, n, wc, trials)) in pairs(CASES)
    TF = Float32
    sys_h = make_system(6100 + ci, n, TF)
    sys_d = make_system(6100 + ci, n, TF)

    opts = FM.CUDARadixLifecycleOptions(; precision=TF,
        m2l_strategy=FM.ConcatenatedFixedZM2L(), body_type=FM.Point{FM.Vortex})
    hcache = RadixFMMCache(sys_h; expansion_order=P, ell=ell, window_classes=wc,
        options=opts)
    fmm!(sys_h, hcache)

    a = device_build_args(hcache)
    LH = typeof(hcache).parameters[2]
    dcache = ext.ka_radix_cache_device_build(DEV_BACKEND, (sys_d,),
        hcache.expansion_order, ell, hcache.x_min, hcache.h0, hcache.max_n_bodies,
        hcache.options, hcache.policy, hcache.accepted_offsets,
        hcache.rejected_offsets, hcache.max_cells, hcache.max_nodes,
        hcache.route_capacity, hcache.direct_capacity,
        hcache.state.multipoles.basis_info, Val(LH);
        hierarchical_tables=a.tables, class_level=a.class_level,
        class_offset=a.class_offset, hierarchical_level_class_of=a.level_class_of,
        hierarchical_level_radii2=a.level_radii2,
        max_level_nodes=a.max_level_nodes, hessian=hcache.hessian,
        ell_axes=hcache.ell_axes, box_extent=hcache.box_extent,
        root_level=a.root_level, first_m2l_level=a.first_m2l_level)
    switches = FM.DerivativesSwitch(FM.to_vector(false, 1), FM.to_vector(true, 1),
        FM.to_vector(false, 1), (sys_d,))

    st = dcache.state

    fmm_step = () -> (ext.ka_radix_cache_device_step!(dcache, (sys_d,), switches);
                      KA.synchronize(DEV_BACKEND))
    fmm_step()
    out_fmm = Array(st.output)[2:4, :]      # gradient rows, FMM

    direct_step = make_direct_arm(st, 64, 64)
    direct_step(); KA.synchronize(DEV_BACKEND)
    out_dir = Array(st.output)[2:4, :]      # gradient rows, all-pairs

    # both arms wrote the SAME state.output over the SAME body ordering, so
    # this is a direct elementwise comparison with no permutation in between
    s = maximum(abs.(out_dir))
    relerr = s == 0 ? maximum(abs.(out_fmm .- out_dir)) :
        maximum(abs.(out_fmm .- out_dir)) / s

    tf = timeit(fmm_step, trials, 3)
    td = timeit(() -> (direct_step(); KA.synchronize(DEV_BACKEND)), trials, 3)

    @printf("%-8d %4d %4d | %10.2f %10.2f | %10.2f %10.2f | %7.2fx | %9.2e\n",
        n, P, ell, tf.min, tf.med, td.min, td.med, td.min / tf.min, relerr)
    flush(stdout)
end
