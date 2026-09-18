# What is the 33% nearfield actually spending its time on?
#
# `ka_direct_pairs_functor_kernel!` (ext:2908) runs one workgroup per
# (target_cell, source_cell) pair, threads striding the target bodies, and each
# thread loops over EVERY source body reading it from global memory. Then it
# does 13 `KA.@atomic` global adds per target body per pair.
#
# Two candidate costs, two ablations:
#   (a) redundant source-body traffic  -> vary WG (changes threads/pair, not the
#       total target x source interaction count)
#   (b) atomic contention              -> rerun with the atomics replaced by
#       plain (racy, WRONG) stores. Times the atomic overhead directly; the
#       result is numerically garbage and is never compared for accuracy.
#
# Ablation (b) is the deciding measurement for whether the fused target-owned
# CSR shape (deliberately skipped at ext:2893) is worth building: that shape
# exists precisely to turn ~53 atomic-accumulating pairs per target cell into
# one non-atomic write.

include("ka_backend.jl")
include("pipeline_field.jl")

using FastMultipole, Printf
const FM = FastMultipole
const V = FLOWVPM
import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index, @Const

dev_functional() || (println("$(DEV_NAME) not functional; skipping"); exit(0))
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
ext === nothing && error("the KA extension is not loaded")

const STEP = parse(Int, get(ENV, "UJ_STEP", "36"))
const P = 5
const CALLS = 12

host = load_wake(STEP; TF=Float64, P=P)
NP = V.get_np(host)
dev = V.ParticleField(host.maxparticles, Float32; arraytype=devmatrix, np=host.np,
    fmm=V.FMM(; p=P+1, autotune_p=false, autotune_ncrit=false,
                autotune_reg_error=false, default_rho_over_sigma=1.0))
dev.particles .= devarray(Float32.(Array(host.particles)))
V.radix_fmm_settings!(dev; m2l_strategy=:concat)
V.UJ_fmm(dev)

st = V._radix_fmm_couplings[dev]
state = st.cache.state
backend = KA.get_backend(state.output)
npairs = state.counts.n_direct
ncell = state.counts.n_cells
TF = eltype(state.output)

ranges = Array(state.cell_ranges)
dt = Array(state.direct_targets); ds = Array(state.direct_sources)
interactions = sum(ranges[2, dt[k]] * ranges[2, ds[k]] for k in 1:npairs)
pairs_per_target = npairs / ncell

println("=== nearfield cost probe ===")
@printf("np=%d  n_cells=%d  n_direct(pairs)=%d  pairs/target cell=%.1f\n",
        NP, ncell, npairs, pairs_per_target)
@printf("target x source interactions = %d\n", interactions)
@printf("hessian rows = %s  -> %d atomic adds per target body per pair\n",
        size(state.output,1) >= 13, size(state.output,1) >= 13 ? 13 : 4)
@printf("atomic adds per step = %d\n\n",
        sum(ranges[2, dt[k]] for k in 1:npairs) * (size(state.output,1) >= 13 ? 13 : 4))

# ---------------- ablation kernel: identical, atomics removed ----------------
@kernel function probe_direct_pairs_noatomic!(kernel, output, @Const(source_bodies),
        @Const(cell_ranges), @Const(direct_targets), @Const(direct_sources),
        npairs, ::Type{T}, ::Val{HS}, ::Val{WG}) where {T,HS,WG}
    pair_i = @index(Group)
    tid = @index(Local)
    ep = FastMultipole._emits_potential(kernel)
    ghv = Val(:shipped)
    @inbounds begin
        target_cell = direct_targets[pair_i]
        source_cell = direct_sources[pair_i]
        tfirst = cell_ranges[1, target_cell]
        tlast = tfirst + cell_ranges[2, target_cell] - 1
        sfirst = cell_ranges[1, source_cell]
        slast = sfirst + cell_ranges[2, source_cell] - 1
        i = tfirst + tid - 1
        while i <= tlast
            xi = source_bodies[1, i]; yi = source_bodies[2, i]; zi = source_bodies[3, i]
            u = zero(T); gx = zero(T); gy = zero(T); gz = zero(T)
            h1 = zero(T); h2 = zero(T); h3 = zero(T)
            h4 = zero(T); h5 = zero(T); h6 = zero(T)
            h7 = zero(T); h8 = zero(T); h9 = zero(T)
            for j in sfirst:slast
                if i != j
                    dx = xi - source_bodies[1, j]
                    dy = yi - source_bodies[2, j]
                    dz = zi - source_bodies[3, j]
                    r2 = dx * dx + dy * dy + dz * dz
                    if r2 > zero(r2)
                        invr = inv(sqrt(r2))
                        if HS
                            du, dgx, dgy, dgz, dh1, dh2, dh3, dh4, dh5, dh6, dh7, dh8, dh9 =
                                FastMultipole._direct_pair_ugh(kernel, dx, dy, dz, r2, invr,
                                    source_bodies, j, ghv)
                            u += du; gx += dgx; gy += dgy; gz += dgz
                            h1 += dh1; h2 += dh2; h3 += dh3
                            h4 += dh4; h5 += dh5; h6 += dh6
                            h7 += dh7; h8 += dh8; h9 += dh9
                        else
                            du, dgx, dgy, dgz = FastMultipole._direct_pair_ug(kernel,
                                dx, dy, dz, r2, invr, source_bodies, j, ghv)
                            u += du; gx += dgx; gy += dgy; gz += dgz
                        end
                    end
                end
            end
            # ABLATION: non-atomic, racy on purpose -- times the atomic away
            if ep
                output[1, i] += u
            end
            output[2, i] += gx
            output[3, i] += gy
            output[4, i] += gz
            if HS
                output[5, i] += h1; output[6, i] += h2; output[7, i] += h3
                output[8, i] += h4; output[9, i] += h5; output[10, i] += h6
                output[11, i] += h7; output[12, i] += h8; output[13, i] += h9
            end
            i += WG
        end
    end
end


# ---------- ablation (c): g/h series skipped, singular (g=1,h=-3) ----------
# Bounds what a :lut / cheapened g-h port could ever recover in this kernel.
@kernel function probe_direct_pairs_nogh!(kernel, output, @Const(source_bodies),
        @Const(cell_ranges), @Const(direct_targets), @Const(direct_sources),
        npairs, ::Type{T}, ::Val{HS}, ::Val{WG}) where {T,HS,WG}
    pair_i = @index(Group)
    tid = @index(Local)
    @inbounds begin
        target_cell = direct_targets[pair_i]
        source_cell = direct_sources[pair_i]
        tfirst = cell_ranges[1, target_cell]
        tlast = tfirst + cell_ranges[2, target_cell] - 1
        sfirst = cell_ranges[1, source_cell]
        slast = sfirst + cell_ranges[2, source_cell] - 1
        i = tfirst + tid - 1
        while i <= tlast
            xi = source_bodies[1, i]; yi = source_bodies[2, i]; zi = source_bodies[3, i]
            u = zero(T); gx = zero(T); gy = zero(T); gz = zero(T)
            h1 = zero(T); h2 = zero(T); h3 = zero(T)
            h4 = zero(T); h5 = zero(T); h6 = zero(T)
            h7 = zero(T); h8 = zero(T); h9 = zero(T)
            for j in sfirst:slast
                if i != j
                    dx = xi - source_bodies[1, j]
                    dy = yi - source_bodies[2, j]
                    dz = zi - source_bodies[3, j]
                    r2 = dx * dx + dy * dy + dz * dz
                    if r2 > zero(r2)
                        invr = inv(sqrt(r2))
                        gsx = source_bodies[5, j]; gsy = source_bodies[6, j]
                        gsz = source_bodies[7, j]
                        du, dgx, dgy, dgz, dh1, dh2, dh3, dh4, dh5, dh6, dh7, dh8, dh9 =
                            FastMultipole._vortex_pair_ugh(dx, dy, dz, r2, invr,
                                gsx, gsy, gsz, one(T), -T(3))
                        u += du; gx += dgx; gy += dgy; gz += dgz
                        h1 += dh1; h2 += dh2; h3 += dh3
                        h4 += dh4; h5 += dh5; h6 += dh6
                        h7 += dh7; h8 += dh8; h9 += dh9
                    end
                end
            end
            KA.@atomic output[2, i] += gx
            KA.@atomic output[3, i] += gy
            KA.@atomic output[4, i] += gz
            if HS
                KA.@atomic output[5, i]  += h1; KA.@atomic output[6, i]  += h2
                KA.@atomic output[7, i]  += h3; KA.@atomic output[8, i]  += h4
                KA.@atomic output[9, i]  += h5; KA.@atomic output[10, i] += h6
                KA.@atomic output[11, i] += h7; KA.@atomic output[12, i] += h8
                KA.@atomic output[13, i] += h9
            end
            i += WG
        end
    end
end

# ---------- ablation (d): SERIES ONLY removed, sigma load + rho divide kept ----------
# (c) above also drops the `sigma` fetch and the rho divide, so its 36% is an
# upper bound on "series + operand traffic". `:lut` keeps both and only replaces
# the series -- and measured ~0%. (d) separates the two: it does every memory
# access and branch the real kernel does, and only skips the 13-term evaluation.
@kernel function probe_direct_pairs_noseries!(kernel, output, @Const(source_bodies),
        @Const(cell_ranges), @Const(direct_targets), @Const(direct_sources),
        npairs, ::Type{T}, ::Val{HS}, ::Val{WG}) where {T,HS,WG}
    pair_i = @index(Group)
    tid = @index(Local)
    @inbounds begin
        target_cell = direct_targets[pair_i]
        source_cell = direct_sources[pair_i]
        tfirst = cell_ranges[1, target_cell]
        tlast = tfirst + cell_ranges[2, target_cell] - 1
        sfirst = cell_ranges[1, source_cell]
        slast = sfirst + cell_ranges[2, source_cell] - 1
        i = tfirst + tid - 1
        while i <= tlast
            xi = source_bodies[1, i]; yi = source_bodies[2, i]; zi = source_bodies[3, i]
            u = zero(T); gx = zero(T); gy = zero(T); gz = zero(T)
            h1 = zero(T); h2 = zero(T); h3 = zero(T)
            h4 = zero(T); h5 = zero(T); h6 = zero(T)
            h7 = zero(T); h8 = zero(T); h9 = zero(T)
            for j in sfirst:slast
                if i != j
                    dx = xi - source_bodies[1, j]
                    dy = yi - source_bodies[2, j]
                    dz = zi - source_bodies[3, j]
                    r2 = dx * dx + dy * dy + dz * dz
                    if r2 > zero(r2)
                        invr = inv(sqrt(r2))
                        gsx = source_bodies[5, j]; gsy = source_bodies[6, j]
                        gsz = source_bodies[7, j]
                        sigma = source_bodies[kernel.sigma_row, j]
                        g = one(T); h = -T(3)
                        if sigma > zero(T)
                            rho = r2 * invr / sigma
                            if rho <= kernel.rho_t
                                g = T(0.5); h = -T(2.5)   # stand-in for the series
                            end
                        end
                        du, dgx, dgy, dgz, dh1, dh2, dh3, dh4, dh5, dh6, dh7, dh8, dh9 =
                            FastMultipole._vortex_pair_ugh(dx, dy, dz, r2, invr,
                                gsx, gsy, gsz, g, h)
                        u += du; gx += dgx; gy += dgy; gz += dgz
                        h1 += dh1; h2 += dh2; h3 += dh3
                        h4 += dh4; h5 += dh5; h6 += dh6
                        h7 += dh7; h8 += dh8; h9 += dh9
                    end
                end
            end
            KA.@atomic output[2, i] += gx
            KA.@atomic output[3, i] += gy
            KA.@atomic output[4, i] += gz
            if HS
                KA.@atomic output[5, i]  += h1; KA.@atomic output[6, i]  += h2
                KA.@atomic output[7, i]  += h3; KA.@atomic output[8, i]  += h4
                KA.@atomic output[9, i]  += h5; KA.@atomic output[10, i] += h6
                KA.@atomic output[11, i] += h7; KA.@atomic output[12, i] += h8
                KA.@atomic output[13, i] += h9
            end
            i += WG
        end
    end
end

# (g)/(h) THE DECIDING PAIR for the sigma-divide question. Both carry the REAL
# 13-term series, inlined exactly as (d)/(e) inline it, so the only difference
# between them is `/ sigma` vs `* sigma`. (e)-minus-(d) measured the divide with
# the series ALREADY REMOVED, which over-credits it: on the real kernel the
# series occupies the same dependency chain and can hide the divide's latency.
# realdiv-minus-realmul is what storing 1/sigma could actually buy.
@kernel function probe_direct_pairs_realdiv!(kernel, output, @Const(source_bodies),
        @Const(cell_ranges), @Const(direct_targets), @Const(direct_sources),
        npairs, ::Type{T}, ::Val{HS}, ::Val{WG}) where {T,HS,WG}
    pair_i = @index(Group)
    tid = @index(Local)
    @inbounds begin
        target_cell = direct_targets[pair_i]
        source_cell = direct_sources[pair_i]
        tfirst = cell_ranges[1, target_cell]
        tlast = tfirst + cell_ranges[2, target_cell] - 1
        sfirst = cell_ranges[1, source_cell]
        slast = sfirst + cell_ranges[2, source_cell] - 1
        i = tfirst + tid - 1
        while i <= tlast
            xi = source_bodies[1, i]; yi = source_bodies[2, i]; zi = source_bodies[3, i]
            u = zero(T); gx = zero(T); gy = zero(T); gz = zero(T)
            h1 = zero(T); h2 = zero(T); h3 = zero(T)
            h4 = zero(T); h5 = zero(T); h6 = zero(T)
            h7 = zero(T); h8 = zero(T); h9 = zero(T)
            for j in sfirst:slast
                if i != j
                    dx = xi - source_bodies[1, j]
                    dy = yi - source_bodies[2, j]
                    dz = zi - source_bodies[3, j]
                    r2 = dx * dx + dy * dy + dz * dz
                    if r2 > zero(r2)
                        invr = inv(sqrt(r2))
                        gsx = source_bodies[5, j]; gsy = source_bodies[6, j]
                        gsz = source_bodies[7, j]
                        sigma = source_bodies[kernel.sigma_row, j]
                        g = one(T); h = -T(3)
                        if sigma > zero(T)
                            rho = r2 * invr / sigma
                            if rho <= kernel.rho_t
                                g, h = FastMultipole._gaussianerf_g_h(rho)
                            end
                        end
                        du, dgx, dgy, dgz, dh1, dh2, dh3, dh4, dh5, dh6, dh7, dh8, dh9 =
                            FastMultipole._vortex_pair_ugh(dx, dy, dz, r2, invr,
                                gsx, gsy, gsz, g, h)
                        u += du; gx += dgx; gy += dgy; gz += dgz
                        h1 += dh1; h2 += dh2; h3 += dh3
                        h4 += dh4; h5 += dh5; h6 += dh6
                        h7 += dh7; h8 += dh8; h9 += dh9
                    end
                end
            end
            KA.@atomic output[2, i] += gx
            KA.@atomic output[3, i] += gy
            KA.@atomic output[4, i] += gz
            if HS
                KA.@atomic output[5, i]  += h1; KA.@atomic output[6, i]  += h2
                KA.@atomic output[7, i]  += h3; KA.@atomic output[8, i]  += h4
                KA.@atomic output[9, i]  += h5; KA.@atomic output[10, i] += h6
                KA.@atomic output[11, i] += h7; KA.@atomic output[12, i] += h8
                KA.@atomic output[13, i] += h9
            end
            i += WG
        end
    end
end

@kernel function probe_direct_pairs_realmul!(kernel, output, @Const(source_bodies),
        @Const(cell_ranges), @Const(direct_targets), @Const(direct_sources),
        npairs, ::Type{T}, ::Val{HS}, ::Val{WG}) where {T,HS,WG}
    pair_i = @index(Group)
    tid = @index(Local)
    @inbounds begin
        target_cell = direct_targets[pair_i]
        source_cell = direct_sources[pair_i]
        tfirst = cell_ranges[1, target_cell]
        tlast = tfirst + cell_ranges[2, target_cell] - 1
        sfirst = cell_ranges[1, source_cell]
        slast = sfirst + cell_ranges[2, source_cell] - 1
        i = tfirst + tid - 1
        while i <= tlast
            xi = source_bodies[1, i]; yi = source_bodies[2, i]; zi = source_bodies[3, i]
            u = zero(T); gx = zero(T); gy = zero(T); gz = zero(T)
            h1 = zero(T); h2 = zero(T); h3 = zero(T)
            h4 = zero(T); h5 = zero(T); h6 = zero(T)
            h7 = zero(T); h8 = zero(T); h9 = zero(T)
            for j in sfirst:slast
                if i != j
                    dx = xi - source_bodies[1, j]
                    dy = yi - source_bodies[2, j]
                    dz = zi - source_bodies[3, j]
                    r2 = dx * dx + dy * dy + dz * dz
                    if r2 > zero(r2)
                        invr = inv(sqrt(r2))
                        gsx = source_bodies[5, j]; gsy = source_bodies[6, j]
                        gsz = source_bodies[7, j]
                        sigma = source_bodies[kernel.sigma_row, j]
                        g = one(T); h = -T(3)
                        if sigma > zero(T)
                            rho = r2 * invr * sigma
                            if rho <= kernel.rho_t
                                g, h = FastMultipole._gaussianerf_g_h(rho)
                            end
                        end
                        du, dgx, dgy, dgz, dh1, dh2, dh3, dh4, dh5, dh6, dh7, dh8, dh9 =
                            FastMultipole._vortex_pair_ugh(dx, dy, dz, r2, invr,
                                gsx, gsy, gsz, g, h)
                        u += du; gx += dgx; gy += dgy; gz += dgz
                        h1 += dh1; h2 += dh2; h3 += dh3
                        h4 += dh4; h5 += dh5; h6 += dh6
                        h7 += dh7; h8 += dh8; h9 += dh9
                    end
                end
            end
            KA.@atomic output[2, i] += gx
            KA.@atomic output[3, i] += gy
            KA.@atomic output[4, i] += gz
            if HS
                KA.@atomic output[5, i]  += h1; KA.@atomic output[6, i]  += h2
                KA.@atomic output[7, i]  += h3; KA.@atomic output[8, i]  += h4
                KA.@atomic output[9, i]  += h5; KA.@atomic output[10, i] += h6
                KA.@atomic output[11, i] += h7; KA.@atomic output[12, i] += h8
                KA.@atomic output[13, i] += h9
            end
            i += WG
        end
    end
end


# (e) no-series AND the rho divide replaced by a multiply -- isolates the
# Float32 divide on the per-interaction dependency chain.
# (f) no-series AND sigma never fetched (constant) -- isolates the operand load.
@kernel function probe_direct_pairs_nodiv!(kernel, output, @Const(source_bodies),
        @Const(cell_ranges), @Const(direct_targets), @Const(direct_sources),
        npairs, ::Type{T}, ::Val{HS}, ::Val{WG}) where {T,HS,WG}
    pair_i = @index(Group)
    tid = @index(Local)
    @inbounds begin
        target_cell = direct_targets[pair_i]
        source_cell = direct_sources[pair_i]
        tfirst = cell_ranges[1, target_cell]
        tlast = tfirst + cell_ranges[2, target_cell] - 1
        sfirst = cell_ranges[1, source_cell]
        slast = sfirst + cell_ranges[2, source_cell] - 1
        i = tfirst + tid - 1
        while i <= tlast
            xi = source_bodies[1, i]; yi = source_bodies[2, i]; zi = source_bodies[3, i]
            u = zero(T); gx = zero(T); gy = zero(T); gz = zero(T)
            h1 = zero(T); h2 = zero(T); h3 = zero(T)
            h4 = zero(T); h5 = zero(T); h6 = zero(T)
            h7 = zero(T); h8 = zero(T); h9 = zero(T)
            for j in sfirst:slast
                if i != j
                    dx = xi - source_bodies[1, j]
                    dy = yi - source_bodies[2, j]
                    dz = zi - source_bodies[3, j]
                    r2 = dx * dx + dy * dy + dz * dz
                    if r2 > zero(r2)
                        invr = inv(sqrt(r2))
                        gsx = source_bodies[5, j]; gsy = source_bodies[6, j]
                        gsz = source_bodies[7, j]
                        sigma = source_bodies[kernel.sigma_row, j]
                        g = one(T); h = -T(3)
                        if sigma > zero(T)
                            rho = r2 * invr * sigma
                            if rho <= kernel.rho_t
                                g = T(0.5); h = -T(2.5)   # stand-in for the series
                            end
                        end
                        du, dgx, dgy, dgz, dh1, dh2, dh3, dh4, dh5, dh6, dh7, dh8, dh9 =
                            FastMultipole._vortex_pair_ugh(dx, dy, dz, r2, invr,
                                gsx, gsy, gsz, g, h)
                        u += du; gx += dgx; gy += dgy; gz += dgz
                        h1 += dh1; h2 += dh2; h3 += dh3
                        h4 += dh4; h5 += dh5; h6 += dh6
                        h7 += dh7; h8 += dh8; h9 += dh9
                    end
                end
            end
            KA.@atomic output[2, i] += gx
            KA.@atomic output[3, i] += gy
            KA.@atomic output[4, i] += gz
            if HS
                KA.@atomic output[5, i]  += h1; KA.@atomic output[6, i]  += h2
                KA.@atomic output[7, i]  += h3; KA.@atomic output[8, i]  += h4
                KA.@atomic output[9, i]  += h5; KA.@atomic output[10, i] += h6
                KA.@atomic output[11, i] += h7; KA.@atomic output[12, i] += h8
                KA.@atomic output[13, i] += h9
            end
            i += WG
        end
    end
end


@kernel function probe_direct_pairs_nosigma!(kernel, output, @Const(source_bodies),
        @Const(cell_ranges), @Const(direct_targets), @Const(direct_sources),
        npairs, ::Type{T}, ::Val{HS}, ::Val{WG}) where {T,HS,WG}
    pair_i = @index(Group)
    tid = @index(Local)
    @inbounds begin
        target_cell = direct_targets[pair_i]
        source_cell = direct_sources[pair_i]
        tfirst = cell_ranges[1, target_cell]
        tlast = tfirst + cell_ranges[2, target_cell] - 1
        sfirst = cell_ranges[1, source_cell]
        slast = sfirst + cell_ranges[2, source_cell] - 1
        i = tfirst + tid - 1
        while i <= tlast
            xi = source_bodies[1, i]; yi = source_bodies[2, i]; zi = source_bodies[3, i]
            u = zero(T); gx = zero(T); gy = zero(T); gz = zero(T)
            h1 = zero(T); h2 = zero(T); h3 = zero(T)
            h4 = zero(T); h5 = zero(T); h6 = zero(T)
            h7 = zero(T); h8 = zero(T); h9 = zero(T)
            for j in sfirst:slast
                if i != j
                    dx = xi - source_bodies[1, j]
                    dy = yi - source_bodies[2, j]
                    dz = zi - source_bodies[3, j]
                    r2 = dx * dx + dy * dy + dz * dz
                    if r2 > zero(r2)
                        invr = inv(sqrt(r2))
                        gsx = source_bodies[5, j]; gsy = source_bodies[6, j]
                        gsz = source_bodies[7, j]
                        sigma = T(0.1)
                        g = one(T); h = -T(3)
                        if sigma > zero(T)
                            rho = r2 * invr / sigma
                            if rho <= kernel.rho_t
                                g = T(0.5); h = -T(2.5)   # stand-in for the series
                            end
                        end
                        du, dgx, dgy, dgz, dh1, dh2, dh3, dh4, dh5, dh6, dh7, dh8, dh9 =
                            FastMultipole._vortex_pair_ugh(dx, dy, dz, r2, invr,
                                gsx, gsy, gsz, g, h)
                        u += du; gx += dgx; gy += dgy; gz += dgz
                        h1 += dh1; h2 += dh2; h3 += dh3
                        h4 += dh4; h5 += dh5; h6 += dh6
                        h7 += dh7; h8 += dh8; h9 += dh9
                    end
                end
            end
            KA.@atomic output[2, i] += gx
            KA.@atomic output[3, i] += gy
            KA.@atomic output[4, i] += gz
            if HS
                KA.@atomic output[5, i]  += h1; KA.@atomic output[6, i]  += h2
                KA.@atomic output[7, i]  += h3; KA.@atomic output[8, i]  += h4
                KA.@atomic output[9, i]  += h5; KA.@atomic output[10, i] += h6
                KA.@atomic output[11, i] += h7; KA.@atomic output[12, i] += h8
                KA.@atomic output[13, i] += h9
            end
            i += WG
        end
    end
end


const HS = size(state.output, 1) >= 13
dkernel = ext._ka_device_direct_kernel(state.options.direct_kernel, TF)

function time_it(f)
    f()
    ts = Float64[]
    for _ in 1:CALLS
        t0 = time_ns(); f(); push!(ts, (time_ns() - t0)/1e9)
    end
    sort(ts[cld(CALLS,2)+1:end])[1]
end

function time_real(wg)
    kern = ext._cached_kernel(ext.ka_direct_pairs_functor_kernel!, backend, wg)
    time_it() do
        fill!(state.output, zero(TF))
        kern(dkernel, state.output, state.source_bodies, state.cell_ranges,
             state.direct_targets, state.direct_sources, npairs, TF, Val(HS), Val(wg);
             ndrange=npairs*wg)
        KA.synchronize(backend)
    end
end

function time_nogh(wg)
    kern = ext._cached_kernel(probe_direct_pairs_nogh!, backend, wg)
    time_it() do
        fill!(state.output, zero(TF))
        kern(dkernel, state.output, state.source_bodies, state.cell_ranges,
             state.direct_targets, state.direct_sources, npairs, TF, Val(HS), Val(wg);
             ndrange=npairs*wg)
        KA.synchronize(backend)
    end
end

function time_noseries(wg)
    kern = ext._cached_kernel(probe_direct_pairs_noseries!, backend, wg)
    time_it() do
        fill!(state.output, zero(TF))
        kern(dkernel, state.output, state.source_bodies, state.cell_ranges,
             state.direct_targets, state.direct_sources, npairs, TF, Val(HS), Val(wg);
             ndrange=npairs*wg)
        KA.synchronize(backend)
    end
end

function _mk(kern_f)
    return function (wg)
        kern = ext._cached_kernel(kern_f, backend, wg)
        time_it() do
            fill!(state.output, zero(TF))
            kern(dkernel, state.output, state.source_bodies, state.cell_ranges,
                 state.direct_targets, state.direct_sources, npairs, TF, Val(HS), Val(wg);
                 ndrange=npairs*wg)
            KA.synchronize(backend)
        end
    end
end
const time_nodiv   = _mk(probe_direct_pairs_nodiv!)
const time_nosigma = _mk(probe_direct_pairs_nosigma!)
const time_realdiv = _mk(probe_direct_pairs_realdiv!)
const time_realmul = _mk(probe_direct_pairs_realmul!)

function time_noatomic(wg)
    kern = ext._cached_kernel(probe_direct_pairs_noatomic!, backend, wg)
    time_it() do
        fill!(state.output, zero(TF))
        kern(dkernel, state.output, state.source_bodies, state.cell_ranges,
             state.direct_targets, state.direct_sources, npairs, TF, Val(HS), Val(wg);
             ndrange=npairs*wg)
        KA.synchronize(backend)
    end
end

@printf("  %-6s %10s %8s %9s %10s %8s %9s %8s\n",
        "WG", "real s", "atomic%", "no-g/h%", "no-series%", "no-div%", "no-sigma%", "Gint/s")
for wg in (64, 128)
    tr = time_real(wg)
    tn = time_noatomic(wg)
    tg = time_nogh(wg)
    ts = time_noseries(wg)
    td = time_nodiv(wg)
    tq = time_nosigma(wg)
    @printf("  %-6d %10.5f %7.1f%% %8.1f%% %9.1f%% %7.1f%% %8.1f%% %8.2f\n",
            wg, tr, 100*(tr-tn)/tr, 100*(tr-tg)/tr, 100*(tr-ts)/tr,
            100*(tr-td)/tr, 100*(tr-tq)/tr, interactions/tr/1e9)
    flush(stdout)
    # deciding pair: real series both sides, divide vs multiply
    trd = time_realdiv(wg); trm = time_realmul(wg)
    @printf("      -> realdiv %.5f s   realmul %.5f s   divide = %.1f%% of this arm, %.1f%% of real\n",
            trd, trm, 100*(trd-trm)/trd, 100*(trd-trm)/tr)
    flush(stdout)
end
fill!(state.output, zero(TF))
