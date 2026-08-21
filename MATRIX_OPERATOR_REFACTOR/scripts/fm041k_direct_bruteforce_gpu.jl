# fm041k_direct_bruteforce_gpu.jl — CUDA half of the 041k brute-force ceiling
# driver. Included by fm041k_direct_bruteforce.jl only when CUDA is functional;
# shares the pair math (uj_pair, zeta_sgm_r2, g_dgdr, custom_erf) defined there.

@inline _rinv(r2::Float64) = one(r2) / sqrt(r2)
@inline _rinv(r2::Float32) = CUDA.rsqrt(r2)   # F32 lever; never used at F64

# --- opt-variant levers (041k amendment, user direction 2026-08-20) ---------
# Lever 2: hardware/libdevice transcendentals for the (rare) near pairs.
# F32 uses the SFU fast exp (rel err ~2^-21) and libdevice erff; F64 keeps
# the accurate exp and uses libdevice erf (ulp-level match to the vendored
# polynomial, so the 1e-11 F64 gate still applies to the opt variant).
@inline fast_exp(x::Float32) = ccall("extern __nv_fast_expf", llvmcall, Cfloat, (Cfloat,), x)
@inline fast_exp(x::Float64) = exp(x)
@inline fast_erf(x::Float32) = ccall("extern __nv_erff", llvmcall, Cfloat, (Cfloat,), x)
@inline fast_erf(x::Float64) = ccall("extern __nv_erf", llvmcall, Cdouble, (Cdouble,), x)

# Lever 1: far-field saturation cutoff on rho^2 = (r/sigma)^2. Beyond it,
# |1-g| and dgdr (and zeta) are below working-precision epsilon, so the pair
# takes the singular path (g=1, dgdr=0; zeta pair skipped) with zero accuracy
# loss at that precision. F32: rho=6.5 -> |1-g| ~ 3.5e-9 < 2^-24.
# F64: rho=9 -> |1-g| ~ 1.9e-17 < eps.
@inline rho2_cut(::Type{Float32}) = 42.25f0
@inline rho2_cut(::Type{Float64}) = 81.0

@inline function uj_pair_opt(dx::T, dy::T, dz::T, r2::T, rinv::T,
                             gx::T, gy::T, gz::T, si::T, rc2::T) where T
    rho2 = r2 * si * si
    g = one(T); dgdr = zero(T)
    if rho2 <= rc2
        rho = r2 * rinv * si
        aux = T(K2) * rho * fast_exp(-rho2 / 2)
        g = fast_erf(rho / T(SQR2)) - aux
        dgdr = rho * aux
    end
    return uj_tail(dx, dy, dz, r2, rinv, g, dgdr, gx, gy, gz, si)
end

function uj_naive_kernel!(U, J, P, G, n, si::T) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        xi = P[1, i]; yi = P[2, i]; zi = P[3, i]
        u1 = u2 = u3 = zero(T)
        j1 = j2 = j3 = j4 = j5 = j6 = j7 = j8 = j9 = zero(T)
        for q in 1:n
            dx = xi - P[1, q]; dy = yi - P[2, q]; dz = zi - P[3, q]
            r2 = dx * dx + dy * dy + dz * dz
            if r2 > zero(T)
                ux, uy, uz, a1, a2, a3, a4, a5, a6, a7, a8, a9 =
                    uj_pair(dx, dy, dz, r2, _rinv(r2), G[1, q], G[2, q], G[3, q], si)
                u1 += ux; u2 += uy; u3 += uz
                j1 += a1; j2 += a2; j3 += a3; j4 += a4; j5 += a5
                j6 += a6; j7 += a7; j8 += a8; j9 += a9
            end
        end
        U[1, i] = u1; U[2, i] = u2; U[3, i] = u3
        J[1, i] = j1; J[2, i] = j2; J[3, i] = j3
        J[4, i] = j4; J[5, i] = j5; J[6, i] = j6
        J[7, i] = j7; J[8, i] = j8; J[9, i] = j9
    end
    return
end

function uj_tiled_kernel!(U, J, P, G, n, si::T) where T
    tid = threadIdx().x
    i = (blockIdx().x - 1) * blockDim().x + tid
    shP = CuStaticSharedArray(T, (3, TILE))
    shG = CuStaticSharedArray(T, (3, TILE))
    xi = yi = zi = zero(T)
    if i <= n
        xi = P[1, i]; yi = P[2, i]; zi = P[3, i]
    end
    u1 = u2 = u3 = zero(T)
    j1 = j2 = j3 = j4 = j5 = j6 = j7 = j8 = j9 = zero(T)
    for t in 1:cld(n, TILE)
        q0 = (t - 1) * TILE
        ql = q0 + tid
        if ql <= n
            shP[1, tid] = P[1, ql]; shP[2, tid] = P[2, ql]; shP[3, tid] = P[3, ql]
            shG[1, tid] = G[1, ql]; shG[2, tid] = G[2, ql]; shG[3, tid] = G[3, ql]
        end
        sync_threads()
        if i <= n
            for k in 1:min(TILE, n - q0)
                dx = xi - shP[1, k]; dy = yi - shP[2, k]; dz = zi - shP[3, k]
                r2 = dx * dx + dy * dy + dz * dz
                if r2 > zero(T)
                    ux, uy, uz, a1, a2, a3, a4, a5, a6, a7, a8, a9 =
                        uj_pair(dx, dy, dz, r2, _rinv(r2), shG[1, k], shG[2, k], shG[3, k], si)
                    u1 += ux; u2 += uy; u3 += uz
                    j1 += a1; j2 += a2; j3 += a3; j4 += a4; j5 += a5
                    j6 += a6; j7 += a7; j8 += a8; j9 += a9
                end
            end
        end
        sync_threads()
    end
    if i <= n
        U[1, i] = u1; U[2, i] = u2; U[3, i] = u3
        J[1, i] = j1; J[2, i] = j2; J[3, i] = j3
        J[4, i] = j4; J[5, i] = j5; J[6, i] = j6
        J[7, i] = j7; J[8, i] = j8; J[9, i] = j9
    end
    return
end

# Lever 3: register blocking — each thread owns TWO targets (a at base+tid,
# b at base+TILE+tid), so every shared-memory source load is reused twice.
# Combined with levers 1-2 via uj_pair_opt. Threads past n compute into dead
# registers (targets default to the origin) and never write.
function uj_opt_kernel!(U, J, P, G, n, si::T, rc2::T) where T
    tid = threadIdx().x
    base = (blockIdx().x - 1) * (2 * TILE)
    ia = base + tid
    ib = base + TILE + tid
    shP = CuStaticSharedArray(T, (3, TILE))
    shG = CuStaticSharedArray(T, (3, TILE))
    xa = ya = za = xb = yb = zb = zero(T)
    if ia <= n
        xa = P[1, ia]; ya = P[2, ia]; za = P[3, ia]
    end
    if ib <= n
        xb = P[1, ib]; yb = P[2, ib]; zb = P[3, ib]
    end
    ua1 = ua2 = ua3 = zero(T)
    ja1 = ja2 = ja3 = ja4 = ja5 = ja6 = ja7 = ja8 = ja9 = zero(T)
    ub1 = ub2 = ub3 = zero(T)
    jb1 = jb2 = jb3 = jb4 = jb5 = jb6 = jb7 = jb8 = jb9 = zero(T)
    for t in 1:cld(n, TILE)
        q0 = (t - 1) * TILE
        ql = q0 + tid
        if ql <= n
            shP[1, tid] = P[1, ql]; shP[2, tid] = P[2, ql]; shP[3, tid] = P[3, ql]
            shG[1, tid] = G[1, ql]; shG[2, tid] = G[2, ql]; shG[3, tid] = G[3, ql]
        end
        sync_threads()
        for k in 1:min(TILE, n - q0)
            sx = shP[1, k]; sy = shP[2, k]; sz = shP[3, k]
            gx = shG[1, k]; gy = shG[2, k]; gz = shG[3, k]
            dx = xa - sx; dy = ya - sy; dz = za - sz
            r2 = dx * dx + dy * dy + dz * dz
            if r2 > zero(T)
                ux, uy, uz, a1, a2, a3, a4, a5, a6, a7, a8, a9 =
                    uj_pair_opt(dx, dy, dz, r2, _rinv(r2), gx, gy, gz, si, rc2)
                ua1 += ux; ua2 += uy; ua3 += uz
                ja1 += a1; ja2 += a2; ja3 += a3; ja4 += a4; ja5 += a5
                ja6 += a6; ja7 += a7; ja8 += a8; ja9 += a9
            end
            dx = xb - sx; dy = yb - sy; dz = zb - sz
            r2 = dx * dx + dy * dy + dz * dz
            if r2 > zero(T)
                ux, uy, uz, a1, a2, a3, a4, a5, a6, a7, a8, a9 =
                    uj_pair_opt(dx, dy, dz, r2, _rinv(r2), gx, gy, gz, si, rc2)
                ub1 += ux; ub2 += uy; ub3 += uz
                jb1 += a1; jb2 += a2; jb3 += a3; jb4 += a4; jb5 += a5
                jb6 += a6; jb7 += a7; jb8 += a8; jb9 += a9
            end
        end
        sync_threads()
    end
    if ia <= n
        U[1, ia] = ua1; U[2, ia] = ua2; U[3, ia] = ua3
        J[1, ia] = ja1; J[2, ia] = ja2; J[3, ia] = ja3
        J[4, ia] = ja4; J[5, ia] = ja5; J[6, ia] = ja6
        J[7, ia] = ja7; J[8, ia] = ja8; J[9, ia] = ja9
    end
    if ib <= n
        U[1, ib] = ub1; U[2, ib] = ub2; U[3, ib] = ub3
        J[1, ib] = jb1; J[2, ib] = jb2; J[3, ib] = jb3
        J[4, ib] = jb4; J[5, ib] = jb5; J[6, ib] = jb6
        J[7, ib] = jb7; J[8, ib] = jb8; J[9, ib] = jb9
    end
    return
end

function zeta_opt_kernel!(OM, Q, P, G, TG, n, si::T, rc2::T) where T
    tid = threadIdx().x
    base = (blockIdx().x - 1) * (2 * TILE)
    ia = base + tid
    ib = base + TILE + tid
    shP = CuStaticSharedArray(T, (3, TILE))
    shG = CuStaticSharedArray(T, (3, TILE))
    shT = CuStaticSharedArray(T, (3, TILE))
    si2 = si * si
    si3 = si * si * si
    xa = ya = za = xb = yb = zb = zero(T)
    if ia <= n
        xa = P[1, ia]; ya = P[2, ia]; za = P[3, ia]
    end
    if ib <= n
        xb = P[1, ib]; yb = P[2, ib]; zb = P[3, ib]
    end
    oa1 = oa2 = oa3 = qa1 = qa2 = qa3 = zero(T)
    ob1 = ob2 = ob3 = qb1 = qb2 = qb3 = zero(T)
    for t in 1:cld(n, TILE)
        q0 = (t - 1) * TILE
        ql = q0 + tid
        if ql <= n
            shP[1, tid] = P[1, ql]; shP[2, tid] = P[2, ql]; shP[3, tid] = P[3, ql]
            shG[1, tid] = G[1, ql]; shG[2, tid] = G[2, ql]; shG[3, tid] = G[3, ql]
            shT[1, tid] = TG[1, ql]; shT[2, tid] = TG[2, ql]; shT[3, tid] = TG[3, ql]
        end
        sync_threads()
        for k in 1:min(TILE, n - q0)
            sx = shP[1, k]; sy = shP[2, k]; sz = shP[3, k]
            dx = xa - sx; dy = ya - sy; dz = za - sz
            rho2 = (dx * dx + dy * dy + dz * dz) * si2
            if rho2 <= rc2
                z = T(K1) * fast_exp(-rho2 / 2) * si3
                oa1 += z * shG[1, k]; oa2 += z * shG[2, k]; oa3 += z * shG[3, k]
                qa1 += z * shT[1, k]; qa2 += z * shT[2, k]; qa3 += z * shT[3, k]
            end
            dx = xb - sx; dy = yb - sy; dz = zb - sz
            rho2 = (dx * dx + dy * dy + dz * dz) * si2
            if rho2 <= rc2
                z = T(K1) * fast_exp(-rho2 / 2) * si3
                ob1 += z * shG[1, k]; ob2 += z * shG[2, k]; ob3 += z * shG[3, k]
                qb1 += z * shT[1, k]; qb2 += z * shT[2, k]; qb3 += z * shT[3, k]
            end
        end
        sync_threads()
    end
    if ia <= n
        OM[1, ia] = oa1; OM[2, ia] = oa2; OM[3, ia] = oa3
        Q[1, ia] = qa1; Q[2, ia] = qa2; Q[3, ia] = qa3
    end
    if ib <= n
        OM[1, ib] = ob1; OM[2, ib] = ob2; OM[3, ib] = ob3
        Q[1, ib] = qb1; Q[2, ib] = qb2; Q[3, ib] = qb3
    end
    return
end

function tg_kernel!(TG, J, G, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        g1 = G[1, i]; g2 = G[2, i]; g3 = G[3, i]
        TG[1, i] = J[1, i] * g1 + J[2, i] * g2 + J[3, i] * g3
        TG[2, i] = J[4, i] * g1 + J[5, i] * g2 + J[6, i] * g3
        TG[3, i] = J[7, i] * g1 + J[8, i] * g2 + J[9, i] * g3
    end
    return
end

function zeta_naive_kernel!(OM, Q, P, G, TG, n, si::T) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        si3 = si * si * si
        xi = P[1, i]; yi = P[2, i]; zi = P[3, i]
        o1 = o2 = o3 = q1 = q2 = q3 = zero(T)
        for q in 1:n
            dx = xi - P[1, q]; dy = yi - P[2, q]; dz = zi - P[3, q]
            z = zeta_sgm_r2(dx * dx + dy * dy + dz * dz, si, si3)
            o1 += z * G[1, q]; o2 += z * G[2, q]; o3 += z * G[3, q]
            q1 += z * TG[1, q]; q2 += z * TG[2, q]; q3 += z * TG[3, q]
        end
        OM[1, i] = o1; OM[2, i] = o2; OM[3, i] = o3
        Q[1, i] = q1; Q[2, i] = q2; Q[3, i] = q3
    end
    return
end

function zeta_tiled_kernel!(OM, Q, P, G, TG, n, si::T) where T
    tid = threadIdx().x
    i = (blockIdx().x - 1) * blockDim().x + tid
    shP = CuStaticSharedArray(T, (3, TILE))
    shG = CuStaticSharedArray(T, (3, TILE))
    shT = CuStaticSharedArray(T, (3, TILE))
    si3 = si * si * si
    xi = yi = zi = zero(T)
    if i <= n
        xi = P[1, i]; yi = P[2, i]; zi = P[3, i]
    end
    o1 = o2 = o3 = q1 = q2 = q3 = zero(T)
    for t in 1:cld(n, TILE)
        q0 = (t - 1) * TILE
        ql = q0 + tid
        if ql <= n
            shP[1, tid] = P[1, ql]; shP[2, tid] = P[2, ql]; shP[3, tid] = P[3, ql]
            shG[1, tid] = G[1, ql]; shG[2, tid] = G[2, ql]; shG[3, tid] = G[3, ql]
            shT[1, tid] = TG[1, ql]; shT[2, tid] = TG[2, ql]; shT[3, tid] = TG[3, ql]
        end
        sync_threads()
        if i <= n
            for k in 1:min(TILE, n - q0)
                dx = xi - shP[1, k]; dy = yi - shP[2, k]; dz = zi - shP[3, k]
                z = zeta_sgm_r2(dx * dx + dy * dy + dz * dz, si, si3)
                o1 += z * shG[1, k]; o2 += z * shG[2, k]; o3 += z * shG[3, k]
                q1 += z * shT[1, k]; q2 += z * shT[2, k]; q3 += z * shT[3, k]
            end
        end
        sync_threads()
    end
    if i <= n
        OM[1, i] = o1; OM[2, i] = o2; OM[3, i] = o3
        Q[1, i] = q1; Q[2, i] = q2; Q[3, i] = q3
    end
    return
end

# --- device state + harness -------------------------------------------------

struct DevState{T}
    P::CuMatrix{T}; G::CuMatrix{T}
    U::CuMatrix{T}; J::CuMatrix{T}
    TG::CuMatrix{T}; OM::CuMatrix{T}; Q::CuMatrix{T}
end

function upload(::Type{T}, P64, G64) where T
    n = size(P64, 2)
    t0 = time_ns()
    P = CuMatrix{T}(P64); G = CuMatrix{T}(G64)
    CUDA.synchronize()
    h2d = (time_ns() - t0) / 1e9
    st = DevState{T}(P, G, CUDA.zeros(T, 3, n), CUDA.zeros(T, 9, n),
                     CUDA.zeros(T, 3, n), CUDA.zeros(T, 3, n), CUDA.zeros(T, 3, n))
    return st, h2d
end

function make_run(st::DevState{T}, n, si::T, kern::Symbol, var::Symbol) where T
    if var == :opt
        B = cld(n, 2 * TILE)
        Bt = cld(n, TILE)
        rc2 = rho2_cut(T)
        if kern == :uj
            return () -> begin
                @cuda threads=TILE blocks=B uj_opt_kernel!(st.U, st.J, st.P, st.G, n, si, rc2)
                CUDA.synchronize()
            end
        else
            return () -> begin
                @cuda threads=TILE blocks=B uj_opt_kernel!(st.U, st.J, st.P, st.G, n, si, rc2)
                @cuda threads=TILE blocks=Bt tg_kernel!(st.TG, st.J, st.G, n)
                @cuda threads=TILE blocks=B zeta_opt_kernel!(st.OM, st.Q, st.P, st.G, st.TG, n, si, rc2)
                CUDA.synchronize()
            end
        end
    end
    B = cld(n, TILE)
    ujk = var == :tiled ? uj_tiled_kernel! : uj_naive_kernel!
    zk = var == :tiled ? zeta_tiled_kernel! : zeta_naive_kernel!
    if kern == :uj
        return () -> begin
            @cuda threads=TILE blocks=B ujk(st.U, st.J, st.P, st.G, n, si)
            CUDA.synchronize()
        end
    else
        return () -> begin
            @cuda threads=TILE blocks=B ujk(st.U, st.J, st.P, st.G, n, si)
            @cuda threads=TILE blocks=B tg_kernel!(st.TG, st.J, st.G, n)
            @cuda threads=TILE blocks=B zk(st.OM, st.Q, st.P, st.G, st.TG, n, si)
            CUDA.synchronize()
        end
    end
end

function bench(run!)
    t1 = @elapsed run!()
    if t1 > 15.0
        ts = [(@elapsed run!()) for _ in 1:2]
        return median(ts), minimum(ts), 2
    end
    run!()
    ts = [(@elapsed run!()) for _ in 1:5]
    return median(ts), minimum(ts), 5
end

function d2h_time(st::DevState)
    t0 = time_ns()
    Array(st.U); Array(st.J)
    CUDA.synchronize()
    return (time_ns() - t0) / 1e9
end

# --- GPU accuracy vs the CPU reference (tiled ujsfs, both precisions) -------

function gpu_accuracy!(io, ref::RefBlocks)
    gate_ok = true
    for var in (:tiled, :opt), T in (Float64, Float32)
        st, _ = upload(T, ref.P, ref.G)
        run! = make_run(st, ref.n, T(ref.si), :ujsfs, var)
        run!()
        gU = Array(st.U); gJ = Array(st.J); gOM = Array(st.OM); gQ = Array(st.Q)
        gE = form_E(Float64.(gJ), Float64.(gOM), Float64.(gQ))
        tag = (T === Float64 ? "f64" : "f32") * (var == :opt ? "_opt" : "")
        for (blk, gpu, cpu) in (("U", gU, ref.U), ("J", gJ, ref.J),
                                ("Omega", gOM, ref.OM), ("Q", gQ, ref.Q),
                                ("E", gE, ref.E))
            m, r = relerr(gpu, cpu)
            @printf(io, "%d,%s,%s,%.3e,%.3e\n", ref.n, tag, blk, m, r)
            @printf("acc n=%d %-7s %-5s max rel %.3e\n", ref.n, tag, blk, m)
            if T === Float64 && m > 1e-11
                gate_ok = false
            end
        end
        CUDA.reclaim()
    end
    return gate_ok
end

# --- sweep ------------------------------------------------------------------

function run_sweep(io, variants)
    println(io, "kernel,variant,precision,n,reps,median_s,min_s,h2d_s,d2h_s,pairs_per_s,gflops_nominal,jobid")
    flush(io)
    series = [(k, v, T) for k in (:uj, :ujsfs) for v in variants
                        for T in (Float64, Float32)]
    active = Dict(s => true for s in series)
    for e in EXPS
        n = round(Int, 10.0^e)
        any(values(active)) || break
        P64, G64, sigma = gen_bodies(n)
        si = 1.0 / sigma
        for T in (Float64, Float32)
            any(active[(k, v, T)] for k in (:uj, :ujsfs) for v in variants) || continue
            st, h2d = upload(T, P64, G64)
            d2h = d2h_time(st)
            tag = T === Float64 ? "f64" : "f32"
            for k in (:uj, :ujsfs), v in variants
                active[(k, v, T)] || continue
                med, mn, reps = bench(make_run(st, n, T(si), k, v))
                pairs = Float64(n)^2 / med
                fl = k == :uj ? FLOPS_UJ : FLOPS_UJ + FLOPS_ZETA
                @printf(io, "%s,%s,%s,%d,%d,%.6e,%.6e,%.4e,%.4e,%.4e,%.1f,%s\n",
                        k, v, tag, n, reps, med, mn, h2d, d2h, pairs,
                        Float64(n)^2 * fl / med / 1e9, JOBID)
                flush(io)
                @printf("%-5s %-5s %s n=%.3g  median %.4f s  (%.3g pairs/s)\n",
                        k, v, tag, n, med, pairs)
                if med > TIME_CAP_S
                    active[(k, v, T)] = false
                    @printf("  -> series %s/%s/%s stops (>%g s)\n", k, v, tag, TIME_CAP_S)
                end
            end
            st = nothing
            CUDA.reclaim()
        end
    end
end
