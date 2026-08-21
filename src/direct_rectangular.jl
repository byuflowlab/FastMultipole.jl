#------- RECTANGULAR (SOURCE-SET -> DISTINCT TARGET-SET) DIRECT EVALUATION (task 051 stage 1) -------#
#
# Standalone brute-force cross-pass evaluation OUTSIDE the radix FMM framework:
# a distinct source set induces velocity (+ optional velocity gradient) at a
# distinct target set. Motivated by the FLOWPanel 018 rotor step's two cross
# passes (BRAINSTORM 023 profiling):
#   pass 1: gaussianerf-regularized vortex particles -> arbitrary target points
#   pass 2: FLOWPanel panel elements -> arbitrary target points
#
# Both passes request velocity always and the velocity gradient optionally, and
# NEVER the scalar potential (FLOWPanel_simulate.jl `_sa_wake_influence!` /
# `_sa_body_influence!` pass scalar_potential=false unconditionally;
# velocity_gradient is gated on requires_hessian / body_hessian_to_particles).
# The functors below therefore emit U (+J when `gradient=true`) and no
# potential.
#
# API (host here; CUDA methods ride the lazy load in translate_batched_cuda.jl):
#
#   direct_rectangular!(out, targets, functor, sources; gradient=false)
#
# - `targets`  : TF matrix, >= 3 rows; rows 1:3 = target position, one column
#                per target. Extra rows are ignored.
# - `sources`  : TF matrix, one column per source; row layout is fixed by the
#                functor (see below).
# - `out`      : TF matrix, one column per target; rows 1:3 accumulate U.
#                With `gradient=true`, rows 4:12 accumulate J in FLOWVPM's
#                order J[(j-1)*3+i] = du_i/dx_j (column-major 3x3), and `out`
#                must have >= 12 rows. Results are ACCUMULATED (+=) so several
#                source systems can be applied in sequence; zero `out` first
#                for a fresh evaluation.
# - `functor`  : isbits kernel functor (compile-time pair math selection, same
#                mechanism as the `AbstractDirectKernel` nearfield functors in
#                containers.jl — one kernel instantiation per functor type,
#                never a runtime branch in the pair loop).
#
# Precision: TF = Float64 primary; the point functor is precision-generic and
# supports Float32 end-to-end (kept behind an explicit caller choice — pass
# Float32 matrices).

abstract type AbstractRectangularKernel end

"""
    RectangularGaussianErfVortex()

Rectangular pair kernel for gaussianerf-regularized vortex particles
(FLOWVPM's default kernel), kerneloffset-free. Source row layout (7 rows):

| rows | content |
|---|---|
| 1:3 | particle position |
| 4:6 | vector strength Gamma |
| 7   | smoothing radius sigma (> 0) |

Pair math transcribed from FLOWVPM's CPU reference `fmm.direct!` overload
(FLOWVPM.jl/src/FLOWVPM_fmm.jl:144-218) with `g_dgdr_gauserf`
(FLOWVPM_kernel.jl:54-57): exact match including the vendored fdlibm
`custom_erf` (FLOWVPM_gpu_erf.jl), so host/device/CPU-FLOWVPM agree to
roundoff. The only excluded pair is exact coincidence (`r2 == 0`), matching
the CPU semantics — NOT the absolute `eps2 = 1e-6` guard used by the FLOWVPM
CUDA extension's `gpu_interaction!`.
"""
struct RectangularGaussianErfVortex <: AbstractRectangularKernel end

"""
    RectangularPanelInfluence(filament_reg=1)
    RectangularPanelInfluence(:vatistas | :compact | :gaussian)

Rectangular pair kernel for FLOWPanel panel elements (the 018 rotor element
set), transcribed from FLOWPanel.jl `src/FLOWPanel_elements_fmm.jl` at commit
75b45c7, plus the working-tree (branch fastmultipole, 2026-08-20) selectable
filament-regularization families for the vortex-ring branch. `filament_reg`
selects the bound-vortex filament family (FLOWPanel's
`FilamentRegularization` enum, working-tree elements_fmm.jl:910-915):

- `1` = Vatistas n=2 (legacy/HEAD; default here)
- `2` = compact support (working-tree elements_fmm.jl:971-976, 1017-1029)
- `3` = Gaussian / Lamb-Oseen (working-tree elements_fmm.jl:977-982, 1030-1042;
  FLOWPanel's working-tree DEFAULT)

The family is stamped into the pair loop at compile time (one kernel
instantiation per family — no runtime branch per edge, matching FLOWPanel's
`Val(FILAMENT_REGULARIZATION[])` function-barrier contract at
elements_fmm.jl:936-944). Source row layout (17 rows, fixed max-vertex
layout):

| row | content |
|---|---|
| 1   | element tag: 1=ConstantSource, 2=ConstantDoublet, 3=VortexRing, 4=ConstantSource+VortexRing, 5=ConstantSource+ConstantDoublet |
| 2   | vertex count nv (3 for tags 1/2/4/5; 3 or 4 for tag 3) |
| 3:5 | vertex 1 |
| 6:8 | vertex 2 |
| 9:11| vertex 3 |
| 12:14 | vertex 4 (ignored when nv == 3) |
| 15  | strength 1 (sigma for tags 1/4/5; mu for tag 2; Gamma for tag 3) |
| 16  | strength 2 (Gamma for tag 4; mu for tag 5; ignored otherwise) |
| 17  | kerneloffset (per-source regularization radius) |

Emits velocity (+gradient when armed), never the scalar potential (the 018
passes request scalar_potential=false; see file header). Includes FLOWPanel's
self-pair short-circuit (`_is_self_pair` / `_self_limit` velocity behavior,
gradient zeroed) so targets sitting exactly on a panel centroid reproduce the
CPU `direct!` surface limits. TE-wake attachments (`_induced_wake` for
`RigidWakeBody` TE panels) are NOT evaluated here; pack each TE wake quad as
an extra tag-3 ring source instead.
"""
struct RectangularPanelInfluence <: AbstractRectangularKernel
    filament_reg::Int32
end

RectangularPanelInfluence() = RectangularPanelInfluence(Int32(1))
function RectangularPanelInfluence(family::Symbol)
    family === :vatistas && return RectangularPanelInfluence(Int32(1))
    family === :compact && return RectangularPanelInfluence(Int32(2))
    family === :gaussian && return RectangularPanelInfluence(Int32(3))
    throw(ArgumentError("unknown filament regularization $(repr(family)); " *
        "use :vatistas, :compact, or :gaussian"))
end

# Int -> Val barrier (one dynamic dispatch per direct_rectangular! call / CUDA
# launch, never inside the pair loop)
@inline function _rect_reg_val(reg::Integer)
    reg == 2 && return Val(2)
    reg == 3 && return Val(3)
    return Val(1)
end

rect_source_rows(::RectangularGaussianErfVortex) = 7
rect_source_rows(::RectangularPanelInfluence) = 17
rect_output_rows(gradient::Bool) = gradient ? 12 : 3

#------- vendored fdlibm erf (FLOWVPM.jl/src/FLOWVPM_gpu_erf.jl:159-191, 124-156) -------#
# Vendored so the point kernel matches FLOWVPM's gaussianerf bit-for-bit on
# host and device without a FLOWVPM/SpecialFunctions dependency. Branch-only
# polynomial evaluation: GPU-compilable as-is.

const _RECT_ERF64 = (
    erx = 8.45062911510467529297e-01,
    pp0 = 1.28379167095512558561e-01, pp1 = -3.25042107247001499370e-01,
    pp2 = -2.84817495755985104766e-02, pp3 = -5.77027029648944159157e-03,
    pp4 = -2.37630166566501626084e-05,
    qq1 = 3.97917223959155352819e-01, qq2 = 6.50222499887672944485e-02,
    qq3 = 5.08130628187576562776e-03, qq4 = 1.32494738004321644526e-04,
    qq5 = -3.96022827877536812320e-06,
    pa0 = -2.36211856075265944077e-03, pa1 = 4.14856118683748331666e-01,
    pa2 = -3.72207876035701323847e-01, pa3 = 3.18346619901161753674e-01,
    pa4 = -1.10894694282396677476e-01, pa5 = 3.54783043256182359371e-02,
    pa6 = -2.16637559486879084300e-03,
    qa1 = 1.06420880400844228286e-01, qa2 = 5.40397917702171048937e-01,
    qa3 = 7.18286544141962662868e-02, qa4 = 1.26171219808761642112e-01,
    qa5 = 1.36370839120290507362e-02, qa6 = 1.19844998467991074170e-02,
    ra0 = -9.86494403484714822705e-03, ra1 = -6.93858572707181764372e-01,
    ra2 = -1.05586262253232909814e+01, ra3 = -6.23753324503260060396e+01,
    ra4 = -1.62396669462573470355e+02, ra5 = -1.84605092906711035994e+02,
    ra6 = -8.12874355063065934246e+01, ra7 = -9.81432934416914548592e+00,
    sa1 = 1.96512716674392571292e+01, sa2 = 1.37657754143519042600e+02,
    sa3 = 4.34565877475229228821e+02, sa4 = 6.45387271733267880336e+02,
    sa5 = 4.29008140027567833386e+02, sa6 = 1.08635005541779435134e+02,
    sa7 = 6.57024977031928170135e+00, sa8 = -6.04244152148580987438e-02,
    rb0 = -9.86494292470009928597e-03, rb1 = -7.99283237680523006574e-01,
    rb2 = -1.77579549177547519889e+01, rb3 = -1.60636384855821916062e+02,
    rb4 = -6.37566443368389627722e+02, rb5 = -1.02509513161107724954e+03,
    rb6 = -4.83519191608651397019e+02,
    sb1 = 3.03380607434824582924e+01, sb2 = 3.25792512996573918826e+02,
    sb3 = 1.53672958608443695994e+03, sb4 = 3.19985821950859553908e+03,
    sb5 = 2.55305040643316442583e+03, sb6 = 4.74528541206955367215e+02,
    sb7 = -2.24409524465858183362e+01,
)

@inline function _rect_erf(x::T) where T<:Union{Float32,Float64}
    C = _RECT_ERF64
    xabs = abs(x)
    sgn = sign(x)
    oneval = one(x)
    val = sgn * oneval
    if xabs < T(0.84375)
        z = x * x
        r = T(C.pp0) + z*(T(C.pp1) + z*(T(C.pp2) + z*(T(C.pp3) + z*T(C.pp4))))
        s = oneval + z*(T(C.qq1) + z*(T(C.qq2) + z*(T(C.qq3) + z*(T(C.qq4) + z*T(C.qq5)))))
        y = r / s
        val = sgn * (xabs + xabs*y)
    elseif xabs < T(1.25)
        s = xabs - oneval
        P = T(C.pa0) + s*(T(C.pa1) + s*(T(C.pa2) + s*(T(C.pa3) + s*(T(C.pa4) + s*(T(C.pa5) + s*T(C.pa6))))))
        Q = oneval + s*(T(C.qa1) + s*(T(C.qa2) + s*(T(C.qa3) + s*(T(C.qa4) + s*(T(C.qa5) + s*T(C.qa6))))))
        val = sgn * (T(C.erx) + P/Q)
    elseif xabs < T(2.857142857142857)
        s = oneval / (x*x)
        R = T(C.ra0) + s*(T(C.ra1) + s*(T(C.ra2) + s*(T(C.ra3) + s*(T(C.ra4) + s*(T(C.ra5) + s*(T(C.ra6) + s*T(C.ra7)))))))
        S = oneval + s*(T(C.sa1) + s*(T(C.sa2) + s*(T(C.sa3) + s*(T(C.sa4) + s*(T(C.sa5) + s*(T(C.sa6) + s*(T(C.sa7) + s*T(C.sa8))))))))
        r = exp(-x*x - T(0.5625) + R/S)
        val = sgn * (oneval - r/xabs)
    elseif xabs < T(6.0)
        s = oneval / (x*x)
        R = T(C.rb0) + s*(T(C.rb1) + s*(T(C.rb2) + s*(T(C.rb3) + s*(T(C.rb4) + s*(T(C.rb5) + s*T(C.rb6))))))
        S = oneval + s*(T(C.sb1) + s*(T(C.sb2) + s*(T(C.sb3) + s*(T(C.sb4) + s*(T(C.sb5) + s*(T(C.sb6) + s*T(C.sb7)))))))
        r = exp(-x*x - T(0.5625) + R/S)
        val = sgn * (oneval - r/xabs)
    end
    return val
end

# gaussianerf g(rho), rho*g'(rho): FLOWVPM_kernel.jl:54-57 g_dgdr_gauserf with
# const2 = sqrt(2/pi), sqr2 = sqrt(2) (FLOWVPM.jl:77,80)
@inline function _rect_g_dgdr_gauserf(rho::T) where T
    aux = T(0.7978845608028654) * rho * exp(-rho*rho/2)   # sqrt(2/pi)*rho*exp(-rho^2/2)
    return _rect_erf(rho / T(1.4142135623730951)) - aux, rho * aux
end

#------- point (gaussianerf vortex) pair math -------#
# FLOWVPM_fmm.jl:144-218 (fmm.direct! for ParticleField), const4 = 1/(4pi).
# Returns (U, J-cols) as scalars; J order J[(j-1)*3+i] = du_i/dx_j.

@inline function _rect_point_pair(::RectangularGaussianErfVortex, tx::T, ty, tz,
        sx, sy, sz, gamma_x, gamma_y, gamma_z, sigma, ::Val{GRAD}) where {T,GRAD}
    dx = tx - sx
    dy = ty - sy
    dz = tz - sz
    r2 = dx*dx + dy*dy + dz*dz
    z3 = zero(T)
    if iszero(r2)   # self-distance guard: exact coincidence only (CPU semantics)
        return z3, z3, z3, z3, z3, z3, z3, z3, z3, z3, z3, z3
    end
    r = sqrt(r2)
    g_sgm, dg_sgmdr = _rect_g_dgdr_gauserf(r / sigma)   # FLOWVPM_fmm.jl:165
    r3inv = one(T) / (r2 * r)                           # FLOWVPM_fmm.jl:168
    c4 = T(ONE_OVER_4π)
    crss1 = -c4 * r3inv * (dy*gamma_z - dz*gamma_y)     # FLOWVPM_fmm.jl:169-171
    crss2 = -c4 * r3inv * (dz*gamma_x - dx*gamma_z)
    crss3 = -c4 * r3inv * (dx*gamma_y - dy*gamma_x)
    Ux = g_sgm * crss1                                  # FLOWVPM_fmm.jl:175-177
    Uy = g_sgm * crss2
    Uz = g_sgm * crss3
    if !GRAD
        return Ux, Uy, Uz, z3, z3, z3, z3, z3, z3, z3, z3, z3
    end
    aux = dg_sgmdr / (sigma*r) - 3*g_sgm / r2           # FLOWVPM_fmm.jl:186
    aux2 = -c4 * g_sgm * r3inv                          # FLOWVPM_fmm.jl:189
    du1x1 = aux * crss1 * dx                            # FLOWVPM_fmm.jl:191-201
    du2x1 = aux * crss2 * dx - aux2 * gamma_z
    du3x1 = aux * crss3 * dx + aux2 * gamma_y
    du1x2 = aux * crss1 * dy + aux2 * gamma_z
    du2x2 = aux * crss2 * dy
    du3x2 = aux * crss3 * dy - aux2 * gamma_x
    du1x3 = aux * crss1 * dz - aux2 * gamma_y
    du2x3 = aux * crss2 * dz + aux2 * gamma_x
    du3x3 = aux * crss3 * dz
    return Ux, Uy, Uz, du1x1, du2x1, du3x1, du1x2, du2x2, du3x2, du1x3, du2x3, du3x3
end

#------- panel element math (FLOWPanel_elements_fmm.jl @ 75b45c7) -------#
# All functions below are faithful transcriptions; citations are
# FLOWPanel.jl/src/FLOWPanel_elements_fmm.jl line numbers at commit 75b45c7.

# rotate_to_panel (elements_fmm.jl:39-74): returns panel-frame axes as SVectors
@inline function _rect_rotate_to_panel(v1::SVector{3,T}, v2::SVector{3,T}, v3::SVector{3,T}) where T
    e1 = v2 - v1
    e2 = v3 - v1
    nz = SVector{3,T}(e1[2]*e2[3] - e1[3]*e2[2],
                      e1[3]*e2[1] - e1[1]*e2[3],
                      e1[1]*e2[2] - e1[2]*e2[1])
    nznorm = sqrt(nz[1]*nz[1] + nz[2]*nz[2] + nz[3]*nz[3])
    nz = nz / (nznorm + eps(nznorm))
    nx = e2
    nxnorm = sqrt(nx[1]*nx[1] + nx[2]*nx[2] + nx[3]*nx[3])
    nx = nx / (nxnorm + eps(nxnorm))
    ny = SVector{3,T}(nz[2]*nx[3] - nz[3]*nx[2],
                      nz[3]*nx[1] - nz[1]*nx[3],
                      nz[1]*nx[2] - nz[2]*nx[1])
    return nx, ny, nz   # columns of R
end

# minimum_distance (elements_fmm.jl:335-366): point-to-segment distance
@inline function _rect_minimum_distance(A::SVector{3,T}, B::SVector{3,T}, p::SVector{3,T}) where T
    AB = B - A
    Ap = p - A
    denom = AB[1]*AB[1] + AB[2]*AB[2] + AB[3]*AB[3]
    proj = (Ap[1]*AB[1] + Ap[2]*AB[2] + Ap[3]*AB[3]) / denom
    proj = clamp(proj, zero(T), one(T))
    d = p - (A + proj * AB)
    return sqrt(d[1]*d[1] + d[2]*d[2] + d[3]*d[3])
end

# regularize (elements_fmm.jl:954-958): compact-support doublet regularizer
@inline function _rect_regularize(distance::T, core_size::T) where T
    return distance < core_size ? (distance - core_size)*(distance - core_size) : zero(T)
end

# recurse_source_dipole (elements_fmm.jl:403-430)
@inline function _rect_edge_prelims(tRx::T, tRy, tRz, vx_i, vy_i, vx_ip1, vy_ip1) where T
    dxp = tRx - vx_ip1
    dyp = tRy - vy_ip1
    eip1 = dxp*dxp + tRz*tRz
    hip1 = dxp*dyp
    rip1 = sqrt(eip1 + dyp*dyp)
    dxi = tRx - vx_i
    dyi = tRy - vy_i
    ei = dxi*dxi + tRz*tRz
    hi = dxi*dyi
    ri = sqrt(ei + dyi*dyi)
    dx = vx_ip1 - vx_i
    dy = vy_ip1 - vy_i
    ds = sqrt(dx*dx + dy*dy)
    R_dot_s = dx*dxi + dy*dyi
    return eip1, hip1, rip1, ei, hi, ri, ds, dx, dy, R_dot_s
end

# shared tan_term with the extension-singularity guard (elements_fmm.jl:446-461)
@inline function _rect_solid_angle_tan(tRx::T, tRy, tRz, ei, hi, ri, eip1, hip1, rip1,
        ds, dx, dy, R_dot_s) where T
    if (tRx == zero(T) && tRy == zero(T) && tRz == zero(T)) ||
       abs(abs(R_dot_s) - ri*ds) <= T(1e-12) * ri * ds
        return zero(T)
    else
        arg1 = (dy*ei - hi*dx) / ri
        arg2 = (dy*eip1 - hip1*dx) / rip1
        num = dx * tRz * (arg1 - arg2)
        den = tRz*tRz*dx*dx + arg1*arg2
        return atan(num, den)
    end
end

# ConstantSource per-edge velocity/gradient in the panel frame
# (compute_source_dipole for ConstantSource, elements_fmm.jl:434-512; VS/GS
# blocks only — the 018 passes never request the potential)
@inline function _rect_edge_source(tRx::T, tRy, tRz, vx_i, vy_i, vx_ip1, vy_ip1,
        ::Val{GRAD}) where {T,GRAD}
    eip1, hip1, rip1, ei, hi, ri, ds, dx, dy, R_dot_s =
        _rect_edge_prelims(tRx, tRy, tRz, vx_i, vy_i, vx_ip1, vy_ip1)
    num = max(eps(T), ri + rip1 - ds)                      # elements_fmm.jl:439
    log_term = log(num / (ri + rip1 + ds))
    tan_term = _rect_solid_angle_tan(tRx, tRy, tRz, ei, hi, ri, eip1, hip1, rip1,
        ds, dx, dy, R_dot_s)
    u = SVector{3,T}(dy/ds*log_term, -dx/ds*log_term, tan_term)  # elements_fmm.jl:471-475
    if !GRAD
        return u, zero(SMatrix{3,3,T,9})
    end
    d2 = ds*ds                                             # elements_fmm.jl:480-508
    r_plus_rp1 = ri + rip1
    r_plus_rp1_2 = r_plus_rp1 * r_plus_rp1
    r_times_rp1 = ri * rip1
    rho = r_times_rp1 + (tRx - vx_i)*(tRx - vx_ip1) + (tRy - vy_i)*(tRy - vy_ip1) + tRz*tRz
    lambda = (tRx - vx_i)*(tRy - vy_ip1) - (tRx - vx_ip1)*(tRy - vy_i)
    ri_inv = 1/ri
    rip1_inv = 1/rip1
    val1 = r_plus_rp1_2 - d2
    val2 = (tRx - vx_i)*ri_inv + (tRx - vx_ip1)*rip1_inv
    val3 = (tRy - vy_i)*ri_inv + (tRy - vy_ip1)*rip1_inv
    val4 = r_plus_rp1 / (r_times_rp1 * rho)
    phi_xx = 2*dy/val1*val2
    phi_xy = -2*dx/val1*val2
    phi_xz = tRz*dy*val4
    phi_yy = -2*dx/val1*val3
    phi_yz = -tRz*dx*val4
    phi_zz = lambda*val4
    g = SMatrix{3,3,T,9}(phi_xx, phi_xy, phi_xz,
                         phi_xy, phi_yy, phi_yz,
                         phi_xz, phi_yz, phi_zz)
    return u, g
end

# ConstantDoublet per-edge velocity/gradient in the panel frame
# (compute_source_dipole for ConstantDoublet, elements_fmm.jl:514-590; VS/GS
# blocks only). `reg_term` regularizes the velocity denominator only
# (elements_fmm.jl:545), exactly as the CPU does.
@inline function _rect_edge_doublet(tRx::T, tRy, tRz, vx_i, vy_i, vx_ip1, vy_ip1,
        reg_term::T, ::Val{GRAD}) where {T,GRAD}
    eip1, hip1, rip1, ei, hi, ri, ds, dx, dy, R_dot_s =
        _rect_edge_prelims(tRx, tRy, tRz, vx_i, vy_i, vx_ip1, vy_ip1)
    tan_term = _rect_solid_angle_tan(tRx, tRy, tRz, ei, hi, ri, eip1, hip1, rip1,
        ds, dx, dy, R_dot_s)
    r_plus_rp1 = ri + rip1
    r_times_rp1 = ri * rip1
    rho = r_times_rp1 + (tRx - vx_i)*(tRx - vx_ip1) + (tRy - vy_i)*(tRy - vy_ip1) + tRz*tRz
    lambda = (tRx - vx_i)*(tRy - vy_ip1) - (tRx - vx_ip1)*(tRy - vy_i)
    val4v = r_plus_rp1 / (r_times_rp1 * rho + reg_term)    # elements_fmm.jl:545
    u = -SVector{3,T}(tRz*dy*val4v, -tRz*dx*val4v, lambda*val4v)  # elements_fmm.jl:546-550
    if !GRAD
        return u, zero(SMatrix{3,3,T,9})
    end
    r_plus_rp1_2 = r_plus_rp1 * r_plus_rp1                 # elements_fmm.jl:554-586
    val1 = r_times_rp1 * r_plus_rp1_2 + rho * rip1 * rip1
    val1 /= rho * ri * r_plus_rp1
    val2 = r_times_rp1 * r_plus_rp1_2 + rho * ri * ri
    val2 /= rho * rip1 * r_plus_rp1
    val3 = r_plus_rp1 / (rho * r_times_rp1 * r_times_rp1)
    psi_xx = tRz*dy*val3*((tRx - vx_i)*val1 + (tRx - vx_ip1)*val2)
    psi_xy = tRz*dy*val3*((tRy - vy_i)*val1 + (tRy - vy_ip1)*val2)
    psi_yy = -tRz*dx*val3*((tRy - vy_i)*val1 + (tRy - vy_ip1)*val2)
    val4 = r_plus_rp1_2 / rho
    val5 = (ri*ri - r_times_rp1 + rip1*rip1) / r_times_rp1
    val6 = tRz*(val4 + val5)
    psi_zz = lambda*val3*val6
    val7 = r_times_rp1 - tRz*val6
    val8 = val3*val7
    psi_xz = -dy*val8
    psi_yz = dx*val8
    g = SMatrix{3,3,T,9}(psi_xx, psi_xy, psi_xz,
                         psi_xy, psi_yy, psi_yz,
                         psi_xz, psi_yz, psi_zz)
    return u, g
end

# planar tri source/doublet influence (_induced for
# ConstantSource/ConstantDoublet, elements_fmm.jl:710-806): rotated-frame edge
# loop; final rotation back u <- -1/4pi R u, g <- -1/4pi R g R^T
# (elements_fmm.jl:796-803). Per-strength-unit result (caller multiplies).
@inline function _rect_tri_source_doublet(target::SVector{3,T}, v1::SVector{3,T},
        v2::SVector{3,T}, v3::SVector{3,T}, kerneloffset::T,
        ::Val{DOUBLET}, ::Val{GRAD}) where {T,DOUBLET,GRAD}
    nx, ny, nz = _rect_rotate_to_panel(v1, v2, v3)
    centroid = (v1 + v2 + v3) * T(0.3333333333333333)
    tc = target - centroid
    tRx = nx[1]*tc[1] + nx[2]*tc[2] + nx[3]*tc[3]          # transpose(R)*(target-centroid)
    tRy = ny[1]*tc[1] + ny[2]*tc[2] + ny[3]*tc[3]
    tRz = nz[1]*tc[1] + nz[2]*tc[2] + nz[3]*tc[3]
    u = zero(SVector{3,T})
    g = zero(SMatrix{3,3,T,9})
    # panel-frame vertex coordinates (elements_fmm.jl:722-741)
    w1 = v1 - centroid
    w2 = v2 - centroid
    w3 = v3 - centroid
    vx1 = nx[1]*w1[1] + nx[2]*w1[2] + nx[3]*w1[3]; vy1 = ny[1]*w1[1] + ny[2]*w1[2] + ny[3]*w1[3]
    vx2 = nx[1]*w2[1] + nx[2]*w2[2] + nx[3]*w2[3]; vy2 = ny[1]*w2[1] + ny[2]*w2[2] + ny[3]*w2[3]
    vx3 = nx[1]*w3[1] + nx[2]*w3[2] + nx[3]*w3[3]; vy3 = ny[1]*w3[1] + ny[2]*w3[2] + ny[3]*w3[3]
    for i in 1:3
        vxa, vya, wa = i == 1 ? (vx1, vy1, w1) : (i == 2 ? (vx2, vy2, w2) : (vx3, vy3, w3))
        vxb, vyb, wb = i == 1 ? (vx2, vy2, w2) : (i == 2 ? (vx3, vy3, w3) : (vx1, vy1, w1))
        if DOUBLET
            # reg_term from the side's minimum distance (elements_fmm.jl:743-746)
            m_dist = _rect_minimum_distance(wa, wb, tc)
            reg_term = _rect_regularize(m_dist, kerneloffset)
            ue, ge = _rect_edge_doublet(tRx, tRy, tRz, vxa, vya, vxb, vyb, reg_term, Val(GRAD))
        else
            ue, ge = _rect_edge_source(tRx, tRy, tRz, vxa, vya, vxb, vyb, Val(GRAD))
        end
        u += ue
        GRAD && (g += ge)
    end
    # rotate back with the -1/(4pi) factor (elements_fmm.jl:796-803)
    c = -T(ONE_OVER_4π)
    R = SMatrix{3,3,T,9}(nx[1], nx[2], nx[3], ny[1], ny[2], ny[3], nz[1], nz[2], nz[3])
    u_out = c * (R * u)
    g_out = GRAD ? c * (R * g * transpose(R)) : zero(SMatrix{3,3,T,9})
    return u_out, g_out
end

# Bound-vortex filament velocity, per-family regularization. Vatistas (REG=1)
# is the HEAD kernel (elements_fmm.jl:867-898 @ 75b45c7); compact (REG=2) and
# Gaussian (REG=3) are the working-tree families (branch fastmultipole,
# 2026-08-20): _bound_vortex_velocity ::Val{F} at working-tree
# elements_fmm.jl:945-987, finite_core=true path (elements_fmm.jl:966-985).
@inline function _rect_bound_vortex_velocity(r1::SVector{3,T}, r2::SVector{3,T},
        core_size::T, ::Val{REG}) where {T,REG}
    nr1 = sqrt(r1[1]*r1[1] + r1[2]*r1[2] + r1[3]*r1[3])
    nr2 = sqrt(r2[1]*r2[1] + r2[2]*r2[2] + r2[3]*r2[3])
    if nr1 < 5*eps(T) || nr2 < 5*eps(T)                    # elements_fmm.jl:873
        return zero(SVector{3,T})
    end
    num = SVector{3,T}(r1[2]*r2[3] - r1[3]*r2[2],
                       r1[3]*r2[1] - r1[1]*r2[3],
                       r1[1]*r2[2] - r1[2]*r2[1])
    r0 = r1 - r2
    dotrixrj = num[1]*num[1] + num[2]*num[2] + num[3]*num[3]   # A = |r1×r2|²
    r0sqr = r0[1]*r0[1] + r0[2]*r0[2] + r0[3]*r0[3]            # B = |r0|²
    rh = r1/nr1 - r2/nr2
    rijdothat = r0[1]*rh[1] + r0[2]*rh[2] + r0[3]*rh[3]
    if REG == 2
        # compact support (working-tree elements_fmm.jl:971-976):
        # 1/h² → 1/(h² + δ(h)), δ = (h-rc)² inside support, 0 beyond;
        # D = A + δB
        h = sqrt(dotrixrj / r0sqr)
        D = h < core_size ?
            dotrixrj + (h - core_size)*(h - core_size) * r0sqr : dotrixrj
        return num * rijdothat / D / (4*T(pi))
    elseif REG == 3
        # Gaussian / Lamb-Oseen (working-tree elements_fmm.jl:977-982):
        # u = c*q*g(h)/(4π A), g = 1 - exp(-h²/2rc²); evaluated as
        # g/A = (g/x²)/(B rc²), x² = (h/rc)², exact h → 0 limit
        x2 = dotrixrj / (r0sqr * core_size * core_size)
        gscaled = x2 < T(1e-12) ? T(0.5) : T(-expm1(-x2/2) / x2)
        return num * rijdothat * gscaled / (r0sqr * core_size * core_size) / (4*T(pi))
    else
        # Vatistas n=2 (HEAD elements_fmm.jl:884 / working tree :968-970)
        rc4 = core_size*core_size*core_size*core_size
        return num * rijdothat / sqrt(dotrixrj*dotrixrj + rc4*r0sqr*r0sqr) / (4*T(pi))
    end
end

# Bound-vortex filament gradient, per-family regularization. Vatistas (REG=1)
# is the HEAD kernel (elements_fmm.jl:900-939 @ 75b45c7); compact (REG=2) and
# Gaussian (REG=3) transcribe the working-tree _bound_vortex_gradient
# ::Val{F} per-family D and ∇D = κ ∇A blocks (working-tree
# elements_fmm.jl:1010-1043), ∇A = 2 s×c.
@inline function _rect_bound_vortex_gradient(r1::SVector{3,T}, r2::SVector{3,T},
        core_size::T, ::Val{REG}) where {T,REG}
    nr1 = sqrt(r1[1]*r1[1] + r1[2]*r1[2] + r1[3]*r1[3])
    nr2 = sqrt(r2[1]*r2[1] + r2[2]*r2[2] + r2[3]*r2[3])
    if nr1 < 5*eps(T) || nr2 < 5*eps(T)                    # elements_fmm.jl:905
        return zero(SMatrix{3,3,T,9})
    end
    c = SVector{3,T}(r1[2]*r2[3] - r1[3]*r2[2],
                     r1[3]*r2[1] - r1[1]*r2[3],
                     r1[1]*r2[2] - r1[2]*r2[1])
    s = r1 - r2
    A = c[1]*c[1] + c[2]*c[2] + c[3]*c[3]
    B = s[1]*s[1] + s[2]*s[2] + s[3]*s[3]
    rh = r1/nr1 - r2/nr2
    q = s[1]*rh[1] + s[2]*rh[2] + s[3]*rh[3]
    if REG == 2
        # compact support (working-tree elements_fmm.jl:1017-1029)
        B == zero(B) && return zero(SMatrix{3,3,T,9})
        h = sqrt(A / B)
        if h < core_size
            D = A + (h - core_size)*(h - core_size) * B
            # κ = 2 - rc/h; h → 0 clamp keeps κ finite where ∇A → 0 anyway
            kappa = 2 - core_size / max(h, eps(T)*core_size)
        else
            D = A
            kappa = one(T)
        end
        D == zero(D) && return zero(SMatrix{3,3,T,9})
    elseif REG == 3
        # Gaussian (working-tree elements_fmm.jl:1030-1042)
        B == zero(B) && return zero(SMatrix{3,3,T,9})
        x2 = A / (B * core_size * core_size)               # (h/rc)²
        if x2 < T(1e-12)
            # series limits: D → 2 B rc², κ → 1/2
            D = 2 * B * core_size * core_size
            kappa = T(0.5)
        else
            gg = T(-expm1(-x2/2))
            D = A / gg
            kappa = (1 - x2 * exp(-x2/2) / (2*gg)) / gg
        end
    else
        # Vatistas n=2 (HEAD elements_fmm.jl:916-919 / working tree :1012-1016)
        rc4 = core_size*core_size*core_size*core_size
        D = sqrt(A*A + rc4*B*B)
        D == zero(D) && return zero(SMatrix{3,3,T,9})
        kappa = A / D
    end
    sxc = SVector{3,T}(s[2]*c[3] - s[3]*c[2],
                       s[3]*c[1] - s[1]*c[3],
                       s[1]*c[2] - s[2]*c[1])
    dD_coeff = kappa * (2 * sxc)                           # elements_fmm.jl:1043
    r1hat = r1 / nr1
    r2hat = r2 / nr2
    # dq_coeff = -((I - r1hat r1hat^T) s)/nr1 + ((I - r2hat r2hat^T) s)/nr2
    # (elements_fmm.jl:928-929)
    r1hs = r1hat[1]*s[1] + r1hat[2]*s[2] + r1hat[3]*s[3]
    r2hs = r2hat[1]*s[1] + r2hat[2]*s[2] + r2hat[3]*s[3]
    dq_coeff = -(s - r1hat*r1hs)/nr1 + (s - r2hat*r2hs)/nr2
    dc_dx = SMatrix{3,3,T,9}(zero(T), -s[3], s[2],
                             s[3], zero(T), -s[1],
                             -s[2], s[1], zero(T))         # elements_fmm.jl:931-935
    df_coeff = dq_coeff/D - q*dD_coeff/(D*D)
    return T(ONE_OVER_4π) * (dc_dx * (q/D) + c * transpose(df_coeff))  # elements_fmm.jl:938
end

# vortex-ring panel velocity/gradient (_induced for VortexRing,
# elements_fmm.jl:811-861; VS/GS blocks). Per-unit-Gamma result.
@inline function _rect_ring(target::SVector{3,T}, v1::SVector{3,T}, v2::SVector{3,T},
        v3::SVector{3,T}, v4::SVector{3,T}, nv::Int, core_size::T,
        ::Val{GRAD}, ::Val{REG}) where {T,GRAD,REG}
    u = zero(SVector{3,T})
    g = zero(SMatrix{3,3,T,9})
    for i in 1:nv
        va = i == 1 ? v1 : (i == 2 ? v2 : (i == 3 ? v3 : v4))
        ip1 = i < nv ? i + 1 : 1
        vb = ip1 == 1 ? v1 : (ip1 == 2 ? v2 : (ip1 == 3 ? v3 : v4))
        r1 = va - target                                   # elements_fmm.jl:838-839
        r2 = vb - target
        u += _rect_bound_vortex_velocity(r1, r2, core_size, Val(REG))
        GRAD && (g += _rect_bound_vortex_gradient(r1, r2, core_size, Val(REG)))
    end
    return u, g
end

# self-pair detection (elements_fmm.jl:201-216): relative tolerance vs sqrt(A)
@inline function _rect_is_self_pair(target::SVector{3,T}, control_point::SVector{3,T},
        v1::SVector{3,T}, v2::SVector{3,T}, v3::SVector{3,T}) where T
    e1 = v2 - v1
    e2 = v3 - v1
    nxc = e1[2]*e2[3] - e1[3]*e2[2]
    nyc = e1[3]*e2[1] - e1[1]*e2[3]
    nzc = e1[1]*e2[2] - e1[2]*e2[1]
    area = T(0.5) * sqrt(nxc*nxc + nyc*nyc + nzc*nzc)
    d = target - control_point
    return (d[1]*d[1] + d[2]*d[2] + d[3]*d[3]) < T(1e-12)*T(1e-12)*area
end

@inline function _rect_panel_normal(v1::SVector{3,T}, v2::SVector{3,T}, v3::SVector{3,T}) where T
    e1 = v2 - v1
    e2 = v3 - v1
    nxc = e1[2]*e2[3] - e1[3]*e2[2]
    nyc = e1[3]*e2[1] - e1[1]*e2[3]
    nzc = e1[1]*e2[2] - e1[2]*e2[1]
    inv_n = one(T) / sqrt(nxc*nxc + nyc*nyc + nzc*nzc)
    return SVector{3,T}(nxc*inv_n, nyc*inv_n, nzc*inv_n)   # elements_fmm.jl:218-227
end

# full per-source panel influence, dispatching on the runtime tag. Applies the
# velocity part of FLOWPanel's `_self_limit` (elements_fmm.jl:156-198) at self
# pairs and zeroes the gradient there, matching `induced` (elements_fmm.jl:250-253).
@inline function _rect_panel_pair(::RectangularPanelInfluence, target::SVector{3,T},
        tag::Int, nv::Int, v1::SVector{3,T}, v2::SVector{3,T}, v3::SVector{3,T},
        v4::SVector{3,T}, s1::T, s2::T, kerneloffset::T, ::Val{GRAD},
        ::Val{REG}=Val(1)) where {T,GRAD,REG}
    u = zero(SVector{3,T})
    g = zero(SMatrix{3,3,T,9})
    if tag == 1 || tag == 4 || tag == 5      # ConstantSource part
        us, gs = _rect_tri_source_doublet(target, v1, v2, v3, kerneloffset,
            Val(false), Val(GRAD))
        u += s1 * us
        GRAD && (g += s1 * gs)
    end
    if tag == 2 || tag == 5                  # ConstantDoublet part
        mu = tag == 2 ? s1 : s2
        ud, gd = _rect_tri_source_doublet(target, v1, v2, v3, kerneloffset,
            Val(true), Val(GRAD))
        u += mu * ud
        GRAD && (g += mu * gd)
    end
    if tag == 3 || tag == 4                  # VortexRing part
        gam = tag == 3 ? s1 : s2
        nvr = tag == 4 ? 3 : nv              # combined tag is a tri panel
        ur, gr = _rect_ring(target, v1, v2, v3, v4, nvr, kerneloffset, Val(GRAD), Val(REG))
        u += gam * ur
        GRAD && (g += gam * gr)
    end
    # self-pair short-circuit (elements_fmm.jl:250-253 / 299-304 + 156-198):
    # velocity gets the fixed exterior surface limit for the source component;
    # doublet/ring velocity is already the clean PV; gradient is zeroed.
    control_point = (v1 + v2 + v3) * T(0.3333333333333333)
    if _rect_is_self_pair(target, control_point, v1, v2, v3)
        n_gt = _rect_panel_normal(v1, v2, v3)
        if tag == 1
            # u_self = u + (sigma/2 - u.n) n (elements_fmm.jl:156-163)
            un = u[1]*n_gt[1] + u[2]*n_gt[2] + u[3]*n_gt[3]
            u = u + (s1*T(0.5) - un) * n_gt
        elseif tag == 4 || tag == 5
            # u_self = u + sigma/2 n (elements_fmm.jl:179-198)
            u = u + s1*T(0.5) * n_gt
        end
        # tags 2, 3: velocity is continuous at the centroid (u unchanged)
        g = zero(SMatrix{3,3,T,9})           # gradient self-limit zeroed
    end
    return u, g
end

#------- host implementation (threaded) -------#

function _rect_check_args(out, targets, kernel, sources, gradient::Bool)
    size(targets, 1) >= 3 || throw(ArgumentError(
        "targets must have at least 3 rows (position); got $(size(targets, 1))"))
    size(sources, 1) >= rect_source_rows(kernel) || throw(ArgumentError(
        "$(typeof(kernel)) sources require $(rect_source_rows(kernel)) rows; " *
        "got $(size(sources, 1))"))
    size(out, 2) == size(targets, 2) || throw(ArgumentError(
        "out has $(size(out, 2)) columns but targets has $(size(targets, 2))"))
    size(out, 1) >= rect_output_rows(gradient) || throw(ArgumentError(
        "out must have at least $(rect_output_rows(gradient)) rows for " *
        "gradient=$gradient; got $(size(out, 1))"))
    return nothing
end

"""
    direct_rectangular!(out, targets, kernel, sources; gradient=false)

Brute-force rectangular direct evaluation: every source column of `sources`
influences every target column of `targets`, accumulating (+=) velocity into
`out[1:3, :]` and, when `gradient=true`, the velocity gradient into
`out[4:12, :]` (order `out[3 + (j-1)*3 + i] = du_i/dx_j`). Host method is
threaded over targets; CUDA methods (CuMatrix arguments) are installed by
`load_cuda_radix_lifecycle!()`. See [`RectangularGaussianErfVortex`](@ref) and
[`RectangularPanelInfluence`](@ref) for the source row layouts.
"""
function direct_rectangular!(out::AbstractMatrix{T}, targets::AbstractMatrix{T},
        kernel::RectangularGaussianErfVortex, sources::AbstractMatrix{T};
        gradient::Bool=false) where T
    _rect_check_args(out, targets, kernel, sources, gradient)
    n_targets = size(targets, 2)
    n_sources = size(sources, 2)
    if gradient
        _rect_points_host!(out, targets, sources, n_targets, n_sources, Val(true))
    else
        _rect_points_host!(out, targets, sources, n_targets, n_sources, Val(false))
    end
    return out
end

function _rect_points_host!(out, targets, sources, n_targets, n_sources, ::Val{GRAD}) where GRAD
    T = eltype(out)
    Threads.@threads :static for i in 1:n_targets
        @inbounds begin
            tx = targets[1, i]; ty = targets[2, i]; tz = targets[3, i]
            u1 = u2 = u3 = zero(T)
            j1 = j2 = j3 = j4 = j5 = j6 = j7 = j8 = j9 = zero(T)
            for q in 1:n_sources
                Ux, Uy, Uz, a1, a2, a3, a4, a5, a6, a7, a8, a9 =
                    _rect_point_pair(RectangularGaussianErfVortex(), tx, ty, tz,
                        sources[1, q], sources[2, q], sources[3, q],
                        sources[4, q], sources[5, q], sources[6, q],
                        sources[7, q], Val(GRAD))
                u1 += Ux; u2 += Uy; u3 += Uz
                if GRAD
                    j1 += a1; j2 += a2; j3 += a3; j4 += a4; j5 += a5
                    j6 += a6; j7 += a7; j8 += a8; j9 += a9
                end
            end
            out[1, i] += u1; out[2, i] += u2; out[3, i] += u3
            if GRAD
                out[4, i] += j1; out[5, i] += j2; out[6, i] += j3
                out[7, i] += j4; out[8, i] += j5; out[9, i] += j6
                out[10, i] += j7; out[11, i] += j8; out[12, i] += j9
            end
        end
    end
    return nothing
end

function direct_rectangular!(out::AbstractMatrix{T}, targets::AbstractMatrix{T},
        kernel::RectangularPanelInfluence, sources::AbstractMatrix{T};
        gradient::Bool=false) where T
    _rect_check_args(out, targets, kernel, sources, gradient)
    n_targets = size(targets, 2)
    n_sources = size(sources, 2)
    regv = _rect_reg_val(kernel.filament_reg)
    if gradient
        _rect_panels_host!(out, targets, sources, n_targets, n_sources, Val(true), regv)
    else
        _rect_panels_host!(out, targets, sources, n_targets, n_sources, Val(false), regv)
    end
    return out
end

@inline function _rect_load_panel_source(sources, q, ::Type{T}) where T
    @inbounds begin
        tag = Int(sources[1, q])
        nv = Int(sources[2, q])
        v1 = SVector{3,T}(sources[3, q], sources[4, q], sources[5, q])
        v2 = SVector{3,T}(sources[6, q], sources[7, q], sources[8, q])
        v3 = SVector{3,T}(sources[9, q], sources[10, q], sources[11, q])
        v4 = SVector{3,T}(sources[12, q], sources[13, q], sources[14, q])
        s1 = sources[15, q]
        s2 = sources[16, q]
        koff = sources[17, q]
    end
    return tag, nv, v1, v2, v3, v4, s1, s2, koff
end

function _rect_panels_host!(out, targets, sources, n_targets, n_sources,
        ::Val{GRAD}, ::Val{REG}=Val(1)) where {GRAD,REG}
    T = eltype(out)
    Threads.@threads :static for i in 1:n_targets
        @inbounds begin
            target = SVector{3,T}(targets[1, i], targets[2, i], targets[3, i])
            u = zero(SVector{3,T})
            g = zero(SMatrix{3,3,T,9})
            for q in 1:n_sources
                tag, nv, v1, v2, v3, v4, s1, s2, koff =
                    _rect_load_panel_source(sources, q, T)
                uq, gq = _rect_panel_pair(RectangularPanelInfluence(), target,
                    tag, nv, v1, v2, v3, v4, s1, s2, koff, Val(GRAD), Val(REG))
                u += uq
                GRAD && (g += gq)
            end
            out[1, i] += u[1]; out[2, i] += u[2]; out[3, i] += u[3]
            if GRAD
                for j in 1:3, k in 1:3
                    out[3 + (j-1)*3 + k, i] += g[k, j]
                end
            end
        end
    end
    return nothing
end
