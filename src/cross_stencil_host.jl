#------- CROSS-PASS STENCIL TABLES (task 052d Stage A) -------#
# Host-side construction for the two-occupancy cross pass (panels → particles)
# on the shared radix grid. Builds the per-level phase-mask tables the device
# route producers consume: the uniform-q push set split into far (M2L) and
# guard-demoted (direct) offsets per level, plus the unmasked near shell.
#
# Guard demotion (052d P3.2): a route at level L with integer offset `o` is
# demoted to direct evaluation when the minimum gap between the two cell boxes,
#
# $$ \mathrm{gap}(o, w_L) = w_L \,\bigl\| \max(|o| - 1, 0) \bigr\|_2, $$
#
# is strictly below `R_guard` (strict `<`: demotion stays OFF when the gap sits
# exactly at `R_guard` — D2 ruling 2026-08-28). Splitting the push set by a
# level-dependent predicate preserves exact-once coverage: far and demoted
# routes partition the same emission set, they only land in different lists.

"""
    CrossStencilTables(q, ell_x, h0, R_guard)

Level-masked stencil tables for the cross pass. `q` is the uniform near radius
(`|o|^2 <= q` — MAC ties resolve to NEAR, D2 ruling), `ell_x` the cross leaf
level, `h0` the shared root half-width, `R_guard` the physical demotion radius.

Fields `level_class_far` / `level_class_demoted` are `(8, K, ell_x + 1)` phase
masks in the `level_class_of` convention (entry `k` when offset `k` is admitted
for that source phase and level, else 0); their nonzero supports partition the
base `tables.class_of` support at every level. `near_class` admits the near
shell for every phase and level (the near set is never phase-masked).
"""
struct CrossStencilTables
    q::Int
    ell_x::Int
    h0::Float64
    R_guard::Float64
    tables::RigidHierarchicalTables
    level_class_far::Array{Int32,3}
    level_class_demoted::Array{Int32,3}
    near_class::Array{Int32,3}
end

"Minimum distance between two level-`L` cell boxes at integer offset `o` (cell
width `w`): per-axis gap `w * max(|o_a| - 1, 0)`, Euclidean-combined."
@inline function _cross_box_gap(o, w::Float64)
    gx = w * max(abs(Int(o[1])) - 1, 0)
    gy = w * max(abs(Int(o[2])) - 1, 0)
    gz = w * max(abs(Int(o[3])) - 1, 0)
    return sqrt(gx * gx + gy * gy + gz * gz)
end

"Demotion predicate: strict `<` so `R_guard` exactly on the box gap keeps the
route in M2L (D2)."
@inline _cross_demoted(o, w::Float64, R_guard::Float64) =
    _cross_box_gap(o, w) < R_guard

function CrossStencilTables(q::Integer, ell_x::Integer, h0::Real, R_guard::Real)
    2 <= ell_x || throw(ArgumentError("cross pass requires ell_x >= 2; got $ell_x"))
    tables = RigidHierarchicalTables(Int(q))
    K = length(tables.push_offsets)
    Kn = length(tables.near_offsets)
    level_class_far = zeros(Int32, 8, K, ell_x + 1)
    level_class_demoted = zeros(Int32, 8, K, ell_x + 1)
    near_class = zeros(Int32, 8, Kn, ell_x + 1)
    for L in 0:ell_x
        w_L = 2 * Float64(h0) / (1 << L)
        for k in 1:K
            demoted = _cross_demoted(tables.push_offsets[k], w_L, Float64(R_guard))
            for phase in 1:8
                tables.class_of[phase, k] == 0 && continue
                if demoted
                    level_class_demoted[phase, k, L + 1] = Int32(k)
                else
                    level_class_far[phase, k, L + 1] = Int32(k)
                end
            end
        end
        for k in 1:Kn, phase in 1:8
            near_class[phase, k, L + 1] = Int32(k)
        end
    end
    return CrossStencilTables(Int(q), Int(ell_x), Float64(h0), Float64(R_guard),
        tables, level_class_far, level_class_demoted, near_class)
end

#------- Stage B: tag → B2M arm mapping (device kernel + CPU tests) -------#
#
# Per-tag far-field equivalences, mirroring the host body_to_multipole!
# shims: 1 → Source(s1); 2 → Dipole(s1); 3 → Dipole(s1) (a closed vortex ring
# is exactly its dipole-panel equivalent — the pure-VortexRing overload,
# FLOWPanel_liftingbody.jl:804-910); 4/5 → Source(s1) + Dipole(s2)
# (Panel{SourceDipole}). Callers must reject nv < 3 first (tag-3 nv=2 is an
# OPEN filament with no dipole equivalent).

"tag → (do_source, do_dipole, s_source, s_dipole) for the cross-pass B2M."
@inline function _cross_b2m_arms(tag::Int, s1, s2)
    do_source = tag == 1 || tag == 4 || tag == 5
    do_dipole = tag != 1
    s_dipole = (tag == 2 || tag == 3) ? s1 : s2
    return do_source, do_dipole, s1, s_dipole
end

#------- Stage B: dense per-(level, octant) M2M operators -------#
#
# On the radix grid a child→parent M2M edge has only 8 distinct geometries per
# level (the child's octant), so the upward operators are occupancy-independent
# and can be materialized ONCE at construction — this is what keeps the cross
# pass device-native per step (D1) without rebuilding rotation groups on host.
# The dense matrices are built by probing the production host M2M
# (`multipole_to_multipole!`) with unit coefficient vectors, so they are
# convention-exact against the host oracle by construction.

_cross_dummy_branch(center) =
    Branch(1:1, 0, 1:0, 0, 1, center, 0.0, SVector(0.0, 0.0, 0.0))

"""
    cross_m2m_operators(P, h0, ell_x)

Dense phi-channel M2M matrices `ops[row, col, octant, L_child]` with
`row/col = 2*(harmonic_index - 1) + (1 re | 2 im)`, `octant = 1 + (cx & 1) +
2*(cy & 1) + 4*(cz & 1)` of the child cell, `L_child in 1:ell_x`. The child
center sits at `(u - 1/2) * w_child` relative to its parent center. Panel
sources are phi-only (`Panel{SourceDipole}`), so the chi channel is not
materialized (Lamb-Helmholtz off).
"""
function cross_m2m_operators(P::Integer, h0::Real, ell_x::Integer)
    H = ((P + 1) * (P + 2)) >> 1
    D = 2 * H
    ops = Array{Float64,4}(undef, D, D, 8, ell_x)
    update_Hs_π2!(Hs_π2, P)
    update_ζs_mag!(ζs_mag, P)
    w1 = initialize_expansion(P)
    w2 = initialize_expansion(P)
    Ts = zeros(length_Ts(P))
    eimϕs = zeros(2, P + 1)
    ce = initialize_expansion(P)
    pe = initialize_expansion(P)
    lhv = Val(false)
    pb = _cross_dummy_branch(SVector(0.0, 0.0, 0.0))
    for Lc in 1:ell_x
        wc = 2 * Float64(h0) / (1 << Lc)
        for phase in 0:7
            u = SVector(phase & 1, (phase >> 1) & 1, (phase >> 2) & 1)
            cb = _cross_dummy_branch(SVector{3,Float64}((u .- 0.5) .* wc))
            for col in 1:D
                i = (col + 1) >> 1
                reim = 2 - (col & 1)
                fill!(ce, 0.0)
                ce[reim, 1, i] = 1.0
                fill!(pe, 0.0)
                multipole_to_multipole!(pe, pb, ce, cb, w1, w2, Ts, eimϕs,
                    ζs_mag, Hs_π2, Int(P), lhv)
                for i2 in 1:H
                    ops[2 * (i2 - 1) + 1, col, phase + 1, Lc] = pe[1, 1, i2]
                    ops[2 * (i2 - 1) + 2, col, phase + 1, Lc] = pe[2, 1, i2]
                end
            end
        end
    end
    return ops
end

#------- Stage C: dense per-offset-class M2L operators (scale-covariant) -------#
#
# A far route at level L with push offset `o` is the M2L translation
# `t_L = o * w_L` (target center − source center; the producers emit
# `target = source + o`). The Laplace M2L is exactly scale-covariant: the
# (n, n') degree block of the operator scales as `1 / s^(n + n' + 1)` under
# `t -> s t` (rotations depend only on the direction of `o`, the z-translate
# on inverse powers of |t|). So ONE reference level (`_CROSS_M2L_L_REF = 2`,
# the coarsest route level) is probed per active offset class, and deeper
# levels apply the separable factor
#
# $$ f_L(n, n') = 2^{(L - 2)(n + n' + 1)}
#              = \bigl[2^{(L-2) n'}\, 2^{L-2}\bigr] \cdot 2^{(L-2) n}, $$
#
# which the device kernel composes from a tiny `scale2[row, L]` table. This
# cuts table memory and probe count by (ell_x - 1)× vs per-level tables
# (q = 12 has K = 1740 push offsets → per-level would be ~175 MB at P = 6).
#
# Like the M2M operators, entries are probed from the PRODUCTION host
# `multipole_to_local!` with unit coefficient vectors in the classic-phi row
# basis `row = 2*(harmonic_index - 1) + (1 re | 2 im)` — the same convention
# Stage B emits — so the tables are convention-exact by construction.

const _CROSS_M2L_L_REF = 2

"""
    cross_m2l_class_slots(ct)

Compact slot numbering over offset classes that appear in the FAR mask at any
route level (2:ell_x) for any phase: returns `(class_slot, n_slots)` with
`class_slot[k] == 0` for classes that never route far.
"""
function cross_m2l_class_slots(ct::CrossStencilTables)
    K = length(ct.tables.push_offsets)
    class_slot = zeros(Int32, K)
    n_slots = 0
    for k in 1:K
        active = any(ct.level_class_far[phase, k, L + 1] != 0
            for L in 2:ct.ell_x, phase in 1:8)
        active && (class_slot[k] = Int32(n_slots += 1))
    end
    return class_slot, n_slots
end

"""
    cross_m2l_operators(P, h0, ct)

Dense phi-channel M2L matrices `ops[row, col, slot]` at the REFERENCE level
`_CROSS_M2L_L_REF`, slot-compacted over `cross_m2l_class_slots(ct)` (returns
`(ops, class_slot)`). Deeper levels are exact diagonal rescalings — see the
Stage-C header note. Probed from the production `multipole_to_local!`
(error_method = nothing, Lamb-Helmholtz off).
"""
function cross_m2l_operators(P::Integer, h0::Real, ct::CrossStencilTables)
    H = ((P + 1) * (P + 2)) >> 1
    D = 2 * H
    class_slot, n_slots = cross_m2l_class_slots(ct)
    ops = Array{Float64,3}(undef, D, D, n_slots)
    update_Hs_π2!(Hs_π2, P)
    update_ζs_mag!(ζs_mag, P)
    update_ηs_mag!(ηs_mag, P)
    update_M̃!(M̃, P)
    update_L̃!(L̃, P)
    w1 = initialize_expansion(P)
    w2 = initialize_expansion(P)
    w3 = initialize_expansion(P)
    Ts = zeros(length_Ts(P))
    eimϕs = zeros(2, P + 1)
    se = initialize_expansion(P)
    te = initialize_expansion(P)
    lhv = Val(false)
    sb = _cross_dummy_branch(SVector(0.0, 0.0, 0.0))
    w_ref = 2 * Float64(h0) / (1 << _CROSS_M2L_L_REF)
    for k in 1:length(class_slot)
        slot = class_slot[k]
        slot == 0 && continue
        o = ct.tables.push_offsets[k]
        tb = _cross_dummy_branch(SVector{3,Float64}(o) * w_ref)
        for col in 1:D
            i = (col + 1) >> 1
            reim = 2 - (col & 1)
            fill!(se, 0.0)
            se[reim, 1, i] = 1.0
            fill!(te, 0.0)
            multipole_to_local!(te, tb, se, sb, w1, w2, w3, Ts, eimϕs,
                ζs_mag, ηs_mag, Hs_π2, M̃, L̃, Int(P), lhv, nothing)
            for i2 in 1:H
                ops[2 * (i2 - 1) + 1, col, slot] = te[1, 1, i2]
                ops[2 * (i2 - 1) + 2, col, slot] = te[2, 1, i2]
            end
        end
    end
    return ops, class_slot
end

"Degree of each classic-phi coefficient row (`row = 2*(harmonic - 1) + re/im`)."
function cross_row_degrees(P::Integer)
    H = ((P + 1) * (P + 2)) >> 1
    row_n = Vector{Int32}(undef, 2 * H)
    i = 1
    for n in 0:P, m in 0:n
        row_n[2 * (i - 1) + 1] = Int32(n)
        row_n[2 * (i - 1) + 2] = Int32(n)
        i += 1
    end
    return row_n
end

"""
    cross_m2l_level_scales(P, ell_x)

`scale2[row, L + 1] = 2.0^((L - _CROSS_M2L_L_REF) * degree(row))` for the
separable in-kernel level rescaling, plus `pow2lvl[L + 1] =
2.0^(L - _CROSS_M2L_L_REF)`: the full factor on output row `r` of a level-`L`
route is `scale2[r, L+1] * pow2lvl[L+1] * scale2[c, L+1]` over input column c.
"""
function cross_m2l_level_scales(P::Integer, ell_x::Integer)
    row_n = cross_row_degrees(P)
    D = length(row_n)
    scale2 = Array{Float64,2}(undef, D, ell_x + 1)
    pow2lvl = Vector{Float64}(undef, ell_x + 1)
    for L in 0:ell_x
        pow2lvl[L + 1] = exp2(L - _CROSS_M2L_L_REF)
        for r in 1:D
            scale2[r, L + 1] = exp2((L - _CROSS_M2L_L_REF) * Int(row_n[r]))
        end
    end
    return scale2, pow2lvl
end

#------- Stage D: dense per-(level, octant) L2L operators -------#
#
# Mirror of the Stage-B M2M octant trick on the downward side: a parent→child
# L2L edge has 8 distinct geometries per child level, so the operators are
# occupancy-independent and probed ONCE from the production host
# `local_to_local!` with unit coefficient vectors (classic-phi row basis,
# convention-exact by construction). The child center sits at
# `(u - 1/2) * w_child` relative to its parent center.

"""
    cross_l2l_operators(P, h0, ell_x)

Dense phi-channel L2L matrices `ops[row, col, octant, L_child]` with the same
row/octant conventions as `cross_m2m_operators`, `L_child in 1:ell_x`
(the cross downward pass uses child levels `3:ell_x`; locals start at the
coarsest route level 2).
"""
function cross_l2l_operators(P::Integer, h0::Real, ell_x::Integer)
    H = ((P + 1) * (P + 2)) >> 1
    D = 2 * H
    ops = Array{Float64,4}(undef, D, D, 8, ell_x)
    update_Hs_π2!(Hs_π2, P)
    update_ηs_mag!(ηs_mag, P)
    w1 = initialize_expansion(P)
    w2 = initialize_expansion(P)
    Ts = zeros(length_Ts(P))
    eimϕs = zeros(2, P + 1)
    se = initialize_expansion(P)
    te = initialize_expansion(P)
    lhv = Val(false)
    pb = _cross_dummy_branch(SVector(0.0, 0.0, 0.0))
    for Lc in 1:ell_x
        wc = 2 * Float64(h0) / (1 << Lc)
        for phase in 0:7
            u = SVector(phase & 1, (phase >> 1) & 1, (phase >> 2) & 1)
            cb = _cross_dummy_branch(SVector{3,Float64}((u .- 0.5) .* wc))
            for col in 1:D
                i = (col + 1) >> 1
                reim = 2 - (col & 1)
                fill!(se, 0.0)
                se[reim, 1, i] = 1.0
                fill!(te, 0.0)
                local_to_local!(te, cb, se, pb, w1, w2, Ts, eimϕs,
                    ηs_mag, Hs_π2, Int(P), lhv)
                for i2 in 1:H
                    ops[2 * (i2 - 1) + 1, col, phase + 1, Lc] = te[1, 1, i2]
                    ops[2 * (i2 - 1) + 2, col, phase + 1, Lc] = te[2, 1, i2]
                end
            end
        end
    end
    return ops
end

"Number of demoted push offsets per level (union over phases), for stage logs."
function cross_demotion_census(ct::CrossStencilTables)
    census = zeros(Int, ct.ell_x + 1)
    for L in 0:ct.ell_x, k in 1:length(ct.tables.push_offsets)
        any(ct.level_class_demoted[phase, k, L + 1] != 0 for phase in 1:8) &&
            (census[L + 1] += 1)
    end
    return census
end
