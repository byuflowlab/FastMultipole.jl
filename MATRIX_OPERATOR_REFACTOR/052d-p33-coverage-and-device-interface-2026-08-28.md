# 052d P3.3 — cross-pass exact-once note + device producer interface memo (2026-08-28)

Companion to `052d-plan-2026-08-26.md` ("Cross-pass design review", P3.3).
Scope per amendments A2/A3 and the device-native-producers ruling: (1) the
uniform-q two-occupancy exact-once argument; (2) the device producer
interfaces the cross pass consumes + host-as-oracle parity checks; (3) the
`ell_x > ell` data path. Status markers `[P3.1]` are filled from the sweep.

## 1. Exact-once coverage, uniform-q two-occupancy (amendment A3)

Setting: one shared radix grid `(x_min, h0)`; panel cells and particle cells
are two sparse subsets of the same level-`L` lattices, `L = 0..ell_x`.
Uniform near radius `q >= 3`; first M2L level 2. For a panel (source) cell
`A` and particle (target) cell `B` at level `L`, the integer offset is
`o_L = coord_L(B) - coord_L(A)`; "near at `L`" means `|o_L|^2 <= q`. The
list rule (identical to the shipped single-occupancy rule,
`interaction_list_batched.jl:158-206`, evaluated per body PAIR through the
cells containing it):

- **near field**: leaf pairs with `|o_{ell_x}|^2 <= q`;
- **M2L at level `L`**: pairs far at `L` and near at `L-1`
  (equivalently: `o_L` lies in the source-phase push set — the table's
  membership test `_rigid_near(fld(o + phi_src, 2), q)` computes exactly the
  parent offset, since `coord_{L-1} = fld(coord_L, 2)` and
  `fld(o + phi_src, 2) = fld(coord_L(B), 2) - fld(coord_L(A), 2)`).

**Claim.** Every (panel, particle) body pair is covered exactly once by
the union of M2L cell-pair subtree products and near-field leaf pairs.

**Proof.** Fix a body pair; it determines one cell pair per level (the cells
containing the bodies), hence one offset `o_L` per level. Two observations:

1. *Root anchoring*: at level 1 every offset is near — level-1 coords lie in
   `{0,1}` so `|o_1|^2 <= 3 <= q`.
2. *Monotonicity (near levels form a prefix from the root)*: per component,
   `|fld(c + phi, 2)| <= |c|` for every integer `c` and `phi in {0,1}`
   (for `c >= 1`, `fld(c+1,2) <= (c+1)/2 <= c`; for `c <= -1`,
   `fld(c+phi,2) in {ceil(c/2)-?}` has magnitude `<= |c|`; for `c = 0` it is
   `0` or `0`). Hence `|o_{L-1}|^2 <= |o_L|^2` componentwise-summed, so
   near at `L` implies near at `L-1`.

By (1) and (2) there is a unique deepest near level
`m in {1, ..., ell_x}`: the pair is near at every `L <= m` and far at every
`L > m`. The coverage count is
`[m = ell_x]` (near field) `+ #{L in 2..ell_x : far at L, near at L-1}`
`= [m = ell_x] + [m < ell_x]` (the only transition level is `L = m+1`,
and `m+1 >= 2` always) `= 1`. ∎

Notes: the argument never references occupancy — it holds for any two body
sets on the shared grid, including the degenerate cases (both bodies in one
cell: `o = 0`, near at every level, covered once by the near field). It
requires only `q >= 3` (root anchoring) and uniformity of `q` (monotonicity
across levels is stated for one fixed `q`; a non-increasing-with-depth
schedule preserves it — same argument as task 025 — but v1 is uniform per
ruling R6). Empirical certification: P3.1 runs (i) the exact count identity
`sum |A||B| = ns * nt` over all 8.89e9 pairs, (ii) brute-force per-pair
coverage for 400 particles x all 36,752 panels, (iii) the prototype's full
coverage matrix on a 2k x 3k subset from the materialized lists — per config.
[P3.1: all 18 configs (q in {3,5,12} x ell_x in {4..9}) PASSED all three.]

## 2. Device producer interfaces (device-native ruling, 2026-08-28)

The cross pass is a self-contained device pass invoked from the
PANEL_INFLUENCE_FMM seam (ruling R3). Inputs it consumes each step:

**Already resident (no new movement):**
- Particle positions (current, device) — the self-pass keeps them current.
- The 17-row panel buffer uploaded per step for the dense near-field kernel —
  device panel B2M and the near-field wrapper both read it.
- Root box `(x_min, h0)`: shared with the self-pass grid (ruling R1);
  device asserts panel containment (kernel reduction on panel coords, one
  flag readback; on failure fall back to a union root box + rebuild the
  cross tables for that step).

**Built once at construction (host, uploaded):**
- Uniform-q push tables for the chosen `q`: per-phase offset lists
  (`RigidHierarchicalTables(q)`, `interaction_list_batched.jl:158-206`) and
  the near-offset shell.
- Per-(level, offset)-class M2L operator tables at the cross-pass `P`
  (own tables — cross `P` may differ from self-pass `P`; ruling R2, A6) via
  the existing per-class operator-table code path
  (`translate_batched_cuda.jl:3560-4667`).

**Produced on device each step:**
1. *Cross keying*: panel centroid coords + particle coords at `ell_x`
   (existing depth-parametric kernel `_cuda_radix_keys_checked_kernel!`,
   `translate_batched_cuda.jl:138-159`; new storage: two key/perm buffers).
2. *Sorts*: panel keys (36,752 — trivial); particle keys at `ell_x`. If
   `ell_x <= ell` the particle sort is FREE (self-pass ell-sorted order is
   already grouped by any coarser prefix — reuse perm + derive cell ranges
   by prefix compression). If `ell_x > ell` a genuine re-sort of particle
   keys is needed: the counting-sort scratch sized `1 << 3*ell`
   (cuda:6483-6488) does not extend, so use a non-counting sort (radix sort
   over 3*ell_x-bit keys, e.g. 8-bit-pass LSD as on host, or a
   segmented sort within ell-cells — keys within an ell-cell share the
   high prefix, so only the low `3*(ell_x - ell)` bits need sorting,
   segment by existing ell-cell ranges). [P3.1 verdict on whether
   `ell_x > ell` is needed: YES — see §3.]
3. *Occupancy*: per-level unique-prefix compression of the two sorted key
   sets (existing pattern) → two `LevelCells`-shaped arrays
   (codes + body ranges per level, panels and particles separately).
4. *Two-occupancy route lists*: the `_cuda_hier_route_flags_kernel!` /
   compact pattern with source = panel occupancy enumerating
   (cell x per-phase offsets) and target membership answered by the
   PARTICLE occupancy (dense `node_at` at `ell_x <= dense cap`, else sorted
   binary search). Direct pairs likewise from panel leaf cells x near shell.
5. *Panel B2M*: thread-per-panel over the 17-row buffer, tags 1..5
   (mixed Source/Dipole/Vortex kernels), phi+chi channels, block-per-cell
   shared-memory accumulate into the cross-pass multipole buffer
   (`FlatCoefficientBuffer` over cross nodes at cross `P`).
6. Upward M2M on cross nodes; cross-M2L over the route lists (per-class
   GEMM drivers); downward L2L; U-only L2B through the cross particle
   permutation (existing no-hessian kernel `_cuda_l2b_output_kernel!`
   variant, cuda:3372); block-sparse near field wrapping
   `_rect_panel_pair` (direct_rectangular.jl:681) driven by the pair list.

**Host-as-oracle parity checks (validation mode, not production path):**
- *List bit-compare*: host builds the same lists with the P3.1
  `CrossStencil` builder (prototypes/052d_cross_stencil/CrossStencil.jl)
  from the same positions; device lists are downloaded, canonicalized
  (sort by (level, src code, tgt code)), and compared bit-exact. The
  builder is deterministic and FP-free (integer lattice classification), so
  exact equality is the pass criterion — no tolerance. The theta-MAC
  adaptive prototype (prototypes/052d_shared_radix/) remains the SECOND,
  independent oracle for pair-partition (count identity + sampled
  coverage), inheriting the exact-<= tie convention (amendment A1 — ties
  live only in this oracle).
- *B2M coefficient parity*: host panel B2M (bodytomultipole.jl:645-869) per
  cross leaf cell vs downloaded device coefficients, relative tolerance at
  FP64/FP32-mix level (match the existing device B2M parity harness
  pattern); per-cell worst-case reported.
- *End-to-end*: sampled dense reference velocities (the P3.2 harness
  sample) vs device cross-pass output, relRMS <= the P3.2-certified
  operating-point error + margin.

## 3. The `ell_x > ell` data path (amendment A2)

[P3.1 verdict: modeled device cost falls monotonically with depth over the
swept range (near-field fraction 1.2-6.6% of dense at ell_x=4 collapses to
<0.005% by ell_x=7; M2L route counts stay tiny, 28k-260k), so every
ell_x >= 6 clears the 0.6 s gate by >= 1000x and the cost model alone
prefers the deepest depth. The binding constraint on depth is ACCURACY
(cell width at ell_x=8 is 1.77e-3 — comparable to the 1e-3 target core
size, so deep near shells sit at the erf-regularization scale) — P3.2
decides the depth. Against self-pass ell = 2 (frozen, production) or 4-5
(adequacy-gate rebuild values), any viable ell_x is STRICTLY DEEPER, so
`ell_x > ell` is CONFIRMED as the operating regime and the non-counting
sort + indirection path below is required for the port.]

- *Keying*: depth-parametric kernel, pass `ell_x` (no new kernel).
- *Particle sort*: segmented LSD radix over the low `3*(ell_x - ell)` bits
  within existing ell-cell segments (high bits are the ell-prefix, already
  ordered). Produces the cross permutation `perm_x` (Int32, ~1 MB at 242k)
  and per-cell ranges at `ell_x`.
- *Gather/scatter*: L2B and near-field kernels index particles through
  `perm_x` (gather positions, scatter U accumulation). The self-pass body
  arrays are NOT reordered — the cross pass owns `perm_x` and never
  perturbs self-pass state (A6/R1: full independence).
- *Occupancy lookup at deep levels*: dense `node_at` at `ell_x` costs
  `8^ell_x` Int32 (`ell_x = 8` → 64 MB — at the shipped
  `dense_occupancy_max_ell = 8` boundary; `ell_x = 9` → 512 MB, over
  budget): use dense `node_at` for `ell_x <= 8` and the sorted-key binary
  search fallback (`_hierarchical_node_lookup`, already implemented
  host-side; same shape on device) beyond.
- `ell_x <= ell` degenerates to the free path: reuse self-pass perm,
  prefix-compressed ranges, `node_at` slice at `level_base[ell_x + 1]`.

## 3b. Regularization guard (P3.2 finding — REQUIRED for accuracy)

P3.2 proved the production Gaussian filament kernel deviates from the
singular kernel inside an rc-cylinder around every edge's INFINITE LINE
(g(h), h = perpendicular line distance — FLOWPanel_elements_fmm.jl:997),
producing an aggregate far-field mismatch that needs a physical exclusion
radius R_guard (value set by the solved-strength mismatch curve; see the
plan doc's P3.1-P3.3 section). Design consequence for the port:

- Route classifier gains ONE extra test: demote route (A, B, level L) to
  direct when the box gap `w_L * |max(|o|-1, 0)|_2 < R_guard` — equivalently
  an integer per-level threshold `t_L` on `|o|^2`, precomputable host-side
  (t_L values form the physical-radius schedule q_L = (R_guard/w_L +
  sqrt(3))^2, exactly self-consistent across levels, so exact-once holds by
  the §1 argument with q -> q_L).
- Demoted routes join the direct-pair list (panel subtree x particle
  subtree at that level; subtrees are contiguous in the two sorted orders,
  so the block-sparse near-field wrapper consumes them unchanged).
- With SOLVED strengths (the production case) truncation re-emerges and the
  measured operating point is q=12 / ell_x=5 / P=6 / R_guard=0.06 m
  (relRMS 9.7e-6, 10.3x margin; near+demoted ~3.3% of dense ~ 0.11 s
  device). q=3 demotes ~4x more pairs than q=12 at the same R_guard (its
  small stencil shell leaves more of the guard ball to route demotion), so
  the larger stencil is CHEAPER here — opposite of the unguarded intuition.
- The oracle inherits the same demotion rule (bit-compare still exact).

## 4. Standing items this memo does NOT cover

- The reverse leg (particles→panels, ruling update 2026-08-28): sequenced
  LAST; same machinery with roles swapped and no B2M (self-pass multipoles
  reused); target-side occupancy over ~37k control points.
- Hierarchical (non-increasing) cross q schedules: ruled out for v1 (R6);
  the §1 argument extends per task 025 if revisited.
- The operating point (q, P, ell_x): P3.2's deliverable.
