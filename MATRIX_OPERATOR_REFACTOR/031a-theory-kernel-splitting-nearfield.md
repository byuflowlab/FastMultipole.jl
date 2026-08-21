# 031a Theory: Partitioned Nearfield for Regularized Biot-Savart

## Status and Entry Gate

**Added by user direction on `2026-08-05`; revised by user direction later the
same day to replace the rejected singular-minus-correction design.** Scoped
derivation row, confined to `theory/`, `scripts/`, and `data/`; no production
`src/` changes and no reopening of the Theory Phase hard gate.

Done and clear-context approved (2026-08-05, fifth review — record at the end
of this file). `031` is likewise approved, so the approval chain into `032a`
is complete.

Kernel scope: FLOWVPM's default `gaussianerf` kernel only. The SFS
(`ζ`/`Estr`) kernel remains deferred.

## Objective

Derive a numerically stable partitioned nearfield for regularized Biot-Savart
U and J. The far field remains the classical singular FMM. Direct pairs inside
a conservative smoothing cutoff are evaluated once with stable regularized
formulas; remaining direct pairs use the cheaper singular kernel.

Deliverables:

1. Explicit component formulas for regularized U and all nine J entries.
2. Cancellation-safe small-`ρ` series for `g(ρ)` and
   `h(ρ)=ρg'(ρ)-3g(ρ)`.
3. Conservative U/J cutoff radii `ρ_t(ε)` and an exact-once geometry contract
   ensuring no cutoff pair remains represented by M2L.
4. A cost model against regularized-everywhere nearfield evaluation.
5. Stdlib-only Float64/Float32 validation against a 256-bit reference,
   covering `ρ ∈ [1e-6,10]` and normalizing errors to regularized U/J.

## Dependencies and Reading

- `START_HERE.md`, including the Integration Phase amendment.
- `integration-api-spec.md` §5.
- `../FLOWVPM.jl/src/FLOWVPM_kernel.jl` and
  `../FLOWVPM.jl/src/FLOWVPM_fmm.jl:102-168`.
- `031-integration-api-design.md`, signed off and complete; fresh
  clear-context approval is pending after its review correction and must be
  obtained before this row can be approved.

## Work Record (2026-08-05) — Done

The completed derivation is
`theory/kernel-splitting-nearfield.md`. It specifies:

- exact regularized and singular U/J formulas in FLOWVPM component order;
- stable precision-specialized Horner series below `ρ=0.5` (six terms in
  Float32, ten in Float64);
- partitioned replacement at `ρ_t`, with the explicit source-directed AABB
  predicate `d_min(B_t,B_s) ≤ ρ_t max(σ_src)` selecting direct and
  construction-time exact-once validation;
- conservative bounds `E_U=ḡ` and `E_J=ḡ+ρg'/2`, reserving half the
  requested error budget, plus the regularized-relative proof
  `E/(1-E) ≤ ε`;
- (§5.1, added by the 2026-08-05 review correction below) the stencil adequacy
  rule `n/8^ℓ > (ρ_t β / g_min)³` with `g_min = √5` for the `θ=0.5` stencil and
  `1` for the classic stencil, the finding that the shipped `|o|²≤12` leaf
  stencil is **not adequate** at `n=1e6, ℓ=5, β=2, ε=1e-3`, and the conclusion
  that the correct remedy is enlarging the deepest-level near set (179 → 389
  classes) rather than reducing `ℓ`; and
- a cost model on adequate geometry predicting a 1.30-1.71x pair-kernel
  speedup at the shipped `ℓ=5` operating point (1.45-2.26x at `ℓ=3`, decaying
  to 1.19-1.40x at `ℓ=6`) over regularized-everywhere evaluation, before GPU
  branch/routing costs.

Generated cutoff evidence remains in `data/kernel_splitting/rt_table.csv`.
At the fixed `ε=1e-3` phase tolerance the U/J radii are `4.211/4.789`; at
`ε=1e-6` they are `5.665/6.182`.

`scripts/validate_031a_kernel_split.jl` now validates only the partitioned
design. Results in `data/kernel_splitting/partitioned_replacement.csv`:

- Float64 close-series relative error: U `8.12e-16`, J `6.15e-16`;
- Float32 close-series relative error: U `3.35e-7`, J `3.08e-7`;
- ordinary regularized branch: at most `5.82e-15` Float64 and `1.63e-6`
  Float32;
- singular tail beyond the `ε=1e-6` J cutoff: below `3.45e-7` in both
  precisions for the validation geometries.

`data/kernel_splitting/geometry_coverage.csv` records the deterministic grid
enumeration for two regimes — a sub-cell cutoff (`ρ_t σ_max = 0.955 h`) and the
overlap-2 leaf regime (`3.065 h`) added by the review correction. In both, every
ordered non-self pair appears exactly once, with zero missing pairs,
duplicates, or cutoff pairs assigned to M2L.

`data/kernel_splitting/near_set_adequacy.csv` (added by the review correction)
tabulates stencil adequacy and the §6 cost model over
`n = 1e3..1e6`, `ℓ = 0..6`, `β ∈ {1.5, 2, 2.5}`, `ε ∈ {1e-3, 1e-4, 1e-6}` for
both the `θ=0.5` and classic stencils.

The earlier two-pass singular-minus-correction derivation, runtime-erfc
requirement, validation, and generated data were removed by explicit user
direction. They are not candidates for `032a`.

Task is Done. Because the derivation materially changed, clear-context
approval by a different agent remains pending.

## Clear-Context Review Correction (2026-08-05)

The clear-context review re-derived §1's `g`, `g'`, `a`, `b` and all nine `J`
components, both §3 Horner series, and the §4 `E_J = ḡ + ρg'/2` bound and
`E/(1-E) ≤ ε` half-budget proof: all correct and unchanged. It found one
substantive error, now fixed.

**Error.** §2 argued the shipped near set needs no enlargement because
`ρ_t σ ≈ 9.58d < R_near ≈ 10.8d`. But `R_near = √12 h` is the stencil's outer
centre-to-centre radius, whereas §5's own predicate binds on the *minimum AABB
gap* over M2L cell pairs. For the `|o|²≤12` stencil that minimum is at
`o = (3,2,0)` and equals `√5 h = 6.99d`, well below `9.58d`. The shipped leaf
stencil is therefore **not adequate** at `n=1e6, ℓ=5, β=2, ε=1e-3`; its
worst-case M2L pairs sit at `ρ = 3.49`, where the doc's own bounds give
`E_U = 6.8e-3` and `E_J = 4.5e-2` against a `1e-3` phase gate. §6's
`f = (ρ_t σ / R_near)³ ≈ 0.70` and its `1.11-1.22x` prediction inherited the
same wrong radius.

**Corrections applied** (all in `theory/kernel-splitting-nearfield.md`):

- §2 now states the gap-based criterion, the enumerated `g_min` per stencil,
  and the non-adequacy finding with its error magnitudes; and records that the
  requirement applies to **both** nearfield strategies, since the far field is
  singular either way — a regularized-everywhere kernel incurs the identical
  tail error on cutoff pairs its near set misses.
- New §5.1 derives the level-invariant adequacy rule
  `n/8^ℓ > (ρ_t β / g_min)³` (79 bodies/cell for `θ=0.5`, 879 for classic at
  `β=2, ε=1e-3`), shows the constraint binds at the leaf level only, and
  compares the two remedies.
- §6's cost model is recomputed on adequate geometry.

**Result that reverses the expected remedy.** Reducing `ℓ` restores compliance
but is the expensive option: it multiplies bodies per cell by 8 while the
required class count grows only about 3x, so compliant direct work *falls*
with depth (52,734 pairs/target at `ℓ=3`, 28,564 at `ℓ=4`, 11,871 at `ℓ=5`).
The cheap remedy is enlarging the deepest-level near set at fixed `ℓ`
(179 → 389 classes at `ℓ=5`). Partitioning's relative advantage moves the other
way, shrinking as the tree deepens, so `032a` must A/B the two strategies at a
fixed adequate geometry rather than at each strategy's own optimum.

**Validation strengthened.** `scripts/validate_031a_kernel_split.jl` gained a
target-regime geometry case (`ρ_t σ_max = 3.065 h`, where the previous test ran
at `0.955 h` and never exercised the failing regime) and a stencil-adequacy
enumeration asserting that the gap test and the closed-form bodies-per-cell
floor agree across the whole grid. `rt_table.csv` and
`partitioned_replacement.csv` regenerate byte-identical; `geometry_coverage.csv`
gained a `case` column and the new row; `near_set_adequacy.csv` is new.

**Downstream.** `integration-api-spec.md` §5/§10 were amended so the geometry
requirement attaches to the `032` regularized-everywhere baseline as well as to
`032a`'s partitioned kernel. `032a` must verify adequacy against its production
route construction, and `034`/`035` depth tuning is now bounded below by the
§5.1 floor.

Because this reviewer changed the artifacts, it did not approve them. Row
`031a` remains Done but not Approved until a different clear-context agent
approves the corrected theory document, script, and data.

## Third Clear-Context Review (2026-08-05)

The third clear-context review independently re-derived §1's `a`, `b`, and all
nine `J` components (confirmed against `../FLOWVPM.jl/src/FLOWVPM_fmm.jl:129-160`),
both §3 Horner series and every printed coefficient, the §4 `E_J` bound and
half-budget proof, the §2/§5.1 gap enumeration (`g_min = √5` at `o=(3,2,0)` for
`|o|²≤12`, `1` for classic; adequate class counts 27/117/389/1839 at
`ℓ=3/4/5/6`), the bodies-per-cell floor arithmetic, and the entire §6 cost
table. All reproduce. The script runs stdlib-only and every CSV regenerated
byte-identical to the committed copies.

One artifact defect was found and fixed. `required_direct_count` enumerated a
fixed half-width `reach=8`, so any grid row whose required gap exceeded `7h`
silently clipped its minimal adequate class count at `17³ = 4913`, corrupting
the derived `direct_pairs_per_target`, `f_regularized`, and speedup columns —
138 of 504 rows in `near_set_adequacy.csv`. None of those rows back any claim
in this document (all are degenerate, well under one body per cell, and every
cited row satisfied `required_gap_h ≤ 6.13`), but they were published without a
flag and are exactly the kind of row a downstream geometry choice in `032a`
might read. The function now derives its own exact bound from
`gap(o) ≥ |o|_∞ - 1` and carries a self-check against direct enumeration on the
shipped radii. `rt_table.csv`, `partitioned_replacement.csv`, and
`geometry_coverage.csv` are unchanged; every row cited in §5.1 and §6 of the
theory document is unchanged.

Because this reviewer changed the script and data, it did not approve them.

## Two-Pass Additive Correction Added as a Third `032a` Candidate (user direction 2026-08-05)

During the third review the user asked whether the exact-once concern above is
specific to a single fused nearfield pass, and directed that the two-pass route
be derived as a third candidate. It is: an additive correction rides on top of a
**complete** singular FMM, so it never asks which route owned a pair and touches
no `025` routing invariant. New `theory/kernel-splitting-nearfield.md` §6.1
derives it. Key results:

- **Operator.** `ΔU_i = -ḡ C_i`, `ΔJ = Δa C_i Δx_j + Δb ε_ijk Γ_k` with
  `Δa = (ρg' + 3ḡ)/r²`, `Δb = ḡ/(4πr³)`. No `erfc` is needed: `ḡ = 1-g` loses
  digits only where `ḡ` is already negligible (`4.2e-5` at `ρ_t`).
- **The §5.1 reach requirement moves rather than disappears.** Pass 2 must still
  visit every `r/σ_s ≤ ρ_t` pair — 389 classes at the shipped operating point.
  Reusing the 179-class direct route list inherits the identical inadequacy.
  The relief is that **pass 1 needs no enlargement**.
- **Conditioning.** The subtraction lands in the accumulator across two kernels,
  so no series can repair it; amplification is `1/g` for U and
  `2/|ρg'-2g|` for J, both `~ρ⁻³`. Measured: Float64 stays below `5.2e-10` at
  `ρ=0.01` and never reaches `1e-3` above `ρ=1e-4` (expected occurrence at
  `n=1e6`: `~1e-8` pairs), so **plain two-pass is safe in Float64**, and only
  the singular direct term and its correction need F64 — the far field may stay
  FP16-WMMA/Float32. In Float32 per-pair J error crosses `1e-3` below
  `ρ≈0.0765` (`~7.5e3` pairs at `n=1e6`); because the phase gate is a sampled
  RMS this is a **tail/stability exposure, not a gate failure**.
- **`ρ_c = 2` hybrid.** Evaluating `ρ ≤ ρ_c` with the §3 stable form inside
  pass 1 removes the cancellation entirely in Float32 (measured at working
  precision at every `ρ`). `ρ_c` must clear the `ρg'-2g` sign crossing at
  `ρ=1.3688`, and pass 1 must be adequate for `ρ_c` — a 5.7 bodies/cell floor
  that the shipped 179-class stencil meets at `ℓ=5`.
- **Cost.** The transcendental count is the physics floor `N_t` under both split
  strategies, so two-pass buys no transcendental work; it trades singular
  evaluations for extra visits, crossing over at `λ* = -0.065 / -0.119 / 0.548 /
  2.434` for `ℓ = 3/4/5/6`. **Two-pass and partitioning have opposite depth
  trends**: two-pass cannot win at `ℓ≤4` (the stencil already contains the
  cutoff ball) and wins for any plausible `λ` at `ℓ=6`. `032a` must measure both
  at fixed adequate geometry.

One inconsistency in the existing §6 model was corrected in the same pass: it
divided the physics floor by the *minimal adequate* class count, but a
production near set can never be smaller than the stencil's own class count
(179 at `θ=0.5`), which is fixed by far-field multipole accuracy. The model now
uses `max(stencil, adequate)`; only the `ℓ=3,4` `θ=0.5` rows change, and they
move in partitioning's favor (1.45–2.26x → 1.49–2.46x at `ℓ=3`).

New artifacts: `data/kernel_splitting/two_pass_conditioning.csv` and
`data/kernel_splitting/two_pass_cost_model.csv`;
`near_set_adequacy.csv` gained five `*_production` columns.
`rt_table.csv`, `partitioned_replacement.csv`, and `geometry_coverage.csv` are
unchanged.

## Three Levers That Outrank the Strategy Choice (user direction 2026-08-05)

The user asked whether two-pass reduces the number of `erf`/Gaussian
evaluations. It does not, relative to partitioned replacement: the set needing
`g` is physical (`ρ ≤ ρ_t`), so both split strategies hit the same floor
`N_reg = 3,681` pairs/target against `11,871` for regularized-everywhere.
Splitting buys 3.2x fewer transcendental evaluations; the choice *between* split
strategies only trades cheap singular evaluations against extra visits. That
prompted three additions (`theory/kernel-splitting-nearfield.md` §§6.2-6.4),
each worth more than the strategy choice:

- **§6.2 — the `erf` can be removed entirely.** Above `ρ_c` the result is `O(1)`,
  so `ḡ` is needed only to *absolute* tolerance. Writing
  `ḡ = e^{-ρ²/2}(Aρ + s(ρ))` with `s = erfc(ρ/√2)e^{ρ²/2}` moves the error
  function into `s`, which falls smoothly 0.3362 → 0.1601 across the shell. A
  degree-3 polynomial in `1/ρ²` holds `|δḡ| = 2.1e-4` against the `3.69e-4`
  budget, so the outer branch is **one hardware `exp` plus four FMAs** — no
  FDLIBM `custom_erf` port. Applies to partitioned too (via `g = 1 - ḡ` above
  `ρ≈2`), and below `ρ≈2` the §3 series is already erf-free.
- **§6.3 — warp divergence inverts the verdict for a naive implementation.** At
  `f = 0.310` the probability a 32-lane warp is branch-homogeneous is `6.9e-6`,
  so an unbinned partitioned kernel pays both paths on every warp: `λ+q+1`
  instead of `λ+fq+(1-f)`, measured **1.56x slower than
  regularized-everywhere**. Cell-level classification does not rescue it — only
  **19 of 389** direct classes at `ℓ=5` are entirely inside the cutoff, 370 are
  mixed. Either split strategy therefore needs a distance-binned or sorted pair
  stream, settled *before* the `032a` A/B, or it loses to the `032` baseline.
- **§6.4 — tighten `ρ_t` itself.** `N_reg ∝ ρ_t³`, and the §4 radii bound the
  *per-pair* error while the phase gate is a sampled RMS. Solving the
  incoherent-accumulation integrals gives `ρ_t(J) = 4.252` instead of `4.789` at
  `ε=1e-3`: **30% fewer expensive pairs**, leaf near set `389 → 275` classes,
  and the §5.1 adequacy floor relaxed from 79 to 55 bodies/cell. This is a
  statistical bound, not the §4 worst-case guarantee, so the recommendation is
  to keep the per-pair radii as the default and let `032a` adopt the RMS radii
  only after its sampled-direct measurement confirms them.

A latent bug was fixed in passing: `erfc_cf` used its Lentz continued fraction
at all arguments, and below `x≈1` that converges slowly enough for the
termination test to fire early (`2.7e-9` relative at `x=0.3`). It now uses the
existing `erf_series` below `x=2`. No published number moves — every cutoff root
sits at `x ≈ 3-6.4`, where the CF is accurate — and `rt_table.csv`,
`partitioned_replacement.csv`, and `geometry_coverage.csv` regenerate
byte-identical after the fix.

New artifacts: `cheap_gbar_fit.csv`, `divergence_model.csv`,
`rt_accumulated.csv`.

Row `031a` remains Done and not Approved; a different clear-context agent must
approve the corrected and extended theory document, script, and data.

## Fourth Clear-Context Review (2026-08-05)

The fourth clear-context review independently re-derived, rather than
re-checked, §1's `g`, `g'`, `a`, `b` and all nine `J` components; both §3 Horner
series (including the resummation of `h = ρg' − 3g` to
`Aρ⁵Σ(−1)^{k+1}ρ^{2k}/((2k+5)2^k k!)`); the §4 `E_J = ḡ + ρg'/2` bound (by
enumerating every `J` component with `Δx ∥ ẑ`, where the transverse entries give
`(ρg'−2g)kΓ` against singular `−2kΓ` and the `bΓ` entries give the smaller `ḡ`)
and the half-budget proof; the `ε=1e-3` radii 4.211/4.789 to three digits by
hand; the §6.1 `Δa`, `Δb`, and the `amp_J = 2/|ρg'−2g|` amplification; the §2
`g_min = √5` claim by exhausting every offset with gap² ≤ 4 and confirming all
lie inside `|o|²≤12`; the §5.1 rule and its 79/879 floors; the 27/117/389 class
counts by direct lattice enumeration; `N_reg = 3681`; and every speedup and
`λ*` entry in §6/§6.1. All reproduce. The script runs stdlib-only, exit 0, and
all nine CSVs regenerated byte-identical before any edit. The third review's
`required_direct_count` fix is sound (`reach = floor(gap)+1` is exactly right
given `gap(o) ≥ |o|_∞ − 1`) and self-checks against brute enumeration.

Three defects were found and fixed.

**1. The `032` adequacy assertion was stated in a form that only holds for a
box-filling field (substantive).** `integration-api-spec.md` §5 told `032` to
assert `n/8^ℓ > (ρ_t β/g_min)³`. The overlap `β` was never the weak point —
every `033` case sets it explicitly and visibly. The broken step is `n/8^ℓ`: the
§5.1 floor counts bodies per **occupied** cell, and that equals the all-cell
average only when the bodies fill their bounding cube. New §5.2 gives the
general occupied-cell form and the equivalent depth ceiling
`2^ℓ < g_min·L_box/(ρ_t σ_max)`, which needs no occupancy estimate at all and is
computable from the cache box plus a max-reduction over the packed σ row. New
`data/kernel_splitting/case_adequacy.csv` works both `033` cases at all seven
resolutions; the script asserts the two forms agree exactly on the box-filling
cube and that the uniform form never exceeds the geometric ceiling on the
sparse case.

The finding was first established on the then-current vortex ring (1.9% fill of
its bounding cube, uniform rule three levels low). Investigating it prompted a
review of the ring's fitness as a benchmark, and the user replaced it with a
helical wake cylinder on `2026-08-05` — see `033`'s wake amendment for the full
rationale. The finding survives the case change intact: the AR=5 wake fills
3.14% of its bounding cube, its occupied cells hold ~32× the all-cell average,
and the uniform rule under-reports its admissible depth by one to two levels
(`ℓ≤2/3/5/6` geometric against `ℓ≤1/2/3/4` uniform at `n=1e3/1e4/1e5/1e6`).
§5.2 and `case_adequacy.csv` are stated on the wake; the ring is gone from both.
Because both cases now use `β=2` against the local mean spacing, the two forms
differ *purely* through occupancy, which makes the comparison cleaner than it
was on the ring.

§5.2 also records what the ceiling does **not** imply: the finest admissible
cell `h_min = ρ_t σ_max/g_min ≈ 4.28·s` depends on particle spacing alone, so
the occupied-cell count at `h_min` is box-shape-independent. An oversized box
costs extra tree levels and inflates structures sized by total rather than
occupied cells — it does not directly inflate far-field work. That distinction
is why the elongated-domain penalty is staged as a *measured* `035` lever
(§1a of that row) rather than asserted.

**2. §6.3's divergence table published the fully-diverged bound at every `ℓ`.**
It reported 1.56x on all four rows while the adjacent column gave
`p_homogeneous_warp = 0.71` at `ℓ=3`. Warp-weighting the two homogeneous
outcomes gives 0.96x at `ℓ=3` — the unbinned kernel is *not* slower there — and
1.51/1.56/1.56x at `ℓ=4/5/6`; the `unbinned_beats_reg_everywhere` flag was
consequently wrong on that row. `divergence_model.csv` gained
`cost_unbinned_warpweighted_q15` and `unbinned_over_reg_everywhere_weighted`,
and the flag is now computed from the weighted cost. The binding conclusion is
unchanged: at the shipped `ℓ=5` operating point `p = 6.9e-6`, so a binned or
sorted stream is still a first-order requirement — and §6.3 now notes that the
loss relative to a *correct* implementation is in fact largest at `ℓ=3`, where
binning would have paid 1.49-2.46x.

**3. Two numeric nits.** §6.1's "`~10^{-8}` pairs" below `ρ=1e-4` was unsourced;
the script's scan finds no Float64 breakdown in `[1e-4, 1]` and printed a
meaningless `0`. It now reports the scan-floor bound, `1.7e-5` unordered pairs
at `n=1e6, β=2`, and the text quotes that. The `ℓ=4` pass-2 pair count
`28,565` is `28,564` (`117 × 244.140625`); corrected in §5.1, §6, §6.1, and in
the third-review record above.

After the fixes the script still runs stdlib-only and exits 0.
`divergence_model.csv` changed as described and `case_adequacy.csv` is new;
`rt_table.csv`, `partitioned_replacement.csv`, `geometry_coverage.csv`,
`near_set_adequacy.csv`, `two_pass_conditioning.csv`,
`two_pass_cost_model.csv`, `cheap_gbar_fit.csv`, and `rt_accumulated.csv`
regenerate byte-identical.

**Downstream.** `integration-api-spec.md` §5 (acceptance item), §7 (the
`034`/`035` depth-tuning bullet, now a ceiling with both cases' numbers), and
gap row `8a` were amended to the geometric form, with the uniform rule
explicitly demoted to design-time sizing.

Because this reviewer changed the artifacts, it did not approve them. Row `031a`
remains Done and not Approved.

## Fifth Clear-Context Review — APPROVED (2026-08-05)

A fifth clear-context agent reviewed `theory/kernel-splitting-nearfield.md`,
`scripts/validate_031a_kernel_split.jl`, and all ten CSVs and **approved them
without changing anything**. Row `031a` is Done and Approved.

Independently re-derived (by hand, not re-checked against the document):

- §1: `g' = Aρ²e^{−ρ²/2}` from `d/dρ erf(ρ/√2) = Ae^{−ρ²/2}`; `a = h/r²` via
  `1/(σr) = ρ/r²`; all nine `J` components from `J_ij = aC_iΔx_j + bε_{ijk}Γ_k`.
  Confirmed transcription-exact against `../FLOWVPM.jl/src/FLOWVPM_fmm.jl:102-168`
  (`aux`, `aux2`, and every `du*` sign), and that FLOWVPM regularizes on the
  **source** `σ`, which is what makes the §5 directional predicate correct.
- §3: both series resummed from scratch —
  `g = Aρ³Σ(−1)^k ρ^{2k}/((2k+3)2^k k!)` and
  `h = Aρ⁵Σ(−1)^{k+1}ρ^{2k}/((2k+5)2^k k!)` — and every printed coefficient
  (1/3, −1/10, 1/56, −1/432, 1/4224; −1/5, 1/14, −1/72, 1/528, −1/4992).
- §4: `E_J = ḡ + ρg'/2` by aligning `Δx ∥ ẑ` (transverse entries
  `(ρg'−2g)κΓ` against singular `−2κΓ`, the `bΓ` entries only `ḡ`); the
  half-budget proof; and the radii `4.211`/`4.789` evaluated by hand to
  `E_U = 5.001e-4`, `E_J = 4.995e-4`.
- §2/§5.1: `g_min = √5` at `o=(3,2,0)` by exhausting every offset with
  `gap² ≤ 4`; the floors 79/879; the ℓ=5 failure (`3.065h` against `2.236h`)
  and its `E_U = 6.8e-3`, `E_J = 4.5e-2`.
- §5.2: both `033` cases worked from `L_box = 5D`, `σ = 2(V_cyl/n)^{1/3}` —
  ceilings 2/3/5/6 (wake) against 1/2/3/4 (uniform rule), 3.14% fill, ~32×
  concentration.
- §6/§6.1: `N_reg = 3681`; every `f` and both speedup columns at
  `q = 1.5/2.5`; `Δa`, `Δb`, `amp_J = 2/|ρg'−2g|`; the `ρg'−2g` crossing near
  1.3688; `amp_J(2) = 3.26`; the `ρ_c=2` pass-1 floor `(4/√5)³ = 5.7`; and all
  four `λ*` crossovers (−0.065, −0.119, 0.548, 2.434) from the class counts.
- §6.2: `ḡ = e^{−ρ²/2}(Aρ + s)`, `s(2) = 0.3362`, `s(4.789) = 0.1601`, budget
  `3.69e-4`. §6.3: `p_hom` at all four levels and the warp-weighted 0.96/1.51/
  1.56/1.56 ratios. §6.4: the `×0.700` pair reduction and the relaxed 55-body
  floor.
- Every lattice count reproduced by independent enumeration: 179 (`|o|²≤12`),
  27/117/389/1839 adequate classes, 275 (`ρ_c` pass 1 at ℓ=6 and the §6.4 RMS
  radius at ℓ=5), and the 19/389 divergence-free counts.

Script and data: runs stdlib-only, exit 0, and all ten CSVs regenerated
**byte-identical** to the committed copies. The fourth review's
`required_direct_count` fix holds — no row is clipped at `17³ = 4913`, and the
largest row (`required_gap_h = 98.91`) reports 4,239,579 classes, consistent
with `(4π/3)(gap+1)³`. Assertions are live `error(...)` calls at the 16 sites
listed, including the two §5.2 cross-form checks.

No defects were found and nothing was changed, so this review approves the
row. The `§6.2`/`§6.3` hand-off into `032`/`032a` is recorded in those rows'
task files.
