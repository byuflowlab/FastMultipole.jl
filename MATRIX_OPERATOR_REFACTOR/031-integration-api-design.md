# 031 Integration API Design: Generalizable Device-Resident System Interface

## Status and Entry Gate

**Added by user request on `2026-08-04`** as the first row of the Integration
Phase. Done and clear-context approved (2026-08-05, fourth review — record at
the end of this file).

Entry is unblocked: the only dependency (`019a`) is Done and clear-context
approved. This row does not block on `030`; the two may proceed in parallel.

This is a **design row**: it produces specification and analysis artifacts
under `MATRIX_OPERATOR_REFACTOR/` only. It makes no changes to production
`src/` in either repository. Like `008b`, it is **user-in-the-loop**: the
final interface spec requires explicit user sign-off before `032` may start.

## Objective

Design the generalizable device-resident system interface that lets an
external code keep its particle states on the GPU and drive the resident FMM
lifecycle with no per-step host/device body transfer — with FLOWVPM as the
first, concrete consumer. Where a transfer-based coupling is instead
appropriate for a consumer, specify measured guidelines for that path.

Deliverables:

1. **Interface specification** (`theory/` or a top-level
   `integration-api-spec.md` artifact in this directory): the storage layout
   and ownership of positions, strengths (scalar and 3-vector), extra states
   (e.g. smoothing radius `σ` for finite-core models), scalar potential,
   gradient, and **9-component hessian** (user decision `2026-08-04`: no
   6-component symmetric variant — the Lamb-Helmholtz / vector-potential
   velocity-gradient contribution is not symmetric in general, and FLOWVPM
   requires Lamb-Helmholtz); the `residency` trait surface and the
   `source_to_buffer!` / `buffer_to_target!` device contracts; and the
   capacity/no-reallocation contract (the existing `RadixFMMCache`
   `max_n_bodies` + derived-capacity + `RadixStepCounts` valid-prefix design
   is the mechanism — generalize and document it, do not reinvent it).
2. **Gap-analysis table**: FastMultipole resident path as shipped vs FLOWVPM
   requirements. Must cover at minimum: scalar-strength 5-row body packing
   (`[x, y, z, radius, strength]`) vs FLOWVPM's 8-row packing with vector `Γ`
   and `σ`; device B2M for `Point{Vortex}`; the Lamb-Helmholtz channel
   end-to-end on the device lifecycle; hessian output (currently absent —
   `fmm!` throws on `hessian=true` for the resident path); output
   accumulation semantics into FLOWVPM's `U_INDEX`/`J_INDEX` rows; and the
   FLOWVPM 46×N column-major `CuArray` particle matrix ↔ FastMultipole buffer
   mapping (aim for direct row-view mapping without intermediate copies where
   possible).
3. **Resident-vs-transfer decision framework**: when should a consumer keep
   state device-resident vs pay per-step transfers? Cite the existing `028`
   measurements (fully host-resident was ~1.7x the verdict-boundary cost at
   `n = 1e6`; per-step transfer costs were measured separately) rather than
   re-measuring; state the guideline as a function of `n` and step cost, and
   note what a consumer must implement in each mode.
4. **User sign-off checkpoint**: present the spec, the gap analysis, and the
   recommended `032` scope to the user; record the decisions in this file.
   `032` may not start until this sign-off is recorded.

## Dependencies

- `019a-milestone-review-final-roadmap.md`, complete and clear-context
  approved.
- Reference material (not blocking): `028`'s residency/transfer measurements,
  `scripts/fm028_device_system.jl` (the current device-system reference
  implementation), and the shipped resident lifecycle
  (`src/translate_batched_resident.jl`, `src/translate_batched_cuda.jl`,
  `src/containers.jl` residency trait and cache types).

## Mandatory Reading Gate

Before drafting the spec, read in full:

1. `START_HERE.md` (this directory), including the Integration Phase preamble.
2. `../FLOWVPM.jl/CLAUDE.md` — required before any FLOWVPM-facing design
   statement.
3. `../FLOWVPM.jl/src/FLOWVPM_fmm.jl` and the particle-layout section of
   `../FLOWVPM.jl/src/FLOWVPM_particlefield.jl` (row-index constants,
   accessors, `arraytype`).
4. `scripts/fm028_device_system.jl` and the residency-contract sections of
   `028-performance-feasibility-1m-in-10ms.md`.
5. The `RadixFMMCache` docstring and capacity contract in
   `src/translate_batched_resident.jl` and `src/containers.jl`.

## Task-Local Requirements

- The spec must be written for a general external consumer first, with FLOWVPM
  as a worked example — not a FLOWVPM-only design.
- Every interface function in the spec must state: who allocates, who owns,
  when it is called in the step lifecycle, and its allocation behavior
  (steady-state must be zero-allocation).
- The spec must state explicitly how a consumer with a known maximum particle
  count pre-sizes the cache once and runs a varying live count `n <=
  max_n_bodies` across steps.
- Record any discovered blocker that would force `032` scope changes in this
  file immediately, not in `032`.

## Work Record (2026-08-04)

Mandatory reading gate completed: `START_HERE.md` (incl. Integration Phase
preamble), `../FLOWVPM.jl/CLAUDE.md`, `../FLOWVPM.jl/src/FLOWVPM_fmm.jl` (full
read), the FLOWVPM particle-layout constants, `scripts/fm028_device_system.jl`
(full read), the `RadixFMMCache` docstring/constructor
(`src/translate_batched_resident.jl:803-930`), and the `028` residency
contract sections.

Deliverables:

1. **Interface specification**: `integration-api-spec.md` (this directory) —
   consumer surface and per-function contracts (§2), packing/output layout
   (§3), capacity/no-realloc contract incl. the domain-box design point (§4),
   far-field/nearfield kernel generalization (§5), 9-component hessian
   decision rationale (§6), FLOWVPM worked example (§7), resident-vs-transfer
   framework (§8), sign-off items (§9).
2. **Gap analysis**: `integration-api-spec.md` §10 — nine confirmed gaps with
   file:line evidence. Two gaps were discovered beyond the entry-gate list:
   the resident nearfield direct kernel is hard-coded singular scalar `1/r`
   with no user-kernel hook (FLOWVPM needs regularized Biot-Savart with per
   body σ) — this is a first-order `032` work item; and the device-resident
   source refresh allocates a fresh `CuArray` per step, violating the
   zero-allocation contract for device-resident consumers.
3. **Resident-vs-transfer decision framework**: `integration-api-spec.md` §8,
   citing the `028` measurements only (no new runs).

Findings affecting later rows (non-blocking):

- FLOWVPM maps FMM *gradient* → `U` and FMM *hessian* → `J`; one RK3 step
  makes **three** full `fmm!` evaluations (with per-substep tree refresh) —
  `035` must report the per-evaluation and per-step multiplier separately.
- `solve_ρ_over_σ` (a Roots.jl solve per body inside `source_to_buffer!`)
  is constant when regularization autotuning is off; the autotuned path needs
  a device-safe form — flagged to `034`.
- The legacy dynamic-`P` error-tolerance machinery has no resident
  equivalent; accuracy on the resident path is set by `expansion_order` +
  stencil geometry, tuned in `035` against the `033` gate.

## User Sign-Off (recorded 2026-08-04)

The user reviewed the spec summary and decided the four §9 items:

- (a) **Domain box**: ship `recenter!` in `032`, and pursue a scaling
  operation putting all bodies on the unit cube (normalized internal
  coordinates). The user asked whether scaling strengths would also be
  appropriate; resolution recorded in the spec §4: strengths stay physical —
  strength scaling can make only one output derivative order pass through
  unscaled, so all scaling lives in the coordinate map (positions, radius, σ
  by `1/L`) plus per-derivative-order output factors (potential `1/L`,
  gradient `1/L²`, hessian `1/L³`); the payoff is box-size-invariant operator
  tables so `recenter!` never rebuilds them.
- (b) **Nearfield kernel**: try the isbits-functor trait; `032` must
  benchmark it against the hard-coded kernel and report to the user if it is
  slow.
- (c) **Packed rows**: the later roadmap review reduced speculative work.
  Implement the canonical all-`data_per_body`-row layout first. Implement the
  5-row core + side-table alternative only if a profile or bandwidth model
  predicts at least a 5% end-to-end U/J-solve improvement; ship it only if
  measurement confirms the gain.
- (d) **032 scope**: approved as specified, with `recenter!` moved into
  scope per (a).

Task complete (`Done`). Clear-context approval by a separate agent is
pending per protocol.

## Clear-Context Review Correction (2026-08-05)

The first review found that the signed-off `recenter!` addition did not yet
have the same complete lifecycle/ownership/allocation contract as the other
interface functions. `integration-api-spec.md` §2/§4 now specifies explicit
call timing, deterministic caller-supplied bounds, allocation-free host/device
bounds derivation, validation/exception behavior, ownership, and the exact
state invalidated before the next ordinary refresh. It also distinguishes the
cache-lifetime invariants from the physical coordinate map that only an
explicit `recenter!` may mutate. No production files changed.

Because that reviewer changed the specification, it did not approve its own
correction. Row `031` remains Done but not Approved until a different
clear-context agent approves the corrected artifacts.

## Second Clear-Context Review Correction (2026-08-05)

The second clear-context review verified the spec's objectives, its
`resident`/capacity/ownership contracts, the normalized-coordinate scaling
argument (potential `1/L`, gradient `1/L²`, hessian `1/L³`), and every
`file:line` citation in §§2-10 — including `src/compatibility.jl:713-737`
(9-component `set_hessian!`), `src/translate_batched_cuda.jl:922-936` (5-row
truncation, zeroed radius row) and `:4005` (per-step `CuArray` allocation),
`src/fmm.jl:876-878` (hessian throw),
`src/translate_batched_resident.jl:664` (out-of-box `ArgumentError`),
`src/compatibility.jl:53,69` (deprecated hooks), and the FLOWVPM row mapping
(`X_INDEX 1:3`, `GAMMA_INDEX 4:6`, `SIGMA_INDEX 7`, `U_INDEX 10:12`,
`J_INDEX 16:24`). All resolve as described.

One correction was required, inherited from the `031a` review. The spec's §5
amendment framed the near-set coverage requirement (direct geometry must
contain every pair with `r/σ_src ≤ ρ_t`) as specific to `032a`'s partitioned
kernel. It is not: the FMM far field is singular under both nearfield
strategies, so the regularized-everywhere `RegularizedVortex` baseline shipped
by `032` needs the identical guarantee, and the shipped `|o|²≤12` leaf stencil
does not meet it at `n=1e6, ℓ=5, β=2, ε=1e-3`. As written, `032` would have
shipped its baseline on an inadequate stencil and silently missed the phase
accuracy gate. §5 now carries the adequacy rule and a `032` acceptance item,
§10 gained gap row `8a`, and §7 records the resulting lower bound on `034`/`035`
depth tuning. Derivation and evidence are in
`theory/kernel-splitting-nearfield.md` §5.1 and
`data/kernel_splitting/near_set_adequacy.csv`. No production files changed.

Because this reviewer also changed the specification, it did not approve it.
Row `031` remains Done but not Approved until a different clear-context agent
approves the corrected artifacts.

## Third Clear-Context Review Correction (2026-08-05)

The third clear-context review re-verified the spec's objectives and contracts,
re-derived the normalized-coordinate scaling from the definition
(`φ'(x') = Lφ(x)` and `∂' = L∂` give potential `1/L`, gradient `1/L²`, hessian
`1/L³`), and independently re-checked the load-bearing `file:line` citations —
`translate_batched_cuda.jl:922-936` (5-row truncation, zeroed radius row),
`:4005` (per-step `CuArray` allocation), `src/fmm.jl:876-878` (hessian throw),
`translate_batched_resident.jl:654-662` (`targets === sources`). All resolve as
described.

One correction was required, again inherited from the `031a` review. §5's `032`
acceptance item stated the near-set adequacy assertion as
`n/8^ℓ > (ρ_t β / g_min)³`. The overlap `β` is not the problem — every `033`
case sets it explicitly. The problem is `n/8^ℓ`: the underlying floor counts
bodies per **occupied** cell, which equals the all-cell average only for a field
that fills its bounding cube. The `033` wake cylinder (AR=5) fills 3.14% of
its cube, so the uniform form under-reports its admissible depth by one to two
levels (`ℓ≤2/3/5/6` geometric against `ℓ≤1/2/3/4` uniform at
`n=1e3/1e4/1e5/1e6`). Because compliant direct work falls steeply with depth
(`031a` §5.1), shipping that form would have forced one of the two mandated test
cases into a more expensive configuration; it is also blind to a σ grown by
`CoreSpreading`. (The defect was originally found on the vortex ring, which the
user replaced with the wake cylinder on `2026-08-05` — `033`'s wake amendment
has the rationale. The conclusion is unchanged; both cases now use `β=2` against
the local mean spacing, so the two forms differ purely through occupancy.)

§5 now specifies the primitive geometric test `g_min·h_leaf > ρ_t·σ_max`
(equivalently the depth ceiling `2^ℓ < g_min·L_box/(ρ_t σ_max)`), computable
from the cache box and a device max-reduction over the packed σ row, and
demotes the uniform rule to design-time sizing. §7's `034`/`035` bullet now
gives the ceiling with both cases' numbers, and gap row `8a` was updated to
match. Derivation and evidence: `theory/kernel-splitting-nearfield.md` §5.2 and
`data/kernel_splitting/case_adequacy.csv`. No production files changed.

Because this reviewer also changed the specification, it did not approve it.
Row `031` remains Done but not Approved until a different clear-context agent
approves the corrected artifacts.

## Fourth Clear-Context Review — APPROVED (2026-08-05)

A fourth clear-context agent reviewed `integration-api-spec.md` against the
`031` objectives, the Integration Phase preamble, and the production source,
and **approved it without changing it**. Row `031` is now Done and Approved;
`032` is unblocked.

Verified independently in this review:

- **Objectives.** All four deliverables are present and answer the entry-gate
  list: per-function contracts with caller/owner/timing/allocation (§2), the
  packing and 13-row output layout (§3), the capacity/no-realloc contract and
  the `recenter!` lifecycle added by the first review (§4), kernel
  generalization (§5), the 9-component decision (§6), the FLOWVPM worked
  example (§7), the resident-vs-transfer framework citing `028` only (§8), the
  recorded sign-off (§9), and nine + one gap rows with evidence (§10).
- **Scaling argument (§4).** Re-derived from `φ'(x') = Lφ(x)`, `∂' = L∂`:
  output factors `1/L`, `1/L²`, `1/L³`, and the claim that strength rescaling
  can normalize only one derivative order. Correct as stated.
- **Citations.** `translate_batched_cuda.jl:922-936` (5-row truncation, hard
  zeroed radius row 4), `:4005` (per-step `CuArray`), `src/fmm.jl:876-878`
  (hessian `ArgumentError`), `translate_batched_resident.jl:654-662`
  (`targets === sources`) and `:664` (out-of-box assert),
  `src/compatibility.jl:713-737` (9-component `set_hessian!`). All resolve.
- **§5/§7/§10 adequacy rule (the third review's correction).** The geometric
  ceiling `2^ℓ < g_min·L_box/(ρ_t σ_max)` was re-derived from the `031a` §2 gap
  formula and re-evaluated by hand for both `033` cases: cube `ℓ ≤ 1/2/3/4`
  and wake `ℓ ≤ 2/3/5/6` at `n = 1e3/1e4/1e5/1e6`, matching §7 and
  `data/kernel_splitting/case_adequacy.csv` exactly. The demotion of the
  `n/8^ℓ` form to design-time sizing is correct: it is an all-cell average and
  the wake fills 3.14% of its bounding cube.
- **FLOWVPM mapping (§7).** Confirmed against `gpu-full`:
  `../FLOWVPM.jl/src/FLOWVPM_fmm.jl:102-168` uses the **source** `σ` only
  (`source_buffer[8, i]`), its `aux`/`aux2` are exactly the spec's and `031a`'s
  `a` and `b`, and its nine `du*` entries match the derived `J` component
  order sign for sign. Gradient→`U`, hessian→`J` is correct.

One non-blocking hand-off gap was found — the spec (2026-08-04) predates
`031a` §§6.2-6.3, so `032`'s mandated FDLIBM `custom_erf` port and `032a`'s
divergence handling do not reflect them. It is recorded in the downstream
rows' own task files rather than by amending this approved spec.
