# 053 row 2 — default-behavior-change enumeration (UNREVIEWED DRAFT, generated 2026-09-03)

This is a first-pass, machine-assisted enumeration of commits in phases 046–052 (across FastMultipole, FLOWPanel.jl, FLOWVPM.jl) that plausibly change a **default behavior** — i.e. what happens when a user does not explicitly override an option. It is generated for 053 audit row 2 sign-off and has **not** been reviewed by Ryan. Every row needs confirmation; "approval evidence" only reflects what a commit message or in-code comment claims, not verified consent.

## Source-of-truth caveats (read before trusting row counts)

- **FastMultipole**: the cluster's `unified-052`/`origin/flowpanel-20260817` branches do **not** contain the 046–052 phase-tagged commit history at all. The coherent history only exists on the **local** `flowpanel-20260817` branch (Dropbox mirror), which is 10 commits ahead of `origin/flowpanel-20260817`. That local branch was used as source of truth for this repo.
- **FLOWPanel.jl**: on `unified-052`, phases 046–051 are not individually tagged commits — only phase-052 sub-letters (a/b/c/d/h) exist (16 commits, 2026-08-31 through 2026-09-02). The bulk of 046–052's actual work is bundled into a single squash commit `3d490e5` ("snapshot of 052-h200 silo state + gh200 dispatcher fix", 138 files, 31,044 insertions) with no incremental diff history. Only keyword-grep auditing of that squash was possible within budget — **higher-confidence follow-up recommended**, specifically diffing `FLOWPanel_solver.jl`, `FLOWPanel_wake.jl`, `FLOWPanel_formulation.jl` against pre-052 state for numeric-literal changes not caught by textual keywords.
- **FLOWVPM.jl**: on `unified-052`, there are only **3** phase/silo-tagged commits total (`4f6e805`, `6c8cda4`, `186bff4`), each a large squashed snapshot rather than incremental work. A finer-grained phase-by-phase audit is not possible from this clone; it would require the original silo branches that produced `4f6e805`, if they still exist.

Because of the above, **absence of a row below is not evidence of absence** — it may reflect squash-commit opacity rather than "no default changed."

## Table

| repo | commit | phase | what changed | old default → new default | user-approval evidence |
|---|---|---|---|---|---|
| FastMultipole | `d938ba68` | 052f | Near-set adequacy failure for a hierarchical regularized-kernel cache no longer throws; demotes cache to an all-direct zero-M2L geometry (with `@warn`, `maxlog=4`) and continues. Follow-on fix `2c6dd60f` (052g) needed to stop this path crashing the device M2L launcher. | throws `ArgumentError` → silently (with warning) falls back to all-direct zero-M2L evaluation | "Task 052f (user decision 2026-08-29)" cited in-code (`src/translate_batched_resident.jl:2111`) — approved |
| FastMultipole | `11b1edcc` | 051 | New `RectangularPanelInfluence` kernel's selectable filament-regularization family defaults to Vatistas n=2 (family 1), diverging from FLOWPanel's own working-tree source (default Gaussian/Lamb-Oseen, family 3) it was transcribed from. | new feature; FastMultipole default = `:vatistas` (family 1) vs. FLOWPanel upstream default = `:gaussian` (family 3) | NEEDS RYAN |
| FastMultipole | `9c812a92` | 052 | Device bounds check loosened from exact `x_min <= p <= x_min+extent` to a 4-ulp tolerance band, to stop false out-of-bounds flags on valid bodies. Low-significance numerical bugfix but does change a boundary-check constant. | exact bound → 4-ulp tolerance band | NEEDS RYAN (borderline; likely uncontroversial bugfix) |
| FLOWPanel | `3d490e5` | 052d | Default vortex-filament regularization kernel changed. | `GaussianRegularization` → `LineGaussRegularization` | "Ryan ruling 2026-08-29, task 052d" cited in source doc comment — approved |
| FLOWPanel | `3d490e5` | 052 (sub-phase undated) | New `FMM_RADIUS_TOL` mechanism inflates panel radii used in multipole-buffer admission by default (tol=1e-6), changing near-field/expansion boundary behavior vs. pre-2026-08-13. | no inflation (raw panel radius) → inflation on, tol=1e-6 (`Inf` disables) | NEEDS RYAN (comment references "pre-2026-08-13 radii" but cites no explicit ruling) |
| FLOWPanel | `3d490e5` | 052 (sub-phase undated) | `GPU_ALLOW_FALLBACK` env var gates whether GPU-path failures silently fall back to CPU; defaults to `"true"` (fallback permitted). | unspecified/no gate → fallback allowed by default | NEEDS RYAN |
| FLOWVPM | `4f6e805` | 052 | GPU-backed particle fields now route `UJ_fmm` through the new radix-FMM pipeline (`UJ_fmm_gpu!`) instead of the legacy `nearfield_device=true` forwarding path, whose own added code comment says it "silently DROPPED the nearfield contribution." | legacy `nearfield_device` path (silently missing nearfield for GPU fields) → new radix-FMM device pipeline (`src/FLOWVPM_fmm_radix.jl`) | NEEDS RYAN (framed in-code as a bugfix, not marked approved) |
| FLOWVPM | `4f6e805` | 052 | Two-system `UJ_direct(source,target)` direct-sum call changed from bare `fmm.direct!(target,source)` (framework defaults) to explicit `scalar_potential=false, gradient=true, hessian=true`. | framework default `DerivativesSwitch` → `scalar_potential=false`, `hessian=true` always computed | NEEDS RYAN |
| FLOWVPM | `4f6e805` | 052 | New `RadixFMMSettings` struct ships default constants for the now-mandatory-for-GPU radix FMM path: `expansion_order=6`, `near_radius2=6`, `direct_kernel=:partitioned`, `m2l_strategy=:dense`, `accuracy_margin=1.03`, `rho_t` fallback `4.789`. Docstring claims `expansion_order` "defaults to 4" but the `@kwdef` code default is actually 6 — doc/code mismatch. | no prior GPU FMM default existed (new subsystem) → these become the unconditional GPU default | NEEDS RYAN |

## Not flagged (checked, default preserved / out of scope)

- FastMultipole `2accbf56` (047): construction-lock registry now throws on post-build setting mutation — behavior-on-misuse change, not a default value.
- FastMultipole `22376fea` (051): `RectangularPanelInfluence` Float64-only hard requirement + on-plane snap — new kernel, no prior default to diverge from.
- FastMultipole `d7251cc5` (052c): commit message mentions "partitioned rho_t default (3.668 → 4.789)" but this is a test-fixture value only; actual `PartitionedVortex.rho_t` source default remains `4.252` (`src/containers.jl:2044`), untouched.
- FLOWPanel `6a64402`, `a146193`, and 13 other individually-tagged commits: env-gated opt-in features explicitly defaulting off (e.g. `GS_VERBOSE`, `PANEL_WAKE_FMM`/`PANEL_WAKE_FMM_DEVICE`), preserving prior behavior.
- FLOWVPM `6c8cda4`: new `sigma_guard` key `:ceil`, default `Inf` — docstring states empty/default reproduces old behavior bit-exactly.
- FLOWVPM `4f6e805`: `Project.toml` Julia compat bump 1.6→1.9 + new CUDA weakdep — environment requirement, not runtime default. Also `FLOWVPM_gpu_erf.jl` `sb7`→`sb7s` — plain typo fix.
- FLOWVPM `186bff4`: scripts-only, no source changes.

## Row 3 status update (2026-09-03 ~07:45 MDT)

Three-repo local suites run (4 threads): FastMultipole GREEN;
FLOWVPM GREEN after updating the stale `sigma-outgrown rebuild` sub-case
to the 052f demotion contract (warn, not throw); FLOWPanel 656/658 —
2 OPEN failures in `explicit jump fallback (:jump)`
(test/runtests_unit_kutta.jl:539-540, bitwise-identity to the legacy
A/jump trajectory broken; likely 7fbd68a; not caused by this session —
see 2026-09-02 §Decision log). A `Logging` test dep was also added to
FLOWPanel test/Project.toml (was a hard load error masking the suite).
