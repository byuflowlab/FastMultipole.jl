# Plan: 041i — Census: Sigma-Question Closure at the Shipped Operating Point

> **Approved execution plan, 2026-08-18.** Self-contained: the "Key facts from
> exploration" section below records everything a fresh agent needs (file
> paths, line numbers, API signatures, seeds, conventions) — start from Step 1
> without re-exploring. Read `START_HERE.md` and
> `041i-census-sigma-closure.md` first per protocol; this plan implements that
> task file plus the one user decision recorded in Context.

## Context

Task file: `MATRIX_OPERATOR_REFACTOR/041i-census-sigma-closure.md` (staged 2026-08-18; unblocked — 041g and 041a both Done+Approved; blocks 042). The user hypothesized that large-particle sigma contaminates the multipole representation, inflating the direct list. The existing record contradicts this for the rotor (041g: zero sigma demotions at both registered counts), but two channels are unmeasured:

- **Part A (operating-point gap):** the 041g demotion census ran at `near_radius2 q=12`, `rho_t=4.252`; the shipped defaults are `q=5` (`RADIX_DEFAULT_NEAR_RADIUS2`, `src/containers.jl:347`) and `rho_t=4.789` (`RegularizedVortex`, `src/containers.jl:2002`), a 2.24× tighter adequacy margin where rotor ell=7 leaf width < `rho_t·sigma_max`. Zero demotions has never been verified there.
- **Part B (split-veto channel):** the sigma depth cap (`src/tree_batched.jl:803-812`, in `_adaptive_build_pool!`) has never been censused: depth forgone, larger leaves, extra U pairs.

Constraints: measurement row only — artifacts under `MATRIX_OPERATOR_REFACTOR/scripts/` and `data/` only, **no `src/` or FLOWVPM changes**, ≤4 local threads, H200 spot-check only if Part B's u_pairs delta > 1% on some case.

**User decision (2026-08-18, this session):** the task file's "gate active (shipped behavior)" wording for Part B is factually off — `AdaptiveTreePolicy` ships `split_veto=false` ("default OFF, pending user ratification", rationale at `src/containers.jl:557-568`; the CUDA path throws if enabled, `src/tree_batched_cuda.jl:70-73`). Proceed as specced but reframe the verdict: the Part B delta prices *ratifying the veto ON*; the shipped sigma cost flows only through the Part A demotion channel.

## Key facts from exploration

- `scripts/sigma_class_m2l_census.jl` (041g, 550 lines) is **self-contained pure Julia**: it includes the reference implementation `scripts/adaptive_octree_verify.jl` (`Tree`, `build_tree`, `balance!`, `build_lists`, `sigma_upward`, `check_exact_once`, `work_counts`) and touches no `src/` code. `q=12` is hardcoded at `build_lists(t, 12; ...)` (lines 321, 413); `rho_t` hardcoded to `RHO_UJ_RMS = 4.252`; `ell` caps hardcoded in `build_tree` calls (lines 320, 407). No CLI surface — all consts.
- 041g rotor loading: **not** the `.bin` snapshots — `load_rotor_iterator!()` string-slices `scripts/benchmark_033_common.jl` (sha256-pinned) and replays `fm033_rotor_foreach` with `MersenneTwister(33025 + 104729 + n)` → `(xs, sigma, strength)`. The 041a/041e convention instead reads `data/rotor_wake/rotor_snapshot_n$(n).bin` (layout: `Int64 n`, `X(3,n)`, `G(3,n)`, `s(n)` Float64) — e.g. `scripts/fm041e_target_owned_bench.jl:273-300`.
- 041g's cube/wake "controls" were `make_uniform`/`make_filament` at n=4096 — *not* the 041a cases. The 041a registered cube/wake are `make_unitcube(n; seed=39101)` / `make_wake(n; seed=39102)` (copy-pasted into `fm041e_target_owned_bench.jl:108,113` and `fm041a_host_widen.jl:47,52`) at n ∈ {1e5, 1e6}; adaptive K_max winners: cube 64, wake 256, rotor 64; `ell_max=10`, `near_radius2=5`.
- 041g exact-once class-partition oracle: `compact_oracle_rows()` (`sigma_class_m2l_census.jl:314-345`) on tiny cases (n=96/128) — `check_exact_once` cover-matrix + per-demoted-pair accepted/direct partition with zero omissions/duplicates; script exit code is the oracle pass.
- 041g census.csv columns include `demoted_body_pairs, reclaimed_body_pairs, ..., sigma_min, sigma_max`; rotor rows have `demoted_body_pairs = 0`.
- Production split veto: `veto_active = p.split_veto && tree.sigma_armed`; refuses a split when `gate_gmin * delta_child < rho_t * smax` with `smax` the on-the-fly subtree max of caller-supplied per-body sigma; `gate_gmin = _ball_stencil_min_gap(near_radius2)`. `sigma_armed = (sigma !== nothing && rho_t > 0)` (`src/tree_batched.jl:710`).
- Production entry points for Part B (host only): `AdaptiveRadixTree(systems; policy, sigma=σvec, TF=Float64)` (`src/tree_batched.jl:658`), `AdaptiveInteractionLists(tree)` + `build_adaptive_interaction_lists!(lists, tree)` (`src/interaction_list_batched.jl:1085, 1142`). U body-pair totals: sum `pop(a)·pop(b)` over `L.u_targets/u_sources` (pattern: `work_counts` in `adaptive_octree_verify.jl:658-666`). Demotion counter `L.n_dem`.
- Supported near radii: q=5 and q=12 both legal (`src/containers.jl:339`).
- `rho_t` provenance: 4.789 = 031a §4 per-pair worst-case radius (RegularizedVortex default = shipped adaptive nearfield); 4.252 = §6.4 RMS J radius (PartitionedVortex/041g).

## Implementation

All new/edited files under `MATRIX_OPERATOR_REFACTOR/`; output dir `data/sigma_closure_census/`.

### Step 1 — Parameterize the 041g census script (no behavior change)

Edit `scripts/sigma_class_m2l_census.jl` minimally: thread `q` (near_radius2), `rho_t`, and the two `ell` caps through as keyword arguments on `write_case!`, the oracle, and the internal `build_tree`/`build_lists` call sites, **defaulting to the current hardcoded values** so an argument-free run reproduces 041g byte-for-byte. Do not change `main()` behavior. (Task file authorizes "Part A reuses `scripts/sigma_class_m2l_census.jl` with parameter overrides".)

### Step 2 — Part A driver: `scripts/fm041i_sigma_closure_census.jl` (new)

- Includes `sigma_class_m2l_census.jl` (which includes `adaptive_octree_verify.jl`) without running its `main()` (guard: give the 041g script a `abspath(PROGRAM_FILE) == @__FILE__` main guard in Step 1 if it doesn't have one).
- Cases:
  - **rotor n=1e5, 1e6**: load from `data/rotor_wake/rotor_snapshot_n$(n).bin` (checksummed provenance per task file). First verify the snapshot equals the 041g regeneration (`rotor_arrays(n)`) — positions/sigma exact or hash-equal; record the check in the report. If they differ, run the q=12 control on the 041g regeneration (bit-for-bit gate) and the q=5 rows on both, documenting.
  - **cube and wake at 041a registered counts (n=1e5, 1e6)**: replicate `make_unitcube(seed 39101)` / `make_wake(seed 39102)` generators (copy the registered definitions from `fm041e_target_owned_bench.jl:108-118`, citing provenance), with their per-body sigma.
- For each case run two operating points: **shipped** `(q=5, rho_t=4.789)` and **control** `(q=12, rho_t=4.252)`; K_max = 041a winners (rotor 64, cube 64, wake 256; 041g rotor used K=128 — run the q=12 rotor control at K=128 to reproduce the 041g zeros exactly, and add K=64 rows at both points for the shipped-config statement).
- Record per row (extend beyond census.csv as needed, computed from `build_lists` output): `demoted_body_pairs`, demoted fraction of total direct body-pair work (`u_pairs`), demotion counts by tree level, plus sigma stats. Write `data/sigma_closure_census/partA_census.csv`.
- **Reproduction gate:** q=12 control rows for rotor must match `data/sigma_class_m2l/census.csv` zeros exactly (`demoted_body_pairs == 0` and matching sigma_min/sigma_max fields); mismatch ⇒ stop the row for a coordination fix.
- Run the 041g compact exact-once class-partition oracle at both operating points; write `partA_oracle.csv`; nonzero ⇒ fail.

### Step 3 — Part B driver: `scripts/fm041i_split_veto_census.jl` (new)

Read-only census against **production host** types (this is what ratification would enable; no `src/` edits — the disarm handle is the `split_veto` policy flag):

- Same case set as Part A (rotor 1e5/1e6, cube 1e5/1e6, wake 1e5/1e6), sigma supplied as the per-body vector.
- Per case build two `AdaptiveRadixTree`s with identical bodies/sigma/policy except `split_veto`: gate-on (`split_veto=true, rho_t=4.789`) vs gate-off (`split_veto=false`, sigma still armed so the demotion gate stays identical — matching the task's "sigma forced to 0/eps in the gate only" intent through the public API). Policy: `K_max` = 041a winner, `ell_max=10`, `near_radius2=5`, capacities per the `fm041e` `adaptive_policy` pattern (`node_capacity = 8*cld(n,K)+1024`, `u_capacity` generous — gate-on trees have fatter leaves, so oversize `ufac` and retry-on-throw with a larger factor).
- Report per case:
  - depth histogram and leaf-population histogram for both trees;
  - veto-refusal count by level — recomputed read-only: for each gate-on leaf with `pop > K_max && level < ell_max`, evaluate `gate_gmin·delta_child < rho_t·max(sigma[range])` to attribute it to the veto (assert it holds; `gate_gmin` recomputed in-script from the q=5 ball stencil min gap, cross-checked against `tree.gate_gmin` if accessible);
  - `u_pairs` totals (sum `pop·pop` over U list from `build_adaptive_interaction_lists!`) and the gate-on/gate-off delta; also `n_dem`, v/w/x route counts as context;
  - coverage check: both trees partition the identical body set exactly once (leaf ranges partition `1:n`; sorted permutations of both trees contain each body once; leaf key intervals disjoint).
- Write `data/sigma_closure_census/partB_veto_census.csv` (+ per-level histogram CSV).
- **Conditional H200 spot-check** (only if the veto fires AND u_pairs delta > 1% on any case): same-job A/B at shipped settings with 035 critical-path pricing, submitted to the cluster (rc.byu.edu conventions; local runs stay ≤4 threads). Expectation from Part B geometry: rotor ell=7 leaf width 0.0094 < 4.789·3.108e-3 = 0.0149, so the veto likely fires on the rotor — the delta size decides.

### Step 4 — Report and verdict: `data/sigma_closure_census/report.md`

- Tables for Parts A and B, the reproduction-gate check, oracle results, snapshot-provenance checks, and checksums file (`checksums.sha256`, matching the 041g convention).
- **Verdict** (signed, per the task's three options), with the ratified reframing:
  - shipped-defaults sigma cost = Part A demotion fraction (the veto is OFF in shipped config, so no forgone-depth cost exists in production today);
  - Part B delta = the price/benefit of ratifying `split_veto=true` at `rho_t=4.789`;
  - distinguish the cube/wake overlap-physics floor (`rho_t·sigma ≈ 9.6` spacings at beta=2) from the rotor regime — state per case, never pooled.

### Step 5 — Close out

- Update `041i-census-sigma-closure.md` status to Done with results summary and verification notes; tick the `Done` box in `START_HERE.md` row 041i (leave `Approved` for a clear-context agent).
- Offer (not write) a notebook entry per the notebook protocol.

## Files

| File | Action |
|---|---|
| `MATRIX_OPERATOR_REFACTOR/scripts/sigma_class_m2l_census.jl` | Parameterize q/rho_t/ell (defaults unchanged) + main guard |
| `MATRIX_OPERATOR_REFACTOR/scripts/fm041i_sigma_closure_census.jl` | New — Part A driver |
| `MATRIX_OPERATOR_REFACTOR/scripts/fm041i_split_veto_census.jl` | New — Part B driver |
| `MATRIX_OPERATOR_REFACTOR/data/sigma_closure_census/` | New — partA_census.csv, partA_oracle.csv, partB_veto_census.csv, partB_levels.csv, report.md, checksums.sha256 |
| `MATRIX_OPERATOR_REFACTOR/041i-census-sigma-closure.md`, `START_HERE.md` | Status updates at close |

No `src/` or FLOWVPM changes. No production defaults touched.

## Verification

1. Argument-free `sigma_class_m2l_census.jl` still passes its own oracle exit (defaults untouched); no re-run of the full 041g suite needed — spot-check one rotor row reproduces census.csv values.
2. Part A: q=12 rotor control reproduces `data/sigma_class_m2l/census.csv` zero-demotion rows (hard gate); compact exact-once oracle exit 0 at both operating points.
3. Part B: coverage checks pass on every case (identical body set, exactly-once leaf partition, both trees); veto-attribution assertion holds for every over-populated gate-on leaf.
4. All local runs at ≤4 threads (`Threads.nthreads() <= 4` guard in both drivers).
5. If the H200 spot-check triggers: same-job anchors and error bars in the timing CSV, 035 critical-path pricing.

## Runtime estimate

Moderate: six census cases × two operating points, pure-Julia reference machinery at n up to 1e6 (041g already ran rotor n=1e6 through the same machinery locally). Part B uses production host builds (fast). H200 job only conditionally.
