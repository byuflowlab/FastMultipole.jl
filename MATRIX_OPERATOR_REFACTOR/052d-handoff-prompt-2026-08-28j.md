# 052d handoff (session 2026-08-28j): Step-4 review fixes LANDED; Step-5 H200 job 13507743 IN FLIGHT

## Prompt for the next agent

You are picking up task 052d mid-Step-5. The Step-4 external-review fix plan
(`/Users/ryan/.claude/plans/review-work-done-on-cryptic-rossum.md`) is **fully
implemented and CPU-verified**; do NOT redo any of it. Your job is to (1)
collect and report the results of Slurm job **13507743** on the ORC cluster,
(2) triage any failures, and (3) offer the user a notebook entry draft.

## What already landed this session (all local edits, uncommitted, branch `flowpanel-20260817`)

FastMultipole (`/Users/ryan/Dropbox/research/projects/FastMultipole`):

- `src/cross_stencil_cuda.jl` — A1: Stage-B kernel accepts tag 3 (nv >= 3),
  dipole arm strength s1 via new `_cross_b2m_arms` helper; verified against
  the pure-VortexRing host `body_to_multipole!` overload
  (FLOWPanel_liftingbody.jl:804-910 — which DOES carry a TE-wake arm, strength
  s1, so the existing wake plumbing covers tag 3). Bonus fix: Stage-E near
  kernel read the wrong wake dipole strength for tag 3 (s2 = 0 → silent zero
  wake); now `tag == 2 || tag == 3 ? s1 : s2`. A2: `refresh_cross_producers!`
  never mutates x_min/h0 in place; on containment failure it sets
  `ctx.needs_rebuild` + `rebuild_x_min`/`rebuild_h0`, zeroes the lists, and
  returns (struct field `rebuilt_root_box` REPLACED by these).
  B1: `_cross_near_kernel!` gained `::Val{POT}`; `apply_cross_near!` kwarg
  `potential::Bool=true` (oracles unchanged; production seam passes false).
  Float64-only guard in `device_cross_expansion_state`.
- `src/cross_stencil_host.jl` — hosts `_cross_b2m_arms` (CPU-testable).
- `test/cross_stencil_test.jl` — new testsets "B2M tag arms" and "demotion
  census changes with h0" (R_guard derived from the tables' min box gap);
  file wired into `test/runtests.jl` (runs ~11 s). NOTE: a linter/user touched
  this file after my edits (`using FastMultipole.StaticArrays`); keep it.
- `MATRIX_OPERATOR_REFACTOR/prototypes/052d_cross_stencil/p35_stageB_oracle.jl`
  — tag-3 parity: host ref maps tag 3 → Panel{Dipole}(s1); case 2 cycles all
  five tags, expected_skipped now 1 (only the tag-3 nv=2 open filament).
  `p34_stageA_oracle.jl` — field rename `rebuilt_root_box` → `needs_rebuild`.
- `MATRIX_OPERATOR_REFACTOR/scripts/fp052d_step5_oracles_run.sh` — C2: awk
  enforces every `relU=` <= 1e-4 (NaN fails), numeric hit-count check.

FLOWPanel (`/Users/ryan/Dropbox/research/projects/FLOWPanel.jl`, branch
`fastmultipole`), `src/FLOWPanel_gpu_influence.jl`:

- A3: `_cross_config()` validates (2<=ell_x<=8, 1<=P<=8, q>=0, finite Rg>=0),
  stored on `_CrossPassEntry.cfg`, part of the cache key; np-only rebuild uses
  `e.cfg.P` (P-skew dead). Config logged once; validated up front in
  `_panel_cross_device!` so bad env fails loudly.
- A2 seam: on `ctx.needs_rebuild`, `_cross_run_body!` deletes the entry and
  reconstructs everything (fresh padded box → new tables/masks/operators),
  reruns producers, errors if still escaping.
- A1 seam: throws on nonzero `xs.n_skipped` after Stage B.
- A4: per-body work isolated in `_cross_run_body!`; the `.+=` into particle U
  is the LAST statement; loop wrapped — throw before any accumulation → clean
  host-fmm! fallback; after partial accumulation → explicit
  "partial write, field must not be trusted" error.
- B1: passes `potential=false`. B2: persistent device mirrors
  `d_srcmat/d_cent/d_wakemat` on the entry, refreshed with `copyto!` per step
  (pos_d still a per-call device temp — deliberate deviation, it never touches
  host).

## Verification already done (do not repeat)

- All edited files parse; script passes `bash -n`.
- `cross_stencil_test.jl`: 487,693/487,693 pass standalone AND inside the full
  suite. Full `Pkg.test()`: PASSED, zero failures
  (log: scratchpad `full_suite2.log`; run via
  `julia --project=. -e 'using Pkg; Pkg.test()'` — running runtests.jl
  directly fails on test-only dep ForwardDiff).
- FLOWPanel loads/precompiles; `_cross_config()` = (q=12, ell_x=5, P=6,
  R_guard=0.006).

## Cluster state (ORC, `ssh orc`, user rander39)

Read memory `orc-cluster-access.md` first (bash -lc for slurm, banner noise,
keyboard-interactive auth expiry — if ssh hangs/fails, have the user run
`! ssh orc echo ok`; rsync must use `--checksum`; NOTE macOS rsync has no
`--info`, use `-i`).

Already staged (rsync --checksum, verified):
- `~/FastMultipole-052-h200/`: full `src/`, updated tests, ALL of
  `MATRIX_OPERATOR_REFACTOR/prototypes/052d_cross_stencil/` AND
  `052d_shared_radix/` (the latter was missed at first — its absence broke
  `CrossStencil.jl`'s include and killed all five oracles in job 13507741,
  which was cancelled), hardened Step-5 script.
- `~/FLOWPanel-052-h200/`: full `src/` + `examples/rotor_hover_pressure_comparison.jl`.
- `~/snapshot472-052d/`: all snapshot472 .bin files + SNAPSHOT_INDEX.md
  (local source: scratchpad of session 1a9c539a…, path baked into oracle
  defaults; SNAPDIR overrides it).

**Job 13507743** (`fp052d5`, partition `eng`, `--qos=eng`, 3 h limit,
H200 on eng-1-1 expected): submitted ~20:12 on 2026-08-28 with
`SNAPDIR=$HOME/snapshot472-052d FP052D_FMDIR=$HOME/FastMultipole-052-h200
FP052D_FPDIR=$HOME/FLOWPanel-052-h200 FP052D_ENV=$HOME/fm052env-h200`.
Output: `~/FastMultipole-052-h200/fp052d5-13507743.out`; per-stage logs
`fp052d5_p3?_*_13507743.log` + `fp052d5_xverify_13507743.log` in the same dir.
A local background watcher may or may not still be alive across the context
reset — just check directly:

```
ssh orc 'bash -lc "source /etc/profile; sacct -j 13507743 -X -n -o State,Elapsed"'
ssh orc 'bash -lc "grep -E \"PASS|FAIL|relU|hits|complete\" FastMultipole-052-h200/fp052d5-13507743.out | tail -40"'
```

## Success criteria for job 13507743

- p34–p38 oracles all print PASS (p35 now includes tag-3 parity,
  expected_skipped: case1=0, case2=1, case3/3b=0).
- Stage-F XVERIFY smoke: `panel_cross_xverify ... relU=` lines every step,
  all <= 1e-4 (expected ~1e-5); script's own threshold gate prints
  "xverify relU threshold (1e-4) PASS"; final line
  "fp052d step-5 job complete: ALL STAGES PASS".
- Also expect the new one-time `panel_cross config: q=12 ell_x=5 P=6
  R_guard=0.006` line in the xverify log.

## Failure triage hints

- Oracle FAIL with include/SystemError → still a staging gap; find the missing
  file locally and rsync it (check `include(` lines in the failing prototype).
- p35 skipped-count mismatch → tag-3 gate/arm regression in
  `_cross_panel_b2m_kernel!` or the oracle's host ref.
- relU >> 1e-4 → real port bug; compare against per-stage logs; the certified
  reference operating point is p32g (7.2e-6 vs dense).
- Seam throw "Stage B skipped N panel(s)" during XVERIFY → some production
  panel hits the skip gate; inspect tags in the packed buffer.

## Remaining work after the job

1. Report results to the user (they authorized this one submission; do NOT
   submit further GPU jobs without asking).
2. If green: offer (don't write) a notebook entry via notebook-drafter
   covering the review triage + fixes + Step-5 results; ask how much detail.
3. Deferred items (recorded in the plan §D, NOT for this pass): Stage-B
   shared-memory reduction, Stage-C class-batched GEMM, capacity geometric
   resize, genuine Float32, objectid-Dict lifecycle/`clear_cross_pass_state!`
   teardown call, handoff-doc wording "code-complete, GPU validation pending".
4. Nothing is committed; the user decides when/what to commit.

## Addendum 2026-08-29 (from the σ/VPM illustration session)

New evidence file: `052-gpu-memory-growth-longrun-2026-08-29.md` — a 40-rev
GH200 shedding run died at step 819/1475 on retained per-step device
allocations (~35 MB/step, pool_used 15.5→31.9 GB) while physically healthy.
If the staged memory-allocation optimization touches pool/caching, use that
run (`scr_p019_s038v_gpu40`, wrapper+arm already in the gh200 silo) as the
regression test.
