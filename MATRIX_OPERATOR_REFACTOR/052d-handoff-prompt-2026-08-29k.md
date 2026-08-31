# 052d handoff (session 2026-08-29k): oracles green after ulp fix; relU attribution job 13509236 IN FLIGHT

## Prompt for the next agent

You are picking up task 052d mid-Step-5. Job 13507743's failures are fully
root-caused and fixed (do NOT redo the analysis or the fixes below). Your job:
(1) collect results of combined Slurm job **13509236** (`fp052d5c`, partition
`eng`, qos `eng`, H200, ~2h limit) on ORC, (2) rsync its state dumps back and
run the local CPU attribution script `p39_relU_attribution.jl`, (3) report the
attribution table, (4) offer a notebook entry. Read memory `orc-cluster-access.md`
before touching ssh (bash -lc for slurm, banner noise, `rsync --checksum`,
macOS rsync has no `--info` — use `-i`).

Check the job:
```
ssh orc 'bash -lc "source /etc/profile; sacct -j 13509236 -X -n -o State,Elapsed"'
ssh orc 'bash -lc "grep -E \"ORACLES|PASS|FAIL|dumps present|complete|exit code\" FastMultipole-052-h200/fp052d5c-13509236.out | tail -25"'
```
A background watcher may already have fired a task notification with this.

## What job 13509236 runs (script: MATRIX_OPERATOR_REFACTOR/scripts/fp052d_step5c_recert_dump.sh, staged on ORC)

1. All five oracles p34–p38. In job 13508865 (after the fixes) they were
   p34 24/24, p35 9/9, p36 4/4, p38 6/6 PASS; p37 case 1 (real snapshot)
   PASS at 1.8e-15 but case 2 FAILed on a stale host-ref tag-3 gate —
   IDENTICAL to p36's, fixed the same way (see below) and staged. Expect ALL
   GREEN now; the script prints "ALL ORACLES GREEN".
2. xverify run (production rotor, device cross pass + host-fmm! reference each
   step) with the NEW state-dump hook: at np thresholds 3500/12500/28000 it
   writes to `$HOME/FastMultipole-052-h200/relU_dumps_13509236/` per dump:
   `dump_np<NP>_{positions,hostU,deviceU}_3xN_f64.bin`,
   `dump_np<NP>_body1_{srcmat_17xS,wakemat_8xS,cent_3xS}_f64.bin`,
   `dump_np<NP>_meta.txt`. Bounded by `timeout 60m` — **exit code 124 with
   "dumps present: 3 / 3" is SUCCESS**, not a failure.

## Then: local attribution (Task B of the approved plan)

```
rsync --checksum -avi orc:FastMultipole-052-h200/relU_dumps_13509236/ \
  <scratchpad>/relU_dumps_gpu/
cd MATRIX_OPERATOR_REFACTOR/prototypes/052d_cross_stencil
JULIA_NUM_THREADS=4 DUMPDIR=<scratchpad>/relU_dumps_gpu \
  julia --project=/Users/ryan/Dropbox/research/projects/FLOWPanel.jl p39_relU_attribution.jl
```
(Local runs: max 4 threads, house rule.) p39 computes a CPU dense reference
(exact; p38-recipe `FM._rect_panel_pair` over body columns + reconstructed
TE-wake arm columns) on each dumped state and prints relL2 of
{deviceU, hostU, host CrossStencil prototype} vs dense, plus dev_vs_hfmm
(must reproduce the job's printed relU at that np — consistency check) and
proto_vs_dev (prototype≡device parity on real states). Sample: all particles
at np<=5000, else 5000 (seed 39). Before reporting, re-run via the `verifier`
agent per the user's review-before-reporting rule.

**Interpretation** (the whole point — user explicitly rejected trajectory-replay
attribution as confounded; these dumps are from the ACTUAL GPU trajectory, both
routes saw identical inputs): whichever of dev_vs_dns / hfmm_vs_dns carries the
~9e-4 seen at small np owns the error. hostfmm owns it → xverify's 1e-4 gate was
measuring the reference's truncation; propose gating via CPU dense spot-checks
instead (user's stated preference: "give the GPU as much time as needed, spot
check specific timesteps on the cpu"). device owns it → real small-np accuracy
bug in the cross stencil; investigate before any full sim. Background expected
values: certified device-vs-dense at np=242k (step 472) is 7.2e-6 (p32g);
cluster relU was 9.2e-4 worst at np≈3.9k falling to ~1e-4 by np≈200k.

## What landed THIS session (2026-08-29, all uncommitted — do not redo)

FastMultipole (branch `flowpanel-20260817`):
- `src/translate_batched_cuda.jl` (~line 144): `_cuda_radix_keys_checked_kernel!`
  tolerates 4-ulp-per-axis overshoot (`tol = 4*eps(max(|lo|,|hi|))`) before
  raising oob. ROOT CAUSE of job 13507743's p34 19-FAIL cascade / p36 case-1
  zero routes / p37 grid-dim-0 crash: the tight box `x_min = center - h0` left
  the max-x snapshot particle 1.4e-17 above `x_min + 2h0` (verified numerically
  on the snapshot bins); the A2 needs_rebuild path then zeroed all lists.
  VALIDATED: job 13508865 → p34 24/24 PASS with exact route parity.
- `src/cross_stencil_cuda.jl:997` (`finish_cross_locals!`): `max_per_level > 0
  || return ls` guard (zero-dim launch protection).
- `p36_stageC_oracle.jl` AND `p37_stageD_oracle.jl` host refs: gate
  `(1 <= tag <= 5) && nv >= 3`; `tag == 2 || tag == 3` → `Panel{Dipole}(s1)`
  (tag 3 = closed vortex ring ≡ dipole s1; device includes it post-A1, old refs
  skipped it → p36 was validated green in 13508865; p37 fix staged for 13509236).
- NEW `p39_relU_attribution.jl` (see above).
- NEW scripts `fp052d_step5b_oracles_only.sh` (job 13508865, done) and
  `fp052d_step5c_recert_dump.sh` (job 13509236).

FLOWPanel (`/Users/ryan/Dropbox/research/projects/FLOWPanel.jl`, branch
`fastmultipole`), `src/FLOWPanel_gpu_influence.jl` (staged on ORC):
- `_panel_fmm_maybe_dump(out, tgt, fmm_bodies, np; deviceU=nothing)` +
  `_PANEL_FMM_DUMPED` (near `_cross_root_box`): env-gated
  (`PANEL_FMM_DUMP_DIR`, `PANEL_FMM_DUMP_NP=n1,n2,...`) state dump — packs
  srcmat/wakemat/cent via `_cross_fill_inputs!` on a throwaway `_CrossPassEntry`.
  Called from (a) the host-target fmm leg (~line 754) and (b) the device
  xverify block in `_panel_cross_device!` with `deviceU=xver`.

## Session history / dead ends (context)

- Job 13507743 (fp052d5): p35+p38 PASS, rest failed (see root causes above);
  xverify TIMED OUT at np≈208.7k of 242k; relU peaked 9.2e-4 (np≈3.5–4.2k),
  plateaued 5e-5–1.5e-4 late. Gate line never printed. Its VTK dirs on ORC are
  EMPTY (timeout killed before flush) — no restart states there.
- Job 13508865 (fp052d5b, oracles only): validated ulp+p36 fixes; p37 case-2
  stale-ref FAIL → fixed, recert in 13509236.
- A local CPU trajectory-replay attribution run was built, run, and KILLED:
  the user correctly identified replay divergence as a confound; superseded by
  the GPU-state dumps. Leftover: scratchpad `dump_run*.log`, empty
  `relU_dumps/` dir — ignore. The host-path dump hook remains in the seam
  (harmless, env-gated).
- Background watcher tasks keep getting killed by the harness — always verify
  job/process state directly instead of trusting a dead watcher.

## Standing constraints

- Plan file: `~/.claude/plans/task-notification-task-id-b934sl05b-tas-jaunty-stonebraker.md`.
- Do NOT submit further GPU jobs beyond 13509236 without asking. The xverify
  gate decision and any full-sim job sizing are explicitly deferred to
  discussion with the user after attribution.
- If 13509236's xverify stage produced <3 dumps (e.g. 60m timeout before
  np=28000): run p39 on whatever dumps exist (it auto-discovers), report,
  and ask before any follow-up job.
- Nothing is committed anywhere; the user decides when/what to commit.
- Notebook: offer a notebook-drafter entry (review fixes, 13507743 triage +
  ulp root cause, oracle green results, attribution) — ask how much detail;
  NEVER write to the notebook without approval.
