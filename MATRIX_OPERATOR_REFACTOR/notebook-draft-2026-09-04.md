# Notebook draft 2026-09-04 (HELD FOR RYAN'S APPROVAL — not written to journal)

Insertion target: `~/Dropbox/research/notebooks/journals/20260901.md`, append
at end of file (line 63; no `# 20260904` header exists yet; no overflow).

Drafter's open questions for Ryan: (1) verbosity per topic (draft is at full
setup+result+trace+conclusions level; trim any section to bullets?); (2) keep
the 052c trial-1d/1e note as its own checklist item or fold to a one-line
aside (status, not finalized result)?; (3) should the "open Ryan rulings"
become explicit sub-tasks instead of prose asides?

---

# 20260904

- [ ] Diagnose 052 phase 2r block-GS residual plateau: test the "FMM accuracy floor" hypothesis
- [ ] Assess 052 phase 1r IGE GPU accept-run (13568974) convergence over the hover window
- [ ] Bisect the FLOWPanel kutta `:jump` bitwise regression to a specific commit
- [ ] Track 052c trial-1d/1e sigma_guard `:ceil` port resubmission after node failure

## 2r block-GS residual plateau: FMM accuracy floor hypothesis refuted

**Setup.** Diagnostic job `gsdiag4` (Slurm 13582074, mgh GH200) re-ran the
`p022g_2r_ige` smoke solve with body FMM expansion order 20 / multipole
acceptance 0.5, to compare against the prior `gsdiag2` (Slurm 13568975) run
at expansion order 17 / acceptance 0.7. The question: does tightening FMM
accuracy move the block Gauss-Seidel residual plateau observed in earlier
2r runs?

**Result.** The normalized block residual trajectory is bit-identical to 16
digits between the two runs: plateau at 5.8224e-4 from outer iteration 2
onward, with the strength delta contracting geometrically to machine
epsilon by iteration ~6, then noise at ~2e-15 through iteration 30.

**Root cause (code trace, FLOWPanel.jl).** `influence!(..., ::FastMultipoleBackend)`
(`src/FLOWPanel_fmm.jl:60-79`) is intercepted by `_gpu_rect_influence!`
(`src/FLOWPanel_gpu_influence.jl:618-650`) before the backend's
`expansion_order`/`multipole_acceptance` fields are ever read. The carrier
exports `FLOWPANEL_GPU_INFLUENCE=cuda`, and every Gauss-Seidel
cross-influence/residual call passes a `production_route`, so all
cross-influence runs through the exact dense Float64 CUDA batch
(`_gpu_direct_batch!`), never `FastMultipole.fmm!`. Changing the FMM
expansion order/acceptance therefore cannot touch this code path at all.

**Conclusions.**
- The 5.8224e-4 plateau is not body-FMM truncation error — cross-influence
  on this route is exact dense, not FMM-approximated.
- "Pay for more FMM accuracy" is not a viable remedy on the GPU route.
- Since the GS fixed point is exact (dense), the pinned residual measures a
  ~5.8e-4 inconsistency between the residual's assembled operator and the
  operator implied by the block solves. Root cause is unidentified;
  candidates: wake-route contributions, Kutta/damping rows, or the
  residual-assembly formulation itself.
- `GS_TOL=1e-8` is unattainable on this route regardless of FMM settings.

**Open Ryan rulings:** (a) raise `GS_TOL` above the floor, vs (b) gate
convergence on strength delta instead of residual, vs (c) fund an
operator-mismatch investigation to find the ~5.8e-4 discrepancy.

## 1r IGE GPU accept run: completed healthy, not converged even over hover-only window

**Setup.** Job 13568974 ran all 413/413 steps (wall time 2:12:24, per-step
times 14-19.5 s) and failed only gate checks (no crash). Offline recompute
from `data/p022g_1r_ige/p022g_1r_ige_CT_per_rev.csv` (11 complete
revolutions, NT=36, RPM 6000). Hover begins at revolution 6.5 (freestream
ramp 1.0 rev + hold 1.5 revs + withdraw 4.0 revs).

**Hover-window per-rev thrust coefficient ($C_T$, thrust convention):**

| Rev | Mean $C_T$ |
|---|---|
| 7 | 0.0897 |
| 8 | 0.0920 |
| 9 | 0.0888 |
| 10 | 0.0801 |

Window mean $C_T = 0.0877$, cycle std 5.9%.

**Phase-2e convergence criterion, evaluated:**

| Metric | Value | Tolerance | Result |
|---|---:|---:|---|
| Max per-rev mean spread | 0.086 | 0.005 | FAIL |
| Max within-rev peak-to-peak / mean | 0.47 | 0.02 | FAIL |

$C_T$ is still drifting over the hover window — could be a slow transient or
the onset of a limit cycle; with only 4.5 hover revs available the two
cannot be distinguished.

**Confound in the original verdict.** The in-job `CONVERGED=false` result
was additionally confounded: the run used `CONVERGENCE_REVS=10` on an
11-rev run, so the readout window started at revolution 1.0 (still in the
ramp/withdraw transient), and the driver's `window_in_hover` guard forced
failure regardless of spread — independent of the genuine non-convergence
found above.

**Gate note.** The 7200 s elapsed gate is hardcoded at
`examples/run_rotor_multi_ground_effect_gpu.slurm.sh:203`; actual elapsed
was 7917 s.

**Open Ryan rulings:** gate policy (raise/relax the 7200 s hardcoded
limit), and whether to extend hover revolutions vs adopt a cycle-mean
acceptance criterion instead of per-rev spread.

## FLOWPanel kutta `:jump` bitwise regression: pinned to commit 7fbd68a

**Method.** Worktree bisect with FastMultipole and FLOWVPM dev-pathed to
the current local trees on both sides of the bisect (FMM/VPM held
constant, isolating the FLOWPanel side).

| Commit | `test/runtests_unit_kutta.jl` result | `:jump` fallback testset |
|---|---|---|
| 8b07f96 (parent) | 658/658 green | 6/6 green |
| 7fbd68a | 656/658 | same 2 failures as HEAD |

Failures are at `runtests_unit_kutta.jl:539-540` — body and wake strength
equality vs. the legacy A/jump trajectory.

**Quantitative signature.** The `:jump` fallback's strengths come out
approximately half the legacy values:

| Quantity | Fallback (7fbd68a/HEAD) | Legacy expected |
|---|---:|---:|
| Body strength (first entry) | -0.111 | -0.571 |
| Wake strength (first entry) | -0.313 | -0.645 |

This ratio pattern (not noise-like) suggests a factor-of-2 or
double-counted/halved influence introduced by 7fbd68a's Dirichlet
self-potential / wake-row convention change, rather than a numerical
sensitivity issue.

**Open Ryan ruling:** update `_kutta_trial!` to match the new convention,
vs. relax the bitwise-equality contract to a tolerance (a canonicality
decision, not just a bugfix).

## 052c sigma_guard `:ceil` port — trial-1d node failure, resubmitted as trial-1e

Trial-1d (Slurm 13582076) validated the sigma_guard `:ceil` port, running
clean to step 693/1079, but then died to a genuine node failure (mgh-1-2
taken to maintenance) rather than a code or logic fault. Resubmitted as
trial-1e (Slurm 13582234), which **COMPLETED (exit 0, 2:58:15) — 052c
trial 1 PASSES**: "artifact and monitor gates passed for indices 0:1079".

**Locked correctness gates (window 720-755):**

| Gate | Measured | Ceiling | Result |
|---|---:|---:|---|
| CT cycle-mean | 6.565e-4 | 1.800e-3 | PASS |
| Gamma M2 max | 1.317e-3 | 2.934e-3 | PASS |
| Gamma M2 RMS | 4.484e-4 | 1.498e-3 | PASS |

Guard config: dt·Z cap 0.5, sigma floor 1% of shed sigma (4.451e-5 m),
ceil Inf. $\sigma_{min}$ contracted monotonically from the shed value
4.451e-3 m to 9.558e-5 m at step 983 (ratio 0.0215 — the floor never
clamped), recovering to 1.303e-4 m by step 1080; the historical
step-~1015 collapse ($dt \cdot Z$ reaching 164) did not recur. Bonus
validation: mid-run the radix-FMM geometry went inadequate and the 052f
demotion fired (fallback to all-direct zero-M2L at $\ell = 2$) — the
exact scenario that hard-crashed the original run 2 at step 894 — and the
run continued healthy. Phase-2e CT convergence printed CONVERGED=false
(known non-fatal readout item; cycle-mean $C_T = 0.072526 \pm 1.04\%$
over 10 revs). Commit-plan proposal recorded in the 052c ledger (needs
Ryan): upstream the ceil port; adopt dt·Z cap 0.5 + floor 1% as the GPU
rotor-acceptance default; fold remaining OFAT candidates into 053.
