# 052c plan review — proposed improvements (2026-08-26, NOT YET APPLIED)

Status update (Ryan rulings, 2026-08-26 in-session): **I2 APPROVED**,
**I4 APPROVED** (recenter! events counted by the running timer probe,
job 13494373), **I5 APPROVED**, **I6 PENDING** (Ryan asked for the
rationale before ruling). I1's io half was done earlier as P1.5; its
monitors half remains unruled. I3 was approved earlier (see plan).

Status: review only. Nothing below has been implemented; each item awaits
Ryan's approval. Evidence re-derived 2026-08-26 from the full 473-step
`data/052c_phase1/` telemetry (the plan's table used 398 steps) plus a
source exploration of FLOWPanel.jl (file:line cited per item).

## Summary of the budget problem the plan misses

Refit of all 473 steps (script:
`data/052c_phase1/052c_review_fits.py`, power-law fit t = a·N^b per
category):

| category | b (refit) | plan's b | s/step @230k |
|---|---|---|---|
| wake_sfs | 1.59 | 1.48 | 20.8 |
| g_influence_pass1 | 1.69 | 1.58 | 11.5 |
| io | 0.52 | — | 2.1 |
| monitors | 0.72 | — | 1.1 |
| body_influence (pass3) | 0.30 | "flat" | 2.6 |
| solve | 0.02 | flat | 0.8 |

Categories P2 does **not** restructure sum to **~6.8 s/step at N=230k**
(io 2.1 + monitors 1.1 + pass3 2.6 + solve 0.8 + misc 0.2), leaving only
**~3.2 s** of the 10 s target for the GPU FMM (UJ+SFS, incl. per-step tree
build) plus the retained linear direct terms. The plan's stated ceiling
("~3.5 s from body_influence+solve") ignores io+monitors entirely. With
item I1 below, the floor drops to ~4 s and the FMM gets ~6 s — comfortable.

## I1 — add an explicit workstream for io+monitors (~3.2 s/step, growing) [HIGH]

The plan never addresses io/monitors. Root causes found in source:

- **io (2.1 s @230k, N^0.52, every step, no interval knob):** the `:io`
  block (`FLOWPanel_simulate.jl:1332-1388`) writes per-step body VTKs,
  panel-wake VTK, a metadata TOML append, and — dominant — the **full
  particle cloud as .vtp with 9 point-data arrays** for all np particles
  (`FLOWPanel_wake.jl:2255-2277`). On a device-backed pfield this also
  forces a full D2H mirror per step (`_wake_monitor_host_pfield`,
  `FLOWPanel_wake.jl:2258`) — i.e. after P2 the cost gets *worse*, not
  better. The only knob is all-or-nothing `SAVE_VTK`
  (`rotor_hover_pressure_comparison.jl:25`); there is **no
  nsteps_save/interval option anywhere** (grep verified).
  - **Measured (2026-08-26, local M-series, exact write shape at 230k
    particles, host arrays only):** `compress=true` (current default)
    = 1.79–1.90 s → 52.4 MB; `compress=false` = 0.17–0.24 s → 57.0 MB.
    The zlib pass costs ~10× the write for ~8% size (float64 particle
    data barely compresses) and accounts for essentially the whole 2.1 s
    io category. Both modes are binary (`format="appended"`, raw
    little-endian block; verified in header) — no ASCII fallback.
  - **APPROVED (Ryan 2026-08-26): fix = (1) `compress=false` +
    (2) Float32 output arrays** (WriteVTK emits `type="Float32"` when
    handed Float32 data; convert views at the write site,
    `FLOWPanel_wake.jl:2264-2279`). Expected io: ~2.0 → ~0.1 s/step;
    disk ~28 MB/step. Output-only → outside the science fingerprint; no
    reference regeneration needed.
  - **REJECTED (Ryan 2026-08-26): save-interval knob** — the VTK series
    generates visuals that need full temporal resolution. Consequence:
    the per-step D2H mirror refresh remains once the pfield is
    device-resident (P2). Mirror mechanics (FLOWPanel_gpu_wake.jl):
    host mirror is a full `ParticleField` at `maxparticles` capacity,
    allocated once and cached in the global `_GPU_PFIELD_MIRRORS` Dict
    keyed by `objectid(pfield)` (`:42-58`, cleared on wake rebuild);
    each refresh is one contiguous `copyto!` of the live rows×np prefix
    (fast CuArray→Array memcpy path, `:74-88`), ~67 MB / ~15-25 ms at
    production shape per the file header. **No within-step memoization:**
    wake-health (`FLOWPanel_simulate_monitors.jl:3653`), attribution
    (`:3798`), inventory (`:3915`), and the VTK writer (2263) each
    re-sync, so ~4 redundant D2H transfers/step (~60-100 ms).
    **DEFERRED (Ryan 2026-08-26): address the redundant transfers later**
    — candidate fix: memoize the sync on `pfield.nt` (mind the
    maintenance seam's mid-step write-back ordering). Not part of the
    initial I1 change.
  - **Float32-for-free note (Ryan 2026-08-26):** if P2 lands the
    device-resident pfield (and hence its host mirror) as Float32
    end-to-end, the write site already hands Float32 views to WriteVTK
    and the ~2× write/size win comes with **zero conversion cost** —
    no `Float32.(...)` copies needed. Keep that possibility open when
    choosing P2 precision (048 measured F32 `e_delivered ~ 1e-3` vs
    F64 `~5e-4`; precision choice IS inside the science fingerprint,
    unlike the output-only conversion). If the run stays F64, do the
    conversion at the write site as planned.
- **monitors (1.1 s @230k, N^0.72, every step, no frequency gate):**
  `_run_monitor!` has no decimation (`FLOWPanel_simulate.jl:1316-1328`,
  `FLOWPanel_simulate_monitors.jl:85-92`). WakeHealthMonitor does one
  O(np) scan **plus an optional second O(np) pass + sort** for
  attribution (`FLOWPanel_simulate_monitors.jl:3652-3706, 3903-3921`);
  WakeInventoryMonitor is another O(np) binning scan with sort-based
  per-cell quantiles (`:3912-3934`); each monitor open/append/closes its
  CSV every step (`:110-119`).
  - Fix: keep the cheap scalar wake-health scan per-step (gates depend on
    the per-step CSV), but decimate attribution + inventory (every k
    steps), and/or run the scans on-device post-P2 (data is already on
    GPU; the per-step D2H mirror disappears too). Caution: confirm which
    monitor columns the gate comparisons consume before decimating.

## I2 — budget at true peak N, not 230k; refresh the plan's exponents [MED]

N was still climbing at the last measured step: 242,348 at step 454,
growing ~126/step. The CPU reference window (steps 720–755) sits at
~209k, so **N peaks somewhere in steps 455–720 at >242k — the peak was
never observed**. The plan's "plateau ≤10 s/step at N≈230k" target and
P3's "~230k snapshot" both use a number below the peak; with wake terms
scaling at N^1.6–1.7 the peak is what the 4 h/12 h wall-time math must
survive.
- Estimate peak N first (cheap: the CPU reference run's full wake-health
  series if archived, or a shedding-minus-clipping model), and take the
  P3 tuning snapshot from a near-peak step, not merely a "late" one.
- Update the plan's Context table to the 473-step fits (above) — the
  quadratic terms are steeper than recorded.

## I3 — P2 should reuse the existing split-call pattern in `_sa_wake_influence!` [MED]

The 2 mixed-target `influence_pass1` calls come from the default branch
of `_sa_wake_influence!` (`FLOWPanel_simulate.jl:746-837`; default flags
at `rotor_hover_pressure_comparison.jl:433-446`): both calls pass the
**full combined targets tuple including the particle field**
(`:764-770` panel-wake-row source; `:774-780` particle-field source).
The particle-target/small-fixed-target **split P2 needs already exists**
in the non-default `!particle_hessian_self` branch (`:782-796, 810-825`),
and `_gpu_rect_influence!` already loops per target system
(`FLOWPanel_gpu_influence.jl:619-650`). So P2's routing change is:
particle-field→particles goes to the radix FMM (targets===sources holds);
the remaining dense calls keep only the small fixed targets — a refactor
following an existing in-repo pattern, not new plumbing. Worth stating in
P2 to scope it correctly.

## I4 — deprioritize in-place recenter! pending one measurement [LOW]

From measured count growth (~126/step late, 5% padding at 242k ⇒ escape
every ~96 steps) the exception-triggered rebuild (~0.95 s/call,
FUTURE_IMPROVEMENTS.local.md) amortizes to ~0.01 s/step — negligible.
Caveat: *spatial* escape (wake convecting steadily downward) could
trigger far more often than count growth suggests. Recommendation: just
count recenter! events in the first 36-step mature-gate run; only pull in
the in-place design if cadence is ≳1/10 steps. Saves P2 scope.

## I5 — make the acceptance restartable and add a peak-N probe gate [MED]

P4 validates only with the 36-step mature gate (steps 720–755, ~209k) —
it never exercises the peak-N region (455–720) where cost is highest —
then goes straight to a 12 h monolithic 1080-step attempt.
- 052b Phase A.1 already verified shared-field save/restart: give the
  stage-d driver periodic checkpoints so a failed 12 h attempt resumes
  instead of restarting (also converts the 3-day MaxWall into usable
  headroom).
- Before the full attempt, run a short probe (~20–30 steps) restarted
  near peak N to confirm the post-P2 s/step at the *worst* N, not just
  the mature window.

## I6 — start P3 now, in parallel; regenerate the reference exactly once [LOW]

P3 needs no GPU and nothing from P2 — it can run locally today, off the
critical path. Sequencing note worth making explicit in the plan: the
retuned triple changes arithmetic for whatever remains CPU-side and the
near-field structure of the GPU FMM, so the pinned CPU reference should
be regenerated **once**, after both the P2 arithmetic changes and the P3
triple are final (regenerating after P2 and again after P3 wastes a
multi-hour deterministic CPU run). P3's direct_list pair-count sweep
should use the near-peak snapshot from I2.

## Sanity check of the plan's own numbers (no change needed)

- Timer coverage: confirmed, total_step − Σ categories ≈ 0.
- "body_influence+solve ceiling ~3.5 s": matches refit (2.6+0.8);
  note pass3 grows slowly (b≈0.30, panels→particles part scales with N),
  so AIC caching (panels→panels only) won't shrink it much — the ceiling
  is real but it is a floor, too.
- Fused nearfield ≥400k: peak N ~250–260k ⇒ correctly not planned.
