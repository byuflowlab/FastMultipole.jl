# 028 Phase A.2 — Pilot H200 job

Submit a short pilot before the full sweep. Purpose: validate at n=1e6 what has
never been constructed hierarchically —

- capacity sizing (route/direct capacity math at
  `src/translate_batched_resident.jl:852-861`; hier persistent memory was only
  1.6-1.8 GB at n=2e5/ell=6-7 so 140 GB should be ample, but n=1e6 construction is
  unverified),
- the device convection step + counter contract on the verdict boundary,
- sampled-direct error plumbing at n=1e6,
- rough construct time (flat was ~28-31 s/case → budget the sweep).

Pilot matrix: n=1e6, F64, {hier12 dense ell=5 K=256} + {flat dense ell=4} (the
024b baseline row, as a cross-check that the harness reproduces ~0.42 s), a few
convection steps each. Short `--time` (2h).

Fetch, sanity-check the CSV (counters flat, error within 10x of 1.19e-4 for the
flat row, no OOM), then adjust the sweep matrix (`04-phase-sweep.md`) with what
the pilot reveals (e.g. best ell bracket, per-case runtime).

## Progress

- [x] Pilot submitted (job id: **12996475**, 2026-07-31, mode=pilot: hier12 dense ell=5 K=256 n=1e6 F64 + flat dense ell=4 cross-check, STEPS=3)
- [x] Fetched + sanity-checked — **both cases fit=true**, all gates pass
- [x] Sweep matrix adjusted (tiered; see below and `04-phase-sweep.md`)

### Pilot results (`MOR/data/feasibility_1m_10ms/cuda_m13h-1-1_2026073{1-082602,1-082722}.csv`)

| case | eval | verdict | host | grad rel RMS | construct | peak dev |
|---|---|---|---|---|---|---|
| hier12 dense ell=5 K=256 | 113.49 | **127.21** | 246.83 | 3.185e-4 | 19.27 s | 1.65 GB |
| flat dense ell=4 (024b) | 349.83 | 363.50 | 482.75 | 1.187e-4 | 19.87 s | 1.99 GB |

Sanity checks, all green:
- **Counter contract**: `body_uploads=0`, `metadata_downloads=0`, `expansion_host_copies=0`;
  route/operator uploads flat across recurring steps (the harness `error()`s otherwise, and
  `fit=true`).
- **024b cross-check**: flat grad rel RMS **1.187295e-4** vs the 024b record 1.1873e-4 — exact
  to 5 digits, so the harness reproduces the 024b flat configuration. `host_step_ms` 482.7 vs
  the recorded 424.6 ms is +13.7%, explained by median (here) vs `step_min` (024b); not a defect.
- **Error gate**: gate is ≤10x of 1.19e-4 = 1.19e-3. flat 1.187e-4 ✓, hier12 3.185e-4 ✓
  (2.7x the P=4 truncation error, comfortably inside).
- **Methodology guard**: `ref_cross_check_grad_rel = 2.76e-14` — the on-device direct kernel
  agrees with the checksummed 024b CSV to machine precision.
- **Convection**: after 3 Euler steps grad rel RMS 3.1842e-4 vs step-0 3.1852e-4 — field stays
  accurate; host path agrees with device to 13 digits (`host_err_gradient_rel_rms`).
- **No OOM**: peak 1.65/1.99 GB of 140 GB. Capacity is a non-issue at n=1e6; ell=6/7 are
  affordable memory-wise.
- **Per-case cost**: ~19-20 s construct, ~2.5 min wall per case including reps and references.

### Verdict-step budget at the presumptive verdict config (hier12 dense ell=5 K=256, n=1e6, F64)

127.21 ms = **12.7x over the 10 ms target**.

| stage | ms | share |
|---|---|---|
| L2B+nearfield (fused) | 52.83 | 41.5% |
| route_gen | 24.19 | 19.0% |
| M2L leaf (L5) | 22.85 | 18.0% |
| unaccounted (host alloc/GC) | 11.07 | 8.7% |
| refresh | 5.43 | 4.3% |
| M2L L2-L4 | 3.70 | 2.9% |
| grid | 3.44 | 2.7% |
| B2M / M2M / L2L | 1.05 / 0.99 / 0.99 | 2.4% |
| direct_gen + groups + occupancy + finalize + euler | 0.82 | 0.6% |

Bottleneck is **distributed across three stages** (L2B+near, route_gen, leaf M2L = 78.5%),
not concentrated — so no single lever reaches 10 ms.

### Pilot findings that reshape the sweep

1. **ell is exhausted as a knob.** Measured L5 M2L rate is 1.370 G routes/s and routes grow
   ~10.9x per level (L4→L5); nearfield scales with bodies/leaf (244 / 30.5 / 3.8 at
   ell=4/5/6). Extrapolating: ell=6 ⇒ leaf M2L ~183 ms (nearfield only ~6.6 ms) ⇒ ~220 ms;
   ell=4 ⇒ nearfield ~420 ms ⇒ ~430 ms. **ell=5 is the integer optimum and both neighbours are
   ~2-3.4x worse.** Because each ell step moves the two terms by 8x in opposite directions, the
   nearfield:leaf-M2L ratio of 2.3 cannot be balanced better. ⇒ Sweep ell={4,6} at **one case
   each** to confirm the bracket, not across the full cross product.
2. **The per-level M2L strategy-mix lever is worth ≤1.1 ms and should be demoted.** Class
   occupancy (`.classes.csv`) shows coarse levels are thin (L2 mean 4.6 routes/class, L3 mean
   92) exactly as 027 predicted — but those levels cost only 0.579+0.511 = **1.09 ms total**.
   The leaf level is fat (mean 17,993, p90 26,784), which is dense's home ground. Swapping
   factored/concat in at thin coarse levels can therefore recover ~1 ms of 127. This is a
   useful negative result against a lever the 028 task file named.
3. **`nonempty_classes` is 1740 at every level ≥3** — it is the stencil class count, not K.
   K=256 vs 1740 changes window grouping (and hence route_gen sync count), so the K sweep is
   still essential and untested at n=1e6.
4. **Stale-tree policy is a weak lever**: `stale_step_ms` 113.76 vs verdict 127.21 — skipping
   the refresh entirely saves only 13.4 ms (10.6%) and costs accuracy. Low priority, but
   measure `STALE>0` accuracy once.
5. Every major stage appears to run at **single-digit % of H200 roofline** (rough first pass:
   nearfield ~1.8 TFLOP/s, leaf M2L ~2.8 TFLOP/s and ~350 GB/s vs 4.8 TB/s HBM3e) — the 12.7x
   gap looks like an efficiency gap rather than a fundamental one. **This is a first-pass
   estimate only; the report must redo it properly per `05-phase-report.md`.**
- Notes:
  - First pilot 12994864 (m13h-1-1, 6m55s) failed fast: `FM028DeviceSystem` lacked the
    required `has_vector_potential` overload (`fmm!` checks it at `src/fmm.jl:882` even
    with explicit `lamb_helmholtz`). Fixed in `fm028_device_system.jl`. Good news from
    that run: lifecycle test 215/215 green on the synced tree; harness plumbing
    (per-case try/catch, CSV write, mode presets) worked.
  - The remote `FastMultipole-023` tree **vanished** between that job and resubmission
    (cluster archive migration suspected; a login-node julia segfault occurred mid-window).
    `fm023env` and `FastMultipole-026` (old 024b-era dir, holds pre-026 snapshots — do
    not delete) survived. `cuda_028_submit.sh` now recreates the remote tree and also
    syncs `data/cpu_gpu_scaling/references/` (needed by the accuracy gate; data was
    never part of the 027 sync list).
