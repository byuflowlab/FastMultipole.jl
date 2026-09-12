# 052e.2a real-simulation addendum — supersession record (2026-09-11)

**Authority: Ryan, 2026-09-11** ("just change the W2 arm to 2.4 overlap
across the board"), following the registered-run G1 failure report and the
root-cause diagnosis he directed. This record supersedes the W2 shedding
parameter of the LOCKED prereg
`052e2a-addendum-realsim-preregistration-2026-09-08.md` per its own
supersession clause (new dated document, reason recorded; Tier 0A
precedent). Every other locked value is unchanged.

## Reason

The registered run (launched 2026-09-10 20:03) failed gate **G1** at
BW2-VTS-L4: `block Gauss-Seidel produced a nonfinite physical residual at
outer iteration 1` ~56 min into the case. Root cause (diagnosed 2026-09-11,
Ryan-authorized, evidence in the failed run's log and the session
diagnostic scripts): **vortex-stretching runaway of the vortex-particle
wake at L4 core sizes** under `OverlapPPS(1.3, 2)` — particle |Γ| amplified
~10×/step (1.3e6 → 4.2e8 over steps 37–40 in a faithful repro), the
ReformulatedVPM σ evolution blew up (σ negative, |σ| up to ~8e4), particles
were ejected km from the 2.7 m wing, terminating in Inf/NaN particle state
→ NaN wake-induced velocity → nonfinite VTS RHS. L1–L3 (larger cores)
were stable. Ryan's hypothesis — overlap 1.3 is borderline unstable;
overlap > 2 stabilizes — was **confirmed** by an otherwise-identical
BW2-VTS-L4 repro with `OverlapPPS(2.4, 2)`: all 61 steps completed, max|Γ|
bounded 0.15–0.21, σ ∈ [0.089, 0.93] m positive throughout, max|X| = t·U∞,
zero nonfinite values.

## Change (the only one)

- W2 arm shedding, both `method_trailing` and `method_unsteady`, all levels:
  `OverlapPPS(1.3, 2)` → **`OverlapPPS(2.4, 2)`**.

## Realization

- New dated harness: `scripts/addendum_052e2a_realsim_2026-09-11.jl` —
  byte-identical to the 2026-09-09 harness except the OverlapPPS values,
  the output directory, and the R12 header note.
- Outputs to `data/052e2a-addendum-realsim-v2/` so the failed registered
  run's data/log/gates remain untouched as evidence.
- Full re-run of all cases (W1/Phase-A cases are unaffected by the
  parameter but re-run for single-provenance consistency). W2 results at
  L1–L3 shift with the core-size change; P1–P4 evidence and all gates are
  evaluated on the v2 dataset only — no mixing across preregs.
- Gates G1–G4 and predictions P1–P4 unchanged. Ruling remains Ryan's.
