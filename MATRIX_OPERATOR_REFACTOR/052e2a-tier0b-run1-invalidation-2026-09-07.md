# 052e.2a Tier 0B — run 1 INVALIDATED (harness defect) — 2026-09-07

**Status:** Run 1 (PID 51333, launched 2026-09-07, completed ~13:58) is
INVALID due to a harness defect. No gate value from run 1 may be
interpreted; the KILL-RULE triggers it printed are artifacts. The locked
preregistration `052e2a-tier0b-preregistration-2026-09-07.md` is unchanged
— this is a harness fix + clean relaunch per the handoff protocol ("a
harness (not kernel/formulation) defect is fixable, but then the whole
registered run must be relaunched cleanly and the fix recorded — no
partial reuse").

## Symptom

`gates.txt` (run 1) showed degenerate values across all levels/cases:
E_q ∈ {NaN, exactly 1.0}, flux=NaN, lambda=-0.0, residual=0, gauge
defect=0, Hodge defect=0, while the report-only cross-check numbers were
~1e-12–1e-15. CSV column audit (by script, not direct read) showed:
control points x,y,z ≡ 0 for every panel, sigma ≡ -0.0, q_ref constant
per case, q_green ≡ 0. Areas and TE-adjacency masks were correct.

## Root cause

FLOWPanel fills `body.controlpoints` and `body.normals` **lazily**: the
constructor leaves them zero-initialized and the solver populates them via
`calc_normals!` / `calc_controlpoints!` (see
`FLOWPanel_solver.jl:251-252`). The harness called these on the **oracle**
body (`make_oracle`) but never on the system-under-test body from
`make_body` (which calls `build_pitching_wing_body`, which does not
populate them either). Consequences:

- all control points at the origin → oracle evaluated at a single point →
  `q_ref` constant (zero at AOA 0° → NaN relative errors; nonzero at 7° →
  relative error exactly 1.0);
- all normals zero → `sigma = -n·u_wake ≡ -0.0` → zero RHS → `q = 0`,
  `lambda = 0`, residual/telemetry all degenerate;
- both the oracle and the production `influence!` cross-check route were
  evaluated at the same (wrong) points, so the cross-check alone looked
  clean — it compared two evaluations at the origin.

The TIER0B_SMOKE mechanical check did not catch this because, by design,
it verified exit status only and no gate values were viewed.

## Fix

`scripts/tier0b_052e2a_bordered_formulation.jl`, `make_body`: call
`pnl.calc_normals!(body)` and `pnl.calc_controlpoints!(body)` on the
constructed body before returning it. No locked fixture constant, gate,
or formulation code was touched.

## Disposition

- Run-1 outputs archived to
  `data/052e2a-tier0b/run1-invalid-harness/` (log, gates.txt, 8 CSVs).
- Fixed harness re-smoke-tested (TIER0B_SMOKE=1, exit status only), then
  the full registered run relaunched cleanly as run 2
  (`tier0b_run2.log`); new `script_sha256` recorded in run-2 `gates.txt`.
