# Higher Derivatives Project

The source of truth is `REVIEW_PLAN.md`. The public contract preserves the dense 9-row
Hessian and adds `T[i,j,k] = ∂H[i,j]/∂x[k]` in component-major packed-18 form, with
the symmetric derivative pairs `(xx,xy,xz,yy,yz,zz)` for each `i`.

## Working rules

1. Follow current user instructions, then `REVIEW_PLAN.md`, this file, the selected task,
   and finally theory/codebase-map artifacts.
2. Verify cited locations before editing and inspect connected interfaces whenever the map
   may be stale.
3. Do not change unrelated `MATRIX_OPERATOR_REFACTOR/` files.
4. Use focused tests for individual changes and full-suite runs at interface milestones.
5. Theory work belongs in `theory/`, `scripts/`, and `data/`; production changes belong to
   the I/G tasks.
6. Update the checklist below whenever an item's state changes: keep its 1-sentence summary to
   exactly one sentence and overwrite that sentence in place rather than appending status history.
7. After finishing an item task, make sure to check the appropriate box.

## Context-reset handoff

- On 2026-09-07 a fresh clear-context session reviewed and accepted the I6 fix, the four
  docstrings, I7's records, and the I8 LH implementation (see those item files), then
  implemented I9: three premise-guarded LH testsets appended to
  `test/third_derivative_test.jl` with thresholds calibrated by a P-sweep (rel err 3.0e-10 at
  P=6 → 8.4e-16 at P=20; see I9's item file), passing full `Pkg.test` suites at 1 and 4
  threads (1,382,726 / 1,382,870 pass, exactly +30 over the I7 baselines). A second fresh
  session on 2026-09-07 performed the clear-context review of I9 — checking the testsets
  against the REVIEW_PLAN clause and independently reproducing the focused bounds-checked run
  (0 fail, all 30 LH assertions) — and accepted it without changes, closing the CPU phase.
  The next step is G2 (level-three GPU cache layout, gated by the reviewed G1 census).
  Focused runs should include `--check-bounds=yes`.
- The worktree is intentionally dirty and the whole `HIGHER_DERIVATIVES/` directory is untracked;
  preserve all existing source, test, documentation, and unrelated `MATRIX_OPERATOR_REFACTOR/` work.
- Verification on 2026-09-07 passed the ForwardDiff formula script under `--project=test` and the
  complete package suite with four Julia threads; CUDA execution was unavailable on macOS.
- Read this file, the I6 item file, and the I6 clauses in `REVIEW_PLAN.md`, then inspect only the
  connected test and implementation surfaces needed for the next missing acceptance case.

## Contract

- `DerivativesSwitch{PS,GS,HS,NO,NM,TS}` appends `TS`; three-Boolean constructors remain.
- Hessians remain dense 3×3 values in 9 rows.
- Third derivatives use 18 rows: `(xx,xy,xz,yy,yz,zz)` for each vector component.
- `ThirdDerivativeTensor <: AbstractArray{T,3}` exposes symmetric `(j,k)` indexing;
  `packed_data` returns its 18 values and `dense` explicitly materializes 27 values.
- `third_derivative` is available on CPU `fmm!` and standalone `direct!`; unsupported
  target/source pairs fail via `supports_third_derivative` before computation.
- `solve!` does not expose third-order output.
- Radix uses 4/13/31-row derivative levels only after its gated GPU tasks are complete.

## Sequence

Theory: T1–T4, then T6; T5 independently gates LH work. CPU: I1–I7, followed by I8–I9.
GPU/Radix: G1–G4 only after the CPU milestones. Detailed dependencies and acceptance
criteria are in `REVIEW_PLAN.md`; task files give the per-item scope.

## Progress checklists

This is the authoritative item-status view, refreshed 2026-09-07 from the current worktree
and verification results. `Technical` means the item's scoped implementation or artifact is
present; `Clear-context review` means a fresh review found its acceptance evidence sufficient,
and went through a check seeking significant improvements in terms of correctness, performance,
robustness, user-friendliness, and minimal invasiveness (in order of decreasing importance). If
changes are made, another clear-context review must be performed.
An unchecked box is intentionally not a claim of failure—the final sentence states the known gap.

### Theory phase

| Item | Technical | Clear-context review | Very brief title | Gated by | 1-sentence summary |
|---|:---:|:---:|---|---|---|
| [T1](T1-theory-third-derivative-recurrence.md) | [x] | [x] | Derive scalar recurrence | — | The recurrence and scalar symmetry are documented and exercised by the scalar packed-output verification. |
| [T2](T2-theory-symmetric-packing.md) | [x] | [x] | Fix packed-18 contract | — | The 18-slot layout, tensor indexing, compact rows, and unchanged 9-row Hessian contract are documented and tested. |
| [T3](T3-theory-real-basis-contraction.md) | [x] | [x] | Derive real contraction | T1 | The real-basis path uses the deeper coefficient recurrence and passes scalar and LH parity checks against the complex path. |
| [T4](T4-theory-direct-kernel-formulas.md) | [x] | [x] | Build scalar direct oracle | T2 | The packed scalar formula is documented, implemented, and agrees with nested ForwardDiff across multiple scales. |
| [T5](T5-theory-lamb-helmholtz-third-derivative.md) | [x] | [x] | Derive LH tensor | T1, T2 | The point-vortex formula, derivative-index symmetry, and recurrence flow are documented and agree with ForwardDiff. |
| [T6](T6-milestone-review-theory.md) | [x] | [x] | Review theory milestone | T1, T2, T3, T4, T5 | The theory artifacts and executable direct-oracle checks have been rerun and are sufficient to support the implemented CPU paths. |

### CPU implementation phase

| Item | Technical | Clear-context review | Very brief title | Gated by | 1-sentence summary |
|---|:---:|:---:|---|---|---|
| [I1](I1-impl-hessian-repack.md) | [x] | [x] | Add compressed tensor API | T6 | The tensor, accessors, exports, probes, compatibility constructors, allocation checks, and unchanged Hessian layout are present and passing. |
| [I2](I2-impl-switch-extension.md) | [x] | [x] | Plumb and guard TS | I1 | TS is appended through switches and public CPU entry points with capability preflight, plan/cache integration, rotation sensitivity, docs, and a Radix rejection guard. |
| [I3](I3-impl-l2b-complex.md) | [x] | [x] | Implement complex scalar L2B | I2 | Complex L2B computes packed third derivatives independently of lower outputs with scratch allocated only for TS requests. |
| [I4](I4-impl-direct-reference.md) | [x] | [x] | Implement scalar direct oracle | I2 | Gravitational direct evaluation now accumulates analytic Hessian and packed third derivatives with storage, writeback, opt-in, and ForwardDiff coverage. |
| [I5](I5-impl-l2b-real-basis.md) | [x] | [x] | Implement real scalar L2B | I3, I4 | Real-basis evaluation emits packed third derivatives and passes complex-basis parity coverage. |
| [I6](I6-impl-integrated-tests.md) | [x] | [x] | Complete scalar integration matrix | I5 | The bounds-checked legacy-overload fix passed a fresh clear-context review and re-run (1451-pass focused suite under `--check-bounds=yes`). |
| [I7](I7-milestone-review-implementation.md) | [x] | [x] | Close scalar milestone | I6 | A fresh clear-context review found all milestone records (full suites at 1/4 threads, verify script re-run PASS, disabled-path ratio 1.0024, memory record, docs status, four added docstrings) complete and sufficient. |
| [I8](I8-impl-lamb-helmholtz.md) | [x] | [x] | Implement LH CPU support | T5, I7 | A fresh clear-context review verified the vorton direct formula by hand, the L2B recurrence slot-by-slot, scratch/guard/opt-in plumbing, and three-way agreement including a quantitative FMM-vs-direct P-sweep to machine precision; accepted without changes. |
| [I9](I9-impl-lh-tests.md) | [x] | [x] | Close LH milestone | I8 | A fresh clear-context review checked the three premise-guarded LH testsets against the REVIEW_PLAN clause and independently reproduced the focused bounds-checked run (0 fail, all 30 LH assertions passing); accepted without changes. |

### GPU/Radix phase

| Item | Technical | Clear-context review | Very brief title | Gated by | 1-sentence summary |
|---|:---:|:---:|---|---|---|
| [G1](G1-gpu-census-and-layout.md) | [x] | [x] | Census GPU routes | I9 | The route capability matrix and 4/13/31-row target layout are documented and reviewed, and with I9 closed the GPU phase (G2) is unblocked. |
| [G2](G2-gpu-hessian-repack.md) | [ ] | [ ] | Add level-three cache layout | G1 | Radix still uses its existing Hessian Boolean and 4/13-row buffers, so maximum derivative levels and 31-row lifecycle plumbing remain unimplemented. |
| [G3](G3-gpu-third-derivative-kernels.md) | [ ] | [ ] | Implement GPU kernels | G2 | No packed-18 scalar or vortex GPU near-field or far-field kernels have been implemented. |
| [G4](G4-gpu-parity-tests.md) | [ ] | [ ] | Qualify GPU parity | G3 | GPU parity, lifecycle, performance, and memory acceptance remain pending after kernel implementation. |
