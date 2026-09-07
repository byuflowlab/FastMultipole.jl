# 052e handoff — 2026-09-07 (context reset)

## Immediate state

- **Tier 0A kernel convention test: PASS (run 3).** All A1–A10 gates passed;
  the result, superseding preregistration, script, and provenance are in
  `052e0-tier0a-results-2026-09-07.md`,
  `052e0-tier0a-preregistration-v2-2026-09-07.md`, and
  `scripts/tier0a_052e_doublet_convention.jl`. Tier 0A validates the low-level
  constant-doublet kernel and wrapper, not the full `PanelWake` → body-trace
  reconstruction.
- **Solver-sequencing ruling adopted 2026-09-07.** Keep the full
  area-augmented bordered `:area_mean` system authoritative until Tier 0B
  proves that the formulation reconstructs the induced potential of a directly
  evaluated doublet-panel wake. Least squares is excluded from 052e testing and
  certification; the existing `:lsq` API need not be removed.
- **The reduced solver is deliberately not implemented yet.** After an
  explicit Tier 0B pass, implement the implicit-Householder
  $(N-1)\times(N-1)$ reduction and pass a separate bordered-parity gate before
  Tiers 1, 1.5, 2, or Stage B use it.

## Governing documents changed this session

All three changes are currently uncommitted in the FastMultipole worktree:

- `052e-theory-velocity-to-potential-trace.md`: keeps the bordered solve as the
  validation reference; derives the exact reduced square solve, implicit
  projection, recovered multiplier, cost/memory properties, pitfalls, and
  alternatives with pros/cons.
- `052e-accuracy-plan-v2-draft-2026-09-05.md`: records Ryan's sequencing ruling,
  makes `:area_mean` the only certified gauge, defines the direct
  doublet-panel-wake Tier 0B proof, and adds Tier 0B-R for post-proof parity.
  The solver sequencing is ratified; numerical gates still marked PROPOSED are
  not.
- `052e-impl-hybrid-wake-potential-experimental.md`: splits 052e.2 into
  **.2a bordered formulation proof** and **.2b reduced algebraic parity**.
  Dependency chain is now
  `.0 → .2a → .2b → {.3,.4} → .5`, with `.1` feeding `.2a` and `.6`
  independent.

`git diff --check` passes for the documentation changes.

## Mathematical decision to preserve

Let $A=I-B$, $b=S\sigma$, and let $a$ contain panel areas. The current
reference solves

$$
\begin{bmatrix}A&a\\a^T&0\end{bmatrix}
\begin{bmatrix}q\\\lambda\end{bmatrix}
=
\begin{bmatrix}b\\0\end{bmatrix}.
$$

After Tier 0B passes, normalize $\widehat a=a/\|a\|_2$ and construct an
implicit, cancellation-safe Householder reflector $H$ with
$H\widehat a=s e_N$. With $Z=H[:,1:N-1]$, solve the square problem

$$
(Z^TAZ)y=Z^Tb,\qquad q=Zy.
$$

This is an exact coordinate reduction of the bordered discrete equations, not
least squares. Do not form dense $Z$. Apply $H=I-2vv^T$ implicitly, and recover
the original bordered multiplier from the omitted transformed equation:

$$
\lambda=\frac{s}{\|a\|_2}
\left(\widetilde b_N-\widetilde A_{N,1:N-1}y\right),
\qquad \widetilde A=HAH^T,\quad \widetilde b=Hb.
$$

Projecting only the unknown or only the RHS is not equivalent for discretely
incompatible data. Factoring the full $PAP$ with
$P=I-aa^T/(a^Ta)$ is also wrong because it remains singular in $N$
coordinates.

## Exact next action: prepare 052e.2a / Tier 0B

1. Review and formally accept the updated theory note; do not interpret Tier
   0B results before that theory gate closes.
2. Write and ratify a Tier 0B preregistration before viewing results. The proof
   fixture must use a closed simply connected body and an external doublet-panel
   wake whose directly evaluated scalar potential and normal velocity are both
   sampled at the same body control points. Wake support/branch surfaces must
   remain outside the body.
3. Run only the existing full bordered `:area_mean` reconstruction. Record
   gauge-aligned trace error and refinement, flux compatibility, bordered
   $\lambda$, Green residual/gauge defect, and the Hodge diagnostics. Preserve
   the analytic-normal-velocity structural kill rule and the staged
   filament/particle/extraction decomposition in the accuracy plan.
4. End 052e.2a with an explicit pass/continue or retire ruling. Do not begin
   reduced-solver source edits merely because Tier 0A passed.
5. Only after a pass, begin 052e.2b: implement the in-place Householder route,
   retain the border as oracle/fallback, and test roundoff-scaled parity for
   compatible and deliberately incompatible RHSs, refined/distorted-area
   meshes, independently gauged disconnected bodies, and physical outputs.
   Benchmark setup/recurring cost and both retained and peak memory. On parity
   or conditioning failure, keep the border; do not silently substitute a
   rank-one, pinned-row, iterative, or least-squares method.

## Repository/worktree cautions

- FastMultipole is heavily dirty with unrelated higher-derivative source,
  documentation, and test work plus the untracked `HIGHER_DERIVATIVES/` tree.
  Touch only the 052e documents/scripts needed for this campaign; do not clean,
  revert, stage, or commit unrelated changes.
- `../FLOWPanel.jl` is also dirty with unrelated BRAINSTORM/example work. Its
  `src/FLOWPanel_formulation.jl` and `test/formulation_test.jl` currently contain
  an uncommitted 052e.1 hard unpaired-edge initialization error and negative
  regression. Preserve those edits. The reduced solve has not been added.
- Before FLOWPanel code work, read its `AGENTS.md`, `CLAUDE.md`, and
  `agent_policies/WORKFLOW.md`; before testing, also read
  `agent_policies/TESTING.md`. Local jobs must use at most four threads.
- Do not run an official campaign from either live dirty checkout. Official
  acceptance work requires committed, annotated-tagged campaign worktrees and
  recorded dependency pins. No run, deployment, push, notebook entry, or HPC
  submission is authorized merely by this handoff.

## Still open but separate from this handoff

- Formal acceptance of the revised theory note and ratification of the Tier 0A
  PASS record.
- 052e.1 regression completion after 052b closes.
- 052e.6 topology-aware global gauge recovery; until separately verified,
  promotion remains limited to gauge-invariant circulation, exterior velocity,
  and integrated loads—not absolute $C_p$, unsteady Bernoulli pressure, or
  acoustics.
- Historical 052b/053/storage/notebook issues remain recorded in
  `052e-handoff-prompt-2026-09-06.md`; do not infer that this focused handoff
  resolves them.
