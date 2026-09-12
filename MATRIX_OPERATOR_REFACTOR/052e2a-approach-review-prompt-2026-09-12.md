# Prompt: independent review of the 052e.2a approach and results (2026-09-12)

You are an independent reviewer with a strong background in boundary-element
methods, potential theory, and vortex methods. You had no part in designing
or running this item. Your job is to evaluate (1) whether the approach taken
is sound and effective, (2) whether the process that produced the results is
trustworthy, and (3) whether a more effective alternative approach exists —
or whether the current approach should be endorsed as-is. Ryan (the PI)
makes all rulings; you produce an assessment and recommendation only.

## The engineering goal (context)

FLOWPanel.jl couples a panel-method body to a wake (doublet panels, or
vortex particles from FLOWVPM). The legacy coupling, `VelocityThroughSources`
(VTS), absorbs the wake's induced velocity into the body's source strengths
(sigma_body = -(U_inf + u_w)·n). The hypothesis under test: VTS is wrong for
lifting flows because sources cannot carry the wake's circulation content,
and the correct coupling is `GreenReconstruction` (GR): sample the wake's
normal velocity on the body, set sigma = -u_w·n, solve the gauged bordered
Green-identity system (I-B)q = S·sigma for the wake's potential trace q
(Neumann-to-Dirichlet map; area-mean gauge a'q=0, production Householder
reduction), and feed q into the Dirichlet doublet RHS. Motivation for
reconstructing from velocity: vortex-particle wakes have no directly
evaluable scalar potential, so the potential trace must come from velocity
data if it is to come from anywhere.

## Read these, in this order (all paths relative to
`/Users/ryan/Dropbox/research/projects/FastMultipole/MATRIX_OPERATOR_REFACTOR/`)

1. `052e-theory-velocity-to-potential-trace.md` — the theory note
   (Green identity, bordered system, gauge, Householder form).
2. `052e2a-addendum-realsim-preregistration-2026-09-08.md` — LOCKED prereg:
   fixture, routes, phases, wakes, gates G1-G4, predictions P1-P4.
3. `052e2a-addendum-realsim-supersession-2026-09-11.md` — the single
   registered change (W2 shedding OverlapPPS 1.3 -> 2.4) and why.
4. `052e2a-addendum-realsim-results-2026-09-11.md` — the v2 registered-run
   results: gate report, Gamma(y) tables, P1/P2 gap tables, P3 oracle
   table, P4 ratios, diagnostics, evidence summary, (possibly ruled)
   ruling checklist.
5. For deeper method validation history if needed: `052e2a-tier0b-*.md`,
   `052e2b-tier0br-*.md` (synthetic-fixture certification of the bordered
   formulation and Householder parity).
6. Harness (read-only, do not edit):
   `scripts/addendum_052e2a_realsim_2026-09-11.jl`.

Rules: never read files under `data/**` (CSVs) directly — if you need their
contents, write a small script that prints a summary. Do not edit any of the
above documents or scripts; the prereg is LOCKED and the harnesses are
sha-registered. Do not launch runs longer than a few minutes; no more than
4 threads. Your deliverable is a markdown report (see Output).

## Headline results you are evaluating

- G1/G2/G3 PASS (36/36 solves; bit-identical wake hashes across routes;
  Green residual <= 1.22e-14, gauge defect <= 2.2e-16).
- G4 FAIL on threshold: oracle error E_q (dy-weighted rms of reconstructed
  vs directly-evaluated true panel-wake potential trace, R-GR/W1) is
  monotone under refinement but reaches only 5.851e-2 (Phase B) /
  4.895e-2 (Phase A) at L4 = 19,384 panels vs the registered <= 1e-2 gate;
  empirical decay ~N^-0.6 (~h^1.2 in panel size).
- P1 evidence: GR-NEU Gamma gap shrinks (Phase A grms 2.02e-2 -> 5.39e-3
  L1->L4; Phase B shrinks then plateaus ~5.5e-3 with dGtot crossing zero).
- P2 evidence: VTS-NEU gap 0.89-0.99 grms at every level, persistent/growing.
- P4 evidence: particle-wake vs panel-wake trace difference small
  (<= 1.68e-2) but growing under refinement (7.4e-3 -> 1.68e-2).

## Questions to answer

A. **Soundness.** Is the Green-trace formulation mathematically correct and
   well-posed as implemented (interior harmonicity assumption, gauge
   treatment, TE/Kutta interaction, endcap geometry)? Any hidden
   assumptions that the real-sim fixture violates?

B. **The G4 rate.** Is ~N^-0.6 convergence of E_q what one should EXPECT
   for flat triangular panels with collocation on this metric (e.g.
   double-layer operator accuracy, TE-adjacent singularity, max-norm
   pollution: note Einf stays O(0.4-0.6) at L4), or does it indicate a
   defect? Was the 1e-2-at-L4 gate realistic a priori? Distinguish
   "formulation converges but discretization is low-order" from
   "formulation has an error floor."

C. **Alternatives.** For the actual production goal (correct
   particle-wake -> body coupling in unsteady simulate!), are there
   approaches likely to be MORE effective than velocity->Green-trace
   reconstruction? Consider at least: direct evaluation of the wake
   potential where it exists (W1 doublet wakes) with GR reserved for
   particles; solving the body problem in Neumann form everywhere;
   higher-order or desingularized panel quadrature for B and S;
   least-squares (rectangular) trace solve; alternative gauges;
   regularized/vector-potential couplings; hybrid schemes. For each,
   assess accuracy potential, cost, intrusiveness into FLOWPanel, and
   whether it addresses the G4 rate or only the constant.

D. **Process.** Was the prereg/gate/supersession discipline sound (one
   registered parameter change after a G1 crash, evidence preserved,
   no retuning)? Any threats to validity in the fixture (single geometry,
   single AOA, laptop tier, referee body uncapped vs capped)?

E. **Recommendation.** One of: (i) approach effective as-is — proceed to
   052e.3 hybrid fixture; (ii) approach sound but needs a specific
   improvement first (name it and the cheapest decisive experiment);
   (iii) a different approach should be tried (name it and a minimal
   falsifying test). Justify against the evidence, not taste.

## Output

Write your report to
`MATRIX_OPERATOR_REFACTOR/052e2a-approach-review-2026-09-12.md` (new file;
do not overwrite anything). Structure: verdict summary (<= 10 lines),
then sections A-E with specific citations (file:line for claims about the
docs/code; exact numbers for claims about results). Flag every claim you
could not verify. Keep speculation clearly labeled. Ryan rules on any
follow-up; do not modify code, docs, or data.
