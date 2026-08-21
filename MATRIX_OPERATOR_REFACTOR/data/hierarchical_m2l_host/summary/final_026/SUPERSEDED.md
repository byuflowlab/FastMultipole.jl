# SUPERSEDED — hierarchical `precomputed_y` rows are mislabeled

Every `strategy=precomputed_y` row with `policy=hierarchical_12` or
`policy=hierarchical_3` in this campaign actually measured the **concat
engine** (bug B1: a construction-time override replaced the precomputed-y
apply plan; see `../../../026-impl-hierarchical-m2l-host.md`, Approval
Notes / Post-Review Fixes). Accuracy columns are unaffected
(bit-identical to the re-measurement); performance attribution for those
rows — including the `crossovers.csv` `not_observed` verdicts for
hierarchical precomputed-y — is wrong.

Genuine hierarchical precomputed-y measurements: jobs
`12953685` (scaling), `12954300` (tuning), `12954303` (accuracy) under
`../12953685/`, `../12954300/`, `../12954303/`. All other strategies and
policies in this campaign remain valid.
