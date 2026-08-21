# SUPERSEDED — hierarchical `precomputed_y` rows are mislabeled

The `strategy=precomputed_y` cases with hierarchical policies in this raw
campaign measured the concat engine (bug B1, fixed 2026-07-29; see
`../../../026-impl-hierarchical-m2l-host.md`). Accuracy is unaffected;
performance attribution is wrong for those rows only. Genuine
re-measurements: `../12953685/`, `../12954300/`, `../12954303/`.
