# macOS smoke summary — 2026-07-18

Host: Apple/macOS, Julia 1.12.5. This is a correctness/allocation smoke run,
not the required non-macOS production tuning run. Configuration was `P=4`,
`N=80`, one timing repetition, both LH modes. Runtime BLAS thread reporting was
8 despite requesting 1, which is another reason not to use these rows as final
crossover evidence.

| LH | Variant | M2L ms | M2L alloc | Full step ms | Operator bytes | Scratch bytes |
|---:|---|---:|---:|---:|---:|---:|
| off | 023a factored | 1.109 | 1,808 | 2.224 | 7,170,728 | 41,760 |
| off | precomputed scalar | 0.478 | 2,928 | 1.550 | 2,057,760 | 1,594,240 |
| off | precomputed GEMM | 0.531 | 2,928 | 1.604 | 2,057,760 | 1,594,240 |
| off | precomputed mixed (12) | 0.458 | 2,928 | 1.538 | 2,057,760 | 1,594,240 |
| on | 023a factored | 2.429 | 1,808 | 4.205 | 8,226,792 | 64,800 |
| on | precomputed scalar | 1.106 | 3,216 | 2.978 | 4,338,304 | 1,950,144 |
| on | precomputed GEMM | 1.208 | 3,216 | 2.830 | 4,338,304 | 1,950,144 |
| on | precomputed mixed (12) | 1.466 | 3,216 | 2.730 | 4,338,304 | 1,950,144 |

The sparse run had 143–157 occupied angle classes with mean occupancy about
4.1 and maxima 13–22. It confirms that a scalar fallback is necessary for
narrow classes. Construction measurements include first-use compilation for
the first variant and should not be compared; the non-macOS runner starts
separate single- and multi-thread processes and provides the production data.
