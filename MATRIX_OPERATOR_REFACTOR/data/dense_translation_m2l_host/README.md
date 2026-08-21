# Dense-translation resident M2L host benchmarks

Task 023e artifacts are labeled `functional_baseline` or `optimized`. Each CSV is
one fresh Julia process/configuration so dense-plan construction and peak RSS are
not contaminated by abandoned plans. The rows record git commit/tree and dirty
status, host/CPU/Julia/BLAS provenance, actual BLAS thread count, strategy options,
route occupancy, construction and warmed stage/full-step timings and allocations,
and exact dense payload bytes. `plan_summary_bytes` is diagnostic object-inclusive
storage. Byte fields are bytes; divide by 1,048,576 for MiB. `process_peak_rss_raw`
is retained in the platform-reported units and is not mixed with the exact payload
fields.

See [`verification_summary.md`](verification_summary.md) for the matched
functional/final analysis, memory table, retained/rejected candidates, and job IDs.
