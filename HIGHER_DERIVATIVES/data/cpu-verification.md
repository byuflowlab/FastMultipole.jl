# CPU verification snapshot — 2026-09-07

- Packed tensor construction/get/index/packed setters allocate zero bytes after warm-up.
- Scalar separated-cluster FMM/direct relative maximum error: `3.35e-11` at P=8.
- Point-vortex separated-cluster FMM/direct relative maximum error: `9.01e-11` at P=8.
- Scalar and point-vortex analytic formulas agree with nested ForwardDiff checks.
- Four-thread `Pkg.test()` completed successfully; CUDA tests were skipped on macOS where
  CUDA is unavailable.
