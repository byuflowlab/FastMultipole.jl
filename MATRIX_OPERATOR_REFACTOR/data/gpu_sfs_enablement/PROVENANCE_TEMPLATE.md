# 048 corrected-device artifact provenance

No corrected-path device result exists yet. `cuda_048_run.sh` must produce
`fm048_ab_<job>.csv`, `fm048_ab_<job>.log`, and
`fm048_ab_<job>.provenance`. Preserve all three here without editing.

Required provenance fields (the run script emits them):

- FastMultipole and FLOWVPM Git SHAs when `.git` is available, plus deterministic
  SHA-256 digests of each rsynced source/test tree (required for dirty/no-Git trees).
- Exact Julia command; Julia version; loaded CUDA module; GPU name, UUID, and
  driver version.
- Seed, particle count, repetitions, P, precision, and resolved settings. The
  first four are CSV columns; resolved radix settings must also be printed in
  the raw log before acceptance.
- Raw log path and SHA-256; result CSV path and SHA-256.

Acceptance requires eight synchronized warmed synthetic A/B rows:
P=4/P=8 × Float32/Float64 × rho_t=4.211/4.789 on the same seeded state, plus
two real-p018 rows at production P=4/F64 (one per candidate). Both arms must have zero
device allocation, remain within the explicit host-bookkeeping budget, and
keep body/expansion-transfer counters at zero.
