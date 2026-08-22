# Task 049 corrected rotor verification

- Job: `13305555`; GPU UUID: `GPU-942a2d09-ae6b-96c8-3185-03b8cafaafcd`
- Julia: `1.11.7`; threads: `8`
- CUDA runtime/driver: `12.8.0` / `13.0.0`; device: `NVIDIA H200`
- Command: `/home/rander39/FLOWVPM-046/data/fm049/p018_710_particles.bin /home/rander39/FLOWVPM-046/data/fm049/p018_711_particles.bin /home/rander39/FLOWVPM-046/data/fm049/p018_712_particles.bin /home/rander39/FLOWVPM-046/data/fm049/p018_713_particles.bin /home/rander39/FLOWVPM-046/data/fm049/p018_714_particles.bin /home/rander39/FLOWVPM-046/data/fm049/p018_715_particles.bin /home/rander39/FLOWVPM-046/data/fm049/p018_716_particles.bin /home/rander39/FLOWVPM-046/data/fm049/p018_717_particles.bin /home/rander39/FLOWVPM-046/data/fm049/p018_718_particles.bin /home/rander39/FLOWVPM-046/data/fm049/p018_719_particles.bin`; seed: `49049`; reps: `5`; sample: `2000`; dt: `0.00030864197530864197`
- Matrix: P=`(4, 8)`, precision=`(Float64, Float32)`, rho_t=`(4.211, 4.789)`; benchmark snapshot: `710`; residency snapshots: `710:719`; residency/budget/profile at production settings P=`6`, rho_t=`4.789` (D14, 2026-08-22)
- Project: `/home/rander39/fm048env/Project.toml`; Manifest SHA-256: `c3848caf9e6012dffbb05fdec20e6cad1a2f45642388b8ae980614f5994e12db`
- Harness: `/home/rander39/FLOWVPM-046/scripts/fm049_rotor_verify.jl`; SHA-256: `864879da21f0531a5f10a7bec665f2cdaa45a8646716bb9d627553ea897b7444`
- FLOWVPM commit: `unavailable`; FastMultipole commit: `unavailable`
- Input manifest: `/home/rander39/FLOWVPM-046/data/fm049/manifest.csv`; SHA-256: `9452b62d31514ba6d4b78adee3ff4a80f40fc4d0a2859e4f975d4eab03d03639`

Results include the complete matrix, hard direct-reference gates, contracts, hashes, and true residency A/B. Budget rows include end-to-end, contiguous-transfer, clean ordered-stage, complete-UJ+SFS, and RK3 residual timings. The run script hashes results, budget, report, raw log, synced-source manifests, and submission provenance.

041a anchors: 7.40 ms at 1e5 and 92.3 ms at 1e6; CPU production baseline: 170–230 s/step; target: 3.3 s/step.

## Residency checkpoint

No threshold chooses a default. Present both A/B results to the user and ask which mode should ship.
