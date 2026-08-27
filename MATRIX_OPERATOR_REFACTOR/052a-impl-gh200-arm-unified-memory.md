# 052a — GH200 execution: ARM offline environment + unified-memory assessment

**Status: IN PROGRESS `2026-08-25` — Phase A COMPLETE (offline ARM env
verified; probe passed, job 13477052). Phase B: testset preflight 11/11
(job 13477924), smoke passed (job 13477793); mature DEFERRED — the
reference gate correctly rejects the session's three FLOWPanel fixes vs the
canonical CPU provenance, and per Ryan (2026-08-25) mature waits until the
per-arch checkout changes are consolidated into one code base (after the
other arch campaigns run). Phase C CLOSED (verdict below, approved
2026-08-25): NO unified-memory adoption; discrete path stays. Next: arch
campaigns → consolidation → CPU-reference regen → mature → Phase D.** The
lower-level plan is `052a-plan-2026-08-25.md`; its top HANDOFF block is the
entry point.

## Scope

Split out from `052-impl-flowpanel-018-driver-gpu.md` (multi-architecture
extension, 2026-08-25): actually execute the `gh200` slug end-to-end, and
assess whether the GH200's fused CPU–GPU memory warrants a different memory
approach than the discrete-transfer strategy inherited from the H200.

The 052 harness (`fm052_arch_prepare.sh`, `fm052_arch_run.sh`, probe → smoke →
mature → cross-arch, naming/provenance/eligibility policy) is taken as given
and is NOT modified here except where the GH200 offline-install reality
requires finishing `prepare`'s ARM pre-instantiation. The acceptance policy is
immutable: Float64, P=16/SFS/FMM, the 46.064 GiB eligibility bound, and all
gates stand. Unified-memory experiments are measurement/exploration stages —
they do not weaken or substitute for any acceptance gate.

## Constraints (confirmed with user, 2026-08-25)

- GH200 partition: `mgh` / `gh200`, `--constraint=arm`, ARM64 Grace 72
  CPU/node. Compute nodes have **no internet**; only the x86_64 login node
  does; `/home` is shared between login and GH200 nodes.
- aarch64 Julia 1.11.7 at `/home/rander39/julia/julia-1.11.7/bin/julia`
  (verified ELF aarch64; launcher SHA-256 recorded in 052-impl). The harness
  already treats this as the default Julia for the `gh200` slug.
- Memory question is **baseline first with a decision gate** — no committed
  unified/managed-memory changes up front.
- Run target: short validation first (probe → smoke → short mature window),
  full acceptance as the final stage.

## Hardware framing

GH200 vs H200: HBM is *smaller* (96 GB vs 141 GB) but the Grace side exposes
~480 GB LPDDR5X to the GPU over NVLink-C2C (~900 GB/s), far faster than PCIe
staging. The code's steady-state transfer volume is already near zero by
design (device-resident radix state; per step only a pinned body-buffer
upload, an influence-prefix download, and pinned 4-byte scalars — instrumented
by `CUDARadixTransferCounters` and asserted by tests), so the plausible GH200
wins are (a) **capacity**: oversized one-shot buffers (e.g. the 10.805 GB S
matrix, or larger future problems) living in Grace memory and read over C2C,
and (b) **faster construction-time uploads and staging copies**. A hard
constraint stands: CUDA.jl managed per-array synchronization is disabled on
the side-stream working set and is illegal under CUDA graph capture
(`CUDA.enable_synchronization!(arr, false)`,
`src/translate_batched_cuda.jl:6535-6551`, CUDA error 900, job 13060540) — no
managed arrays may enter the graph-captured step.

## Phases

### Phase A — Offline aarch64 environment (gating unknown; lower-level plan first)

Finish `fm052_arch_prepare.sh`'s ARM pre-instantiation so that a GH200 node
with no network can load the stack:

- Populate the isolated per-slug depot **from the x86 login node** using Pkg
  artifact platform-override (`Base.BinaryPlatforms.Platform("aarch64",
  "linux"; ...)` with the correct `cuda`/`libc`/`julia_version` tags) so
  aarch64 JLL artifacts (CUDA runtime, OpenBLAS, etc.) download without
  executing ARM code. Registry and package sources are arch-independent.
- CUDA runtime decision: pin the CUDA runtime artifact version explicitly so
  the x86-side download matches the ARM node's driver; fallback is the
  cluster-local toolkit via `CUDA.set_runtime_version!(local_toolkit=true)`
  in a slug-side `LocalPreferences.toml`.
- Precompilation cannot run on x86 for aarch64: first `using` on the GH200
  node precompiles offline (artifacts already present). Keep x86/aarch64
  compile caches from colliding (separate depots per slug already; verify
  Julia's per-arch cache slugs suffice if a depot is ever shared).
- **Exit criterion:** on a GH200 node with no network,
  `/home/rander39/julia/julia-1.11.7/bin/julia --project=<fm052env-gh200> -e
  'using FastMultipole, CUDA; CUDA.versioninfo()'` succeeds.

**Phase A COMPLETE `2026-08-25` (job 13477052; exit criterion first met in
job 13476578).** CUDA runtime pinned to the 12.6 artifact (sbsa) with
driver `local=true`; node driver 580.159.04 supports CUDA 13.3. The probe
stage also passed in job 13477052 (`status = "pass"`, 54 JLLs, lbt BLAS,
97,871 MiB VRAM ≥ the 46.064 GiB bound), satisfying the first Phase-B gate.
Implementation notes: prefs must be written without loading CUDA.jl on the
x86 login node, the pinned JLLs must be registered in the env Project's
`[extras]` for the augmentation hooks to see the pin, cross-platform
instantiate needs explicit `cuda`/`cuda_platform` tags (hooks don't run for
foreign platforms), and two latent bugs in `fm052_arch_probe.jl` were fixed
(PackageInfo `.uuid` field access; wrong hardcoded BLAS expectation) — the
h100/b200/l40s copies of the probe need the same re-sync before their
probes run (prepare now rsyncs scripts with `--checksum`).

### Phase B — Probe + short validation (existing discrete-memory path, unchanged)

- Run the harness `probe` stage ARM-native (per 052 policy: "GH200 is not
  feasible unless this ARM-native probe actually passes"), then `smoke`
  (one upload, three GPU-S gemvs, CPU-S parity).
- Preflight the standard `test/cuda_*_test.jl` set on the GH200; the
  transfer-counter contract should hold unchanged — the discrete path is
  valid on GH200, C2C just makes copies faster.
- `mature` 36-step window with per-phase timing; provenance-gated reuse of
  the canonical CPU arm; cross-arch report normalized to H200.
- Record actual post-upload free HBM at 96 GB (eligibility bound 46.064 GiB
  passes on paper; measure the real margin, especially the 16 GiB
  mature-tail minimum).

### Phase C — Memory-effectiveness assessment (measure, then decision gate)

Collected on the GH200 without code changes:

- Transfer-counter volumes + measured H2D/D2H bandwidth for the pinned
  staging copies (quantify the C2C advantage against the near-zero
  steady-state volume).
- HBM headroom with the S matrix resident vs not; identify any configuration
  of interest that is capacity-blocked at 96 GB.
- Whether `CUDA.pin` on Grace-side staging buffers still matters on
  fused-memory hardware (first-touch/pageable behavior).

**Decision gate — adopt targeted unified/managed memory only on a
demonstrated capacity or bandwidth win**, e.g. the S matrix or other
oversized one-shot buffers allocated host-addressable/managed and read over
C2C when HBM is tight. Any such change touches only construction-time
factories (`_to_cuda_array`, the `DeviceResidentRadixState`/adaptive-context
constructors) — never the graph-captured/side-stream working set (error-900
constraint above). Candidate mechanisms: unified memory for specific
buffers, or `JULIA_CUDA_MEMORY_POOL` experiments confined to a benchmark
stage. If no win: record the negative result and keep the discrete-style
path as the multi-arch default.

#### Phase C verdict — CLOSED 2026-08-25 (approved by Ryan): negative result, keep the discrete path

No unified/managed-memory adoption: the gate's "demonstrated capacity or
bandwidth win" is not met. Measurements: membench job 13477889 (GH200 C2C)
and the identical benchmark on an x86 H200 PCIe node, job 13478001 (same
`fm052a_gh200_membench.jl`, thin x86 wrapper `fm052a_h200_membench.sh`;
CSVs + sha-stamped TOMLs under
`FLOWPanel-052-{gh200,h200}/data/fm052_multiarch/<arch>/membench/job-<id>/`).
Note the previously assumed "H200 PCIe report numbers" never existed — the
H200 GPU-S run had crashed pre-upload on the `CUDA.MemoryInfo` API break —
so job 13478001 is their first measurement.

| Quantity | GH200 C2C (13477889) | H200 PCIe (13478001) | C2C/PCIe |
|---|---|---|---|
| H2D, ≥1 GiB, pinned | 374–380 GB/s | 55.4 GB/s | 6.8× |
| H2D, ≥1 GiB, pageable | 377–381 GB/s | 9.8–10.0 GB/s | ~38× |
| D2H, ≥1 GiB, pinned | 294–296 GB/s | 42.2–42.9 GB/s | 6.9× |
| Pinning benefit at ≥1 GiB | none (≤1%) | 5.6× H2D / 2.6× D2H | — |
| One-shot S upload (10.805 GB, pinned) | 0.0287 s | 0.1958 s | 6.8× |
| Free HBM with S resident | 84.363 of 95.577 GiB | 129.173 of 139.801 GiB | — |

Reasoning: (1) no bandwidth win is possible — discrete staging on C2C
already runs at link speed (pinning irrelevant there), and managed pages
migrate over the same link; (2) the cost to remove is already negligible —
0.029 s one-shot S upload, per-step traffic in the tens of µs; (3) capacity
is not binding — 84.4 GiB free with S resident leaves room for ~5.8× larger
S (~2.4× panel count) inside the locked 32+4 GiB reserves, and LPDDR
oversubscription would serve gemv at C2C (~0.38 TB/s) instead of HBM
(~4 TB/s) speed, a ~10× loss, besides touching the forbidden
graph-captured working set (error-900 constraint).

Operational side note (Ryan, 2026-08-25): independent of this verdict, the
GH200 remains an attractive *architecture choice* for the campaign — its
mgh partition sits idle (jobs start in seconds) while H200/H100 partitions
queue, and its measured discrete-path staging is 6.8× faster than H200
PCIe. The verdict rejects unified memory, not the GH200.

The 1,080-step GH200 acceptance run under the unchanged 052 policy (manual
submission after passing probe/smoke/mature IDs; no auto-submits). Stage the
GPU workloads into one multi-stage sbatch where sensible — ARM-partition
queue waits favor combining.

## Deliverables

- Working offline `fm052env-gh200` environment (Phase A exit criterion met).
- GH200 probe/smoke/mature/cross-arch reports under
  `data/fm052_multiarch/gh200/`.
- A written memory-effectiveness verdict (Phase C) with numbers, and either
  a targeted unified-memory change set (construction-time only) or a
  recorded negative result.
- GH200 acceptance verdict feeding the 052/053 architecture comparison.
