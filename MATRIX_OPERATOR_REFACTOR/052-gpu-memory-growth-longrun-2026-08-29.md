# 052 evidence: unbounded per-step device-memory growth kills long shedding runs (2026-08-29)

Filed from the σ/VPM illustration campaign (FLOWPanel
`plans/sigma_vpm_illustrations_20260827/`). A 40-rev DJI9443 hover screen case
(`scr_p019_s038v_gpu40`, gh200 silo `~/FLOWPanel-018-gpu-gh200`, GH200 mgh-1-1,
resident-S GPU + device pfield, ~6.3 s/step) died at **step 819/1475** with

```
GPU free memory below resident-S emergency margin before gemv 820:
free=4201119744, margin=4294967296
```

while the *physics* was healthy (u_max 17.9 m/s, max ΔtZ 0.075, 194k particles).
Slurm job **13508681**; full log
`~/FLOWPanel-018-gpu-gh200/logs/slurm/slurm-fp-il-s038v-gpu40-13508681.err`
(`source_s_gpu_memory` sample lines, interval 10 gemvs).

## Measured memory trajectory (GH200, total 102.0 GB device-visible)

| point | free | pool_reserved | pool_used |
|---|---|---|---|
| before S upload | 36.6 GB | 0.23 GB | 0.23 GB |
| gemv 60 | 20.35 GB | 16.27 GB | 15.55 GB |
| gemv 220 | 20.35 GB | 16.27 GB | 14.57 GB |
| gemv 300 | 18.67 GB | 17.95 GB | 16.89 GB |
| gemv 540 | 13.51 GB | 23.05 GB | 22.81 GB |
| gemv 820 (death) | 4.20 GB | 32.21 GB | 31.93 GB |

Decomposition:

1. **Startup baseline ~59 GB** (free is already down to 36.6 GB before the S
   upload): CUDA context + kernels, FMM device radix state, cuBLAS workspaces,
   preallocated device pfield. Static. (Also why the default 32 GiB resident-S
   post-upload reserve failed outright on this case — free 36.6 < 10.8 + 32;
   run used `SCR_GPU_RESERVE_GIB=16` via the screen GPU wrapper hook.)
2. **Resident S 10.8 GB**, uploaded once. Static (inside pool_used).
3. **Leak: pool_used grows ~35 MB/step, linearly, from ~step 250 to death**
   (15.5 → 31.9 GB over ~570 steps). This is *outstanding* pool allocations —
   live references, not cached free blocks — i.e. something allocated per step
   is retained. Steps 60–220 are flat (healthy reuse) before the growth begins.

## Hypothesis (unverified — needs a read of the CUDA lifecycle code)

The per-step growth pattern + the fact that the particle count increases every
shedding step implicates size-keyed caching in the CUDA FMM seam
(`CUDA_CACHED_WINDOWS` / `CUDA_GRAPH_LIFECYCLE` / DeviceResidentRadixState):
each new particle-count "window" captures/caches device buffers that are never
evicted, so a monotonically growing N leaks one window per step. The flat
early phase would be the spinup/merge regime where N was not sweeping upward.

## Consequence

Any long shedding run on the current stack has a hard horizon of ~800 steps at
this footprint, independent of physics. If the staged 052 memory-allocation
optimization touches the pool/caching path, this is a ready-made regression
test: rerun `scr_p019_s038v_gpu40` (case arm + wrapper already in the gh200
silo, `examples/run_p018_screen_gpu.slurm.sh`) and require pool_used to plateau
after spinup. The physics run itself was re-launched on CPU
(`scr_p019_s038v_cpu40`, main checkout) and is not blocked on this.
