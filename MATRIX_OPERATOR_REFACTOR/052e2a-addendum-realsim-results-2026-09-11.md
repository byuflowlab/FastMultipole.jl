# 052e.2a real-simulation addendum — RESULTS (v2 registered run)

Drafted 2026-09-12 by agent; ruling pending (Ryan). Dataset: **v2 registered run only** — no mixing with the failed 2026-09-10 run.

## Provenance

- Run: launched 2026-09-11 22:58 MDT (nohup-detached, PID 59407), completed 2026-09-12 ~12:00 MDT (~13.1 h wall, laptop, 4 threads), julia 1.12.5.
- Harness: `scripts/addendum_052e2a_realsim_2026-09-11.jl`, SHA-256 (12) `e0e76531e492`.
- Pins: FLOWPanel `8dce66c09d7bd9a89680ad72fd023f417273411b` DIRTY (tracked-diff `ce3423d51c15`); FastMultipole `da1bd13a07db3ed3914720b814bd14882e7670f7` DIRTY (tracked-diff `2ebca3df17eb`). Laptop formulation-proof tier, not an official campaign.
- Outputs: `data/052e2a-addendum-realsim-v2/` (`run_20260911_2258.log`, `gates.txt`, per-case `gamma_*/steps_*/grdiag_*` CSVs, `p4_L*.csv`, `trace_{A,B}_L*.csv`).
- Prereg: `052e2a-addendum-realsim-preregistration-2026-09-08.md` (LOCKED). Supersession: `052e2a-addendum-realsim-supersession-2026-09-11.md`.

## Run history (failed run + supersession)

The original registered run (2026-09-10 20:03) **failed G1** at BW2-VTS-L4: block Gauss–Seidel produced a nonfinite physical residual at outer iteration 1, ~5.85 h in, L1–L3 complete. Root cause (diagnosed 2026-09-11, Ryan-authorized): vortex-stretching runaway of the particle wake at L4 core sizes under `OverlapPPS(1.3,2)` — particle |Γ| amplified ~10×/step (1.3e6→4.2e8 over steps 37–40 of a faithful repro), ReformulatedVPM σ driven negative (|σ| to 8e4), particles ejected km from the body, → Inf/NaN wake velocity → NaN VTS RHS. Same signature as the 052 gpu40 ignition root cause. Repro at overlap 2.4: all 61 steps clean, max|Γ| 0.15–0.21, σ ∈ [0.089, 0.93] positive.

**Supersession (Ryan ruling, 2026-09-11):** W2 shedding `OverlapPPS(1.3,2)` → `OverlapPPS(2.4,2)`, both arms, all levels — the only change. Gates G1–G4 and predictions P1–P4 unchanged. Failed-run evidence preserved untouched in `data/052e2a-addendum-realsim/`.

## Fixture (from LOCKED prereg, abbreviated)

Capped NACA0012 rectangular wing (b=2.7 m, c=0.76 m, t/c=0.12), AOA=30°, |U∞|=1. Levels L1–L4 = 1,744 / 3,816 / 8,960 / 19,384 panels (Neumann referee body uncapped: 1,120 / 2,400 / 5,400 / 11,880). Routes: R-VTS (Dirichlet capped, `VelocityThroughSources`), R-GR (Dirichlet capped, `GreenReconstruction(gauge=:area_mean)`, production Householder), R-NEU (Neumann uncapped, doublets only). Wakes: W1 = doublet-panel wake; W2 = vortex-particle wake (production shedding, `OverlapPPS(2.4,2)` per supersession). Phases: A = prescribed/frozen flat wake fixed point (NITER_A=80, TOL_A=1e-8); B = free convecting wake via `simulate!` (NSAMP_B=61, DT=0.38). Solver: `Backslash` dense, outer hard-fail on. Kutta: `RigidTransitionAttachment`+`JumpKutta` everywhere.

Case labels below: `{A,B}{W1,W2}-{GR,NEU,VTS}-L{1..4}` (Phase A W2 not run per R9, allowed by lock L2). 36 solves total.

## Gate report

| Gate | Result | Raw value | Threshold |
|---|---|---|---|
| G1 | **PASS** | all 36 solves converged (hard-fail on; all Phase A fixed points converged) | required |
| G2 | **PASS** | in-solve wake-node hashes = prescribed, all levels: `a175dd6c180f7d43` / `42cbe2a3abd10c4a` / `bd7dcd2c02f94e8c` / `47f91e5c7a904745` | required |
| G3 | **PASS** | max Green residual 1.22e-14; max gauge defect 2.20e-16 (measurement solves) | res ≤ 1e-10, gd ≤ 1e-11 |
| G4 | **FAIL** | Phase B E_q = [2.133e-1, 1.533e-1, 9.665e-2, **5.851e-2**] at L1–L4: monotone ✓, finest ≤ 1e-2 ✗ (exceeds by ~5.9×) | finest ≤ 1e-2, monotone |

G4 note: Phase A recorded E_q = [1.968e-1, 1.385e-1, 8.435e-2, 4.895e-2]. Observed decay ≈ N^(−0.58) overall (N^(−0.70) on L3→L4) — the ~0.5-order rate flagged by the smoke test and by the failed run's L1–L3. Per the LOCKED prereg, a G4 failure stops the tier and is reported with raw numbers before any code or fixture change; no in-place gate retuning. This is a registered outcome.

The failed run's crash case, BW2-VTS-L4, completed cleanly under overlap 2.4: Gtot = −3.753445 vs BW1-VTS-L4 −3.753336 (0.003% apart), drift 7.37e-5.

## Γ(y) tables (dy-weighted spanwise circulation)

Figures: `~/Dropbox/research/notebooks/img/20260912_052e2a_realsim/gamma_L4.tex` (+ `eq_convergence.tex`), backing CSVs in same-named subdirectories.

### L1 (7 stations)
| y | AW1-GR | AW1-NEU | AW1-VTS | BW1-GR | BW1-NEU | BW1-VTS | BW2-GR | BW2-NEU | BW2-VTS |
|---|---|---|---|---|---|---|---|---|---|
| -1.1571 | -0.55053 | -0.55663 | -1.09177 | -0.52205 | -0.51991 | -0.97929 | -0.52152 | -0.51943 | -0.97768 |
| -0.7714 | -0.74767 | -0.76169 | -1.48000 | -0.72018 | -0.72733 | -1.37333 | -0.71942 | -0.72661 | -1.37085 |
| -0.3857 | -0.83381 | -0.85214 | -1.64898 | -0.80873 | -0.82125 | -1.55494 | -0.80784 | -0.82041 | -1.55197 |
| +0.0000 | -0.86193 | -0.88191 | -1.70395 | -0.83770 | -0.85225 | -1.61463 | -0.83676 | -0.85136 | -1.61148 |
| +0.3857 | -0.84451 | -0.86385 | -1.66966 | -0.81926 | -0.83292 | -1.57532 | -0.81837 | -0.83207 | -1.57235 |
| +0.7714 | -0.77099 | -0.78710 | -1.52521 | -0.74279 | -0.75220 | -1.41579 | -0.74200 | -0.75147 | -1.41331 |
| +1.1571 | -0.58948 | -0.59857 | -1.16773 | -0.55850 | -0.55930 | -1.04320 | -0.55795 | -0.55880 | -1.04156 |

### L2 (10 stations)
| y | AW1-GR | AW1-NEU | AW1-VTS | BW1-GR | BW1-NEU | BW1-VTS | BW2-GR | BW2-NEU | BW2-VTS |
|---|---|---|---|---|---|---|---|---|---|
| -1.2150 | -0.48171 | -0.48203 | -0.95836 | -0.45969 | -0.45432 | -0.87036 | -0.45919 | -0.45386 | -0.86906 |
| -0.9450 | -0.67586 | -0.68188 | -1.34299 | -0.65167 | -0.65230 | -1.24731 | -0.65093 | -0.65159 | -1.24525 |
| -0.6750 | -0.77862 | -0.78853 | -1.54609 | -0.75540 | -0.76060 | -1.45647 | -0.75450 | -0.75974 | -1.45387 |
| -0.4050 | -0.83492 | -0.84720 | -1.65717 | -0.81273 | -0.82078 | -1.57389 | -0.81173 | -0.81981 | -1.57092 |
| -0.1350 | -0.86139 | -0.87487 | -1.70930 | -0.83971 | -0.84919 | -1.62938 | -0.83865 | -0.84818 | -1.62621 |
| +0.1350 | -0.86427 | -0.87798 | -1.71492 | -0.84248 | -0.85222 | -1.63479 | -0.84143 | -0.85120 | -1.63161 |
| +0.4050 | -0.84385 | -0.85683 | -1.67457 | -0.82130 | -0.83009 | -1.59048 | -0.82029 | -0.82912 | -1.58748 |
| +0.6750 | -0.79447 | -0.80558 | -1.57703 | -0.77051 | -0.77696 | -1.48525 | -0.76959 | -0.77610 | -1.48261 |
| +0.9450 | -0.70017 | -0.70789 | -1.39057 | -0.67453 | -0.67686 | -1.28943 | -0.67377 | -0.67615 | -1.28732 |
| +1.2150 | -0.51569 | -0.51794 | -1.02515 | -0.49103 | -0.48738 | -0.92537 | -0.49051 | -0.48690 | -0.92401 |

### L3 (15 stations)
| y | AW1-GR | AW1-NEU | AW1-VTS | BW1-GR | BW1-NEU | BW1-VTS | BW2-GR | BW2-NEU | BW2-VTS |
|---|---|---|---|---|---|---|---|---|---|
| -1.2600 | -0.40957 | -0.40592 | -0.81648 | -0.39451 | -0.38732 | -0.75545 | -0.39406 | -0.38693 | -0.75480 |
| -1.0800 | -0.59004 | -0.59000 | -1.17534 | -0.57096 | -0.56704 | -1.09956 | -0.57028 | -0.56644 | -1.09846 |
| -0.9000 | -0.69773 | -0.70079 | -1.38920 | -0.67774 | -0.67717 | -1.31141 | -0.67689 | -0.67641 | -1.30993 |
| -0.7200 | -0.76791 | -0.77326 | -1.52842 | -0.74801 | -0.75002 | -1.45292 | -0.74703 | -0.74915 | -1.45111 |
| -0.5400 | -0.81466 | -0.82162 | -1.62108 | -0.79505 | -0.79893 | -1.54860 | -0.79399 | -0.79797 | -1.54650 |
| -0.3600 | -0.84493 | -0.85298 | -1.68104 | -0.82559 | -0.83072 | -1.61102 | -0.82446 | -0.82970 | -1.60872 |
| -0.1800 | -0.86243 | -0.87114 | -1.71570 | -0.84323 | -0.84913 | -1.64717 | -0.84206 | -0.84807 | -1.64473 |
| +0.0000 | -0.86905 | -0.87803 | -1.72878 | -0.84984 | -0.85604 | -1.66065 | -0.84866 | -0.85497 | -1.65817 |
| +0.1800 | -0.86540 | -0.87430 | -1.72152 | -0.84602 | -0.85211 | -1.65267 | -0.84484 | -0.85105 | -1.65021 |
| +0.3600 | -0.85100 | -0.85944 | -1.69296 | -0.83127 | -0.83681 | -1.62223 | -0.83014 | -0.83579 | -1.61988 |
| +0.5400 | -0.82411 | -0.83168 | -1.63966 | -0.80389 | -0.80839 | -1.56591 | -0.80282 | -0.80742 | -1.56375 |
| +0.7200 | -0.78122 | -0.78739 | -1.55458 | -0.76040 | -0.76324 | -1.47690 | -0.75941 | -0.76236 | -1.47501 |
| +0.9000 | -0.71555 | -0.71966 | -1.42428 | -0.69425 | -0.69470 | -1.34272 | -0.69339 | -0.69394 | -1.34116 |
| +1.0800 | -0.61323 | -0.61439 | -1.22106 | -0.59230 | -0.58951 | -1.13879 | -0.59160 | -0.58890 | -1.13761 |
| +1.2600 | -0.43822 | -0.43559 | -0.87310 | -0.42088 | -0.41453 | -0.80267 | -0.42041 | -0.41412 | -0.80194 |

### L4 (22 stations)
| y | AW1-GR | AW1-NEU | AW1-VTS | BW1-GR | BW1-NEU | BW1-VTS | BW2-GR | BW2-NEU | BW2-VTS |
|---|---|---|---|---|---|---|---|---|---|
| -1.2886 | -0.34961 | -0.34354 | -0.69779 | -0.33974 | -0.33148 | -0.65667 | -0.33940 | -0.33121 | -0.65700 |
| -1.1659 | -0.51217 | -0.50798 | -1.02177 | -0.49824 | -0.49133 | -0.96581 | -0.49771 | -0.49090 | -0.96620 |
| -1.0432 | -0.61608 | -0.61411 | -1.22869 | -0.60009 | -0.59536 | -1.16625 | -0.59941 | -0.59481 | -1.16661 |
| -0.9205 | -0.68954 | -0.68948 | -1.37489 | -0.67265 | -0.66993 | -1.31080 | -0.67186 | -0.66928 | -1.31108 |
| -0.7977 | -0.74380 | -0.74527 | -1.48283 | -0.72657 | -0.72553 | -1.41926 | -0.72568 | -0.72479 | -1.41945 |
| -0.6750 | -0.78465 | -0.78732 | -1.56406 | -0.76732 | -0.76762 | -1.50183 | -0.76635 | -0.76682 | -1.50192 |
| -0.5523 | -0.81547 | -0.81907 | -1.62532 | -0.79813 | -0.79949 | -1.56458 | -0.79710 | -0.79862 | -1.56458 |
| -0.4295 | -0.83838 | -0.84267 | -1.67085 | -0.82107 | -0.82323 | -1.61146 | -0.81999 | -0.82231 | -1.61136 |
| -0.3068 | -0.85476 | -0.85956 | -1.70338 | -0.83748 | -0.84021 | -1.64505 | -0.83636 | -0.83926 | -1.64488 |
| -0.1841 | -0.86548 | -0.87062 | -1.72468 | -0.84821 | -0.85132 | -1.66707 | -0.84706 | -0.85035 | -1.66685 |
| -0.0614 | -0.87109 | -0.87641 | -1.73582 | -0.85379 | -0.85712 | -1.67857 | -0.85263 | -0.85613 | -1.67832 |
| +0.0614 | -0.87185 | -0.87721 | -1.73732 | -0.85449 | -0.85786 | -1.68005 | -0.85333 | -0.85687 | -1.67979 |
| +0.1841 | -0.86777 | -0.87302 | -1.72920 | -0.85032 | -0.85355 | -1.67154 | -0.84917 | -0.85257 | -1.67129 |
| +0.3068 | -0.85863 | -0.86362 | -1.71102 | -0.84104 | -0.84397 | -1.65258 | -0.83991 | -0.84302 | -1.65237 |
| +0.4295 | -0.84392 | -0.84849 | -1.68178 | -0.82617 | -0.82861 | -1.62219 | -0.82508 | -0.82769 | -1.62203 |
| +0.5523 | -0.82280 | -0.82676 | -1.63981 | -0.80488 | -0.80661 | -1.57869 | -0.80384 | -0.80573 | -1.57860 |
| +0.6750 | -0.79396 | -0.79707 | -1.58245 | -0.77586 | -0.77663 | -1.51953 | -0.77489 | -0.77581 | -1.51952 |
| +0.7977 | -0.75531 | -0.75732 | -1.50559 | -0.73713 | -0.73663 | -1.44081 | -0.73623 | -0.73588 | -1.44089 |
| +0.9205 | -0.70357 | -0.70412 | -1.40263 | -0.68549 | -0.68339 | -1.33646 | -0.68468 | -0.68273 | -1.33662 |
| +1.0432 | -0.63301 | -0.63172 | -1.26219 | -0.61556 | -0.61148 | -1.19632 | -0.61486 | -0.61091 | -1.19654 |
| +1.1659 | -0.53239 | -0.52884 | -1.06183 | -0.51675 | -0.51042 | -1.00066 | -0.51619 | -0.50997 | -1.00093 |
| +1.2886 | -0.37277 | -0.36700 | -0.74374 | -0.36128 | -0.35321 | -0.69656 | -0.36091 | -0.35292 | -0.69680 |

## Pairwise-gap refinement tables (P1/P2 evidence)

Gaps vs R-NEU, dy-weighted rms (grms) and max (gmax) of Γ(y), plus dGtot and dCL.

### GR − NEU (P1)
| case | level | grms | gmax | dGtot | dCL |
|---|---|---|---|---|---|
| AW1 | L1 | 2.0233e-02 | 2.2658e-02 | +3.9724e-02 | +0.03872 |
| AW1 | L2 | 1.3293e-02 | 1.5621e-02 | +2.4238e-02 | +0.02362 |
| AW1 | L3 | 8.3776e-03 | 1.0224e-02 | +1.2805e-02 | +0.01248 |
| AW1 | L4 | 5.3902e-03 | 6.9266e-03 | +4.1953e-03 | +0.00409 |
| BW1 | L1 | 1.3619e-02 | 1.7074e-02 | +2.1577e-02 | +0.02103 |
| BW1 | L2 | 9.1318e-03 | 1.1424e-02 | +1.1249e-02 | +0.01096 |
| BW1 | L3 | 6.4511e-03 | 8.3983e-03 | +3.9113e-03 | +0.00381 |
| BW1 | L4 | 5.4673e-03 | 9.6256e-03 | −2.1165e-03 | −0.00206 |
| BW2 | L1 | 1.3686e-02 | 1.7148e-02 | +2.1708e-02 | +0.02116 |
| BW2 | L2 | 9.1773e-03 | 1.1483e-02 | +1.1359e-02 | +0.01107 |
| BW2 | L3 | 6.5215e-03 | 8.3355e-03 | +4.1756e-03 | +0.00407 |
| BW2 | L4 | 5.4768e-03 | 9.5522e-03 | −1.7273e-03 | −0.00168 |

### VTS − NEU (P2)
| case | level | grms | gmax | dGtot | dCL |
|---|---|---|---|---|---|
| AW1 | L1 | 9.3903e-01 | 9.3211e-01 | −1.9229e+00 | −1.87421 |
| AW1 | L2 | 9.6010e-01 | 9.5325e-01 | −1.9320e+00 | −1.88300 |
| AW1 | L3 | 9.7557e-01 | 9.6893e-01 | −1.9381e+00 | −1.88895 |
| AW1 | L4 | 9.8710e-01 | 9.8051e-01 | −1.9460e+00 | −1.89671 |
| BW1 | L1 | 8.8782e-01 | 8.9455e-01 | −1.7324e+00 | −1.68848 |
| BW1 | L2 | 9.1424e-01 | 9.1828e-01 | −1.7663e+00 | −1.72158 |
| BW1 | L3 | 9.3790e-01 | 9.3991e-01 | −1.8023e+00 | −1.75667 |
| BW1 | L4 | 9.5802e-01 | 9.5842e-01 | −1.8369e+00 | −1.79040 |
| BW2 | L1 | 8.8625e-01 | 8.9283e-01 | −1.7276e+00 | −1.68386 |
| BW2 | L2 | 9.1296e-01 | 9.1683e-01 | −1.7619e+00 | −1.71729 |
| BW2 | L3 | 9.3762e-01 | 9.3944e-01 | −1.7998e+00 | −1.75417 |
| BW2 | L4 | 9.6012e-01 | 9.6038e-01 | −1.8391e+00 | −1.79247 |

## Oracle-check table (P3; G4 metric)

R-GR, W1. E_q = dy-weighted rms source error vs oracle; Einf = max-norm (all / excluding TE-adjacent — identical here); xchk = cross-check residuals; sigx = σ cross-check; clr = clearance interval.

| phase | level | N | E_q | Einf(all) | Einf(ex) | xchk | sigx |
|---|---|---|---|---|---|---|---|
| A | L1 | 1744 | 1.9680e-01 | 1.0764e+00 | 1.0764e+00 | (1.02e-12, 1.30e-15) | 1.28e-16 |
| A | L2 | 3816 | 1.3854e-01 | 8.1600e-01 | 8.1600e-01 | (2.64e-12, 1.66e-15) | 1.24e-16 |
| A | L3 | 8960 | 8.4347e-02 | 5.6452e-01 | 5.6452e-01 | (1.49e-12, 1.84e-15) | 1.25e-16 |
| A | L4 | 19384 | 4.8951e-02 | 3.7653e-01 | 3.7653e-01 | (2.97e-12, 2.17e-15) | 1.26e-16 |
| B | L1 | 1744 | 2.1333e-01 | 1.2346e+00 | 1.2346e+00 | (3.10e-14, 1.60e-15) | 1.31e-16 |
| B | L2 | 3816 | 1.5330e-01 | 1.0054e+00 | 1.0054e+00 | (4.45e-14, 2.00e-15) | 1.30e-16 |
| B | L3 | 8960 | 9.6650e-02 | 7.7031e-01 | 7.7031e-01 | (4.77e-14, 2.17e-15) | 1.25e-16 |
| B | L4 | 19384 | 5.8507e-02 | 5.7989e-01 | 5.7989e-01 | (6.80e-14, 2.64e-15) | 1.26e-16 |

## P4 ratios (W2 vs W1 trace difference)

rmsA(q_W2 − q_W1)/rmsA(q_W1), R-GR, matched Phase B final step:

| level | ratio |
|---|---|
| L1 | 7.3735e-03 |
| L2 | 9.3952e-03 |
| L3 | 1.2851e-02 |
| L4 | 1.6798e-02 |

Small (≤1.7%) but **growing** under refinement, continuing the failed run's L1–L3 trend (3.6e-3/5.5e-3/8.1e-3 there is NOT comparable — that run used overlap 1.3; v2 values stand alone).

## Solve scalars and diagnostics

Gtot / CL per case (Phase A dA = final fixed-point delta; Phase B drift = late-window circulation drift; np = final particle count; kutta_x = 0 everywhere, R8 xcheck):

| case | L1 Gtot / CL | L2 Gtot / CL | L3 Gtot / CL | L4 Gtot / CL |
|---|---|---|---|---|
| AW1-GR | −2.00529 / −1.95448 | −1.98476 / −1.93446 | −1.97011 / −1.92019 | −1.96327 / −1.91352 |
| AW1-NEU | −2.04502 / −1.99319 | −2.00900 / −1.95809 | −1.98292 / −1.93267 | −1.96747 / −1.91761 |
| AW1-VTS | −3.96796 / −3.86741 | −3.94096 / −3.84109 | −3.92098 / −3.82162 | −3.91349 / −3.81431 |
| BW1-GR | −1.93213 / −1.88316 | −1.92214 / −1.87343 | −1.91771 / −1.86911 | −1.91850 / −1.86989 |
| BW1-NEU | −1.95370 / −1.90419 | −1.93339 / −1.88440 | −1.92162 / −1.87292 | −1.91639 / −1.86782 |
| BW1-VTS | −3.68608 / −3.59267 | −3.69974 / −3.60598 | −3.72396 / −3.62959 | −3.75334 / −3.65822 |
| BW2-GR | −1.93006 / −1.88115 | −1.91986 / −1.87121 | −1.91521 / −1.86667 | −1.91610 / −1.86754 |
| BW2-NEU | −1.95177 / −1.90231 | −1.93122 / −1.88228 | −1.91938 / −1.87074 | −1.91437 / −1.86586 |
| BW2-VTS | −3.67941 / −3.58616 | −3.69316 / −3.59957 | −3.71916 / −3.62491 | −3.75345 / −3.65833 |

Diagnostics: Phase A dA ∈ [2.9e-14, 2.5e-13] (all converged well under TOL_A=1e-8, 80 iterations); Phase B drift ∈ [4.9e-5, 8.1e-5] across all 24 B cases; W2 particle counts np = 1680/2352/3472/5040 at L1–L4; kutta_x = 0 everywhere. Phase A delta histories show clean geometric contraction (4.2e-1 at iter 1 → ~1e-13 by iter 79 at every level).

## Call chains (as registered in gates.txt)

`simulate!`/`Backslash`/`DirectBackend` | GR: `state.green.q` + `_green_lambda` | telemetry via `step_telemetry_callback` | G3 by harness: `_source_potential!` + `_green_B_product!` | oracle: `pnl.induced` sums.

## Realization notes (R1–R10 from gates.txt; R12 from v2 harness header)

- R1: Phase A fixed point via `maneuver!`.
- R2: Phase B production defaults.
- R3: CL via Kutta–Joukowski.
- R4: G3 evaluated at the measurement solve.
- R5: G4 on the Phase B W1 sequence.
- R6: G2 on in-solve hashes.
- R7: TE mask from shedding nodes.
- R8: JumpKutta c==0 by construction + `_get_wakestrength_mu` cross-check.
- R9: Phase A W2 frozen-particle variant NOT RUN (allowed by lock L2).
- R10: stage-1 metrics/oracle definitions reused.
- R12: v2 harness header note recording the supersession (overlap 2.4, output dir v2).

## Evidence summary for ruling (agent recommendation — Ryan rules)

- **P1** (GR ≈ NEU, gap shrinking/plateauing): Phase A gap shrinks monotonically through L4 (grms 2.02e-2 → 5.39e-3, ~1.9e-3 per doubling at the end). Phase B gaps shrink L1→L3 then plateau at ~5.5e-3 grms at L4 with dGtot crossing zero (BW1: +3.9e-3 → −2.1e-3). "Shrinking/plateauing" as registered. **Recommend CONFIRMED.**
- **P2** (VTS ≠ NEU, persistent gap): grms 0.89–0.99 at every level and mildly growing; |Γ| ~92–99% above referee; dCL ≈ −1.7 to −1.9 throughout. **Recommend CONFIRMED.**
- **P3** (Green-trace reconstruction matches true trace, converging): monotone convergence at every level in both phases (Phase A 1.97e-1 → 4.90e-2, Phase B 2.13e-1 → 5.85e-2), but decay is ~N^(−0.6), not the rate implied by the 1e-2-at-L4 gate — G4 FAIL on threshold, monotone clause satisfied. Whether "converging under refinement at body-discretization level" is met at these resolutions is the judgment call. **Recommend: Ryan rules CONFIRMED-with-caveat vs INCONCLUSIVE; agent notes the monotone trend is unambiguous, the registered threshold was not met.**
- **P4** (W2 trace ≈ W1 trace, difference = wake-representation error only): ratio ≤ 1.68e-2 (small) but growing under refinement (7.4e-3 → 1.68e-2), not shrinking. Small in absolute terms; trend direction was not registered. **Recommend: Ryan rules; agent notes both facts.**

### Ruling (Ryan, 2026-09-12)

- P1: ☑ CONFIRMED
- P2: ☑ CONFIRMED
- P3: ☑ CONFIRMED
- P4: ☑ CONFIRMED
- CONTINUE to 052e.3 (hybrid fixture): ☑ YES

Ruled "pass" on P1–P4 with CONTINUE (Ryan, 2026-09-12, in-session). G4
remains a recorded FAIL on its registered threshold; the ruling accepts the
monotone convergence evidence as confirming the predictions.

## Post-ruling external review (2026-09-12)

An independent approach review was commissioned and delivered; Ryan's
disposition: **no course change for now — noted for possible return if
issues arise.**

1. `052e2a-approach-review-2026-09-12.md` — assesses Green reconstruction,
   convergence evidence, gauges, and iterative conditioning; recommends
   preconditioned gauge-consistent GMRES and direct panel-wake potential
   evaluation; proposes investigating a second-kind source-Neumann
   formulation with separate circulation/Kutta unknowns (that lifting
   formulation remains incomplete).
2. `052e2a-unsteady-pressure-followup-2026-09-12.md` — potential
   differentiation and acceleration-potential methods; recommends surface
   Euler pressure reconstruction for rotational particle wakes with
   multigrid preconditioning; derives six precomputed adjoint solves for
   inexpensive rigid-body force/moment recovery; uniform pressure-gauge
   shifts cancel from total closed-body loads.
3. `052e2a-vortical-field-layer-potential-clarification-2026-09-12.md` —
   qualifies the first review: a harmonic body correction preserves the
   actual wake's vorticity; an auxiliary Green-reconstructed trace can
   correctly enforce impermeability even when it is not the physical wake
   potential — so its time derivative is not automatically valid for
   Bernoulli pressure.
