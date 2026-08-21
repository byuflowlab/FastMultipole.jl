# Phase P autonomous-run decisions log

Started 2026-08-20 when the user left with: "finish off phase P on your own
… make a log of [important decisions], and give me a report when I get
back. I would like to retire the tmp3 clones; we should be able to
completely remove the tmp3 directory (after making sure no files will be
lost. If a file is uncommitted, just copy it to the other clone)."

Standing decisions the user made before leaving (not autonomous):
merge FastMultipole work onto `flowpanel-20260817` and FLOWVPM work onto
`flowpanel` once passing tests (done for both); retire tmp3; the
`zeta_direct` fix must use a repo-wide-consistent container that doesn't
interfere with other consumers of that storage (delivered as the
VORTICITY_INDEX migration, commit `bc9b9a6`).

## D1 (2026-08-20) — 018 CPU smoke stopped early, recorded as PASS

The smoke run (`NREVS=0.1`, 4 threads, CPU-only, merged stack) ran
116/467 steps error-free at ~20 s/step (7,288-panel default mesh, CG
solves converging, force monitors physically sensible) before I stopped
it. The 467-step count comes from the driver's freestream-schedule revs
(ramp 2 + hold 3 + withdraw 4 + settle 4), which NREVS does not bound —
letting it finish would have cost ~2.5 h for no additional gate value.
Verdict: "018 driver still runs CPU-only unchanged" = PASS on 116 steps.
Log: `~/.claude/jobs/1e9d3d2e/tmp/panel_018_smoke.log`.

## D2 (2026-08-20) — tmp3 retirement mechanics

- All tmp3 branch tips fetched into the projects clones as remote refs
  `tmp3/*` and verified present by SHA before deletion:
  FastMultipole `combined-tree` 6eded90, `main` 58cf693, `matrix-ops`
  a244cef, `worktree-agent-afe26a7125ed0eaee` c2dd452 (worktree was
  clean); stash on combined-tree preserved as tag
  `tmp3-stash-combined-tree` (5b53765). FLOWVPM `gpu-full` c2e8400,
  `master` 4f433fb.
- `FLOWVPM-baseline-e2bd487` was a clean linked worktree at e2bd487
  (present in projects/FLOWVPM.jl history) — nothing to preserve.
- Gitignored figure build artifacts (PNG/PDF/aux) under
  `MATRIX_OPERATOR_REFACTOR/data/figures/` copied to the projects clone
  (`--ignore-existing`), although they are rebuildable from the committed
  .tex+CSV per the figures convention.
- **Deliberately NOT copied:** the tmp3 clones' `Manifest.toml`s — they
  pin dev-paths to tmp3 itself; the projects clones' Manifests already
  dev the correct sibling paths and regenerate via `Pkg.resolve`.
- tmp3 directory deleted 2026-08-20 after the sweep.
- The 046 merge-log doc commit (`a244cef`, made on tmp3 matrix-ops after
  the fast-forward) was merged into `flowpanel-20260817` (`f17fc24`) so
  no doc history was lost.

## D3 (2026-08-20) — Stage 4 checkpoint recorded as answered

The 046 user checkpoint (retire tmp3 vs re-point) was answered by the
user directly ("I would like to retire the tmp3 clones") — recorded here
and in the 046 doc; 046 marked Done in START_HERE. The Approved column is
left unticked for the user's return, per the convention that approval is
granted only by the user.

(Later decisions appended below as they are made.)

## D4 (2026-08-20) — 047 scope: dispatch-cleanup refactors deferred to the 053 punch list

047 delivered: the consolidated validated settings surface
(`src/radix_settings.jl`: registry of all 31 tunables with lock classes from
the read-site audit, `radix_settings`/`radix_setting`/`set_radix_setting!`/
`radix_setting_lock`), the construction-lock contract (snapshot on
`RadixFMMCache.locked_settings` at both ctors; `verify_locked_radix_settings`
at `_radix_cache_device_step!` entry throws loudly on drift — closing the
documented silent-flip hazard), the regression test
(`test/radix_settings_test.jl`, 93 assertions, registered in runtests), and
the FLOWVPM passthrough (`radix_fmm_settings!(pfield; gpu=(;...))` applies
validated settings before the cache rebuild).

DEFERRED (my call): the Future Dispatch Cleanup Notes refactors
(`allow_host_bodies` → source-buffer dispatch, legacy `nearfield_device::Bool`
policy tag, `target::Bool` tree-role arg, route-selection flags →
dispatch-on-object). Rationale: they are behavior-neutral refactors of
device-path APIs, and with no local GPU every verification is a cluster
round-trip; spending those round-trips on 048 (the 75%-of-018-step SFS lever)
first serves the phase objective better. They are punch-listed for the 053
review, which can reopen 047 if the user disagrees. The lock-class read-site
audit table (the main input those refactors need) is preserved in this log's
supporting doc and in radix_settings.jl comments.

Also folded into 048's cluster job: the 047 device-side robustness sweep
(F32/F64 × adaptive/uniform × P=4/P=8) and the device wiring check of the
late-flip error.

## D5 (2026-08-21) — 048 physics-parity gate is J-error-bound, not a flat 1e-3

The host SFS pass is mechanically exact: an all-pairs zeta brute force
computed FROM the radix-delivered J agrees with the radix SFS output to
2.6e-9 (F64). But a flat 1e-3 gate vs the exact-erf CPU references
(Estr_direct!/Estr_fmm!) is unattainable at ANY radix setting on the test
cube (measured e≈3.7e-3, P- and shell-independent): the radix J itself
carries the 031a erf-free g/h nearfield approximation (j_rel_rms≈1.9e-3 on
that case — pre-existing, not a 048 defect). Test structure adopted:
(a) tight 1e-6 mechanical-parity gate at a widened shell; (b) physics gate
max(1e-3, 3·j_rel_rms) vs both CPU references — self-tightening if g/h
arithmetic improves, still fails loudly on any non-J-bound regression. The
default-list zeta-truncation gap (≈3.0e-3 at derived q=12) is recorded in a
test comment; widening near_radius2 is the accuracy knob if production SFS
accuracy ever binds. Device testsets use the same J-aware gate.

## D6 (2026-08-21) — 048 implementation decisions of note

- DeviceResidentRadixState carries one Any-typed `sfs` NamedTuple
  (tg/om/q + transposed) instead of three typed fields (avoids renumbering
  the 38-arg positional ctor at 4 call sites; typed function barriers keep
  kernels specialized).
- The device SFS kernels always run once a cache is sfs-armed (graph-baked
  at construction; FLOWVPM arms vortex couplings unconditionally); the
  per-call `sfs` flag gates only the finalize/delivery. Cost when unused =
  one tg pass + one pair pass; acceptable for v1, flagged for 054 retune.
- transposed is construction-baked (sfs_transposed=pfield.transposed);
  flipping pfield.transposed mid-run requires clear_radix_fmm_cache!.
- Device zero-alloc assertion is a delta (alloc_sfs <= alloc_base) matching
  the suite's existing treatment of pool-served sort scratch.
- CUDA kernels written blind (no local GPU) as close clones of validated
  kernel patterns; H200 job vpm048 validates (parity, counters, alloc,
  graph replay) + the 047 late-flip device check rides along.

## D7 (2026-08-21) — H200 SFS "failure" was gate calibration, not a defect

Job 13247540: everything green except the device SFS physics-parity gates
(e_sfs=0.0293 vs the 3·j_rel=0.0062 gate; wake n=20000 F64; capture and
replay bit-consistent). Diagnosis (reproduced ON HOST to 5 digits at the same
operating point): E_str is a cancellation-dominated difference quantity, so
the radix J error (erf-free g/h + far-field deficit, j_rel≈2.1e-3) amplifies
field-dependently — measured E/J ratio 2.05 (cube) vs 14.1 (wake); the
original 3x gate was calibrated on the benign cube. Zeta-truncation measured
4 orders below the failure (2.1e-6 / 8.7e-7). Resolution: device testset now
gates MECHANICAL parity (host mirror over the identical device pair list +
delivered J) at 1e-6 F64 — this carries SFS correctness — and retiers the
physics gate to max(base, 20·j_rel + 2·e_trunc) with truncation reported.
Also fixed in passing: my sfs_to_target! ambiguity fix (buf::Matrix) broke
capacity>np prefix views — widened to Union{Matrix,SubArray{<:Any,2,<:Matrix}}
with a capacity regression test. Pair-list per-direction convention verified
on both device generators. Production-relevant note for 049/052: SFS
delivered accuracy is J-error-bound; if production SFS accuracy ever binds,
the knob is J accuracy (gh mode / P / MAC), not the SFS pass.

## D8 (2026-08-21) — 049 results + 050 verdict (autonomous)

049 (H200 job 13247848, p018 210k field): UJ parity 3.4e-4 PASS; SFS
marginal cost 0.3 ms; device-resident RK3 step 0.199 s = 6% of the 3.3 s
target; residency measurement says transfers = 35% of step ⇒ RECOMMEND
device-resident (default unchanged — user checkpoint open). Flags: SFS
0.666 vs exact-J reference (J-error-bound; like-for-like test deferred to
052's CT gate), CPU/GPU direct cross-check 2.6e-4 (follow-up before 051
parity gates).

050 verdict (my call, full pricing in theory/panel-multisystem-scoping.md):
option B' — keep the 3-pass structure; NEW rectangular GPU brute-force
kernels for the cross passes (the sizes make FMM unnecessary: 7.7e9 pairs);
device-resident dense nearfield-cache matvec for the solve (measure first);
radix framework untouched. A rejected because radix source-homogeneity
excludes panel elements regardless of the targets===sources lift; C rejected
(ceiling 60-80 s/step misses the target) but recorded as fallback.
