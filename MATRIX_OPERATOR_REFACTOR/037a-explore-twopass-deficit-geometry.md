# 037a Nearfield Reduction I: Two-Pass Deficit Splitting at Co-Designed Geometry

## Status and Entry Gate

**Staged by user direction `2026-08-13`; not started.** Precedes `038` (the
adaptive-octree derivation) in the roadmap order by that same direction.

Entry gate: `037` Done and clear-context approved (satisfied `2026-08-13`).
No production default changes without explicit user approval; exploration,
measurement, and pre-registered sweeps do not need per-step approval.

## Motivation

The nearfield is 66-95% of the shipped U/J solve and IS the critical path
(unlike the far-field levers of `037` — see the standing lesson: price
levers against the overlapped critical path, where nearfield reductions
translate ~1:1 until the far-field chain surfaces at roughly 2-5 ms). The
pair count exists because the σ-adequacy gate
`g_min(q)·h_leaf >= rho_t·sigma` forces the direct stencil to reach
`rho_t·sigma ≈ 3.7σ` so that everything beyond it is pure singular math.

Hypothesis under test: **two-pass deficit splitting decouples the stencil
from `rho_t`.** Singular `1/r` math runs everywhere the FMM/stencil
geometry requires (expansion-validity reach only — the scalar path runs
q² = 4-6 one level deeper); a pairwise deficit correction
(`(g(ρ)-1)`-weighted Biot-Savart, smooth, short-ranged) runs only within
`rho_c·sigma` with `rho_c ≈ 2`. Naive scaling `(rho_c/rho_t)^3 ≈ 0.16`
suggests up to ~6x fewer correction pairs plus a deeper admissible tree —
but `032a` Stage D measured partitioned beating two-pass *at matched
geometry*, so the open question is precisely the co-design: two-pass at its
OWN best (ell, q, rho_c, P). This is the same coupled-lever trap that 035
cycle 3 exposed for P/rho_t/stencil; resolve it the same way —
measurement-first.

`TwoPassVortex` already exists as a production kernel surface
(`direct_kernel=:twopass` reaches it from FLOWVPM); the deficit-kernel
theory is in `031a`/`theory/kernel-splitting-nearfield.md`.

## Objective

Determine, by pre-registered measurement, whether two-pass deficit
splitting at co-designed geometry reduces the end-to-end U/J solve by a
reportable margin on the campaign cases and on a realistic rotor wake; ship
it as a default only on explicit user approval; produce an exact
per-approach speedup report; and render the go/no-go evidence for `037b`
(mesh/FFT deficit evaluation).

## Campaign

1. **Realistic rotor-wake case (shared deliverable, consumed by `037b`).**
   Construct a rotor-wake particle field from an actual rotor — e.g. the
   DJI 9443 rotor available in `~/Dropbox/research/projects/FLOWPanel.jl`
   (`examples/dji9443_*.jl`; blade circulation seeding a FLOWVPM wake, or a
   FLOWUnsteady-generated snapshot if one is more direct). Requirements:
   deterministic construction (seeded), n at 1e5 and 1e6, σ distribution
   and aspect ratio documented (real rotor wakes are longer and more
   σ-heterogeneous than the AR=5 helical cylinder of 033), sampled-direct
   velocity references computed once with the 033 erf-oracle machinery and
   sha256-checksummed. This case also produces the multi-scale-density
   evidence (or lack of it) that `038`'s entry gate asks for — record that
   verdict explicitly either way.
2. **Accuracy instrument first.** Extend the 3C error-decomposition oracle
   to the two-pass field: separate truncated-deficit error (pairs beyond
   `rho_c·sigma` whose correction is dropped) from FMM truncation error;
   governing gate remains the conservative sum `< 1e-3` on sampled velocity
   RMS, J diagnostic. Pre-register the gate before any timing sweep.
3. **Pre-registered co-design sweep** (3A pattern; cube + wake + rotor wake
   at the representative n): grid over `rho_c` (≈1.7-2.5), depth
   (adequacy under `rho_c`, incl. one-level-deeper candidates), leaf `q`
   (down to expansion-validity), `P` (literature 4/5/6), both precisions at
   the winner. Same-job anchors at the shipped defaults
   (P5/3.668/partitioned/dense, cubic). All standard gates per row
   (checksummed references, flat 023 counters, zero recurring allocation).
4. **Optimization cycles** (035 rules): ranked levers with a ≥5% expected
   end-to-end bar priced against the overlapped critical path; production
   implementation only with explicit user approval; realized-vs-expected
   recorded per cycle.
5. **Definitive speedup report.** Per case (cube / wake / rotor wake) and
   scale: U/J solve and RK3 medians (15 warmed reps policy), speedup vs the
   shipped-default anchor measured in the same job, per-stage profiles
   showing where the reduction landed, accuracy decomposition per winner,
   and the co-design map (what `rho_c` bought at which depth). Figures per
   the 024a conventions. **Every claimed speedup must be attributable to
   this approach alone** — anchors and winners share everything except the
   split/geometry under test.
6. **037b gate verdict.** Recommend opening `037b` iff the measured
   evidence says mesh evaluation of the deficit beats pair evaluation —
   e.g. the correction-pass pair cost remains a material fraction of the
   solve at the best `rho_c`, or accuracy forces `rho_c` high enough to
   cap the gain. Recommend against iff two-pass captures most of the
   available reduction or the deficit cost is already negligible. Record
   the quantitative basis either way.

## Dependencies and Reading

- `037` (Done + approved), `035` Final Report §3 and cycle-3 records (the
  co-design method and the overlap lesson), `031a` +
  `theory/kernel-splitting-nearfield.md` (deficit kernel, `rho_c` hybrid),
  `032a` Stage D (the matched-geometry two-pass result being superseded),
  `033` (case/reference machinery), `benchmark_035_gpu.jl` harness
  (extend; keep the campaign-file/CSV discipline).
- `~/Dropbox/research/projects/FLOWPanel.jl` — rotor geometry/circulation
  source for the wake case (read its CLAUDE.md/examples before building).

## Verification Gates

- Sampled velocity RMS ≤ 1e-3 (conservative sum form once the decomposition
  instrument exists) against sha256-checksummed references on every
  reported row; J logged as diagnostic.
- 023 counter/allocation contracts unchanged; scalar 028/030 no-regression
  reruns mandatory if FastMultipole `src/` is touched.
- Speedup claims only from same-job anchor/winner pairs; the report states
  the warmup/repeat/median policy and prices every lever against the
  overlapped critical path.
