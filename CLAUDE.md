# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run tests:**
```bash
julia --project=. -e 'using Pkg; Pkg.test()'
# Or run a single test file:
julia --project=. test/fmm_test.jl
```

**Run tests with threads:**
```bash
julia --project=. --threads=4 -e 'using Pkg; Pkg.test()'
```

**Build docs:**
```bash
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

**Activate environment in REPL:**
```julia
using Pkg; Pkg.activate("."); Pkg.instantiate()
using FastMultipole
```

The test suite in `test/runtests.jl` includes these test files in order: `auxilliary_test`, `direct_test`, `harmonics_test`, `rotate_test`, `bodytomultipole_test`, `multipole_power_test`, `translate_multipole_test`, `translate_multipole_to_local_test`, `translate_local_test`, `evaluate_expansions_test`, `lamb_helmholtz_test`, `tree_test`, `dynamic_expansion_order_test`, `interaction_list_test`, `fmm_test`, `solve_test`.

## Architecture

FastMultipole.jl is a pure-Julia implementation of the **Fast Multipole Method (FMM)** for N-body problems governed by the Laplace (1/r) Green's function. It supports scalar-plus-vector potential problems (v = ∇φ + ∇×ψ), covering gravitational, electrostatic, magnetostatic, and vortex-flow applications.

### Primary Entry Points

- `fmm!(system)` / `fmm!(target_systems, source_systems)` — main accelerated N-body solver
- `direct!(system)` — brute-force O(N²) reference (also parallelized)
- `tune_fmm(...)` — auto-tunes expansion order, leaf size, and multipole acceptance criterion
- `build_interaction_lists(...)` — constructs M2L and direct interaction lists

### User Interface Pattern

Users define their own system type and overload ~6 dispatch functions in `compatibility.jl`:
- `source_system_to_buffer!(buffer, system, ...)` — pack body data into flat Matrix
- `get_position(system, i)` — return SVector{3} position of body i
- `get_n_bodies(system)` — return number of bodies
- `body_to_multipole!(branch, system, ...)` — accumulate multipole coefficients
- `direct!(target, source, ...)` — nearfield direct interaction
- `buffer_to_target_system!(system, buffer, ...)` — write results back from buffer
- `has_vector_potential(system)` — whether system uses Lamb-Helmholtz χ channel

See `test/gravitational.jl` as the canonical example of implementing a user system.

### FMM Pipeline (`fmm.jl`)

1. **Upward pass**: B2M (body-to-multipole at leaves) + M2M (multipole-to-multipole up tree)
2. **Horizontal pass**: M2L (multipole-to-local, one per interaction pair)
3. **Downward pass**: L2L (local-to-local down tree) + L2B (local-to-body at leaves)
4. **Nearfield**: direct interactions between close branches

All passes have single-thread and multi-thread variants.

### Key Source Files

| File | Role |
|---|---|
| `containers.jl` | All core data structures: `Branch`, `Tree`, `InteractionList`, `Cache`, `DerivativesSwitch` |
| `compatibility.jl` | User interface layer — all functions users must overload |
| `harmonics.jl` | Regular and irregular solid harmonics (Gumerov normalization) |
| `rotate.jl` | Wigner D-matrix rotations; precomputes rotation matrices by π/2 |
| `translate.jl` | M2M, M2L, L2L translations; also Lamb-Helmholtz χ transforms |
| `bodytomultipole.jl` | B2M dispatch by kernel/geometry combination |
| `evaluate_expansions.jl` | L2B: evaluates local expansion at target positions |
| `tree.jl` | Octree construction via recursive pigeonhole sort |
| `interaction_list.jl` | Builds M2L and direct lists using multipole acceptance criterion (MAC) |
| `fmm.jl` | Full FMM pipeline orchestration |
| `error.jl` | Error bounds for dynamic expansion order |
| `dynamic_expansion_order.jl` | Per-interaction expansion order selection (`get_P`) |
| `autotune.jl` | Parameter sweep over (leaf_size, expansion_order, MAC) |
| `solve.jl` | `FastGaussSeidel` iterative BEM solver using FMM-accelerated mat-vec |

### Key Data Structures

**`Branch{TF,N}`**: Octree node with `bodies_index::SVector{N,UnitRange}` (one range per system), `center`, `radius`, `box` (half-widths for MAC), and min potential/gradient values.

**`Tree{TF,N}`**: Complete octree. Critical fields:
- `expansions::Array{TF,4}` — indexed `[re/im=1:2, component=1:2, harmonic_index, branch_index]`; component 1 = scalar φ, component 2 = Lamb-Helmholtz χ
- `buffers::NTuple{N,Matrix{TF}}` — flat column-major body data; rows: `[1:3]=position, 4=scalar_potential, 5:7=gradient, 8:16=hessian, 17:18=previous values`
- `levels_index`, `leaf_index` — branch indices by tree level
- `sort_index_list`, `inverse_sort_index_list` — permutation for sorted body ordering

**`DerivativesSwitch{PS,GS,HS}`**: Compile-time boolean type parameters — `PS` (scalar potential), `GS` (gradient), `HS` (Hessian). Eliminates dead code paths via Julia specialization.

### Important Design Decisions

- **Multi-system octree**: The `N` type parameter in `Tree{TF,N}` and `Branch{TF,N}` allows N different system types in one tree simultaneously — no separate trees needed.
- **Buffer layout**: All body data flows through flat `Matrix{TF}` buffers (columns = bodies). The tree sorts bodies by permutation index without physically moving data; `buffer_to_target_system!` writes results back.
- **M2L algorithm**: O(p³) via rotation trick — rotate to align z-axis (Wigner D-matrix), z-axis translate (O(p²) recurrence), rotate back. Precomputed `Hs_π2` matrices live on the `Tree`.
- **Lamb-Helmholtz flag**: `Val{true/false}` type parameter routes through dual-channel (φ+χ) code paths in M2M, M2L, L2L, L2B for vortex/vector-potential systems.
- **`Cache` pattern**: Pass a `Cache` object to `fmm!` across time steps to avoid repeated allocation. Use `tune_fmm(...)` once at startup to find optimal parameters.
- **MAC methods**: `Barba` (classic: `(r_src + r_tgt)^2 / dist^2 < θ²`) and `SelfTuning` (adaptive, based on actual leaf sizes).
- **Error bounds for dynamic P**: `UnequalSpheres`, `PowerAbsolutePotential`, `RotatedCoefficientsAbsoluteGradient`, `DehnenAbsoluteGradient` — different accuracy/cost tradeoffs.


## Token Discipline

- Delegate instead of doing inline: test/script runs → `julia-test-runner` agent; questions about `MATRIX_OPERATOR_REFACTOR/` docs → `refactor-docs-librarian` agent; re-checking a claimed result before reporting → `verifier` agent; lab-notebook drafts → `notebook-drafter` agent.
- When spawning exploration subagents, choose the model by task: `haiku` for mechanical symbol/file searches, `sonnet` for conceptual exploration ("how does X work", "where does Y flow"), `opus` only when the question requires subtle cross-file reasoning. Always instruct the agent to return a brief synthesis plus an indexed `file:line` list, quoting no more than a few lines of code.
- Any command with potentially long output: redirect to a scratchpad log (`cmd > log 2>&1`), then grep/tail the log. Never let raw test or build output into the main context.
- Never read `MATRIX_OPERATOR_REFACTOR/data/**`, `*.csv`, or `*.bin` files directly; if their contents are needed, write a small script that prints a summary.

## Cluster Jobs

- Combine GPU HPC jobs where appropriate: queue wait on the H200 partition is often long, so when multiple GPU workloads are ready at the same time (e.g. device testsets + a benchmark + a parity harness), prefer staging them into ONE sbatch script (the `cuda_048_run.sh` multi-stage pattern) over submitting separate jobs — unless they need different envs/resources or a stage's outcome should gate whether the next is worth running.
