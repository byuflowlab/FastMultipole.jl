# GPU/Radix capability census

The Radix path is independent of CPU `DerivativesSwitch` storage. At this implementation
snapshot it retains 4 rows (potential plus gradient) or 13 rows (plus dense Hessian), and
the public Radix entry point rejects third-order requests. A future level-three cache must
use 31 rows and preserve the first 13 exactly.

| Surface | Route | Scalar | Vortex/LH | Current output gate | Level-three work |
|---|---|---|---|---|---|
| `translate_batched_resident.jl` | host direct | singular/regularized pair kernels | singular/regularized pair kernels | `size(output,1) >= 13` | packed-18 pair formulas and rows 14:31 |
| `translate_batched_resident.jl` | host L2B far field | flat Hessian evaluator | flat LH Hessian evaluator | 4/13-row dispatch | third recurrence and 31-row writeback |
| `translate_batched_cuda.jl` | CUDA direct, atomic/ballot/two-pass | scalar Hessian variants | vortex Hessian variants | `Val(HS)`, 13-slot accumulators | 31-slot/level-specialized accumulators and atomics |
| `translate_batched_cuda.jl` | CUDA L2B/M2T | flat Hessian evaluators | LH evaluators where supported | `size(output,1) >= 13` | packed third recurrence and rows 14:31 |
| `cross_stencil_cuda.jl` | panel cross-stencil | 4-row potential/gradient | route-specific | fixed 4 rows | explicitly unsupported or new analytic third kernel |
| `direct_rectangular.jl` | rectangular direct | potential/gradient, body-dependent Hessian | functor-dependent | `rect_output_rows`, potential row 4/13 | level-aware row count and kernel trait |
| `interaction_list_batched.jl` | route/list construction | layout-neutral traversal | layout-neutral traversal | coupled through resident state | propagate maximum derivative level only |

Lifecycle construction sites allocate output in `translate_batched_resident.jl` and
`translate_batched_cuda.jl` from a Hessian Boolean. Finalization copies rows 1:4 and 5:13
through CPU switch-relative ranges. SFS consumes the 13-row velocity Jacobian and must
continue to require at least Hessian level; it does not itself require third-order rows.

GPU implementation and parity remain gated because CUDA is unavailable in the current
macOS environment. The CPU implementation therefore keeps the explicit Radix guard rather
than advertising unverified device support.
