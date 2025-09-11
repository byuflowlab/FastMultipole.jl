# FastMultipole

*A fast, multi-system, multi-kernel, differentiable implementation of the fast multipole method for use with scalar-plus-vector potential N-body problems in pure Julia.*

**Author**: Ryan Anderson

**Features**:

* solves ``N``-body problems governed by the Laplace (``1/r``) kernel, with work planned to support the Helmholtz kernel in the future
* incorporates seamlessly into existing Julia code without modifications (just the addition of a few interface functions)
* offers convenience functions for determining the expansion coefficients for source, dipole, and vortex points, filaments, and panels (this list is growing!)
* provides velocity and velocity gradient (or their equivalent for non-fluids problems) obtained using analytic expressions (no finite difference)
* uses ``O(p^3)`` translation operators (where ``p`` is the expansion order)
* automated CPU-parallelization of expansions and direct interactions
* supports GPU-parallelization of direct interactions with user-defined kernels
* [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) and [ReverseDiff](https://github.com/JuliaDiff/ReverseDiff.jl) compatible

**Installation**:

```julia
pkg> add https://github.com/byuflowlab/FastMultipole.git
```

**Documentation**:

* learn basic useage in the [Quick Start](quickstart.md) tutorial
* see how to implement `FastMultipole` in your code in the [Gravitational Example](guided_examples.md)
* explore more details in the [Vortex Filament Example](vortex_filament.md)
* learn about the [Tuning Parameters](tuning.md)
* run FMM simultaneously on multiple systems in [Multiple Systems](advanced_usage.md) section
* fine-tune performance in the [Automated Tuning](advanced_usage_2.md)
* see the full [API](reference.md)
