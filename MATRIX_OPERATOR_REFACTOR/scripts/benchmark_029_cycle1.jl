# Task 029 cycle 1 driver: A/B the occupancy-epoch window cache and the
# graph-captured far-field chain around the UNCHANGED benchmark_028_feasibility
# harness. The production defaults are ON; this wrapper only pins the two
# runtime flags from env before the harness runs, so "off" rows measure the
# exact pre-029 launch pattern on the same manifest.
#
#   FM029_CACHED  "1"/"0"  CUDA_CACHED_WINDOWS   (default 1)
#   FM029_GRAPH   "1"/"0"  CUDA_GRAPH_LIFECYCLE  (default 1)
#
# All FM028_* env knobs pass through untouched.

using FastMultipole

FastMultipole.load_cuda_radix_lifecycle!() ||
    error("CUDA radix lifecycle failed to load: $(FastMultipole.cuda_radix_status())")

FastMultipole.CUDA_CACHED_WINDOWS[] = get(ENV, "FM029_CACHED", "1") == "1"
FastMultipole.CUDA_GRAPH_LIFECYCLE[] = get(ENV, "FM029_GRAPH", "1") == "1"
@info "029 cycle-1 flags" cached_windows = FastMultipole.CUDA_CACHED_WINDOWS[] graph_lifecycle = FastMultipole.CUDA_GRAPH_LIFECYCLE[]

include(joinpath(@__DIR__, "benchmark_028_feasibility.jl"))
