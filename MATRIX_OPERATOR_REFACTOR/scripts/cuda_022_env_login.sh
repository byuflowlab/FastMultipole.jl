#!/bin/bash
# Item 022: login-node environment prep (network available, GPU absent).
# Installs CUDA.jl into the global env, sets local-toolkit preferences for all
# three CUDA JLLs, and instantiates the project WITHOUT precompiling (pkgimages
# must be built on the GPU node's CPU target; see cuda-hpc-setup notes).
source /etc/profile
set -eo pipefail
module load julia

export JULIA_PKG_PRECOMPILE_AUTO=0

julia -e 'using Pkg; Pkg.add(["CUDA", "Preferences",
    "CUDA_Runtime_jll", "CUDA_Compiler_jll", "CUDA_Driver_jll"]); Pkg.status()'

julia -e '
using Preferences
for p in ("CUDA_Runtime_jll", "CUDA_Compiler_jll", "CUDA_Driver_jll")
    set_preferences!(p, "local" => "true"; force=true)
    println(p, " => local=true")
end'

# Dedicated validation env: dev the synced repo + CUDA as a direct dep, so
# everything resolves coherently under the cluster Julia (the repo Manifest.toml
# was resolved under a different Julia version and is ignored by dev).
julia -e '
using Pkg
Pkg.activate(joinpath(homedir(), "fm022env"))
Pkg.develop(path=joinpath(homedir(), "projects", "FastMultipole-022"))
Pkg.add(["CUDA", "Random", "Printf", "Test"])
Pkg.status()'

echo "LOGIN_PREP_DONE"
