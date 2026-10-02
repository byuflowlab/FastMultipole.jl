#!/usr/bin/env bash
# Compact runner for the ka_*_correctness.jl suites.
#
# Prints ONE line per suite; full output goes to logs/<suite>.log and is echoed
# only for suites that fail. Exists so that running the gate does not dump
# hundreds of lines into a terminal (or an agent's context).
#
#   bash run_suites.sh              # all correctness suites
#   bash run_suites.sh m2m l2l      # only suites matching these substrings
#
# Exit status is the number of failing suites (0 = all green). The work is
# done by run_suites.jl in a single Julia process; see there for the
# pass/fail rules.

set -uo pipefail
cd "$(dirname "$0")"

# One Julia process for every suite (run_suites.jl): package load and Metal
# kernel compilation are paid once. --startup-file=no because under Pkg.test
# the child's load path has no Pkg stdlib and a startup.jl that imports Pkg
# would abort every suite before it starts.
# Pkg.test hands its child a restricted JULIA_LOAD_PATH ("@"), which hides the
# standard library from the suites and makes every one of them fail before it
# starts; the project comes from --project here, so drop both inherited vars.
unset JULIA_LOAD_PATH JULIA_PROJECT

exec julia --startup-file=no --project="${FASTMULTIPOLE_GPU_TEST_PROJECT:-.}" run_suites.jl "$@"
