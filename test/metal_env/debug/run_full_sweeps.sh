#!/bin/bash
# The full parameter sweeps behind the unit suites: larger fields, deeper
# trees, more expansion orders. Run this when a suite fails, or before a
# release; the default run keeps one small case per structural path.
#   bash test/metal_env/debug/run_full_sweeps.sh [suite-name-pattern ...]
set -uo pipefail
cd "$(dirname "$0")/.."
FM_FULL_SWEEP=1 exec bash run_suites.sh "$@"
