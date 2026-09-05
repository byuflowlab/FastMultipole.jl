#!/usr/bin/env python3
"""Add RELAX_SCHEME knob to the 052-h200 driver and a scr_p019_s038v_lgpz case."""
import sys

drv = "/home/rander39/FLOWPanel-052-h200/examples/rotor_hover_pressure_comparison.jl"
tab = "/home/rander39/FLOWPanel-052-h200/examples/run_p018_screen_hpc.slurm.sh"

old_drv = """stock_relaxation = pnl.FLOWVPM.relaxation_correctedpedrizzetti
"""
new_drv = """# 052 ignition study: RELAX_SCHEME selects the Pedrizzetti variant
# ("pedrizzetti" = plain, magnitude-DAMPING rotation toward omega;
#  "correctedpedrizzetti" = magnitude-preserving, stock default).
stock_relaxation = let _s = lowercase(get(ENV, "RELAX_SCHEME", "correctedpedrizzetti"))
    _s == "pedrizzetti" ? pnl.FLOWVPM.relaxation_pedrizzetti :
    _s == "correctedpedrizzetti" ? pnl.FLOWVPM.relaxation_correctedpedrizzetti :
    error("unknown RELAX_SCHEME=$(_s)")
end
println("relaxation scheme: $(stock_relaxation.relax)")
"""

s = open(drv).read()
if "RELAX_SCHEME" in s:
    print("driver already patched")
else:
    assert s.count(old_drv) == 1, "driver anchor not unique"
    open(drv, "w").write(s.replace(old_drv, new_drv))
    print("driver patched")

anchor = "  scr_p019_s038v_gpu40) export OVERLAP=2.4; export P_PER_STEP=11; export MERGE_R_FACTOR=0.00524; export NWAKEROWS=1; export DAS_UNIFORM_DSIGMA=3.4; export WAKE_HEALTH_DTZ=true; export CORE_SPREADING_ACTIVE=true; export WAKE_CORE_BETA=1e9; export NREVS=40 ;;"
newcase = anchor + "\n  # lgpz: identical to s038v_gpu40 but PLAIN Pedrizzetti relaxation (052 ignition study 2026-08-31)\n  scr_p019_s038v_lgpz) export OVERLAP=2.4; export P_PER_STEP=11; export MERGE_R_FACTOR=0.00524; export NWAKEROWS=1; export DAS_UNIFORM_DSIGMA=3.4; export WAKE_HEALTH_DTZ=true; export CORE_SPREADING_ACTIVE=true; export WAKE_CORE_BETA=1e9; export NREVS=40; export RELAX_SCHEME=pedrizzetti ;;"

t = open(tab).read()
if "scr_p019_s038v_lgpz" in t:
    print("case table already patched")
else:
    assert t.count(anchor) == 1, "case anchor not unique"
    open(tab, "w").write(t.replace(anchor, newcase))
    print("case table patched")
