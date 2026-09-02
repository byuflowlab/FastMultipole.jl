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
# Exit status is the number of failing suites (0 = all green).
#
# Pass/fail detection: no suite calls exit(1), and the summary lines are not
# uniform, so a suite is PASS only if julia exited 0 AND the log carries no
# failure marker. `error()` and a failing `@test` both throw, which julia
# reports as a nonzero exit, so the two checks together are what cover it.

set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs

FAIL_RE='FAIL|THREW|Test Failed|ERROR:|✗|MethodError|not functional'

suites=()
for f in ka_*_correctness.jl; do
    [ -e "$f" ] || continue
    if [ $# -eq 0 ]; then
        suites+=("$f")
    else
        for pat in "$@"; do
            case "$f" in *"$pat"*) suites+=("$f"); break;; esac
        done
    fi
done

[ ${#suites[@]} -eq 0 ] && { echo "no suites matched: $*"; exit 0; }

nfail=0
for f in "${suites[@]}"; do
    name="${f%.jl}"
    log="logs/$name.log"
    start=$(date +%s)
    julia --project="${FASTMULTIPOLE_GPU_TEST_PROJECT:-.}" "$f" >"$log" 2>&1
    rc=$?
    dt=$(( $(date +%s) - start ))

    if [ $rc -eq 0 ] && ! grep -qE "$FAIL_RE" "$log"; then
        # the suites' own summary line, if they printed one
        summary=$(grep -oE '[0-9]+/[0-9]+ pass|All .* passed|gate passed' "$log" | tail -1)
        printf '%-42s PASS  %3ds  %s\n' "$name" "$dt" "$summary"
    else
        nfail=$((nfail + 1))
        printf '%-42s FAIL  %3ds  (rc=%d) -> %s\n' "$name" "$dt" "$rc" "$log"
        grep -nE "$FAIL_RE" "$log" | head -5 | sed 's/^/    /'
    fi
done

echo "---"
if [ $nfail -eq 0 ]; then
    echo "all ${#suites[@]} suites PASS"
else
    echo "$nfail of ${#suites[@]} suites FAILED (full output in logs/)"
fi
exit $nfail
