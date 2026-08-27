#!/usr/bin/env python3
"""Phase 1: parse fp052chain stage-d log -> per-category time vs step."""
import re, sys, csv, statistics as st
from collections import defaultdict

LOG = sys.argv[1]
ansi = re.compile(r"\x1b\[[0-9;]*m")

step_timer = re.compile(r"step_timer(_nested)?\s+(\S+)\s+step=(\d+)\s+([0-9.eE+-]+)\s*s")
gpu_timer = re.compile(r"gpu_timer\s+(\S+)\s+([0-9.eE+-]+)\s*s")
step_line = re.compile(r"step\s+(\d+)/1079 at time\s+([0-9.eE+-]+)\s+\[([0-9.]+)\s*s\]")

# stage d starts after "stages=d" marker; only parse after it
lines = []
started = False
for raw in open(LOG, errors="replace"):
    line = ansi.sub("", raw).rstrip("\r\n")
    if "stages=d" in line:
        started = True
        lines = []  # reset: keep only stage-d portion
        continue
    lines.append(line)
if not started:
    print("WARNING: no stages=d marker; parsing whole file")

cur_step = None
cat = defaultdict(dict)      # cat[name][step] = seconds (step_timer, top-level)
nested = defaultdict(dict)   # nested timers
gpu = defaultdict(lambda: defaultdict(float))  # gpu[name][step] += s
gpu_cnt = defaultdict(lambda: defaultdict(int))
wall = {}                    # step -> bracketed wall s from "step N/1079 ... [x s]"

for line in lines:
    m = step_line.search(line)
    if m:
        s = int(m.group(1)); wall[s] = float(m.group(3)); cur_step = s
        continue
    m = step_timer.search(line)
    if m:
        is_nested, name, s, v = m.group(1), m.group(2), int(m.group(3)), float(m.group(4))
        (nested if is_nested else cat)[name][s] = v
        cur_step = s
        continue
    m = gpu_timer.search(line)
    if m and cur_step is not None:
        gpu[m.group(1)][cur_step] += float(m.group(2))
        gpu_cnt[m.group(1)][cur_step] += 1

steps = sorted(wall)
print(f"stage-d steps with wall time: {len(steps)} (range {steps[0]}..{steps[-1]})" if steps else "no step lines")

def block_means(d, blocks):
    out = []
    for lo, hi in blocks:
        vs = [d[s] for s in d if lo <= s <= hi]
        out.append(st.mean(vs) if vs else float("nan"))
    return out

if steps:
    lo, hi = steps[0], steps[-1]
    n = 6
    edges = [lo + i * (hi - lo) // n for i in range(n + 1)]
    blocks = [(edges[i], edges[i + 1]) for i in range(n)]
    hdr = " ".join(f"[{a}-{b}]" for a, b in blocks)
    print(f"\nmean seconds/step by step-block:\n{'category':28s} {hdr}")
    def show(name, d):
        bm = block_means(d, blocks)
        print(f"{name:28s} " + " ".join(f"{v:9.2f}" for v in bm))
    show("WALL (bracketed)", wall)
    for name in sorted(cat, key=lambda k: -max(cat[k].values())):
        show(f"t:{name}", cat[name])
    for name in sorted(nested, key=lambda k: -max(nested[k].values())):
        show(f"n:{name}", nested[name])
    for name in sorted(gpu, key=lambda k: -max(gpu[k].values())):
        show(f"g:{name}", {s: v for s, v in gpu[name].items()})
    # gpu call counts per step, last block
    print("\ngpu_timer calls/step (last block):")
    a, b = blocks[-1]
    for name in sorted(gpu_cnt):
        cs = [gpu_cnt[name][s] for s in gpu_cnt[name] if a <= s <= b]
        if cs:
            print(f"  {name:28s} mean {st.mean(cs):8.1f}  max {max(cs)}")
    # coverage: does sum of top-level categories explain total_step?
    if "total_step" in cat:
        tot = cat["total_step"]
        others = [k for k in cat if k != "total_step"]
        gaps = []
        for s in tot:
            ssum = sum(cat[k].get(s, 0.0) for k in others)
            gaps.append((s, tot[s], ssum, tot[s] - ssum))
        gaps.sort()
        late = [g for g in gaps if g[0] >= edges[-2]]
        if late:
            print(f"\ntotal_step vs sum(other top-level cats), last block mean: "
                  f"total={st.mean([g[1] for g in late]):.2f}s "
                  f"sum={st.mean([g[2] for g in late]):.2f}s "
                  f"unexplained={st.mean([g[3] for g in late]):.2f}s")
    # dump per-step csv for later plotting
    with open(sys.argv[2], "w", newline="") as f:
        w = csv.writer(f)
        names = sorted(set(list(cat) + list(nested)))
        w.writerow(["step", "wall_s"] + names + [f"g_{k}" for k in sorted(gpu)])
        for s in steps:
            w.writerow([s, wall.get(s, "")] +
                       [cat.get(k, {}).get(s, nested.get(k, {}).get(s, "")) for k in names] +
                       [round(gpu[k].get(s, 0.0), 5) for k in sorted(gpu)])
    print(f"\nwrote per-step CSV: {sys.argv[2]}")

# CPU reference wall_s summary
if len(sys.argv) > 3:
    import csv as _csv
    with open(sys.argv[3]) as f:
        r = list(_csv.DictReader(f))
    cols = r[0].keys()
    wcol = next((c for c in cols if "wall" in c.lower()), None)
    ncol = next((c for c in cols if c.lower() in ("np", "n_particles", "nparticles", "count", "n")), None)
    print(f"\nCPU reference ({len(r)} rows) cols={list(cols)[:8]}...")
    if wcol:
        ws = [float(x[wcol]) for x in r if x[wcol]]
        print(f"CPU wall_s per step (plateau ~209k): mean {st.mean(ws):.2f}  "
              f"median {st.median(ws):.2f}  min {min(ws):.2f}  max {max(ws):.2f}")
    if ncol:
        ns = [float(x[ncol]) for x in r if x[ncol]]
        print(f"CPU particle count: first {ns[0]:.0f} last {ns[-1]:.0f}")
