#!/usr/bin/env python3
"""Review runs on 128 GPUs (2026-11-12): paulistrings-rs completed noisy kicked Ising at 7e-7 and 5e-7 on 128 GPUs,
cutoffs that SectorQ had not been run at, and TFIM 11x11/XXZ comparisons need SectorQ below 1e-6. Writes
queue_z: SectorQ (bin/pp_engine_i, the capacity settings of the kicked-Ising runs n12.ki.*.g128) on
noisy kicked Ising 7e-7, 5e-7, 3e-7 and TFIM 7e-7, 5e-7 on 32 nodes. The runner (jobs/z_run.py) writes its logs to
logs/s, where figures/make_eval_figures.big_logs reads the runs on 32 GPUs and more.

  env/bin/python3 jobs/z_items.py            noisy kicked Ising -> queue_z/todo_z1.jsonl
  env/bin/python3 jobs/z_items.py --tfim     TFIM -> queue_z/todo_z2.jsonl (written by jobs/z_review128.sh after the
                                             matched XXZ runs, so that a long TFIM run cannot take their time)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CIRC = {"noise": "--circuit ki_noisy --noise 0.005 --target 6000000",
        "tfim": "--circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20 --target 4000000"}
TFIM = "--tfim" in sys.argv
CUTS = {"tfim": ("7e-7", "5e-7")} if TFIM else {"noise": ("7e-7", "5e-7", "3e-7")}
OUT = "todo_z2.jsonl" if TFIM else "todo_z1.jsonl"
items = []
for c, cuts in CUTS.items():
    for x in cuts:
        items.append(dict(id=f"n12.{c}.{x}.g128", nodes=32, kind="eng", bin="bin/pp_engine_i", per=4, limit=7200, coef=float(x),
                          est=900, ladder=f"n12.{c}.g128",
                          args=f"{CIRC[c]} --coef {x} --trace 1 --vram-frac 0.9 --gmax 12 --chunk 65536"))
os.makedirs(os.path.join(ROOT, "queue_z"), exist_ok=True)
with open(os.path.join(ROOT, "queue_z", OUT), "w") as f:
    for i in items:
        f.write(json.dumps(i) + "\n")
print(f"wrote {len(items)} runs to queue_z/{OUT}")
