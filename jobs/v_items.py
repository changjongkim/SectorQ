#!/usr/bin/env python3
"""Run list of the second review round (2026-11-13, queue_v, logs/v), bin/pp_engine_k3 on four nodes (16 GPUs).

  python3 jobs/v_items.py            writes queue_v/todo_v.jsonl (the runner keeps finished runs in queue_v/done)

E7  rows planned before the run against rows chosen during the run. "on" is SectorQ as evaluated (rows drawn from the
    valid rows of each window, best of eight candidates on the live-term sample, adaptive sector count). "off<F>" uses
    no live-term information for the rows or the count: one candidate drawn from the valid rows of each window
    (--cands 1, the sample is not consulted) and a constant count of F sectors (--fixed-sectors F, active once the
    adaptive count first reaches one sector per GPU). F = 16, 64, 256. Seven circuits, cutoffs 1e-5 and 3e-6.
E8  random k-local circuits (Table 7) against paulistrings-rs under its truncation rule (after every operation):
    SectorQ with --trunc-every 1 and paulistrings-rs with one GPU per rank on 16 ranks (mode mg16 of jobs/p_drive.py),
    on the circuit files that bin/rpauli_ir writes (results/f_ext/ir/rpauli_k<k>.ir.json; the 2,000 generators of
    each match the engine's --dump-gens).
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "queue_v")
BIN = "bin/pp_engine_k3"

CIRC = {
    "ki": "--circuit ki_lab --weight 0 --target 6000000",
    "tfim": "--circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20 --target 4000000",
    "noise": "--circuit ki_noisy --noise 0.005 --target 6000000",
    "tfimsq": "--circuit tfim_sq --lx 11 --ly 11 --theta 0.2 --phi -0.3 --steps 10 --target 4000000",
    "xxz": "--circuit xxz --lx 8 --ly 8 --jxy 0.1 --jz 0.15 --steps 6 --target 4000000",
    "qaoa": "--circuit qaoa --n 100 --layers 4 --target 4000000",
    "hea": "--circuit hea --n 100 --layers 16 --amp 0.4 --target 4000000",
}
COMMON = "--trace 1 --vram-frac 0.9"
VARIANTS = {"on": "", "off16": "--cands 1 --fixed-sectors 16", "off64": "--cands 1 --fixed-sectors 64",
            "off256": "--cands 1 --fixed-sectors 256"}
RCUTS = ("1e-3", "3e-4", "1e-4", "3e-5", "1e-5")


def item(id_, args, kind="eng", limit=2400, est=300, ladder=None, **extra):
    d = dict(id=id_, kind=kind, nodes=4, per=4, args=" ".join(args.split()), limit=limit, est=est, bin=BIN, **extra)
    if ladder:
        d["ladder"] = ladder
    if "--coef" in args:
        d["coef"] = float(args.split("--coef")[1].split()[0])
    return d


def main():
    for d in ("done", "lock"):
        os.makedirs(os.path.join(Q, d), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "logs", "v"), exist_ok=True)
    e7, e8 = [], []
    for c, args in CIRC.items():
        for v, extra in VARIANTS.items():
            for x in ("1e-5", "3e-6"):
                e7.append(item(f"v.e7.{v}.{c}.{x}", f"{args} --coef {x} {COMMON} {extra}", ladder=f"v.e7.{v}.{c}", est=300))
    for k in (2, 4, 8):
        rp = f"--circuit rpauli --n 100 --layers 20 --locality {k} --amp 0.1 --seed 1 --target 4000000"
        for x in RCUTS:
            e8.append(item(f"v.e8.sq1.k{k}.{x}", f"{rp} --coef {x} --trunc-every 1 {COMMON}", ladder=f"v.e8.sq1.k{k}",
                           limit=2400, est=200))
            e8.append(item(f"v.e8.prs.k{k}.{x}", f"--cutoff {x} --case rpauli_k{k}", kind="prs", ladder=f"v.e8.prs.k{k}",
                           limit=2400, est=600, mode="mg16", circuit=f"rpk{k}", cut=x))
    # smoke tests first: one step of each new configuration
    early = ("v.e8.sq1.k2.1e-3", "v.e8.prs.k2.1e-3", "v.e7.off16.qaoa.1e-5")
    items = e7 + e8
    items = [i for i in items if i["id"] in early] + [i for i in items if i["id"] not in early]
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids))
    with open(os.path.join(Q, "todo_v.jsonl"), "w") as f:
        for i in items:
            f.write(json.dumps(i) + "\n")
    print(f"{len(items)} runs: E7 {len(e7)}, E8 {len(e8)}")


if __name__ == "__main__":
    main()
