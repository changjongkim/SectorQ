#!/usr/bin/env python3
"""Run list of the review experiments of 2026-11-12 (queue_y, logs/y), all with bin/pp_engine_k on at most four nodes.

  python3 jobs/y_items.py            writes queue_y/todo_y.jsonl (the runner keeps finished runs in queue_y/done)

E1  fixed rows against rows recomputed for every window, on 16 GPUs: re (recomputed, the default), fx (--fixed-rows 1: the
    rows of the first window that divides the terms), bf (--fixed-rows 2: the rows that the most rotations of the whole
    circuit conserve). The circuits and ladders of the former ablation (jobs/x_ablate.sh, analysis/paper_fixed_rows.json),
    rerun with one engine so that the three modes compare. Noisy kicked Ising is left out (no fixed-row mode).
E2  the window rule: --ann-min at 1/4, 1/2 and 3/2 of the qubit count (the default is the qubit count, which E1 re
    measures), four circuits, cutoffs 1e-5 and 3e-6, 16 GPUs.
E3  retained terms at window boundaries: a checkpoint after the truncation that ends window W, from one GPU with one
    sector (--gmax 0, every merge global) and from 16 GPUs with sectors; analysis/retained_terms.py compares them.
E4  the sample's prediction of the largest sector: the trace field pred_largest_share of the E1 re runs, plus noisy
    kicked Ising.
E5  random k-local Pauli rotations (--circuit rpauli), k = 2, 4, 8: SectorQ and cuPauliProp's distributed mode on 16 GPUs.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "queue_y")
BIN = "bin/pp_engine_k"

CIRC = {
    "ki": ("--circuit ki_lab --weight 0 --target 6000000", 127),
    "tfim": ("--circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20 --target 4000000", 127),
    "tfimsq": ("--circuit tfim_sq --lx 11 --ly 11 --theta 0.2 --phi -0.3 --steps 10 --target 4000000", 121),
    "xxz": ("--circuit xxz --lx 8 --ly 8 --jxy 0.1 --jz 0.15 --steps 6 --target 4000000", 64),
    "qaoa": ("--circuit qaoa --n 100 --layers 4 --target 4000000", 100),
    "hea": ("--circuit hea --n 100 --layers 16 --amp 0.4 --target 4000000", 100),
    "noise": ("--circuit ki_noisy --noise 0.005 --target 6000000", 127),
}
# the 16-GPU ladders of the former ablation (jobs/x_ablate.sh)
LADDER = {"ki": ("1e-4", "3e-5", "1e-5", "3e-6"), "tfim": ("1e-4", "3e-5", "1e-5", "3e-6"),
          "tfimsq": ("1e-4", "3e-5", "1e-5", "3e-6"), "xxz": ("1e-4", "3e-5", "1e-5", "3e-6"),
          "qaoa": ("1e-4", "3e-5", "1e-5", "3e-6", "1e-6"), "hea": ("1e-4", "3e-5", "1e-5", "3e-6", "1e-6")}
COMMON = "--trace 1 --vram-frac 0.9"


def item(id_, args, nodes=4, per=4, kind="eng", limit=2400, est=300, ladder=None):
    d = dict(id=id_, kind=kind, nodes=nodes, per=per, args=args, limit=limit, est=est, bin=BIN)
    if ladder:
        d["ladder"] = ladder
    if "--coef" in args:
        d["coef"] = float(args.split("--coef")[1].split()[0])
    return d


def e3():
    """Checkpoints after the window ends W (cutoff 3e-5): one GPU, one sector; 16 GPUs, sectors."""
    wins = {"ki": (5, 10, 15), "tfim": (10, 22, 35), "xxz": (5, 11, 17), "qaoa": (2, 4, 6)}
    one, many = [], []
    for c, ws in wins.items():
        for w in ws:
            for tag, nodes, per, extra in (("g1", 1, 1, "--gmax 0"), ("g16", 4, 4, "")):
                rid = f"e3.{c}.w{w}.{tag}"
                ck = f"logs/y/ckpt/{rid}"
                args = f"{CIRC[c][0]} --coef 3e-5 --vram-frac 0.9 {extra} --ckpt-window {w} --ckpt-file {ck} --stop-at-ckpt 1"
                (one if nodes == 1 else many).append(item(rid, " ".join(args.split()), nodes, per, limit=1800, est=120))
    return one, many


def main():
    os.makedirs(os.path.join(Q, "done"), exist_ok=True)
    os.makedirs(os.path.join(Q, "lock"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "logs", "y", "ckpt"), exist_ok=True)
    items = []
    one, many = e3()
    items += one                                                            # one-node runs (run first, side by side)
    # E1 re (also E4) and E4 noisy kicked Ising
    for c, cuts in LADDER.items():
        for x in cuts:
            items.append(item(f"e1.re.{c}.{x}", f"{CIRC[c][0]} --coef {x} {COMMON}", ladder=f"e1.re.{c}", est=60))
    for x in ("1e-5", "3e-6"):
        items.append(item(f"e4.re.noise.{x}", f"{CIRC['noise'][0]} --coef {x} {COMMON}", ladder="e4.re.noise", est=120))
    # E2
    for c in ("ki", "tfim", "xxz", "qaoa"):
        args, n = CIRC[c]
        for f in (0.25, 0.5, 1.5):
            a = int(round(f * n))
            for x in ("1e-5", "3e-6"):
                items.append(item(f"e2.ann{f:g}.{c}.{x}", f"{args} --coef {x} {COMMON} --ann-min {a}",
                                  ladder=f"e2.ann{f:g}.{c}", est=200))
    items += many                                                           # E3, 16 GPUs
    # E1 bf
    for c, cuts in LADDER.items():
        for x in cuts:
            items.append(item(f"e1.bf.{c}.{x}", f"{CIRC[c][0]} --coef {x} {COMMON} --fixed-rows 2", ladder=f"e1.bf.{c}", est=400))
    # E5: SectorQ and cuPauliProp, k = 2, 4, 8
    for k in (2, 4, 8):
        rp = f"--circuit rpauli --n 100 --layers 20 --locality {k} --amp 0.1 --seed 1 --target 4000000"
        for x in ("1e-3", "3e-4", "1e-4", "3e-5", "1e-5"):
            items.append(item(f"e5.eng.k{k}.{x}", f"{rp} --coef {x} {COMMON}", ladder=f"e5.eng.k{k}", limit=1800, est=200))
            items.append(item(f"e5.cupp.k{k}.{x}", f"{rp} --coef {x} --vram-frac 0.9", kind="cupp", ladder=f"e5.cupp.k{k}",
                              limit=1800, est=200))
    # E1 fx (the longest runs last)
    for c, cuts in LADDER.items():
        for x in cuts:
            items.append(item(f"e1.fx.{c}.{x}", f"{CIRC[c][0]} --coef {x} {COMMON} --fixed-rows 1", ladder=f"e1.fx.{c}", est=600))
    # smoke tests of the new code first: the first step of one bf ladder and of one rpauli ladder of each system
    early = ("e1.bf.qaoa.1e-4", "e5.eng.k2.1e-3", "e5.cupp.k2.1e-3")
    first = [i for i in items if i["id"] in early]
    rest = [i for i in items if i["id"] not in early]
    k = sum(1 for i in rest if i["nodes"] == 1)
    items = rest[:k] + first + rest[k:]
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids))
    with open(os.path.join(Q, "todo_y.jsonl"), "w") as f:
        for i in items:
            f.write(json.dumps(i) + "\n")
    print(f"{len(items)} runs: {sum(i['nodes'] == 1 for i in items)} on one node, {sum(i['nodes'] == 4 for i in items)} on four")


if __name__ == "__main__":
    main()
