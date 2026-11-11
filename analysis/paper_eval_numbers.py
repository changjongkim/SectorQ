#!/usr/bin/env python3
"""Numbers of the evaluation text (2026-11-10 update) from the values the paper figures plot.

Single node (Figure ev_single), weak and strong scaling (ev_weak, ev_strong), seven circuits on 16 and 128 GPUs
(ev_circuits, ev_circuits128) and the 1,024-qubit point of ev_size. Every value comes from the same records and
selection functions as the figures (figures/make_eval_figures.py, make_deck_figures.py, make_paper_figures.py).
paulistrings-rs truncates after every operation and SectorQ every 10 operations, so their times at one cutoff
compare runs that keep different numbers of terms (as for MonoProp).

  ../env/bin/python3 analysis/paper_eval_numbers.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(HERE), "figures")
sys.path.insert(0, FIG)
sys.argv = sys.argv[:1]
os.chdir(FIG)
import make_paper_figures as P            # noqa: E402

mf, md = P.mf, P.md
CIRCUITS = ("ki", "tfim", "noise", "tfimsq", "xxz", "qaoa", "hea")
KINDS = ("ours", "cupp", "defer", "orqa", "monoprop", "prs")


def rng(vals, fmt="%.1f"):
    vals = sorted(vals)
    return (fmt % vals[0]) if len(vals) == 1 or fmt % vals[0] == fmt % vals[-1] else (fmt + "-" + fmt) % (vals[0], vals[-1])


def deepest(series):
    return min(series, key=float) if series else None


def single():
    print("== single node, kicked Ising (Figure ev_single)")
    G = md.campaign()
    fig, ax = plt.subplots()
    rows = {label: dict(pts) for label, pts, _ in mf.g_single_node(G, ax=ax)}
    plt.close(fig)
    for g, sq, pr in ((4, "SectorQ (4 GPUs)", "paulistrings-rs (4 GPUs)"), (1, "SectorQ (1 GPU)", "paulistrings-rs (1 GPU)")):
        shared = [x for x in rows[pr] if x in rows[sq]]
        r = {x: rows[pr][x] / rows[sq][x] for x in shared}
        print(f"  {g} GPU(s): prs/SectorQ over {len(r)} shared cutoffs {rng(r.values())}; faster prs at {[x for x in r if r[x] < 1]}")
        print(f"     deepest: SectorQ {deepest(rows[sq])} ({rows[sq][deepest(rows[sq])]:.0f} s), prs {deepest(rows[pr])} ({rows[pr][deepest(rows[pr])]:.0f} s)")
        x = deepest(shared)
        print(f"     at the deepest shared cutoff {x}: SectorQ {rows[sq][x]:.1f} s, prs {rows[pr][x]:.1f} s ({r[x]:.1f}x)")


def weak():
    print("== weak scaling (Figure ev_weak): prs time / SectorQ time")
    G, big = md.campaign(), md.scaling_jobs()
    fig, grid = plt.subplots(3, 2)
    rows = mf.g_weak(G, big, axes=[[grid[j][0], grid[j][1]] for j in range(3)])
    plt.close(fig)
    t = {}
    for load, title, kind, g, x, s in rows:
        if mf.gok(s):
            t[load, title, kind, g] = s["wall"]
    for load in ("S", "L"):
        for g in (16, 64):
            r = [t[load, title, "paulistrings-rs", g] / t[load, title, "SectorQ", g] for _, title in mf.G3
                 if (load, title, "paulistrings-rs", g) in t and (load, title, "SectorQ", g) in t]
            print(f"  load {load}, {g} GPUs: {rng(r)} over {len(r)} circuits")
        gs = sorted({g for (l, _, k, g) in t if l == load and k == "paulistrings-rs"})
        print(f"  load {load}: prs points on {gs} GPUs")
        for kind in ("MonoProp",):
            r = [t[load, title, kind, 64] / t[load, title, "SectorQ", 64] for _, title in mf.G3 if (load, title, kind, 64) in t]
            print(f"  load {load}, 64 GPUs: {kind}/SectorQ {rng(r)} over {len(r)} circuits")


def strong():
    print("== strong scaling, kicked Ising (Figure ev_strong): prs")
    G, big = md.campaign(), md.scaling_jobs()
    for g in P.STRONG_GPUS:
        sq = {x: P.strong_run(G, big, g, "ours", x)[0] for x in P.STRONG_CUTS}
        pr = {x: P.strong_run(G, big, g, "prs", x)[0] for x in P.STRONG_CUTS}
        shared = [x for x in P.STRONG_CUTS if sq[x] and pr[x]]
        if not shared:
            print(f"  {g} GPUs: no paulistrings-rs runs")
            continue
        r = {x: pr[x] / sq[x] for x in shared}
        x = shared[-1]
        print(f"  {g} GPUs: prs/SectorQ {rng(r.values())} over {len(r)} cutoffs; at its deepest shared {x}: {sq[x]:.1f} vs {pr[x]:.1f} s ({r[x]:.1f}x); "
              f"prs faster at {[c for c in r if r[c] < 1]}")


def circuits16():
    print("== seven circuits on 16 GPUs (Figure ev_circuits)")
    shown = md.circuits16("d_circuits_all", kinds=KINDS, keep=P.every_finished)
    plt.close("all")
    P.CAUGHT.clear()
    for kind in ("orqa", "monoprop", "prs"):
        rs, deeper = [], []
        for c in CIRCUITS:
            a, b = shown.get((c, "ours"), {}), shown.get((c, kind), {})
            shared = [x for x in b if x in a]
            if shared:
                x = deepest(shared)
                rs.append(b[x] / a[x])
                print(f"    {kind} {c}: lowest shared cutoff {x}, time/SectorQ {b[x] / a[x]:.2f}")
            if b and float(deepest(b)) < float(deepest(a)):
                deeper.append(c)
        faster = {c: ["%s %.2fx" % (x, v / shown[c, "ours"][x]) for x, v in shown.get((c, kind), {}).items()
                      if x in shown.get((c, "ours"), {}) and v < shown[c, "ours"][x]] for c in CIRCUITS}
        faster = {c: v for c, v in faster.items() if v}
        print(f"  {kind}: time/SectorQ at the deepest shared cutoff {rng(rs)} over {len(rs)} circuits; faster at {faster}; deeper than SectorQ in {deeper}")
    for c in CIRCUITS:
        print("   ", c, {k: deepest(shown.get((c, k), {})) for k in KINDS})


def circuits128():
    print("== seven circuits on 128 GPUs (Figure ev_circuits128)")
    ladder = ("1e-4", "3e-5", "1e-5", "5e-6", "3e-6", "2e-6", "1e-6", "7e-7", "5e-7", "3e-7", "2e-7", "1e-7", "7e-8", "5e-8", "3e-8")
    circuits = [(c, c) for c in CIRCUITS]
    rows = md.circuits_big(128, ladder=ladder, original_baseline=True, circuits=circuits, kinds=KINDS)
    plt.close("all")
    P.CAUGHT.clear()
    r = {}
    for c in CIRCUITS:
        a, b = rows[c, "ours"], rows[c, "cupp"]
        x = deepest([x for x in b if x in a])
        r[c] = (x, a[x], b[x], b[x] / a[x])
    print("  original-circuit cuPP / SectorQ at the cuPP's deepest cutoff:", {c: "%s %.1f/%.1f s %.1fx" % v for c, v in r.items()}, "range", rng([v[3] for v in r.values()]))
    for kind in ("defer", "orqa", "monoprop"):
        rs = []
        for c in CIRCUITS:
            a, b = rows[c, "ours"], rows.get((c, kind), {})
            shared = [x for x in b if x in a]
            if shared:
                x = deepest(shared)
                rs.append((c, x, b[x] / a[x]))
        print(f"  {kind}: time/SectorQ at its deepest shared cutoff", {c: "%s %.1fx" % (x, v) for c, x, v in rs}, "range", rng([v for _, _, v in rs]))
        faster = {c: ["%s %.2fx" % (x, v / rows[c, "ours"][x]) for x, v in rows.get((c, kind), {}).items() if x in rows[c, "ours"] and v < rows[c, "ours"][x]]
                  for c in CIRCUITS}
        print(f"  {kind}: faster than SectorQ at", {c: v for c, v in faster.items() if v})
    for c in CIRCUITS:
        print("   ", c, {k: deepest(rows.get((c, k), {})) for k in KINDS}, "SectorQ deepest time %.0f s" % rows[c, "ours"][deepest(rows[c, "ours"])])


def qubits():
    print("== qubit sweep at 1e-4 (Figure ev_size)")
    G = md.campaign()
    for L in mf.SIZE_L:
        e, c = G.get(f"m.eng.L{L}.1e-4.g16"), G.get(f"m.cupp.L{L}.1e-4.g16")
        print(f"  {L * L} qubits: SectorQ {e and e.get('status')} {e and e.get('wall')} {e and e.get('limit_note', '')}; cuPP {c and c.get('status')} {c and c.get('wall')}")


if __name__ == "__main__":
    single()
    weak()
    strong()
    circuits16()
    circuits128()
    qubits()
