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
import re
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


def xxz_splits128():
    print("== XXZ on 128 GPUs: sector divisions and waiting for the slowest node (engine logs)")
    import json
    import re
    for x in ("3e-6", "2e-6", "1e-6", "7e-7"):
        rec = json.load(open(os.path.join(os.path.dirname(FIG), "queue_s", "done", f"eng.xxz.{x}.g128")))
        t = open(os.path.join(os.path.dirname(FIG), rec["log"])).read()
        g, h = (int(v) for v in re.search(r"splits_on_gpu=(\d+) splits_on_host=(\d+)", t).groups())
        wall = float(re.search(r"^wall=([\d.]+) s", t, re.M)[1])
        wait = float(re.search(r"waiting for the slowest node=([\d.]+)", t)[1])
        print(f"  {x}: {rec.get('bin', 'pp_engine_i')} wall {wall:.1f} s, divisions {g} on GPUs + {h} on hosts = {g + h}, "
              f"waiting for the slowest node {wait:.1f} s ({100 * wait / wall:.0f}% of the wall)")


def qubits():
    print("== qubit sweep at 1e-4 (Figure ev_size)")
    G = md.campaign()
    for L in mf.SIZE_L:
        e, c = G.get(f"m.eng.L{L}.1e-4.g16"), G.get(f"m.cupp.L{L}.1e-4.g16")
        print(f"  {L * L} qubits: SectorQ {e and e.get('status')} {e and e.get('wall')} {e and e.get('limit_note', '')}; cuPP {c and c.get('status')} {c and c.get('wall')}")


def runtime_profiles():
    print("== time shares of the 64- and 256-GPU profiles (Figure ev_runtime, analysis/paper_runtime_profiles.json)")
    import json
    rows = json.load(open(os.path.join(os.path.dirname(FIG), "analysis", "paper_runtime_profiles.json")))
    share = lambda r, k: 100 * r[k] / r["wall"]
    rng2 = lambda v: f"{min(v):.0f}-{max(v):.0f}%"
    for g in (64, 256):
        done = [r for r in rows if r["gpus"] == g and r["status"] == "complete"]
        small = [r for r in done if r["point"] == 0]
        large = [r for r in done if r["point"] == 2]
        print(f"  {g} GPUs: small reference idle {rng2([share(r, 'idle') for r in small])}, "
              f"largest points library calls {rng2([share(r, 'operations') for r in large])}, "
              f"exchange {rng2([share(r, 'exchange') for r in large])}, GPU-host copies {rng2([share(r, 'copies') for r in large])}, "
              f"idle {max(share(r, 'idle') for r in large):.1f}% at most")
    small = [r for r in rows if r["gpus"] in (64, 256) and r["point"] == 0 and r["status"] == "complete"]
    print(f"  64 and 256 GPUs: small reference idle {rng2([share(r, 'idle') for r in small])} over {len(small)} profiles")


def weak_efficiency():
    print("== weak scaling, large load (Figure ev_weak d-f): boundary-term throughput per GPU relative to one GPU")
    # Work is the number of term checks of the integrity line (terms at the window boundaries summed over the run),
    # as in analysis/scaling_table.py; efficiency = work / wall / GPUs over the same quantity on one GPU.
    import make_eval_figures as mf
    G, big = md.campaign(), md.scaling_jobs()
    fig, ax = plt.subplots(3, 2)
    rows = mf.g_weak(G, big, axes=[[ax[j][0], ax[j][1]] for j in range(3)])
    plt.close(fig)
    names = {t: c for c, t in mf.G3}
    rate = {}
    for load, title, kind, g, x, s in rows:
        if load != "L" or kind != mf.SQ_NAME["ours"] or not mf.gok(s):
            continue
        rec = mf.big_run(big, names[title], x, g) if g > 16 else mf.gdetail(s)
        if rec and rec.get("checks"):
            rate.setdefault(title, {})[g] = float(rec["checks"]) / s["wall"] / g
    for g in sorted({g for d in rate.values() for g in d}):
        if g > 256:                                  # the paper shows 512 GPUs only in its Extreme Scale-Out section
            continue
        e = [100 * d[g] / d[1] for d in rate.values() if g in d and 1 in d]
        print(f"  {g:3d} GPUs: {min(e):.0f}-{max(e):.0f}% over {len(e)} circuits")
    # idle time at window boundaries until the slowest node finishes (the "waiting" part of the trace), large load
    for g in (64, 256):
        w = [100 * mf.big_run(big, c, mf.BIG_WEAK[g], g)["parts"]["waiting"] / mf.big_run(big, c, mf.BIG_WEAK[g], g)["wall"]
             for c, _ in mf.G3]
        print(f"  {g:3d} GPUs: nodes idle at window boundaries for {min(w):.0f}-{max(w):.0f}% of the wall time")


def cap256():
    print("== kicked Ising 3e-6 on 256 GPUs: performance setting against a 1,024-sector cap (Sections 4.6 and 4.9)")
    import make_eval_figures as mf
    big = md.scaling_jobs()
    for tag, log in (("performance setting (cap 256)", "s/i.ref.ki.3e-6.g256.out"), ("cap 1,024", "s/n.eng10.ki.3e-6.g256.out")):
        r = next(r for r in big if r["log"] == log and mf.ok(r))
        p = r["parts"]
        runs = re.search(r"bucket_runs=(\d+)", open(os.path.join(os.path.dirname(FIG), "logs", log)).read()).group(1)
        print(f"  {tag}: {r['wall']:.1f} s, worker idle {100 * p['idle'] / p['workers'] / r['wall']:.0f}%, "
              f"{int(runs):,} sector tasks, binary {r.get('binary')}")


def ki_values():
    print("== kicked Ising expectation values at the low cutoffs, 256 GPUs or fewer (Section 4.8)")
    import make_eval_figures as mf
    big = md.scaling_jobs()
    for x in (1e-6, 7e-7, 5e-7, 3e-7, 2e-7):
        v = {round(r["value"], 6) for r in big if r["kind"] == "engine" and r["circuit"] == "ki_lab" and r["gpus"] <= 256
             and r["coef"] == x and mf.ok(r)}
        print(f"  {x:.0e}: {sorted(v)}")


def accuracy_time():
    """Figure ev_accuracy (30-qubit TFIM, 4 and 6 Trotter steps, exact reference): the error of the value at each cutoff
    (the tick labels) and, per implementation, the lowest completed cutoff with its time (make_eval_figures.g_accuracy_time)."""
    G = md.campaign()
    impl = [("SectorQ 16 GPUs", "eng", "g16"), ("SectorQ 4 GPUs", "eng", "g4"), ("SectorQ 1 GPU", "eng", "g1"),
            ("cuPauliProp 1 GPU", "ref", "g1"), ("SPD 1 GPU", "spd", "n1"), ("ORQA reproduction 4 nodes", "orqa", "n4"),
            ("MonoProp 4 nodes", "monoprop", "n4"), ("PauliPropagation.jl 64 cores", "julia", "n1")]
    print("accuracy-time (Figure ev_accuracy)")
    for steps in (4, 6):
        errs = []
        for x in mf.ACC_CUTS:
            vals = [G[i]["expectation"] for i in (f"acc.eng.s{steps}.{x}.g16", f"acc.eng.s{steps}.{x}.g4", f"acc.eng.s{steps}.{x}.g1") if mf.gok(G.get(i))]
            if vals:
                errs.append((x, abs(vals[0] - mf.EXACT_G[steps])))
        print(f"  {steps} steps: error " + ", ".join(f"{x} {e:.1e}" for x, e in errs))
        for name, tag, suffix in impl:
            done = [(x, G[f"acc.{tag}.s{steps}.{x}.{suffix}"]["wall"]) for x in mf.ACC_CUTS if mf.gok(G.get(f"acc.{tag}.s{steps}.{x}.{suffix}"))]
            if done:
                x, w = done[-1]
                print(f"    {name:30s} lowest {x} in {w:.2f} s; at 1e-5 " + (f"{dict(done)['1e-5']:.2f} s" if "1e-5" in dict(done) else "-"))


if __name__ == "__main__":
    single()
    weak()
    strong()
    circuits16()
    circuits128()
    xxz_splits128()
    qubits()
    runtime_profiles()
    weak_efficiency()
    cap256()
    ki_values()
    accuracy_time()
