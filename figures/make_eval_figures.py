#!/usr/bin/env python3
"""Background and evaluation figures of the README.

  make_eval_figures.py           every figure of the README
  make_eval_figures.py --g       only the figures of the 1-16 GPU campaign (queue_g/done, logs/g); --print lists the numbers
  make_eval_figures.py --old     the evaluation figures before that campaign, from the earlier logs
  --out DIR                      writes the figures to DIR instead of this folder (drafts)

Run times, expectation values and time breakdowns are read from the run logs in ../logs
(blocks that start with "===== engine|cupp nodes=... args: ..."). The numbers of the
background figure and of the exact-value check come from earlier logs with other formats
and are listed below with their source.
"""
import glob
import json
import os
import re
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

BIG_LOG = "pp_e10-59483161.out"
HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "..", "logs")

OURS, BASE, TUNED, INK, GRID = "#1B9E8F", "#8A8780", "#C9B99B", "#1F2328", "#DDDDD5"
plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 14, "xtick.labelsize": 12.5, "ytick.labelsize": 12.5,
    "axes.edgecolor": "#BDBDB5", "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "legend.frameon": False, "savefig.dpi": 200,
})


# ---------------------------------------------------------------------------------------
# log reader
# ---------------------------------------------------------------------------------------
def parse(name):
    """Returns the runs of one log as a list of dicts."""
    path = os.path.join(LOGS, name)
    if not os.path.exists(path):
        return []
    runs, cur = [], None
    for line in open(path, errors="replace"):
        m = re.match(r"===== (engine|cupp) nodes=(\d+) gpus=(\d+) args: (.*)", line)
        if m:
            args = m.group(4).split()
            opt = {args[i]: args[i + 1] for i in range(len(args) - 1) if args[i].startswith("--")}
            gpus = int(m.group(3))
            if m.group(1) == "engine" and "--gpus" in opt:
                gpus = int(opt["--gpus"]) * int(m.group(2))
            cur = dict(kind=m.group(1), nodes=int(m.group(2)), gpus=gpus, opt=opt,
                       circuit=opt.get("--circuit"), coef=float(opt.get("--coef", "1e-4")),
                       tuned=opt.get("--cupp-tuned") == "1", vram=float(opt.get("--vram-frac", "0.9")),
                       wall=None, value=None, exit=None, fatal=False, parts={}, log=name, binary=None)
            runs.append(cur)
            continue
        if cur is None:
            continue
        if line.startswith("binary="):
            cur["binary"] = line.strip().split("=", 1)[1]
        if "FATAL" in line or "failed at operation" in line or "out of memory" in line:
            cur["fatal"] = True
        m = re.match(r"expectation=(\S+)", line)
        if m:
            cur["value"] = float(m.group(1))
        m = re.match(r"wall=([\d.]+) s", line)
        if m:
            cur["wall"] = float(m.group(1))
            for k, v in re.findall(r"(operations|partition|host->gpu|gpu->host|idle|planning)=([\d.]+)", line):
                cur["parts"][k] = float(v)
            w = re.search(r"summed over (\d+) GPU workers", line)
            if w:
                cur["parts"]["workers"] = int(w.group(1))
        m = re.search(r"between nodes: ([\d.]+) GB sent, ([\d.]+) s per node", line)
        if m:
            cur["parts"]["exchange"] = float(m.group(2))
        m = re.search(r"windows running=([\d.]+), waiting for the slowest node=([\d.]+)", line)
        if m:
            cur["parts"]["running"], cur["parts"]["waiting"] = float(m.group(1)), float(m.group(2))
        m = re.match(r"solve_epoch=([\d.]+)\.\.([\d.]+)", line)
        if m:
            cur["epoch"] = (float(m.group(1)), float(m.group(2)))
        if line.startswith("cupp phases"):
            cur["phases"] = {k: (float(a), float(b)) for k, a, b in re.findall(r"(\w+)=([\d.]+) \(([\d.]+)\)", line)}
        m = re.match(r"cupp mpi, seconds per rank: collective=([\d.]+) point_to_point=([\d.]+) total=([\d.]+)", line)
        if m:
            cur["mpi"] = {"collective": float(m.group(1)), "p2p": float(m.group(2)), "total": float(m.group(3))}
        m = re.match(r"cupp mpi (\w+): calls/rank=(\d+) \(([\d.]+) per operation\) seconds/rank=([\d.]+) GB/rank=([\d.]+)", line)
        if m:
            cur.setdefault("mpi_kinds", {})[m.group(1)] = dict(calls=int(m.group(2)), per_op=float(m.group(3)), seconds=float(m.group(4)), gb=float(m.group(5)))
        m = re.match(r"max_terms_at_window_boundary=(\d+)", line)
        if m:
            cur["boundary"] = int(m.group(1))
        m = re.match(r"max_terms_inside_a_window=(\d+)", line)
        if m:
            cur["inside"] = int(m.group(1))
        m = re.match(r"host memory in chunks: ([\d.]+) GiB on the busiest node, ([\d.]+) GiB over all nodes", line)
        if m:
            cur["busiest_gib"], cur["host_gib"] = float(m.group(1)), float(m.group(2))
        m = re.match(r"integrity: (\d+) term checks", line)
        if m:
            cur["checks"] = int(m.group(1))
        m = re.match(r"exit=(\d+)", line)
        if m:
            cur["exit"] = int(m.group(1))
    return runs


# options that mark a run as a variant of the engine (a default switched off or changed)
VARIANT = ("--early-per-node", "--share-above", "--defer-dedup", "--min-bucket", "--growth-cap", "--busy-window", "--min-bucket-early")


def ok(r):
    return r is not None and r["wall"] is not None and r["exit"] == 0 and not r["fatal"]


def find(runs, kind, circuit, coef, gpus=16, tuned=False, last=True, **opt):
    hits = [r for r in runs if r["kind"] == kind and r["circuit"] == circuit and abs(r["coef"] / coef - 1) < 1e-9
            and r["gpus"] == gpus and r["tuned"] == tuned and all(r["opt"].get(k) == v for k, v in opt.items())
            and not any(k in r["opt"] for k in VARIANT)]
    done = [r for r in hits if ok(r)]
    if tuned and done:          # the hand-tuned baseline: the fastest deduplication interval that fits
        return min(done, key=lambda r: r["wall"])
    pool = done or hits
    return (pool[-1] if last else pool[0]) if pool else None


def secs(v):
    return f"{v:.2f}" if v < 1 else (f"{v:.1f}" if v < 100 else f"{v:.0f}")


def times(b, e):
    x = b / e
    return f"{x:.1f}×" if x < 10 else f"{x:.0f}×"


def clean(ax, ygrid=True):
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=1.0)
    ax.tick_params(length=0)


OUT = HERE          # --out DIR writes the figures elsewhere (drafts)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------------------
# background: cuPauliProp's distributed mode (logs/pp_prof1-59470347.out, pp_prof2-59470887.out,
# pp_prof4n-59470888.out; kicked Ising, cutoff 1e-4, 22.5 million terms, original circuit)
# ---------------------------------------------------------------------------------------
def background():
    gpus = ["1", "4", "8", "16"]
    panels = [
        ("bg_scaling.png", [2.92, 32.9, 64.8, 71.2], "Time (s)", "{:.0f}", (1, 300)),
        ("bg_gatecost.png", [0.05, 5.9, 22.5, 29.8], "Fixed Cost per Gate (ms)", "{:.1f}", (0.01, 200)),
        ("bg_capacity.png", [96.0, 95.8, 103.7, 87.9], "Terms per GPU\nat the Failure Point (millions)", "{:.0f}", (1, 1000)),
    ]
    for name, vals, ylabel, fmt, ylim in panels:
        fig, ax = plt.subplots(figsize=(4.3, 3.5))
        ax.bar(gpus, vals, width=0.62, color=BASE)
        for x, v in zip(gpus, vals):
            ax.text(x, v * 1.12, ("%.2f" % v) if v < 1 else fmt.format(v), ha="center", va="bottom", fontsize=13)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(*ylim)
        ax.set_xlabel("GPUs (cuPauliProp distributed)")
        ax.set_ylabel(ylabel)
        clean(ax)
        save(fig, name)


# ---------------------------------------------------------------------------------------
# speed against cuPauliProp's distributed mode, 16 GPUs
# ---------------------------------------------------------------------------------------
MEDIUM = [("ki_lab", 1e-4, "Kicked\nIsing", {}), ("tfim_hh", 1e-4, "TFIM\nheavy-hex", {}),
          ("tfim_sq", 1e-4, "TFIM\n11×11", {"--lx": "11"}), ("xxz", 1e-3, "XXZ\n6×6", {}),
          ("qaoa", 1e-4, "QAOA", {}), ("hea", 1e-5, "Ansatz", {}), ("ki_noisy", 1e-4, "Noisy\nKicked\nIsing", {})]
LARGE3 = [("ki_lab", 1e-5, "Kicked\nIsing", {}), ("tfim_hh", 1e-5, "TFIM\nheavy-hex", {}), ("ki_noisy", 1e-5, "Noisy\nKicked\nIsing", {})]
LARGE = [("ki_lab", 1e-5, "Kicked\nIsing", {}), ("tfim_hh", 1e-5, "TFIM\nheavy-hex", {}),
         ("tfim_sq", 1e-5, "TFIM\n11×11", {"--lx": "11"}), ("ki_noisy", 1e-5, "Noisy\nKicked\nIsing", {})]


def triple(runs_first, runs_all, c, coef, opt, gpus=16):
    """(engine, tuned baseline, default baseline) for one setting; prefers runs on the same nodes."""
    out = []
    for kind, tuned in (("engine", False), ("cupp", True), ("cupp", False)):
        r = find(runs_first, kind, c, coef, gpus=gpus, tuned=tuned, **opt)
        if not ok(r):
            r2 = find(runs_all, kind, c, coef, gpus=gpus, tuned=tuned, **opt)
            r = r2 if (ok(r2) or r is None) else r
        out.append(r)
    return out


def speed(same16, all16, big):
    """Three panels: fits one GPU (16 GPUs); beyond one GPU on 16 and on 32 GPUs of the same nodes."""
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 3.9), gridspec_kw={"width_ratios": [7, 3.3, 3.3], "wspace": 0.13})
    table = []
    panels = ((axes[0], MEDIUM, "Fits One GPU, 16 GPUs", same16, all16, 16),
              (axes[1], LARGE3, "Beyond One GPU, 16 GPUs", big, big + all16, 16),
              (axes[2], LARGE3, "Beyond One GPU, 32 GPUs", big, big, 32))
    for ax, sets, title, first, pool, g in panels:
        w = 0.25
        for i, (c, coef, label, opt) in enumerate(sets):
            e, t, d = triple(first, pool, c, coef, opt, gpus=g)
            table.append((label.replace("\n", " "), coef, g, e, t, d))
            for j, (r, col) in enumerate(((e, OURS), (t, TUNED), (d, BASE))):
                x = i + (j - 1) * w
                if ok(r):
                    ax.bar(x, r["wall"], width=w * 0.94, color=col)
                    if j > 0 and ok(e):   # multiples: the tuned label grows to the left, the default label to the right
                        ax.text(x + (w * 0.47 if j == 1 else -w * 0.47), r["wall"] * 1.12, times(r["wall"], e["wall"]),
                                ha="right" if j == 1 else "left", va="bottom", fontsize=10.5)
                elif r is not None:
                    ax.text(x, 0.075, "out of\nmemory" if r["fatal"] else "time\nlimit", ha="center", va="bottom", fontsize=9.5, rotation=90, color=col)
        ax.set_yscale("log")
        ax.set_ylim(0.06, 400)
        ax.set_xlim(-0.6, len(sets) - 0.4)
        ax.set_xticks(range(len(sets)))
        ax.set_xticklabels([s[2] for s in sets], fontsize=11.5)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title, fontsize=13.5, pad=8)
        clean(ax)
    axes[0].set_ylabel("Time (s)")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (OURS, TUNED, BASE)]
    fig.legend(handles, ["SectorQ", "cuPauliProp distributed, dedup deferred", "cuPauliProp distributed"], loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.08), fontsize=12.5, columnspacing=2.2, handlelength=1.4)
    save(fig, "eval_speed.png")
    return table


# ---------------------------------------------------------------------------------------
# number of GPUs and reach (kicked Ising)
# ---------------------------------------------------------------------------------------
def gpus_and_reach(run4, run5, allruns, big):
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 3.8), gridspec_kw={"width_ratios": [6.4, 3.0], "wspace": 0.2})
    ax = axes[0]
    counts = [1, 2, 4, 8, 16, 32]
    rows = []
    w = 0.26
    for i, g in enumerate(counts):
        e = find(run5 + big, "engine", "ki_lab", 1e-5, gpus=g)                       # one GPU: batch 5; the rest: final job
        d = find(run4 + big, "cupp", "ki_lab", 1e-5, gpus=g)
        t = find(run5 + big, "cupp", "ki_lab", 1e-5, gpus=g, tuned=True)
        rows.append((g, e, t, d))
        for j, (r, col) in enumerate(((e, OURS), (t, TUNED), (d, BASE))):
            x = i + (j - 1) * w
            if ok(r):
                ax.bar(x, r["wall"], width=w * 0.94, color=col)
                label = secs(r["wall"]) if j == 0 else times(r["wall"], e["wall"])
                # the engine's time and the tuned multiple grow to the left, the default multiple to the right
                ax.text(x + (-w * 0.47 if j == 2 else w * 0.47), r["wall"] * 1.1, label,
                        ha="left" if j == 2 else "right", va="bottom", fontsize=10.5)
            elif r is not None or (g == 1 and j == 2):   # one GPU, default: fails at 96 million terms (logs/pp_prof2-59470887.out)
                ax.text(x, 6.3, "out of memory", ha="center", va="bottom", fontsize=9.5, rotation=90, color=col)
    ax.set_yscale("log")
    ax.set_ylim(5.5, 700)
    ax.set_xlim(-0.6, len(counts) - 0.4)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([str(g) for g in counts])
    ax.set_xlabel("GPUs (kicked Ising, cutoff 1e-5, up to 1.4 billion terms)")
    ax.set_ylabel("Time (s)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    clean(ax)

    ax = axes[1]
    reach = []
    for i, coef in enumerate([1e-5, 3e-6, 1e-6]):
        e = find(allruns, "engine", "ki_lab", coef)
        w = 0.38
        cands = [r for r in allruns if r["kind"] == "cupp" and r["circuit"] == "ki_lab" and abs(r["coef"] / coef - 1) < 1e-9 and r["gpus"] == 16 and not r["tuned"]]
        d = next((r for r in cands if ok(r)), cands[-1] if cands else None)
        reach.append((coef, e, d))
        if ok(e):
            ax.bar(i - w / 2, e["wall"], width=w * 0.94, color=OURS)
            ax.text(i - w / 2, e["wall"] * 1.1, secs(e["wall"]), ha="center", va="bottom", fontsize=11.5)
        elif e is not None and e["exit"] == 124:
            ax.text(i - w / 2, 6.3, "over 25 minutes", ha="center", va="bottom", fontsize=10.5, rotation=90, color=OURS)
        if ok(d):
            ax.bar(i + w / 2, d["wall"], width=w * 0.94, color=BASE)
            ax.text(i + w / 2, d["wall"] * 1.1, times(d["wall"], e["wall"]) if ok(e) else secs(d["wall"]), ha="center", va="bottom", fontsize=11.5)
        elif d is not None:
            ax.text(i + w / 2, 6.3, "out of memory" if d["fatal"] else "time limit", ha="center", va="bottom", fontsize=10.5, rotation=90, color=BASE)
    ax.set_yscale("log")
    ax.set_ylim(5.5, 700)
    ax.set_xlim(-0.6, 2.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["1e-5", "3e-6", "1e-6"])
    ax.set_xlabel("Cutoff (kicked Ising, 16 GPUs)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    clean(ax)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in (OURS, TUNED, BASE)], ["SectorQ", "cuPauliProp distributed, dedup deferred", "cuPauliProp distributed"],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07), fontsize=12.5, columnspacing=2.2, handlelength=1.4)
    save(fig, "eval_gpus.png")
    return rows, reach


# ---------------------------------------------------------------------------------------
# scaling on 4-128 GPUs, 40 GB nodes at 36 GB per GPU: engine from job set e13 (default settings),
# cuPauliProp from job set e12
# ---------------------------------------------------------------------------------------
SCALE3 = [("ki_lab", "Kicked Ising", 3e-6), ("tfim_hh", "TFIM heavy-hex", 5e-6), ("ki_noisy", "Noisy Kicked Ising", 5e-6)]
WEAK_CUTOFF = {4: 1e-5, 8: 7e-6, 16: 4.8e-6, 32: 3.4e-6, 64: 2.3e-6, 128: 1.6e-6}


def cut(v):
    """A cutoff as 3e-6 instead of 3e-06."""
    return f"{v:g}".replace("e-0", "e-")


def many(pattern):
    out = []
    for path in sorted(glob.glob(os.path.join(LOGS, pattern))):
        out += parse(os.path.basename(path))
    return out


# Runs that try a setting of the sector rules (the small-problem tests of 2026-10-12) are not results of the campaign.
TEST_FLAGS = ("--min-bucket", "--min-bucket-early", "--early-per-node", "--min-bucket-early-end")
NEW_ENGINE = "pp_engine_i"     # the exchange between nodes in rounds (2026-10-12): where a problem was run again with it, that run counts


def newest(runs):
    """The runs of the newer engine among `runs` if there are any, else all of them."""
    return [r for r in runs if r.get("binary") == NEW_ENGINE] or runs


def scale_rec(*ids):
    """The first of these runs of the scale campaign that has a record (queue_s/done/<id>), as in campaign g (status, wall);
    runs that were not made (skipped after a failure, predicted not to fit) count as absent."""
    for i in ids:
        try:
            r = json.load(open(os.path.join(os.path.dirname(HERE), "queue_s", "done", i)))
        except (OSError, ValueError):
            continue
        if r.get("status") in ("skipped_after_failure", "predicted_oom", "predicted_tl", "superseded"):
            continue
        # Some adapters write a solver result before MPI teardown hangs.
        # Preserve its queue record but do not plot a failed process as success.
        if r.get("status") == "ok" and r.get("exit") not in (None, 0):
            return dict(r, status="result_exit_nonzero", wall=None)
        return r
    return None


def as_big(r):
    """A record of the scale campaign in the form of the logs of the scaling jobs (wall, exit, fatal), for ok()."""
    if r is None:
        return None
    good = r.get("status") == "ok" and r.get("wall") is not None
    return dict(wall=r.get("wall") if good else None, exit=0 if good else 1, fatal=r.get("status") == "oom")


def big_logs():
    """The runs on 32 GPUs and more: the scaling jobs e15 and the scale campaign (queue_s; one log per run in logs/s).
    A run that the campaign took over from e15 appears twice with the same values."""
    out = many("pp_e15_g*-*.out")
    for path in sorted(glob.glob(os.path.join(LOGS, "s", "*.out"))):
        out += parse(os.path.join("s", os.path.basename(path)))
    return [r for r in out if not any(k in r["opt"] for k in TEST_FLAGS)]


def default_run(runs, circuit, coef, gpus):
    """The engine with its default sector limit and chunk size."""
    hits = [r for r in runs if r["kind"] == "engine" and r["circuit"] == circuit and abs(r["coef"] / coef - 1) < 1e-9
            and r["gpus"] == gpus and "--gmax" not in r["opt"] and "--chunk" not in r["opt"] and ok(r)]
    return newest(hits)[-1] if hits else None


def scaling(e13):
    counts = [g for g in (4, 8, 16, 32, 64, 128) if default_run(e13, "ki_lab", 3e-6, g)]
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 6.9), gridspec_kw={"wspace": 0.2, "hspace": 0.5})
    strong, weak = [], []
    for j, (c, title, coef) in enumerate(SCALE3):
        # strong scaling: the same problem on every GPU count
        ax = axes[0][j]
        rs = [default_run(e13, c, coef, g) for g in counts]
        t0, g0 = rs[0]["wall"], counts[0]
        for i, (g, r) in enumerate(zip(counts, rs)):
            if r is None:
                continue
            ax.bar(i, r["wall"], width=0.62, color=OURS)
            ideal = t0 * g0 / g
            ax.plot([i - 0.38, i + 0.38], [ideal, ideal], color=INK, linewidth=1.6)
            ax.text(i, r["wall"] * 1.12, secs(r["wall"]) + " s" if i == 0 else times(t0, r["wall"]), ha="center", va="bottom", fontsize=12)
            strong.append((title, coef, g, r))
        ax.set_yscale("log")
        ax.set_ylim(t0 / (counts[-1] / g0) / 2.2, t0 * 2.6)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels([str(g) for g in counts])
        ax.set_xlabel(f"GPUs (cutoff {cut(coef)})")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        if j == 0:
            ax.set_ylabel("Time (s)")
        clean(ax)
        # weak scaling: the cutoff is lowered with the GPU count, the rate is terms per second and GPU
        ax = axes[1][j]
        rs = [default_run(e13, c, WEAK_CUTOFF[g], g) for g in counts]
        rate = [None if r is None else r["checks"] / r["wall"] / g for g, r in zip(counts, rs)]
        for i, (g, r) in enumerate(zip(counts, rs)):
            if r is None:
                continue
            pct = 100 * rate[i] / rate[0]
            ax.bar(i, pct, width=0.62, color=OURS)
            ax.text(i, 103, f"{pct:.0f}%", ha="center", va="bottom", fontsize=12 if len(counts) <= 5 else 11)      # all labels in one row above the ideal line
            weak.append((title, g, r, rate[i], pct))
        ax.axhline(100, color=INK, linewidth=1.6)
        ax.set_ylim(0, 118)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels([f"{g}\n{cut(WEAK_CUTOFF[g])}" for g in counts], fontsize=11 if len(counts) <= 5 else 9.5)
        ax.set_xlabel("GPUs and Cutoff of the Run")
        if j == 0:
            ax.set_ylabel("Terms per Second per GPU\n(% of 4 GPUs)")
        clean(ax)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=OURS), plt.Line2D([0], [0], color=INK, linewidth=1.6)], ["SectorQ", "Ideal scaling from 4 GPUs"],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.985), fontsize=12.5, columnspacing=2.2, handlelength=1.4)
    for ax, label in ((axes[0][0], "Strong Scaling"), (axes[1][0], "Weak Scaling")):
        box = ax.get_position()
        fig.text(0.028, (box.y0 + box.y1) / 2, label, fontsize=13.5, fontweight="bold", rotation=90, ha="center", va="center")
    save(fig, "eval_scaling.png")
    return strong, weak


def scaling_baselines(e13, e12):
    """Cutoff 1e-5 on 16 GPUs and more: the engine against both cuPauliProp modes on the same kind of node."""
    counts = [g for g in (16, 32, 64, 128) if default_run(e13, "ki_lab", 1e-5, g)]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.8), gridspec_kw={"wspace": 0.16})
    rows = []
    w = 0.27
    for j, (c, title, _) in enumerate(SCALE3):
        ax = axes[j]
        for i, g in enumerate(counts):
            e = default_run(e13, c, 1e-5, g)
            t = find(e12, "cupp", c, 1e-5, gpus=g, tuned=True)
            d = find(e12, "cupp", c, 1e-5, gpus=g)
            rows.append((title, g, e, t, d))
            for k, (r, col) in enumerate(((e, OURS), (t, TUNED), (d, BASE))):
                if not ok(r):
                    continue
                x = i + (k - 1) * w
                ax.bar(x, r["wall"], width=w * 0.94, color=col)
                # the engine's time and the tuned multiple grow to the left, the default multiple to the right
                ax.text(x + (-w * 0.47 if k == 2 else w * 0.47), r["wall"] * 1.1, secs(r["wall"]) if k == 0 else times(r["wall"], e["wall"]),
                        ha="left" if k == 2 else "right", va="bottom", fontsize=10.5)
        ax.set_yscale("log")
        ax.set_ylim(4.5, 220)
        ax.set_xlim(-0.6, len(counts) - 0.4)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels([str(g) for g in counts])
        ax.set_xlabel("GPUs (cutoff 1e-5)")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        if j == 0:
            ax.set_ylabel("Time (s)")
        else:
            ax.set_yticklabels([])
        clean(ax)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in (OURS, TUNED, BASE)], ["SectorQ", "cuPauliProp distributed, dedup deferred", "cuPauliProp distributed"],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.1), fontsize=12.5, columnspacing=2.2, handlelength=1.4)
    save(fig, "eval_scaling_base.png")
    return rows


# ---------------------------------------------------------------------------------------
# exact-value check: 5x6 square-lattice TFIM, 30 qubits (logs/pp_e2b-59475514.out,
# logs/inter_run2.out, logs/inter_run3.out; exact values from analysis/exact_sv.py)
# ---------------------------------------------------------------------------------------
EXACT = {4: 0.802867136639, 6: 0.790133273060}
PP = {4: {1e-3: 0.78343, 1e-4: 0.79018, 1e-5: 0.79905, 1e-6: 0.80155, 1e-7: 0.80280},
      6: {1e-3: 0.72858, 1e-4: 0.74777, 1e-5: 0.78088, 3e-6: 0.77863, 1e-6: 0.78341}}


def accuracy():
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    for steps, col, mk in ((4, OURS, "o"), (6, "#0E5A52", "s")):
        cs = sorted(PP[steps], reverse=True)
        err = [abs(PP[steps][c] - EXACT[steps]) for c in cs]
        ax.plot(cs, err, color=col, marker=mk, markersize=8, linewidth=2.2, label=f"{steps} steps")
        ax.text(cs[-1] / 1.25, err[-1], f"{err[-1]:.5f}" if err[-1] < 1e-3 else f"{err[-1]:.4f}", ha="left", va="center", fontsize=11.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2e-3, 2.2e-8)
    ax.set_ylim(3e-5, 0.2)
    ax.set_xlabel("Cutoff (smaller keeps more terms)")
    ax.set_ylabel("Error against the Exact Value")
    ax.grid(color=GRID, linewidth=1.0)
    ax.tick_params(length=0)
    ax.legend(loc="lower left", fontsize=12.5)
    save(fig, "eval_accuracy.png")


# ---------------------------------------------------------------------------------------
# where the time goes, 16 GPUs
# ---------------------------------------------------------------------------------------
def breakdown(big):
    sets = [("ki_lab", 1e-5, 16, "Kicked Ising, 16 GPUs"), ("ki_lab", 1e-5, 32, "Kicked Ising, 32 GPUs"),
            ("tfim_hh", 1e-5, 32, "TFIM heavy-hex, 32 GPUs"), ("ki_noisy", 1e-5, 32, "Noisy Kicked Ising, 32 GPUs")]
    parts = [("Gates on GPU", "operations", OURS), ("Exchange Between Nodes", "exchange", "#C0432F"),
             ("GPU-Host Copies", "copies", "#F5B553"), ("Partition", "partition", "#7FC8BE"),
             ("Idle GPU", "idle", "#B9BDC4"), ("Waiting for Other Nodes", "waiting", "#8E96A3"), ("Planning and Other", "other", "#E3E4E8")]
    fig, ax = plt.subplots(figsize=(10.8, 2.9))
    out, ylabels = [], []
    for c, coef, g, label in sets:
        r = find(big, "engine", c, coef, gpus=g)
        if not ok(r) or "workers" not in r["parts"]:
            continue
        p, n = r["parts"], r["parts"]["workers"]
        share = {"operations": p.get("operations", 0.0) / n, "copies": (p.get("gpu->host", 0.0) + p.get("host->gpu", 0.0)) / n,
                 "partition": p.get("partition", 0.0) / n, "idle": p.get("idle", 0.0) / n,
                 "waiting": p.get("waiting", 0.0), "exchange": p.get("exchange", 0.0)}
        share = {k: v / r["wall"] for k, v in share.items()}
        share["other"] = max(0.0, 1.0 - sum(share.values()))
        out.append((label, r["wall"], share))
        left, i = 0.0, len(ylabels)
        for name, key, col in parts:
            ax.barh(i, share[key] * 100, left=left, color=col, height=0.62, label=name if i == 0 else None)
            if share[key] >= 0.06:
                ax.text(left + share[key] * 50, i, f"{share[key] * 100:.0f}%", ha="center", va="center", fontsize=11.5,
                        color="white" if key in ("operations", "exchange", "waiting") else INK)
            left += share[key] * 100
        ylabels.append(f"{label}\n({secs(r['wall'])} s)")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the Run Time (%)")
    ax.tick_params(length=0)
    ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.42, 1.36), fontsize=11.5, columnspacing=1.4, handlelength=1.2)
    save(fig, "eval_breakdown.png")
    return out


# ---------------------------------------------------------------------------------------
# where the time goes on 16 GPUs: SectorQ next to cuPauliProp's distributed mode (logs/u7, jobs/u7_breakdown.sh)
# ---------------------------------------------------------------------------------------
def gpu_busy(prefix, window):
    """Mean GPU utilization (%) of all GPUs of all nodes inside the timed part of a run (nvidia-smi samples)."""
    if not window:
        return None
    vals = []
    for path in glob.glob(prefix + ".*.csv"):
        for line in open(path, errors="replace"):
            f = [x.strip() for x in line.split(",")]
            try:
                ts = datetime.strptime(f[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()   # local time, as on the nodes
                if window[0] <= ts <= window[1]:
                    vals.append(float(f[2]))
            except (ValueError, IndexError):
                continue
    return sum(vals) / len(vals) if vals else None


BASE_PARTS = [("Gates on GPU", "gpu", OURS), ("Library on the Host, GPU Idle", "host", "#8E96A3"),
              ("Collective Calls Among All GPUs", "collective", "#5B3A8C"), ("Moving Terms Between GPUs", "moving", "#C0432F"),
              ("GPU-Host Copies and Partition", "copies", "#F5B553"), ("Idle or Waiting GPU", "idle", "#CFD3D9"),
              ("Other", "other", "#ECEDEF")]


def shares(kind, r, smi_prefix):
    """Seconds and shares of one run by BASE_PARTS. cuPauliProp: seconds per rank from the profiling plugin
    (every MPI call of the library) and from timers around the library calls; the time inside the library calls
    and outside MPI is split by the sampled GPU utilization into the part with a busy GPU and the rest."""
    if not ok(r):
        return None
    wall = r["wall"]
    busy = gpu_busy(smi_prefix, r.get("epoch"))
    if kind == "eng":
        # time inside the single-GPU library calls of the workers, split by the sampled GPU utilization like the time
        # inside the library calls of cuPauliProp's distributed mode: the same rule for both (without samples: not split)
        p, n = r["parts"], r["parts"]["workers"]
        lib = p.get("operations", 0.0) / n
        gpu = lib if busy is None else min(lib, busy / 100.0 * wall)
        sec = {"gpu": gpu, "host": lib - gpu, "collective": 0.0, "moving": p.get("exchange", 0.0),
               "copies": (p.get("gpu->host", 0.0) + p.get("host->gpu", 0.0) + p.get("partition", 0.0)) / n,
               "idle": p.get("idle", 0.0) / n + p.get("waiting", 0.0)}
    else:
        if "phases" not in r or "mpi" not in r:
            return None
        inside = sum(a for a, _ in r["phases"].values())
        local = inside - sum(b for _, b in r["phases"].values())      # inside the library calls, outside MPI
        gpu = min(local, (busy or 0.0) / 100.0 * wall)
        sec = {"gpu": gpu, "host": local - gpu, "collective": r["mpi"]["collective"], "moving": r["mpi"]["p2p"],
               "copies": 0.0, "idle": 0.0}
    sec["other"] = max(0.0, wall - sum(sec.values()))
    return dict(wall=wall, sec=sec, share={k: v / wall for k, v in sec.items()}, busy=busy, run=r)


def draw_shares(groups, name):
    """Stacked bars: groups is [(title, [(label, data of shares()), ...]), ...]."""
    rows = [(title if j == 0 else None, lab, d) for title, group in groups for j, (lab, d) in enumerate([x for x in group if x[1]])]
    if not rows:
        return []
    fig, ax = plt.subplots(figsize=(10.8, 0.52 * len(rows) + 0.5 * len(groups) + 1.3))
    y, ticks, labels, out, seen = 0.0, [], [], [], set()
    for title, lab, d in rows:
        if title:
            y += 0.75 if ticks else 0.0
            ax.text(0, y - 0.62, title, ha="left", va="center", fontsize=12.5, fontweight="bold")
        left = 0.0
        for pname, key, col in BASE_PARTS:
            v = d["share"][key] * 100
            if v <= 0:
                continue
            ax.barh(y, v, left=left, color=col, height=0.62, label=pname if pname not in seen else None)
            seen.add(pname)
            if v >= 6:
                ax.text(left + v / 2, y, f"{v:.0f}%", ha="center", va="center", fontsize=11.5,
                        color="white" if key in ("gpu", "host", "collective", "moving") else INK)
            left += v
        ticks.append(y)
        labels.append(f"{lab} ({secs(d['wall'])} s)")
        out.append((title, lab, d))
        y += 1.0
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(ticks[-1] + 0.55, ticks[0] - 1.0)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the Run Time (%)")
    ax.tick_params(length=0)
    handles, names = ax.get_legend_handles_labels()
    order = [names.index(pn) for pn, _, _ in BASE_PARTS if pn in names]
    ax.legend([handles[i] for i in order], [names[i] for i in order], loc="lower center", ncol=4, bbox_to_anchor=(0.40, 1.0),
              fontsize=11.5, columnspacing=1.3, handlelength=1.2)
    save(fig, name)
    return out


def breakdown_base(folder="u7"):
    """Stacked bars of the run time on 16 GPUs: SectorQ, cuPauliProp distributed (default), and cuPauliProp with
    the deduplication deferred to the truncations."""
    sets = [("ki.1e-4", "Kicked Ising, Cutoff 1e-4"), ("ki.1e-5", "Kicked Ising, Cutoff 1e-5"),
            ("tfim.1e-5", "TFIM Heavy-Hex, Cutoff 1e-5"), ("noise.1e-5", "Noisy Kicked Ising, Cutoff 1e-5")]

    def one(kind, name):
        runs = parse(os.path.join(folder, f"{kind}.{name}.out"))
        return shares(kind, runs[-1] if runs else None, os.path.join(LOGS, folder, f"smi.{kind}.{name}"))

    return draw_shares([(title, [("SectorQ", one("eng", name)), ("cuPauliProp", one("cupp", name)),
                                 ("cuPauliProp, Merge Deferred", one("cupp", name + ".defer"))]) for name, title in sets], "eval_breakdown_base.png")


# ---------------------------------------------------------------------------------------
# campaign g: everything on 1-16 GPUs (40 GB nodes at 36 GB per GPU), one run per item of ../queue_g. The record
# of a run that ended is ../queue_g/done/<id> (status, solve seconds, expectation value, log); engine and
# cuPauliProp logs are ../logs/g/<id>.out, external baselines ../results/g_ext/<id>.launch.json
# ---------------------------------------------------------------------------------------
QG = os.path.join(HERE, "..", "queue_g")
G3 = [("ki", "Kicked Ising"), ("tfim", "TFIM heavy-hex"), ("noise", "Noisy Kicked Ising")]
G7 = [("ki", "Kicked\nIsing"), ("tfim", "TFIM\nheavy-hex"), ("tfimsq", "TFIM\n11×11"), ("xxz", "XXZ\n8×8"),
      ("qaoa", "QAOA"), ("hea", "Ansatz"), ("noise", "Noisy\nKicked Ising")]
GPUS_G = (1, 2, 4, 8, 16)
WHY = {"oom": "out of\nmemory", "tl": "time\nlimit", "predicted_tl": "time\nlimit"}
WEAK_G = {1: "2e-5", 2: "1.4e-5", 4: "1e-5", 8: "7e-6", 16: "4.8e-6"}


QR = os.path.join(HERE, "..", "queue_r")
QO = os.path.join(HERE, "..", "queue_o")     # campaign o: cuPauliProp on the original circuits (jobs/o_items.py)


def campaign_g():
    """{item id: record} of the runs on 1-16 GPUs that ended: campaign g and, over it, the runs that campaign r made
    again with the engine whose exchange between nodes runs in rounds (jobs/r_items.py, r_trim.py: the multi-node
    runs of 30 s and more, those that ended at a limit, and the time-breakdown runs). A record of campaign r with
    "reused" is the record of campaign g. The short runs that campaign r repeated before it was trimmed (the same
    values, 0.97-1.03 times the time) do not replace those of campaign g. Campaign o ran cuPauliProp's single-GPU and
    default distributed runs of the four circuits with Clifford gates on the circuit as it is; those records replace
    the ones of campaign g, in which cuPauliProp was given the operations that SectorQ's Clifford frame leaves."""
    out = {}
    for queue, again in ((QG, False), (QR, True), (QO, False)):      # campaign o replaces the cuPauliProp runs it repeated
        done = os.path.join(queue, "done")
        if not os.path.isdir(done):
            continue
        for name in os.listdir(done):
            try:
                rec = json.load(open(os.path.join(done, name)))
            except ValueError:
                continue
            if again:
                old = out.get(name)
                short = old is not None and old.get("status") == "ok" and (old.get("wall") or 0) < 30 and not name.startswith("bd.")
                if rec.get("reused") or short:
                    continue
            out[name] = rec
    return out


def gok(s):
    return bool(s) and s.get("status") == "ok" and s.get("wall") is not None


def gdetail(s):
    """The fields of parse() for a run with an engine or cuPauliProp log."""
    log = (s or {}).get("log") or ""
    runs = parse(log[len("logs/"):]) if log.startswith("logs/") and log.endswith(".out") else []
    return runs[-1] if runs else None


def gladder(G, prefix, suffix):
    """Cutoffs of the items <prefix>.<cutoff>.<suffix> that have a record, the largest (smallest problem) first."""
    pat = re.compile(re.escape(prefix) + r"\.([0-9.]+e-[0-9]+)\." + re.escape(suffix) + "$")
    return sorted((m.group(1) for m in map(pat.match, G) if m), key=float, reverse=True)


def greached(G, prefix, suffix, cutoffs):
    """`cutoffs` without the tail that no run reached: a ladder stops at its first failure, the steps after it were not started."""
    out = list(cutoffs)
    while out and (G.get(f"{prefix}.{out[-1]}.{suffix}") or {}).get("status") in (None, "skipped_after_failure"):
        out.pop()
    return out


def gwhy(G, prefix, coef, suffix):
    """Why a run has no time. A run that was not started takes the reason of the step of its ladder that failed."""
    s = G.get(f"{prefix}.{coef}.{suffix}")
    if s is None or gok(s):
        return None
    st = s["status"]
    if st == "skipped_after_failure":
        for x in gladder(G, prefix, suffix):
            if float(x) <= float(coef):
                break
            t = G[f"{prefix}.{x}.{suffix}"]["status"]
            st = t if t not in ("ok", "skipped_after_failure") else st
    return WHY.get(st, "failed")


def gdeferred(G, c, coef, g):
    """The deferred-deduplication baseline: the fastest of its three intervals that completed."""
    cands = [G.get(f"{t}.{c}.{coef}.g{g}") for t in ("defer", "defer3", "defer2")]
    done = [s for s in cands if gok(s)]
    return min(done, key=lambda s: s["wall"]) if done else next((s for s in cands if s), None)


def gterms(G, c, coef):
    """Terms at the window boundaries of one problem (from any SectorQ run of it)."""
    for g in reversed(GPUS_G):
        s = G.get(f"eng.{c}.{coef}.g{g}")
        if gok(s):
            d = gdetail(s)
            if d and d.get("boundary"):
                return d["boundary"]
    return None


def count(n):
    return f"{n / 1e9:.3g} B" if n >= 1e9 else (f"{n / 1e6:.3g} M" if n >= 1e6 else f"{n / 1e3:.3g} K")


# The look of these figures follows the evaluation of ScaleQsim (SIGMETRICS '26): time against the problem size or
# the GPU count, one line per implementation with a fixed color and marker, a legend strip above the panels, in the
# order single node -> weak scaling -> strong scaling -> other circuits -> scale-out.
SQ = {"ours": ("#1f77b4", "o"), "cupp": ("#d62728", "^"), "defer": ("#8c564b", "s"), "orqa": ("#800080", "v"), "monoprop": ("#FFA500", "d"),
      "spd": ("#008000", "X"), "julia": ("#17becf", "P"), "ppvm": ("#7f7f7f", "h"), "qiskit": ("#e377c2", "*")}
SQ_NAME = {"ours": "SectorQ", "cupp": "cuPauliProp", "defer": "cuPauliProp, dedup deferred", "orqa": "ORQA (reproduction)", "monoprop": "MonoProp",
           "spd": "SPD", "julia": "PauliPropagation.jl", "ppvm": "ppvm", "qiskit": "Qiskit pauli-prop"}
SHORT_WHY = {"oom": "out of memory", "tl": "over 40 min", "predicted_tl": "over 40 min"}
STOPS = False     # True: a cross where an implementation stops; False: its line just ends at the last run that finished
SIZES = False     # True: the size of the terms in bytes as a third line under a cutoff; False: the number of terms only


def xl_cut():
    return "Cutoff (# of Terms) (Size of Terms)" if SIZES else "Cutoff (# of Terms)"


def sq_axes(ax, xlabel=None, ylabel=None, log=True, size=16):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("black")
        s.set_linewidth(1.0)
    ax.grid(True, which="major", color="#B8B8B8", linewidth=0.6, linestyle="--")
    if log:
        ax.set_yscale("log")
        ax.grid(True, which="minor", axis="y", color="#DADADA", linewidth=0.4, linestyle="--")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(length=0, labelsize=size - 3)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=size, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=size, fontweight="bold")


def sq_runs(G, kind, c, cutoffs, suffix):
    """The records of one implementation over the cutoffs (None where there is no item or no record yet)."""
    if kind == "defer":
        return [gdeferred(G, c, x, int(suffix[1:])) for x in cutoffs]
    return [G.get(f"{'eng' if kind == 'ours' else kind}.{c}.{x}.{suffix}") for x in cutoffs]


def sq_fail(G, kind, c, cutoffs, suffix, recs):
    """(position, reason) of the first cutoff after the last finished one that failed, or None."""
    last = max((i for i, s in enumerate(recs) if gok(s)), default=-1)
    for i in range(last + 1, len(recs)):
        s = recs[i]
        if s is None or gok(s):
            continue
        st = s["status"]
        if st == "skipped_after_failure":
            tag = "defer" if kind == "defer" else ("eng" if kind == "ours" else kind)
            return i, {"out of\nmemory": "out of memory", "time\nlimit": "over 40 min"}.get(gwhy(G, f"{tag}.{c}", cutoffs[i], suffix), "failed")
        return i, SHORT_WHY.get(st, "failed")
    return None


def sq_line(ax, G, kind, c, cutoffs, suffix, label=None, ls="-", lw=3.6, ms=9.5, xs=None, mark_end=True):
    recs = sq_runs(G, kind, c, cutoffs, suffix)
    xs = list(range(len(cutoffs))) if xs is None else xs
    pts = [(xs[i], s["wall"]) for i, s in enumerate(recs) if gok(s)]
    col, mk = SQ[kind]
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker=mk, markersize=ms, linewidth=lw, linestyle=ls, label=label,
                zorder=4 if kind == "ours" else 3, solid_capstyle="round")
    fail = sq_fail(G, kind, c, cutoffs, suffix, recs)
    if pts and fail and mark_end and STOPS:           # the implementation stops here: a cross at the height of its last point
        ax.plot([xs[fail[0]]], [pts[-1][1]], color=col, marker="x", markersize=ms + 1, markeredgewidth=2.6, linestyle="none", zorder=5)
    return recs, pts, fail


def sq_reached(cutoffs, *records):
    """The cutoffs up to the last one that any of the record lists finished (all of them where crosses are drawn)."""
    if STOPS:
        return len(cutoffs)
    return 1 + max((i for recs in records for i, s in enumerate(recs) if gok(s)), default=len(cutoffs) - 1)


def sq_legend(fig, ax, y, ncol, extra=True, size=13.5):
    handles, names = ax.get_legend_handles_labels()
    if extra and STOPS:
        handles.append(plt.Line2D([0], [0], color="black", marker="x", markersize=10, markeredgewidth=2.6, linestyle="none"))
        names.append("stops here: out of memory or over 40 min")
    fig.legend(handles, names, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, y), fontsize=size, columnspacing=1.6, handlelength=2.4, frameon=False)


QUBITS = {"ki": 127, "tfim": 127, "noise": 127, "tfimsq": 121, "xxz": 64, "qaoa": 100, "hea": 100}


def term_bytes(c):
    """Bytes of one term: the X and Z bits of the Pauli string in 64-bit words, and a double coefficient."""
    return 2 * ((QUBITS[c] + 63) // 64) * 8 + 8


def data_size(nbytes):
    for unit, scale in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6)):
        if nbytes >= scale:
            return f"{nbytes / scale:.3g}{unit}"
    return f"{nbytes / 1e3:.3g}KB"


def cut_ticks(G, c, cutoffs, extra=None, more=None):
    """Tick labels: the cutoff and, below it, the number of terms of that problem and their size (or the text of `extra`).
    `more`: {cutoff: terms} of the problems that no run of campaign g finished."""
    out = []
    for i, x in enumerate(cutoffs):
        n = gterms(G, c, x) or (more or {}).get(x)
        second = extra[i] if extra is not None else ((f"({count(n).replace(' ', '')})" + (f"\n({data_size(n * term_bytes(c))})" if SIZES else "")) if n else "")
        out.append(f"{x}\n{second}")
    return out


def g_single_node(G, c="ki", title="Kicked Ising", ax=None):
    """One node (4 GPUs, 64 cores): SectorQ and every other implementation, time against the problem size.
    With `ax` (a panel of another figure, make_deck_figures.py) only the panel is drawn: no legend, no file."""
    cutoffs = gladder(G, f"eng.{c}", "g4") or gladder(G, f"eng.{c}", "g1")
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(15.2, 5.6))
    lines = [("ours", "g4", "-", "SectorQ (4 GPUs)"), ("ours", "g1", "--", "SectorQ (1 GPU)"),
             ("cupp", "g4", "-", "cuPauliProp (4 GPUs)"), ("cupp", "g1", "--", "cuPauliProp (1 GPU)"),
             ("defer", "g4", "-", "cuPauliProp, dedup deferred (4 GPUs)"), ("spd", "n1", "--", "SPD (1 GPU)"),
             ("orqa", "n1", "-", "ORQA, reproduction (64 cores)"), ("monoprop", "n1", "-", "MonoProp (64 cores)"),
             ("julia", "n1", "-", "PauliPropagation.jl (64 cores)"), ("ppvm", "n1", "-", "ppvm (64 cores)"), ("qiskit", "n1", "-", "Qiskit pauli-prop (1 core)")]
    rows, got = [], []
    for kind, suffix, ls, label in lines:
        recs, pts, fail = sq_line(ax, G, kind, c, cutoffs, suffix, label=label, ls=ls, lw=3.4 if kind == "ours" else 2.6, ms=9 if kind == "ours" else 8)
        got.append(recs)
        rows.append((label, [(cutoffs[i], s["wall"]) for i, s in enumerate(recs) if gok(s)], (cutoffs[fail[0]], fail[1]) if fail else None))
    n = sq_reached(cutoffs, *got)
    sq_axes(ax, xl_cut(), "Simulation Time (s)")
    ax.set_xticks(range(n))
    ax.set_xticklabels(cut_ticks(G, c, cutoffs[:n]))
    ax.set_xlim(-0.4, n - 0.6)
    ax.set_ylim(0.06, 5000)
    if fig is None:
        return rows
    sq_legend(fig, ax, 1.1, 4)
    save(fig, "eval_single_node.png")
    return rows


WEAK_S = {1: "6.2e-5", 2: "4.3e-5", 4: "3e-5", 8: "2.1e-5", 16: "1.5e-5"}
# 32-256 GPUs: the scaling jobs e15 (logs/pp_e15_g*-*.out), one log per GPU count; the weak-scaling cutoffs continue the large load
BIG_WEAK = {32: "3.4e-6", 64: "2.3e-6", 128: "1.6e-6", 256: "1.15e-6", 512: "8e-7"}
BIG_WEAK_S = {32: "1e-5", 64: "7.1e-6", 128: "4.9e-6", 256: "3.5e-6", 512: "2.4e-6"}    # the small load: 3e-5 * (GPUs/4)^-0.52


def big_weak(kind, load, c, x, g, big):
    """One point of weak scaling on more than 16 GPUs, as a record of campaign g (status, wall). SectorQ: its default
    settings in the scaling jobs and the scale campaign; cuPauliProp: given the original circuit where that run exists,
    else the list of the Clifford frame; the CPU implementations on the same nodes. On 32 GPUs the small load is cutoff
    1e-5 of the sweep, whose runs have the ids of the sweep."""
    sweep = load == "S" and x == "1e-5"
    if kind == "ours":
        b = big_run(big, c, x, g)
        return dict(status="ok", wall=b["wall"]) if b else None
    if kind == "cupp":
        ids = (f"o.cupp.{c}.{x}.g{g}", f"cupp.{c}.{x}.g{g}") if sweep else (f"o.weak{load}c.{c}.{x}.g{g}", f"weak{load}c.{c}.{x}.g{g}")
    elif kind == "defer":
        ids = (f"defer.{c}.{x}.g{g}",) if sweep else (f"weak{load}d.{c}.{x}.g{g}",)
    else:
        ids = (f"{kind}.{c}.{x}.n{g // 4}",) if sweep else (f"weak{load}.{kind}.{c}.{x}.g{g}",)
    return scale_rec(*ids)
BIG_NAME = {"ki": "ki_lab", "tfim": "tfim_hh", "noise": "ki_noisy"}


def big_run(big, c, coef, g):
    """SectorQ with its default settings on 32-256 GPUs."""
    return default_run(big, BIG_NAME[c], float(coef), g)


def g_weak(G, big=(), axes=None):
    """Weak scaling: the cutoff is lowered with the GPU count. One panel per circuit and load, one line per implementation.
    The large load goes on to the GPU counts of the scaling jobs (SectorQ only: the other implementations were not run there).
    With `axes` (panels of another figure, make_deck_figures.py; axes[circuit][load]) only the panels are drawn: no legend, no file."""
    fig = None
    if axes is None:
        fig, axes = plt.subplots(3, 2, figsize=(15.6, 12.4), gridspec_kw={"wspace": 0.14, "hspace": 0.52})
    rows = []
    for r, (load, cuts) in enumerate((("S", WEAK_S), ("L", WEAK_G))):
        extra = BIG_WEAK if load == "L" else BIG_WEAK_S
        more = [g for g in sorted(extra) if any(big_run(big, c, extra[g], g) for c, _ in G3)]
        counts = list(GPUS_G) + more
        cuts = {**cuts, **extra}
        for j, (c, title) in enumerate(G3):
            ax = axes[j][r]
            for kind in ("ours", "cupp", "defer", "orqa", "monoprop"):
                pts, stop = [], None
                for i, g in enumerate(counts):
                    x = cuts[g]
                    if g > 16:
                        s = big_weak(kind, load, c, x, g, big)
                    elif kind == "ours":
                        s = G.get(f"weak{load}.{c}.{x}.g{g}") or G.get(f"eng.{c}.{x}.g{g}")
                    elif kind == "cupp":
                        s = G.get(f"weak{load}c.{c}.{x}.g{g}") or G.get(f"cupp.{c}.{x}.g{g}")
                    elif kind == "defer":
                        s = None if g == 1 else (G.get(f"weak{load}d.{c}.{x}.g{g}") or gdeferred(G, c, x, g))
                    else:
                        s = G.get(f"{kind}.{c}.{x}.n1") if g == 4 else G.get(f"weak{load}.{kind}.{c}.{x}.g{g}")      # 4 GPUs: the sweep's run on one node
                    if gok(s):
                        pts.append((i, s["wall"]))
                    elif s is not None and stop is None and (not pts or i > pts[-1][0]):
                        stop = i
                    rows.append((load, title, SQ_NAME[kind], g, x, s))
                col, mk = SQ[kind]
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker=mk, markersize=10, linewidth=3.8 if kind == "ours" else 3.0,
                            label=SQ_NAME[kind] if (r == 0 and j == 0) else None, zorder=4 if kind == "ours" else 3)
                    if STOPS and stop is not None and stop > pts[-1][0]:
                        ax.plot([stop], [pts[-1][1]], color=col, marker="x", markersize=11, markeredgewidth=2.6, linestyle="none", zorder=5)
            sq_axes(ax, "# of GPUs (Cutoff)", "Simulation Time (s)" if r == 0 else None)
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels([f"{g}\n({cuts[g]})" for g in counts], fontsize=13 if len(counts) <= 5 else 10)
            ax.set_xlim(-0.35, len(counts) - 0.65)
            ax.set_ylim((1.5, 500) if load == "S" else (15, 1500))
            ax.set_title(f"({'abcdef'[2 * j + r]}) {title}, {'small' if load == 'S' else 'large'} load", fontsize=14.5, pad=8)
    if fig is None:
        return rows
    sq_legend(fig, axes[0][0], 0.945, 6)
    save(fig, "eval_weak.png")
    return rows


def g_strong(G, c="ki", cutoffs=("1e-4", "3e-5", "1e-5", "3e-6", "1e-6"), axes=None):
    """Strong scaling in the form of ScaleQsim: one panel per GPU count, time against the problem size; below each cutoff
    the speedup of SectorQ over cuPauliProp's distributed mode on the same GPUs.
    With `axes` (one panel per GPU count of another figure, make_deck_figures.py) only the panels are drawn."""
    panels = ((4, "g4", "n1", "4 GPUs (1 node)"), (8, "g8", "n2", "8 GPUs (2 nodes)"), (16, "g16", "n4", "16 GPUs (4 nodes)"))
    panels = [p for p in panels if any(gok(G.get(f"eng.{c}.{x}.{p[1]}")) for x in cutoffs)]
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.9), gridspec_kw={"wspace": 0.2}, squeeze=False)
        axes = axes[0]
    rows = []
    for j, (g, gs, ns, title) in enumerate(panels):
        ax = axes[j]
        got = {}
        for kind, suffix in (("ours", gs), ("cupp", gs), ("defer", gs), ("orqa", ns), ("monoprop", ns)):
            recs, pts, fail = sq_line(ax, G, kind, c, cutoffs, suffix, label=SQ_NAME[kind] if j == 0 else None, lw=3.8 if kind == "ours" else 3.0, ms=10)
            got[kind] = recs
            rows.append((title, SQ_NAME[kind], [(cutoffs[i], s["wall"]) for i, s in enumerate(recs) if gok(s)], (cutoffs[fail[0]], fail[1]) if fail else None))
        extra = []
        for a, b in zip(got["ours"], got["cupp"]):
            extra.append("(" + times(b["wall"], a["wall"]).replace("×", "x") + ")" if gok(a) and gok(b) else ("(OOM)" if gok(a) and b is not None else ""))
        n = sq_reached(cutoffs, *got.values())
        sq_axes(ax, "Cutoff / Speedup of SectorQ", "Simulation Time (s)" if j == 0 else None)
        ax.set_xticks(range(n))
        ax.set_xticklabels(cut_ticks(G, c, cutoffs[:n], extra[:n]))
        ax.set_xlim(-0.4, n - 0.6)
        ax.set_ylim(0.3, 4000)
        ax.set_title(f"({'abc'[j]}) {title}", fontsize=14.5, pad=8)
    if fig is None:
        return rows
    sq_legend(fig, axes[0], 1.1, 3 if len(panels) < 3 else 6)
    save(fig, "eval_strong.png")
    return rows


def g_circuits(G, g=16, axes=None, which=None, keep=None, sizes=None):
    """Seven circuits on 16 GPUs: SectorQ against cuPauliProp's two distributed modes, time against the problem size.
    Every cutoff of a panel has its tick; below it the speedup of SectorQ over cuPauliProp's distributed mode.
    With `axes` (panels of another figure, make_deck_figures.py) only the circuits `which` are drawn, one per panel;
    `keep(G, circuit, cutoffs, g)` then chooses the cutoffs of a panel (a slide shows fewer steps than the README).
    `sizes` ({circuit: {cutoff: terms}} for the problems no run of campaign g finished, or {}): the tick labels give
    the number of terms and their size instead of the speedup; the caller prints the speedups at the points."""
    fig, flat = None, axes
    if axes is None:
        fig, axes = plt.subplots(4, 2, figsize=(15.6, 15.4), gridspec_kw={"wspace": 0.14, "hspace": 0.5})
        flat = [a for row in axes for a in row]
    rows = []
    for j, (c, name) in enumerate(G7 if which is None else [(c, dict(G7)[c]) for c in which]):
        ax = flat[j]
        cutoffs = gladder(G, f"eng.{c}", f"g{g}")
        cutoffs = greached(G, f"eng.{c}", f"g{g}", [x for x in cutoffs if "." not in x and x not in ("15e-7", "12e-7")])
        if keep is not None:
            cutoffs = keep(G, c, cutoffs, g)
        got = {}
        for kind in ("ours", "cupp", "defer"):
            recs, pts, fail = sq_line(ax, G, kind, c, cutoffs, f"g{g}", label=SQ_NAME[kind] if j == 0 else None, lw=3.8 if kind == "ours" else 3.0, ms=9)
            got[kind] = recs
            rows.append((name.replace("\n", " "), SQ_NAME[kind], [(cutoffs[i], s["wall"]) for i, s in enumerate(recs) if gok(s)], (cutoffs[fail[0]], fail[1]) if fail else None))
        extra, stopped = [], False
        for a_, b_ in zip(got["ours"], got["cupp"]):
            stopped = stopped or (b_ is not None and not gok(b_))
            extra.append("(" + times(b_["wall"], a_["wall"]).replace("×", "x") + ")" if gok(a_) and gok(b_) else ("(OOM)" if gok(a_) and stopped else ""))
        n = sq_reached(cutoffs, *got.values()) if keep is None else len(cutoffs)
        sq_axes(ax, "Cutoff / Speedup of SectorQ" if sizes is None else xl_cut(), "Simulation Time (s)" if j % 2 == 0 else None, size=14)
        ax.set_xticks(range(n))
        labels = [f"{x}\n{e}" for x, e in zip(cutoffs, extra)] if sizes is None else cut_ticks(G, c, cutoffs, more=sizes.get(c))
        ax.set_xticklabels(labels[:n], fontsize=11 if n <= 8 else (9.4 if n <= 12 else 8.6))
        ax.set_xlim(-0.4, n - 0.6)
        ax.set_ylim(0.2, 3000)
        ax.set_title(f"({'abcdefg'[j]}) {name.replace(chr(10), ' ')}", fontsize=14.5, pad=8)
    if fig is None:
        return rows
    flat[7].axis("off")
    sq_legend(fig, flat[0], 0.93, 4)
    save(fig, "eval_circuits.png")
    return rows


def g_scaleout(G, big=(), c="ki", title="Kicked Ising", ax=None):
    """Scale-out of SectorQ: time against the problem size, one line per GPU count (1-16 GPUs: campaign g; more: the
    scaling jobs e15 and the scale campaign). On 32 GPUs and more the runs at cutoff 1e-6 and below are those with
    the sector limit 1,024 where there is one (the ladder of the scale campaign down to the memory limit).
    With `ax` (a panel of another figure, make_deck_figures.py) only the panel is drawn: no legend, no file."""
    mine = [r for r in big if r["kind"] == "engine" and r["circuit"] == BIG_NAME[c]]

    def at(x, g):
        """The run of one point. At cutoff 1e-6 and below: the ladder of the scale campaign (sector limit 4,096 with small
        chunks on 128 GPUs and more, 1,024 on 32 and 64 GPUs); above: the default settings."""
        hits = [r for r in mine if r["gpus"] == g and abs(r["coef"] / float(x) - 1) < 1e-9]
        plain = [r for r in hits if "--gmax" not in r["opt"] and "--chunk" not in r["opt"]]
        wide = [r for r in hits if r["opt"].get("--gmax") == "12"] + [r for r in hits if r["opt"].get("--gmax") == "10" and "--chunk" not in r["opt"]]
        order = (wide + plain) if float(x) <= 1.0000001e-6 else (plain + wide)
        done = [r for r in order if ok(r)]
        return newest(done)[0] if done else (order[0] if order else None)

    cutoffs = greached(G, f"eng.{c}", "g16", [x for x in gladder(G, f"eng.{c}", "g16") if x not in ("15e-7", "12e-7")])[4:]         # from cutoff 1e-4
    lower = {r["opt"]["--coef"] for r in mine if r["coef"] < float(cutoffs[-1]) * 0.999 and (ok(r) or r["fatal"])}
    cutoffs += sorted(lower, key=float, reverse=True)                # problems that only more than 16 GPUs were given
    more = {}
    for r in mine:
        if ok(r) and r.get("boundary"):
            more[r["opt"]["--coef"]] = r["boundary"]
    colors = {1: "#1f77b4", 2: "#d62728", 4: "#FFA500", 8: "#5b4a8a", 16: "#2ca02c", 32: "#8c564b", 64: "#e377c2", 128: "#7f7f7f", 256: "#17becf", 512: "#bcbd22"}
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(10.4, 1.2 * len(cutoffs)), 4.9))           # room for the three-line tick labels
    rows = []
    for g in list(GPUS_G) + sorted(set(BIG_WEAK) | {r["gpus"] for r in mine if r["gpus"] > 16}):
        if g > 16:
            found = [at(x, g) for x in cutoffs]
            recs = [None if r is None else dict(status="ok", wall=r["wall"]) if ok(r) else dict(status="oom" if r["fatal"] else "tl") for r in found]
        else:
            recs = [G.get(f"eng.{c}.{x}.g{g}") for x in cutoffs]
        pts = [(i, s["wall"]) for i, s in enumerate(recs) if gok(s)]
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colors[g], marker="s", markersize=10, linewidth=3.8, label=f"{g} GPU" + ("s" if g > 1 else ""))
        fail = next((i for i in range(pts[-1][0] + 1, len(recs)) if recs[i] is not None), None)
        if STOPS and fail is not None:
            ax.plot([fail], [pts[-1][1]], color=colors[g], marker="x", markersize=11, markeredgewidth=2.6, linestyle="none", zorder=5)
        rows.append((g, [(cutoffs[i], w) for i, w in pts]))
    sq_axes(ax, xl_cut(), "Simulation Time (s)")
    ax.set_xticks(range(len(cutoffs)))
    ax.set_xticklabels(cut_ticks(G, c, cutoffs, more=more))
    ax.set_xlim(-0.4, len(cutoffs) - 0.6)
    ax.set_ylim(0.5, 4000)
    if fig is None:
        return rows
    sq_legend(fig, ax, 1.09 if len(rows) > 5 else 1.03, 5, extra=True, size=13)
    save(fig, "eval_scaleout.png")
    return rows


def g_breakdown(G, g=16):
    """Where the time goes on 16 GPUs (runs bd: SectorQ, bdc: cuPauliProp with the profiling plugin, bdd: the same, deferred)."""
    # the largest setting that cuPauliProp's distributed mode finishes on 16 GPUs, for the two noiseless circuits
    sets = []
    for c, x, name in (("ki", "1e-5", "Kicked Ising"), ("tfim", "1e-5", "TFIM Heavy-Hex")):
        n = gterms(G, c, x)
        sets.append((c, x, f"{name}, Cutoff {x}" + (f" ({count(n).replace(' ', '')} Terms, {data_size(n * term_bytes(c))})" if n else "")))

    def one(kind, tag, c, coef):
        item = f"{tag}.{c}.{coef}.g{g}"
        folder = os.path.dirname((G.get(item) or {}).get("log") or "logs/g")          # logs/r where campaign r measured the run again
        return shares(kind, gdetail(G.get(item)) if gok(G.get(item)) else None, os.path.join(LOGS, os.path.basename(folder), "smi." + item))

    return draw_shares([(title, [("SectorQ", one("eng", "bd", c, coef)), ("cuPauliProp", one("cupp", "bdc", c, coef)),
                                 ("cuPauliProp, Dedup Deferred", one("cupp", "bdd", c, coef))]) for c, coef, title in sets], "eval_breakdown_base.png")


SIZE_PARTS = [("Gate Application (Library Calls)", "operations", OURS), ("Exchange Between Nodes", "exchange", "#C0432F"),
              ("GPU-Host Copies", "copies", "#F5B553"), ("Re-sectoring", "partition", "#7FC8BE"),
              ("Idle GPU", "idle", "#B9BDC4"), ("Waiting for Other Nodes", "waiting", "#8E96A3"), ("Other", "other", "#E3E4E8")]


def g_breakdown_sizes(G, g=16, cutoffs=("1e-4", "3e-5", "1e-5", "3e-6", "1e-6"), axes=None, height=0.66):
    """SectorQ on 16 GPUs: shares of the run time as the problem grows (sweep runs; times summed over the GPU workers
    are divided by the number of workers).
    With `axes` (three panels of another figure, make_deck_figures.py) only the panels are drawn, with bars of `height`."""
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(15.6, 3.1), gridspec_kw={"wspace": 0.34})
    out = []
    for j, (c, title) in enumerate(G3):
        ax = axes[j]
        labels = []
        for coef in cutoffs:
            s = G.get(f"eng.{c}.{coef}.g{g}")
            r = gdetail(s) if gok(s) else None
            if not r or "workers" not in r["parts"]:
                continue
            p, n = r["parts"], r["parts"]["workers"]
            share = {"operations": p.get("operations", 0.0) / n, "copies": (p.get("gpu->host", 0.0) + p.get("host->gpu", 0.0)) / n,
                     "partition": p.get("partition", 0.0) / n, "idle": p.get("idle", 0.0) / n,
                     "waiting": p.get("waiting", 0.0), "exchange": p.get("exchange", 0.0)}
            share = {k: v / r["wall"] for k, v in share.items()}
            share["other"] = max(0.0, 1.0 - sum(share.values()))
            out.append((title, coef, r["wall"], share))
            left, i = 0.0, len(labels)
            for name, key, col in SIZE_PARTS:
                ax.barh(i, share[key] * 100, left=left, color=col, height=height, label=name if (j == 0 and i == 0) else None)
                if share[key] >= 0.07:
                    ax.text(left + share[key] * 50, i, f"{share[key] * 100:.0f}%", ha="center", va="center", fontsize=11,
                            color="white" if key in ("operations", "exchange", "waiting") else INK)
                left += share[key] * 100
            labels.append(f"{coef} ({secs(r['wall'])} s)")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=11.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of the Run Time (%)")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Cutoff (run time)")
    if fig is None:
        return out
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=7, bbox_to_anchor=(0.5, 1.13), fontsize=11.5, columnspacing=1.4, handlelength=1.2)
    save(fig, "eval_breakdown.png")
    return out


def g_accuracy(G):
    """The exact-value check with the values of campaign g where it has them (the answer does not depend on the engine)."""
    for steps in PP:
        for coef in gladder(G, f"exact.tfim30.s{steps}", "g4"):
            s = G[f"exact.tfim30.s{steps}.{coef}.g4"]
            if gok(s) and s.get("expectation") is not None:
                PP[steps][float(coef)] = s["expectation"]
    accuracy()


# ---- circuit size and accuracy (campaign g, lists todo_e and todo_f) ---------------------------------------------------
SIZE_L = (6, 8, 11, 16, 23, 32)                     # 36 ... 1,024 qubits; the library takes no more than 1,024
STEPS_D = (4, 8, 12, 16, 20, 28, 40, 60)
ACC_CUTS = ("1e-3", "3e-4", "1e-4", "3e-5", "1e-5", "3e-6", "1e-6", "3e-7", "1e-7")
EXACT_G = {4: 0.802867136639, 6: 0.790133273060}
WHY_SIZE = {"oom": "out of memory", "tl": "over 20 min", "predicted_tl": "over 20 min", "fail": "not supported"}


def size_line(ax, G, ids, kind, label, ls="-"):
    """One implementation over a ladder of item ids: a line through the finished runs and a cross where it stops."""
    recs = [G.get(i) for i in ids]
    pts = [(k, s["wall"]) for k, s in enumerate(recs) if gok(s)]
    col, mk = SQ[kind]
    if not pts:
        return
    ours = kind == "ours"
    ax.plot([a for a, _ in pts], [b for _, b in pts], color=col, marker=mk, markersize=9 if ours else 8, linewidth=3.4 if ours else 2.6,
            linestyle=ls, label=label, zorder=4 if ours else 3, solid_capstyle="round")
    nxt = pts[-1][0] + 1
    if STOPS and nxt < len(recs) and recs[nxt] is not None and not gok(recs[nxt]):
        ax.plot([nxt], [pts[-1][1]], color=col, marker="x", markersize=10, markeredgewidth=2.6, linestyle="none", zorder=5)


SIZE_LINES = [("ours", "eng", "g16", "-", "SectorQ (16 GPUs)"), ("ours", "eng", "g4", "--", "SectorQ (4 GPUs)"), ("ours", "eng", "g1", ":", "SectorQ (1 GPU)"),
              ("cupp", "cupp", "g16", "-", "cuPauliProp (16 GPUs)"), ("defer", "defer", "g16", "-", "cuPauliProp, dedup deferred (16 GPUs)"),
              ("cupp", "ref", "g1", ":", "cuPauliProp (1 GPU)"), ("spd", "spd", "n1", ":", "SPD (1 GPU)"),
              ("orqa", "orqa", "n4", "-", "ORQA, reproduction (4 nodes)"), ("monoprop", "monoprop", "n4", "-", "MonoProp (4 nodes)"),
              ("julia", "julia", "n1", "-", "PauliPropagation.jl (64 cores)"), ("ppvm", "ppvm", "n1", "-", "ppvm (64 cores)")]


def g_qubits(G, axes=None):
    """Qubit sweep: total magnetization of the LxL TFIM (one observable term per qubit), time against the number of qubits.
    With `axes` (two panels of another figure, make_deck_figures.py) only the panels are drawn; returns the legend entries."""
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(15.6, 5.4), sharey=True)
    for ax, x in zip(axes, ("1e-3", "1e-4")):
        for kind, tag, suffix, ls, label in SIZE_LINES:
            size_line(ax, G, [f"m.{tag}.L{L}.{x}.{suffix}" for L in SIZE_L + (45,)], kind, label, ls)
        sq_axes(ax, "# of Qubits", "Simulation Time (s)" if ax is axes[0] else None)
        ax.set_xticks(range(len(SIZE_L)))
        ax.set_xticklabels([f"{L * L:,}" for L in SIZE_L])
        ax.set_xlim(-0.4, len(SIZE_L) - 0.4)
        ax.set_ylim(0.1, 3000)
        ax.set_title(f"cutoff {x}", fontsize=15, fontweight="bold")
    if fig is not None:
        sq_legend(fig, axes[0], 1.2, 4, extra=False)
    h, n = axes[0].get_legend_handles_labels()
    for leg in (fig.legends if fig is not None else ()):
        leg.remove()
    if STOPS:
        h.append(plt.Line2D([0], [0], color="black", marker="x", markersize=10, markeredgewidth=2.6, linestyle="none"))
        n.append("stops here: out of memory, over 20 min, or qubits not supported")
    if fig is None:
        return h, n
    fig.legend(h, n, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.2), fontsize=12.5, columnspacing=1.4, handlelength=2.4, frameon=False)
    fig.subplots_adjust(wspace=0.05)
    save(fig, "eval_qubits.png")


def g_depth(G, x="3e-5", ax=None):
    """Depth sweep: kicked Ising on 127 qubits, time against the number of Trotter steps.
    With `ax` (a panel of another figure, make_deck_figures.py) only the panel is drawn; returns the legend entries."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(11.6, 5.2))
    lines = [l for l in SIZE_LINES if l[1] not in ("ppvm",)]
    for kind, tag, suffix, ls, label in lines:
        size_line(ax, G, [f"d.{tag}.s{s}.{x}.{suffix}" for s in STEPS_D], kind, label, ls)
    sq_axes(ax, "# of Trotter Steps", "Simulation Time (s)")
    ax.set_xticks(range(len(STEPS_D)))
    ax.set_xticklabels([str(s) for s in STEPS_D])
    ax.set_xlim(-0.4, len(STEPS_D) - 0.6)
    ax.set_ylim(0.01, 2000)
    h, n = ax.get_legend_handles_labels()
    if STOPS:
        h.append(plt.Line2D([0], [0], color="black", marker="x", markersize=10, markeredgewidth=2.6, linestyle="none"))
        n.append("stops here: out of memory")
    if fig is None:
        return h, n
    fig.legend(h, n, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.22), fontsize=12.5, columnspacing=1.4, handlelength=2.4, frameon=False)
    save(fig, "eval_depth.png")


def g_accuracy_time(G, axes=None):
    """Accuracy against the exact value (30-qubit TFIM): time against the cutoff; below each cutoff the error of the
    value that all implementations with the same truncation rule return (MonoProp truncates by its own rule).
    With `axes` (panels of another figure, make_deck_figures.py) only the panels are drawn; returns the legend entries."""
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(15.6, 5.4), sharey=True)
    lines = [("ours", "eng", "g16", "-", "SectorQ (16 GPUs)"), ("ours", "eng", "g4", "--", "SectorQ (4 GPUs)"), ("ours", "eng", "g1", ":", "SectorQ (1 GPU)"),
             ("cupp", "ref", "g1", ":", "cuPauliProp (1 GPU)"), ("spd", "spd", "n1", ":", "SPD (1 GPU)"), ("orqa", "orqa", "n4", "-", "ORQA, reproduction (4 nodes)"),
             ("monoprop", "monoprop", "n4", "-", "MonoProp (4 nodes, own truncation)"), ("julia", "julia", "n1", "-", "PauliPropagation.jl (64 cores)")]
    for ax, steps in zip(axes, (4, 6)):
        for kind, tag, suffix, ls, label in lines:
            size_line(ax, G, [f"acc.{tag}.s{steps}.{x}.{suffix}" for x in ACC_CUTS], kind, label, ls)
        ticks = []
        for x in ACC_CUTS:
            vals = [G[i]["expectation"] for i in (f"acc.eng.s{steps}.{x}.g16", f"acc.eng.s{steps}.{x}.g4", f"acc.eng.s{steps}.{x}.g1") if gok(G.get(i))]
            ticks.append(f"{x}\n({abs(vals[0] - EXACT_G[steps]):.1e})".replace("e-0", "e-") if vals else f"{x}\n")
        sq_axes(ax, "Cutoff (Error against the Exact Value)", "Simulation Time (s)" if ax is axes[0] else None)
        ax.set_xticks(range(len(ACC_CUTS)))
        ax.set_xticklabels(ticks, fontsize=10.5)
        ax.set_xlim(-0.4, len(ACC_CUTS) - 0.6)
        ax.set_ylim(0.01, 2000)
        ax.set_title(f"{steps} Trotter steps", fontsize=15, fontweight="bold")
    h, n = axes[0].get_legend_handles_labels()
    if STOPS:
        h.append(plt.Line2D([0], [0], color="black", marker="x", markersize=10, markeredgewidth=2.6, linestyle="none"))
        n.append("stops here: out of memory or over 20 min")
    if fig is None:
        return h, n
    fig.legend(h, n, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.2), fontsize=12.5, columnspacing=1.4, handlelength=2.4, frameon=False)
    fig.subplots_adjust(wspace=0.05)
    save(fig, "eval_accuracy.png")


VAR_SIZES = ("1e-4", "5e-5", "3e-5", "2e-5", "1e-5")
VAR_AT = "3e-5"
VAR_GPUS = (1, 2, 4, 8, 16)
VAR_REPS = 10


def var_runs(G, kind, x, g):
    """Seconds of the repeated runs of one setting (runs var.*, jobs/g_var_items.py), None where a repeat is missing."""
    return [G[i]["wall"] if gok(G.get(i)) else None for i in (f"var.{kind}.ki.{x}.g{g}.r{r:02d}" for r in range(1, VAR_REPS + 1))]


def var_stat(t):
    """(mean, min, max, (max - min) / mean) of the repeats that finished."""
    v = [x for x in t if x is not None]
    return (sum(v) / len(v), min(v), max(v), (max(v) - min(v)) / (sum(v) / len(v))) if v else None


def g_variability(G):
    """Run-to-run variability, ten executions per setting: heat maps as in the ScaleQsim paper (one cell per run).
    Left: 16 GPUs over the problem size. Right: one problem over the GPU count. Under each column the spread
    (max - min) / mean of its ten runs; OOM where cuPauliProp's distributed mode does not fit."""
    import numpy as np
    from matplotlib.colors import LogNorm
    parts = [("Cutoff", [(x, 16) for x in VAR_SIZES], [x for x in VAR_SIZES]),
             ("# of GPUs", [(VAR_AT, g) for g in VAR_GPUS], [str(g) for g in VAR_GPUS])]
    if not any(v is not None for _, cols, _ in parts for x, g in cols for v in var_runs(G, "eng", x, g)):
        return None
    fig = plt.figure(figsize=(17.6, 5.0))
    gs = fig.add_gridspec(1, 7, width_ratios=[1, 1, 0.055, 0.5, 1, 1, 0.055], wspace=0.07)
    heads = ("(a) 16 GPUs, Problem Size Varied", f"(b) Cutoff {VAR_AT}, GPU Count Varied")
    rows = []
    for k, (xlabel, cols, names) in enumerate(parts):
        data = {}
        for kind, tag in (("ours", "eng"), ("cupp", "cupp")):
            data[kind] = np.array([[np.nan if v is None else v for v in var_runs(G, tag, x, g)] for x, g in cols], dtype=float).T
        seen = np.concatenate([d[np.isfinite(d)] for d in data.values()])
        norm = LogNorm(vmin=seen.min() * 0.9, vmax=seen.max() * 1.1) if seen.size else LogNorm(vmin=1, vmax=10)   # a part not measured yet
        im = None
        for j, kind in enumerate(("ours", "cupp")):
            ax = fig.add_subplot(gs[0, 4 * k + j])
            d = data[kind]
            cmap = plt.get_cmap("YlOrRd").copy()
            cmap.set_bad("#E6E6E6")
            im = ax.imshow(np.ma.masked_invalid(d), aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
            ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
            ax.set_yticks(np.arange(-0.5, VAR_REPS), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.2)
            ax.tick_params(which="both", length=0)
            ticks = []
            for c, ((x, g), name) in enumerate(zip(cols, names)):
                st = var_stat(list(d[:, c][np.isfinite(d[:, c])]))
                oom = (G.get(f"{'eng' if kind == 'ours' else 'cupp'}.ki.{x}.g{g}") or {}).get("status") == "oom"     # the sweep's run of the setting
                ticks.append(f"{name}\n({st[3] * 100:.1f}%)" if st else f"{name}\n(OOM)" if oom else f"{name}\n")
                if st:
                    rows.append((xlabel, SQ_NAME[kind], name, len(d[:, c][np.isfinite(d[:, c])])) + st)
                elif oom:
                    ax.text(c, (VAR_REPS - 1) / 2, "out of memory", ha="center", va="center", rotation=90, fontsize=12, color="#555555")
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels(ticks, fontsize=10.5)
            ax.set_yticks(range(VAR_REPS))
            ax.set_yticklabels([str(r) for r in range(1, VAR_REPS + 1)] if j == 0 else [], fontsize=12.5)
            ax.set_xlabel(f"{xlabel} (Spread)", fontsize=15, fontweight="bold")
            if j == 0:
                ax.set_ylabel("Execution #", fontsize=15, fontweight="bold")
            ax.set_title(SQ_NAME[kind], fontsize=15.5, fontweight="bold", pad=7)
            if j == 0:                                   # one heading over the pair of panels
                ax.text(1.03, 1.14, heads[k], transform=ax.transAxes, ha="center", va="bottom", fontsize=15)
            for sp in ax.spines.values():
                sp.set_visible(False)
        cax = fig.add_subplot(gs[0, 4 * k + 2])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Simulation Time (s)", fontsize=14, fontweight="bold")
        cb.ax.tick_params(labelsize=12)
        cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        cb.ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if f"{v:e}"[0] in "25" else ""))
    save(fig, "eval_variability.png")
    return rows


def main_g():
    G = campaign_g()
    big = big_logs()
    single = g_single_node(G)
    weak = g_weak(G, big)
    strong = g_strong(G)
    circuits = g_circuits(G)
    scaleout = g_scaleout(G, big)
    parts = g_breakdown(G)
    sizes = g_breakdown_sizes(G)
    g_qubits(G)
    g_depth(G)
    g_accuracy_time(G)
    g_variability(G)
    if "--print" in sys.argv:
        f = lambda s: "-" if s is None else (secs(s["wall"]) if gok(s) else s["status"])
        show = lambda pts, stop: "  ".join(f"{x}: {secs(w)}" for x, w in pts) + (f"  | stops at {stop[0]} ({stop[1]})" if stop else "")
        print("# one node, kicked Ising: implementation | cutoff: seconds")
        for label, pts, stop in single:
            print(f"{label:40s} {show(pts, stop)}")
        print("\n# weak scaling: load | circuit | implementation | GPUs | cutoff | seconds")
        for load, title, name, g, x, s in weak:
            if s is not None:
                print(f"{load} {title:20s} {name:28s} {g:3d} {x:7s} {f(s)}")
        print("\n# strong scaling, kicked Ising: GPUs | implementation | cutoff: seconds")
        for title, name, pts, stop in strong:
            print(f"{title:18s} {name:28s} {show(pts, stop)}")
        print("\n# seven circuits, 16 GPUs: circuit | implementation | cutoff: seconds")
        for title, name, pts, stop in circuits:
            print(f"{title:20s} {name:28s} {show(pts, stop)}")
        print("\n# scale-out, kicked Ising: GPUs | cutoff: seconds")
        for g, pts in scaleout:
            print(f"{g:3d}  " + "  ".join(f"{x}: {secs(w)}" for x, w in pts))
        print("\n# where the time goes, 16 GPUs")
        for title, lab, d in parts:
            print(f"{title or '':32s} {lab:28s} {d['wall']:7.2f} s  " + "  ".join(f"{k}={v * 100:.0f}%" for k, v in d["share"].items())
                  + (f"  gpu_busy={d['busy']:.0f}%" if d["busy"] is not None else ""))
        print("\n# SectorQ on 16 GPUs by problem size: circuit | cutoff | s | shares")
        for title, coef, wall, share in sizes:
            print(f"{title:20s} {coef:6s} {wall:8.1f}  " + "  ".join(f"{k}={v * 100:.0f}%" for k, v in share.items()))


def main():
    global OUT
    if "--out" in sys.argv:
        OUT = sys.argv[sys.argv.index("--out") + 1]
        os.makedirs(OUT, exist_ok=True)
    if "--g" in sys.argv:
        return main_g()
    if "--base-only" in sys.argv:
        for title, lab, d in breakdown_base():
            r = d["run"]
            print(f"{title or '':32s} {lab:28s} {d['wall']:7.2f} s  " + "  ".join(f"{k}={v * 100:.0f}%" for k, v in d["share"].items())
                  + (f"  gpu_busy={d['busy']:.0f}%" if d["busy"] is not None else ""))
            for k, v in sorted(r.get("mpi_kinds", {}).items()):
                print(f"{'':62s} mpi {k}: {v['per_op']:.2f} calls per operation, {v['seconds']:.2f} s, {v['gb']:.2f} GB per rank")
            if "phases" in r:
                print(f"{'':62s} phases: " + ", ".join(f"{k} {a:.2f} s ({b:.2f} in MPI)" for k, (a, b) in r["phases"].items()))
        return
    e12, e13 = many("pp_e12_g*-*.out"), many("pp_e13_g*-*.out")
    if "--old" not in sys.argv:
        # the figures of the README: background, then campaign g (1-16 GPUs) with the scaling jobs e15 (32-128 GPUs)
        background()
        return main_g()
    # the evaluation figures before campaign g, from the earlier logs (they overwrite eval_breakdown*.png and eval_accuracy.png)
    run4, run5, e3 = parse("inter_run4.out"), parse("inter_run5.out"), parse("pp_e3-59475516.out")
    run6 = parse("inter_run6.out")
    big = parse(BIG_LOG)        # 16 and 32 GPUs of the same eight nodes: engine and both baselines
    e2a, e2b, run2, run3 = parse("pp_e2a-59475513.out"), parse("pp_e2b-59475514.out"), parse("inter_run2.out"), parse("inter_run3.out")
    allruns = e2a + e2b + e3 + run2 + run3 + run4 + run5 + run6
    table = speed(run5 + run6 + [r for r in big if r["kind"] == "engine"], e3 + run3 + run4 + run5 + run6, big)
    rows, reach = gpus_and_reach(run4, run5, run4 + run5 + run6, big)
    strong, weak = scaling(e13)
    base = scaling_baselines(e13, e12)
    accuracy()
    parts = breakdown(big)
    breakdown_base()
    if "--print" in sys.argv:
        print("\n# speed, 16 GPUs: setting | engine | tuned | default | value")
        for label, coef, g, e, t, d in table:
            f = lambda r: ("-" if r is None else (("%.2f" % r["wall"]) + (" [%s%s]" % (r["log"][:10], " k=" + r["opt"]["--cupp-dedup-every"] if "--cupp-dedup-every" in r["opt"] else "")) if ok(r) else ("OOM" if r["fatal"] else "exit %s" % r["exit"])))
            v = e["value"] if ok(e) else None
            agree = [r["value"] == v for r in (t, d) if ok(r)]
            print(f"{label:22s} {coef:g} {g:2d} GPUs | {f(e)} | {f(t)} | {f(d)} | {v} same={agree}")
        print("\n# GPUs: engine | cuPauliProp")
        for g, e, t, d in rows:
            f2 = lambda r: ("%.2f [%s]" % (r["wall"], r["log"][:10])) if ok(r) else (r and ("OOM" if r["fatal"] else r["exit"]))
            print(g, f2(e), "| tuned", f2(t), "| default", f2(d))
        print("\n# reach: cutoff | engine | cuPauliProp")
        for coef, e, d in reach:
            print(coef, (secs(e["wall"]), e["value"], e.get("boundary")) if ok(e) else e and e["exit"], secs(d["wall"]) if ok(d) else (d and ("OOM" if d["fatal"] else d["exit"])))
        print("\n# strong scaling: circuit | cutoff | GPUs | time | gate time per GPU | exchange | host GiB")
        for title, coef, g, r in strong:
            print(f"{title:20s} {coef:g} {g:4d} {r['wall']:8.1f} {r['parts']['operations'] / g:8.1f} {r['parts'].get('exchange', 0):6.1f}")
        print("\n# weak scaling: circuit | GPUs | cutoff | time | work (1e9 terms) | rate (1e6 terms/s/GPU) | % of 4 GPUs")
        for title, g, r, rate, pct in weak:
            print(f"{title:20s} {g:4d} {r['coef']:g} {r['wall']:8.1f} {r['checks'] / 1e9:8.2f} {rate / 1e6:6.2f} {pct:5.0f}%")
        print("\n# cutoff 1e-5 on 16 GPUs and more: circuit | GPUs | engine | tuned | default")
        for title, g, e, t, d in base:
            f3 = lambda r: ("%.1f" % r["wall"]) if ok(r) else "-"
            print(f"{title:20s} {g:4d} {f3(e)} | {f3(t)} | {f3(d)}")
        print("\n# breakdown")
        for label, wall, share in parts:
            print(label, wall, {k: round(v * 100, 1) for k, v in share.items()})


if __name__ == "__main__":
    main()
