#!/usr/bin/env python3
"""Data charts of the SectorQ presentation deck, in the look of the evaluation figures of the README.

  make_deck_figures.py             writes the charts to figures/deck/
  make_deck_figures.py n13 d_      only the charts whose names start with these
  --out DIR                        writes the charts to DIR instead (drafts)

  n02, n11, n13, n14               measured values, entered by hand below
  n16, d_*                         read from the records of the 1-16 GPU campaign (make_eval_figures.campaign_g);
                                   the d_* charts are the evaluation figures of the README in a layout for a slide:
                                   the panels are drawn by the functions of make_eval_figures.py, this file places them

The look is the one of make_eval_figures.py (sq_axes, the colors and markers of SQ): lines with a fixed color and
marker per implementation, a black box around the panel, a dashed gray grid, bold axis labels, the legend in a strip
above the panels. Every chart has the size of its slide box (10.6 in wide, 200 dpi); the panels are moved until
they fill it. Each chart is checked while it is written: a text that leaves the picture, two texts on top of each
other, or a printed value that lies on a line is reported on stderr as "CHECK <chart>: ...".
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
ARGS, sys.argv = sys.argv[1:], sys.argv[:1]         # make_eval_figures reads sys.argv in its main: it does not see ours
sys.path.insert(0, HERE)
import make_eval_figures as mf                      # noqa: E402  (also sets the fonts of the evaluation figures)

OUT = os.path.join(HERE, "deck")
if "--out" in ARGS:
    OUT = ARGS.pop(ARGS.index("--out") + 1)
    ARGS.remove("--out")
mf.OUT = OUT                                        # nothing of this file is written next to the figures of the README

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 200, "figure.facecolor": "white", "savefig.facecolor": "white"})

W, H = 10.6, 5.6                                    # inches: the slide box is 1060 x 560
BLUE = mf.SQ["ours"][0]
LINK = "#8C8C8C"                                    # a line that only connects points of different implementations
HALO = [patheffects.withStroke(linewidth=3.4, foreground="white")]
CROSS = dict(color="black", marker="x", markersize=10, markeredgewidth=2.6, linestyle="none")
mf.STOPS = False         # no crosses in the deck figures: a line ends at the last run that finished
mf.SIZES = False         # under a cutoff the number of terms only (the size in bytes reads as the memory the run needs, which it is not)
SIZES = {}
_G = []


def campaign():
    if not _G:
        _G.append(mf.campaign_g())
    return _G[0]


def scaling_jobs():
    """The runs on 32-256 GPUs (jobs e15), as the evaluation figures read them."""
    if len(_G) < 2:
        campaign()
        _G.append(mf.big_logs())
    return _G[1]


# ---------------------------------------------------------------------------------------
# layout: legend strip, panels that fill the picture, tick labels that do not meet
# ---------------------------------------------------------------------------------------
def strip(fig, rows, size=13.0, ncol=None, pad=0.07, side=0.18, handle=2.4, apart=1.5):
    """The legend strip at the top of the figure: one centered legend per item of `rows` (a list of (handle, label)).
    An item is one line of entries unless `ncol` gives its number of columns. The text is made smaller until the
    widest item fits the figure with `side` inches left at both ends. Returns the lower edge of the strip as a
    fraction of the figure height."""
    w, h = fig.get_size_inches()
    rows = [row for row in rows if row]                 # a line of the strip may be empty when the crosses are not drawn
    while True:
        fig.draw_without_rendering()
        r = fig.canvas.get_renderer()
        made, y = [], 1 - pad / h
        for row in rows:
            leg = fig.legend([a for a, _ in row], [b for _, b in row], loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol or len(row),
                             fontsize=size, columnspacing=apart, handlelength=handle, handletextpad=0.6, labelspacing=0.42,
                             borderpad=0, borderaxespad=0, frameon=False)
            box = leg.get_window_extent(r)
            made.append((leg, box))
            y = box.y0 / fig.dpi / h - 0.42 * size / 72 / h
        if max(b.width for _, b in made) <= (w - 2 * side) * fig.dpi or size <= 8:
            fig.strip_size = size
            return y
        for leg, _ in made:
            leg.remove()
        size -= 0.25


def cell_legend(ax, entries, size=13.0):
    """The legend in a free cell of the grid of panels."""
    ax.axis("off")
    ax.legend([a for a, _ in entries], [b for _, b in entries], loc="center", fontsize=size, handlelength=2.4, handletextpad=0.7,
              labelspacing=0.95, borderpad=0, borderaxespad=0, frameon=False)


def entries(fig, ax):
    """The legend entries of a panel as sq_legend of the evaluation figures lists them: its lines, then the cross."""
    h, n = ax.get_legend_handles_labels()
    mf.sq_legend(fig, ax, 1.0, 1)
    leg = fig.legends[-1]
    note = leg.get_texts()[-1].get_text()
    leg.remove()
    return list(zip(h, n)) + ([(Line2D([0], [0], **CROSS), note)] if mf.STOPS else [])


def tight(fig, r, hide=()):
    """(place in the grid, box with everything drawn around it) of every panel; the artists `hide` are left out."""
    for a in hide:
        a.set_visible(False)
    box = [(a.get_subplotspec(), a.get_tightbbox(r)) for a in fig.axes]
    for a in hide:
        a.set_visible(True)
    return [(s, b) for s, b in box if b is not None and b.width > 0 and b.height > 0]


def fit(fig, top=None, pad=0.07, hgap=0.13, wgap=0.13, free=()):
    """Moves the panels until they fill the figure: with tick labels, axis labels and titles they reach `pad` inches
    from the edges and `top` (the lower edge of the legend strip; without one the upper edge); what is drawn around
    two neighboring rows (columns) of panels is `hgap` (`wgap`) inches apart. The artists `free` (a label that is
    longer than its panel and has no neighbor to meet) do not count for these gaps."""
    w, h = fig.get_size_inches()
    top = 1 - pad / h if top is None else top
    nr, nc = fig.axes[0].get_gridspec().get_geometry()
    for _ in range(10):
        fig.draw_without_rendering()
        r = fig.canvas.get_renderer()
        box = tight(fig, r)
        sp, d = fig.subplotpars, fig.dpi
        new = dict(left=sp.left + pad / w - min(b.x0 for _, b in box) / d / w, right=sp.right + 1 - pad / w - max(b.x1 for _, b in box) / d / w,
                   bottom=sp.bottom + pad / h - min(b.y0 for _, b in box) / d / h, top=sp.top + top - max(b.y1 for _, b in box) / d / h)
        box = tight(fig, r, free) if free else box
        if nr > 1:
            lo = {i: min(b.y0 for s, b in box if s.rowspan.start == i) for i in range(nr)}
            hi = {i: max(b.y1 for s, b in box if s.rowspan.start == i) for i in range(nr)}
            gap = min(lo[i] - hi[i + 1] for i in range(nr - 1)) / d
            between = sp.hspace * (sp.top - sp.bottom) * h / (nr + sp.hspace * (nr - 1)) - gap + hgap       # inches between two rows of axes
            new["hspace"] = between * nr / ((new["top"] - new["bottom"]) * h - between * (nr - 1))
        if nc > 1:
            lo = {i: min(b.x0 for s, b in box if s.colspan.start == i) for i in range(nc)}
            hi = {i: max(b.x1 for s, b in box if s.colspan.start == i) for i in range(nc)}
            gap = min(lo[i + 1] - hi[i] for i in range(nc - 1)) / d
            between = sp.wspace * (sp.right - sp.left) * w / (nc + sp.wspace * (nc - 1)) - gap + wgap
            new["wspace"] = between * nc / ((new["right"] - new["left"]) * w - between * (nc - 1))
        if new["right"] - new["left"] < 0.2 or new["top"] - new["bottom"] < 0.2 or min(new.get("hspace", 0), new.get("wspace", 0)) < 0:
            break                                       # what is drawn around the panels does not fit: the check reports it
        moved = max(abs(v - getattr(sp, k)) for k, v in new.items())
        fig.subplots_adjust(**new)
        if moved < 2e-4:
            break


def xlabels(ax, minor=False):
    lo, hi = sorted(ax.get_xlim())
    ticks = ax.xaxis.get_minor_ticks() if minor else ax.xaxis.get_major_ticks()
    return [t.label1 for t in ticks if t.label1.get_visible() and t.label1.get_text().strip() and lo <= t.get_loc() <= hi]


def shrink_ticks(fig, gap=0.4):
    """Makes the x tick labels of a panel smaller until neighbors on the same line are `gap` em apart."""
    fig.draw_without_rendering()
    r = fig.canvas.get_renderer()
    changed = False
    for ax in fig.axes:
        groups = [g for g in (xlabels(ax), xlabels(ax, minor=True)) if g] if ax.axison else []
        if not groups:
            continue
        now = size = groups[0][0].get_fontsize()
        for labels in groups:
            boxes = sorted((t.get_window_extent(r) for t in labels), key=lambda b: b.x0)
            for a, b in zip(boxes, boxes[1:]):
                apart = (b.x0 + b.x1) / 2 - (a.x0 + a.x1) / 2
                size = min(size, apart / ((a.width + b.width) / 2 / now + gap * fig.dpi / 72))
        size = math.floor(size * 4) / 4
        if size < now:
            for t in ax.xaxis.get_ticklabels(which="both"):
                t.set_fontsize(size)
            changed = True
    return changed


def stagger(ax, size, lines=2):
    """Every second x tick label moves `lines` text lines down, with a thin line from its tick to it, so that the
    labels of a panel with many ticks keep their size."""
    ticks = list(ax.get_xticks())
    labels = [t.get_text() for t in ax.get_xticklabels()]
    ax.set_xticks(ticks[0::2])
    ax.set_xticklabels(labels[0::2], fontsize=size)
    ax.set_xticks(ticks[1::2], minor=True)
    ax.set_xticklabels(labels[1::2], minor=True, fontsize=size)
    ax.tick_params(axis="x", which="minor", length=lines * size * 1.36, width=0.6, color="#B8B8B8", pad=2.5, labelsize=size)
    ax.grid(True, which="minor", axis="x", color="#B8B8B8", linewidth=0.6, linestyle="--")


def one_row(ax, size):
    """The x tick labels of a panel with many ticks stay on one row: every tick has the first line of its label (the
    GPU count) and every second tick the second line (the cutoff) below it. The second lines are the labels of minor
    ticks at the same places, so that shrink_ticks compares them with each other only."""
    ticks = list(ax.get_xticks())
    labels = [t.get_text().split("\n") for t in ax.get_xticklabels()]
    ax.set_xticks(ticks)
    ax.set_xticklabels([l[0] for l in labels], fontsize=size)
    ax.xaxis.remove_overlapping_locs = False                    # the minor ticks are at the places of major ticks
    ax.set_xticks(ticks[0::2], minor=True)
    ax.set_xticklabels([l[1] for l in labels[0::2]], minor=True, fontsize=size)
    major = ax.xaxis.get_major_ticks()[0]
    ax.tick_params(axis="x", which="minor", length=0, labelsize=size,
                   pad=major.get_pad() + major.tick1line.get_markersize() + 1.22 * size)        # one text line below the GPU counts


def value(ax, x, y, s, dx=0, dy=9, ha="center", size=13):
    """A measured value printed at its point (offsets in points); the check looks at every one of them."""
    t = ax.annotate(s, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="bottom" if dy >= 0 else "top",
                    fontsize=size, color=mf.INK, path_effects=HALO, zorder=6)
    t.set_gid("value")
    return t


def copy_label(src, dst, axis="y"):
    lab = getattr(src, axis + "axis").label
    getattr(dst, f"set_{axis}label")(lab.get_text(), fontsize=lab.get_fontsize(), fontweight=lab.get_fontweight())


# ---------------------------------------------------------------------------------------
# checks: text inside the picture, no two texts on top of each other, no printed value on a line
# ---------------------------------------------------------------------------------------
def shown_texts(fig):
    out = list(fig.texts)
    legs = list(fig.legends)
    for ax in fig.axes:
        out += list(ax.texts)
        if ax.get_legend() is not None:
            legs.append(ax.get_legend())
        if not ax.axison:
            continue
        out += [ax.title, ax.xaxis.label, ax.yaxis.label] + xlabels(ax) + xlabels(ax, minor=True)
        lo, hi = sorted(ax.get_ylim())
        out += [t.label1 for t in ax.yaxis.get_major_ticks() if t.label1.get_visible() and lo <= t.get_loc() <= hi]
    for leg in legs:
        out += leg.get_texts()
    return [t for t in out if t.get_visible() and t.get_text().strip()], legs


def check(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    cw, ch = fig.canvas.get_width_height()
    texts, legs = shown_texts(fig)
    ext = [(t, t.get_window_extent(r)) for t in texts]
    said = lambda t: t.get_text().replace("\n", " / ")
    bad = []
    for t, b in ext + [(None, leg.get_window_extent(r)) for leg in legs]:
        if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > cw + 0.5 or b.y1 > ch + 0.5:
            bad.append(f"leaves the picture: {said(t) if t is not None else 'a legend'!r}")
    for i, (t, a) in enumerate(ext):
        for u, b in ext[i + 1:]:
            if min(a.x1, b.x1) - max(a.x0, b.x0) > 1 and min(a.y1, b.y1) - max(a.y0, b.y0) > 1:
                bad.append(f"texts overlap: {said(t)!r} and {said(u)!r}")
    for leg in legs:                                    # a legend does not reach into a panel or into a text of a panel
        a, own = leg.get_window_extent(r), set(leg.get_texts())
        others = [(said(t), b) for t, b in ext if t not in own] + [("a panel", x.get_window_extent(r)) for x in fig.axes if x.axison]
        for what, b in others:
            if min(a.x1, b.x1) - max(a.x0, b.x0) > 1 and min(a.y1, b.y1) - max(a.y0, b.y0) > 1:
                bad.append(f"a legend overlaps {what!r}")
    vals = [(t, b) for t, b in ext if t.get_gid() == "value"]
    if vals:                                            # the picture without the printed values: what lies under them?
        for t, _ in vals:
            t.set_visible(False)
        fig.canvas.draw()
        dark = np.asarray(fig.canvas.buffer_rgba())[..., :3].min(axis=2) < 150      # lines, markers, frame, text; not the grid
        for t, b in vals:
            t.set_visible(True)
            under = dark[max(0, ch - math.ceil(b.y1)):ch - int(b.y0), max(0, int(b.x0)):math.ceil(b.x1)]
            if under.size and under.mean() > 0.004:
                bad.append(f"value {said(t)!r} lies on a line or marker ({under.mean() * 100:.1f}% of its box)")
    for line in bad:
        print(f"CHECK {name}: {line}", file=sys.stderr)
    return len(bad)


def finish(fig, name, top=None, place=True, **kw):
    """Fills the picture, lets no two tick labels meet, checks, writes. `place=False`: the figure places its own
    panels (constrained layout)."""
    if place:
        fit(fig, top, **kw)
    if shrink_ticks(fig) and place:
        fit(fig, top, **kw)
    n = check(fig, name)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200, facecolor="white")
    cw, ch = fig.canvas.get_width_height()
    ticks = sorted({t.get_fontsize() for ax in fig.axes if ax.axison for t in xlabels(ax) + xlabels(ax, minor=True)})
    SIZES[name] = (cw, ch, n, getattr(fig, "strip_size", None), ticks)
    plt.close(fig)


def short_count(v, _=None):
    """1, 100, 10K, 1M, 100M, 10B: the numbers of terms as the tick labels of the evaluation figures write them."""
    return mf.count(v).replace(" ", "") if v >= 1e3 else f"{v:g}"


# ---------------------------------------------------------------------------------------
# n02  more kept terms give a more accurate answer
#   (a) kicked Ising, 127 qubits, 20 steps: largest number of terms of the run and the expectation value at each cutoff
#   (b) 5x6 TFIM, 30 qubits, 4 steps: |value - exact value|; the same for every implementation with this truncation rule
# ---------------------------------------------------------------------------------------
def n02():
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, H), gridspec_kw={"width_ratios": [1, 1.34]})
    col, mk = mf.SQ["ours"]
    cut = ["1e-4", "1e-5", "3e-6", "1e-6"]
    terms = ["22.5M", "1.4B", "12.6B", "100.1B"]
    val = [0.153, 0.229, 0.289, 0.322]
    a.plot(range(4), val, color=col, marker=mk, markersize=10, linewidth=3.6, solid_capstyle="round", zorder=4)
    for i, v in enumerate(val):
        value(a, i, v, f"{v:.3f}", dx=-8, dy=5, ha="right")
    mf.sq_axes(a, "Cutoff (Largest # of Terms)", "Expectation Value", log=False, size=15)
    a.tick_params(labelsize=13)
    a.set_xticks(range(4))
    a.set_xticklabels([f"{c}\n({t})" for c, t in zip(cut, terms)])
    a.set_xlim(-0.85, 3.4)
    a.set_ylim(0, 0.4)
    a.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    a.set_title("(a) 127-Qubit Kicked Ising", fontsize=14.5, pad=8)

    cut = ["1e-3", "3e-4", "1e-4", "3e-5", "1e-5", "3e-6", "1e-6", "3e-7", "1e-7"]
    err = [1.9e-2, 5.9e-3, 1.3e-2, 1.0e-2, 3.8e-3, 4.8e-4, 1.3e-3, 3.5e-4, 7.2e-5]
    b.plot(range(9), err, color=col, marker=mk, markersize=10, linewidth=3.6, solid_capstyle="round", zorder=4)
    value(b, 0, err[0], "0.019", dx=8, dy=5, ha="left")
    value(b, 8, err[8], "0.000072", dx=-9, dy=-4, ha="right")
    mf.sq_axes(b, "Cutoff", "Error Against the Exact Value", size=15)
    b.tick_params(labelsize=13)
    b.set_xticks(range(9))
    b.set_xticklabels(cut)
    b.set_xlim(-0.5, 8.5)
    b.set_ylim(3e-5, 5e-2)
    b.set_title("(b) 30-Qubit TFIM, 4 Steps (Exact Value Known)", fontsize=14.5, pad=8)
    fig.align_xlabels([a, b])
    finish(fig, "n02_accuracy_terms", wgap=0.2)


# ---------------------------------------------------------------------------------------
# n11  windows end where the operator is smallest (kicked Ising, cutoff 1e-5, 16 GPUs)
# ---------------------------------------------------------------------------------------
def n11():
    terms = [1, 2, 10, 48, 2582, 2146372, 35930513, 108287276, 151203327, 158924060, 146839005, 133005210,
             120522709, 110316483, 95647971, 81236389, 90049862, 103334041, 92371508, 77582584]
    sectors = [1, 1, 1, 1, 1, 4, 16, 64, 128, 64, 64, 32, 32, 32, 32, 32, 32, 32, 32, 32]
    peak = 1407075677
    fig, ax = plt.subplots(figsize=(W, H))
    col, mk = mf.SQ["ours"]
    ax.plot(range(20), terms, color=col, marker=mk, markersize=9.5, linewidth=3.6, solid_capstyle="round", zorder=4)
    ax.axhline(peak, color="black", linewidth=1.6, linestyle=(0, (5, 3)), zorder=3)
    ax.annotate(f"Largest Count Inside a Window: {peak / 1e9:.2f}B", (-0.45, peak), xytext=(0, 6), textcoords="offset points",
                ha="left", va="bottom", fontsize=13.5, color=mf.INK, path_effects=HALO, zorder=6).set_gid("value")
    mf.sq_axes(ax, "Window (# of Sectors)", "Terms at the Start of the Window")
    ax.yaxis.set_major_locator(FixedLocator([1, 1e2, 1e4, 1e6, 1e8, 1e10]))
    ax.yaxis.set_minor_locator(FixedLocator([1e1, 1e3, 1e5, 1e7, 1e9]))
    ax.yaxis.set_major_formatter(FuncFormatter(short_count))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(range(20))
    ax.set_xticklabels([f"{k}\n({n})" for k, n in enumerate(sectors)])
    ax.set_xlim(-0.6, 19.6)
    ax.set_ylim(0.5, 3e10)
    finish(fig, "n11_window_terms")


# ---------------------------------------------------------------------------------------
# n13  run time changes little with the tuning values (kicked Ising, cutoff 1e-5); the values as reported
# ---------------------------------------------------------------------------------------
def n13():
    style = {16: ("-", "o"), 4: ("--", "s"), 1: (":", "^")}
    panels = [("Target Sector Size", ["1.5M", "3M", "6M\n(Ours)", "12M"],
               {16: ["14.7", "12.9", "12.2", "11.7"], 4: ["51.1", "47.0", "46.7", "44.6"]}),
              ("Sectors per GPU", ["1", "2\n(Ours)", "4", "8"],
               {16: ["12.3", "12.2", "12.6", "13.6"], 4: ["44.5", "46.7", "46.8", "46.7"]}),
              ("GPU Memory Limit (GB)", ["36\n(Ours)", "18", "8", "4"],
               {1: ["168", "177", "219", "274"], 4: ["46.7", "45.7", "55.6", "71.1"]})]
    fig, axes = plt.subplots(1, 3, figsize=(W, H), sharey=True)
    for ax, (xlabel, ticks, series) in zip(axes, panels):
        for g, shown in series.items():
            ls, mk = style[g]
            ys = [float(v) for v in shown]
            ax.plot(range(4), ys, color=BLUE, marker=mk, markersize=10, linewidth=3.4, linestyle=ls, zorder=4)
            for i, y in enumerate(ys):
                value(ax, i, y, shown[i], dy=9)
        mf.sq_axes(ax, xlabel, "Simulation Time (s)" if ax is axes[0] else None, size=15)
        ax.tick_params(labelsize=13)
        ax.set_xticks(range(4))
        ax.set_xticklabels([t if "\n" in t else t + "\n" for t in ticks])         # two lines everywhere: the upper lines stand at one height
        ax.set_xlim(-0.45, 3.45)
    axes[0].set_ylim(8, 450)
    axes[0].yaxis.set_major_locator(FixedLocator([10, 20, 50, 100, 200, 400]))
    line = lambda g: Line2D([0], [0], color=BLUE, marker=style[g][1], markersize=10, linewidth=3.4, linestyle=style[g][0])
    top = strip(fig, [[(line(16), "SectorQ, 16 GPUs"), (line(4), "SectorQ, 4 GPUs"), (line(1), "SectorQ, 1 GPU")]], size=13.5)
    finish(fig, "n13_knobs", top - 0.1 / H, wgap=0.16)


# ---------------------------------------------------------------------------------------
# n14  which element gives which gain (kicked Ising, 16 GPUs): one more element at every step to the right
# ---------------------------------------------------------------------------------------
def n14():
    steps = ["cuPauliProp,\nOriginal Circuit", "+ Clifford Gates Not\nApplied to the Data", "+ Dedup Only\nBefore Truncation",
             "+ Conserved-Parity\nSectors (SectorQ)"]
    kinds = ["cupp", "cupp", "defer", "ours"]
    series = [("Cutoff 1e-5 (1.4B Terms)", "-", ["95.0", "54.3", "48.8", "12.2"], +1),
              ("Cutoff 1e-4 (22.5M Terms)", "--", ["53.0", "28.3", "6.5", "0.81"], -1)]
    fig, ax = plt.subplots(figsize=(W, H))
    for label, ls, shown, side in series:
        ys = [float(v) for v in shown]
        ax.plot(range(4), ys, color=LINK, linewidth=3.4, linestyle=ls, zorder=3)
        for i, (y, kind) in enumerate(zip(ys, kinds)):
            col, mk = mf.SQ[kind]
            ax.plot([i], [y], color=col, marker=mk, markersize=13, linestyle="none", zorder=4)
            value(ax, i, y, shown[i], dx=7 * side, dy=7 * side, ha="left" if side > 0 else "right", size=13.5)   # the lines fall to the right
    mf.sq_axes(ax, None, "Simulation Time (s)")
    ax.set_xticks(range(4))
    ax.set_xticklabels(steps, fontsize=13.5)
    ax.set_xlim(-0.42, 3.42)
    ax.set_ylim(0.4, 200)
    cuts = [(Line2D([0], [0], color=LINK, linewidth=3.4, linestyle=ls), label) for label, ls, _, _ in reversed(series)]
    kinds3 = [(Line2D([0], [0], color=mf.SQ[k][0], marker=mf.SQ[k][1], markersize=12, linestyle="none"), mf.SQ_NAME[k]) for k in ("cupp", "defer", "ours")]
    top = strip(fig, [cuts, kinds3], size=13.5)
    finish(fig, "n14_staircase", top - 0.1 / H)


# ---------------------------------------------------------------------------------------
# n16  small problems gain little from more GPUs (kicked Ising, the sweep runs of the campaign)
# ---------------------------------------------------------------------------------------
def n16():
    G = campaign()
    style = {16: ("-", "o"), 8: ("-.", "D"), 4: ("--", "s"), 2: (":", "v")}
    ladder = [x for x in mf.gladder(G, "eng.ki", "g16") if x not in ("15e-7", "12e-7")]          # the cutoffs of g_scaleout, from 1e-3
    time = lambda x, g: G[f"eng.ki.{x}.g{g}"]["wall"] if mf.gok(G.get(f"eng.ki.{x}.g{g}")) else None
    gain = {g: [time(x, 1) / time(x, g) if time(x, 1) and time(x, g) else None for x in ladder] for g in style}
    cutoffs = [x for i, x in enumerate(ladder) if any(gain[g][i] for g in gain)]
    fig, ax = plt.subplots(figsize=(W, H))
    for y in (2, 4, 8, 16):
        ax.axhline(y, color="black", linewidth=1.1, linestyle=(0, (5, 3)), zorder=2)
    for g in sorted(style):
        ls, mk = style[g]
        pts = [(i, gain[g][ladder.index(x)]) for i, x in enumerate(cutoffs) if gain[g][ladder.index(x)]]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=BLUE, marker=mk, markersize=9.5, linewidth=3.4, linestyle=ls,
                label=f"{g} GPUs", zorder=3 + g / 16, solid_capstyle="round")
        if "--print" in ARGS:
            print(f"n16 {g:2d} GPUs  " + "  ".join(f"{cutoffs[i]}: {v:.2f}" for i, v in pts))
    mf.sq_axes(ax, mf.xl_cut(), "Speedup Over 1 GPU", log=False)
    ax.set_xticks(range(len(cutoffs)))
    ax.set_xticklabels(mf.cut_ticks(G, "ki", cutoffs))
    ax.set_xlim(-0.4, len(cutoffs) - 0.6)
    ax.set_ylim(0, 17)
    ax.set_yticks(range(0, 17, 2))
    top = strip(fig, [list(zip(*ax.get_legend_handles_labels()))], size=13.5)
    finish(fig, "n16_gain_by_size", top - 0.1 / H)


# ---------------------------------------------------------------------------------------
# d_*  the evaluation figures of the README, placed for a slide
# ---------------------------------------------------------------------------------------
def d_single_node():
    fig, ax = plt.subplots(figsize=(W, H))
    mf.g_single_node(campaign(), ax=ax)
    found = dict((n, h) for h, n in entries(fig, ax))
    # three columns that read row by row: the 4-GPU runs, the 1-GPU runs, then the implementations on the host
    # (13 entries in three columns of 5, 4 and 4: the GPU implementations, cuPauliProp's modes and MonoProp, the CPU ones)
    grid = [["SectorQ (4 GPUs)", "SectorQ (1 GPU)", "paulistrings-rs (4 GPUs)", "paulistrings-rs (1 GPU)", "SPD (1 GPU)"],
            ["cuPauliProp (4 GPUs)", "cuPauliProp (1 GPU)", "cuPauliProp, deferred dedup (4 GPUs)", "MonoProp (64 cores)"],
            ["ORQA reproduction (64 cores)", "PauliPropagation.jl (64 cores)", "ppvm (64 cores)", "Qiskit pauli-prop (1 core)"]]
    order = [n for column in grid for n in column if n in found]
    order += [n for n in found if n not in order]                    # the cross, and any line this list does not know
    top = strip(fig, [[(found[n], n) for n in order]], size=13, ncol=3)
    finish(fig, "d_single_node", top - 0.1 / H)


def d_strong():
    fig, axes = plt.subplots(2, 2, figsize=(W, H))
    panels = [axes[0][0], axes[0][1], axes[1][0]]
    mf.g_strong(campaign(), axes=panels)
    copy_label(axes[0][0], axes[1][0])
    for ax in panels:
        ax.xaxis.label.set_fontsize(14.5)
        ax.yaxis.label.set_fontsize(14.5)
    cell_legend(axes[1][1], entries(fig, axes[0][0]), size=13)
    finish(fig, "d_strong", wgap=0.18)


def three_lines(e):
    """Legend entries in three lines of about one length, in the order of the README's legend but for cuPauliProp (1 GPU)."""
    upper = [x for x in e if x[1].startswith(("SectorQ", "cuPauliProp ("))]
    lower = [x for x in e if x[1].startswith(("PauliPropagation", "ppvm", "stops here"))]
    middle = [x for x in e if x not in upper and x not in lower]
    return [row for row in (upper, middle, lower) if row]


def d_qubits():
    fig, axes = plt.subplots(1, 2, figsize=(W, H), sharey=True)
    h, n = mf.g_qubits(campaign(), axes=axes)
    top = strip(fig, three_lines(list(zip(h, n))), size=12.5, side=0.1, handle=2.1, apart=1.2)
    finish(fig, "d_qubits", top - 0.08 / H, wgap=0.16)


def d_breakdown():
    fig, axes = plt.subplots(3, 1, figsize=(W, H), sharex=True)
    mf.g_breakdown_sizes(campaign(), axes=axes, height=0.76)
    axes[0].get_gridspec().set_height_ratios([max(1, len(ax.get_yticks())) for ax in axes])       # bars of one thickness in all rows
    copy_label(axes[0], axes[1])
    axes[0].set_ylabel("")
    for ax in axes:
        ax.set_xlabel("" if ax is not axes[-1] else ax.get_xlabel())
        ax.tick_params(labelsize=12.5)
        ax.title.set_fontsize(14.5)
        for t in ax.texts:
            t.set_fontsize(12)
    h, n = axes[0].get_legend_handles_labels()
    e = list(zip(h, n))
    top = strip(fig, [e[:4], e[4:]], size=12.5)
    for leg in fig.legends:
        for p in leg.legend_handles:
            p.set_height(9)
    finish(fig, "d_breakdown", top - 0.08 / H, hgap=0.12, free=[axes[1].yaxis.label])


def slide_ladder(G, c, cutoffs, g):
    """The cutoffs of a panel on a slide, few enough for one line of tick labels with the number of terms and their
    size: from 1e-4 down, the steps 1e-k and 3e-k, the last cutoff each implementation finished, and the last step
    of the ladder (where SectorQ stops)."""
    last, done_any = set(), set()
    for kind in ("ours", "cupp", "defer"):
        done = [x for x, rec in zip(cutoffs, mf.sq_runs(G, kind, c, cutoffs, f"g{g}")) if mf.gok(rec)]
        last.update(done[-1:])
        done_any.update(done)
    return [x for x in cutoffs if float(x) <= 1.0000001e-4 and (x[0] in "13" or x in last or x == cutoffs[-1]) and (mf.STOPS or x in done_any)]


def d_circuits(name, which, legend_cell, scale=1.0):
    """`scale` > 1: a larger picture of the same shape (smaller text on the slide)."""
    fig, axes = plt.subplots(2, 2, figsize=(W * scale, H * scale))
    flat = [a for row in axes for a in row]
    panels = flat[:len(which)]
    more = {}                                           # terms of the problems only more than 16 GPUs finished
    for r in scaling_jobs():
        if r["kind"] == "engine" and mf.ok(r) and r.get("boundary"):
            more.setdefault({v: k for k, v in mf.BIG_NAME.items()}.get(r["circuit"]), {})[r["opt"]["--coef"]] = r["boundary"]
    rows = mf.g_circuits(campaign(), axes=panels, which=which, keep=slide_ladder, sizes=more)
    for j, ax in enumerate(panels):
        if len(ax.get_xticks()) > 9:
            stagger(ax, 10.5)
        else:
            for t in ax.get_xticklabels():
                t.set_fontsize(11.5)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_yticks([1, 10, 100, 1000])
        # how many times as long cuPauliProp's default mode takes as SectorQ, printed above its point
        ticks = [t.get_text().split("\n")[0] for t in ax.get_xticklabels()]
        ours, cupp = dict(rows[3 * j][2]), dict(rows[3 * j + 1][2])
        for x in ticks:
            if x in ours and x in cupp:
                value(ax, ticks.index(x), cupp[x], mf.times(cupp[x], ours[x]), dy=8, size=12.5)
    note = (Line2D([0], [0], color="none"), "N×: cuPauliProp time / SectorQ time")
    if legend_cell:
        cell_legend(flat[3], entries(fig, panels[0]) + [note], size=13)
        top = None
    else:
        for ax in axes[0]:                                  # four panels: the x label once per column
            ax.set_xlabel("")
        e = entries(fig, panels[0])
        top = strip(fig, [e[:3] + [note], e[3:]], size=12.5) - 0.08 / (H * scale)
    fig.align_xlabels(axes[1])
    finish(fig, name, top, wgap=0.18)


def d_circuits_a():
    d_circuits("d_circuits_a", ("ki", "tfim", "noise", "tfimsq"), False)


def d_circuits_b():
    d_circuits("d_circuits_b", ("xxz", "qaoa", "hea"), True)


def peaks_of(c, cutoffs):
    """{cutoff: the largest number of terms inside a window} of one circuit, from any SectorQ run of the problem (the
    number belongs to the problem: runs on different GPU counts report the same one)."""
    G, out = campaign(), {}
    for x in cutoffs:
        for n in reversed(mf.GPUS_G):
            run = G.get(f"eng.{c}.{x}.g{n}")
            det = mf.gdetail(run) if mf.gok(run) else None
            if det and det.get("inside"):
                out[x] = det["inside"]
                break
    for r in scaling_jobs():                                # problems that only more than 16 GPUs finished
        if r["kind"] == "engine" and r["circuit"] == mf.BIG_NAME.get(c) and mf.ok(r) and r.get("inside"):
            out.setdefault(r["opt"]["--coef"], r["inside"])
    return out


GPU_TERMS = 95e6      # 40-byte terms one 36-GB GPU holds when the whole list is on the GPUs: cuPauliProp reports "insufficient
                      # workspace" with 92-110 million local terms (default mode finishes up to 89 million, fails from 98 million)
GPU_USED = 36e9       # bytes of a 40-GB GPU that the runs use (--vram-frac 0.9)
NEED = GPU_USED / (GPU_TERMS * 40)       # GPU memory per byte of terms: the terms, the output buffer and the scratch space (9.5)
BEYOND = "#E6E9ED"


def size_ticks(ax, c, peaks, g):
    """Shades the part of a panel where the whole list of terms no longer fits on `g` GPUs: the cutoffs at which the
    largest number of terms of the run is more than the GPUs hold at once (GPU_TERMS each, for 40-byte terms). The
    tick labels stay the cutoff and the number of terms."""
    cuts = [t.get_text().split("\n")[0] for t in ax.get_xticklabels()]
    over = [i for i, x in enumerate(cuts) if peaks.get(x, 0) * mf.term_bytes(c) / 40 > g * GPU_TERMS]
    if over:
        ax.axvspan(over[0] - 0.5, ax.get_xlim()[1], color=BEYOND, zorder=0, linewidth=0)


XL_NEED = "Cutoff (# of Terms)"


def stops(reasons):
    """The legend text of the crosses: what stopped the runs of this figure."""
    reasons = {r for r in reasons if r}
    return "out of memory" if reasons == {"out of memory"} else "out of memory or over 40 min"




def circuits16(name, per_term=False, kinds=("ours", "cupp", "defer"), keep=None):
    """All seven circuits on 16 GPUs in one picture as wide as a slide (1744 x 580 px there). Two rows of panels whose
    widths follow their numbers of cutoffs, so that the tick labels have one size: the three main circuits above,
    the other four below. One legend line, one x label, the y axis once per row.
    per_term: the same runs as the time divided by the number of terms (ns): a flat line is a time that grows in
    proportion to the problem. Returns {(circuit, kind): {cutoff: value plotted}}."""
    which = ("ki", "tfim", "noise", "tfimsq", "xxz", "qaoa", "hea")
    widths = ((6, 6, 5), (4, 4, 5, 6))                  # cutoffs per panel (slide_ladder)
    fig = plt.figure(figsize=(17.44, 5.8), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.05, h_pad=0.04, wspace=0.03, hspace=0.07, rect=(0, 0, 1, 0.935))     # the legend line above
    outer = fig.add_gridspec(2, 1)
    panels, first = [], []
    for r, ratio in enumerate(widths):
        sub = outer[r].subgridspec(1, len(ratio), width_ratios=ratio)
        first.append(len(panels))
        panels += [fig.add_subplot(sub[0, i]) for i in range(len(ratio))]
    more = {}                                           # terms of the problems only more than 16 GPUs finished
    for r in scaling_jobs():
        if r["kind"] == "engine" and mf.ok(r) and r.get("boundary"):
            more.setdefault({v: k for k, v in mf.BIG_NAME.items()}.get(r["circuit"]), {})[r["opt"]["--coef"]] = r["boundary"]
    G = campaign()
    rows = mf.g_circuits(G, axes=panels, which=which, keep=keep or slide_ladder, sizes=more, kinds=kinds)
    shown = {}
    for j, ax in enumerate(panels):
        c = which[j]
        for t in ax.get_xticklabels():
            t.set_fontsize(13.5)
        ax.set_title(ax.get_title(), fontsize=15, pad=6)
        ax.set_ylabel(("Time per Term (ns)" if per_term else "Time (s)") if j in first else "", fontsize=15, fontweight="bold")      # the full label is longer than a panel is tall
        ax.tick_params(axis="y", labelsize=12.5, labelleft=(j in first))
        ax.set_xlabel("")
        ticks = [t.get_text().split("\n")[0] for t in ax.get_xticklabels()]
        size_ticks(ax, c, peaks_of(c, ticks), 16)
        times = {kind: dict(rows[len(kinds) * j + k][2]) for k, kind in enumerate(kinds)}
        if per_term:
            for line in list(ax.lines):
                line.remove()
            for kind in ("ours", "cupp", "defer"):
                at = [(i, times[kind][x] / mf.gterms(G, c, x) * 1e9) for i, x in enumerate(ticks) if x in times[kind] and mf.gterms(G, c, x)]
                col, mk = mf.SQ[kind]
                ax.plot([a for a, _ in at], [b for _, b in at], color=col, marker=mk, markersize=9.5, linewidth=3.6 if kind == "ours" else 3.0,
                        label=mf.SQ_NAME[kind] if j == 0 else None, zorder=4 if kind == "ours" else 3, solid_capstyle="round")
                shown[c, kind] = {ticks[a]: b for a, b in at}
            ax.set_ylim(10, 60000)
            ax.set_yticks([10, 100, 1000, 10000])
        else:
            ax.set_yticks([1, 10, 100, 1000])
            for kind in times:
                shown[c, kind] = {x: times[kind][x] for x in ticks if x in times[kind]}
            for x in ticks:
                if x in times["ours"] and x in times["cupp"]:
                    value(ax, ticks.index(x), times["cupp"][x], mf.times(times["cupp"][x], times["ours"][x]), dy=8, size=13)
    fig.supxlabel(XL_NEED, fontsize=15, fontweight="bold")
    e = entries(fig, panels[0])
    if mf.STOPS:
        e[-1] = (e[-1][0], stops(r[3][1] for r in rows if r[3]))
    e += [(Patch(facecolor=BEYOND, edgecolor="#B5BCC4"), "does not fit on the 16 GPUs")]
    if not per_term:
        e += [(Line2D([0], [0], color="none"), "N×: cuPauliProp / SectorQ time")]
    fig.legend([a for a, _ in e], [b for _, b in e], loc="upper center", bbox_to_anchor=(0.5, 1 - 0.07 / 5.8), ncol=len(e), fontsize=13,
               columnspacing=1.3, handlelength=2.0, handletextpad=0.5, borderpad=0, borderaxespad=0, frameon=False)
    fig.draw_without_rendering()
    if shrink_ticks(fig):                                   # one size of tick labels in all panels
        size = min(t.get_fontsize() for ax in panels for t in ax.get_xticklabels())
        for ax in panels:
            for t in ax.get_xticklabels():
                t.set_fontsize(size)
    finish(fig, name, place=False)
    return shown


def d_circuits_all():
    return circuits16("d_circuits_all")


def d_cost_all():
    return circuits16("d_cost_all", per_term=True)


def circuits_big(g, ladder=("1e-4", "3e-5", "1e-5", "5e-6", "3e-6", "2e-6", "1e-6", "7e-7", "5e-7", "3e-7", "2e-7"), original_baseline=False,
                 circuits=None, kinds=("ours", "cupp", "defer", "orqa", "monoprop")):
    """The three circuits of the scaling jobs on `g` GPUs in one picture as wide as a slide (1744 x 580 px there): time
    against the cutoff for SectorQ and the two modes of cuPauliProp on the same nodes, in the form of d_circuits_all.
    SectorQ at cutoff 1e-6 and below: the runs of the ladder of the scale campaign (a higher sector limit), as in
    d_scaleout. A cross: the next cutoff of that implementation failed."""
    big = [r for r in scaling_jobs() if r["gpus"] == g and (mf.ok(r) or r["fatal"] or r["exit"] is not None)]      # not a run that is still going
    G = campaign()

    circuits = circuits or mf.G3

    def run_new(kind, c, x):
        """The circuits without runs of the scaling jobs (TFIM 11x11, XXZ, QAOA, ansatz): the records of the scale
        campaign (jobs/u_items.py), cuPauliProp on the original circuit first (o.: --frame 0 where the frame removes
        Clifford gates); SectorQ with the terms of its log for the tick labels and the shading."""
        if kind == "prs":
            return mf.as_big(mf.prs_rec(c, x, g))
        ids = {"orqa": (f"orqa.{c}.{x}.n{g // 4}",), "monoprop": (f"monoprop.{c}.{x}.n{g // 4}",),
               "cupp": (f"o.cupp.{c}.{x}.g{g}", f"cupp.{c}.{x}.g{g}"), "defer": (f"defer.{c}.{x}.g{g}",), "ours": (f"eng.{c}.{x}.g{g}",)}[kind]
        rec = mf.scale_rec(*ids)
        out = mf.as_big(rec)
        if kind == "ours" and out and mf.ok(out):
            d = mf.gdetail(rec) or {}
            out.update(boundary=d.get("boundary"), inside=d.get("inside"))
        return out

    def run(kind, c, x):
        if c not in mf.BIG_NAME:
            return run_new(kind, c, x)
        if kind == "prs":
            return mf.as_big(mf.prs_rec(c, x, g))
        if kind in ("orqa", "monoprop"):                 # the CPU implementations on the same nodes (queue_s)
            return mf.as_big(mf.scale_rec(f"{kind}.{c}.{x}.n{g // 4}"))
        if original_baseline and kind == "cupp" and c in ("ki", "noise"):
            # Do not replace missing original-circuit measurements with frame-input runs.
            return mf.as_big(mf.scale_rec(f"o.cupp.{c}.{x}.g{g}"))
        if kind != "ours":
            return mf.find(big, "cupp", mf.BIG_NAME[c], float(x), gpus=g, tuned=(kind == "defer"))
        hits = [r for r in big if r["kind"] == "engine" and r["circuit"] == mf.BIG_NAME[c] and abs(r["coef"] / float(x) - 1) < 1e-9]
        plain = [r for r in hits if "--gmax" not in r["opt"] and "--chunk" not in r["opt"]]
        wide = [r for r in hits if r["opt"].get("--gmax") == "12"] + [r for r in hits if r["opt"].get("--gmax") == "10" and "--chunk" not in r["opt"]]
        order = (wide + plain) if float(x) <= 1.0000001e-6 else (plain + wide)
        done = [r for r in order if mf.ok(r)]
        return mf.newest(done)[0] if done else (order[0] if order else None)

    cuts = {c: [x for x in ladder if any(run(k, c, x) if mf.STOPS else mf.ok(run(k, c, x)) for k in kinds)] for c, _ in circuits}
    fig = plt.figure(figsize=(17.44, 5.8 if len(circuits) <= 3 else 10.4), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.05, h_pad=0.04, wspace=0.03, rect=(0, 0, 1, 0.935))     # the legend line above
    if len(circuits) <= 3:
        grid = fig.add_gridspec(1, 3, width_ratios=[len(cuts[c]) for c, _ in circuits])
        cells = [grid[0, j] for j in range(len(circuits))]
    else:                                               # three circuits above, the others below
        grid = fig.add_gridspec(2, 4)
        cells = [grid[0, j] for j in range(3)] + [grid[1, j] for j in range(len(circuits) - 3)]
    panels, rows, why = [], {}, set()
    for j, (c, title) in enumerate(circuits):
        ax = fig.add_subplot(cells[j])
        panels.append(ax)
        more, peak = {}, {}
        for kind in kinds:
            recs = [run(kind, c, x) for x in cuts[c]]
            pts = [(i, r["wall"]) for i, r in enumerate(recs) if mf.ok(r)]
            col, mk = mf.SQ[kind]
            if pts:
                named = any(l.get_label() == mf.SQ_NAME[kind] for a in panels[:-1] for l in a.get_lines())      # first panel with points
                ax.plot([q[0] for q in pts], [q[1] for q in pts], color=col, marker=mk, markersize=11, linewidth=4.0 if kind == "ours" else 3.2,
                        label=None if named else mf.SQ_NAME[kind], zorder=4 if kind == "ours" else 3, solid_capstyle="round")
                fail = next((i for i in range(pts[-1][0] + 1, len(recs)) if recs[i] is not None), None)
                if fail is not None and mf.STOPS:
                    ax.plot([fail], [pts[-1][1]], color=col, marker="x", markersize=12, markeredgewidth=2.8, linestyle="none", zorder=5)
                    why.add("out of memory" if recs[fail]["fatal"] else "over 40 min")
            rows[c, kind] = {cuts[c][i]: w for i, w in pts}
            if kind == "ours":
                more = {cuts[c][i]: r["boundary"] for i, r in enumerate(recs) if mf.ok(r) and r.get("boundary")}
                peak = {cuts[c][i]: r["inside"] for i, r in enumerate(recs) if mf.ok(r) and r.get("inside")}
        mf.sq_axes(ax, None, "Simulation Time (s)" if j == 0 else None, size=15)
        ax.set_xticks(range(len(cuts[c])))
        ax.set_xticklabels(mf.cut_ticks(G, c, cuts[c], more=more), fontsize=13.5)
        ax.set_xlim(-0.4, len(cuts[c]) - 0.6)
        size_ticks(ax, c, peak, g)
        ax.set_ylim(0.1, 3000)                              # MonoProp finishes the smallest problems in a fraction of a second
        ax.set_yticks([0.1, 1, 10, 100, 1000])
        ax.tick_params(axis="y", labelsize=13, labelleft=(j == 0))
        ax.set_title(f"({'abcdefg'[j]}) {title}", fontsize=16, pad=7)
        for i, x in enumerate(cuts[c]):                     # how many times as long the default mode takes as SectorQ
            if x in rows[c, "ours"] and x in rows[c, "cupp"]:
                value(ax, i, rows[c, "cupp"][x], mf.times(rows[c, "cupp"][x], rows[c, "ours"][x]), dy=9, size=14)
    fig.supxlabel(XL_NEED, fontsize=15, fontweight="bold")
    seen = {}                                           # every line once, whichever panel names it
    for a in panels:
        for h, l in zip(*a.get_legend_handles_labels()):
            seen.setdefault(l, h)
    e = [(seen[mf.SQ_NAME[k]], mf.SQ_NAME[k]) for k in kinds if mf.SQ_NAME[k] in seen]
    if mf.STOPS:
        e.append((Line2D([0], [0], **CROSS), stops(why)))
    e += [(Patch(facecolor=BEYOND, edgecolor="#B5BCC4"), f"does not fit on the {g} GPUs"),
          (Line2D([0], [0], color="none"), "N×: cuPauliProp / SectorQ time")]
    fig.legend([a for a, _ in e], [b for _, b in e], loc="upper center", bbox_to_anchor=(0.5, 1 - 0.07 / 5.8), ncol=len(e), fontsize=13,
               columnspacing=1.3, handlelength=2.0, handletextpad=0.5, borderpad=0, borderaxespad=0, frameon=False)
    fig.draw_without_rendering()
    if shrink_ticks(fig):                                   # one size of tick labels in all panels
        size = min(t.get_fontsize() for ax in panels for t in ax.get_xticklabels())
        for ax in panels:
            for t in ax.get_xticklabels():
                t.set_fontsize(size)
    finish(fig, f"d_circuits_g{g}", place=False)
    return rows


def d_circuits_g128():
    return circuits_big(128)


def d_scaleout():
    fig, ax = plt.subplots(figsize=(W, H))
    mf.g_scaleout(campaign(), scaling_jobs(), ax=ax)
    e = entries(fig, ax)
    half = (len(e) + 1) // 2
    top = strip(fig, [e[:half], e[half:]], size=13.5)
    finish(fig, "d_scaleout", top - 0.1 / H)


def d_weak():
    # rows: the two loads, columns: the three circuits (the README has them the other way round); one y axis per row
    fig, grid = plt.subplots(2, 3, figsize=(W, H))
    mf.g_weak(campaign(), scaling_jobs(), axes=[[grid[0][j], grid[1][j]] for j in range(3)])
    names = [name.replace("\n", " ") for _, name in mf.G3]
    for r in range(2):
        for j in range(3):
            ax = grid[r][j]
            ax.set_title(f"({'abcdef'[3 * r + j]}) {names[j]}, {'Small' if r == 0 else 'Large'} Load", fontsize=12.5, pad=6)
            ax.set_ylabel("Time (s)" if j == 0 else "", fontsize=14, fontweight="bold")        # the full label is longer than a panel is tall
            ax.tick_params(axis="y", labelsize=12, labelleft=(j == 0))
            if r == 0:
                ax.set_xlabel("")
            else:
                ax.xaxis.label.set_fontsize(14)
            one_row(ax, 11.5)                                  # both rows run to 512 GPUs: GPU counts on one line, the cutoff under every second
    e = entries(fig, grid[0][0])
    top = strip(fig, [e[:3], e[3:]], size=13)
    finish(fig, "d_weak", top - 0.08 / H, wgap=0.1)


def d_depth():
    fig, ax = plt.subplots(figsize=(W, H))
    h, n = mf.g_depth(campaign(), ax=ax)
    top = strip(fig, three_lines(list(zip(h, n))), size=13, side=0.1, handle=2.1, apart=1.2)
    finish(fig, "d_depth", top - 0.1 / H)


def d_accuracy():
    fig, axes = plt.subplots(1, 2, figsize=(W, H), sharey=True)
    h, n = mf.g_accuracy_time(campaign(), axes=axes)
    for ax in axes:
        stagger(ax, 11.5)
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo - 0.25, hi)                               # the first label stays clear of the lowest label of the y axis
        ax.xaxis.label.set_fontsize(14.5)
    top = strip(fig, three_lines(list(zip(h, n))), size=12.5, side=0.1, handle=2.1, apart=1.2)
    finish(fig, "d_accuracy", top - 0.08 / H, wgap=0.16)


# ---------------------------------------------------------------------------------------
# d_reach  the largest kicked Ising problem each implementation finished, by GPU count (16-256 GPUs)
#   the size of a problem is its number of terms at window boundaries, taken from a SectorQ run of the same cutoff
# ---------------------------------------------------------------------------------------
REACH_GPUS = (16, 32, 64, 128, 256, 512)
REACH_LADDER = ("1e-4", "3e-5", "1e-5", "5e-6", "3e-6", "2e-6", "1e-6")      # the cutoffs of cuPauliProp's ladders on 32 GPUs and more


def reach():
    """{GPU count: {"ours": (cutoff, terms, GiB of host memory in use), "cupp": (cutoff, terms), "defer": (cutoff, terms)}}"""
    G, big = campaign(), scaling_jobs()
    low = lambda xs: min(xs, key=float) if xs else None
    out = {}
    for g in REACH_GPUS:
        if g == 16:
            x = low([x for x in mf.gladder(G, "eng.ki", "g16") if mf.gok(G[f"eng.ki.{x}.g16"])])
            d = mf.gdetail(G[f"eng.ki.{x}.g16"])
            ours = (x, d["boundary"], d.get("host_gib"))
            base = {"cupp": low([x for x in mf.gladder(G, "cupp.ki", "g16") if mf.gok(G[f"cupp.ki.{x}.g16"])]),
                    "defer": low([x for x in mf.gladder(G, "cupp.ki", "g16") if mf.gok(mf.gdeferred(G, "ki", x, 16))])}
        else:
            runs = [r for r in big if r["gpus"] == g and r["circuit"] == "ki_lab"]
            r = min((r for r in runs if r["kind"] == "engine" and mf.ok(r)), key=lambda r: (r["coef"], r.get("binary") != mf.NEW_ENGINE))
            ours = (r["opt"]["--coef"], r["boundary"], r.get("host_gib"))
            base = {tag: low([q["opt"]["--coef"] for q in runs if q["kind"] == "cupp" and q["tuned"] == tuned and mf.ok(q)
                              and q["opt"]["--coef"] in REACH_LADDER]) for tag, tuned in (("cupp", False), ("defer", True))}
        # the CPU implementations on the same nodes: campaign g on 4 nodes, the scale campaign above
        for b in ("orqa", "monoprop"):
            if g == 16:
                base[b] = low([x for x in mf.gladder(G, f"{b}.ki", "n4") if mf.gok(G.get(f"{b}.ki.{x}.n4"))])
            else:
                base[b] = low([x for x in CPU_LADDER if mf.gok(mf.scale_rec(f"{b}.ki.{x}.n{g // 4}"))])
        out[g] = dict(ours=ours, **{tag: (x, terms(G, big, x)) for tag, x in base.items() if x})
    return out


CPU_LADDER = ("1e-4", "3e-5", "1e-5", "5e-6", "3e-6", "2e-6", "1e-6", "7e-7", "5e-7", "3e-7", "2e-7")   # jobs/s_items.py, jobs/t_items.py


def terms(G, big, x):
    """Terms at the window boundaries of kicked Ising at cutoff x: from campaign g, else from a SectorQ run on more GPUs."""
    n = mf.gterms(G, "ki", x)
    if n:
        return n
    hits = [r for r in big if r["kind"] == "engine" and r["circuit"] == "ki_lab" and abs(r["coef"] / float(x) - 1) < 1e-9 and mf.ok(r) and r.get("boundary")]
    return hits[0]["boundary"] if hits else None


def d_reach():
    R = reach()
    pos = {g: i for i, g in enumerate(REACH_GPUS)}
    fig, ax = plt.subplots(figsize=(W, H))
    for kind in ("ours", "cupp", "defer", "orqa", "monoprop"):
        col, mk = mf.SQ[kind]
        pts = [(pos[g], R[g][kind][1]) for g in REACH_GPUS if kind in R[g] and R[g][kind][1]]
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker=mk, markersize=11, linewidth=3.4,
                linestyle="--" if kind == "defer" else "-", label=mf.SQ_NAME[kind], zorder=5 if kind == "ours" else 4 if kind == "cupp" else 3)
    for g in REACH_GPUS:
        x, n, host = R[g]["ours"]
        rise = g != REACH_GPUS[-1] and R[REACH_GPUS[pos[g] + 1]]["ours"][1] > 1.5 * n      # the line climbs to the right: keep the label off it
        value(ax, pos[g], n, f"{short_count(n)}\n(Cutoff {x})", dx=-14 if rise and g != REACH_GPUS[0] else 0, dy=11, size=13)
        c, d = R[g]["cupp"][1], R[g].get("defer", (None, None))[1]
        value(ax, pos[g], c, short_count(c), dy=10, size=12.5)
        if d and d != c:
            value(ax, pos[g], d, short_count(d), dy=-11, size=12.5)
        mid = math.sqrt(n * c)
        ax.annotate("", (pos[g] + 0.2, n), xytext=(pos[g] + 0.2, c), arrowprops=dict(arrowstyle="<->", color=mf.INK, linewidth=1.3, shrinkA=7, shrinkB=7), zorder=2)
        t = ax.text(pos[g] + 0.24, mid, f"{n / c:.0f}×", ha="left", va="center", fontsize=14.5, fontweight="bold", color=mf.INK, path_effects=HALO, zorder=6)
        t.set_gid("value")
        if "--print" in ARGS:
            print(f"d_reach {g:3d} GPUs  SectorQ {x} {n:,} terms ({host:,.0f} GiB)  cuPauliProp {R[g]['cupp'][0]} {c:,}  deferred {R[g].get('defer')}  ratio {n / c:.1f}")
    mf.sq_axes(ax, "# of GPUs (Host Memory SectorQ Used / Installed, GiB)", "Largest Finished Problem (# of Terms)")
    ax.set_xticks(range(len(REACH_GPUS)))
    ax.set_xticklabels([f"{g}\n({R[g]['ours'][2]:,.0f} / {g // 4 * 256:,})" for g in REACH_GPUS])
    ax.set_xlim(-0.5, len(REACH_GPUS) - 0.25)          # room right of the last arrow for its label
    ax.set_ylim(6e7, 2.4e12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e12:g}T" if v >= 1e12 else short_count(v)))
    top = strip(fig, [list(zip(*ax.get_legend_handles_labels()))], size=13.5)
    finish(fig, "d_reach", top - 0.1 / H)


# ---------------------------------------------------------------------------------------
# d_variability  ten runs per setting (kicked Ising): every run as a point, its time relative to the mean of its ten runs
#   (a) 16 GPUs over the problem size, (b) cutoff 3e-5 over the GPU count; at each group its spread (max - min) / mean
# ---------------------------------------------------------------------------------------
def d_variability():
    G = campaign()
    parts = [("(a) 16 GPUs, Problem Size Varied", mf.xl_cut(), [(x, 16) for x in mf.VAR_SIZES], mf.cut_ticks(G, "ki", list(mf.VAR_SIZES))),
             (f"(b) Cutoff {mf.VAR_AT}, GPU Count Varied", "# of GPUs", [(mf.VAR_AT, g) for g in mf.VAR_GPUS], [str(g) for g in mf.VAR_GPUS])]
    fig, axes = plt.subplots(1, 2, figsize=(W, H), sharey=True)
    side = {"ours": -0.2, "cupp": 0.2}
    for ax, (title, xlabel, cols, names) in zip(axes, parts):
        ax.axhline(1.0, color="black", linewidth=1.1, linestyle=(0, (5, 3)), zorder=2)
        for kind, tag in (("ours", "eng"), ("cupp", "cupp")):
            col, mk = mf.SQ[kind]
            for i, (x, g) in enumerate(cols):
                t = [v for v in mf.var_runs(G, tag, x, g) if v is not None]
                if not t:
                    if (G.get(f"{tag}.ki.{x}.g{g}") or {}).get("status") == "oom":          # the sweep's run of the setting
                        ax.text(i + side[kind], 1.0, "out of memory", rotation=90, ha="center", va="center", fontsize=12.5, color="#666666",
                                path_effects=HALO, zorder=6)
                    continue
                mean, lo, hi, spread = mf.var_stat(t)
                ax.plot([i + side[kind] + 0.1 * (k / (len(t) - 1) - 0.5) for k in range(len(t))], [v / mean for v in t], color=col, marker=mk,
                        markersize=8.5, linestyle="none", alpha=0.8, label=mf.SQ_NAME[kind] if i == 0 else None, zorder=4)
                if kind == "ours":                       # the two labels of a setting do not meet: one under its group, one above
                    value(ax, i + side[kind], lo / mean, f"{spread * 100:.1f}%", dy=-8, size=12.5).set_color(col)
                else:
                    value(ax, i + side[kind], hi / mean, f"{spread * 100:.1f}%", dy=8, size=12.5).set_color(col)
                if "--print" in ARGS:
                    print(f"d_variability {tag:4s} {x} g{g:<2d} mean {mean:.3g} s  {lo:.3g}-{hi:.3g} s  spread {spread * 100:.1f}%")
        mf.sq_axes(ax, xlabel, "Run Time / Mean of 10 Runs" if ax is axes[0] else None, log=False)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(names, fontsize=12)
        ax.set_xlim(-0.55, len(cols) - 0.45)
        ax.set_title(title, fontsize=14.5, pad=7)
        ax.xaxis.label.set_fontsize(14.5)
    axes[0].set_ylim(0.93, 1.17)
    axes[0].yaxis.set_major_locator(FixedLocator([0.95, 1.0, 1.05, 1.1, 1.15]))
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    h, n = axes[0].get_legend_handles_labels()
    top = strip(fig, [list(zip(h, n)) + [(Line2D([0], [0], color="none"), "Number at Each Group: (Max − Min) / Mean")]], size=13.5)
    finish(fig, "d_variability", top - 0.1 / H, wgap=0.16)


FIGURES = [n02, n11, n13, n14, n16, d_single_node, d_strong, d_qubits, d_breakdown, d_circuits_a, d_circuits_b, d_circuits_all, d_cost_all, d_circuits_g128, d_scaleout, d_reach, d_weak, d_depth,
           d_accuracy, d_variability]

if __name__ == "__main__":
    want = [a for a in ARGS if not a.startswith("-")]
    for f in FIGURES:
        if not want or any(f.__name__.startswith(w) for w in want):
            f()
    for name, (w, h, n, legend, ticks) in SIZES.items():
        sizes = (f"  x tick labels {', '.join(f'{t:g}' for t in ticks)} pt" + (f", legend {legend:g} pt" if legend else "")) if "--print" in ARGS else ""
        print(f"{name}.png  {w} x {h} px  ({w / h:.2f}:1){sizes}" + (f"  {n} CHECK lines" if n else ""))
