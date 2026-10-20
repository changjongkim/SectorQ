#!/usr/bin/env python3
"""Data figures of the paper (paper/figures/*.pdf): one PDF per panel and one legend strip per figure.

  ../env/bin/python3 make_paper_figures.py [name ...]      (default: every figure; names from FIGURES below)

The panels are drawn by the functions of the README and deck figures (make_eval_figures.py, make_deck_figures.py: the
same data, colors and markers); their save step is intercepted and every panel is placed again on a canvas of its own:
  - the paper prints every panel and legend at SCALE of its size, so one font size on the canvas is one size in print:
    T points for tick labels and legends, V points for printed values, bold axis titles only;
  - a panel has the canvas of its place in the paper (PLACE: four, three, two or one panel per text width), with the
    tick labels, axis labels and values inside it; cutoff labels have the parenthesized
    term count or error directly below, centered at the same tick; left, middle and right labels are shown;
  - no title (it becomes the subfloat caption, printed here), the common x label and the y label on independent panels,
    shared y labels once per comparison row, and the legend in a strip of its own as wide as the text.
Prints, per figure, the files with their captions, and the overlaps that the check of the deck figures still finds.
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.text import Text
from matplotlib.ticker import LogFormatterMathtext

HERE = os.path.dirname(os.path.abspath(__file__))
ARGS, sys.argv = sys.argv[1:], sys.argv[:1]
sys.path.insert(0, HERE)
import make_eval_figures as mf                      # noqa: E402
import make_deck_figures as md                      # noqa: E402
import make_prs_figure as mp                        # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "paper", "figures")
os.makedirs(OUT, exist_ok=True)
SCALE = 0.6                                         # \includegraphics[scale=0.6] for every file of this script
TEXT_W = 5.48 / SCALE                               # the text width of acmsmall (395.8 pt) on the canvas
T, V, AXIS = 11, 10, 11.5                           # points on the canvas: 6.6, 6.0 and 6.9 points in print
PLACE = {4: (2.17, 2.0), 3: (2.95, 2.0), 2: (4.5, 2.1), 1: (TEXT_W, 2.3)}      # canvas of a panel, by panels per text width
HALO = [patheffects.withStroke(linewidth=1.4, foreground="white")]
CAUGHT = {}


def catch(fig, name, *a, **k):
    CAUGHT[name] = fig


md.finish = catch
mf.save = catch


# ----------------------------------------------------------------------------------------------- one panel
def panels_of(fig):
    return [ax for ax in fig.axes if ax.axison and (ax.lines or ax.patches or ax.collections)]


def legend_items(fig):
    """(handle, label) of every legend of the figure, in order, without repeats."""
    out, seen = [], set()
    legs = list(fig.legends) + [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    for leg in legs:
        for h, t in zip(leg.legend_handles, leg.get_texts()):
            if t.get_text() not in seen:
                seen.add(t.get_text())
                out.append((h, t.get_text()))
    return out


def restyle(ax):
    """Sizes for print: text, lines, markers, frame."""
    for t in [ax.xaxis.label, ax.yaxis.label]:
        t.set_fontsize(AXIS)
        t.set_fontweight("bold")
    ax.tick_params(axis="both", which="major", labelsize=T, length=3.5, width=0.8, pad=2.5)
    minor = [t for t in ax.xaxis.get_minor_ticks() if t.label1.get_text().strip()]
    if minor:                                       # the second line of md.one_row, or the lower row of md.stagger
        length = ax.xaxis.get_minor_ticks()[0].tick1line.get_markersize()
        if length > 5:
            ax.tick_params(axis="x", which="minor", length=2 * T * 1.36, width=0.5, labelsize=T, pad=2.5)
        else:
            ax.tick_params(axis="x", which="minor", length=0, labelsize=T, pad=2.5 + 3.5 + 1.22 * T)
    else:
        ax.tick_params(axis="both", which="minor", length=2, width=0.6)
    for t in ax.get_xticklabels(which="both") + ax.get_yticklabels(which="both"):
        t.set_fontweight("normal")
    for t in ax.get_xticklabels(which="both"):
        t.set_horizontalalignment("center")
        t.set_multialignment("center")
        t.set_linespacing(1.1)
    for t in (ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()):
        t.set_fontsize(T)
        t.set_fontweight("normal")
    for t in ax.texts:
        t.set_fontsize(V)
        t.set_fontweight("normal")
        if t.get_path_effects():
            t.set_path_effects(HALO)
    for s in ax.spines.values():
        s.set_linewidth(0.8)


def thin_lines(ax):
    """Lines and markers for print (once per panel)."""
    for line in ax.lines:
        if line.get_linewidth() > 1.4:
            line.set_linewidth(max(1.3, line.get_linewidth() * 0.5))
        line.set_markersize(line.get_markersize() * 0.6)
        line.set_markeredgewidth(min(line.get_markeredgewidth(), 1.0))


def place(fig, ax, w, h, pad=0.04):
    """Moves the panel until its labels reach `pad` inches from the canvas edges."""
    fig.set_size_inches(w, h)
    ax.set_position([0.2, 0.2, 0.7, 0.7])
    for _ in range(6):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        tb, bb, d = ax.get_tightbbox(r), ax.get_window_extent(r), fig.dpi
        x0 = (bb.x0 - tb.x0) / d + pad
        x1 = w - (tb.x1 - bb.x1) / d - pad
        y0 = (bb.y0 - tb.y0) / d + pad
        y1 = h - (tb.y1 - bb.y1) / d - pad
        new = [x0 / w, y0 / h, max(0.05, (x1 - x0) / w), max(0.05, (y1 - y0) / h)]
        old = ax.get_position().bounds
        ax.set_position(new)
        if max(abs(a - b) for a, b in zip(new, old)) < 1e-4:
            break


def ticks_meet(fig, ax):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for labels in (md.xlabels(ax), md.xlabels(ax, minor=True)):
        boxes = sorted((t.get_window_extent(r) for t in labels), key=lambda b: b.x0)
        gap = 0.25 * T * fig.dpi / 72
        if any(min(a.x1, b.x1) - max(a.x0, b.x0) > -gap and min(a.y1, b.y1) - max(a.y0, b.y0) > 0
               for i, a in enumerate(boxes) for b in boxes[i + 1:]):
            return True
    return False


LABELS = {"Simulation Time (s)": "Time (s)", "Terms at the Start of the Window": "Terms at Window Start",
          "Error Against the Exact Value": "Error", "Run Time / Mean of 10 Runs": "Time / Mean",
          "Time per Term (ns)": "Time/Term (ns)", "Cutoff (Largest # of Terms)": "Cutoff (# of Terms)",
          "Cutoff (Error against the Exact Value)": "Cutoff (Error)", "Fixed Cost per Gate (ms)": "Cost per Gate (ms)"}


def merge_rows(ax, drop_minor):
    """The labels of minor ticks (the lower row of md.stagger, the second lines of md.one_row) back into one row of
    major labels, or dropped (`drop_minor`)."""
    minor = [(t.get_loc(), t.label1.get_text()) for t in ax.xaxis.get_minor_ticks() if t.label1.get_text().strip()]
    if not minor:
        return
    major = [(loc, t.get_text()) for loc, t in zip(ax.get_xticks(), ax.get_xticklabels())]
    if drop_minor:
        rows = major
    else:
        at = dict(major)
        for loc, text in minor:
            at[loc] = (at[loc] + "\n" + text) if at.get(loc) else text
        rows = sorted(at.items())
    ax.set_xticks([l for l, _ in rows])
    ax.set_xticklabels([t for _, t in rows])
    ax.set_xticks([], minor=True)
    ax.grid(False, which="minor", axis="x")


def stacked_tick_labels(labels):
    """Center a numeric tick and its parenthesized quantity on two aligned lines."""
    numeric = all(re.match(r"[0-9]", s) for s in labels if s.strip())
    secondary = numeric and any("(" in s for s in labels)
    if not secondary:
        return labels, False
    out = []
    for text in labels:
        if not text.strip():
            out.append("")
            continue
        top, sep, bottom = text.partition("(")
        quantity = sep + " ".join(bottom.split()) if sep else ""
        count = re.fullmatch(r"\(([\d.]+)\s*([KMBT])\)", quantity)
        if count:
            # Two significant digits keep three counts legible in narrow panels.
            value = float(f"{float(count.group(1)):.2g}")
            quantity = f"({value:g}{count.group(2)})"
        # An unmeasured point reserves the lower line without inventing a count.
        out.append(top.strip() + "\n" + quantity)
    return out, True


def representative_labels(labels, ticks):
    """Keep endpoints and the available tick nearest their horizontal midpoint."""
    visible = [i for i, s in enumerate(labels) if s.strip()]
    if len(visible) <= 3:
        return labels
    first, last = visible[0], visible[-1]
    midpoint = (ticks[first] + ticks[last]) / 2
    middle = min(visible[1:-1], key=lambda i: abs(ticks[i] - midpoint))
    keep = {first, middle, last}
    return [s if i in keep else "" for i, s in enumerate(labels)]


def tick_choices(labels, ticks, representative=False):
    """Keep each label intact; paired quantities use three positions from the outset."""
    if representative:
        yield representative_labels(labels, ticks)
        return
    yield labels
    if all(re.match(r"[0-9]", s) for s in labels if s.strip()):
        yield representative_labels(labels, ticks)


def spread_values(fig, ax):
    """Printed values along a line that meet: the first and the last stay, a value between them stays only when it
    meets none of the values kept."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    vals = sorted((t for t in ax.texts if t.get_visible() and t.get_text().strip() and hasattr(t, "xyann")),
                  key=lambda t: t.get_window_extent(r).x0)
    if len(vals) < 3:
        return
    box = {id(t): t.get_window_extent(r) for t in vals}
    meet = lambda a, b: min(box[id(a)].x1, box[id(b)].x1) - max(box[id(a)].x0, box[id(b)].x0) > -2 and \
        min(box[id(a)].y1, box[id(b)].y1) - max(box[id(a)].y0, box[id(b)].y0) > -2
    kept = [vals[0], vals[-1]]
    for t in vals[1:-1]:
        if any(meet(t, k) for k in kept):
            t.set_visible(False)
        else:
            kept.append(t)


def yticks_meet(fig, ax):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    lo, hi = sorted(ax.get_ylim())
    boxes = sorted((t.label1.get_window_extent(r) for t in ax.yaxis.get_major_ticks()
                    if t.label1.get_visible() and t.label1.get_text().strip() and lo <= t.get_loc() <= hi), key=lambda b: b.y0)
    return any(b.y0 - a.y1 < 0 for a, b in zip(boxes, boxes[1:]))


def corner_meets(fig, ax):
    """The first x tick label meets the lowest y tick label."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    xs = md.xlabels(ax)
    lo, hi = sorted(ax.get_ylim())
    ys = [t.label1 for t in ax.yaxis.get_major_ticks() if t.label1.get_visible() and t.label1.get_text().strip() and lo <= t.get_loc() <= hi]
    if not xs or not ys:
        return False
    a = min(xs, key=lambda t: t.get_window_extent(r).x0).get_window_extent(r)
    return any(min(a.x1, b.x1) - max(a.x0, b.x0) > 0 and min(a.y1, b.y1) - max(a.y0, b.y0) > 0
               for b in (t.get_window_extent(r) for t in ys))


def contain_values(fig, ax, name):
    """Keep data labels inside the plot by moving their text, preserving their data anchors."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = ax.get_window_extent(renderer)
    pad = 2 * fig.dpi / 72
    for text in ax.texts:
        if not text.get_visible() or not text.get_text().strip():
            continue
        bounds = text.get_window_extent(renderer)
        dx = max(0, box.x0 + pad - bounds.x0) - max(0, bounds.x1 - box.x1 + pad)
        dy = max(0, box.y0 + pad - bounds.y0) - max(0, bounds.y1 - box.y1 + pad)
        if dx or dy:
            transform = text.get_transform()
            x, y = transform.transform(text.get_position())
            text.set_position(transform.inverted().transform((x + dx, y + dy)))
    fig.canvas.draw()
    for text in ax.texts:
        if text.get_visible() and text.get_text().strip():
            bounds = text.get_window_extent(fig.canvas.get_renderer())
            if bounds.x0 < box.x0 - 0.5 or bounds.y0 < box.y0 - 0.5 or bounds.x1 > box.x1 + 0.5 or bounds.y1 > box.y1 + 0.5:
                print(f"CHECK {name}: data label leaves the plot: {text.get_text()!r}", file=sys.stderr)


def panel(fig, ax, name, w, h, xlabel=None, ylabel=None, before=None, drop_minor=False, position=None):
    """Writes the panel `ax` of `fig` alone on a w x h canvas as OUT/<name>.pdf; returns its title."""
    def fit():
        if position is None:
            place(fig, ax, w, h)
        else:
            fig.set_size_inches(w, h)
            ax.set_position(position)

    axes = fig.axes
    state = [(a, a.get_visible(), a.axison) for a in axes]
    texts = [(t, t.get_visible()) for a in axes for t in a.texts] + [(t, t.get_visible()) for t in fig.texts]
    legs = list(fig.legends) + [a.get_legend() for a in axes if a.get_legend() is not None]
    for a in axes:
        if a is not ax:
            a.set_visible(False)
            a.set_axis_off()
            for t in a.texts:
                t.set_visible(False)
    for t in fig.texts:
        t.set_visible(False)
    for leg in legs:
        leg.set_visible(False)
    groups = getattr(fig, "_align_label_groups", None)
    if groups:                                      # labels aligned with other panels (fig.align_xlabels) stay put
        from matplotlib import cbook
        for k in groups:
            groups[k] = cbook.Grouper()
    title = re.sub(r"^\([a-z]\)\s*", "", ax.get_title())
    ax.set_title("")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_xlabel(LABELS.get(ax.get_xlabel(), ax.get_xlabel()))
    ax.set_ylabel(LABELS.get(ax.get_ylabel(), ax.get_ylabel()))
    ax.tick_params(axis="y", labelleft=True)
    if before:
        before(ax)
    merge_rows(ax, drop_minor)
    restyle(ax)
    thin_lines(ax)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    ticks = list(ax.get_xticks())
    labels, secondary = stacked_tick_labels(labels)
    representative = secondary or ax.get_xlabel().startswith("Cutoff")
    choices = tick_choices(labels, ticks, representative)
    for choice in choices:
        ax.set_xticks(ticks)
        ax.set_xticklabels(choice)
        restyle(ax)
        fit()
        if not ticks_meet(fig, ax):
            break
    if yticks_meet(fig, ax):                        # every second y tick
        ax.set_yticks(ax.get_yticks()[::2])
        fit()
    for _ in range(4):
        if not corner_meets(fig, ax):
            break
        for tick in ax.xaxis.get_major_ticks():
            tick.set_pad(tick.get_pad() + 2)
        fit()
    spread_values(fig, ax)
    fit()
    contain_values(fig, ax, name)
    shown = [t.get_text() for t in ax.get_xticklabels()]
    assert all(not current or current == original for current, original in zip(shown, labels)), name
    if labels:
        assert shown[0] == labels[0] and shown[-1] == labels[-1], name
    if representative:
        assert shown == representative_labels(labels, ticks), name
        assert not ticks_meet(fig, ax), name
    if secondary:
        assert all(s.count("\n") == 1 for s in shown if s), name
        assert all(not s.split("\n")[1] or s.split("\n")[1].startswith("(") for s in shown if s), name
    if getattr(ax, '_paper_require_all_xticks', False):
        assert shown == labels, name
    figlegs, axlegs = list(fig.legends), [(a, a.legend_) for a in axes]
    fig.legends.clear()                             # hidden legends are not part of this panel
    for a in axes:
        a.legend_ = None
    md.check(fig, name)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.legends.extend(figlegs)
    for a, leg in axlegs:
        a.legend_ = leg
    for a, vis, on in state:
        a.set_visible(vis)
        if on:
            a.set_axis_on()
    for t, vis in texts:
        t.set_visible(vis)
    return title


def strip(items, name, width=TEXT_W, ncol=None):
    """The legend strip of a figure: the entries in as few rows as fit the text width."""
    items = [(h, l) for h, l in items if l and not l.startswith("_")]
    for n in ([ncol] if ncol else range(len(items), 0, -1)):
        fig = plt.figure(figsize=(width, 1.0))
        leg = fig.legend([h for h, _ in items], [l for _, l in items], loc="center", ncol=n, fontsize=T, frameon=False,
                         columnspacing=1.2, handlelength=2.0, handletextpad=0.5, borderpad=0.1, labelspacing=0.3)
        for hd in leg.legend_handles:
            if hasattr(hd, "set_linewidth") and hasattr(hd, "get_marker"):
                hd.set_linewidth(max(1.3, hd.get_linewidth() * 0.5) if hd.get_linewidth() > 1.4 else hd.get_linewidth())
                hd.set_markersize(hd.get_markersize() * 0.55)
        for t in leg.get_texts():
            t.set_fontweight("normal")
        fig.canvas.draw()
        bb = leg.get_window_extent(fig.canvas.get_renderer())
        if bb.width <= width * fig.dpi or n == 1:
            box = bb.transformed(fig.dpi_scale_trans.inverted()).padded(0.02)
            fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches=box)
            plt.close(fig)
            return n
        plt.close(fig)


def split(fig, base, per_row, xlabel=None, ylabel=None, legend=True, size=None, before=None, only=None, drop_minor=False):
    """Every panel of `fig` as OUT/<base>_<letter>.pdf on the canvas of `per_row` panels per text width, and the legend
    strip as OUT/<base>_legend.pdf. Returns [(file, title)]."""
    w, h = size or PLACE[per_row]
    axes = panels_of(fig)
    sup = getattr(fig, "_supxlabel", None)
    common_x = xlabel or (sup.get_text() if sup is not None else None)
    common_y = ylabel or next((a.get_ylabel() for a in axes if a.get_ylabel()), None)
    items = legend_items(fig) if legend else []
    out = []
    for i, ax in enumerate(axes):
        if only is not None and i not in only:
            continue
        name = f"{base}_{chr(97 + len(out))}"
        x = None if ax.get_xlabel() and not xlabel else common_x
        y = None if ax.get_ylabel() and not ylabel else common_y
        out.append((name, panel(fig, ax, name, w, h, x, y, before, drop_minor)))
    if items:
        strip(items, base + "_legend")
        out.append((base + "_legend", ""))
    plt.close(fig)
    print(f"{base}: " + "  ".join(f"{n} [{t}]" for n, t in out))
    return out, items


def comparison_panels(fig, base, items=None):
    """Seven comparisons in full-width rows of four and three, with a shared y scale per row."""
    axes = panels_of(fig)
    assert len(axes) == 7
    assert all(ax.get_yscale() == "log" and ax.get_ylim() == axes[0].get_ylim() for ax in axes)
    entries = legend_items(fig) if items is None else items
    fig.set_layout_engine(None)
    out, index = [], 0
    gap, first_left, other_left, right = 0.09, 0.66, 0.23, 0.22
    height, bottom, top = 2.02, 0.72, 0.09
    for columns in (4, 3):
        plot_width = (TEXT_W - gap * (columns - 1) - first_left - other_left * (columns - 1)
                      - right * columns) / columns
        for column in range(columns):
            ax = axes[index]
            left = first_left if column == 0 else other_left
            width = left + plot_width + right
            position = [left / width, bottom / height, plot_width / width, (height - bottom - top) / height]

            def style(a, first=(column == 0)):
                count = len(a.get_xticks())
                margin = 0.08 if count == 4 else 0.45
                a.set_xlim(-margin, count - 1 + margin)
                a.set_ylabel("Time (s)" if first else "")
                a.tick_params(axis="y", labelleft=first)
                a.set_yticks([1, 10, 100, 1000])
                a.yaxis.set_major_formatter(LogFormatterMathtext())
                a.grid(False, which="both", axis="both")
                a.grid(True, which="major", axis="y", color="#d0d4da", linestyle="--", linewidth=0.65)
                for spine in a.spines.values():
                    spine.set_edgecolor("#737b85")
                for text in a.texts:
                    if hasattr(text, "xyann"):
                        dx, dy = text.xyann
                        if text.xy[0] == a.get_xticks()[0]:
                            dx = max(dx, 2)
                        text.xyann = (dx, max(dy, 11))

            name = f"{base}_{chr(97 + index)}"
            title = panel(fig, ax, name, width, height, xlabel="Cutoff (# of Terms)", before=style, position=position)
            out.append((name, title))
            index += 1
    strip(entries, base + "_legend")
    plt.close(fig)
    print(f"{base}: " + "  ".join(f"{name} [{title}]" for name, title in out))


# ----------------------------------------------------------------------------------------------- background
def bg_anatomy():
    """cuPauliProp's distributed mode on kicked Ising: time, fixed cost per gate, terms per GPU at the failure."""
    mf.background()
    col = mf.SQ["cupp"][0]
    for name, base, ylabel in (("bg_scaling.png", "bg_time", None), ("bg_gatecost.png", "bg_gatecost", None),
                               ("bg_capacity.png", "bg_capacity", "Terms per GPU (M)")):
        fig = CAUGHT.pop(name)
        for p in fig.axes[0].patches:
            p.set_facecolor(col)
        fig.axes[0].set_ylim(fig.axes[0].get_ylim()[0], fig.axes[0].get_ylim()[1] * 2)
        split(fig, base, 3, xlabel="# of GPUs", ylabel=ylabel, legend=False)


def bg_accuracy():
    md.n02()
    def style(ax):
        if ax.get_ylabel().startswith("Expectation"):
            ax.set_yticks([0, 0.2, 0.4])
            ax.set_xlim(-0.12, 3.15)
            for text in ax.texts:
                if hasattr(text, 'xyann') and text.xy[0] == 0:
                    text.xyann = (5, -8)
                    text.set_horizontalalignment('left')
                    text.set_verticalalignment('top')
    # Keep both data panels beside the unchanged operations diagram in Figure 1.
    split(CAUGHT.pop("n02_accuracy_terms"), "bg_accuracy", 4, size=(2.6, 2.0), legend=False, before=style)


def bg_windows():
    """(a) gates, gates at which 16 fixed parts move terms, and windows per circuit; (b) terms at the window starts."""
    import json
    d = json.load(open(os.path.join(os.path.dirname(HERE), "analysis", "fixed_rows.json")))
    logs = {"Kicked Ising": (5420, 20), "TFIM heavy-hex": (5420, 45), "Noisy kicked Ising": (7960, 20), "TFIM 11x11": (3410, 24),
            "XXZ 8x8": (2048, 23), "QAOA": (1100, 8), "Ansatz": (4784, 34)}           # gates and windows of the engine runs
    names = list(logs)
    short = ["KI", "TFIM\nHH", "Noisy\nKI", "TFIM\n2D", "XXZ", "QAOA", "Ansatz"]
    fig, ax = plt.subplots()
    x = range(len(names))
    w = 0.27
    series = (([logs[n][0] for n in names], "#9aa3ad", "Gates"),
              ([d[n]["fixed_random"] * logs[n][0] / d[n]["gates"] for n in names], mf.SQ["cupp"][0], "Gates That Move Terms, Fixed Rows"),
              ([logs[n][1] for n in names], mf.SQ["ours"][0], "Windows of SectorQ"))
    for k, (vals, col, lab) in enumerate(series):
        ax.bar([i + (k - 1) * w for i in x], vals, width=w, color=col, label=lab, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(3, 30000)
    ax.set_xticks(list(x))
    ax.set_xticklabels(short)
    ax.set_xlim(-0.45, len(names) - 0.55)
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle=":", color="#c8ccd2", zorder=0)
    fig.legend(*ax.get_legend_handles_labels())
    split(fig, "bg_windows", 2)
    md.n11()
    fig = CAUGHT.pop("n11_window_terms")
    fig.axes[0].set_ylim(0.5, 3e11)                   # room above the dashed reference for its label
    split(fig, "bg_window_terms", 2, ylabel="Terms at\nWindow Start", legend=False)


# ----------------------------------------------------------------------------------------------- evaluation
def ev_single():
    md.d_single_node()
    split(CAUGHT.pop("d_single_node"), "ev_single", 1, size=(TEXT_W, 2.5))


def ev_circuits():
    md.d_circuits_all()
    comparison_panels(CAUGHT.pop("d_circuits_all"), "ev_circuits")


def ev_cost():
    md.d_cost_all()
    split(CAUGHT.pop("d_cost_all"), "ev_cost", 3)


def ev_prs():
    fig = plt.figure()
    axes = [fig.add_subplot(2, 4, k + 1) for k in range(7)]
    mp.draw(axes, 13)
    for ax in axes:
        ax.set_ylabel("Time (s)")
        ax.set_xlabel(mf.xl_cut())
    handles, names = axes[0].get_legend_handles_labels()
    items = list(zip(handles, names))
    items.append((plt.Line2D([], [], color="none"), "N×: paulistrings-rs / SectorQ time"))
    comparison_panels(fig, "ev_prs", items=items)


def ev_size():
    """Qubits (two cutoffs) and depth in one figure of three panels, one legend."""
    md.d_qubits()
    fig = CAUGHT.pop("d_qubits")
    items = legend_items(fig)
    split(fig, "ev_size", 3, legend=False)
    md.d_depth()
    fig = CAUGHT.pop("d_depth")
    ax = panels_of(fig)[0]
    title = panel(fig, ax, "ev_size_c", *PLACE[3])
    seen = {l for _, l in items}
    union = items + [(h, l) for h, l in legend_items(fig) if l not in seen]
    strip(union, "ev_size_legend")
    plt.close(fig)
    print(f"ev_size: + ev_size_c [{title}] ev_size_legend")


def ev_accuracy():
    md.d_accuracy()
    split(CAUGHT.pop("d_accuracy"), "ev_accuracy", 2)


def ev_breakdown():
    """The existing cutoff sweeps as three vertical stacks, matching ev_large_time."""
    source, source_axes = plt.subplots(1, 3)
    rows = mf.g_breakdown_sizes(md.campaign(), axes=source_axes)
    plt.close(source)
    fig, axes = plt.subplots(1, 3)
    for ax, (_, title) in zip(axes, mf.G3):
        samples = [(coef, share) for name, coef, wall, share in rows if name == title]
        for i, (coef, share) in enumerate(samples):
            bottom = 0
            for label, key, color in mf.SIZE_PARTS:
                label = "Idle Workers" if key == "idle" else label
                value = 100 * share[key]
                ax.bar(i, value, bottom=bottom, width=0.65, color=color, label=label if i == 0 else None)
                bottom += value
        ax.set_xticks(range(len(samples)), [coef for coef, share in samples])
        ax.set_xlabel("Cutoff")
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 50, 100])
        ax.set_ylabel("Time Share (%)")
        ax.set_title(title)
    fig.legend(*axes[0].get_legend_handles_labels())
    split(fig, "ev_breakdown", 3)


def ev_variability():
    md.d_variability()
    split(CAUGHT.pop("d_variability"), "ev_variability", 2)


def ev_staircase():
    md.n14()
    fig = CAUGHT.pop("n14_staircase")
    ax = panels_of(fig)[0]
    ax.set_xticks(range(4))
    ax.set_xticklabels(["cuPauliProp", "+ Clifford\nFrame", "+ Deferred\nDedup", "+ Sectors\n(SectorQ)"])
    # The paper reports boundary populations, not summed intermediate expansion.
    for legend in fig.legends:
        for label in legend.get_texts():
            label.set_text(label.get_text().replace("1.4B Terms", "159M Terms").replace("22.5M Terms", "2.33M Terms"))
    split(fig, "ev_staircase", 2, xlabel="Execution Configuration", size=(PLACE[2][0], 2.3))


def ev_knobs():
    md.n13()
    fig = CAUGHT.pop("n13_knobs")
    fig.axes[0].set_ylim(8, 900)                     # shared headroom for labels of the 219/274 s points
    split(fig, "ev_knobs", 3)


def ev_weak():
    md.d_weak()
    split(CAUGHT.pop("d_weak"), "ev_weak", 3, drop_minor=True, xlabel="# of GPUs")


def ev_scaleout():
    md.d_scaleout()
    split(CAUGHT.pop("d_scaleout"), "ev_scaleout", 1, size=(TEXT_W, 2.8))


def ev_circuits128():
    md.circuits_big(128, original_baseline=True)
    fig = CAUGHT.pop("d_circuits_g128")
    for ax in panels_of(fig):
        # The ratios are stated in the text; keep the new narrow panels legible.
        for label in list(ax.texts):
            label.remove()
        lo, hi = ax.get_ylim()
        ax.set_yticks([y for y in (1, 10, 100, 1000, 10000) if lo <= y <= hi])
    for leg in fig.legends:
        for label in leg.get_texts():
            if label.get_text().startswith("N×:"):
                label.set_text("")
    split(fig, "ev_circuits128", 3, size=(PLACE[3][0], 2.3))


def ev_fixed_rows():
    """Clean, same-engine fixed-row ablation; archived contention logs are excluded by the reader."""
    import json
    data = json.load(open(os.path.join(os.path.dirname(HERE), "analysis", "paper_fixed_rows.json")))
    data = [r for r in data if r["gpus"] == 16]
    fig, axes = plt.subplots(2, 3)
    cases = [("ki", "Kicked Ising"), ("tfim", "TFIM heavy-hex"), ("tfimsq", "TFIM 11x11"),
             ("xxz", "XXZ 8x8"), ("qaoa", "QAOA"), ("hea", "Ansatz")]
    for ax, (c, title) in zip(axes.flat, cases):
        runs = {(r["mode"], r["cutoff"]): r for r in data if r["circuit"] == c}
        cuts = sorted({x for m, x in runs if runs[m, x]["exit"] == 0}, key=float, reverse=True)
        for mode, label, color, marker in [("re", "Recomputed Rows (SectorQ)", mf.SQ["ours"][0], "o"),
                                            ("fx", "Fixed Rows", mf.SQ["cupp"][0], "s")]:
            points = [(i, runs[mode, x]["wall"]) for i, x in enumerate(cuts)
                      if (mode, x) in runs and runs[mode, x]["exit"] == 0]
            ax.plot([p[0] for p in points], [p[1] for p in points], color=color, marker=marker,
                    linewidth=2, label=label)
        ticks = []
        for i, x in enumerate(cuts):
            a, b = runs.get(("re", x)), runs.get(("fx", x))
            terms = a["terms"] if a and a["exit"] == 0 else None
            ticks.append(x + ("\n(" + mf.count(terms).replace(" ", "") + ")" if terms else ""))
            if a and b and a["exit"] == b["exit"] == 0:
                ax.annotate(f'{b["wall"] / a["wall"]:.1f}', (i, b["wall"]), xytext=(0, 5),
                            textcoords="offset points", ha="center", fontsize=V)
        ax.set_xticks(range(len(cuts)), ticks)
        ax.set_yscale("log")
        ax.set_ylim(0.1, 4000)
        ax.set_xlabel("Cutoff (# of Terms)")
        ax.set_ylabel("Time (s)")
        ax.set_title(title)
        ax.grid(axis="y", linestyle=":", color="#c8ccd2")
    fig.legend(*axes.flat[0].get_legend_handles_labels())
    split(fig, "ev_fixed_rows", 3, size=(PLACE[3][0], 2.3))


def ev_large_time():
    """Worker-time averages and node-time averages, using the existing breakdown palette."""
    import json
    rows = json.load(open(os.path.join(os.path.dirname(HERE), "analysis", "paper_large_time.json")))
    by_id = {r["id"]: r for r in rows}
    fig, axes = plt.subplots(1, 3)
    for j, (c, title) in enumerate(mf.G3):
        ax = axes[j]
        for i, g in enumerate((128, 256, 512)):
            prefix = "i.ref" if c == "ki" else "n.eng"
            r = by_id[f"{prefix}.{c}.3e-6.g{g}"]
            values = dict(operations=r["operations_avg"], copies=r["h2d_avg"] + r["d2h_avg"],
                          partition=r["partition_avg"], idle=r["idle_avg"], waiting=r["sync"],
                          exchange=r["exchange"], other=r["residual"])
            assert values["other"] >= -0.01, r["id"]
            bottom = 0
            for label, key, color in mf.SIZE_PARTS:
                label = "Idle Workers" if key == "idle" else label
                ax.bar(i, values[key], bottom=bottom, width=0.65, color=color, label=label if i == 0 else None)
                bottom += values[key]
        ax.set_xticks(range(3), ["128", "256", "512"])
        ax.set_xlabel("# of GPUs")
        ax.set_ylabel("Time (s)")
        ax.set_title(title)
    fig.legend(*axes[0].get_legend_handles_labels())
    split(fig, "ev_large_time", 3)


def ev_runtime():
    """Activity shares at a fixed small case and larger workloads for larger allocations."""
    import json
    rows = json.load(open(os.path.join(os.path.dirname(HERE), "analysis", "paper_runtime_profiles.json")))
    fig, axes = plt.subplots(3, 3)
    for row, g in enumerate((16, 64, 256)):
        for column, (c, title) in enumerate(mf.G3):
            ax = axes[row, column]
            samples = sorted((r for r in rows if r["gpus"] == g and r["circuit"] == c), key=lambda r: r['point'])
            assert len(samples) == 3
            ticks = []
            for i, r in enumerate(samples):
                label = r['cutoff'] + ('*' if r['sector_cap'] > 256 else '')
                if r['status'] != 'complete':
                    ticks.append(label)
                    ax.text(i, 50, r['status'].title(), ha="center", va="center", color="#666666")
                    continue
                label += "\n(" + mf.count(r['terms']).replace(" ", "") + ")"
                ticks.append(label)
                bottom = 0
                for label, key, color in mf.SIZE_PARTS:
                    label = "Idle Workers" if key == "idle" else label
                    share = 100 * r[key] / r["wall"]
                    ax.bar(i, share, width=0.6, bottom=bottom,
                           color=color, label=label if i == 0 else None)
                    bottom += share
                assert abs(bottom - 100) < 1e-8, r["id"]
            ax.set_xlim(-0.5, 2.5)
            ax.set_xticks(range(3), ticks)
            ax._paper_require_all_xticks = True
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 50, 100])
            ax.set_xlabel("Cutoff (# of Terms)")
            ax.set_ylabel("Time Share (%)")
            ax.set_title(f"{title}, {g} GPUs")
    fig.legend(*axes[0, 0].get_legend_handles_labels())
    split(fig, "ev_runtime", 3)


def ev_resector():
    """All 30 candidate pairs and six triplicated overlap settings from one allocation.

    Candidate points are single matched measurements. Overlap markers are ratios
    of mean wall times; whiskers span the three individual paired ratios, not CIs.
    Each imbalance is the maximum, over windows, of largest / mean occupied sector.
    """
    import json
    from matplotlib.lines import Line2D

    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "analysis"))
    from paper_n2 import compare

    path = os.path.join(os.path.dirname(HERE), "analysis", "paper_n2.59842747.json")
    with open(path) as source:
        data = json.load(source)
    rows = data["rows"]
    pairs = compare(rows)
    assert len(rows) == 96 and len(pairs) == 48
    assert all(r["status"] == "complete" and r["allocation"] == "59842747"
               and r["gpus"] == 16 for r in rows)
    assert all(p["status"] == "complete" for p in pairs)
    runs = {(r["mode"], r["circuit"], r["cutoff"], r["repetition"]): r for r in rows}
    cases = [("ki", "Kicked Ising"), ("tfim", "TFIM HH"), ("noise", "Noisy KI"),
             ("tfimsq", r"TFIM $11\times11$"), ("xxz", r"XXZ $8\times8$"),
             ("qaoa", "QAOA"), ("hea", "Ansatz")]
    styles = [("1e-4", r"$10^{-4}$", "#0072B2", "o"),
              ("3e-5", r"$3\times10^{-5}$", "#E69F00", "s"),
              ("1e-5", r"$10^{-5}$", "#009E73", "^"),
              ("3e-6", r"$3\times10^{-6}$", "#CC79A7", "D"),
              ("1e-6", r"$10^{-6}$", "#6E6E6E", "v")]
    fig, axes = plt.subplots(1, 3)
    plotted = 0
    for i, (circuit, _) in enumerate(cases):
        for j, (cutoff, _, color, marker) in enumerate(styles):
            a = runs.get(("re", circuit, cutoff, 0))
            b = runs.get(("rb", circuit, cutoff, 0))
            if a is None or b is None:
                continue
            y = i + (j - 2) * 0.15
            values = [b["max_imbalance_nonempty"] / a["max_imbalance_nonempty"],
                      b["wall"] / a["wall"]]
            for ax, value in zip(axes[:2], values):
                ax.plot(value, y, marker=marker, color=color, markersize=7,
                        linestyle="none", markeredgewidth=0.5)
            plotted += 1
    assert plotted == 30
    for i, (circuit, _) in enumerate(cases[:3]):
        for offset, j in zip((-0.14, 0.14), (0, 2)):
            cutoff, _, color, marker = styles[j]
            on = [runs[("ov1", circuit, cutoff, rep)]["wall"] for rep in (1, 2, 3)]
            off = [runs[("ov0", circuit, cutoff, rep)]["wall"] for rep in (1, 2, 3)]
            ratio = sum(off) / sum(on)
            ratios = [b / a for a, b in zip(on, off)]
            axes[2].errorbar(ratio, i + offset,
                            xerr=[[ratio - min(ratios)], [max(ratios) - ratio]],
                            fmt=marker, color=color, markersize=7, capsize=2.5,
                            elinewidth=1, markeredgewidth=0.5)
    settings = [((0.94, 1.76), [1.0, 1.2, 1.4, 1.6], "Imbalance ratio (1 / 8)"),
                ((0.62, 1.10), [0.7, 0.8, 0.9, 1.0, 1.1], "Time ratio (1 / 8)"),
                ((0.97, 1.53), [1.0, 1.2, 1.4], "Time ratio (off / on)")]
    for k, (ax, (limits, ticks, xlabel)) in enumerate(zip(axes, settings)):
        labels = cases if k < 2 else cases[:3]
        ax.set_yticks(range(len(labels)), [label for _, label in labels])
        ax.set_ylim(len(labels) - 0.45, -0.55)
        ax.set_xlim(*limits)
        ax.set_xticks(ticks)
        ax.set_xlabel(xlabel)
        ax.axvline(1, color="#52575E", linestyle="--", linewidth=1)
        ax.grid(axis="y", color="#E1E4E8", linewidth=0.6)
    handles = [Line2D([], [], marker=m, color=c, linestyle="none", markersize=7)
               for _, _, c, m in styles]
    labels = [label for _, label, _, _ in styles]
    labels[0] = "Cutoff " + labels[0]
    fig.legend(handles, labels)
    split(fig, "ev_resector", 3, size=(PLACE[3][0], 2.55))


FIGURES = [bg_anatomy, bg_accuracy, bg_windows, ev_single, ev_circuits, ev_cost, ev_prs, ev_size, ev_accuracy, ev_breakdown, ev_variability,
           ev_staircase, ev_knobs, ev_weak, ev_scaleout, ev_fixed_rows, ev_large_time, ev_circuits128, ev_runtime, ev_resector]

if __name__ == "__main__":
    want = [a for a in ARGS if not a.startswith("-")]
    for f in FIGURES:
        if not want or f.__name__ in want:
            f()
