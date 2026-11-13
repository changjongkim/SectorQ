#!/usr/bin/env python3
"""Numbers of the Sensitivity Analysis subsection (review experiments of 2026-11-12, jobs/y_items.py, queue_y, logs/y,
bin/pp_engine_k3, 16 GPUs on four nodes unless stated).

  env/bin/python3 analysis/paper_review_y.py [--latex]        (writes analysis/paper_review_y.json)

E1  fixed rows on the same engine: --fixed-rows 1 (rows of the first partitioned window, "fx") and --fixed-rows 2
    (rows conserved by the most later rotations at the sample balance of fx, "bf") against recomputed rows ("re").
E2  reserved rows of the window rule (--ann-min n/4, n/2, 3n/2 against n).
E3  retained terms at window boundaries, one GPU with every merge global against 16 GPUs (analysis/retained_terms.py,
    which writes analysis/retained_terms.json; this script reads that file).
E4  predicted (pred_largest_share of the trace, from the sample) against measured largest sector of each window
    (largest_sector_in / terms_in) in the recomputed-row runs of E1 and the noisy kicked Ising runs, windows with at
    least 10^6 input terms and more than one sector.
E5  random k-local circuits (--circuit rpauli --n 100 --layers 20 --amp 0.1 --seed 1), SectorQ against cuPauliProp.
"""
import glob
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DONE = os.path.join(ROOT, "queue_y", "done")
LOGS = os.path.join(ROOT, "logs", "y")
CIRCUITS = ("ki", "tfim", "tfimsq", "xxz", "qaoa", "hea")
CUTS = ("1e-4", "3e-5", "1e-5", "3e-6", "1e-6")


def rec(i):
    try:
        r = json.load(open(os.path.join(DONE, i)))
    except (OSError, ValueError):
        return None
    path = os.path.join(LOGS, i + ".out")
    t = open(path, errors="replace").read() if os.path.exists(path) else ""

    def get(pat, f=float):
        m = re.search(pat, t, re.M)
        return f(m.group(1)) if m else None
    r.update(windows=get(r"^windows=(\d+)", int), boundary=get(r"max_terms_at_window_boundary=(\d+)", int),
             host_splits=get(r"splits_on_host=(\d+)", int), gpu_splits=get(r"splits_on_gpu=(\d+)", int),
             exchange=get(r"between nodes: [\d.]+ GB sent, ([\d.]+) s per node", float),
             traffic=get(r"between nodes: ([\d.]+) GB sent", float), rows=get(r"smallest row space (\d+) of", int),
             stop="no row divides an overflowing piece" in t, text=t)
    return r


def ok(r):
    return r is not None and r.get("status") == "ok"


def gmean(a):
    return math.exp(sum(map(math.log, a)) / len(a))


def e1():
    out = dict(pairs={}, windows={}, boundary={})
    for pol in ("fx", "bf"):
        q, diffs, failed = [], [], []
        for c in CIRCUITS:
            for x in CUTS:
                re_, f = rec(f"e1.re.{c}.{x}"), rec(f"e1.{pol}.{c}.{x}")
                if not ok(re_) or f is None:
                    continue
                if ok(f):
                    q.append(f["wall"] / re_["wall"])
                    diffs.append(abs(f["expectation"] - re_["expectation"]))
                else:
                    failed.append(dict(circuit=c, cutoff=x, status=f["status"], recomputed_s=re_["wall"]))
        out["pairs"][pol] = dict(n=len(q), lo=min(q), hi=max(q), gmean=gmean(q), max_exp_diff=max(diffs), failed=failed)
        ws = [rec(i)["windows"] for i in (os.path.basename(p) for p in glob.glob(os.path.join(DONE, f"e1.{pol}.*"))) if ok(rec(i))]
        out["windows"][pol] = [min(ws), max(ws)]
    ws = [rec(os.path.basename(p))["windows"] for p in glob.glob(os.path.join(DONE, "e1.re.*")) if ok(rec(os.path.basename(p)))]
    out["windows"]["re"] = [min(ws), max(ws)]
    for pol in ("re", "fx", "bf"):
        r = rec(f"e1.{pol}.xxz.1e-5")
        out["boundary"][pol] = dict(xxz_1e5=r["boundary"], traffic_gb=r["traffic"])
    m = re.search(r"(\d+) of (\d+) later branching rotations conserve them", rec("e1.bf.xxz.1e-5")["text"])
    out["bf_xxz_conserving"] = [int(m.group(1)), int(m.group(2))] if m else None
    conserve = []
    for c in CIRCUITS:
        r = rec(f"e1.bf.{c}.1e-5")
        m = re.search(r"(\d+) of (\d+) later branching rotations conserve them", r["text"]) if r else None
        if m:
            conserve.append((c, int(m.group(1)), int(m.group(2))))
    out["bf_conserving"] = conserve
    out["bf_conserving_share"] = [min(a / b for _, a, b in conserve), max(a / b for _, a, b in conserve)]
    return out


def e2():
    rows, cells = [], []
    for c in ("ki", "tfim", "xxz", "qaoa"):
        for x in ("1e-5", "3e-6"):
            row = dict(circuit=c, cutoff=x)
            for a in ("0.25", "0.5", "1", "1.5"):
                r = rec(f"e1.re.{c}.{x}") if a == "1" else rec(f"e2.ann{a}.{c}.{x}")
                row[a] = None if r is None else dict(status=r["status"], wall=r.get("wall"), windows=r["windows"], rows=r["rows"],
                                                     host_splits=r["host_splits"], exchange=r["exchange"], stop=r["stop"])
            rows.append(row)
    fastest = sum(1 for r in rows if all(r[a] is None or r[a]["status"] != "ok" or r[a]["wall"] >= r["1"]["wall"] for a in ("0.25", "0.5", "1.5")))
    stopped = [(r["circuit"], r["cutoff"], a) for r in rows for a in ("0.25", "0.5") if r[a] and r[a]["stop"]]
    skipped = [(r["circuit"], r["cutoff"], a) for r in rows for a in ("0.25", "0.5") if r[a] and r[a]["status"] == "skipped_after_failure"]
    more = [r["1.5"]["wall"] / r["1"]["wall"] - 1 for r in rows]
    exch = [r["1.5"]["exchange"] / r["1"]["exchange"] for r in rows]
    beaten = [(r["circuit"], r["cutoff"], a, r["1"]["wall"] / r[a]["wall"]) for r in rows for a in ("0.25", "0.5", "1.5")
              if r[a] and r[a]["status"] == "ok" and r[a]["wall"] < r["1"]["wall"]]
    xs = next(r for r in rows if r["circuit"] == "xxz" and r["cutoff"] == "3e-6")
    return dict(rows=rows, fastest=fastest, settings=len(rows), stopped=stopped, skipped=skipped,
                fewer_rows_runs=sum(1 for r in rows for a in ("0.25", "0.5") if r[a]),
                fewer_rows_ok=sum(1 for r in rows for a in ("0.25", "0.5") if r[a] and r[a]["status"] == "ok"),
                more_rows_time=[min(more), max(more)], more_rows_exchange=[min(exch), max(exch)], beaten=beaten,
                xxz_3e6=dict(n=xs["1"], quarter=xs["0.25"], half=xs["0.5"]))


def e3():
    path = os.path.join(ROOT, "analysis", "retained_terms.json")
    pairs = json.load(open(path))
    terms = [p["terms"][0] for p in pairs]
    return dict(pairs=len(pairs), circuits=sorted({p["circuit"] for p in pairs}), identical=sum(p["only_one_gpu"] + p["only_16_gpus"] == 0 for p in pairs),
                max_abs_diff=max(p["max_abs_diff"] for p in pairs), duplicates=sum(sum(p["duplicates"]) for p in pairs),
                terms=[min(terms), max(terms)], cutoff=pairs[0]["cutoff"] if pairs else None)


def e4():
    pat = re.compile(r"trace window=(\d+) .*?sectors=(\d+) largest_sector_in=(\d+) terms_in=(\d+) .*pred_largest_share=([-\d.]+)")
    per, allw = {}, []
    files = sorted(glob.glob(os.path.join(LOGS, "e1.re.*.out")) + glob.glob(os.path.join(LOGS, "e4.re.*.out")))
    for f in files:
        name = os.path.basename(f)[:-4]
        if not ok(rec(name)):
            continue
        for m in pat.finditer(open(f).read()):
            _, sec, lg, tin, pred = m.groups()
            sec, tin, pred = int(sec), int(tin), float(pred)
            if pred < 0 or tin < 10 ** 6 or sec < 2:
                continue
            act = int(lg) / tin
            w = dict(run=name, actual=act * sec, predicted=pred * sec, ratio=act / pred)
            allw.append(w)
            per.setdefault(name, []).append(w)
    xxz = [w for w in allw if ".xxz." in w["run"]]
    other = [w for w in allw if ".xxz." not in w["run"]]
    return dict(runs=len(per), windows=len(allw), actual_other=[min(w["actual"] for w in other), max(w["actual"] for w in other)],
                actual_xxz=[min(w["actual"] for w in xxz), max(w["actual"] for w in xxz)],
                actual_xxz_1e5=max(w["actual"] for w in xxz if w["run"].endswith(".1e-5")),
                ratio=[min(w["ratio"] for w in allw), max(w["ratio"] for w in allw)],
                under=sum(1 for w in allw if w["ratio"] > 1), at_or_below=sum(1 for w in allw if w["ratio"] <= 1), under_max=max(w["ratio"] for w in allw) - 1,
                over_max=1 / min(w["ratio"] for w in allw),
                median_ratio=sorted(w["ratio"] for w in allw)[len(allw) // 2])


def e5():
    rows, ratios, both = [], [], {}
    for k in ("k2", "k4", "k8"):
        for x in ("1e-3", "3e-4", "1e-4", "3e-5", "1e-5"):
            e, c = rec(f"e5.eng.{k}.{x}"), rec(f"e5.cupp.{k}.{x}")
            row = dict(k=int(k[1:]), cutoff=x, sectorq=e["wall"] if ok(e) else e["status"] if e else None,
                       cupp=c["wall"] if ok(c) else c["status"] if c else None, windows=e["windows"] if e else None)
            if ok(e) and ok(c):
                row["ratio"] = c["wall"] / e["wall"]
                row["exp_diff"] = abs(c["expectation"] - e["expectation"])
                ratios.append(row["ratio"])
                both[k] = row
            rows.append(row)
    cupp_1e3 = [r["cupp"] for r in rows if r["cutoff"] == "1e-3"]
    return dict(rows=rows, ratio=[min(ratios), max(ratios)], lowest_shared={k: (v["cutoff"], v["ratio"]) for k, v in both.items()},
                max_exp_diff=max(r["exp_diff"] for r in rows if "exp_diff" in r), windows=sorted({r["windows"] for r in rows if r["windows"]}),
                cupp_1e3=[min(cupp_1e3), max(cupp_1e3)],
                cupp_oom=[(r["k"], r["cutoff"], r["sectorq"]) for r in rows if r["cupp"] == "oom"])


def latex(out):
    name = {"ki": "Kicked Ising", "tfim": "TFIM heavy-hex", "xxz": "XXZ $8\\times8$", "qaoa": "QAOA"}
    cut = {"1e-5": "$10^{-5}$", "3e-6": "$3\\times10^{-6}$"}
    print("% Table: reserved rows (E2)")
    for r in out["e2"]["rows"]:
        cells = []
        for a in ("0.25", "0.5", "1", "1.5"):
            v = r[a]
            if v is None:
                cells.append("--")
            elif v["status"] == "ok":
                cells.append(f"{v['wall']:.1f}")
            elif v["stop"]:
                cells.append("stop")
            else:
                cells.append("--")
        print(f"{name[r['circuit']]} & {cut[r['cutoff']]} & " + " & ".join(cells) + " \\\\")
    print("% Table: random k-local circuits (E5), one row per cutoff, SectorQ and cuPauliProp for k = 2, 4, 8")
    tex = {"1e-3": "$10^{-3}$", "3e-4": "$3\\times10^{-4}$", "1e-4": "$10^{-4}$", "3e-5": "$3\\times10^{-5}$", "1e-5": "$10^{-5}$"}
    val = lambda v: f"{v:.2f}" if isinstance(v, float) else {"oom": "OOM"}.get(v, str(v))
    for x in tex:
        cells = []
        for k in (2, 4, 8):
            r = next(r for r in out["e5"]["rows"] if r["k"] == k and r["cutoff"] == x)
            cells += [val(r["sectorq"]), val(r["cupp"])]
        print(f"{tex[x]} & " + " & ".join(cells) + " \\\\")


def main():
    out = dict(e1=e1(), e2=e2(), e3=e3(), e4=e4(), e5=e5())
    json.dump(out, open(os.path.join(ROOT, "analysis", "paper_review_y.json"), "w"), indent=1, default=str)
    e = out["e1"]
    for pol in ("fx", "bf"):
        p = e["pairs"][pol]
        print(f"E1 {pol}: {p['lo']:.1f}-{p['hi']:.1f}x slower over {p['n']} pairs, geometric mean {p['gmean']:.2f}x, max |dE| {p['max_exp_diff']:.1e}; "
              f"failed {p['failed']}; windows {e['windows'][pol]}")
    print(f"E1 re windows {e['windows']['re']}; xxz 1e-5 boundary/traffic {e['boundary']}; bf conserving {e['bf_conserving']} "
          f"share {e['bf_conserving_share'][0]:.2f}-{e['bf_conserving_share'][1]:.2f}")
    e = out["e2"]
    print(f"E2: n fastest in {e['fastest']} of {e['settings']}; beaten {e['beaten']}; fewer rows: {e['fewer_rows_ok']} of {e['fewer_rows_runs']} ok, "
          f"stopped {e['stopped']}, skipped {e['skipped']}; 3n/2 time +{100 * e['more_rows_time'][0]:.0f}-{100 * e['more_rows_time'][1]:.0f}%, "
          f"exchange {e['more_rows_exchange'][0]:.2f}-{e['more_rows_exchange'][1]:.2f}x")
    x = e["xxz_3e6"]
    print(f"E2 xxz 3e-6: n {x['n']['wall']} s, {x['n']['host_splits']} host splits, {x['n']['rows']} rows; n/2 {x['half']['wall']} s, "
          f"{x['half']['host_splits']} host splits, {x['half']['rows']} rows; n/4 {x['quarter']['wall']} s, {x['quarter']['host_splits']} host splits, "
          f"{x['quarter']['rows']} rows, {x['quarter']['wall'] / x['n']['wall']:.1f}x")
    print("E3:", out["e3"])
    e = out["e4"]
    print(f"E4: {e['windows']} windows of {e['runs']} runs; actual/mean {e['actual_other'][0]:.2f}-{e['actual_other'][1]:.2f} (not XXZ), "
          f"XXZ {e['actual_xxz'][0]:.2f}-{e['actual_xxz'][1]:.2f} (1e-5 max {e['actual_xxz_1e5']:.2f}); actual/predicted {e['ratio'][0]:.2f}-{e['ratio'][1]:.2f}, "
          f"median {e['median_ratio']:.2f}; {e['at_or_below']} windows at or below the prediction, {e['under']} windows above the prediction, by at most {100 * e['under_max']:.1f}%; overestimate up to {e['over_max']:.2f}x")
    e = out["e5"]
    print(f"E5: {e['ratio'][0]:.1f}-{e['ratio'][1]:.1f}x; lowest shared {e['lowest_shared']}; windows {e['windows']}; max |dE| {e['max_exp_diff']}; "
          f"cuPP at 1e-3 {e['cupp_1e3'][0]:.1f}-{e['cupp_1e3'][1]:.1f} s; cuPP oom {e['cupp_oom']}")
    if "--latex" in sys.argv:
        latex(out)


if __name__ == "__main__":
    main()
