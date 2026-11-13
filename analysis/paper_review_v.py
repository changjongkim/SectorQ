#!/usr/bin/env python3
"""Numbers of the second review round (2026-11-13) for the paper.

  env/bin/python3 analysis/paper_review_v.py          (prints; writes analysis/paper_review_v.json)

E9  overlap of row selection with the running window (logs/b/ov1.* with overlap, ov0.* without; three repetitions,
    16 GPUs): the selection time on node 0 ("planning=") and the time the 16 workers waited for the rows (summed),
    so the hidden share of the selection is 1 - (wait / 16) / planning.
E7  rows planned before the run (queue_v, v.e7.off<F>.*) against rows chosen during the run (v.e7.on.*).
E8  random k-local circuits: SectorQ under truncation after every operation (v.e8.sq1.*) against paulistrings-rs on
    16 GPUs (v.e8.prs.*).
"""
import glob
import json
import math
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def grab(path, pat):
    m = re.search(pat, open(path, errors="replace").read(), re.M)
    return float(m.group(1)) if m else None


def e9():
    rows = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "logs/b/ov[01].*.out"))):
        tag, c, g, cut, rep = os.path.basename(f)[:-4].split(".", 4)[0:5] if False else (None,) * 5
        name = os.path.basename(f)[:-4]
        parts = name.split(".")
        mode, circ = parts[0], parts[1]
        cut = ".".join(parts[3:-1])
        rows.setdefault((circ, cut), {}).setdefault(mode, []).append(dict(
            wall=grab(f, r"^wall=([0-9.]+)"), plan=grab(f, r"planning=([0-9.]+)"),
            wait=grab(f, r"workers waited ([0-9.]+) s")))
    out = []
    for (c, cut), d in sorted(rows.items()):
        on, off = d.get("ov1", []), d.get("ov0", [])
        mean = lambda xs: sum(xs) / len(xs) if xs else None
        plan = mean([r["plan"] for r in on])
        wait = mean([r["wait"] for r in on]) / 16.0
        rec = dict(circuit=c, cutoff=cut, plan_s=round(plan, 3), wait_per_worker_s=round(wait, 4),
                   hidden=round(1 - wait / plan, 3), wall_on=round(mean([r["wall"] for r in on]), 3),
                   wall_off=round(mean([r["wall"] for r in off]), 3))
        rec["off_over_on"] = round(rec["wall_off"] / rec["wall_on"], 3)
        out.append(rec)
    print("E9 overlap (16 GPUs, mean of three runs)")
    for r in out:
        print(f"  {r['circuit']:6s} {r['cutoff']:5s} selection {r['plan_s']:.3f} s, wait per worker {r['wait_per_worker_s']:.4f} s, "
              f"hidden {100 * r['hidden']:.1f}%, wall on {r['wall_on']:.3f} off {r['wall_off']:.3f} ({r['off_over_on']:.3f}x)")
    hid = [r["hidden"] for r in out]
    print(f"  hidden share {100 * min(hid):.1f}-{100 * max(hid):.1f}%, largest wait per worker "
          f"{max(r['wait_per_worker_s'] for r in out):.3f} s, selection {min(r['plan_s'] for r in out):.2f}-{max(r['plan_s'] for r in out):.2f} s")
    return out


def done(prefix):
    recs = {}
    for f in glob.glob(os.path.join(ROOT, "queue_v/done", prefix + "*")):
        try:
            recs[os.path.basename(f)] = json.load(open(f))
        except ValueError:
            pass
    return recs


def e7():
    recs = done("v.e7.")
    if not recs:
        print("E7: no runs yet")
        return []
    out = []
    for rid, r in sorted(recs.items()):
        _, _, var, c, cut = rid.split(".", 4)
        out.append(dict(variant=var, circuit=c, cutoff=cut, status=r.get("status"), wall=r.get("wall"),
                        expectation=r.get("expectation")))
    on = {(o["circuit"], o["cutoff"]): o for o in out if o["variant"] == "on" and o["status"] == "ok"}
    print("E7 rows before the run (off<F>) against rows during the run (on), 16 GPUs")
    for o in out:
        ref = on.get((o["circuit"], o["cutoff"]))
        ratio = o["wall"] / ref["wall"] if ref and o["wall"] and o["status"] == "ok" else None
        same = (ref is not None and o["expectation"] is not None and ref["expectation"] == o["expectation"])
        o["ratio"] = round(ratio, 3) if ratio else None
        print(f"  {o['variant']:6s} {o['circuit']:6s} {o['cutoff']:5s} {o['status']:>8s} wall {o['wall']} "
              f"ratio {o['ratio']} same_value {same}")
    for var in ("off16", "off64", "off256"):
        rs = [o["ratio"] for o in out if o["variant"] == var and o["ratio"]]
        if rs:
            g = math.exp(sum(math.log(x) for x in rs) / len(rs))
            print(f"  {var}: {min(rs):.2f}-{max(rs):.2f}x of on, geometric mean {g:.2f}x over {len(rs)} pairs; "
                  f"not ok: {[o['circuit'] + ' ' + o['cutoff'] + ' ' + str(o['status']) for o in out if o['variant'] == var and o['status'] != 'ok']}")
    return out


def e8():
    recs = done("v.e8.")
    if not recs:
        print("E8: no runs yet")
        return []
    out = []
    for rid, r in sorted(recs.items()):
        _, _, sysname, k, cut = rid.split(".", 4)
        wall = r.get("wall") if sysname == "sq1" else r.get("wall")
        out.append(dict(system=sysname, k=k, cutoff=cut, status=r.get("status"), wall=wall,
                        expectation=r.get("expectation")))
    print("E8 random k-local, truncation after every operation, 16 GPUs: SectorQ (sq1) and paulistrings-rs (prs)")
    by = {(o["system"], o["k"], o["cutoff"]): o for o in out}
    for k in ("k2", "k4", "k8"):
        for cut in ("1e-3", "3e-4", "1e-4", "3e-5", "1e-5"):
            s, p = by.get(("sq1", k, cut)), by.get(("prs", k, cut))
            if not s and not p:
                continue
            ratio = (p["wall"] / s["wall"]) if (s and p and s["status"] == "ok" and p["status"] == "ok") else None
            diff = (abs(p["expectation"] - s["expectation"]) if (s and p and s.get("expectation") is not None
                                                                   and p.get("expectation") is not None) else None)
            print(f"  {k} {cut:5s} SectorQ {s and s['status']} {s and s['wall']} | paulistrings-rs {p and p['status']} "
                  f"{p and p['wall']} | ratio {ratio and round(ratio, 2)} | |diff| {diff}")
    return out


if __name__ == "__main__":
    res = dict(e9=e9(), e7=e7(), e8=e8())
    json.dump(res, open(os.path.join(ROOT, "analysis/paper_review_v.json"), "w"), indent=1)
