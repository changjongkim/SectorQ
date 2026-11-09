#!/usr/bin/env python3
"""Runs that fill the data points the evaluation still lacks (plan: handoff/2026-11-08_eval_scaleqsim_plan.md, P1-P3;
512 GPUs keep the SectorQ scale-out runs only, nothing more is run there).

  env/bin/python3 jobs/u_items.py            # print the runs by queue and node count
  env/bin/python3 jobs/u_items.py --write    # write queue_g/todo_w.jsonl and queue_s/todo_u_fill.jsonl

queue_g (16 GPUs = 4 nodes; jobs/w_fill16.sh):
  P1-a  SectorQ on the four smaller circuits down to its first failure (their ladders ended on a success), and
        MonoProp on the ansatz (its own truncation rule; its ladder also ended on a success).
  P1-b  the cutoffs between the half-decade steps of those circuits (5e-5, 2e-5, 5e-6, 2e-6, 5e-7) for every
        implementation on 16 GPUs / 4 nodes; for each implementation only the steps above its first failure.
        One extra ladder per implementation (<ladder>.mid), from the largest cutoff down.
  P1-d  the 1,024-qubit TFIM at cutoff 1e-4 on 16 GPUs with two hours (it reached the 20-minute limit).
queue_s (32-256 GPUs; jobs/s_scale_w.sh):
  P2    TFIM 11x11, XXZ 8x8, QAOA and the ansatz on 128 GPUs (32 nodes): SectorQ, cuPauliProp (original circuit
        where it has Clifford gates), deferred deduplication, the ORQA reproduction and MonoProp, each down to its
        first failure.
  P3    the ladders of cuPauliProp and deferred deduplication that ended on a success on 64-256 GPUs, continued.
        (The nine open ORQA runs on 64 nodes are already in todo_t_fill.jsonl.)
New SectorQ runs use bin/pp_engine_i (exchange in rounds), as every run on more than one node since 2026-10-12.
"""
import copy
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIMIT = 2400
ENGINE = "bin/pp_engine_i"
CIRC = {
    "tfimsq": "--circuit tfim_sq --lx 11 --ly 11 --theta 0.2 --phi -0.3 --steps 10",
    "xxz": "--circuit xxz --lx 8 --ly 8 --jxy 0.1 --jz 0.15 --steps 6",
    "qaoa": "--circuit qaoa --n 100 --layers 4",
    "hea": "--circuit hea --n 100 --layers 16 --amp 0.4",
}
CLIFFORD = ("qaoa", "hea")                 # circuits whose original form differs from the list the Clifford frame leaves
SMALL = tuple(CIRC)


def lists(queue):
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, queue, "todo*.jsonl"))):
        for line in open(path):
            if line.strip():
                out.append(json.loads(line))
    return out


EXIST = {q: lists(q) for q in ("queue_g", "queue_r", "queue_o", "queue_s")}


def template(queue, item_id):
    hits = [i for i in EXIST[queue] if i["id"] == item_id]
    if not hits:
        raise KeyError(f"{queue}: no item {item_id}")
    return copy.deepcopy(hits[-1])


def status(queue, item_id):
    try:
        return json.load(open(os.path.join(ROOT, queue, "done", item_id))).get("status")
    except (OSError, ValueError):
        return None


def at(item, item_id, coef, ladder=None, **more):
    """A copy of an existing run at another cutoff."""
    i = copy.deepcopy(item)
    i["id"], i["coef"] = item_id, float(coef)
    if "args" in i:
        i["args"] = re.sub(r"--coef \S+", f"--coef {coef}", i["args"])
    if ladder is not None:
        i["ladder"] = ladder
    i.pop("est", None) if i.get("ladder") else i.setdefault("est", 600)
    i.update(more)
    return i


G, S = [], []                               # runs for queue_g and queue_s

# ---------------------------------------------------------------- queue_g: 16 GPUs / 4 nodes
# P1-a: the ladders that ended on a success, continued
DEEPER = {"tfimsq": ("2e-6", "1e-6", "7e-7", "5e-7"), "xxz": ("2e-6", "1e-6", "7e-7", "5e-7"),
          "qaoa": ("5e-7", "3e-7", "2e-7", "1e-7", "7e-8", "5e-8"), "hea": ("2e-7", "1e-7", "7e-8", "5e-8", "3e-8")}
LAST = {"tfimsq": "3e-6", "xxz": "3e-6", "qaoa": "1e-6", "hea": "3e-7"}
for c, xs in DEEPER.items():
    t = template("queue_g", f"eng.{c}.{LAST[c]}.g16")
    assert status("queue_g", t["id"]) == "ok"
    for x in xs:
        G.append(at(t, f"eng.{c}.{x}.g16", x, bin=ENGINE))
t = template("queue_g", "monoprop.hea.3e-7.n4")
assert status("queue_g", t["id"]) == "ok"
for x in ("2e-7", "1e-7", "7e-8", "5e-8"):
    G.append(at(t, f"monoprop.hea.{x}.n4", x))

# P1-b: cutoffs between the half-decade steps
MID = {"tfimsq": ("5e-5", "2e-5", "5e-6"), "xxz": ("5e-5", "2e-5", "5e-6"),
       "qaoa": ("5e-5", "2e-5", "5e-6", "2e-6"), "hea": ("5e-5", "2e-5", "5e-6", "2e-6", "5e-7")}
STEPS = {"tfimsq": ("1e-3", "3e-4", "1e-4", "3e-5", "1e-5", "3e-6"), "xxz": ("1e-3", "3e-4", "1e-4", "3e-5", "1e-5", "3e-6"),
         "qaoa": ("1e-3", "3e-4", "1e-4", "3e-5", "1e-5", "3e-6", "1e-6"), "hea": ("1e-4", "3e-5", "1e-5", "3e-6", "1e-6", "3e-7")}


def first_failure(queues, fmt, c):
    """The largest cutoff of the existing ladder whose run did not succeed (None: every step succeeded)."""
    for x in STEPS[c]:
        s = None
        for q in queues:                    # the later queue replaces the earlier one (campaign o over campaign g)
            s = status(q, fmt.format(c=c, x=x)) or s
        if s is not None and s != "ok":
            return float(x)
    return None


IMPL16 = [  # tag, id format, queues whose records count, template id format and its queue
    ("eng", "eng.{c}.{x}.g16", ("queue_g", "queue_r"), ("queue_g", "eng.{c}.1e-4.g16")),
    ("cupp", "cupp.{c}.{x}.g16", ("queue_g", "queue_o"), None),
    ("defer", "defer.{c}.{x}.g16", ("queue_g",), ("queue_g", "defer.{c}.1e-4.g16")),
    ("orqa", "orqa.{c}.{x}.n4", ("queue_g",), ("queue_g", "orqa.{c}.1e-4.n4")),
    ("monoprop", "monoprop.{c}.{x}.n4", ("queue_g",), ("queue_g", "monoprop.{c}.1e-4.n4")),
]
for c in SMALL:
    for tag, fmt, queues, tpl in IMPL16:
        if tag == "cupp":                   # the default distributed mode on the original circuit (campaign o's input)
            tpl = ("queue_o", "cupp.{c}.1e-4.g16") if c in CLIFFORD else ("queue_g", "cupp.{c}.1e-4.g16")
        t = template(tpl[0], tpl[1].format(c=c))
        stop = first_failure(queues, fmt, c)
        xs = [x for x in MID[c] if stop is None or float(x) > stop]
        for x in xs:
            more = {"bin": ENGINE} if tag == "eng" else {}
            G.append(at(t, fmt.format(c=c, x=x), x, ladder=f"{t['ladder']}.mid", **more))

# P1-d: 1,024 qubits at 1e-4 with two hours (the run of campaign r reached its 20-minute limit)
t = template("queue_r", "m.eng.L32.1e-4.g16")
i = at(t, "m.eng.L32.1e-4.g16.t2h", "1e-4", limit=7200, est=7200)
i.pop("ladder", None)
G.append(i)

# ---------------------------------------------------------------- queue_s: 64-256 GPUs
LADDER128 = {
    "tfimsq": ("1e-4", "3e-5", "1e-5", "5e-6", "3e-6", "2e-6", "1e-6", "7e-7", "5e-7", "3e-7", "2e-7"),
    "xxz": ("1e-4", "3e-5", "1e-5", "5e-6", "3e-6", "2e-6", "1e-6", "7e-7", "5e-7", "3e-7", "2e-7"),
    "qaoa": ("1e-4", "3e-5", "1e-5", "5e-6", "3e-6", "2e-6", "1e-6", "5e-7", "3e-7", "2e-7", "1e-7", "7e-8", "5e-8"),
    "hea": ("1e-4", "3e-5", "1e-5", "5e-6", "3e-6", "2e-6", "1e-6", "5e-7", "3e-7", "2e-7", "1e-7", "7e-8", "5e-8", "3e-8"),
}
g, n = 128, 32
for c in SMALL:                             # P2, short ladders first: cuPauliProp's two modes, SectorQ, the CPU implementations
    xs = LADDER128[c]
    pre = "o." if c in CLIFFORD else ""
    for x in xs:
        S.append(dict(id=f"{pre}cupp.{c}.{x}.g{g}", nodes=n, kind="cupp", limit=LIMIT, coef=float(x), per=4, plugin="orig",
                      args=f"{CIRC[c]} --coef {x} --vram-frac 0.9" + (" --frame 0" if pre else ""), ladder=f"{pre}cupp.{c}.g{g}"))
    for x in xs:
        S.append(dict(id=f"defer.{c}.{x}.g{g}", nodes=n, kind="cupp", limit=LIMIT, coef=float(x), per=4, plugin="orig",
                      args=f"{CIRC[c]} --coef {x} --vram-frac 0.9 --cupp-tuned 1", ladder=f"defer.{c}.g{g}"))
    for x in xs:
        S.append(dict(id=f"eng.{c}.{x}.g{g}", nodes=n, kind="eng", bin=ENGINE, per=4, limit=LIMIT, coef=float(x),
                      args=f"{CIRC[c]} --coef {x} --target 4000000 --trace 1 --vram-frac 0.9", ladder=f"eng.{c}.g{g}"))
    for b, policy, rpn, threads in (("monoprop", "native", 8, 8), ("orqa", "matched", 64, 1)):
        for x in xs:
            S.append(dict(id=f"{b}.{c}.{x}.n{n}", nodes=n, kind="ext", limit=LIMIT, coef=float(x), backend=b, case=c,
                          policy=policy, rpn=rpn, threads=threads, predict=True, ladder=f"x.{b}.{c}.n{n}"))

# P3: ladders that ended on a success, continued two steps
for item_id in ("cupp.tfim.3e-6.g256", "defer.tfim.3e-6.g256", "o.cupp.noise.3e-6.g64", "o.cupp.noise.3e-6.g128",
                "o.cupp.noise.3e-6.g256", "defer.noise.3e-6.g128", "defer.noise.3e-6.g256"):
    t = template("queue_s", item_id)
    assert status("queue_s", item_id) == "ok", item_id
    for x in ("2e-6", "1e-6"):
        S.append(at(t, item_id.replace(".3e-6.", f".{x}."), x))

if __name__ == "__main__":
    for name, new, queue in (("G", G, "queue_g"), ("S", S, "queue_s")):
        ids = [i["id"] for i in new]
        assert len(set(ids)) == len(ids), name
        clash = set(ids) & {i["id"] for i in EXIST[queue]}
        assert not clash, sorted(clash)[:5]
        by = {}
        for i in new:
            by.setdefault((i["nodes"], i["kind"], i.get("backend", "")), []).append(i["id"])
        print(f"== {queue}: {len(new)} runs")
        for k, v in sorted(by.items()):
            print(f"   nodes={k[0]:3d} {k[1]:5s} {k[2]:9s} {len(v):3d}  e.g. {v[0]}")
    if "--write" in sys.argv:
        for new, out in ((G, "queue_g/todo_w.jsonl"), (S, "queue_s/todo_u_fill.jsonl")):
            path = os.path.join(ROOT, out)
            with open(path + ".tmp", "w") as f:
                for i in new:
                    f.write(json.dumps(i) + "\n")
            os.replace(path + ".tmp", path)
            print("wrote", path)
