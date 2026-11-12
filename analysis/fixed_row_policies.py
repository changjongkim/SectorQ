#!/usr/bin/env python3
"""E1 without GPUs (review plan 2026-11-12): how many branching rotations change a label under fixed-row policies, in the
coordinates of the Clifford frame, against the window count of SectorQ.

  env/bin/python3 analysis/fixed_row_policies.py [--rows 8] [--seeds 5]      (writes analysis/fixed_row_policies.json)

The generators come from bin/pp_engine_k --dump-gens (at position 0, the frame applied), the windows from --plan-only.
A rotation whose generator has an odd overlap with a fixed row moves the terms it branches to another sector, so under
fixed rows every such rotation ends a window (jobs/x_ablate.sh, --fixed-rows). Policies, each with G rows:
  random       G random rows over all x and z bits
  x-only       G random rows over the x bits (the x-bit rows of paulistrings-rs)
  z-only       G random rows over the z bits
  first window G rows of the annihilator of the window in which --fixed-rows 1 chose its rows in the former ablation
               (the operation printed in logs/x/fx.<circuit>.g16.1e-4.out)
  dividing     at the same operation b, rows conserved by the most later rotations among those that still divide the
               terms at b (the terms at b differ only by the generators before b, span A; a row orthogonal to A gives
               every term one label), without a balance condition: a lower bound on the breaks of useful fixed rows
               under this greedy order. The engine's --fixed-rows 2 adds the balance of --fixed-rows 1 (pp_engine_k3).
The row space conserved by every rotation of the circuit is listed too (whole-circuit rows): every term lies in the
observable plus the span of all generators, so these rows give every term the same label.
Random policies report the mean over the seeds. Balance needs the live terms and is measured by the engine runs.
"""
import argparse
import json
import os
import random
import re
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(ROOT, "bin", "pp_engine_k")
CIRC = {
    "ki": "--circuit ki_lab --weight 0",
    "tfim": "--circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20",
    "noise": "--circuit ki_noisy --noise 0.005",
    "tfimsq": "--circuit tfim_sq --lx 11 --ly 11 --theta 0.2 --phi -0.3 --steps 10",
    "xxz": "--circuit xxz --lx 8 --ly 8 --jxy 0.1 --jz 0.15 --steps 6",
    "qaoa": "--circuit qaoa --n 100 --layers 4",
    "hea": "--circuit hea --n 100 --layers 16 --amp 0.4",
}


def engine(args):
    env = dict(os.environ, MPICH_GPU_SUPPORT_ENABLED="0")
    return subprocess.run([ENGINE] + args.split(), capture_output=True, text=True, env=env, cwd=ROOT, check=True).stdout


def load(c):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "gens")
        engine(f"{CIRC[c]} --dump-gens {path}")
        lines = open(path).read().split("\n")
    n, words = map(int, re.match(r"qubits (\d+) words (\d+)", lines[0]).groups())
    nw = words // 2
    gens = []
    for line in lines[1:]:
        if line.strip():
            parts = line.split()
            v = 0
            for i, h in enumerate(parts[1:]):
                v |= int(h, 16) << (64 * i)
            gens.append((int(parts[0]), v))
    plan = engine(f"{CIRC[c]} --plan-only 1 --trace 1")
    windows = [tuple(map(int, m)) for m in re.findall(r"plan window=\d+ ops=(\d+)\.\.(\d+)", plan)]
    return n, nw, gens, windows


def bits(n, nw, part):
    xs = [q for q in range(n)]
    zs = [64 * nw + q for q in range(n)]
    return {"all": xs + zs, "x": xs, "z": zs}[part]


class Span:
    def __init__(self):
        self.rows = {}                                  # pivot bit -> reduced row

    def reduce(self, v):
        for p, r in self.rows.items():
            if v >> p & 1:
                v ^= r
        return v

    def add(self, v):
        v = self.reduce(v)
        if not v:
            return False
        p = v.bit_length() - 1
        for q in list(self.rows):
            if self.rows[q] >> p & 1:
                self.rows[q] ^= v
        self.rows[p] = v
        return True

    def annihilator(self, cols):
        out = []
        for j in cols:
            if j in self.rows:
                continue
            r = 1 << j
            for p, row in self.rows.items():
                if row >> j & 1:
                    r |= 1 << p
            out.append(r)
        return out


def parity(a, b):
    return bin(a & b).count("1") & 1


def combos(basis, g, rng):
    rows = []
    for _ in range(g):
        r = 0
        while not r:
            for b in basis:
                if rng.random() < 0.5:
                    r ^= b
        rows.append(r)
    return rows


def breaks(gens, rows):
    return sum(1 for _, q in gens if any(parity(r, q) for r in rows))


def whole_circuit_rows(gens, n, nw):
    S = Span()
    for _, q in gens:
        S.add(q)
    return len(S.annihilator(bits(n, nw, "all")))


def best_fixed(gens, n, nw, g, b):
    """Dividing rows conserved by the most later rotations (the rows of pp_engine_k2's oracleRows): generators before b span A; later generators enter S by decreasing use, one in S + A only
    while dim(S + A) - dim(S) > g + 4. Returns representatives of the annihilator of S modulo that of S + A."""
    S, SA = Span(), Span()
    for k, q in gens:
        if k < b:
            SA.add(q)
    uses = {}
    for k, q in gens:
        if k >= b:
            if q in uses:
                uses[q][0] += 1
            else:
                uses[q] = [1, k]
    for q, _ in sorted(uses.items(), key=lambda e: (-e[1][0], e[1][1])):
        if not S.reduce(q):
            continue
        if SA.reduce(q):
            S.add(q)
            SA.add(q)
        elif len(SA.rows) - len(S.rows) > g + 4:
            S.add(q)
    cols = bits(n, nw, "all")
    Q, reps = Span(), []
    for v in SA.annihilator(cols):
        Q.add(v)
    for v in S.annihilator(cols):
        if Q.add(v):
            reps.append(v)
    return reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    chosen = {}                                          # operation where --fixed-rows 1 chose its rows (former ablation)
    for c in CIRC:
        p = os.path.join(ROOT, "logs", "x", f"fx.{c}.g16.1e-4.out")
        if os.path.exists(p):
            m = re.search(r"fixed rows: \d+ chosen at operation (\d+)", open(p).read())
            if m:
                chosen[c] = int(m.group(1))
    out = []
    print(f"{'circuit':8s} {'rotations':>9s} {'windows':>7s} {'random':>7s} {'x-only':>7s} {'z-only':>7s} {'first':>7s} {'divid.':>7s} {'after b':>7s} {'whole-circuit rows':>18s}")
    for c in CIRC:
        n, nw, gens, windows = load(c)
        g = a.rows
        row = dict(circuit=c, rotations=len(gens), windows=len(windows))
        for part in ("all", "x", "z"):
            vals = []
            for s in range(a.seeds):
                rng = random.Random(s)
                cols = bits(n, nw, part)
                rows = [sum(1 << j for j in cols if rng.random() < 0.5) or (1 << cols[0]) for _ in range(g)]
                vals.append(breaks(gens, rows))
            row["random" if part == "all" else part + "-only"] = sum(vals) / len(vals)
        if c in chosen:
            b = chosen[c]
            w = next((w for w in windows if w[0] <= b < w[1]), windows[0])
            S = Span()
            for k, q in gens:
                if w[0] <= k < w[1]:
                    S.add(q)
            ann = S.annihilator(bits(n, nw, "all"))
            vals = [breaks([(k, q) for k, q in gens if k >= b], combos(ann, g, random.Random(s))) for s in range(a.seeds)]
            row["first window"] = sum(vals) / len(vals)
            row["first window at"] = b
            reps = best_fixed(gens, n, nw, g, b)
            vals = [breaks([(k, q) for k, q in gens if k >= b], combos(reps, g, random.Random(s))) for s in range(a.seeds)]
            row["dividing"] = sum(vals) / len(vals)
            row["rotations after b"] = sum(1 for k, _ in gens if k >= b)
        row["whole-circuit rows"] = whole_circuit_rows(gens, n, nw)
        out.append(row)
        f = lambda k: f"{row[k]:7.0f}" if k in row else f"{'-':>7s}"
        print(f"{c:8s} {row['rotations']:9d} {row['windows']:7d} {f('random')} {f('x-only')} {f('z-only')} {f('first window')} {f('dividing')} {f('rotations after b')} {row['whole-circuit rows']:18d}")
    json.dump(out, open(os.path.join(ROOT, "analysis", "fixed_row_policies.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
