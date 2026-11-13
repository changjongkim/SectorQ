#!/usr/bin/env python3
"""Parities that a full circuit conserves, evaluated on every term of its observable (review 2026-11-13).

  env/bin/python3 analysis/whole_circuit_rows.py      (prints; writes analysis/whole_circuit_rows.json)

Every live term equals a term of the observable plus a sum of generators (in the coordinates at position 0, where the
observable starts). A row orthogonal to every generator of the circuit therefore gives every live term the parity of
the observable term it descends from. The rows divide no live terms when all terms of the observable have the same
parities under them. Six circuits have a one-term observable (Z on one qubit); QAOA has the sum of Z_a Z_b over the
edges. The generators come from bin/pp_engine_k3 --dump-gens, the QAOA observable from results/f_ext/ir/qaoa.ir.json
(the circuit file the external implementations read, 100 qubits, 4 layers).
"""
import json
import os
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(ROOT, "bin", "pp_engine_k3")
CIRC = {
    "ki": "--circuit ki_lab --weight 0",
    "tfim": "--circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20",
    "noise": "--circuit ki_noisy --noise 0.005",
    "tfimsq": "--circuit tfim_sq --lx 11 --ly 11 --theta 0.2 --phi -0.3 --steps 10",
    "xxz": "--circuit xxz --lx 8 --ly 8 --jxy 0.1 --jz 0.15 --steps 6",
    "qaoa": "--circuit qaoa --n 100 --layers 4",
    "hea": "--circuit hea --n 100 --layers 16 --amp 0.4",
}


def generators(args):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "g.txt")
        subprocess.run([ENGINE] + args.split() + ["--dump-gens", path], cwd=ROOT, check=True, capture_output=True)
        lines = open(path).read().split("\n")
    n, words = int(lines[0].split()[1]), int(lines[0].split()[3])
    half = words // 2
    gens = []
    for l in lines[1:]:
        if not l.strip():
            continue
        w = [int(x, 16) for x in l.split()[1:]]
        x = sum(v << (64 * i) for i, v in enumerate(w[:half]))
        z = sum(v << (64 * i) for i, v in enumerate(w[half:]))
        gens.append(x | (z << n))                 # bit q: x of qubit q, bit n + q: z of qubit q
    return n, gens


def annihilator(vecs, bits):
    """Basis of {r : r . v = 0 for every v} over GF(2), with vectors as integers of `bits` bits."""
    rows, piv = [], []                            # reduced row echelon form of the generator span
    for v in vecs:
        for r, p in zip(rows, piv):
            if v >> p & 1:
                v ^= r
        if v:
            p = v.bit_length() - 1
            for i in range(len(rows)):
                if rows[i] >> p & 1:
                    rows[i] ^= v
            rows.append(v)
            piv.append(p)
    free = [b for b in range(bits) if b not in set(piv)]
    out = []
    for f in free:                                # one kernel vector per free column of the span matrix (transposed)
        r = 1 << f
        for row, p in zip(rows, piv):
            if row >> f & 1:
                r |= 1 << p
        out.append(r)
    return out


def parity(a, b):
    return bin(a & b).count("1") & 1


def pauli_bits(s, n):
    v = 0
    for q, c in enumerate(s):
        if c in "XY":
            v |= 1 << q
        if c in "ZY":
            v |= 1 << (n + q)
    return v


def main():
    res = {}
    for c, args in CIRC.items():
        n, gens = generators(args)
        ann = annihilator(gens, 2 * n)
        assert all(parity(r, g) == 0 for r in ann for g in gens)
        if c == "qaoa":
            ir = json.load(open(os.path.join(ROOT, "results", "f_ext", "ir", "qaoa.ir.json")))
            assert ir["qubits"] == n
            obs = [pauli_bits(t["pauli"], n) for t in ir["observable"]]
        else:
            obs = [None]                           # one term: a single value under every row
        labels = {tuple(parity(r, o) for r in ann) for o in obs} if obs[0] is not None else {()}
        res[c] = dict(qubits=n, generators=len(gens), whole_circuit_rows=len(ann), observable_terms=len(obs),
                      distinct_labels_of_observable_terms=len(labels))
        print(f"{c:7s} rows conserved by the full circuit {len(ann):3d}, observable terms {len(obs):4d}, "
              f"distinct labels of the observable terms {len(labels)}")
    json.dump(res, open(os.path.join(ROOT, "analysis", "whole_circuit_rows.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
