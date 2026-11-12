#!/usr/bin/env python3
"""E3 (review plan 2026-11-12): the retained terms after the truncation that ends a window, from one GPU with one sector
(every merge global) against 16 GPUs with sectors (jobs/y_items.py, logs/y/ckpt/e3.<circuit>.w<W>.<g1|g16>.r<rank>).

  env/bin/python3 analysis/retained_terms.py          (writes analysis/retained_terms.json)

For each pair: the term counts, the strings found in only one of the two, duplicated strings within one checkpoint, and
the largest absolute and relative difference of the coefficients of the shared strings. A string found in only one run
has a coefficient next to the cutoff when the summation order decided its truncation; the ratio |c| / cutoff is listed.

Checkpoint (src/engine/pp_engine.cpp writeCheckpoint): int32 header[6] = {magic, words, position, operations, window,
ranks}; int32 window[6]; int64 payload length; uint64 payload; int64 buckets; per bucket int64 n, n*words uint64 strings,
n doubles.
"""
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = os.path.join(ROOT, "logs", "y", "ckpt")


def read(prefix):
    keys, coefs, pos = [], [], None
    for path in sorted(glob.glob(prefix + ".r*")):
        with open(path, "rb") as f:
            hdr = np.fromfile(f, np.int32, 6)
            assert hdr[0] == 0x50504B45, path
            words, pos = int(hdr[1]), int(hdr[2])
            np.fromfile(f, np.int32, 6)
            wl = int(np.fromfile(f, np.int64, 1)[0])
            np.fromfile(f, np.uint64, wl)
            nb = int(np.fromfile(f, np.int64, 1)[0])
            for _ in range(nb):
                n = int(np.fromfile(f, np.int64, 1)[0])
                if n:
                    keys.append(np.fromfile(f, np.uint64, n * words).reshape(n, words))
                    coefs.append(np.fromfile(f, np.float64, n))
    k = np.concatenate(keys) if keys else np.zeros((0, 1), np.uint64)
    return np.ascontiguousarray(k), (np.concatenate(coefs) if coefs else np.zeros(0)), pos


def as_void(k):
    return k.view(np.dtype((np.void, k.dtype.itemsize * k.shape[1]))).ravel()


def compare(a, b, cutoff):
    ka, ca, pa = read(a)
    kb, cb, pb = read(b)
    va, vb = as_void(ka), as_void(kb)
    ua, ia = np.unique(va, return_index=True)
    ub, ib = np.unique(vb, return_index=True)
    common, xa, xb = np.intersect1d(ua, ub, assume_unique=True, return_indices=True)
    da, db = ca[ia[xa]], cb[ib[xb]]
    diff = np.abs(da - db)
    rel = diff / np.maximum(np.abs(da), 1e-300)
    only_a = np.setdiff1d(np.arange(len(ua)), xa)
    only_b = np.setdiff1d(np.arange(len(ub)), xb)
    near = [float(abs(x)) / cutoff for x in np.concatenate([ca[ia[only_a]], cb[ib[only_b]]])]
    return dict(position=[pa, pb], terms=[int(len(va)), int(len(vb))], duplicates=[int(len(va) - len(ua)), int(len(vb) - len(ub))],
                shared=int(len(common)), only_one_gpu=int(len(only_a)), only_16_gpus=int(len(only_b)),
                max_abs_diff=float(diff.max()) if len(diff) else 0.0, max_rel_diff=float(rel.max()) if len(rel) else 0.0,
                unshared_over_cutoff=[min(near), max(near)] if near else None)


def main():
    out = []
    for g1 in sorted(glob.glob(os.path.join(CK, "e3.*.g1.r0"))):
        base = g1[:-len(".g1.r0")]
        if not glob.glob(base + ".g16.r*"):
            continue
        m = re.search(r"e3\.(\w+)\.w(\d+)$", base)
        cutoff = 3e-5                                     # jobs/y_items.py e3()
        r = dict(circuit=m.group(1), window=int(m.group(2)), cutoff=cutoff, **compare(base + ".g1", base + ".g16", cutoff))
        out.append(r)
        print(f"{r['circuit']:5s} w{r['window']:<3d} op {r['position'][0]:5d}/{r['position'][1]:<5d} terms {r['terms'][0]:>11,d} {r['terms'][1]:>11,d}"
              f"  only1 {r['only_one_gpu']:4d} only16 {r['only_16_gpus']:4d} dup {r['duplicates']}  max|dc| {r['max_abs_diff']:.2e}"
              f"  rel {r['max_rel_diff']:.2e}  unshared |c|/cut {r['unshared_over_cutoff']}")
    json.dump(out, open(os.path.join(ROOT, "analysis", "retained_terms.json"), "w"), indent=1)
    if out:
        print(f"{len(out)} pairs: {sum(r['only_one_gpu'] + r['only_16_gpus'] == 0 for r in out)} with identical string sets")


if __name__ == "__main__":
    main()
