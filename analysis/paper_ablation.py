#!/usr/bin/env python3
"""Fixed-row ablation of Figure 15 and Table 5 (fixed against recomputed rows on 16 GPUs).

Since 2026-11-12 the runs are the re-measurement with the current engine (bin/pp_engine_k3, jobs/y_items.py E1,
logs/y/e1.<mode>.<circuit>.<cutoff>.out): re = recomputed rows, fx = fixed rows of the first partitioned window,
bf = rotation-aware fixed rows (--fixed-rows 2, Section 4.10). The former runs of logs/x (pp_engine_x, 4 and 16 GPUs)
are no longer read.
"""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def parse_log(path):
    text = path.read_text(errors='replace')
    def num(pattern, cast=float):
        match = re.search(pattern, text, re.M)
        return cast(match.group(1)) if match else None
    result = dict(log=str(path.relative_to(ROOT)))
    for key, pattern, cast in [
        ('exit', r'^exit=(\d+)\s*$', int), ('wall', r'^wall=([\d.]+)', float),
        ('expectation', r'^expectation=([^\s]+)', float), ('windows', r'^windows=(\d+)', int),
        ('terms', r'^max_terms_at_window_boundary=(\d+)', int),
        ('workers', r'summed over (\d+) GPU workers:', int),
        ('operations', r'operations=([\d.]+)', float), ('partition', r'partition=([\d.]+)', float),
        ('h2d', r'host->gpu=([\d.]+)', float), ('d2h', r'gpu->host=([\d.]+)', float),
        ('idle', r'idle=([\d.]+)', float), ('planning', r'planning=([\d.]+)', float),
        ('row_wait', r'workers waited ([\d.]+) s', float),
        ('sent_gb', r'between nodes: ([\d.]+) GB', float),
        ('exchange', r'between nodes: [\d.]+ GB sent, ([\d.]+) s', float),
        ('sync', r'waiting for the slowest node=([\d.]+)', float),
        ('counts', r'exchange per node on average: counts ([\d.]+)', float),
        ('posting', r'posting ([\d.]+) s', float), ('transfer', r'transfer ([\d.]+) s', float),
        ('new_chunks', r'of it new host chunks ([\d.]+) s', float),
        ('gpu_splits', r'splits_on_gpu=(\d+)', int), ('host_splits', r'splits_on_host=(\d+)', int),
        ('host_gib', r'GiB on the busiest node, ([\d.]+) GiB', float),
    ]:
        result[key] = num(pattern, cast)
    traces = []
    for line in text.splitlines():
        if line.startswith('trace window='):
            trace = {k: float(v) for k, v in re.findall(r'(\w+)=(-?\d+(?:\.\d+)?)(?=\s|$)', line)}
            if trace.get('terms_in', 0):
                trace['imbalance_nonempty'] = trace['largest_sector_in'] * trace['sectors'] / trace['terms_in']
                trace['imbalance_all'] = trace['largest_sector_in'] * (2 ** trace['rows']) / trace['terms_in']
            traces.append(trace)
    result['trace'] = traces
    return result

def main():
    rows = []
    for path in sorted((ROOT / 'logs/y').glob('e1.*.out')):
        match = re.fullmatch(r'e1\.(re|fx|bf)\.(\w+)\.([\de.-]+)\.out', path.name)
        if not match:
            continue
        mode, circuit, cutoff = match.groups()
        gpus = 16
        row = parse_log(path)
        # Preserve the source path and summary counters without duplicating all
        # per-window logs in the checked-in table. parse_log retains traces for
        # callers that analyze occupancy or candidate selection.
        row['trace_windows'] = len(row.pop('trace'))
        row.update(mode=mode, circuit=circuit, gpus=int(gpus), cutoff=cutoff)
        rows.append(row)
    out = ROOT / 'analysis/paper_fixed_rows.json'
    out.write_text(json.dumps(rows, indent=2) + '\n')
    fields = [k for k in rows[0] if k != 'trace']
    with (ROOT / 'analysis/paper_fixed_rows.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    for fixed in ('fx', 'bf'):
        every = []
        for c in ('ki', 'tfim', 'tfimsq', 'xxz', 'qaoa', 'hea'):
            runs = {(r['mode'], r['cutoff']): r for r in rows if r['circuit'] == c}
            pairs = [(r, runs.get((fixed, x))) for (m, x), r in runs.items() if m == 're']
            failed = [(c, a['cutoff']) for a, b in pairs if b and a['exit'] == 0 and b['exit'] != 0]
            pairs = [(a, b) for a, b in pairs if b and a['exit'] == b['exit'] == 0]
            ratios = [b['wall'] / a['wall'] for a, b in pairs]
            every += ratios
            errors = [abs(a['expectation'] - b['expectation']) for a, b in pairs]
            print(16, fixed, c, 'pairs', len(pairs), 'speedup', round(min(ratios), 2), round(max(ratios), 2), 'max answer diff', max(errors),
                  'fixed failed', failed)
        print(16, fixed, 'all', 'pairs', len(every), 'speedup', round(min(every), 1), round(max(every), 1))
    print('Table 5 (cutoff 1e-5): circuit, windows re fx, traffic GB re fx')
    for c in ('ki', 'tfim', 'tfimsq', 'xxz', 'qaoa', 'hea'):
        a = next(r for r in rows if r['circuit'] == c and r['mode'] == 're' and r['cutoff'] == '1e-5')
        b = next(r for r in rows if r['circuit'] == c and r['mode'] == 'fx' and r['cutoff'] == '1e-5')
        print(f"{c:7s} {a['windows']:>5,d} {b['windows']:>6,d} {a['sent_gb']:8.1f} {b['sent_gb']:8.1f}   re wall {a['wall']:.2f} s")

if __name__ == '__main__':
    main()
