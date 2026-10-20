#!/usr/bin/env python3
"""Read and validate the Figure 18 count-policy campaign, including unfinished runs."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re

from paper_ablation import ROOT, parse_log

POLICIES = ('adaptive', 'fixed16', 'fixed32', 'fixed64', 'fixed128', 'fixed256')


def read_run(path):
    path = Path(path).resolve()
    text = path.read_text(errors='replace')
    first = text.splitlines()[0] if text else ''
    if not first.startswith('# count_run '):
        raise ValueError('missing provenance: ' + str(path))
    meta = json.loads(first[len('# count_run '):])
    row = parse_log(path)
    exit_match = re.search(r'^exit=(-?\d+)\s*$', text, re.M)
    if exit_match:
        row['exit'] = int(exit_match.group(1))
    row['meta'] = meta
    row['schedule'] = re.findall(r'^trace window=(\d+) ops=(\d+)\.\.(\d+)', text, re.M)
    row['audit'] = [dict(re.findall(r'(\w+)=([^\s]+)', line))
                    for line in text.splitlines() if line.startswith('count_policy ')]
    row['status'] = 'incomplete' if row['exit'] is None else 'failed'
    row['failure_reason'] = None
    if row['status'] == 'failed':
        if re.search(r'out.of.memory|bad_alloc|oom.kill|host allocation failed', text, re.I):
            row['failure_reason'] = 'oom'
        elif row['exit'] in (124, 137):
            row['failure_reason'] = 'timeout_or_kill'
        else:
            row['failure_reason'] = 'process_error'
    if row['exit'] == 0:
        valid = (row['wall'] is not None and row['wall'] > 0 and math.isfinite(row['wall'])
                 and row['expectation'] is not None and math.isfinite(row['expectation'])
                 and row['workers'] == 16 and len(row['schedule']) == row['windows']
                 and len(row['trace']) == row['windows'] and meta.get('binary_sha256'))
        row['status'] = 'complete' if valid else 'invalid'
        if meta['policy'] not in ('old', 'legacy', 'warmup'):
            try:
                validate_count(row)
            except (ValueError, KeyError, AssertionError) as e:
                row['status'] = 'invalid'
                row['validation_error'] = str(e)
    return row


def validate_count(row):
    meta = row['meta']
    audit, traces = row['audit'], row['trace']
    if len(audit) != len(traces) or not audit:
        raise ValueError('missing count-decision audit')
    fixed = int(meta['policy'][5:]) if meta['policy'].startswith('fixed') else 0
    target_rows = int(math.log2(fixed)) if fixed else None
    active = False
    for i, (a, t) in enumerate(zip(audit, traces)):
        g = int(a['rows'])
        assert int(a['window']) == int(t['window']) and g == int(t['rows'])
        assert int(a['labels']) == 2 ** g and 0 <= int(a['row_rank']) <= g
        assert len(a['row_hash']) == 16
        last = i == len(traces) - 1
        if not last and fixed and int(a['adaptive_next_rows']) >= 4:
            active = True
        assert int(a['next_fixed_active']) == int(active), 'activation state mismatch'
        if not last:
            expected = target_rows if active else int(a['adaptive_next_rows'])
            assert int(a['requested_next_rows']) == expected, 'requested count mismatch'
            assert int(t['next_rows']) == min(expected, int(traces[i + 1]['valid_rows']))
            assert int(traces[i + 1]['rows']) == int(t['next_rows'])
        else:
            assert int(a['requested_next_rows']) == -1


def check_answers(a, b):
    if a['status'] != 'complete' or b['status'] != 'complete':
        raise ValueError('comparison requires successful validated runs')
    if abs(a['expectation'] - b['expectation']) > 1e-9:
        raise ValueError('expectation mismatch')
    if a['schedule'] != b['schedule']:
        raise ValueError('window boundaries mismatch')


def check_compat(a, b):
    check_answers(a, b)
    if a['meta']['allocation'] != b['meta']['allocation']:
        raise ValueError('compatibility controls from different allocations')
    for x, y in zip(a['trace'], b['trace']):
        for key in ('rows', 'next_rows', 'terms_in', 'terms_out', 'sectors', 'largest_sector_in'):
            if x[key] != y[key]:
                raise ValueError('legacy trace mismatch: ' + key)


def check_block(rows):
    by = {r['meta']['policy']: r for r in rows}
    if set(by) != set(POLICIES) or len(rows) != 6:
        raise ValueError('block must contain exactly six policies')
    base = by['adaptive']
    keys = ('allocation', 'binary_sha256', 'circuit', 'cutoff', 'policy_seed', 'phase')
    first_fixed = None
    for name in POLICIES:
        r = by[name]
        check_answers(base, r)
        for key in keys:
            if r['meta'][key] != base['meta'][key]:
                raise ValueError('block provenance mismatch: ' + key)
        for key in ('source_commit', 'campaign'):
            if r['meta'].get(key) != base['meta'].get(key):
                raise ValueError('block provenance mismatch: ' + key)
        # The only changed input flag must be the fixed-sector count.
        def opts(x):
            args = x['meta']['args']
            return {k: v for k, v in zip(args[::2], args[1::2]) if k != '--fixed-sectors'}
        if opts(base) != opts(r):
            raise ValueError('block option mismatch')
        if name == 'adaptive':
            continue
        activation = next((i for i, a in enumerate(r['audit']) if int(a['next_fixed_active'])), len(r['audit']))
        if first_fixed is None:
            first_fixed = activation
        elif first_fixed != activation:
            raise ValueError('different bootstrap activation windows')
        # The current window is still shared when the *next* count first changes.
        for i in range(min(activation + 1, len(r['audit']))):
            if r['audit'][i]['row_hash'] != base['audit'][i]['row_hash']:
                raise ValueError('bootstrap row hash mismatch')
            for key in ('rows', 'terms_in', 'terms_out', 'sectors', 'largest_sector_in'):
                if r['trace'][i][key] != base['trace'][i][key]:
                    raise ValueError('bootstrap population mismatch: ' + key)
        if activation < len(r['audit']) and int(base['audit'][activation]['adaptive_next_rows']) < 4:
            raise ValueError('adaptive reference lacks bootstrap activation')
    return dict(phase=base['meta']['phase'], circuit=base['meta']['circuit'], cutoff=base['meta']['cutoff'],
                policy_seed=base['meta']['policy_seed'], activation_window=first_fixed,
                wall={k: by[k]['wall'] for k in POLICIES},
                reference_over_adaptive={k: by[k]['wall'] / base['wall'] for k in POLICIES[1:]})


def summarize(directory):
    rows = [read_run(p) for p in sorted(Path(directory).glob('*.out'))]
    return summarize_rows(rows)


def summarize_rows(rows):
    groups = {}
    for r in rows:
        m = r['meta']
        if m['phase'] in ('pilot', 'validation', 'confirmatory'):
            groups.setdefault((m['phase'], m['circuit'], m['cutoff'], m['policy_seed']), []).append(r)
    blocks = []
    for key, group in sorted(groups.items()):
        if len(group) == 6 and all(r['status'] == 'complete' for r in group):
            try:
                blocks.append(dict(status='complete', **check_block(group)))
            except ValueError as e:
                blocks.append(dict(status='invalid', key=key, error=str(e)))
        else:
            status = 'incomplete'
            error = None
            if any(r['status'] == 'invalid' for r in group):
                status = 'invalid'
            elif len(group) == 6 and all(r['status'] in ('complete', 'failed') for r in group):
                # A measured failure is a result, not a reason to retry until it wins.
                status = 'failed'
                successful = [r for r in group if r['status'] == 'complete']
                try:
                    for row in successful[1:]:
                        check_answers(successful[0], row)
                except ValueError as exc:
                    status, error = 'invalid', str(exc)
            blocks.append(dict(status=status, key=key, phase=key[0], circuit=key[1],
                               cutoff=key[2], policy_seed=key[3], error=error,
                               run_statuses=dict(Counter(r['status'] for r in group))))
    return dict(rows=rows, blocks=blocks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allocation', required=True)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    if not args.allocation.isdigit():
        parser.error('numeric allocation required')
    directory = ROOT / 'logs/n2count' / args.allocation
    result = summarize(directory)
    print('Runs:', dict(Counter(r['status'] for r in result['rows'])))
    print('Pilot runs:', dict(Counter(r['status'] for r in result['rows'] if r['meta']['phase'] == 'pilot')))
    print('Confirmatory runs:', dict(Counter(r['status'] for r in result['rows'] if r['meta']['phase'] == 'confirmatory')))
    for block in result['blocks']:
        print(json.dumps(block, sort_keys=True))
    if args.write:
        path = ROOT / 'analysis' / ('paper_n2count.' + args.allocation + '.json')
        path.write_text(json.dumps(result, indent=2) + '\n')
        print('Saved', path)
    if any(r['status'] == 'invalid' for r in result['rows']) or any(b['status'] == 'invalid' for b in result['blocks']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
