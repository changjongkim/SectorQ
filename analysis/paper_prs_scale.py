"""Audit completed 32/64-GPU matched-truncation jobs without changing raw data."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
JOBS = {32: '59769001', 64: '59769002'}
CUTS = {
    'ki': '1e-3 1e-4 3e-5 1e-5 3e-6 1e-6 5e-7',
    'tfim': '1e-4 3e-5 1e-5 3e-6 1e-6 5e-7',
    'noise': '1e-4 3e-5 1e-5 3e-6 1e-6 5e-7',
    'tfimsq': '1e-4 3e-5 1e-5 3e-6 1e-6',
    'xxz': '1e-4 3e-5 1e-5 3e-6 1e-6',
    'qaoa': '1e-4 3e-5 1e-5 3e-6 1e-6 5e-7',
    'hea': '1e-4 3e-5 1e-5 3e-6 1e-6 3e-7 1e-7',
}
OOM = re.compile(r'out of memory|OutOfMemory|oom_kill|oom-kill|MemoryError', re.I)


def audit():
    rows, sources = [], {}

    def read(name):
        data = (ROOT / name).read_bytes()
        sources[name] = hashlib.sha256(data).hexdigest()
        return data.decode('utf-8', errors='replace')

    for g, job in JOBS.items():
        batch = read('logs/pp_p_g%d-%s.out' % (g, job))
        for circuit, ladder in CUTS.items():
            for cutoff in ladder.split():
                stem = '' if circuit == 'ki' else circuit + '.'
                sqlog = 'logs/p/sq.e1.%sg%d.%s.out' % (stem, g, cutoff)
                sq = read(sqlog)
                exits = re.findall(r'^exit=(\d+)$', sq, re.M)
                assert len(exits) == 1, sqlog
                code = int(exits[0])
                assert 'SectorQ %s %d GPUs %s:' % (circuit, g, cutoff) in batch
                assert 'nodes=%d gpus/node=4' % (g // 4) in sq
                assert 'truncEvery=1' in sq
                status = ('ok' if code == 0 else 'timeout' if code == 124
                          else 'oom' if OOM.search(sq) else 'process_error')
                row = dict(gpus=g, job=job, circuit=circuit, cutoff=cutoff,
                           sectorq_status=status, sectorq_exit=code,
                           sectorq_log=sqlog)
                if status == 'ok':
                    row.update(sectorq_seconds=float(re.search(r'^wall=([\d.]+) s', sq, re.M)[1]),
                               sectorq_expectation=float(re.search(r'^expectation=([-\d.e+]+)', sq, re.M)[1]))
                record = 'queue_p/done/p.mg%d.%s.%s.json' % (g, circuit, cutoff)
                prs = json.loads(read(record))
                row.update(prs_status=prs['status'], prs_record=record, prs_exit=prs.get('exit'))
                if prs['status'] != 'skipped_after_failure':
                    assert str(prs['job']) == job, record
                    log = read(prs['log'])
                    row['prs_log'] = prs['log']
                    if prs['status'] == 'ok':
                        assert prs['exit'] == 0 and not prs.get('left_running'), record
                        result = json.loads(re.findall(r'^RESULT (\{.*\})$', log, re.M)[-1])
                        assert all(result[k] == prs[k] for k in result), record
                        assert prs['ranks'] == g and prs['cutoff'] == float(cutoff)
                        row.update(prs_seconds=prs['wall'], prs_expectation=prs['expectation'])
                    elif prs['status'] == 'oom':
                        assert OOM.search(log), record
                if status == 'ok' and prs['status'] == 'ok':
                    row['expectation_abs_difference'] = abs(row['sectorq_expectation'] - prs['expectation'])
                    assert row['expectation_abs_difference'] < 1e-9, row
                    row['prs_over_sectorq_time'] = prs['wall'] / row['sectorq_seconds']
                rows.append(row)
    summary = {}
    for g in JOBS:
        group = [r for r in rows if r['gpus'] == g]
        pairs = [r for r in group if 'prs_over_sectorq_time' in r]
        eval_pairs = [r for r in pairs if float(r['cutoff']) <= 1e-4]
        largest = [min([r for r in pairs if r['circuit'] == c], key=lambda r: float(r['cutoff'])) for c in CUTS]
        summary[g] = dict(
            sectorq_status=dict(collections.Counter(r['sectorq_status'] for r in group)),
            prs_status=dict(collections.Counter(r['prs_status'] for r in group)),
            matched_pairs=len(pairs),
            maximum_expectation_abs_difference=max(r['expectation_abs_difference'] for r in pairs),
            speedup_range=[min(r['prs_over_sectorq_time'] for r in pairs), max(r['prs_over_sectorq_time'] for r in pairs)],
            cutoff_at_most_1e4_speedup_range=[min(r['prs_over_sectorq_time'] for r in eval_pairs), max(r['prs_over_sectorq_time'] for r in eval_pairs)],
            largest_shared_by_circuit=[{k: r[k] for k in ('circuit', 'cutoff', 'sectorq_seconds', 'prs_seconds', 'prs_over_sectorq_time')} for r in largest],
            sectorq_success_prs_oom=[{k: r[k] for k in ('circuit', 'cutoff', 'sectorq_seconds')} for r in group if r['sectorq_status'] == 'ok' and r['prs_status'] == 'oom'])
    return dict(jobs=JOBS, notes=[
        'Matched per-operation truncation; do not mix with default truncation-interval-10 timings.',
        'One timing sample per setting; paulistrings-rs timings include lazy GPU kernel compilation.',
        'Skipped points are unmeasured, not measured OOM; nonzero exits are never successful timings.',
        'Source hashes capture audited files, not historical binary provenance.'
    ], summary=summary, rows=rows, source_sha256=sources)


if __name__ == '__main__':
    result = audit()
    dest = ROOT / 'analysis' / 'paper_prs_scale.json'
    dest.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    with dest.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for r in result['rows'] for k in r}))
        writer.writeheader()
        writer.writerows(result['rows'])
    print(json.dumps(result['summary'], indent=2))
