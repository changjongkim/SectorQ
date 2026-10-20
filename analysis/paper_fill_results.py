#!/usr/bin/env python3
"""Audit the six completed scale-fill allocations, retaining every outcome.

Reads only named queue files and their recorded logs. No Slurm or experiment
mutation. CSVs include failed and unexecuted cases, never synthetic timings.
"""
import collections
import csv
import hashlib
import json
import re

from paper_ablation import ROOT, parse_log

JOBS = ('59768854', '59768995', '59768996', '59768998', '59768999', '59769000')
OUT = ROOT / 'analysis'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def check_expectations(rows):
    """Check available independent engine-i results under matched truncation."""
    references = collections.defaultdict(list)
    for path in sorted((ROOT / 'logs/s').glob('*.out')):
        if not re.fullmatch(r'(?:i\.ref|n\.eng|n\.weak[SL]|n10|n12|r12|i12|i\.runtime(?:10|12))'
                            r'\.(ki|tfim|noise)\.[\de.-]+\.g\d+\.out', path.name):
            continue
        text = path.read_text(errors='replace')
        if '\nbinary=pp_engine_i\n' not in text or 'truncEvery=10 ' not in text:
            continue
        result = parse_log(path)
        if result['exit'] != 0 or result['expectation'] is None:
            continue
        circuit = re.search(r'--circuit (\S+)', text).group(1)
        cutoff = float(re.search(r'--coef (\S+)', text).group(1))
        references[circuit, cutoff].append(result)
    checks = []
    for row in rows:
        if row['status'] != 'ok' or row['backend'] == 'monoprop':
            continue
        circuit = {'ki': 'ki_lab', 'tfim': 'tfim_hh', 'noise': 'ki_noisy'}[row['circuit']]
        candidates = [r for r in references[circuit, row['cutoff']] if r['log'] != row['log']]
        if candidates:
            reference = candidates[0]
            error = abs(row['expectation'] - reference['expectation'])
            assert error < 1e-9, (row['id'], error)
            checks.append(dict(id=row['id'], reference=reference['log'], difference=error))
        else:
            checks.append(dict(id=row['id'], reference=None, difference=None))
    return checks


def main():
    records = [json.loads(line) for line in (ROOT / 'queue_s/runner.log').read_text().splitlines()]
    records = [r for r in records if str(r.get('job')) in JOBS]
    assert len({r['id'] for r in records}) == len(records)
    rows = []
    for r in records:
        done = json.loads((ROOT / 'queue_s/done' / r['id']).read_text())
        for k in ('job', 'status', 'exit', 'wall', 'expectation', 'log'):
            assert done[k] == r[k], (r['id'], k)
        path = ROOT / r['log']
        row = {k: r.get(k) for k in ('job', 'id', 'status', 'exit', 'wall', 'expectation', 'log', 'time')}
        row['sha256'] = digest(path)
        if path.suffix == '.json':
            launch = json.loads(path.read_text())
            assert str(launch['job']) == str(r['job'])
            assert launch['status'] == ('success' if r['status'] == 'ok' else r['status'])
            assert launch['exit_code'] == r['exit']
            row.update(backend=launch['backend'], circuit=launch['case'],
                       cutoff=launch['cutoff'], nodes=len(launch['hosts_used']),
                       gpus_used=0, truncation=launch['policy'])
            if r['status'] == 'ok':
                assert launch['solver_seconds'] == r['wall']
                assert launch['expectation'] == r['expectation']
            else:
                err = path.with_name(path.name.replace('.launch.json', '.stderr.txt'))
                evidence = err.read_text(errors='replace')
                assert re.search(r'oom|out.of.memory|bad_alloc|cannot allocate', evidence, re.I), r['id']
                row['failure_evidence'] = str(err.relative_to(ROOT))
        else:
            text = path.read_text()
            parsed = parse_log(path)
            for k in ('exit', 'wall', 'expectation'):
                assert parsed[k] == r[k], (r['id'], k)
            header = re.search(r'^===== (engine|cupp) nodes=(\d+) gpus=(\d+) args: (.*)$', text, re.M)
            kind, nodes, gpus, args = header.groups()
            opts = dict(zip(args.split()[::2], args.split()[1::2]))
            row.update(backend='sectorq' if kind == 'engine' else 'cupp',
                       circuit={'ki_lab': 'ki', 'tfim_hh': 'tfim', 'ki_noisy': 'noise'}[opts['--circuit']],
                       cutoff=float(opts['--coef']), nodes=int(nodes), gpus_used=int(gpus),
                       truncation='interval10', binary=r['bin'], args=args)
            row['input'] = 'window' if kind == 'engine' else 'original' if row['circuit'] == 'tfim' or opts.get('--frame') == '0' else 'frame'
            if kind == 'engine':
                row.update(boundary_terms=parsed['terms'], host_pool_gib=parsed['host_gib'],
                           sector_cap=2 ** int(opts.get('--gmax', 8)),
                           chunk_terms=int(opts.get('--chunk', 262144)))
            if r['status'] != 'ok':
                assert re.search(r'out.of.memory|bad_alloc|cannot allocate', text, re.I), r['id']
                row['failure_evidence'] = r['log']
        assert r['status'] in ('ok', 'oom'), r['id']
        if r['status'] == 'ok':
            assert r['wall'] > 0 and r['expectation'] is not None
            if r['exit'] != 0:
                row.update(status='result_exit_nonzero', recorded_status=r['status'],
                           reported_solver_seconds=r['wall'], wall=None,
                           failure_evidence=str(path.with_name(path.name.replace('.launch.json', '.stderr.txt')).relative_to(ROOT)))
        else:
            assert r['wall'] is None and r['expectation'] is None
        rows.append(row)

    # Separate inventory of the fill plan: missing is not pending or a failure.
    plan = [json.loads(line) for line in (ROOT / 'queue_s/todo_t_fill.jsonl').read_text().splitlines()]
    coverage = []
    for item in plan:
        path = ROOT / 'queue_s/done' / item['id']
        record = json.loads(path.read_text()) if path.exists() else {}
        status = record.get('status', 'unexecuted')
        if status == 'ok' and record.get('exit') not in (None, 0):
            status = 'result_exit_nonzero'
        coverage.append(dict(id=item['id'], nodes=item['nodes'],
                             status=status,
                             job=record.get('job'), wall=record.get('wall') if status == 'ok' else None,
                             log=record.get('log')))
    checks = check_expectations(rows)
    payload = dict(jobs=JOBS, measurements=rows, fill_plan=coverage,
                   independent_expectation_checks=checks,
                   notes=['MonoProp uses native truncation; its expectations are not matched-rule results.',
                          'Frame-input cuPauliProp is not the primary original-circuit baseline.',
                          'Unexecuted entries are not claimed to be queued.',
                          'Skipped and predicted entries contain no measured failure or time.'])
    (OUT / 'paper_fill_results.json').write_text(json.dumps(payload, indent=2) + '\n')
    write_csv('paper_fill_results.csv', rows)
    write_csv('paper_fill_coverage.csv', coverage)
    for job in JOBS:
        print(job, dict(collections.Counter(r['status'] for r in rows if r['job'] == job)))
    print('all measurements', len(rows), dict(collections.Counter(r['status'] for r in rows)))
    print('fill plan', len(coverage), dict(collections.Counter(r['status'] for r in coverage)))
    checked = [r for r in checks if r['reference']]
    print('independent matched-rule checks', len(checked), 'max difference',
          max(r['difference'] for r in checked))


if __name__ == '__main__':
    main()
