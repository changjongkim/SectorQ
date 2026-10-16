#!/usr/bin/env python3
"""Compatibility gates and the predeclared 36-run count-policy pilot. No automatic full campaign."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
from paper_n2count import POLICIES, check_answers, check_block, check_compat, read_run, summarize

CIRCUITS = {
    'ki': '--circuit ki_lab --weight 0 --target 6000000',
    'tfim': '--circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20 --target 4000000',
    'noise': '--circuit ki_noisy --noise 0.005 --target 6000000',
    'tfimsq': '--circuit tfim_sq --lx 11 --ly 11 --theta 0.2 --phi -0.3 --steps 10 --target 4000000',
    'xxz': '--circuit xxz --lx 8 --ly 8 --jxy 0.1 --jz 0.15 --steps 6 --target 4000000',
    'qaoa': '--circuit qaoa --n 100 --layers 4 --target 4000000',
    'hea': '--circuit hea --n 100 --layers 16 --amp 0.4 --target 4000000',
}
BASE = ROOT / 'logs/n2count'
SCHEDULE_SEED = 180016


def say(message):
    print(datetime.datetime.utcnow().isoformat() + 'Z ' + message, flush=True)


def output(args):
    return subprocess.check_output(args, cwd=str(ROOT), universal_newlines=True, timeout=30).strip()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan():
    design = json.loads((ROOT / 'jobs/figure18_design.json').read_text())
    blocks = []
    for cutoff in design['pilot']['cutoffs']:
        for circuit in design['pilot']['circuits']:
            order = list(POLICIES)
            random.Random('{}:{}:{}'.format(SCHEDULE_SEED, circuit, cutoff)).shuffle(order)
            blocks.append(dict(circuit=circuit, cutoff=cutoff, policy_seed=7, order=order))
    assert sum(len(b['order']) for b in blocks) == design['pilot']['measured_runs'] == 36
    return blocks


def remaining_seconds(job):
    remaining = output(['squeue', '-h', '-j', job, '-o', '%L'])
    days, stamp = (remaining.split('-', 1) if '-' in remaining else ('0', remaining))
    fields = [int(x) for x in stamp.split(':')]
    if len(fields) == 2:
        fields.insert(0, 0)
    if len(fields) != 3:
        raise ValueError('unrecognized allocation time: ' + remaining)
    return int(days) * 86400 + fields[0] * 3600 + fields[1] * 60 + fields[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--batch', action='store_true', help='use this Slurm batch allocation; do not call salloc')
    parser.add_argument('--source-commit')
    parser.add_argument('--binary-sha256')
    parser.add_argument('--reference-sha256')
    args = parser.parse_args()
    blocks = plan()
    if args.dry_run:
        print(json.dumps(dict(warmups=1, compatibility_runs=14, matched_input_checks=7,
                             pilot_runs=36, blocks=blocks), indent=2))
        return
    os.chdir(str(ROOT))
    BASE.mkdir(exist_ok=True)
    lock = (BASE / 'loop.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    job = None
    interactive_owned = False
    directory = None
    binaries = {name: ROOT / 'bin' / name for name in ('pp_engine_x', 'pp_engine_n2count')}
    hashes = {name: digest(path) for name, path in binaries.items()}
    source_commit = args.source_commit or output(['git', 'rev-parse', 'HEAD'])
    clean_paths = ['src/engine/pp_engine.cpp', 'src/engine/count_policy.hpp',
                   'src/engine/circuit.hpp', 'src/engine/gf2.hpp', 'src/engine/noise_fusion.hpp',
                   'jobs/n2_count.py', 'jobs/n2_count.sh', 'jobs/n2_count_batch.sh', 'jobs/env.sh',
                   'analysis/paper_n2count.py', 'analysis/paper_ablation.py']
    if output(['git', 'diff', source_commit, '--'] + clean_paths):
        raise RuntimeError('experiment sources differ from the submission commit')
    if args.binary_sha256 and hashes['pp_engine_n2count'] != args.binary_sha256:
        raise RuntimeError('experiment binary differs from the submitted hash')
    if args.reference_sha256 and hashes['pp_engine_x'] != args.reference_sha256:
        raise RuntimeError('reference binary differs from the submitted hash')
    if args.batch and not all((args.source_commit, args.binary_sha256, args.reference_sha256)):
        raise RuntimeError('batch execution requires pinned source and both binary hashes')
    submitted_design = json.loads(output(['git', 'show', source_commit + ':jobs/figure18_design.json']))
    current_design = json.loads((ROOT / 'jobs/figure18_design.json').read_text())
    if submitted_design['pilot'] != current_design['pilot']:
        raise RuntimeError('pilot matrix differs from the submission commit')

    def stop(signum, frame):
        raise RuntimeError('signal ' + str(signum))
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    (BASE / 'loop.pid').write_text(str(os.getpid()) + '\n')
    try:
        if args.batch:
            job = os.environ.get('SLURM_JOB_ID', '')
            if not job.isdigit():
                raise RuntimeError('--batch requires a numeric SLURM_JOB_ID')
        else:
            request = subprocess.run(['salloc', '--immediate=30', '-A', 'm5320_g', '-C', 'gpu&hbm40g',
                                      '-q', 'interactive', '-N', '4', '-t', '04:00:00',
                                      '--gpus-per-node=4', '--no-shell', '-J', 'pp_n2count'],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     universal_newlines=True, timeout=60)
            say(request.stdout.strip())
            match = re.search(r'Granted job allocation (\d+)', request.stdout)
            if match:
                job = match.group(1)
                interactive_owned = True
            if request.returncode or not job:
                raise RuntimeError('allocation request failed; no automatic retry')
        state = output(['squeue', '-h', '-j', job, '-o', '%j %T %D']).split()
        if state != ['pp_n2count', 'RUNNING', '4']:
            raise RuntimeError('unexpected allocation: ' + repr(state))
        allocation_lock = (BASE / ('allocation.' + job + '.lock')).open('w')
        fcntl.flock(allocation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        hosts = output(['scontrol', 'show', 'hostnames', output(['squeue', '-h', '-j', job, '-o', '%N'])]).splitlines()
        directory = BASE / job
        directory.mkdir(exist_ok=False)
        (BASE / 'allocation.id').write_text(job + '\n')
        (directory / 'plan.json').write_text(json.dumps(blocks, indent=2) + '\n')
        (directory / 'source_commit').write_text(source_commit + '\n')
        say('allocation={} loop={} hosts={} source={}'.format(job, os.getpid(), ','.join(hosts), source_commit))

        def steps():
            values = output(['squeue', '-h', '--steps', '-j', job, '-o', '%i']).splitlines()
            return [v for v in values if not v.endswith(('.extern', '.batch'))]
        if steps():
            raise RuntimeError('allocation already has an active step')

        def run(phase, circuit, cutoff, policy, seed=7, limit=1800):
            if steps():
                raise RuntimeError('previous step still active')
            binary = 'pp_engine_x' if policy in ('old', 'warmup') else 'pp_engine_n2count'
            argv = ['--gpus', '4'] + CIRCUITS[circuit].split() + [
                '--coef', cutoff, '--trace', '1', '--vram-frac', '0.9', '--seed', '7',
                '--frame', '1', '--trunc-every', '10', '--gmax', '8', '--chunk', '262144',
                '--cands', '8', '--overlap-plan', '1']
            if policy not in ('old', 'legacy', 'warmup'):
                count = policy[5:] if policy.startswith('fixed') else '0'
                argv += ['--policy-seed', str(seed), '--fixed-sectors', count, '--policy-audit', '1']
            name = '{}.{}.{}.{}.p{}.out'.format(phase, circuit, cutoff, policy, seed)
            path = directory / name
            meta = dict(allocation=job, phase=phase, circuit=circuit, cutoff=cutoff, policy=policy,
                        policy_seed=seed, binary=binary, binary_sha256=hashes[binary], args=argv,
                        source_commit=source_commit, hosts=hosts, schedule_seed=SCHEDULE_SEED)
            if digest(binaries[binary]) != hashes[binary]:
                raise RuntimeError('binary changed during allocation')
            say('start ' + name)
            with path.open('x') as log:
                log.write('# count_run ' + json.dumps(meta, sort_keys=True) + '\n')
                log.flush()
                cmd = ['timeout', '--signal=TERM', '--kill-after=30', str(limit),
                       'srun', '--jobid=' + job, '--nodelist=' + ','.join(hosts), '--exclusive', '--exact',
                       '-N', '4', '-n', '4', '--ntasks-per-node=1', '-c', '128',
                       '--gpus-per-node=4', '--cpu-bind=none', 'stdbuf', '-oL', str(binaries[binary])] + argv
                code = subprocess.call(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                state = output(['squeue', '-h', '-j', job, '-o', '%T'])
                if state != 'RUNNING':
                    raise RuntimeError('allocation ended; incomplete run kept without exit marker')
                log.write('exit={}\n'.format(code))
            if steps():
                raise RuntimeError('unfinished step after run; aborting allocation')
            row = read_run(path)
            say('finish {} status={} wall={} expectation={}'.format(name, row['status'], row['wall'], row['expectation']))
            if row['status'] == 'invalid':
                raise RuntimeError('invalid measurement: ' + str(row.get('validation_error')))
            return row

        warmup = run('warmup', 'ki', '1e-4', 'warmup', limit=180)
        if warmup['status'] != 'complete':
            raise RuntimeError('warmup failed')
        for circuit in CIRCUITS:
            old = run('compat', circuit, '1e-4', 'old', limit=180)
            legacy = run('compat', circuit, '1e-4', 'legacy', limit=180)
            check_compat(old, legacy)
            matched = run('inputcheck', circuit, '1e-4', 'adaptive', limit=180)
            check_answers(old, matched)
            say('compatibility and matched-stream input check passed: ' + circuit)
        (directory / 'compatibility.ok').write_text('Seven circuits: legacy traces and matched-stream answers/windows passed.\n')
        # Admit whole blocks using conservative prior-time estimates plus cleanup.
        # The timeout is an upper bound, not a promise that every block will fit.
        for block in blocks:
            cutoff = block['cutoff']
            limit = 180 if cutoff == '1e-4' else 1800
            # Reserve observed adaptive time * 3 plus launch time per large run.
            # A timeout or a slower-than-estimated block remains explicit, not a speedup.
            estimates = {'ki': 120, 'tfim': 132, 'xxz': 242}
            block_budget = 6 * (30 if cutoff == '1e-4' else min(limit, estimates[block['circuit']] * 3 + 30)) + 300
            if remaining_seconds(job) < block_budget:
                say('insufficient allocation time for complete block: ' + json.dumps(block))
                break
            group = [run('pilot', block['circuit'], cutoff, policy, block['policy_seed'], limit)
                     for policy in block['order']]
            if all(r['status'] == 'complete' for r in group):
                say('validated block ' + json.dumps(check_block(group), sort_keys=True))
            else:
                say('block contains failed runs; excluded from full-policy speedups')
            result = summarize(directory)
            (directory / 'progress.json').write_text(json.dumps(result, indent=2) + '\n')
        say('pilot allocation finished; full campaign requires evidence review')
    finally:
        if directory is not None:
            try:
                result = summarize(directory)
                (ROOT / 'analysis' / ('paper_n2count.' + job + '.json')).write_text(json.dumps(result, indent=2) + '\n')
            except Exception as exc:
                say('summary error: ' + repr(exc))
        if job and interactive_owned:
            # Interactive allocations are explicitly released; batch exits release
            # their allocation normally, preserving COMPLETED/FAILED accounting.
            try:
                if output(['squeue', '-h', '-j', job, '-o', '%j']) == 'pp_n2count':
                    subprocess.run(['scancel', job], timeout=30, check=True)
                    say('released allocation ' + job)
            except Exception as exc:
                say('allocation cleanup requires inspection: ' + repr(exc))
        pid = BASE / 'loop.pid'
        if pid.exists() and pid.read_text().strip() == str(os.getpid()):
            pid.unlink()


if __name__ == '__main__':
    main()
