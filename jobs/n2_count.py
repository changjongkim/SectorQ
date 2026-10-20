#!/usr/bin/env python3
"""Correctness gates followed by the full count-policy campaign on four nodes.

Whole paired blocks remain within one allocation. Normal time-budget exits
submit one dependent continuation, with a finite allocation limit. Timing wins
are never an admission condition. Failed policies remain measured failures.
"""
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
DESIGN_KEYS = ('controls', 'policies', 'validation', 'confirmatory', 'execution_flow')


class AllocationEnding(Exception):
    pass


def say(message):
    print(datetime.datetime.utcnow().isoformat() + 'Z ' + message, flush=True)


def output(args):
    return subprocess.check_output(args, cwd=str(ROOT), universal_newlines=True, timeout=30).strip()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def design():
    return json.loads((ROOT / 'jobs/figure18_design.json').read_text())


def plan(spec=None):
    spec = spec or design()
    blocks = []
    suite = spec['confirmatory']
    for seed in suite['paired_block_policy_seeds']:
        workloads = [(c, x) for c in suite['circuits'] for x in suite['cutoffs']]
        random.Random('{}:blocks:{}'.format(SCHEDULE_SEED, seed)).shuffle(workloads)
        for circuit, cutoff in workloads:
            order = list(POLICIES)
            random.Random('{}:{}:{}:{}'.format(SCHEDULE_SEED, seed, circuit, cutoff)).shuffle(order)
            blocks.append(dict(circuit=circuit, cutoff=cutoff, policy_seed=seed, order=order))
    assert sum(len(b['order']) for b in blocks) == suite['measured_runs'] == 630
    assert len({block_key(b) for b in blocks}) == 105
    return blocks


def block_key(block):
    return (block['circuit'], block['cutoff'], block['policy_seed'])


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def completed_blocks(results):
    """A partial block never contributes members to a later allocation's block."""
    finished = {}
    for result in results:
        if any(r['status'] == 'invalid' for r in result['rows']):
            raise ValueError('invalid earlier run; campaign requires inspection')
        for block in result['blocks']:
            if block.get('status') == 'invalid':
                raise ValueError('invalid earlier block; campaign requires inspection')
            if block.get('phase') != 'confirmatory':
                continue
            if block['status'] in ('complete', 'failed'):
                key = block_key(block)
                if key in finished:
                    raise ValueError('duplicate finished block across allocations: ' + repr(key))
                finished[key] = block
    return finished


def block_budget(block, prior_results):
    """Admission only: estimate all six policies, independent of speedup sign."""
    prior = json.loads((ROOT / 'analysis/paper_n2.59842747.json').read_text())
    old = next(r['wall'] for r in prior['rows'] if r['mode'] == 're'
               and r['circuit'] == block['circuit'] and r['cutoff'] == block['cutoff'])
    ceiling = design()['confirmatory']['per_run_timeout_seconds']
    costs = {p: min(ceiling, max(30, old * 3)) for p in POLICIES}
    for result in prior_results:
        for row in result['rows']:
            meta = row['meta']
            if meta['phase'] != 'confirmatory' or meta['circuit'] != block['circuit'] or meta['cutoff'] != block['cutoff']:
                continue
            if row['status'] == 'complete':
                costs[meta['policy']] = max(costs[meta['policy']], min(ceiling, row['wall'] * 1.5))
            elif row['status'] == 'failed':
                costs[meta['policy']] = ceiling
    return sum(costs.values()) + 6 * 30 + 300


def continuation_command(job, campaign, source_commit, hashes):
    exports = ['ALL', 'PP_N2_SOURCE_COMMIT=' + source_commit,
               'PP_N2_BINARY_SHA256=' + hashes['pp_engine_n2count'],
               'PP_N2_REFERENCE_SHA256=' + hashes['pp_engine_x'],
               'PP_N2_CAMPAIGN=' + campaign]
    return ['sbatch', '--parsable', '--dependency=afterany:' + job,
            '--export=' + ','.join(exports), 'jobs/n2_count_batch.sh']


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
    parser.add_argument('--campaign', help='stable identifier shared by sequential allocations')
    args = parser.parse_args()
    spec = design()
    blocks = plan(spec)
    if args.dry_run:
        print(json.dumps(dict(warmups=1, compatibility_runs=14, policy_validation_runs=42,
                             confirmatory_runs=630, blocks=blocks,
                             execution_flow=spec['execution_flow']), indent=2))
        return
    if not args.campaign or not re.fullmatch(r'[a-zA-Z0-9_-]+', args.campaign):
        parser.error('--campaign must be a simple stable identifier')
    os.chdir(str(ROOT))
    BASE.mkdir(exist_ok=True)
    lock = (BASE / 'loop.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    job = None
    allocation_lock = None
    interactive_owned = False
    directory = None
    active = None
    stopping = False
    stop_signal = None
    validation_passed = False
    history = []
    manifest_path = BASE / ('campaign.' + args.campaign + '.json')
    manifest = None
    binaries = {name: ROOT / 'bin' / name for name in ('pp_engine_x', 'pp_engine_n2count')}
    hashes = {name: digest(path) for name, path in binaries.items()}
    source_commit = args.source_commit or output(['git', 'rev-parse', 'HEAD'])
    clean_paths = ['src/engine/pp_engine.cpp', 'src/engine/count_policy.hpp',
                   'src/engine/circuit.hpp', 'src/engine/gf2.hpp', 'src/engine/noise_fusion.hpp',
                   'jobs/n2_count.py', 'jobs/n2_count.sh', 'jobs/n2_count_batch.sh', 'jobs/env.sh',
                   'analysis/paper_n2count.py', 'analysis/paper_ablation.py',
                   'analysis/paper_n2.59842747.json']
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
    if any(submitted_design[k] != current_design[k] for k in DESIGN_KEYS):
        raise RuntimeError('campaign design differs from the submission commit')
    pin = dict(source_commit=source_commit, binary_sha256=hashes,
               design_sha256=hashlib.sha256(json.dumps(
                   {k: spec[k] for k in DESIGN_KEYS}, sort_keys=True).encode()).hexdigest())
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['pin'] != pin:
            raise RuntimeError('campaign continuation has different source, binaries or design')
        if manifest['status'] in ('complete', 'halted', 'allocation_limit'):
            raise RuntimeError('campaign is closed: ' + manifest['status'])
        for entry in manifest['allocations']:
            result = summarize(BASE / entry['job'])
            for row in result['rows']:
                meta = row['meta']
                if (meta.get('campaign') != args.campaign or meta.get('allocation') != entry['job']
                        or meta.get('source_commit') != source_commit
                        or meta.get('binary_sha256') != hashes.get(meta.get('binary'))):
                    raise RuntimeError('earlier allocation provenance mismatch')
            history.append(result)
        completed_blocks(history)
    else:
        manifest = dict(campaign=args.campaign, pin=pin, status='ready', allocations=[], submissions=[])

    def stop(signum, frame):
        nonlocal stopping, stop_signal
        stopping = True
        stop_signal = signum
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
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
        if len(manifest['allocations']) >= spec['execution_flow']['max_allocations']:
            raise RuntimeError('campaign allocation limit reached')
        allocation_lock = (BASE / ('allocation.' + job + '.lock')).open('w')
        fcntl.flock(allocation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        hosts = output(['scontrol', 'show', 'hostnames', output(['squeue', '-h', '-j', job, '-o', '%N'])]).splitlines()
        directory = BASE / job
        directory.mkdir(exist_ok=False)
        (BASE / 'allocation.id').write_text(job + '\n')
        (directory / 'plan.json').write_text(json.dumps(blocks, indent=2) + '\n')
        (directory / 'source_commit').write_text(source_commit + '\n')
        manifest['allocations'].append(dict(job=job, directory=str(directory.relative_to(ROOT))))
        manifest['status'] = 'running'
        atomic_json(manifest_path, manifest)
        say('allocation={} loop={} hosts={} source={}'.format(job, os.getpid(), ','.join(hosts), source_commit))

        def steps():
            values = output(['squeue', '-h', '--steps', '-j', job, '-o', '%i']).splitlines()
            return [v for v in values if not v.endswith(('.extern', '.batch'))]
        if steps():
            raise RuntimeError('allocation already has an active step')

        def run(phase, circuit, cutoff, policy, seed=7, limit=1800):
            nonlocal active
            if stopping or remaining_seconds(job) < 210:
                raise AllocationEnding('cleanup reserve reached')
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
                        source_commit=source_commit, hosts=hosts, schedule_seed=SCHEDULE_SEED,
                        campaign=args.campaign)
            if digest(binaries[binary]) != hashes[binary]:
                raise RuntimeError('binary changed during allocation')
            say('start ' + name)
            with path.open('x') as log:
                log.write('# count_run ' + json.dumps(meta, sort_keys=True) + '\n')
                log.flush()
                allowed = min(limit, remaining_seconds(job) - 180)
                cmd = ['timeout', '--signal=TERM', '--kill-after=30', str(allowed),
                       'srun', '--jobid=' + job, '--nodelist=' + ','.join(hosts), '--exclusive', '--exact',
                       '-N', '4', '-n', '4', '--ntasks-per-node=1', '-c', '128',
                       '--gpus-per-node=4', '--cpu-bind=none', 'stdbuf', '-oL', str(binaries[binary])] + argv
                active = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
                code = active.wait()
                active = None
                if stopping or (allowed < limit and code in (124, 137)):
                    raise AllocationEnding('allocation reserve interrupted a run; keep it incomplete')
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
            validation = [run('validation', circuit, '1e-4', policy, limit=180) for policy in POLICIES]
            matched = validation[0]
            check_answers(old, matched)
            check_block(validation)
            say('legacy compatibility and all six count policies validated: ' + circuit)
        validation_passed = True
        (directory / 'compatibility.ok').write_text('Seven circuits: legacy compatibility and all six count policies passed.\n')
        finished = completed_blocks(history)
        budget_history = history
        for block in blocks:
            if block_key(block) in finished:
                continue
            if stopping or remaining_seconds(job) < block_budget(block, budget_history):
                say('insufficient allocation time for complete block: ' + json.dumps(block))
                break
            group = [run('confirmatory', block['circuit'], block['cutoff'], policy, block['policy_seed'],
                         spec['confirmatory']['per_run_timeout_seconds'])
                     for policy in block['order']]
            if all(r['status'] == 'complete' for r in group):
                say('validated block ' + json.dumps(check_block(group), sort_keys=True))
            else:
                say('block contains failed runs; excluded from full-policy speedups')
            result = summarize(directory)
            completed_blocks([result])  # Detect answer mismatches even when another policy failed.
            budget_history = history + [result]
            finished = completed_blocks(budget_history)
            (directory / 'progress.json').write_text(json.dumps(result, indent=2) + '\n')
        say('confirmatory allocation finished; no performance-based stopping rule')
    except AllocationEnding as exc:
        say(str(exc))
        deadline_signal = not stopping
        if stopping:
            try:
                # A manual cancellation must not silently respawn the campaign.
                deadline_signal = (stop_signal == signal.SIGTERM
                                   and output(['squeue', '-h', '-j', job, '-o', '%T']) == 'RUNNING'
                                   and remaining_seconds(job) <= 180)
            except Exception:
                deadline_signal = False
        if not validation_passed or not deadline_signal:
            manifest['status'] = 'halted'
    except BaseException:
        if manifest is not None:
            manifest['status'] = 'halted'
        raise
    finally:
        # Preserve the allocation-local samples before deciding on continuation.
        if directory is not None:
            try:
                result = summarize(directory)
                (ROOT / 'analysis' / ('paper_n2count.' + job + '.json')).write_text(json.dumps(result, indent=2) + '\n')
                finished = completed_blocks(history + [result])
                manifest['finished_blocks'] = len(finished)
                manifest['remaining_blocks'] = len(blocks) - len(finished)
                manifest['block_outcomes'] = list(finished.values())
                if manifest['status'] != 'halted' and validation_passed:
                    if len(finished) == len(blocks):
                        manifest['status'] = 'complete'
                    elif len(manifest['allocations']) >= spec['execution_flow']['max_allocations']:
                        manifest['status'] = 'allocation_limit'
                    elif not any(r['meta']['phase'] == 'confirmatory' for r in result['rows']):
                        manifest['status'] = 'halted'
                        manifest['error'] = 'No confirmatory progress; inspect block admission before resubmission.'
                    elif args.batch:
                        manifest['status'] = 'submitting_continuation'
                        atomic_json(manifest_path, manifest)
                        command = continuation_command(job, args.campaign, source_commit, hashes)
                        submitted = output(command)
                        next_job = submitted.split(';')[0]
                        if not next_job.isdigit():
                            raise RuntimeError('unrecognized sbatch receipt: ' + submitted)
                        manifest['submissions'].append(dict(job=next_job, after=job, command=command))
                        manifest['status'] = 'queued_continuation'
                        say('queued dependent four-node continuation ' + next_job)
                    else:
                        manifest['status'] = 'needs_continuation'
            except Exception as exc:
                manifest['status'] = 'halted'
                manifest['error'] = repr(exc)
                say('summary error: ' + repr(exc))
            atomic_json(manifest_path, manifest)
            say('campaign status={} finished={} remaining={}'.format(
                manifest['status'], manifest.get('finished_blocks'), manifest.get('remaining_blocks')))
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
        if allocation_lock is not None:
            allocation_lock.close()
        lock.close()


if __name__ == '__main__':
    main()
