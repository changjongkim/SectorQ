#!/usr/bin/env python3
"""Exercise sequential allocation/resume and correctness gates without Slurm."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'jobs'))
import n2_count as runner
from paper_n2count import summarize_rows
from test_n2count import row


class RunnerTest(unittest.TestCase):
    def setUp(self):
        folder = ROOT / 'logs/n2count/test-runs'
        folder.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=str(folder))
        self.root = Path(self.temp.name)
        for name in ('jobs', 'analysis', 'bin', 'logs/n2count'):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.spec = runner.design()
        for name in ('jobs/figure18_design.json', 'analysis/paper_n2.59842747.json'):
            (self.root / name).write_bytes((ROOT / name).read_bytes())
        self.base = self.root / 'logs/n2count'
        self.rows = {}
        self.commands = []
        self.job = '111'
        self.stop_after = 6
        self.fail_policy = None
        self.invalid_validation = False
        self.manual_cancel = False
        self.blocks = runner.plan()[:2]
        self.previous_cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self.previous_cwd)
        self.temp.cleanup()

    def output(self, cmd):
        self.commands.append(cmd)
        if cmd[0] == 'git':
            if cmd[1] == 'diff':
                return ''
            if cmd[1] == 'show':
                return json.dumps(self.spec)
        if cmd[0] == 'scontrol':
            return 'n1\nn2\nn3\nn4'
        if cmd[0] == 'squeue':
            if '--steps' in cmd:
                return ''
            return {'%j %T %D': 'pp_n2count RUNNING 4', '%N': 'n[1-4]',
                    '%T': 'RUNNING'}[cmd[-1]]
        if cmd[0] == 'sbatch':
            return '222'
        raise AssertionError(cmd)

    def read(self, path):
        meta = json.loads(path.read_text().splitlines()[0][len('# count_run '):])
        r = row(meta['policy'])
        r['meta'] = meta
        # The adaptive variant is deliberately slower: continuation must ignore this.
        r['wall'] = 2.0 if meta['policy'] == 'adaptive' else 1.0
        if meta['phase'] == 'confirmatory' and meta['policy'] == self.fail_policy:
            r.update(status='failed', wall=None, expectation=None)
        if self.invalid_validation and meta['phase'] == 'validation' and meta['policy'] == 'fixed32':
            r['expectation'] += 0.1
        self.rows[str(path)] = copy.deepcopy(r)
        return r

    def summary(self, directory):
        return summarize_rows([r for name, r in self.rows.items() if Path(name).parent == directory])

    def remaining(self, job):
        n = sum(r['meta']['phase'] == 'confirmatory' and r['meta']['allocation'] == job
                for r in self.rows.values())
        return 14000 if n < self.stop_after else 100

    def execute(self):
        handlers = {}
        test = self
        class Process:
            def __init__(self, *args, **kwargs):
                self.pid = 123
            def wait(self):
                if test.manual_cancel and len(test.rows) == 57:
                    handlers[runner.signal.SIGTERM](runner.signal.SIGTERM, None)
                return 0
            def poll(self):
                return 0
        (self.root / 'jobs/figure18_design.json').write_text(json.dumps(self.spec))
        argv = ['n2_count.py', '--batch', '--source-commit', 'commit',
                '--binary-sha256', 'pp_engine_n2count', '--reference-sha256', 'pp_engine_x',
                '--campaign', 'test']
        with patch.object(runner, 'ROOT', self.root), patch.object(runner, 'BASE', self.base), \
             patch.object(runner, 'output', self.output), \
             patch.object(runner, 'digest', lambda path: Path(path).name), \
             patch.object(runner, 'plan', lambda spec=None: self.blocks), \
             patch.object(runner, 'remaining_seconds', self.remaining), \
             patch.object(runner, 'read_run', self.read), \
             patch.object(runner, 'summarize', self.summary), \
             patch.object(runner.subprocess, 'Popen', Process), \
             patch.object(runner.signal, 'signal', lambda sig, handler: handlers.update({sig: handler})), \
             patch.dict(os.environ, {'SLURM_JOB_ID': self.job}), \
             patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
            runner.main()

    def manifest(self):
        return json.loads((self.base / 'campaign.test.json').read_text())

    def test_negative_results_continue_and_resume_without_remeasurement(self):
        self.execute()
        state = self.manifest()
        self.assertEqual((state['status'], state['finished_blocks']), ('queued_continuation', 1))
        submissions = [c for c in self.commands if c[0] == 'sbatch']
        self.assertEqual(len(submissions), 1)
        self.assertIn('--dependency=afterany:111', submissions[0])
        self.assertTrue(all(v < 1 for v in state['block_outcomes'][0]['reference_over_adaptive'].values()))
        self.job = '222'
        self.stop_after = 100
        self.execute()
        self.assertEqual(self.manifest()['status'], 'complete')
        counts = [r for r in self.rows.values() if r['meta']['phase'] == 'confirmatory']
        self.assertEqual(len(counts), 12)
        self.assertEqual(len([c for c in self.commands if c[0] == 'sbatch']), 1)

    def test_partial_block_restarts_all_policies_on_one_allocation(self):
        self.stop_after = 3
        self.execute()
        self.assertEqual(self.manifest()['finished_blocks'], 0)
        self.job = '222'
        self.stop_after = 100
        self.execute()
        self.assertEqual(self.manifest()['status'], 'complete')
        counts = [r for r in self.rows.values() if r['meta']['phase'] == 'confirmatory']
        self.assertEqual(len(counts), 15)  # retain three partial records, plus two full blocks
        self.assertEqual(len([r for r in counts if r['meta']['allocation'] == '222']), 12)

    def test_measured_failure_is_preserved_and_not_retried(self):
        self.fail_policy = 'fixed16'
        self.execute()
        self.assertEqual(self.manifest()['block_outcomes'][0]['status'], 'failed')
        self.job = '222'
        self.stop_after = 100
        self.execute()
        self.assertEqual(self.manifest()['finished_blocks'], 2)
        self.assertEqual(len([r for r in self.rows.values() if r['meta']['phase'] == 'confirmatory']), 12)

    def test_correctness_failure_blocks_performance_and_submission(self):
        self.invalid_validation = True
        with self.assertRaises(ValueError):
            self.execute()
        self.assertEqual(self.manifest()['status'], 'halted')
        self.assertFalse(any(r['meta']['phase'] == 'confirmatory' for r in self.rows.values()))
        self.assertFalse(any(c[0] == 'sbatch' for c in self.commands))

    def test_allocation_limit_stops_submission(self):
        self.spec['execution_flow']['max_allocations'] = 1
        self.execute()
        self.assertEqual(self.manifest()['status'], 'allocation_limit')
        self.assertEqual(self.manifest()['remaining_blocks'], 1)
        self.assertFalse(any(c[0] == 'sbatch' for c in self.commands))

    def test_no_progress_does_not_queue_repeated_empty_allocations(self):
        self.stop_after = 0
        # Let validation run, but deny the first confirmatory block.
        normal_remaining = self.remaining
        self.remaining = lambda job: 100 if len(self.rows) >= 57 else 14000
        self.execute()
        self.assertEqual(self.manifest()['status'], 'halted')
        self.assertFalse(any(c[0] == 'sbatch' for c in self.commands))
        self.remaining = normal_remaining

    def test_manual_cancellation_does_not_respawn_campaign(self):
        self.manual_cancel = True
        self.execute()
        self.assertEqual(self.manifest()['status'], 'halted')
        self.assertFalse(any(c[0] == 'sbatch' for c in self.commands))

    def test_full_plan_is_predeclared_and_repeatable(self):
        blocks = runner.plan()
        self.assertEqual(blocks, runner.plan())
        self.assertEqual(len(blocks), 105)
        self.assertEqual(len(set(runner.block_key(b) for b in blocks)), 105)
        self.assertEqual(sum(len(b['order']) for b in blocks), 630)
        self.assertEqual(set(b['policy_seed'] for b in blocks), {11, 23, 37, 53, 71})
        for b in blocks:
            self.assertEqual(set(b['order']), set(runner.POLICIES))


if __name__ == '__main__':
    unittest.main()
