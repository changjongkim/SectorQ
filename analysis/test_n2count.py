#!/usr/bin/env python3
"""Reject invalid causal comparisons before the new count-policy pilot runs."""
import copy
import math
import unittest
from paper_n2count import POLICIES, check_block, check_compat, validate_count


def row(policy):
    fixed = int(policy[5:]) if policy.startswith('fixed') else 0
    g = int(math.log2(fixed)) if fixed else 4
    counts = [0, g, g if fixed else 3]
    trace, audit = [], []
    for i in range(3):
        nxt = counts[i + 1] if i < 2 else -1
        trace.append(dict(window=i, rows=counts[i], next_rows=nxt, valid_rows=8,
                          terms_in=10 ** i, terms_out=10 ** (i + 1),
                          sectors=1 if i == 0 else min(2 ** counts[i], 10 ** i),
                          largest_sector_in=1 if i == 0 else 10))
        audit.append(dict(window=str(i), rows=str(counts[i]), labels=str(2 ** counts[i]),
                          row_rank=str(counts[i]), row_hash='0000000000000000',
                          adaptive_next_rows=str([4, 3, -1][i]), requested_next_rows=str(nxt),
                          next_fixed_active=str(int(bool(fixed)))))
    return dict(meta=dict(policy=policy, allocation='123', binary_sha256='abc', circuit='ki',
                          cutoff='1e-4', policy_seed=7, phase='pilot',
                          args=['--seed', '7', '--fixed-sectors', str(fixed)]),
                trace=trace, audit=audit, status='complete', expectation=0.25, wall=1.0,
                schedule=[(str(i), str(i * 3), str(i * 3 + 3)) for i in range(3)])


class AuditTest(unittest.TestCase):
    def test_valid_activation_and_block(self):
        rows = [row(p) for p in POLICIES]
        for r in rows:
            validate_count(r)
        self.assertEqual(check_block(rows)['activation_window'], 0)

    def test_reject_answer_and_provenance(self):
        for key, value in [('allocation', 'other'), ('binary_sha256', 'other'),
                           ('policy_seed', 8), ('cutoff', '3e-6')]:
            rows = [row(p) for p in POLICIES]
            rows[-1]['meta'][key] = value
            with self.assertRaises(ValueError):
                check_block(rows)
        rows = [row(p) for p in POLICIES]
        rows[-1]['expectation'] += 1e-5
        with self.assertRaises(ValueError):
            check_block(rows)

    def test_reject_bootstrap_and_schedules(self):
        rows = [row(p) for p in POLICIES]
        rows[-1]['audit'][0]['row_hash'] = '0000000000000001'
        with self.assertRaises(ValueError):
            check_block(rows)
        rows = [row(p) for p in POLICIES]
        rows[-1]['schedule'][0] = ('0', '0', '4')
        with self.assertRaises(ValueError):
            check_block(rows)

    def test_reject_wrong_fixed_semantics(self):
        r = row('fixed64')
        r['audit'][1]['requested_next_rows'] = '3'
        with self.assertRaises(AssertionError):
            validate_count(r)
        r = row('fixed64')
        r['audit'][1]['next_fixed_active'] = '0'
        with self.assertRaises(AssertionError):
            validate_count(r)

    def test_compat_and_failure(self):
        a = row('adaptive'); b = copy.deepcopy(a)
        check_compat(a, b)
        b['trace'][1]['terms_in'] += 1
        with self.assertRaises(ValueError):
            check_compat(a, b)
        rows = [row(p) for p in POLICIES]
        rows[-1]['status'] = 'failed'
        with self.assertRaises(ValueError):
            check_block(rows)


if __name__ == '__main__':
    unittest.main()
