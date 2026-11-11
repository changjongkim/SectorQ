#!/usr/bin/env python3
"""Sector-count campaign (jobs/figure18_design.json, campaign full-20261109T060832Z): adaptive against fixed counts.

Reads the 105 confirmatory blocks of the campaign manifest (7 circuits x 3 cutoffs x 5 policy seeds; every block holds
the six policies measured together in one allocation) and applies the analysis the design declared before the runs:
  - per workload (circuit, cutoff): the mean time of each policy over its five blocks;
  - references: fixed16 (one nominal task per GPU) and the fastest tested fixed count of that workload (a
    retrospective, workload-tuned reference, not a deployable setting);
  - ratio = mean reference time / mean adaptive time (above 1: adaptive is faster), all 21 workloads;
  - aggregate: equal-weight geometric mean over the 21 workloads, wins and losses, the worst slowdown;
  - uncertainty: paired bootstrap over the five seed blocks, resampled jointly across the workloads, with the best
    fixed count chosen again in every replicate.

  ../env/bin/python3 analysis/paper_n2count_summary.py      # writes analysis/paper_n2count_summary.json
"""
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = 'full-20261109T060832Z'
FIXED = ('fixed16', 'fixed32', 'fixed64', 'fixed128', 'fixed256')
CIRCUITS = ('ki', 'tfim', 'noise', 'tfimsq', 'xxz', 'qaoa', 'hea')
CUTOFFS = ('1e-4', '1e-5', '3e-6')
SEEDS = (11, 23, 37, 53, 71)
REPLICATES = 10000


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def load():
    manifest = json.loads((ROOT / 'logs' / 'n2count' / ('campaign.%s.json' % CAMPAIGN)).read_text())
    assert manifest['status'] == 'complete', manifest['status']
    walls = {}
    for b in manifest['block_outcomes']:
        assert b['status'] == 'complete' and b['phase'] == 'confirmatory', b
        key = (b['circuit'], b['cutoff'], b['policy_seed'])
        assert key not in walls, key
        assert set(b['wall']) == {'adaptive'} | set(FIXED), key
        walls[key] = b['wall']
    assert len(walls) == len(CIRCUITS) * len(CUTOFFS) * len(SEEDS)
    return manifest, walls


def ratios(walls, seeds):
    """Per workload: means over the given seed blocks (a seed may repeat), the best fixed count, both ratios."""
    out = {}
    for c in CIRCUITS:
        for x in CUTOFFS:
            mean = {p: sum(walls[c, x, s][p] for s in seeds) / len(seeds) for p in ('adaptive',) + FIXED}
            best = min(FIXED, key=lambda p: mean[p])
            out[c, x] = dict(mean=mean, best=best, fixed16=mean['fixed16'] / mean['adaptive'], best_fixed=mean[best] / mean['adaptive'])
    return out


def mechanism(manifest):
    """Mean GPU splits, host splits and summed worker idle seconds per policy and workload, from the runs of the
    complete blocks (analysis/paper_n2count.<allocation>.json; a partial block was measured again in a later one)."""
    acc = {}
    for a in manifest['allocations']:
        data = json.loads((ROOT / 'analysis' / ('paper_n2count.%s.json' % a['job'])).read_text())
        done = {(b['circuit'], b['cutoff'], b['policy_seed']) for b in data['blocks'] if b['status'] == 'complete' and b['phase'] == 'confirmatory'}
        for r in data['rows']:
            m = r['meta']
            if m.get('phase') == 'confirmatory' and (m['circuit'], m['cutoff'], m['policy_seed']) in done:
                acc.setdefault((m['circuit'], m['cutoff'], m['policy']), []).append(r)
    out = {}
    for (c, x, p), rs in acc.items():
        assert len(rs) == len(SEEDS), (c, x, p, len(rs))
        out.setdefault('%s %s' % (c, x), {})[p] = {k: sum(r[k] for r in rs) / len(rs) for k in ('gpu_splits', 'host_splits', 'idle')}
    return out


def main():
    manifest, walls = load()
    mech = mechanism(manifest)
    point = ratios(walls, SEEDS)
    rng = random.Random(20261110)
    boot = {'fixed16': [], 'best_fixed': []}
    for _ in range(REPLICATES):
        seeds = [rng.choice(SEEDS) for _ in SEEDS]
        r = ratios(walls, seeds)
        for k in boot:
            boot[k].append(geomean([v[k] for v in r.values()]))

    def interval(values):
        v = sorted(values)
        return [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]]

    aggregate = {}
    for k in ('fixed16', 'best_fixed'):
        vals = [v[k] for v in point.values()]
        aggregate[k] = dict(geomean=geomean(vals), interval95=interval(boot[k]), wins=sum(v > 1 for v in vals),
                            losses=sum(v < 1 for v in vals), worst=min(vals), best=max(vals),
                            by_cutoff={x: geomean([point[c, x][k] for c in CIRCUITS]) for x in CUTOFFS})
    workloads = []
    for (c, x), v in point.items():
        blocks = {p: [walls[c, x, s][p] for s in SEEDS] for p in ('adaptive',) + FIXED}
        workloads.append(dict(circuit=c, cutoff=x, mean=v['mean'], best_fixed_policy=v['best'], fixed16_over_adaptive=v['fixed16'],
                              best_fixed_over_adaptive=v['best_fixed'], blocks=blocks))
    result = dict(campaign=CAMPAIGN, allocations=[a['job'] for a in manifest['allocations']], blocks=len(walls), runs=6 * len(walls),
                  bootstrap=dict(replicates=REPLICATES, seed=20261110, scope='five seed blocks resampled jointly across workloads'),
                  aggregate=aggregate, best_fixed_counts={p: sum(w['best_fixed_policy'] == p for w in workloads) for p in FIXED},
                  workloads=workloads, mechanism=mech)
    (ROOT / 'analysis' / 'paper_n2count_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    for k, a in aggregate.items():
        print('%-10s geomean %.3f [%.3f, %.3f]  wins %d losses %d  worst %.3f best %.3f  by cutoff %s'
              % (k, a['geomean'], a['interval95'][0], a['interval95'][1], a['wins'], a['losses'], a['worst'], a['best'],
                 ' '.join('%s:%.3f' % (x, v) for x, v in a['by_cutoff'].items())))
    print('best fixed count per workload:', result['best_fixed_counts'])
    for k in ('fixed16', 'best_fixed'):
        a = aggregate[k]
        print('%-10s adaptive time / reference time: %.3f (geomean), at most %.3f (worst workload)' % (k, 1 / a['geomean'], 1 / a['worst']))
        for x in CUTOFFS:
            vals = [point[c, x][k] for c in CIRCUITS]
            print('   cutoff %s: adaptive faster in %d, slower in %d of %d' % (x, sum(v > 1 for v in vals), sum(v < 1 for v in vals), len(vals)))
    split16 = {c: (mech['%s 3e-6' % c]['fixed16']['gpu_splits'], mech['%s 3e-6' % c]['adaptive']['gpu_splits']) for c in CIRCUITS}
    print('GPU splits per run at 3e-6, fixed16 vs adaptive:', {c: '%.0f/%.0f' % v for c, v in split16.items()})
    for w in workloads:
        k = mech['%s %s' % (w['circuit'], w['cutoff'])]
        print('%-7s %-5s adaptive %8.2f s  fixed16 %.3f  best %-8s %.3f | splits gpu/host, idle s: adaptive %.0f/%.0f %.0f  fixed16 %.0f/%.0f %.0f  fixed256 %.0f/%.0f %.0f'
              % (w['circuit'], w['cutoff'], w['mean']['adaptive'], w['fixed16_over_adaptive'], w['best_fixed_policy'], w['best_fixed_over_adaptive'],
                 *(k[p][f] for p in ('adaptive', 'fixed16', 'fixed256') for f in ('gpu_splits', 'host_splits', 'idle'))))


if __name__ == '__main__':
    main()
