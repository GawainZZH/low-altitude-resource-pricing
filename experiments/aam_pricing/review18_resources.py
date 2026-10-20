"""Exact two-stage FIFO example; no general multi-resource solver claims."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

CAPACITY = (2/3, 1/3)
QUALITY = (10., 12.)
COST = (3., 1.)
POLICIES = {'NC': (0., 0.), 'scalar': (1., 1.), 'vector': (0., 3.)}
OWNED = (
    'experiments/aam_pricing/review18_resources.py',
    'tests/test_review18_resources.py',
    'scripts/run_review18_resources.py',
    'docs/review18_resources_design.md',
    'docs/review18_resources_report.md',
)


def _pair(values, nonnegative=False):
    if len(values) != 2 or any(not math.isfinite(x) for x in values):
        raise ValueError('Exactly two finite values required')
    if nonnegative and min(values) < 0:
        raise ValueError('Nonnegative values required')
    return tuple(float(x) for x in values)


def _fares(p):
    p = _pair(p, True)
    if max(p) > 7:
        raise ValueError('Closed formula is proved only on [0,7]^2')
    return p


def resource_work(y):
    return [y[0] + y[1], y[1]]


def choice(p, wait):
    u = (0., QUALITY[0]-p[0]-wait[0], QUALITY[1]-p[1]-wait[0]-wait[1])
    peak = max(u)
    z = [math.exp(v-peak) for v in u]
    total = sum(z)
    return [z[1]/total, z[2]/total]


def response(p):
    p = _fares(p)
    difference = 2 + p[0] - p[1]
    if difference >= 0:
        y = [1/3, 1/3]
        wait = [10-p[0], difference]
    else:
        ratio = math.exp(difference)
        y = [(2/3)/(1+ratio), (2/3)*ratio/(1+ratio)]
        wait = [math.log((math.exp(10-p[0])+math.exp(12-p[1]))/2), 0.]
    return {'y': y, 'wait': wait, 'resource': resource_work(y)}


def _decreasing_root(function):
    lo, hi = 0., 1.
    if function(lo) <= 0:
        return 0.
    while function(hi) > 0:
        hi *= 2
        if hi > 1024:
            raise ArithmeticError('Clearing root outside bounded diagnostic range')
    for _ in range(65):
        mid = (lo+hi)/2
        if function(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def independent_response(p):
    """Nested resource roots from original logit, not the closed formula."""
    p = _fares(p)

    def first_wait(w2):
        return _decreasing_root(lambda w1: sum(choice(p, (w1, w2)))-CAPACITY[0])

    w2 = _decreasing_root(lambda w: choice(p, (first_wait(w), w))[1]-CAPACITY[1])
    w1 = first_wait(w2)
    y = choice(p, (w1, w2))
    return {'y': y, 'wait': [w1, w2], 'resource': resource_work(y)}


def residuals(p, r):
    return {
        'logit': max(abs(a-b) for a, b in zip(choice(p, r['wait']), r['y'])),
        'capacity_violation': max(0., *(r['resource'][b]-CAPACITY[b] for b in range(2))),
        'negative_wait': max(0., *(-w for w in r['wait'])),
        'complementarity': max(abs(r['wait'][b]*(CAPACITY[b]-r['resource'][b]))
                               for b in range(2)),
    }


def effective_costs(tariff):
    t = _pair(tariff, True)
    return [3+t[0], 1+t[0]+t[1]]


def accounts(p, tariff):
    p = _fares(p)
    t = _pair(tariff, True)
    r = response(p)
    y, w = r['y'], r['wait']
    bill = [t[0], t[0]+t[1]]
    profit = [(p[m]-COST[m]-bill[m])*y[m] for m in range(2)]
    revenue = sum(t[b]*r['resource'][b] for b in range(2))
    utilities = [0., 10-p[0]-w[0], 12-p[1]-w[0]-w[1]]
    peak = max(utilities)
    cs = peak + math.log(sum(math.exp(u-peak) for u in utilities))
    return {**r, 'prices': list(p), 'tariff': list(t), 'charge_per_service': bill,
            'profit': profit, 'revenue': revenue, 'CS': cs,
            'TS': cs+sum(profit)+revenue,
            'waiting_person_time': sum(y)*w[0]+y[1]*w[1]}


def wait_jacobian_determinant(y):
    # det(A)=1 and det(diag(y)-y*y')=y0*y1*y2 for unit potential demand.
    return (1-sum(y))*y[0]*y[1]


def common_wait_choice(p, wait):
    total = sum(wait)
    u = [0., 10-p[0]-total, 12-p[1]-total]
    peak = max(u)
    z = [math.exp(v-peak) for v in u]
    return [z[1]/sum(z), z[2]/sum(z)]


def common_wait_rank():
    return 1


def global_nash_proof(tariff):
    """Return the domain of the analytical global proof, not a grid claim."""
    d = effective_costs(tariff)
    if d[0] < 3 or max(d) >= 7:
        raise ValueError('Proof needs d1>=3 and both effective costs strictly below 7')
    return {'global_full_fare_interval': True, 'fare_interval': [0., 7.],
            'equilibrium': [7., 7.], 'effective_costs': d,
            'firm1_second_stage_release_fare': 5.,
            'firm1_slack_branch_derivative_factor_lower': 0.,
            'firm1_active_branch_profit_derivative': 1/3,
            'firm2_all_fares_profit_derivative': 1/3,
            'withdrawal_strictly_worse': True,
            'firm1_strictly_dominant_cap_when_participating': True,
            'unique_equilibrium_including_withdrawal': True,
            'source': 'Analytical derivative proof in review18_resources_report.md'}


def source_hashes(root):
    return {name: hashlib.sha256((Path(root)/name).read_bytes()).hexdigest()
            for name in OWNED if (Path(root)/name).is_file()}


def write_results(root, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new review18_resources output directory')
    output.mkdir(parents=True)
    comparisons = {name: accounts((7., 7.), t) for name, t in POLICIES.items()}
    largest_residual = 0.
    independent_error = 0.
    closure_rows = []
    for i in range(11):
        for j in range(11):
            p = (.7*i, .7*j)
            r, check = response(p), independent_response(p)
            error = max(abs(a-b) for key in ('y', 'wait', 'resource')
                        for a, b in zip(r[key], check[key]))
            res = max(residuals(p, r).values())
            independent_error = max(independent_error, error)
            largest_residual = max(largest_residual, res)
            closure_rows.append([*p, *r['y'], *r['wait'], *r['resource'], res, error])
    deviations = []
    max_gains = {}
    for name, tariff in POLICIES.items():
        gains = []
        for m in range(2):
            values = []
            for k in range(281):
                p = [7., 7.]
                p[m] = k/40
                r = accounts(p, tariff)
                gain = r['profit'][m]-comparisons[name]['profit'][m]
                values.append(gain)
                deviations.append([name, m+1, p[m], *r['y'], *r['wait'],
                                   *r['resource'], r['profit'][m], gain])
            gains.append(max(values))
        max_gains[name] = gains
    with (output/'closure_grid.csv').open('w', newline='', encoding='ascii') as f:
        writer = csv.writer(f)
        writer.writerow(['p1', 'p2', 'y1', 'y2', 'w1', 'w2', 'R1', 'R2',
                         'complementarity_logit_max_residual', 'independent_max_error'])
        writer.writerows(closure_rows)
    with (output/'global_deviation_curves.csv').open('w', newline='', encoding='ascii') as f:
        writer = csv.writer(f)
        writer.writerow(['policy', 'platform', 'deviation_fare', 'y1', 'y2', 'w1',
                         'w2', 'R1', 'R2', 'own_profit', 'gain_from_equilibrium'])
        writer.writerows(deviations)
    summary = {
        'design': {'A_resource_rows': [[1, 1], [0, 1]], 'capacity': CAPACITY,
                   'quality_net_of_fixed_processing_time': QUALITY, 'cost': COST,
                   'potential_demand': 1, 'outside_utility': 0, 'logit_scale': 1,
                   'fare_box': [[0, 7], [0, 7]], 'waiting_value': 1,
                   'quality_reliability_feedback': False,
                   'hard_resource_constraints': 2, 'optimized_tariff_menus': 0},
        'comparisons': comparisons,
        'global_proofs': {n: global_nash_proof(t) for n, t in POLICIES.items()},
        'diagnostics': {'closure_grid_points': len(closure_rows),
                        'actual_deviation_points': len(deviations),
                        'max_logit_complementarity_residual': largest_residual,
                        'max_independent_clearing_error': independent_error,
                        'max_actual_deviation_gain_by_platform': max_gains},
        'decision': 'Controlled two-hard-resource tariff neutrality, not multi-hard welfare validation',
        'primary_interpretation': 'One specified pooled ground access process; not arbitrary airspace aggregation',
        'source_sha256': source_hashes(root),
    }
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='ascii')
    return summary


def replay(output):
    """Read-only reproduction of saved comparison accounts and curve rows."""
    output = Path(output)
    saved = json.loads((output/'summary.json').read_text(encoding='ascii'))
    for name, tariff in POLICIES.items():
        current = accounts((7., 7.), tariff)
        if current != saved['comparisons'][name]:
            raise AssertionError(f'Changed comparison {name}')
    rows = 0
    with (output/'global_deviation_curves.csv').open(newline='', encoding='ascii') as f:
        for row in csv.DictReader(f):
            p = [7., 7.]
            m = int(row['platform'])-1
            p[m] = float(row['deviation_fare'])
            r = accounts(p, POLICIES[row['policy']])
            if abs(r['profit'][m]-float(row['own_profit'])) > 1e-12:
                raise AssertionError('Changed deviation row')
            rows += 1
    return {'replay': 'PASS', 'comparison_policies': len(POLICIES),
            'deviation_rows_recomputed': rows}
