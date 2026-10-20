"""Exact two-platform correspondence; no historical engine or optimizer calls.

The continuum argument is in docs/TRB_REVIEW16_SELECTION_RANGE_20261019.md.
Directed Decimal arithmetic bounds its constants and five frozen root pairs.
Floating-point fare comparisons are explicitly secondary diagnostics.
"""
from __future__ import annotations

from decimal import Decimal, Context, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_EVEN
import csv
import hashlib
import json
from math import exp, log, log1p
from pathlib import Path
import sys


OUTPUT = 'results/trb_review16_selection_range_v1'
CHARGES = ('1', '1.025', '1.05', '1.075', '1.1')
DOWN = Context(prec=72, rounding=ROUND_FLOOR)
UP = Context(prec=72, rounding=ROUND_CEILING)
NEAR = Context(prec=72, rounding=ROUND_HALF_EVEN)
ROOT_TOL = Decimal('1e-38')


def exact(value):
    if isinstance(value, (str, int, Decimal)):
        result = Decimal(value)
        if result.is_finite():
            return result
        raise ValueError('Finite exact decimal required')
    raise TypeError('Use strings, integers or Decimals, not binary floats')


class I:
    """Outward basic operations and adjacent correctly-rounded ln/exp."""

    def __init__(self, lo, hi=None):
        self.lo = exact(lo)
        self.hi = exact(lo if hi is None else hi)
        if self.lo > self.hi:
            raise ValueError('Reversed interval')

    def __add__(self, other):
        b = as_interval(other)
        return I(DOWN.add(self.lo, b.lo), UP.add(self.hi, b.hi))

    __radd__ = __add__

    def __neg__(self):
        return I(self.hi.copy_negate(), self.lo.copy_negate())

    def __sub__(self, other):
        return self + -as_interval(other)

    def __rsub__(self, other):
        return as_interval(other) + -self

    def __mul__(self, other):
        b = as_interval(other)
        pairs = [(a, c) for a in (self.lo, self.hi) for c in (b.lo, b.hi)]
        return I(min(DOWN.multiply(a, c) for a, c in pairs),
                 max(UP.multiply(a, c) for a, c in pairs))

    __rmul__ = __mul__

    def __truediv__(self, other):
        b = as_interval(other)
        if b.lo <= 0 <= b.hi:
            raise ValueError('Division interval contains zero')
        return self * I(DOWN.divide(Decimal(1), b.hi), UP.divide(Decimal(1), b.lo))

    def log(self):
        if self.lo <= 0:
            raise ValueError('Positive logarithm domain required')
        if self.lo == self.hi == 1:
            return I(0)
        return I(NEAR.next_minus(NEAR.ln(self.lo)), NEAR.next_plus(NEAR.ln(self.hi)))

    def exp(self):
        if self.lo == self.hi == 0:
            return I(1)
        return I(NEAR.next_minus(NEAR.exp(self.lo)), NEAR.next_plus(NEAR.exp(self.hi)))

    def serialized(self):
        return [str(self.lo), str(self.hi)]


def as_interval(value):
    return value if isinstance(value, I) else I(value)


def midpoint(value):
    return NEAR.divide(NEAR.add(value.lo, value.hi), Decimal(2))


def design():
    return dict(charges=list(CHARGES), epsilon_domain=['0', '0.1'],
                fare_bounds=['5', '44'], mass='2', capacity='1',
                resource=['1', '2'], cost=['5', '5'],
                qualities=['8+ln(2)', '9'], outside_utility='ln(5)',
                lambda_mu_v=['1', '1', '1'], ground_margin='0', penalty='0',
                scope='full_fixed_participation_Nash_set', new_engine_calls=0,
                decimal_precision=72, root_width=str(ROOT_TOL),
                source_section='app:release-selection-example (current C.3.3)',
                proof_domain='every epsilon in [0,0.1], not just frozen charges')


def capacity_state(d, epsilon):
    d, e = as_interval(d), as_interval(epsilon)
    if d.lo <= -1 or d.hi >= DOWN.divide(Decimal(2), Decimal(3)):
        raise ValueError('Positive weights require -1 < d < 2/3')
    l1, l2 = (1-I('1.5')*d).log(), (1+d).log()
    p1, p2 = 8-l1, 9-l2
    y1, y2 = (2-3*d)/(4-d), (1+d)/(4-d)
    x1, x2 = y1, 2*y2
    m1, m2 = 2-e-l1, 2-2*e-l2
    cs = 2*(8-2*d).log()
    welfare = cs+(p1-5)*y1+(p2-5)*y2
    return dict(p1=p1, p2=p2, y1=y1, y2=y2, x1=x1, x2=x2,
                m1=m1, m2=m2, cs=cs, welfare=welfare,
                profit1=m1*y1, profit2=m2*y2,
                binding1=m1*x2-1, binding2=m2*x1-1,
                free1_slack=m1*(1-y1/2)-1,
                free2_slack=m2*(1-y2/2)-1,
                welfare_d=5*((p2-5)-2*(p1-5))/((4-d)*(4-d)))


def binding_residual(platform, d, epsilon):
    if platform not in (1, 2):
        raise ValueError('Platform must be 1 or 2')
    d, e = as_interval(d), as_interval(epsilon)
    if platform == 1:
        return (2-e-(1-I('1.5')*d).log())*2*(1+d)/(4-d)-1
    return (2-2*e-(1+d).log())*(2-3*d)/(4-d)-1


def _root(platform, epsilon):
    lo, hi = (Decimal('0'), Decimal('0.1')) if platform == 1 else (Decimal('-0.1'), Decimal('0'))
    orientation = 1 if platform == 1 else -1
    left = binding_residual(platform, I(lo), I(epsilon))*orientation
    right = binding_residual(platform, I(hi), I(epsilon))*orientation
    if not (left.hi < 0 < right.lo):
        raise ArithmeticError('Root endpoint signs not proved')
    while UP.subtract(hi, lo) > ROOT_TOL:
        mid = midpoint(I(lo, hi))
        value = binding_residual(platform, I(mid), I(epsilon))*orientation
        if value.hi < 0:
            lo = mid
        elif value.lo > 0:
            hi = mid
        else:
            raise ArithmeticError('Precision insufficient to decide bisection sign')
    return I(lo, hi)


def root_brackets(epsilon):
    e = exact(epsilon)
    if not Decimal(0) <= e <= Decimal('0.1'):
        raise ValueError('Epsilon outside the proved domain')
    if e == 0:
        return I(0), I(0)
    return _root(2, e), _root(1, e)


def continuum_bounds():
    d, e = I('-0.1', '0.1'), I('0', '0.1')
    s = capacity_state(d, e)
    den2 = (4-d)*(4-d)
    derivative1 = I('1.5')/(1-I('1.5')*d)*s['x2']+10*s['m1']/den2
    derivative2 = -s['x1']/(1+d)-10*s['m2']/den2
    # Share derivatives locate exact extrema on the entire slab.
    free1 = (I('1.9')-I('1.15').log())*(1-(I(23)/41)/2)-1
    free2 = (I('1.8')-I('1.1').log())*(1-(I(11)/39)/2)-1
    odds_max = I(40).exp()
    bounds = dict(
        margin1_global_lower=I('1.9')-I('2.5').log(),
        margin2_global_lower=I('1.8')-(I(5)/3).log(),
        binding1_d=derivative1, binding2_d=derivative2,
        free1_slack=free1, free2_slack=free2,
        welfare_d=s['welfare_d'],
        root1_left=binding_residual(1, '-0.1', '0'),
        root1_right=binding_residual(1, '0.1', '0.1'),
        root2_left=binding_residual(2, '-0.1', '0.1'),
        root2_right=binding_residual(2, '0.1', '0'),
        slack_request_excess=2*I('-0.1').exp()+3*(I(7)/15).exp()-5,
        fare44_free1_gradient=1-I('37.9')/2,
        fare44_free2_gradient=1-I('36.8')*3/4,
        fare44_capacity1_gradient=1-I('37.9')*8/11,
        fare44_capacity2_gradient=1-I('36.8')/2,
        auxiliary_contraction=odds_max/(1+odds_max),
        upper_fare_z1=2*I(-36).exp(), upper_fare_z2=I(-35).exp())
    positive = ('margin1_global_lower', 'margin2_global_lower', 'binding1_d',
                'free1_slack', 'free2_slack', 'root1_right', 'root2_left',
                'slack_request_excess')
    negative = ('binding2_d', 'welfare_d', 'root1_left', 'root2_right',
                'fare44_free1_gradient', 'fare44_free2_gradient',
                'fare44_capacity1_gradient', 'fare44_capacity2_gradient')
    valid = (all(bounds[k].lo > 0 for k in positive)
             and all(bounds[k].hi < 0 for k in negative)
             and bounds['auxiliary_contraction'].hi < 1
             and bounds['upper_fare_z1'].hi < 1 and bounds['upper_fare_z2'].hi < 1)
    return dict(valid=valid, domain=dict(epsilon=['0', '0.1'], d_slab=['-0.1', '0.1']),
                bounds={k: v.serialized() for k, v in bounds.items()},
                root_nonemptiness='F1(0,e)=-e/2 <= 0; F2(0,e)=-e <= 0',
                root_direction='d_min strictly decreases; d_max strictly increases',
                analytic_steps=['positive_profit_excludes_lower_fare',
                                'left_gradients_exclude_upper_fare_at_slack_or_capacity',
                                'slack_FOC_implies_weighted_odds_contradiction',
                                'positive_wait_requires_unique_projected_auxiliary_fixed_point',
                                'auxiliary_fixed_point_above_release_has_subcapacity_requests',
                                'capacity_curve_global_margin_and_root_monotonicity',
                                'free_conditions_hold_on_entire_enclosing_slab',
                                'one_crossing_and_regime_unimodality_imply_global_Nash'])


def selection_ranges():
    release = capacity_state('0', '0')['welfare']
    rows = []
    for charge in CHARGES:
        epsilon = NEAR.subtract(Decimal(charge), Decimal(1))
        low, high = root_brackets(epsilon)
        best, worst = capacity_state(low, epsilon), capacity_state(high, epsilon)
        rows.append(dict(charge=charge, epsilon=str(epsilon),
                         d_min=low.serialized(), d_max=high.serialized(),
                         best_welfare=best['welfare'].serialized(),
                         worst_welfare=worst['welfare'].serialized(),
                         best_gain_from_release=(best['welfare']-release if epsilon else I(0)).serialized(),
                         worst_gain_from_release=(worst['welfare']-release if epsilon else I(0)).serialized(),
                         p1_min=best['p1'].serialized(), p1_max=worst['p1'].serialized(),
                         p2_min=worst['p2'].serialized(), p2_max=best['p2'].serialized(),
                         best_endpoint={k: v.serialized() for k, v in best.items()},
                         worst_endpoint={k: v.serialized() for k, v in worst.items()},
                         constant_selection_prices=['8', '9'],
                         constant_selection_welfare=release.serialized()))
    return rows


def policy_comparison(rows):
    worst = all(Decimal(a['worst_welfare'][0]) > Decimal(b['worst_welfare'][1])
                for a, b in zip(rows, rows[1:]))
    best = all(Decimal(a['best_welfare'][1]) < Decimal(b['best_welfare'][0])
               for a, b in zip(rows, rows[1:]))
    if not (worst and best):
        raise ArithmeticError('Policy rankings not separated by intervals')
    return dict(menu=[r['charge'] for r in rows],
                worst_Nash_maximizers=[rows[0]['charge']],
                optimistic_maximizers=[rows[-1]['charge']],
                constant_selection_maximizers=[r['charge'] for r in rows],
                interpretation='post_release_frozen_menu_only; not a global tariff recommendation')


def actual_allocation(p1, p2):
    """Independent scalar closure for diagnostic fare comparisons only."""
    z1, z2 = 2*exp(8-p1), exp(9-p2)
    if z1+3*z2 >= 5:
        return z1/(z1+2*z2), z2/(z1+2*z2), log((z1+3*z2)/5)
    return 2*z1/(5+z1+z2), 2*z2/(5+z1+z2), 0.


def counterchecks():
    inside, outside, directed = [], [], []
    for row in selection_ranges():
        e, charge = float(row['epsilon']), float(row['charge'])
        a, b = [sum(map(float, row[k]))/2 for k in ('d_min', 'd_max')]
        for d in (a, (a+b)/2, b):
            prices = [8-log1p(-1.5*d), 9-log1p(d)]
            service = actual_allocation(*prices)[:2]
            for i in range(2):
                baseline = (prices[i]-5-(i+1)*charge)*service[i]
                candidates = [5+39*k/256 for k in range(257)]
                candidates += [prices[i], prices[i]-1e-7, prices[i]+1e-7]
                gain = 0.
                for fare in candidates:
                    deviating = prices.copy()
                    deviating[i] = fare
                    y = actual_allocation(*deviating)[i]
                    gain = max(gain, (fare-5-(i+1)*charge)*y-baseline)
                inside.append(dict(charge=row['charge'], d=d, platform=i+1, gain=gain))
        for d, i in ((a-.0001, 1), (b+.0001, 0)):
            prices = [8-log1p(-1.5*d), 9-log1p(d)]
            baseline = (prices[i]-5-(i+1)*charge)*actual_allocation(*prices)[i]
            deviating = prices.copy()
            deviating[i] -= 1e-6
            gain = ((deviating[i]-5-(i+1)*charge)*actual_allocation(*deviating)[i]-baseline)
            outside.append(dict(charge=row['charge'], d=d, platform=i+1,
                                fare_decrease=1e-6, improvement=gain))
        for key, shift, i in (('d_min', '-0.0001', 1), ('d_max', '0.0001', 0)):
            d = I(*row[key])+I(shift)
            state = capacity_state(d, row['epsilon'])
            prices = [state['p1'], state['p2']]
            prices[i] = prices[i]-I('0.000001')
            z1, z2 = 2*(8-prices[0]).exp(), (9-prices[1]).exp()
            excess = z1+3*z2-5
            service = [z1/(z1+2*z2), z2/(z1+2*z2)][i]
            improvement = (prices[i]-5-(i+1)*I(row['charge']))*service-state['profit'+str(i+1)]
            directed.append(dict(charge=row['charge'], d=d.serialized(), platform=i+1,
                                 fare_decrease='0.000001',
                                 deviating_request_excess=excess.serialized(),
                                 profit_improvement=improvement.serialized()))
    maxgain = max(r['gain'] for r in inside)
    mingain = min(r['improvement'] for r in outside)
    directed_valid = all(Decimal(r['deviating_request_excess'][0]) > 0
                         and Decimal(r['profit_improvement'][0]) > 0 for r in directed)
    return dict(valid=maxgain < 1e-11 and mingain > 0 and directed_valid, inside_profiles=len(inside)//2,
                outside_profiles=len(outside), max_sampled_inside_gain=maxgain,
                min_exterior_improvement=mingain,
                role='diagnostic_only_not_the_global_proof',
                own_fare_grid_points=257, inside=inside, outside=outside,
                directed_exterior_witnesses=directed)


def bundle():
    proof, rows, diagnostic = continuum_bounds(), selection_ranges(), counterchecks()
    policies = policy_comparison(rows)
    valid = proof['valid'] and diagnostic['valid']
    if not valid:
        raise ArithmeticError('Proof constants or secondary diagnostics failed')
    return dict(design=design(), continuum_bounds=proof, ranges=rows,
                policy_comparison=policies, counterchecks=diagnostic,
                verification=dict(valid=valid, scope=design()['scope'], new_engine_calls=0,
                                  numerical_evidence='outward_enclosures_not_float_tolerances',
                                  theorem_evidence='continuum_derivation_in_owned_document'))


def _owned_destination(root, destination):
    owned = (Path(root)/OUTPUT).resolve()
    destination = Path(destination).resolve()
    if not destination.is_relative_to(owned):
        raise ValueError('Output must remain within the owned result directory')
    return destination


def _json_text(value):
    return json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False)+'\n'


def run(root, destination=None):
    root = Path(root).resolve()
    destination = _owned_destination(root, destination or root/OUTPUT)
    data = bundle()
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in data.items():
        (destination/(name+'.json')).write_text(_json_text(value), encoding='utf-8')
    with (destination/'ranges.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['charge', 'epsilon', 'd_min', 'd_max', 'p1_min', 'p1_max',
                  'p2_min', 'p2_max', 'worst_welfare', 'best_welfare',
                  'worst_gain_from_release', 'best_gain_from_release']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in data['ranges']:
            writer.writerow({key: json.dumps(row[key]) if isinstance(row[key], list) else row[key]
                             for key in fields})
    owned_sources = ['experiments/aam_pricing/review16_selection_range.py',
                     'scripts/run_review16_selection_range.py', 'tests/test_review16_selection_range.py',
                     'docs/TRB_REVIEW16_SELECTION_RANGE_20261019.md']
    files = [destination/(key+'.json') for key in data]+[destination/'ranges.csv']
    manifest = dict(python=sys.version, new_engine_calls=0,
                    source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest()
                                   for p in owned_sources if (root/p).exists()},
                    result_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (destination/'manifest.json').write_text(_json_text(manifest), encoding='utf-8')
    return data['verification']


def replay(root, destination=None):
    root = Path(root).resolve()
    destination = _owned_destination(root, destination or root/OUTPUT)
    fresh = bundle()
    manifest = json.loads((destination/'manifest.json').read_text(encoding='utf-8'))
    for name, expected in fresh.items():
        actual = json.loads((destination/(name+'.json')).read_text(encoding='utf-8'))
        if actual != expected:
            raise AssertionError('Replay mismatch: '+name)
    for name, digest in manifest['result_sha256'].items():
        if hashlib.sha256((destination/name).read_bytes()).hexdigest() != digest:
            raise AssertionError('Result hash mismatch: '+name)
    for name, digest in manifest['source_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise AssertionError('Source hash mismatch: '+name)
    return dict(valid=True, replay='all analytical arithmetic and recorded hashes agree',
                scope=fresh['verification']['scope'], new_engine_calls=0)
