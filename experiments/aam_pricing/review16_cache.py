"""Read-only cache admission, NOT a substitute for complete leaf replay.

The incumbent and saved best-fare point enclosures are recomputed. Leaf
metadata is inspected, but no leaf bound or unilateral optimization is rerun.
No Nash-distance or welfare-error bound follows from admission.
"""
from decimal import Decimal, DecimalException
import math
import re

from .review16_equilibrium import AccessProblem, behavior_key
from .review15_bounds import I, UP, DOWN, verify_partition


SCIENTIFIC_SOURCES = (
    'experiments/aam_pricing/review13_equilibrium.py',
    'experiments/aam_pricing/review15_bounds.py',
    'experiments/aam_pricing/review16_equilibrium.py',
)
SOURCE_DATA = (
    'results/trb_review10_core_v1/paths.csv',
    'results/trb_review12_signals_v1/normalized_cases.json',
    'results/trb_review14_policy_v1/normalized_menus.json',
    'results/trb_review6_access_v1/saved_inputs.json',
)


def _number(value, label, serialized=False):
    if isinstance(value, bool):
        raise ValueError(label+': Boolean is not a numeric value')
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(label+': finite value required')
        result = Decimal.from_float(value)
    elif isinstance(value, int):
        result = Decimal(value)
    elif serialized and isinstance(value, (str, Decimal)):
        result = Decimal(value)
    else:
        raise ValueError(label+': numeric value required')
    if not result.is_finite():
        raise ValueError(label+': finite value required')
    return result


def _interval(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(label+': two endpoints required')
    lo, hi = (_number(v, label, serialized=True) for v in value)
    if lo > hi:
        raise ValueError(label+': reversed endpoints')
    return I(lo, hi)


def _numeric_tree(value, label):
    if isinstance(value, dict):
        return {k: _numeric_tree(v, label+'.'+str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_numeric_tree(v, label) for v in value]
    return _number(value, label)


def _row(row):
    if not isinstance(row, dict) or not isinstance(row['case'], dict):
        raise ValueError('Input row and case must be mappings')
    case = _numeric_tree({k: v for k, v in row['case'].items() if k != 'name'}, 'case')
    prices = _numeric_tree(row['prices'], 'prices')
    waiting = _numeric_tree(row['waiting_coefficients'], 'waiting coefficients')
    if (not isinstance(prices, list) or len(prices) != 3
            or not isinstance(waiting, list) or len(waiting) != 3 or min(waiting) <= 0):
        raise ValueError('Three prices and three positive waiting coefficients required')
    excluded = row['excluded']
    if (not isinstance(excluded, (list, tuple))
            or any(type(j) is not int or j not in range(3) for j in excluded)
            or len(set(excluded)) != len(excluded) or len(excluded) == 3):
        raise ValueError('Invalid excluded platform set')
    charge = _number(row['charge'], 'charge')
    lower, upper = case['price_lower'], case['price_upper']
    if charge < 0 or lower >= upper or any(p < lower or p > upper for p in prices):
        raise ValueError('Charge or prices outside the permitted domain')
    return dict(case=case, prices=prices, waiting=waiting, charge=charge, excluded=list(excluded))


def _compatible_designs(source, target):
    if (source is None) != (target is None):
        raise ValueError('Supply both source_design and target_design, or neither')
    if source is None:
        return False
    for design in (source, target):
        if not isinstance(design, dict):
            raise ValueError('Design must be a mapping with SHA256 records')
        for section, paths in (('source_sha256', SCIENTIFIC_SOURCES), ('data_sha256', SOURCE_DATA)):
            if not isinstance(design.get(section), dict):
                raise ValueError('Missing or invalid SHA256 mapping: '+section)
            for path in paths:
                digest = design[section].get(path)
                if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', digest):
                    raise ValueError('Missing or invalid SHA256: '+path)
        for name, expected in (('absolute', Decimal('.01')), ('normalized', Decimal('.00001')),
                               ('decimal_precision', Decimal(40))):
            if name in design:
                _number(design[name], 'protocol '+name)
                # Production converts these policy tolerances through str(), not binary I().
                if Decimal(str(design[name])) != expected:
                    raise ValueError('Incompatible directed protocol: '+name)
    for section, paths in (('source_sha256', SCIENTIFIC_SOURCES), ('data_sha256', SOURCE_DATA)):
        for path in paths:
            if source[section][path].lower() != target[section][path].lower():
                raise ValueError('Incompatible SHA256: '+path)
    return True


def _leaves(leaves, lower, upper):
    if not isinstance(leaves, list) or not leaves:
        raise ValueError('Nonempty complete leaf partition required')
    uppers = []
    for leaf in leaves:
        fare = _interval(leaf['fare'], 'leaf fare')
        direct = _interval(leaf['direct'], 'leaf direct profit')
        center_profit = _interval(leaf['center_profit'], 'leaf center profit')
        center = _number(leaf['center'], 'leaf center', serialized=True)
        bound = _number(leaf['upper'], 'leaf upper', serialized=True)
        if not fare.lo <= center <= fare.hi:
            raise ValueError('Leaf center outside its fare interval')
        if bound > direct.hi or bound < direct.lo or bound < center_profit.lo:
            raise ValueError('Inconsistent leaf profit metadata')
        method = leaf['method']
        gradient = None if leaf['gradient'] is None else _interval(leaf['gradient'], 'leaf gradient')
        if method == 'direct':
            if bound != direct.hi:
                raise ValueError('Direct leaf upper differs from direct enclosure')
        elif method == 'increasing':
            if gradient is None or gradient.lo < 0:
                raise ValueError('Inconsistent increasing leaf gradient')
        elif method == 'decreasing':
            if gradient is None or gradient.hi > 0:
                raise ValueError('Inconsistent decreasing leaf gradient')
        elif method == 'mean_value':
            if gradient is None or not gradient.lo < 0 < gradient.hi:
                raise ValueError('Inconsistent mean-value leaf gradient')
        else:
            raise ValueError('Unknown leaf method')
        uppers.append(bound)
    if not verify_partition(leaves, lower, upper):
        raise ValueError('Incomplete full-domain leaf partition')
    return max(uppers)


def _coordinate(bound, row, problem, lower, upper):
    j = bound['platform']
    if (_numeric_tree(bound['prices'], 'coordinate prices') != _numeric_tree(row['prices'], 'prices')
            or _number(bound['charge'], 'coordinate charge') != _number(row['charge'], 'charge')):
        raise ValueError('Coordinate prices or charge differ from expected input')
    if bound['coverage_complete'] is not True:
        raise ValueError('Full-domain coverage flag must be true')
    maximum = _leaves(bound['leaves'], lower, upper)
    if _number(bound['profit_upper'], 'profit upper', True) != maximum:
        raise ValueError('Global upper is not the maximum leaf upper')
    baseline = _interval(bound['baseline'], 'baseline')
    denominator = max(Decimal(1), Decimal(0) if baseline.lo <= 0 <= baseline.hi
                      else min(baseline.lo.copy_abs(), baseline.hi.copy_abs()))
    gain = max(Decimal(0), UP.subtract(maximum, baseline.lo))
    normalized = UP.divide(gain, denominator)
    tolerance = min(Decimal('.01'), DOWN.multiply(Decimal('.00001'), denominator))
    for name, expected in (('gain_upper', gain), ('normalized_upper', normalized), ('tolerance', tolerance)):
        if _number(bound[name], name, True) != expected:
            raise ValueError('Directed '+name+' differs')
    if 'denominator' in bound and _number(bound['denominator'], 'denominator', True) != denominator:
        raise ValueError('Normalization denominator differs')
    if type(bound['closed']) is not bool or bound['closed'] != (gain <= tolerance):
        raise ValueError('Coordinate closed flag differs')
    observed = _number(bound['observed_gain_lower'], 'observed gain lower', True)
    gap = _number(bound['profit_maximum_error'], 'profit maximum error', True)
    best_fare = _number(bound['best_fare'], 'best fare', True)
    if not lower <= best_fare <= upper or not 0 <= observed <= gain or not 0 <= gap <= gain:
        raise ValueError('Observed lower, upper, gap or best fare inconsistent')

    prices = [I(v) for v in row['prices']]
    actual_base = problem.profit(prices, row['charge'], j, point=True)
    if actual_base.lo != baseline.lo or actual_base.hi != baseline.hi:
        raise ValueError('Recomputed directed baseline differs')
    prices[j] = I(best_fare)
    best = problem.profit(prices, row['charge'], j, point=True)
    if best.lo < baseline.lo or best.lo > maximum:
        raise ValueError('Saved best fare does not support a best-profit lower bound')
    expected_observed = max(Decimal(0), DOWN.subtract(best.lo, baseline.hi))
    expected_gap = max(Decimal(0), UP.subtract(maximum, best.lo))
    if observed != expected_observed or gap != expected_gap:
        raise ValueError('Recomputed observed lower or profit gap differs')
    return dict(platform=j, leaves=len(bound['leaves']), denominator=str(denominator),
                gain_upper=str(gain), observed_gain_lower=str(observed), closed=bound['closed'])


def validate_cached(saved, expected_row, source_design=None, target_design=None):
    """Admit consistent cache metadata or raise ValueError, without writes.

    Both designs are required for cross-design reuse. Only three scientific
    module hashes and four data hashes must match; runner hashes and budgets
    are intentionally irrelevant. Actual file hashes are the caller's duty.

    Successful admission recomputes two point-profit enclosures per active
    platform, NOT every leaf. The caller must still run complete directed
    leaf replay before describing the global profit bounds as verified.
    """
    try:
        compatible = _compatible_designs(source_design, target_design)
        if not isinstance(saved, dict):
            raise ValueError('Saved response must be a mapping')
        expected, actual = _row(expected_row), _row(saved['input'])
        if expected != actual:
            raise ValueError('Expected numeric parameters, prices, charge or platforms differ')
        key = behavior_key(expected_row)
        if behavior_key(saved['input']) != key:
            raise ValueError('Behavior key differs')
        if 'behavior_key' in saved and saved['behavior_key'] != key:
            raise ValueError('Saved behavior key differs')
        active = [j for j in range(3) if j not in expected['excluded']]
        bounds = saved['coordinates']
        if (not isinstance(bounds, list)
                or any(type(b['platform']) is not int for b in bounds)
                or sorted(b['platform'] for b in bounds) != active):
            raise ValueError('Incomplete or duplicate active platform coordinates')
        problem = AccessProblem(expected_row['case'], expected_row['excluded'],
                                expected_row['waiting_coefficients'])
        lower, upper = expected['case']['price_lower'], expected['case']['price_upper']
        checks = [_coordinate(b, expected_row, problem, lower, upper) for b in bounds]
        if (type(saved['all_closed']) is not bool
                or saved['all_closed'] != all(b['closed'] for b in checks)):
            raise ValueError('Response all_closed flag differs')
        return dict(accepted=True, behavior_key=key, coordinates=len(checks), checks=checks,
                    all_closed=saved['all_closed'], source_design_compatible=compatible,
                    baseline_and_best_points_recomputed=True, every_leaf_recomputed=False,
                    requires_leaf_replay=True, welfare_error_bound=False)
    except (KeyError, TypeError, IndexError, DecimalException, OverflowError) as error:
        raise ValueError('Malformed or incompatible cache: '+str(error)) from error
