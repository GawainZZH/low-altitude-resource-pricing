"""Accounting identities and finite-candidate calibration boundaries."""
from __future__ import annotations

import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from .comprehensive_state_grid import build_state_grid
from .hard_capacity_reequilibrium import HardCapacityMarketEngine


def activation_threshold(candidates, tolerance=1.e-9):
    """Each candidate is (charge, welfare intercept, externality relief)."""
    ordered = sorted(candidates)
    if any(b < -tolerance for _, _, b in ordered):
        raise ValueError('Threshold characterization requires nonnegative relief')
    active_zero = any(a > tolerance for _, a, _ in ordered)
    finite = [(max(0., -a / b), charge) for charge, a, b in ordered if b > tolerance]
    first = ordered[0] if ordered else None
    first_threshold = None
    if first and first[2] > tolerance:
        first_threshold = max(0., -first[1] / first[2])
    if active_zero:
        threshold, charge = 0., min(q for q, a, b in ordered if a > tolerance)
    elif finite:
        threshold, charge = min(finite)
    else:
        threshold, charge = None, None
    return {'threshold': threshold, 'charge_at_boundary': charge,
            'active_at_zero': active_zero, 'first_charge_threshold': first_threshold}


def logit_surplus(engine, prices, reliability, availability):
    scale = getattr(engine, 'logit_scale', engine.case.logit_scale)
    utility = (engine.platform_affinity
               - engine.price_sensitivity[:, None] * np.asarray(prices)
               - engine.time_sensitivity[:, None] * engine.time
               - engine.reliability_sensitivity[:, None] * np.asarray(reliability))
    outside = (engine.outside_affinity - engine.price_sensitivity * engine.case.outside_price
               - engine.time_sensitivity * engine.case.outside_time)
    normalized = np.column_stack((outside / scale,
                                 utility / scale + math.log(availability)))
    return float(np.dot(engine.demand_base * scale / engine.price_sensitivity,
                        logsumexp(normalized, axis=1)))


def surplus_shapley(engine, p0, l0, phi0, p1, l1, phi1):
    """Symmetric endpoint decomposition, not a causal attribution."""
    ends = ((p0, p1), (l0, l1), (phi0, phi1))
    values = {}
    for bits in itertools.product((0, 1), repeat=3):
        args = [ends[j][bit] for j, bit in enumerate(bits)]
        values[bits] = logit_surplus(engine, *args)
    contributions = np.zeros(3)
    for order in itertools.permutations(range(3)):
        bits = [0, 0, 0]
        previous = values[tuple(bits)]
        for index in order:
            bits[index] = 1
            current = values[tuple(bits)]
            contributions[index] += (current - previous) / 6.
            previous = current
    return dict(zip(('fare', 'reliability', 'availability'), contributions.tolist()))


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resource_penalty(rho):
    return .15 * rho * rho + max(rho - .9, 0.) ** 3


def reuse_aware_search(curves, families, tolerance=1.e-8):
    lengths = {len(values) for values in curves.values()}
    if len(lengths) != 1 or min(lengths) < 2:
        raise ValueError('A common charge set with at least two candidates is required')
    active = {families[s] for s, values in curves.items() if values[1] - values[0] > tolerance}
    standard, reuse = {}, {}
    sd_indices, standard_indices, reuse_indices = {}, {}, {}
    for state, values in curves.items():
        best = max(range(len(values)), key=values.__getitem__)
        sd_indices[state] = best
        standard_indices[state] = best if values[1] - values[0] > tolerance else 0
        reuse_indices[state] = best if families[state] in active else 0
        standard[state] = values[standard_indices[state]]
        reuse[state] = values[reuse_indices[state]]
        assert reuse[state] >= standard[state] - tolerance
    return {'standard_values': standard, 'reuse_values': reuse,
            'sd_indices': sd_indices, 'standard_indices': standard_indices,
            'reuse_indices': reuse_indices, 'active_family_ids': active,
            'distinct_games': 2 * len(set(families.values())) + (next(iter(lengths)) - 2) * len(active),
            'active_families': len(active),
            'missed_states': [s for s, values in curves.items() if max(values) - reuse[s] > tolerance]}


def reuse_summaries(rows):
    """Compare searches on one lowest-E selected response per family and charge.

    Cache rows are unchanged. Choice diagnostics compare exact smallest-charge
    maximizers; gate and missed-improvement diagnostics retain the 1e-8 tolerance.
    """
    tolerance = 1.e-8
    scenarios = defaultdict(lambda: defaultdict(list))
    for row in rows:
        numbers = [float(row[key]) for key in
                   ('charge', 'externality_weight', 'welfare', 'utilization')]
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError('Canonical reuse requires finite charge, weight, welfare and utilization')
        if numbers[0] < 0 or numbers[1] < 0 or numbers[3] < 0:
            raise ValueError('Canonical reuse requires nonnegative charge, weight and utilization')
        scenarios[row.get('scenario', 'baseline')][row['case']].append(row)
    result = []
    for scenario, states in sorted(scenarios.items()):
        ordered = {state: sorted(group, key=lambda r: float(r['charge'])) for state, group in states.items()}
        charges = tuple(float(r['charge']) for r in next(iter(ordered.values())))
        if len(charges) < 2 or charges[0] != 0. or len(set(charges)) != len(charges):
            raise ValueError('A unique charge set beginning at zero is required')
        families = {state: state.rsplit('_e', 1)[0] for state in ordered}
        references = {}
        for state, group in ordered.items():
            if tuple(float(r['charge']) for r in group) != charges:
                raise ValueError('Every state must have the same unique charge set')
            if len({float(r['externality_weight']) for r in group}) != 1:
                raise ValueError('Externality weight must be constant within each state')
            for row in group:
                key = families[state], float(row['charge'])
                previous = references.get(key)
                if previous is None or (float(row['externality_weight']), state) < (
                        float(previous['externality_weight']), previous['case']):
                    references[key] = row

        raw_curves, curves = {}, {}
        maximum_difference = 0.
        for state, group in ordered.items():
            raw_curves[state] = [float(r['welfare']) for r in group]
            canonical = []
            for row in group:
                reference = references[families[state], float(row['charge'])]
                value = (float(reference['welfare'])
                         - (float(row['externality_weight']) - float(reference['externality_weight']))
                         * resource_penalty(float(reference['utilization'])))
                if not math.isfinite(value):
                    raise ValueError('Canonical welfare reconstruction must remain finite')
                canonical.append(value)
                maximum_difference = max(maximum_difference, abs(value - float(row['welfare'])))
            curves[state] = canonical

        raw_search = reuse_aware_search(raw_curves, families)
        search = reuse_aware_search(curves, families)
        diagnostics = {'maximum_canonical_welfare_difference': maximum_difference,
                       'first_step_status_change_count': sum(
                           (curves[s][1] - curves[s][0] > tolerance)
                           != (raw_curves[s][1] - raw_curves[s][0] > tolerance) for s in curves),
                       'active_family_status_change_count': len(
                           search['active_family_ids'] ^ raw_search['active_family_ids']),
                       'raw_two_stage_distinct_games': raw_search['distinct_games']}
        for policy in ('sd', 'standard', 'reuse'):
            indices, raw_indices = search[f'{policy}_indices'], raw_search[f'{policy}_indices']
            diagnostics[f'{policy}_charge_change_count'] = sum(
                indices[s] != raw_indices[s] for s in curves)
            diagnostics[f'{policy}_positive_charge_status_change_count'] = sum(
                (indices[s] > 0) != (raw_indices[s] > 0) for s in curves)
        n = len(curves)
        result.append({'scenario': scenario, 'states': n, 'families': len(set(families.values())),
                       'full_distinct_games': len(set(families.values())) * len(next(iter(curves.values()))),
                       'two_stage_distinct_games': search['distinct_games'],
                       'active_families': search['active_families'],
                       'standard_false_negative_count': sum(max(v) > v[0] + 1.e-8 and v[1] <= v[0] + 1.e-8 for v in curves.values()),
                       'reuse_false_negative_count': len(search['missed_states']),
                       'sd_average_gain': sum(max(v) - v[0] for v in curves.values()) / n,
                       'standard_airp_average_gain': sum(search['standard_values'][s] - curves[s][0] for s in curves) / n,
                       'reuse_average_gain': sum(search['reuse_values'][s] - curves[s][0] for s in curves) / n,
                       'standard_max_regret': max(max(v) - search['standard_values'][s] for s, v in curves.items()),
                       'reuse_max_regret': max(max(v) - search['reuse_values'][s] for s, v in curves.items()),
                       **diagnostics})
    return result


def calibration_rows(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row['case'].rsplit('_e', 1)[0]][float(row['charge'])].append(row)
    result = []
    max_affine_error = 0.
    max_fare_replica_difference = 0.
    for profile, by_charge in sorted(grouped.items()):
        ordered = {q: sorted(v, key=lambda r: float(r['externality_weight'])) for q, v in by_charge.items()}
        nc = ordered[0.][0]
        e0 = float(nc['externality_weight'])
        g0 = resource_penalty(float(nc['utilization']))
        fare0 = sum(float(nc[f'price_{i}']) * float(nc[f'demand_{i}']) for i in (1, 2, 3))
        for criterion in ('US', 'TS'):
            lines = []
            for charge, replicas in sorted(ordered.items()):
                if charge == 0.:
                    continue
                row = replicas[0]
                relief = g0 - resource_penalty(float(row['utilization']))
                intercept = float(row['welfare']) - float(nc['welfare']) - e0 * relief
                revenue = sum(float(row[f'price_{i}']) * float(row[f'demand_{i}']) for i in (1, 2, 3))
                if criterion == 'TS':
                    intercept += revenue - fare0
                lines.append((charge, intercept, relief))
                for other in replicas:
                    e = float(other['externality_weight'])
                    nc_e = next(v for v in ordered[0.] if float(v['externality_weight']) == e)
                    actual = float(other['welfare']) - float(nc_e['welfare'])
                    if criterion == 'TS':
                        actual += (sum(float(other[f'price_{i}']) * float(other[f'demand_{i}']) for i in (1, 2, 3))
                                   - sum(float(nc_e[f'price_{i}']) * float(nc_e[f'demand_{i}']) for i in (1, 2, 3)))
                    max_affine_error = max(max_affine_error, abs(actual - intercept - e * relief))
                    max_fare_replica_difference = max(max_fare_replica_difference,
                        max(abs(float(other[f'price_{i}']) - float(row[f'price_{i}'])) for i in (1, 2, 3)))
            boundary = activation_threshold(lines)
            result.append({'profile': profile, 'criterion': criterion,
                           'demand_scale': nc['demand_scale'], 'capacity_scale': nc['capacity_scale'],
                           'reliability_power': nc['reliability_power'], 'nc_utilization': nc['utilization'],
                           **boundary})
    return result, {'maximum_affine_reconstruction_error': max_affine_error,
                    'maximum_replica_fare_difference': max_fare_replica_difference}


def hard_capacity_decomposition(rows):
    from .review_benchmark_correction import evaluate_domain

    if any(float(row.get('repair_verified', 0)) != 1. for row in rows):
        raise ValueError('Hard decomposition requires verified corrected cache rows')
    policy_rows = evaluate_domain(rows, 'US', 'unrestricted_utilization')['policy_rows']
    policies = {(row['case'], row['policy']): row for row in policy_rows}
    cases = {c.name: c for c in build_state_grid('full')}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['case']].append(row)
    output = []
    for name, group in sorted(grouped.items()):
        engine = HardCapacityMarketEngine(cases[name])
        nc = min(group, key=lambda r: float(r['charge']))
        sd = max(group, key=lambda r: (float(r['welfare']), -float(r['charge'])))
        p0 = [float(nc[f'price_{j}']) for j in (1, 2, 3)]
        p1 = [float(sd[f'price_{j}']) for j in (1, 2, 3)]
        rho0, rho1 = float(nc['utilization']), float(sd['utilization'])
        l0 = engine.base_loss + engine.exposure * engine.reliability_function(rho0)
        l1 = engine.base_loss + engine.exposure * engine.reliability_function(rho1)
        phi0, phi1 = float(nc['hard_availability']), float(sd['hard_availability'])
        parts = surplus_shapley(engine, p0, l0, phi0, p1, l1, phi1)
        cost0 = engine.operating_cost([float(nc[f'demand_{j}']) for j in (1, 2, 3)])
        cost1 = engine.operating_cost([float(sd[f'demand_{j}']) for j in (1, 2, 3)])
        operating = cost0 - cost1
        system = engine.system_cost(rho0) - engine.system_cost(rho1)
        delta = float(sd['welfare']) - float(nc['welfare'])
        nc_gap = max(0., float(nc['unrestricted_resource_demand'])
                     - float(nc['served_resource_demand']))
        diagnostics = {'nc_counterfactual_unserved_gap': nc_gap}
        for policy in ('SD', 'AIR', 'UT'):
            selected = policies[name, policy]
            prefix = policy.lower()
            gap = max(0., float(selected['unrestricted_resource_demand'])
                      - float(selected['served_resource_demand']))
            diagnostics.update({prefix + '_charge': float(selected['charge']),
                                prefix + '_availability': float(selected['hard_availability']),
                                prefix + '_counterfactual_unserved_gap': gap,
                                prefix + '_counterfactual_unserved_gap_relief': nc_gap - gap})
        first = sorted(group, key=lambda r: float(r['charge']))[1]
        first_increment = float(first['welfare']) - float(nc['welfare'])
        diagnostics['hard_activation_first_step_increment'] = first_increment
        diagnostics['hard_activation_first_step_false_negative'] = int(
            delta > 1.e-8 and first_increment <= 1.e-8)
        output.append({'case': name, 'charge': float(sd['charge']),
                       'nc_utilization': rho0, 'sd_utilization': rho1,
                       'nc_availability': phi0, 'sd_availability': phi1,
                       'both_binding': int(phi0 < 1.-1.e-8 and phi1 < 1.-1.e-8),
                       **parts, 'operating_saving': operating, 'system_relief': system,
                       'welfare_difference': delta,
                       'identity_error': delta - sum(parts.values()) - operating - system,
                       **diagnostics})
    return output


def run(root):
    output = root / 'results/trb_review_mechanisms_v1'
    output.mkdir(exist_ok=True)
    base = read_csv(root / 'results/comprehensive_state_grid_v1_full/comprehensive_state_grid_charge_search_records.csv')
    threshold, calibration = calibration_rows(base)
    write_csv(output / 'global_externality_boundaries.csv', threshold)
    mixture = read_csv(root / 'results/out_of_model_demand_robustness_v1/taste_mixture_charge_records.csv')
    reuse = reuse_summaries(base) + reuse_summaries(mixture)
    write_csv(output / 'reuse_aware_search_summary.csv', reuse)
    hard = hard_capacity_decomposition(read_csv(root / 'results/trb_review_hard_capacity_repair_v1/corrected_hard_charge_records.csv'))
    write_csv(output / 'hard_capacity_welfare_decomposition.csv', hard)
    summary = {'calibration': calibration, 'thresholds': {}, 'reuse': reuse,
               'hard': {'states': len(hard), 'both_binding_states': sum(r['both_binding'] for r in hard),
                        'maximum_identity_error': max(abs(r['identity_error']) for r in hard),
                        'average_components': {key: sum(r[key] for r in hard) / len(hard)
                                               for key in ('fare', 'reliability', 'availability', 'operating_saving', 'system_relief', 'welfare_difference')}}}
    for criterion in ('US', 'TS'):
        subset = [r for r in threshold if r['criterion'] == criterion]
        summary['thresholds'][criterion] = {
            'profiles': len(subset), 'active_at_zero_severity': sum(r['active_at_zero'] for r in subset),
            'thresholds_within_6500': sum(r['threshold'] is not None and r['threshold'] <= 6500 for r in subset),
            'largest_first_step_threshold_excess': max((r['first_charge_threshold'] or 0.) - (r['threshold'] or 0.) for r in subset),
            'first_step_boundaries_exact': sum(r['threshold'] is not None and r['first_charge_threshold'] is not None
                                              and abs(r['first_charge_threshold'] - r['threshold']) <= 1.e-6 for r in subset)}
    assert calibration['maximum_affine_reconstruction_error'] < 1.e-5
    assert summary['hard']['maximum_identity_error'] < 1.e-4
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary
