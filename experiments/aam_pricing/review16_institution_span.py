"""Solver-free matched payment ledgers on twelve frozen analytical paths."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

SOURCE = 'results/trb_review11_finite_paths_v1'
INPUT = SOURCE + '/inputs.json'
SUMMARY = SOURCE + '/summary.json'
WAIT_SOURCE = SOURCE + '/execution_source/experiments/aam_pricing/review11_finite_paths.py'
PROOF_SOURCE = SOURCE + '/reference_source/submission_packages/TRA_V290_LaTeX_Project/MAIN.tex'
PINNED = {
    INPUT: '5e5286f16ed96dcd1a0fec712241cada0d13b8421d1d411eb0a00d9671a157e6',
    SUMMARY: '7851419c34ccf07658b65f3998d4f51ee4328bbf599aada84b4518616a66f8cf',
    WAIT_SOURCE: '8e750d62c4a28010e0b3bdb49fdd2b0a120838d8e343f7533b2f355d620ea58b',
    PROOF_SOURCE: '6f2b369f419bedc30f5f442b3606e05ec2e49c50d9ebadf17f581a103b3d4ae6',
}
OWNED = ('experiments/aam_pricing/review16_institution_span.py',
         'scripts/run_review16_institution_span.py', 'tests/test_review16_institution_span.py')
GENERATED = ('results.json', 'source_mapping.json', 'menu_ledgers.csv', 'candidate_comparisons.csv', 'policies.csv',
             'comparisons.csv', 'summaries.csv', 'group_CS.csv', 'operational_span.json')
KAPPAS = (0., .5, 1.)
TOL = 2e-7
PRIMITIVE_KEYS = {'lam', 'mu', 'value', 'capacity', 'resource', 'costs', 'platform_quality',
                  'mass', 'group_quality', 'reliability_slopes', 'outside_utility',
                  'outside_price', 'fare_bounds'}


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Finite scalar required: ' + label)
    return float(value)


def _tree_finite(value):
    if isinstance(value, dict):
        for item in value.values():
            _tree_finite(item)
    elif isinstance(value, list):
        for item in value:
            _tree_finite(item)
    elif isinstance(value, (int, float)) and not math.isfinite(value):
        raise ValueError('Nonfinite saved value')


def _close(actual, expected, label, atol=TOL):
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ValueError('Dimension mismatch: ' + label)
        for a, b in zip(actual, expected):
            _close(a, b, label, atol)
    elif not math.isclose(_finite(actual, label), _finite(expected, label),
                          abs_tol=atol, rel_tol=2e-10):
        raise ValueError('Saved arithmetic mismatch: ' + label)


def validate_contract(c, *, waiting_coefficients=None, real_share=0., recipient_share=1.):
    """Fail closed outside the frozen common-value, full-recipient institution."""
    if set(c) != PRIMITIVE_KEYS:
        raise ValueError('Unknown or incomplete primitive/coefficient specification')
    for key in ('lam', 'mu', 'value', 'capacity'):
        if _finite(c[key], key) <= 0:
            raise ValueError('Positive primitive required: ' + key)
    if _finite(c['outside_price'], 'outside_price') < 0:
        raise ValueError('Nonnegative ground payment required')
    m, g = len(c['resource']), len(c['mass'])
    if m < 2 or g < 1:
        raise ValueError('At least two platforms and one group required')
    for key in ('resource', 'costs', 'platform_quality', 'mass', 'group_quality',
                'reliability_slopes', 'outside_utility', 'fare_bounds'):
        expected = m if key in ('resource', 'costs', 'platform_quality') else 2 if key == 'fare_bounds' else g
        if not isinstance(c[key], list) or len(c[key]) != expected:
            raise ValueError('Invalid primitive dimension: ' + key)
        for value in c[key]:
            _finite(value, key)
    if (min(c['resource']) <= 0 or min(c['mass']) <= c['capacity'] / min(c['resource'])
            or min(c['costs']) < 0 or min(c['reliability_slopes']) < 0
            or not 0 <= c['fare_bounds'][0] < c['fare_bounds'][1]
            or c['outside_price'] > c['mu'] / c['lam']):
        raise ValueError('Analytical actual-game premises failed')
    if _finite(real_share, 'real_share') != 0 or _finite(recipient_share, 'recipient_share') != 1:
        raise ValueError('Only pure payments with full independent recipient credit supported')
    b = [c['lam'] * c['value']] * g
    if waiting_coefficients is not None:
        _close(waiting_coefficients, b, 'common b=lambda*v', atol=1e-14)
    return dict(waiting_coefficients=b, payment_coefficient=c['lam'], real_share=0.,
                recipient_share=1., payment_rule='r=v*w', separate_recipient=True,
                payoff_equivalence='all feasible fare vectors and charges under the same complementarity')


def _choice(c, prices, rho, access):
    probabilities, group_cs = [], []
    for mass, quality, slope, outside in zip(c['mass'], c['group_quality'],
                                            c['reliability_slopes'], c['outside_utility']):
        air = [(quality - slope * rho + a - c['lam'] * p - access) / c['mu']
               for a, p in zip(c['platform_quality'], prices)]
        values = air + [outside / c['mu']]
        top = max(values)
        den = top + math.log(math.fsum(math.exp(v - top) for v in values))
        probabilities.append([math.exp(v - den) for v in air])
        group_cs.append(mass * c['mu'] / c['lam'] * den)
    return probabilities, group_cs


def reconstruct(c, state, institution, kappa):
    """Substitute saved states into utilities, auxiliary equations and both ledgers."""
    contract = validate_contract(c)
    if institution not in ('waiting', 'payment') or kappa not in KAPPAS:
        raise ValueError('Unsupported institution or ground accounting')
    _tree_finite(state)
    q, w, rho = (_finite(state[k], k) for k in ('charge', 'wait', 'rho'))
    if q < 0 or w < 0 or rho != 1.:
        raise ValueError('Only full-capacity saved pre-release states supported')
    p, x, a, costs = state['prices'], state['resource_shares'], c['resource'], c['costs']
    m = len(a)
    if (len(p) != m or len(x) != m or any(not 0 < z < 1 for z in x)
            or any(not c['fare_bounds'][0] < v < c['fare_bounds'][1] for v in p)):
        raise ValueError('Strictly interior fares and resource shares required')
    _close(math.fsum(x), 1., 'resource-share sum', 1e-11)
    h = c['lam'] / c['mu']
    expected_prices = [cost + work * q + 1 / (h * (1 - share))
                       for cost, work, share in zip(costs, a, x)]
    _close(p, expected_prices, 'auxiliary global markup', 1e-9)
    zeta = [math.log(share) + 1 / (1 - share) - (quality - c['lam'] * cost) / c['mu']
            - math.log(work) + h * work * q
            for share, quality, cost, work in zip(x, c['platform_quality'], costs, a)]
    _close(zeta, [zeta[0]] * m, 'auxiliary inverse-share equations', 1e-9)

    wait_prob, wait_cs = _choice(c, p, rho, c['lam'] * c['value'] * w)
    reservation = c['value'] * w
    pay_prob, pay_cs = _choice(c, p, rho, c['lam'] * reservation)
    _close(pay_prob, wait_prob, 'all-group payment choice equivalence', 1e-12)
    _close(pay_cs, wait_cs, 'payment logsum equivalence', 1e-9)
    _close(state['probabilities'], wait_prob, 'saved probabilities', 1e-11)
    y = [math.fsum(mass * prob[j] for mass, prob in zip(c['mass'], wait_prob)) for j in range(m)]
    _close(state['demand'], y, 'saved platform demand', 1e-8)
    _close(state['conditional_demand'], [c['capacity'] * share / work for share, work in zip(x, a)],
           'auxiliary full-capacity demand', 1e-8)
    _close(y, state['conditional_demand'], 'actual vs auxiliary demand', 1e-8)
    work = math.fsum(v * z for v, z in zip(a, y))
    _close(work, c['capacity'], 'capacity work', 1e-8)
    Y, N = math.fsum(y), math.fsum(c['mass'])
    _close(state['total_mass'], Y, 'saved total passengers', 1e-8)
    _close(state['ground_mass'], N - Y, 'saved ground passengers', 1e-8)
    profits = [(fare - cost - q * z) * v for fare, cost, z, v in zip(p, costs, a, y)]
    if min(profits) <= 0:
        raise ValueError('Positive incumbent profits required for global comparison')
    _close(state['profits'], profits, 'platform profits', 1e-8)
    fare_payment = math.fsum(v * z for v, z in zip(p, y))
    operating_cost = math.fsum(v * z for v, z in zip(costs, y))
    regulator_revenue = q * work
    waiting_cs = math.fsum(wait_cs)
    _close(state['net_surplus'], waiting_cs, 'saved monetary logsum', 1e-8)
    for k in KAPPAS:
        expected_ts = waiting_cs + math.fsum(profits) + regulator_revenue + k * c['outside_price'] * (N - Y)
        _close(state['TS'][str(k)], expected_ts, 'saved TS kappa=' + str(k), 1e-8)
    zero_prob, _ = _choice(c, p, 1., 0.)
    zero_work = math.fsum(mass * math.fsum(z * t for z, t in zip(a, prob))
                          for mass, prob in zip(c['mass'], zero_prob))
    _close(state['zero_wait_excess'], zero_work - c['capacity'], 'zero-wait resource excess', 1e-8)

    # Store original probabilities only after independent substitution validates both utilities.
    prob = state['probabilities']
    group_cs = wait_cs if institution == 'waiting' else pay_cs
    group_y = [mass * math.fsum(row) for mass, row in zip(c['mass'], prob)]
    group_work = [mass * math.fsum(t * z for t, z in zip(row, a)) for mass, row in zip(c['mass'], prob)]
    group_fares = [mass * math.fsum(t * z for t, z in zip(row, p)) for mass, row in zip(c['mass'], prob)]
    physical_wait = w if institution == 'waiting' else 0.
    r = reservation if institution == 'payment' else 0.
    recipient = r * Y
    ground_margin = kappa * c['outside_price'] * (N - Y)
    cs = math.fsum(group_cs)
    ts = cs + math.fsum(profits) + regulator_revenue + recipient + ground_margin
    return dict(institution=institution, kappa=kappa, charge=q, prices=list(p),
        resource_shares=list(x), probabilities=prob, demand=list(state['demand']),
        group_mass=list(c['mass']), group_CS=group_cs,
        group_CS_per_potential=[v / mass for v, mass in zip(group_cs, c['mass'])],
        group_air_probability=[math.fsum(row) for row in prob], group_air_passengers=group_y,
        group_resource_work=group_work, group_fare_payment=group_fares,
        group_access_payment=[r * v for v in group_y],
        group_wait_cost=[c['value'] * physical_wait * v for v in group_y],
        air_passengers=Y, ground_passengers=N - Y, resource_work=work, rho=rho,
        source_wait_minutes=w, physical_wait_minutes=physical_wait,
        passenger_wait_minutes=physical_wait * Y, real_wait_cost=c['value'] * physical_wait * Y,
        reservation_price=r, recipient_revenue=recipient, fare_payment=fare_payment,
        traveler_payment=fare_payment + recipient, operating_resource_cost=operating_cost,
        public_charge_revenue=regulator_revenue, platform_profits=profits,
        platform_profit=math.fsum(profits), ground_margin=ground_margin, external_cost=0.,
        CS=cs, TS=ts, waiting_coefficients=contract['waiting_coefficients'],
        transfer_cancellation_error=-fare_payment - recipient + math.fsum(profits)
            + regulator_revenue + recipient + operating_cost,
        choice_equivalence_error=max(abs(u - v) for urow, vrow in zip(wait_prob, pay_prob)
                                     for u, v in zip(urow, vrow)),
        group_CS_equivalence_error=max(abs(u - v) for u, v in zip(wait_cs, pay_cs)))


def reduction(waiting_gain, payment_gain):
    w, p = _finite(waiting_gain, 'waiting gain'), _finite(payment_gain, 'payment gain')
    status = 'zero' if w == 0 else 'negative' if w < 0 else 'near_zero' if w <= 1e-8 else 'positive'
    return dict(denominator_status=status,
                reduction_percent=100 * (1 - p / w) if status == 'positive' else None)


def select_policy(menu):
    if not menu or menu[0]['charge'] != 0 or any(b['charge'] <= a['charge'] for a, b in zip(menu, menu[1:])):
        raise ValueError('Complete increasing menu with own NC required')
    for row in menu:
        _finite(row['TS'], 'menu welfare')
    best = min(menu, key=lambda r: (-r['TS'], r['charge']))
    gains = [row['TS'] - menu[0]['TS'] for row in menu]
    return dict(NC=menu[0], SD=best, gain=best['TS'] - menu[0]['TS'], candidate_gains=gains,
                negative_candidate_gains=sum(g < 0 for g in gains),
                zero_candidate_gains=sum(g == 0 for g in gains))


def _validate_path(case, row):
    c, r = case['primitives'], row['result']
    validate_contract(c)
    if (r['status'] != 'completed' or r['classification'] != 'release_first'
            or r['qualifying_release'] is not True or r['validation']['valid'] is not True
            or r['post_domain_candidates_evaluated'] is not False or c['fare_bounds'] != [5., 120.]):
        raise ValueError('Completed analytical pre-release path required')
    end, states = r['endpoints'], r['states']
    stop = _finite(r['stop_charge'], 'stop charge')
    if (end['eligible'] is not True or len(states) != 33 or stop <= 0
            or not stop < end['fare_charge']):
        raise ValueError('All 33 states through interior first release required')
    _close(end['release_charge'], stop, 'first release charge', 1e-10)
    _close(end['release']['charge'], stop, 'endpoint charge witness', 1e-10)
    _close(end['release']['zero_wait_excess'], 0., 'endpoint zero-wait witness', 1e-8)
    _close([s['charge'] for s in states], [stop * i / 32 for i in range(33)], '33 candidate charges', 1e-10)
    if states[-1]['wait'] != 0. or any(s['wait'] <= 0 for s in states[:-1]):
        raise ValueError('Positive waiting through the first zero-wait endpoint required')
    if any(b['wait'] >= a['wait'] for a, b in zip(states, states[1:])):
        raise ValueError('Saved waiting path is not strictly decreasing')
    return states


def _span(values):
    return dict(minimum=min(values), maximum=max(values))


def _summarize(comparisons):
    groups = defaultdict(list)
    for row in comparisons:
        strata = ('all', 'resource:' + str(row['resource_index']), 'cost:' + str(row['cost_index']),
                  'quality:' + str(row['alignment']))
        for stratum in strata:
            groups[stratum, row['kappa']].append(row)
    summaries = []
    for (stratum, kappa), rows in groups.items():
        n = len(rows)
        wg = math.fsum(r['waiting_gain'] for r in rows)
        pg = math.fsum(r['payment_gain'] for r in rows)
        valid = [r['reduction_percent'] for r in rows if r['reduction_percent'] is not None]
        data = dict(stratum=stratum, kappa=kappa, economies=n, waiting_gain_total=wg,
                    payment_gain_total=pg, waiting_gain=wg / n, payment_gain=pg / n,
                    **reduction(wg, pg), valid_case_ratios=len(valid),
                    undefined_case_ratios=n - len(valid),
                    negative_case_ratios=sum(v < 0 for v in valid),
                    zero_case_ratios=sum(v == 0 for v in valid),
                    case_reduction_range=_span(valid) if valid else None,
                    candidate_negative_waiting=sum(r['waiting_negative_candidates'] for r in rows),
                    candidate_negative_payment=sum(r['payment_negative_candidates'] for r in rows))
        for inst in ('waiting', 'payment'):
            for policy in ('NC', 'SD'):
                for metric in ('TS', 'CS', 'air_passengers', 'source_wait_minutes', 'physical_wait_minutes',
                               'resource_work', 'recipient_revenue', 'platform_profit', 'ground_margin'):
                    data[f'{inst}_{policy}_{metric}'] = math.fsum(r[f'{inst}_{policy}'][metric] for r in rows) / n
        summaries.append(data)
    return summaries


def compute(inputs, saved):
    if inputs['path_points'] != 33 or inputs['kappas'] != list(KAPPAS):
        raise ValueError('Frozen path grid or kappa quantifier differs')
    cases, paths = {}, {}
    for i, case in enumerate(inputs['cases']):
        if case['id'] in cases:
            raise ValueError('Duplicate input economy')
        cases[case['id']] = (i, case)
    for i, row in enumerate(saved['rows']):
        if row['id'] in paths:
            raise ValueError('Duplicate saved path')
        paths[row['id']] = (i, row)
    if set(cases) != set(paths):
        raise ValueError('Input/path inventory mismatch')
    selected = [(i, c) for i, c in cases.values() if c['primitives']['fare_bounds'][1] == 120.]
    signatures = {(c['resource_index'], c['cost_index'], c['alignment']) for _, c in selected}
    expected = {(r, c, v) for r in (0, 1) for c in (0, 1) for v in (-.75, 0., .75)}
    if len(selected) != 12 or signatures != expected:
        raise ValueError('Exactly twelve prespecified analytical economies required')

    ledgers, policies, comparisons, economies, candidate_comparisons = [], [], [], [], []
    rent_errors = []
    for input_index, case in selected:
        path_index, path = paths[case['id']]
        c = case['primitives']
        states = _validate_path(case, path)
        contract = validate_contract(c)
        menus = {(inst, k): [] for inst in ('waiting', 'payment') for k in KAPPAS}
        for j, s in enumerate(states):
            for k in KAPPAS:
                pair = {}
                for inst in ('waiting', 'payment'):
                    ledger = dict(case=case['id'], state_index=j,
                        source_input_pointer=f'/cases/{input_index}/primitives',
                        source_state_pointer=f'/rows/{path_index}/result/states/{j}',
                        **reconstruct(c, s, inst, k))
                    menus[inst, k].append(ledger)
                    ledgers.append(ledger)
                    pair[inst] = ledger
                diff = pair['payment']['TS'] - pair['waiting']['TS']
                _close(diff, pair['waiting']['real_wait_cost'], 'payment rent identity', 1e-8)
                rent_errors.append(abs(diff - pair['waiting']['real_wait_cost']))
        for k in KAPPAS:
            for inst in ('waiting', 'payment'):
                nc = menus[inst, k][0]
                for point in menus[inst, k]:
                    point['own_NC_TS'] = nc['TS']
                    point['gain_from_own_NC'] = point['TS'] - nc['TS']
                    point['own_NC_group_CS'] = list(nc['group_CS'])
                    point['group_CS_change_from_own_NC'] = [v - u for v, u in zip(point['group_CS'], nc['group_CS'])]
            nc_advantage = menus['payment', k][0]['TS'] - menus['waiting', k][0]['TS']
            for w_point, p_point in zip(menus['waiting', k], menus['payment', k]):
                wg, pg = w_point['gain_from_own_NC'], p_point['gain_from_own_NC']
                candidate_comparisons.append(dict(case=case['id'], kappa=k,
                    state_index=w_point['state_index'], charge=w_point['charge'],
                    waiting_TS=w_point['TS'], payment_TS=p_point['TS'],
                    waiting_NC_TS=w_point['own_NC_TS'], payment_NC_TS=p_point['own_NC_TS'],
                    waiting_gain=wg, payment_gain=pg, **reduction(wg, pg),
                    baseline_advantage=nc_advantage,
                    same_charge_advantage=p_point['TS'] - w_point['TS'], gain_difference=wg - pg))
            pair = {}
            for inst in ('waiting', 'payment'):
                policy = dict(case=case['id'], institution=inst, kappa=k, **select_policy(menus[inst, k]))
                pair[inst] = policy
                policies.append(policy)
            w, p = pair['waiting'], pair['payment']
            baseline = p['NC']['TS'] - w['NC']['TS']
            optimized = p['SD']['TS'] - w['SD']['TS']
            gap = w['gain'] - p['gain']
            _close(gap, baseline - optimized, 'own-NC increment identity', 1e-8)
            comparisons.append(dict(case=case['id'], resource_index=case['resource_index'],
                cost_index=case['cost_index'], alignment=case['alignment'], kappa=k,
                waiting_gain=w['gain'], payment_gain=p['gain'], **reduction(w['gain'], p['gain']),
                baseline_advantage=baseline, optimized_advantage=optimized, gain_difference=gap,
                waiting_negative_candidates=w['negative_candidate_gains'],
                payment_negative_candidates=p['negative_candidate_gains'],
                waiting_NC=w['NC'], waiting_SD=w['SD'], payment_NC=p['NC'], payment_SD=p['SD']))
        economies.append(dict(case=case['id'], resource_index=case['resource_index'], cost_index=case['cost_index'],
            alignment=case['alignment'], primitives=c, transfer_contract=contract,
            candidate_charges=[s['charge'] for s in states], release_charge=states[-1]['charge'],
            initial_wait_minutes=states[0]['wait'], initial_passengers=states[0]['total_mass'],
            terminal_passengers=states[-1]['total_mass'],
            observed_wait_span=_span([s['wait'] for s in states]),
            observed_fare_span=_span([p for s in states for p in s['prices']]),
            group_participation_span=_span([math.fsum(p) for s in states for p in s['probabilities']]),
            global_Nash_basis='analytical_actual_game_with_payoff_equivalence',
            finite_deviation_search_is_global_proof=False,
            scope='same 33 saved candidate charges from NC to first zero wait; no postrelease optimum'))
    span = dict(resource_vectors=sorted({tuple(e['primitives']['resource']) for e in economies}),
        cost_vectors=sorted({tuple(e['primitives']['costs']) for e in economies}),
        quality_vectors=sorted({tuple(e['primitives']['platform_quality']) for e in economies}),
        resource_per_request=_span([a for e in economies for a in e['primitives']['resource']]),
        marginal_cost=_span([a for e in economies for a in e['primitives']['costs']]),
        platform_quality=_span([a for e in economies for a in e['primitives']['platform_quality']]),
        initial_wait_minutes=_span([e['initial_wait_minutes'] for e in economies]),
        release_charge=_span([e['release_charge'] for e in economies]),
        fares=_span([p for e in economies for p in e['observed_fare_span'].values()]),
        value_per_minute=.4, waiting_coefficient=.04, lambda_common=.1, mu=.5,
        capacity=60., group_mass=[300., 280., 260.], outside_price=5., fare_bounds=[5., 120.])
    result = dict(schema=1, coverage=dict(economies=12, candidates_per_economy=33,
        saved_behavioral_points=396, independent_markets=False, nonmatched_new_economies=0,
        kappas=list(KAPPAS), institutional_ledger_rows=len(ledgers), postrelease_candidates=0,
        excluded_fare_cap=44., excluded_paths=len(cases) - 12),
        new_Nash_solves=0, new_root_solves=0, new_fixed_point_solves=0, new_behavioral_responses=0,
        economies=economies, menu_ledgers=ledgers, candidate_comparisons=candidate_comparisons,
        policies=policies, comparisons=comparisons,
        summaries=_summarize(comparisons), operational_span=span,
        audit=dict(complete_analytic_transfer=True, verified_common_charge_pairs=len(rent_errors),
            max_transfer_identity_error=max(rent_errors),
            max_payment_choice_error=max(r['choice_equivalence_error'] for r in ledgers),
            max_payment_group_CS_error=max(r['group_CS_equivalence_error'] for r in ledgers),
            undefined_candidate_ratios=sum(r['reduction_percent'] is None for r in candidate_comparisons),
            negative_candidate_ratios=sum(r['reduction_percent'] is not None and r['reduction_percent'] < 0
                                          for r in candidate_comparisons),
            max_transfer_cancellation_error=max(abs(r['transfer_cancellation_error']) for r in ledgers)),
        interpretation='Controlled matched analytical operational span, not new native nonmatched data. '
            'Kappa revalues the same responses; finite candidate optima end at first release.')
    # JSON normalization keeps bundle replay independent of Python tuple serialization.
    return json.loads(json.dumps(result, allow_nan=False))


def mapping():
    return dict(pinned_sources=PINNED, selected='cap120 only; all 2 resource x 2 cost x 3 quality economies',
        wait_utility='(Ag(rho)+Am-lambda*p-lambda*v*w)/mu',
        payment_utility='(Ag(rho)+Am-lambda*p-lambda*r)/mu, r=v*w',
        game_transfer='Same complementarity and payoff at every feasible fare vector, including deviations; '
            'positive-margin actual profit bounded by K*x_m/a_m auxiliary profit in frozen analytical proof.',
        proof_reference='Frozen MAIN.tex app:primitive-nash-proof and cor:capacity-transition-location',
        proof_status='Analytical inheritance conditional on verified primitives; no new numerical global proof',
        group_CS='Full group monetary logsum, including outside travelers; recipient receipts not assigned to groups',
        TS_wait='CS+platform_profit+public_charge_revenue+kappa*p0*ground_passengers',
        TS_pay='TS_wait+v*w*Y; all access receipts credited once to independent recipient',
        real_wait='Already included in logsum; not subtracted twice', external_cost='Zero in source analytical economies',
        selection='Each institution own NC and exact maximum TS on same 33 saved charges; exact ties smallest charge',
        denominator='Sum gains for aggregate ratio; zero, negative and near-zero denominators flagged, never dropped',
        inference='12 controlled analytical configurations; 396 within-path responses are not independent markets')


def _read(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + '\n', encoding='utf-8')


def _cell(value):
    if value is None:
        return ''
    return json.dumps(value, sort_keys=True, allow_nan=False) if isinstance(value, (dict, list)) else str(value)


def _csv_rows(result):
    policy_rows, group_rows = [], []
    for row in result['policies']:
        flat = {k: row[k] for k in ('case', 'institution', 'kappa', 'gain', 'negative_candidate_gains',
                                    'zero_candidate_gains', 'candidate_gains')}
        for policy in ('NC', 'SD'):
            point = row[policy]
            for key in ('charge', 'TS', 'CS', 'air_passengers', 'resource_work', 'physical_wait_minutes',
                        'source_wait_minutes', 'recipient_revenue', 'platform_profit', 'ground_margin'):
                flat[policy + '_' + key] = point[key]
            for g, mass in enumerate(point['group_mass']):
                group_rows.append(dict(case=row['case'], institution=row['institution'], kappa=row['kappa'],
                    policy=policy, charge=point['charge'], group=g, group_mass=mass,
                    CS=point['group_CS'][g], CS_per_potential=point['group_CS_per_potential'][g],
                    CS_change_from_own_NC=point['group_CS'][g] - row['NC']['group_CS'][g],
                    CS_change_per_potential=(point['group_CS'][g] - row['NC']['group_CS'][g]) / mass,
                    air_passengers=point['group_air_passengers'][g], resource_work=point['group_resource_work'][g],
                    fare_payment=point['group_fare_payment'][g], access_payment=point['group_access_payment'][g]))
        policy_rows.append(flat)
    comparison_rows = [{k: v for k, v in r.items() if k not in ('waiting_NC', 'waiting_SD', 'payment_NC', 'payment_SD')}
                       for r in result['comparisons']]
    return {'menu_ledgers.csv': result['menu_ledgers'], 'candidate_comparisons.csv': result['candidate_comparisons'],
            'policies.csv': policy_rows,
            'comparisons.csv': comparison_rows, 'summaries.csv': result['summaries'], 'group_CS.csv': group_rows}


def _outputs(folder, result):
    _dump(folder / 'results.json', result)
    _dump(folder / 'source_mapping.json', mapping())
    _dump(folder / 'operational_span.json', result['operational_span'])
    for name, rows in _csv_rows(result).items():
        with (folder / name).open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({k: _cell(v) for k, v in r.items()} for r in rows)


def _old_hashes(base):
    files = sorted((base / SOURCE).rglob('*'))
    files.extend(base / n for n in ('experiments/aam_pricing/review11_finite_paths.py',
                                   'scripts/run_review11_finite_paths.py', 'tests/test_review11_finite_paths.py'))
    return {p.relative_to(base).as_posix(): _digest(p) for p in files if p.is_file()}


def run(base, folder):
    base, folder = Path(base).resolve(), Path(folder).resolve()
    if folder.exists():
        raise FileExistsError(folder)
    if folder.is_relative_to(base / SOURCE):
        raise ValueError('Output must not modify original analytical archive')
    for name, digest in PINNED.items():
        if _digest(base / name) != digest:
            raise ValueError('Pinned source hash mismatch: ' + name)
    old = _old_hashes(base)
    owned_hashes = {name: _digest(base / name) for name in OWNED}
    result = compute(_read(base / INPUT), _read(base / SUMMARY))
    folder.mkdir(parents=True, exist_ok=False)
    for names, prefix in (((INPUT, SUMMARY), 'inputs'), ((WAIT_SOURCE, PROOF_SOURCE), 'reference_source'),
                          (OWNED, 'execution_source')):
        for name in names:
            target = folder / prefix / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(base / name, target)
    _outputs(folder, result)
    if _old_hashes(base) != old:
        raise ValueError('Original analytical archive changed during arithmetic generation')
    if {name: _digest(base / name) for name in OWNED} != owned_hashes:
        raise ValueError('Analysis source changed during generation')
    manifest = dict(schema=1, pinned_sources=PINNED, execution=owned_hashes,
        outputs={name: _digest(folder / name) for name in GENERATED},
        protected_old_files=old, protected_old_files_unchanged=True,
        new_Nash_solves=0, new_root_solves=0,
        replay_command='python -B execution_source/scripts/run_review16_institution_span.py --replay-dir .')
    _dump(folder / 'manifest.json', manifest)
    replay(folder)
    return result


def replay(folder):
    """Validate bundle bytes and redo only arithmetic with frozen analysis source."""
    folder = Path(folder).resolve()
    manifest = _read(folder / 'manifest.json')
    if (manifest.get('schema') != 1 or manifest.get('pinned_sources') != PINNED
            or set(manifest.get('execution', {})) != set(OWNED)
            or set(manifest.get('outputs', {})) != set(GENERATED)):
        raise ValueError('Incomplete or incompatible manifest hash inventory')
    for names, prefix in (((INPUT, SUMMARY), 'inputs'), ((WAIT_SOURCE, PROOF_SOURCE), 'reference_source')):
        for name in names:
            if _digest(folder / prefix / name) != PINNED[name]:
                raise ValueError('Frozen input/reference hash mismatch: ' + name)
    for key, prefix in (('execution', 'execution_source'), ('outputs', '')):
        for name, digest in manifest[key].items():
            if _digest(folder / prefix / name) != digest:
                raise ValueError('Bundle hash mismatch: ' + name)
    source = folder / 'execution_source' / OWNED[0]
    namespace = {'__file__': str(source), '__name__': 'review16_frozen_arithmetic'}
    exec(compile(source.read_bytes(), str(source), 'exec'), namespace)
    result = namespace['compute'](_read(folder / 'inputs' / INPUT), _read(folder / 'inputs' / SUMMARY))
    if result != _read(folder / 'results.json'):
        raise ValueError('Arithmetic replay differs from saved result')
    if namespace['mapping']() != _read(folder / 'source_mapping.json'):
        raise ValueError('Source mapping replay differs')
    if result['operational_span'] != _read(folder / 'operational_span.json'):
        raise ValueError('Operational span replay differs')
    for name, rows in namespace['_csv_rows'](result).items():
        with (folder / name).open(encoding='utf-8', newline='') as handle:
            actual = list(csv.DictReader(handle))
        if actual != [{k: _cell(v) for k, v in r.items()} for r in rows]:
            raise ValueError('CSV arithmetic replay differs: ' + name)
    return result
