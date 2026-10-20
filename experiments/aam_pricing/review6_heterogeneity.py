"""Frozen, bounded strong-heterogeneity waiting games and solve-free replay.

Finite best-response searches are numerical epsilon evidence, not Nash proofs.
The 268 games are fixed before the pilot and never extended by observed signs.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np

from .review3_ground import load_parameters
from .review5_differentiated import local_response, welfare_balance
from .review90_waiting import WaitingEngine
from . import review_hard_capacity_repair as repair

OUTPUT = 'results/trb_review6_heterogeneity_v1'
CONTEXTS = ((1.35, .9, 1.7), (1.7, .7, 1.7))
Q = (0, 2, 4, 8, 12, 16, 20)
KAPPAS = (0., .5, 1.)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def append(path, data):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(data, allow_nan=False)+'\n')
        f.flush()


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def freeze_design(root):
    loaded = load_parameters(Path(root))
    cases = loaded['cases']
    templates = [next(c for c in cases.values() if
                     (c.demand_scale, c.capacity_scale, c.reliability_power, c.congestion_cost)
                     == (*profile, 90.)) for profile in CONTEXTS]
    families = [dict(id='baseline', axis='baseline', level=0)]
    resources = ([.7, 1., 1.3], [.4, 1., 1.8], [.25, 1., 2.5])
    for axis, values in [('resource', resources), ('cost', (10., 15., 20.)),
                         ('preference', (.5, 1., 1.5)), ('dispersion', (.8, .5, .3))]:
        for i, value in enumerate(values, 1):
            families.append(dict(id=f'{axis}_{i}', axis=axis, level=i, value=value))
    for i in range(1, 4):
        families.append(dict(id=f'joint_{i}', axis='joint', level=i,
                             resource=resources[i-1], cost=(10., 15., 20.)[i-1],
                             preference=(.5, 1., 1.5)[i-1], mu=(.8, .5, .3)[i-1]))

    def transform(c, f, name):
        changes = dict(name=name)
        axis = f['axis']
        if axis in ('resource', 'joint'):
            changes['platform_resource'] = f['value'] if axis == 'resource' else f['resource']
        if axis in ('cost', 'joint'):
            high = f['value'] if axis == 'cost' else f['cost']
            actual = np.array([6.2, (6.2+high)/2, high])
            changes['base_cost'] = (actual-np.array([0., c.cost_gap, .5*c.cost_gap])).tolist()
        if axis in ('preference', 'joint'):
            shift = f['value'] if axis == 'preference' else f['preference']
            changes['platform_affinity'] = (np.array(c.platform_affinity)+[-shift, 0., shift]).tolist()
        if axis in ('dispersion', 'joint'):
            changes['logit_scale'] = f['value'] if axis == 'dispersion' else f['mu']
        return replace(c, **changes)

    menus = []
    for f in families:
        for i, c in enumerate(templates):
            name = f"{f['id']}_c{i}"
            case = transform(c, f, name)
            menus.append(dict(id=name, family=f['id'], kind='stress',
                              profile=list(CONTEXTS[i]), charges=list(Q), case=asdict(case)))
    offgrid = ((1.12, .83, 1.93), (1.48, 1.03, 2.13),
               (1.61, .79, 2.47), (.97, .91, 2.63))
    for i, (profile, family) in enumerate(zip(offgrid, ('resource_2', 'cost_2', 'preference_2', 'joint_2'))):
        name = f'offgrid_{i+1}'
        c = replace(templates[0], demand_scale=profile[0], capacity_scale=profile[1], reliability_power=profile[2])
        case = transform(c, next(f for f in families if f['id'] == family), name)
        menus.append(dict(id=name, family=family, kind='offgrid', profile=list(profile),
                          charges=list(range(0, 22, 2)), case=asdict(case)))
    return dict(schema=1, families=families, menus=menus, games=268, wall_seconds=600,
                pilot_ids=['baseline_c0:q0', 'baseline_c1:q12', 'resource_3_c0:q0',
                           'cost_3_c1:q4', 'preference_3_c0:q0', 'joint_3_c1:q8'],
                game_seconds=8, repair_passes=2, audit_limit=6,
                audit_rule='At most six accepted strong cases: negative actual, then nonpositive bound, then closest ratio to one; ties by game id. No new games.',
                selection='Fixed menus; continuation nearest accepted anchor; no welfare-based game/profile selection',
                dispersion_interpretation='mu is competition/taste dispersion; market power is not inferred without measured markups')


def scheduled_games(plan):
    games = [dict(game=f"{menu['id']}:q{q}", menu=menu['id'], charge=q)
             for menu in plan['menus'] for q in menu['charges']]
    lookup = {g['game']: g for g in games}
    return [lookup[k] for k in plan['pilot_ids']]+[g for g in games if g['game'] not in plan['pilot_ids']]


def make_engine(menu, budget=None):
    from .config import MarketCase
    return WaitingEngine(MarketCase(**menu['case']), budget)


def balance(prices, costs, resource, dy, wdot, value, kappa=0., ground_price=0.):
    r = welfare_balance(prices, costs, resource, dy, wdot, value, kappa, ground_price)
    adverse = r['spread']*r['turnover']
    r.update(adverse_bound=adverse, ratio=adverse/r['relief'] if r['relief'] > 0 else None)
    return r


def classify(row):
    if row['TS_derivative'] < 0:
        return 'negative_actual'
    if row['lower_bound'] > 0:
        return 'positive_bound'
    return 'nonpositive_bound_positive_actual' if row['TS_derivative'] > 0 else 'nonpositive_bound_zero_actual'


def replay_tangent(saved):
    """Independent envelope calculation using saved arrays, with no engine calls."""
    mass, lam, beta, prob = [np.array(saved[k], dtype=float) for k in ('mass', 'lam', 'beta', 'prob')]
    prices, costs, resource, pdot, dy = [np.array(saved[k], dtype=float) for k in
        ('prices', 'cost', 'resource', 'price_derivative', 'demand_derivative')]
    demand = mass@prob
    value = float(np.sum(mass*prob.sum(axis=1)*beta/lam))
    csdot = float(-demand@pdot-value*saved['friction_derivative'])
    rows = []
    for kappa in KAPPAS:
        r = balance(prices, costs, resource, dy, saved['friction_derivative'], value, kappa, saved['outside_price'])
        actual = float(csdot+demand@pdot+(prices-costs-kappa*saved['outside_price'])@dy)
        r.update(kappa=kappa, envelope_TS_derivative=actual, cs_derivative=csdot,
                 class_name=classify(r), near_bound=abs(r['lower_bound']) <= .1*max(1e-12, abs(r['relief'])))
        rows.append(r)
    return rows


def saved_ift(saved, charge, mu):
    """Analytic strict-branch IFT from saved probabilities, without demand roots.

    Differentiate fixed-w logit derivatives B,z and capacity-implied w_p,
    then differentiate F_m=y_m+(p_m-c_m-q*a_m)*J_mm.
    """
    mass, lam, beta, prob = [np.array(saved[k], dtype=float) for k in ('mass', 'lam', 'beta', 'prob')]
    p, c, a = [np.array(saved[k], dtype=float) for k in ('prices', 'cost', 'resource')]
    theta = prob.sum(axis=1)
    l, b = lam/mu, beta/mu
    B = np.einsum('g,gm,gi->mi', mass*l, prob, prob)-np.diag((mass*l)@prob)
    z = -(mass*b*(1-theta))@prob
    wp = -(a@B)/float(a@z)
    J = B+np.outer(z, wp)
    n = len(p)
    dJ = np.empty((n, n, n))
    for j in range(n):
        dp = prob*(l[:, None]*(prob[:, j, None]-np.eye(n)[j])
                   -b[:, None]*(1-theta[:, None])*wp[j])
        dt = dp.sum(axis=1)
        dB = (np.einsum('g,gm,gi->mi', mass*l, dp, prob)
              +np.einsum('g,gm,gi->mi', mass*l, prob, dp)-np.diag((mass*l)@dp))
        dz = (mass*b*dt)@prob-(mass*b*(1-theta))@dp
        dwp = -(a@dB+float(a@dz)*wp)/float(a@z)
        dJ[:, :, j] = dB+np.outer(dz, wp)+np.outer(z, dwp)
    margin = p-c-charge*a
    A = J+np.diag(np.diag(J))+margin[:, None]*np.array([dJ[m, m, :] for m in range(n)])
    gq = -a*np.diag(J)
    pdot = np.linalg.solve(A, -gq)
    return dict(profit_jacobian=A.tolist(), demand_jacobian=J.tolist(),
                price_derivative=pdot.tolist(), demand_derivative=(J@pdot).tolist(),
                friction_derivative=float(wp@pdot), jacobian_condition=float(np.linalg.cond(A)),
                gradient=(mass@prob+margin*np.diag(J)).tolist())


def pilot_gate(times, elapsed, total):
    projected = elapsed+(total-len(times))*float(np.mean(times))*1.35+30 if times else None
    return dict(pilot_games=len(times), elapsed_seconds=elapsed,
                projected_seconds=projected, remaining_games=total-len(times),
                proceed=bool(len(times) == 6 and projected < 600),
                rule='6 pilot games; mean complete-game time times 1.35 plus 30 seconds reserve; timing only')


def freeze_rescue(games):
    return dict(game_ids=[g['game'] for g in games if not g['selected']],
                attempts_per_game=1, wall_seconds=90, game_seconds=8,
                gain_tolerance=1e-6, max_nfev=80,
                anchor='Original audited candidate with smallest BR41 gain; original order for exact ties',
                procedure='Exact common-shift capacity projection; if rejected, one least-squares fit of capacity equality and one-sided gradient violations, then exact projection and unchanged BR41 verification',
                no_new_games=True, no_tolerance_relaxation=True,
                full_interval_audit='All rescued candidates, subject to the same rescue wall deadline')


def rescue(root, out):
    """One authorized second attempt on original rejections; preserve first batch."""
    from scipy.optimize import least_squares
    from .review5_best_response import audit_candidate
    root, out = Path(root), Path(out)
    if (out/'rescue_plan.json').exists():
        raise FileExistsError('Second attempt is frozen and may run only once')
    plan = json.loads((out/'design.json').read_text(encoding='utf-8'))
    original = [json.loads(x) for x in (out/'games.jsonl').read_text(encoding='utf-8').splitlines()]
    design = freeze_rescue(original)
    initial_status = json.loads((out/'run_status.json').read_text(encoding='utf-8'))
    if initial_status['elapsed_seconds']+design['wall_seconds'] > 600:
        raise ValueError('No remaining numerical wall budget for frozen rescue')
    design['initial_games_sha256'] = digest(out/'games.jsonl')
    design['source_sha256'] = digest(root/'experiments/aam_pricing/review6_heterogeneity.py')
    write_json(out/'rescue_plan.json', design)
    write_json(out/'initial_summary.json', replay(out)['summary'])
    started = perf_counter()
    deadline = started+design['wall_seconds']
    for gid in design['game_ids']:
        if perf_counter()+2 >= deadline:
            break
        g = next(g for g in original if g['game'] == gid)
        menu = next(m for m in plan['menus'] if m['id'] == g['menu'])
        anchor = min(g['audited_candidates'], key=lambda a: a['max_gain'])['prices']
        begin = perf_counter()
        e = make_engine(menu, repair.EvaluationBudget(limit=20000, deadline=min(deadline, begin+design['game_seconds'])))
        r = dict(game=gid, menu=g['menu'], family=g['family'], kind=g['kind'], charge=g['charge'],
                 selected=None, attempts=[], audited_candidates=[], anchor=anchor, attempt_number=2,
                 root_cause_hypothesis='Near capacity kink, not regular smooth stationarity')
        def verify(p, label):
            a = repair.verify_candidate(e, p, g['charge'])
            a['candidate_source'] = label
            r['attempts'].append(a)
            r['audited_candidates'].append(a)
            return a
        try:
            old = e.outcome(anchor)
            r['anchor_wait'] = old.wait_minutes
            r['anchor_gap'] = e.transition_gap(anchor)
            projection = repair.project_transition(e, anchor)
            r['projection'] = projection
            if projection is None:
                raise ValueError('No common-shift capacity projection within fare box')
            a = verify(projection['prices'], 'exact_projection_of_saved_candidate')
            if not a['verified']:
                def residual(p):
                    left, right = repair.side_gradients(e, p, g['charge'])
                    gap = e.transition_gap(p)
                    return np.r_[100*gap, np.minimum(left, 0), np.maximum(right, 0)]
                opt = least_squares(residual, projection['prices'],
                                    bounds=(e.case.price_lower, e.case.price_upper),
                                    max_nfev=design['max_nfev'], xtol=1e-11, ftol=1e-11, gtol=1e-11)
                r['one_sided_fit'] = dict(prices=opt.x.tolist(), nfev=opt.nfev,
                                         residual=residual(opt.x).tolist(), success=bool(opt.success))
                projected = repair.project_transition(e, opt.x)
                if projected is not None:
                    a = verify(projected['prices'], 'one_bounded_one_sided_fit_exact_projection')
            if a['verified']:
                r['selected'] = a
                r['diagnostic'] = diagnose(e, a, g['charge'])
                r['full_interval_audit'] = audit_candidate(e, a['prices'], g['charge'], deadline=min(deadline, e.budget.deadline))
        except Exception as exc:
            r['error'] = repr(exc)
        r.update(elapsed=perf_counter()-begin, evaluations=e.budget.used,
                 choice_calls=e.budget.choice_calls)
        append(out/'rescue_games.jsonl', r)
        print(json.dumps(dict(game=gid, rescued=bool(r['selected']),
                              status=r.get('diagnostic', {}).get('status', 'rejected'),
                              seconds=r['elapsed'])), flush=True)
    rescues = [json.loads(x) for x in (out/'rescue_games.jsonl').read_text(encoding='utf-8').splitlines()]
    write_json(out/'rescue_status.json', dict(elapsed_seconds=perf_counter()-started,
        cumulative_numerical_seconds=initial_status['elapsed_seconds']+perf_counter()-started,
        attempted=len(rescues), rescued=sum(bool(g['selected']) for g in rescues),
        original_games_unchanged=digest(out/'games.jsonl') == design['initial_games_sha256']))
    return replay(out, write=True)


def snapshot(e, prices, o):
    p = np.array(prices)
    revenue = float(p@o.demand)
    cost = float(e.cost@o.demand)
    G = float(e.case.congestion_cost*(.15*o.rho**2+max(o.rho-.9, 0)**3))
    ground = float(e.demand_base.sum()-o.demand.sum())
    return dict(prices=p.tolist(), demand=o.demand.tolist(), prob=o.prob.tolist(),
                rho=float(o.rho), binding=bool(o.binding), wait_minutes=float(o.wait_minutes),
                closure=float(o.residual), cs=float(o.cs), revenue=revenue, cost=cost, G=G,
                ground=ground, TS=float(o.cs+revenue-cost-G),
                potential=float(e.demand_base.sum()), served=float(o.demand.sum()),
                capacity=float(e.capacity), markup=(p-e.cost).tolist())


def diagnose(e, selected, charge):
    p = selected['prices']
    o = e.outcome(p)
    result = dict(state=snapshot(e, p, o), status='not_strict_interior')
    if not o.binding or o.wait_minutes <= 1e-6:
        result['status'] = 'nonbinding_or_transition'
        return result
    if min(np.array(p)-e.case.price_lower) <= 2e-4 or min(e.case.price_upper-np.array(p)) <= 2e-4:
        return result
    try:
        first = local_response(e, p, charge)
        second = local_response(e, p, charge, jacobian_step=1e-4)
        saved = dict(prices=p, cost=e.cost.tolist(), resource=e.resource.tolist(),
                     mass=e.demand_base.tolist(), lam=e.price_sensitivity.tolist(), beta=e.beta.tolist(),
                     prob=o.prob.tolist(), outside_price=e.case.outside_price,
                     price_derivative=first['price_derivative'], demand_derivative=first['demand_derivative'],
                     friction_derivative=first['friction_derivative'])
        rows = replay_tangent(saved)
        for row in rows:
            native = local_response(e, p, charge, row['kappa'])
            row.update(finite_derivative=native['finite_derivative'], derivative_error=native['derivative_error'])
        step_error = float(np.max(abs(np.array(first['price_derivative'])-second['price_derivative'])))
        result.update(status='valid_IFT', tangent=saved, rows=rows, local=first,
                      jacobian_step_error=step_error)
        if step_error > 1e-3 or max(r['derivative_error'] for r in rows) > 1e-3:
            result['status'] = 'invalid_IFT_numerical_agreement'
    except (ValueError, np.linalg.LinAlgError) as exc:
        result.update(status='invalid_IFT', error=str(exc))
    return result


def solve_one(menu, game, anchor, deadline, plan):
    """Bounded existing-engine repair; preserve all verify and BR call records."""
    from copy import deepcopy
    from unittest.mock import patch
    started = perf_counter()
    budget = repair.EvaluationBudget(limit=20000, deadline=min(deadline, started+plan['game_seconds']))
    e = make_engine(menu, budget)
    attempts, br_calls = [], []
    original_verify, original_br = repair.verify_candidate, repair.coordinate_br

    def track_verify(*args, **kwargs):
        r = original_verify(*args, **kwargs)
        attempts.append(deepcopy(r))
        return r

    def track_br(*args, **kwargs):
        r = original_br(*args, **kwargs)
        br_calls.append(r)
        return r

    r = dict(**game, family=menu['family'], kind=menu['kind'], selected=None)
    try:
        with patch.object(repair, 'verify_candidate', track_verify), patch.object(repair, 'coordinate_br', track_br):
            r.update(repair.repair_game(e, anchor, game['charge'], game['game'], max_passes=plan['repair_passes']))
            if not r['selected'] and r['attempts']:
                final = track_verify(e, repair.smooth_polish(e, r['attempts'][-1]['prices'], game['charge']), game['charge'])
                final['candidate_source'] = 'one_predeclared_final_polish'
                r['attempts'].append(final)
                if final['verified']:
                    r['selected'] = final
        if r['selected']:
            r['diagnostic'] = diagnose(e, r['selected'], game['charge'])
    except Exception as exc:
        r['error'] = repr(exc)
    r.update(audited_candidates=attempts, coordinate_br_calls=br_calls,
             evaluations=budget.used, choice_calls=budget.choice_calls,
             elapsed=perf_counter()-started)
    return r


def effective(e):
    return dict(resource=e.resource.tolist(), cost=e.cost.tolist(), mass=e.demand_base.tolist(),
                capacity=float(e.capacity), platform_affinity=e.platform_affinity.tolist(),
                lam=e.price_sensitivity.tolist(), beta=e.beta.tolist(), mu=e.logit_scale,
                time=e.time.tolist(), reliability_sensitivity=e.reliability_sensitivity.tolist(),
                outside_affinity=e.outside_affinity.tolist(), base_loss=e.base_loss.tolist(),
                exposure=e.exposure.tolist())


def summarize(plan, games):
    rows, states, coverage, diagnostics = [], [], [], []
    for g in games:
        d = g.get('diagnostic', {})
        selected = g.get('selected')
        candidates = g.get('audited_candidates', [])
        best = min(candidates, key=lambda x: x['max_gain']) if candidates else None
        diagnostics.append(dict(game=g['game'], family=g['family'], kind=g['kind'],
            menu=g['menu'], charge=g['charge'], accepted=bool(selected),
            status=d.get('status', 'rejected'), error=g.get('error', d.get('error')),
            candidate_N=len(candidates), elapsed_seconds=g['elapsed'],
            max_BR41_gain=selected['max_gain'] if selected else None,
            best_rejected_candidate_gain=best['max_gain'] if best and not selected else None,
            best_rejected_candidate_KKT=best['kkt'] if best and not selected else None))
        if d.get('state'):
            s = d['state']
            menu = next(m for m in plan['menus'] if m['id'] == g['menu'])
            for k in KAPPAS:
                ts = s['TS']+k*menu['case']['outside_price']*s['ground']
                states.append(dict(game=g['game'], menu=g['menu'], family=g['family'], kind=g['kind'],
                                   charge=g['charge'], kappa=k, TS=ts, wait_minutes=s['wait_minutes'],
                                   rho=s['rho'], served=s['served'], potential=s['potential'],
                                   TS_per_potential=ts/s['potential'], TS_per_served=ts/s['served'],
                                   TS_per_capacity=ts/s['capacity'], max_BR41_gain=g['selected']['max_gain'],
                                   **{f'markup_{i+1}': v for i, v in enumerate(s['markup'])},
                                   **{f'net_markup_{i+1}': v-g['charge']*menu['effective']['resource'][i]
                                      for i, v in enumerate(s['markup'])}))
        if d.get('status') == 'valid_IFT':
            rows += [dict(game=g['game'], menu=g['menu'], family=g['family'], kind=g['kind'], charge=g['charge'],
                          jacobian_condition=d['local']['jacobian_condition'],
                          jacobian_step_error=d['jacobian_step_error'], **r) for r in d['rows']]
    for kind in ('stress', 'offgrid'):
        for fam in sorted({g['family'] for g in games if g['kind'] == kind}):
            group = [g for g in games if g['kind'] == kind and g['family'] == fam]
            for k in KAPPAS:
                subset = [r for r in rows if r['kind'] == kind and r['family'] == fam and r['kappa'] == k]
                ratios = [r['ratio'] for r in subset if r['ratio'] is not None]
                coverage.append(dict(kind=kind, family=fam, kappa=k, attempted_games=len(group),
                    accepted_games=sum(bool(g['selected']) for g in group),
                    strict_interior_valid_IFT=len(subset),
                    rejected_games=sum(not bool(g['selected']) for g in group),
                    invalid_IFT=sum(g.get('diagnostic', {}).get('status', '').startswith('invalid_IFT') for g in group),
                    fare_boundary=sum(g.get('diagnostic', {}).get('status') == 'not_strict_interior' for g in group),
                    nonbinding_or_transition=sum(g.get('diagnostic', {}).get('status') == 'nonbinding_or_transition' for g in group),
                    positive_bound=sum(r['lower_bound'] > 0 for r in subset),
                    nonpositive_bound=sum(r['lower_bound'] <= 0 for r in subset),
                    negative_actual=sum(r['TS_derivative'] < 0 for r in subset),
                    near_bound=sum(r['near_bound'] for r in subset),
                    nonpositive_relief=sum(r['relief'] <= 0 for r in subset),
                    max_ratio=max(ratios, default=None),
                    min_lower_bound=min((r['lower_bound'] for r in subset), default=None),
                    min_actual_derivative=min((r['TS_derivative'] for r in subset), default=None)))
    nc = {(s['menu'], s['kappa']): s for s in states if s['charge'] == 0}
    for s in states:
        base = nc.get((s['menu'], s['kappa']))
        gain = s['TS']-base['TS'] if base else None
        s.update(NC_available=base is not None, TS_gain=gain,
                 TS_gain_per_potential=gain/s['potential'] if base else None,
                 TS_gain_per_served_NC=gain/base['served'] if base else None,
                 TS_gain_per_capacity=gain/next(m['effective']['capacity'] for m in plan['menus'] if m['id'] == s['menu']) if base else None)
    menus = []
    for m in plan['menus']:
        group = [g for g in games if g['menu'] == m['id']]
        accepted = [g for g in group if g['selected'] and g.get('diagnostic', {}).get('state')]
        menus.append(dict(menu=m['id'], family=m['family'], kind=m['kind'],
                          planned=len(m['charges']), attempted=len(group), accepted=len(accepted),
                          complete=len(accepted) == len(m['charges'])))
    axes = []
    for kind in ('stress', 'offgrid'):
        for axis in ('baseline', 'resource', 'cost', 'preference', 'dispersion', 'joint'):
            group = [r for r in diagnostics if r['kind'] == kind and r['family'].split('_')[0] == axis]
            if group:
                axes.append(dict(kind=kind, axis=axis, games=len(group),
                    accepted=sum(r['accepted'] for r in group),
                    rejected=sum(not r['accepted'] for r in group),
                    interior_IFT=sum(r['status'] == 'valid_IFT' for r in group),
                    fare_boundary=sum(r['status'] == 'not_strict_interior' for r in group),
                    nonbinding_or_transition=sum(r['status'] == 'nonbinding_or_transition' for r in group),
                    invalid_IFT=sum(r['status'].startswith('invalid_IFT') for r in group)))
    summary = dict(planned_games=plan['games'], attempted_games=len(games),
                   accepted_games=sum(bool(g['selected']) for g in games),
                   rejected_games=sum(not bool(g['selected']) for g in games),
                   diagnostic_status=dict(Counter(g.get('diagnostic', {}).get('status', 'rejected') for g in games)),
                   complete_menus=sum(m['complete'] for m in menus), total_menus=len(menus),
                   max_BR41_gain=max((g['selected']['max_gain'] for g in games if g['selected']), default=None),
                   coverage=coverage, new_Nash_solves_on_replay=0, formal_global_Nash_claim=False)
    return dict(summary=summary, derivatives=rows, states=states, coverage=coverage, menus=menus,
                diagnostics=diagnostics, axes=axes)


def replay(out, write=False):
    """Rebuild tables and independently check saved arithmetic; never solve games."""
    out = Path(out)
    plan = json.loads((out/'design.json').read_text(encoding='utf-8'))
    original = [json.loads(line) for line in (out/'games.jsonl').read_text(encoding='utf-8').splitlines()]
    rescues = []
    if (out/'rescue_games.jsonl').exists():
        rescues = [json.loads(line) for line in (out/'rescue_games.jsonl').read_text(encoding='utf-8').splitlines()]
        allowed = {g['game'] for g in original if not g['selected']}
        if len({g['game'] for g in rescues}) != len(rescues) or any(g['game'] not in allowed for g in rescues):
            raise ValueError('Second attempts may only cover original rejections once')
    replacement = {g['game']: g for g in rescues if g['selected']}
    games = [replacement.get(g['game'], g) for g in original]
    expected = {g['game'] for g in scheduled_games(plan)}
    if len({g['game'] for g in games}) != len(games) or any(g['game'] not in expected for g in games):
        raise ValueError('Duplicate or unplanned game')
    errors, ift_errors = [], []
    def check(a, b):
        errors.append(abs(float(a)-float(b)))
        if not np.isclose(a, b, rtol=1e-8, atol=1e-7):
            raise ValueError(f'Saved arithmetic differs: {a}, {b}')
    for g in games:
        d = g.get('diagnostic', {})
        if d.get('state'):
            s = d['state']
            m = next(m for m in plan['menus'] if m['id'] == g['menu'])
            e = m['effective']
            p, y, mass, prob = [np.array(v) for v in (s['prices'], s['demand'], e['mass'], s['prob'])]
            for a, b in zip(y, mass@prob):
                check(a, b)
            check(s['cost'], np.array(e['cost'])@y)
            check(s['revenue'], p@y)
            check(s['TS'], s['cs']+s['revenue']-s['cost']-s['G'])
            check(s['G'], m['case']['congestion_cost']*(.15*s['rho']**2+max(s['rho']-.9, 0)**3))
            check(s['ground'], mass.sum()-y.sum())
            check(np.array(e['resource'])@y, e['capacity']*s['rho'])
            # Reconstruct the waiting logsum from fixed outside utility and shares.
            outside = (np.array(e['outside_affinity'])-np.array(e['lam'])*m['case']['outside_price']
                       -np.array(e['beta'])*m['case']['outside_time'])/e['mu']
            check(s['cs'], np.sum(mass*e['mu']/np.array(e['lam'])*(outside-np.log1p(-prob.sum(axis=1)))))
        if d.get('tangent'):
            rebuilt = replay_tangent(d['tangent'])
            menu = next(m for m in plan['menus'] if m['id'] == g['menu'])
            analytic = saved_ift(d['tangent'], g['charge'], menu['effective']['mu'])
            for field in ('price_derivative', 'demand_derivative', 'friction_derivative'):
                err = float(np.max(abs(np.array(analytic[field])-d['tangent'][field])))
                ift_errors.append(err)
                if err > 2e-5:
                    raise ValueError(f'Analytic saved IFT differs at {g["game"]}: {field} {err}')
            for a, b in zip(rebuilt, d['rows']):
                for k in ('TS_derivative', 'lower_bound', 'relief', 'adverse_bound', 'envelope_TS_derivative'):
                    check(a[k], b[k])
                if a['ratio'] is not None:
                    check(a['ratio'], b['ratio'])
    result = summarize(plan, games)
    result['summary']['replay_max_arithmetic_error'] = max(errors, default=0.)
    result['summary']['analytic_IFT_max_error'] = max(ift_errors, default=0.)
    result['summary'].update(initial_accepted_games=sum(bool(g['selected']) for g in original),
                             second_attempt_games=len(rescues), rescued_games=len(replacement),
                             total_game_attempts=len(original)+len(rescues))
    if write:
        for name in ('derivatives', 'states', 'coverage', 'menus', 'diagnostics', 'axes'):
            write_csv(out/f'{name}.csv', result[name])
        write_json(out/'summary.json', result['summary'])
    return result


def audit_selected(out, plan, games, deadline):
    from .review5_best_response import audit_candidate
    eligible = [g for g in games if g['family'] != 'baseline' and
                g.get('diagnostic', {}).get('status') == 'valid_IFT']
    def key(g):
        rows = g['diagnostic']['rows']
        return (not any(r['TS_derivative'] < 0 for r in rows),
                not any(r['lower_bound'] <= 0 for r in rows),
                min(abs(1-r['ratio']) for r in rows if r['ratio'] is not None), g['game'])
    chosen = sorted(eligible, key=key)[:plan['audit_limit']]
    write_json(out/'audit_selection.json', [g['game'] for g in chosen])
    for g in chosen:
        if perf_counter()+2 >= deadline:
            break
        menu = next(m for m in plan['menus'] if m['id'] == g['menu'])
        e = make_engine(menu, repair.EvaluationBudget(deadline=deadline))
        try:
            audit = audit_candidate(e, g['selected']['prices'], g['charge'], deadline=deadline)
        except Exception as exc:
            audit = dict(error=repr(exc))
        append(out/'full_interval_audits.jsonl', dict(game=g['game'], audit=audit))


def run(root, out=None):
    root = Path(root)
    out = Path(out) if out else root/OUTPUT
    if out.exists():
        raise FileExistsError('Frozen run already exists; use replay, never silently extend')
    out.mkdir(parents=True)
    plan = freeze_design(root)
    for m in plan['menus']:
        m['effective'] = effective(make_engine(m))
    write_json(out/'design.json', plan)
    dependencies = ['experiments/aam_pricing/'+n for n in
        ('engine.py', 'config.py', 'review_equilibrium.py', 'review_hard_capacity_repair.py',
         'review90_waiting.py', 'review3_ground.py', 'review5_differentiated.py', 'review5_best_response.py')]
    dependencies += ['results/trb_review_parameters_v1/full_parameters.json',
                     'submission_packages/TRA_V290_LaTeX_Project/MAIN.tex',
                     'submission_packages/TRA_V290_LaTeX_Project/MAIN.pdf']
    source = dependencies+['experiments/aam_pricing/review6_heterogeneity.py',
                           'scripts/run_review6_heterogeneity.py', 'tests/test_review6_heterogeneity.py']
    hashes = {p: digest(root/p) for p in source}
    write_json(out/'provenance.json', dict(sha256=hashes, design_sha256=digest(out / 'design.json')))
    started = perf_counter()
    deadline = started+600
    games, anchors, gate = [], {}, None
    for index, spec in enumerate(scheduled_games(plan)):
        if perf_counter()+2 >= deadline:
            break
        menu = next(m for m in plan['menus'] if m['id'] == spec['menu'])
        g = solve_one(menu, spec, anchors.get(menu['id'], [22., 22., 22.]), deadline, plan)
        games.append(g)
        append(out/'games.jsonl', g)
        if g['selected']:
            anchors[menu['id']] = g['selected']['prices']
        write_json(out/'checkpoint.json', dict(completed=len(games), planned=268,
                    elapsed_seconds=perf_counter()-started, last_game=g['game']))
        print(json.dumps(dict(completed=len(games), game=g['game'], accepted=bool(g['selected']),
                              status=g.get('diagnostic', {}).get('status', 'rejected'), seconds=g['elapsed'])), flush=True)
        if index == 5:
            gate = pilot_gate([x['elapsed'] for x in games], perf_counter()-started, 268)
            write_json(out/'pilot.json', gate)
            print('PILOT '+json.dumps(gate), flush=True)
            if not gate['proceed']:
                break
    audit_selected(out, plan, games, deadline)
    result = replay(out, write=True)
    protected_changes = [p for p in dependencies if digest(root/p) != hashes[p]]
    write_json(out/'run_status.json', dict(elapsed_seconds=perf_counter()-started, pilot=gate,
              completed_games=len(games), planned_games=268, protected_changes=protected_changes,
              incomplete_reason=None if len(games) == 268 else 'pilot time gate or 600-second deadline',
              formal_global_Nash_claim=False))
    print(json.dumps(result['summary']), flush=True)
    return result
