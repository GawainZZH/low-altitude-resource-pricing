"""Report84 authorized opt-out: exactly 14 saved IDs, no extension or rescue.

Plan: RED contracts; implement masked native/vector closure and active-only
search; freeze inputs/code/budgets; run once; closed replay; policy comparison.
Original fixed-incumbent responses remain the reference, not an exit model.
"""
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
from scipy.special import logsumexp

from .config import MarketCase
from .review90_waiting import WaitingEngine
from .review_hard_capacity_repair import (
    EvaluationBudget, BudgetExceeded, coordinate_br, side_gradients)
from .review_equilibrium import projected_gradient
from .review5_best_response import (
    VectorResponse, FareBudget, AuditBudgetExceeded, segmented_search, root_distance)
from .review10_core import closed_arithmetic, digest, dump, export
from .review11_participation_audit import HET, CORE, OUTPUT as AUDIT, merge_attempts, read, jsonl, csv_rows

OUTPUT = 'results/trb_review11_optout_v1'
OWNED = ['experiments/aam_pricing/review11_optout.py',
         'scripts/run_review11_optout.py', 'tests/test_review11_optout.py']
DEPENDENCIES = ['config.py', 'review90_waiting.py', 'review_hard_capacity_repair.py',
                'review_equilibrium.py', 'review5_best_response.py', 'review10_core.py',
                'review11_participation_audit.py', 'engine.py', 'out_of_model_demand.py',
                'review90_allocation.py']


def load_frozen(root):
    root = Path(root)
    spec = read(root/CORE/'inputs'/HET/'design.json')
    merged = merge_attempts(jsonl(root/CORE/'inputs'/HET/'games.jsonl'),
                            jsonl(root/CORE/'inputs'/HET/'rescue_games.jsonl'))
    economics = {g['game']:g for g in csv_rows(root/AUDIT/'games.csv')}
    games = []
    for g in merged:
        row = economics[g['game']]
        games.append(dict(game=g['game'], menu=g['menu'], charge=g['charge'],
            accepted=bool(g['selected']), negative_margin_platforms=int(row['negative_margin_platforms']),
            min_profit=float(row['min_profit']) if row['min_profit'] else None,
            state=g.get('diagnostic', {}).get('state') if g['selected'] else None))
    return spec, games


def design(root):
    _, games = load_frozen(root)
    ids = [g['game'] for g in games if g['negative_margin_platforms']]
    if len(ids) != 14:
        raise ValueError('Authorized 14-game loss set differs')
    return dict(schema=1, games=ids, max_new_games=14, game_seconds=20, wall_seconds=360,
        reserve_seconds=80, starts=['saved_active_fares', 'fixed22'], fixed_fare=22.,
        gain_tolerance=1e-6, smooth_KKT_tolerance=1e-6, kink_one_sided_tolerance=1e-5,
        full_fare_points=[129, 257], native_profit_agreement=1e-6,
        max_nfev=80, coordinate_passes_per_start=6, evaluation_limit_per_game=40000,
        max_fare_evaluations=10000, rescue_attempts=0,
        selection='Nearest saved active-fare vector, lexicographic tie break; never welfare based',
        exclusion='Only platforms with actual operating cost + q*resource > upper fare',
        reuse='All other accepted fixed-incumbent responses have strictly positive operating profit; unilateral zero exit cannot improve them',
        inference='Two-start finite unilateral-profit evidence, not uniqueness or the complete Nash correspondence')


def exclusion_proof(case, charge):
    costs = np.array(case['base_cost']) + [0., case['cost_gap'], .5*case['cost_gap']]
    maximum_margin = case['price_upper']-costs-charge*np.array(case['platform_resource'])
    return [dict(platform=i+1, max_unit_margin=float(margin), no_profitable_entry=True)
            for i, margin in enumerate(maximum_margin) if margin < 0]


class OptOutEngine(WaitingEngine):
    def __init__(self, case, excluded, budget=None):
        self.excluded = list(excluded)
        self.active = [i for i in range(3) if i not in self.excluded]
        if not self.active or not self.excluded:
            raise ValueError('Nonempty active and excluded sets required')
        super().__init__(case, budget)

    def pack(self, active_fares):
        x = np.asarray(active_fares, float)
        if x.shape != (len(self.active),):
            raise ValueError('Optimize active coordinates only')
        p = np.full(3, self.case.price_lower, dtype=float)
        p[self.active] = x
        return p

    def choice(self, prices, rho, wait_minutes=0.):
        self.budget.choice_calls += 1
        loss = self.base_loss+self.exposure*self.source.reliability_function(rho)
        air = (self.platform_affinity-self.price_sensitivity[:, None]*prices
            -self.time_sensitivity[:, None]*self.time
            -self.reliability_sensitivity[:, None]*loss
            -self.beta[:, None]*wait_minutes)/self.logit_scale
        air[:, self.excluded] = -np.inf
        outside = (self.outside_affinity-self.price_sensitivity*self.case.outside_price
                   -self.time_sensitivity*self.case.outside_time)/self.logit_scale
        denominator = np.logaddexp(outside, logsumexp(air, axis=1))
        prob = np.exp(air-denominator[:, None])
        return self.demand_base@prob, prob, float((self.demand_base*self.logit_scale/self.price_sensitivity)@denominator)


class OptOutResponse(VectorResponse):
    def __init__(self, engine, prices, charge, platform, budget=None):
        if platform not in engine.active:
            raise ValueError('Excluded platform has analytic no-entry proof, not a fare coordinate')
        super().__init__(engine, prices, charge, platform, budget)
        self.base[:, engine.excluded] = -np.inf


def verify_active(engine, prices, charge, spec):
    p = np.asarray(prices, float)
    o = engine.outcome(p)
    gap = engine.transition_gap(p)
    near = abs(gap) <= 1e-10
    gradient = engine.gradient(p, charge, o)
    ix = engine.active
    kkt = float(np.max(abs(projected_gradient(p, gradient, engine.case.price_lower, engine.case.price_upper)[ix])))
    left, right = side_gradients(engine, p, charge) if near else (gradient, gradient)
    lv = np.where(p > engine.case.price_lower+1e-8, np.maximum(-left, 0), 0)[ix]
    rv = np.where(p < engine.case.price_upper-1e-8, np.maximum(right, 0), 0)[ix]
    first_order = max(np.max(lv), np.max(rv)) <= spec['kink_one_sided_tolerance'] if near else kkt <= spec['smooth_KKT_tolerance']
    br = [coordinate_br(engine, p, charge, m) for m in ix]
    gain = max(b['gain'] for b in br)
    profit = (p-engine.cost-charge*engine.resource)*o.demand
    verified = bool(first_order and gain <= spec['gain_tolerance'] and np.min(profit[ix]) >= 0)
    return dict(prices=p.tolist(), verified=verified, kkt=kkt, near_transition=near,
        transition_gap=gap, first_order_ok=bool(first_order), max_gain=gain,
        left_gradient=left.tolist(), right_gradient=right.tolist(),
        active_profits=profit[ix].tolist(), br=br)


def full_fare_audit(engine, prices, charge, spec, deadline):
    budget = FareBudget(spec['max_fare_evaluations'], deadline)
    coordinates = []
    p = np.asarray(prices, float)
    for m in engine.active:
        response = OptOutResponse(engine, p, charge, m, budget)
        passes, event_passes = [], []
        for points in spec['full_fare_points']:
            events = response.events(engine.case.price_lower, engine.case.price_upper, points)
            event_passes.append(events)
            passes.append(segmented_search(response.evaluate, engine.case.price_lower,
                engine.case.price_upper, p[m], points, [v['fare'] for v in events],
                max_evaluations=spec['max_fare_evaluations'], deadline=deadline))
        a, b = passes
        best = max(passes, key=lambda v:v['best_profit'])
        deviation = p.copy()
        deviation[m] = best['best_fare']
        native_base = engine.profit(p, m, charge)
        native_best = engine.profit(deviation, m, charge)
        coordinates.append(dict(platform=m+1, passes=passes, events=event_passes,
            native_base=native_base, native_best=native_best, gain=max(0., native_best-native_base),
            native_profit_error=max(abs(native_best-best['best_profit']), abs(native_base-best['baseline'])),
            gain_difference=abs(a['gain']-b['gain']),
            root_count_difference=len(b['stationary_roots'])-len(a['stationary_roots']),
            root_distance=root_distance(a['stationary_roots'], b['stationary_roots']),
            event_count_difference=len(event_passes[1])-len(event_passes[0])))
    stable = all(c['gain_difference'] <= 1e-7 and c['root_count_difference'] == 0 and
        c['event_count_difference'] == 0 and (c['root_distance'] is not None and c['root_distance'] <= 1e-6)
        for c in coordinates)
    epsilon = max(c['gain'] for c in coordinates)
    agreement = max(c['native_profit_error'] for c in coordinates)
    return dict(accepted=bool(stable and epsilon <= spec['gain_tolerance'] and agreement <= spec['native_profit_agreement']),
        epsilon=epsilon, native_profit_error=agreement, refinement_stable=stable,
        coordinates=coordinates, evaluations=budget.used, formal_interval_bound=False)


def polish_active(engine, initial, charge, spec):
    lo, hi = engine.case.price_lower, engine.case.price_upper
    def residual(x):
        p = engine.pack(x)
        return projected_gradient(p, engine.gradient(p, charge), lo, hi)[engine.active]
    fit = least_squares(residual, np.clip(initial, lo+1e-9, hi-1e-9), bounds=(lo, hi),
        max_nfev=spec['max_nfev'], xtol=1e-11, ftol=1e-11, gtol=1e-11)
    p = engine.pack(fit.x)
    gradient = engine.gradient(p, charge)
    snap = p.copy()
    for m in engine.active:
        if hi-p[m] < 1e-5 and gradient[m] > 0:
            snap[m] = hi
        elif p[m]-lo < 1e-5 and gradient[m] < 0:
            snap[m] = lo
    if np.max(abs(residual(snap[engine.active]))) <= np.max(abs(residual(p[engine.active]))):
        p = snap
    return p, dict(success=bool(fit.success), nfev=fit.nfev, status=int(fit.status))


def snapshot(engine, prices, charge):
    p = np.asarray(prices, float)
    o = engine.outcome(p)
    revenue, operating = float(p@o.demand), float(engine.cost@o.demand)
    external = float(engine.case.congestion_cost*(.15*o.rho**2+max(o.rho-.9, 0.)**3))
    profits = (p-engine.cost-charge*engine.resource)*o.demand
    return dict(prices=p.tolist(), demand=o.demand.tolist(), prob=o.prob.tolist(),
        rho=float(o.rho), binding=bool(o.binding), wait_minutes=float(o.wait_minutes),
        closure=float(o.residual), cs=float(o.cs), revenue=revenue, cost=operating, G=external,
        TS=float(o.cs+revenue-operating-external), served=float(sum(o.demand)),
        potential=float(sum(engine.demand_base)), profits=profits.tolist(),
        excluded_platforms=[m+1 for m in engine.excluded],
        placeholder_fare='Excluded fare is bookkeeping only; utility is -inf')


def solve_one(menu, game, spec, wall_deadline):
    begin = perf_counter()
    deadline = min(begin+spec['game_seconds'], wall_deadline)
    proof = exclusion_proof(menu['case'], game['charge'])
    excluded = [r['platform']-1 for r in proof]
    row = dict(game=game['game'], menu=game['menu'], charge=game['charge'],
        exclusion_proof=proof, selected=None, state=None, attempts=[], errors=[])
    e = OptOutEngine(MarketCase(**menu['case']), excluded,
        EvaluationBudget(limit=spec['evaluation_limit_per_game'], deadline=deadline))
    anchor = np.array(game['state']['prices'])[e.active]
    for label, start in zip(spec['starts'], (anchor, np.full(len(e.active), spec['fixed_fare']))):
        attempt = dict(start=label, initial_active_fares=start.tolist(), candidates=[])
        row['attempts'].append(attempt)
        try:
            p, fit = polish_active(e, start, game['charge'], spec)
            attempt['least_squares'] = fit
            for iteration in range(spec['coordinate_passes_per_start']+1):
                checked = verify_active(e, p, game['charge'], spec)
                checked['coordinate_pass'] = iteration
                attempt['candidates'].append(checked)
                if checked['verified']:
                    checked['full_fare'] = full_fare_audit(e, p, game['charge'], spec, deadline)
                    if checked['full_fare']['accepted']:
                        break
                if iteration < spec['coordinate_passes_per_start']:
                    for m in e.active:
                        p[m] = coordinate_br(e, p, game['charge'], m)['best_fare']
                    p, _ = polish_active(e, p[e.active], game['charge'], spec)
        except (BudgetExceeded, AuditBudgetExceeded, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            attempt['error'] = str(exc)
            row['errors'].append(label+': '+str(exc))
    accepted = [c for a in row['attempts'] for c in a['candidates'] if c['verified'] and c.get('full_fare', {}).get('accepted')]
    if accepted:
        chosen = min(accepted, key=lambda c:(float(np.linalg.norm(np.array(c['prices'])[e.active]-anchor)), tuple(c['prices'])))
        row['selected'] = chosen
        # This is a cached state; no post-deadline root search is needed.
        row['state'] = snapshot(e, chosen['prices'], game['charge'])
    row.update(elapsed_seconds=perf_counter()-begin, evaluations=e.budget.used, choice_calls=e.budget.choice_calls)
    return row


def policy_comparison(root, old, new, spec=None):
    if spec is None:
        spec = load_frozen(root)[0]
    changed = {g['game']:g for g in new if g['selected']}
    rows = []
    for m in spec['menus']:
        if max(m['case']['platform_resource'])/min(m['case']['platform_resource']) < 4.5-1e-12:
            continue
        original = [g for g in old if g['menu'] == m['id']]
        complete = all(g['accepted'] for g in original)
        known = [g for g in original if g['accepted']]
        best = max(known, key=lambda g:(g['state']['TS'], -g['charge']))
        nc = next((g for g in known if g['charge'] == 0), None)
        replacements = []
        for g in original:
            if not g['accepted']:
                continue
            state = changed[g['game']]['state'] if g['game'] in changed else (g['state'] if not g['negative_margin_platforms'] else None)
            if state:
                replacements.append(dict(g, state=state))
        new_complete = complete and len(replacements) == len(original)
        opt = max(replacements, key=lambda g:(g['state']['TS'], -g['charge'])) if new_complete else None
        row = dict(menu=m['id'], original_complete=complete, new_complete=new_complete,
            original_best_accepted_charge=best['charge'], original_best_accepted_TS=best['state']['TS'],
            original_optimum_charge=best['charge'] if complete else None,
            original_gain=best['state']['TS']-nc['state']['TS'] if complete and nc else None,
            original_loss_at_best=best['negative_margin_platforms'] > 0,
            new_optimum_charge=opt['charge'] if opt else None,
            new_optimum_TS=opt['state']['TS'] if opt else None,
            new_gain=opt['state']['TS']-nc['state']['TS'] if opt and nc else None,
            new_optimum_wait=opt['state']['wait_minutes'] if opt else None,
            new_optimum_served=opt['state']['served'] if opt else None,
            unavailable_ids=[g['game'] for g in original if not g['accepted'] or (g['negative_margin_platforms'] and g['game'] not in changed)])
        row['gain_change'] = row['new_gain']-row['original_gain'] if opt and row['original_gain'] is not None else None
        rows.append(row)
    return rows


def comparison_rows(old, new):
    before = {g['game']:g for g in old}
    rows = []
    for g in new:
        base = before[g['game']]['state']
        now = g['state']
        r = dict(game=g['game'], menu=g['menu'], charge=g['charge'], accepted=bool(g['selected']),
            original_TS=base['TS'], original_served=base['served'], original_wait=base['wait_minutes'],
            original_rho=base['rho'], original_demand=base['demand'], original_prices=base['prices'],
            new_TS=now['TS'] if now else None, new_served=now['served'] if now else None,
            new_wait=now['wait_minutes'] if now else None, new_rho=now['rho'] if now else None,
            new_demand=now['demand'] if now else None, new_prices=now['prices'] if now else None,
            new_profits=now['profits'] if now else None, new_binding=now['binding'] if now else None)
        r.update(TS_change=now['TS']-base['TS'] if now else None,
                 service_change=now['served']-base['served'] if now else None)
        rows.append(r)
    return rows


def source_hashes(root):
    root = Path(root)
    paths = [HET+'/'+name for name in ('design.json', 'games.jsonl', 'rescue_games.jsonl')]
    paths += [AUDIT+'/'+name for name in ('manifest.json', 'summary.json', 'games.csv')]
    paths += OWNED
    paths += ['experiments/aam_pricing/'+name for name in DEPENDENCIES]
    return {name:digest(root/name) for name in paths}


def freeze(root, folder):
    root, folder = Path(root), Path(folder)
    if folder.exists():
        raise FileExistsError('Frozen opt-out output cannot be overwritten')
    spec, games = load_frozen(root)
    plan = design(root)
    for g in games:
        if g['negative_margin_platforms']:
            menu = next(m for m in spec['menus'] if m['id'] == g['menu'])
            if exclusion_proof(menu['case'], g['charge']) != [dict(platform=3,
                    max_unit_margin=float(menu['case']['price_upper']-menu['effective']['cost'][2]-g['charge']*menu['effective']['resource'][2]), no_profitable_entry=True)]:
                raise ValueError('Uniform no-entry proof differs from authorized platform 3')
        elif g['accepted'] and g['min_profit'] <= 0:
            raise ValueError('Unaffected reuse requires positive incumbent profits')
    folder.mkdir(parents=True)
    dump(folder/'design.json', plan)
    dump(folder/'inputs.json', dict(spec=spec, games=games))
    hashes = source_hashes(root)
    for name in OWNED:
        dst = folder/'execution_source'/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/name, dst)
    dump(folder/'pre_run_freeze.json', dict(schema=1, max_new_games=14,
        frozen_before_execution=True, design_sha256=digest(folder/'design.json'),
        inputs_sha256=digest(folder/'inputs.json'), source_sha256=hashes,
        authorization='User authorized exactly the 14 audited saved losing IDs; 20 seconds/game, 360 total, two starts, no extension/rescue'))
    return plan


def verify_freeze(root, folder):
    folder = Path(folder)
    frozen = read(folder/'pre_run_freeze.json')
    if digest(folder/'design.json') != frozen['design_sha256'] or read(folder/'design.json') != design(root):
        raise ValueError('Frozen design hash or specification differs')
    if digest(folder/'inputs.json') != frozen['inputs_sha256'] or source_hashes(root) != frozen['source_sha256']:
        raise ValueError('Frozen input or source hash differs')
    return frozen


def finish(folder, inputs, results, elapsed):
    spec, old = inputs['spec'], inputs['games']
    accepted = [g for g in results if g['selected']]
    policies = policy_comparison(None, old, results, spec)
    summary = dict(planned_games=14, attempted_games=sum(bool(g['attempts']) for g in results),
        accepted_games=len(accepted), failed_games=[g['game'] for g in results if not g['selected']],
        elapsed_seconds=elapsed, no_rescue_extension=True, new_Nash_games=sum(bool(g['attempts']) for g in results),
        max_BR41_gain=max((g['selected']['max_gain'] for g in accepted), default=None),
        max_full_fare_gain=max((g['selected']['full_fare']['epsilon'] for g in accepted), default=None),
        max_native_profit_error=max((g['selected']['full_fare']['native_profit_error'] for g in accepted), default=None),
        min_active_profit=min((min(g['selected']['active_profits']) for g in accepted), default=None),
        complete_original_strong_menus=sum(r['original_complete'] for r in policies),
        complete_new_strong_menus=sum(r['new_complete'] for r in policies),
        changed_complete_menu_optima=[r['menu'] for r in policies if r['new_complete'] and r['original_optimum_charge'] != r['new_optimum_charge']],
        interpretation='Focused selected-response participation sensitivity, not a complete equilibrium correspondence')
    export(folder/'comparisons.csv', comparison_rows(old, results))
    export(folder/'policies.csv', policies)
    dump(folder/'summary.json', summary)
    dump(folder/'manifest.json', dict(files_sha256={p.relative_to(folder).as_posix():digest(p)
        for p in folder.rglob('*') if p.is_file() and p.name != 'manifest.json'}))
    return summary


def run(root, folder):
    root, folder = Path(root), Path(folder)
    verify_freeze(root, folder)
    if (folder/'games.jsonl').exists() or (folder/'run_started.json').exists():
        raise FileExistsError('One execution only; no rescue or restart extension')
    inputs, plan = read(folder/'inputs.json'), read(folder/'design.json')
    dump(folder/'run_started.json', dict(frozen_design_sha256=digest(folder/'design.json')))
    started = perf_counter()
    deadline = started+plan['wall_seconds']-plan['reserve_seconds']
    by_id = {g['game']:g for g in inputs['games']}
    menus = {m['id']:m for m in inputs['spec']['menus']}
    results = []
    for gid in plan['games']:
        g = by_id[gid]
        if perf_counter() >= deadline:
            row = dict(game=gid, menu=g['menu'], charge=g['charge'], attempts=[], selected=None,
                       state=None, errors=['Total wall budget: not attempted'])
        else:
            row = solve_one(menus[g['menu']], g, plan, deadline)
        results.append(row)
        with (folder/'games.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(dict(game=gid, accepted=bool(row['selected']),
            elapsed_seconds=row.get('elapsed_seconds'), TS=row['state']['TS'] if row['state'] else None)), flush=True)
    verify_freeze(root, folder)
    return finish(folder, inputs, results, perf_counter()-started)


def check_saved_state(menu, game):
    case, s = menu['case'], game['state']
    p, y, prob = (np.asarray(s[k], float) for k in ('prices', 'demand', 'prob'))
    mass = np.array(case['base_demand'])*case['demand_scale']
    lam, beta, gamma = (np.asarray(case[k]) for k in ('price_sensitivity', 'time_sensitivity', 'reliability_sensitivity'))
    loss = np.array(case['base_loss'])+np.array(case['exposure'])*(.55*s['rho']**case['reliability_power']+1.10*max(s['rho']-.88, 0.)**2)
    alpha = np.array(case['platform_affinity'])+np.eye(len(mass), 3)*case['differentiation']
    air = (alpha-lam[:, None]*p-beta[:, None]*(np.array(case['service_time'])+s['wait_minutes'])-gamma[:, None]*loss)/case['logit_scale']
    excluded = [r['platform']-1 for r in game['exclusion_proof']]
    air[:, excluded] = -np.inf
    outside = (np.array(case['outside_affinity'])-lam*case['outside_price']-beta*case['outside_time'])/case['logit_scale']
    den = np.logaddexp(outside, logsumexp(air, axis=1))
    rebuilt = np.exp(air-den[:, None])
    cost = np.array(case['base_cost'])+[0., case['cost_gap'], .5*case['cost_gap']]
    cs = float((mass*case['logit_scale']/lam)@den)
    external = case['congestion_cost']*(.15*s['rho']**2+max(s['rho']-.9, 0.)**3)
    profits = (p-cost-game['charge']*np.array(case['platform_resource']))*y
    tests = [(prob, rebuilt), (y, mass@prob), (s['cs'], cs), (s['profits'], profits),
             (s['TS'], cs+p@y-cost@y-external),
             (np.array(case['platform_resource'])@y, case['base_capacity']*case['capacity_scale']*s['rho'])]
    if any(np.max(abs(np.array(a)-np.array(b))) > 1e-7 for a, b in tests):
        raise ValueError('Opt-out saved-state arithmetic differs: '+game['game'])
    if any(y[i] != 0 for i in excluded) or min(profits[i] for i in range(3) if i not in excluded) < 0:
        raise ValueError('Participation condition differs')
    if game['exclusion_proof'] != exclusion_proof(case, game['charge']):
        raise ValueError('No-entry proof differs')
    return max(float(np.max(abs(np.array(a)-np.array(b)))) for a, b in tests)


def replay(folder):
    folder = Path(folder)
    for name, h in read(folder/'manifest.json')['files_sha256'].items():
        if digest(folder/name) != h:
            raise ValueError('Opt-out artifact hash differs: '+name)
    inputs = read(folder/'inputs.json')
    results = jsonl(folder/'games.jsonl')
    plan = read(folder/'design.json')
    if [g['game'] for g in results] != plan['games']:
        raise ValueError('Opt-out scheduled coverage differs')
    menus = {m['id']:m for m in inputs['spec']['menus']}
    with closed_arithmetic():
        errors = [check_saved_state(menus[g['menu']], g) for g in results if g['selected']]
        policies = policy_comparison(None, inputs['games'], results, inputs['spec'])
        comparisons = comparison_rows(inputs['games'], results)
    # CSV serialization matches the producer exactly without writing a file.
    for name, rows in [('policies', policies), ('comparisons', comparisons)]:
        saved = csv_rows(folder/(name+'.csv'))
        encoded = [{k:json.dumps(v, allow_nan=False) if isinstance(v, (list, dict)) else '' if v is None else str(v) for k, v in r.items()} for r in rows]
        if saved != encoded:
            raise ValueError('Opt-out replay table differs: '+name)
    return dict(summary=read(folder/'summary.json'), max_arithmetic_error=max(errors, default=0.),
                new_Nash_games=0, replay_engine_calls=0)
