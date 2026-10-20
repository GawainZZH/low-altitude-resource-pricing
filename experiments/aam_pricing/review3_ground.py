"""Fixed-ground-game surplus sensitivity on frozen finite response menus.

No price optimization is called. kappa*p0 is the assumed ground net receipt
p0-c0, not gross revenue in addition to an unspecified resource cost.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy

from .config import MarketCase
from .review_benchmark_correction import (
    EXPANDED_SLOPES, THRESHOLDS, group_cases, read_csv, select_ut_row,
)
from .review_equilibrium import ReviewEngine
from .review_mechanisms import resource_penalty
from .review90_allocation import group_surplus_difference
from .review90_waiting import WaitingEngine, group_logsum

SOFT = 'results/comprehensive_state_grid_v1_full/comprehensive_state_grid_charge_search_records.csv'
HARD = 'results/trb_review_hard_capacity_repair_v1/corrected_hard_charge_records.csv'
PARAMETERS = 'results/trb_review_parameters_v1/full_parameters.json'
WAITING = 'results/trb_review90_waiting_v1/waiting_games.jsonl'
CORRIDOR = 'results/calibrated_anchor_case_v1/anchor_charge_search_records.csv'
SEQUENCE = 'results/dynamic_sequence_v1/dynamic_sequence_charge_search_records.csv'
MANUSCRIPT = 'submission_packages/TRA_V290_LaTeX_Project/MAIN.tex'
OUTPUT = 'results/trb_review3_ground_v1'
KAPPAS = (0., .5, 1.)
TOL = 1.e-8


def load_parameters(root):
    raw = json.loads((root/PARAMETERS).read_text(encoding='utf-8'))
    return dict(cases={v['dataclass']['name']: MarketCase(**v['dataclass'])
                       for v in raw['full_state_grid']},
                anchor=MarketCase(**raw['anchor']['dataclass']), raw=raw)


def accounting(row, case, kappa):
    if not math.isfinite(kappa) or not 0 <= kappa <= 1:
        raise ValueError('Ground net margin fraction must be in [0,1]')
    demand = np.array([float(row[f'demand_{j}']) for j in (1, 2, 3)])
    prices = np.array([float(row[f'price_{j}']) for j in (1, 2, 3)])
    mass = math.fsum(case.base_demand)*case.demand_scale
    us = float(row['welfare'])
    if (not np.all(np.isfinite(demand)) or not np.all(np.isfinite(prices))
            or not math.isfinite(us) or np.min(demand) < 0
            or not math.isfinite(mass) or mass <= 0
            or not math.isfinite(case.outside_price) or case.outside_price < 0
            or demand.sum() > mass+TOL):
        raise ValueError('Finite, feasible passenger masses and monetary inputs required')
    ground = max(0., mass-float(demand.sum()))
    receipts = float(prices @ demand)
    net = kappa*case.outside_price*ground
    return dict(US=us, TS=us+receipts, W=us+receipts+net,
                air_fare_receipts=receipts, ground_demand=ground,
                ground_price=case.outside_price,
                ground_cost_per_trip=(1-kappa)*case.outside_price,
                ground_net_receipt=net)


def optimize_menu(rows, cases, kappa, signal='utilization'):
    grouped = group_cases(rows)
    scored = {}
    for name, group in grouped.items():
        values = [dict(r, charge=float(r['charge']), **accounting(r, cases[name], kappa)) for r in group]
        zero = values[0]['W']
        for value in values:
            value['gain'] = value['W']-zero
        scored[name] = values
    # Existing UT rule and deterministic candidate ties are retained verbatim.
    tuning = []
    for threshold in THRESHOLDS:
        for slope in EXPANDED_SLOPES:
            selected = [select_ut_row(g, threshold, slope, signal) for g in scored.values()]
            tuning.append(dict(threshold=threshold, slope=slope,
                               mean_welfare=math.fsum(r['W'] for r in selected)/len(selected),
                               mean_gain=math.fsum(r['gain'] for r in selected)/len(selected),
                               mean_charge=math.fsum(r['charge'] for r in selected)/len(selected)))
    best = max(tuning, key=lambda r: (r['mean_welfare'], -r['mean_charge'], -r['slope'], r['threshold']))
    policies = {p: {} for p in ('NC', 'SD', 'UT', 'RM')}
    for name, group in scored.items():
        resource = np.array(cases[name].platform_resource)
        def revenue(r):
            return r['charge']*float(resource @ [float(r[f'demand_{j}']) for j in (1, 2, 3)])
        policies['NC'][name] = group[0]
        policies['SD'][name] = max(group, key=lambda r: (r['W'], -r['charge']))
        policies['UT'][name] = select_ut_row(group, best['threshold'], best['slope'], signal)
        policies['RM'][name] = max(group, key=lambda r: (revenue(r), -r['charge']))
    return dict(policies=policies, best_ut=best, ut_candidates=tuning, signal=signal, scored=scored)


def policy_summary(sd, selected, old):
    if not sd or sd.keys() != selected.keys() or sd.keys() != old.keys():
        raise ValueError('Identical nonempty state denominators required')
    names = sorted(sd)
    gains = np.array([selected[s]['gain'] for s in names])
    regret = np.array([sd[s]['gain']-selected[s]['gain'] for s in names])
    if not np.all(np.isfinite(gains)) or not np.all(np.isfinite(regret)) or min(regret) < -TOL:
        raise ValueError('Finite gains and a dominating SD reference required')
    positive = np.array([selected[s]['charge'] > TOL for s in names])
    previous = np.array([float(old[s]['charge']) > TOL for s in names])
    tail = sorted(names, key=lambda s: (-(sd[s]['gain']-selected[s]['gain']), s))[:math.ceil(.05*len(names))]
    sd_gain = math.fsum(sd[s]['gain'] for s in names)/len(names)
    return dict(states=len(names), positive_count=int(sum(positive)), positive_share=float(np.mean(positive)),
                mean_charge=float(np.mean([selected[s]['charge'] for s in names])),
                mean_gain=float(np.mean(gains)), mean_regret=float(np.mean(regret)),
                recovery_vs_sd=float(np.mean(gains))/sd_gain if sd_gain > TOL else None,
                p90_regret=float(np.quantile(regret, .90)), p95_regret=float(np.quantile(regret, .95)),
                p99_regret=float(np.quantile(regret, .99)), max_regret=float(max(regret)),
                negative_states=int(sum(gains < -TOL)), min_gain=float(min(gains)),
                old_positive_new_positive=int(sum(previous & positive)),
                old_positive_new_zero=int(sum(previous & ~positive)),
                old_zero_new_positive=int(sum(~previous & positive)),
                old_zero_new_zero=int(sum(~previous & ~positive)),
                same_charge_count=sum(float(old[s]['charge']) == selected[s]['charge'] for s in names),
                tail_count=len(tail), tail_states=tail,
                tail_mean_regret=float(np.mean([sd[s]['gain']-selected[s]['gain'] for s in tail])),
                tail_total_regret=math.fsum(sd[s]['gain']-selected[s]['gain'] for s in tail),
                tail_share_of_total_regret=(math.fsum(sd[s]['gain']-selected[s]['gain'] for s in tail)
                                           /float(sum(regret)) if sum(regret) > TOL else None))


def group_changes(engine, before, after):
    prices = [np.array([float(r[f'price_{j}']) for j in (1, 2, 3)]) for r in (before, after)]
    lo, hi = [engine.outcome(p) for p in prices]
    shares0, shares1 = lo.prob.sum(axis=1), hi.prob.sum(axis=1)
    cs = group_surplus_difference(engine.demand_base, engine.price_sensitivity,
                                  shares0, shares1, engine.logit_scale)
    direct = group_logsum(engine, prices[1], hi)-group_logsum(engine, prices[0], lo)
    served0, served1 = engine.demand_base*shares0, engine.demand_base*shares1
    return [dict(group=g+1, mass=float(engine.demand_base[g]),
                 service_probability_before=float(shares0[g]), service_probability_after=float(shares1[g]),
                 service_probability_change=float(shares1[g]-shares0[g]),
                 served_before=float(served0[g]), served_after=float(served1[g]),
                 served_change=float(served1[g]-served0[g]),
                 served_composition_change=float(served1[g]/sum(served1)-served0[g]/sum(served0)),
                 cs_change=float(cs[g]), cs_per_potential_person=float(cs[g]/engine.demand_base[g]),
                 identity_error=float(cs[g]-direct[g]),
                 wait_change=float(getattr(hi, 'wait_minutes', 0.)-getattr(lo, 'wait_minutes', 0.)))
            for g in range(3)]


def corridor_analysis(rows, case):
    group = group_cases(rows)[case.name]
    nc = group[0]
    def components(r):
        demand0, demand1 = [np.array([float(v[f'demand_{j}']) for j in (1, 2, 3)]) for v in (nc, r)]
        cost = np.array(case.base_cost)+np.array((0., case.cost_gap, .5*case.cost_gap))
        operating = float(cost @ (demand0-demand1))
        system = case.congestion_cost*(resource_penalty(float(nc['utilization']))-resource_penalty(float(r['utilization'])))
        us = float(r['welfare'])-float(nc['welfare'])
        fare = accounting(r, case, 0.)['air_fare_receipts']-accounting(nc, case, 0.)['air_fare_receipts']
        return dict(net_traveler_surplus=us-operating-system, operating_saving=operating,
                    system_relief=system, fare_receipt_change=fare)
    result = dict(charge_grid=[float(r['charge']) for r in group], monetary_unit_usd=5.,
                  source='frozen saved fares and demand; no Nash rerun', NC_TS=accounting(nc, case, 0.)['TS'])
    for objective in ('US', 'TS'):
        chosen = max(group, key=lambda r: (accounting(r, case, 0.)[objective], -float(r['charge'])))
        gain = accounting(chosen, case, 0.)[objective]-accounting(nc, case, 0.)[objective]
        parts = components(chosen)
        result[objective] = dict(charge=float(chosen['charge']), gain=gain, **parts,
                                gain_under_TS=accounting(chosen, case, 0.)['TS']-result['NC_TS'],
                                component_identity_error=(accounting(chosen, case, 0.)['TS']-result['NC_TS']-sum(parts.values())))
    result['TS_gain_at_US20'] = accounting(next(r for r in group if float(r['charge']) == 20.), case, 0.)['TS']-result['NC_TS']
    result['TS_gain_advantage_vs_US20'] = result['TS']['gain']-result['TS_gain_at_US20']
    return result


def smooth_sequence(rows, weight=1.):
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('Nonnegative finite smoothing weight required')
    grouped = defaultdict(list)
    for row in rows:
        r = dict(row, case=str(int(row['period'])))
        r['welfare'] = float(r['welfare'])+sum(float(r[f'price_{j}'])*float(r[f'demand_{j}']) for j in (1, 2, 3))
        grouped[int(row['period'])].append(r)
    group_cases([r for g in grouped.values() for r in g])
    periods = sorted(grouped)
    if periods != list(range(1, len(periods)+1)):
        raise ValueError('Consecutive periods starting at one required')
    for group in grouped.values():
        group.sort(key=lambda r: float(r['charge']))
    # Same initial charge 0, squared movement cost, no terminal charge penalty.
    scores = {0.: (0., [])}
    for period in periods:
        new = {}
        for r in grouped[period]:
            q = float(r['charge'])
            best = max(scores, key=lambda prev: (scores[prev][0]-weight*(q-prev)**2, -prev))
            new[q] = (scores[best][0]+r['welfare']-weight*(q-best)**2, scores[best][1]+[r])
        scores = new
    sm = scores[max(scores, key=lambda q: (scores[q][0], -q))][1]
    nc = [grouped[t][0] for t in periods]
    sd = [max(grouped[t], key=lambda r: (r['welfare'], -float(r['charge']))) for t in periods]
    base = sum(r['welfare'] for r in nc)
    result = dict(periods=len(periods), weight=weight, criterion='TS', initial_charge=0., terminal_penalty=False)
    for policy, selected in (('NC', nc), ('SD', sd), ('SM', sm)):
        charges = [float(r['charge']) for r in selected]
        changes = np.diff([0.]+charges)
        total = sum(r['welfare'] for r in selected)
        result[policy] = dict(charges=charges, total_welfare=total, gain=total-base,
                              total_absolute_movement=float(sum(abs(changes))),
                              adjusted_objective=total-weight*float(changes @ changes))
    result['gain_retention'] = result['SM']['gain']/result['SD']['gain'] if result['SD']['gain'] > TOL else None
    result['movement_reduction'] = (1-result['SM']['total_absolute_movement']/result['SD']['total_absolute_movement']
                                    if result['SD']['total_absolute_movement'] > TOL else None)
    return result


def waiting_menu(root, cases):
    profiles = defaultdict(dict)
    with (root/WAITING).open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            game = json.loads(line)
            if game['mode'] != 'native':
                continue
            profile = tuple(game['profile'])
            q = float(game['game'].rsplit(':q', 1)[1])
            if q in profiles[profile] or not game.get('selected'):
                raise ValueError('Duplicate or unverified native waiting game')
            profiles[profile][q] = (game, line_number)
    if len(profiles) != 18 or any(sorted(g) != list(range(0, 22, 2)) for g in profiles.values()):
        raise ValueError('Complete frozen native 18x11 waiting games required')
    rows = []
    diagnostics = []
    for case in cases.values():
        key = case.demand_scale, case.capacity_scale, case.reliability_power
        if key not in profiles:
            continue
        for q, (game, line_number) in sorted(profiles[key].items()):
            saved = game['waiting_outcome']
            prices = game['selected']['prices']
            y = saved['demand']
            cost = np.array(case.base_cost)+[0., case.cost_gap, .5*case.cost_gap]
            us = saved['cs']-float(cost @ y)-case.congestion_cost*resource_penalty(saved['rho'])
            r = dict(case=case.name, charge=q, welfare=us, utilization=saved['rho'],
                     wait_minutes=saved['wait_minutes'], source_line=line_number,
                     source_file=WAITING, consumer_surplus=saved['cs'])
            r.update({f'price_{j}': prices[j-1] for j in (1, 2, 3)})
            r.update({f'demand_{j}': y[j-1] for j in (1, 2, 3)})
            # Keep the historical hard-capacity UT input convention for waiting.
            # This is a fixed-fare soft counterfactual, never a new price game.
            e = ReviewEngine(case, 'soft')
            r['unrestricted_utilization'] = e.outcome(prices).rho
            rows.append(r)
            native = WaitingEngine(case)
            o = native.outcome(prices)
            diagnostics.append(dict(case=case.name, charge=q,
                                    demand_error=float(max(abs(o.demand-y))),
                                    cs_error=o.cs-saved['cs'], wait_error=o.wait_minutes-saved['wait_minutes']))
    return rows, diagnostics


def source_hashes(root, paths):
    return {str(Path(p).as_posix()): hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sorted(set(paths))}


def source_closure(root):
    pending = ['experiments/aam_pricing/review3_ground.py']
    seen = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        tree = ast.parse((root/path).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                target = f'experiments/aam_pricing/{node.module.replace(".", "/")}.py'
                if (root/target).is_file():
                    pending.append(target)
    return sorted(seen)


def persist(path, payload):
    """A second run verifies identical bytes; never overwrite different evidence."""
    data = payload.encode('utf-8')
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f'Frozen output differs; refusing overwrite: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def export_csv(path, rows):
    import csv
    import io
    handle = io.StringIO(newline='')
    fields = list(dict.fromkeys(k for r in rows for k in r))
    writer = csv.DictWriter(handle, fields)
    writer.writeheader()
    writer.writerows(rows)
    persist(path, handle.getvalue())


def run(root, output_dir=None):
    root = Path(root)
    old_policies = {
        'soft': 'results/trb_review_benchmark_correction_v1/baseline_TS/policy_rows.csv',
        'hard': 'results/trb_review_hard_capacity_repair_v1/hardcap_TS/policy_rows.csv',
    }
    inputs = [SOFT, HARD, PARAMETERS, WAITING, CORRIDOR, SEQUENCE, MANUSCRIPT,
              'results/trb_review90_waiting_v1/waiting_policy_pairs.csv', *old_policies.values()]
    hashes = source_hashes(root, inputs+source_closure(root)+['scripts/run_review3_ground.py', 'tests/test_review3_ground.py'])
    params = load_parameters(root)
    cases = params['cases']
    if len(cases) != 450:
        raise ValueError('Frozen full parameter grid must have 450 cases')
    waiting, rechecks = waiting_menu(root, cases)
    if len(waiting) != 990 or max(max(abs(r[k]) for k in ('demand_error', 'cs_error', 'wait_error')) for r in rechecks) > 1.e-7:
        raise ValueError('Saved native waiting replay mismatch')
    domains = {'soft': (read_csv(root/SOFT), 'utilization'),
               'hard': (read_csv(root/HARD), 'unrestricted_utilization'),
               'waiting': (waiting, 'unrestricted_utilization')}
    if any(float(r['repair_verified']) != 1 for r in domains['hard'][0]):
        raise ValueError('Only corrected verified hard rows allowed')
    summaries, policies, tuning, menus, tails, groups, group_summary = [], [], [], [], [], [], []
    replay = []
    for domain, (rows, signal) in domains.items():
        expected = 90 if domain == 'waiting' else 450
        results = {k: optimize_menu(rows, cases, k, signal) for k in KAPPAS}
        zero = results[0.]['policies']
        if len(zero['SD']) != expected:
            raise ValueError('Incomplete frozen state denominator')
        if domain in old_policies:
            old = read_csv(root/old_policies[domain])
            for policy in ('SD', 'UT', 'RM'):
                reference = {r['case']: r for r in old if r['policy'] == policy}
                if reference.keys() != zero[policy].keys():
                    raise ValueError('Historical TS denominator mismatch')
                charge_errors = sum(float(reference[s]['charge']) != r['charge'] for s, r in zero[policy].items())
                gain_error = max(abs(float(reference[s]['delta_w_vs_nc'])-r['gain']) for s, r in zero[policy].items())
                replay.append(dict(domain=domain, policy=policy, charge_errors=charge_errors, max_gain_error=gain_error))
                if charge_errors or gain_error > 1.e-7:
                    raise ValueError('kappa=0 historical policy replay mismatch')
        else:
            reference = {r['case']: r for r in read_csv(root/'results/trb_review90_waiting_v1/waiting_policy_pairs.csv')
                         if r['mode'] == 'native' and r['objective'] == 'TS'}
            err = max(abs(float(reference[s]['objective_gain'])-r['gain']) for s, r in zero['SD'].items())
            changed = sum(float(reference[s]['charge_SD']) != r['charge'] for s, r in zero['SD'].items())
            replay.append(dict(domain=domain, policy='SD', charge_errors=changed, max_gain_error=err))
            if changed or err > 1.e-7:
                raise ValueError('Native TS SD replay mismatch')
        original_us = {s: max(g, key=lambda r: (r['US'], -r['charge'])) for s, g in results[0.]['scored'].items()}
        for name, group in results[0.]['scored'].items():
            for r in group:
                menus.append(dict(domain=domain, case=name, charge=r['charge'], US=r['US'], TS=r['TS'],
                                  ground_demand=r['ground_demand'], ground_price=r['ground_price'],
                                  W_0=r['W'], W_half=r['W']+.5*r['ground_price']*r['ground_demand'],
                                  W_full=r['W']+r['ground_price']*r['ground_demand']))
        for kappa, result in results.items():
            sd = result['policies']['SD']
            tuning.extend(dict(domain=domain, kappa=kappa, signal=signal, **r) for r in result['ut_candidates'])
            for policy, choices in result['policies'].items():
                info = policy_summary(sd, choices, zero[policy])
                summaries.append(dict(domain=domain, kappa=kappa, policy=policy,
                                      old_status_reference='same policy at kappa=0 TS', signal=signal,
                                      ut_threshold=result['best_ut']['threshold'], ut_slope=result['best_ut']['slope'],
                                      old_US_SD_positive_overlap=sum(original_us[s]['charge'] > TOL and choices[s]['charge'] > TOL for s in choices),
                                      old_US_SD_positive_count=sum(r['charge'] > TOL for r in original_us.values()),
                                      **info))
                for name, chosen in choices.items():
                    policies.append(dict(domain=domain, kappa=kappa, policy=policy, case=name,
                                         charge=chosen['charge'], W=chosen['W'], gain=chosen['gain'],
                                         regret=sd[name]['W']-chosen['W'], TS=chosen['TS'],
                                         ground_demand=chosen['ground_demand'], ground_net_receipt=chosen['ground_net_receipt'],
                                         ground_cost_per_trip=chosen['ground_cost_per_trip'],
                                         **{f'price_{j}': chosen[f'price_{j}'] for j in (1, 2, 3)}))
            info = policy_summary(sd, result['policies']['UT'], zero['UT'])
            for rank, name in enumerate(info['tail_states'], 1):
                ut, best = result['policies']['UT'][name], sd[name]
                tails.append(dict(domain=domain, kappa=kappa, rank=rank, case=name,
                                  sd_charge=best['charge'], ut_charge=ut['charge'],
                                  sd_gain=best['gain'], ut_gain=ut['gain'], regret=best['W']-ut['W']))
            # Group distribution refers to TS-SD, i.e. kappa=0, not another objective.
            if kappa == 0.:
                for name in info['tail_states']:
                    engine = WaitingEngine(cases[name]) if domain == 'waiting' else ReviewEngine(cases[name], domain)
                    ut, best, nc = result['policies']['UT'][name], sd[name], result['policies']['NC'][name]
                    for contrast, before, after in (('SD_minus_UT', ut, best), ('UT_minus_NC', nc, ut), ('SD_minus_NC', nc, best)):
                        groups.extend(dict(domain=domain, case=name, contrast=contrast,
                                           sd_charge=best['charge'], ut_charge=ut['charge'],
                                           ts_regret=best['W']-ut['W'], **r)
                                      for r in group_changes(engine, before, after))
        print(f'{domain}: {expected} states, 3 accounting criteria, frozen menus only', flush=True)
    for domain in domains:
        for contrast in ('SD_minus_UT', 'UT_minus_NC', 'SD_minus_NC'):
            for g in (1, 2, 3):
                subset = [r for r in groups if r['domain'] == domain and r['contrast'] == contrast and r['group'] == g]
                group_summary.append(dict(domain=domain, contrast=contrast, group=g, states=len(subset),
                                          **{key: float(np.mean([r[key] for r in subset])) for key in
                                             ('service_probability_change', 'served_change', 'served_composition_change',
                                              'cs_change', 'cs_per_potential_person', 'wait_change')}))
    if max(abs(r['identity_error']) for r in groups) > 1.e-7:
        raise ValueError('Group surplus identity failed')
    corridor = corridor_analysis(read_csv(root/CORRIDOR), params['anchor'])
    if corridor['charge_grid'] != list(map(float, range(0, 44, 4))) or abs(corridor['TS']['component_identity_error']) > 1.e-7:
        raise ValueError('Corridor menu/component mismatch')
    smoothing = smooth_sequence(read_csv(root/SEQUENCE))
    if source_hashes(root, hashes) != hashes:
        raise RuntimeError('Frozen inputs or implementation changed during analysis')
    out = root/OUTPUT if output_dir is None else Path(output_dir)
    for filename, data in [('policy_summary.csv', [{k: v for k, v in r.items() if k != 'tail_states'} for r in summaries]),
                           ('policy_states.csv', policies), ('ut_tuning.csv', tuning), ('accounting_menu.csv', menus),
                           ('ut_worst5_states.csv', tails), ('ts_worst5_group_changes.csv', groups),
                           ('ts_worst5_group_summary.csv', group_summary), ('waiting_rechecks.csv', rechecks),
                           ('historical_policy_replay.csv', replay)]:
        export_csv(out/filename, data)
    summary = dict(schema_version=1, no_new_nash=True,
                   assumption='p0-c0=kappa*p0; ground price, outside utility, demand response and fares fixed; no ground supply/congestion game',
                   tail_definition='largest ceil(0.05*n) absolute SD-UT welfare gaps; ties by ascending case name; NumPy linear quantiles',
                   RM_definition='maximize regulatory charge revenue over the same menu; accounting changes evaluation, not the RM criterion',
                   waiting_UT_signal='counterfactual unrationed soft utilization at saved NC waiting fares; same convention as hard-capacity UT',
                   states_are_design_points_not_probabilities=True,
                   behavioral_families=dict(soft=90, hard=90, waiting=18),
                   policies=summaries, corridor=corridor, smoothing=smoothing,
                   group_summary=group_summary, historical_replay=replay,
                   maximum_group_identity_error=max(abs(r['identity_error']) for r in groups),
                   maximum_waiting_replay_error=max(max(abs(r[k]) for k in ('demand_error', 'cs_error', 'wait_error')) for r in rechecks))
    persist(out/'summary.json', json.dumps(summary, indent=2, allow_nan=False)+'\n')
    outputs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file() and p.name != 'manifest.json'}
    manifest = dict(inputs_sha256=hashes, outputs_sha256=outputs,
                    runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                    command='py -3.13 -B scripts/run_review3_ground.py --output-dir PATH',
                    repeat_behavior='recompute and compare exact output bytes; refuse differing overwrite',
                    manuscript_sha256=hashes[MANUSCRIPT], frozen_inputs_unchanged=True)
    persist(out/'manifest.json', json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    return summary
