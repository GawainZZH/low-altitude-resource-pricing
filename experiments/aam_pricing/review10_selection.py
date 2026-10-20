"""Arithmetic over frozen candidates, not a new equilibrium or BR search."""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil


OUTPUT = 'results/trb_review10_selection_v1'
FIFTH = 'results/trb_review5_best_response_v1/full_attempt_01'
SIXTH = 'results/trb_review6_branches_v1'
SOURCES = {
    'inventory': FIFTH+'/inventory.json',
    'audits': FIFTH+'/games.jsonl',
    'fifth_design': FIFTH+'/design.json',
    'fifth_summary': FIFTH+'/summary.json',
    'parameters': 'results/trb_review_parameters_v1/full_parameters.json',
    'waiting': 'results/trb_review90_waiting_v1/waiting_games.jsonl',
    'initial_waiting': 'results/trb_review90_waiting_v1/initial_attempt/waiting_games.jsonl',
    'waiting_design': 'results/trb_review90_waiting_v1/design.json',
    'hard': 'results/trb_review_hard_capacity_repair_v1/full_candidates.jsonl',
    'hard_design': 'results/trb_review_hard_capacity_repair_v1/repair_manifest.json',
    'branch_points': SIXTH+'/points.jsonl',
    'branch_attempts': SIXTH+'/attempts.jsonl',
    'branch_audits': SIXTH+'/diagnostics.jsonl',
    'branch_extra_audits': SIXTH+'/candidate_audits.jsonl',
    'branch_design': SIXTH+'/design.json',
    'branch_summary': SIXTH+'/summary.json',
}
OWNED = ('experiments/aam_pricing/review10_selection.py',
         'scripts/run_review10_selection.py', 'tests/test_review10_selection.py')
ES = (90., 1200., 2800., 4200., 6500.)
KAPPAS = (0., .5, 1.)
GAIN_TOL = 1e-5
POLICY_TOL = 1e-7
CONVENTIONS = {
    'hard': 'Nearest verified discovered candidate to saved producer old_prices; lexicographic fare tie break.',
    'native': 'Saved q0 source anchor, then ascending-charge continuation; nearest verified discovered candidate; saved final-polish exception retained.',
    'equal_vot': 'Same waiting producer; equal-VOT waiting beta, not native beta.',
    'equal_beta': 'Same waiting producer; common waiting beta 0.038, not native beta.',
    'refined': 'Exact old response where present, otherwise first accepted forward continuation; fine starts at nearest lower coarse response; never welfare selection.',
}


def design():
    return dict(version=1, alternatives=408, affected_menus=36, charges=list(range(0, 22, 2)),
        externalities=list(ES), kappas=list(KAPPAS), gain_tolerance=GAIN_TOL,
        deduplication_fare_tolerance=1e-6, audit_match_tolerance=1e-7,
        policy_tolerance=POLICY_TOL, numerical_scope='A+B_numerical',
        hypothetical_scope='A+B+D_hypothetical', complete_Nash_set=False,
        new_games=0, new_roots=0, new_BR_audits=0,
        qualification='A original accepted selected plus saved full search; B full-search epsilon pass with original FOC rejection; C saved profitable deviation; D no full search and original FOC failure.',
        selection_conventions=CONVENTIONS)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    with path.open(encoding='utf-8-sig') as f:
        return [json.loads(line) for line in f if line.strip()] if path.suffix == '.jsonl' else json.load(f)


def dump(path, data):
    Path(path).write_text(json.dumps(data, allow_nan=False, ensure_ascii=True, separators=(',', ':'))+'\n', encoding='utf-8')


def load_inputs(root):
    return {name:read(Path(root)/path) for name, path in SOURCES.items()}


def distance(a, b):
    return max(abs(x-y) for x, y in zip(a, b))


def close(a, b, tolerance=1e-7):
    if not math.isfinite(a) or not math.isfinite(b) or abs(a-b) > tolerance:
        raise ValueError(f'Inconsistent saved arithmetic: {a} != {b}')


def solve3(matrix, rhs):
    a = [list(row)+[value] for row, value in zip(matrix, rhs)]
    scale = max(abs(v) for row in matrix for v in row)
    for j in range(3):
        pivot = max(range(j, 3), key=lambda i:abs(a[i][j]))
        if abs(a[pivot][j]) <= scale*1e-12:
            raise ValueError('Singular saved-demand recovery matrix')
        a[j], a[pivot] = a[pivot], a[j]
        div = a[j][j]
        a[j] = [v/div for v in a[j]]
        for i in range(3):
            if i != j:
                div = a[i][j]
                a[i] = [x-div*y for x, y in zip(a[i], a[j])]
    return [row[-1] for row in a]


def restore_state(e, kind, power, prices, charge, baselines):
    """Invert saved profits and, at capacity, a 3x3 mixture; no scalar root."""
    try:
        margin = [p-c-charge*a for p, c, a in zip(prices, e['c'], e['a'])]
        if min(abs(v) for v in margin) < 1e-8:
            raise ValueError('Zero or ill-conditioned saved profit margin')
        y = [v/m for v, m in zip(baselines, margin)]
        rho = sum(a*v for a, v in zip(e['a'], y))/e['K']
        if not 0 <= rho <= 1+1e-8 or min(y) < 0:
            raise ValueError('Nonphysical recovered demand/resource use')
        at_capacity = abs(rho-1) <= 1e-8
        r = 1. if at_capacity else rho
        h = .55*r**power+1.10*max(r-.88, 0.)**2
        outside = [(a-l*e['outside_price']-b*e['outside_time'])/e['mu']
                   for a, l, b in zip(e['alpha_outside'], e['lambda'], e['beta'])]
        air = [[(e['alpha'][g][m]-e['lambda'][g]*prices[m]-e['beta'][g]*e['service_time'][m]
                 -e['gamma'][g]*(e['ell'][m]+e['chi'][m]*h))/e['mu']
                for m in range(3)] for g in range(3)]
        A = [[math.exp(v-outside[g]) for v in row] for g, row in enumerate(air)]
        beta = list(e['beta'])
        if kind == 'equal_beta':
            beta = [.038]*3
        elif kind == 'equal_vot':
            ratio = sum(d*b for d, b in zip(e['D'], beta))/sum(d*l for d, l in zip(e['D'], e['lambda']))
            beta = [l*ratio for l in e['lambda']]
        wait, logphi, common_error, condition = 0., 0., 0., None
        if at_capacity:
            matrix = [[e['D'][g]*A[g][m] for g in range(3)] for m in range(3)]
            z = solve3(matrix, y)
            shares0 = [1-sum(row)*v for row, v in zip(A, z)]
            if min(z+shares0) <= 0:
                raise ValueError('Nonphysical recovered group probabilities')
            logt = [math.log(v/s) for v, s in zip(z, shares0)]
            if kind == 'hard':
                logphi = sum(logt)/3
                common_error = max(logt)-min(logt)
                if logphi > 1e-7:
                    raise ValueError('Positive availability log multiplier')
            else:
                waits = [-e['mu']*v/b for v, b in zip(logt, beta)]
                wait = sum(waits)/3
                common_error = max(waits)-min(waits)
                if wait < -1e-7:
                    raise ValueError('Negative recovered waiting')
            inverse_columns = [solve3(matrix, [float(i == j) for i in range(3)]) for j in range(3)]
            condition = max(sum(abs(v) for v in row) for row in matrix)*max(
                sum(abs(inverse_columns[j][i]) for j in range(3)) for i in range(3))
            if condition > 1e10 or common_error > 1e-7:
                raise ValueError('Unresolved common dual or recovery conditioning')
        prob, ground_prob, denoms = [], [], []
        for g in range(3):
            dual = logphi if kind == 'hard' else -beta[g]*wait/e['mu']
            row = [a*math.exp(dual) for a in A[g]]
            denominator = 1+sum(row)
            prob.append([v/denominator for v in row]); ground_prob.append(1/denominator)
            denoms.append(outside[g]+math.log(denominator))
        rebuilt = [sum(e['D'][g]*prob[g][m] for g in range(3)) for m in range(3)]
        error = distance(y, rebuilt)
        closure = abs(sum(a*v for a, v in zip(e['a'], rebuilt))/e['K']-r)
        if error > 1e-7 or closure > 1e-8:
            raise ValueError('Recovered state fails direct utility/demand closure')
        cs = sum(d*e['mu']/l*v for d, l, v in zip(e['D'], e['lambda'], denoms))
        ground = sum(d*p for d, p in zip(e['D'], ground_prob))
        revenue = sum(p*v for p, v in zip(prices, rebuilt))
        cost = sum(c*v for c, v in zip(e['c'], rebuilt))
        return dict(state_status='valid', state=dict(demand=rebuilt, rho=r, recovered_rho=rho,
            prob=prob, ground_prob=ground_prob, wait_minutes=wait, logphi=logphi,
            mechanism='availability' if kind == 'hard' else 'minutes_waiting', cs=cs,
            ground=ground, ground_payment=e['outside_price']*ground, revenue=revenue, cost=cost,
            base_TS=cs+revenue-cost, base_US=cs-cost,
            damage=.15*r*r+max(r-.90, 0.)**3, demand_error=error,
            closure_error=closure, common_dual_error=common_error, condition_inf=condition))
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        return dict(state_status='unresolved', error=str(exc), state=None)


def score(state, E, kappa):
    return state['base_TS']-E*state['damage']+kappa*state['ground_payment']


def policy_metrics(scores, selected):
    qs = sorted(scores)
    if not qs or 0. not in scores or any(not scores[q] for q in qs) or set(qs) != set(selected):
        raise ValueError('Incomplete candidate policy menu')
    lo = {q:min(scores[q]) for q in qs}; hi = {q:max(scores[q]) for q in qs}
    choose = lambda values:max(qs, key=lambda q:(values[q], -q))
    q = choose(selected)
    lower, upper = choose(lo), choose(hi)
    rivals = [v for v in qs if v != q]
    competing = max((hi[v] for v in rivals), default=lo[q])
    activation_lower = max(lo[v] for v in qs if v > 0)-hi[0.]
    activation_upper = max(hi[v] for v in qs if v > 0)-lo[0.]
    activation = ('robust_positive_within_saved_scope' if activation_lower > POLICY_TOL else
                  'no_observed_positive_gain' if activation_upper <= POLICY_TOL else 'ambiguous_within_saved_scope')
    return dict(selected_charge=q, selected_TS=selected[q], lower_charge=lower, upper_charge=upper,
        selected_ties=[v for v in qs if selected[v] >= selected[q]-POLICY_TOL],
        lower_ties=[v for v in qs if lo[v] >= lo[lower]-POLICY_TOL],
        upper_ties=[v for v in qs if hi[v] >= hi[upper]-POLICY_TOL],
        possible_optimal_charges=[v for v in qs if hi[v] >= max((lo[w] for w in qs if w != v), default=lo[v])-POLICY_TOL],
        selected_charge_robust=lo[q] >= competing-POLICY_TOL,
        worst_observed_regret=max(0., competing-lo[q]), activation_lower=activation_lower,
        activation_upper=activation_upper, activation=activation,
        selected_upper_endpoint=q == qs[-1], lower_upper_endpoint=lower == qs[-1],
        upper_upper_endpoint=upper == qs[-1], lower_TS=lo[lower], upper_TS=hi[upper])


def finite_gains(candidate):
    br = candidate['br']
    if len(br) != 3 or [r['platform'] for r in br] != [0, 1, 2]:
        raise ValueError('Missing platform finite deviation coverage')
    gains = []
    for row in br:
        gain = max(0., max(v['profit'] for v in row['trace'])-row['baseline'])
        close(gain, row['gain'], 1e-8)
        gains.append(gain)
    close(max(gains), candidate['max_gain'], 1e-8)
    return max(gains)


def profile(game):
    c = game['case']
    return tuple(c[k] for k in ('demand_scale', 'capacity_scale', 'reliability_power'))


def menu_id(game):
    return game['kind']+':'+','.join(f'{x:g}' for x in profile(game))


def saved_candidates(inventory):
    rows = []
    for g in inventory:
        seen = [g['prices']]
        for index, candidate in enumerate(g['branches']):
            p = candidate.get('prices')
            if p is None or not all(math.isfinite(x) for x in p):
                continue
            if any(distance(p, s) <= 1e-6 for s in seen):
                continue
            seen.append(p)
            rows.append((g, index, candidate))
    return rows


def check_audit(audit, candidate, charge):
    if audit['charge'] != charge or distance(audit['prices'], candidate['prices']) > 1e-12:
        raise ValueError('Wrong full-search candidate linkage')
    if len(audit['coordinates']) != 3:
        raise ValueError('Missing full-search coordinate')
    gains = []
    for m, coordinate in enumerate(audit['coordinates']):
        if coordinate['platform'] != m+1:
            raise ValueError('Wrong full-search platform linkage')
        close(coordinate['native_baseline'], candidate['br'][m]['baseline'], 1e-6)
        if [p['points'] for p in coordinate['passes']] != [129, 257]:
            raise ValueError('Missing full-search refinement')
        if any(p['interval'] != [5., 44.] for p in coordinate['passes']):
            raise ValueError('Wrong full fare interval')
        gains.append(max(0., coordinate['native_best']-coordinate['native_baseline']))
    close(max(gains), audit['epsilon_absolute'], 1e-8)
    if (audit['status'] == 'epsilon_search_pass') != (max(gains) <= GAIN_TOL and audit['refinement_stable']):
        raise ValueError('Full-search status disagrees with saved gains/refinement')


def make_envelopes(rows, scope, E_values):
    groups = defaultdict(list)
    for row in rows:
        groups[row['menu']].append(row)
    policies, envelopes = [], []
    for name, group in sorted(groups.items()):
        byq = defaultdict(list)
        for row in group:
            byq[row['charge']].append(row)
        for E in E_values:
            for k in KAPPAS:
                scores, selected = {}, {}
                complete = True
                for q, candidates in sorted(byq.items()):
                    valid = [r for r in candidates if r['state_status'] == 'valid']
                    complete &= len(valid) == len(candidates)
                    scores[q] = [score(r['state'], E, k) for r in valid]
                    chosen = [r for r in valid if r['tier'] == 'A']
                    if len(chosen) != 1:
                        complete = False
                    else:
                        selected[q] = score(chosen[0]['state'], E, k)
                    counts = Counter(r['tier'] for r in candidates)
                    envelopes.append(dict(menu=name, scope=scope, externality=E, kappa=k, charge=q,
                        candidates=len(candidates), recovered=len(valid), unresolved=len(candidates)-len(valid),
                        A_candidates=counts['A'], B_candidates=counts['B'], C_candidates=counts['C'],
                        D_candidates=counts['D'], R_candidates=counts['R'], lower_TS=min(scores[q], default=None),
                        upper_TS=max(scores[q], default=None)))
                header = dict(menu=name, scope=scope, externality=E, kappa=k,
                              complete_saved_scope=complete, complete_Nash_set=False)
                policies.append(dict(header, **policy_metrics(scores, selected)) if complete else
                    dict(header, activation='unresolved_saved_states', selected_charge=None))
    return policies, envelopes


def analyze(b):
    inventory, audits = b['inventory'], b['audits']
    if len(inventory) != 2255 or len({g['id'] for g in inventory}) != 2255 or len(audits) != 2500:
        raise ValueError('Frozen game/audit coverage differs')
    if sum(len(g['branches']) for g in inventory) != 1787:
        raise ValueError('Frozen historical attempt coverage differs')
    candidates = saved_candidates(inventory)
    if len(candidates) != 408:
        raise ValueError('Frozen 408-candidate coverage differs')
    menu_audits = {a['id']:a for a in audits if a['role'] == 'menu'}
    branch_audits = {(a['id'], a['branch']['index']):a for a in audits if a['role'] == 'branch'}
    if len(menu_audits) != 2255 or len(branch_audits) != 245 or set(menu_audits) != {g['id'] for g in inventory}:
        raise ValueError('Missing or duplicated original audit')
    keys = {(g['id'], index) for g, index, _ in candidates}
    if not set(branch_audits) <= keys:
        raise ValueError('Unlinked full-search alternative')
    affected = {menu_id(g) for g, _, _ in candidates}
    games = [g for g in inventory if menu_id(g) in affected]
    if len(affected) != 36 or len(games) != 396:
        raise ValueError('Affected menu count differs')
    for menu in affected:
        if sorted(g['charge'] for g in games if menu_id(g) == menu) != list(range(0, 22, 2)):
            raise ValueError('Incomplete original charge menu')
    params = {r['dataclass']['name']:r for r in b['parameters']['full_state_grid']}
    wait = {r['game']:r for r in b['waiting']}
    hard = {}
    for row in b['hard']:
        if row['game'].startswith('coarse:'):
            hard[row['case'], row['charge']] = row
    selected, alternatives = [], []

    def row_for(game, index, candidate, tier, audit, producer):
        c = game['case']; stored = params[c['name']]
        if stored['dataclass'] != c:
            raise ValueError('Case/effective parameter mismatch')
        finite_gain = finite_gains(candidate)
        data = dict(menu=menu_id(game), game=game['id'], kind=game['kind'], profile=list(profile(game)),
            charge=game['charge'], index=index, tier=tier, prices=candidate['prices'],
            original_verified=candidate.get('verified', False),
            original_first_order_ok=candidate.get('first_order_ok', False), original_kkt=candidate.get('kkt'),
            candidate_source=candidate.get('candidate_source'), saved_finite_gain=finite_gain,
            full_interval_status=audit['status'] if audit else None,
            full_interval_gain=audit['epsilon_absolute'] if audit else None,
            producer_anchor=producer['old_prices'], selection_convention=CONVENTIONS[game['kind']])
        if audit:
            check_audit(audit, candidate, game['charge'])
        data.update(restore_state(stored['effective'], game['kind'], c['reliability_power'],
                                  candidate['prices'], game['charge'], [r['baseline'] for r in candidate['br']]))
        if data['state_status'] == 'valid' and audit:
            close(data['state']['rho'], audit['rho'], 1e-8)
        if data['state_status'] == 'valid' and index is None:
            saved = producer['outcome'] if game['kind'] == 'hard' else producer['waiting_outcome']
            state = data['state']
            close(state['cs'], saved['cs'], 1e-6)
            close(distance(state['demand'], saved['demand']), 0., 1e-7)
            close(state['rho'], saved['rho'], 1e-8)
            if game['kind'] == 'hard':
                close(state['logphi'], saved['logphi'], 1e-7)
            else:
                close(state['wait_minutes'], saved['wait_minutes'], 1e-7)
        return data

    producers = {}
    for game in games:
        source = hard.get((game['case']['name'], game['charge'])) if game['kind'] == 'hard' else wait.get(game['id'])
        if not source or not source.get('selected') or not source['selected']['verified']:
            raise ValueError('Missing original accepted selected producer')
        if distance(source['selected']['prices'], game['prices']) > 1e-12:
            raise ValueError('Selected inventory/producer mismatch')
        if menu_audits[game['id']]['status'] != 'epsilon_search_pass':
            raise ValueError('Tier A lacks saved full-search pass')
        producers[game['id']] = source
        selected.append(row_for(game, None, source['selected'], 'A', menu_audits[game['id']], source))
    for game, index, candidate in candidates:
        gain = finite_gains(candidate)
        audit = branch_audits.get((game['id'], index))
        if audit:
            tier = 'B' if audit['status'] == 'epsilon_search_pass' else 'C'
        else:
            tier = 'C' if gain > GAIN_TOL else 'D'
        if tier == 'C' and max(gain, audit['epsilon_absolute'] if audit else 0.) <= GAIN_TOL:
            raise ValueError('Tier C lacks profitable saved witness')
        alternatives.append(row_for(game, index, candidate, tier, audit, producers[game['id']]))
    tiers = dict(sorted(Counter(r['tier'] for r in alternatives).items()))
    if tiers != dict(B=58, C=302, D=48):
        raise ValueError('Frozen candidate qualification counts differ')
    if any(r['original_verified'] or r['original_first_order_ok'] for r in alternatives):
        raise ValueError('Frozen alternative first-order rejection differs')
    all_rows = selected+alternatives
    policies, envelopes = [], []
    for label, allowed in [('A+B_numerical', {'A', 'B'}), ('A+B+D_hypothetical', {'A', 'B', 'D'})]:
        p, e = make_envelopes([r for r in all_rows if r['tier'] in allowed], label, ES)
        policies += p; envelopes += e
    refined, rejected, refined_policies, refined_envelopes = refined_rows(b)
    failed = [dict(corpus='initial_waiting', game=r['game'], record=r) for r in b['initial_waiting'] if not r.get('selected')]
    failed += [dict(corpus='refined_branch', game=r['role']+':q'+str(r['q']), record=r) for r in rejected]
    unresolved = [dict(corpus='original', game=r['game'], index=r['index'], tier=r['tier'], error=r['error'])
                  for r in all_rows if r['state_status'] != 'valid']
    unresolved += [dict(corpus='refined', game=r['menu'], index=r['index'], tier=r['tier'], error=r['error'])
                   for r in refined if r['state_status'] != 'valid']
    summary = dict(alternatives=len(alternatives), alternative_tiers=tiers, selected=len(selected),
        affected_menus=len(affected), original_menu_games=2255,
        unaffected_selected_only_games=2255-len(selected),
        original_first_order_rejected_alternatives=sum(not r['original_first_order_ok'] for r in alternatives),
        recovered_alternatives=sum(r['state_status'] == 'valid' for r in alternatives),
        unresolved_states=len(unresolved), state_failures=unresolved,
        preserved_initial_unselected_games=sum(not r.get('selected') for r in b['initial_waiting']),
        preserved_initial_attempts=sum(len(r.get('attempts', [])) for r in b['initial_waiting']),
        refined_selected=61, refined_accepted_records=len(refined),
        refined_exact_vectors=len({(r['menu'], r['charge'], tuple(r['prices'])) for r in refined}),
        refined_rejected_phases=len(rejected), menu_policy_appraisals=len(policies),
        refined_policy_appraisals=len(refined_policies), new_games=0, new_roots=0, new_BR_audits=0,
        complete_Nash_set=False, queue_unlinked_starts_not_in_this_corpus=51,
        max_demand_reconstruction_error=max((r['state']['demand_error'] for r in all_rows+refined if r['state_status'] == 'valid'), default=None),
        max_resource_closure_error=max((r['state']['closure_error'] for r in all_rows+refined if r['state_status'] == 'valid'), default=None))
    summary['policy_scope_totals'] = [dict(scope=label, appraisals=len(rows),
        robust_positive=sum(r.get('activation') == 'robust_positive_within_saved_scope' for r in rows),
        ambiguous_activation=sum(r.get('activation') == 'ambiguous_within_saved_scope' for r in rows),
        no_observed_positive=sum(r.get('activation') == 'no_observed_positive_gain' for r in rows),
        unresolved=sum(not r['complete_saved_scope'] for r in rows),
        selected_charge_not_robust=sum(r.get('selected_charge_robust') is False for r in rows),
        max_observed_regret=max((r.get('worst_observed_regret', 0.) for r in rows), default=None))
        for label in ('A+B_numerical', 'A+B+D_hypothetical') for rows in [[p for p in policies if p['scope'] == label]]]
    return dict(design=design(), summary=summary, selected=selected, alternatives=alternatives,
        policies=policies, envelopes=envelopes, refined_candidates=refined,
        refined_policies=refined_policies, refined_envelopes=refined_envelopes,
        failed_cases=unresolved, historical_failures=failed)


def refined_rows(b):
    points = b['branch_points']; attempts = b['branch_attempts']
    finished = [a for a in attempts if a['status'] != 'started']
    started = [a['attempt'] for a in attempts if a['status'] == 'started']
    if len(points) != 61 or len(finished) != 186 or sorted(started) != sorted(a['attempt'] for a in finished):
        raise ValueError('Incomplete frozen refined branch')
    source = [dict(r, tier='A', index=None, candidate=r['finite41']) for r in points]
    source += [dict(r, tier='R', index=r['attempt'], candidate=r) for r in finished if r.get('verified')]
    audits = b['branch_audits']+b['branch_extra_audits']
    if len(audits) != 83:
        raise ValueError('Missing refined full-search evidence')
    params = b['parameters']['full_state_grid']; specs = {r['role']:r for r in b['branch_design']['profiles']}
    rows = []
    for s in source:
        matched = [r['audit'] for r in audits if r['role'] == s['role'] and r['q'] == s['q']
                   and distance(r['audit']['prices'], s['prices']) <= 1e-7
                   and r['audit']['status'] == 'epsilon_search_pass']
        if not matched or not s['candidate']['verified']:
            raise ValueError('Refined accepted candidate lacks qualified evidence')
        case = next(r for r in params if [r['dataclass'][v] for v in ('demand_scale', 'capacity_scale', 'reliability_power')]
                    == specs[s['role']]['profile'] and r['dataclass']['congestion_cost'] == 2800.)
        record = dict(menu=s['role'], charge=s['q'], prices=s['prices'], tier=s['tier'], index=s['index'],
                      original_verified=True, original_first_order_ok=s['candidate']['first_order_ok'],
                      full_interval_scope='matched_within_1e-7', original_start=s.get('start', s.get('origin')),
                      selection_convention=CONVENTIONS['refined'])
        finite_gains(s['candidate'])
        record.update(restore_state(case['effective'], 'native', case['dataclass']['reliability_power'],
            s['prices'], s['q'], [r['baseline'] for r in s['candidate']['br']]))
        if record['state_status'] == 'valid':
            st = record['state']
            close(st['cs'], s['cs'], 1e-6); close(st['wait_minutes'], s['wait'], 1e-6)
            close(distance(st['demand'], s['demand']), 0., 1e-7)
            for k, ts in zip(KAPPAS, s['TS']):
                close(score(st, 2800., k), ts, 1e-6)
        rows.append(record)
    # Duplicate attempts remain in the ledger, not as independent candidate states.
    unique = {}
    for row in rows:
        key = row['menu'], row['charge'], tuple(row['prices'])
        if key not in unique or row['tier'] == 'A':
            unique[key] = row
    policies, envelopes = make_envelopes(list(unique.values()), 'refined_original_accepted', (2800.,))
    return rows, [r for r in finished if not r.get('verified')], policies, envelopes


def export_csv(path, rows):
    if not rows:
        Path(path).write_text('status\n', encoding='utf-8')
        return
    names = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, names); writer.writeheader()
        writer.writerows({k:json.dumps(v, allow_nan=False) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows)


def safe_copy_path(folder, relative):
    target = (folder/relative).resolve()
    if not target.is_relative_to(folder.resolve()):
        raise ValueError('Frozen input path escapes output directory')
    return target


def replay(folder):
    folder = Path(folder)
    raw = read(folder/'raw.json')
    if raw['design'] != design() or set(raw['inputs']) != set(SOURCES):
        raise ValueError('Frozen selection design/input coverage differs')
    inputs = {}
    for name, entry in raw['inputs'].items():
        path = safe_copy_path(folder, entry['copy'])
        if digest(path) != entry['sha256']:
            raise ValueError('Copied source hash differs: '+name)
        inputs[name] = read(path)
    result = analyze(inputs)
    if result != read(folder/'diagnostics.json'):
        raise ValueError('Saved selection arithmetic differs')
    manifest = read(folder/'manifest.json')
    for p, h in manifest['outputs_sha256'].items():
        if digest(safe_copy_path(folder, p)) != h:
            raise ValueError('Output hash differs: '+p)
    return result


def run(root, output=None):
    root = Path(root); output = root/OUTPUT if output is None else Path(output)
    if output.exists():
        raise FileExistsError('Immutable selection output already exists')
    inputs = load_inputs(root)
    fifth = inputs['fifth_design']
    if digest(root/SOURCES['inventory']) != fifth['inventory_sha256'] or digest(root/SOURCES['audits']) != inputs['fifth_summary']['games_sha256']:
        raise ValueError('Frozen fifth-review source hash differs')
    for p, h in inputs['branch_design']['source_hashes'].items():
        if p.startswith('results/') and digest(root/p) != h:
            raise ValueError('Frozen refined source hash differs')
    dependencies = {p for p in fifth['source_hashes'] if p.endswith('.py')}
    dependencies.update(p for p in inputs['branch_design']['source_hashes'] if p.endswith('.py'))
    dependencies.update(('experiments/aam_pricing/review6_branches.py', 'scripts/run_review6_branches.py',
                         'scripts/run_review90_waiting.py', 'scripts/run_review_hard_capacity_repair.py'))
    protected = set(SOURCES.values()) | dependencies
    details = [root/FIFTH/a['detail'] for a in inputs['audits']]
    protected.update(p.relative_to(root).as_posix() for p in details)
    hashes = {p:digest(root/p) for p in sorted(protected)}
    for audit, path in zip(inputs['audits'], details):
        if hashes[path.relative_to(root).as_posix()] != audit['detail_sha256']:
            raise ValueError('Frozen full-search detail hash differs')
    result = analyze(inputs)
    output.mkdir(parents=True, exist_ok=False)
    entries = {}
    for name, p in SOURCES.items():
        dest = output/'inputs'/name/Path(p).name
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(root/p, dest)
        entries[name] = dict(source=p, copy=dest.relative_to(output).as_posix(), sha256=hashes[p])
    for p in dependencies | set(OWNED):
        dest = output/'execution_code'/p
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(root/p, dest)
    for path in details:
        dest = output/'full_search_details'/path.name
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, dest)
    own_hashes = {p:digest(root/p) for p in OWNED}
    dump(output/'raw.json', dict(design=design(), inputs=entries, source_sha256=hashes,
        historical_execution_sha256=dict(fifth=fifth['source_hashes'], refined=inputs['branch_design']['source_hashes']),
        current_arithmetic_code_sha256=own_hashes, runtime_dependencies='Python standard library only',
        created_utc=datetime.now(timezone.utc).isoformat()))
    dump(output/'diagnostics.json', result); dump(output/'summary.json', result['summary'])
    for name in ('selected', 'alternatives', 'policies', 'envelopes', 'refined_candidates',
                 'refined_policies', 'refined_envelopes', 'failed_cases', 'historical_failures'):
        export_csv(output/(name+'.csv'), result[name])
    dump(output/'failed_cases.json', dict(unresolved=result['failed_cases'], historical=result['historical_failures']))
    if any(digest(root/p) != h for p, h in hashes.items()):
        raise ValueError('An original file changed during arithmetic extraction')
    dump(output/'manifest.json', dict(originals_unchanged=True, source_sha256=hashes,
        execution_code_sha256=own_hashes, new_games=0, new_roots=0, new_BR_audits=0,
        outputs_sha256={p.relative_to(output).as_posix():digest(p) for p in output.rglob('*') if p.is_file()}))
    return result
