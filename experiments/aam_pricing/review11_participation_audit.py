"""Closed arithmetic for report84 participation; no behavior or Nash solves.

Keep all 268 planned IDs. Rescue replaces rejected originals only. Unknown
states stay null, while the ex-ante fare-cap comparison remains available.
"""
from collections import defaultdict
import csv
import json
from pathlib import Path

from .review10_core import closed_arithmetic, digest, dump, export, verify_manifest

OUTPUT = 'results/trb_review11_participation_audit_v1'
HET = 'results/trb_review6_heterogeneity_v1'
CORE = 'results/trb_review10_core_v1'
REF = 'results/trb_review5_refinement_v1'
OWNED = ['experiments/aam_pricing/review11_participation_audit.py',
         'scripts/run_review11_participation_audit.py',
         'tests/test_review11_participation_audit.py']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def csv_rows(path):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def platform_rows(case, charge, prices=None, demand=None):
    if (prices is None) != (demand is None):
        raise ValueError('Prices and demand must both be saved or both unknown')
    costs = [v + offset for v, offset in zip(case['base_cost'], [0., case['cost_gap'], .5*case['cost_gap']])]
    rows = []
    for i, (cost, resource) in enumerate(zip(costs, case['platform_resource'])):
        expenditure = cost + resource*charge
        p, y = (prices[i], demand[i]) if prices is not None else (None, None)
        margin = p-expenditure if p is not None else None
        rows.append(dict(platform=i+1, cost=cost, resource=resource,
            unit_expenditure=expenditure, cap_nonpositive=expenditure >= case['price_upper'],
            cap_negative=expenditure > case['price_upper'], price=p, demand=y,
            margin=margin, profit=margin*y if margin is not None else None,
            upper_active=p >= case['price_upper']-1e-8 if p is not None else None,
            lower_active=p <= case['price_lower']+1e-8 if p is not None else None))
    return rows


def merge_attempts(original, rescue):
    selected = {g['game']: g for g in original}
    if len(selected) != len(original) or len({g['game'] for g in rescue}) != len(rescue):
        raise ValueError('Duplicate game or rescue alias')
    for g in rescue:
        if g['game'] not in selected or selected[g['game']]['selected']:
            raise ValueError('Rescue must reference an original not accepted')
        if g['selected']:
            selected[g['game']] = g
    return [selected[g['game']] for g in original]


def flags(rows):
    known = [r for r in rows if r['margin'] is not None]
    return dict(cap_nonpositive_platforms=sum(r['cap_nonpositive'] for r in rows),
        cap_negative_platforms=sum(r['cap_negative'] for r in rows),
        negative_margin_platforms=sum(r['margin'] < 0 for r in known),
        zero_margin_platforms=sum(r['margin'] == 0 for r in known),
        nonpositive_margin_platforms=sum(r['margin'] <= 0 for r in known),
        negative_profit_platforms=sum(r['profit'] < 0 for r in known),
        zero_profit_platforms=sum(r['profit'] == 0 for r in known),
        nonpositive_profit_platforms=sum(r['profit'] <= 0 for r in known),
        upper_active_platforms=sum(r['upper_active'] for r in known),
        lower_active_platforms=sum(r['lower_active'] for r in known),
        min_margin=min((r['margin'] for r in known), default=None),
        min_profit=min((r['profit'] for r in known), default=None))


def incidence(rows):
    keys = [k for k in rows[0] if k.endswith('_platforms')]
    result = {k: sum(r[k] for r in rows) for k in keys}
    result.update({k.removesuffix('_platforms')+'_games': sum(r[k] > 0 for r in rows) for k in keys})
    result['negative_margin_positive_wait_games'] = sum(r['negative_margin_platforms'] > 0 and
        r['wait_minutes'] > 1e-6 for r in rows if r['accepted'])
    result['negative_margin_strict_positive_wait_games'] = sum(r['negative_margin_platforms'] > 0 and
        r['strict_positive_wait'] for r in rows)
    return result


def analyze(root):
    root = Path(root)
    core = root/CORE
    verify_manifest(core)
    copied = core/'inputs'/HET
    hash_pairs = {name: digest(root/HET/name) == digest(copied/name)
                  for name in ('design.json', 'games.jsonl', 'rescue_games.jsonl')}
    if not all(hash_pairs.values()):
        raise ValueError('Current heterogeneity raw files differ from core copies')
    spec = read(copied/'design.json')
    original, rescue = jsonl(copied/'games.jsonl'), jsonl(copied/'rescue_games.jsonl')
    selected = merge_attempts(original, rescue)
    expected = {f"{m['id']}:q{q}" for m in spec['menus'] for q in m['charges']}
    if {g['game'] for g in selected} != expected or len(selected) != spec['games']:
        raise ValueError('Planned game coverage differs')
    menus_by_id = {m['id']: m for m in spec['menus']}
    originals_by_id = {g['game']: g for g in original}
    rescue_by_id = {g['game']: g for g in rescue}
    games, platforms, unresolved = [], [], []
    profit_error = 0.
    for g in selected:
        m = menus_by_id[g['menu']]
        case = m['case']
        strong = max(case['platform_resource'])/min(case['platform_resource']) >= 4.5-1e-12
        s = g.get('diagnostic', {}).get('state') if g['selected'] else None
        rows = platform_rows(case, g['charge'], s['prices'] if s else None, s['demand'] if s else None)
        for r, effective in zip(rows, m['effective']['cost']):
            if abs(r['cost']-effective) > 1e-12:
                raise ValueError('Actual operating cost differs from effective saved cost')
        if s:
            for r, br in zip(rows, g['selected']['br']):
                error = abs(r['profit']-br['baseline'])
                profit_error = max(profit_error, error)
                if error > 1e-7:
                    raise ValueError('Saved BR baseline profit differs from arithmetic')
        meta = dict(game=g['game'], menu=g['menu'], family=m['family'], charge=g['charge'],
                    strong=strong, accepted=bool(g['selected']), rescued=g['game'] in rescue_by_id and bool(g['selected']) and not originals_by_id[g['game']]['selected'])
        platforms.extend(dict(meta, **r) for r in rows)
        f = flags(rows)
        status = g.get('diagnostic', {}).get('status', 'unresolved')
        games.append(dict(meta, status=status, wait_minutes=s['wait_minutes'] if s else None,
            TS=s['TS'] if s else None, served=s['served'] if s else None,
            strict_positive_wait=bool(s and s['wait_minutes'] > 1e-6 and
                not f['upper_active_platforms'] and not f['lower_active_platforms'] and status == 'valid_IFT'), **f))
        if not s:
            old = originals_by_id[g['game']]
            retried = rescue_by_id.get(g['game'], {})
            row = dict(game=g['game'], menu=g['menu'], charge=g['charge'],
                demand_scale=case['demand_scale'], potential=sum(m['effective']['mass']),
                capacity_scale=case['capacity_scale'], capacity=m['effective']['capacity'],
                reliability_power=case['reliability_power'], mu=case['logit_scale'],
                cost=m['effective']['cost'], resource=case['platform_resource'],
                cap_nonpositive_platforms=f['cap_nonpositive_platforms'])
            for label, attempt in [('original', old), ('rescue', retried)]:
                candidates = attempt.get('audited_candidates', [])
                best = min(candidates, key=lambda c:c['max_gain']) if candidates else None
                row.update({label+'_candidates':len(candidates),
                    label+'_best_gain':best['max_gain'] if best else None,
                    label+'_best_gain_kkt':best['kkt'] if best else None,
                    label+'_best_gain_first_order_ok':best['first_order_ok'] if best else None,
                    label+'_best_gain_transition_gap':best['transition_gap'] if best else None})
            unresolved.append(row)
    by_menu = defaultdict(list)
    for g in games:
        by_menu[g['menu']].append(g)
    menus, intervals = [], []
    for m in spec['menus']:
        group = sorted(by_menu[m['id']], key=lambda g:g['charge'])
        good = [g for g in group if g['accepted']]
        best = max(good, key=lambda g:(g['TS'], -g['charge']))
        nc = next((g for g in good if g['charge'] == 0), None)
        for left, right in zip(group, group[1:]):
            if not (left['accepted'] and right['accepted']):
                continue
            intervals.append(dict(menu=m['id'], strong=group[0]['strong'],
                qL=left['charge'], qR=right['charge'], TS_change=right['TS']-left['TS'],
                both_positive_wait=left['wait_minutes'] > 1e-6 and right['wait_minutes'] > 1e-6,
                both_strict=left['strict_positive_wait'] and right['strict_positive_wait'],
                any_negative_margin=bool(left['negative_margin_platforms'] or right['negative_margin_platforms']),
                any_upper_active=bool(left['upper_active_platforms'] or right['upper_active_platforms'])))
        menus.append(dict(menu=m['id'], family=m['family'], strong=group[0]['strong'],
            primary=m['id'] in read(core/'design.json')['primary_paths'], complete=len(good) == len(group),
            planned=len(group), accepted=len(good), failed_charges=[g['charge'] for g in group if not g['accepted']],
            costs=m['effective']['cost'], resource=m['effective']['resource'], mu=m['effective']['mu'],
            profile=m['profile'], potential=sum(m['effective']['mass']),
            cap_nonpositive_q=[g['charge'] for g in group if g['cap_nonpositive_platforms']],
            negative_margin_q=[g['charge'] for g in good if g['negative_margin_platforms']],
            upper_active_q=[g['charge'] for g in good if g['upper_active_platforms']],
            strict_positive_wait_q=[g['charge'] for g in good if g['strict_positive_wait']],
            positive_wait_q=[g['charge'] for g in good if g['wait_minutes'] > 1e-6],
            optimum_charge=best['charge'], optimum_TS=best['TS'],
            optimum_TS_gain=best['TS']-nc['TS'] if nc else None,
            optimum_wait=best['wait_minutes'], optimum_min_margin=best['min_margin'],
            optimum_min_profit=best['min_profit'], optimum_negative_margin=best['negative_margin_platforms'] > 0,
            optimum_upper_active=best['upper_active_platforms'] > 0))
    by_game = {g['game']:g for g in games}
    paths = csv_rows(core/'paths.csv')
    if len(paths) != sum(g['strong'] and g['accepted'] for g in games):
        raise ValueError('Core paths coverage differs')
    for p in paths:
        g = by_game[p['game']]
        if abs(float(p['TS'])-g['TS']) > 1e-7 or (p['fare_upper_active'] == 'True') != bool(g['upper_active_platforms']):
            raise ValueError('Core paths state or fare flag differs')
    source = csv_rows(core/'inputs'/REF/'source.csv')
    cases = {c['dataclass']['name']:c['dataclass'] for c in read(core/'inputs'/REF/'parameters.json')['main']}
    native = {}
    native_rows = []
    for r in source:
        if r['cohort'] != 'main':
            continue
        case = cases[r['case']]
        ps = [float(r[f'price_{i}']) for i in (1, 2, 3)]
        ys = [float(r[f'demand_{i}']) for i in (1, 2, 3)]
        row = dict(case=r['case'], charge=float(r['charge']), accepted=r['accepted'] == 'True',
            wait_minutes=float(r['wait_minutes']), TS=float(r['TS']), strict_positive_wait=False,
            **flags(platform_rows(case, float(r['charge']), ps, ys)))
        native[(r['case'], float(r['charge']))] = row
        native_rows.append(row)
    policies = []
    for p in csv_rows(core/'policies.csv'):
        for label in ('NC', 'UT', 'SD'):
            r = native[(p['case'], float(p[label+'_charge']))]
            if abs(r['TS']-float(p[label+'_TS'])) > 1e-7:
                raise ValueError('Core policy TS differs from its raw native row')
            prices = json.loads(p[label+'_prices'])
            raw = next(x for x in source if x['cohort'] == 'main' and x['case'] == p['case'] and float(x['charge']) == r['charge'])
            if any(abs(prices[i]-float(raw[f'price_{i+1}'])) > 1e-8 for i in range(3)):
                raise ValueError('Core policy fare differs from raw')
            policies.append(dict(policy=label, **r))
    strong_games = [g for g in games if g['strong']]
    strong_menus = [m for m in menus if m['strong']]
    summary = dict(schema=1, planned_games=len(games), accepted_games=sum(g['accepted'] for g in games),
        original_accepted=sum(bool(g['selected']) for g in original), rescue_attempts=len(rescue),
        rescued_games=sum(bool(g['selected']) for g in rescue), planned_menus=len(menus),
        strong_menus=len(strong_menus), strong_complete=sum(m['complete'] for m in strong_menus),
        strong_planned_games=len(strong_games), strong_accepted_games=sum(g['accepted'] for g in strong_games),
        all=incidence(games), strong=incidence(strong_games),
        native_candidates=len(native_rows), native=incidence(native_rows),
        native_policies=len(policies), native_policy=incidence(policies),
        core_path_rows=len(paths), saved_BR_profit_max_error=profit_error, copy_hashes_match=all(hash_pairs.values()),
        new_Nash_games=0, live_engine_calls=0,
        interpretation='Saved selected candidates only; finite FOC/BR acceptance is not mathematical Nash uniqueness; no opt-out response computed')
    return dict(summary=summary, games=games, platforms=platforms, menus=menus,
                unresolved=unresolved, intervals=intervals, policies=policies)


def input_hashes(root):
    root = Path(root)
    paths = [p for folder in (root/HET, root/CORE) for p in folder.rglob('*') if p.is_file()]
    return {p.relative_to(root).as_posix():digest(p) for p in sorted(paths)}


def run(root, folder=None):
    root = Path(root)
    folder = Path(folder) if folder is not None else root/OUTPUT
    if folder.exists():
        raise FileExistsError('Audit output cannot overwrite an existing directory')
    before = input_hashes(root)
    with closed_arithmetic():
        out = analyze(root)
    if input_hashes(root) != before:
        raise ValueError('Historical audit inputs changed during arithmetic')
    folder.mkdir(parents=True)
    dump(folder/'design.json', dict(schema=1, unit_margin='fare - actual_operating_cost - charge*resource',
        strict_cap='actual_operating_cost + charge*resource > fare_upper',
        weak_cap='actual_operating_cost + charge*resource >= fare_upper',
        fare_tolerance=1e-8, wait_tolerance=1e-6, new_Nash_games=0,
        missing_state='null, never inferred from a rejected candidate',
        strong_resource_ratio_min=4.5, optimization='max saved TS, lower charge tie break; incomplete menus descriptive only'))
    for key in out:
        if key != 'summary':
            export(folder/(key+'.csv'), out[key])
    dump(folder/'summary.json', out['summary'])
    dump(folder/'provenance.json', dict(inputs_sha256=before, unchanged_after_arithmetic=True,
        source_sha256={p:digest(root/p) for p in OWNED}))
    dump(folder/'manifest.json', dict(files_sha256={p.name:digest(p) for p in folder.iterdir() if p.is_file()}))
    return out


def replay(root, folder):
    root, folder = Path(root), Path(folder)
    for name, h in read(folder/'manifest.json')['files_sha256'].items():
        if digest(folder/name) != h:
            raise ValueError('Audit artifact changed: '+name)
    if input_hashes(root) != read(folder/'provenance.json')['inputs_sha256']:
        raise ValueError('Historical audit inputs changed')
    with closed_arithmetic():
        out = analyze(root)
    if out['summary'] != read(folder/'summary.json'):
        raise ValueError('Replayed audit differs')
    return out
