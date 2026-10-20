"""Frozen cached capacity-clearing evidence; no new behavioral computation.

Plan: observe RED contracts; copy complete inputs and failed attempts; rebuild
finite paths/policy arithmetic; verify copied replay under a solver tripwire.
Release intervals are sampled finite differences, not causal channel estimates.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

import numpy as np

OUTPUT = 'results/trb_review10_core_v1'
HET = 'results/trb_review6_heterogeneity_v1'
REF = 'results/trb_review5_refinement_v1'
UT = 'results/trb_review7_policy_controls_v1'
INPUTS = [HET+'/'+n for n in ('design.json', 'games.jsonl', 'rescue_games.jsonl')]
INPUTS += [REF+'/'+n for n in ('parameters.json', 'source.csv', 'policies.csv',
                              'old_games.jsonl', 'new_games.jsonl', 'prespecified_design.json')]
INPUTS += [UT+'/'+n for n in ('summary.json', 'training_tuning.csv', 'frozen_design.json', 'full_parameters.json')]
OWNED = ['experiments/aam_pricing/review10_core.py', 'scripts/run_review10_core.py',
         'tests/test_review10_core.py']


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=True, allow_nan=False,
                                   separators=(',', ':'))+'\n', encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def export(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, allow_nan=False) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()} for row in rows)


@contextmanager
def closed_arithmetic():
    """Fail on live behavior, BR or optimization, including imported aliases."""
    old = sys.getprofile()

    def guard(frame, event, arg):
        if event != 'call':
            return
        module = frame.f_globals.get('__name__', '')
        name = frame.f_code.co_name
        owner = frame.f_locals.get('self')
        engine = owner is not None and any('Engine' in c.__name__ for c in type(owner).__mro__)
        if (module.startswith('scipy.optimize') or
            (module.startswith('experiments.aam_pricing') and
             (name in {'solve_game', 'solve_candidate', 'whole_box', 'wide_deviations',
                       'outcome', 'choice', 'profit', 'gradient', 'jacobian'} or
              (name == '__init__' and engine)))):
            raise RuntimeError('Closed arithmetic replay blocked live behavior: '+module+'.'+name)
    sys.setprofile(guard)
    try:
        yield
    finally:
        sys.setprofile(old)


def freeze_inputs(root, folder, paths, owned):
    root, folder = Path(root), Path(folder)
    hashes = {}
    for name in paths:
        dst = folder/'inputs'/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        hashes[name] = digest(root/name)
        shutil.copyfile(root/name, dst)
        if digest(dst) != hashes[name]:
            raise ValueError('Input copy hash differs')
    sources = {name: digest(root/name) for name in owned}
    for name in owned:
        dst = folder/'execution_source'/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/name, dst)
    dump(folder/'frozen_inputs.json', dict(inputs_sha256=hashes, execution_source_sha256=sources))
    return hashes, sources


def finish_manifest(folder):
    folder = Path(folder)
    files = {p.relative_to(folder).as_posix(): digest(p) for p in sorted(folder.rglob('*'))
             if p.is_file() and p.name != 'manifest.json'}
    dump(folder/'manifest.json', dict(schema=1, files_sha256=files,
        provenance='Execution source identity is frozen; current replay validator may be newer'))


def verify_manifest(folder):
    folder = Path(folder)
    for name, h in read(folder/'manifest.json')['files_sha256'].items():
        if digest(folder/name) != h:
            raise ValueError('Frozen artifact hash differs: '+name)
    for name, h in read(folder/'frozen_inputs.json')['inputs_sha256'].items():
        if digest(folder/'inputs'/name) != h:
            raise ValueError('Frozen input hash differs: '+name)


def capacity_state(case, prices, rho, wait):
    """Direct MNL/accounting at supplied state, never an implicit closure solve."""
    p = np.asarray(prices, float)
    D = np.asarray(case['base_demand'])*case['demand_scale']
    lam, beta, gamma = (np.asarray(case[k]) for k in
                       ('price_sensitivity', 'time_sensitivity', 'reliability_sensitivity'))
    mu = case['logit_scale']
    alpha = np.asarray(case['platform_affinity'])+np.eye(len(D), 3)*case['differentiation']
    loss = np.asarray(case['base_loss'])+np.asarray(case['exposure'])*(
        .55*rho**case['reliability_power']+1.10*max(rho-.88, 0.)**2)
    air = (alpha-lam[:, None]*p-beta[:, None]*(np.asarray(case['service_time'])+wait)
           -gamma[:, None]*loss)/mu
    outside = (np.asarray(case['outside_affinity'])-lam*case['outside_price']-beta*case['outside_time'])/mu
    peak = np.maximum(outside, air.max(axis=1))
    den = peak+np.log(np.exp(outside-peak)+np.exp(air-peak[:, None]).sum(axis=1))
    prob = np.exp(air-den[:, None]); y = D@prob
    cost = np.asarray(case['base_cost'])+np.array([0., case['cost_gap'], .5*case['cost_gap']])
    cs, receipts, C = float((D*mu/lam)@den), float(p@y), float(cost@y)
    G = case['congestion_cost']*(.15*rho*rho+max(rho-.9, 0.)**3)
    return dict(demand=y.tolist(), CS=cs, revenue=receipts, operating_cost=C,
                external_cost=G, TS=cs+receipts-C-G, US=cs-C-G,
                offered=float(y@case['platform_resource']/(case['base_capacity']*case['capacity_scale'])))


def close(a, b, tolerance=1e-7):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or np.max(abs(a-b)) > tolerance:
        raise ValueError('Saved arithmetic state differs')


def partition(rows):
    rows = sorted(rows, key=lambda r: r['charge'])
    if not rows or rows[0]['charge'] != 0:
        raise ValueError('Own NC required')
    nc = rows[0]; opt = max(rows, key=lambda r: (r['TS'], -r['charge']))
    zero = next((r for r in rows if r['wait_minutes'] <= 1e-6), None)
    before = [r for r in rows if zero and r['charge'] < zero['charge'] and r['wait_minutes'] > 1e-6]
    left = before[-1] if before else None
    valid = left is not None and opt['charge'] >= zero['charge']
    pre = left['TS']-nc['TS'] if left else None
    cross = zero['TS']-left['TS'] if left else None
    post = opt['TS']-zero['TS'] if valid else None
    gain = opt['TS']-nc['TS']
    return dict(qL=left['charge'] if left else None, qR=zero['charge'] if zero else None,
        qSD=opt['charge'], pre=pre, cross=cross, post=post, gain=gain,
        identity_error=gain-pre-cross-post if valid else None,
        NC_waiting=nc['wait_minutes'] > 1e-6, sampled_not_causal=True)


def design():
    return dict(schema=1, strong_resource_ratio_min=4.5,
        primary_paths=['joint_2_c0', 'resource_2_c1', 'resource_3_c1'],
        wait_tolerance=1e-6, zero_gain_tolerance=1e-8,
        frozen_UT=dict(training_states=450, threshold=.9, slope=20., signal='offered'),
        signal_groups=['below_0.9', '0.9_to_1', '1_to_1.2', 'at_least_1.2'],
        inference='Saved finite candidates, no causal channel attribution or continuous/global Nash claims',
        new_Nash_games=0)


def group_stats(rows):
    gaps = [r['SD_minus_UT'] for r in rows]
    total = sum(r['SD_gain'] for r in rows)
    zero = sum(abs(r['SD_gain']) <= 1e-8 for r in rows)
    return dict(n=len(rows), positive_gap=sum(g > 1e-8 for g in gaps),
        zero_gap=sum(abs(g) <= 1e-8 for g in gaps), zero_SD_gain=zero,
        positive_SD_gain=sum(r['SD_gain'] > 1e-8 for r in rows),
        adverse_UT=sum(r['UT_gain'] < -1e-8 for r in rows),
        mean_gap=float(np.mean(gaps)) if gaps else None, max_gap=max(gaps, default=None),
        gap_quantiles=np.quantile(gaps, [0, .25, .5, .75, .9, 1]).tolist() if gaps else [],
        loss_at_least_10pct=sum(r['SD_gain'] > 1e-8 and r['SD_minus_UT']/r['SD_gain'] >= .1 for r in rows),
        total_gap=sum(gaps), total_SD_gain=total,
        retained_gain=1-sum(gaps)/total if total > 1e-8 else None,
        admin_break_even_per_potential=sum(gaps)/sum(r['potential'] for r in rows) if rows else None)


def analyze(inputs):
    inputs = Path(inputs)
    spec = read(inputs/HET/'design.json')
    originals = [json.loads(s) for s in (inputs/HET/'games.jsonl').read_text().splitlines() if s.strip()]
    rescues = [json.loads(s) for s in (inputs/HET/'rescue_games.jsonl').read_text().splitlines() if s.strip()]
    selected = {r['game']: r for r in originals}
    if len(selected) != len(originals):
        raise ValueError('Duplicate original game alias')
    for r in rescues:
        if r.get('selected'):
            if selected[r['game']].get('selected'):
                raise ValueError('Rescue cannot replace an accepted original')
            selected[r['game']] = r
    paths, scope = [], []
    for menu in spec['menus']:
        case = menu['case']; a = case['platform_resource']
        if max(a)/min(a) < 4.5-1e-12:
            continue
        games = sorted([r for r in selected.values() if r['menu'] == menu['id']], key=lambda r:r['charge'])
        if [g['charge'] for g in games] != menu['charges']:
            raise ValueError('Planned heterogeneous menu aliases differ')
        complete = all(g.get('selected') for g in games)
        good = [g for g in games if g.get('selected')]
        nc = next((g['diagnostic']['state']['TS'] for g in good if g['charge'] == 0), None)
        scope.append(dict(menu=menu['id'], complete=complete, planned=len(games), accepted=len(good),
            failed_charges=[g['charge'] for g in games if not g.get('selected')],
            case=case, resource_ratio=max(a)/min(a)))
        for g in good:
            s = g['diagnostic']['state']
            rebuilt = capacity_state(case, s['prices'], s['rho'], s['wait_minutes'])
            for key, value in [('demand', s['demand']), ('CS', s['cs']), ('TS', s['TS'])]:
                close(rebuilt[key], value)
            close(sum(s['demand']), s['served'])
            paths.append(dict(menu=menu['id'], family=menu['family'], charge=g['charge'], complete=complete,
                core_path=menu['id'] in design()['primary_paths'], rho=s['rho'], wait_minutes=s['wait_minutes'],
                Y=s['served'], TS=s['TS'], TS_gain=s['TS']-nc if nc is not None else None,
                prices=s['prices'], demand=s['demand'], resource_intensity=a,
                fare_upper_active=max(s['prices']) >= case['price_upper']-1e-8,
                fare_lower_active=min(s['prices']) <= case['price_lower']+1e-8,
                status=g['diagnostic']['status'], closure=s['closure'], game=g['game']))
    cases = {c['dataclass']['name']: c['dataclass'] for c in read(inputs/REF/'parameters.json')['main']}
    source = list(csv.DictReader((inputs/REF/'source.csv').open(encoding='utf-8', newline='')))
    menus = defaultdict(list)
    aliases = set()
    for row in source:
        if row['cohort'] != 'main':
            continue
        if row['accepted'] != 'True':
            raise ValueError('Incomplete native menu cannot enter policies')
        case = cases[row['case']]
        nums = {k: float(row[k]) for k in ('charge', 'TS', 'CS', 'US', 'wait_minutes', 'rho',
                                           'revenue', 'operating_cost', 'external_cost', 'unrestricted_gap')}
        prices, y = ([float(row[f'{k}_{j}']) for j in range(1, 4)] for k in ('price', 'demand'))
        alias = (row['case'], nums['charge'])
        if alias in aliases:
            raise ValueError('Duplicate native case/charge alias')
        aliases.add(alias)
        rebuilt = capacity_state(case, prices, nums['rho'], nums['wait_minutes'])
        for key in ('TS', 'CS', 'US', 'revenue', 'operating_cost', 'external_cost'):
            close(rebuilt[key], nums[key])
        close(rebuilt['demand'], y)
        close(rebuilt['offered'], nums['rho'], 1e-8)
        menus[row['case']].append(dict(nums, prices=prices, Y=sum(y), family=row['family'],
                                        candidate_source=row['source'], game=row['game']))
    rule = read(inputs/UT/'summary.json')['training_rules']['common_offered_load_TS']
    if {k: rule[k] for k in design()['frozen_UT']} != design()['frozen_UT']:
        raise ValueError('Frozen UT rule differs')
    policies, parts = [], []
    for name, rows in sorted(menus.items()):
        rows.sort(key=lambda r:r['charge'])
        if len(rows) not in range(15, 21) or rows[0]['charge'] != 0 or rows[-1]['charge'] != 20:
            raise ValueError('Refined complete 15-20 point menu required')
        case = cases[name]; nc = rows[0]
        offered = capacity_state(case, nc['prices'], 1., 0.)['offered']
        close(offered, 1+nc['unrestricted_gap'], 1e-10)
        target = rule['slope']*max(0., offered-rule['threshold'])
        ut = min(rows, key=lambda r:(abs(r['charge']-target), r['charge']))
        sd = max(rows, key=lambda r:(r['TS'], -r['charge']))
        sg, ug = sd['TS']-nc['TS'], ut['TS']-nc['TS']
        signal_group = ('below_0.9' if offered < .9 else '0.9_to_1' if offered < 1 else
                        '1_to_1.2' if offered < 1.2 else 'at_least_1.2')
        p = dict(case=name, family=nc['family'], candidate_count=len(rows),
            demand_scale=case['demand_scale'], capacity_scale=case['capacity_scale'],
            curvature=case['reliability_power'], externality=case['congestion_cost'],
            offered_signal=offered, signal_group=signal_group, UT_target=target,
            UT_projection_distance=abs(ut['charge']-target), NC_waiting=nc['wait_minutes'] > 1e-6,
            potential=sum(case['base_demand'])*case['demand_scale'], SD_gain=sg, UT_gain=ug,
            SD_minus_UT=sg-ug, relative_gain_loss=(sg-ug)/sg if sg > 1e-8 else None,
            zero_SD_gain=abs(sg) <= 1e-8, candidate_charges=[r['charge'] for r in rows])
        for label, r in [('NC', nc), ('UT', ut), ('SD', sd)]:
            p.update({label+'_'+k:r[k] for k in ('charge', 'TS', 'rho', 'wait_minutes', 'Y', 'prices')})
        p['admin_break_even_per_potential'] = (sg-ug)/p['potential']
        p['admin_break_even_per_UT_served'] = (sg-ug)/ut['Y']
        policies.append(p)
        parts.append(dict(case=name, **partition(rows)))
    groups = {'all': group_stats(policies)}
    group_rows = []
    dimensions = [('NC_wait', lambda r:'NC_waiting' if r['NC_waiting'] else 'NC_zero_wait'),
                  ('signal', lambda r:r['signal_group']), ('curvature', lambda r:str(r['curvature'])),
                  ('externality', lambda r:str(r['externality']))]
    for dim, fn in dimensions:
        for label in sorted(set(fn(r) for r in policies)):
            s = group_stats([r for r in policies if fn(r) == label])
            group_rows.append(dict(dimension=dim, group=label, **s))
            groups[label if dim == 'NC_wait' else dim+':'+label] = s
    matched = []
    for p in policies:
        matched.append(dict(matched_background=f"d{p['demand_scale']}_k{p['capacity_scale']}",
            **{k:p[k] for k in ('case', 'curvature', 'externality', 'offered_signal', 'NC_wait_minutes',
                                'UT_charge', 'SD_charge', 'SD_gain', 'SD_minus_UT', 'zero_SD_gain')}))
    valid = [r for r in parts if r['identity_error'] is not None]
    totals = {k:sum(r[k] for r in valid) for k in ('pre', 'cross', 'post')}
    summary = dict(strong_planned=len(scope), strong_complete=sum(r['complete'] for r in scope),
        strong_incomplete=[r['menu'] for r in scope if not r['complete']], native_states=len(policies),
        native_candidates=sum(len(v) for v in menus.values()), native_NC_waiting=sum(r['NC_waiting'] for r in policies),
        native_NC_zero_wait=sum(not r['NC_waiting'] for r in policies),
        total_SD_gain=sum(r['SD_gain'] for r in policies), partition_totals=totals,
        partition_complete=len(valid), max_partition_error=max(abs(r['identity_error']) for r in valid),
        UT_groups=groups, new_Nash_games=0, replay_engine_calls=0,
        inference=design()['inference'], administration='Break-even thresholds only; no implementation-cost estimate')
    return dict(summary=summary, paths=paths, policies=policies, partitions=parts, scope=scope,
                groups=group_rows, matched_families=matched)


def run(root, folder=None):
    root = Path(root); folder = root/OUTPUT if folder is None else Path(folder)
    if folder.exists():
        raise FileExistsError('Frozen output cannot be overwritten')
    folder.mkdir(parents=True)
    dump(folder/'design.json', design())
    hashes, _ = freeze_inputs(root, folder, INPUTS, OWNED)
    with closed_arithmetic():
        out = analyze(folder/'inputs')
    for key in ('paths', 'policies', 'partitions', 'scope', 'groups', 'matched_families'):
        export(folder/(key+'.csv'), out[key])
    dump(folder/'summary.json', out['summary'])
    dump(folder/'figure_schema.json', dict(fig1='paths.csv: menu,charge,TS_gain,rho,wait_minutes,Y,prices,fare_upper_active,status,complete',
        fig2='policies.csv: case,NC_waiting,signal_group,curvature,externality,SD_minus_UT,SD_gain,zero_SD_gain,relative_gain_loss',
        raw_aliases='inputs preserve case and game aliases and all failed historical attempts'))
    if any(digest(root/p) != h for p, h in hashes.items()):
        raise ValueError('Original input changed during cached analysis')
    finish_manifest(folder)
    return out


def replay(folder):
    folder = Path(folder)
    verify_manifest(folder)
    if read(folder/'design.json') != design():
        raise ValueError('Frozen core design differs')
    with closed_arithmetic():
        out = analyze(folder/'inputs')
    if out['summary'] != read(folder/'summary.json'):
        raise ValueError('Rebuilt core summary differs')
    return out

