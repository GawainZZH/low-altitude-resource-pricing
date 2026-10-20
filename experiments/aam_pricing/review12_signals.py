"""Training-frozen extra-state signal benchmark on saved finite menus only."""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import io
from itertools import product
import json
import math
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np

from .review10_core import capacity_state, close

OUTPUT = 'results/trb_review12_signals_v1'
PARAMS = 'results/trb_review_parameters_v1/full_parameters.json'
TRAIN = 'results/trb_review_hard_capacity_repair_v1/corrected_hard_charge_records.csv'
REF = 'results/trb_review5_refinement_v1'
UT = 'results/trb_review7_policy_controls_v1'
DESIGN_NOTE = 'docs/qa/trb_twelfth_review_20261018/signals_design.md'
INPUTS = [PARAMS, TRAIN, UT+'/frozen_design.json', UT+'/summary.json']
INPUTS += [REF+'/'+name for name in ('parameters.json', 'source.csv', 'old_games.jsonl',
                                   'new_games.jsonl', 'prespecified_design.json')]
INPUTS += [DESIGN_NOTE]
OWNED = ['experiments/aam_pricing/review12_signals.py', 'scripts/run_review12_signals.py',
         'tests/test_review12_signals.py']
SOURCES = [*OWNED, 'experiments/__init__.py', 'experiments/aam_pricing/__init__.py',
           'experiments/aam_pricing/review10_core.py']
BASELINE = dict(threshold=.9, slope=20., eta=0.)
POLICIES = ('UT', 'UT_grid', 'UTE')
TOL = 1e-8


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def json_bytes(obj):
    return (json.dumps(obj, indent=2, ensure_ascii=True, allow_nan=False)+'\n').encode('utf-8')


def dump(path, obj):
    Path(path).write_bytes(json_bytes(obj))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csv_bytes(rows):
    out = io.StringIO(newline='')
    keys = list(dict.fromkeys(k for r in rows for k in r))
    writer = csv.DictWriter(out, fieldnames=keys)
    writer.writeheader()
    writer.writerows({k: json.dumps(v, allow_nan=False) if isinstance(v, (dict, list)) else v
                     for k, v in r.items()} for r in rows)
    return out.getvalue().encode('utf-8')


def rule_grid():
    return [dict(threshold=t, slope=s, eta=e) for t, s, e in
            product((.8, .9, 1.), (0., 5., 10., 15., 20., 25., 30.),
                    (-4., -2., 0., 2., 4., 8., 12.))]


def design():
    return dict(schema=1, externality_reference=2800., rule_count=147, rules=rule_grid(),
        baseline=BASELINE, administration_kappa=0., training_states=450,
        test_states=90, test_candidates=1635, budget_seconds=30.,
        formula='slope*max(0,offered_NC-threshold)+eta*(E/2800-1)',
        externality_field='case.congestion_cost; cross-check CSV externality',
        signal='direct MNL at saved NC fares, rho=1, access expense=0',
        charge_domain=[0., 20.], projection='nearest actual candidate; lower exact-distance ties',
        training_objective='unweighted aggregate original availability TS at kappa=0',
        selection_tie='max total_TS, min total_charge, min abs(eta), min eta, min slope, max threshold',
        nested_control='UT_grid selected from the 21 eta=0 rules within the 147; UT unchanged',
        training_institution='availability', evaluation_institution='waiting',
        selection_stage='before_waiting_evaluation', optional_held30='not evaluated',
        inference='Finite cached menus and institutional transfer, not independent market observations; '
                  'no inherent global Nash guarantee.',
        validation_dependency='Parent validation separately audits the 150 refinement games; '
                              'not performed or assumed passed by this benchmark.',
        administration='Incremental SD-policy benefit thresholds only, not observed expenses',
        units=dict(money='model money per period; USD 5 per model money unit',
            charge='model money per resource unit', request='potential requests per model period',
            time='minutes', externality='model money coefficient E; E/2800 dimensionless'))


@contextmanager
def closed_arithmetic():
    """Catch live calls and previously imported aliases without constructing engines."""
    old = sys.getprofile()
    counts = dict(blocked_calls=0)
    forbidden = {'outcome', 'choice', 'profit', 'platform_profit', 'gradient', 'jacobian',
                 'solve_game', 'solve_candidate', 'solve_platform_prices', 'repair_game',
                 'whole_box', 'wide_deviations', 'coordinate_br', 'audit_candidate'}

    def guard(frame, event, arg):
        if event != 'call':
            return
        module = frame.f_globals.get('__name__', '')
        name = frame.f_code.co_name
        owner = frame.f_locals.get('self')
        engine = owner is not None and any('Engine' in c.__name__ for c in type(owner).__mro__)
        live = module.startswith('experiments.aam_pricing') and (
            name in forbidden or name.startswith(('solve_demand', 'solve_hard', 'solve_waiting')) or
            engine)
        if module.startswith('scipy.optimize') or live:
            counts['blocked_calls'] += 1
            raise RuntimeError('Closed arithmetic blocked live engine/BR/root call: '+module+'.'+name)
    sys.setprofile(guard)
    try:
        yield counts
    finally:
        sys.setprofile(old)


def offered_pressure(case, prices):
    """Exact review7 offered-pressure arithmetic, no closure root or engine."""
    mass = np.asarray(case['base_demand'])*case['demand_scale']
    affinity = np.asarray(case['platform_affinity'])+np.eye(len(mass), 3)*case['differentiation']
    h1 = .55+1.10*(1.-.88)**2
    loss = np.asarray(case['base_loss'])+np.asarray(case['exposure'])*h1
    air = (affinity-np.asarray(case['price_sensitivity'])[:, None]*np.asarray(prices)
           -np.asarray(case['time_sensitivity'])[:, None]*case['service_time']
           -np.asarray(case['reliability_sensitivity'])[:, None]*loss)/case['logit_scale']
    outside = (np.asarray(case['outside_affinity'])-np.asarray(case['price_sensitivity'])*case['outside_price']
               -np.asarray(case['time_sensitivity'])*case['outside_time'])/case['logit_scale']
    peak = np.maximum(outside, air.max(axis=1))
    weights = np.exp(air-peak[:, None])
    probabilities = weights/(np.exp(outside-peak)+weights.sum(axis=1))[:, None]
    return float((mass@probabilities)@case['platform_resource']/(case['base_capacity']*case['capacity_scale']))


def validate_menu(m):
    rows = m['candidates']
    charges = [r['charge'] for r in rows]
    if not charges or charges != sorted(set(charges)) or charges[0] != 0.:
        raise ValueError('Sorted unique candidate menu with own zero NC required')
    if charges[-1] > 20.:
        raise ValueError('Candidate outside frozen charge domain')
    values = [m['offered_signal'], m['externality'], m['potential']]
    values += [r[k] for r in rows for k in ('charge', 'TS', 'served')]
    if not all(math.isfinite(x) for x in values) or m['potential'] <= 0:
        raise ValueError('Nonfinite signal or candidate, or nonpositive request mass')
    if m['externality'] != m['case']['congestion_cost']:
        raise ValueError('Externality must come from actual case E')


def normalized_menu(name, case, institution, candidates, raw_cases):
    candidates.sort(key=lambda r:r['charge'])
    nc = candidates[0]
    result = dict(id=institution+':'+name, case_id=name, institution=institution,
        case=case, raw_case=raw_cases[name], candidates=candidates,
        offered_signal=offered_pressure(case, nc['prices']), externality=case['congestion_cost'],
        potential=sum(case['base_demand'])*case['demand_scale'],
        behavior_profile=[case[k] for k in ('demand_scale', 'capacity_scale', 'reliability_power')],
        NC_waiting=nc['wait_minutes'] > 1e-6 if institution == 'waiting' else None)
    validate_menu(result)
    return result


def load_training(inputs):
    inputs = Path(inputs)
    parameters = read(inputs/PARAMS)
    if parameters['units']['money_usd_per_model_unit'] != 5.:
        raise ValueError('Frozen monetary units differ')
    raw_cases = {r['dataclass']['name']:r for r in parameters['full_state_grid']}
    grouped = defaultdict(list)
    for r in read_csv(inputs/TRAIN):
        case = raw_cases[r['case']]['dataclass']
        if r['repair_verified'] != '1':
            raise ValueError('Unaccepted training candidate')
        if float(r['externality_weight']) != case['congestion_cost']:
            raise ValueError('Training externality field differs from case E')
        prices, y = ([float(r[f'{k}_{i}']) for i in (1, 2, 3)] for k in ('price', 'demand'))
        receipts = float(np.dot(prices, y))
        ts = float(r['welfare'])+receipts
        close(ts, float(r['TS']))
        close(float(r['welfare']), float(r['US']))
        grouped[r['case']].append(dict(charge=float(r['charge']), TS=ts, served=sum(y),
            prices=prices, demand=y, rho=float(r['utilization']), wait_minutes=None,
            availability=float(r['hard_availability']), source='repaired_availability',
            game=r['repair_game']))
    menus = [normalized_menu(name, raw_cases[name]['dataclass'], 'availability', rows, raw_cases)
             for name, rows in sorted(grouped.items())]
    if len(menus) != 450 or set(grouped) != set(raw_cases):
        raise ValueError('Training requires complete original 450-case grid')
    if any([r['charge'] for r in m['candidates']] != list(range(0, 21, 2)) for m in menus):
        raise ValueError('Original 11-charge training menu differs')
    return menus


def project(menu, rule):
    raw = rule['slope']*max(0., menu['offered_signal']-rule['threshold'])
    raw += rule['eta']*(menu['externality']/2800.-1.)
    target = min(20., max(0., raw))
    selected = min(menu['candidates'], key=lambda r:(abs(r['charge']-target), r['charge']))
    return dict(raw_target=raw, target=target, selected=selected)


def training_rank(score):
    return (score['total_TS'], -score['total_charge'], -abs(score['eta']),
            -score['eta'], -score['slope'], score['threshold'])


def select_training(menus):
    if not menus:
        raise ValueError('Empty training sample')
    for menu in menus:
        validate_menu(menu)
        if menu['institution'] != 'availability':
            raise ValueError('Only original availability menus can select parameters')
    scores = []
    # The original menus share a charge grid; fallback handles synthetic tests.
    lengths = {len(m['candidates']) for m in menus}
    if len(lengths) != 1:
        raise ValueError('Training candidate lengths differ')
    charges = np.array([[r['charge'] for r in m['candidates']] for m in menus])
    values = np.array([[r['TS'] for r in m['candidates']] for m in menus])
    signals = np.array([m['offered_signal'] for m in menus])
    externality = np.array([m['externality'] for m in menus])
    for rule in rule_grid():
        target = np.clip(rule['slope']*np.maximum(0., signals-rule['threshold'])
                         +rule['eta']*(externality/2800.-1.), 0., 20.)
        index = np.argmin(abs(charges-target[:, None]), axis=1)
        selected_values, selected_q = values[np.arange(len(menus)), index], charges[np.arange(len(menus)), index]
        total, total_q = math.fsum(selected_values), math.fsum(selected_q)
        scores.append(dict(rule, total_TS=total, total_charge=total_q,
            mean_ts=total/len(menus), mean_charge=total_q/len(menus)))
    best = max(scores, key=training_rank)
    matched = max((r for r in scores if r['eta'] == 0.), key=training_rank)
    params = lambda r: {k:r[k] for k in ('threshold', 'slope', 'eta')}
    return dict(UT=dict(BASELINE), UT_grid=params(matched), UTE=params(best), scores=scores,
        training_only=True, training_states=len(menus), selection_stage='before_waiting_evaluation')


def baseline_identity(inputs, training, selection):
    old = read(Path(inputs)/UT/'summary.json')['training_rules']['common_offered_load_TS']
    frozen = read(Path(inputs)/UT/'frozen_design.json')['rules']['common_offered_load_TS']
    expected = dict(threshold=.9, slope=20., signal='offered', training_states=450)
    for obj in (old, frozen):
        if {k:obj[k] for k in expected} != expected:
            raise ValueError('Original frozen offered-load UT identity differs')
    score = next(r for r in selection['scores'] if all(r[k] == v for k, v in BASELINE.items()))
    close(score['mean_ts'], old['mean_ts'], 1e-9)
    close(score['mean_charge'], old['mean_charge'], 1e-12)
    # Explicitly verify vectorized scoring against literal original projection.
    chosen = [project(m, BASELINE)['selected'] for m in training]
    close(math.fsum(r['TS'] for r in chosen), score['total_TS'], 1e-9)
    return dict(expected, mean_ts=score['mean_ts'], mean_charge=score['mean_charge'],
                exact_projection_equivalence=True, source='review7 common_offered_load_TS')


def load_test(inputs):
    inputs = Path(inputs)
    raw_cases = {r['dataclass']['name']:r for r in read(inputs/REF/'parameters.json')['main']}
    grouped = defaultdict(list)
    for r in read_csv(inputs/REF/'source.csv'):
        if r['cohort'] != 'main':
            continue
        if r['accepted'] != 'True':
            raise ValueError('Incomplete waiting candidate cannot be omitted')
        case = raw_cases[r['case']]['dataclass']
        if float(r['externality']) != case['congestion_cost']:
            raise ValueError('Waiting externality field differs from case E')
        fields = ('charge', 'TS', 'CS', 'US', 'revenue', 'operating_cost', 'external_cost',
                  'rho', 'wait_minutes', 'unrestricted_gap')
        nums = {k:float(r[k]) for k in fields}
        prices, y = ([float(r[f'{k}_{i}']) for i in (1, 2, 3)] for k in ('price', 'demand'))
        rebuilt = capacity_state(case, prices, nums['rho'], nums['wait_minutes'])
        for key in ('TS', 'CS', 'US', 'revenue', 'operating_cost', 'external_cost'):
            close(rebuilt[key], nums[key])
        close(rebuilt['demand'], y)
        close(rebuilt['offered'], nums['rho'], 1e-8)
        grouped[r['case']].append(dict(nums, prices=prices, demand=y, served=sum(y),
            game=r['game'], source=r['source'], family=r['family']))
    menus = [normalized_menu(name, raw_cases[name]['dataclass'], 'waiting', rows, raw_cases)
             for name, rows in sorted(grouped.items())]
    if len(menus) != 90 or set(grouped) != set(raw_cases):
        raise ValueError('Complete native 90 waiting menus required')
    if sum(len(m['candidates']) for m in menus) != 1635:
        raise ValueError('Refined waiting candidate set differs')
    for m in menus:
        if not 15 <= len(m['candidates']) <= 20 or m['candidates'][-1]['charge'] != 20.:
            raise ValueError('Refined waiting charge menu differs')
        close(m['offered_signal'], 1+m['candidates'][0]['unrestricted_gap'], 1e-10)
    if sum(m['NC_waiting'] for m in menus) != 60:
        raise ValueError('Frozen NC waiting partition differs')
    return menus


def evaluate(menus, rules):
    rows = []
    for menu in menus:
        validate_menu(menu)
        nc = menu['candidates'][0]
        sd = max(menu['candidates'], key=lambda r:(r['TS'], -r['charge']))
        row = dict(menu=menu['id'], case_id=menu['case_id'], institution=menu['institution'],
            externality=menu['externality'], offered_signal=menu['offered_signal'],
            NC_waiting=menu['NC_waiting'], potential=menu['potential'],
            candidate_charges=[r['charge'] for r in menu['candidates']],
            NC_TS=nc['TS'], SD_TS=sd['TS'], SD_gain=sd['TS']-nc['TS'], SD_charge=sd['charge'])
        for name in POLICIES:
            p = project(menu, rules[name]); selected = p['selected']
            gap = sd['TS']-selected['TS']
            row.update({name+'_raw_target':p['raw_target'], name+'_target':p['target'],
                name+'_charge':selected['charge'], name+'_TS':selected['TS'],
                name+'_gain':selected['TS']-nc['TS'], name+'_served':selected['served'],
                'SD_minus_'+name:gap,
                name+'_admin_per_potential':gap/menu['potential'],
                name+'_admin_per_served':gap/selected['served'] if selected['served'] > 0 else None})
        row['UTE_minus_UT'] = row['UTE_TS']-row['UT_TS']
        row['UTE_minus_UT_grid'] = row['UTE_TS']-row['UT_grid_TS']
        row['UT_grid_minus_UT'] = row['UT_grid_TS']-row['UT_TS']
        rows.append(row)
    return rows


def group_stats(rows):
    sd = math.fsum(r['SD_gain'] for r in rows)
    mass = math.fsum(r['potential'] for r in rows)
    result = dict(states=len(rows), potential=mass, SD_gain=sd,
                  positive_SD_states=sum(r['SD_gain'] > TOL for r in rows))
    for name in POLICIES:
        gain = math.fsum(r[name+'_gain'] for r in rows)
        gaps = [r['SD_minus_'+name] for r in rows]
        served = math.fsum(r[name+'_served'] for r in rows)
        result[name] = dict(gain=gain, retention=gain/sd if sd > TOL else None,
            total_gap=math.fsum(gaps), max_absolute_gap=max(map(abs, gaps), default=None),
            adverse_states=sum(r[name+'_gain'] < -TOL for r in rows),
            strict_gap_states=sum(x > TOL for x in gaps),
            admin_break_even_per_potential=math.fsum(gaps)/mass if mass else None,
            admin_break_even_per_served=math.fsum(gaps)/served if served else None,
            max_admin_per_potential=max((r[name+'_admin_per_potential'] for r in rows), default=None),
            max_admin_per_served=max((r[name+'_admin_per_served'] for r in rows
                                      if r[name+'_admin_per_served'] is not None), default=None))
    for name in ('UTE_minus_UT', 'UTE_minus_UT_grid', 'UT_grid_minus_UT'):
        vals = [r[name] for r in rows]
        result[name] = dict(total=math.fsum(vals), positive_states=sum(v > TOL for v in vals),
            negative_states=sum(v < -TOL for v in vals), zero_states=sum(abs(v) <= TOL for v in vals),
            minimum=min(vals, default=None), maximum=max(vals, default=None))
    return result


def finish_analysis(training, test, selection, identity):
    rules = {k:selection[k] for k in POLICIES}
    rows = evaluate(test, rules)
    train_rows = evaluate(training, rules)
    groups = dict(all=group_stats(rows),
        NC_positive_wait=group_stats([r for r in rows if r['NC_waiting']]),
        NC_zero_wait=group_stats([r for r in rows if not r['NC_waiting']]))
    training_ids, test_ids = {m['id'] for m in training}, {m['id'] for m in test}
    raw_overlap = sorted({m['case_id'] for m in training} & {m['case_id'] for m in test})
    overlap = dict(institution_label_overlap=sorted(training_ids & test_ids),
        raw_case_name_overlap=raw_overlap, raw_case_name_overlap_count=len(raw_overlap),
        test_behavior_profiles=len({tuple(m['behavior_profile']) for m in test}),
        independent_market_observations=False,
        interpretation='Institutional transfer on shared calibrated backgrounds and repeated E values')
    if overlap['institution_label_overlap']:
        raise ValueError('Training/evaluation institution labels overlap')
    summary = dict(schema=1, training_states=len(training), test_states=len(test),
        test_candidates=sum(len(m['candidates']) for m in test), rule_count=147,
        selected_parameters=rules, baseline_identity=identity,
        training=group_stats(train_rows), groups=groups, overlap=overlap,
        new_nash_games=0, new_br_searches=0, new_root_calls=0, new_engine_calls=0,
        inference=design()['inference'], validation_dependency=design()['validation_dependency'],
        administration=design()['administration'], units=design()['units'],
        optional_held30=design()['optional_held30'])
    return dict(summary=summary, training_selection=selection,
        normalized=dict(training=training, test=test), policies=rows, training_policies=train_rows)


def markdown(summary):
    lines = ['# Frozen Cached Signal Benchmark', '',
        'Original UT and all fitted parameters are fixed before waiting-menu evaluation.', '',
        '```json', json.dumps(summary['selected_parameters'], indent=2), '```', '',
        '| Group | Policy | n | Aggregate gain | Retention | Max absolute SD gap | Break-even/request | Adverse |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for label, g in summary['groups'].items():
        for name in POLICIES:
            p = g[name]
            retained = f"{100*p['retention']:.6f}%" if p['retention'] is not None else 'NA'
            lines.append(f"| {label} | {name} | {g['states']} | {p['gain']:.9f} | {retained} | "
                f"{p['max_absolute_gap']:.9f} | {p['admin_break_even_per_potential']:.9f} | {p['adverse_states']} |")
    lines += ['', 'Incremental contrasts (model money per period, summed over design states):', '']
    for label, g in summary['groups'].items():
        lines.append(f"- {label}: UTE-UT = {g['UTE_minus_UT']['total']:.9f}; "
            f"UTE-UT_grid = {g['UTE_minus_UT_grid']['total']:.9f}; "
            f"UT_grid-UT = {g['UT_grid_minus_UT']['total']:.9f}.")
    lines += ['', 'SD is the finite-menu TS optimum. UT_grid controls for the revised training grid. '
        'The UTE contrast is evidence about one frozen signal family, not an unrestricted information value.',
        'Potential-request denominators describe incremental SD-policy administration break-even '
        'benefits; served denominators are in summary.json and policies.csv. No measured costs or '
        'operating probabilities support an implementation recommendation.',
        'Training and evaluation labels have different institution prefixes. Shared backgrounds and '
        'five externality levels per behavior profile are not independent market observations.',
        summary['validation_dependency'], summary['inference'],
        'Optional held30 was not evaluated. Negative and zero gains remain in the tables.']
    return '\n'.join(lines)+'\n'


def artifacts(out):
    return {'summary.json':json_bytes(out['summary']),
        'summary.md':markdown(out['summary']).encode('utf-8'),
        'normalized_cases.json':json_bytes(out['normalized']),
        'policies.csv':csv_bytes(out['policies']),
        'training_policies.csv':csv_bytes(out['training_policies']),
        'training_scores.csv':csv_bytes(out['training_selection']['scores'])}


def verify_manifest(folder):
    folder = Path(folder)
    manifest = read(folder/'manifest.json')
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*')
              if p.is_file() and p.relative_to(folder).as_posix() != 'manifest.json'}
    if actual != set(manifest['files_sha256']):
        raise ValueError('Frozen artifact hash inventory differs')
    for name, h in manifest['files_sha256'].items():
        if digest(folder/name) != h:
            raise ValueError('Frozen artifact hash differs: '+name)
    frozen = read(folder/'frozen_inputs.json')
    if set(frozen['inputs_sha256']) != set(INPUTS) or set(frozen['execution_source_sha256']) != set(SOURCES):
        raise ValueError('Frozen input/source hash inventory differs')
    for directory, key in [('inputs', 'inputs_sha256'), ('execution_source', 'execution_source_sha256')]:
        for name, h in frozen[key].items():
            if digest(folder/directory/name) != h:
                raise ValueError('Frozen input/source hash differs: '+name)


def run(root, folder=None):
    start = perf_counter()
    root = Path(root)
    folder = root/OUTPUT if folder is None else Path(folder)
    if folder.exists():
        raise FileExistsError('Frozen signals output cannot be overwritten; use replay')
    folder.mkdir(parents=True)
    dump(folder/'design.json', dict(design(), frozen_at=datetime.now(timezone.utc).isoformat()))
    hashes = {}
    for name in INPUTS:
        dst = folder/'inputs'/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        hashes[name] = digest(root/name)
        shutil.copyfile(root/name, dst)
        if digest(dst) != hashes[name]:
            raise ValueError('Input changed during snapshot: '+name)
    sources = {}
    for name in SOURCES:
        dst = folder/'execution_source'/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        sources[name] = digest(root/name)
        shutil.copyfile(root/name, dst)
        if digest(dst) != sources[name]:
            raise ValueError('Source changed during snapshot: '+name)
    dump(folder/'frozen_inputs.json', dict(inputs_sha256=hashes, execution_source_sha256=sources))
    with closed_arithmetic() as counts:
        training = load_training(folder/'inputs')
        selection = select_training(training)
        identity = baseline_identity(folder/'inputs', training, selection)
        dump(folder/'selected_training.json', selection)
        if perf_counter()-start >= 30.:
            raise TimeoutError('Bounded signals batch exhausted before evaluation; freeze preserved')
        test = load_test(folder/'inputs')
        out = finish_analysis(training, test, selection, identity)
    for name, data in artifacts(out).items():
        (folder/name).write_bytes(data)
    if any(digest(root/name) != h for name, h in {**hashes, **sources}.items()):
        raise ValueError('Original input or source changed during cached run')
    elapsed = perf_counter()-start
    dump(folder/'run_report.json', dict(elapsed_seconds=elapsed, budget_seconds=30.,
        within_budget=elapsed < 30., original_inputs_unchanged=True, rule_count=147,
        training_freeze_sha256=digest(folder/'selected_training.json'), tripwire=counts,
        new_nash_games=0, new_br_searches=0, new_root_calls=0, new_engine_calls=0))
    if elapsed >= 30.:
        raise TimeoutError('Bounded signals batch exceeded 30 seconds; outputs preserved')
    files = {p.relative_to(folder).as_posix():digest(p) for p in sorted(folder.rglob('*')) if p.is_file()}
    dump(folder/'manifest.json', dict(schema=1, files_sha256=files,
        design_sha256=digest(folder/'design.json'), selected_training_sha256=digest(folder/'selected_training.json')))
    return out


def replay(folder):
    folder = Path(folder)
    verify_manifest(folder)
    frozen = read(folder/'design.json')
    frozen.pop('frozen_at')
    if frozen != design():
        raise ValueError('Frozen signals design differs')
    with closed_arithmetic():
        training = load_training(folder/'inputs')
        selection = select_training(training)
        if selection != read(folder/'selected_training.json'):
            raise ValueError('Rebuilt training freeze differs')
        identity = baseline_identity(folder/'inputs', training, selection)
        test = load_test(folder/'inputs')
        out = finish_analysis(training, test, selection, identity)
    for name, data in artifacts(out).items():
        if (folder/name).read_bytes() != data:
            raise ValueError('Rebuilt cached arithmetic differs: '+name)
    return out
