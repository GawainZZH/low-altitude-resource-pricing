"""Frozen information-input sensitivity, conditional signals and true saved TS."""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback

OUTPUT = 'results/trb_review13_information_v1'
PARENT = 'results/trb_review12_signals_v1'
REF = 'results/trb_review5_refinement_v1'
INITIAL = 'results/trb_review90_waiting_v1/initial_attempt'
DESIGN = 'docs/qa/trb_thirteenth_review_20261018/information_design.md'
INPUTS = [PARENT+'/'+n for n in ('normalized_cases.json', 'selected_training.json',
    'summary.json', 'policies.csv', 'manifest.json', 'frozen_inputs.json')]
INPUTS += [REF+'/'+n for n in ('parameters.json', 'source.csv', 'old_games.jsonl',
                              'new_games.jsonl', 'prespecified_design.json')]
INPUTS += [INITIAL+'/'+n for n in ('waiting_game_status.csv', 'waiting_games.jsonl')]
INPUTS += [DESIGN]
SOURCES = ['experiments/aam_pricing/review13_information.py',
    'scripts/run_review13_information.py', 'tests/test_review13_information.py',
    'experiments/__init__.py', 'experiments/aam_pricing/__init__.py']
RULES = dict(UT=dict(threshold=.9, slope=20., eta=0.),
             UTE=dict(threshold=.8, slope=20., eta=2.))
ERRORS = (-.2, -.1, .1, .2)
FAMILIES = ('demand_forecast', 'effective_capacity', 'preference_input')
TOL = 1e-8


def design():
    return dict(schema=1, rules=deepcopy(RULES), errors=list(ERRORS), families=list(FAMILIES),
        scenarios=13, decisions=2340, test_states=90, test_candidates=1635,
        budget_seconds=120., reference_rho=1., reference_wait=0.,
        preference='lambda multiplied in BOTH air and outside utility; fixed true NC fares',
        truth='unchanged saved true menu outcomes; SD true finite-menu oracle',
        no_training=True, no_new_equilibrium=True, independent_markets=False,
        primary_objective='TS', money_usd_per_model_unit=5.)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def json_bytes(obj):
    return (json.dumps(obj, indent=2, ensure_ascii=True, allow_nan=False)+'\n').encode()


def dump(path, obj):
    Path(path).write_bytes(json_bytes(obj))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csv_bytes(rows):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
    writer.writeheader()
    writer.writerows({k:json.dumps(v, allow_nan=False) if isinstance(v, (dict, list)) else v
                     for k, v in r.items()} for r in rows)
    return out.getvalue().encode()


def close(a, b, tolerance=1e-7):
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise ValueError('Saved arithmetic dimensions differ')
        for x, y in zip(a, b):
            close(x, y, tolerance)
    elif not (math.isfinite(a) and math.isfinite(b)) or abs(a-b) > tolerance:
        raise ValueError('Saved arithmetic differs: '+str((a, b)))


@contextmanager
def arithmetic_guard():
    """Allow this direct evaluator only; catch preimported Python aliases too."""
    old = sys.getprofile()
    counts = dict(blocked_calls=0)

    def guard(frame, event, arg):
        if event != 'call':
            return
        module = frame.f_globals.get('__name__', '')
        legacy = module.startswith('experiments.aam_pricing.') and module != __name__
        if legacy or module.startswith(('scipy.optimize', 'subprocess', 'multiprocessing',
                                         'concurrent.futures', 'cvxpy', 'gurobipy')):
            counts['blocked_calls'] += 1
            raise RuntimeError('Information arithmetic blocked: '+module+'.'+frame.f_code.co_name)
    sys.setprofile(guard)
    try:
        yield counts
    finally:
        sys.setprofile(old)


def direct_state(case, prices, rho, wait):
    """Original MNL/accounting evaluated at supplied state; no closure equation."""
    mu = case['logit_scale']
    loss = [ell+chi*(.55*rho**case['reliability_power']+1.10*max(rho-.88, 0.)**2)
            for ell, chi in zip(case['base_loss'], case['exposure'])]
    demand = [0., 0., 0.]
    cs = 0.
    for g, base in enumerate(case['base_demand']):
        mass = base*case['demand_scale']
        lam, beta, gamma = [case[k][g] for k in
                            ('price_sensitivity', 'time_sensitivity', 'reliability_sensitivity')]
        air = [(case['platform_affinity'][g][j]+(case['differentiation'] if g == j else 0.)
                -lam*prices[j]-beta*(case['service_time'][j]+wait)-gamma*loss[j])/mu
               for j in range(3)]
        outside = (case['outside_affinity'][g]-lam*case['outside_price']
                   -beta*case['outside_time'])/mu
        peak = max(outside, *air)
        weights = [math.exp(v-peak) for v in air]
        total = math.exp(outside-peak)+sum(weights)
        for j in range(3):
            demand[j] += mass*weights[j]/total
        cs += mass*mu/lam*(peak+math.log(total))
    costs = [case['base_cost'][j]+(0., case['cost_gap'], .5*case['cost_gap'])[j] for j in range(3)]
    receipts = sum(p*y for p, y in zip(prices, demand))
    operating = sum(c*y for c, y in zip(costs, demand))
    external = case['congestion_cost']*(.15*rho*rho+max(rho-.9, 0.)**3)
    work = sum(y*a for y, a in zip(demand, case['platform_resource']))
    return dict(demand=demand, CS=cs, revenue=receipts, operating_cost=operating,
        external_cost=external, TS=cs+receipts-operating-external, US=cs-operating-external,
        work=work, pressure=work/(case['base_capacity']*case['capacity_scale']))


def scenario_grid():
    return [('baseline', 'baseline', 0.)]+[(f'{f}:{e:+.0%}', f, e) for f in FAMILIES for e in ERRORS]


def signal(menu, family, error):
    if (family == 'baseline' and error != 0.) or (family != 'baseline' and
            (family not in FAMILIES or error not in ERRORS)):
        raise ValueError('Only frozen one-factor scenarios allowed')
    case = deepcopy(menu['case'])
    if family == 'demand_forecast':
        case['demand_scale'] *= 1+error
    elif family == 'effective_capacity':
        case['capacity_scale'] *= 1+error
    elif family == 'preference_input':
        case['price_sensitivity'] = [v*(1+error) for v in case['price_sensitivity']]
    state = direct_state(case, menu['candidates'][0]['prices'], 1., 0.)
    true = menu['offered_signal']
    return dict(pressure=state['pressure'], offered_work=state['work'],
        capacity=case['base_capacity']*case['capacity_scale'], relative_bias=state['pressure']/true-1.)


def project(menu, rule, pressure):
    raw = rule['slope']*max(0., pressure-rule['threshold'])+rule['eta']*(menu['externality']/2800.-1.)
    target = min(20., max(0., raw))
    selected = min(menu['candidates'], key=lambda r:(abs(r['charge']-target), r['charge']))
    return dict(raw_target=raw, target=target, selected=selected)


def validate_rules(selection):
    if {k:selection.get(k) for k in RULES} != RULES:
        raise ValueError('Training-frozen UT/UTE parameters differ')


def validate_menus(menus):
    if len(menus) != 90 or len({m['id'] for m in menus}) != 90:
        raise ValueError('Complete unique 90-state menu required')
    for m in menus:
        rows = m['candidates']
        q = [r['charge'] for r in rows]
        if not q or q != sorted(set(q)) or q[0] != 0. or q[-1] != 20. or not 15 <= len(q) <= 20:
            raise ValueError('Incomplete actual charge menu or own NC absent')
        if m['institution'] != 'waiting' or m['case_id'] != m['case']['name']:
            raise ValueError('Waiting menu identity differs')
        close(m['externality'], m['case']['congestion_cost'], 0.)
        close(m['potential'], sum(m['case']['base_demand'])*m['case']['demand_scale'])
        if m['NC_waiting'] != (rows[0]['wait_minutes'] > 1e-6):
            raise ValueError('True NC waiting partition differs')
        close(signal(m, 'baseline', 0.)['pressure'], m['offered_signal'], 1e-12)
    if sum(len(m['candidates']) for m in menus) != 1635:
        raise ValueError('Expected exactly 1635 menu candidates')
    profiles = defaultdict(set)
    for m in menus:
        profiles[tuple(m['behavior_profile'])].add(m['externality'])
    if len(profiles) != 18 or any(v != {90., 1200., 2800., 4200., 6500.} for v in profiles.values()):
        raise ValueError('Expected 18 profiles x 5 externality weights')
    if sum(m['NC_waiting'] for m in menus) != 60:
        raise ValueError('Expected 60/30 true NC partition')


def load_inputs(inputs):
    inputs = Path(inputs)
    manifest = read(inputs/PARENT/'manifest.json')['files_sha256']
    for name in ('normalized_cases.json', 'selected_training.json', 'summary.json', 'policies.csv',
                 'frozen_inputs.json'):
        if digest(inputs/PARENT/name) != manifest[name]:
            raise ValueError('Parent artifact hash differs: '+name)
    old_freeze = read(inputs/PARENT/'frozen_inputs.json')['inputs_sha256']
    for name in INPUTS:
        if name.startswith(REF+'/') and digest(inputs/name) != old_freeze[name]:
            raise ValueError('Parent raw input hash differs: '+name)
    normalized = read(inputs/PARENT/'normalized_cases.json')
    selection = read(inputs/PARENT/'selected_training.json')
    validate_rules(selection)
    validate_menus(normalized['test'])
    if not selection['training_only'] or selection['selection_stage'] != 'before_waiting_evaluation':
        raise ValueError('Saved selection provenance differs')
    training = normalized['training']
    if len(training) != 450 or len({tuple(m['behavior_profile']) for m in training}) != 90:
        raise ValueError('Original training denominator differs')
    cases = {r['dataclass']['name']:r['dataclass'] for r in read(inputs/REF/'parameters.json')['main']}
    raw = [r for r in read_csv(inputs/REF/'source.csv') if r['cohort'] == 'main']
    index = {(r['case'], float(r['charge'])):r for r in raw}
    if len(raw) != len(index) or len(raw) != 1635 or any(r['accepted'] != 'True' for r in raw):
        raise ValueError('Incomplete/duplicate/rejected raw evaluation rows')
    maximum = defaultdict(float)
    for m in normalized['test']:
        if m['case'] != cases[m['case_id']]:
            raise ValueError('Raw true parameters differ')
        for row in m['candidates']:
            original = index[(m['case_id'], row['charge'])]
            if original['game'] != row['game']:
                raise ValueError('Saved game identity differs')
            for key in ('prices', 'demand'):
                stem = 'price' if key == 'prices' else 'demand'
                close(row[key], [float(original[f'{stem}_{j}']) for j in (1, 2, 3)], 0.)
            rebuilt = direct_state(m['case'], row['prices'], row['rho'], row['wait_minutes'])
            for key in ('TS', 'CS', 'US', 'revenue', 'operating_cost', 'external_cost'):
                close(row[key], float(original[key]), 0.)
                close(row[key], rebuilt[key])
                maximum[key] = max(maximum[key], abs(row[key]-rebuilt[key]))
            close(row['demand'], rebuilt['demand'])
            close(row['served'], sum(row['demand']))
            close(row['rho'], float(original['rho']), 0.)
            close(row['wait_minutes'], float(original['wait_minutes']), 0.)
            close(rebuilt['pressure'], row['rho'], 1e-8)
    failed = [r for r in read_csv(inputs/INITIAL/'waiting_game_status.csv') if r['verified'] != 'True']
    if len(failed) != 23:
        raise ValueError('Historical failure denominator differs')
    return normalized, read(inputs/PARENT/'summary.json'), dict(maximum), failed


def evaluate(menus):
    rows = []
    for menu in menus:
        nc = menu['candidates'][0]
        sd = max(menu['candidates'], key=lambda r:(r['TS'], -r['charge']))
        baseline = {p:project(menu, rule, signal(menu, 'baseline', 0.)['pressure'])['selected']['charge']
                    for p, rule in RULES.items()}
        for name, family, error in scenario_grid():
            s = signal(menu, family, error)
            for policy, rule in RULES.items():
                decision = project(menu, rule, s['pressure'])
                selected = decision['selected']
                rows.append(dict(scenario=name, family=family, error=error, case_id=menu['case_id'],
                    behavior_profile=menu['behavior_profile'], externality=menu['externality'],
                    NC_waiting=menu['NC_waiting'], policy=policy,
                    true_signal=menu['offered_signal'], input_signal=s['pressure'],
                    signal_relative_bias=s['relative_bias'], raw_target=decision['raw_target'],
                    target=decision['target'], charge=selected['charge'], game=selected['game'],
                    prices=selected['prices'], demand=selected['demand'], wait_minutes=selected['wait_minutes'],
                    NC_TS=nc['TS'], SD_TS=sd['TS'], SD_charge=sd['charge'], TS=selected['TS'],
                    gain=selected['TS']-nc['TS'], SD_gain=sd['TS']-nc['TS'],
                    gap=sd['TS']-selected['TS'], changed=selected['charge'] != baseline[policy],
                    potential=menu['potential'], served=selected['served']))
    return rows


def stats(rows):
    sd = math.fsum(r['SD_gain'] for r in rows)
    gain = math.fsum(r['gain'] for r in rows)
    loss = [-r['gain'] for r in rows if r['gain'] < -TOL]
    gap = math.fsum(r['gap'] for r in rows)
    potential = math.fsum(r['potential'] for r in rows)
    served = math.fsum(r['served'] for r in rows)
    return dict(states=len(rows), positive_SD_states=sum(r['SD_gain'] > TOL for r in rows),
        SD_gain=sd, gain=gain, retention=gain/sd if sd > TOL else None,
        loss_states=len(loss), loss_frequency=len(loss)/len(rows) if rows else None,
        loss_total=math.fsum(loss), loss_mean_conditional=math.fsum(loss)/len(loss) if loss else 0.,
        loss_max=max(loss, default=0.), totalgap=gap, meangap=gap/len(rows) if rows else None,
        maxgap=max((r['gap'] for r in rows), default=None),
        changed_states=sum(r['changed'] for r in rows), potential=potential, served=served,
        break_even_per_potential=gap/potential if potential else None,
        break_even_per_served=gap/served if served else None)


def information_requirements():
    return [dict(policy='SD', formula_inputs='full candidate TS menu; one realized state',
        observable='fares, bookings, resource work, realized waiting, capacity and operating accounts',
        estimated='group masses; affinities and outside option; price/time/reliability coefficients; E and costs',
        counterfactual='platform response and demand closure for each candidate charge; TS accounting and finite comparison',
        this_run='1635 saved outcomes read and reconstructed; no new market solve; true-menu oracle only'),
        dict(policy='UT', formula_inputs='one offered NC pressure; frozen threshold and slope',
        observable='effective K and resource coefficients; actual NC fares only if NC observed',
        estimated='group masses; air/outside preferences and sensitivities; reference quality losses',
        counterfactual='NC equilibrium fares if NC unobserved, then offered work at rho=1 and zero access cost; not capped utilization',
        this_run='saved true NC fares; direct MNL, no closure; conditional construction errors only'),
        dict(policy='UTE', formula_inputs='UT pressure plus E/2800; frozen threshold, slope and eta',
        observable='same as UT; E is not generally an observed booking statistic',
        estimated='same as UT plus externality valuation E',
        counterfactual='same as UT; additional E term; training cost already incurred, not zero',
        this_run='same conditional construction errors; E held true; no E uncertainty or net administration-cost estimate')]


def analyze(inputs):
    normalized, parent, audit, failed = load_inputs(inputs)
    menus = normalized['test']
    decisions = evaluate(menus)
    groups = {}
    compact, preference, signals = [], [], []
    for name, family, error in scenario_grid():
        policies = {}
        compact_row = dict(scenario=name, error=error)
        for policy in RULES:
            rows = [r for r in decisions if r['scenario'] == name and r['policy'] == policy]
            policies[policy] = dict(all=stats(rows),
                NC_positive_wait=stats([r for r in rows if r['NC_waiting']]),
                NC_zero_wait=stats([r for r in rows if not r['NC_waiting']]),
                by_E={str(e):stats([r for r in rows if r['externality'] == e])
                      for e in (90., 1200., 2800., 4200., 6500.)})
            for key in ('retention', 'loss_states', 'loss_total', 'loss_mean_conditional',
                        'loss_max', 'maxgap', 'changed_states'):
                compact_row[policy+'_'+key] = policies[policy]['all'][key]
            if policy == 'UT':
                for r in rows:
                    signals.append({k:r[k] for k in ('scenario', 'family', 'error', 'case_id',
                        'behavior_profile', 'externality', 'true_signal', 'input_signal', 'signal_relative_bias')})
        groups[name] = policies
        (preference if family == 'preference_input' else compact).append(compact_row)
    for policy in RULES:
        actual, old = groups['baseline'][policy]['all'], parent['groups']['all'][policy]
        close(actual['retention'], old['retention'], 1e-12)
        close(actual['maxgap'], old['max_absolute_gap'])
        if actual['loss_states'] != old['adverse_states']:
            raise ValueError('Baseline adverse decisions differ from parent')
    summary = dict(schema=1, states=90, candidates=1635, behavior_profiles=18, training_states=450,
        training_profiles=90, case_overlap=len({m['case_id'] for m in menus} &
            {m['case_id'] for m in normalized['training']}), independent_markets=False,
        rules=deepcopy(RULES), scenario_count=13, decision_count=len(decisions),
        historical_failed_attempts=len(failed), scenarios=groups, true_accounting_max_errors=audit,
        new_nash_games=0, new_br_searches=0, new_optimizers=0, new_closure_calls=0,
        inference='Conditional signal-input robustness on 18 shared designed profiles x 5 E, not independent market transfer.',
        preference_scope='Fixed-NC-fare monetary-sensitivity input bias, not identified preference effects or re-equilibrium.',
        SD_scope='True finite-menu oracle; misspecified reoptimized SD not evaluated.',
        administration='SD-rule gap is a break-even implementation budget, not measured net benefit.')
    if summary['case_overlap'] != 90:
        raise ValueError('Training/evaluation overlap differs')
    return dict(summary=summary, decisions=decisions, signals=signals, compact=compact,
                preference=preference, information=information_requirements(), historical_failures=failed)


def markdown(out):
    lines = ['# Task B: conditional information-input sensitivity', '',
        'True TS from 90 complete saved menus (18 profiles x 5 E), 1635 candidates. '
        'All 90 states retained; 60 NC-waiting / 30 NC-zero-wait. No independent-market inference.', '',
        'Loss is TS below own NC by more than 1e-8. Retention uses aggregate gains. '
        'Monetary amounts are model money per period (USD 5 per unit).', '',
        '| Input error | UT retained % | UT losses /90 | UT loss sum / max | UT max SD gap | '
        'UTE retained % | UTE losses /90 | UTE loss sum / max | UTE max SD gap |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def table(row):
        fields = [row['scenario']]
        for p in RULES:
            fields += [f"{100*row[p+'_retention']:.2f}", str(row[p+'_loss_states']),
                f"{row[p+'_loss_total']:.2f} / {row[p+'_loss_max']:.2f}", f"{row[p+'_maxgap']:.2f}"]
        return '| '+' | '.join(fields)+' |'
    lines += [table(r) for r in out['compact']]
    lines += ['', '## Preference-input Appendix', '',
        'Only lambda-input errors at fixed true NC fares. Both air and outside utility are '
        'recomputed. This is not preference re-equilibrium or empirical identification.', '', lines[6], lines[7]]
    lines += [table(r) for r in out['preference']]
    lines += ['', '## Information Requirements', '',
        '| Rule | Formula inputs | Observed / estimated | Counterfactual computation |',
        '|---|---|---|---|']
    lines += ['| '+r['policy']+' | '+r['formula_inputs']+' | '+r['observable']+'; ESTIMATED: '+
              r['estimated']+' | '+r['counterfactual']+' |' for r in out['information']]
    lines += ['', '## Evidence Boundary', '',
        'Demand-pool and capacity changes are conditional analytic MNL/denominator responses; '
        'the six saved demand/capacity pairs change jointly, so saved profiles alone do not '
        'separately identify those effects. The separate saved reliability powers do not change '
        'the reference loss at rho=1. No saved independent lambda variation identifies the '
        'preference effect. signals.csv reports the calculable pressure bias instead.', '',
        'Actual NC utilization and waiting are not the offered-pressure signal. All true fare '
        'responses, quantities and TS remain the original menu outcomes; errors affect only rule '
        'inputs. SD is a true finite-menu oracle, not an error-robust SD experiment.', '',
        'No new optimizer, Nash, best response or demand closure; no refitting. Formula input '
        'count excludes demand estimation, NC counterfactual fare computation when NC is not '
        'observed, and the cost already incurred in rule training. No measured administration '
        'cost is available. Retained gains are not net administrative benefits.', '',
        'The 23 historical failed attempts are copied intact and listed separately; final-menu '
        'coverage is 90/90 and 1635/1635, without dropping negative or zero-gain states.', '',
        'Reproduce: `python -B scripts/run_review13_information.py --replay '
        '--output-dir results/trb_review13_information_v1`', '']
    return '\n'.join(lines)


def artifacts(out):
    return {'summary.json':json_bytes(out['summary']), 'report.md':markdown(out).encode(),
        'decisions.csv':csv_bytes(out['decisions']), 'signals.csv':csv_bytes(out['signals']),
        'compact_summary_9.csv':csv_bytes(out['compact']),
        'preference_input_appendix_4.csv':csv_bytes(out['preference']),
        'information_requirements.csv':csv_bytes(out['information']),
        'historical_failures.csv':csv_bytes(out['historical_failures'])}


def finish_manifest(folder):
    paths = [p for p in sorted(folder.rglob('*')) if p.is_file() and p != folder/'manifest.json']
    dump(folder/'manifest.json', dict(files_sha256={p.relative_to(folder).as_posix():digest(p) for p in paths}))


def verify_manifest(folder):
    manifest = read(folder/'manifest.json')['files_sha256']
    paths = {p.relative_to(folder).as_posix() for p in folder.rglob('*')
             if p.is_file() and p != folder/'manifest.json'}
    if set(manifest) != paths:
        raise ValueError('Frozen hash inventory differs')
    for name, h in manifest.items():
        if digest(folder/name) != h:
            raise ValueError('Frozen hash differs: '+name)
    frozen = read(folder/'frozen_inputs.json')
    if set(frozen['inputs_sha256']) != set(INPUTS) or set(frozen['sources_sha256']) != set(SOURCES):
        raise ValueError('Frozen input/source inventory differs')
    for subdir, key in (('inputs', 'inputs_sha256'), ('execution_source', 'sources_sha256')):
        for name, h in frozen[key].items():
            if digest(folder/subdir/name) != h:
                raise ValueError('Frozen source/input hash differs: '+name)


def run(root, folder=None):
    start = perf_counter()
    root = Path(root)
    folder = root/OUTPUT if folder is None else Path(folder)
    if folder.exists():
        raise FileExistsError('Frozen information output exists; use replay')
    folder.mkdir(parents=True)
    try:
        dump(folder/'design.json', design())
        frozen = {}
        for subdir, key, paths in (('inputs', 'inputs_sha256', INPUTS),
                                   ('execution_source', 'sources_sha256', SOURCES)):
            hashes = {}
            for name in paths:
                hashes[name] = digest(root/name)
                dst = folder/subdir/name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root/name, dst)
                if digest(dst) != hashes[name]:
                    raise ValueError('Snapshot hash differs: '+name)
            frozen[key] = hashes
        dump(folder/'frozen_inputs.json', frozen)
        with arithmetic_guard() as guard:
            out = analyze(folder/'inputs')
        for name, data in artifacts(out).items():
            (folder/name).write_bytes(data)
        for name, h in {**frozen['inputs_sha256'], **frozen['sources_sha256']}.items():
            if digest(root/name) != h:
                raise ValueError('Original input/source changed: '+name)
        elapsed = perf_counter()-start
        dump(folder/'run_report.json', dict(status='complete', elapsed_seconds=elapsed,
            budget_seconds=120., within_budget=elapsed < 120., tripwire=guard,
            input_source_unchanged=True, new_nash_games=0, new_br_searches=0,
            new_optimizers=0, new_closure_calls=0))
        if elapsed >= 120.:
            raise TimeoutError('Frozen 120-second budget exceeded; evidence preserved')
        finish_manifest(folder)
        return out
    except Exception as exc:
        dump(folder/'failure.json', dict(status='failed', error_type=type(exc).__name__,
            error=str(exc), traceback=traceback.format_exc(), elapsed_seconds=perf_counter()-start))
        finish_manifest(folder)
        raise


def replay(folder):
    folder = Path(folder)
    verify_manifest(folder)
    if read(folder/'design.json') != design():
        raise ValueError('Frozen information design differs')
    with arithmetic_guard():
        out = analyze(folder/'inputs')
    for name, data in artifacts(out).items():
        if (folder/name).read_bytes() != data:
            raise ValueError('Rebuilt arithmetic output differs: '+name)
    return out
