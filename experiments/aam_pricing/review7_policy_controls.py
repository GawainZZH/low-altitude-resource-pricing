"""One cached policy-control batch. No engine, closure, BR or Nash evaluation."""
from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
from itertools import product
import json
import math
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

from scripts.replay_review6 import no_new_solving

OUTPUT = 'results/trb_review7_policy_controls_v1'
PARAMS = 'results/trb_review_parameters_v1/full_parameters.json'
SOFT = 'results/comprehensive_state_grid_v1_full/comprehensive_state_grid_charge_search_records.csv'
HARD = 'results/trb_review_hard_capacity_repair_v1/corrected_hard_charge_records.csv'
MIXTURE = 'results/out_of_model_demand_robustness_v1/taste_mixture_charge_records.csv'
RESOURCE = 'results/trb_review6_resources_v1'
HETERO = 'results/trb_review6_heterogeneity_v1'
INPUTS = (PARAMS, SOFT, HARD, MIXTURE, RESOURCE+'/raw.json', RESOURCE+'/design.json',
          RESOURCE+'/summary.json', RESOURCE+'/nc_tie_replay_correction.json',
          HETERO+'/design.json', HETERO+'/games.jsonl', HETERO+'/rescue_games.jsonl',
          HETERO+'/summary.json', HETERO+'/menus.csv', HETERO+'/provenance.json',
          HETERO+'/full_interval_audits.jsonl', HETERO+'/audit_selection.json')
Q2 = tuple(product((0.,8.,16.), repeat=2))
THRESHOLDS = (.90,.95,1.,1.05,1.10,1.15,1.20)
SLOPES = (0.,10.,20.,30.,40.,50.,60.)
TOL = 1e-8


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def json_bytes(data):
    return (json.dumps(data,indent=2,allow_nan=False)+'\n').encode('utf-8')


def write_json(path,data):
    Path(path).write_bytes(json_bytes(data))


def csv_bytes(rows):
    text = io.StringIO(newline='')
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        writer = csv.DictWriter(text,fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list,tuple)) else v
                         for k,v in r.items()} for r in rows)
    return text.getvalue().encode('utf-8')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def distribution(values):
    a = np.asarray(values,dtype=float)
    if not len(a):
        return dict(n=0,strict_positive=0,zeros=0)
    return dict(n=len(a),strict_positive=int(sum(a>TOL)),positive_share=float(np.mean(a>TOL)),
        zeros=int(sum(abs(a)<=TOL)),mean=float(np.mean(a)),minimum=float(min(a)),
        median=float(np.median(a)),p90=float(np.quantile(a,.9)),p95=float(np.quantile(a,.95)),
        p99=float(np.quantile(a,.99)),maximum=float(max(a)))


def resource_controls(rows):
    policies,feasibility,excluded,switches = [],[],[],[]
    for name in sorted({r['profile'] for r in rows}):
        group = [r for r in rows if r['profile']==name]
        if (len(group)!=9 or {tuple(r['charge']) for r in group}!=set(Q2)
                or not all(r['accepted'] for r in group)):
            excluded.append(name)
            continue
        nc = next(r for r in group if sum(r['charge'])==0)
        a,y = np.asarray(nc['A']),np.asarray(nc['demand'])
        if not all(np.array_equal(r['A'],nc['A']) for r in group):
            raise ValueError('Resource consumption changed across tariffs')
        usage = a.T@y
        cap = 16.*float(max(usage))
        u0 = np.asarray(nc['utilization'])
        basis = int(np.flatnonzero(max(u0)-u0<=TOL)[0])
        switch = any((r['utilization'][0]-r['utilization'][1])*(u0[0]-u0[1])<0
                     and abs(r['utilization'][0]-r['utilization'][1])>TOL
                     and abs(u0[0]-u0[1])>TOL for r in group if sum(r['charge'])>0)
        if switch:
            switches.append(name)
        for constraint in ('full_grid','coefficient_sum_16','nc_exposure_cap'):
            allowed = []
            for r in group:
                exposure = float(usage@r['charge'])
                ok = (constraint=='full_grid' or
                      constraint=='coefficient_sum_16' and sum(r['charge'])<=16.+TOL or
                      constraint=='nc_exposure_cap' and exposure<=cap+TOL)
                feasibility.append(dict(profile=name,constraint=constraint,charge=r['charge'],
                    feasible=bool(ok),nc_exposure=exposure,cap=cap if constraint=='nc_exposure_cap' else
                    16. if constraint=='coefficient_sum_16' else None))
                if ok:
                    allowed.append(r)
            axes = [r for r in group if min(r['charge'])==0]
            if not all(r in allowed for r in axes):
                raise ValueError('Controlled vector menu must contain both scalar axes')
            menus = dict(NC=[nc],fixed=[r for r in axes if r['charge'][1]==0],
                nc_adaptive=[r for r in axes if r['charge'][1-basis]==0],best_axis=axes,vector=allowed)
            for objective in ('TS','US'):
                key = objective.lower()
                best = {p:max(g,key=lambda r:(r[key],-sum(r['charge']),tuple(-x for x in r['charge'])))
                        for p,g in menus.items()}
                vector = best['vector'][key]
                for policy,r in best.items():
                    gap = float(vector-r[key])
                    if gap < -TOL:
                        raise ValueError('Scalar escaped controlled vector menu')
                    policies.append(dict(profile=name,constraint=constraint,objective=objective,
                        policy=policy,basis=basis if policy=='nc_adaptive' else 0 if policy=='fixed' else None,
                        charge=r['charge'],value=r[key],gain_vs_nc=r[key]-nc[key],gap_to_vector=gap,
                        gap_per_potential=gap/nc['potential_requests'],nc_exposure=float(usage@r['charge']),
                        vector_feasible_count=len(allowed),switch_subset=bool(switch)))
    summaries = []
    for constraint,objective,policy in product(('full_grid','coefficient_sum_16','nc_exposure_cap'),
                                               ('TS','US'),('fixed','nc_adaptive','best_axis')):
        group = [r for r in policies if (r['constraint'],r['objective'],r['policy'])==
                 (constraint,objective,policy)]
        for subset in ('all','switch','no_switch'):
            chosen = [r for r in group if subset=='all' or r['switch_subset']==(subset=='switch')]
            summaries.append(dict(constraint=constraint,objective=objective,policy=policy,subset=subset,
                gap=distribution([r['gap_to_vector'] for r in chosen]),
                own_nc_gain=distribution([r['gain_vs_nc'] for r in chosen]),
                gap_per_potential=distribution([r['gap_per_potential'] for r in chosen])))
    return dict(policies=policies,feasibility=feasibility,summary=summaries,excluded=excluded,
                switch_profiles=switches,proportional_control=proportional_control())


def proportional_control():
    a,omega = np.array([.5,1.5,1.]),np.array([1.,1.1])
    matrix = np.outer(a,omega)
    error = max(float(np.max(abs(matrix@q-a*float(omega@q)))) for q in Q2)
    return dict(new_games=0,welfare_comparison_available=False,
        maximum_payment_identity_error=error,
        reason='No proportional-A response menu is saved. Changing A while reusing nonproportional '
               'fares/demands would not be a valid equilibrium control. This is an algebra check only.')


def offered_pressure(case, prices):
    """Direct MNL at rho=1, zero access expense; no demand-closure root."""
    mass = np.asarray(case['base_demand'])*case['demand_scale']
    affinity = np.asarray(case['platform_affinity'])+np.eye(len(mass),3)*case['differentiation']
    h1 = .55+1.10*(1.-.88)**2
    loss = np.asarray(case['base_loss'])+np.asarray(case['exposure'])*h1
    air = (affinity-np.asarray(case['price_sensitivity'])[:,None]*np.asarray(prices)
           -np.asarray(case['time_sensitivity'])[:,None]*case['service_time']
           -np.asarray(case['reliability_sensitivity'])[:,None]*loss)/case['logit_scale']
    outside = (np.asarray(case['outside_affinity'])-np.asarray(case['price_sensitivity'])*case['outside_price']
               -np.asarray(case['time_sensitivity'])*case['outside_time'])/case['logit_scale']
    peak = np.maximum(outside,air.max(axis=1))
    weights = np.exp(air-peak[:,None])
    probabilities = weights/(np.exp(outside-peak)+weights.sum(axis=1))[:,None]
    return float((mass@probabilities)@case['platform_resource']/(case['base_capacity']*case['capacity_scale']))


def nearest(candidates,target):
    return min(candidates,key=lambda r:(abs(r['charge']-target),r['charge']))


def tune(menus,signal,thresholds=THRESHOLDS,slopes=SLOPES):
    tuning = []
    for threshold,slope in product(thresholds,slopes):
        chosen = [nearest(m['candidates'],slope*max(0.,next(c for c in m['candidates']
                   if c['charge']==0)[signal]-threshold)) for m in menus]
        tuning.append(dict(threshold=threshold,slope=slope,signal=signal,training_states=len(menus),
            mean_ts=math.fsum(r['ts'] for r in chosen)/len(chosen),
            mean_charge=math.fsum(r['charge'] for r in chosen)/len(chosen)))
    best = max(tuning,key=lambda r:(r['mean_ts'],-r['mean_charge'],-r['slope'],r['threshold']))
    return dict(best,training_candidates=tuning)


def transfer(menus,rules):
    result = []
    for m in menus:
        nc = next(c for c in m['candidates'] if c['charge']==0)
        sd = max(m['candidates'],key=lambda c:(c['ts'],-c['charge']))
        for name,rule in rules.items():
            target = rule['slope']*max(0.,nc[rule['signal']]-rule['threshold'])
            ut = nearest(m['candidates'],target)
            gap = sd['ts']-ut['ts']
            result.append(dict(menu=m['id'],split=m['split'],family=m['family'],rule=name,
                potential=m['mass'],candidate_count=len(m['candidates']),
                nc_signal=nc[rule['signal']],signal=rule['signal'],ut_target=target,
                ut_charge=ut['charge'],projection_distance=abs(ut['charge']-target),sd_charge=sd['charge'],
                nc_ts=nc['ts'],ut_ts=ut['ts'],sd_ts=sd['ts'],ut_gain=ut['ts']-nc['ts'],
                sd_gain=sd['ts']-nc['ts'],sd_minus_ut=gap,
                admin_break_even_total=gap,admin_break_even_per_potential=gap/m['mass'],
                admin_break_even_per_ut_served=gap/ut['served'] if ut['served']>0 else None))
    return result


def transfer_summary(rows):
    groups = sorted({(r['rule'],r['split']) for r in rows})
    groups += [(r,'heldout_all') for r in sorted({r['rule'] for r in rows})
               if any(x['rule']==r and x['split'] in ('offgrid','heterogeneity') for x in rows)]
    summaries = []
    for rule,split in groups:
        group = [r for r in rows if r['rule']==rule and (r['split']==split or
                 split=='heldout_all' and r['split'] in ('offgrid','heterogeneity'))]
        adverse = [max(0.,-r['ut_gain']) for r in group]
        n = len(group)
        summaries.append(dict(rule=rule,split=split,states=n,
            gap=distribution([r['sd_minus_ut'] for r in group]),
            ut_gain=distribution([r['ut_gain'] for r in group]),
            sd_gain=distribution([r['sd_gain'] for r in group]),
            adverse_ut_count=sum(r['ut_gain']<-TOL for r in group),
            adverse_ut_share=sum(r['ut_gain']<-TOL for r in group)/n,
            adverse_loss=distribution(adverse),
            aggregate_admin_break_even_per_potential=math.fsum(r['sd_minus_ut'] for r in group)
                /math.fsum(r['potential'] for r in group),
            admin_break_even_per_potential=distribution([r['admin_break_even_per_potential'] for r in group])))
    return summaries


def native_menus(rows,parameters,split,reference_load=True):
    cases = {c['dataclass']['name']:c['dataclass'] for c in parameters['full_state_grid']}
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['case']].append(r)
    menus = []
    for name, group in sorted(grouped.items()):
        case = cases[name]
        if len({float(r['charge']) for r in group})!=len(group):
            raise ValueError('Duplicate native candidate')
        candidates = []
        for r in sorted(group,key=lambda r:float(r['charge'])):
            prices = [float(r[f'price_{i}']) for i in (1,2,3)]
            demand = [float(r[f'demand_{i}']) for i in (1,2,3)]
            q = float(r['charge'])
            candidates.append(dict(charge=q,ts=float(r['welfare'])+float(np.dot(prices,demand)),
                served=sum(demand),rho=float(r['utilization']),
                offered=offered_pressure(case,prices) if q==0 and reference_load else None))
        if not any(c['charge']==0 for c in candidates):
            raise ValueError('No native NC')
        menus.append(dict(id=name,split=split,family='original',candidates=candidates,
                          mass=sum(case['base_demand'])*case['demand_scale']))
    return menus


def heldout_menus(plan,games,rescues):
    original = {g['game']:g for g in games}
    if len(original)!=len(games) or len({g['game'] for g in rescues})!=len(rescues):
        raise ValueError('Duplicate saved game or rescue')
    for r in rescues:
        if r['game'] not in original or original[r['game']].get('selected'):
            raise ValueError('Unplanned rescue or replacement of accepted response')
    replacement = {r['game']:r for r in rescues if r.get('selected')}
    effective = [replacement.get(g['game'],g) for g in games]
    plans = {m['id']:m for m in plan['menus']}
    grouped = defaultdict(dict)
    for g in effective:
        if g['menu'] not in plans or g['charge'] not in plans[g['menu']]['charges']:
            raise ValueError('Unplanned held-out candidate')
        if g['charge'] in grouped[g['menu']]:
            raise ValueError('Duplicate held-out tariff')
        grouped[g['menu']][g['charge']] = g
    menus, excluded = [],[]
    for name,m in plans.items():
        group = grouped[name]
        missing = [q for q in m['charges'] if q not in group or
                   not group[q].get('selected') or group[q]['selected'].get('verified') is not True]
        if missing:
            excluded.append(dict(menu=name,kind=m['kind'],family=m['family'],
                                 missing_or_failed_charges=missing,planned=len(m['charges'])))
            continue
        candidates = []
        for q in sorted(m['charges']):
            g = group[q]
            s = g['diagnostic']['state']
            check = s['cs']+s['revenue']-s['cost']-s['G']
            if abs(check-s['TS'])>1e-7 or abs(float(np.dot(s['prices'],s['demand']))-s['revenue'])>1e-7:
                raise ValueError('Held-out TS arithmetic changed')
            if not np.allclose(g['selected']['prices'],s['prices'],rtol=0,atol=1e-9):
                raise ValueError('Held-out state does not match accepted fare')
            offered = offered_pressure(m['case'],s['prices']) if q==0 else None
            if q==0 and abs(offered-1.-g['selected']['transition_gap'])>1e-9:
                raise ValueError('Offered load differs from saved no-wait reference')
            candidates.append(dict(charge=float(q),ts=s['TS'],served=s['served'],rho=s['rho'],offered=offered))
        split = 'offgrid' if m['kind']=='offgrid' else 'baseline_control' if m['family']=='baseline' else 'heterogeneity'
        menus.append(dict(id=name,split=split,family=m['family'],candidates=candidates,
                          mass=sum(m['case']['base_demand'])*m['case']['demand_scale']))
    return menus,excluded


def prepare(root,output):
    """Copy immutable inputs and freeze training-only rules before held-out scoring."""
    from scripts.package_review5_revision import dependency_paths
    root,output = Path(root),Path(output)
    if output.exists():
        raise FileExistsError('Batch already prepared; replay it rather than expanding it')
    output.mkdir(parents=True)
    input_hashes = {}
    for name in INPUTS:
        target = output/'saved_inputs'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/name,target)
        input_hashes[name] = digest(target)
        if digest(root/name)!=input_hashes[name]:
            raise ValueError('Input changed during copy: '+name)
    params = read_json(output/'saved_inputs'/PARAMS)
    # Keep the convenient full-parameter alias byte-for-byte, not reserialized.
    shutil.copy2(output/'saved_inputs'/PARAMS,output/'full_parameters.json')
    with no_new_solving() as counts:
        soft = native_menus(read_csv(output/'saved_inputs'/SOFT),params,'training_soft')
        hard = native_menus(read_csv(output/'saved_inputs'/HARD),params,'training_hard')
        if len(soft)!=450 or len(hard)!=450:
            raise ValueError('Training must use exactly the original 450 states')
        rules = dict(legacy_soft_TS=tune(soft,'rho'),common_offered_load_TS=tune(hard,'offered'))
    code = dependency_paths(root,['experiments/aam_pricing/review7_policy_controls.py',
                                  'scripts/run_review7_policy_controls.py',
                                  'tests/test_review7_policy_controls.py'])
    # The guard imports these modules dynamically, outside AST import traversal.
    from scripts.replay_review6 import GUARD_MODULES
    code = dependency_paths(root,[*code,*(f'experiments/aam_pricing/{n}.py' for n in GUARD_MODULES)])
    source_hashes = {}
    for name in code:
        target = output/'code_snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/name,target)
        source_hashes[name] = digest(target)
    design = dict(schema=1,frozen_at=datetime.now(timezone.utc).isoformat(),
        input_hashes=input_hashes,source_hashes=source_hashes,rules=rules,
        training_tripwire=counts,training_only=True,
        resource_price_grid=[list(q) for q in Q2],resource_sum_cap=16.,nc_tie_tolerance=TOL,
        nc_exposure_cap='16 * max_b sum_m A_mb*y_m_NC; fixed before tariff choices',
        thresholds=list(THRESHOLDS),slopes=list(SLOPES),
        projection='nearest saved tariff; exact distance ties choose lower tariff; no interpolation',
        heldout='all complete accepted review6 heterogeneity menus; baseline controls separated',
        policy_gaps='all states including zeros; design frequencies, not operating probabilities',
        proportional='algebra only; no valid saved proportional-A response menu',
        administration='incremental SD-minus-UT break-even thresholds, not monetary cost estimates')
    write_json(output/'frozen_design.json',design)
    return design


def compact(summary):
    lines = ['# Cached Policy Controls','',
        'No new Nash/BR/root/engine calls. Original inputs remain unchanged.', '',
        '## Resource Controls','',
        '| Constraint | Objective | Scalar | n | Strict gaps | Mean gap | Max gap |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in summary['resources']['summary']:
        if r['subset']=='all' and r['constraint']!='full_grid':
            g = r['gap']
            lines.append(f"| {r['constraint']} | {r['objective']} | {r['policy']} | {g['n']} | "
                         f"{g['strict_positive']} | {g.get('mean',0):.6f} | {g.get('maximum',0):.6f} |")
    lines += ['', '## Frozen TS Rule Transfer','',
        '| Rule | Split | n | Strict gaps | Mean SD-UT | Max SD-UT | Adverse UT | Pooled threshold/request |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary['transfer']:
        g = r['gap']
        lines.append(f"| {r['rule']} | {r['split']} | {r['states']} | {g['strict_positive']} | "
                     f"{g['mean']:.6f} | {g['maximum']:.6f} | {r['adverse_ut_count']} | "
                     f"{r['aggregate_admin_break_even_per_potential']:.6f} |")
    lines += ['', 'Coefficient-sum caps use declared resource-unit tariff bases, not equal money. '
        'NC exposure is a frozen hypothetical bill, not realized receipts. Vector weak superiority '
        'follows menu inclusion; strict gaps quantify its size.',
        'Legacy soft UT transfers its served-utilization signal literally, which saturates at one '
        'under binding waiting. The common offered-load rule is separately trained on old hard TS '
        'states at rho=1 and zero access cost. Institutional and parameter changes are not separated.',
        'Administrative thresholds are incremental SD-versus-UT benefits per potential request. '
        'No implementation cost, operating frequency or external-validity estimate is supplied.']
    return '\n'.join(lines)+'\n'


def analyze(output,design):
    data = Path(output)/'saved_inputs'
    params = read_json(data/PARAMS)
    resource = resource_controls(read_json(data/RESOURCE/'raw.json'))
    plan = read_json(data/HETERO/'design.json')
    games = [json.loads(x) for x in (data/HETERO/'games.jsonl').read_text().splitlines()]
    rescues = [json.loads(x) for x in (data/HETERO/'rescue_games.jsonl').read_text().splitlines()]
    heldout,excluded = heldout_menus(plan,games,rescues)
    rules = design['rules']
    transfer_rows = transfer(heldout,rules)
    mixtures = read_csv(data/MIXTURE)
    mixture_menus = []
    for scenario in sorted({r['scenario'] for r in mixtures}):
        values = native_menus([r for r in mixtures if r['scenario']==scenario],params,scenario,reference_load=False)
        mixture_menus.extend(values)
        transfer_rows.extend(transfer(values,{'legacy_soft_TS':rules['legacy_soft_TS']}))
    summary = dict(schema=1,new_nash_games=0,new_br_searches=0,new_root_calls=0,
        resources={k:v for k,v in resource.items() if k not in ('policies','feasibility')},
        training_rules={n:{k:v for k,v in r.items() if k!='training_candidates'} for n,r in rules.items()},
        transfer=transfer_summary(transfer_rows),excluded_menus=excluded,
        complete_menus=len(heldout),heldout_complete_menus=sum(m['split']!='baseline_control' for m in heldout),
        baseline_controls=sum(m['split']=='baseline_control' for m in heldout),
        planned_menus=len(plan['menus']),original_games=len(games),rescue_attempts=len(rescues),
        proportional_control=resource['proportional_control'])
    tuning = [dict(rule=name,**r) for name,rule in rules.items() for r in rule['training_candidates']]
    artifacts = {'resource_policies.csv':csv_bytes(resource['policies']),
        'resource_feasibility.csv':csv_bytes(resource['feasibility']),
        'transfer_states.csv':csv_bytes(transfer_rows),'training_tuning.csv':csv_bytes(tuning),
        'excluded_menus.json':json_bytes(excluded), 'summary.json':json_bytes(summary),
        'normalized_candidates.json':json_bytes(dict(heldout=heldout,mixtures=mixture_menus)),
        'summary.md':compact(summary).encode('utf-8')}
    return summary,artifacts


def replay(output,write=False):
    output = Path(output)
    design = read_json(output/'frozen_design.json')
    for name,expected in design['input_hashes'].items():
        if digest(output/'saved_inputs'/name)!=expected:
            raise ValueError('Saved input hash differs: '+name)
    if PARAMS in design['input_hashes'] and (output/'full_parameters.json').read_bytes()!=(output/'saved_inputs'/PARAMS).read_bytes():
        raise ValueError('Full parameter alias differs from frozen bytes')
    for name,expected in design['source_hashes'].items():
        if digest(output/'code_snapshot'/name)!=expected:
            raise ValueError('Code snapshot hash differs: '+name)
    with no_new_solving() as counts:
        summary,artifacts = analyze(output,design)
    if counts['blocked_calls']:
        raise ValueError('Unexpected replay solver invocation')
    for name,data in artifacts.items():
        target = output/name
        if write:
            target.write_bytes(data)
        elif not target.is_file() or target.read_bytes()!=data:
            raise ValueError('Saved arithmetic output mismatch: '+name)
    return summary


def run(root,output):
    start = perf_counter()
    design = prepare(root,output)
    summary = replay(output,write=True)
    unchanged = all(digest(Path(root)/p)==h for p,h in design['input_hashes'].items())
    if not unchanged:
        raise ValueError('Original input changed during bounded batch')
    write_json(Path(output)/'run_report.json',dict(new_nash_games=0,new_br_searches=0,new_root_calls=0,
        original_hashes_unchanged=True,source_files=len(design['source_hashes']),
        input_files=len(design['input_hashes']),elapsed=perf_counter()-start,
        frozen_design_sha256=digest(Path(output)/'frozen_design.json')))
    return summary
