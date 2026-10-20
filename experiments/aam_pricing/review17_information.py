"""Common-belief policy comparison with fresh forecast NC fares and bounded solves."""
from copy import deepcopy
import ast
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback

import numpy as np

from .config import MarketCase
from .review90_waiting import WaitingEngine
from . import review_hard_capacity_repair as repair
from .review12_signals import offered_pressure
from .review13_information import direct_state

ACCESS = 'results/trb_review6_access_v1/saved_inputs.json'
TRAINING = 'results/trb_review12_signals_v1/selected_training.json'
CHARGES = tuple(range(0,22,2))
RULES = dict(UT=dict(threshold=.9,slope=20.,eta=0.), UTE=dict(threshold=.8,slope=20.,eta=2.))
OWN = ('experiments/aam_pricing/review17_information.py', 'scripts/run_review17_information.py',
       'tests/test_review17_information.py')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dump(path, value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def _append(path, value):
    with Path(path).open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(value,allow_nan=False)+'\n')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _behavior(case):
    return {k:v for k,v in case.items() if k not in ('name','congestion_cost')}


def belief_case(case, axis, error):
    if (axis=='control' and error!=0) or axis not in ('control','demand','lambda'):
        raise ValueError('Frozen belief axis required')
    if axis!='control' and error not in (-.2,.2):
        raise ValueError('Frozen one-factor error required')
    result=deepcopy(case)
    if axis=='demand': result['demand_scale']*=1+error
    if axis=='lambda': result['price_sensitivity']=[x*(1+error) for x in result['price_sensitivity']]
    return result


def design(raw):
    backgrounds=[]
    for label,profile in [('free',(.85,1.12,2.2)),('congested',(1.7,.7,2.2))]:
        c=next(x['dataclass'] for x in raw['records'] if x['dataclass']['congestion_cost']==4200
               and tuple(x['dataclass'][k] for k in ('demand_scale','capacity_scale','reliability_power'))==profile)
        backgrounds.append(dict(id=label,profile=list(profile),case=deepcopy(c)))
    beliefs=[]
    for axis,error in [('control',0.),('demand',-.2),('demand',.2),('lambda',-.2),('lambda',.2)]:
        for b in backgrounds:
            beliefs.append(dict(id=f"{b['id']}__{axis}_{error:+.1f}",background=b['id'],axis=axis,
                                error=error,case=belief_case(b['case'],axis,error)))
    return dict(schema=1,backgrounds=backgrounds,beliefs=beliefs,charges=list(CHARGES),rules=deepcopy(RULES),
        budget_seconds=900,market_seconds=90,game_seconds=7,max_passes=2,
        selection='Ascending continuation; closest verified candidate to input anchor, lexicographic fare tie',
        NC='Fresh solve for every nonzero belief; exact-model caches only for zero-error controls',
        comparison='All policies use the same belief; true full-information optimum is diagnostic only',
        new_global_nash_claim=False,all_equilibria_claim=False)


def ledger(case, prices, rho, wait, charge):
    if not (0<=rho<=1+1e-10 and wait>=0 and all(math.isfinite(x) for x in [rho,wait,charge,*prices])):
        raise ValueError('Invalid saved response state')
    s=direct_state(case,prices,rho,wait)
    K=case['base_capacity']*case['capacity_scale']
    if abs(s['work']/K-rho)>1e-7 or abs(wait*(1-rho))>1e-6:
        raise ValueError('Saved state does not clear the true/belief market')
    costs=np.array(case['base_cost'])+[0.,case['cost_gap'],.5*case['cost_gap']]
    y=np.array(s['demand']); p=np.array(prices)
    profit=((p-costs-charge*np.array(case['platform_resource']))*y).tolist()
    revenue=charge*s['work']
    if abs(s['TS']-(s['CS']+sum(profit)+revenue-s['external_cost']))>1e-7:
        raise ValueError('Transfer ledger mismatch')
    return dict(charge=float(charge),prices=list(prices),rho=float(rho),wait_minutes=float(wait),
        TS=s['TS'],CS=s['CS'],demand=s['demand'],served=float(sum(y)),resource_work=s['work'],
        platform_profit=profit,charge_revenue=revenue,operating_cost=s['operating_cost'],
        external_cost=s['external_cost'],global_nash_claim=False)


def true_menu(raw, case):
    record=next(x['dataclass'] for x in raw['records'] if x['dataclass']['name']==case['name'])
    profile=tuple(case[k] for k in ('demand_scale','capacity_scale','reliability_power'))
    original=next(x['dataclass'] for x in raw['records'] if x['dataclass']['congestion_cost']==90
                  and tuple(x['dataclass'][k] for k in ('demand_scale','capacity_scale','reliability_power'))==profile)
    if _behavior(record)!=_behavior(case) or _behavior(original)!=_behavior(case):
        raise ValueError('True cache behavioral identity differs')
    games=sorted([g for g in raw['games'] if g['mode']=='native' and tuple(g['profile'])==profile],key=lambda g:g['charge'])
    if [g['charge'] for g in games]!=list(CHARGES): raise ValueError('Incomplete exact native cache')
    engine=WaitingEngine(MarketCase(**case)); rows=[]
    for g in games:
        selected=g['selected']
        if not selected or not selected['verified'] or selected['max_gain']>1e-6 or len(selected['br'])!=3:
            raise ValueError('True cache lacks original finite verification')
        p=np.array(selected['prices']); o=engine.outcome(p)
        if not np.allclose(o.demand,g['waiting_outcome']['demand'],atol=1e-7,rtol=1e-10):
            raise ValueError('True cache demand reconstruction differs')
        rows.append(dict(ledger(case,p.tolist(),o.rho,o.wait_minutes,g['charge']),status='accepted',
            source='exact_native_cache_revalued',cache_game=g['game'],finite_check=deepcopy(selected)))
    return rows


def solve_menu(seed, solve_one):
    previous=list(seed); rows=[]
    for q in CHARGES:
        row=solve_one(float(q),list(previous)); rows.append(row)
        if row['status']=='accepted': previous=list(row['prices'])
    return rows


def predict_policies(case, menu):
    if ([r['charge'] for r in menu]!=list(CHARGES)
            or any(r['status']!='accepted' or not math.isfinite(r['TS']) for r in menu)):
        raise ValueError('A complete accepted predicted menu is required')
    nc=menu[0]
    pressure=offered_pressure(case,nc['prices'])
    charges={'SD':max(menu,key=lambda r:(r['TS'],-r['charge']))['charge']}
    targets={}
    for name,rule in RULES.items():
        target=min(20.,max(0.,rule['slope']*max(0.,pressure-rule['threshold'])
                            +rule['eta']*(case['congestion_cost']/2800.-1.)))
        charges[name]=min(CHARGES,key=lambda q:(abs(q-target),q)); targets[name]=target
    return dict(charges=charges,rule_targets=targets,pressure=pressure,belief_nc_prices=nc['prices'],
                belief_nc_TS=nc['TS'],information_set_sha256=hashlib.sha256(
                    json.dumps(case,sort_keys=True).encode()).hexdigest())


def realized_rows(belief_id, choices, truth):
    if [r['charge'] for r in truth]!=list(CHARGES): raise ValueError('Complete common true menu required')
    nc=truth[0]; index={r['charge']:r for r in truth}
    oracle=max(truth,key=lambda r:(r['TS'],-r['charge']))
    rows=[]
    for policy,q in choices['charges'].items():
        actual=index[q]
        rows.append(dict(belief=belief_id,policy=policy,status='evaluated',charge=q,
            predicted_nc_prices=choices['belief_nc_prices'],true_nc_prices=nc['prices'],
            NC_fare_max_error=max(abs(a-b) for a,b in zip(choices['belief_nc_prices'],nc['prices'])),
            pressure=choices['pressure'],information_set_sha256=choices['information_set_sha256'],
            true_nc_TS=nc['TS'],true_TS=actual['TS'],true_gain=actual['TS']-nc['TS'],
            true_prices=actual['prices'],true_wait=actual['wait_minutes'],true_served=actual.get('served'),
            true_full_information_charge_diagnostic=oracle['charge'],
            true_full_information_gain_diagnostic=oracle['TS']-nc['TS'],
            new_global_nash_claim=False))
    return rows


def worker_allowance(remaining): return max(0.,min(90.,remaining))


def budget_remaining(start, now, budget): return max(0.,budget-(now-start))


def _belief_worker(belief, seed, directory, allowance):
    folder=Path(directory); started=perf_counter(); deadline=started+max(0.,allowance-2.)
    budget=repair.EvaluationBudget(limit=250000,deadline=deadline,
        callback=lambda e:_append(folder/'evaluations.jsonl',e))
    engine=WaitingEngine(MarketCase(**belief['case']),budget)
    original_verify=repair.verify_candidate

    def logged_verify(e,p,q,*args,**kwargs):
        label=e.budget.stage
        _append(folder/'attempts.jsonl',dict(event='start',charge=q,stage=label,prices=np.asarray(p).tolist()))
        try:
            result=original_verify(e,p,q,*args,**kwargs)
            _append(folder/'attempts.jsonl',dict(event='complete',charge=q,stage=label,result=result))
            return result
        except Exception as error:
            _append(folder/'attempts.jsonl',dict(event='failure',charge=q,stage=label,error=repr(error)))
            raise

    repair.verify_candidate=logged_verify
    try:
        def one(q,anchor):
            t=perf_counter()
            if t>=deadline:
                row=dict(charge=q,status='not_attempted_market_budget',prices=None,TS=None)
            else:
                budget.deadline=min(deadline,t+7.)
                _append(folder/'game_starts.jsonl',dict(charge=q,seed_prices=anchor,
                                                       elapsed=t-started))
                try:
                    raw=repair.repair_game(engine,anchor,q,belief['id']+f':q{q:g}',max_passes=2)
                    selected=raw['selected']
                    row=dict(charge=q,status='failed_finite_check',prices=None,TS=None,raw=raw)
                    if selected:
                        prices=selected['prices']; o=engine.outcome(np.array(prices))
                        row=dict(ledger(belief['case'],prices,o.rho,o.wait_minutes,q),status='accepted',
                                 source='fresh_belief_solve',raw=raw)
                except Exception as error:
                    row=dict(charge=q,status='failed',prices=None,TS=None,error=repr(error),
                             traceback=traceback.format_exc(),last_stage=budget.stage)
            row.update(seed_prices=anchor,elapsed=perf_counter()-t)
            _dump(folder/f'game_{q:g}.json',row)
            return row
        rows=solve_menu(seed,one)
        _dump(folder/'worker_summary.json',dict(status='finished',elapsed=perf_counter()-started,
            evaluations=budget.used,choice_calls=budget.choice_calls,rows=len(rows)))
    finally:
        repair.verify_candidate=original_verify


def _run_belief(belief, seed, folder, remaining):
    folder.mkdir(parents=True,exist_ok=False)
    allowance=worker_allowance(remaining)
    if allowance<=0: return dict(status='not_attempted_global_budget',rows=[])
    context=mp.get_context('spawn')
    process=context.Process(target=_belief_worker,args=(belief,seed,str(folder),allowance))
    start=perf_counter(); process.start(); process.join(timeout=allowance)
    terminated=process.is_alive()
    if terminated:
        process.terminate(); process.join(timeout=2.)
        if process.is_alive(): process.kill(); process.join()
    rows=[]
    for q in CHARGES:
        path=folder/f'game_{q:g}.json'
        rows.append(_read(path) if path.exists() else dict(charge=float(q),status='not_saved_before_worker_stop',
                                                         prices=None,TS=None))
    return dict(status='worker_time_budget' if terminated else
                ('finished' if process.exitcode==0 else 'worker_failed'),exitcode=process.exitcode,
                elapsed=perf_counter()-start,rows=rows)


def _source_closure(root):
    """Freeze relative/absolute project imports, including imports inside functions."""
    pending=list(OWN); found=set()
    while pending:
        name=pending.pop()
        if name in found or not (root/name).is_file(): continue
        found.add(name)
        parts=Path(name).with_suffix('').parts
        package=list(parts[:-1])
        for node in ast.walk(ast.parse((root/name).read_text(encoding='utf-8-sig'))):
            names=[]
            if isinstance(node,ast.ImportFrom):
                base=package[:len(package)-node.level+1] if node.level else []
                base+=node.module.split('.') if node.module else []
                names=[base]+[base+[a.name] for a in node.names]
            elif isinstance(node,ast.Import): names=[a.name.split('.') for a in node.names]
            for imported in names:
                if not imported or imported[0]!='experiments': continue
                candidate=Path(*imported).with_suffix('.py').as_posix()
                if (root/candidate).is_file(): pending.append(candidate)
    for name in ('experiments/__init__.py','experiments/aam_pricing/__init__.py'):
        if (root/name).exists(): found.add(name)
    return sorted(found)


def analyze(plan, truth, beliefs):
    comparisons=[]; details=[]
    saved_by_id={b['id']:b for b in beliefs}
    if len(saved_by_id)!=len(beliefs) or set(saved_by_id)-{b['id'] for b in plan['beliefs']}:
        raise ValueError('Duplicate or unplanned belief')
    for b in plan['beliefs']:
        saved=saved_by_id.get(b['id'],dict(rows=[]))
        try:
            choices=predict_policies(b['case'],saved['rows'])
            rows=realized_rows(b['id'],choices,truth[b['background']])
            sd=next(r['true_gain'] for r in rows if r['policy']=='SD')
            for row in rows:
                oracle=row['true_full_information_gain_diagnostic']
                row.update(background=b['background'],axis=b['axis'],error=b['error'],
                    gain_retention_vs_common_belief_SD=row['true_gain']/sd if sd>1e-8 else None,
                    gain_retention_vs_true_full_information_SD_diagnostic=(
                        row['true_gain']/oracle if oracle>1e-8 else None))
            comparisons+=rows
            details.append(dict(id=b['id'],status='complete',choices=choices))
        except ValueError as error:
            details.append(dict(id=b['id'],status='incomplete',reason=str(error)))
            for policy in ('SD','UT','UTE'):
                comparisons.append(dict(belief=b['id'],policy=policy,status='incomplete',background=b['background'],
                    axis=b['axis'],error=b['error'],charge=None,true_TS=None,true_gain=None,
                    gain_retention_vs_common_belief_SD=None,
                    gain_retention_vs_true_full_information_SD_diagnostic=None))
    summaries=[]
    for axis,error in [('control',0.),('demand',-.2),('demand',.2),('lambda',-.2),('lambda',.2)]:
        for policy in ('SD','UT','UTE'):
            group=[r for r in comparisons if r['axis']==axis and r['error']==error and r['policy']==policy]
            valid=[r for r in group if r['status']=='evaluated']
            sd_group=[r for r in comparisons if r['axis']==axis and r['error']==error and r['policy']=='SD']
            sd_total=math.fsum(r['true_gain'] for r in sd_group) if all(
                r['status']=='evaluated' for r in sd_group) else None
            summaries.append(dict(axis=axis,error=error,policy=policy,planned=len(group),evaluated=len(valid),
                sum_true_gain=math.fsum(r['true_gain'] for r in valid) if len(valid)==len(group) else None,
                aggregate_retention_vs_common_belief_SD=(math.fsum(r['true_gain'] for r in valid)/sd_total
                    if len(valid)==len(group) and sd_total is not None and sd_total>1e-8 else None),
                negative_gain_cases=sum(r['true_gain'] < -1e-8 for r in valid)))
    return dict(comparisons=comparisons,belief_decisions=details,compact=summaries,
        summary=dict(planned_beliefs=10,complete_beliefs=sum(x['status']=='complete' for x in details),
                     incomplete_beliefs=sum(x['status']!='complete' for x in details),
                     evaluated_policy_rows=sum(r['status']=='evaluated' for r in comparisons),
                     negative_policy_gains=sum(r.get('true_gain') is not None and r['true_gain'] < -1e-8 for r in comparisons),
                     common_information=True,true_NC_fares_held_fixed_in_forecast=False,
                     new_global_nash_claim=False))


def _write_analysis(folder, result):
    for name,value in result.items(): _dump(folder/(name+'.json'),value)
    columns=list(dict.fromkeys(k for row in result['comparisons'] for k in row))
    with (folder/'comparisons.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns); writer.writeheader()
        for row in result['comparisons']:
            writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def run(root, folder, budget_seconds=900.):
    if not 0<budget_seconds<=900: raise ValueError('Budget must be positive and at most 900 seconds')
    root,folder=Path(root),Path(folder); folder.mkdir(parents=True,exist_ok=False)
    start=perf_counter(); data=_read(root/ACCESS); trained=_read(root/TRAINING)
    if {k:trained[k] for k in RULES}!=RULES: raise ValueError('Frozen training rules differ')
    plan=design(data); plan['budget_seconds']=budget_seconds
    source_files=_source_closure(root)
    (folder/'inputs').mkdir()
    for src,name in ((ACCESS,'saved_inputs.json'),(TRAINING,'selected_training.json')):
        shutil.copyfile(root/src,folder/'inputs'/name)
    for name in source_files:
        target=folder/'execution_source'/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/name,target)
    plan.update(source_sha256={n:_sha(root/n) for n in source_files},
                data_sha256={ACCESS:_sha(root/ACCESS),TRAINING:_sha(root/TRAINING)},python=sys.executable)
    _dump(folder/'design.json',plan)
    truth={b['id']:true_menu(data,b['case']) for b in plan['backgrounds']}
    _dump(folder/'true_menus.json',truth)
    beliefs=[]
    for b in plan['beliefs']:
        remaining=budget_remaining(start,perf_counter(),max(0.,budget_seconds-3.))
        if b['axis']=='control':
            result=dict(status='exact_zero_error_cache',rows=deepcopy(truth[b['background']]),elapsed=0.)
        elif remaining<=0:
            result=dict(status='not_attempted_global_budget',rows=[])
        else:
            for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
            result=_run_belief(b,truth[b['background']][0]['prices'],folder/'beliefs'/b['id'],remaining)
        result['id']=b['id']; beliefs.append(result); _dump(folder/'belief_menus.json',beliefs)
        print(json.dumps(dict(belief=b['id'],status=result['status'],
            accepted=sum(r['status']=='accepted' for r in result['rows']),elapsed=round(perf_counter()-start,2))),flush=True)
    result=analyze(plan,truth,beliefs); _write_analysis(folder,result)
    execution=dict(wall_seconds=perf_counter()-start,budget_seconds=budget_seconds,
        fresh_prediction_NC_attempts=sum(any(r['charge']==0 and r['status'] not in ('not_saved_before_worker_stop','not_attempted_market_budget') for r in x['rows'])
            for x in beliefs if not x['id'].endswith('control_+0.0')),
        fresh_accepted_games=sum(r['status']=='accepted' for x in beliefs if x['status']!='exact_zero_error_cache' for r in x['rows']),
        fresh_accepted_prediction_NC=sum(any(r['charge']==0 and r['status']=='accepted' for r in x['rows'])
            for x in beliefs if x['status']!='exact_zero_error_cache'),
        fresh_game_status_counts={status:sum(r['status']==status for x in beliefs
            if x['status']!='exact_zero_error_cache' for r in x['rows']) for status in sorted(
                {r['status'] for x in beliefs if x['status']!='exact_zero_error_cache' for r in x['rows']})},
        exact_true_cache_games=22,selective_reruns=0)
    _dump(folder/'execution.json',execution)
    _dump(folder/'manifest.json',dict(files={p.relative_to(folder).as_posix():_sha(p) for p in sorted(folder.rglob('*'))
                                           if p.is_file()}))
    return dict(result['summary'],execution=execution)


def rebuild_menu(case, rows):
    rebuilt=[]
    for saved in rows:
        row=deepcopy(saved)
        if row['status']=='accepted':
            fresh=ledger(case,row['prices'],row['rho'],row['wait_minutes'],row['charge'])
            for k,value in fresh.items():
                if isinstance(value,(float,list)) and not np.allclose(value,row[k],atol=1e-8,rtol=1e-12):
                    raise ValueError('Saved ledger differs: '+k)
            row.update(fresh)
        rebuilt.append(row)
    return rebuilt


def replay(folder):
    """Read-only, original-utility arithmetic; no optimizer or scalar root calls."""
    folder=Path(folder)
    for name,digest in _read(folder/'manifest.json')['files'].items():
        if _sha(folder/name)!=digest: raise ValueError('Artifact hash differs: '+name)
    plan=_read(folder/'design.json')
    for name,digest in plan['source_sha256'].items():
        if _sha(folder/'execution_source'/name)!=digest:
            raise ValueError('Frozen source hash differs: '+name)
    for name,digest in plan['data_sha256'].items():
        if _sha(folder/'inputs'/Path(name).name)!=digest:
            raise ValueError('Frozen input hash differs: '+name)
    truth={b['id']:rebuild_menu(b['case'],_read(folder/'true_menus.json')[b['id']]) for b in plan['backgrounds']}
    specs={b['id']:b for b in plan['beliefs']}; beliefs=_read(folder/'belief_menus.json')
    rebuilt=[dict(x,rows=rebuild_menu(specs[x['id']]['case'],x['rows'])) for x in beliefs]
    result=analyze(plan,truth,rebuilt)
    for name,value in result.items():
        if value!=_read(folder/(name+'.json')): raise ValueError('Policy arithmetic replay differs: '+name)
    return dict(result['summary'],execution=_read(folder/'execution.json'),
                replay_new_Nash_solves=0,replay_new_scalar_roots=0)
