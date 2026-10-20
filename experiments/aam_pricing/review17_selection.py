"""Candidate selection sensitivity with exact historical full-bound matching.

Replay is standard-library arithmetic only. The bounded discovery engine is
imported inside discover(), never inside the saved-candidate appraisal path.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from decimal import Decimal
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

from .review10_selection import restore_state

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = 'results/trb_review17_selection_v1'
BOUNDS = 'results/trb_review16_equilibrium_full_v1'
PARAMS = 'results/trb_review_parameters_v1/full_parameters.json'
WAITING = 'results/trb_review90_waiting_v1/waiting_games.jsonl'
INITIAL = 'results/trb_review90_waiting_v1/initial_attempt/waiting_games.jsonl'
REFINE = 'results/trb_review5_refinement_v1/new_games.jsonl'
NEAR = 'results/trb_review11_selection_v1/diagnostics.json'
BRANCH = 'results/trb_review6_branches_v1/attempts.jsonl'
NORMAL = 'results/trb_review12_signals_v1/normalized_cases.json'
UT = 'results/trb_review12_signals_v1/policies.csv'
OWNED = ('experiments/aam_pricing/review17_selection.py','scripts/run_review17_selection.py',
         'tests/test_review17_selection.py','experiments/__init__.py',
         'experiments/aam_pricing/__init__.py','experiments/aam_pricing/review10_selection.py')
ENGINE_SOURCES = ('experiments/aam_pricing/review90_waiting.py',
                  'experiments/aam_pricing/review_hard_capacity_repair.py',
                  'experiments/aam_pricing/review_equilibrium.py','experiments/aam_pricing/config.py')
TOL = 1e-7


def design():
    return dict(profiles=[[.85,1.12,2.2],[1.35,.9,2.2],[1.7,.7,2.2]],
                modes=['native','equal_vot'],charges=list(range(0,22,2)),starts=[12.,22.,36.],
                max_starts=198,wall_seconds=600,kappas=[0.,.5,1.],
                rules=['pessimistic','optimistic'],scopes=['all_accepted','full_bound_exact_key'],
                complete_Nash_set=False,full_bound_leaf_replay=False,
                no_retry=True,policy_tolerance=TOL,
                acceptance='Existing finite41/transition/KKT criteria; not a full-domain bound',
                criterion='Pointwise welfare min/max; same criterion at NC and positive tariffs',
                execution='historical-only; discovery disabled; legacy start grid not executed')


def read(path):
    p=Path(path)
    with p.open(encoding='utf-8-sig',newline='') as f:
        if p.suffix=='.jsonl': return [json.loads(x) for x in f if x.strip()]
        if p.suffix=='.csv': return list(csv.DictReader(f))
        return json.load(f)


def dump(path,data,exclusive=False):
    with Path(path).open('x' if exclusive else 'w',encoding='utf-8') as f:
        json.dump(data,f,indent=2,allow_nan=False)
        f.write('\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def behavior_key(row):
    c={k:v for k,v in row['case'].items() if k not in ('name','congestion_cost')}
    return hashlib.sha256(json.dumps([c,row['prices'],row['charge'],row['excluded'],
        row['waiting_coefficients']],sort_keys=True).encode()).hexdigest()


def close(a,b,label,tol=1e-6):
    if isinstance(a,(list,tuple)):
        if len(a)!=len(b): raise ValueError('Dimension mismatch: '+label)
        for x,y in zip(a,b): close(x,y,label,tol)
    elif not math.isfinite(a) or not math.isfinite(b) or abs(a-b)>tol:
        raise ValueError('Saved arithmetic mismatch: '+label)


def validate_bound_index(rows):
    if len(rows)!=441 or len({r['key'] for r in rows})!=441:
        raise ValueError('Historical 441 full-bound population differs')
    for r in rows:
        if (behavior_key(r['input'])!=r['key']
                or behavior_key(r['historical_response_input'])!=r['key'] or not r['all_closed']):
            raise ValueError('Full-bound exact identity or closure differs')
        active=set(range(3))-set(r['input']['excluded'])
        if {x['platform'] for x in r['coordinates']}!=active or len(r['coordinates'])!=len(active):
            raise ValueError('Full-bound coordinate coverage differs')
        for c in r['coordinates']:
            if (not c['closed'] or not c['coverage_complete']
                or Decimal(c['gain_upper'])>Decimal('.01')
                or Decimal(c['normalized_upper'])>Decimal('.00001')):
                raise ValueError('Historical full-bound threshold failed')
            if c['prices']!=r['input']['prices'] or c['charge']!=r['input']['charge']:
                raise ValueError('Bound coordinates refer to another response')
    if sum(len(r['coordinates']) for r in rows)!=1309:
        raise ValueError('Historical coordinate total differs')
    return True


def build_bound_index(root):
    root=Path(root); inputs=read(root/BOUNDS/'inputs.json')
    prior=read(root/BOUNDS/'replay_verified.json')
    if not (prior['responses']==prior['closed_responses']==441 and prior['coordinates']==1309
            and prior['complete_domain'] and prior['population_mapping'] and prior['no_profitable_entry']):
        raise ValueError('Historical replay metadata incomplete')
    reports={r['id']:r for r in prior['rows']}
    rows=[]
    for key,group in inputs['groups'].items():
        p=root/BOUNDS/'responses'/(key+'.json')
        saved=read(p)
        report=reports.get(group['input']['id'],reports.get(saved['input']['id']))
        if (behavior_key(saved['input'])!=key or behavior_key(group['input'])!=key
                or not report or not report['all_closed']):
            raise ValueError('Historical input/replay linkage differs')
        # Deliberately do not evaluate, iterate, or copy the saved leaf objects.
        coords=[{k:v for k,v in c.items() if k!='leaves'} for c in saved['coordinates']]
        rows.append(dict(key=key,input=group['input'],aliases=group['aliases'],
                         coordinates=coords,all_closed=saved['all_closed'],
                         historical_response_input=saved['input'],
                         historical_response_file=p.relative_to(root).as_posix(),response_sha256=sha(p)))
    validate_bound_index(rows)
    return rows


def accepted(c):
    if c.get('verified') is not True: return False
    if c.get('first_order_ok') is not True: raise ValueError('Acceptance contradicts first-order rejection')
    near=abs(c['transition_gap'])<=1e-10
    if c['near_transition']!=near: raise ValueError('Near-transition flag differs')
    if near:
        sides=[max(-l,0.) if p>5+1e-8 else 0. for p,l in zip(c['prices'],c['left_gradient'])]
        sides += [max(r,0.) if p<44-1e-8 else 0. for p,r in zip(c['prices'],c['right_gradient'])]
        if max(sides)>1e-5: raise ValueError('One-sided acceptance witness failed')
    elif c['kkt']>1e-6: raise ValueError('KKT acceptance witness failed')
    if len(c['br'])!=3 or {r['platform'] for r in c['br']}!={0,1,2}:
        raise ValueError('Missing finite deviation witnesses')
    gains=[]
    for b in c['br']:
        best=max(x['profit'] for x in b['trace'])
        close(best,b['best_profit'],'best saved deviation',1e-7)
        gain=max(0.,best-b['baseline'])
        close(gain,b['gain'],'coordinate gain',1e-7)
        gains.append(gain)
    close(max(gains),c['max_gain'],'maximum gain',1e-7)
    if max(gains)>1e-6: raise ValueError('Finite deviation acceptance failed')
    return True


def compact_candidate(c):
    names=('prices','verified','first_order_ok','near_transition','transition_gap','kkt',
           'left_gradient','right_gradient','max_gain','verification_type','candidate_source')
    out={k:c[k] for k in names if k in c}
    out['br']=[dict(platform=b['platform'],baseline=b['baseline'],best_profit=b['best_profit'],gain=b['gain'],
                    trace=[{k:t[k] for k in ('fare','profit') if k in t} for t in b['trace']]) for b in c.get('br',[])]
    return out


def effective(c):
    return dict(D=[d*c['demand_scale'] for d in c['base_demand']],
        alpha=[[v+(c['differentiation'] if g==j else 0.) for j,v in enumerate(row)]
               for g,row in enumerate(c['platform_affinity'])],alpha_outside=c['outside_affinity'],
        beta=c['time_sensitivity'],gamma=c['reliability_sensitivity'],mu=c['logit_scale'],
        c=[v+(0. if j==0 else c['cost_gap'] if j==1 else .5*c['cost_gap']) for j,v in enumerate(c['base_cost'])],
        chi=c['exposure'],ell=c['base_loss'],a=c['platform_resource'],K=c['base_capacity']*c['capacity_scale'],
        service_time=c['service_time'],outside_price=c['outside_price'],outside_time=c['outside_time'],
        **{'lambda':c['price_sensitivity']})


def waiting_coefficients(c,mode):
    e=effective(c)
    v=sum(d*b for d,b in zip(e['D'],e['beta']))/sum(d*l for d,l in zip(e['D'],e['lambda']))
    return [l*v for l in e['lambda']] if mode=='equal_vot' else list(e['beta'])


def response_input(c,mode,q,prices):
    return dict(case=c,prices=list(prices),charge=float(q),excluded=[],
                waiting_coefficients=waiting_coefficients(c,mode))


def market_key(c,mode):
    return behavior_key(response_input(c,mode,0.,[]))


def reconstruct(row):
    c,p,q=row['case'],row['candidate']['prices'],row['charge']; e=effective(c)
    if 'saved_outcome' in row:
        rho,w=row['saved_outcome']['rho'],row['saved_outcome']['wait_minutes']
    else:
        rebuilt=restore_state(e,row['mode'],c['reliability_power'],p,q,
                              [b['baseline'] for b in sorted(row['candidate']['br'],key=lambda b:b['platform'])])
        if rebuilt['state_status']!='valid': raise ValueError('Saved-state recovery: '+rebuilt['error'])
        rho,w=rebuilt['state']['rho'],rebuilt['state']['wait_minutes']
    if not 0<=rho<=1 or w<0: raise ValueError('Invalid saved utilization or waiting')
    h=.55*rho**c['reliability_power']+1.1*max(rho-.88,0.)**2
    prob=[]; group_cs=[]
    beta=waiting_coefficients(c,row['mode'])
    for g in range(3):
        air=[(e['alpha'][g][j]-e['lambda'][g]*p[j]-e['beta'][g]*e['service_time'][j]
            -e['gamma'][g]*(e['ell'][j]+e['chi'][j]*h)-beta[g]*w)/e['mu'] for j in range(3)]
        outside=(e['alpha_outside'][g]-e['lambda'][g]*e['outside_price']-e['beta'][g]*e['outside_time'])/e['mu']
        peak=max(air+[outside]); den=peak+math.log(sum(math.exp(v-peak) for v in air+[outside]))
        prob.append([math.exp(v-den) for v in air]); group_cs.append(e['D'][g]*e['mu']/e['lambda'][g]*den)
    y=[sum(e['D'][g]*prob[g][j] for g in range(3)) for j in range(3)]
    work=sum(a*v for a,v in zip(e['a'],y)); close(work/e['K'],rho,'resource closure',1e-8)
    close(w*(1-rho),0.,'complementarity',1e-8)
    for b in row['candidate']['br']:
        j=b['platform']; close((p[j]-e['c'][j]-q*e['a'][j])*y[j],b['baseline'],'profit baseline')
    v=sum(d*b for d,b in zip(e['D'],e['beta']))/sum(d*l for d,l in zip(e['D'],e['lambda']))
    return dict(base_TS=sum(group_cs)+sum((x-z)*yy for x,z,yy in zip(p,e['c'],y)),
                damage=.15*rho*rho+max(rho-.9,0.)**3,ground_payment=e['outside_price']*(sum(e['D'])-sum(y)),
                payment_receipts=v*w*sum(y),rho=rho,wait_minutes=w,passengers=sum(y),
                group_CS=group_cs,demand=y)


def collect_saved(root):
    root=Path(root); normal=read(root/NORMAL)['test']; policy_rows=read(root/UT)
    cases={tuple(m['behavior_profile']):m['case'] for m in normal if m['externality']==90}
    records=[]
    def add(c,mode,q,candidate,source,outcome=None):
        if mode not in ('native','equal_vot'): return
        row=dict(case=c,mode=mode,charge=float(q),candidate=compact_candidate(candidate),source=source)
        if outcome is not None: row['saved_outcome']={k:outcome[k] for k in ('rho','wait_minutes')}
        records.append(row)
    for name in (WAITING,INITIAL,REFINE):
        for i,g in enumerate(read(root/name)):
            mode=g.get('mode','native'); profile=tuple(g['profile'])
            if profile not in cases: continue
            discoveries=list(enumerate(g.get('attempts',[])))
            discoveries += [('audited:'+str(j),a) for j,a in enumerate(g.get('audited_candidates',[]))]
            for j,a in discoveries:
                outcome=g.get('waiting_outcome') if g.get('selected') and a['prices']==g['selected']['prices'] else None
                add(cases[profile],mode,g['charge'],a,dict(file=name,row=i,attempt=j,game=g['game']),outcome)
            if g.get('selected'):
                add(cases[profile],mode,g['charge'],g['selected'],dict(file=name,row=i,selected=True,game=g['game']),g.get('waiting_outcome'))
    near=read(root/NEAR)
    roles={'shortest':(1.35,.9,1.7),'longest':(1.7,.7,1.7)}
    for i,r in enumerate(read(root/BRANCH)):
        if r['status']=='started': continue
        c=cases[roles[r['role']]]
        if 'prices' in r:
            outcome=dict(rho=r['rho'],wait_minutes=r['wait']) if 'rho' in r else None
            add(c,'native',r['q'],r,dict(file=BRANCH,row=i,attempt=r['attempt'],role=r['role']),outcome)
        else:
            records.append(dict(case=c,mode='native',charge=float(r['q']),
                candidate=dict(verified=False),source=dict(file=BRANCH,row=i,attempt=r['attempt']),error=r.get('error',r['status'])))
    policies={r['menu']:r for r in policy_rows}
    scenarios=[]
    for m in normal:
        c=m['case']; p=policies[m['id']]
        scenarios.append(dict(id=m['id'],family='native_UT_SD',case=c,mode='native',institution='native_waiting',
                              charges=[r['charge'] for r in m['candidates']],UT_charge=float(p['UT_charge']),
                              original_SD=float(p['SD_TS']),original_NC=float(p['NC_TS']),
                              selected=[dict(charge=r['charge'],prices=r['prices']) for r in m['candidates']]))
    for profile in design()['profiles']:
        for E in (90.,1200.,2800.,4200.,6500.):
            for mode,inst in (('native','native_waiting'),('equal_vot','payment')):
                c=deepcopy(cases[tuple(profile)]); c['congestion_cost']=E
                scenarios.append(dict(id=f'figure4:{profile}:{E}:{inst}',family='figure4',case=c,mode=mode,
                    institution=inst,charges=[float(q) for q in design()['charges']],UT_charge=None))
    for role,profile in roles.items():
        c=deepcopy(cases[profile]); c['congestion_cost']=2800.
        scenarios.append(dict(id='near_release:'+role,family='near_release',case=c,mode='native',
            institution='native_waiting',charges=sorted({r['charge'] for r in near['candidates'] if r['menu']==role}),UT_charge=None))
    return dict(records=records,scenarios=scenarios,cases=list(cases.values()),near_reference=near['summary'])


def policy(by_q,rule,scope,ut_charge=None):
    if rule not in ('pessimistic','optimistic') or 0. not in by_q: raise ValueError('Unknown rule or no NC')
    missing=[q for q,rows in sorted(by_q.items()) if not rows]
    if missing: return dict(status='incomplete',scope=scope,rule=rule,missing_charges=missing,SD_charge=None)
    chosen={q:min(rows,key=lambda r:((r['W'] if rule=='pessimistic' else -r['W']),tuple(r['prices']),r['key']))
            for q,rows in by_q.items()}
    best=max(r['W'] for r in chosen.values())
    qs=sorted(q for q,r in chosen.items() if r['W']>=best-TOL); q=qs[0]
    nc=chosen[0.]['W']; lo={q:min(r['W'] for r in rows) for q,rows in by_q.items()}
    hi={q:max(r['W'] for r in rows) for q,rows in by_q.items()}
    possible=sorted(q for q in by_q if hi[q]>=max(lo.values())-TOL)
    return dict(status='complete',scope=scope,rule=rule,missing_charges=[],NC=nc,NC_key=chosen[0.]['key'],
        SD_charge=q,optimal_charges=qs,SD_TS=chosen[q]['W'],SD_key=chosen[q]['key'],SD_gain=chosen[q]['W']-nc,
        UT_charge=ut_charge,UT_TS=chosen[ut_charge]['W'] if ut_charge is not None else None,
        UT_gain=chosen[ut_charge]['W']-nc if ut_charge is not None else None,
        possible_optimal_charges=possible,possible_charge_interval=[min(possible),max(possible)],
        maximum_candidate_spread=max(hi[q]-lo[q] for q in by_q),
        minimum_candidate_TS=min(lo.values()),maximum_candidate_TS=max(hi.values()))


def candidate_pool(records,index):
    registry={r['key']:r for r in index}; pool={}; excluded=[]
    for i,row in enumerate(records):
        c=row['candidate']
        if not accepted(c):
            excluded.append(dict(record=i,source=row['source'],reason=row.get('error','not accepted by original finite criteria'),
                                 verified=c.get('verified',False),prices=c.get('prices'),charge=row['charge'],mode=row['mode']))
            continue
        inp=response_input(row['case'],row['mode'],row['charge'],c['prices']); key=behavior_key(inp)
        if key in pool:
            pool[key]['sources'].append(row['source']); continue
        try: state=reconstruct(row)
        except ValueError as e:
            excluded.append(dict(record=i,source=row['source'],reason='state reconstruction unresolved: '+str(e),
                                 verified=True,prices=c['prices'],charge=row['charge'],mode=row['mode']))
            continue
        pool[key]=dict(key=key,input=inp,mode=row['mode'],market=market_key(row['case'],row['mode']),
            charge=row['charge'],prices=c['prices'],state=state,sources=[row['source']],
            level='full_bound_exact_key' if key in registry else 'accepted_finite_deviation',
            full_bound_match=key if key in registry else None)
    return list(pool.values()),excluded


def aggregate_policies(rows):
    groups=defaultdict(list)
    for r in rows: groups[r['family'],r['scope'],r['rule'],r['kappa']].append(r)
    result=[]
    for (family,scope,rule,k),group in sorted(groups.items()):
        valid=[r for r in group if r['status']=='complete']
        base=dict(family=family,scope=scope,rule=rule,kappa=k,states=len(group),complete=len(valid))
        if len(valid)!=len(group):
            result.append(dict(base,status='incomplete')); continue
        if family=='figure4':
            w=[r for r in valid if r['institution']=='native_waiting']
            p=[r for r in valid if r['institution']=='payment']
            wg=sum(r['SD_gain'] for r in w)/len(w); pg=sum(r['SD_gain'] for r in p)/len(p)
            result.append(dict(base,status='complete',waiting_NC=sum(r['NC'] for r in w)/len(w),
                payment_NC=sum(r['NC'] for r in p)/len(p),waiting_SD=sum(r['SD_TS'] for r in w)/len(w),
                payment_SD=sum(r['SD_TS'] for r in p)/len(p),waiting_gain=wg,payment_gain=pg,
                reduction_percent=100*(1-pg/wg) if wg>TOL else None))
        else:
            sd=sum(r['SD_gain'] for r in valid)
            ut=sum(r['UT_gain'] for r in valid) if family=='native_UT_SD' else None
            result.append(dict(base,status='complete',total_SD_gain=sd,total_UT_gain=ut,
                UT_retention_percent=100*ut/sd if ut is not None and sd>TOL else None,
                negative_UT_states=sum(r['UT_gain']<-TOL for r in valid) if ut is not None else None,
                minimum_NC=min(r['NC'] for r in valid),maximum_NC=max(r['NC'] for r in valid)))
    return result


def analyze(payload,new_records,index):
    validate_bound_index(index)
    pool,excluded=candidate_pool(payload['records']+new_records,index)
    grouped=defaultdict(list)
    for r in pool: grouped[r['market'],r['charge']].append(r)
    appraisals=[]; envelopes=[]; coverage=[]
    for s in payload['scenarios']:
        market=market_key(s['case'],s['mode']); E=s['case']['congestion_cost']
        for k in design()['kappas']:
            by_q={q:[dict(key=r['key'],prices=r['prices'],level=r['level'],
                W=r['state']['base_TS']-E*r['state']['damage']+k*r['state']['ground_payment']
                  +(r['state']['payment_receipts'] if s['institution']=='payment' else 0.))
                for r in grouped[market,q]] for q in s['charges']}
            for scope in design()['scopes']:
                selected={q:[r for r in rows if scope=='all_accepted' or r['level']=='full_bound_exact_key']
                          for q,rows in by_q.items()}
                for q,rows in selected.items():
                    values=[r['W'] for r in rows]
                    envelopes.append(dict(scenario=s['id'],family=s['family'],institution=s['institution'],
                        profile=[s['case'][x] for x in ('demand_scale','capacity_scale','reliability_power')],
                        externality=E,kappa=k,scope=scope,charge=q,candidates=len(rows),
                        min_TS=min(values) if values else None,max_TS=max(values) if values else None,
                        welfare_spread=max(values)-min(values) if values else None,keys=[r['key'] for r in rows]))
                for rule in design()['rules']:
                    r=policy(selected,rule,scope,s['UT_charge'])
                    appraisals.append(dict(scenario=s['id'],family=s['family'],institution=s['institution'],
                        profile=[s['case'][x] for x in ('demand_scale','capacity_scale','reliability_power')],
                        externality=E,kappa=k,**r))
            if k==0.:
                coverage.append(dict(scenario=s['id'],family=s['family'],charges=len(by_q),
                    accepted_vectors=sum(len(v) for v in by_q.values()),
                    full_bound_vectors=sum(r['level']=='full_bound_exact_key' for v in by_q.values() for r in v),
                    multi_candidate_charges=sum(len(v)>1 for v in by_q.values())))
                if 'selected' in s:
                    original=[]
                    for r in s['selected']:
                        key=behavior_key(response_input(s['case'],s['mode'],r['charge'],r['prices']))
                        matches=[v for v in by_q[r['charge']] if v['key']==key]
                        if len(matches)!=1: raise ValueError('Normalized selected response is not an exactly accepted candidate')
                        original.append(matches[0]['W'])
                    close(original[0],s['original_NC'],'original native NC',1e-5)
                    close(max(original),s['original_SD'],'original native SD',1e-5)
    summary=dict(historical_full_bounds=len(index),historical_coordinates=1309,
        raw_candidate_records=len(payload['records'])+len(new_records),unique_accepted_vectors=len(pool),
        accepted_levels=dict(Counter(r['level'] for r in pool)),excluded_records=len(excluded),
        scenarios=len(payload['scenarios']),complete_Nash_set=False,
        new_bound_computations=0,full_bound_leaves_recomputed=0,new_Nash_solves=0,
        new_BR_searches=0,new_scalar_roots=0)
    keys={r['key'] for r in pool}
    matches=[dict(key=r['key'],historical_aliases=r['aliases'],
        historical_response_id=r['historical_response_input']['id'],coordinates=len(r['coordinates']),
        exact_target_candidate=r['key'] in keys,
        scope='matched_target' if r['key'] in keys else 'outside_target_candidate_population',
        historical_response_sha256=r['response_sha256']) for r in index]
    policy_mapping=[dict(alias=a,**r) for r in matches for a in r['historical_aliases'] if a.startswith('policy:')]
    if len(policy_mapping)!=327 or not all(r['exact_target_candidate'] for r in policy_mapping):
        raise ValueError('Main 327 policy response mapping incomplete')
    exits=[dict(key=r['key'],aliases=r['aliases'],excluded=r['input']['excluded'],
                charge=r['input']['charge'],prices=r['input']['prices'],
                level='historical_full_bound_with_historical_entry_check',
                candidate_scope='outside_target_candidate_population',
                source=r['historical_response_file'],sha256=r['response_sha256'])
           for r in index if r['input']['excluded']]
    return dict(summary=summary,candidates=pool,excluded=excluded,envelopes=envelopes,
                policies=appraisals,coverage=coverage,aggregates=aggregate_policies(appraisals),
                full_bound_matches=matches,policy_response_mapping=policy_mapping,exit_ledger=exits)


def prepare(root,output):
    root,output=Path(root),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'frozen_inputs.json').exists(): raise FileExistsError('Frozen input already exists')
    files=(PARAMS,WAITING,INITIAL,REFINE,NEAR,BRANCH,NORMAL,UT,BOUNDS+'/inputs.json',
           BOUNDS+'/replay_verified.json',BOUNDS+'/summary.json',BOUNDS+'/design.json')
    before={n:sha(root/n) for n in files}
    payload=collect_saved(root); index=build_bound_index(root)
    dump(output/'frozen_inputs.json',payload,True); dump(output/'full_bound_index.json',index,True)
    dump(output/'design.json',design(),True)
    for name in OWNED+ENGINE_SOURCES:
        p=output/'execution_source'/name; p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/name,p)
    reference=output/'reference_source'; reference.mkdir(exist_ok=True)
    pass
    dump(output/'source_manifest.json',dict(original_sources=before,
        execution_sources={n:sha(root/n) for n in OWNED+ENGINE_SOURCES},
        frozen_files={n:sha(output/n) for n in ('frozen_inputs.json','full_bound_index.json','design.json')},
        historical_bound_index_scope='Exact saved metadata and source hashes; no leaf evaluation'),True)
    if any(sha(root/n)!=v for n,v in before.items()): raise ValueError('Old source changed during preparation')
    return dict(prepared=True,records=len(payload['records']),bound_responses=len(index))


def discover(payload,output):
    """The only function permitted to import and execute an equilibrium engine."""
    raise RuntimeError('historical-only reconstruction: discovery disabled')
    import numpy as np
    from .config import MarketCase
    from .review90_waiting import WaitingEngine
    from .review_hard_capacity_repair import EvaluationBudget, BudgetExceeded, smooth_polish, verify_candidate
    output=Path(output)
    started=time.perf_counter(); deadline=started+design()['wall_seconds']
    budget=EvaluationBudget(limit=2000000,deadline=deadline)
    cases={tuple(c[k] for k in ('demand_scale','capacity_scale','reliability_power')):c for c in payload['cases']}
    results=[]; records=[]; engines={}; expired=False
    for q in design()['charges']:
        for profile in design()['profiles']:
            c=cases[tuple(profile)]
            for mode in design()['modes']:
                slot=(tuple(profile),mode)
                if slot not in engines:
                    engines[slot]=WaitingEngine(MarketCase(**c),budget,beta=waiting_coefficients(c,mode))
                engine=engines[slot]
                for initial in design()['starts']:
                    i=len(results); row=dict(attempt=i,profile=profile,mode=mode,charge=q,initial=initial)
                    if expired or time.perf_counter()>=deadline:
                        row.update(status='not_attempted_budget'); expired=True
                    else:
                        try:
                            p=smooth_polish(engine,np.full(3,initial),q)
                            candidate=verify_candidate(engine,p,q)
                            o=engine.outcome(p)
                            saved=dict(rho=float(o.rho),wait_minutes=float(o.wait_minutes))
                            record=dict(case=c,mode=mode,charge=float(q),candidate=compact_candidate(candidate),
                                        source=dict(file='discovery.jsonl',attempt=i,initial=initial),saved_outcome=saved)
                            records.append(record)
                            row.update(status='finished',verified=candidate['verified'],record=record)
                        except BudgetExceeded as e:
                            row.update(status='budget_exhausted',error=str(e)); expired=True
                        except Exception as e:
                            row.update(status='failed',error=type(e).__name__+': '+str(e))
                    results.append(row)
                    with (output/'discovery.jsonl').open('a',encoding='utf-8') as f:
                        f.write(json.dumps(row,allow_nan=False)+'\n')
    return dict(records=records,attempts=results,elapsed_seconds=time.perf_counter()-started,
                elapsed_cap_seconds=600,counters=dict(Counter(r['status'] for r in results)),
                candidate_solve_starts=sum(r['status']!='not_attempted_budget' for r in results))


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        w.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)


def run(root,output):
    root,output=Path(root),Path(output)
    manifest=read(output/'source_manifest.json')
    for n,h in manifest['frozen_files'].items():
        if sha(output/n)!=h: raise ValueError('Frozen input changed')
    dump(output/'attempt.json',dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),retry=False),True)
    payload=read(output/'frozen_inputs.json'); index=read(output/'full_bound_index.json')
    discovery=discover(payload,output); dump(output/'discovery.json',discovery,True)
    result=analyze(payload,discovery['records'],index)
    result['summary']['discovery']={k:v for k,v in discovery.items() if k not in ('records','attempts')}
    dump(output/'results.json',result,True)
    for name in ('policies','envelopes','coverage','excluded','aggregates','full_bound_matches'):
        write_csv(output/(name+'.csv'),result[name])
    for n,h in manifest['original_sources'].items():
        if sha(root/n)!=h: raise ValueError('Old science changed during execution: '+n)
    outputs={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}
    dump(output/'manifest.json',dict(files=outputs,old_sources_unchanged=True,
         new_bound_computations=0,leaf_replay=False),True)
    return result['summary']


def replay(output):
    """No engines, no Nash, no BR, no scalar root, no writes, no leaf replay."""
    output=Path(output); manifest=read(output/'manifest.json')
    for n,h in manifest['files'].items():
        p=output/n
        if not p.resolve().is_relative_to(output.resolve()) or sha(p)!=h:
            raise ValueError('Archive hash differs: '+n)
    if read(output/'design.json')!=design(): raise ValueError('Frozen design differs')
    payload=read(output/'frozen_inputs.json')
    fresh=analyze(payload,[],read(output/'full_bound_index.json'))
    if fresh!=read(output/'results.json'): raise ValueError('Candidate appraisal replay differs')
    return dict(valid=True,new_Nash_solves=0,new_BR_searches=0,new_scalar_roots=0,
                full_bound_leaves_recomputed=0,writes=0,summary=fresh['summary'])


def reconstruct_saved(root,output):
    """Preserve interrupted artifacts, then rebuild from original historical records."""
    root,output=Path(root),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    snapshot=output/'interrupted_worker_snapshot'
    if not snapshot.exists():
        snapshot.mkdir()
        for p in list(output.iterdir()):
            if p==snapshot: continue
            if p.is_dir(): shutil.copytree(p,snapshot/p.name)
            else: shutil.copy2(p,snapshot/p.name)
    payload=collect_saved(root); index=build_bound_index(root)
    result=analyze(payload,[],index)
    for name,data in (('frozen_inputs.json',payload),('full_bound_index.json',index),
                      ('design.json',design()),('results.json',result)):
        dump(output/name,data)
    for name in ('policies','envelopes','coverage','excluded','aggregates','full_bound_matches',
                 'policy_response_mapping','exit_ledger'):
        write_csv(output/(name+'.csv'),result[name])
    dump(output/'discovery.json',dict(records=[],attempts=[],candidate_solve_starts=0,
        status='disabled_historical_only',legacy_artifacts='interrupted_worker_snapshot; excluded'))
    for name in OWNED:
        p=output/'execution_source'/name; p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/name,p)
    sources=(PARAMS,WAITING,INITIAL,REFINE,NEAR,BRANCH,NORMAL,UT)
    dump(output/'source_manifest.json',dict(original_sources={n:sha(root/n) for n in sources},
        historical_bound_index_scope='441 exact keys; metadata and hashes only; no leaf replay',
        legacy_discovery_excluded=True))
    files={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*')
           if p.is_file() and p.name!='manifest.json' and snapshot not in p.parents}
    dump(output/'manifest.json',dict(files=files,new_bound_computations=0,leaf_replay=False,
        legacy_snapshot_excluded=True))
    return result['summary']
