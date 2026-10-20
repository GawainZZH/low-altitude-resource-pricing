"""Capped auxiliary Nash construction; actual-game support is conditional.

Participation is fixed until a firm has no positive feasible margin. Removing
that firm explicitly changes the game to the voluntary-exit extension.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

from .review8_resource import _primitives
from .review11_finite_paths import design as original_design

OUTPUT = 'results/trb_review15_capped_v1'
MODULE = 'experiments/aam_pricing/review15_capped.py'
SCRIPT = 'scripts/run_review15_capped.py'
TEST = 'tests/test_review15_capped.py'
SOURCES = (MODULE,SCRIPT,TEST,'experiments/__init__.py',
           'experiments/aam_pricing/__init__.py',
           'experiments/aam_pricing/review11_finite_paths.py',
           'experiments/aam_pricing/review8_resource.py',
           'experiments/aam_pricing/review7_structure.py')
PROTECTED = SOURCES[3:] + (
    'submission_packages/TRA_V290_LaTeX_Project/MAIN.tex',
    'submission_packages/TRA_V290_LaTeX_Project/MAIN.pdf',
    'results/trb_review11_finite_paths_v1/inputs.json',
    'results/trb_review11_finite_paths_v1/summary.json',
    'results/trb_review11_finite_paths_v1/paths_compact.csv')
KAPPAS = (0.,.5,1.)
KKT_TOL = 2e-9


def design():
    cases = [deepcopy(r) for r in original_design()['cases']
             if r['alignment'] == 0 and r['primitives']['fare_bounds'][1] == 44.]
    return dict(schema_version=1,cases=cases,charges=list(range(0,21,2)),
                kappas=list(KAPPAS),max_auxiliary_vectors=44,runtime_cap_seconds=600,
                new_general_Nash_solves=0,participation_rule='exit iff adjusted cost >= upper fare',
                numerical_interpretation='Analytical theorem with floating-point residual witnesses; not interval arithmetic',
                policy_scope='Selected constructed responses on qualifying points of this restricted menu only')


def _budget(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError('Frozen 600-second production cap reached; no expansion')


def _case(case,q):
    c = _primitives(case)
    if not np.isfinite(q) or q < 0:
        raise ValueError('Finite nonnegative charge required')
    return c


def shares(case,prices,active):
    a = np.asarray(case['resource'],float)
    p, active = np.asarray(prices,float),np.asarray(active,bool)
    x = np.zeros(len(a))
    if active.any():
        z = np.log(a[active])+(np.asarray(case['platform_quality'])[active]-case['lam']*p[active])/case['mu']
        x[active] = np.exp(z-logsumexp(z))
    return x


def auxiliary_profit(case,q,prices,active):
    a = np.asarray(case['resource'])
    return case['capacity']/a*(np.asarray(prices)-np.asarray(case['costs'])-a*q)*shares(case,prices,active)


def _H(case,q,prices,m,active):
    x = shares(case,prices,active)[m]
    return float(prices[m]-case['costs'][m]-case['resource'][m]*q-1/(case['lam']/case['mu']*(1-x)))


def best_response(case,q,prices,m,active,deadline=None):
    _budget(deadline)
    lo,hi = case['fare_bounds']
    if not active[m]:
        raise ValueError('No fare best response for an exited firm')
    if sum(active) == 1:
        return float(hi)
    p = np.asarray(prices,float).copy()
    def residual(v):
        _budget(deadline)
        p[m] = v
        return _H(case,q,p,m,active)
    if residual(lo) >= 0:
        return float(lo)
    if residual(hi) <= 0:
        return float(hi)
    return float(brentq(residual,lo,hi,xtol=5e-14,rtol=1e-14))


def kkt_witness(case,q,p,active):
    """Global single-peaked BR error bounds from saved prices, without solving."""
    lo,hi = case['fare_bounds']
    a, costs = np.asarray(case['resource']),np.asarray(case['costs'])
    k, h = costs+a*q,case['lam']/case['mu']
    rows = []
    n = sum(active)
    for m in np.flatnonzero(active):
        if p[m] < lo or p[m] > hi:
            raise ValueError('Fare outside fixed box')
        if n == 1:
            H, residual = None,float(hi-p[m])
            mode = 'singleton_upper'
        else:
            H = _H(case,q,p,m,active)
            if p[m] == lo:
                mode,residual = 'lower',max(0.,-H)
            elif p[m] == hi:
                mode,residual = 'upper',max(0.,H)
            else:
                mode,residual = 'interior',abs(H)
        lip = case['capacity']/a[m]*(1+h*max(abs(lo-k[m]),abs(hi-k[m])))
        rows.append(dict(firm=int(m),mode=mode,H=H,global_BR_price_error_bound=float(residual),
                         global_auxiliary_profit_gain_bound=float(lip*residual)))
    # Box-wide BR row sums are bounded by the maximum own share at lo vs rivals hi.
    limits = []
    if n >= 2:
        for m in np.flatnonzero(active):
            corner = np.full(len(a),hi); corner[m] = lo
            limits.append(float(shares(case,corner,active)[m]))
    return dict(active=list(map(bool,active)),rows=rows,
                maximum_kkt_price_residual=max((r['global_BR_price_error_bound'] for r in rows),default=0.),
                maximum_auxiliary_profit_gain_bound=max((r['global_auxiliary_profit_gain_bound'] for r in rows),default=0.),
                uniform_BR_contraction_bound=max(limits,default=0.))


def solve_auxiliary(case,q,deadline=None):
    _budget(deadline)
    c = _case(case,q)
    k = c['costs']+c['resource']*q
    active = k < c['fare_bounds'][1]
    p = np.full(len(k),float(c['fare_bounds'][1]))
    for _ in range(20000):
        _budget(deadline)
        new = p.copy()
        for m in np.flatnonzero(active):
            new[m] = best_response(c,q,p,m,active,deadline)
        p = new
        witness = kkt_witness(c,q,p,active)
        if witness['maximum_kkt_price_residual'] < 2e-12:
            return p,witness
    raise RuntimeError('Auxiliary BR iteration budget exhausted; no rescue or extra vector')


def choice(case,prices,active,rho,wait):
    c = case
    mass = np.asarray(c['mass'])
    active = np.asarray(active,bool)
    p = np.asarray(prices,float)
    air = np.full((len(mass),len(p)),-np.inf)
    if active.any():
        group = np.asarray(c['group_quality'])-np.asarray(c['reliability_slopes'])*rho
        air[:,active] = (group[:,None]+np.asarray(c['platform_quality'])[active]
                        -c['lam']*p[active]-c['lam']*c['value']*wait)/c['mu']
    den = np.logaddexp(np.asarray(c['outside_utility'])/c['mu'],logsumexp(air,axis=1))
    prob = np.exp(air-den[:,None])
    return dict(probabilities=prob.tolist(),demand=(mass@prob).tolist(),
                net_surplus=float((mass*c['mu']/c['lam'])@den))


def close_response(case,p,active,deadline=None):
    a,K = np.asarray(case['resource']),case['capacity']
    def excess(rho,w):
        _budget(deadline)
        return float(a@choice(case,p,active,rho,w)['demand']-K*rho)
    initial = excess(1.,0.)
    if abs(initial) <= 1e-10:
        return 1.,0.
    if initial < 0:
        return float(brentq(lambda r:excess(r,0.),0.,1.,xtol=2e-13)),0.
    upper = 1.
    for _ in range(64):
        if excess(1.,upper) <= 0:
            return 1.,float(brentq(lambda w:excess(1.,w),0.,upper,xtol=2e-12))
        upper *= 2
    raise RuntimeError('Clearing wait bracket unavailable')


def _comparative(case,q,p,active,x,y,prob,wait,full,witness):
    if not full or not active.any():
        return dict(smooth_active_set=False,interpretation='No full-capacity derivative after release')
    a,costs = np.asarray(case['resource']),np.asarray(case['costs'])
    h,K,v = case['lam']/case['mu'],case['capacity'],case['value']
    interior = np.zeros(len(a),bool)
    kink = False
    for row in witness['rows']:
        interior[row['firm']] = row['mode'] == 'interior'
        if row['mode'] in ('upper','lower') and abs(row['H']) < 1e-7:
            kink = True
    if np.any(np.abs(costs+a*q-case['fare_bounds'][1]) < 1e-10):
        kink = True
    eta = x/(1-x+x*x)
    T = float(np.sum(x[interior]*(1-eta[interior])*a[interior])/
              (1-np.sum(x[interior]*eta[interior])))
    dp = np.zeros(len(a)); dp[interior] = (1-eta[interior])*a[interior]+eta[interior]*T
    dx = h*x*(T-dp)
    dy = K*dx/a
    Y,dY = float(y.sum()),float(dy.sum())
    theta = prob.sum(axis=1)
    L = float(np.asarray(case['mass'])@(theta*(1-theta)))
    dw = -float(y@dp/Y+dY/(h*L))/v
    relief = -v*Y*dw
    allocation = {str(k):float((p-costs-k*case['outside_price'])@dy) for k in KAPPAS}
    ts = {str(k):relief+allocation[str(k)] for k in KAPPAS}
    return dict(smooth_active_set=bool(not kink and wait > 0),
                interpretation='Open fixed-participation and fixed-fare-active-set segment; no two-sided claim at kinks',
                resource_share_slope=dx.tolist(),fare_slope=dp.tolist(),demand_slope=dy.tolist(),
                total_mass_slope=dY,wait_slope=dw,participation_L=L,weighted_fare_slope=T,
                relief_slope=relief,allocation_slopes=allocation,ts_derivatives=ts)


def _assemble(case,q,p,active,rho,wait):
    c = _case(case,q)
    a,costs = c['resource'],c['costs']
    active = np.asarray(active,bool)
    expected = costs+a*q < c['fare_bounds'][1]
    if not np.array_equal(active,expected):
        raise ValueError('Active set violates explicit nonpositive-margin exit rule')
    if not 0 <= rho <= 1 or wait < 0 or (wait > 0 and rho != 1):
        raise ValueError('Invalid closure/complementarity')
    witness = kkt_witness(c,q,p,active)
    if witness['maximum_kkt_price_residual'] > KKT_TOL:
        raise ValueError('Saved prices fail global auxiliary BR residual')
    x = shares(c,p,active)
    o = choice(c,p,active,rho,wait)
    y,prob = np.asarray(o['demand']),np.asarray(o['probabilities'])
    R = float(a@y)
    if not np.isclose(R,c['capacity']*rho,atol=2e-7,rtol=2e-10):
        raise ValueError('Primitive closure residual')
    zero = float(a@choice(c,p,active,1.,0.)['demand']-c['capacity'])
    margin = p-costs-a*q
    full = bool(rho == 1. and abs(R-c['capacity']) <= 2e-7)
    positive = bool(active.any() and np.all(margin[active] > 0))
    equality = bool(active.any() and full and np.allclose(y,c['capacity']*x/a,atol=2e-7,rtol=2e-10))
    profits = margin*y
    aux = c['capacity']/a*margin*x
    supported = bool(full and positive and equality)
    TS = {str(k):float(o['net_surplus']+(p-costs)@y+k*c['outside_price']*(c['mass'].sum()-y.sum()))
          for k in KAPPAS}
    noentry = []
    for m in range(len(a)):
        top = float(c['fare_bounds'][1]-costs[m]-a[m]*q)
        noentry.append(dict(firm=m,exited=bool(not active[m]),maximum_feasible_margin=top,
                            nonpositive_all_fares=bool(top <= 0),strictly_negative_all_fares=bool(top < 0),
                            positive_profit_reentry_excluded=bool(not active[m] and top <= 0),
                            active_exit_gain=float(-profits[m]) if active[m] else 0.))
    premises = dict(common_lambda=True,common_b_equals_lambda_v=True,additive_preferences=True,
                    resource_identity=bool(np.allclose(y,R*x/a,atol=2e-7,rtol=2e-10)),
                    resource_at_most_capacity=bool(R <= c['capacity']+2e-7),
                    all_nonnegative_margin_deviations_profit_dominated=True,
                    negative_margin_deviations_nonpositive=True,
                    active_positive_margins=positive,full_capacity=full,
                    candidate_profit_equality=equality,clearing_wait_nonnegative=bool(wait >= 0),
                    excluded_firms_no_positive_profit_reentry=all(r['positive_profit_reentry_excluded'] for r in noentry if r['exited']))
    classification = ('no_active_service' if not active.any() else
                      'actual_global_nash_supported' if supported else 'auxiliary_only_after_release')
    return dict(charge=float(q),prices=[float(v) if active[m] else None for m,v in enumerate(p)],
                active=active.tolist(),game_scope='fixed_participation' if active.all() else 'voluntary_exit_extension',
                classification=classification,actual_global_nash_supported=supported,
                numerical_exact_nash_claim=False,premises=premises,exit_noentry=noentry,
                auxiliary_witness=witness,auxiliary_resource_shares=x.tolist(),
                resource_shares=(a*y/R).tolist() if R > 0 else [0.]*len(a),
                resource_use=R,resource_use_slope=None,zero_wait_excess=zero,rho=float(rho),wait=float(wait),
                total_mass=float(y.sum()),ground_mass=float(c['mass'].sum()-y.sum()),
                active_margins=[float(v) if active[m] else None for m,v in enumerate(margin)],
                actual_profits=profits.tolist(),auxiliary_profits=aux.tolist(),TS=TS,
                **o,comparative_statics=_comparative(c,q,p,active,x,y,prob,wait,full,witness))


def evaluate_point(case,q,deadline=None):
    _budget(deadline)
    p,witness = solve_auxiliary(case,q,deadline)
    active = np.asarray(witness['active'],bool)
    rho,wait = close_response(case,p,active,deadline)
    return _assemble(case,q,p,active,rho,wait)


def _equal(saved,fresh,path='point'):
    if isinstance(fresh,dict):
        if not isinstance(saved,dict) or set(saved) != set(fresh):
            raise ValueError('Saved keys mismatch: '+path)
        for k in fresh: _equal(saved[k],fresh[k],path+'.'+k)
    elif isinstance(fresh,list):
        if not isinstance(saved,list) or len(saved) != len(fresh):
            raise ValueError('Saved length mismatch: '+path)
        for i,v in enumerate(fresh): _equal(saved[i],v,path+'.'+str(i))
    elif isinstance(fresh,(str,bool)) or fresh is None:
        if saved != fresh: raise ValueError('Saved label mismatch: '+path)
    elif not np.isfinite(saved) or not np.isclose(saved,fresh,atol=2e-8,rtol=2e-11):
        raise ValueError('Saved arithmetic mismatch: '+path)


def replay_point(case,saved):
    active = np.asarray(saved['active'],bool)
    p = np.array([case['fare_bounds'][1] if v is None else v for v in saved['prices']])
    fresh = _assemble(case,saved['charge'],p,active,saved['rho'],saved['wait'])
    _equal(saved,fresh)
    return dict(valid=True,scalar_roots_called=0,optimizations_called=0,
                global_auxiliary_BR_proof='Unimodality and boundary KKT residual; no sampled-deviation inference')


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def prepare(root,out):
    root,out = Path(root),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    d = design()
    target = out/'inputs.json'
    if target.exists():
        _equal(json.loads(target.read_text(encoding='utf-8')),d,'frozen design')
    else:
        _write(target,d)
    hashes = {f:_hash(root/f) for f in SOURCES}
    frozen = out/'source_snapshot'
    for f in SOURCES:
        dest = frozen/f
        if dest.exists() and _hash(dest) != hashes[f]:
            raise ValueError('Existing frozen source differs: '+f)
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists(): shutil.copyfile(root/f,dest)
    _write(out/'source_manifest.json',hashes)
    return dict(prepared=True,auxiliary_vector_budget=44)


def summarize(d,rows):
    cases = []
    for case in d['cases']:
        points = [r['state'] for r in rows if r['case_id'] == case['id'] and r['status'] == 'completed']
        first = next((i for i,s in enumerate(points) if any(v == 44 for v in s['prices'])),None)
        qualifying = [s for s in points if s['actual_global_nash_supported']]
        optima = []
        for k in KAPPAS:
            key = str(k)
            best = max(qualifying,key=lambda s:(s['TS'][key],-s['charge'])) if qualifying else None
            diagnostic = max(points,key=lambda s:(s['TS'][key],-s['charge'])) if points else None
            optima.append(dict(kappa=k,qualifying_restricted_menu_charge=None if best is None else best['charge'],
                               qualifying_TS=None if best is None else best['TS'][key],
                               all_auxiliary_diagnostic_max_charge=None if diagnostic is None else diagnostic['charge'],
                               all_auxiliary_diagnostic_max_is_actual_supported=False if diagnostic is None else diagnostic['actual_global_nash_supported'],
                               full_menu_or_all_Nash_policy_optimum=False))
        negative = [s['charge'] for s in qualifying if s['comparative_statics'].get('smooth_active_set')
                    and s['comparative_statics'].get('total_mass_slope',0) < -1e-9]
        cases.append(dict(case_id=case['id'],completed=len(points),qualifying_charges=[s['charge'] for s in qualifying],
                          first_menu_fare_contact=None if first is None else points[first]['charge'],
                          fare_contact_bracket=None if first is None else [points[max(0,first-1)]['charge'],points[first]['charge']],
                          contact_is_exact_continuous_threshold=False,
                          comparison_indices=[] if first is None else list(range(max(0,first-1),min(len(points),first+2))),
                          negative_Ydot_qualifying_charges=negative,restricted_optima=optima))
    complete = sum(r['status'] == 'completed' for r in rows)
    return dict(completed_points=complete,planned_points=44,attempted_vectors=len(rows),
                all_completed=complete == 44,new_general_Nash_solves=0,
                actual_global_nash_supported_points=sum(r.get('state',{}).get('actual_global_nash_supported',False) for r in rows),
                post_release_auxiliary_points=sum(r.get('state',{}).get('classification') == 'auxiliary_only_after_release' for r in rows),
                voluntary_exit_extension_points=sum(r.get('state',{}).get('game_scope') == 'voluntary_exit_extension' for r in rows),
                cases=cases,scope=d['policy_scope'])


def _csv(out,rows):
    fields = ['case_id','q','scope','classification','prices','active','resource_shares','served_Y','w','rho',
              'TS_k0','TS_k05','TS_k1','Ydot','wdot','global_auxiliary_gain_bound']
    with (out/'points.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for r in rows:
            if r['status'] != 'completed': continue
            s = r['state']
            writer.writerow(dict(case_id=r['case_id'],q=s['charge'],scope=s['game_scope'],classification=s['classification'],
                                 prices=json.dumps(s['prices']),active=json.dumps(s['active']),resource_shares=json.dumps(s['resource_shares']),
                                 served_Y=s['total_mass'],w=s['wait'],rho=s['rho'],TS_k0=s['TS']['0.0'],TS_k05=s['TS']['0.5'],TS_k1=s['TS']['1.0'],
                                 Ydot=s['comparative_statics'].get('total_mass_slope'),wdot=s['comparative_statics'].get('wait_slope'),
                                 global_auxiliary_gain_bound=s['auxiliary_witness']['maximum_auxiliary_profit_gain_bound']))


def run(root,out):
    root,out = Path(root),Path(out)
    if (out/'production_started.json').exists():
        raise FileExistsError('One-shot production already started; replay only, no replacement run')
    prepare(root,out)
    d = json.loads((out/'inputs.json').read_text(encoding='utf-8'))
    before = {f:_hash(root/f) for f in PROTECTED}
    _write(out/'production_started.json',dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                                            max_vectors=44,runtime_cap_seconds=600))
    start = time.monotonic(); deadline = start+600.
    rows = []
    try:
        for case in d['cases']:
            for q in d['charges']:
                _budget(deadline)
                if len(rows) >= 44: raise RuntimeError('Frozen vector budget exhausted')
                try:
                    state = evaluate_point(case['primitives'],q,deadline)
                    replay_point(case['primitives'],state)
                    rows.append(dict(case_id=case['id'],status='completed',state=state))
                except Exception as exc:
                    rows.append(dict(case_id=case['id'],charge=q,status='failed',error=repr(exc)))
                    raise
                finally:
                    _write(out/'points.json',rows)
    finally:
        _write(out/'points.json',rows)
        after = {f:_hash(root/f) for f in PROTECTED}
        _write(out/'protected_files.json',dict(before=before,after=after,unchanged=before == after,
                                               scope='Named old sources, active MAIN pair and original finite-path input/summary/CSV; not a claim about concurrent owners'))
        summary = summarize(d,rows)
        _write(out/'summary.json',summary)
        _csv(out,rows)
        _write(out/'execution.json',dict(elapsed_seconds=time.monotonic()-start,
                                        attempted_vectors=len(rows),runtime_cap_seconds=600,new_general_Nash_solves=0))
        _write(out/'result_manifest.json',{f:_hash(out/f) for f in ('inputs.json','points.json','summary.json','points.csv')})
    if before != after:
        raise ValueError('Protected files changed during production; inspect concurrent ownership')
    return summary


def replay(out):
    out = Path(out)
    for f,digest in json.loads((out/'result_manifest.json').read_text(encoding='utf-8')).items():
        if _hash(out/f) != digest: raise ValueError('Result manifest mismatch: '+f)
    for f,digest in json.loads((out/'source_manifest.json').read_text(encoding='utf-8')).items():
        if _hash(out/'source_snapshot'/f) != digest: raise ValueError('Frozen source mismatch: '+f)
    d = json.loads((out/'inputs.json').read_text(encoding='utf-8'))
    _equal(d,design(),'design')
    rows = json.loads((out/'points.json').read_text(encoding='utf-8'))
    expected = [(c['id'],q) for c in d['cases'] for q in d['charges']]
    got = [(r['case_id'],r.get('state',{}).get('charge',r.get('charge'))) for r in rows]
    if got != expected or any(r['status'] != 'completed' for r in rows):
        raise ValueError('Incomplete or reordered frozen menu; cannot pass complete replay')
    cases = {c['id']:c['primitives'] for c in d['cases']}
    for r in rows: replay_point(cases[r['case_id']],r['state'])
    summary = summarize(d,rows)
    _equal(json.loads((out/'summary.json').read_text(encoding='utf-8')),summary,'summary')
    return dict(valid=True,points_verified=len(rows),scalar_roots_called=0,optimizations_called=0,
                new_general_Nash_solves=0,summary=summary)
