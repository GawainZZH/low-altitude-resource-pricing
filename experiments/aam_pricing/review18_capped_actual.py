"""Actual capped fare games at exactly five historical unsupported charges."""
from __future__ import annotations

from decimal import Decimal
import hashlib
import heapq
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize_scalar
from numpy.polynomial import Polynomial

from . import review15_capped as cp
from .review13_equilibrium import Interval, interval, midpoint, UP

OUTPUT = 'results/review18_capped_actual_v1'
OLD = 'results/trb_review15_capped_v1'
TOL = Decimal('0.000001')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def budget(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError('Frozen review18 computation cap reached; no expansion')


def design(root):
    d = read(Path(root)/OLD/'inputs.json')
    rows = read(Path(root)/OLD/'points.json')
    targets = [dict(case_id=r['case_id'], charge=r['state']['charge']) for r in rows
               if not r['state']['actual_global_nash_supported']]
    if len(targets) != 5 or len(rows) != 44:
        raise ValueError('Historical five-target/44-point design changed')
    return dict(cases=d['cases'], charges=d['charges'], targets=targets,
                reused_points=39, runtime_cap_seconds=900, gain_tolerance=str(TOL),
                max_candidate_sweeps=120, max_leaves_per_coordinate=4096,
                all_equilibria_enumerated=False)


def state(c, q, p, active):
    p, active = np.asarray(p, float), np.asarray(active, bool)
    rho, w = cp.close_response(c, p, active)
    o = cp.choice(c, p, active, rho, w)
    y = np.asarray(o['demand']); a = np.asarray(c['resource'])
    costs = np.asarray(c['costs'])
    return dict(charge=float(q), prices=[float(x) if active[i] else None for i,x in enumerate(p)],
                active=active.tolist(), rho=rho, wait=w, resource_use=float(a@y),
                total_mass=float(y.sum()), profits=((p-costs-a*q)*y).tolist(),
                TS={str(k):float(o['net_surplus']+(p-costs)@y+
                    k*c['outside_price']*(sum(c['mass'])-y.sum())) for k in cp.KAPPAS}, **o)


def gradient(c, q, p, active):
    s = state(c,q,p,active); h = c['lam']/c['mu']
    a = np.asarray(c['resource']); p = np.asarray(p)
    prob = np.asarray(s['probabilities']); n = np.asarray(c['mass'])
    if s['wait'] > 0:
        x = cp.shares(c,p,active)
        dy = -h*c['capacity']/a*x*(1-x)
    else:
        theta = prob.sum(axis=1); work = prob@a
        b = np.asarray(c['reliability_slopes'])/c['mu']
        yr = -(n*b*(1-theta))@prob
        Rp = -h*(n[:,None]*prob*(a-work[:,None])).sum(axis=0)
        Rr = -float((n*b*work)@(1-theta))
        dy = -h*(n[:,None]*prob*(1-prob)).sum(axis=0)+yr*Rp/(c['capacity']-Rr)
    return np.asarray(s['demand'])+(p-np.asarray(c['costs'])-a*q)*dy


def regime_breakpoints(c, p, active, m):
    """All zero-wait contacts: cubic in own positive attraction z."""
    active=np.asarray(active,bool); p=np.asarray(p,float)
    t=np.exp((np.asarray(c['platform_quality'])-c['lam']*p)/c['mu'])*active
    t[m]=0.; T=t.sum(); B=np.asarray(c['resource'])@t; a=c['resource'][m]
    b=np.exp((np.asarray(c['group_quality'])-np.asarray(c['reliability_slopes'])-
              np.asarray(c['outside_utility']))/c['mu'])
    factors=[Polynomial([1+bg*T,bg]) for bg in b]
    den=Polynomial([1.])
    for factor in factors: den*=factor
    numerator=Polynomial([0.])
    for g, bg in enumerate(b):
        term=Polynomial([c['mass'][g]*bg])
        for j,factor in enumerate(factors):
            if j != g: term*=factor
        numerator+=term
    poly=Polynomial([B,a])*numerator-c['capacity']*den
    lo,hi=c['fare_bounds']; roots=[]
    for z in poly.roots():
        if abs(z.imag)<1e-8 and z.real>0:
            fare=(c['platform_quality'][m]-c['mu']*np.log(z.real))/c['lam']
            if lo<fare<hi: roots.append(float(fare))
    return sorted(roots)


def solve(c,q,initial,active,deadline=None):
    budget(deadline)
    p=np.asarray(initial,float).copy(); active=np.asarray(active,bool)
    for _ in range(120):
        old=p.copy()
        for m in np.flatnonzero(active):
            budget(deadline)
            cuts=[c['fare_bounds'][0],*regime_breakpoints(c,p,active,m),c['fare_bounds'][1]]
            def objective(fare):
                budget(deadline)
                dev=p.copy(); dev[m]=fare
                return -state(c,q,dev,active)['profits'][m]
            candidates=cuts+[p[m]]
            for l,r in zip(cuts,cuts[1:]):
                fit=minimize_scalar(objective,bounds=(l,r),method='bounded',
                                    options={'xatol':1e-11,'maxiter':120})
                candidates.append(float(fit.x))
            p[m]=min(candidates,key=objective)
        if np.max(abs(p-old))<2e-7: return p
    return p


def hull(a,b):
    return Interval(min(a.lo,b.lo),max(a.hi,b.hi))


class FareEnvelope:
    """Directed bounds for binding/slack closures and unilateral derivatives."""
    def __init__(self,c,q,p,active,m):
        self.c,self.q,self.p,self.active,self.m=c,q,np.asarray(p,float),np.asarray(active,bool),int(m)
        self.a=[Interval(x) for x in c['resource']]
        self.K=Interval(c['capacity']); self.h=Interval(c['lam'])/Interval(c['mu'])
        self.k=Interval(c['costs'][m])+self.a[m]*Interval(q)
        self.t=[((Interval(c['platform_quality'][j])-Interval(c['lam'])*Interval(p[j]))/
                 Interval(c['mu'])).exp() if self.active[j] else Interval(0) for j in range(len(p))]

    def attractions(self,fare):
        t=self.t.copy(); m=self.m
        t[m]=((Interval(self.c['platform_quality'][m])-Interval(self.c['lam'])*fare)/
              Interval(self.c['mu'])).exp()
        return t

    def choice(self,t,rho):
        c=self.c; T=sum(t,Interval(0)); y=[Interval(0) for _ in t]
        probs=[]; R=Interval(0)
        for g,n in enumerate(c['mass']):
            b=((Interval(c['group_quality'][g])-Interval(c['reliability_slopes'][g])*rho-
                Interval(c['outside_utility'][g]))/Interval(c['mu'])).exp()
            den=1+b*T
            ps=[b*x/den for x in t]
            probs.append(ps)
            for j,x in enumerate(ps): y[j]+=Interval(n)*x
        R=sum((a*v for a,v in zip(self.a,y)),Interval(0))
        return y,probs,R

    def free(self,t):
        rho=Interval(0,1)
        for _ in range(18):
            y,ps,R=self.choice(t,rho)
            new=rho.intersection(R/self.K)
            if new is None: return None
            if new.lo==rho.lo and new.hi==rho.hi: break
            rho=new
        y,ps,R=self.choice(t,rho)
        return rho,y,ps,R

    def data(self,fare):
        t=self.attractions(fare); m=self.m
        atone=self.choice(t,Interval(1))[2]
        x=self.a[m]*t[m]/sum((a*v for a,v in zip(self.a,t)),Interval(0))
        full_y=self.K/self.a[m]*x
        full_d=-self.h*full_y*(1-x)
        free=self.free(t) if atone.lo<=self.K.hi else None
        if atone.hi<self.K.lo:
            if free is None: raise ValueError('Lost free closure')
            modes=['slack']
        elif atone.lo>self.K.hi: modes=['binding']
        else: modes=['binding','slack'] if free is not None else ['binding']
        y,dy=full_y,full_d
        if free is not None:
            _,ys,ps,_=free; direct=Interval(0); yr=Interval(0); Rp=Interval(0); Rr=Interval(0)
            for g,n in enumerate(self.c['mass']):
                pi=ps[g][m]; theta=sum(ps[g],Interval(0))
                work=sum((a*v for a,v in zip(self.a,ps[g])),Interval(0))
                b=Interval(self.c['reliability_slopes'][g])/Interval(self.c['mu'])
                direct-=self.h*Interval(n)*pi*(1-pi)
                yr-=Interval(n)*b*pi*(1-theta)
                Rp-=self.h*Interval(n)*pi*(self.a[m]-work)
                Rr-=Interval(n)*b*work*(1-theta)
            # True R_rho <= 0. Dependency can inflate this enclosure.
            Rr=Interval(Rr.lo,min(Decimal(0),Rr.hi))
            fd=direct+yr*Rp/(self.K-Rr)
            if modes==['slack']: y,dy=ys[m],fd
            elif 'slack' in modes: y,dy=hull(y,ys[m]),hull(dy,fd)
        return y,dy,modes

    def profit(self,fare):
        y,_,_=self.data(fare)
        return (fare-self.k)*y

    def profit_upper(self,lo,hi):
        fare=Interval(lo,hi); y,dy,modes=self.data(fare)
        natural=(fare-self.k)*y
        derivative=y+(fare-self.k)*dy
        mid=midpoint(fare); center=self.profit(Interval(mid))
        radius=max(UP.subtract(fare.hi,mid),UP.subtract(mid,fare.lo))
        delta=Interval(radius.copy_negate(),radius)
        upper=min(natural.hi,(center+derivative*delta).hi)
        if derivative.lo>=0: upper=min(upper,self.profit(Interval(hi)).hi)
        if derivative.hi<=0: upper=min(upper,self.profit(Interval(lo)).hi)
        return Interval(min(natural.lo,upper),upper)


def bound_game(c,q,p,active,deadline=None):
    coordinates=[]
    for m in np.flatnonzero(active):
        budget(deadline); env=FareEnvelope(c,q,p,active,m)
        base=env.profit(Interval(p[m])); serial=0
        lo,hi=(Interval(v).lo for v in c['fare_bounds'])
        upper=env.profit_upper(lo,hi).hi
        heap=[(upper.copy_negate(),serial,lo,hi)]; splits=0
        while max(Decimal(0),UP.subtract(heap[0][0].copy_negate(),base.lo))>TOL and len(heap)<4096:
            budget(deadline)
            _,_,l,r=heapq.heappop(heap); cut=midpoint(Interval(l,r))
            for ll,rr in ((l,cut),(cut,r)):
                ub=env.profit_upper(ll,rr).hi; serial+=1
                heapq.heappush(heap,(ub.copy_negate(),serial,ll,rr))
            splits+=1
        leaves=[dict(fare=[str(l),str(r)],profit_upper=str(v.copy_negate())) for v,_,l,r in sorted(heap,key=lambda z:z[2])]
        ub=heap[0][0].copy_negate(); gap=max(Decimal(0),UP.subtract(ub,base.lo))
        coordinates.append(dict(firm=int(m),baseline=base.serialized(),profit_upper=str(ub),
                                gain_upper=str(gap),leaves=leaves,splits=splits))
    gap=max((float(x['gain_upper']) for x in coordinates),default=0.)
    noentry=[dict(firm=int(m),maximum_margin=(Interval(c['fare_bounds'][1])-
             Interval(c['costs'][m])-Interval(c['resource'][m])*Interval(q)).serialized())
             for m in np.flatnonzero(~np.asarray(active,bool))]
    if any(Decimal(r['maximum_margin'][1])>0 for r in noentry):
        raise ValueError('Excluded firm can earn a positive margin')
    return dict(coordinates=coordinates,noentry=noentry,coverage_complete=True,maximum_gain_upper=gap,
                epsilon_nash_closed=gap<=float(TOL),arithmetic='Decimal40 outward; binary input primitives',
                exact_nash_claim=False,all_equilibria_enumerated=False)


def replay_bounds(c,q,p,active,proof):
    expected=list(map(int,np.flatnonzero(active)))
    if [r['firm'] for r in proof['coordinates']]!=expected: raise ValueError('Coordinate coverage')
    gains=[]
    for row in proof['coordinates']:
        env=FareEnvelope(c,q,p,active,row['firm']); base=env.profit(Interval(p[row['firm']]))
        last=Interval(c['fare_bounds'][0]).lo; bounds=[]
        for leaf in row['leaves']:
            l,r=map(Decimal,leaf['fare'])
            if l!=last or r<=l: raise ValueError('Fare coverage gap')
            value=env.profit_upper(l,r).hi
            if str(value)!=leaf['profit_upper']: raise ValueError('Profit bound changed')
            last=r; bounds.append(value)
        if last!=Interval(c['fare_bounds'][1]).hi: raise ValueError('Fare endpoint gap')
        upper=max(bounds); gain=max(Decimal(0),UP.subtract(upper,base.lo))
        if row['baseline']!=base.serialized() or row['profit_upper']!=str(upper) or row['gain_upper']!=str(gain):
            raise ValueError('Baseline/gain mismatch')
        gains.append(float(gain))
    if proof['maximum_gain_upper']!=max(gains,default=0.): raise ValueError('Maximum mismatch')
    if proof['epsilon_nash_closed']!=(max(gains,default=0.)<=float(TOL)): raise ValueError('Closure label')
    noentry=[dict(firm=int(m),maximum_margin=(Interval(c['fare_bounds'][1])-
             Interval(c['costs'][m])-Interval(c['resource'][m])*Interval(q)).serialized())
             for m in np.flatnonzero(~np.asarray(active,bool))]
    if proof['noentry']!=noentry or any(Decimal(r['maximum_margin'][1])>0 for r in noentry):
        raise ValueError('Withdrawal/reentry bound')
    return dict(valid=True,coordinates=len(expected),coverage_complete=True)


def summarize(root, rows):
    d=design(root)
    expected={(r['case_id'],r['charge']) for r in d['targets']}
    if len(rows)!=5 or {(r['case_id'],r['state']['charge']) for r in rows}!=expected:
        raise ValueError('Exactly five actual target games required')
    old=read(Path(root)/OLD/'points.json'); new={(r['case_id'],r['state']['charge']):r for r in rows}
    cases=[]
    for case in d['cases']:
        historical=[r['state'] for r in old if r['case_id']==case['id']]
        supported=[s for s in historical if s['actual_global_nash_supported']]
        points=[]
        for s in historical:
            replacement=new.get((case['id'],s['charge']))
            points.append(replacement['state'] if replacement else s)
        first=lambda pred:next((s['charge'] for s in points if pred(s)),None)
        choices=[]
        closed=all(r['proof']['epsilon_nash_closed'] for r in rows if r['case_id']==case['id'])
        for k in cp.KAPPAS:
            key=str(k); sub=max(supported,key=lambda s:(s['TS'][key],-s['charge']))
            best=max(points,key=lambda s:(s['TS'][key],-s['charge']))
            ordered=sorted(points,key=lambda s:s['TS'][key],reverse=True)
            choices.append(dict(kappa=k,supported_submenu_charge=sub['charge'],supported_submenu_TS=sub['TS'][key],
                                full_selected_menu_charge=best['charge'],full_selected_menu_TS=best['TS'][key],
                                increment_over_submenu=best['TS'][key]-sub['TS'][key],
                                runner_up_gap=ordered[0]['TS'][key]-ordered[1]['TS'][key],
                                exact_exhaustive_menu_comparison=True,all_response_bounds_closed=closed,
                                all_Nash_policy_optimum=False,exact_real_arithmetic_claim=False))
        cases.append(dict(case_id=case['id'],menu_points=len(points),optima=choices,
                          first_menu_fare_contact=first(lambda s:any(p==44 for p in s['prices'])),
                          first_menu_zero_wait=first(lambda s:s['wait']<=1e-8),
                          first_menu_zero_wait_at_capacity=first(lambda s:s['wait']<=1e-8 and abs(s['rho']-1)<1e-8),
                          first_menu_slack=first(lambda s:s['rho']<1-1e-8),
                          first_menu_withdrawal=first(lambda s:not all(s['active'])),
                          continuous_withdrawal_thresholds=[(44-c)/a for a,c in
                              zip(case['primitives']['resource'],case['primitives']['costs'])],
                          continuous_fare_or_release_threshold_claim=False))
    return dict(target_games=5,reused_supported_responses=39,total_menu_responses=44,
                bounded_active_coordinates=sum(len(r['proof']['coordinates']) for r in rows),
                excluded_coordinates=sum(len(r['proof']['noentry']) for r in rows),
                maximum_gain_upper=max(r['proof']['maximum_gain_upper'] for r in rows),
                epsilon_nash_closed_games=sum(r['proof']['epsilon_nash_closed'] for r in rows),
                exact_nash_claim=False,all_equilibria_enumerated=False,cases=cases,
                scope='All 11 charges per selected-response menu; not the full Nash correspondence')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected(root):
    paths=[]
    for folder in ('experiments/aam_pricing','scripts','tests',OLD,
                   'results/trb_review11_finite_paths_v1','results/trb_review13_equilibrium_v1',
                   'submission_packages/TRA_V290_LaTeX_Project'):
        paths.extend(p for p in (Path(root)/folder).glob('*') if p.is_file() and
                     'review18_capped' not in p.as_posix())
    paths.extend(p for p in Path(root).glob('MAIN.*') if p.is_file())
    return {p.relative_to(root).as_posix():digest(p) for p in paths}


def run(root,out):
    root,out=Path(root),Path(out)
    if (out/'production_started.json').exists():
        raise FileExistsError('One-shot production already started; replay only')
    out.mkdir(parents=True,exist_ok=True)
    d=design(root)
    if (out/'inputs.json').exists() and read(out/'inputs.json')!=d:
        raise ValueError('Prepared design changed')
    write(out/'inputs.json',d)
    before=protected(root); write(out/'historical_before.json',before)
    sourcefiles=['experiments/aam_pricing/review18_capped_actual.py',
                 'scripts/run_review18_capped_actual.py','tests/test_review18_capped_actual.py',
                 'experiments/aam_pricing/review11_finite_paths.py',
                 'experiments/aam_pricing/review13_equilibrium.py',
                 'experiments/aam_pricing/review15_capped.py']
    write(out/'source_manifest.json',{p:digest(root/p) for p in sourcefiles})
    cases={c['id']:c['primitives'] for c in d['cases']}
    old=read(root/OLD/'points.json'); rows=[]
    start=time.monotonic(); deadline=start+600
    write(out/'production_started.json',dict(max_games=5,max_seconds=600,
         reserved_for_tests_replay_and_initialization=300,
         initialization_note='First recursive archival hash enumeration interrupted before any game; scoped read-only hashes instead'))
    try:
        for target in d['targets']:
            budget(deadline); case_id,q=target['case_id'],target['charge']; c=cases[case_id]
            previous=next(r['state'] for r in old if r['case_id']==case_id and r['state']['charge']==q)
            active=np.asarray(previous['active'],bool)
            p=[v if v is not None else c['fare_bounds'][1] for v in previous['prices']]
            t=time.monotonic(); p=solve(c,q,p,active,deadline); s=state(c,q,p,active)
            proof=bound_game(c,q,p,active,deadline)
            rows.append(dict(case_id=case_id,state=s,proof=proof,elapsed_seconds=time.monotonic()-t,
                             auxiliary_state=previous,
                             search='One historical start; actual coordinate responses over cubic regime segments',
                             breakpoints_at_candidate={str(m):regime_breakpoints(c,p,active,m)
                                                       for m in np.flatnonzero(active)}))
            write(out/'points.json',rows)
    finally:
        write(out/'points.json',rows)
        after={p:digest(root/p) for p in before}
        write(out/'historical_preservation.json',dict(files=len(before),unchanged=before==after,
              changed=[p for p in before if before[p]!=after[p]]))
        write(out/'execution.json',dict(elapsed_seconds=time.monotonic()-start,attempted_games=len(rows),
                                       max_seconds=600,expansion=False))
    summary=summarize(root,rows); write(out/'summary.json',summary)
    write(out/'result_manifest.json',{p:digest(out/p) for p in
          ('inputs.json','points.json','summary.json','execution.json','historical_before.json','historical_preservation.json')})
    return summary


def replay(root,out):
    root,out=Path(root),Path(out)
    for p,h in read(out/'result_manifest.json').items():
        if digest(out/p)!=h: raise ValueError('Result hash mismatch: '+p)
    for p,h in read(out/'source_manifest.json').items():
        if digest(root/p)!=h: raise ValueError('Source hash mismatch: '+p)
    d=design(root)
    if read(out/'inputs.json')!=d: raise ValueError('Frozen design mismatch')
    cases={c['id']:c['primitives'] for c in d['cases']}; rows=read(out/'points.json')
    count=0
    for r in rows:
        s=r['state']; c=cases[r['case_id']]
        p=[44 if x is None else x for x in s['prices']]
        replay_bounds(c,s['charge'],p,s['active'],r['proof']); count+=len(r['proof']['coordinates'])
        o=cp.choice(c,p,s['active'],s['rho'],s['wait'])
        for key in ('demand','net_surplus','probabilities'):
            if not np.allclose(o[key],s[key],atol=1e-10,rtol=1e-12): raise ValueError('State arithmetic')
        work=float(np.asarray(c['resource'])@o['demand'])
        if abs(work-c['capacity']*s['rho'])>1e-7: raise ValueError('Closure feasibility')
    summary=summarize(root,rows)
    if summary!=read(out/'summary.json'): raise ValueError('Menu summary mismatch')
    return dict(valid=True,games=5,active_coordinates=count,optimizations_called=0,
                scalar_roots_called=0,summary=summary)
