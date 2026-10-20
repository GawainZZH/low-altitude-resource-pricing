"""Whole-fare profit enclosures using scalar closure and implicit derivatives.

No concavity or completeness of sampled stationary roots is assumed. All
coefficients are the exact effective binary primitives of the numerical model.
"""
from decimal import Decimal
import heapq
import hashlib
import json
import math
from pathlib import Path
import shutil
from time import perf_counter

from .review13_equilibrium import Interval as I, UP, DOWN, NEAR, midpoint


def hull(values):
    return I(min(x.lo for x in values),max(x.hi for x in values))


def positive(x):
    return I(max(Decimal(0),x.lo),max(Decimal(0),x.hi))


def heap_key(value):
    return value.copy_negate()


def cover_radius(lo,hi,center):
    return max(UP.subtract(center,lo),UP.subtract(hi,center))


def power(x,n):
    if n.lo == n.hi == 0:
        return I(1)
    if x.lo == x.hi == 0:
        return I(0)
    if x.lo == 0:
        return I(0,(I(x.hi).log()*n).exp().hi)
    return (x.log()*n).exp()


class Problem:
    def __init__(self,case,excluded=()):
        self.case=dict(case)
        self.active=[i for i in range(3) if i not in excluded]
        self.excluded=list(excluded)
        self.effective=dict(D=[float(v*case['demand_scale']) for v in case['base_demand']],
            K=float(case['base_capacity']*case['capacity_scale']),
            alpha=[[float(v+(case['differentiation'] if i==g else 0.)) for i,v in enumerate(row)]
                   for g,row in enumerate(case['platform_affinity'])],
            cost=[float(v+d) for v,d in zip(case['base_cost'],(0.,case['cost_gap'],.5*case['cost_gap']))])
        self.D=list(map(I,self.effective['D'])); self.K=I(self.effective['K'])
        self.a=list(map(I,case['platform_resource'])); self.chi=list(map(I,case['exposure']))
        self.cost=list(map(I,self.effective['cost'])); self.mu=I(case['logit_scale'])
        self.lam=list(map(I,case['price_sensitivity'])); self.beta=list(map(I,case['time_sensitivity']))
        self.gamma=list(map(I,case['reliability_sensitivity'])); self.nu=I(case['reliability_power'])
        values=[v for row in self.effective['alpha'] for v in row]+self.effective['D']+[
            self.effective['K'],*self.effective['cost'],case['logit_scale'],case['reliability_power'],*case['platform_resource'],
            *case['time_sensitivity'],*case['price_sensitivity'],*case['reliability_sensitivity'],
            *case['exposure'],*case['base_loss'],*case['service_time'],*case['outside_affinity'],
            case['outside_price'],case['outside_time'],case['price_lower'],case['price_upper']]
        if (not self.active or not all(math.isfinite(v) for v in values)
                or min(self.effective['D']+[self.effective['K'],case['logit_scale'],
                    *case['platform_resource'],*case['time_sensitivity'],*case['price_sensitivity']])<=0
                or case['reliability_power']<1 or min(case['exposure']+case['reliability_sensitivity'])<0):
            raise ValueError('Invalid positive primitives for closure')
        adverse=sum((positive(-(self.a[i]-self.a[j])*(self.chi[i]-self.chi[j]))
                     for i in self.active for j in self.active if i<j),I(0))/4
        hp=I(.55)*self.nu+I(2.20)*positive(I(1)-I(.88))
        self.closure_slope_upper=hp*adverse*sum((d*g/self.mu for d,g in zip(self.D,self.gamma)),I(0))/self.K
        self.unique_closure=self.closure_slope_upper.hi<1
        if not self.unique_closure:
            raise ValueError('Unproved unique scalar closure')
        self._cache={}

    def h(self,r):
        extra=positive(r-I(.88))
        return I(.55)*power(r,self.nu)+I(1.10)*extra*extra

    def hp(self,r):
        return I(.55)*self.nu*power(r,self.nu-1)+I(2.20)*positive(r-I(.88))

    def base(self,p,rho):
        h=self.h(rho); rows=[]
        for g in range(3):
            outside=(I(self.case['outside_affinity'][g])-self.lam[g]*self.case['outside_price']
                     -self.beta[g]*self.case['outside_time'])/self.mu
            weights={j:((I(self.effective['alpha'][g][j])-self.lam[g]*p[j]
                -self.beta[g]*self.case['service_time'][j]
                -self.gamma[g]*(I(self.case['base_loss'][j])+self.chi[j]*h))/self.mu-outside).exp()
                for j in self.active}
            total=sum(weights.values(),I(0)); shares=[]
            for j in range(3):
                if j not in self.active:
                    shares.append(I(0)); continue
                lo=I(weights[j].lo)/(I(weights[j].lo)+sum((I(w.hi) for k,w in weights.items() if k!=j),I(0)))
                hi=I(weights[j].hi)/(I(weights[j].hi)+sum((I(w.lo) for k,w in weights.items() if k!=j),I(0)))
                shares.append(I(max(Decimal(0),lo.lo),min(Decimal(1),hi.hi)))
            average=sum((self.a[j]*shares[j] for j in self.active),I(0))
            average=I(max(average.lo,min(self.a[j].lo for j in self.active)),
                      min(average.hi,max(self.a[j].hi for j in self.active)))
            rows.append((total,shares,average))
        return rows

    def evaluate(self,base,wait):
        probabilities=[]; thetas=[]; work=I(0)
        for g,(total,shares,average) in enumerate(base):
            z=(-self.beta[g]*wait/self.mu).exp(); odds=total*z
            theta=I((I(odds.lo)/(1+I(odds.lo))).lo,(I(odds.hi)/(1+I(odds.hi))).hi)
            theta=I(max(Decimal(0),theta.lo),min(Decimal(1),theta.hi))
            probabilities.append([theta*s for s in shares]); thetas.append(theta)
            work+=self.D[g]*theta*average
        y=[sum((self.D[g]*probabilities[g][j] for g in range(3)),I(0)) for j in range(3)]
        return dict(prob=probabilities,theta=thetas,demand=y,work=work,base=base)

    def enclose(self,prices,point=False):
        key=(tuple((p.lo,p.hi) for p in prices),point)
        if key in self._cache:
            return self._cache[key]
        base1=self.base(prices,I(1)); zero=self.evaluate(base1,I(0))
        result=[]; iterations=54 if point else 27
        if zero['work'].lo<=self.K.hi:
            ll,lh,ul,uh=Decimal(0),Decimal(1),Decimal(0),Decimal(1)
            for _ in range(iterations):
                m=NEAR.divide(NEAR.add(ll,lh),Decimal(2))
                v=self.evaluate(self.base(prices,I(m)),I(0))['work']-self.K*I(m)
                if v.lo>0: ll=m
                else: lh=m
                m=NEAR.divide(NEAR.add(ul,uh),Decimal(2))
                v=self.evaluate(self.base(prices,I(m)),I(0))['work']-self.K*I(m)
                if v.hi<0: uh=m
                else: ul=m
            rho=I(ll,uh)
            state=self.evaluate(self.base(prices,rho),I(0))
            state.update(regime='free',rho=rho,wait=I(0)); result.append(state)
        if zero['work'].hi>=self.K.lo:
            upper=Decimal(1)
            for _ in range(40):
                if self.evaluate(base1,I(upper))['work'].hi<self.K.lo: break
                upper=UP.multiply(upper,Decimal(2))
            else: raise ValueError('No finite wait enclosure')
            ll,lh,ul,uh=Decimal(0),upper,Decimal(0),upper
            for _ in range(iterations):
                m=NEAR.divide(NEAR.add(ll,lh),Decimal(2))
                if self.evaluate(base1,I(m))['work'].lo>self.K.hi: ll=m
                else: lh=m
                m=NEAR.divide(NEAR.add(ul,uh),Decimal(2))
                if self.evaluate(base1,I(m))['work'].hi<self.K.lo: uh=m
                else: ul=m
            wait=I(ll,uh); state=self.evaluate(base1,wait)
            state.update(regime='binding',rho=I(1),wait=wait); result.append(state)
        if not result: raise ValueError('Empty closure cover')
        self._cache[key]=result
        return result

    def profit(self,p,q,j,point=False):
        margin=p[j]-self.cost[j]-I(q)*self.a[j]
        return hull([margin*s['demand'][j] for s in self.enclose(p,point)])

    def welfare(self,p,kappa=0.,point=False):
        values=[]
        for s in self.enclose(p,point):
            cs=I(0)
            for g,(total,_,_) in enumerate(s['base']):
                outside=(I(self.case['outside_affinity'][g])-self.lam[g]*self.case['outside_price']
                         -self.beta[g]*self.case['outside_time'])/self.mu
                odds=total*(-self.beta[g]*s['wait']/self.mu).exp()
                cs+=self.D[g]*self.mu/self.lam[g]*(outside+(1+odds).log())
            excess=positive(s['rho']-I(.9))
            external=I(self.case['congestion_cost'])*(I(.15)*s['rho']*s['rho']+excess*excess*excess)
            value=cs+sum(((p[j]-self.cost[j])*s['demand'][j] for j in self.active),I(0))-external
            value+=I(kappa)*I(self.case['outside_price'])*(sum(self.D,I(0))-sum(s['demand'],I(0)))
            values.append(value)
        return hull(values)

    def gradient(self,p,q,j):
        gradients=[]
        for s in self.enclose(p):
            prob,theta=s['prob'],s['theta']
            direct=-sum((self.D[g]*self.lam[g]/self.mu*prob[g][j]*(1-prob[g][j]) for g in range(3)),I(0))
            work_price=-sum((self.D[g]*self.lam[g]/self.mu*prob[g][j]
                *(self.a[j]-sum((self.a[k]*prob[g][k] for k in self.active),I(0))) for g in range(3)),I(0))
            if s['regime']=='binding':
                wait_j=-sum((self.D[g]*self.beta[g]/self.mu*prob[g][j]*(1-theta[g]) for g in range(3)),I(0))
                wait_R=-sum((self.D[g]*self.beta[g]/self.mu*theta[g]*(1-theta[g])*s['base'][g][2]
                    for g in range(3)),I(0))
                dy=direct-wait_j*work_price/wait_R
            else:
                partial=[]
                for k in range(3):
                    partial.append(-self.hp(s['rho'])*sum((self.D[g]*self.gamma[g]/self.mu*prob[g][k]
                        *(self.chi[k]-sum((self.chi[h]*prob[g][h] for h in self.active),I(0)))
                        for g in range(3)),I(0)))
                Rrho=sum((self.a[k]*partial[k] for k in self.active),I(0))
                Rrho=I(Rrho.lo,min(Rrho.hi,(self.K*self.closure_slope_upper).hi))
                dy=direct+partial[j]*work_price/(self.K-Rrho)
            gradients.append(s['demand'][j]+(p[j]-self.cost[j]-I(q)*self.a[j])*dy)
        return hull(gradients)

    def leaf(self,fares,q,j,lo,hi):
        p=[I(v) for v in fares]; p[j]=I(lo,hi)
        direct=self.profit(p,q,j); maximum=direct.hi
        midpoint_price=midpoint(p[j]); center=p.copy(); center[j]=I(midpoint_price)
        value=self.profit(center,q,j,point=True)
        method='direct'; derivative=None
        try:
            derivative=self.gradient(p,q,j)
            if derivative.lo>=0 or derivative.hi<=0:
                end=p.copy(); end[j]=I(hi if derivative.lo>=0 else lo)
                endpoint=self.profit(end,q,j,point=True)
                maximum=min(maximum,endpoint.hi); method='increasing' if derivative.lo>=0 else 'decreasing'
            else:
                radius=I(cover_radius(lo,hi,midpoint_price))
                taylor=I(value.hi)+I(max(derivative.lo.copy_abs(),derivative.hi.copy_abs()))*radius
                maximum=min(maximum,taylor.hi); method='mean_value'
        except ValueError:
            pass
        return dict(fare=[str(lo),str(hi)],upper=str(maximum),direct=direct.serialized(),
            gradient=derivative.serialized() if derivative else None,method=method,
            center=str(midpoint_price),center_profit=value.serialized())

    def unilateral(self,fares,q,j,seconds=60.,absolute=.01,relative=1e-5,max_leaves=600):
        started=perf_counter(); domain=(I(self.case['price_lower']).lo,I(self.case['price_upper']).hi)
        base=self.profit([I(v) for v in fares],q,j,point=True)
        denominator=max(Decimal(1),Decimal(0) if base.lo<=0<=base.hi else min(base.lo.copy_abs(),base.hi.copy_abs()))
        tolerance=min(Decimal(str(absolute)),DOWN.multiply(Decimal(str(relative)),denominator))
        leaves=[]; serial=0
        first=self.leaf(fares,q,j,*domain); heapq.heappush(leaves,(heap_key(Decimal(first['upper'])),serial,first))
        best=base.lo; best_fare=I(fares[j]).lo
        while True:
            upper=heap_key(leaves[0][0])
            if UP.subtract(upper,base.lo)<=tolerance: reason='small_deviation'; break
            if UP.subtract(upper,best)<=tolerance: reason='profit_maximum_resolved'; break
            if perf_counter()-started>=seconds: reason='time_budget'; break
            if len(leaves)>=max_leaves: reason='leaf_budget'; break
            old=leaves[0][2]; lo,hi=map(Decimal,old['fare']); cut=NEAR.divide(NEAR.add(lo,hi),Decimal(2))
            children=[self.leaf(fares,q,j,lo,cut),self.leaf(fares,q,j,cut,hi)]
            heapq.heappop(leaves)
            for child in children:
                serial+=1; heapq.heappush(leaves,(heap_key(Decimal(child['upper'])),serial,child))
                if Decimal(child['center_profit'][0])>best:
                    best=Decimal(child['center_profit'][0]); best_fare=Decimal(child['center'])
        upper=heap_key(leaves[0][0]); gain=max(Decimal(0),UP.subtract(upper,base.lo))
        result=dict(platform=j,charge=q,prices=list(fares),baseline=base.serialized(),
            profit_upper=str(upper),gain_upper=str(gain),normalized_upper=str(UP.divide(gain,denominator)),
            observed_gain_lower=str(max(Decimal(0),DOWN.subtract(best,base.hi))),
            best_fare=str(best_fare),profit_maximum_error=str(max(Decimal(0),UP.subtract(upper,best))),
            tolerance=str(tolerance),closed=gain<=tolerance,stop=reason,
            seconds=perf_counter()-started,closure_slope_upper=str(self.closure_slope_upper.hi),
            coverage_complete=True,leaves=sorted((v[2] for v in leaves),key=lambda r:Decimal(r['fare'][0])))
        assert verify_partition(result['leaves'],*domain)
        return result


def verify_partition(leaves,lower,upper):
    if not leaves: return False
    endpoints=[tuple(map(Decimal,r['fare'])) for r in leaves]
    return (endpoints[0][0]==Decimal(str(lower)) and endpoints[-1][1]==Decimal(str(upper))
        and all(a<b for a,b in endpoints) and all(a[1]==b[0] for a,b in zip(endpoints,endpoints[1:])))


MENUS=('resource_2_c0','resource_3_c0','joint_3_c1')
INPUT='results/trb_review14_policy_v1/normalized_menus.json'
OWNED=('experiments/aam_pricing/review15_bounds.py','scripts/run_review15_bounds.py','tests/test_review15_bounds.py')


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(root,folder,pilot=False):
    root,folder=Path(root),Path(folder)
    folder.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(root/INPUT,folder/'inputs.json')
    data=json.loads((folder/'inputs.json').read_text())
    design=dict(menus=list(MENUS),pilot=pilot,absolute_tolerance=.01,relative_tolerance=1e-5,
        coordinate_seconds=85 if pilot else 60,batch_seconds=180 if pilot else 1800,
        max_leaves=600,fare_perturbation=1e-6,all_Nash_characterized=False,
        inference='Full-domain gain enclosures at selected numerical responses; not a welfare bound over all epsilon-Nash points')
    dump(folder/'design.json',design)
    started=perf_counter(); rows=[]; states=[]; pending=[]
    for menu in data:
        if menu['id'] not in MENUS: continue
        for row in menu['candidates']:
            problem=Problem(menu['case'],row['excluded'])
            selected=list(problem.active)
            if pilot:
                if menu['id']!=MENUS[0] or row['charge'] not in (0.,2.): continue
                selected=[0]
            for j in selected:
                remaining=design['batch_seconds']-(perf_counter()-started)
                if remaining<=0:
                    pending.append(dict(menu=menu['id'],charge=row['charge'],platform=j)); continue
                result=problem.unilateral(row['prices'],row['charge'],j,
                    seconds=min(design['coordinate_seconds'],remaining),max_leaves=design['max_leaves'])
                result.update(menu=menu['id'],excluded=row['excluded'])
                rows.append(result)
                dump(folder/'coordinates.json',rows)
                print(json.dumps({k:result[k] for k in ('menu','charge','platform','gain_upper','normalized_upper','closed','stop','seconds')}),flush=True)
            p=[I(v) for v in row['prices']]
            welfare=problem.welfare(p,point=True)
            perturbed=[I(max(menu['case']['price_lower'],v-1e-6),min(menu['case']['price_upper'],v+1e-6)) for v in row['prices']]
            state=dict(menu=menu['id'],charge=row['charge'],prices=row['prices'],excluded=row['excluded'],
                welfare=welfare.serialized(),perturbed_welfare=problem.welfare(perturbed).serialized(),
                no_entry=all(menu['case']['price_upper']-problem.effective['cost'][j]-row['charge']*menu['case']['platform_resource'][j]<=0 for j in row['excluded']))
            states.append(state)
    summary=summarize(rows,states,pending)
    dump(folder/'coordinates.json',rows); dump(folder/'states.json',states)
    dump(folder/'pending.json',pending); dump(folder/'summary.json',summary)
    for name in OWNED+('experiments/aam_pricing/review13_equilibrium.py',):
        target=folder/'execution_source'/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/name,target)
    files={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file()}
    dump(folder/'manifest.json',dict(files=files,source_sha256={n:sha(root/n) for n in OWNED},parent_input=INPUT,
        parent_input_sha256=sha(root/INPUT),production_seconds=perf_counter()-started))
    return summary


def summarize(rows,states,pending):
    rankings=[]
    for menu in MENUS:
        group=[s for s in states if s['menu']==menu]
        if not group: continue
        best=max(group,key=lambda s:(Decimal(s['welfare'][0]),-s['charge']))
        others=[s for s in group if s is not best]
        margin=DOWN.subtract(Decimal(best['perturbed_welfare'][0]),max((Decimal(s['perturbed_welfare'][1]) for s in others),default=Decimal('-Infinity')))
        # Singleton pilot panels have no tariff-ranking interpretation.
        rankings.append(dict(menu=menu,states=len(group),selected_charge=best['charge'],
            separation_under_fare_perturbation=str(margin) if others else None,
            stable=bool(others and margin>0),comparison_scope='Selected saved candidate responses, not every approximate Nash selection'))
    return dict(coordinates=len(rows),closed_coordinates=sum(r['closed'] for r in rows),
        maximum_gain_upper=max((r['gain_upper'] for r in rows),key=Decimal,default=None),
        maximum_normalized_upper=max((r['normalized_upper'] for r in rows),key=Decimal,default=None),
        profitable_deviation_coordinates=sum(Decimal(r['observed_gain_lower'])>Decimal(r['tolerance']) for r in rows),
        pending_coordinates=len(pending),rankings=rankings,
        new_Nash_searches=0,interval_closure_enclosures=True,full_fare_domain=[5.,44.],
        scope='Directed interval derivative/closure proof arithmetic; no quasiconcavity or global welfare-error implication')


def replay(folder):
    folder=Path(folder); manifest=json.loads((folder/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        path=folder/name
        if not path.is_file() or sha(path)!=digest: raise ValueError('Evidence hash mismatch: '+name)
    source_name='experiments/aam_pricing/review15_bounds.py'
    if source_name in manifest.get('source_sha256',{}) and sha(Path(__file__))!=manifest['source_sha256'][source_name]:
        raise ValueError('Reviewed bound source hash differs')
    data={m['id']:m for m in json.loads((folder/'inputs.json').read_text())}
    rows=json.loads((folder/'coordinates.json').read_text())
    for row in rows:
        problem=Problem(data[row['menu']]['case'],row['excluded'])
        if not verify_partition(row['leaves'],5,44): raise ValueError('Incomplete full-fare cover')
        for leaf in row['leaves']:
            actual=problem.leaf(row['prices'],row['charge'],row['platform'],*map(Decimal,leaf['fare']))
            if actual!=leaf: raise ValueError('Interval arithmetic differs at saved leaf')
        baseline=problem.profit([I(p) for p in row['prices']],row['charge'],row['platform'],point=True)
        if baseline.serialized()!=row['baseline']: raise ValueError('Baseline closure enclosure differs')
        upper=max(Decimal(r['upper']) for r in row['leaves'])
        if str(upper)!=row['profit_upper']: raise ValueError('Full-domain maximum upper differs')
    states=json.loads((folder/'states.json').read_text())
    for state in states:
        case=data[state['menu']]['case']; problem=Problem(case,state['excluded'])
        value=problem.welfare([I(v) for v in state['prices']],point=True)
        perturbed=[I(max(case['price_lower'],v-1e-6),min(case['price_upper'],v+1e-6)) for v in state['prices']]
        if value.serialized()!=state['welfare'] or problem.welfare(perturbed).serialized()!=state['perturbed_welfare']:
            raise ValueError('Welfare enclosure differs')
    result=summarize(rows,states,json.loads((folder/'pending.json').read_text()))
    if result!=json.loads((folder/'summary.json').read_text()): raise ValueError('Bound summary differs')
    return result
