"""Bounded fixed-quotation fluid transitions with entry-cohort welfare accrual."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil

from scipy.integrate import solve_ivp

from experiments.aam_pricing import review17_fluid_validity as old

ROOT = Path(__file__).resolve().parents[2]
DURATIONS = [30., 60., 120., 240., 480., 960.]
OWNED = ('experiments/aam_pricing/review18_transition.py',
         'scripts/run_review18_transition.py', 'tests/test_review18_transition.py')


def integrate_workload(work, welfare, capacity, initial_wait, duration, horizon,
                       stationary_welfare, stationary_wait, rtol=2e-11):
    """Integrate until the reflected queue empties, then accrue the slack state."""
    if min(capacity, duration, horizon) <= 0 or min(initial_wait, stationary_wait) < 0:
        raise ValueError('Positive scales and nonnegative waits required')
    if initial_wait == 0 and work(0.) <= capacity:
        return dict(end_wait=0., welfare_integral=stationary_welfare*duration/horizon,
                    clearance_minutes=0., busy_duration=0.)

    def rhs(t, z):
        w = max(0., z[0])
        return [(work(w)-capacity)/capacity, welfare(w)/horizon]

    def empty(t, z):
        return z[0]
    empty.terminal = True
    empty.direction = -1
    sol = solve_ivp(rhs, (0., duration), [initial_wait, 0.], method='DOP853',
                    rtol=rtol, atol=[1e-11, 1e-9], events=empty, max_step=10.)
    if not sol.success:
        raise ArithmeticError(sol.message)
    clear = float(sol.t[-1]) if len(sol.t_events[0]) else None
    value = float(sol.y[1, -1])
    if clear is not None:
        value += stationary_welfare*(duration-clear)/horizon
    return dict(end_wait=0. if clear is not None else float(sol.y[0, -1]),
                welfare_integral=value, clearance_minutes=clear,
                busy_duration=duration if clear is None else clear)


def load_inputs(root):
    return old.payload(Path(root))


def metrics(c, fares, rho, wait):
    base = old.choice(c, fares, rho, wait)
    mu = c['logit_scale']
    h = .55*rho**c['reliability_power']+1.10*max(rho-.88, 0.)**2
    loss = [b+e*h for b,e in zip(c['base_loss'], c['exposure'])]
    waiting_cost = 0.
    for g in range(3):
        lam, beta, gamma = [c[k][g] for k in
                            ('price_sensitivity', 'time_sensitivity', 'reliability_sensitivity')]
        u = [(c['platform_affinity'][g][j]+(c['differentiation'] if j == g else 0.)
              -lam*fares[j]-beta*(c['service_time'][j]+wait)-gamma*loss[j])/mu
             for j in range(3)]
        u0 = (c['outside_affinity'][g]-lam*c['outside_price']-beta*c['outside_time'])/mu
        peak = max(u0, *u)
        aerial = sum(math.exp(z-peak) for z in u)
        aerial /= aerial+math.exp(u0-peak)
        waiting_cost += c['base_demand'][g]*c['demand_scale']*beta/lam*aerial*wait
    return dict(**base, waiting_cost=waiting_cost, passengers=sum(base['demand']))


def discrepancy_bound(c, fares, w0, state, duration):
    """Conservative exact-form bound, not an observed adjustment-error estimate."""
    masses = [z*c['demand_scale'] for z in c['base_demand']]
    b = c['time_sensitivity']
    costs = [z+v for z,v in zip(c['base_cost'], [0., c['cost_gap'], .5*c['cost_gap']])]
    lipschitz = sum(d*x/l for d,x,l in zip(masses,b,c['price_sensitivity']))
    lipschitz += sum(abs(p-v) for p,v in zip(fares,costs))*sum(d*x for d,x in zip(masses,b))/(4*c['logit_scale'])
    k = c['base_capacity']*c['capacity_scale']
    ws = state['wait_minutes']
    busy_zero = old.choice(c,fares,1.,0.)
    if ws > 1e-7 or (state['rho'] >= 1-1e-9 and abs(k-busy_zero['work']) < 1e-7):
        lo, hi = sorted([w0, ws])
        left, right = old.choice(c,fares,1.,lo), old.choice(c,fares,1.,hi)
        m = sum(masses[g]*b[g]/c['logit_scale']*left['outside'][g]*right['group_work'][g] for g in range(3))
        rate = m/k
        bound = lipschitz*abs(w0-ws)*(-math.expm1(-rate*duration))/(rate*duration)
        return dict(normalized_absolute_error_bound=bound, derivative_bound=lipschitz,
                    contraction_rate=rate, clearance_upper_minutes=None)
    busy = busy_zero
    gap = k-busy['work']
    star = old.choice(c,fares,state['rho'],ws)['ts']
    if gap <= 0:
        raise ValueError('Finite menu has no strictly slack endpoint for this bound')
    upper = k*w0/gap
    amplitude = abs(busy['ts']-star)+lipschitz*w0
    return dict(normalized_absolute_error_bound=amplitude*min(duration,upper)/duration,
                derivative_bound=lipschitz, contraction_rate=None,
                clearance_upper_minutes=upper)


def compute(data):
    rows, decisions = [], []
    for c in data['cases']:
        profile = [c[k] for k in ('demand_scale','capacity_scale','reliability_power')]
        menu = sorted([z for z in data['fluid'] if z['profile'] == profile],key=lambda z:z['charge'])
        if [z['charge'] for z in menu] != [0.,4.,8.,12.,16.,20.]:
            raise ValueError('Exactly the existing six-charge menus required')
        nc = menu[0]
        initial = nc['state']['wait_minutes']
        k = c['base_capacity']*c['capacity_scale']
        baseline = metrics(c,nc['fares'],nc['state']['rho'],initial)
        for z in menu:
            fares, state = z['fares'],z['state']
            star = metrics(c,fares,state['rho'],state['wait_minutes'])
            busy = lambda w: metrics(c,fares,1.,w)
            for duration in DURATIONS:
                out = integrate_workload(lambda w:busy(w)['work'],lambda w:busy(w)['ts'],k,
                                         initial,duration,old.H,star['ts'],state['wait_minutes'])
                waiting = integrate_workload(lambda w:busy(w)['work'],lambda w:busy(w)['waiting_cost'],k,
                                         initial,duration,old.H,star['waiting_cost'],state['wait_minutes'])
                gain = out['welfare_integral']-baseline['ts']*duration/old.H
                normalized = gain*old.H/duration
                error = normalized-(star['ts']-baseline['ts'])
                bound = discrepancy_bound(c,fares,initial,state,duration)
                if abs(error) > bound['normalized_absolute_error_bound']+1e-5:
                    raise ArithmeticError('Transition discrepancy exceeds proven conservative bound')
                rows.append(dict(profile=c['name'],charge=z['charge'],duration=duration,
                    initial_wait=initial,stationary_wait=state['wait_minutes'],
                    stationary_gain=star['ts']-baseline['ts'],gain=gain,
                    normalized_gain_per_H=normalized,stationary_approximation_error=error,
                    entry_waiting_cost=waiting['welfare_integral'],legacy_waiting_difference=0.,
                    **out,**bound))
        for duration in DURATIONS:
            candidates = [r for r in rows if r['profile']==c['name'] and r['duration']==duration]
            best = max(candidates,key=lambda r:r['gain'])
            steady = max(candidates,key=lambda r:r['stationary_gain'])
            decisions.append(dict(profile=c['name'],duration=duration,best_charge=best['charge'],
                gain=best['gain'],normalized_gain=best['normalized_gain_per_H'],
                stationary_charge=steady['charge'],stationary_gain=steady['stationary_gain'],
                implementation_gain_at_stationary_charge=steady['gain'],
                stationary_charge_gain_fraction=(steady['normalized_gain_per_H']/steady['stationary_gain']
                                                if steady['stationary_gain'] else None)))
    return dict(durations_minutes=DURATIONS,evaluations=rows,decisions=decisions,
                new_Nash_solves=0,fare_adjustment='Immediate fixed stationary quotations; no transient Nash claim',
                welfare_accrual='Joining-cohort utility and operating accounts including future access wait; calendar system cost only within the appraisal window',
                legacy_assumption='Same initial FIFO cohort, constant processing rate, no reneging or priority; its waiting costs cancel')


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_results(root, output):
    root, output = Path(root),Path(output)
    if output.exists():
        raise FileExistsError(output)
    data = load_inputs(root)
    result = compute(data)
    output.mkdir(parents=True)
    for name,value in [('inputs.json',data),('results.json',result)]:
        (output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for source in OWNED:
        target = output/'execution_source'/source
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/source,target)
    manifest = {str(p.relative_to(output)).replace('\\','/'):digest(p) for p in output.rglob('*') if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return result


def replay(folder):
    folder = Path(folder)
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if any(digest(folder/p)!=h for p,h in manifest.items()):
        raise ValueError('Frozen payload mismatch')
    data = json.loads((folder/'inputs.json').read_text(encoding='utf-8'))
    saved = json.loads((folder/'results.json').read_text(encoding='utf-8'))
    if compute(data) != saved:
        raise ValueError('Transition arithmetic replay mismatch')
    return dict(verified=True,evaluations=72,new_Nash_solves=0)
