"""Fixed-fare fluid relaxation and frozen-flow Poisson diagnostics, stdlib only."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil

SOURCE = 'results/trb_review7_queue_v1'
INPUTS = (SOURCE+'/static_reference_inputs.json', SOURCE+'/raw.json')
OWNED = ('experiments/aam_pricing/review17_fluid_validity.py',
         'scripts/run_review17_fluid_validity.py', 'tests/test_review17_fluid_validity.py')
H = 120.


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positive(*values):
    if any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError('Finite positive scales required')


def relaxation(horizon, capacity, work_slope):
    positive(horizon, capacity, -work_slope)
    normalized = work_slope*horizon/capacity
    return dict(minutes=capacity/(-work_slope), normalized_slope=normalized,
                via_normalized_minutes=horizon/(-normalized))


def exp_bound(initial_error, tolerance, rate):
    positive(tolerance, rate)
    if initial_error < 0 or not math.isfinite(initial_error):
        raise ValueError('Nonnegative finite initial error required')
    return max(0., math.log(max(initial_error/tolerance, 1.))/rate)


def pk(y, a, capacity, horizon, cv2):
    positive(capacity, horizon)
    if len(y) != len(a) or not y or any(z < 0 or not math.isfinite(z) for z in y):
        raise ValueError('Nonnegative finite flow vector required')
    positive(*a)
    if cv2 < 0 or not math.isfinite(cv2):
        raise ValueError('Finite nonnegative conditional variance required')
    total = math.fsum(y)
    work = math.fsum(v*b for v, b in zip(y, a))
    second = (1+cv2)*math.fsum(v*b*b for v, b in zip(y, a))
    rho = work/capacity
    return dict(rho=rho, status='stable' if rho < 1 else 'unstable',
                wait_minutes=horizon*second/(2*capacity**2*(1-rho)) if rho < 1 else None,
                arrivals_per_minute=total/horizon, arrival_count_variance_per_H=total,
                work_variance_per_H=second,
                work_cv_per_H=math.sqrt(second)/work if work else 0.,
                service_means_minutes=[horizon*b/capacity for b in a],
                pooled_service_cv2=second*total/work**2-1 if work else 0.)


def payment_equivalence(a, value, wait):
    positive(*a, value)
    if wait < 0:
        raise ValueError('Nonnegative wait required')
    r = value*wait
    return dict(common_request_payment=r, resource_unit_rates=[r/x for x in a],
                single_resource_rate_equivalent=(r == 0 or max(a) == min(a)))


def choice(c, fares, rho, wait):
    """Native saved logit specification on 0<=rho<=1, without closure solving."""
    if not 0 <= rho <= 1 or wait < 0:
        raise ValueError('Invalid fixed state')
    mu = c['logit_scale']
    h = .55*rho**c['reliability_power']+1.10*max(rho-.88, 0.)**2
    loss = [b+e*h for b, e in zip(c['base_loss'], c['exposure'])]
    probs, outside, logs = [], [], []
    for g in range(3):
        lam, beta, gamma = (c[k][g] for k in
                            ('price_sensitivity', 'time_sensitivity', 'reliability_sensitivity'))
        u = [(c['platform_affinity'][g][j]+(c['differentiation'] if j == g else 0.)
              -lam*fares[j]-beta*(c['service_time'][j]+wait)-gamma*loss[j])/mu
             for j in range(3)]
        u0 = (c['outside_affinity'][g]-lam*c['outside_price']-beta*c['outside_time'])/mu
        peak = max(u0, *u)
        logden = peak+math.log(math.fsum(math.exp(z-peak) for z in [u0]+u))
        probs.append([math.exp(z-logden) for z in u])
        outside.append(math.exp(u0-logden))
        logs.append(logden)
    masses = [x*c['demand_scale'] for x in c['base_demand']]
    a = c['platform_resource']
    y = [math.fsum(masses[g]*probs[g][j] for g in range(3)) for j in range(3)]
    group_work = [math.fsum(x*z for x, z in zip(a, p)) for p in probs]
    slope = -math.fsum(masses[g]*c['time_sensitivity'][g]/mu*outside[g]*group_work[g]
                       for g in range(3))
    cost = [b+d for b, d in zip(c['base_cost'], [0., c['cost_gap'], .5*c['cost_gap']])]
    cs = math.fsum(masses[g]*mu/c['price_sensitivity'][g]*logs[g] for g in range(3))
    system = c['congestion_cost']*(.15*rho*rho+max(rho-.90, 0.)**3)
    ts = cs+math.fsum((p-v)*n for p, v, n in zip(fares, cost, y))-system
    return dict(demand=y, outside=outside, group_work=group_work,
                work=math.fsum(x*z for x, z in zip(a, y)), work_slope=slope, ts=ts)


def transient(c, fares, star_wait, initial_work, tolerance=1.):
    k = c['base_capacity']*c['capacity_scale']
    w0 = H*initial_work/k
    low, high = sorted([w0, star_wait])
    left, right = choice(c, fares, 1., low), choice(c, fares, 1., high)
    masses = [x*c['demand_scale'] for x in c['base_demand']]
    lower_slope = math.fsum(masses[g]*c['time_sensitivity'][g]/c['logit_scale']
                           *left['outside'][g]*right['group_work'][g] for g in range(3))
    if star_wait > 1e-7:
        return dict(initial_work=initial_work, initial_wait=w0, tolerance_minutes=tolerance,
                    method='endpoint_logit_exponential_bound',
                    derivative_lower_bound=lower_slope,
                    time_upper_minutes=exp_bound(abs(w0-star_wait), tolerance, lower_slope/k))
    gap = k-choice(c, fares, 1., 0.)['work']
    if gap > 1e-7:
        return dict(initial_work=initial_work, initial_wait=w0, tolerance_minutes=tolerance,
                    method='strict_underload_drain_bound', busy_work_gap=gap,
                    time_upper_minutes=k*max(0., w0-tolerance)/gap,
                    exact_zero_time_upper_minutes=k*w0/gap)
    if gap < -1e-7:
        raise ValueError('Saved zero-wait state conflicts with busy work closure')
    return dict(initial_work=initial_work, initial_wait=w0, tolerance_minutes=tolerance,
                method='critical_boundary_exponential_bound',
                derivative_lower_bound=lower_slope,
                time_upper_minutes=exp_bound(w0, tolerance, lower_slope/k),
                exact_zero_time_upper_minutes=None)


def payload(root):
    f = json.loads((root/INPUTS[0]).read_text(encoding='utf-8'))
    s = json.loads((root/INPUTS[1]).read_text(encoding='utf-8'))
    names = {c['name'] for c in f['cases']}
    return dict(cases=f['cases'], fluid=[dict(profile=r['profile'], charge=r['charge'],
                fares=r['selected']['prices'], state=r['waiting_outcome'])
                for r in f['historical_games']],
                queue=[{k:r[k] for k in ('profile', 'charge', 'prices', 'demand', 'rho',
                        'wait_minutes', 'ts')} for r in s['games'] if r['profile'] in names and r['accepted']])


def compute(data):
    cases = {c['name']: c for c in data['cases']}
    pairs, variability, max_choice, max_queue = [], [], 0., 0.
    if len(data['fluid']) != 12 or len(data['queue']) != 12:
        raise ValueError('Exactly two complete six-charge saved menus required')
    for i, f in enumerate(data['fluid']):
        c = next(c for c in cases.values() if
                 [c[k] for k in ('demand_scale', 'capacity_scale', 'reliability_power')] == f['profile'])
        q = next(q for q in data['queue'] if q['profile'] == c['name'] and q['charge'] == f['charge'])
        k = c['base_capacity']*c['capacity_scale']
        state = f['state']
        r = choice(c, f['fares'], state['rho'], state['wait_minutes'])
        z = choice(c, q['prices'], q['rho'], q['wait_minutes'])
        err = max(abs(x-y) for x,y in zip(r['demand']+z['demand'], state['demand']+q['demand']))
        max_choice = max(max_choice, err)
        queue = pk(q['demand'], c['platform_resource'], k, H, 1.)
        max_queue = max(max_queue, abs(queue['wait_minutes']-q['wait_minutes']))
        if err > 1e-7 or abs(queue['rho']-q['rho']) > 1e-9 or max_queue > 1e-7:
            raise ValueError('Saved arithmetic mismatch')
        busy = choice(c, f['fares'], 1., state['wait_minutes'])
        local = relaxation(H, k, busy['work_slope']) if state['wait_minutes'] > 1e-7 else None
        fluid_pk = pk(state['demand'], c['platform_resource'], k, H, 0.)
        if abs(state['rho']-1.) < 1e-8:
            # Saved complementarity identifies exact unit load despite summation roundoff.
            fluid_pk.update(arithmetic_rho=fluid_pk['rho'], rho=1., status='unstable',
                            wait_minutes=None, reason='Exact full fluid load; no stationary M/G/1 law')
        pairs.append(dict(profile=c['name'], charge=f['charge'], source_index=i,
            fluid_fares=f['fares'], stochastic_fares=q['prices'], same_fares=False,
            fluid_wait=state['wait_minutes'], stochastic_wait=q['wait_minutes'],
            fluid_rho=state['rho'], stochastic_rho=q['rho'],
            fluid_passengers=sum(state['demand']), stochastic_passengers=sum(q['demand']),
            fluid_TS=r['ts'], stochastic_TS=q['ts'], local_relaxation=local,
            local_95_percent_minutes=math.log(20)*local['minutes'] if local else None,
            initial_backlogs=[transient(c, f['fares'], state['wait_minutes'], b) for b in (0., k/2)],
            frozen_fluid_flow_PK=fluid_pk))
        if f['charge'] in (0., 12., 20.):
            for cv2 in (0., 1., 2.):
                variability.append(dict(profile=c['name'], charge=f['charge'], conditional_cv2=cv2,
                    fixed_fares=q['prices'], fixed_arrivals=q['demand'],
                    reference_saved_wait=q['wait_minutes'],
                    reference_fluid_wait=state['wait_minutes'],
                    **pk(q['demand'], c['platform_resource'], k, H, cv2)))
    if len(variability) != 18:
        raise ValueError('Variability budget must be exactly 18 frozen-flow moment evaluations')
    return dict(H_minutes=H, paired_saved_states=pairs, fixed_flow_variability=variability,
        max_saved_choice_error=max_choice, max_saved_queue_wait_error=max_queue,
        new_Nash_solves=0, new_behavioral_closures=0,
        interpretation='Saved institution menus have independent fares; new variability rows freeze queue fares AND arrivals, not new equilibria.',
        counterexample=pk([90.], [1.], 120., 120., 0.),
        payment_example=payment_equivalence([1., 2.], .4, 10.))


def analyze(root):
    return compute(payload(Path(root)))


def write_results(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    hashes = {p:digest(root/p) for p in INPUTS}
    data = payload(root)
    result = compute(data)
    output.mkdir(parents=True)
    for name, item in (('inputs.json', data), ('results.json', result)):
        (output/name).write_text(json.dumps(item, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for p in OWNED:
        target = output/'execution_source'/p
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/p, target)
    proof = root/'submission_packages/TRA_V290_LaTeX_Project/MAIN.tex'
    shutil.copyfile(proof, output/'MAIN_readonly_reference.tex')
    if hashes != {p:digest(root/p) for p in INPUTS}:
        raise ValueError('Old input changed during arithmetic generation')
    manifest = dict(old_sources=hashes, files={str(p.relative_to(output)).replace('\\','/'):digest(p)
                    for p in output.rglob('*') if p.is_file()}, old_sources_unchanged=True)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return result


def replay(folder):
    folder = Path(folder)
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    for p, expected in manifest['files'].items():
        if digest(folder/p) != expected:
            raise ValueError('Frozen hash mismatch: '+p)
    data = json.loads((folder/'inputs.json').read_text(encoding='utf-8'))
    saved = json.loads((folder/'results.json').read_text(encoding='utf-8'))
    if compute(data) != saved:
        raise ValueError('Frozen arithmetic replay mismatch')
    return dict(verified=True, paired_states=12, fixed_flow_evaluations=18,
                new_Nash_solves=0, new_behavioral_closures=0, writes=0)
