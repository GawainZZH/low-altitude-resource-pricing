"""Bounded primitive finite paths; closed replay never invokes scalar solvers."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from itertools import product
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

from .review8_resource import _primitives, candidate_shares

OUTPUT = 'results/trb_review11_finite_paths_v1'
MODULE = 'experiments/aam_pricing/review11_finite_paths.py'
SCRIPT = 'scripts/run_review11_finite_paths.py'
TEST = 'tests/test_review11_finite_paths.py'
SOURCE_FILES = (MODULE, SCRIPT, TEST, 'experiments/__init__.py',
                'experiments/aam_pricing/__init__.py',
                'experiments/aam_pricing/review8_resource.py',
                'experiments/aam_pricing/review7_structure.py')
KAPPAS = (0., .5, 1.)
ATOL = 2e-7
RTOL = 2e-10


def design():
    cases = []
    for ri, ci, alignment in product(range(2), range(2), (-.75, 0., .75)):
        for cap in (44., 120.):
            c = dict(lam=.1, mu=.5, value=.4, capacity=60.,
                     resource=([.7,1.,1.4], [.25,1.,2.5])[ri],
                     costs=([9.,9.,9.], [6.,9.,12.])[ci],
                     platform_quality=[3.6-alignment,3.6,3.6+alignment],
                     mass=[300.,280.,260.], group_quality=[0.,.4,-.2],
                     reliability_slopes=[.1,.2,.3], outside_utility=[1.7,1.9,1.5],
                     outside_price=5., fare_bounds=[5.,cap])
            cases.append(dict(id=f'r{ri}_c{ci}_a{alignment:+.2f}_cap{cap:g}',
                              resource_index=ri, cost_index=ci, alignment=alignment,
                              primitives=deepcopy(c)))
    return dict(schema_version=1, cases=cases, kappas=list(KAPPAS),
                initial_charge=0., path_points=33, deviation_fares=65,
                quadrature_orders=[8,16], quadrature_tolerance=1e-9,
                quadrature_max_leaves=128, runtime_cap_seconds=600,
                interpretation='Purely analytical illustrations, not corridor calibration',
                new_general_Nash_solves=0)


def validate(case, kappas=KAPPAS):
    c = _primitives(case)
    ks = np.asarray(kappas, float)
    if (c['mass'].min() <= c['capacity']/c['resource'].min()
            or ks.ndim != 1 or len(ks) == 0 or not np.isfinite(ks).all()
            or np.any(ks < 0) or np.any(ks > 1)
            or np.any(ks*c['outside_price'] > c['mu']/c['lam'])):
        raise ValueError('All group-mass and ground-accounting premises required')
    return c


def _budget(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError('Frozen 600-second batch deadline reached; STOP')


def _close(actual, expected, label, atol=ATOL):
    if not np.allclose(actual, expected, atol=atol, rtol=RTOL):
        raise ValueError(f'Saved arithmetic mismatch: {label}')


def choice(case, prices, rho, wait):
    """Direct primitive logit and logsum, with no closure or share inversion."""
    c = case
    mass, A, g, slopes, out = (np.asarray(c[k], float) for k in
        ('mass','platform_quality','group_quality','reliability_slopes','outside_utility'))
    p = np.asarray(prices, float)
    air = (g[:,None]-slopes[:,None]*rho+A-c['lam']*p-c['lam']*c['value']*wait)/c['mu']
    den = np.logaddexp(out/c['mu'], logsumexp(air, axis=1))
    prob = np.exp(air-den[:,None])
    y = mass@prob
    cs = float((mass*c['mu']/c['lam'])@den)
    return dict(probabilities=prob.tolist(), demand=y.tolist(), net_surplus=cs)


def geometry(case, charge, shares):
    """Analytic full-capacity geometry from supplied shares; replay-safe."""
    a, costs, A = (np.asarray(case[k], float) for k in ('resource','costs','platform_quality'))
    x = np.asarray(shares, float)
    if (x.shape != a.shape or not np.isfinite(x).all() or np.any(x <= 0)
            or np.any(x >= 1) or not np.isfinite(charge) or charge < 0):
        raise ValueError('Finite positive interior shares and nonnegative charge required')
    h, K = case['lam']/case['mu'], case['capacity']
    zeta = np.log(x)+1/(1-x)-(A-case['lam']*costs)/case['mu']-np.log(a)+h*a*charge
    _close(x.sum(), 1., 'share sum', 1e-11)
    _close(zeta, zeta.mean(), 'exact share equations', 1e-10)
    p = costs+a*charge+1/(h*(1-x))
    logits = np.log(a)+(A-case['lam']*p)/case['mu']
    _close(np.exp(logits-logsumexp(logits)), x, 'resource-weighted logits', 1e-11)
    eta = x/(1-x+x*x)
    weights = eta*(1-x)**2
    abar = float(weights@a/weights.sum())
    dx = h*weights*(abar-a)
    dp = (1-eta)*a+eta*abar
    dy = K*dx/a
    pair = sum(weights[i]*weights[j]*(a[i]-a[j])**2/(a[i]*a[j])
               for i in range(len(a)) for j in range(i))
    dY = float(h*K*pair/weights.sum())
    D = float(1+eta@(abar/a-1))
    u, B = float(eta.sum()-1), float(weights.sum())
    floor = u+B+u*u/B
    if not D >= floor-1e-10 > 0 or min(dp) <= 0 or dY < 0:
        raise ValueError('Analytical sign or Cauchy bound failed')
    return dict(charge=float(charge), resource_shares=x.tolist(), prices=p.tolist(),
                conditional_demand=(K*x/a).tolist(), resource_share_slope=dx.tolist(),
                fare_slope=dp.tolist(), demand_slope=dy.tolist(), total_mass_slope=dY,
                fare_allocation_factor=D, cauchy_floor=floor)


def _auxiliary(case, q, deadline=None):
    _budget(deadline)
    s = geometry(case, q, candidate_shares(case, q))
    return dict(s, **_zero_wait_witness(case,s))


def _zero_wait_witness(case, s):
    o = choice(case, s['prices'], 1., 0.)
    prob = np.asarray(o['probabilities'])
    a,dp = np.asarray(case['resource']),np.asarray(s['fare_slope'])
    slope = -case['lam']/case['mu']*float(np.asarray(case['mass'])@(
        (prob*(a*dp)).sum(axis=1)-(prob*a).sum(axis=1)*(prob*dp).sum(axis=1)))
    return dict(zero_wait_excess=float(a@o['demand']-case['capacity']),
                zero_wait_excess_slope=slope)


def _ineligible_reasons(case, initial):
    p = np.asarray(initial['prices'])
    reasons = []
    if np.any(p <= case['fare_bounds'][0]) or np.any(p >= case['fare_bounds'][1]):
        reasons.append('empty initial interior fare domain')
    if initial['zero_wait_excess'] <= 0:
        reasons.append('empty initial strict capacity domain')
    return reasons


def locate_endpoints(case, deadline=None):
    validate(case)
    initial = _auxiliary(case, 0., deadline)
    reasons = _ineligible_reasons(case, initial)
    if reasons:
        return dict(eligible=False, reasons=reasons, initial=initial)

    def solve(fun):
        upper = 1.
        for _ in range(64):
            _budget(deadline)
            if fun(upper) <= 0:
                return float(brentq(fun, 0., upper, xtol=2e-12, rtol=1e-14))
            upper *= 2
        raise ValueError('Monotone endpoint bracket unavailable')

    release = solve(lambda q: _auxiliary(case, q, deadline)['zero_wait_excess'])
    fare = solve(lambda q: case['fare_bounds'][1]-max(_auxiliary(case, q, deadline)['prices']))
    return dict(eligible=True, initial=initial, release_charge=release, fare_charge=fare,
                release=_auxiliary(case, release, deadline), fare=_auxiliary(case, fare, deadline))


def _classification(endpoints):
    r, f = endpoints['release_charge'], endpoints['fare_charge']
    if abs(r-f) <= 1e-9:
        return 'coincident'
    return 'release_first' if r < f else 'fare_first'


def _state_from_choice(case, s, wait, rho=1.):
    o = choice(case, s['prices'], rho, wait)
    y, p, costs = (np.asarray(v, float) for v in (o['demand'], s['prices'], case['costs']))
    Y, N, h = float(y.sum()), sum(case['mass']), case['lam']/case['mu']
    theta = np.asarray(o['probabilities']).sum(axis=1)
    L = float(np.asarray(case['mass'])@(theta*(1-theta)))
    dY = s['total_mass_slope']
    dw = -float(y@np.asarray(s['fare_slope'])/Y+dY/(h*L))/case['value']
    s = dict(s, **o, wait=float(wait), rho=float(rho), total_mass=Y,
             ground_mass=N-Y, L=L, wait_slope=dw,
             profits=((p-costs-s['charge']*np.asarray(case['resource']))*y).tolist(),
             TS={}, ts_derivatives={}, relief_allocation_derivatives={})
    for k in KAPPAS:
        key = str(k)
        s['TS'][key] = float(o['net_surplus']+(p-costs)@y+k*case['outside_price']*(N-Y))
        s['ts_derivatives'][key] = float(case['capacity']*s['fare_allocation_factor']+
                                        (Y/(h*L)-k*case['outside_price'])*dY)
        s['relief_allocation_derivatives'][key] = float(-case['value']*Y*dw+
                                  (p-costs-k*case['outside_price'])@np.asarray(s['demand_slope']))
    return s


def path_state(case, charge, endpoint=None, deadline=None):
    if endpoint not in (None,'release','fare','coincident'):
        raise ValueError('Unknown endpoint extension')
    validate(case)
    s = _auxiliary(case, charge, deadline)
    p = np.asarray(s['prices'])
    lo, hi = case['fare_bounds']
    if np.any(p <= lo) or np.any(p > hi+1e-9) or (endpoint is None and np.any(p >= hi)):
        raise ValueError('Candidate outside interior fare domain')
    excess = s['zero_wait_excess']
    if endpoint in ('release','coincident'):
        _close(excess, 0., 'analytical zero-wait endpoint')
        wait = 0.
    else:
        if excess <= 0:
            raise ValueError('Candidate outside strict capacity domain')
        upper = 1.
        def residual(w):
            _budget(deadline)
            return float(np.dot(case['resource'], choice(case, p, 1., w)['demand'])-case['capacity'])
        for _ in range(64):
            if residual(upper) < 0:
                break
            upper *= 2
        else:
            raise ValueError('Strict waiting bracket unavailable')
        wait = float(brentq(residual, 0., upper, xtol=2e-12, rtol=1e-14))
    if endpoint in ('fare','coincident'):
        _close(max(p), hi, 'upper fare endpoint', 1e-9)
    s = _state_from_choice(case, s, wait)
    _close(s['demand'], s['conditional_demand'], 'capacity shares vs primitive choice')
    return s


def actual_state(case, prices, deadline=None):
    """Behavioral scalar closure for fresh unilateral checks only."""
    a, K = case['resource'], case['capacity']
    def excess(rho, w):
        _budget(deadline)
        return float(np.dot(a, choice(case, prices, rho, w)['demand'])-K*rho)
    e = excess(1., 0.)
    if abs(e) <= 1e-10:
        rho, w, regime = 1., 0., 'transition'
    elif e > 0:
        upper = 1.
        for _ in range(64):
            if excess(1., upper) < 0:
                break
            upper *= 2
        else:
            raise ValueError('Actual-response wait bracket unavailable')
        rho, w, regime = 1., float(brentq(lambda w: excess(1.,w), 0.,upper, xtol=2e-12)), 'strict'
    else:
        rho = float(brentq(lambda r: excess(r,0.), 0.,1., xtol=2e-13))
        w, regime = 0., 'free'
    return dict(**choice(case, prices, rho, w), rho=rho, wait=w, regime=regime)


def adaptive_integral(case, stop, deadline=None):
    """Adaptive paired Gaussian quadrature with complete arithmetic witnesses."""
    rules = {}
    for order in (8,16):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        rules[str(order)] = dict(nodes=nodes.tolist(), weights=weights.tolist())

    def leaf(left, right):
        result = dict(left=float(left), right=float(right))
        for label, order in (('low',8),('high',16)):
            rows = []
            for node, weight in zip(rules[str(order)]['nodes'], rules[str(order)]['weights']):
                _budget(deadline)
                q = (left+right)/2+(right-left)*node/2
                s = path_state(case, q, deadline=deadline)
                rows.append(dict(charge=q, weight=weight*(right-left)/2, state=s))
            result[label] = rows
        result['integral'] = sum(r['weight']*r['state']['fare_allocation_factor'] for r in result['high'])
        low = sum(r['weight']*r['state']['fare_allocation_factor'] for r in result['low'])
        result['error_estimate'] = abs(result['integral']-low)
        return result

    leaves = [leaf(0., stop)]
    while True:
        value = sum(v['integral'] for v in leaves)
        error = sum(v['error_estimate'] for v in leaves)
        if error <= 1e-9*max(1.,abs(value)):
            break
        if len(leaves) >= 128:
            raise ValueError('Frozen quadrature leaf budget exhausted')
        index = max(range(len(leaves)), key=lambda i: leaves[i]['error_estimate'])
        old = leaves.pop(index)
        mid = (old['left']+old['right'])/2
        leaves.extend((leaf(old['left'],mid), leaf(mid,old['right'])))
    leaves.sort(key=lambda v: v['left'])
    return dict(rules=rules, leaves=leaves, integral=value, error_estimate=error,
                error_interpretation='Paired-rule numerical estimate, not a rigorous interval enclosure')


def _bounds(case, states, quadrature):
    start, end = states[0], states[-1]
    deltaY = end['total_mass']-start['total_mass']
    rows = []
    for k in KAPPAS:
        key = str(k)
        gain = end['TS'][key]-start['TS'][key]
        bound = case['capacity']*quadrature['integral']+(case['mu']/case['lam']-k*case['outside_price'])*deltaY
        derivative = sum(r['weight']*r['state']['ts_derivatives'][key]
                         for leaf in quadrature['leaves'] for r in leaf['high'])
        rows.append(dict(kappa=k, finite_gain=gain, lower_bound=bound, slack=gain-bound,
                         tightness=bound/gain, deltaY=deltaY, integrated_derivative=derivative,
                         bound_quadrature_error_estimate=case['capacity']*quadrature['error_estimate']))
    return rows


def evaluate_case(case, deadline=None):
    _budget(deadline)
    validate(case)
    ends = locate_endpoints(case, deadline)
    if not ends['eligible']:
        return dict(status='ineligible', reasons=ends['reasons'], endpoints=ends,
                    qualifying_release=False, new_general_Nash_solves=0)
    kind = _classification(ends)
    stop = min(ends['fare_charge'], ends['release_charge'])
    endpoint = {'release_first':'release','fare_first':'fare','coincident':'coincident'}[kind]
    states = [path_state(case, float(q), endpoint=endpoint if i == 32 else None, deadline=deadline)
              for i,q in enumerate(np.linspace(0.,stop,33))]
    quadrature = adaptive_integral(case, stop, deadline)
    deviations = []
    maximum = 0.
    for index in (0,16,32):
        s = states[index]
        for m in range(len(case['resource'])):
            baseline = s['profits'][m]
            trace = []
            for fare in np.unique(np.r_[np.linspace(*case['fare_bounds'],65),s['prices'][m]]):
                _budget(deadline)
                p = np.array(s['prices']); p[m] = fare
                o = actual_state(case, p, deadline)
                profit = float((fare-case['costs'][m]-s['charge']*case['resource'][m])*o['demand'][m])
                trace.append(dict(fare=float(fare), profit=profit, response=o))
                maximum = max(maximum,profit-baseline)
            deviations.append(dict(state_index=index, platform=m, baseline=baseline, trace=trace))
    result = dict(status='completed', classification=kind, stop_charge=stop, endpoints=ends,
                  qualifying_release=kind == 'release_first', states=states, quadrature=quadrature,
                  bounds=_bounds(case, states, quadrature), deviations=deviations,
                  maximum_sampled_unilateral_gain=maximum, new_general_Nash_solves=0,
                  numerical_global_Nash_proof=False, post_domain_candidates_evaluated=False)
    result['validation'] = replay_case(case,result)
    return result


def _replay_auxiliary(case, saved):
    s = geometry(case,saved['charge'],saved['resource_shares'])
    for key in s:
        _close(saved[key],s[key],key)
    witness = _zero_wait_witness(case,s)
    for key,value in witness.items():
        _close(saved[key],value,key)
    if witness['zero_wait_excess_slope'] >= 0:
        raise ValueError('Nonmonotone zero-wait auxiliary resource excess')
    return witness['zero_wait_excess']


def _replay_state(case, saved):
    _replay_auxiliary(case,saved)
    s = geometry(case,saved['charge'],saved['resource_shares'])
    if saved['wait'] < 0 or saved['rho'] != 1.:
        raise ValueError('Invalid saved capacity state')
    s = _state_from_choice(case,s,saved['wait'],saved['rho'])
    for key in s:
        if isinstance(s[key],dict):
            if set(s[key]) != set(saved[key]):
                raise ValueError('Missing kappa quantifier')
            for k in s[key]:
                _close(saved[key][k],s[key][k],key+':'+k)
        else:
            _close(saved[key],s[key],key)
    _close(s['demand'],s['conditional_demand'],'exact full-capacity shares')
    _close(np.dot(case['resource'],s['demand']),case['capacity'],'capacity')
    for k in s['TS']:
        _close(s['ts_derivatives'][k],s['relief_allocation_derivatives'][k],'two analytic derivatives')
    if min(s['profits']) <= 0 or max(s['wait_slope'], -min(s['fare_slope'])) >= 0:
        raise ValueError('Profit or comparative static validation failed')
    return s


def replay_case(case, result):
    """Validate saved roots by substitution, never re-solve their locations."""
    validate(case)
    ends = result['endpoints']
    _replay_auxiliary(case,ends['initial'])
    reasons = _ineligible_reasons(case,ends['initial'])
    if result['status'] == 'ineligible':
        if not reasons or ends['eligible'] or result['reasons'] != reasons or result['qualifying_release']:
            raise ValueError('Unsupported ineligible classification')
        return dict(valid=True, status='ineligible', scalar_roots_called=0)
    if result['status'] != 'completed' or reasons or not ends['eligible']:
        raise ValueError('Incomplete/failed outcome cannot pass closed arithmetic validation')
    for key in ('release','fare'):
        _replay_auxiliary(case,ends[key])
        _close(ends[key]['charge'],ends[key+'_charge'],key+' charge')
    _close(ends['release']['zero_wait_excess'],0.,'release equality')
    _close(max(ends['fare']['prices']),case['fare_bounds'][1],'fare equality')
    kind = _classification(ends)
    if result['classification'] != kind or result['qualifying_release'] != (kind == 'release_first'):
        raise ValueError('Fare-first exclusion or endpoint classification mismatch')
    stop = min(ends['release_charge'],ends['fare_charge'])
    _close(result['stop_charge'],stop,'stop charge')
    states = result['states']
    if len(states) != 33:
        raise ValueError('Full frozen 33-point path required')
    _close([s['charge'] for s in states],np.linspace(0.,stop,33),'entire path interval')
    for i,s in enumerate(states):
        _replay_state(case,s)
        p = np.array(s['prices'])
        if np.any(p <= case['fare_bounds'][0]) or np.any(p > case['fare_bounds'][1]+1e-9):
            raise ValueError('Path exceeds fare domain')
        if i < 32 and (s['wait'] <= 0 or s['zero_wait_excess'] <= 0 or max(p) >= case['fare_bounds'][1]):
            raise ValueError('Non-strict interior path point')
    last = states[-1]
    if kind in ('release_first','coincident'):
        _close(last['wait'],0.,'zero wait at release',1e-12)
        _close(last['zero_wait_excess'],0.,'release resource equality')
        if kind == 'release_first' and max(last['prices']) >= case['fare_bounds'][1]:
            raise ValueError('Release claim requires strictly interior endpoint fares')
    else:
        _close(max(last['prices']),case['fare_bounds'][1],'first upper fare equality',1e-9)
        if last['wait'] <= 0:
            raise ValueError('Fare-first endpoint must still have positive wait')
    for low,high in zip(states,states[1:]):
        if (np.any(np.array(high['prices']) <= low['prices']) or high['wait'] >= low['wait']
                or high['total_mass'] < low['total_mass']-ATOL
                or any(high['TS'][k] <= low['TS'][k] for k in low['TS'])):
            raise ValueError('Finite path monotonicity failed')
    quad = result['quadrature']
    rules = quad['rules']
    for order in (8,16):
        rule = rules[str(order)]
        x,w = np.array(rule['nodes']),np.array(rule['weights'])
        if len(x) != order or len(w) != order or min(w) <= 0 or max(abs(x)) >= 1 or np.any(np.diff(x) <= 0):
            raise ValueError('Invalid Gaussian rule')
        for degree in range(2*order):
            _close(w@(x**degree),0. if degree%2 else 2/(degree+1),'Gaussian polynomial moment',1e-12)
    cursor, integral, error = 0.,0.,0.
    if not 1 <= len(quad['leaves']) <= 128:
        raise ValueError('Quadrature leaf budget invalid')
    for leaf in quad['leaves']:
        left,right = leaf['left'],leaf['right']
        _close(left,cursor,'quadrature partition',1e-12)
        if right <= left or right > stop+1e-12:
            raise ValueError('Invalid quadrature interval')
        cursor = right
        sums = {}
        for label,order in (('low',8),('high',16)):
            rows,rule = leaf[label],rules[str(order)]
            if len(rows) != order:
                raise ValueError('Incomplete quadrature witness')
            for row,node,weight in zip(rows,rule['nodes'],rule['weights']):
                _close(row['charge'],(left+right)/2+(right-left)*node/2,'quadrature node',1e-12)
                _close(row['weight'],weight*(right-left)/2,'quadrature weight',1e-12)
                _close(row['state']['charge'],row['charge'],'quadrature state charge',1e-12)
                _replay_state(case,row['state'])
                if row['state']['wait'] <= 0 or not 0 < row['charge'] < stop:
                    raise ValueError('Quadrature evaluated outside strict path')
                if not case['fare_bounds'][0] < min(row['state']['prices']) <= max(row['state']['prices']) < case['fare_bounds'][1]:
                    raise ValueError('Quadrature outside interior fare domain')
            sums[label] = sum(r['weight']*r['state']['fare_allocation_factor'] for r in rows)
        estimate = abs(sums['high']-sums['low'])
        _close(leaf['integral'],sums['high'],'leaf integral',1e-11)
        _close(leaf['error_estimate'],estimate,'paired quadrature error',1e-12)
        integral += sums['high']; error += estimate
    _close(cursor,stop,'full quadrature coverage',1e-12)
    _close(quad['integral'],integral,'integral',1e-11)
    _close(quad['error_estimate'],error,'total quadrature error',1e-12)
    if error > 1e-9*max(1.,abs(integral)):
        raise ValueError('Quadrature did not meet frozen tolerance')
    expected = _bounds(case,states,quad)
    if len(result['bounds']) != len(KAPPAS):
        raise ValueError('Bound must quantify all three kappas')
    for saved,b in zip(result['bounds'],expected):
        for key in b:
            _close(saved[key],b[key],'finite bound '+key)
        if b['finite_gain'] < b['lower_bound']-ATOL or b['lower_bound'] <= 0:
            raise ValueError('Finite welfare bound failed')
        _close(b['finite_gain'],b['integrated_derivative'],'direct vs integrated derivative',2e-6)
    deviations = result['deviations']
    expected_keys = [(i,m) for i in (0,16,32) for m in range(len(case['resource']))]
    if [(d['state_index'],d['platform']) for d in deviations] != expected_keys:
        raise ValueError('Incomplete start/mid/end unilateral comparisons')
    maximum = 0.
    for d in deviations:
        s,m = states[d['state_index']],d['platform']
        _close(d['baseline'],s['profits'][m],'incumbent profit')
        fares = np.unique(np.r_[np.linspace(*case['fare_bounds'],65),s['prices'][m]])
        _close([t['fare'] for t in d['trace']],fares,'whole interval 65-fare coverage',1e-12)
        for t in d['trace']:
            o = t['response']; p = np.array(s['prices']); p[m] = t['fare']
            if not 0 <= o['rho'] <= 1 or o['wait'] < 0:
                raise ValueError('Invalid saved actual unilateral closure')
            rebuilt = choice(case,p,o['rho'],o['wait'])
            for key in rebuilt:
                _close(o[key],rebuilt[key],'actual choice '+key)
            R = float(np.dot(case['resource'],rebuilt['demand']))
            _close(R,case['capacity']*o['rho'],'actual resource closure')
            if o['wait'] > 0:
                _close(o['rho'],1.,'waiting complementarity',1e-12)
            regime = 'strict' if o['wait'] > 0 else ('transition' if o['rho'] == 1 else 'free')
            if o['regime'] != regime:
                raise ValueError('Actual regime label mismatch')
            profit = (t['fare']-case['costs'][m]-s['charge']*case['resource'][m])*rebuilt['demand'][m]
            _close(t['profit'],profit,'unilateral actual profit')
            maximum = max(maximum,profit-d['baseline'])
    _close(result['maximum_sampled_unilateral_gain'],maximum,'maximum sampled gain')
    if maximum > 1e-7 or result['new_general_Nash_solves'] != 0 or result['numerical_global_Nash_proof']:
        raise ValueError('Finite deviation failure or invalid proof claim')
    if result['post_domain_candidates_evaluated']:
        raise ValueError('Post-domain Nash evaluation forbidden')
    return dict(valid=True, status='completed', scalar_roots_called=0,
                qualifying_release=kind == 'release_first', states=33,
                maximum_sampled_unilateral_gain=maximum,
                maximum_capacity_error=max(abs(np.dot(case['resource'],s['demand'])-case['capacity']) for s in states),
                maximum_share_choice_error=max(float(np.max(abs(np.array(s['demand'])-s['conditional_demand']))) for s in states))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dump(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def _prior_hashes(root):
    return {str(p.relative_to(root)).replace('\\','/'):sha(p)
            for folder in ('experiments/aam_pricing','scripts','tests')
            for p in sorted((root/folder).rglob('*.py'))
            if 'review11' not in p.name.lower()}


def historical_audit(before, after):
    """Exclude every concurrent review11 owner, not merely this owner's files."""
    names = set(before)|set(after)
    historical = sorted(k for k in names if 'review11' not in Path(k).name.lower())
    changes = [k for k in historical if before.get(k) != after.get(k)]
    excluded = sorted(k for k in names if k not in historical and before.get(k) != after.get(k))
    return dict(unchanged=not changes, changes=changes, historical_modules=len(historical),
                excluded_concurrent_review11_changes=excluded)


def prepare(root, output):
    """Cache inputs and immutable sources before any fresh path evaluations."""
    root,output = Path(root),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'execution_manifest.json').exists() or (output/'inputs.json').exists():
        raise FileExistsError('Frozen execution already exists; use --replay, never overwrite')
    if not (output/'design_freeze.md').exists():
        raise FileNotFoundError('Design freeze must be present before preparation')
    _dump(output/'inputs.json',design())
    snapshot = output/'execution_source'
    for name in SOURCE_FILES:
        dest = snapshot/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/name,dest)
    references = ('submission_packages/TRA_V290_LaTeX_Project/MAIN.tex',
                  'docs/qa/trb_eleventh_review_20261017/before/TRB_Second_Round_Reviewer_Report_Major_Revision_84.md',
                  'results/trb_review8_resource_v1/inputs.json')
    for name in references:
        dest = output/'reference_source'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/name,dest)
    files = [p for folder in ('execution_source','reference_source') for p in (output/folder).rglob('*') if p.is_file()]
    files += [output/'design_freeze.md',output/'inputs.json']
    for name in ('tdd_red.txt','tdd_red_assertion.txt','tdd_red_endpoint_witness.txt'):
        if (output/name).exists():
            files.append(output/name)
    _dump(output/'execution_manifest.json',dict(schema_version=1,
        frozen_files={p.relative_to(output).as_posix():sha(p) for p in files},
        prior_modules=_prior_hashes(root), created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        source_snapshot_separate_from_replay=True))
    return dict(prepared=True,cases=24,new_general_Nash_solves=0)


def _manifest_check(output):
    manifest = json.loads((output/'execution_manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['frozen_files'].items():
        if sha(output/name) != digest:
            raise ValueError('Frozen source/input hash mismatch: '+name)
    return manifest


def freeze_replay_source(root, output):
    """Freeze integration replay code separately; never rewrite executed sources."""
    root,output = Path(root),Path(output)
    _manifest_check(output)
    if (output/'replay_source_manifest.json').exists():
        raise FileExistsError('Closed replay source already frozen')
    snapshot = output/'replay_source'
    for name in SOURCE_FILES:
        dest = snapshot/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/name,dest)
    files = [p for p in snapshot.rglob('*') if p.is_file()]
    _dump(output/'replay_source_manifest.json',dict(schema_version=1,
        execution_manifest_sha256=sha(output/'execution_manifest.json'),
        frozen_files={p.relative_to(output).as_posix():sha(p) for p in files},
        reason='Concurrency scope correction only; production was not rerun',
        replay_is_closed_arithmetic=True))
    return dict(replay_source_frozen=True,execution_sources_preserved=True)


def summarize(rows):
    counts = {}
    for row in rows:
        r = row['result']
        kind = r.get('classification',r['status'])
        counts[kind] = counts.get(kind,0)+1
    complete = [r['result'] for r in rows if r['result']['status'] == 'completed']
    return dict(paths=len(rows), outcomes=counts,
                qualifying_release_paths=sum(r['qualifying_release'] for r in complete),
                all_completed=len(complete) == len(rows),
                maximum_sampled_unilateral_gain=max((r['maximum_sampled_unilateral_gain'] for r in complete),default=0.),
                new_general_Nash_solves=0, finite_deviations_are_global_proof=False,
                parent_approval_required_before_new_cases=not any(r['qualifying_release'] for r in complete))


def run(root, output):
    root,output = Path(root),Path(output)
    manifest = _manifest_check(output)
    if (output/'summary.json').exists():
        raise FileExistsError('Saved run exists; immutable evidence cannot be overwritten')
    inputs = json.loads((output/'inputs.json').read_text(encoding='utf-8'))
    if inputs != design():
        raise ValueError('Cached inputs differ from frozen design')
    started = time.monotonic()
    deadline = started+600
    rows = []
    for item in inputs['cases']:
        try:
            r = evaluate_case(item['primitives'],deadline)
        except (TimeoutError, KeyboardInterrupt) as exc:
            r = dict(status='runtime_cap_STOP',error=str(exc))
        except Exception as exc:
            r = dict(status='failed',error=f'{type(exc).__name__}: {exc}')
        rows.append(dict(id=item['id'],result=r))
        _dump(output/'checkpoint.json',dict(rows=rows,elapsed_seconds=time.monotonic()-started))
        print(item['id']+': '+r.get('classification',r['status']),flush=True)
        if r['status'] == 'runtime_cap_STOP':
            rows.extend(dict(id=v['id'],result=dict(status='not_run_runtime_cap_STOP'))
                        for v in inputs['cases'][len(rows):])
            break
    result = dict(schema_version=1,rows=rows,summary=summarize(rows),
                  elapsed_seconds=time.monotonic()-started)
    after = _prior_hashes(root)
    result['prior_modules_unchanged'] = after == manifest['prior_modules']
    result['prior_module_changes'] = [k for k in set(after)|set(manifest['prior_modules'])
                                    if after.get(k) != manifest['prior_modules'].get(k)]
    _dump(output/'summary.json',result)
    _dump(output/'result_manifest.json',dict(summary_sha256=sha(output/'summary.json'),
        execution_manifest_sha256=sha(output/'execution_manifest.json'),
        prior_modules_after=after))
    return result


def replay(output):
    """Read-only, solver-free validation of a closed saved execution."""
    output = Path(output)
    manifest = _manifest_check(output)
    if (output/'replay_source_manifest.json').exists():
        replay_manifest = json.loads((output/'replay_source_manifest.json').read_text(encoding='utf-8'))
        if replay_manifest['execution_manifest_sha256'] != sha(output/'execution_manifest.json'):
            raise ValueError('Replay source not linked to executed source manifest')
        for name,digest in replay_manifest['frozen_files'].items():
            if sha(output/name) != digest:
                raise ValueError('Closed replay source hash mismatch: '+name)
    result_manifest = json.loads((output/'result_manifest.json').read_text(encoding='utf-8'))
    if (sha(output/'summary.json') != result_manifest['summary_sha256'] or
            sha(output/'execution_manifest.json') != result_manifest['execution_manifest_sha256']):
        raise ValueError('Result/manifest hash mismatch')
    inputs = json.loads((output/'inputs.json').read_text(encoding='utf-8'))
    result = json.loads((output/'summary.json').read_text(encoding='utf-8'))
    if inputs != design() or [r['id'] for r in result['rows']] != [r['id'] for r in inputs['cases']]:
        raise ValueError('Incomplete or changed 24-path design')
    checks = []
    for item,row in zip(inputs['cases'],result['rows']):
        checks.append(replay_case(item['primitives'],row['result']))
    if result['summary'] != summarize(result['rows']):
        raise ValueError('Saved summary does not match all outcomes')
    audit = historical_audit(manifest['prior_modules'],result_manifest['prior_modules_after'])
    if not audit['unchanged']:
        raise ValueError('Prior source modules changed during execution')
    return dict(summary=result['summary'],valid=True,paths_validated=len(checks),
                roots_called=0,adaptive_quadrature_calls=0,writes=0,
                historical_audit=audit,raw_execution_broad_audit_unchanged=result['prior_modules_unchanged'],
                reference='Saved source manifest and primitive arithmetic, no current-manuscript dependency')
