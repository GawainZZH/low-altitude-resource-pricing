"""Explicit unequal-resource Nash construction and primitive welfare signs.

Scalar choice/share inversion is not a general equilibrium search. Global
actual-game Nash existence follows from capacity-profit domination; the extra
population condition is used only for all-regime uniqueness.
"""
from __future__ import annotations

from numbers import Real

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

from .review7_structure import interior_shares


def _finite_tree(value):
    if isinstance(value, dict):
        return all(_finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(v) for v in value)
    return bool(np.isfinite(value)) if isinstance(value, Real) else True


def _primitives(case):
    c = {k: np.asarray(case[k], float) for k in (
        'resource', 'costs', 'platform_quality', 'mass', 'group_quality',
        'reliability_slopes', 'outside_utility', 'fare_bounds')}
    for k in ('lam', 'mu', 'value', 'capacity', 'outside_price'):
        c[k] = float(case[k])
    if any(v.ndim != 1 for v in c.values() if isinstance(v, np.ndarray)):
        raise ValueError('Primitive vectors must be one dimensional')
    m, g = len(c['resource']), len(c['mass'])
    if (m < 2 or g < 1 or any(v.ndim != 1 or not np.isfinite(v).all()
            for k, v in c.items() if isinstance(v, np.ndarray))
            or any(c[k].shape != (m,) for k in ('costs', 'platform_quality'))
            or any(c[k].shape != (g,) for k in ('group_quality', 'reliability_slopes', 'outside_utility'))
            or any(np.any(c[k] <= 0) for k in ('resource', 'mass'))
            or np.any(c['costs'] < 0) or np.any(c['reliability_slopes'] < 0)
            or not np.isfinite([c[k] for k in ('lam', 'mu', 'value', 'capacity', 'outside_price')]).all()
            or min(c[k] for k in ('lam', 'mu', 'value', 'capacity')) <= 0
            or c['outside_price'] < 0 or c['fare_bounds'].shape != (2,)
            or c['fare_bounds'][0] >= c['fare_bounds'][1]):
        raise ValueError('Finite compatible positive primitives required')
    if c['mass'].sum() <= c['capacity']/c['resource'].min():
        raise ValueError('Potential population must exceed K/min(a)')
    return c


def candidate_shares(case, charge):
    c = _primitives(case)
    if not np.isfinite(charge) or charge < 0:
        raise ValueError('Finite nonnegative charge required')
    return interior_shares(c['platform_quality']+c['mu']*np.log(c['resource']),
        c['costs']+c['resource']*charge, c['lam'], c['mu'])


def _choice(c, p, rho, wait):
    group = c['group_quality']-c['reliability_slopes']*rho
    air = (group[:, None]+c['platform_quality']-c['lam']*p-c['lam']*c['value']*wait)/c['mu']
    den = np.logaddexp(c['outside_utility']/c['mu'], logsumexp(air, axis=1))
    prob = np.exp(air-den[:, None])
    cs = float((c['mass']*c['mu']/c['lam'])@den)
    return prob, cs, c['mass']@prob


def _response(c, p):
    K, a = c['capacity'], c['resource']
    excess = float(a@_choice(c, p, 1., 0.)[2]-K)
    if excess > 1e-10:
        upper = 1.
        for _ in range(32):
            if float(a@_choice(c, p, 1., upper)[2]) < K:
                break
            upper *= 2
        else:
            raise ValueError('Finite clearing-wait bracket unavailable')
        wait = brentq(lambda w: float(a@_choice(c, p, 1., w)[2]-K),
                       0., upper, xtol=2e-12, rtol=1e-14)
        rho, regime = 1., 'strict'
    elif abs(excess) <= 1e-10:
        rho, wait, regime = 1., 0., 'transition'
    else:
        rho = brentq(lambda r: float(a@_choice(c, p, r, 0.)[2]/K-r),
                      0., 1., xtol=2e-13, rtol=1e-14)
        wait, regime = 0., 'free'
    prob, cs, y = _choice(c, p, rho, wait)
    return dict(probabilities=prob, demand=y, net_surplus=cs, rho=rho,
                wait=float(wait), regime=regime)


def actual_response(case, prices):
    c = _primitives(case)
    p = np.asarray(prices, float)
    if p.shape != c['resource'].shape or not np.isfinite(p).all():
        raise ValueError('Finite complete fare vector required')
    return _response(c, p)


def resource_state(case, charge):
    c = _primitives(case)
    x = candidate_shares(case, charge)
    a, costs, K = c['resource'], c['costs'], c['capacity']
    h, v = c['lam']/c['mu'], c['value']
    p = costs+a*charge+1/(h*(1-x))
    o = _response(c, p)
    if o['regime'] != 'strict':
        raise ValueError('A strict capacity response is required')
    if np.any(p <= c['fare_bounds'][0]) or np.any(p >= c['fare_bounds'][1]):
        raise ValueError('Candidate fares must be interior')
    y, prob = o['demand'], o['probabilities']
    if not np.allclose(a*y/K, x, rtol=1e-11, atol=1e-11):
        raise ArithmeticError('Resource-share and choice equations differ')
    eta = x/(1-x+x*x)
    weights = eta*(1-x)**2
    abar = float(weights@a/weights.sum())
    differences = a[None, :]-a[:, None]
    delta = (weights[None, :]*differences).sum(axis=1)/weights.sum()
    dx = h*weights*delta
    dp = a+eta*delta
    dy = K*dx/a
    # Pairwise variance avoids cancellation near the common-resource case.
    pairwise = sum(weights[i]*weights[j]*(a[i]-a[j])**2/(a[i]*a[j])
                   for i in range(len(a)) for j in range(i))
    Y, dY = float(y.sum()), float(h*K*pairwise/weights.sum())
    theta = prob.sum(axis=1)
    L = float(c['mass']@(theta*(1-theta)))
    if L <= 0:
        raise ArithmeticError('Positive outside-choice derivative required')
    dw = -float(y@dp/Y+dY/(h*L))/v
    D = 1+float(eta@(abar/a-1))
    t, B = float(eta.sum()-1), float(weights.sum())
    floor = t+B+t*t/B
    ground = float(c['mass'].sum()-Y)
    welfare, derivatives, allocations = {}, {}, {}
    relief = -v*Y*dw
    for kappa in (0., .5, 1.):
        key = str(kappa)
        welfare[key] = float(o['net_surplus']+(p-costs)@y+kappa*c['outside_price']*ground)
        allocations[key] = float((p-costs-kappa*c['outside_price'])@dy)
        derivatives[key] = float(K*D+(Y/(h*L)-kappa*c['outside_price'])*dY)
        if not np.isclose(derivatives[key], relief+allocations[key], atol=1e-7, rtol=1e-10):
            raise ArithmeticError('Two independent welfare derivatives differ')
    J = h*y[:, None]*x[:, None]*x[None, :]/(1-x[:, None])
    np.fill_diagonal(J, -h*y)
    equal = bool(np.ptp(a) == 0)
    result = dict(charge=float(charge), resource=a.tolist(), costs=costs.tolist(),
        prices=p.tolist(), resource_shares=x.tolist(), demand=y.tolist(),
        probabilities=prob.tolist(), group_air_probability=theta.tolist(),
        wait=o['wait'], rho=o['rho'], total_mass=Y, ground_mass=ground,
        net_surplus=o['net_surplus'], TS=welfare,
        resource_share_slope=dx.tolist(), fare_slope=dp.tolist(),
        demand_slope=dy.tolist(), total_mass_slope=dY, wait_slope=dw,
        relief_slope=relief, allocation_slopes=allocations, ts_derivatives=derivatives,
        fare_allocation_factor=D, cauchy_floor=floor,
        outside_receipt_boundary=None if equal else float(Y/(h*L)+K*D/dY),
        profit_jacobian=J.tolist(),
        global_uniqueness_mass_condition=bool(c['mass'].min() > K/a.min()),
        all_resources_equal=equal, constant_system_penalty_omitted=True)
    if not _finite_tree(result):
        raise ArithmeticError('Nonfinite derived resource response is not representable')
    return result


def resource_path(case, charges, deviation_grid=33):
    c = _primitives(case)
    q = np.asarray(charges, float)
    if (q.ndim != 1 or len(q) < 1 or not np.isfinite(q).all()
            or np.any(q < 0) or len(np.unique(q)) != len(q)
            or not isinstance(deviation_grid, int) or deviation_grid < 2):
        raise ValueError('Distinct nonnegative charges and a finite deviation grid required')
    states = [resource_state(case, charge) for charge in sorted(q)]
    max_gain, counts = 0., dict(strict=0, free=0, transition=0)
    a, costs = c['resource'], c['costs']
    for state in states:
        p, charge = np.array(state['prices']), state['charge']
        profits = (p-costs-charge*a)*state['demand']
        for m in range(len(a)):
            for fare in np.unique(np.r_[np.linspace(*c['fare_bounds'], deviation_grid), p[m]]):
                dev = p.copy(); dev[m] = fare
                o = _response(c, dev)
                counts[o['regime']] += 1
                gain = float((fare-costs[m]-charge*a[m])*o['demand'][m]-profits[m])
                max_gain = max(max_gain, gain)
    return dict(states=states, all_states_strict=True,
        maximum_sampled_unilateral_gain=max_gain, deviation_regimes=counts,
        new_general_Nash_solves=0, general_Nash_uniqueness_numerically_proved=False,
        interpretation='Analytic Nash construction with scalar share/choice inversion and finite corroborating deviations')
