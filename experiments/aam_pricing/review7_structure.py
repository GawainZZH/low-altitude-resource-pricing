"""Primitive asymmetric logit branch and saved transition accounting."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logsumexp


def interior_shares(quality, costs, lam, mu):
    quality, costs = np.asarray(quality, float), np.asarray(costs, float)
    if (quality.ndim != 1 or quality.shape != costs.shape or len(costs) < 2
            or not np.all(np.isfinite([quality, costs])) or np.any(costs < 0)
            or not np.isfinite(lam+mu) or min(lam, mu) <= 0):
        raise ValueError('Finite positive primitives and at least two platforms required')
    eta = (quality-lam*costs)/mu

    def f(s):
        return np.log(s)+1/(1-s)

    def inverse(x):
        # In log-odds coordinates, intermediate inverse calls need no share floor.
        left = min(-2., float(x)-2.)
        right = max(2., float(np.log(max(1., x)))+1.)
        root = brentq(lambda t: -np.logaddexp(0., -t)+1+np.exp(t)-x,
                      left, right, xtol=1.e-13)
        return float(expit(root))

    middle = f(1/len(costs))
    left, right = middle-float(eta.max()), middle-float(eta.min())
    zeta = left if left == right else brentq(
        lambda z: sum(inverse(x+z) for x in eta)-1, left, right, xtol=1.e-13)
    return np.array([inverse(x+zeta) for x in eta])


def primitive_branch(case):
    """Explicit strict Nash branch; the actual game permits free deviations."""
    lam, mu, v, a, K = (float(case[k]) for k in ('lam', 'mu', 'value', 'resource', 'capacity'))
    mass, g, outside, A, c, bounds, charges = (np.asarray(case[k], float) for k in
        ('mass', 'group_quality', 'outside_utility', 'platform_quality', 'costs', 'fare_bounds', 'charges'))
    if (not np.all(np.isfinite([lam, mu, v, a, K])) or min(lam, mu, v, a, K) <= 0
            or any(not np.all(np.isfinite(x)) for x in (mass, g, outside, A, c, bounds, charges))
            or mass.ndim != 1 or len(mass) != len(g) or len(g) != len(outside)
            or np.any(mass <= 0) or not K/a < sum(mass) or bounds.shape != (2,)
            or bounds[0] >= bounds[1] or charges.ndim != 1 or not len(charges)
            or np.any(charges < 0) or len(np.unique(charges)) != len(charges)):
        raise ValueError('Invalid primitive branch inputs')
    Y, h = K/a, lam/mu
    shares = interior_shares(A, c, lam, mu)
    zero_min = float(mass@expit((g+mu*logsumexp((A-lam*bounds[1])/mu)-outside)/mu))

    def probabilities(p, w):
        air = (g[:, None]+A-lam*p-lam*v*w)/mu
        den = np.logaddexp(outside/mu, logsumexp(air, axis=1))
        return np.exp(air-den[:, None]), float((mass*mu/lam)@den)

    contraction = []
    for i in range(len(A)):
        p = np.full(len(A), bounds[1]); p[i] = bounds[0]
        contraction.append(float(np.exp((A[i]-lam*p[i])/mu-logsumexp((A-lam*p)/mu))))
    responses = []
    max_gain = 0.
    for q in sorted(charges):
        p = c+a*q+1/(h*(1-shares))
        if np.any(p <= bounds[0]) or np.any(p >= bounds[1]):
            raise ValueError('The specified charge does not have an interior solution')
        if np.any(bounds[0] >= c+a*q+1/h):
            raise ValueError('Lower fare endpoint does not exclude a boundary response')
        zero_demand = float(mass@probabilities(p, 0.)[0].sum(axis=1))
        if zero_demand <= Y:
            raise ValueError('An initial strict response is not established')
        upper = 1.
        while float(mass@probabilities(p, upper)[0].sum(axis=1)) > Y:
            upper *= 2
        w = brentq(lambda x: float(mass@probabilities(p, x)[0].sum(axis=1))-Y,
                   0., upper, xtol=1.e-12)
        prob, cs = probabilities(p, w)
        y = mass@prob
        J = h*Y*np.outer(shares, shares)-np.diag(h*Y*shares)
        F = p-c-a*q
        # At the FOC, the first derivative of demand multiplies a zero bracket.
        profit_J = np.diag(y)@(-h*np.diag(1-shares)+h*F[:, None]*J/Y)
        for m in range(len(p)):
            for x in np.linspace(bounds[0], bounds[1], 401):
                dev = p.copy(); dev[m] = x
                conditional = np.exp((A-lam*dev)/mu-logsumexp((A-lam*dev)/mu))
                unrestricted = float(mass@probabilities(dev, 0.)[0].sum(axis=1))
                actual_y = min(Y, unrestricted)*conditional[m]
                gain = (x-c[m]-a*q)*actual_y-F[m]*y[m]
                max_gain = max(max_gain, float(gain))
        responses.append(dict(charge=float(q), prices=p.tolist(), wait=float(w),
            probabilities=prob.tolist(), demand=y.tolist(), US=float(cs-c@y),
            TS=float(cs+(p-c)@y), profit_jacobian=profit_J.tolist()))
    initial = responses[0]
    first_release = initial['charge']+v*initial['wait']/a
    release_fares = np.array(initial['prices'])+a*(first_release-initial['charge'])
    return dict(shares=shares.tolist(), minimum_zero_wait_demand=zero_min,
                initial_zero_wait_demand=float(mass@probabilities(np.array(initial['prices']), 0.)[0].sum(axis=1)),
                first_release_charge=float(first_release),
                first_release_fares_interior=bool(np.all(release_fares > bounds[0]) and np.all(release_fares < bounds[1])),
                maximum_sampled_unilateral_gain=max_gain,
                response_contraction_bound=max(contraction), total_service=Y,
                constant_system_penalty_omitted=True, responses=responses)


def release_split(points, kappa_index):
    if kappa_index not in (0, 1, 2):
        raise ValueError('Three stated accounting shares required')
    rows = sorted(points, key=lambda r: r['q'])
    if not rows or rows[0]['q'] != 0 or not all(np.isfinite(r['TS'][kappa_index]) for r in rows):
        raise ValueError('Complete finite values with an NC anchor required')
    release = next((r for r in rows if r['wait'] <= 1.e-8), None)
    best = max(rows, key=lambda r: (r['TS'][kappa_index], -r['q']))
    result = dict(optimal_charge=best['q'], total_gain=best['TS'][kappa_index]-rows[0]['TS'][kappa_index],
                  first_release_charge=None, pre_release_gain=None, post_release_gain=None)
    if release is not None:
        result.update(first_release_charge=release['q'],
            pre_release_gain=release['TS'][kappa_index]-rows[0]['TS'][kappa_index],
            post_release_gain=best['TS'][kappa_index]-release['TS'][kappa_index],
            optimum_before_first_release=best['q'] < release['q'])
    return result


def candidate_decision(envelope):
    if not envelope or any(r['min_TS'] > r['max_TS'] for r in envelope):
        raise ValueError('Ordered candidate welfare bounds required')
    worst = max(envelope, key=lambda r: (r['min_TS'], -r['q']))
    optimistic = max(envelope, key=lambda r: (r['max_TS'], -r['q']))
    return dict(maximin_charge=worst['q'], optimistic_charge=optimistic['q'],
        observed_best_worst_welfare=worst['min_TS'], observed_optimistic_welfare=optimistic['max_TS'],
        possible_optimizer_charges=[r['q'] for r in envelope if r['max_TS'] >= worst['min_TS']-1.e-8],
        all_Nash_bound=False)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
