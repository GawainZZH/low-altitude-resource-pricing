"""Strict-capacity comparative statics, not a Nash pass-through assumption.

The reduced comparison has common fare p and fixed aerial attributes. Each
group faces actual waiting disutility beta_g*w. Availability is the special
case beta_g=mu and w=-log(phi), with w dimensionless rather than in time units.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit


def _arrays(*values):
    arrays = tuple(np.asarray(v, dtype=float) for v in values)
    if not arrays or not arrays[0].size or any(a.ndim != 1 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError('Matching one-dimensional group arrays are required')
    if any(not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError('All group values must be finite')
    return arrays


def group_surplus_difference(demand, sensitivity, before, after, scale=1.):
    d, lam, lo, hi = _arrays(demand, sensitivity, before, after)
    if not np.isfinite(scale) or scale <= 0 or np.any(d <= 0) or np.any(lam <= 0):
        raise ValueError('Positive group mass, money sensitivity and scale required')
    if np.any(lo < 0) or np.any(hi < 0) or np.any(lo >= 1) or np.any(hi >= 1):
        raise ValueError('Outside shares must be strictly positive')
    return -d*scale/lam*(np.log1p(-hi)-np.log1p(-lo))


def constrained_response(demand, sensitivity, wait_sensitivity, affinity,
                         price, capacity, scale=1.):
    d, lam, beta, alpha = _arrays(demand, sensitivity, wait_sensitivity, affinity)
    if (not np.all(np.isfinite([price, capacity, scale]))
            or scale <= 0 or capacity <= 0 or np.any(d <= 0) or np.any(lam <= 0)
            or np.any(beta <= 0)):
        raise ValueError('Positive parameters required')

    def shares(delay):
        return expit((alpha-lam*price-beta*delay)/scale)

    if float(d @ shares(0)) <= capacity:
        return {'share': shares(0), 'delay': 0., 'strictly_active': False}
    hi = 1.
    while float(d @ shares(hi)) > capacity:
        hi *= 2.
    delay = brentq(lambda w: float(d @ shares(w))-capacity, 0., hi,
                   xtol=1.e-12, rtol=1.e-14)
    return {'share': shares(delay), 'delay': delay, 'strictly_active': True}


def allocation_derivative(demand, sensitivity, wait_sensitivity, share, scale=1., *, delay):
    """Identical fares, equal resource units/costs and positive clearing delay.

    A valid constrained_response supplies the interior branch; this algebraic
    helper does not verify a Nash response or infer activity from shares alone.
    """
    d, lam, beta, theta = _arrays(demand, sensitivity, wait_sensitivity, share)
    if (not np.all(np.isfinite([scale, delay])) or delay <= 0
            or scale <= 0 or np.any(d <= 0) or np.any(lam <= 0) or np.any(beta <= 0)
            or np.any(theta <= 0) or np.any(theta >= 1)):
        raise ValueError('Interior shares and positive coefficients required')
    served = d*theta
    responsive = served*(1-theta)
    lsum = float(responsive @ lam)
    bsum = float(responsive @ beta)
    delay_prime = -lsum/bsum
    theta_prime = theta*(1-theta)*(-lam-beta*delay_prime)/scale
    group_cs = served*(-1-beta/lam*delay_prime)
    out = {'delay_derivative': delay_prime, 'share_derivative': theta_prime,
           'group_cs_derivative': group_cs,
           'us_derivative': float(sum(group_cs)),
           'ts_derivative': float(sum(group_cs)+sum(served))}
    if len(d) == 2:
        v = beta/lam
        out['two_group_us_formula'] = float(
            np.prod(served)*(v[1]-v[0])
            *(lam[0]*(1-theta[0])-lam[1]*(1-theta[1]))/bsum)
    return out
