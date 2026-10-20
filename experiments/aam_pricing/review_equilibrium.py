"""Read-only diagnostic model: physical closure and analytic own-profit gradients.

Not a replacement for the published engine or a global equilibrium certificate.
The published h(rho) cap at 1.45 is intentionally retained.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import brentq, least_squares, minimize_scalar
from scipy.special import logsumexp

from .engine import MarketEngine
from .out_of_model_demand import TasteMixtureEngine, taste_mixture_specs


@dataclass
class ReviewOutcome:
    demand: np.ndarray
    rho: float
    prob: np.ndarray
    cs: float
    binding: bool
    logphi: float
    residual: float


class ReviewEngine:
    def __init__(self, case, kind="soft"):
        if kind not in ("soft", "mixture", "hard"):
            raise ValueError(kind)
        self.source = (TasteMixtureEngine(case, taste_mixture_specs()[1])
                       if kind == "mixture" else MarketEngine(case))
        for name, value in vars(self.source).items():
            setattr(self, name, value.copy() if isinstance(value, np.ndarray) else value)
        self.kind = kind
        self.logit_scale = getattr(self.source, "logit_scale", case.logit_scale)
        self.rmax = float(sum(self.demand_base) * max(self.resource) / self.capacity)

    def hprime(self, rho):
        if rho < 0 or rho >= 1.45:
            return 0.0
        q = self.case.reliability_power
        return 0.55*q*rho**(q-1) + 2.20*max(rho-0.88, 0)

    def choice(self, prices, rho, logphi=0.0):
        loss = self.base_loss + self.exposure*self.source.reliability_function(rho)
        air = (self.platform_affinity - self.price_sensitivity[:, None]*prices
               - self.time_sensitivity[:, None]*self.time
               - self.reliability_sensitivity[:, None]*loss)/self.logit_scale + logphi
        outside = (self.outside_affinity - self.price_sensitivity*self.case.outside_price
                   - self.time_sensitivity*self.case.outside_time)/self.logit_scale
        denominator = np.logaddexp(outside, logsumexp(air, axis=1))
        prob = np.exp(air-denominator[:, None])
        demand = self.demand_base @ prob
        cs = float(np.dot(self.demand_base*self.logit_scale/self.price_sensitivity, denominator))
        return demand, prob, cs

    def outcome(self, prices):
        p = np.asarray(prices, dtype=float)
        if self.kind == "hard":
            y1, _, _ = self.choice(p, 1.0)
            if self.resource @ y1 > self.capacity:
                def gap(logphi):
                    return float(self.resource @ self.choice(p, 1.0, logphi)[0]/self.capacity-1)
                logphi = brentq(gap, -745., 0., xtol=2e-13, rtol=1e-14)
                y, prob, cs = self.choice(p, 1.0, logphi)
                return ReviewOutcome(y, 1.0, prob, cs, True, logphi,
                                     abs(float(self.resource @ y/self.capacity-1)))
            upper = 1.0
        else:
            upper = self.rmax

        def gap(rho):
            return float(self.resource @ self.choice(p, rho)[0]/self.capacity-rho)

        rho = brentq(gap, 0., upper, xtol=2e-13, rtol=1e-14)
        y, prob, cs = self.choice(p, rho)
        return ReviewOutcome(y, rho, prob, cs, False, 0.,
                             abs(float(self.resource @ y/self.capacity-rho)))

    def jacobian(self, prices, outcome=None):
        o = self.outcome(prices) if outcome is None else outcome
        p = o.prob
        w = self.demand_base*self.price_sensitivity/self.logit_scale
        b = np.einsum("g,gm,gj->mj", w, p, p) - np.diag(w @ p)
        if o.binding:
            z = (self.demand_base*(1-p.sum(axis=1))) @ p
            logphi_p = -(self.resource @ b)/float(self.resource @ z)
            return b + np.outer(z, logphi_p)
        centered = self.exposure[None, :] - (p @ self.exposure)[:, None]
        v = ((-self.demand_base*self.reliability_sensitivity/self.logit_scale)[:, None]
             * p * centered).sum(axis=0)*self.hprime(o.rho)
        omega_p = self.resource @ b/self.capacity
        omega_rho = float(self.resource @ v/self.capacity)
        rho_p = omega_p/(1-omega_rho)
        return b + np.outer(v, rho_p)

    def gradient(self, prices, charge, outcome=None):
        o = self.outcome(prices) if outcome is None else outcome
        return o.demand + (np.asarray(prices)-self.cost-charge*self.resource)*np.diag(self.jacobian(prices, o))

    def profit(self, prices, platform, charge):
        o = self.outcome(prices)
        return float((prices[platform]-self.cost[platform]-charge*self.resource[platform])*o.demand[platform])

    def welfare(self, prices, weight=None, outcome=None):
        o = self.outcome(prices) if outcome is None else outcome
        e = self.case.congestion_cost if weight is None else weight
        us = o.cs - float(self.cost @ o.demand) - e*(0.15*o.rho**2+max(o.rho-.90, 0)**3)
        return float(us), float(us+np.dot(prices, o.demand))


def projected_gradient(prices, gradient, lower, upper):
    p = np.asarray(prices, dtype=float)
    return p - np.clip(p+np.asarray(gradient)/100., lower, upper)


def solve_game(engine, charge, previous=None, max_nfev=120):
    """Two fixed starts; report every attempt, including unsuccessful termination."""
    start_time = perf_counter()
    lo, hi = engine.case.price_lower, engine.case.price_upper
    starts = (previous if previous is not None else (22.,)*3, (22.,)*3)
    attempts = []
    for start in starts:
        def residual(p):
            return projected_gradient(p, engine.gradient(p, charge), lo, hi)
        try:
            opt = least_squares(residual, np.clip(start, lo+1e-9, hi-1e-9),
                                bounds=(lo, hi), xtol=1e-11, ftol=1e-11,
                                gtol=1e-11, max_nfev=max_nfev)
            p = np.clip(opt.x, lo, hi)
            raw_prices = p.tolist()
            raw_kkt = float(np.max(abs(residual(p))))
            # Bounded TRF approaches active bounds asymptotically. Evaluate exact
            # bounds only when close and the analytic gradient points outward.
            g = engine.gradient(p, charge)
            snapped = np.where((hi-p < 1e-5) & (g > 0), hi, p)
            snapped = np.where((snapped-lo < 1e-5) & (g < 0), lo, snapped)
            if np.max(abs(residual(snapped))) <= np.max(abs(residual(p))):
                p = snapped
            kkt = float(np.max(abs(residual(p))))
            attempts.append(dict(start=list(start), prices=p.tolist(), kkt=kkt,
                                 raw_prices=raw_prices, raw_kkt=raw_kkt,
                                 success=bool(opt.success), status=int(opt.status),
                                 nfev=int(opt.nfev), message=opt.message))
        except Exception as exc:
            attempts.append(dict(start=list(start), prices=list(start), kkt=float("inf"),
                                 success=False, status=-999, nfev=0, message=repr(exc)))
    best = min(attempts, key=lambda a: a["kkt"])
    return dict(prices=best["prices"], kkt=best["kkt"],
                accepted=bool(best["success"] and best["kkt"] <= 1e-6),
                start_price_spread=float(np.max(abs(np.array(attempts[0]["prices"])-attempts[1]["prices"]))),
                attempts=attempts, elapsed=perf_counter()-start_time)


def wide_deviations(engine, prices, charge, evaluation_cap=210):
    """Sampled box search, local sampled-maximum refinement, transition candidate.

    The cap includes base profit calls and transition root probes. No certificate.
    """
    start = perf_counter()
    p = np.asarray(prices, dtype=float)
    lo, hi = engine.case.price_lower, engine.case.price_upper
    rows = []
    used = 0
    for m in range(3):
        per_platform_cap = evaluation_cap//3
        cache = {}
        local_used = 0

        def evaluate(x, purpose="local"):
            nonlocal used, local_used
            key = float(x)
            if key in cache:
                return cache[key]
            if local_used >= per_platform_cap:
                raise RuntimeError("deviation budget exhausted")
            q = p.copy()
            q[m] = x
            value = engine.profit(q, m, charge)
            used += 1
            local_used += 1
            cache[key] = value
            rows.append(dict(platform=m, fare=key, profit=value, purpose=purpose))
            return value

        baseline = evaluate(p[m], "base")
        grid = np.linspace(lo, hi, 41)
        vals = [evaluate(x, "grid") for x in grid]
        maxima = [i for i in range(1, 40) if vals[i] >= vals[i-1] and vals[i] >= vals[i+1]]
        # Refine at most the two largest sampled local maxima (never extra starts).
        for i in sorted(maxima, key=lambda j: -vals[j])[:2]:
            budget = min(9, per_platform_cap-local_used-6)
            if budget > 0:
                minimize_scalar(lambda x: -evaluate(x), bounds=(grid[i-1], grid[i+1]),
                                method="bounded", options={"maxiter": budget, "xatol":1e-5})
        if engine.kind == "hard" and local_used < per_platform_cap-2:
            def transition(x):
                nonlocal used, local_used
                if local_used >= per_platform_cap-1:
                    raise RuntimeError("transition budget exhausted")
                q = p.copy()
                q[m] = x
                gap = float(engine.resource @ engine.choice(q, 1.0)[0]/engine.capacity-1)
                used += 1
                local_used += 1
                rows.append(dict(platform=m, fare=float(x), profit=None,
                                 purpose="transition_probe", transition_gap=gap))
                return gap
            left, right = transition(lo), transition(hi)
            if left*right < 0:
                try:
                    root = brentq(transition, lo, hi, xtol=1e-6,
                                  maxiter=max(1, per_platform_cap-local_used-2))
                    evaluate(root, "transition_candidate")
                except RuntimeError:
                    pass
        best_fare, best_profit = max(cache.items(), key=lambda kv: kv[1])
        rows.append(dict(platform=m, fare=best_fare, profit=best_profit, purpose="summary",
                         baseline=baseline, gain=max(0., best_profit-baseline),
                         evaluations=local_used))
    return dict(rows=rows, evaluations=used, elapsed=perf_counter()-start)
