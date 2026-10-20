"""Bounded nonsmooth hard-capacity repair, without global BR certification."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import brentq, least_squares

from .review_equilibrium import ReviewEngine, ReviewOutcome, projected_gradient


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class EvaluationBudget:
    limit: int = 200000
    used: int = 0
    deadline: float = float("inf")
    callback: object = None
    choice_calls: int = 0
    game: str = ""
    stage: str = ""

    def spend(self, kind, prices):
        if self.used >= self.limit or perf_counter() > self.deadline:
            raise BudgetExceeded("evaluation or wall budget exhausted")
        self.used += 1
        if self.callback:
            self.callback(dict(evaluation=self.used, game=self.game, stage=self.stage,
                               kind=kind, price_1=float(prices[0]),
                               price_2=float(prices[1]), price_3=float(prices[2])))


class RepairEngine(ReviewEngine):
    def __init__(self, case, budget=None):
        super().__init__(case, "hard")
        self.budget = budget or EvaluationBudget()
        self.cache = {}

    def choice(self, prices, rho, logphi=0.):
        self.budget.choice_calls += 1
        return super().choice(prices, rho, logphi)

    def outcome(self, prices):
        p = np.asarray(prices, dtype=float)
        key = tuple(p)
        if key not in self.cache:
            self.budget.spend("closed_demand", p)
            self.cache[key] = super().outcome(p)
        return self.cache[key]

    def transition_gap(self, prices):
        self.budget.spend("fixed_rho_choice", prices)
        return float(self.resource @ self.choice(np.asarray(prices), 1.)[0]/self.capacity-1)

    def unrestricted(self, prices):
        self.budget.spend("unrestricted_closed_demand", prices)
        self.kind = "soft"
        try:
            return ReviewEngine.outcome(self, prices)
        finally:
            self.kind = "hard"


def project_transition(engine, prices):
    p = np.asarray(prices, dtype=float)
    lower = float(np.max(engine.case.price_lower-p))
    upper = float(np.min(engine.case.price_upper-p))
    if lower > upper:
        return None
    radius = .5
    while True:
        a, b = max(lower, -radius), min(upper, radius)
        ga, gb = engine.transition_gap(p+a), engine.transition_gap(p+b)
        if ga*gb <= 0:
            delta = brentq(lambda d: engine.transition_gap(p+d), a, b,
                           xtol=2e-12, rtol=1e-14)
            q = p+delta
            return dict(prices=q.tolist(), delta=float(delta),
                        gap=engine.transition_gap(q), bracket=[a,b])
        if a == lower and b == upper:
            return None
        radius *= 2


def side_gradients(engine, prices, charge):
    """At rho=1, cheaper own fare is binding; dearer own fare is nonbinding."""
    p = np.asarray(prices, dtype=float)
    # At a transition rho=1 and phi=1, no additional scalar root is needed.
    y, prob, cs = engine.choice(p, 1.)
    binding = ReviewOutcome(y, 1., prob, cs, True, 0., 0.)
    free = ReviewOutcome(y, 1., prob, cs, False, 0., 0.)
    return engine.gradient(p, charge, binding), engine.gradient(p, charge, free)


def coordinate_br(engine, prices, charge, m, grid_points=41):
    """Whole-box grid plus transition and sampled smooth stationary candidates."""
    p = np.asarray(prices, dtype=float)
    lo, hi = engine.case.price_lower, engine.case.price_upper
    values, deriv, trace = {}, {}, []

    def evaluate(x, purpose):
        x = float(x)
        if x not in values:
            q = p.copy()
            q[m] = x
            o = engine.outcome(q)
            profit = float((x-engine.cost[m]-charge*engine.resource[m])*o.demand[m])
            gradient = float(engine.gradient(q, charge, o)[m])
            values[x], deriv[x] = profit, gradient
            trace.append(dict(platform=m, fare=x, profit=profit, derivative=gradient,
                              purpose=purpose))
        return values[x]

    baseline = evaluate(p[m], "base")
    grid = np.linspace(lo, hi, grid_points)
    profit_grid = [evaluate(x, "grid") for x in grid]
    qlo, qhi = p.copy(), p.copy()
    qlo[m], qhi[m] = lo, hi
    glo, ghi = engine.transition_gap(qlo), engine.transition_gap(qhi)
    transition = None
    if glo*ghi <= 0:
        def gap(x):
            q = p.copy()
            q[m] = x
            return engine.transition_gap(q)
        transition = float(brentq(gap, lo, hi, xtol=2e-12, rtol=1e-14))
        evaluate(transition, "capacity_transition")
    maxima = [i for i in range(1, grid_points-1)
              if profit_grid[i] >= profit_grid[i-1] and profit_grid[i] >= profit_grid[i+1]]
    for i in sorted(maxima, key=lambda j: -profit_grid[j])[:2]:
        a, b = float(grid[i-1]), float(grid[i+1])
        if transition is not None and a < transition < b:
            q = p.copy()
            q[m] = transition
            left, right = side_gradients(engine, q, charge)
            if left[m] >= 0 and right[m] <= 0:
                continue
            # Split the sampled maximum at the known kink; do not smooth it.
            intervals = [(a, transition-1e-8), (transition+1e-8, b)]
        else:
            intervals = [(a,b)]
        for a, b in intervals:
            if a >= b:
                continue
            evaluate(a, "local_bracket")
            evaluate(b, "local_bracket")
            if deriv[a] > 0 and deriv[b] < 0:
                def gradient(x):
                    evaluate(x, "stationary_root_probe")
                    return deriv[float(x)]
                root = brentq(gradient, a, b, xtol=1e-9, rtol=1e-12, maxiter=20)
                evaluate(root, "stationary_candidate")
    best = max(values, key=lambda x:(values[x], -abs(x-p[m]), -x))
    return dict(platform=m, baseline=baseline, best_fare=best,
                best_profit=values[best], gain=max(0., values[best]-baseline),
                transition=transition, trace=trace)


def verify_candidate(engine, prices, charge, gain_tol=1e-6):
    p = np.asarray(prices, dtype=float)
    o = engine.outcome(p)
    gap = engine.transition_gap(p)
    near = abs(gap) <= 1e-10
    gradient = engine.gradient(p, charge, o)
    kkt = float(np.max(abs(projected_gradient(p, gradient,
                engine.case.price_lower, engine.case.price_upper))))
    left, right = side_gradients(engine, p, charge) if near else (gradient, gradient)
    lo, hi = engine.case.price_lower, engine.case.price_upper
    left_violation = np.where(p > lo+1e-8, np.maximum(-left, 0), 0)
    right_violation = np.where(p < hi-1e-8, np.maximum(right, 0), 0)
    first_order = (max(np.max(left_violation), np.max(right_violation)) <= 1e-5
                   if near else kkt <= 1e-6)
    br = [coordinate_br(engine, p, charge, m) for m in range(3)]
    max_gain = max(b["gain"] for b in br)
    return dict(prices=p.tolist(), verified=bool(first_order and max_gain <= gain_tol),
                verification_type="one-sided/BR-verified" if near else "smooth-KKT/BR-verified",
                near_transition=bool(near), transition_gap=gap, kkt=kkt,
                left_gradient=left.tolist(), right_gradient=right.tolist(),
                first_order_ok=bool(first_order), max_gain=max_gain, br=br)


def smooth_polish(engine, prices, charge):
    lo, hi = engine.case.price_lower, engine.case.price_upper
    def residual(p):
        return projected_gradient(p, engine.gradient(p, charge), lo, hi)
    solved = least_squares(residual, np.clip(prices, lo+1e-9, hi-1e-9),
                           bounds=(lo,hi), max_nfev=25, xtol=1e-11, gtol=1e-11,
                           ftol=1e-11)
    p = np.clip(solved.x, lo, hi)
    g = engine.gradient(p, charge)
    snap = np.where((hi-p<1e-5)&(g>0), hi, p)
    snap = np.where((snap-lo<1e-5)&(g<0), lo, snap)
    return snap if np.max(abs(residual(snap))) <= np.max(abs(residual(p))) else p


def repair_game(engine, prices, charge, game_id="", max_passes=20):
    start = perf_counter()
    used0 = engine.budget.used
    old = np.asarray(prices, dtype=float)
    engine.budget.game = game_id
    engine.budget.stage = "projection_or_smooth"
    old_outcome = engine.outcome(old)
    gap = engine.transition_gap(old)
    projected = project_transition(engine, old) if abs(gap) <= .002 else None
    attempts = []

    def check(p, label):
        engine.budget.stage = label
        result = verify_candidate(engine, p, charge)
        result["candidate_source"] = label
        attempts.append(result)
        return result

    if projected:
        candidate = check(projected["prices"], "common_shift_projection")
    else:
        old_grad = engine.gradient(old, charge, old_outcome)
        kkt = float(np.max(abs(projected_gradient(old, old_grad, engine.case.price_lower, engine.case.price_upper))))
        candidate = check(old if kkt <= 1e-6 else smooth_polish(engine, old, charge),
                          "old_smooth" if kkt <= 1e-6 else "analytic_smooth_polish")
    if not candidate["verified"]:
        starts = [("projected", projected["prices"])] if projected else []
        starts += [("original", old.tolist()), ("fixed22", [22.,22.,22.])]
        for label, initial in starts:
            p = np.asarray(initial, dtype=float)
            for pass_index in range(max_passes):
                engine.budget.stage = f"GS_{label}_{pass_index}"
                for m in range(3):
                    br = coordinate_br(engine, p, charge, m)
                    p[m] = br["best_fare"]
                result = check(p, f"GS_{label}_pass{pass_index+1}")
                if result["verified"]:
                    break
            if any(a["verified"] for a in attempts):
                break
    verified = [a for a in attempts if a["verified"]]
    # Independent of E and welfare. Old case/charge continuation is the anchor.
    selected = min(verified, key=lambda a:(float(np.linalg.norm(np.asarray(a["prices"])-old)),
                   tuple(a["prices"]))) if verified else None
    return dict(game=game_id, charge=float(charge), old_prices=old.tolist(),
                old_rho=old_outcome.rho, old_cs=old_outcome.cs,
                old_transition_gap=gap, projection=projected, selected=selected,
                attempts=attempts, evaluations=engine.budget.used-used0,
                elapsed=perf_counter()-start)
