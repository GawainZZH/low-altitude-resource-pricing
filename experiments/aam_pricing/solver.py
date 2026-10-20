from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.optimize import least_squares

from .config import PriceSolution
from .engine import MarketEngine


def _profit_gradient(engine: MarketEngine, prices: np.ndarray, charge: float) -> np.ndarray:
    step = 0.03
    grad = np.zeros(engine.platform_count)
    for m in range(engine.platform_count):
        upper = prices.copy()
        lower = prices.copy()
        upper[m] = min(engine.case.price_upper, upper[m] + step)
        lower[m] = max(engine.case.price_lower, lower[m] - step)
        width = max(upper[m] - lower[m], 1.0e-9)
        grad[m] = (
            engine.platform_profit(upper, m, charge)
            - engine.platform_profit(lower, m, charge)
        ) / width
    return grad / 100.0


def solve_platform_prices(
    engine: MarketEngine,
    charge: float,
    starts: Iterable[Iterable[float]] | None = None,
) -> PriceSolution:
    if starts is None:
        starts = (
            (14.0, 14.5, 15.0),
            (18.0, 18.5, 19.0),
            (23.0, 23.0, 23.0),
            (29.0, 29.0, 29.0),
        )

    lower = np.full(engine.platform_count, engine.case.price_lower)
    upper = np.full(engine.platform_count, engine.case.price_upper)
    candidates = []

    for start in starts:
        initial = np.clip(np.asarray(tuple(start), dtype=float), lower, upper)

        def residual_fn(price_vector: np.ndarray) -> np.ndarray:
            return _profit_gradient(engine, price_vector, charge)

        solved = least_squares(
            residual_fn,
            initial,
            bounds=(lower, upper),
            xtol=1.0e-8,
            ftol=1.0e-8,
            gtol=1.0e-8,
            max_nfev=80,
        )
        prices = np.clip(solved.x, lower, upper)
        residual = float(np.max(np.abs(_profit_gradient(engine, prices, charge))))
        outcome = engine.solve_demand_reliability(prices)
        welfare = engine.welfare(prices)
        candidates.append(
            PriceSolution(
                prices=tuple(float(x) for x in prices),
                demand=outcome.demand,
                utilization=outcome.utilization,
                reliability=outcome.reliability,
                residual=residual,
                welfare=float(welfare),
                charge=float(charge),
            )
        )

    acceptable = [item for item in candidates if item.residual <= 1.0e-4]
    if acceptable:
        return max(acceptable, key=lambda item: item.welfare)
    return min(candidates, key=lambda item: item.residual)
