from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.optimize import brentq

from .config import DemandOutcome, MarketCase


def _as_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(tuple(values), dtype=float)


def _logsumexp(values: np.ndarray) -> float:
    peak = float(np.max(values))
    return peak + math.log(float(np.sum(np.exp(values - peak))))


class MarketEngine:
    def __init__(self, case: MarketCase):
        self.case = case
        self.group_count = 3
        self.platform_count = 3
        self.demand_base = _as_array(case.base_demand) * case.demand_scale
        self.capacity = case.base_capacity * case.capacity_scale
        self.resource = _as_array(case.platform_resource)
        self.time = _as_array(case.service_time)
        self.price_sensitivity = _as_array(case.price_sensitivity)
        self.time_sensitivity = _as_array(case.time_sensitivity)
        self.reliability_sensitivity = _as_array(case.reliability_sensitivity)
        self.outside_affinity = _as_array(case.outside_affinity)
        self.platform_affinity = np.asarray(case.platform_affinity, dtype=float)
        self.platform_affinity = self.platform_affinity + np.eye(3) * case.differentiation
        self.base_loss = _as_array(case.base_loss)
        self.exposure = _as_array(case.exposure)
        self.cost = _as_array(case.base_cost) + np.array((0.0, case.cost_gap, 0.5 * case.cost_gap))

    def reliability_function(self, utilization: float) -> float:
        u = min(max(float(utilization), 0.0), 1.45)
        overload = max(u - 0.88, 0.0)
        return 0.55 * (u ** self.case.reliability_power) + 1.10 * overload * overload

    def _demand_at_utilization(self, prices: np.ndarray, utilization: float) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        reliability = self.base_loss + self.exposure * self.reliability_function(utilization)
        prob = np.zeros((self.group_count, self.platform_count))
        for g in range(self.group_count):
            air_utility = (
                self.platform_affinity[g]
                - self.price_sensitivity[g] * prices
                - self.time_sensitivity[g] * self.time
                - self.reliability_sensitivity[g] * reliability
            )
            outside_utility = (
                self.outside_affinity[g]
                - self.price_sensitivity[g] * self.case.outside_price
                - self.time_sensitivity[g] * self.case.outside_time
            )
            values = np.concatenate(([outside_utility], air_utility)) / self.case.logit_scale
            denom = _logsumexp(values)
            all_prob = np.exp(values - denom)
            prob[g, :] = all_prob[1:]

        demand = self.demand_base @ prob
        next_utilization = float(np.dot(self.resource, demand) / self.capacity)
        return demand, min(max(next_utilization, 0.0), 1.70), reliability, prob

    def _iterated_demand_reliability(self, prices: np.ndarray) -> DemandOutcome:
        p = _as_array(prices)
        utilization = 0.70
        demand = np.zeros(self.platform_count)
        prob = np.zeros((self.group_count, self.platform_count))
        reliability = self.base_loss.copy()
        last_gap = math.inf

        for iteration in range(1, 81):
            new_demand, new_utilization, reliability, prob = self._demand_at_utilization(p, utilization)
            last_gap = abs(new_utilization - utilization)
            demand = new_demand
            utilization = 0.35 * utilization + 0.65 * new_utilization
            if last_gap <= 1.0e-8:
                break

        reliability = self.base_loss + self.exposure * self.reliability_function(utilization)
        return DemandOutcome(
            demand=tuple(float(x) for x in demand),
            utilization=float(utilization),
            reliability=tuple(float(x) for x in reliability),
            choice_prob=tuple(tuple(float(x) for x in row) for row in prob),
            residual=float(last_gap),
            iterations=iteration,
        )

    def solve_demand_reliability(self, prices: Iterable[float]) -> DemandOutcome:
        p = _as_array(prices)
        iterated = self._iterated_demand_reliability(p)
        if iterated.residual <= 1.0e-7:
            return iterated

        def gap(utilization: float) -> float:
            _, next_utilization, _, _ = self._demand_at_utilization(p, utilization)
            return next_utilization - utilization

        grid = np.linspace(0.0, 1.70, 69)
        values = [gap(float(u)) for u in grid]
        bracket = None
        for lower, upper, lower_value, upper_value in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
            if abs(lower_value) <= 1.0e-10:
                root = float(lower)
                break
            if lower_value * upper_value <= 0.0:
                bracket = (float(lower), float(upper))
                root = brentq(gap, bracket[0], bracket[1], xtol=1.0e-10, rtol=1.0e-10, maxiter=80)
                break
        else:
            return self._iterated_demand_reliability(p)

        demand, next_utilization, reliability, prob = self._demand_at_utilization(p, root)
        return DemandOutcome(
            demand=tuple(float(x) for x in demand),
            utilization=float(root),
            reliability=tuple(float(x) for x in reliability),
            choice_prob=tuple(tuple(float(x) for x in row) for row in prob),
            residual=abs(float(next_utilization - root)),
            iterations=0 if bracket is None else 1,
        )

    def platform_profit(self, prices: Iterable[float], platform: int, charge: float) -> float:
        p = _as_array(prices)
        outcome = self.solve_demand_reliability(p)
        y = outcome.demand[platform]
        variable_cost = self.cost[platform] * y
        resource_payment = charge * self.resource[platform] * y
        return float(p[platform] * y - variable_cost - resource_payment)

    def consumer_surplus(self, prices: Iterable[float], reliability: Iterable[float]) -> float:
        p = _as_array(prices)
        loss = _as_array(reliability)
        total = 0.0
        for g in range(self.group_count):
            air_utility = (
                self.platform_affinity[g]
                - self.price_sensitivity[g] * p
                - self.time_sensitivity[g] * self.time
                - self.reliability_sensitivity[g] * loss
            )
            outside_utility = (
                self.outside_affinity[g]
                - self.price_sensitivity[g] * self.case.outside_price
                - self.time_sensitivity[g] * self.case.outside_time
            )
            values = np.concatenate(([outside_utility], air_utility)) / self.case.logit_scale
            total += self.demand_base[g] * (self.case.logit_scale / self.price_sensitivity[g]) * _logsumexp(values)
        return float(total)

    def operating_cost(self, demand: Iterable[float]) -> float:
        y = _as_array(demand)
        return float(np.dot(self.cost, y))

    def system_cost(self, utilization: float) -> float:
        overload = max(float(utilization) - 0.90, 0.0)
        return float(self.case.congestion_cost * (0.15 * utilization * utilization + overload**3))

    def welfare(self, prices: Iterable[float]) -> float:
        outcome = self.solve_demand_reliability(prices)
        return (
            self.consumer_surplus(prices, outcome.reliability)
            - self.operating_cost(outcome.demand)
            - self.system_cost(outcome.utilization)
        )
