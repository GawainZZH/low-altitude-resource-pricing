from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


Vector = Tuple[float, ...]
Matrix = Tuple[Vector, ...]


@dataclass(frozen=True)
class MarketCase:
    name: str
    demand_scale: float
    capacity_scale: float
    reliability_power: float
    differentiation: float
    cost_gap: float
    base_demand: Vector = (150.0, 125.0, 100.0)
    base_capacity: float = 360.0
    platform_resource: Vector = (1.00, 1.04, 0.98)
    service_time: Vector = (15.0, 14.5, 16.0)
    price_sensitivity: Vector = (0.115, 0.090, 0.072)
    time_sensitivity: Vector = (0.030, 0.038, 0.047)
    reliability_sensitivity: Vector = (0.50, 0.65, 0.82)
    outside_affinity: Vector = (1.65, 1.55, 1.45)
    outside_price: float = 8.2
    outside_time: float = 34.0
    platform_affinity: Matrix = (
        (3.05, 2.72, 2.54),
        (2.66, 3.00, 2.70),
        (2.52, 2.68, 3.08),
    )
    base_loss: Vector = (0.06, 0.07, 0.06)
    exposure: Vector = (0.35, 0.42, 0.39)
    base_cost: Vector = (6.2, 6.4, 6.6)
    congestion_cost: float = 90.0
    logit_scale: float = 1.0
    price_lower: float = 5.0
    price_upper: float = 38.0


@dataclass(frozen=True)
class DemandOutcome:
    demand: Vector
    utilization: float
    reliability: Vector
    choice_prob: Matrix
    residual: float
    iterations: int


@dataclass(frozen=True)
class PriceSolution:
    prices: Vector
    demand: Vector
    utilization: float
    reliability: Vector
    residual: float
    welfare: float
    charge: float
