from __future__ import annotations

import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from .comprehensive_state_grid import build_state_grid, charge_grid
from .config import DemandOutcome, MarketCase
from .engine import MarketEngine, _logsumexp
from .solver import solve_platform_prices


TOLERANCE = 1.0e-8
UTILIZATION_THRESHOLDS = (0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20)
UTILIZATION_SLOPES = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _as_float(row: dict, key: str) -> float:
    return float(row[key])


def _average(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= TOLERANCE:
        return 0.0
    return float(np.dot(values, weights) / total)


def _nearest_charge(charges: tuple[float, ...], target: float) -> float:
    capped = min(max(float(target), min(charges)), max(charges))
    return min(charges, key=lambda charge: (abs(charge - capped), charge))


class HardCapacityMarketEngine(MarketEngine):
    """Market engine with direct hard-capacity rationing in demand and profit."""

    def __init__(self, case: MarketCase):
        super().__init__(case)
        self.last_availability = 1.0
        self.last_unrestricted_resource = 0.0
        self.last_unrestricted_utilization = 0.0
        self.last_consumer_surplus = 0.0

    def _resource_demand(self, demand: np.ndarray) -> float:
        return float(np.dot(self.resource, demand))

    def _choice_given_reliability(
        self,
        prices: np.ndarray,
        reliability: np.ndarray,
        availability: float,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        safe_availability = max(float(availability), 1.0e-14)
        log_availability = math.log(safe_availability)
        demand = np.zeros(self.platform_count)
        prob = np.zeros((self.group_count, self.platform_count))
        consumer_surplus = 0.0
        for group in range(self.group_count):
            air_utility = (
                self.platform_affinity[group]
                - self.price_sensitivity[group] * prices
                - self.time_sensitivity[group] * self.time
                - self.reliability_sensitivity[group] * reliability
            )
            outside_utility = (
                self.outside_affinity[group]
                - self.price_sensitivity[group] * self.case.outside_price
                - self.time_sensitivity[group] * self.case.outside_time
            )
            values = np.concatenate(
                ([outside_utility], air_utility + self.case.logit_scale * log_availability)
            ) / self.case.logit_scale
            denominator = _logsumexp(values)
            probabilities = np.exp(values - denominator)
            prob[group, :] = probabilities[1:]
            demand += self.demand_base[group] * probabilities[1:]
            consumer_surplus += (
                self.demand_base[group]
                * (self.case.logit_scale / self.price_sensitivity[group])
                * denominator
            )
        return demand, float(consumer_surplus), prob

    def solve_hard_outcome(self, prices) -> tuple[DemandOutcome, float, float, float, float]:
        p = np.asarray(tuple(prices), dtype=float)
        unrestricted = self.solve_demand_reliability_baseline(p)
        unrestricted_demand = np.asarray(unrestricted.demand, dtype=float)
        unrestricted_resource = self._resource_demand(unrestricted_demand)
        unrestricted_utilization = unrestricted_resource / self.capacity

        if unrestricted_resource <= self.capacity + TOLERANCE:
            reliability = np.asarray(unrestricted.reliability, dtype=float)
            demand, consumer_surplus, prob = self._choice_given_reliability(p, reliability, 1.0)
            availability = 1.0
            residual = max(self._resource_demand(demand) - self.capacity, 0.0)
        else:
            reliability = np.asarray(
                self.base_loss + self.exposure * self.reliability_function(1.0),
                dtype=float,
            )

            def resource_gap(availability: float) -> float:
                demand_at_availability, _surplus, _prob = self._choice_given_reliability(
                    p,
                    reliability,
                    availability,
                )
                return self._resource_demand(demand_at_availability) - self.capacity

            availability = float(
                brentq(resource_gap, 1.0e-12, 1.0, xtol=1.0e-11, rtol=1.0e-11, maxiter=100)
            )
            demand, consumer_surplus, prob = self._choice_given_reliability(
                p,
                reliability,
                availability,
            )
            residual = abs(self._resource_demand(demand) - self.capacity)

        served_resource = self._resource_demand(demand)
        utilization = min(served_resource / self.capacity, 1.0)
        self.last_availability = float(availability)
        self.last_unrestricted_resource = float(unrestricted_resource)
        self.last_unrestricted_utilization = float(unrestricted_utilization)
        self.last_consumer_surplus = float(consumer_surplus)

        outcome = DemandOutcome(
            demand=tuple(float(x) for x in demand),
            utilization=float(utilization),
            reliability=tuple(float(x) for x in reliability),
            choice_prob=tuple(tuple(float(x) for x in row) for row in prob),
            residual=float(residual),
            iterations=1,
        )
        return (
            outcome,
            float(availability),
            float(unrestricted_resource),
            float(unrestricted_utilization),
            float(consumer_surplus),
        )

    def solve_demand_reliability_baseline(self, prices) -> DemandOutcome:
        return super().solve_demand_reliability(prices)

    def solve_demand_reliability(self, prices) -> DemandOutcome:
        outcome, _availability, _resource, _utilization, _surplus = self.solve_hard_outcome(prices)
        return outcome

    def platform_profit(self, prices, platform: int, charge: float) -> float:
        p = np.asarray(tuple(prices), dtype=float)
        outcome = self.solve_demand_reliability(p)
        y = outcome.demand[platform]
        variable_cost = self.cost[platform] * y
        resource_payment = charge * self.resource[platform] * y
        return float(p[platform] * y - variable_cost - resource_payment)

    def consumer_surplus(self, prices, reliability) -> float:
        _outcome, _availability, _resource, _utilization, surplus = self.solve_hard_outcome(prices)
        return float(surplus)

    def welfare(self, prices) -> float:
        outcome, _availability, _resource, _utilization, surplus = self.solve_hard_outcome(prices)
        return float(
            surplus
            - self.operating_cost(outcome.demand)
            - self.system_cost(outcome.utilization)
        )


def _solve_case_charge_rows(args: tuple[MarketCase, tuple[float, ...]]) -> list[dict]:
    case, charges = args
    engine = HardCapacityMarketEngine(case)
    rows = []
    previous_prices: tuple[float, ...] | None = None
    for charge in charges:
        starts = (previous_prices,) if previous_prices is not None else ((22.0, 22.0, 22.0),)
        solved = solve_platform_prices(engine, charge=charge, starts=starts)
        previous_prices = solved.prices
        outcome, availability, unrestricted_resource, unrestricted_utilization, surplus = (
            engine.solve_hard_outcome(solved.prices)
        )
        demand = np.asarray(outcome.demand, dtype=float)
        reliability = np.asarray(outcome.reliability, dtype=float)
        prices = np.asarray(solved.prices, dtype=float)
        served_resource = float(np.dot(engine.resource, demand))
        rows.append(
            {
                "case": case.name,
                "demand_scale": case.demand_scale,
                "capacity_scale": case.capacity_scale,
                "reliability_power": case.reliability_power,
                "externality_weight": case.congestion_cost,
                "charge": float(charge),
                "price_1": solved.prices[0],
                "price_2": solved.prices[1],
                "price_3": solved.prices[2],
                "demand_1": outcome.demand[0],
                "demand_2": outcome.demand[1],
                "demand_3": outcome.demand[2],
                "total_demand": float(np.sum(demand)),
                "served_resource_demand": served_resource,
                "utilization": outcome.utilization,
                "unrestricted_resource_demand": unrestricted_resource,
                "unrestricted_utilization": unrestricted_utilization,
                "hard_availability": availability,
                "hard_binding": int(availability < 1.0 - 1.0e-8),
                "average_fare": _average(prices, demand),
                "average_reliability": _average(reliability, demand),
                "consumer_surplus": surplus,
                "welfare": solved.welfare,
                "charge_revenue": float(charge * served_resource),
                "price_residual": solved.residual,
                "demand_residual": outcome.residual,
            }
        )
    return rows


def _solve_charge_rows(cases: list[MarketCase], charges: tuple[float, ...], workers: int) -> list[dict]:
    tasks = [(case, charges) for case in cases]
    if workers <= 1:
        nested = [_solve_case_charge_rows(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            nested = list(executor.map(_solve_case_charge_rows, tasks))
    rows = [row for case_rows in nested for row in case_rows]
    return sorted(rows, key=lambda row: (row["case"], row["charge"]))


def _group_by_case(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["case"], []).append(row)
    return grouped


def _select_static_charge(charge_rows: list[dict], charges: tuple[float, ...]) -> float:
    welfare_by_charge = {charge: [] for charge in charges}
    for row in charge_rows:
        welfare_by_charge[_as_float(row, "charge")].append(_as_float(row, "welfare"))
    return max(
        welfare_by_charge,
        key=lambda charge: (sum(welfare_by_charge[charge]) / len(welfare_by_charge[charge]), -charge),
    )


def _ut_row_for_case(rows: list[dict], threshold: float, slope: float) -> dict:
    charges = tuple(sorted({_as_float(row, "charge") for row in rows}))
    by_charge = {_as_float(row, "charge"): row for row in rows}
    nc_utilization = _as_float(by_charge[0.0], "unrestricted_utilization")
    target = slope * max(0.0, nc_utilization - threshold)
    return by_charge[_nearest_charge(charges, target)]


def _tune_ut(by_case: dict[str, list[dict]]) -> dict:
    rows = []
    for threshold in UTILIZATION_THRESHOLDS:
        for slope in UTILIZATION_SLOPES:
            selected = [_ut_row_for_case(case_rows, threshold, slope) for case_rows in by_case.values()]
            rows.append(
                {
                    "ut_threshold": threshold,
                    "ut_slope": slope,
                    "mean_welfare": sum(_as_float(row, "welfare") for row in selected) / len(selected),
                    "mean_charge": sum(_as_float(row, "charge") for row in selected) / len(selected),
                    "positive_charge_share": sum(_as_float(row, "charge") > TOLERANCE for row in selected)
                    / len(selected),
                }
            )
    return max(
        rows,
        key=lambda row: (
            row["mean_welfare"],
            -row["mean_charge"],
            -row["positive_charge_share"],
            -row["ut_slope"],
            row["ut_threshold"],
        ),
    )


def _select_air_charges(by_case: dict[str, list[dict]]) -> tuple[dict[str, float], float, int]:
    charges = tuple(sorted({_as_float(row, "charge") for rows in by_case.values() for row in rows}))
    active_cases = []
    for case, rows in by_case.items():
        ordered = sorted(rows, key=lambda row: _as_float(row, "charge"))
        zero = ordered[0]
        first_positive = next(row for row in ordered if _as_float(row, "charge") > 0.0)
        step = _as_float(first_positive, "charge") - _as_float(zero, "charge")
        activation_index = (_as_float(first_positive, "welfare") - _as_float(zero, "welfare")) / step
        if activation_index > TOLERANCE:
            active_cases.append(case)

    if not active_cases:
        return {case: 0.0 for case in by_case}, 0.0, 0

    scores = []
    for charge in charges:
        if charge <= TOLERANCE:
            continue
        selected = []
        for case in active_cases:
            rows = by_case[case]
            selected.append(min(rows, key=lambda row: abs(_as_float(row, "charge") - charge)))
        scores.append(
            (
                sum(_as_float(row, "welfare") for row in selected) / len(selected),
                -charge,
                charge,
            )
        )
    positive_charge = max(scores)[2]
    return (
        {
            case: positive_charge if case in active_cases else 0.0
            for case in by_case
        },
        positive_charge,
        len(active_cases),
    )


def _policy_rows(charge_rows: list[dict], charges: tuple[float, ...]) -> tuple[list[dict], dict]:
    by_case = _group_by_case(charge_rows)
    static_charge = _select_static_charge(charge_rows, charges)
    ut_params = _tune_ut(by_case)
    air_charge_by_case, air_positive_charge, activation_count = _select_air_charges(by_case)
    selected = []
    for case, rows in sorted(by_case.items()):
        nc = min(rows, key=lambda row: abs(_as_float(row, "charge")))
        sc = min(rows, key=lambda row: abs(_as_float(row, "charge") - static_charge))
        ut = _ut_row_for_case(rows, ut_params["ut_threshold"], ut_params["ut_slope"])
        rm = max(rows, key=lambda row: _as_float(row, "charge_revenue"))
        sd = max(rows, key=lambda row: _as_float(row, "welfare"))
        air_target = air_charge_by_case[case]
        air = min(rows, key=lambda row: abs(_as_float(row, "charge") - air_target))
        ordered = sorted(rows, key=lambda row: _as_float(row, "charge"))
        first_positive = next(row for row in ordered if _as_float(row, "charge") > 0.0)
        step = _as_float(first_positive, "charge") - _as_float(nc, "charge")
        activation_index = (_as_float(first_positive, "welfare") - _as_float(nc, "welfare")) / step
        choices = {
            "NC": nc,
            "SC": sc,
            "UT": ut,
            "RM": rm,
            "AIR": air,
            "SD": sd,
        }
        nc_welfare = _as_float(nc, "welfare")
        for policy, source in choices.items():
            row = dict(source)
            row["policy"] = policy
            row["activation_index_fd"] = activation_index
            row["sd_charge"] = _as_float(sd, "charge")
            row["sd_positive_charge"] = int(_as_float(sd, "charge") > TOLERANCE)
            row["false_positive_vs_sd"] = int(
                policy == "AIR"
                and _as_float(row, "charge") > TOLERANCE
                and _as_float(sd, "charge") <= TOLERANCE
            )
            row["false_negative_vs_sd"] = int(
                policy == "AIR"
                and _as_float(row, "charge") <= TOLERANCE
                and _as_float(sd, "charge") > TOLERANCE
            )
            row["delta_w_vs_nc"] = _as_float(row, "welfare") - nc_welfare
            row["delta_w_percent_vs_nc"] = 100.0 * row["delta_w_vs_nc"] / max(abs(nc_welfare), 1.0)
            selected.append(row)

    meta = {
        "selected_static_charge": static_charge,
        "ut_threshold": ut_params["ut_threshold"],
        "ut_slope": ut_params["ut_slope"],
        "air_positive_charge": air_positive_charge,
        "activation_positive_state_count": activation_count,
    }
    return selected, meta


def _summary(charge_rows: list[dict], policy_rows: list[dict], meta: dict, profile: str, charges: tuple[float, ...]) -> dict:
    by_policy: dict[str, list[dict]] = {}
    for row in policy_rows:
        by_policy.setdefault(row["policy"], []).append(row)

    def mean(policy: str, key: str) -> float:
        values = [_as_float(row, key) for row in by_policy[policy]]
        return sum(values) / len(values)

    def share(policy: str, predicate) -> float:
        rows = by_policy[policy]
        return sum(1 for row in rows if predicate(row)) / len(rows)

    sd_gain_total = sum(_as_float(row, "delta_w_vs_nc") for row in by_policy["SD"])
    air_gain_total = sum(_as_float(row, "delta_w_vs_nc") for row in by_policy["AIR"])
    return {
        "profile": profile,
        "state_count": len(by_policy["NC"]),
        "charge_count": len(charges),
        "selected_static_charge": meta["selected_static_charge"],
        "ut_threshold": meta["ut_threshold"],
        "ut_slope": meta["ut_slope"],
        "air_positive_charge": meta["air_positive_charge"],
        "capacity_binding_nc_share": share("NC", lambda row: _as_float(row, "hard_binding") > 0.0),
        "mean_nc_unrestricted_utilization": mean("NC", "unrestricted_utilization"),
        "mean_nc_availability": mean("NC", "hard_availability"),
        "positive_charge_share": {
            policy: share(policy, lambda row: _as_float(row, "charge") > TOLERANCE)
            for policy in sorted(by_policy)
        },
        "mean_charge": {
            policy: mean(policy, "charge")
            for policy in sorted(by_policy)
        },
        "mean_delta_w_vs_nc": {
            policy: mean(policy, "delta_w_vs_nc")
            for policy in sorted(by_policy)
        },
        "mean_delta_w_percent_vs_nc": {
            policy: mean(policy, "delta_w_percent_vs_nc")
            for policy in sorted(by_policy)
        },
        "mean_utilization": {
            policy: mean(policy, "utilization")
            for policy in sorted(by_policy)
        },
        "activation_positive_state_share": meta["activation_positive_state_count"] / len(by_policy["NC"]),
        "air_false_positive_share_vs_sd": sum(int(row["false_positive_vs_sd"]) for row in by_policy["AIR"])
        / len(by_policy["AIR"]),
        "air_false_negative_share_vs_sd": sum(int(row["false_negative_vs_sd"]) for row in by_policy["AIR"])
        / len(by_policy["AIR"]),
        "air_welfare_recovery_vs_sd": air_gain_total / sd_gain_total if sd_gain_total > TOLERANCE else 1.0,
        "max_price_residual": max(_as_float(row, "price_residual") for row in charge_rows),
        "max_demand_residual": max(_as_float(row, "demand_residual") for row in charge_rows),
    }


def _summary_rows(summary: dict) -> list[dict]:
    rows = []
    for policy in sorted(summary["positive_charge_share"]):
        rows.append(
            {
                "policy": policy,
                "positive_charge_share": summary["positive_charge_share"][policy],
                "mean_charge": summary["mean_charge"][policy],
                "mean_delta_w_vs_nc": summary["mean_delta_w_vs_nc"][policy],
                "mean_delta_w_percent_vs_nc": summary["mean_delta_w_percent_vs_nc"][policy],
                "mean_utilization": summary["mean_utilization"][policy],
            }
        )
    return rows


def run_hard_capacity_reequilibrium(
    output_dir: Path,
    profile: str = "full",
    workers: int = 1,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_state_grid(profile)
    charges = charge_grid(profile)
    charge_rows = _solve_charge_rows(cases, charges, workers=workers)
    policy_rows, meta = _policy_rows(charge_rows, charges)
    summary = _summary(charge_rows, policy_rows, meta, profile, charges)

    _write_csv(output_dir / "hard_capacity_reequilibrium_charge_records.csv", charge_rows)
    _write_csv(output_dir / "hard_capacity_reequilibrium_policy_records.csv", policy_rows)
    _write_csv(output_dir / "hard_capacity_reequilibrium_summary_rows.csv", _summary_rows(summary))
    (output_dir / "hard_capacity_reequilibrium_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
