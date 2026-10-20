from __future__ import annotations

import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import brentq

from .comprehensive_state_grid import build_state_grid, charge_grid
from .config import DemandOutcome, MarketCase
from .parameter_misspecification import (
    as_float,
    group_by_case,
    mean,
    nearest_true_row,
    select_air_charges,
    select_ut_charges,
)
from .solver import solve_platform_prices


TOLERANCE = 1.0e-8


@dataclass(frozen=True)
class TasteMixtureSpec:
    name: str
    logit_scale_factor: float
    class_weights: tuple[float, ...]
    price_multipliers: tuple[float, ...]
    time_multipliers: tuple[float, ...]
    reliability_multipliers: tuple[float, ...]
    outside_affinity_shifts: tuple[float, ...]
    platform_affinity_shifts: tuple[float, ...]


def _as_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(tuple(values), dtype=float)


def _logsumexp(values: np.ndarray) -> float:
    peak = float(np.max(values))
    return peak + math.log(float(np.sum(np.exp(values - peak))))


def taste_mixture_specs() -> tuple[TasteMixtureSpec, ...]:
    return (
        TasteMixtureSpec(
            name="tm_correlated_vot_reliability",
            logit_scale_factor=1.00,
            class_weights=(0.40, 0.35, 0.25),
            price_multipliers=(1.15, 1.00, 0.85),
            time_multipliers=(0.90, 1.00, 1.20),
            reliability_multipliers=(0.80, 1.00, 1.35),
            outside_affinity_shifts=(0.05, 0.00, -0.05),
            platform_affinity_shifts=(-0.04, 0.00, 0.07),
        ),
        TasteMixtureSpec(
            name="tm_stronger_substitution",
            logit_scale_factor=0.75,
            class_weights=(0.35, 0.35, 0.30),
            price_multipliers=(1.20, 1.00, 0.82),
            time_multipliers=(0.88, 1.00, 1.25),
            reliability_multipliers=(0.78, 1.00, 1.42),
            outside_affinity_shifts=(0.08, 0.00, -0.08),
            platform_affinity_shifts=(-0.06, 0.00, 0.10),
        ),
    )


class TasteMixtureEngine:
    def __init__(self, case: MarketCase, spec: TasteMixtureSpec):
        self.case = case
        self.spec = spec
        self.platform_count = 3
        self.resource = _as_array(case.platform_resource)
        self.time = _as_array(case.service_time)
        self.base_loss = _as_array(case.base_loss)
        self.exposure = _as_array(case.exposure)
        self.capacity = case.base_capacity * case.capacity_scale
        self.cost = _as_array(case.base_cost) + np.array(
            (0.0, case.cost_gap, 0.5 * case.cost_gap)
        )
        self.logit_scale = case.logit_scale * spec.logit_scale_factor

        demand_base = []
        price_sensitivity = []
        time_sensitivity = []
        reliability_sensitivity = []
        outside_affinity = []
        platform_affinity = []
        base_platform_affinity = np.asarray(case.platform_affinity, dtype=float)

        for original_group in range(3):
            for class_index, class_weight in enumerate(spec.class_weights):
                demand_base.append(
                    case.base_demand[original_group]
                    * case.demand_scale
                    * class_weight
                )
                price_sensitivity.append(
                    case.price_sensitivity[original_group]
                    * spec.price_multipliers[class_index]
                )
                time_sensitivity.append(
                    case.time_sensitivity[original_group]
                    * spec.time_multipliers[class_index]
                )
                reliability_sensitivity.append(
                    case.reliability_sensitivity[original_group]
                    * spec.reliability_multipliers[class_index]
                )
                outside_affinity.append(
                    case.outside_affinity[original_group]
                    + spec.outside_affinity_shifts[class_index]
                )
                row = base_platform_affinity[original_group].copy()
                row[original_group] += case.differentiation
                row = row + spec.platform_affinity_shifts[class_index]
                platform_affinity.append(row)

        self.group_count = len(demand_base)
        self.demand_base = np.asarray(demand_base, dtype=float)
        self.price_sensitivity = np.asarray(price_sensitivity, dtype=float)
        self.time_sensitivity = np.asarray(time_sensitivity, dtype=float)
        self.reliability_sensitivity = np.asarray(
            reliability_sensitivity,
            dtype=float,
        )
        self.outside_affinity = np.asarray(outside_affinity, dtype=float)
        self.platform_affinity = np.asarray(platform_affinity, dtype=float)

    def reliability_function(self, utilization: float) -> float:
        u = min(max(float(utilization), 0.0), 1.45)
        overload = max(u - 0.88, 0.0)
        return 0.55 * (u ** self.case.reliability_power) + 1.10 * overload * overload

    def _demand_at_utilization(
        self,
        prices: np.ndarray,
        utilization: float,
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        reliability = self.base_loss + self.exposure * self.reliability_function(
            utilization
        )
        prob = np.zeros((self.group_count, self.platform_count))
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
            values = np.concatenate(([outside_utility], air_utility)) / self.logit_scale
            denominator = _logsumexp(values)
            all_prob = np.exp(values - denominator)
            prob[group, :] = all_prob[1:]

        demand = self.demand_base @ prob
        next_utilization = float(np.dot(self.resource, demand) / self.capacity)
        return demand, min(max(next_utilization, 0.0), 1.70), reliability, prob

    def _iterated_demand_reliability(self, prices: np.ndarray) -> DemandOutcome:
        utilization = 0.70
        demand = np.zeros(self.platform_count)
        prob = np.zeros((self.group_count, self.platform_count))
        reliability = self.base_loss.copy()
        last_gap = math.inf
        iteration = 0

        for iteration in range(1, 81):
            new_demand, new_utilization, reliability, prob = self._demand_at_utilization(
                prices,
                utilization,
            )
            last_gap = abs(new_utilization - utilization)
            demand = new_demand
            utilization = 0.35 * utilization + 0.65 * new_utilization
            if last_gap <= 1.0e-8:
                break

        reliability = self.base_loss + self.exposure * self.reliability_function(
            utilization
        )
        return DemandOutcome(
            demand=tuple(float(x) for x in demand),
            utilization=float(utilization),
            reliability=tuple(float(x) for x in reliability),
            choice_prob=tuple(tuple(float(x) for x in row) for row in prob),
            residual=float(last_gap),
            iterations=iteration,
        )

    def solve_demand_reliability(self, prices: Iterable[float]) -> DemandOutcome:
        price_vector = _as_array(prices)
        iterated = self._iterated_demand_reliability(price_vector)
        if iterated.residual <= 1.0e-7:
            return iterated

        def gap(utilization: float) -> float:
            _, next_utilization, _, _ = self._demand_at_utilization(
                price_vector,
                utilization,
            )
            return next_utilization - utilization

        values = np.linspace(0.0, 1.70, 69)
        gaps = [gap(float(value)) for value in values]
        bracket = None
        root = None
        for lower, upper, lower_value, upper_value in zip(
            values[:-1],
            values[1:],
            gaps[:-1],
            gaps[1:],
        ):
            if abs(lower_value) <= 1.0e-10:
                root = float(lower)
                break
            if lower_value * upper_value <= 0.0:
                bracket = (float(lower), float(upper))
                root = brentq(
                    gap,
                    bracket[0],
                    bracket[1],
                    xtol=1.0e-10,
                    rtol=1.0e-10,
                    maxiter=80,
                )
                break
        if root is None:
            return iterated

        demand, next_utilization, reliability, prob = self._demand_at_utilization(
            price_vector,
            root,
        )
        return DemandOutcome(
            demand=tuple(float(x) for x in demand),
            utilization=float(root),
            reliability=tuple(float(x) for x in reliability),
            choice_prob=tuple(tuple(float(x) for x in row) for row in prob),
            residual=abs(float(next_utilization - root)),
            iterations=0 if bracket is None else 1,
        )

    def platform_profit(
        self,
        prices: Iterable[float],
        platform: int,
        charge: float,
    ) -> float:
        price_vector = _as_array(prices)
        outcome = self.solve_demand_reliability(price_vector)
        demand = outcome.demand[platform]
        variable_cost = self.cost[platform] * demand
        resource_payment = charge * self.resource[platform] * demand
        return float(price_vector[platform] * demand - variable_cost - resource_payment)

    def consumer_surplus(
        self,
        prices: Iterable[float],
        reliability: Iterable[float],
    ) -> float:
        price_vector = _as_array(prices)
        loss = _as_array(reliability)
        total = 0.0
        for group in range(self.group_count):
            air_utility = (
                self.platform_affinity[group]
                - self.price_sensitivity[group] * price_vector
                - self.time_sensitivity[group] * self.time
                - self.reliability_sensitivity[group] * loss
            )
            outside_utility = (
                self.outside_affinity[group]
                - self.price_sensitivity[group] * self.case.outside_price
                - self.time_sensitivity[group] * self.case.outside_time
            )
            values = np.concatenate(([outside_utility], air_utility)) / self.logit_scale
            total += (
                self.demand_base[group]
                * (self.logit_scale / self.price_sensitivity[group])
                * _logsumexp(values)
            )
        return float(total)

    def operating_cost(self, demand: Iterable[float]) -> float:
        return float(np.dot(self.cost, _as_array(demand)))

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


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def average_fare(prices: tuple[float, ...], demand: tuple[float, ...]) -> float:
    total = sum(demand)
    if total <= 0.0:
        return 0.0
    return sum(price * quantity for price, quantity in zip(prices, demand)) / total


def average_reliability(loss: tuple[float, ...], demand: tuple[float, ...]) -> float:
    total = sum(demand)
    if total <= 0.0:
        return 0.0
    return sum(value * quantity for value, quantity in zip(loss, demand)) / total


def solve_case_charge_rows(
    args: tuple[TasteMixtureSpec, MarketCase, tuple[float, ...]],
) -> list[dict]:
    spec, case, charges = args
    engine = TasteMixtureEngine(case, spec)
    rows = []
    previous_prices: tuple[float, ...] | None = None
    for charge in charges:
        starts = (previous_prices,) if previous_prices is not None else None
        solved = solve_platform_prices(engine, charge=charge, starts=starts)
        previous_prices = solved.prices
        demand_outcome = engine.solve_demand_reliability(solved.prices)
        rows.append(
            {
                "scenario": spec.name,
                "case": case.name,
                "demand_scale": case.demand_scale,
                "capacity_scale": case.capacity_scale,
                "reliability_power": case.reliability_power,
                "externality_weight": case.congestion_cost,
                "charge": float(charge),
                "price_1": solved.prices[0],
                "price_2": solved.prices[1],
                "price_3": solved.prices[2],
                "demand_1": solved.demand[0],
                "demand_2": solved.demand[1],
                "demand_3": solved.demand[2],
                "total_demand": sum(solved.demand),
                "utilization": solved.utilization,
                "average_fare": average_fare(solved.prices, solved.demand),
                "average_reliability": average_reliability(
                    solved.reliability,
                    solved.demand,
                ),
                "welfare": solved.welfare,
                "price_residual": solved.residual,
                "demand_residual": demand_outcome.residual,
            }
        )
    return rows


def solve_charge_rows(
    specs: tuple[TasteMixtureSpec, ...],
    cases: list[MarketCase],
    charges: tuple[float, ...],
    workers: int,
) -> list[dict]:
    tasks = [(spec, case, charges) for spec in specs for case in cases]
    if workers <= 1:
        nested = [solve_case_charge_rows(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            nested = list(executor.map(solve_case_charge_rows, tasks))
    return sorted(
        [row for rows in nested for row in rows],
        key=lambda row: (row["scenario"], row["case"], row["charge"]),
    )


def rows_by_scenario_and_case(rows: list[dict]) -> dict[str, dict[str, list[dict]]]:
    result: dict[str, dict[str, list[dict]]] = {}
    for row in rows:
        result.setdefault(row["scenario"], {}).setdefault(row["case"], []).append(row)
    return result


def true_sd_charges(true_by_case: dict[str, list[dict]]) -> dict[str, float]:
    return {
        case: as_float(max(rows, key=lambda row: as_float(row, "welfare")), "charge")
        for case, rows in true_by_case.items()
    }


def zero_charge_rows(true_by_case: dict[str, list[dict]]) -> dict[str, dict]:
    return {
        case: min(rows, key=lambda row: abs(as_float(row, "charge")))
        for case, rows in true_by_case.items()
    }


def baseline_air_charges(profile: str) -> dict[str, float]:
    if profile != "full":
        return {}
    path = Path("results/activation_index_rule_v1/activation_index_rule_state_records.csv")
    if not path.exists():
        return {}
    return {row["case"]: as_float(row, "charge") for row in read_csv(path)}


def baseline_sd_charges(profile: str) -> dict[str, float]:
    if profile != "full":
        return {}
    path = Path(
        "results/comprehensive_state_grid_v1_full/"
        "comprehensive_state_grid_policy_records.csv"
    )
    if not path.exists():
        return {}
    return {
        row["case"]: as_float(row, "charge")
        for row in read_csv(path)
        if row["policy"] == "SD"
    }


def evaluate_policy(
    *,
    scenario: str,
    policy: str,
    true_by_case: dict[str, list[dict]],
    selected_charge_by_case: dict[str, float],
    alternative_sd_charge_by_case: dict[str, float],
    baseline_sd_charge_by_case: dict[str, float],
) -> list[dict]:
    records = []
    for case, rows in sorted(true_by_case.items()):
        nc = nearest_true_row(rows, 0.0)
        selected = nearest_true_row(rows, selected_charge_by_case[case])
        alternative_sd_charge = alternative_sd_charge_by_case[case]
        selected_charge = as_float(selected, "charge")
        alternative_positive = alternative_sd_charge > TOLERANCE
        selected_positive = selected_charge > TOLERANCE
        baseline_sd_charge = baseline_sd_charge_by_case.get(case, 0.0)
        records.append(
            {
                "scenario": scenario,
                "policy": policy,
                "case": case,
                "selected_charge": selected_charge,
                "alternative_sd_charge": alternative_sd_charge,
                "baseline_sd_charge": baseline_sd_charge,
                "nc_welfare": as_float(nc, "welfare"),
                "selected_welfare": as_float(selected, "welfare"),
                "delta_w_vs_nc": as_float(selected, "welfare") - as_float(nc, "welfare"),
                "alternative_sd_positive": int(alternative_positive),
                "selected_positive": int(selected_positive),
                "baseline_sd_positive": int(baseline_sd_charge > TOLERANCE),
                "false_positive_vs_alternative_sd": int(
                    selected_positive and not alternative_positive
                ),
                "false_negative_vs_alternative_sd": int(
                    (not selected_positive) and alternative_positive
                ),
                "nc_utilization": as_float(nc, "utilization"),
                "selected_utilization": as_float(selected, "utilization"),
                "externality_weight": as_float(nc, "externality_weight"),
                "reliability_power": as_float(nc, "reliability_power"),
                "demand_scale": as_float(nc, "demand_scale"),
                "capacity_scale": as_float(nc, "capacity_scale"),
            }
        )
    return records


def select_zero_charges(true_by_case: dict[str, list[dict]]) -> dict[str, float]:
    return {case: 0.0 for case in true_by_case}


def select_policy_charges(
    true_by_case: dict[str, list[dict]],
    profile: str,
) -> dict[str, dict[str, float]]:
    alternative_sd = true_sd_charges(true_by_case)
    selectors = {
        "NC": select_zero_charges(true_by_case),
        "SD_alt": alternative_sd,
        "AIR_local": select_air_charges(true_by_case),
        "UT_best": select_ut_charges(true_by_case),
    }
    air_base = baseline_air_charges(profile)
    if air_base:
        selectors["AIR_base"] = {
            case: air_base.get(case, 0.0) for case in true_by_case
        }
    return selectors


def summarize_policy_records(policy_records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    alternative_sd_delta_by_scenario: dict[str, float] = {}
    for row in policy_records:
        key = (row["scenario"], row["policy"])
        grouped.setdefault(key, []).append(row)
    for (scenario, policy), rows in grouped.items():
        if policy == "SD_alt":
            alternative_sd_delta_by_scenario[scenario] = mean(
                [as_float(row, "delta_w_vs_nc") for row in rows]
            )

    summaries = []
    for (scenario, policy), rows in sorted(grouped.items()):
        alternative_sd_delta = alternative_sd_delta_by_scenario.get(scenario, 0.0)
        summaries.append(
            {
                "scenario": scenario,
                "policy": policy,
                "state_count": len(rows),
                "positive_charge_share": mean(
                    [float(row["selected_positive"]) for row in rows]
                ),
                "mean_selected_charge": mean(
                    [as_float(row, "selected_charge") for row in rows]
                ),
                "mean_delta_w_vs_nc": mean(
                    [as_float(row, "delta_w_vs_nc") for row in rows]
                ),
                "welfare_recovery_vs_alternative_sd": (
                    mean([as_float(row, "delta_w_vs_nc") for row in rows])
                    / alternative_sd_delta
                    if abs(alternative_sd_delta) > TOLERANCE
                    else 0.0
                ),
                "false_positive_share_vs_alternative_sd": mean(
                    [
                        float(row["false_positive_vs_alternative_sd"])
                        for row in rows
                    ]
                ),
                "false_negative_share_vs_alternative_sd": mean(
                    [
                        float(row["false_negative_vs_alternative_sd"])
                        for row in rows
                    ]
                ),
            }
        )
    return summaries


def summarize_boundary_overlap(policy_records: list[dict]) -> list[dict]:
    records = []
    by_scenario: dict[str, list[dict]] = {}
    for row in policy_records:
        if row["policy"] == "SD_alt":
            by_scenario.setdefault(row["scenario"], []).append(row)
    for scenario, rows in sorted(by_scenario.items()):
        alternative_positive = {
            row["case"] for row in rows if int(row["alternative_sd_positive"]) == 1
        }
        baseline_positive = {
            row["case"] for row in rows if int(row["baseline_sd_positive"]) == 1
        }
        intersection = alternative_positive & baseline_positive
        union = alternative_positive | baseline_positive
        records.append(
            {
                "scenario": scenario,
                "state_count": len(rows),
                "alternative_positive_count": len(alternative_positive),
                "alternative_positive_share": len(alternative_positive) / len(rows),
                "baseline_positive_count": len(baseline_positive),
                "baseline_positive_share": len(baseline_positive) / len(rows),
                "overlap_count": len(intersection),
                "overlap_share_of_alternative_positive": (
                    len(intersection) / len(alternative_positive)
                    if alternative_positive
                    else 0.0
                ),
                "jaccard_overlap": len(intersection) / len(union) if union else 1.0,
            }
        )
    return records


def summarize_run(
    *,
    profile: str,
    specs: tuple[TasteMixtureSpec, ...],
    charge_rows: list[dict],
    policy_records: list[dict],
    summary_rows: list[dict],
    overlap_rows: list[dict],
    charges: tuple[float, ...],
) -> dict:
    sd_rows = [row for row in summary_rows if row["policy"] == "SD_alt"]
    non_nc_rows = [row for row in summary_rows if row["policy"] != "NC"]
    cases_by_scenario: dict[str, set[str]] = {}
    for row in charge_rows:
        cases_by_scenario.setdefault(row["scenario"], set()).add(row["case"])
    return {
        "profile": profile,
        "scenario_count": len(specs),
        "state_count_per_scenario": max(
            (len(cases) for cases in cases_by_scenario.values()),
            default=0,
        ),
        "charge_count": len(charges),
        "charge_record_count": len(charge_rows),
        "policy_record_count": len(policy_records),
        "sd_positive_share_by_scenario": {
            row["scenario"]: row["positive_charge_share"] for row in sd_rows
        },
        "minimum_non_nc_mean_delta_w": min(
            [row["mean_delta_w_vs_nc"] for row in non_nc_rows],
            default=0.0,
        ),
        "air_base_rows": [
            row for row in summary_rows if row["policy"] == "AIR_base"
        ],
        "boundary_overlap": overlap_rows,
        "max_price_residual": max(as_float(row, "price_residual") for row in charge_rows),
        "max_demand_residual": max(
            as_float(row, "demand_residual") for row in charge_rows
        ),
    }


def write_result_summary(
    output_dir: Path,
    summary: dict,
    summary_rows: list[dict],
    overlap_rows: list[dict],
) -> None:
    lines = [
        "# Out-of-Model Demand Robustness",
        "",
        "This experiment replaces the baseline three-group logit demand layer with deterministic taste-mixture demand classes. It does not modify the capacity, cost, reliability, or charge sets.",
        "",
        "## Run Summary",
        "",
        f"- Profile: {summary['profile']}",
        f"- Scenarios: {summary['scenario_count']}",
        f"- Charge records: {summary['charge_record_count']}",
        f"- Policy records: {summary['policy_record_count']}",
        f"- Maximum platform first-order residual: {summary['max_price_residual']:.3e}",
        f"- Maximum demand-closure residual: {summary['max_demand_residual']:.3e}",
        "",
        "## Policy Summary",
        "",
        "| Scenario | Policy | Positive charge share | Mean charge | Mean delta W | Recovery vs alternative SD | FP share | FN share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {scenario} | {policy} | {positive_charge_share:.4f} | "
            "{mean_selected_charge:.4f} | {mean_delta_w_vs_nc:.4f} | "
            "{welfare_recovery_vs_alternative_sd:.4f} | "
            "{false_positive_share_vs_alternative_sd:.4f} | "
            "{false_negative_share_vs_alternative_sd:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Boundary Overlap",
            "",
            "| Scenario | Alternative positive share | Baseline positive share | Overlap share | Jaccard overlap |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in overlap_rows:
        lines.append(
            "| {scenario} | {alternative_positive_share:.4f} | "
            "{baseline_positive_share:.4f} | "
            "{overlap_share_of_alternative_positive:.4f} | "
            "{jaccard_overlap:.4f} |".format(**row)
        )
    output_dir.joinpath("RESULT_SUMMARY.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_out_of_model_demand_robustness(
    output_dir: Path,
    profile: str = "full",
    workers: int = 1,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = taste_mixture_specs()
    cases = build_state_grid(profile)
    charges = charge_grid(profile)
    charge_rows = solve_charge_rows(specs, cases, charges, workers)
    scenario_case_rows = rows_by_scenario_and_case(charge_rows)
    baseline_sd_charge_by_case = baseline_sd_charges(profile)

    policy_records: list[dict] = []
    for scenario, true_by_case in sorted(scenario_case_rows.items()):
        alternative_sd_charge_by_case = true_sd_charges(true_by_case)
        selectors = select_policy_charges(true_by_case, profile)
        for policy, selected_charge_by_case in selectors.items():
            policy_records.extend(
                evaluate_policy(
                    scenario=scenario,
                    policy=policy,
                    true_by_case=true_by_case,
                    selected_charge_by_case=selected_charge_by_case,
                    alternative_sd_charge_by_case=alternative_sd_charge_by_case,
                    baseline_sd_charge_by_case=baseline_sd_charge_by_case,
                )
            )

    summary_rows = summarize_policy_records(policy_records)
    overlap_rows = summarize_boundary_overlap(policy_records)
    summary = summarize_run(
        profile=profile,
        specs=specs,
        charge_rows=charge_rows,
        policy_records=policy_records,
        summary_rows=summary_rows,
        overlap_rows=overlap_rows,
        charges=charges,
    )

    write_csv(output_dir / "taste_mixture_charge_records.csv", charge_rows)
    write_csv(output_dir / "taste_mixture_policy_records.csv", policy_records)
    write_csv(output_dir / "taste_mixture_summary_rows.csv", summary_rows)
    write_csv(output_dir / "taste_mixture_boundary_overlap_rows.csv", overlap_rows)
    output_dir.joinpath("taste_mixture_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_result_summary(output_dir, summary, summary_rows, overlap_rows)
    return summary
