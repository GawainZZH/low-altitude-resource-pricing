from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from .comprehensive_state_grid import build_state_grid, charge_grid
from .config import MarketCase
from .engine import MarketEngine
from .regulatory_benchmarks import (
    UTILIZATION_SLOPES,
    UTILIZATION_THRESHOLDS,
    _nearest_charge,
)
from .solver import solve_platform_prices


TOLERANCE = 1.0e-8


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict, key: str) -> float:
    return float(row[key])


def optional_float(row: dict, key: str) -> float:
    return float(row[key]) if key in row else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def group_by_case(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["case"], []).append(row)
    return grouped


def perturb_case(
    case: MarketCase,
    *,
    externality_factor: float = 1.0,
    reliability_power_factor: float = 1.0,
    price_sensitivity_factor: float = 1.0,
) -> MarketCase:
    return replace(
        case,
        congestion_cost=case.congestion_cost * externality_factor,
        reliability_power=case.reliability_power * reliability_power_factor,
        price_sensitivity=tuple(
            value * price_sensitivity_factor for value in case.price_sensitivity
        ),
    )


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


def solve_case_charge_rows(args: tuple[str, MarketCase, tuple[float, ...]]) -> list[dict]:
    scenario, case, charges = args
    engine = MarketEngine(case)
    rows = []
    previous_prices: tuple[float, ...] | None = None
    for charge in charges:
        starts = (previous_prices,) if previous_prices is not None else None
        solved = solve_platform_prices(engine, charge=charge, starts=starts)
        previous_prices = solved.prices
        demand_outcome = engine.solve_demand_reliability(solved.prices)
        rows.append(
            {
                "scenario": scenario,
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
                    solved.reliability, solved.demand
                ),
                "welfare": solved.welfare,
                "price_residual": solved.residual,
                "demand_residual": demand_outcome.residual,
            }
        )
    return rows


def solve_perceived_rows(
    scenario: str,
    cases: list[MarketCase],
    charges: tuple[float, ...],
    workers: int,
) -> list[dict]:
    tasks = [(scenario, case, charges) for case in cases]
    if workers <= 1:
        nested = [solve_case_charge_rows(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            nested = list(executor.map(solve_case_charge_rows, tasks))
    return sorted(
        [row for rows in nested for row in rows],
        key=lambda row: (row["scenario"], row["case"], row["charge"]),
    )


def select_sd_charges(perceived_by_case: dict[str, list[dict]]) -> dict[str, float]:
    return {
        case: as_float(max(rows, key=lambda row: as_float(row, "welfare")), "charge")
        for case, rows in perceived_by_case.items()
    }


def select_air_charges(perceived_by_case: dict[str, list[dict]]) -> dict[str, float]:
    charges = sorted(
        {as_float(row, "charge") for rows in perceived_by_case.values() for row in rows}
    )
    positive_charges = [charge for charge in charges if charge > TOLERANCE]
    if not positive_charges:
        return {case: 0.0 for case in perceived_by_case}
    first_positive = min(positive_charges)
    active_cases = []
    for case, rows in perceived_by_case.items():
        by_charge = {as_float(row, "charge"): row for row in rows}
        zero = by_charge[0.0]
        first = by_charge[first_positive]
        if as_float(first, "welfare") - as_float(zero, "welfare") > TOLERANCE:
            active_cases.append(case)
    if not active_cases:
        return {case: 0.0 for case in perceived_by_case}

    scores = []
    for charge in positive_charges:
        values = []
        for case in active_cases:
            rows = perceived_by_case[case]
            selected = min(rows, key=lambda row: abs(as_float(row, "charge") - charge))
            values.append(as_float(selected, "welfare"))
        scores.append((mean(values), -charge, charge))
    selected_positive_charge = max(scores)[2]
    return {
        case: selected_positive_charge if case in active_cases else 0.0
        for case in perceived_by_case
    }


def utilization_rule_row(
    group_rows: list[dict],
    threshold: float,
    slope: float,
) -> dict:
    charges = sorted({as_float(row, "charge") for row in group_rows})
    by_charge = {as_float(row, "charge"): row for row in group_rows}
    nc_utilization = as_float(by_charge[0.0], "utilization")
    target = slope * max(0.0, nc_utilization - threshold)
    selected_charge = _nearest_charge(charges, target)
    return by_charge[selected_charge]


def select_ut_charges(perceived_by_case: dict[str, list[dict]]) -> dict[str, float]:
    tuning_rows = []
    for threshold in UTILIZATION_THRESHOLDS:
        for slope in UTILIZATION_SLOPES:
            selected = [
                utilization_rule_row(rows, threshold, slope)
                for rows in perceived_by_case.values()
            ]
            tuning_rows.append(
                (
                    mean([as_float(row, "welfare") for row in selected]),
                    -mean([as_float(row, "charge") for row in selected]),
                    -slope,
                    threshold,
                    slope,
                )
            )
    _, _, _, threshold, slope = max(tuning_rows)
    return {
        case: as_float(utilization_rule_row(rows, threshold, slope), "charge")
        for case, rows in perceived_by_case.items()
    }


def nearest_true_row(true_rows: list[dict], selected_charge: float) -> dict:
    charges = sorted({as_float(row, "charge") for row in true_rows})
    target = _nearest_charge(charges, selected_charge)
    return min(true_rows, key=lambda row: abs(as_float(row, "charge") - target))


def true_sd_charges(true_by_case: dict[str, list[dict]]) -> dict[str, float]:
    return {
        case: as_float(max(rows, key=lambda row: as_float(row, "welfare")), "charge")
        for case, rows in true_by_case.items()
    }


def evaluate_selected_charges(
    *,
    true_by_case: dict[str, list[dict]],
    selected_charge_by_case: dict[str, float],
    true_sd_charge_by_case: dict[str, float],
    scenario: str,
    policy: str,
) -> list[dict]:
    records = []
    for case, rows in sorted(true_by_case.items()):
        nc = nearest_true_row(rows, 0.0)
        selected = nearest_true_row(rows, selected_charge_by_case[case])
        true_sd_charge = true_sd_charge_by_case[case]
        selected_charge = as_float(selected, "charge")
        true_sd_positive = true_sd_charge > TOLERANCE
        selected_positive = selected_charge > TOLERANCE
        records.append(
            {
                "scenario": scenario,
                "policy": policy,
                "case": case,
                "selected_charge": selected_charge,
                "true_sd_charge": true_sd_charge,
                "nc_welfare": as_float(nc, "welfare"),
                "realized_welfare": as_float(selected, "welfare"),
                "realized_delta_w": as_float(selected, "welfare")
                - as_float(nc, "welfare"),
                "true_sd_positive_charge": int(true_sd_positive),
                "selected_positive_charge": int(selected_positive),
                "false_positive_vs_true_sd": int(
                    selected_positive and not true_sd_positive
                ),
                "false_negative_vs_true_sd": int(
                    (not selected_positive) and true_sd_positive
                ),
                "nc_utilization": optional_float(nc, "utilization"),
                "realized_utilization": optional_float(selected, "utilization"),
                "externality_weight": optional_float(nc, "externality_weight"),
                "reliability_power": optional_float(nc, "reliability_power"),
                "demand_scale": optional_float(nc, "demand_scale"),
                "capacity_scale": optional_float(nc, "capacity_scale"),
            }
        )
    return records


def scenario_specs() -> list[dict]:
    specs = []
    for factor in (0.50, 0.75, 1.25, 1.50):
        specs.append(
            {
                "scenario": f"externality_x{factor:.2f}",
                "parameter": "externality_weight",
                "factor": factor,
                "externality_factor": factor,
                "reliability_power_factor": 1.0,
                "price_sensitivity_factor": 1.0,
            }
        )
    for factor in (0.75, 1.25):
        specs.append(
            {
                "scenario": f"reliability_curvature_x{factor:.2f}",
                "parameter": "reliability_curvature",
                "factor": factor,
                "externality_factor": 1.0,
                "reliability_power_factor": factor,
                "price_sensitivity_factor": 1.0,
            }
        )
    for factor in (0.75, 1.25):
        specs.append(
            {
                "scenario": f"price_sensitivity_x{factor:.2f}",
                "parameter": "price_sensitivity",
                "factor": factor,
                "externality_factor": 1.0,
                "reliability_power_factor": 1.0,
                "price_sensitivity_factor": factor,
            }
        )
    return specs


def build_perceived_cases(spec: dict) -> list[MarketCase]:
    cases = []
    for case in build_state_grid("full"):
        cases.append(
            perturb_case(
                case,
                externality_factor=spec["externality_factor"],
                reliability_power_factor=spec["reliability_power_factor"],
                price_sensitivity_factor=spec["price_sensitivity_factor"],
            )
        )
    return cases


def summarize_records(records: list[dict], true_sd_mean_delta: float) -> list[dict]:
    by_key: dict[tuple[str, str], list[dict]] = {}
    for row in records:
        by_key.setdefault((row["scenario"], row["policy"]), []).append(row)

    summaries = []
    for (scenario, policy), rows in sorted(by_key.items()):
        realized_deltas = [float(row["realized_delta_w"]) for row in rows]
        true_positive_rows = [
            row for row in rows if int(row["true_sd_positive_charge"]) == 1
        ]
        summaries.append(
            {
                "scenario": scenario,
                "policy": policy,
                "state_count": len(rows),
                "positive_charge_share": mean(
                    [float(row["selected_positive_charge"]) for row in rows]
                ),
                "false_positive_share": mean(
                    [float(row["false_positive_vs_true_sd"]) for row in rows]
                ),
                "false_negative_share": mean(
                    [float(row["false_negative_vs_true_sd"]) for row in rows]
                ),
                "realized_mean_delta_w": mean(realized_deltas),
                "realized_welfare_recovery_vs_true_sd": mean(realized_deltas)
                / true_sd_mean_delta
                if abs(true_sd_mean_delta) > TOLERANCE
                else 0.0,
                "worst_delta_w_true_positive_states": min(
                    [float(row["realized_delta_w"]) for row in true_positive_rows],
                    default=0.0,
                ),
            }
        )
    return summaries


def run_parameter_misspecification(
    output_dir: Path,
    workers: int = 1,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    true_charge_path = Path(
        "results/comprehensive_state_grid_v1_full/"
        "comprehensive_state_grid_charge_search_records.csv"
    )
    true_rows = read_csv(true_charge_path)
    true_by_case = group_by_case(true_rows)
    charges = charge_grid("full")
    true_sd_charge_by_case = true_sd_charges(true_by_case)
    true_sd_records = evaluate_selected_charges(
        true_by_case=true_by_case,
        selected_charge_by_case=true_sd_charge_by_case,
        true_sd_charge_by_case=true_sd_charge_by_case,
        scenario="correct_specification",
        policy="SD",
    )
    true_sd_mean_delta = mean(
        [float(row["realized_delta_w"]) for row in true_sd_records]
    )

    all_perceived_rows: list[dict] = []
    all_policy_rows: list[dict] = []
    for spec in scenario_specs():
        perceived_rows = solve_perceived_rows(
            spec["scenario"],
            build_perceived_cases(spec),
            charges,
            workers=workers,
        )
        for row in perceived_rows:
            row["misspecified_parameter"] = spec["parameter"]
            row["misspecification_factor"] = spec["factor"]
        all_perceived_rows.extend(perceived_rows)
        perceived_by_case = group_by_case(perceived_rows)
        selectors = {
            "SD": select_sd_charges(perceived_by_case),
            "AIR": select_air_charges(perceived_by_case),
            "UT": select_ut_charges(perceived_by_case),
        }
        for policy, selected in selectors.items():
            policy_rows = evaluate_selected_charges(
                true_by_case=true_by_case,
                selected_charge_by_case=selected,
                true_sd_charge_by_case=true_sd_charge_by_case,
                scenario=spec["scenario"],
                policy=policy,
            )
            for row in policy_rows:
                row["misspecified_parameter"] = spec["parameter"]
                row["misspecification_factor"] = spec["factor"]
            all_policy_rows.extend(policy_rows)
        write_csv(
            output_dir / f"{spec['scenario']}_perceived_charge_records.csv",
            perceived_rows,
        )

    summary_rows = summarize_records(all_policy_rows, true_sd_mean_delta)
    summary = {
        "state_count": len(true_by_case),
        "charge_count": len(charges),
        "scenario_count": len(scenario_specs()),
        "policy_count": 3,
        "true_sd_mean_delta_w": true_sd_mean_delta,
        "min_realized_mean_delta_w": min(
            row["realized_mean_delta_w"] for row in summary_rows
        ),
        "max_false_positive_share": max(
            row["false_positive_share"] for row in summary_rows
        ),
        "max_false_negative_share": max(
            row["false_negative_share"] for row in summary_rows
        ),
        "worst_summary_row": min(
            summary_rows,
            key=lambda row: row["realized_mean_delta_w"],
        ),
    }

    write_csv(
        output_dir / "parameter_misspecification_perceived_charge_records.csv",
        all_perceived_rows,
    )
    write_csv(
        output_dir / "parameter_misspecification_policy_records.csv",
        all_policy_rows,
    )
    write_csv(
        output_dir / "parameter_misspecification_summary_rows.csv",
        summary_rows,
    )
    (output_dir / "parameter_misspecification_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
