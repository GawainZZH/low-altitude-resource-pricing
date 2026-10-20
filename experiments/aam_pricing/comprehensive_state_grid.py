from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .config import MarketCase
from .engine import MarketEngine
from .solver import solve_platform_prices


FULL_CHARGE_GRID = tuple(float(value) for value in range(0, 22, 2))
SMOKE_CHARGE_GRID = (0.0, 4.0, 8.0, 12.0, 16.0)
SMOOTHING_WEIGHT = 1.0


def _grid_values(profile: str) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    if profile == "smoke":
        return (
            (0.85, 1.55),
            (0.70, 1.12),
            (1.70, 2.80),
            (90.0, 4200.0),
        )
    if profile == "full":
        return (
            (0.85, 1.00, 1.15, 1.35, 1.55, 1.70),
            (0.70, 0.80, 0.90, 1.00, 1.12),
            (1.70, 2.20, 2.80),
            (90.0, 1200.0, 2800.0, 4200.0, 6500.0),
        )
    raise ValueError(f"Unknown grid profile: {profile}")


def charge_grid(profile: str) -> tuple[float, ...]:
    return SMOKE_CHARGE_GRID if profile == "smoke" else FULL_CHARGE_GRID


def build_state_grid(profile: str = "full") -> list[MarketCase]:
    demand_values, capacity_values, power_values, externality_values = _grid_values(profile)
    cases = []
    for demand_scale in demand_values:
        for capacity_scale in capacity_values:
            for reliability_power in power_values:
                for externality_weight in externality_values:
                    cases.append(
                        MarketCase(
                            name=(
                                f"grid_d{int(round(100 * demand_scale)):03d}_"
                                f"c{int(round(100 * capacity_scale)):03d}_"
                                f"p{int(round(100 * reliability_power)):03d}_"
                                f"e{int(round(externality_weight)):04d}"
                            ),
                            demand_scale=demand_scale,
                            capacity_scale=capacity_scale,
                            reliability_power=reliability_power,
                            differentiation=0.24,
                            cost_gap=0.35,
                            reliability_sensitivity=(1.35, 1.65, 1.95),
                            outside_affinity=(1.30, 1.20, 1.10),
                            base_loss=(0.08, 0.09, 0.08),
                            exposure=(0.86, 1.02, 0.94),
                            congestion_cost=externality_weight,
                            price_upper=44.0,
                        )
                    )
    return cases


def _average_fare(prices: tuple[float, ...], demand: tuple[float, ...]) -> float:
    total = sum(demand)
    if total <= 0.0:
        return 0.0
    return sum(p * y for p, y in zip(prices, demand)) / total


def _average_reliability(loss: tuple[float, ...], demand: tuple[float, ...]) -> float:
    total = sum(demand)
    if total <= 0.0:
        return 0.0
    return sum(v * y for v, y in zip(loss, demand)) / total


def _pressure_band(utilization: float) -> str:
    if utilization < 0.80:
        return "lt_080"
    if utilization < 1.00:
        return "080_100"
    if utilization < 1.20:
        return "100_120"
    return "ge_120"


def _solve_case_charge_rows(args: tuple[MarketCase, tuple[float, ...]]) -> list[dict]:
    case, charges = args
    engine = MarketEngine(case)
    rows = []
    previous_prices: tuple[float, ...] | None = None
    for charge in charges:
        starts = (previous_prices,) if previous_prices is not None else ((22.0, 22.0, 22.0),)
        solved = solve_platform_prices(engine, charge=charge, starts=starts)
        previous_prices = solved.prices
        demand_outcome = engine.solve_demand_reliability(solved.prices)
        row = {
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
            "average_fare": _average_fare(solved.prices, solved.demand),
            "average_reliability": _average_reliability(solved.reliability, solved.demand),
            "welfare": solved.welfare,
            "price_residual": solved.residual,
            "demand_residual": demand_outcome.residual,
        }
        rows.append(row)
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
        welfare_by_charge[float(row["charge"])].append(row["welfare"])
    mean_by_charge = {
        charge: sum(values) / len(values)
        for charge, values in welfare_by_charge.items()
    }
    return max(mean_by_charge, key=lambda charge: mean_by_charge[charge])


def _policy_rows(charge_rows: list[dict], static_charge: float) -> list[dict]:
    selected = []
    for rows in _group_by_case(charge_rows).values():
        nc = min(rows, key=lambda row: abs(float(row["charge"]))).copy()
        sc = min(rows, key=lambda row: abs(float(row["charge"]) - static_charge)).copy()
        sd = max(rows, key=lambda row: row["welfare"]).copy()
        nc["policy"] = "NC"
        sc["policy"] = "SC"
        sd["policy"] = "SD"
        nc_welfare = nc["welfare"]
        for row in (nc, sc, sd):
            row["welfare_gain_percent_vs_NC"] = 100.0 * (row["welfare"] - nc_welfare) / max(abs(nc_welfare), 1.0)
            row["utilization_relief_vs_NC"] = nc["utilization"] - row["utilization"]
            row["reliability_relief_vs_NC"] = nc["average_reliability"] - row["average_reliability"]
            row["nc_utilization"] = nc["utilization"]
            row["nc_average_reliability"] = nc["average_reliability"]
            row["pressure_band"] = _pressure_band(nc["utilization"])
        selected.extend([nc, sc, sd])
    return sorted(selected, key=lambda row: (row["case"], row["policy"]))


def _boundary_records(policy_rows: list[dict]) -> list[dict]:
    sd_rows = [row for row in policy_rows if row["policy"] == "SD"]
    records = []
    for band in ("lt_080", "080_100", "100_120", "ge_120"):
        rows = [row for row in sd_rows if row["pressure_band"] == band]
        if not rows:
            continue
        records.append(
            {
                "pressure_band": band,
                "state_count": len(rows),
                "sd_nonzero_charge_share": sum(1 for row in rows if float(row["charge"]) > 0.0) / len(rows),
                "sd_positive_gain_share": sum(
                    1 for row in rows if row["welfare_gain_percent_vs_NC"] > 1.0e-8
                )
                / len(rows),
                "mean_sd_gain_percent_vs_NC": sum(row["welfare_gain_percent_vs_NC"] for row in rows) / len(rows),
                "mean_nc_utilization": sum(row["nc_utilization"] for row in rows) / len(rows),
                "mean_nc_reliability": sum(row["nc_average_reliability"] for row in rows) / len(rows),
            }
        )
    return records


def _rows_by_case_and_charge(charge_rows: list[dict]) -> dict[tuple[str, float], dict]:
    return {
        (row["case"], float(row["charge"])): row
        for row in charge_rows
    }


def _case_pressure_scores(policy_rows: list[dict]) -> list[tuple[float, str]]:
    scores = []
    for row in policy_rows:
        if row["policy"] == "NC":
            score = row["nc_utilization"] * max(row["nc_average_reliability"], 1.0e-8) * row["externality_weight"]
            scores.append((score, row["case"]))
    return sorted(scores)


def _dynamic_sequences(policy_rows: list[dict], profile: str) -> list[list[str]]:
    sd_rows = [row for row in policy_rows if row["policy"] == "SD"]
    positive = [row["case"] for row in sd_rows if float(row["charge"]) > 0.0]
    zero_rows = sorted(
        [row for row in sd_rows if float(row["charge"]) == 0.0],
        key=lambda row: row["nc_utilization"],
    )
    if not zero_rows:
        return []
    count = 2 if profile == "smoke" else 24
    low = [row["case"] for row in zero_rows[: max(6, len(zero_rows) // 5)]]
    middle_start = max(0, len(zero_rows) // 2 - max(3, len(zero_rows) // 20))
    middle = [row["case"] for row in zero_rows[middle_start : middle_start + max(6, len(zero_rows) // 5)]]
    high = positive or [row["case"] for row in zero_rows[-max(6, len(zero_rows) // 5) :]]
    sequences = []
    for index in range(count):
        sequence = []
        for step in range(12):
            if step in (0, 5, 9):
                pool = low
            elif step in (1, 4, 8, 11):
                pool = middle
            else:
                pool = high
            sequence.append(pool[(index + step) % len(pool)])
        sequences.append(sequence)
    return sequences


def _select_smoothed_path(period_rows: list[list[dict]]) -> list[dict]:
    previous: dict[tuple[int, float], float | None] = {}
    best_value: dict[float, float] = {}
    for row in period_rows[0]:
        charge = float(row["charge"])
        best_value[charge] = row["welfare"] - SMOOTHING_WEIGHT * charge * charge
        previous[(0, charge)] = None

    for period, rows in enumerate(period_rows[1:], start=1):
        next_value = {}
        for row in rows:
            charge = float(row["charge"])
            best_prev_charge = None
            best_score = None
            for prev_charge, prev_score in best_value.items():
                score = prev_score + row["welfare"] - SMOOTHING_WEIGHT * (charge - prev_charge) ** 2
                if best_score is None or score > best_score:
                    best_score = score
                    best_prev_charge = prev_charge
            next_value[charge] = float(best_score)
            previous[(period, charge)] = best_prev_charge
        best_value = next_value

    current_charge: float | None = max(best_value, key=lambda charge: best_value[charge])
    path: dict[int, float] = {}
    for period in reversed(range(len(period_rows))):
        if current_charge is None:
            break
        path[period] = current_charge
        current_charge = previous[(period, current_charge)]

    return [
        min(rows, key=lambda row: abs(float(row["charge"]) - path[period])).copy()
        for period, rows in enumerate(period_rows)
    ]


def _mean_charge_change(rows: list[dict]) -> float:
    previous = 0.0
    changes = []
    for row in rows:
        charge = float(row["charge"])
        changes.append(abs(charge - previous))
        previous = charge
    return sum(changes) / len(changes)


def _dynamic_records(
    charge_rows: list[dict],
    policy_rows: list[dict],
    charges: tuple[float, ...],
    profile: str,
) -> list[dict]:
    lookup = _rows_by_case_and_charge(charge_rows)
    records = []
    for sequence_id, sequence in enumerate(_dynamic_sequences(policy_rows, profile), start=1):
        period_rows = [
            [lookup[(case, charge)].copy() for charge in charges]
            for case in sequence
        ]
        nc_rows = [min(rows, key=lambda row: abs(float(row["charge"]))).copy() for rows in period_rows]
        sd_rows = [max(rows, key=lambda row: row["welfare"]).copy() for rows in period_rows]
        sm_rows = _select_smoothed_path(period_rows)

        totals = {
            "NC": sum(row["welfare"] for row in nc_rows),
            "SD": sum(row["welfare"] for row in sd_rows),
            "SM": sum(row["welfare"] for row in sm_rows),
        }
        nc_total = totals["NC"]
        sd_move = _mean_charge_change(sd_rows)
        sm_move = _mean_charge_change(sm_rows)
        high_share = sum(1 for rows in period_rows if max(rows, key=lambda row: row["welfare"])["charge"] > 0.0) / len(period_rows)
        records.append(
            {
                "sequence_id": sequence_id,
                "period_count": len(sequence),
                "positive_charge_period_share": high_share,
                "sd_total_welfare": totals["SD"],
                "sm_total_welfare": totals["SM"],
                "sd_total_welfare_gain_percent_vs_NC": 100.0 * (totals["SD"] - nc_total) / max(abs(nc_total), 1.0),
                "sm_total_welfare_gain_percent_vs_NC": 100.0 * (totals["SM"] - nc_total) / max(abs(nc_total), 1.0),
                "sm_welfare_retention_ratio_vs_SD": (
                    (totals["SM"] - nc_total) / max(totals["SD"] - nc_total, 1.0e-8)
                    if totals["SD"] > nc_total
                    else 0.0
                ),
                "sd_mean_absolute_charge_change": sd_move,
                "sm_mean_absolute_charge_change": sm_move,
                "movement_reduction_percent": 100.0 * (sd_move - sm_move) / max(sd_move, 1.0e-8),
            }
        )
    return records


def _summary(
    profile: str,
    charges: tuple[float, ...],
    policy_rows: list[dict],
    charge_rows: list[dict],
    boundary_rows: list[dict],
    dynamic_rows: list[dict],
    static_charge: float,
) -> dict:
    by_policy: dict[str, list[dict]] = {}
    for row in policy_rows:
        by_policy.setdefault(row["policy"], []).append(row)

    def mean(policy: str, key: str) -> float:
        rows = by_policy[policy]
        return sum(row[key] for row in rows) / len(rows)

    sd_rows = by_policy["SD"]
    sm_positive = [
        row for row in dynamic_rows
        if row["sm_total_welfare_gain_percent_vs_NC"] > 0.0
    ]
    sm_reduces = [
        row for row in dynamic_rows
        if row["sm_mean_absolute_charge_change"] < row["sd_mean_absolute_charge_change"]
    ]
    return {
        "profile": profile,
        "state_count": len(sd_rows),
        "charge_count": len(charges),
        "policies": sorted(by_policy),
        "selected_static_charge": static_charge,
        "sd_nonzero_charge_state_share": sum(1 for row in sd_rows if float(row["charge"]) > 0.0) / len(sd_rows),
        "sd_positive_gain_state_share": sum(1 for row in sd_rows if row["welfare_gain_percent_vs_NC"] > 1.0e-8) / len(sd_rows),
        "sc_positive_gain_state_share": sum(
            1 for row in by_policy["SC"] if row["welfare_gain_percent_vs_NC"] > 1.0e-8
        )
        / len(by_policy["SC"]),
        "sd_mean_gain_percent_vs_NC": mean("SD", "welfare_gain_percent_vs_NC"),
        "sc_mean_gain_percent_vs_NC": mean("SC", "welfare_gain_percent_vs_NC"),
        "sd_mean_utilization_relief_vs_NC": mean("SD", "utilization_relief_vs_NC"),
        "sd_mean_reliability_relief_vs_NC": mean("SD", "reliability_relief_vs_NC"),
        "boundary_group_count": len(boundary_rows),
        "sequence_count": len(dynamic_rows),
        "sm_positive_gain_share": len(sm_positive) / len(dynamic_rows) if dynamic_rows else 0.0,
        "sm_reduces_movement_share": len(sm_reduces) / len(dynamic_rows) if dynamic_rows else 0.0,
        "sm_mean_welfare_retention_ratio_vs_SD": (
            sum(row["sm_welfare_retention_ratio_vs_SD"] for row in dynamic_rows) / len(dynamic_rows)
            if dynamic_rows
            else 0.0
        ),
        "mean_movement_reduction_percent": (
            sum(row["movement_reduction_percent"] for row in dynamic_rows) / len(dynamic_rows)
            if dynamic_rows
            else 0.0
        ),
        "max_price_residual": max(row["price_residual"] for row in charge_rows),
        "max_demand_residual": max(row["demand_residual"] for row in charge_rows),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_comprehensive_state_grid(
    output_dir: Path,
    profile: str = "full",
    workers: int = 1,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_state_grid(profile)
    charges = charge_grid(profile)
    charge_rows = _solve_charge_rows(cases, charges, workers=workers)
    static_charge = _select_static_charge(charge_rows, charges)
    policy_rows = _policy_rows(charge_rows, static_charge)
    boundary_rows = _boundary_records(policy_rows)
    dynamic_rows = _dynamic_records(charge_rows, policy_rows, charges, profile)
    summary = _summary(
        profile,
        charges,
        policy_rows,
        charge_rows,
        boundary_rows,
        dynamic_rows,
        static_charge,
    )

    _write_csv(output_dir / "comprehensive_state_grid_charge_search_records.csv", charge_rows)
    _write_csv(output_dir / "comprehensive_state_grid_policy_records.csv", policy_rows)
    _write_csv(output_dir / "comprehensive_state_grid_boundary_records.csv", boundary_rows)
    _write_csv(output_dir / "comprehensive_state_grid_dynamic_records.csv", dynamic_rows)
    (output_dir / "comprehensive_state_grid_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
