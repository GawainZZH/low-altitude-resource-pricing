from __future__ import annotations

import csv
import json
import math
from pathlib import Path


RESOURCE = (1.00, 1.04, 0.98)
UTILIZATION_THRESHOLDS = (0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20)
UTILIZATION_SLOPES = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _as_float(row: dict, key: str) -> float:
    return float(row[key])


def _resource_demand(row: dict) -> float:
    return sum(RESOURCE[index] * _as_float(row, f"demand_{index + 1}") for index in range(3))


def _charge_revenue(row: dict) -> float:
    return _as_float(row, "charge") * _resource_demand(row)


def _nearest_charge(charges: list[float], target: float) -> float:
    capped = min(max(target, min(charges)), max(charges))
    return min(charges, key=lambda charge: (abs(charge - capped), charge))


def _group(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict]]:
    grouped: dict[tuple[str, ...], list[dict]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    return grouped


def _ut_row_for_group(group_rows: list[dict], threshold: float, slope: float) -> dict:
    charges = sorted({_as_float(row, "charge") for row in group_rows})
    by_charge = {_as_float(row, "charge"): row for row in group_rows}
    nc = by_charge[0.0]
    nc_util = _as_float(nc, "utilization")
    ut_target = slope * max(0.0, nc_util - threshold)
    return by_charge[_nearest_charge(charges, ut_target)]


def _tune_ut(rows: list[dict], keys: tuple[str, ...]) -> tuple[dict, list[dict]]:
    grouped = _group(rows, keys)
    tuning_rows = []
    for threshold in UTILIZATION_THRESHOLDS:
        for slope in UTILIZATION_SLOPES:
            choices = [
                _ut_row_for_group(group_rows, threshold, slope)
                for group_rows in grouped.values()
            ]
            mean_welfare = sum(_as_float(row, "welfare") for row in choices) / len(choices)
            mean_charge = sum(_as_float(row, "charge") for row in choices) / len(choices)
            positive_share = sum(1 for row in choices if _as_float(row, "charge") > 0.0) / len(choices)
            tuning_rows.append(
                {
                    "ut_threshold": f"{threshold:.2f}",
                    "ut_slope": f"{slope:.1f}",
                    "mean_welfare": f"{mean_welfare:.10f}",
                    "mean_charge": f"{mean_charge:.10f}",
                    "positive_charge_share": f"{positive_share:.10f}",
                }
            )
    best = max(
        tuning_rows,
        key=lambda row: (
            _as_float(row, "mean_welfare"),
            -_as_float(row, "mean_charge"),
            -_as_float(row, "positive_charge_share"),
            -_as_float(row, "ut_slope"),
            _as_float(row, "ut_threshold"),
        ),
    )
    return best, tuning_rows


def _select_rows(rows: list[dict], keys: tuple[str, ...], ut_params: dict) -> list[dict]:
    selected = []
    ut_threshold = _as_float(ut_params, "ut_threshold")
    ut_slope = _as_float(ut_params, "ut_slope")
    for group_key, group_rows in _group(rows, keys).items():
        charges = sorted({_as_float(row, "charge") for row in group_rows})
        by_charge = {_as_float(row, "charge"): row for row in group_rows}
        nc = by_charge[0.0]
        nc_util = _as_float(nc, "utilization")
        choices = {
            "NC": nc,
            "UT": _ut_row_for_group(group_rows, ut_threshold, ut_slope),
            "RM": max(group_rows, key=_charge_revenue),
            "SD": max(group_rows, key=lambda row: _as_float(row, "welfare")),
        }
        nc_welfare = _as_float(nc, "welfare")
        nc_reliability = _as_float(nc, "average_reliability")
        for policy, source in choices.items():
            row = dict(source)
            row["policy"] = policy
            row["resource_demand"] = f"{_resource_demand(row):.10f}"
            row["charge_revenue"] = f"{_charge_revenue(row):.10f}"
            row["nc_utilization"] = f"{nc_util:.10f}"
            row["ut_threshold"] = f"{ut_threshold:.2f}"
            row["ut_slope"] = f"{ut_slope:.1f}"
            row["welfare_gain_percent_vs_NC"] = f"{100.0 * (_as_float(row, 'welfare') - nc_welfare) / max(abs(nc_welfare), 1.0):.10f}"
            row["utilization_relief_vs_NC"] = f"{nc_util - _as_float(row, 'utilization'):.10f}"
            row["reliability_relief_vs_NC"] = f"{nc_reliability - _as_float(row, 'average_reliability'):.10f}"
            for index, key in enumerate(keys):
                row[f"group_key_{index + 1}"] = group_key[index]
            selected.append(row)
    return sorted(selected, key=lambda row: tuple(row[f"group_key_{index + 1}"] for index in range(len(keys))) + (row["policy"],))


def _summarize(selected: list[dict], ut_params: dict, tuning_rows: list[dict]) -> dict:
    by_policy: dict[str, list[dict]] = {}
    for row in selected:
        by_policy.setdefault(row["policy"], []).append(row)

    def mean(policy: str, key: str) -> float:
        values = [_as_float(row, key) for row in by_policy[policy]]
        return sum(values) / len(values)

    def share(policy: str, predicate) -> float:
        rows = by_policy[policy]
        return sum(1 for row in rows if predicate(row)) / len(rows)

    return {
        "policies": sorted(by_policy),
        "state_count": len(by_policy["NC"]),
        "utilization_threshold_candidates": list(UTILIZATION_THRESHOLDS),
        "utilization_slope_candidates": list(UTILIZATION_SLOPES),
        "best_tuned_utilization_threshold": _as_float(ut_params, "ut_threshold"),
        "best_tuned_utilization_slope": _as_float(ut_params, "ut_slope"),
        "best_tuned_ut_mean_welfare": _as_float(ut_params, "mean_welfare"),
        "best_tuned_ut_mean_charge": _as_float(ut_params, "mean_charge"),
        "best_tuned_ut_positive_charge_share": _as_float(ut_params, "positive_charge_share"),
        "ut_tuning_candidate_count": len(tuning_rows),
        "mean_gain_percent_vs_NC": {
            policy: mean(policy, "welfare_gain_percent_vs_NC")
            for policy in sorted(by_policy)
        },
        "mean_charge": {
            policy: mean(policy, "charge")
            for policy in sorted(by_policy)
        },
        "mean_revenue": {
            policy: mean(policy, "charge_revenue")
            for policy in sorted(by_policy)
        },
        "mean_utilization": {
            policy: mean(policy, "utilization")
            for policy in sorted(by_policy)
        },
        "positive_charge_share": {
            policy: share(policy, lambda row: _as_float(row, "charge") > 0.0)
            for policy in sorted(by_policy)
        },
    }


def _dynamic_summary(selected: list[dict]) -> dict:
    by_policy: dict[str, list[dict]] = {}
    for row in selected:
        by_policy.setdefault(row["policy"], []).append(row)

    def total(policy: str, key: str) -> float:
        return sum(_as_float(row, key) for row in by_policy[policy])

    def movement(policy: str) -> tuple[float, float]:
        rows = sorted(by_policy[policy], key=lambda row: int(row["period"]))
        previous = 0.0
        changes = []
        for row in rows:
            charge = _as_float(row, "charge")
            changes.append(abs(charge - previous))
            previous = charge
        return sum(changes), max(changes)

    nc_welfare = total("NC", "welfare")
    records = {}
    for policy in sorted(by_policy):
        total_move, max_change = movement(policy)
        records[policy] = {
            "total_welfare": total(policy, "welfare"),
            "gain_percent_vs_NC": 100.0 * (total(policy, "welfare") - nc_welfare) / max(abs(nc_welfare), 1.0),
            "total_revenue": total(policy, "charge_revenue"),
            "total_movement": total_move,
            "max_change": max_change,
        }
    return records


def _write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_regulatory_benchmarks(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = {
        "anchor": (Path("results/calibrated_anchor_case_v1/anchor_charge_search_records.csv"), ("case",)),
        "stress": (Path("results/stress_robustness_v1/stress_robustness_charge_search_records.csv"), ("case",)),
        "grid": (Path("results/comprehensive_state_grid_v1_full/comprehensive_state_grid_charge_search_records.csv"), ("case",)),
        "dynamic": (Path("results/dynamic_sequence_v1/dynamic_sequence_charge_search_records.csv"), ("period",)),
    }

    summary = {}
    for name, (source, keys) in tasks.items():
        rows = _read_csv(source)
        ut_params, tuning_rows = _tune_ut(rows, keys)
        selected = _select_rows(rows, keys, ut_params)
        _write_csv(output_dir / f"{name}_ut_tuning_records.csv", tuning_rows)
        _write_csv(output_dir / f"{name}_benchmark_policy_records.csv", selected)
        summary[name] = _summarize(selected, ut_params, tuning_rows)
        if name == "dynamic":
            summary[name]["dynamic_totals"] = _dynamic_summary(selected)

    _write_summary(output_dir / "regulatory_benchmarks_summary.json", summary)
    return summary
