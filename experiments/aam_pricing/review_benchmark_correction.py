"""Bounded UT retuning over saved equilibrium rows, with no solver dependency."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


THRESHOLDS = (0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20)
RESTRICTED_SLOPES = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
EXPANDED_SLOPES = (0.0,) + RESTRICTED_SLOPES
RESOURCE = (1.0, 1.04, 0.98)
TOLERANCE = 1e-8
BASELINE = "results/comprehensive_state_grid_v1_full/comprehensive_state_grid_charge_search_records.csv"
HARDCAP = "results/hard_capacity_reequilibrium_v1/hard_capacity_reequilibrium_charge_records.csv"
MIXTURE = "results/out_of_model_demand_robustness_v1/taste_mixture_charge_records.csv"
PIGOUVIAN = "results/pigouvian_benchmark_v2/pigouvian_benchmark_policy_records.csv"
MIXTURE_POLICIES = "results/out_of_model_demand_robustness_v1/taste_mixture_policy_records.csv"
MISSPEC_DIR = "results/parameter_misspecification_v1"
ABLATION_DIR = "results/mechanism_ablation_v1"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def mean(values) -> float:
    values = list(values)
    return math.fsum(values) / len(values)


def objective_value(row: dict, objective: str) -> float:
    value = float(row["welfare"])
    if objective == "TS":
        value += math.fsum(float(row[f"price_{i}"]) * float(row[f"demand_{i}"])
                           for i in (1, 2, 3))
    elif objective != "US":
        raise ValueError(f"Unknown objective: {objective}")
    if not math.isfinite(value):
        raise ValueError("Nonfinite objective")
    return value


def group_cases(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["case"]].append(row)
    if not grouped:
        raise ValueError("Empty saved rows")
    grid = None
    for case, group in sorted(grouped.items()):
        charges = [float(row["charge"]) for row in group]
        if len(set(charges)) != len(charges):
            raise ValueError(f"duplicate saved charge in {case}")
        if 0.0 not in charges:
            raise ValueError(f"Missing zero charge in {case}")
        if any(not math.isfinite(c) or c < 0 for c in charges):
            raise ValueError(f"Invalid charge in {case}")
        if grid is not None and set(charges) != grid:
            raise ValueError("Nonrectangular saved charge grid")
        grid = set(charges)
    return {case: sorted(group, key=lambda row: float(row["charge"]))
            for case, group in sorted(grouped.items())}


def nearest_saved_row(rows: list[dict], target: float) -> dict:
    if not math.isfinite(target):
        raise ValueError("Nonfinite target")
    return min(rows, key=lambda row: (abs(float(row["charge"]) - target), float(row["charge"])))


def select_ut_row(rows: list[dict], threshold: float, slope: float,
                  signal: str = "utilization") -> dict:
    nc = next(row for row in rows if float(row["charge"]) == 0)
    utilization = float(nc[signal])
    if not math.isfinite(utilization) or slope < 0:
        raise ValueError("Invalid UT signal or slope")
    return nearest_saved_row(rows, slope * max(0.0, utilization - threshold))


def candidate_key(row: dict) -> tuple:
    return (row["mean_welfare"], -row["mean_charge"], -row["ut_slope"], row["ut_threshold"])


def tune_ut(rows: list[dict], objective: str, signal: str = "utilization", *,
            thresholds=THRESHOLDS, slopes=EXPANDED_SLOPES) -> tuple[dict, list[dict], list[dict]]:
    grouped = group_cases(rows)
    candidates, states = [], []
    for threshold in sorted(set(thresholds)):
        for slope in sorted(set(slopes)):
            choices = []
            for case, group in grouped.items():
                selected = select_ut_row(group, threshold, slope, signal)
                nc = group[0]
                value = objective_value(selected, objective)
                delta = value - objective_value(nc, objective)
                choices.append((selected, value, delta))
                states.append(dict(
                    case=case, objective=objective, ut_threshold=threshold, ut_slope=slope,
                    charge=float(selected["charge"]), objective_value=value, delta_w_vs_nc=delta,
                    nc_signal=float(nc[signal]), signal_key=signal,
                    source_file=selected.get("source_file", ""),
                    source_row_number=selected.get("source_row_number", ""),
                    in_restricted_old=int(slope in RESTRICTED_SLOPES),
                ))
            candidates.append(dict(
                objective=objective, signal_key=signal, ut_threshold=threshold, ut_slope=slope,
                state_count=len(grouped), mean_welfare=mean(v for _, v, _ in choices),
                mean_delta_w_vs_nc=mean(d for _, _, d in choices),
                mean_charge=mean(float(r["charge"]) for r, _, _ in choices),
                positive_charge_share=mean(float(float(r["charge"]) > TOLERANCE) for r, _, _ in choices),
                in_restricted_old=int(slope in RESTRICTED_SLOPES),
            ))
    return max(candidates, key=candidate_key), candidates, states


def revenue(row: dict) -> float:
    return float(row["charge"]) * math.fsum(
        r * float(row[f"demand_{i}"]) for i, r in enumerate(RESOURCE, 1))


def best_objective_row(rows: list[dict], objective: str) -> dict:
    return max(rows, key=lambda r: (objective_value(r, objective), -float(r["charge"])))


def summarize_policy_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(row)
    summaries = []
    sd_gain = mean(float(r["delta_w_vs_nc"]) for r in grouped["SD"])
    for policy, group in sorted(grouped.items()):
        n = len(group)
        positive = sum(float(r["charge"]) > TOLERANCE for r in group)
        fp = sum(float(r["charge"]) > TOLERANCE and float(r["sd_charge"]) <= TOLERANCE for r in group)
        fn = sum(float(r["charge"]) <= TOLERANCE and float(r["sd_charge"]) > TOLERANCE for r in group)
        delta = mean(float(r["delta_w_vs_nc"]) for r in group)
        summaries.append(dict(
            policy=policy, state_count=n, denominator="all_states", positive_charge_count=positive,
            positive_charge_share=positive/n, mean_charge=mean(float(r["charge"]) for r in group),
            mean_objective=mean(float(r["objective_value"]) for r in group),
            mean_delta_w_vs_nc=delta, welfare_recovery_vs_sd=delta/sd_gain if sd_gain > TOLERANCE else None,
            positive_delta_w_count=sum(float(r["delta_w_vs_nc"]) > TOLERANCE for r in group),
            negative_delta_w_count=sum(float(r["delta_w_vs_nc"]) < -TOLERANCE for r in group),
            false_positive_count_vs_sd=fp, false_negative_count_vs_sd=fn,
            false_positive_share_vs_sd=fp/n, false_negative_share_vs_sd=fn/n,
        ))
    return summaries


def evaluate_domain(rows: list[dict], objective: str, signal: str = "utilization") -> dict:
    grouped = group_cases(rows)
    best, tuning, candidate_states = tune_ut(rows, objective, signal)
    old = max((r for r in tuning if r["in_restricted_old"]), key=candidate_key)
    charges = sorted(float(r["charge"]) for r in next(iter(grouped.values())))
    static = max(charges, key=lambda c: (
        mean(objective_value(nearest_saved_row(g, c), objective) for g in grouped.values()), -c))
    active = {case for case, g in grouped.items() if len(g) > 1 and
              (objective_value(g[1], objective) - objective_value(g[0], objective)) /
              float(g[1]["charge"]) > TOLERANCE}
    air_charge = max((c for c in charges if c > 0), key=lambda c: (
        mean(objective_value(nearest_saved_row(grouped[case], c), objective) for case in sorted(active)), -c)
    ) if active else 0.0
    policies = []
    for case, group in grouped.items():
        nc = group[0]
        sd = best_objective_row(group, objective)
        choices = dict(NC=nc, SC=nearest_saved_row(group, static), SD=sd,
                       RM=max(group, key=lambda r: (revenue(r), -float(r["charge"]))),
                       AIR=nearest_saved_row(group, air_charge if case in active else 0),
                       UT=select_ut_row(group, best["ut_threshold"], best["ut_slope"], signal),
                       UT_restricted_old=select_ut_row(group, old["ut_threshold"], old["ut_slope"], signal))
        for policy, source in sorted(choices.items()):
            params = old if policy == "UT_restricted_old" else best
            policies.append(dict(source, policy=policy, objective=objective,
                                 objective_value=objective_value(source, objective),
                                 nc_objective=objective_value(nc, objective),
                                 delta_w_vs_nc=objective_value(source, objective)-objective_value(nc, objective),
                                 sd_charge=float(sd["charge"]), nc_signal=float(nc[signal]), signal_key=signal,
                                 ut_threshold=params["ut_threshold"] if policy.startswith("UT") else "",
                                 ut_slope=params["ut_slope"] if policy.startswith("UT") else ""))
    for candidate in tuning:
        candidate["selected_expanded"] = int(candidate is best)
        candidate["selected_restricted_old"] = int(candidate is old)
    return dict(best=best, restricted_old=old, tuning_rows=tuning, candidate_state_rows=candidate_states,
                policy_rows=policies, summary_rows=summarize_policy_rows(policies),
                state_count=len(grouped), static_charge=static, air_positive_charge=air_charge)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_saved(root: Path, relative: str) -> list[dict]:
    return [dict(row, source_file=relative, source_row_number=i)
            for i, row in enumerate(read_csv(root / relative), 2)]


def export_domain(output: Path, name: str, result: dict) -> None:
    directory = output / name
    for key in ("tuning_rows", "candidate_state_rows", "policy_rows", "summary_rows"):
        write_csv(directory / f"{key}.csv", result[key])
    write_json(directory / "summary.json", {k: v for k, v in result.items() if not k.endswith("rows")})


def corrected_pigouvian(original: list[dict], ut_rows: list[dict]) -> list[dict]:
    by_case = {r["case"]: r for r in ut_rows if r["policy"] == "UT"}
    nc = {r["case"]: r for r in original if r["policy"] == "NC"}
    fields = list(original[0])
    output = []
    for row in original:
        if row["policy"] != "UT":
            output.append(dict(row))
            continue
        source = by_case[row["case"]]
        zero = nc[row["case"]]
        new = dict.fromkeys(fields, "")
        new.update({k: source[k] for k in fields if k in source})
        gain = float(source["welfare"]) - float(zero["welfare"])
        util = float(zero["utilization"]) - float(source["utilization"])
        reliability = float(zero["average_reliability"]) - float(source["average_reliability"])
        new.update(policy="UT", delta_w_vs_nc=gain, group_key_1=row["case"],
                   resource_demand=math.fsum(r*float(source[f"demand_{i}"]) for i, r in enumerate(RESOURCE, 1)),
                   charge_revenue=revenue(source), nc_utilization=zero["utilization"],
                   nc_average_reliability=zero["average_reliability"],
                   utilization_relief_vs_NC=util, utilization_relief_vs_nc=util,
                   reliability_relief_vs_NC=reliability, reliability_relief_vs_nc=reliability,
                   welfare_gain_percent_vs_NC=100*gain/max(abs(float(zero["welfare"])), 1.0))
        output.append(new)
    return output


def corrected_mixtures(original: list[dict], results: dict) -> list[dict]:
    lookup = {(scenario, r["case"]): r for scenario, result in results.items()
              for r in result["policy_rows"] if r["policy"] == "UT"}
    output = []
    for row in original:
        if row["policy"] != "UT_best":
            output.append(dict(row))
            continue
        new = dict(row)
        selected = lookup[(row["scenario"], row["case"])]
        charge = float(selected["charge"])
        sd_positive = float(row["alternative_sd_charge"]) > TOLERANCE
        new.update(selected_charge=charge, selected_welfare=selected["welfare"],
                   delta_w_vs_nc=selected["delta_w_vs_nc"], selected_positive=int(charge > TOLERANCE),
                   false_positive_vs_alternative_sd=int(charge > TOLERANCE and not sd_positive),
                   false_negative_vs_alternative_sd=int(charge <= TOLERANCE and sd_positive),
                   selected_utilization=selected["utilization"])
        output.append(new)
    return output


def mixture_summaries(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scenario"], row["policy"]].append(row)
    output = []
    for (scenario, policy), group in sorted(grouped.items()):
        sd_delta = mean(float(r["delta_w_vs_nc"]) for r in grouped[scenario, "SD_alt"])
        delta = mean(float(r["delta_w_vs_nc"]) for r in group)
        output.append(dict(scenario=scenario, policy=policy, state_count=len(group),
                           positive_charge_share=mean(float(r["selected_positive"]) for r in group),
                           mean_selected_charge=mean(float(r["selected_charge"]) for r in group),
                           mean_delta_w_vs_nc=delta,
                           welfare_recovery_vs_alternative_sd=delta/sd_delta if sd_delta > TOLERANCE else None,
                           false_positive_share_vs_alternative_sd=mean(float(r["false_positive_vs_alternative_sd"]) for r in group),
                           false_negative_share_vs_alternative_sd=mean(float(r["false_negative_vs_alternative_sd"]) for r in group)))
    return output


def performance_summaries(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("scenario", ""), row["policy"]].append(row)
    output = []
    for (scenario, policy), group in sorted(grouped.items()):
        record = dict(policy=policy, state_count=len(group),
                      positive_charge_share=mean(float(float(r["charge"]) > TOLERANCE) for r in group),
                      mean_charge=mean(float(r["charge"]) for r in group),
                      mean_delta_w=mean(float(r["delta_w_vs_nc"]) for r in group),
                      mean_utilization_relief=mean(float(r["utilization_relief_vs_nc"]) for r in group),
                      mean_reliability_relief=mean(float(r["reliability_relief_vs_nc"]) for r in group))
        if scenario:
            record = dict(scenario=scenario, **record)
        output.append(record)
    return output


def correct_ablations(root: Path, output: Path) -> dict:
    charge_path = f"{ABLATION_DIR}/mechanism_ablation_charge_records.csv"
    policy_path = f"{ABLATION_DIR}/mechanism_ablation_policy_records.csv"
    source = load_saved(root, charge_path)
    grouped = defaultdict(list)
    for row in source:
        grouped[row["scenario"]].append(row)
    lookup, metadata = {}, {}
    for scenario, rows in sorted(grouped.items()):
        result = evaluate_domain(rows, "US")
        if result["state_count"] != 450:
            raise ValueError(f"Incomplete ablation cache: {scenario}")
        export_domain(output / "ablations", scenario, result)
        metadata[scenario] = {k: result[k] for k in ("best", "restricted_old", "state_count")}
        lookup.update({(scenario, r["case"]): r for r in result["policy_rows"] if r["policy"] == "UT"})
    old = read_csv(root / policy_path)
    corrected, changed = [], []
    for row in old:
        selected = lookup[row["scenario"], row["case"]]
        new = dict(row, ut_threshold=selected["ut_threshold"], ut_slope=selected["ut_slope"])
        if row["policy"] == "UT":
            new.update({k: selected[k] for k in row if k in selected})
            zero = next(r for r in grouped[row["scenario"]] if r["case"] == row["case"] and float(r["charge"]) == 0)
            new.update(delta_w_vs_nc=selected["delta_w_vs_nc"],
                       utilization_relief_vs_nc=float(zero["utilization"])-float(selected["utilization"]),
                       reliability_relief_vs_nc=float(zero["average_reliability"])-float(selected["average_reliability"]))
            if float(new["charge"]) != float(row["charge"]):
                changed.append(dict(new, old_charge=row["charge"], old_delta_w_vs_nc=row["delta_w_vs_nc"]))
        corrected.append(new)
    summaries = performance_summaries(corrected)
    write_csv(output / "mechanism_ablation_policy_records.csv", corrected)
    write_csv(output / "mechanism_ablation_summary_rows.csv", summaries)
    write_csv(output / "mechanism_ablation_changed_ut_rows.csv", changed)
    original_summary_path = f"{ABLATION_DIR}/mechanism_ablation_summary.json"
    summary = json.loads((root / original_summary_path).read_text(encoding="utf-8"))
    for key in ("positive_charge_share", "mean_charge", "mean_delta_w", "mean_utilization_relief", "mean_reliability_relief"):
        summary[key] = {s: {r["policy"]: r[key] for r in summaries if r["scenario"] == s} for s in grouped}
    summary.update(ut_correction=metadata, changed_ut_state_choices=len(changed), source_cache=charge_path)
    write_json(output / "mechanism_ablation_summary.json", summary)
    return dict(scenarios=metadata, changed_selected_charge_count=len(changed),
                source_files=[charge_path, policy_path, original_summary_path])


def export_full_misspecification(root: Path, output: Path, evaluated: list[dict], true_sd_gain: float) -> None:
    original = read_csv(root / f"{MISSPEC_DIR}/parameter_misspecification_policy_records.csv")
    lookup = {(r["scenario"], r["case"]): r for r in evaluated if r["policy"] == "UT"}
    corrected = []
    for row in original:
        if row["policy"] != "UT" or (row["scenario"], row["case"]) not in lookup:
            corrected.append(dict(row))
            continue
        selected = lookup[row["scenario"], row["case"]]
        charge = float(selected["charge"])
        sd_positive = float(selected["sd_charge"]) > TOLERANCE
        corrected.append(dict(row, selected_charge=charge, realized_welfare=selected["welfare"],
                              realized_delta_w=selected["delta_w_vs_nc"], selected_positive_charge=int(charge > TOLERANCE),
                              false_positive_vs_true_sd=int(charge > TOLERANCE and not sd_positive),
                              false_negative_vs_true_sd=int(charge <= TOLERANCE and sd_positive),
                              realized_utilization=selected["utilization"]))
    grouped = defaultdict(list)
    for row in corrected:
        grouped[row["scenario"], row["policy"]].append(row)
    summaries = []
    for (scenario, policy), group in sorted(grouped.items()):
        delta = mean(float(r["realized_delta_w"]) for r in group)
        summaries.append(dict(
            scenario=scenario, policy=policy, state_count=len(group),
            positive_charge_share=mean(float(r["selected_positive_charge"]) for r in group),
            false_positive_share=mean(float(r["false_positive_vs_true_sd"]) for r in group),
            false_negative_share=mean(float(r["false_negative_vs_true_sd"]) for r in group),
            realized_mean_delta_w=delta,
            realized_welfare_recovery_vs_true_sd=delta/true_sd_gain if true_sd_gain > TOLERANCE else 0.0,
            worst_delta_w_true_positive_states=min((float(r["realized_delta_w"]) for r in group
                                                   if int(r["true_sd_positive_charge"])), default=0.0)))
    write_csv(output / "parameter_misspecification_policy_records.csv", corrected)
    write_csv(output / "parameter_misspecification_summary_rows.csv", summaries)
    summary = json.loads((root / f"{MISSPEC_DIR}/parameter_misspecification_summary.json").read_text(encoding="utf-8"))
    summary.update(min_realized_mean_delta_w=min(r["realized_mean_delta_w"] for r in summaries),
                   max_false_positive_share=max(r["false_positive_share"] for r in summaries),
                   max_false_negative_share=max(r["false_negative_share"] for r in summaries),
                   worst_summary_row=min(summaries, key=lambda r: r["realized_mean_delta_w"]),
                   cached_ut_correction=True)
    write_json(output / "parameter_misspecification_summary.json", summary)


def protected_artifacts(root: Path) -> list[Path]:
    protected = []
    for directory in ('results', 'figures'):
        for path in (root / directory).rglob('*'):
            relative = path.relative_to(root / directory)
            if path.is_file() and not relative.parts[0].startswith('trb_review_'):
                protected.append(path)
    if (root / 'MAIN.tex').is_file():
        protected.append(root / 'MAIN.tex')
    return sorted(protected)


def run_review_benchmark_correction(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    allowed = root / "results/trb_review_benchmark_correction_v1"
    if output != allowed:
        raise ValueError(f"Output is bounded to {allowed}")
    protected = protected_artifacts(root)
    hashes = {p.relative_to(root).as_posix(): sha256(p) for p in protected}
    inputs = [BASELINE, HARDCAP, MIXTURE, PIGOUVIAN, MIXTURE_POLICIES,
              f"{MISSPEC_DIR}/parameter_misspecification_policy_records.csv"]
    baseline, hardcap, mixtures = (load_saved(root, name) for name in (BASELINE, HARDCAP, MIXTURE))
    mixture_groups = defaultdict(list)
    for row in mixtures:
        mixture_groups[row["scenario"]].append(row)
    domain_rows = dict(baseline=(baseline, "utilization"), hardcap=(hardcap, "unrestricted_utilization"))
    domain_rows.update({s: (rows, "utilization") for s, rows in sorted(mixture_groups.items())})
    all_summary, metadata, us_mixtures = [], {}, {}
    baseline_result = None
    for domain, (rows, signal) in domain_rows.items():
        if len(group_cases(rows)) != 450:
            raise ValueError(f"Expected all 450 states: {domain}")
        for objective in ("US", "TS"):
            name = f"{domain}_{objective}"
            result = evaluate_domain(rows, objective, signal)
            export_domain(output, name, result)
            all_summary.extend(dict(domain=domain, objective=objective, **r) for r in result["summary_rows"])
            metadata[name] = {k: result[k] for k in ("best", "restricted_old", "state_count")}
            if domain == "baseline" and objective == "US":
                baseline_result = result
            if domain in mixture_groups and objective == "US":
                us_mixtures[domain] = result
    original_pig = read_csv(root / PIGOUVIAN)
    corrected_pig = corrected_pigouvian(original_pig, baseline_result["policy_rows"])
    write_csv(output / "pigouvian_benchmark_policy_records.csv", corrected_pig)
    pig_summary = performance_summaries(corrected_pig)
    write_csv(output / "pigouvian_benchmark_summary_rows.csv", pig_summary)
    pig_json_path = "results/pigouvian_benchmark_v2/pigouvian_benchmark_summary.json"
    pig_json = json.loads((root / pig_json_path).read_text(encoding="utf-8"))
    for key in ("positive_charge_share", "mean_charge", "mean_delta_w", "mean_utilization_relief", "mean_reliability_relief"):
        pig_json[key] = {r["policy"]: r[key] for r in pig_summary}
    pig_json["ut_correction"] = metadata["baseline_US"]
    write_json(output / "pigouvian_benchmark_summary.json", pig_json)
    active_cases = {r["case"] for r in corrected_pig if r["policy"] == "SD" and float(r["charge"]) > TOLERANCE}
    active_rows = [r for r in corrected_pig if r["case"] in active_cases]
    write_csv(output / "figure5_active_state_policy_records.csv", active_rows)
    write_csv(output / "figure5_active_state_summary_rows.csv", performance_summaries(active_rows))
    mixture_policies = corrected_mixtures(read_csv(root / MIXTURE_POLICIES), us_mixtures)
    write_csv(output / "taste_mixture_policy_records.csv", mixture_policies)
    mixture_summary_rows = mixture_summaries(mixture_policies)
    write_csv(output / "taste_mixture_summary_rows.csv", mixture_summary_rows)
    mixture_json_path = "results/out_of_model_demand_robustness_v1/taste_mixture_summary.json"
    mixture_json = json.loads((root / mixture_json_path).read_text(encoding="utf-8"))
    mixture_json.update(minimum_non_nc_mean_delta_w=min(r["mean_delta_w_vs_nc"] for r in mixture_summary_rows if r["policy"] != "NC"),
                        ut_correction={s: metadata[f"{s}_US"] for s in us_mixtures})
    write_json(output / "taste_mixture_summary.json", mixture_json)
    misspec = correct_misspecification(root, output, baseline)
    inputs.extend(misspec.pop("source_files"))
    ablations = correct_ablations(root, output)
    inputs.extend(ablations.pop("source_files"))
    inputs.extend([pig_json_path, mixture_json_path, f"{MISSPEC_DIR}/parameter_misspecification_summary.json"])
    write_csv(output / "policy_summary_rows.csv", all_summary)
    unchanged = all(sha256(root / p) == digest for p, digest in hashes.items())
    if not unchanged:
        raise RuntimeError("Protected source artifact changed during cached correction")
    manifest = dict(
        no_equilibrium_solves=True, outcome_interpolation=False,
        objectives={"US": "saved welfare", "TS": "saved welfare + sum(saved price_i * saved demand_i)"},
        thresholds=list(THRESHOLDS), expanded_slopes=list(EXPANDED_SLOPES),
        restricted_old_slopes=list(RESTRICTED_SLOPES),
        tie_break="max mean objective, min mean charge, min slope, max threshold",
        domains=metadata, ablations=ablations, misspecification=misspec, protected_artifacts_unchanged=unchanged,
        protected_artifact_count=len(hashes),
        input_sha256={p: sha256(root / p) for p in inputs},
        parent_code_freeze_sha256=sha256(root / "docs/qa/trb_review_revision_20261016/code_before.zip"),
        figure5_active_state_count=len(active_cases),
    )
    write_json(output / "protected_source_sha256.json", hashes)
    write_json(output / "manifest.json", manifest)
    return manifest


def correct_misspecification(root: Path, output: Path, true_rows: list[dict]) -> dict:
    original_path = f"{MISSPEC_DIR}/parameter_misspecification_policy_records.csv"
    original = [r for r in read_csv(root / original_path) if r["policy"] == "UT"]
    scenarios = sorted({r["scenario"] for r in original})
    true_groups = group_cases(true_rows)
    original_lookup = {(r["scenario"], r["case"]): r for r in original}
    evaluated, summaries, sources, status = [], [], [], []
    for scenario in scenarios:
        path = f"{MISSPEC_DIR}/{scenario}_perceived_charge_records.csv"
        if not (root / path).exists():
            status.append(dict(scenario=scenario, status="missing_perceived_cache_no_computation"))
            continue
        sources.append(path)
        perceived = load_saved(root, path)
        if set(group_cases(perceived)) != set(true_groups):
            raise ValueError(f"Perceived state coverage mismatch: {scenario}")
        result = evaluate_domain(perceived, "US")
        export_domain(output / "misspecification_perceived", scenario, result)
        changed = 0
        for selected in result["policy_rows"]:
            if selected["policy"] not in ("UT", "UT_restricted_old"):
                continue
            case = selected["case"]
            group = true_groups[case]
            charge = float(selected["charge"])
            true = nearest_saved_row(group, charge)
            if float(true["charge"]) != charge:
                raise ValueError("Perceived selected charge not present in true saved grid")
            old = original_lookup[scenario, case]
            change = int(charge != float(old["selected_charge"]))
            changed += change if selected["policy"] == "UT" else 0
            sd = best_objective_row(group, "US")
            evaluated.append(dict(true, scenario=scenario, policy=selected["policy"], objective="US",
                                  objective_value=objective_value(true, "US"),
                                  delta_w_vs_nc=objective_value(true, "US")-objective_value(group[0], "US"),
                                  sd_charge=float(sd["charge"]),
                                  ut_threshold=selected["ut_threshold"], ut_slope=selected["ut_slope"],
                                  perceived_source_file=selected["source_file"],
                                  perceived_source_row_number=selected["source_row_number"],
                                  perceived_objective_value=selected["objective_value"],
                                  old_selected_charge=old["selected_charge"],
                                  old_realized_delta_w=old["realized_delta_w"], selection_changed=change,
                                  misspecified_parameter=old["misspecified_parameter"],
                                  misspecification_factor=old["misspecification_factor"]))
        scenario_rows = [r for r in evaluated if r["scenario"] == scenario]
        sd_rows = [dict(r, policy="SD", objective_value=objective_value(r, "US"),
                        delta_w_vs_nc=objective_value(r, "US")-objective_value(true_groups[case][0], "US"),
                        sd_charge=float(r["charge"]))
                   for case, group in true_groups.items() for r in [best_objective_row(group, "US")]]
        summaries.extend(dict(scenario=scenario, **r) for r in summarize_policy_rows(scenario_rows + sd_rows)
                         if r["policy"] != "SD")
        status.append(dict(scenario=scenario, status="cached_selection_and_true_evaluation",
                           affected_original_ut_rows=450, changed_selected_charge_count=changed,
                           best=result["best"], restricted_old=result["restricted_old"]))
    write_csv(output / "parameter_misspecification_affected_ut_rows.csv", original)
    write_csv(output / "parameter_misspecification_corrected_ut_records.csv", evaluated)
    write_csv(output / "parameter_misspecification_changed_ut_rows.csv",
              [r for r in evaluated if r["policy"] == "UT" and r["selection_changed"]])
    write_csv(output / "parameter_misspecification_corrected_summary_rows.csv", summaries)
    true_sd_gain = mean(objective_value(best_objective_row(g, "US"), "US")-objective_value(g[0], "US")
                        for g in true_groups.values())
    export_full_misspecification(root, output, evaluated, true_sd_gain)
    return dict(scenarios=status, affected_original_ut_row_count=len(original),
                evaluated_row_count=len(evaluated), source_files=sources)
