"""Validate and aggregate two blinded C2 annotation sheets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABELS = ("FULL", "PARTIAL", "NONE")
SEVERITIES = ("light", "medium", "heavy")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def wilson(success: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    p = success / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - margin, center + margin


def cohen_kappa(a: list[str], b: list[str]) -> tuple[float, float]:
    if not a:
        return float("nan"), float("nan")
    agreement = sum(x == y for x, y in zip(a, b)) / len(a)
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[label] / len(a)) * (cb[label] / len(b)) for label in LABELS)
    kappa = (agreement - expected) / (1 - expected) if expected < 1 else float("nan")
    return agreement, kappa


def exact_binomial_two_sided(success: int, total: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value using the probability-ordering definition."""
    if total == 0:
        return 1.0
    observed = math.comb(total, success) * p**success * (1 - p) ** (total - success)
    probabilities = [math.comb(total, k) * p**k * (1 - p) ** (total - k) for k in range(total + 1)]
    return min(1.0, sum(value for value in probabilities if value <= observed + 1e-15))


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - location) + ordered[upper] * (location - lower)


def clustered_exact_rates(
    rows: list[dict[str, Any]], seed: int = 20261002, repetitions: int = 20_000
) -> list[dict[str, Any]]:
    """Bootstrap exact-reconstruction rates by source anchor, preserving paired raters."""
    output: list[dict[str, Any]] = []
    for severity in (*SEVERITIES, "all"):
        subset = rows if severity == "all" else [row for row in rows if row["severity"] == severity]
        clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in subset:
            clusters[row["source_sample_id"]].append(row)
        keys = sorted(clusters)
        rng = random.Random(seed)
        boot = {"A": [], "B": [], "pooled": []}
        for _ in range(repetitions):
            sampled = [rng.choice(keys) for _ in keys]
            draw = [row for key in sampled for row in clusters[key]]
            exact_a = sum(row["answer_status_A"] == "EXACT" for row in draw)
            exact_b = sum(row["answer_status_B"] == "EXACT" for row in draw)
            boot["A"].append(exact_a / len(draw))
            boot["B"].append(exact_b / len(draw))
            boot["pooled"].append((exact_a + exact_b) / (2 * len(draw)))
        exact_a = sum(row["answer_status_A"] == "EXACT" for row in subset)
        exact_b = sum(row["answer_status_B"] == "EXACT" for row in subset)
        for annotator, rate in (
            ("A", exact_a / len(subset)),
            ("B", exact_b / len(subset)),
            ("pooled", (exact_a + exact_b) / (2 * len(subset))),
        ):
            output.append({
                "annotator": annotator,
                "severity": severity,
                "n_source_anchors": len(keys),
                "rate": rate,
                "cluster_bootstrap_ci95_low": percentile(boot[annotator], 0.025),
                "cluster_bootstrap_ci95_high": percentile(boot[annotator], 0.975),
                "bootstrap_repetitions": repetitions,
                "seed": seed,
            })
    return output


def normalize_json(text: str) -> Any:
    return json.loads(text)


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else key
            out.update(flatten(item, child))
        return out
    if isinstance(value, list):
        out = {}
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            out.update(flatten(item, child))
        return out
    return {prefix: value}


def unwrap_answer(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"final_result"}:
        return value["final_result"]
    return value


def score_answer(answer_text: str, target: Any) -> tuple[str, float, str]:
    text = answer_text.strip()
    if not text:
        return "MISSING", 0.0, "blank answer"
    try:
        answer = unwrap_answer(normalize_json(text))
    except json.JSONDecodeError as exc:
        return "INVALID_JSON", 0.0, str(exc)
    if answer == target:
        return "EXACT", 1.0, ""
    target_flat = flatten(target)
    answer_flat = flatten(answer)
    if not target_flat:
        return "NONEXACT", 0.0, "empty target"
    matched = sum(path in answer_flat and answer_flat[path] == value for path, value in target_flat.items())
    coverage = matched / len(target_flat)
    return ("PARTIAL_MATCH" if matched else "NONMATCH"), coverage, ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator-a", type=Path, required=True)
    parser.add_argument("--annotator-b", type=Path, required=True)
    parser.add_argument("--gold-key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    raw_a = read_csv(args.annotator_a)
    raw_b = read_csv(args.annotator_b)
    raw_gold = read_csv(args.gold_key)
    for name, rows in (("Annotator A", raw_a), ("Annotator B", raw_b), ("Gold key", raw_gold)):
        ids = [row["item_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise SystemExit(f"{name} contains duplicate item IDs")
    rows_a = {row["item_id"]: row for row in raw_a}
    rows_b = {row["item_id"]: row for row in raw_b}
    gold = {row["item_id"]: row for row in raw_gold}
    if set(rows_a) != set(rows_b) or set(rows_a) != set(gold):
        raise SystemExit("Annotator sheets and gold key must contain the same unique item IDs")

    invalid: list[dict[str, Any]] = []
    scored: list[dict[str, Any]] = []
    for item_id in sorted(gold):
        target = json.loads(gold[item_id]["observation_target_json"])
        combined: dict[str, Any] = {
            "item_id": item_id,
            "source_sample_id": gold[item_id]["source_sample_id"],
            "severity": gold[item_id]["severity"],
        }
        for name, source in (("A", rows_a[item_id]), ("B", rows_b[item_id])):
            label = source["recoverability_label"].strip().upper()
            confidence = source["confidence_1_to_5"].strip()
            problems = []
            if label not in LABELS:
                problems.append("label must be FULL/PARTIAL/NONE")
            if not confidence.isdigit() or not 1 <= int(confidence) <= 5:
                problems.append("confidence must be integer 1-5")
            if not source["evidence_citation"].strip():
                problems.append("evidence_citation is required")
            expected_prompt_hash = gold[item_id].get("prompt_sha256", "").strip()
            actual_prompt_hash = hashlib.sha256(source["model_visible_prompt"].encode("utf-8")).hexdigest()
            if expected_prompt_hash and actual_prompt_hash != expected_prompt_hash:
                problems.append("model_visible_prompt differs from the issued sheet")
            status, coverage, error = score_answer(source["recovered_answer_json"], target)
            if label != "NONE" and status in {"MISSING", "INVALID_JSON"}:
                problems.append("FULL/PARTIAL requires valid recovered_answer_json")
            if problems:
                invalid.append({"item_id": item_id, "annotator": name, "problems": "; ".join(problems), "json_error": error})
            combined[f"label_{name}"] = label
            combined[f"confidence_{name}"] = confidence
            combined[f"answer_status_{name}"] = status
            combined[f"answer_coverage_{name}"] = round(coverage, 6)
            combined[f"answer_{name}"] = source["recovered_answer_json"]
            combined[f"notes_{name}"] = source["notes"]
        combined["label_agreement"] = combined["label_A"] == combined["label_B"]
        combined["both_exact"] = combined["answer_status_A"] == "EXACT" and combined["answer_status_B"] == "EXACT"
        combined["needs_adjudication"] = (
            not combined["label_agreement"]
            or combined["answer_status_A"] != combined["answer_status_B"]
            or "PARTIAL_MATCH" in {combined["answer_status_A"], combined["answer_status_B"]}
            or "INVALID_JSON" in {combined["answer_status_A"], combined["answer_status_B"]}
        )
        scored.append(combined)

    invalid_fields = ["item_id", "annotator", "problems", "json_error"]
    write_csv(output / "validation_issues.csv", invalid, invalid_fields)
    if invalid and not args.allow_incomplete:
        raise SystemExit(f"Found {len(invalid)} incomplete/invalid annotations; see validation_issues.csv")

    completed = [row for row in scored if row["label_A"] in LABELS and row["label_B"] in LABELS]
    summary: dict[str, Any] = {"n_expected": len(scored), "n_completed_pair": len(completed), "n_validation_issues": len(invalid)}
    if completed:
        agreement, kappa = cohen_kappa([row["label_A"] for row in completed], [row["label_B"] for row in completed])
        exact_a = [row["answer_status_A"] == "EXACT" for row in completed]
        exact_b = [row["answer_status_B"] == "EXACT" for row in completed]
        exact_agreement, exact_kappa = cohen_kappa(
            ["FULL" if value else "NONE" for value in exact_a],
            ["FULL" if value else "NONE" for value in exact_b],
        )
        a_only = sum(a and not b for a, b in zip(exact_a, exact_b))
        b_only = sum(b and not a for a, b in zip(exact_a, exact_b))
        summary.update({
            "raw_label_agreement": agreement,
            "cohen_kappa": kappa,
            "objective_exact_agreement": exact_agreement,
            "objective_exact_cohen_kappa": exact_kappa,
            "annotator_A_exact": sum(exact_a),
            "annotator_B_exact": sum(exact_b),
            "pooled_exact_trials": sum(exact_a) + sum(exact_b),
            "pooled_trials": 2 * len(completed),
            "pooled_exact_rate": (sum(exact_a) + sum(exact_b)) / (2 * len(completed)),
            "objective_exact_A_only": a_only,
            "objective_exact_B_only": b_only,
            "mcnemar_exact_two_sided_p": exact_binomial_two_sided(a_only, a_only + b_only),
        })
        rates = []
        for annotator in ("A", "B"):
            for severity in (*SEVERITIES, "all"):
                subset = completed if severity == "all" else [row for row in completed if row["severity"] == severity]
                for metric, success in (
                    ("subjective_FULL", sum(row[f"label_{annotator}"] == "FULL" for row in subset)),
                    ("subjective_FULL_or_PARTIAL", sum(row[f"label_{annotator}"] in {"FULL", "PARTIAL"} for row in subset)),
                    ("objective_exact_answer", sum(row[f"answer_status_{annotator}"] == "EXACT" for row in subset)),
                ):
                    lo, hi = wilson(success, len(subset))
                    rates.append({"annotator": annotator, "severity": severity, "metric": metric, "success": success, "n": len(subset), "rate": success / len(subset) if subset else "", "ci95_low": lo, "ci95_high": hi})
        write_csv(output / "rates_by_severity.csv", rates, ["annotator", "severity", "metric", "success", "n", "rate", "ci95_low", "ci95_high"])
        clustered_rates = clustered_exact_rates(completed)
        write_csv(
            output / "objective_exact_cluster_bootstrap.csv",
            clustered_rates,
            [
                "annotator", "severity", "n_source_anchors", "rate",
                "cluster_bootstrap_ci95_low", "cluster_bootstrap_ci95_high",
                "bootstrap_repetitions", "seed",
            ],
        )
        pooled_all = next(
            row for row in clustered_rates if row["annotator"] == "pooled" and row["severity"] == "all"
        )
        summary.update({
            "pooled_exact_cluster_bootstrap_ci95_low": pooled_all["cluster_bootstrap_ci95_low"],
            "pooled_exact_cluster_bootstrap_ci95_high": pooled_all["cluster_bootstrap_ci95_high"],
            "cluster_bootstrap_n_source_anchors": pooled_all["n_source_anchors"],
            "cluster_bootstrap_repetitions": pooled_all["bootstrap_repetitions"],
            "cluster_bootstrap_seed": pooled_all["seed"],
        })

    scored_fields = list(scored[0]) if scored else ["item_id"]
    write_csv(output / "paired_scored_annotations.csv", scored, scored_fields)
    adjudication = [row for row in scored if row["needs_adjudication"]]
    exact_disagreements = [
        row for row in scored
        if (row["answer_status_A"] == "EXACT") != (row["answer_status_B"] == "EXACT")
    ]
    write_csv(output / "objective_exact_disagreements.csv", exact_disagreements, scored_fields)
    for row in adjudication:
        row.update({"adjudicated_label": "", "adjudicated_answer_json": "", "adjudicator_notes": ""})
    adjudication_fields = list(adjudication[0]) if adjudication else scored_fields + ["adjudicated_label", "adjudicated_answer_json", "adjudicator_notes"]
    write_csv(output / "adjudication_queue.csv", adjudication, adjudication_fields)
    summary["n_adjudication_queue"] = len(adjudication)
    summary["n_objective_exact_disagreements"] = len(exact_disagreements)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
