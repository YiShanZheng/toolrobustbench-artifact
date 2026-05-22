"""Report contaminated benchmark records under a results directory.

The report focuses on failures that should not be treated as model-capability
errors:
- API call failures: backend request failed before a usable model response.
- JSON compatibility failures: stage-2 C/D output was empty or not parseable.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


JSON_COMPAT_PATTERNS = {
    "empty observation result content": re.compile(r"empty observation result content", re.I),
    "observation content is not valid json object": re.compile(r"observation content is not valid json object", re.I),
    "empty execution decision content": re.compile(r"empty execution decision content", re.I),
    "execution decision is not valid json object": re.compile(r"execution decision is not valid json object", re.I),
    "finish_reason=length": re.compile(r"finish_reason\s*=\s*length", re.I),
}

API_FAILURE_PATTERNS = {
    "backend request failed": re.compile(r"backend request failed", re.I),
    "rate limit / quota": re.compile(
        r"RateLimitError|429|rate limit|too many requests|quota|insufficient balance|resource has been exhausted",
        re.I,
    ),
    "connection / server error": re.compile(
        r"APIConnectionError|APITimeoutError|InternalServerError|ServiceUnavailable|BadGateway|GatewayTimeout",
        re.I,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="data/results/all-family",
        type=Path,
        help="Result folder to scan. The script searches recursively for records.csv.",
    )
    parser.add_argument("--records-name", default="records.csv")
    parser.add_argument("--format", choices=("table", "csv", "json"), default="table")
    parser.add_argument("--include-zero", action="store_true", help="Include model/family files with no contamination.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output file for csv/json/table text.")
    return parser.parse_args()


def infer_model_family(records_csv: Path, root: Path) -> tuple[str, str]:
    full_parts = records_csv.parts
    for marker in ("all-family", "cross-family"):
        if marker in full_parts:
            index = full_parts.index(marker)
            if index + 2 < len(full_parts):
                return full_parts[index + 1], full_parts[index + 2]
    try:
        relative = records_csv.relative_to(root)
    except ValueError:
        relative = records_csv
    parts = relative.parts
    if len(parts) >= 3:
        model = parts[-3]
        family = parts[-2]
    elif records_csv.parent.parent != records_csv.parent:
        model = records_csv.parent.parent.name
        family = records_csv.parent.name
    else:
        model = parts[0] if len(parts) >= 1 else ""
        family = parts[1] if len(parts) >= 2 else ""
    return model, family


def matched_reasons(notes: str, patterns: dict[str, re.Pattern[str]]) -> list[str]:
    return [name for name, pattern in patterns.items() if pattern.search(notes)]


def summarize_file(records_csv: Path, root: Path) -> list[dict[str, object]]:
    model, family = infer_model_family(records_csv, root)
    with records_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    total = len(rows)
    buckets: dict[str, dict[str, object]] = {
        "api未成功调用": {"count": 0, "reasons": {}},
        "json兼容": {"count": 0, "reasons": {}},
    }
    for row in rows:
        notes = str(row.get("notes") or "")
        api_reasons = matched_reasons(notes, API_FAILURE_PATTERNS)
        json_reasons = matched_reasons(notes, JSON_COMPAT_PATTERNS)
        if api_reasons:
            add_bucket(buckets["api未成功调用"], api_reasons)
        if json_reasons:
            add_bucket(buckets["json兼容"], json_reasons)
    records: list[dict[str, object]] = []
    for issue_type, bucket in buckets.items():
        count = int(bucket["count"])
        if count == 0:
            continue
        records.append(
            {
                "model": model,
                "family": family,
                "total": total,
                "contaminated": count,
                "percent": round((count / total * 100) if total else 0.0, 2),
                "issue_type": issue_type,
                "reasons": format_reasons(bucket["reasons"]),
                "records_csv": str(records_csv),
            }
        )
    if not records:
        records.append(
            {
                "model": model,
                "family": family,
                "total": total,
                "contaminated": 0,
                "percent": 0.0,
                "issue_type": "无污染",
                "reasons": "",
                "records_csv": str(records_csv),
            }
        )
    return records


def add_bucket(bucket: dict[str, object], reasons: list[str]) -> None:
    bucket["count"] = int(bucket["count"]) + 1
    reason_counts = bucket["reasons"]
    assert isinstance(reason_counts, dict)
    for reason in reasons:
        reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1


def format_reasons(reasons: object) -> str:
    if not isinstance(reasons, dict):
        return ""
    return "; ".join(f"{reason}={count}" for reason, count in sorted(reasons.items()))


def render_table(records: list[dict[str, object]]) -> str:
    headers = ["model", "family", "total", "contaminated", "percent", "issue_type", "reasons"]
    widths = {
        header: max(len(header), *(len(str(record[header])) for record in records))
        for header in headers
    }
    lines = []
    lines.append("  ".join(header.ljust(widths[header]) for header in headers))
    lines.append("  ".join("-" * widths[header] for header in headers))
    for record in records:
        lines.append("  ".join(str(record[header]).ljust(widths[header]) for header in headers))
    return "\n".join(lines)


def render_csv(records: list[dict[str, object]]) -> str:
    headers = ["model", "family", "total", "contaminated", "percent", "issue_type", "reasons", "records_csv"]
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=headers, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue()


def main() -> int:
    args = parse_args()
    root = args.results_dir
    if not root.exists():
        raise FileNotFoundError(root)
    records_files = sorted(root.rglob(args.records_name))
    records: list[dict[str, object]] = []
    for records_csv in records_files:
        for record in summarize_file(records_csv, root):
            if args.include_zero or record["contaminated"] != 0:
                records.append(record)
    records.sort(key=lambda item: (-int(item["contaminated"]), str(item["model"]), str(item["family"]), str(item["issue_type"])))
    if args.format == "json":
        output = json.dumps(records, ensure_ascii=False, indent=2)
    elif args.format == "csv":
        output = render_csv(records)
    else:
        output = render_table(records) if records else "No contaminated records found."
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + ("\n" if not output.endswith("\n") else ""), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
