"""Write a concrete per-model paper run config from a base YAML file."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--families", default="")
    parser.add_argument("--family-set", default="")
    parser.add_argument("--composition-ids", default="")
    parser.add_argument("--cross-family", action="store_true")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    payload = _load_yaml(args.base)
    payload.setdefault("experiment", {})["name"] = args.experiment_name
    payload.setdefault("models", {})["include"] = [args.model]

    if args.cross_family:
        family_set = _split_csv(args.family_set)
        payload.setdefault("noise", {})["enabled_families"] = [
            "interface_exposure",
            "user_intent",
            "observation",
            "execution_environment",
        ]
        cross_family = payload.setdefault("cross_family", {})
        cross_family["enabled"] = True
        if family_set:
            if len(family_set) != 2:
                raise ValueError("--family-set must contain exactly two comma-separated families")
            cross_family["family_sets"] = [family_set]
        composition_ids = _split_csv(args.composition_ids)
        if composition_ids:
            cross_family["enabled_compositions"] = composition_ids
        cross_family["include_single_family_baseline"] = False
    else:
        families = _split_csv(args.families)
        if len(families) != 1:
            raise ValueError("--families must contain exactly one family for all-family split runs")
        payload.setdefault("noise", {})["enabled_families"] = families
        payload.setdefault("cross_family", {})["enabled"] = False

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


if __name__ == "__main__":
    main()
