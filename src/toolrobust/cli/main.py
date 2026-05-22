"""CLI entrypoint for ToolRobust v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from toolrobust.cli.progress import build_progress_reporter
from toolrobust.domain.enums import OutputProfile
from toolrobust.domain.schemas import BackendConfig, BenchmarkV3Config
from toolrobust.runner.experiment_runner import ExperimentRunner


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_configs(config_path: Path, models_path: Path) -> tuple[BenchmarkV3Config, dict[str, BackendConfig], dict[str, str]]:
    config = BenchmarkV3Config.model_validate(load_yaml(config_path))
    models_raw = load_yaml(models_path)
    backends = models_raw.get("backends", {})
    backend_configs: dict[str, BackendConfig] = {}
    backend_map: dict[str, str] = {}
    for model_name in config.models.include:
        matched_backend = None
        for backend_name, backend_payload in backends.items():
            if backend_payload.get("model_name") == model_name:
                payload = dict(backend_payload)
                payload.setdefault("request_timeout_sec", config.execution.timeout_sec)
                payload.setdefault("request_max_retries", config.execution.retry_times)
                backend_configs[model_name] = BackendConfig.model_validate(payload)
                matched_backend = backend_name
                break
        if matched_backend is None:
            default_backend = models_raw.get("default_backend", "mock")
            payload = dict(backends[default_backend])
            payload["model_name"] = model_name
            payload.setdefault("request_timeout_sec", config.execution.timeout_sec)
            payload.setdefault("request_max_retries", config.execution.retry_times)
            backend_configs[model_name] = BackendConfig.model_validate(payload)
            matched_backend = default_backend
        backend_map[model_name] = matched_backend
    return config, backend_configs, backend_map


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ToolRobust v3 CLI")
    parser.add_argument("--config", default="configs/benchmark_v3.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--experiment-description", default=None)
    _add_progress_args(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_data_parser = subparsers.add_parser("build-data", help="Build ToolRobust v3 datasets")
    _add_progress_args(build_data_parser)

    run_parser = subparsers.add_parser("run", help="Run ToolRobust v3")
    _add_progress_args(run_parser)
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--results-dir", default=None)
    run_parser.add_argument("--canary", action="store_true", help="Run a small D2/D3 calibration canary instead of the full task set")
    run_parser.add_argument("--canary-subtypes", default="D2 Runtime Variability,D3 Failure Feedback Quality")
    run_parser.add_argument("--canary-samples-per-severity", type=int, default=3)
    run_parser.add_argument("--disable-stage2-compat-guard", action="store_true", help="Disable C/D stage-2 empty/non-JSON compatibility guard")
    run_parser.add_argument("--stage2-compat-guard-min-issues", type=int, default=3)
    run_parser.add_argument("--stage2-compat-guard-min-checked", type=int, default=8)
    run_parser.add_argument("--stage2-compat-guard-failure-rate", type=float, default=0.5)

    plot_parser = subparsers.add_parser("plot", help="Regenerate plots from an existing v3 result directory")
    _add_progress_args(plot_parser)
    plot_parser.add_argument("--results-dir", default=None)
    plot_parser.add_argument("--overwrite", action="store_true")
    return parser


def _add_progress_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--progress-interval", type=float, default=5.0)
    parser.add_argument("--progress-label", default=None)
    parser.add_argument("--progress-offset", type=int, default=0)
    parser.add_argument("--progress-total", type=int, default=0)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    progress = build_progress_reporter(
        command_name=args.command,
        enabled=args.progress,
        interval_seconds=args.progress_interval,
        progress_label=args.progress_label,
        progress_offset=args.progress_offset,
        progress_total=args.progress_total,
    )
    config, backend_configs, backend_map = resolve_configs(Path(args.config), Path(args.models_config))
    if args.experiment_name is not None:
        config.experiment.name = args.experiment_name
    if args.experiment_description is not None:
        config.experiment.description = args.experiment_description
    runner = ExperimentRunner(config, backend_configs, backend_map, output_profile=OutputProfile(config.output.profile), clean_output=config.output.clean_output)

    if args.command == "build-data":
        clean, perturbed = runner.build_data(progress=progress)
        progress.finish_all("[progress] done")
        print(json.dumps({"clean_samples": len(clean), "perturbed_samples": len(perturbed)}, ensure_ascii=False))
        return
    if args.command == "run":
        outputs = runner.run(
            progress=progress,
            resume=args.resume,
            results_dir=Path(args.results_dir) if args.results_dir else None,
            canary_subtypes={item.strip() for item in args.canary_subtypes.split(",") if item.strip()} if args.canary else None,
            canary_samples_per_subtype_severity=args.canary_samples_per_severity if args.canary else None,
            stage2_compat_guard_enabled=not args.disable_stage2_compat_guard,
            stage2_compat_guard_min_issues=args.stage2_compat_guard_min_issues,
            stage2_compat_guard_min_checked=args.stage2_compat_guard_min_checked,
            stage2_compat_guard_failure_rate=args.stage2_compat_guard_failure_rate,
        )
        progress.finish_all("[progress] done")
        print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False))
        return
    if args.command == "plot":
        outputs = runner.plot_existing_results(results_dir=Path(args.results_dir) if args.results_dir else None, overwrite=args.overwrite, output_profile=OutputProfile(config.output.profile), progress=progress)
        progress.finish_all("[progress] done")
        print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False))
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
