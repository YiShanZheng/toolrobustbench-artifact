"""Experiment runner for ToolRobust v3."""

from __future__ import annotations

import ast
import shutil
import time
from pathlib import Path

import pandas as pd

from toolrobust.backends.base import BackendBase
from toolrobust.backends.mock_backend import MockBackend
from toolrobust.datasets.builders import DatasetBuilder
from toolrobust.domain.enums import OutputProfile, SEVERITY_ORDER
from toolrobust.domain.schemas import BackendConfig, BackendResponse, BenchmarkV3Config, EvaluationRecord, ParsedInvocation, RunManifest, TaskSample
from toolrobust.evaluation.metrics import BenchmarkMetricCalculator, write_tables
from toolrobust.evaluation.failure_profiles import family_failure_profiles_payload, stage_failure_profiles_payload
from toolrobust.evaluation.scorer import Scorer
from toolrobust.exports.llamafactory_sharegpt import export_samples_to_sharegpt
from toolrobust.perturbation.pipeline import PerturbationPipeline
from toolrobust.reports.markdown import MarkdownReportGenerator, write_manifest
from toolrobust.reports.output_paths import RunOutputPaths, build_run_output_paths
from toolrobust.reports.plotting import PlotManager


class ExperimentRunner:
    def __init__(self, config: BenchmarkV3Config, backend_configs: dict[str, BackendConfig], backend_map: dict[str, str], output_profile: OutputProfile | None = None, clean_output: bool | None = None) -> None:
        self.config = config
        self.backend_configs = backend_configs
        self.backend_map = backend_map
        self.output_profile = output_profile or config.output.profile
        self.clean_output = config.output.clean_output if clean_output is None else clean_output
        self.pipeline = PerturbationPipeline(config)
        self.scorer = Scorer()
        self.output_paths = self._build_output_paths()
        self.manifest = self._build_manifest()

    def build_data(self, progress: object | None = None) -> tuple[list[TaskSample], list[TaskSample]]:
        return DatasetBuilder(self.config).build(progress=progress)

    def run(
        self,
        progress: object | None = None,
        *,
        resume: bool = False,
        results_dir: Path | None = None,
        canary_subtypes: set[str] | None = None,
        canary_samples_per_subtype_severity: int | None = None,
        stage2_compat_guard_enabled: bool = True,
        stage2_compat_guard_min_issues: int = 3,
        stage2_compat_guard_min_checked: int = 8,
        stage2_compat_guard_failure_rate: float = 0.5,
    ) -> dict[str, Path]:
        self.output_paths = self._build_output_paths(explicit_results_dir=results_dir)
        self.manifest = self._build_manifest()
        clean_tasks, perturbed_tasks = self.build_data(progress=progress)
        if canary_subtypes:
            clean_tasks, perturbed_tasks = self._select_canary_tasks(
                clean_tasks,
                perturbed_tasks,
                canary_subtypes=canary_subtypes,
                samples_per_subtype_severity=canary_samples_per_subtype_severity or 3,
            )
        if self.clean_output and not resume and self.output_paths.results_dir.exists():
            shutil.rmtree(self.output_paths.results_dir)
        self.output_paths.results_dir.mkdir(parents=True, exist_ok=True)
        if not resume and self.output_paths.records_csv.exists():
            self.output_paths.records_csv.unlink()
        existing_records = self._load_existing_records() if resume else pd.DataFrame()
        completed_keys = self._completed_record_keys(existing_records)
        new_records = self._run_tasks(
            clean_tasks + perturbed_tasks,
            progress=progress,
            completed_keys=completed_keys,
            append_records_path=self.output_paths.records_csv,
            stage2_compat_guard_enabled=stage2_compat_guard_enabled,
            stage2_compat_guard_min_issues=stage2_compat_guard_min_issues,
            stage2_compat_guard_min_checked=stage2_compat_guard_min_checked,
            stage2_compat_guard_failure_rate=stage2_compat_guard_failure_rate,
        )
        records_frame = self._merge_records(existing_records, new_records)
        outputs = self._write_outputs(records_frame, progress=progress)
        export_samples_to_sharegpt(clean_tasks + perturbed_tasks, self.config.export_path)
        outputs["llamafactory_export"] = self.config.export_path
        return outputs

    @staticmethod
    def _select_canary_tasks(
        clean_tasks: list[TaskSample],
        perturbed_tasks: list[TaskSample],
        *,
        canary_subtypes: set[str],
        samples_per_subtype_severity: int,
    ) -> tuple[list[TaskSample], list[TaskSample]]:
        selected: list[TaskSample] = []
        counts: dict[tuple[str, str], int] = {}
        limit = max(1, samples_per_subtype_severity)
        for task in perturbed_tasks:
            if task.subtype not in canary_subtypes:
                continue
            key = (task.subtype, task.severity.value)
            if counts.get(key, 0) >= limit:
                continue
            selected.append(task)
            counts[key] = counts.get(key, 0) + 1
        selected_sources = {str(task.metadata.get("source_sample_id", task.sample_id)) for task in selected}
        selected_clean = [task for task in clean_tasks if task.sample_id in selected_sources]
        return selected_clean, selected

    def plot_existing_results(self, *, results_dir: Path | None = None, overwrite: bool = True, output_profile: OutputProfile | None = None, progress: object | None = None) -> dict[str, Path]:
        paths = self._build_output_paths(explicit_results_dir=results_dir)
        plots = PlotManager(paths.results_dir, paths.plots_dir).generate_all(profile=output_profile or self.output_profile, overwrite=overwrite, progress=progress)
        return {"results_dir": paths.results_dir, **plots}

    def _run_tasks(
        self,
        tasks: list[TaskSample],
        progress: object | None = None,
        *,
        completed_keys: set[tuple[str, str]] | None = None,
        append_records_path: Path | None = None,
        stage2_compat_guard_enabled: bool = True,
        stage2_compat_guard_min_issues: int = 3,
        stage2_compat_guard_min_checked: int = 8,
        stage2_compat_guard_failure_rate: float = 0.5,
    ) -> list[EvaluationRecord]:
        registry_index = self.pipeline.registry_index()
        records: list[EvaluationRecord] = []
        completed_keys = completed_keys or set()
        header_written = bool(append_records_path and append_records_path.exists() and append_records_path.stat().st_size > 0)
        total_catalog_items = len(self.backend_map) * len(tasks)
        total_items = sum(1 for model_name in self.backend_map for task in tasks if (model_name, task.sample_id) not in completed_keys)
        compat_guard_state: dict[str, dict[str, int]] = {}
        if progress is not None:
            resume_note = f"resume loaded {len(completed_keys)} completed records; remaining {total_items}/{total_catalog_items}" if completed_keys else "initializing"
            progress.start("running_inference", total_items=total_items, current_item=resume_note)
        for model_name, backend_name in self.backend_map.items():
            backend = self._build_backend(model_name, backend_name)
            for task in tasks:
                if (model_name, task.sample_id) in completed_keys:
                    continue
                registry = registry_index[self.pipeline.registry_key_for_sample(task)]
                started_at = time.perf_counter()
                response = self._predict_with_retries(backend, task, registry.list_tools())
                if progress is not None and hasattr(progress, "record_api_call"):
                    progress.record_api_call(time.perf_counter() - started_at)
                record = self.scorer.score(task, response, registry)
                records.append(record)
                if append_records_path is not None:
                    self._append_record(record, append_records_path, write_header=not header_written)
                    header_written = True
                if stage2_compat_guard_enabled:
                    self._check_stage2_compatibility_guard(
                        model_name=model_name,
                        task=task,
                        response=response,
                        state=compat_guard_state,
                        min_issues=stage2_compat_guard_min_issues,
                        min_checked=stage2_compat_guard_min_checked,
                        failure_rate=stage2_compat_guard_failure_rate,
                    )
                if progress is not None:
                    progress.advance(current_item=f"{model_name}:{task.sample_id}")
        if progress is not None:
            progress.finish_stage("[progress] completed stage=running_inference")
        return records

    @classmethod
    def _check_stage2_compatibility_guard(
        cls,
        *,
        model_name: str,
        task: TaskSample,
        response: BackendResponse,
        state: dict[str, dict[str, int]],
        min_issues: int,
        min_checked: int,
        failure_rate: float,
    ) -> None:
        if not cls._is_stage2_compatibility_guard_candidate(model_name, task, response):
            return
        bucket = state.setdefault(model_name, {"checked": 0, "issues": 0, "consecutive_issues": 0})
        bucket["checked"] += 1
        issue = cls._is_stage2_compatibility_issue(response)
        if issue:
            bucket["issues"] += 1
            bucket["consecutive_issues"] += 1
        else:
            bucket["consecutive_issues"] = 0
        issue_rate = bucket["issues"] / max(1, bucket["checked"])
        consecutive_trip = bucket["consecutive_issues"] >= min_issues
        rate_trip = bucket["checked"] >= min_checked and bucket["issues"] >= min_issues and issue_rate >= failure_rate
        if not consecutive_trip and not rate_trip:
            return
        parser_message = cls._stage2_parser_message(response)
        raise RuntimeError(
            "Stage-2 compatibility guard tripped for "
            f"{model_name} on {task.family}/{task.sample_id}: "
            f"{bucket['issues']}/{bucket['checked']} eligible stage-2 records show empty/non-JSON stage-2 output "
            f"(consecutive={bucket['consecutive_issues']}, last_parser_message={parser_message!r}). "
            "This matches the stage-2 protocol compatibility failure pattern. "
            "The run was stopped before spending more API calls; ordinary tool-selection, parameter, or runtime failures do not trigger this guard."
        )

    @classmethod
    def _is_stage2_compatibility_guard_candidate(cls, model_name: str, task: TaskSample, response: BackendResponse) -> bool:
        _ = model_name
        if not (task.has_family("observation") or task.has_family("execution_environment")):
            return False
        auxiliary = response.auxiliary or {}
        if "runtime_execution_success" not in auxiliary:
            return False
        if auxiliary.get("stage1_predicted_tool_resolved") != task.canonical_tool_id:
            return False
        normalized_args = auxiliary.get("stage1_normalized_arguments")
        if not isinstance(normalized_args, dict):
            return False
        return all(normalized_args.get(key) == value for key, value in task.expected_arguments.items())

    @staticmethod
    def _is_stage2_compatibility_issue(response: BackendResponse) -> bool:
        return ExperimentRunner._stage2_parser_message(response) in {
            "empty observation result content",
            "observation content is not valid json object",
            "empty execution decision content",
            "execution decision is not valid json object",
        }

    @staticmethod
    def _stage2_parser_message(response: BackendResponse) -> str:
        auxiliary = response.auxiliary or {}
        return str(
            auxiliary.get("observation_stage2_parser_message")
            or auxiliary.get("execution_stage2_parser_message")
            or ""
        )

    def _predict_with_retries(self, backend: BackendBase, task: TaskSample, tool_registry: list) -> BackendResponse:
        attempts = max(1, int(self.config.execution.retry_times) + 1)
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return backend.predict(task, tool_registry)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= attempts:
                    break
                time.sleep(min(2.0 * attempt, 5.0))
        assert last_error is not None
        parser_message = (
            f"backend request failed after {attempts} attempt(s): "
            f"{type(last_error).__name__}: {last_error}"
        )
        raw_response = {
            "backend_error": {
                "type": type(last_error).__name__,
                "message": str(last_error),
                "attempts": attempts,
            }
        }
        auxiliary = {
            "backend_exception_type": type(last_error).__name__,
            "backend_exception_message": str(last_error),
            "backend_retry_attempts": attempts,
        }
        if hasattr(backend, "default_parsed"):
            return backend.default_parsed(
                raw_response=raw_response,
                call_present=False,
                format_valid=False,
                parser_message=parser_message,
                auxiliary=auxiliary,
            )
        return BackendResponse(
            backend_name=getattr(backend, "name", "backend"),
            model_name=getattr(getattr(backend, "config", None), "model_name", "unknown-model"),
            raw_response=raw_response,
            auxiliary=auxiliary,
            parsed=ParsedInvocation(
                predicted_tool_raw=None,
                predicted_args_raw={},
                call_present=False,
                format_valid=False,
                parser_message=parser_message,
            ),
        )

    def _write_outputs(self, records_frame: pd.DataFrame, progress: object | None = None) -> dict[str, Path]:
        self.output_paths.results_dir.mkdir(parents=True, exist_ok=True)
        if progress is not None:
            progress.start("aggregating_metrics", total_items=4, current_item=self.output_paths.records_csv.name)
        records_frame.to_csv(self.output_paths.records_csv, index=False)
        if progress is not None:
            progress.advance(current_item="summary_tables")
        tables = BenchmarkMetricCalculator(self.manifest, keep_missing_buckets=self.config.output.keep_missing_buckets).calculate(records_frame)
        self.pipeline.preflight_gate_report_frame().to_csv(self.output_paths.preflight_severity_gate_report_csv, index=False)
        if progress is not None:
            progress.advance(current_item="run_manifest.json")
        written = write_tables(self.output_paths.results_dir, tables)
        write_manifest(self.output_paths.run_manifest_json, self.manifest.model_dump(mode="json"))
        if progress is not None:
            progress.advance(current_item="summary.md")
        MarkdownReportGenerator().generate(self.output_paths.summary_md, manifest=self.manifest, tables=tables)
        if progress is not None:
            progress.advance(current_item="aggregation_complete")
            progress.finish_stage("[progress] completed stage=aggregating_metrics")
        plots: dict[str, Path] = {}
        if progress is not None:
            progress.start("writing_outputs", total_items=2, current_item=self.output_paths.summary_overall_csv.name)
            progress.advance(current_item=self.output_paths.run_manifest_json.name)
            progress.advance(current_item=self.output_paths.summary_md.name)
            progress.finish_stage("[progress] completed stage=writing_outputs")
        if self.config.output.generate_plots:
            plots = PlotManager(self.output_paths.results_dir, self.output_paths.plots_dir).generate_all(profile=self.output_profile, overwrite=True, progress=progress, include_loading_stage=False)
        return {
            "results_dir": self.output_paths.results_dir,
            "records": self.output_paths.records_csv,
            "summary_overall": self.output_paths.summary_overall_csv,
            "summary_by_family": self.output_paths.summary_by_family_csv,
            "summary_by_subtype": self.output_paths.summary_by_subtype_csv,
            "summary_by_stage": self.output_paths.summary_by_stage_csv,
            "summary_by_operator": self.output_paths.summary_by_operator_csv,
            "summary_by_composition": self.output_paths.summary_by_composition_csv,
            "interaction_effect": self.output_paths.interaction_effect_csv,
            "preflight_severity_gate_report": self.output_paths.preflight_severity_gate_report_csv,
            "severity_calibration_report": self.output_paths.severity_calibration_report_csv,
            "failure_taxonomy": self.output_paths.failure_taxonomy_csv,
            "primary_failure_breakdown": self.output_paths.primary_failure_breakdown_csv,
            "cascade_failure_breakdown": self.output_paths.cascade_failure_breakdown_csv,
            "failure_by_family": self.output_paths.failure_by_family_csv,
            "failure_by_composition": self.output_paths.failure_by_composition_csv,
            "paired_case_dump": self.output_paths.paired_case_dump_csv,
            "family_boundary_report": self.output_paths.family_boundary_report_csv,
            "run_manifest": self.output_paths.run_manifest_json,
            "summary_md": self.output_paths.summary_md,
            **written,
            **plots,
        }

    def _load_existing_records(self) -> pd.DataFrame:
        if not self.output_paths.records_csv.exists() or self.output_paths.records_csv.stat().st_size == 0:
            return pd.DataFrame()
        frame = pd.read_csv(self.output_paths.records_csv)
        for column in (
            "operator_names",
            "families_applied",
            "subtypes_applied",
            "stage_targets_applied",
            "expected_arguments",
            "predicted_arguments",
            "expected_result",
        ):
            if column in frame.columns:
                frame[column] = frame[column].apply(self._deserialize_structured_cell)
        return frame

    @staticmethod
    def _completed_record_keys(records_frame: pd.DataFrame) -> set[tuple[str, str]]:
        if records_frame.empty:
            return set()
        return {
            (str(row.model_name), str(row.sample_id))
            for row in records_frame.loc[:, ["model_name", "sample_id"]].itertuples(index=False)
        }

    @staticmethod
    def _append_record(record: EvaluationRecord, output_path: Path, *, write_header: bool) -> None:
        pd.DataFrame([record.model_dump(mode="json")]).to_csv(output_path, mode="a", header=write_header, index=False)

    @staticmethod
    def _merge_records(existing_records: pd.DataFrame, new_records: list[EvaluationRecord]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        if not existing_records.empty:
            frames.append(existing_records)
        if new_records:
            frames.append(pd.DataFrame([record.model_dump(mode="json") for record in new_records]))
        if not frames:
            return pd.DataFrame()
        merged = pd.concat(frames, ignore_index=True)
        return merged.drop_duplicates(subset=["model_name", "sample_id"], keep="last")

    @staticmethod
    def _deserialize_structured_cell(value: object) -> object:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped or stripped.lower() == "nan":
            return None
        if stripped[0] not in "[{(":
            return value
        return ast.literal_eval(stripped)

    def _build_output_paths(self, explicit_results_dir: Path | None = None) -> RunOutputPaths:
        return build_run_output_paths(self.config.output_root, experiment_name=self.config.resolved_experiment_name(), explicit_results_dir=explicit_results_dir)

    def _build_backend(self, model_name: str, backend_name: str) -> BackendBase:
        backend_config = self.backend_configs[model_name]
        if backend_name == "mock":
            return MockBackend(backend_config)
        if backend_name == "openai" or backend_name.startswith("deepseek"):
            from toolrobust.backends.openai_chat import OpenAIChatBackend

            return OpenAIChatBackend(backend_config)
        if backend_name == "llamafactory_openai_compat":
            from toolrobust.backends.llamafactory_openai_compat import LlamaFactoryOpenAICompatBackend

            return LlamaFactoryOpenAICompatBackend(backend_config)
        raise ValueError(f"Unsupported backend: {backend_name}")

    def _build_manifest(self) -> RunManifest:
        sampled_tool_ids = DatasetBuilder(self.config).selected_tool_ids()
        return RunManifest(
            benchmark_version=self.config.benchmark_version,
            operator_catalog_version=self.pipeline.operator_catalog_version,
            experiment_name=self.config.resolved_experiment_name(),
            experiment_description=self.config.experiment.description,
            models=list(self.backend_map),
            dataset={
                "source": self.config.dataset.source,
                "task_scope": self.config.dataset.task_scope,
                "max_clean_per_tool": self.config.dataset.max_clean_per_tool,
                "sample_tool_count": self.config.dataset.sample_tool_count,
                "sampled_tool_ids": sampled_tool_ids,
            },
            enabled_families=self.pipeline.enabled_families(),
            enabled_subtypes=self.config.noise.enabled_subtypes,
            disabled_operators=list(self.config.noise.disabled_operators),
            severities=[severity.value for severity in self.config.noise.severities],
            severity_policy=self.config.noise.severity_policy,
            allow_cross_family_heavy=self.config.noise.allow_cross_family_heavy,
            cross_family={
                "enabled": self.config.cross_family.enabled,
                "strategy": self.config.cross_family.strategy,
                "family_sets": self.config.cross_family.family_sets,
                "severities": [severity.value for severity in self.config.cross_family.severities],
                "samples_per_combo_bucket": self.config.cross_family.samples_per_combo_bucket,
                "max_families_per_sample": self.config.cross_family.max_families_per_sample,
                "include_single_family_baseline": self.config.cross_family.include_single_family_baseline,
                "enabled_compositions": self.config.cross_family.enabled_compositions,
                "composition_order": self.pipeline.composition_order(),
                "composite_stage_mapping": self.pipeline.composite_stage_mapping(),
            },
            max_operators_per_sample=self.config.noise.max_operators_per_sample,
            samples_per_bucket=self.config.noise.samples_per_bucket,
            seed=self.config.dataset.seed,
            output_profile=self.output_profile.value,
            backend_map=self.backend_map,
            severity_order=self.config.output.severity_order or SEVERITY_ORDER,
            family_order=self.pipeline.family_order(),
            subtype_order=self.pipeline.subtype_order(),
            subtype_family_mapping=self.pipeline.subtype_family_mapping(),
            stage_order=self.pipeline.stage_order(),
            stage_mapping=self.pipeline.stage_mapping(),
            family_failure_profiles=family_failure_profiles_payload(),
            stage_failure_profiles=stage_failure_profiles_payload(),
            operator_primary_failure_risks=self.pipeline.operator_failure_risk_mapping(),
            operator_metadata=self.pipeline.operator_metadata_mapping(),
            preflight_gate_rows=self.pipeline.preflight_gate_rows(),
        )
