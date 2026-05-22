"""Output profiles for ToolRobust v3."""

from __future__ import annotations

from dataclasses import dataclass

from toolrobust.domain.enums import OutputProfile


@dataclass(frozen=True)
class OutputProfileSpec:
    profile: OutputProfile
    plots: tuple[str, ...]


PROFILE_SPECS = {
    OutputProfile.LITE: OutputProfileSpec(
        profile=OutputProfile.LITE,
        plots=(
            "overall_robustness_curve",
            "family_wise_robustness_bar",
            "subtype_heatmap",
            "failure_taxonomy_stacked_bar",
            "stage_breakdown",
            "cross_pair_success_heatmap",
            "cross_all_compositions_success_matrix",
            "cross_pair_subtype_success_heatmaps",
        ),
    ),
    OutputProfile.FULL: OutputProfileSpec(
        profile=OutputProfile.FULL,
        plots=(
            "overall_robustness_curve",
            "family_wise_robustness_bar",
            "subtype_heatmap",
            "failure_taxonomy_stacked_bar",
            "stage_breakdown",
            "cross_pair_success_heatmap",
            "cross_all_compositions_success_matrix",
            "cross_pair_subtype_success_heatmaps",
        ),
    ),
}


def get_output_profile_spec(profile: OutputProfile) -> OutputProfileSpec:
    return PROFILE_SPECS[profile]
