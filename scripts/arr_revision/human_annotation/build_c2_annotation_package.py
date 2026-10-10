"""Build a blinded, paired C2 human-recoverability annotation package.

The public sheets reproduce the stage-2 information surface used by the
observation benchmark. Gold outputs and original argument values are kept in
``private/`` and must not be shown to annotators.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from toolrobust.execution.runtime_feedback import (  # noqa: E402
    _apply_observation_operator,
    build_observation_target,
)
from toolrobust.tools.executor import ToolExecutor  # noqa: E402


MODELS = (
    "claude-sonnet-4-6",
    "deepseek-v4-pro",
    "gpt-5.4-mini",
    "qwen-plus",
    "gemini-2.5-pro",
    "deepseek-v4-flash",
    "gemini-2.5-flash",
)
SEVERITIES = ("light", "medium", "heavy")
PUBLIC_FIELDS = (
    "annotation_order",
    "item_id",
    "severity",
    "model_visible_prompt",
    "recovered_answer_json",
    "recoverability_label",
    "confidence_1_to_5",
    "evidence_citation",
    "notes",
    "minutes_spent",
)


def parse_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text.lower() == "nan":
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    raise ValueError(f"Cannot parse structured value: {text[:120]!r}")


def complexity_hint(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            "kind": "object",
            "field_count": len(value),
            "contains_list": any(isinstance(item, list) for item in value.values()),
        }
    if isinstance(value, list):
        return {"kind": "list", "length_hint": len(value)}
    return {"kind": type(value).__name__}


def model_visible_prompt(tool_name: str, args: dict[str, Any], feedback: dict[str, Any], target: Any) -> str:
    invocation_context = {
        "tool_name": tool_name,
        "argument_keys": sorted(args),
        "target_complexity": complexity_hint(target),
    }
    prompt_feedback = {
        "execution_success": feedback["execution_success"],
        "status": feedback["status"],
        "content": feedback["content"],
    }
    return (
        "Initial invocation context:\n"
        + json.dumps(invocation_context, ensure_ascii=False, sort_keys=True)
        + "\n\nRuntime feedback:\n"
        + json.dumps(prompt_feedback, ensure_ascii=False, sort_keys=True)
        + "\n\nRecover only the clean observation target from the noisy runtime feedback. "
        "Use the invocation context only to understand the tool family. "
        "Do not reconstruct missing values from the original arguments unless the runtime feedback itself supports them. "
        'Return only JSON and prefer the wrapper {"final_result": ...}. '
        "Do not emit tool_name or arguments."
    )


def load_reference_rows() -> list[dict[str, str]]:
    path = ROOT / "data" / "results" / "all-family" / MODELS[0] / "observation" / "records.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if "C2" in row.get("subtype", "")]
    if len(rows) != 144:
        raise RuntimeError(f"Expected 144 C2 instances, found {len(rows)} in {path}")

    reference_ids = {row["sample_id"] for row in rows}
    for model in MODELS[1:]:
        other = ROOT / "data" / "results" / "all-family" / model / "observation" / "records.csv"
        with other.open(encoding="utf-8-sig", newline="") as handle:
            other_rows = [row for row in csv.DictReader(handle) if "C2" in row.get("subtype", "")]
        other_ids = {row["sample_id"] for row in other_rows}
        if other_ids != reference_ids:
            raise RuntimeError(f"C2 sample universe differs for {model}")
    return rows


def reconstruct(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    executor = ToolExecutor()
    items: list[dict[str, Any]] = []
    for row in rows:
        args = parse_literal(row["expected_arguments"])
        expected_record = parse_literal(row["expected_result"])
        tool = row["gold_canonical_tool_id"]
        clean = executor.execute(tool, args)
        if clean != expected_record:
            raise RuntimeError(f"Executor/record mismatch for {row['sample_id']}: {clean!r} != {expected_record!r}")

        content: Any = clean
        notes: list[str] = []
        operators = [part.strip() for part in row["operator"].split("+") if part.strip()]
        for operator in operators:
            content, note = _apply_observation_operator(tool, args, content, operator)
            notes.append(note)
        feedback = {
            "execution_success": 1,
            "status": "success",
            "content": content,
            "note": "; ".join(notes),
        }
        target = build_observation_target(tool, clean)
        prompt = model_visible_prompt(tool, args, feedback, target)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        items.append(
            {
                "item_id": row["sample_id"],
                "source_sample_id": row["source_sample_id"],
                "severity": row["severity"],
                "operator": row["operator"],
                "gold_tool": tool,
                "expected_arguments": args,
                "clean_result": clean,
                "observation_target": target,
                "runtime_feedback": feedback,
                "model_visible_prompt": prompt,
                "prompt_sha256": digest,
            }
        )
    return items


def choose_paired_anchors(items: list[dict[str, Any]], seed: int, n_anchors: int = 20) -> list[str]:
    light = [item for item in items if item["severity"] == "light"]
    by_tool: dict[str, list[str]] = {}
    for item in light:
        by_tool.setdefault(item["gold_tool"], []).append(item["source_sample_id"])
    if n_anchors < len(by_tool):
        raise ValueError("n_anchors must cover every tool at least once")
    rng = random.Random(seed)
    chosen: list[str] = []
    for tool in sorted(by_tool):
        candidates = sorted(set(by_tool[tool]))
        chosen.append(rng.choice(candidates))
    remaining = sorted({item["source_sample_id"] for item in light} - set(chosen))
    chosen.extend(rng.sample(remaining, n_anchors - len(chosen)))
    return chosen


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...] | list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def public_rows(items: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(shuffled, start=1):
        rows.append(
            {
                "annotation_order": index,
                "item_id": item["item_id"],
                "severity": item["severity"],
                "model_visible_prompt": item["model_visible_prompt"],
                "recovered_answer_json": "",
                "recoverability_label": "",
                "confidence_1_to_5": "",
                "evidence_citation": "",
                "notes": "",
                "minutes_spent": "",
            }
        )
    return rows


def write_readme(output: Path) -> None:
    text = """# C2 人工可恢复性盲标包

## 核心设计

- 主样本为 **20 个 clean anchors × 3 个 severity = 60 项**，三个严重级别使用同一批 anchors，因此可做配对比较。
- 覆盖全部 16 个工具；抽样不看模型成功/失败，避免结果导向抽样。
- 两名标注者使用独立随机顺序，彼此不能讨论。
- 公开表只包含模型在 observation stage 真正可见的信息。`private/` 内有 gold，绝不能发给标注者。
- 旧版表把 Gold 答案放在判断前，不能作为盲标证据；本包已修复。

## 给标注者的操作步骤

1. A 标注者只打开 `annotator_A.csv`；B 标注者只打开 `annotator_B.csv`。
2. 每行只读 `model_visible_prompt`，严格执行其中限制：不得从原始参数值、常识数据库、搜索引擎或其他外部来源补值。
3. 在 `recovered_answer_json` 写你能从可见证据恢复的最佳 JSON。能恢复多少写多少；完全不能恢复时写 `null`。
4. 在 `recoverability_label` 中三选一：
   - `FULL`：可见证据足以恢复你认为完整且精确的目标；
   - `PARTIAL`：只能恢复部分字段/大致方向，缺少精确值；
   - `NONE`：没有足够证据给出目标内容。
5. `confidence_1_to_5` 填 1–5；`evidence_citation` 必须引用 prompt 中支持答案的字段路径或原文片段；可在 `notes` 说明歧义。
6. 不要打开 `private/`、`all_144_private.jsonl`、其他实验输出或论文中的结果表。

## 质量控制与统计口径

- 两人完成后运行：

  `python aggregate_annotations.py --annotator-a annotator_A.csv --annotator-b annotator_B.csv --gold-key private/gold_key_private.csv --output-dir aggregated`

- 论文主结果应优先报告：**盲标答案与 gold 的 exact JSON recovery rate**，并给 Wilson 95% CI。
- `FULL+PARTIAL` 只能作为宽松的主观证据充足率，必须与 exact recovery 分开。
- 报告 Cohen's kappa、原始一致率、分 severity 结果，并对所有分歧/非 JSON/部分匹配项做第三人 adjudication。
- 建议在论文中明确：机器“至少一个模型成功”只是可解性的保守存在性证据，不是人类可恢复率估计。

## 文件

- `annotator_A.csv`, `annotator_B.csv`：60 项主盲标表。
- `primary_items_public.jsonl`：公开主样本的机器可读版本。
- `extension_84_public.csv`：剩余 84 项，可选扩展验证；页数不影响时建议也标。
- `private/gold_key_private.csv`：主样本 gold 与抽样信息，仅供聚合/裁决。
- `private/all_144_private.jsonl`：全部重建结果，用于可复现审计。
- `aggregate_annotations.py`：校验、IAA、置信区间、裁决队列。
- `analyze_c2_proxy.py`：机器代理与敏感性分析。
"""
    (output / "README_CN.md").write_text(text, encoding="utf-8")


def write_audit(output: Path) -> None:
    text = """# C2 既有标注证据审计

## 结论

工作区中**不存在可追溯到 60 个逐项判断的已完成 C2 专项人工标注文件**，因此论文草稿中的 `41/60 (68.3%)` 目前不能作为已验证人工结果继续使用。

## 找到的证据

1. `scripts/rebuttal/output/human_annotation/annotation_A_c2_recoverability.csv`：29 行，四个人工填写列全部为空。
2. 中文同源表 `实验A_C2可恢复性审核_标注表.csv`：29 行，填写列全部为空。
3. 两个 TXT 表同样保留空白下划线，没有逐项作答。
4. `rebuttal_final_en.md`、`rebuttal_final_zh.md`、`paper/update_report.md` 和 TeX 草稿只保存了聚合数字（YES=29、PARTIAL=12、NO=19），没有 60 行原始标签、标注者 ID、时间戳或裁决记录。
5. 中文回复稿在叙述中写“采样30个”，同一段表格却写 `n=60`，内部不一致。
6. 旧生成脚本计划每个 severity 取 10 项，但 heavy 只存在 1 个“至少一个模型成功”实例；其 8 all-fail + 最多 1 success 的逻辑实际只产生 9 个 heavy，总数因此为 29 而非 30。
7. `results/validation/integrated_validation_annotation_human.csv` 有 140 行已填字段，但它是通用构造/评分审核，仅含 10 个 C2 记录、7 个 unique C2 instance ID，且多条 notes 明示为 provisional/无法验证 observation recovery；它不能还原或替代 60 项 C2 recoverability audit。

## 旧协议的效度问题

- 旧完整版在标注前直接展示 Gold 正确答案，产生严重答案泄漏。
- 旧 CSV 没有直接提供实际扰动后返回；要求标注者“想象”删除字段。
- 旧说明把 YES+PARTIAL 合并成 recoverable，但 benchmark 的严格 scorer 要求精确恢复；这会夸大与模型严格成功率的差距。
- 旧抽样按 all-model-fail 状态挑样，会造成 outcome-conditioned sampling，不适合作为总体人类可恢复率。

## 处理建议

- 在新双人盲标和裁决完成前，将 `41/60`、`68.3%`、按 severity 的人工比例视为 **unsupported draft values**。
- 新协议同时报告 objective exact-answer recovery 与 subjective FULL/PARTIAL/NONE；不要再把二者混成一个指标。
"""
    (output / "C2_AUDIT_REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    private = output / "private"
    private.mkdir(parents=True, exist_ok=True)

    items = reconstruct(load_reference_rows())
    anchors = choose_paired_anchors(items, args.seed, 20)
    anchor_set = set(anchors)
    primary = [item for item in items if item["source_sample_id"] in anchor_set]
    extension = [item for item in items if item["source_sample_id"] not in anchor_set]
    if len(primary) != 60 or len(extension) != 84:
        raise RuntimeError(f"Unexpected split: primary={len(primary)}, extension={len(extension)}")

    write_csv(output / "annotator_A.csv", public_rows(primary, args.seed + 101), PUBLIC_FIELDS)
    write_csv(output / "annotator_B.csv", public_rows(primary, args.seed + 202), PUBLIC_FIELDS)
    extension_fields = ("item_id", "severity", "model_visible_prompt", "recovered_answer_json", "recoverability_label", "confidence_1_to_5", "evidence_citation", "notes", "minutes_spent")
    write_csv(output / "extension_84_public.csv", public_rows(extension, args.seed + 303), extension_fields)

    with (output / "primary_items_public.jsonl").open("w", encoding="utf-8") as handle:
        for item in primary:
            handle.write(json.dumps({"item_id": item["item_id"], "severity": item["severity"], "model_visible_prompt": item["model_visible_prompt"]}, ensure_ascii=False) + "\n")
    with (private / "all_144_private.jsonl").open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    gold_fields = (
        "item_id", "source_sample_id", "severity", "operator", "gold_tool",
        "expected_arguments_json", "clean_result_json", "observation_target_json", "prompt_sha256",
    )
    gold_rows = []
    for item in primary:
        gold_rows.append(
            {
                "item_id": item["item_id"],
                "source_sample_id": item["source_sample_id"],
                "severity": item["severity"],
                "operator": item["operator"],
                "gold_tool": item["gold_tool"],
                "expected_arguments_json": json.dumps(item["expected_arguments"], ensure_ascii=False, sort_keys=True),
                "clean_result_json": json.dumps(item["clean_result"], ensure_ascii=False, sort_keys=True),
                "observation_target_json": json.dumps(item["observation_target"], ensure_ascii=False, sort_keys=True),
                "prompt_sha256": item["prompt_sha256"],
            }
        )
    write_csv(private / "gold_key_private.csv", gold_rows, gold_fields)
    manifest = {
        "seed": args.seed,
        "design": "20 paired source anchors x 3 severities",
        "n_tools": len({item["gold_tool"] for item in primary}),
        "n_primary": len(primary),
        "n_extension": len(extension),
        "selected_source_sample_ids": anchors,
    }
    (private / "sampling_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(output)
    write_audit(output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
