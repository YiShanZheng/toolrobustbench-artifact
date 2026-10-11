# C2 可恢复性人工盲标说明

你会收到一份包含 60 行的 CSV。请独立完成，不与另一名标注者讨论，也不要查看论文、模型成绩、Gold 答案、搜索引擎或项目中的其他文件。

每行只阅读 `model_visible_prompt`：

1. 在 `recovered_answer_json` 中填写仅凭可见证据能恢复出的最佳 JSON；完全不能恢复时填写 `null`。
2. `recoverability_label` 只能填：
   - `FULL`：可见证据足以恢复完整、精确答案；
   - `PARTIAL`：只能恢复部分字段或方向，缺少精确值；
   - `NONE`：没有足够证据恢复目标内容。
3. `confidence_1_to_5` 填 1--5。
4. `evidence_citation` 必须写出支持判断的字段路径或原文片段；不能只写“明显”“看得出来”。
5. 在 `notes` 记录歧义、互相冲突的证据或无法判断的原因。
6. `minutes_spent` 记录该行耗时，可填小数。

不要更改 `annotation_order`、`item_id`、`severity` 或 `model_visible_prompt`。完成后把原 CSV 另存为 UTF-8 CSV 并交回协调者。标注过程中不要打开任何含 `gold`、`private`、`target`、模型成功率或聚合结果的文件。
