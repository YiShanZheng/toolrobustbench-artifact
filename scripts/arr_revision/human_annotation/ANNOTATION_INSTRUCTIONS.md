# C2 Recoverability Blinded-Annotation Instructions

You will receive a CSV file containing 60 rows. Complete the annotations independently. Do not discuss them with the other annotator, and do not consult the paper, model scores, gold answers, search engines, or other files in the project.

For each row, read only `model_visible_prompt`:

1. In `recovered_answer_json`, enter the best JSON answer that can be reconstructed solely from the visible evidence. Enter `null` when no answer can be reconstructed.
2. In `recoverability_label`, enter exactly one of:
   - `FULL`: the visible evidence is sufficient to reconstruct the complete and exact answer;
   - `PARTIAL`: only some fields or the general direction can be reconstructed, while exact values are missing;
   - `NONE`: the visible evidence is insufficient to reconstruct the target content.
3. In `confidence_1_to_5`, enter an integer from 1 to 5.
4. In `evidence_citation`, quote the supporting text or identify the relevant field path. Do not use unsupported statements such as “obvious” or “can be seen.”
5. In `notes`, record any ambiguity, conflicting evidence, or reason that the item cannot be determined.
6. In `minutes_spent`, record the time spent on the row; decimal values are allowed.

Do not modify `annotation_order`, `item_id`, `severity`, or `model_visible_prompt`. When finished, save the original CSV as a UTF-8 CSV and return it to the coordinator. During annotation, do not open any file whose name or contents expose `gold`, `private`, `target`, model success rates, or aggregate results.
