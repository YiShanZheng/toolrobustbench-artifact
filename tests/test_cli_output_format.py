from __future__ import annotations

import json


def test_cli_result_payload_is_single_line_json() -> None:
    payload = {"clean_samples": 12, "perturbed_samples": 108}
    rendered = json.dumps(payload, ensure_ascii=False)
    assert "\n" not in rendered
