"""Tests for SSE utility helpers."""

from __future__ import annotations

import datetime as dt
import json

from paper.backend.sse import _sse_event


def test_sse_event_serialises_dates() -> None:
    """SSE payloads should support date values via JSON encoding."""
    frame = _sse_event(
        "complete",
        {"result": {"publication_date": dt.date(2024, 1, 2), "label": 5}},
    )

    assert frame.startswith("event: complete\n")
    payload_text = frame.splitlines()[1].removeprefix("data: ")
    payload = json.loads(payload_text)
    assert payload["result"]["publication_date"] == "2024-01-02"
    assert payload["result"]["label"] == 5
