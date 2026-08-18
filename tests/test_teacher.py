import json
from pathlib import Path

import pytest

from pdfsum import db
from pdfsum.dataset import teacher


@pytest.fixture
def conn(tmp_path: Path):
    return db.connect(tmp_path / "registry.db")


def test_missing_api_key_raises(conn, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(teacher.MissingApiKeyError):
        teacher.call_teacher(conn, "summarize this")


def test_quota_exceeded_raises_before_network_call(conn, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "unused-because-quota-blocks-first")
    for _ in range(teacher.DAILY_FREE_REQUEST_LIMIT):
        db.record_teacher_request(conn, teacher.FREE_MODELS[0])

    with pytest.raises(teacher.QuotaExceededError):
        teacher.call_teacher(conn, "summarize this")


def test_successful_call_records_request(conn, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": '{"summary": "ok"}'}}]}
            ).encode("utf-8")

    monkeypatch.setattr(
        "pdfsum.dataset.teacher.urllib.request.urlopen", lambda *a, **k: FakeResponse()
    )

    result = teacher.call_teacher(conn, "summarize this")
    assert result == '{"summary": "ok"}'
    assert db.count_teacher_requests_today_utc(conn) == 1
