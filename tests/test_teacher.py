import io
import json
import urllib.error
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


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(
            {"choices": [{"message": {"content": '{"summary": "recovered"}'}}]}
        ).encode("utf-8")


def _http_error(code):
    return urllib.error.HTTPError(
        "https://openrouter.ai", code, "err", {}, io.BytesIO(b'{"error": "boom"}')
    )


def test_retries_transient_error_then_succeeds(conn, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("pdfsum.dataset.teacher.time.sleep", lambda s: None)

    calls = {"n": 0}

    def flaky_urlopen(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _http_error(400)
        return FakeResponse()

    monkeypatch.setattr("pdfsum.dataset.teacher.urllib.request.urlopen", flaky_urlopen)

    result = teacher.call_teacher(conn, "summarize this")
    assert result == '{"summary": "recovered"}'
    assert calls["n"] == 2
    assert db.count_teacher_requests_today_utc(conn) == 1  # only the success is recorded


def test_429_never_retried(conn, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    calls = {"n": 0}

    def always_429(*a, **k):
        calls["n"] += 1
        raise _http_error(429)

    monkeypatch.setattr("pdfsum.dataset.teacher.urllib.request.urlopen", always_429)

    with pytest.raises(teacher.TeacherApiError):
        teacher.call_teacher(conn, "summarize this")
    assert calls["n"] == 1  # no retry attempted for a real quota error
