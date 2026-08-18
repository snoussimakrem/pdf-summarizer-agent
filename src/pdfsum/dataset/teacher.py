"""OpenRouter client for teacher-model calls during dataset generation.

Both models below were license-verified 2026-08-18 (Apache 2.0 and
OpenMDW-1.1 respectively — see project memory) to carry no restriction on
using their outputs to train another model. Do not add a model here without
checking its specific license page first — "free to call" is not the same
as "free to train on".
"""
import json
import os
import sqlite3
import time
import urllib.error
import urllib.request

from pdfsum import db

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
FREE_MODELS = [
    "dots-studio/dots-3-note-preview:free",
    "nvidia/nemotron-3.5-lightning:free",
]
DAILY_FREE_REQUEST_LIMIT = 50
# Transient provider-side errors happen (verified 2026-08-18: a real 400
# "bad request" from the AtlasCloud backend on a chunk that succeeded
# verbatim on retry seconds later) -- retry a couple of times before giving
# up, rather than losing an entire multi-request hierarchical generation to
# one flaky call. 429 (quota) is handled separately and never retried here.
TRANSIENT_RETRY_STATUS_CODES = {400, 500, 502, 503, 504}
MAX_TRANSIENT_RETRIES = 2
RETRY_BACKOFF_SECONDS = 3.0


class QuotaExceededError(RuntimeError):
    pass


class TeacherApiError(RuntimeError):
    pass


class MissingApiKeyError(RuntimeError):
    pass


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise MissingApiKeyError(
            "OPENROUTER_API_KEY is not set. MANUAL_ACTION_REQUIRED: create a free "
            "OpenRouter account at https://openrouter.ai, generate an API key, and "
            "export it as OPENROUTER_API_KEY before running dataset generation."
        )
    return key


def _check_quota(conn: sqlite3.Connection) -> None:
    used = db.count_teacher_requests_today_utc(conn)
    if used >= DAILY_FREE_REQUEST_LIMIT:
        raise QuotaExceededError(
            f"OpenRouter free-tier daily quota ({DAILY_FREE_REQUEST_LIMIT} req/day, "
            "shared across ALL free models) is exhausted for today (UTC) "
            f"({used} requests recorded by us — the account-wide total, including "
            "any usage outside this tool, may be higher). Wait for the UTC-midnight "
            "reset, or stop here — "
            "buying the $10 credit unlock (1000/day) is a paid deviation that "
            "must be explicitly flagged and confirmed before use."
        )


def call_teacher(
    conn: sqlite3.Connection,
    prompt: str,
    model: str = FREE_MODELS[0],
    max_tokens: int = 2000,
    timeout: float = 120.0,
    _already_fell_back: bool = False,
) -> str:
    _check_quota(conn)
    key = _api_key()

    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            # Both free teacher models default to emitting hidden chain-of-thought
            # reasoning tokens before content, which can consume the whole
            # max_tokens budget and leave "content": null (verified 2026-08-18).
            # Disabling it is confirmed to work cleanly for both models.
            "reasoning": {"enabled": False},
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        OPENROUTER_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )

    last_error = None
    for attempt in range(MAX_TRANSIENT_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                payload = json.loads(resp.read())
            break
        except urllib.error.HTTPError as e:
            body_text = e.read().decode()
            last_error = TeacherApiError(f"OpenRouter request failed: {e.code} {body_text}")
            if e.code == 429:
                raise last_error from e  # quota errors are never transient
            if e.code in TRANSIENT_RETRY_STATUS_CODES and attempt < MAX_TRANSIENT_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS)
                continue
            if e.code in TRANSIENT_RETRY_STATUS_CODES and not _already_fell_back:
                # Verified 2026-08-18: a 400 that failed 3/3 times against
                # dots-studio (AtlasCloud backend) on a report-document chunk
                # succeeded immediately on the *first* try against the other
                # free model with identical content -- a provider-specific
                # issue, not randomness. Worth one cross-model attempt before
                # giving up entirely.
                other_model = next((m for m in FREE_MODELS if m != model), None)
                if other_model:
                    return call_teacher(
                        conn, prompt, model=other_model, max_tokens=max_tokens,
                        timeout=timeout, _already_fell_back=True,
                    )
            raise last_error from e

    db.record_teacher_request(conn, model)
    content = payload["choices"][0]["message"]["content"]
    if content is None:
        finish_reason = payload["choices"][0].get("finish_reason")
        raise TeacherApiError(
            f"teacher model returned no content (finish_reason={finish_reason}); "
            "likely truncated by max_tokens — this request was still counted "
            "against the daily quota"
        )
    return content
