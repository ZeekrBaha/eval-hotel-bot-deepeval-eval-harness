"""Tests for judge retry/backoff and JudgeError exception."""

import pytest
from pydantic import BaseModel

from judge.deepseek_judge import DeepSeekJudge, JudgeError


class _Schema(BaseModel):
    verdict: bool


def _judge():
    return DeepSeekJudge(api_key="test-key", base_url="http://fake")


def test_generate_retries_on_bad_json_and_succeeds(monkeypatch):
    """Two bad-JSON responses, then valid JSON — should return parsed schema."""
    monkeypatch.setattr("time.sleep", lambda _: None)
    judge = _judge()
    call_count = 0

    def fake_chat(prompt, json_mode):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return "not valid json!!!"
        return '{"verdict": true}'

    monkeypatch.setattr(judge, "_chat", fake_chat)
    result = judge.generate("test", schema=_Schema)
    assert result.verdict is True
    assert call_count == 3


def test_generate_raises_judge_error_on_persistent_bad_json(monkeypatch):
    """Three consecutive bad-JSON responses should raise JudgeError."""
    monkeypatch.setattr("time.sleep", lambda _: None)
    judge = _judge()

    monkeypatch.setattr(judge, "_chat", lambda *_: "not json")

    with pytest.raises(JudgeError, match="judge failed after 3 attempts"):
        judge.generate("test", schema=_Schema)


def test_generate_raises_judge_error_on_network_failure(monkeypatch):
    """Persistent connection errors should raise JudgeError, not ConnectionError."""
    monkeypatch.setattr("time.sleep", lambda _: None)
    judge = _judge()

    def boom(*_):
        raise ConnectionError("network down")

    monkeypatch.setattr(judge, "_chat", boom)

    with pytest.raises(JudgeError, match="judge failed after 3 attempts"):
        judge.generate("test")


def test_generate_no_retry_for_plain_text(monkeypatch):
    """Without schema, bad text is returned as-is (no JSON parse, no retry needed)."""
    judge = _judge()
    monkeypatch.setattr(judge, "_chat", lambda *_: "some plain text")
    result = judge.generate("test")
    assert result == "some plain text"


def test_judge_error_is_importable():
    from judge.deepseek_judge import JudgeError  # noqa: F401

    assert issubclass(JudgeError, Exception)
