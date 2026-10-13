import threading
import time

import pytest

from convcontract import llm


class HangingClient:
    """Stands in for an OpenRouter call that never returns."""

    calls = 0

    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                HangingClient.calls += 1
                threading.Event().wait(30)  # not time.sleep, which the test patches


def test_hung_request_hits_hard_deadline(monkeypatch):
    monkeypatch.setattr(llm, "_client", lambda: HangingClient)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)  # skip retry backoff (the pool thread uses the real one)
    started = time.time()
    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        llm.chat([{"role": "user", "content": "hi"}], "some/model", max_retries=2, timeout=0)
    assert time.time() - started < 15
    assert HangingClient.calls == 2


def test_parse_json_tolerates_fences_and_prose():
    assert llm.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.parse_json('Sure! {"a": 1} hope that helps') == {"a": 1}


class FakeAnthropic:
    """Records Messages API requests and answers with a canned reply."""

    def __init__(self, stop_reason="end_turn", text="hello"):
        self.requests = []
        self.messages = self
        self.stop_reason, self.text = stop_reason, text

    def create(self, **kwargs):
        from types import SimpleNamespace as NS

        self.requests.append(kwargs)
        content = [] if self.stop_reason == "refusal" else [NS(type="text", text=self.text)]
        return NS(stop_reason=self.stop_reason, content=content, usage=NS(input_tokens=1_000_000, output_tokens=100_000))


@pytest.mark.parametrize("model, thinking_off", [("claude-haiku-5-5", {"type": "disabled"}), ("claude-sonnet-5-5", {"type": "between_tools"})])
def test_claude_request_shape(monkeypatch, model, thinking_off):
    fake = FakeAnthropic()
    monkeypatch.setattr(llm, "_anthropic_client", lambda: fake)
    monkeypatch.setattr(llm, "CLAUDE_THINKING", "off")
    messages = [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}, {"role": "user", "content": "again"}]
    response = llm.chat(messages, model, temperature=0.01, json_mode=True)
    req = fake.requests[0]
    assert req["system"] == "Be terse."
    assert [m["role"] for m in req["messages"]] == ["user", "assistant", "user"]
    assert "temperature" not in req  # rejected by these models; all arms get the default
    assert req["thinking"] == thinking_off
    price_in, price_out = llm.CLAUDE_PRICES[model]
    assert response.text == "hello" and response.cost_usd == pytest.approx(price_in + price_out / 10)


def test_claude_refusal_becomes_a_non_answer(monkeypatch):
    monkeypatch.setattr(llm, "_anthropic_client", lambda: FakeAnthropic(stop_reason="refusal"))
    assert llm.chat([{"role": "user", "content": "hi"}], "claude-haiku-5-5").text == llm.REFUSAL_TEXT
