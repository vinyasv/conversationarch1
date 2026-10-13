import json

import pytest

from convcontract import llm
from convcontract.layer import ContractAssistant
from convcontract.llm import LLMResponse


class ScriptedBackend:
    """Extractor replies come from a script; executor calls are recorded and answered with a counter."""

    def __init__(self, extractor_replies):
        self.extractor_replies = list(extractor_replies)
        self.executor_calls = []

    def __call__(self, messages, model, temperature=None, max_tokens=None, json_mode=False):
        if "You maintain a requirements contract" in messages[-1]["content"]:
            reply = self.extractor_replies.pop(0)
            return LLMResponse(text=reply if isinstance(reply, str) else json.dumps(reply), cost_usd=0.001)
        self.executor_calls.append(messages)
        return LLMResponse(text=f"answer #{len(self.executor_calls)}", cost_usd=0.01)


@pytest.fixture
def backend():
    holder = {}

    def install(replies):
        holder["b"] = ScriptedBackend(replies)
        llm.set_backend(holder["b"])
        return holder["b"]

    yield install
    llm.set_backend(None)


def add(text, ready=True, question=None, refers_to_draft=False):
    return {"goal": None, "add": [{"text": text, "quote": text}], "supersede": [], "remove": [], "material": [], "open_questions": [], "ready": ready, "question": question, "refers_to_draft": refers_to_draft}


def test_executor_sees_only_the_contract_never_the_transcript(backend):
    b = backend([add("Rank customers by order value"), add("Only 2025 orders"), add("Top 5")])
    a = ContractAssistant(system_prompt="You write SQL.", policy="answer")
    for msg in ["top customers pls", "last year only", "just 5"]:
        a.respond(msg)
    last = b.executor_calls[-1]
    assert [m["role"] for m in last] == ["system", "user"]
    assert last[0]["content"] == "You write SQL."
    task = last[1]["content"]
    for req in ["Rank customers by order value", "Only 2025 orders", "Top 5"]:
        assert req in task
    # neither the raw user wording nor the previous answers leak through
    assert "last year only" not in task and "answer #" not in task


def test_clarify_asks_at_most_once_in_a_row(backend):
    b = backend([add("x", ready=False, question="Which year?"), add("y", ready=False, question="Which region?"), add("z", ready=False, question="Q3?")])
    a = ContractAssistant(policy="clarify", max_consecutive_questions=1)
    decisions = [a.respond(m).decision for m in ["a", "b", "c"]]
    assert decisions == ["ask", "answer", "ask"]
    assert len(b.executor_calls) == 1


def test_answer_policy_never_asks(backend):
    backend([add("x", ready=False, question="Which year?")])
    turn = ContractAssistant(policy="answer").respond("a")
    assert turn.decision == "answer" and turn.text == "answer #1"


def test_draft_included_only_when_user_refers_to_it(backend):
    b = backend([add("Write a haiku"), add("Make it about rain", refers_to_draft=True)])
    a = ContractAssistant(policy="answer")
    a.respond("haiku")
    a.respond("make that about rain")
    assert "previous_answer" not in b.executor_calls[0][-1]["content"]
    assert "<previous_answer>\nanswer #1\n</previous_answer>" in b.executor_calls[1][-1]["content"]


@pytest.mark.parametrize(
    "bad_reply",
    [
        "not json",
        {},  # seen in the pilot: valid JSON, no content
        {"goal": "x", "add": [], "supers": [], "remove": [], "ready": True},  # seen in the pilot: invented key
        {"add": "304 humps", "supersede": [], "ready": True},  # wrong type
    ],
)
def test_malformed_extractor_output_keeps_users_words(backend, bad_reply):
    backend([bad_reply] * 12)
    a = ContractAssistant(policy="clarify")
    turn = a.respond("use the orders table")
    assert turn.ops.get("extractor_failed")
    assert [r.text for r in a.contract.active()] == ["use the orders table"]
    assert turn.decision == "answer"


def test_extractor_retries_until_valid(backend):
    b = backend([{}, add("Only 2025 orders")])
    a = ContractAssistant(policy="answer")
    turn = a.respond("last year only")
    assert not turn.ops.get("extractor_failed")
    assert [r.text for r in a.contract.active()] == ["Only 2025 orders"]
    assert b.extractor_replies == []


def test_cost_sums_extractor_and_executor(backend):
    backend([add("x")])
    assert ContractAssistant(policy="answer").respond("a").cost_usd == pytest.approx(0.011)


def test_user_only_ablation_sees_user_messages_but_not_its_replies(backend):
    from convcontract.layer import UserOnlyAssistant

    b = backend([])
    a = UserOnlyAssistant(system_prompt="You write SQL.")
    for msg in ["top customers pls", "last year only", "just 5"]:
        a.respond(msg)
    assert len(b.executor_calls) == 3
    task = b.executor_calls[-1][1]["content"]
    assert "1. top customers pls\n2. last year only\n3. just 5" in task
    assert "answer #" not in task
