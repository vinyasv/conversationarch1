"""ContractAssistant: extractor -> gate -> executor, once per user turn.

The executor never sees the chat transcript. It gets the task's system prompt plus the contract
rendered as a single user message, which is the setting where the paper found models recover
most of their single-turn performance.
"""
from dataclasses import dataclass

from convcontract import config, llm
from convcontract.contract import Contract

POLICIES = ("answer", "clarify")

_EXTRACTOR_PROMPT = (config.PROMPTS_DIR / "extractor.txt").read_text()
_EXECUTOR_PROMPT = (config.PROMPTS_DIR / "executor.txt").read_text()


_OPS_KEYS = {"goal", "add", "supersede", "remove", "material", "open_questions", "ready", "question", "refers_to_draft"}
_LIST_KEYS = ("add", "supersede", "remove", "material", "open_questions")


def valid_ops(ops) -> bool:
    """Structural check on extractor output. Valid JSON is not enough: in the pilot a bad provider
    returned `{}` and invented keys, which silently dropped what the user said."""
    if not isinstance(ops, dict) or not _OPS_KEYS >= set(ops) or not {"add", "supersede", "ready"} <= set(ops):
        return False
    if not isinstance(ops["ready"], bool) or any(not isinstance(ops.get(k) or [], list) for k in _LIST_KEYS):
        return False
    items = (ops.get("add") or []) + (ops.get("supersede") or [])
    return all(isinstance(i, dict) and isinstance(i.get("text"), str) for i in items)


@dataclass
class TurnResult:
    text: str  # what the user sees
    decision: str  # "answer" | "ask"
    ops: dict
    warnings: list[str]
    cost_usd: float
    contract: dict  # snapshot after this turn
    executor_prompt: str | None  # exactly what the executor saw (None when asking)


class ContractAssistant:
    def __init__(
        self,
        system_prompt: str = "",
        executor_model: str = config.EXECUTOR_MODEL,
        extractor_model: str = config.EXTRACTOR_MODEL,
        executor_temperature: float | None = 1.0,
        extractor_temperature: float | None = 0.0,
        policy: str = "clarify",
        max_consecutive_questions: int = 1,
        executor_max_tokens: int | None = None,
        extractor_prompt: str | None = None,
    ):
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
        self.system_prompt = system_prompt
        self.executor_model = executor_model
        self.extractor_model = extractor_model
        self.executor_temperature = executor_temperature
        self.extractor_temperature = extractor_temperature
        self.policy = policy
        self.max_consecutive_questions = max_consecutive_questions
        self.executor_max_tokens = executor_max_tokens
        self.extractor_prompt = extractor_prompt or _EXTRACTOR_PROMPT

        self.contract = Contract()
        self.last_reply: str | None = None
        self.consecutive_questions = 0

    def _extract(self, user_message: str, attempts: int = 3) -> tuple[dict, float]:
        prompt = (
            self.extractor_prompt.replace("[[CONTEXT]]", self.system_prompt)
            .replace("[[CONTRACT]]", self.contract.view_for_extractor())
            .replace("[[ASSISTANT_MESSAGE]]", self.last_reply or "")
            .replace("[[USER_MESSAGE]]", user_message)
        )
        cost = 0.0
        for _ in range(attempts):
            try:
                ops, response = llm.chat_json([{"role": "user", "content": prompt}], self.extractor_model, temperature=self.extractor_temperature)
            except ValueError:
                continue
            cost += response.cost_usd
            if valid_ops(ops):
                return ops, cost
        # Degrade safely: keep the user's words verbatim rather than losing the turn.
        return {"add": [{"text": user_message, "quote": user_message}], "ready": True, "extractor_failed": True}, cost

    def _should_ask(self, ops: dict) -> bool:
        return (
            self.policy == "clarify"
            and not ops.get("ready", True)
            and bool(ops.get("question"))
            and self.consecutive_questions < self.max_consecutive_questions
        )

    def update_contract(self, user_message: str) -> tuple[dict, list[str], float]:
        """Extractor step only: apply the user's message to the contract. Returns (ops, warnings, cost)."""
        ops, cost = self._extract(user_message)
        return ops, self.contract.apply(ops), cost

    def respond(self, user_message: str) -> TurnResult:
        ops, warnings, cost = self.update_contract(user_message)

        if self._should_ask(ops):
            self.consecutive_questions += 1
            text, decision, executor_prompt = ops["question"].strip(), "ask", None
        else:
            self.consecutive_questions = 0
            task = self.contract.render_task(include_draft=bool(ops.get("refers_to_draft")))
            executor_prompt = _EXECUTOR_PROMPT.replace("[[TASK]]", task)
            messages = [{"role": "user", "content": executor_prompt}]
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})
            response = llm.chat(messages, self.executor_model, temperature=self.executor_temperature, max_tokens=self.executor_max_tokens)
            cost += response.cost_usd
            text, decision = response.text, "answer"
            self.contract.draft = text

        self.last_reply = text
        return TurnResult(
            text=text,
            decision=decision,
            ops=ops,
            warnings=warnings,
            cost_usd=cost,
            contract=self.contract.to_dict(),
            executor_prompt=executor_prompt,
        )


class UserOnlyAssistant:
    """Ablation of ContractAssistant: no extractor. The executor gets the same instructions, but in place of
    the contract it sees the user's messages so far, verbatim and in order. Like the contract, it never sees
    its own earlier replies. Comparing the two separates "drop the model's own answers" from "rebuild the
    request as a structured requirements list"."""

    def __init__(self, system_prompt: str = "", executor_model: str = config.EXECUTOR_MODEL, executor_temperature: float | None = 1.0, executor_max_tokens: int | None = None):
        self.system_prompt = system_prompt
        self.executor_model = executor_model
        self.executor_temperature = executor_temperature
        self.executor_max_tokens = executor_max_tokens
        self.user_messages: list[str] = []

    def render_task(self) -> str:
        return "The user's messages, in the order they were sent:\n" + "\n".join(f"{i}. {m}" for i, m in enumerate(self.user_messages, 1))

    def respond(self, user_message: str) -> TurnResult:
        self.user_messages.append(user_message)
        executor_prompt = _EXECUTOR_PROMPT.replace("[[TASK]]", self.render_task())
        messages = [{"role": "user", "content": executor_prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        response = llm.chat(messages, self.executor_model, temperature=self.executor_temperature, max_tokens=self.executor_max_tokens)
        return TurnResult(
            text=response.text,
            decision="answer",
            ops={},
            warnings=[],
            cost_usd=response.cost_usd,
            contract={"user_messages": list(self.user_messages)},
            executor_prompt=executor_prompt,
        )
