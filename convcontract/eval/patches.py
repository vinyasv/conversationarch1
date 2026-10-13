"""Grading fixes applied to the harness at setup, identically for every arm.

1. Math answer extraction ("gen" strategy, math only). The harness asks gpt-4o-mini to copy the final
answer verbatim, then rejects it unless it is an exact substring of the response. The extractor
normalizes numbers ("$1,198" -> "1198"), so any answer written with a thousands separator is rejected
three times and scored 0 even when it is correct (seen in the Luna pilot on GSM8K/369). We also accept
an extraction whose normalized form appears in the normalized response.

2. Math grading. The harness compares strings, so "64.00" fails against "64" unless a "$" is present (seen in the
main run). We also accept the answer when its last number equals the gold number.

3. Tool-call grading (BFCL). The task prompt lists the functions in JSON and never says booleans must be Python
`True`/`False`. The parser reads a bare `true`/`false` as the string "true"/"false", so a correct call written
JSON-style fails with a type error (seen in the Claude pilot: Haiku writes `sort_by_rating=true` about half the
time; gpt-4.1-nano does it too). We read those two bare names as booleans.
"""
import re

import tasks.actions.eval_bfcl as eval_bfcl
from model_openai import generate_json
from system_agent import SystemAgent
from tasks.math.task_math import TaskMath
from utils import extract_conversation

_original_extract_answer = SystemAgent.extract_answer
_original_math_evaluator = TaskMath.evaluator_function
_original_resolve_ast_by_type = eval_bfcl.resolve_ast_by_type


def _normalize(text: str) -> str:
    return re.sub(r"[\s,${}\\]", "", text)


def _extract_answer(self, conversation_so_far):
    if self.answer_extraction_strategy != "gen":
        return _original_extract_answer(self, conversation_so_far)

    assistant_response = [msg["content"] for msg in conversation_so_far if msg["role"] == "assistant"][-1]
    last_turn_text = extract_conversation(conversation_so_far, to_str=True, only_last_turn=True)
    for _ in range(self.max_extraction_attempts):
        response = generate_json([{"role": "user", "content": self.answer_extraction_prompt_gen}], model=self.system_model, return_metadata=True, variables={"ASSISTANT_RESPONSE": last_turn_text, "ANSWER_DESCRIPTION": self.answer_description}, temperature=0.0)
        answer = str(response["message"].get("answer", ""))
        if answer and (answer in assistant_response or _normalize(answer) in _normalize(assistant_response)):
            return answer
    print(f"Failed to extract answer after {self.max_extraction_attempts} attempts")
    return ""


def _last_number(text: str) -> float | None:
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text or "")
    try:
        return float(numbers[-1].replace(",", "").rstrip(".")) if numbers else None
    except ValueError:
        return None


def _math_evaluator(self, extracted_answer, sample):
    result = _original_math_evaluator(self, extracted_answer, sample)
    if result.get("score") != 1.0:
        predicted, gold = _last_number(extracted_answer), _last_number(sample["answer"].split("####")[1])
        if predicted is not None and gold is not None and abs(predicted - gold) < 1e-6:
            return {"score": 1.0, "note": "accepted by numeric comparison"}
    return result


def _resolve_ast_by_type(value):
    import ast

    if isinstance(value, ast.Name) and value.id in ("true", "false"):
        return value.id == "true"
    return _original_resolve_ast_by_type(value)


def apply():
    SystemAgent.extract_answer = _extract_answer
    TaskMath.evaluator_function = _math_evaluator
    eval_bfcl.resolve_ast_by_type = _resolve_ast_by_type
