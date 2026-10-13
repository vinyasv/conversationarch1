"""Scripted stand-in for OpenRouter, for dry runs and tests. Costs nothing and needs no API key.

It recognizes each prompt the pipeline sends (simulated user, turn classifier, answer extractor, our
extractor, executor) by a marker string and returns a well-formed reply. Answers are deliberately
wrong, so a dry run checks wiring and grading, not scores.
"""
import json
import re

from convcontract.llm import LLMResponse

FAKE_CODE = "def solution(x):\n    return x"


def fake_backend(messages, model, temperature=None, max_tokens=None, json_mode=False) -> LLMResponse:
    prompt = messages[-1]["content"]
    everything = "\n".join(m["content"] for m in messages)

    if "You maintain a requirements contract" in prompt:
        user_message = re.search(r"<user_message>\n(.*?)\n</user_message>", prompt, re.S).group(1)
        reply = {"goal": None, "add": [{"text": user_message, "quote": user_message}], "supersede": [], "remove": [], "material": [], "open_questions": [], "ready": True, "question": None, "refers_to_draft": False}
    elif "You are simulating a user" in prompt:
        hidden = re.search(r"have not been revealed yet:\n(.*?)\n\nYou must generate", prompt, re.S).group(1)
        remaining = json.loads(hidden)
        reply = {"response": remaining[0]["shard"], "shard_id": remaining[0]["shard_id"]} if remaining else {"response": "that's all", "shard_id": -1}
    elif '"response_type"' in prompt:
        reply = {"response_type": "answer_attempt"}
    elif "extract verbatim" in prompt:
        reply = {"answer": FAKE_CODE if "def solution" in prompt else "42"}
    elif "Python" in everything:
        return LLMResponse(text=f"```python\n{FAKE_CODE}\n```", prompt_tokens=10, completion_tokens=10)
    else:
        return LLMResponse(text="Final answer: 42", prompt_tokens=10, completion_tokens=5)

    return LLMResponse(text=json.dumps(reply), prompt_tokens=10, completion_tokens=10)
