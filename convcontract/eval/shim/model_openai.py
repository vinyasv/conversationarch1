"""Drop-in replacement for the harness's `model_openai` module, routed through `convcontract.llm`
(OpenRouter, or the Anthropic API for `claude-` models).

`convcontract.eval.harness.setup()` puts this directory ahead of the harness on sys.path, so every
`from model_openai import generate` in the harness, including in code-grading child processes,
resolves here. The harness's original module asserts OPENAI_API_KEY at import time, which would crash
those child processes and silently mark every code test as failed.

Signatures match the original so the harness runs unmodified.
"""
import re

from convcontract import llm


def _model_id(model: str) -> str:
    # The harness defaults to bare OpenAI names like "gpt-4o-mini".
    return model if "/" in model or llm.is_claude(model) else f"openai/{model}"


def format_messages(messages, variables={}):
    # Same behavior as the harness original (MIT, Microsoft): fill [[KEY]] slots in the last user message.
    last_user_msg = [msg for msg in messages if msg["role"] == "user"][-1]
    for k, v in variables.items():
        key_string = f"[[{k}]]"
        if key_string not in last_user_msg["content"]:
            print(f"[prompt] Key {k} not found in prompt; effectively ignored")
        assert type(v) == str, f"[prompt] Variable {k} is not a string"
        last_user_msg["content"] = last_user_msg["content"].replace(key_string, v)
    keys_still_in_prompt = re.findall(r"\[\[([^\]]+)\]\]", last_user_msg["content"])
    if len(keys_still_in_prompt) > 0:
        print(f"[prompt] The following keys were not replaced: {keys_still_in_prompt}")
    return messages


def generate(messages, model="gpt-4o-mini", timeout=30, max_retries=3, temperature=1.0, is_json=False, return_metadata=False, max_tokens=None, variables={}):
    messages = format_messages(messages, variables)
    response = llm.chat(messages, _model_id(model), temperature=temperature, max_tokens=max_tokens, json_mode=is_json, max_retries=max(max_retries, 4))
    if not return_metadata:
        return response.text
    return {
        "message": response.text,
        "total_tokens": response.prompt_tokens + response.completion_tokens,
        "prompt_tokens": response.prompt_tokens,
        "prompt_tokens_cached": 0,
        "completion_tokens": response.completion_tokens,
        "total_usd": response.cost_usd,
    }


def generate_json(messages, model="gpt-4o-mini", **kwargs):
    response = generate(messages, model, is_json=True, **kwargs)
    response["message"] = llm.parse_json(response["message"])
    return response
