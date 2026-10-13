"""LLM client. Every LLM call in the project (ours and the harness's) goes through `chat()`,
so tests and dry runs can swap the backend with `set_backend()`.

Model ids starting with `claude-` go to the Anthropic API; everything else goes to OpenRouter."""
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache

from convcontract import config

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


_backend = None  # callable(messages, model, temperature, max_tokens, json_mode) -> LLMResponse
_model_params: dict[str, dict] = {}  # extra sampling params per model, e.g. frequency_penalty


def set_backend(fn):
    """Route all calls to `fn` instead of OpenRouter (None restores OpenRouter)."""
    global _backend
    _backend = fn


def set_model_params(model: str, **params):
    """Send extra params on every call to `model`, whichever code path makes the call: sampling settings
    (e.g. frequency_penalty, to match another paper's setup) or `provider` routing for OpenRouter."""
    _model_params.setdefault(model, {}).update(params)


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set; copy .env.example to .env and add your key.")
    # max_retries=0: the SDK otherwise retries each request twice on its own, so one hung request
    # could take 3x the timeout before our retry loop even sees it.
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key, max_retries=0)


# Requests run on this pool so each attempt has a hard wall-clock limit. The HTTP read timeout alone isn't
# enough: a provider that keeps the connection alive without answering never trips it (seen in the pilots).
# A timed-out attempt's thread is abandoned and finishes on its own.
_call_pool = ThreadPoolExecutor(max_workers=128, thread_name_prefix="llm")

_log_lock = threading.Lock()


def _log_call(**record):
    """Append one line per API attempt to $CC_CALL_LOG, if set (model, provider, seconds, error)."""
    path = os.environ.get("CC_CALL_LOG")
    if path:
        with _log_lock, open(path, "a") as f:
            f.write(json.dumps(record) + "\n")


def chat(messages, model, temperature=None, max_tokens=None, json_mode=False, max_retries=4, timeout=90) -> LLMResponse:
    # The floor applies to every call so no arm is truncated differently (thinking models spend
    # part of max_tokens on reasoning).
    max_tokens = max(max_tokens or 0, config.MIN_MAX_TOKENS)

    if _backend is not None:
        return _backend(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)
    if is_claude(model):
        return _chat_anthropic(messages, model, max_tokens, max_retries, timeout)

    params = dict(_model_params.get(model, {}))
    # Prefer the fastest provider. In the pilot one Llama 3.3 70B provider sometimes held requests for
    # ~7 minutes (the read timeout never fires while the connection is kept alive).
    # A per-model `provider` setting (e.g. an explicit order) replaces this default.
    provider = dict(params.pop("provider", None) or {"sort": "latency"})
    if json_mode:
        # Only route to providers that implement response_format. Without this, OpenRouter may pick a
        # provider that ignores it (seen in the pilot: `{}` and invented keys from Llama 3.3 70B).
        provider["require_parameters"] = True
    kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens, "timeout": timeout, **params}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    extra_body = {"usage": {"include": True}, "provider": provider}
    if config.REASONING_EFFORT:
        extra_body["reasoning"] = {"effort": config.REASONING_EFFORT}
    kwargs["extra_body"] = extra_body

    last_error = None
    for attempt in range(max_retries):
        started = time.time()
        try:
            response = _call_pool.submit(_client().chat.completions.create, **kwargs).result(timeout=timeout + 5)
            text = (response.choices[0].message.content or "") if response.choices else ""
            served_by = getattr(response, "provider", None)
            if not text.strip():
                raise RuntimeError(f"empty response from {model} via {served_by}")
            _log_call(model=model, provider=served_by, seconds=round(time.time() - started, 1), attempt=attempt, ok=True)
            usage = response.usage
            cost = getattr(usage, "cost", None)
            if cost is None and usage is not None and usage.model_extra:
                cost = usage.model_extra.get("cost")
            return LLMResponse(
                text=text,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                cost_usd=float(cost or 0.0),
            )
        except Exception as e:  # network errors, rate limits, empty completions
            last_error = e
            _log_call(model=model, seconds=round(time.time() - started, 1), attempt=attempt, ok=False, error=str(e)[:200])
            time.sleep(min(2 ** attempt * 2, 30))
    raise RuntimeError(f"OpenRouter call to {model} failed after {max_retries} attempts: {last_error}")


# --- Anthropic ---------------------------------------------------------------------------------

# USD per million tokens (input, output), first-party API, prompts up to 100K tokens.
CLAUDE_PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00)}
# Thinking for every Claude call, all arms and roles alike: "off" or "adaptive". Set by eval.run.
CLAUDE_THINKING = os.environ.get("CC_CLAUDE_THINKING", "off")
# How each model turns thinking off: Sonnet 5.5 rejects {"type": "disabled"} and uses "between_tools".
_THINKING_OFF = {"claude-sonnet-5-5": {"type": "between_tools"}}
# Returned in place of a refused reply, so the turn is graded as a non-answer instead of dropping the
# conversation (dropping would remove different conversations from different arms).
REFUSAL_TEXT = "(no answer: the model declined this request)"


def is_claude(model: str) -> bool:
    return model.startswith("claude-")


@lru_cache(maxsize=1)
def _anthropic_client():
    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set; add it to .env.")
    # max_retries=0 for the same reason as the OpenRouter client: our loop owns retries and deadlines.
    return anthropic.Anthropic(max_retries=0)


def _chat_anthropic(messages, model, max_tokens, max_retries, timeout) -> LLMResponse:
    """Anthropic Messages API. `temperature` is not sent: these models reject non-default sampling values,
    so runs vary and every arm gets the same default. No server-side refusal fallback either: it would
    swap the model under test mid-run."""
    import anthropic

    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    turns = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
    kwargs = {"model": model, "max_tokens": max_tokens, "messages": turns, "timeout": timeout}
    if system:
        kwargs["system"] = system
    if CLAUDE_THINKING == "off":
        kwargs["thinking"] = _THINKING_OFF.get(model, {"type": "disabled"})
    else:
        kwargs["thinking"] = {"type": "adaptive"}

    last_error = None
    for attempt in range(max_retries):
        started = time.time()
        try:
            response = _call_pool.submit(_anthropic_client().messages.create, **kwargs).result(timeout=timeout + 5)
            refused = response.stop_reason == "refusal"
            text = REFUSAL_TEXT if refused else "".join(b.text for b in response.content if b.type == "text")
            if not text.strip():
                raise RuntimeError(f"empty response from {model} (stop_reason={response.stop_reason})")
            _log_call(model=model, provider="anthropic", seconds=round(time.time() - started, 1), attempt=attempt, ok=True, stop_reason=response.stop_reason)
            usage = response.usage
            price_in, price_out = CLAUDE_PRICES.get(model, (0.0, 0.0))
            return LLMResponse(
                text=text,
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                cost_usd=(usage.input_tokens * price_in + usage.output_tokens * price_out) / 1e6,
            )
        except anthropic.BadRequestError:
            raise  # a malformed request fails the same way every time
        except Exception as e:  # rate limits, overload, network errors, timeouts, empty replies
            last_error = e
            _log_call(model=model, provider="anthropic", seconds=round(time.time() - started, 1), attempt=attempt, ok=False, error=str(e)[:200])
            time.sleep(min(2 ** attempt * 2, 30))
    raise RuntimeError(f"Anthropic call to {model} failed after {max_retries} attempts: {last_error}")


def parse_json(text: str) -> dict:
    """Parse a JSON object from a model reply, tolerating code fences and surrounding prose."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def chat_json(messages, model, parse_retries=2, **kwargs) -> tuple[dict, LLMResponse]:
    """Like `chat()`, but returns a parsed JSON object. Retries when the reply is not valid JSON."""
    total_cost = 0.0
    for attempt in range(parse_retries + 1):
        response = chat(messages, model, json_mode=True, **kwargs)
        total_cost += response.cost_usd
        try:
            data = parse_json(response.text)
            if isinstance(data, dict):
                response.cost_usd = total_cost
                return data, response
        except json.JSONDecodeError:
            pass
    raise ValueError(f"{model} did not return valid JSON after {parse_retries + 1} attempts: {response.text[:300]!r}")
