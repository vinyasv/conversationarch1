"""End-to-end wiring through the Microsoft harness with the fake backend (no API calls)."""
import json
import os
import shutil
import uuid

import pytest

from convcontract import config, llm
from convcontract.eval import harness

pytestmark = pytest.mark.skipif(not config.HARNESS_DIR.exists(), reason="harness not cloned into external/")


@pytest.fixture
def in_harness():
    cwd = os.getcwd()
    harness.setup()
    os.chdir(config.HARNESS_DIR)
    yield
    os.chdir(cwd)
    llm.set_backend(None)


def test_code_grading_works_in_child_process(in_harness):
    """The code grader runs tests in a child process that imports the harness's `model_openai`.
    With the harness original (which asserts OPENAI_API_KEY) the child crashes and every test silently
    fails. With the shim first on sys.path, a correct solution must score 1.0."""
    from tasks import get_task

    sample = next(d for d in json.load(open(harness.DATASET)) if d["task_id"] == "sharded-HumanEval/105")
    solution = (
        "def by_length(arr):\n"
        "    names = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine'}\n"
        "    return [names[x] for x in sorted(arr, reverse=True) if 1 <= x <= 9]\n"
    )
    task = get_task("code")
    assert task.evaluator_function(solution, sample)["score"] == 1.0
    assert task.evaluator_function("def by_length(arr):\n    return []\n", sample)["score"] == 0.0


def test_math_extraction_accepts_thousands_separator(in_harness):
    """The harness rejected '1198' because the response says '$1,198', scoring a correct answer 0."""
    from system_agent import SystemAgent
    from tasks import get_task

    from convcontract.llm import LLMResponse

    llm.set_backend(lambda **kw: LLMResponse(text='{"answer": "1198"}'))
    sample = next(d for d in json.load(open(harness.DATASET)) if d["task_id"] == "sharded-GSM8K/369")
    trace = [{"role": "user", "content": "how much?"}, {"role": "assistant", "content": "**Total: $1,056 + $10 + $132 = $1,198.**"}]
    answer = SystemAgent("math", "openai/gpt-4o-mini", sample).extract_answer(trace)
    assert answer == "1198"
    assert get_task("math").evaluator_function(answer, sample)["score"] == 1.0

    llm.set_backend(lambda **kw: LLMResponse(text='{"answer": "1200"}'))
    assert SystemAgent("math", "openai/gpt-4o-mini", sample).extract_answer(trace) == ""


def test_dry_run_all_arms(in_harness):
    from convcontract.eval import run

    run_name = f"pytest-dry-run-{uuid.uuid4().hex[:8]}"
    run_dir = config.RESULTS_DIR / run_name
    try:
        run.main(["--dry-run", "--tasks", "math", "code", "--arms", "full", "concat", "sharded", "contract", "contract-clarify", "--workers", "4", "--run-name", run_name])
        assert not (run_dir / "errors.jsonl").exists(), (run_dir / "errors.jsonl").read_text()
        records = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
        assert len(records) == 2 * 2 * 5  # tasks x instructions x arms (1 run each)

        samples = {d["task_id"]: d for d in json.load(open(harness.DATASET))}
        for r in records:
            if r["arm"] in ("sharded", "contract", "contract-clarify"):
                # the fake user reveals one shard per turn and the fake answers are wrong, so every shard gets revealed
                assert r["n_user_turns"] == len(samples[r["task_id"]]["shards"]), r

        traces = {json.loads(line)["key"]: json.loads(line)["trace"] for line in (run_dir / "traces.jsonl").read_text().splitlines()}
        contract_keys = [k for k in traces if "|contract|" in k]
        assert contract_keys
        for key in contract_keys:
            snapshots = [m["content"]["contract"] for m in traces[key] if m["role"] == "log" and m["content"]["type"] == "contract"]
            task_id = key.split("|")[1]
            assert len(snapshots[-1]["requirements"]) == len(samples[task_id]["shards"])

        # resuming skips finished conversations
        run.main(["--dry-run", "--tasks", "math", "code", "--arms", "full", "concat", "sharded", "contract", "contract-clarify", "--run-name", run_name])
        assert len((run_dir / "results.jsonl").read_text().splitlines()) == len(records)

        summary = json.loads((run_dir / "summary.json").read_text())
        assert {c["arm"] for c in summary["comparisons"]} == {"contract", "contract-clarify"}
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_mtosc_subset_sizes(in_harness):
    from convcontract.eval.run import select_samples

    samples = select_samples(["math", "actions", "code"], limit=None, seed=0, min_shards=6, code_source="humaneval")
    counts = {t: sum(1 for s in samples if s["task"] == t) for t in ("math", "actions", "code")}
    assert counts == {"math": 51, "actions": 61, "code": 27}
    assert all(len(s["shards"]) >= 6 for s in samples)
    assert all(s["source"] == "humaneval" for s in samples if s["task"] == "code")


def test_dry_run_mtosc_preset(in_harness):
    from convcontract.eval import run

    run_name = f"pytest-mtosc-{uuid.uuid4().hex[:8]}"
    run_dir = config.RESULTS_DIR / run_name
    try:
        run.main(["--dry-run", "--preset", "mtosc", "--workers", "4", "--run-name", run_name])
        cfg = json.loads((run_dir / "invocations.jsonl").read_text().splitlines()[-1])
        assert cfg["models"] == ["openai/gpt-4.1-nano"] and cfg["temperature"] == 0.01 and cfg["frequency_penalty"] == 1.0
        assert llm._model_params["openai/gpt-4.1-nano"] == {"frequency_penalty": 1.0}
        assert not (run_dir / "errors.jsonl").exists(), (run_dir / "errors.jsonl").read_text()
        records = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
        assert len(records) == 3 * 2 * 2  # tasks x instructions x arms
        assert {r["arm"] for r in records} == {"sharded", "contract"}
        summary = json.loads((run_dir / "summary.json").read_text())
        assert {m["task"] for m in summary["mtosc"]} == {"math", "actions", "code"}
    finally:
        llm._model_params.clear()
        shutil.rmtree(run_dir, ignore_errors=True)


def test_always_answer_baseline_gets_the_instruction(in_harness):
    from convcontract.eval import run

    run_name = f"pytest-answer-{uuid.uuid4().hex[:8]}"
    run_dir = config.RESULTS_DIR / run_name
    try:
        run.main(["--dry-run", "--tasks", "math", "--arms", "sharded", "sharded-answer", "--run-name", run_name])
        traces = {json.loads(line)["key"]: json.loads(line)["trace"] for line in (run_dir / "traces.jsonl").read_text().splitlines()}
        for key, trace in traces.items():
            assert ("Do not ask questions" in trace[0]["content"]) == ("|sharded-answer|" in key)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_math_grading_accepts_numerically_equal_answers(in_harness):
    from tasks import get_task

    sample = next(d for d in json.load(open(harness.DATASET)) if d["task_id"] == "sharded-GSM8K/1207")  # gold 64
    task = get_task("math")
    assert task.evaluator_function("64.00", sample)["score"] == 1.0
    assert task.evaluator_function("$64", sample)["score"] == 1.0
    assert task.evaluator_function("64.5", sample)["score"] == 0.0


def test_dry_run_claude_preset(in_harness):
    from convcontract.eval import run

    run_name = f"pytest-claude-{uuid.uuid4().hex[:8]}"
    run_dir = config.RESULTS_DIR / run_name
    try:
        run.main(["--dry-run", "--preset", "claude", "--workers", "4", "--run-name", run_name])
        cfg = json.loads((run_dir / "invocations.jsonl").read_text().splitlines()[-1])
        assert cfg["user_model"] == cfg["system_model"] == "claude-haiku-5-5" and cfg["claude_thinking"] == "off"
        assert not (run_dir / "errors.jsonl").exists(), (run_dir / "errors.jsonl").read_text()
        records = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
        assert len(records) == 2 * 2 * 4  # models x instructions x arms
        assert {r["task"] for r in records} == {"actions"}
        traces = {json.loads(line)["key"]: json.loads(line)["trace"] for line in (run_dir / "traces.jsonl").read_text().splitlines()}
        for key, trace in traces.items():
            if "|user-only|" in key:
                prompts = [m["content"]["executor_prompt"] for m in trace if m["role"] == "log" and m["content"]["type"] == "contract"]
                assert "1. " in prompts[-1] and "Requirements" not in prompts[-1]
        summary = json.loads((run_dir / "summary.json").read_text())
        assert ("contract", "user-only") in {(c["arm"], c["baseline"]) for c in summary["comparisons"]}
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_tool_call_grading_accepts_json_style_booleans(in_harness):
    """The Claude pilot's Haiku answer was right but wrote `sort_by_rating=true`, which the parser read as a string."""
    from tasks import get_task

    sample = next(d for d in json.load(open(harness.DATASET)) if d["task_id"] == "sharded-BFCL/parallel_5")
    call = '[streaming_services.shows_list_and_ratings(streaming_service="Netflix", show_list=["Friends"], sort_by_rating={b}), streaming_services.shows_list_and_ratings(streaming_service="Hulu", show_list=["The Office", "Stranger Things"], sort_by_rating={b})]'
    task = get_task("actions")
    assert task.evaluator_function(call.format(b="True"), sample)["is_correct"]
    assert task.evaluator_function(call.format(b="true"), sample)["is_correct"]
    assert not task.evaluator_function(call.format(b="false"), sample)["is_correct"]
