"""Run the comparison: every arm, on the same instructions, same models, same simulated user.

    uv run python -m convcontract.eval.run --tasks math code --limit 20 --n-runs 10
    uv run python -m convcontract.eval.run --preset mtosc --n-runs 3   # comparable to MT-OSC's published Table 5
    uv run python -m convcontract.eval.run --dry-run      # fake LLM, no API key, checks wiring

Presets fill any option you don't pass explicitly:
    mtosc   gpt-4.1-nano with MT-OSC's chat settings (temperature 0.01, frequency_penalty 1), llama-3.3-70b
            extractor (MT-OSC's condenser model), math/actions/code with 6+ shards, HumanEval-only code,
            arms sharded + contract
    claude  Claude Haiku 5.5 and Sonnet 5.5, each its own extractor, Haiku 5.5 as simulated user and grader
            helper, thinking off, tool use (BFCL) with 6+ shards, arms sharded, sharded-answer, user-only,
            contract. Claude models take no temperature (non-default values are rejected), so
            --temperature has no effect on them.

Arms:
    full              fully specified original instruction, single turn (paper's FULL)
    concat            all shards in one prompt, single turn (paper's CONCAT; the ceiling we aim for)
    sharded           raw multi-turn conversation (paper's SHARDED; the baseline)
    sharded-answer    the baseline told to always give a final answer and never ask (same rule the contract's
                      executor gets), to separate "never stalls" from the architecture itself
    user-only         ablation: the contract's executor and instructions, but it sees the user's messages verbatim
                      instead of the contract (no extractor). Never sees its own replies either
    contract          contract layer, always answers
    contract-clarify  contract layer, may ask one clarifying question when the extractor says info is missing

Results go to results/<run-name>/: results.jsonl (one line per conversation), traces.jsonl (full transcripts),
invocations.jsonl (settings of each command that wrote to the run). Re-running with the same --run-name resumes
(finished conversations are skipped); adding arms or tasks to an existing run extends it.
"""
import argparse
import json
import random
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from convcontract import config, llm
from convcontract.eval import harness

ARMS = ("full", "concat", "sharded", "sharded-answer", "user-only", "contract", "contract-clarify")
ALWAYS_ANSWER = (config.PROMPTS_DIR / "always_answer.txt").read_text()


def summarize_trace(trace: list[dict]) -> dict:
    """Turn counts, when the first answer attempt happened, and total cost across every call in a conversation."""
    user_turns, attempts, first_answer_turn, questions, cost, last_type = 0, 0, None, 0, 0.0, None
    for msg in trace:
        cost += msg.get("cost_usd") or 0.0
        if msg["role"] == "user":
            user_turns += 1
        elif msg["role"] == "log" and msg["content"]["type"] == "system-verification":
            last_type = msg["content"]["response"].get("response_type")
            if last_type == "answer_attempt":
                attempts += 1
                first_answer_turn = first_answer_turn or user_turns
        elif msg["role"] == "log" and msg["content"]["type"] == "contract" and msg["content"]["decision"] == "ask":
            questions += 1
    # The last reply wasn't an answer (a question, a partial answer, a hedge): the conversation ended without one.
    ended_without_answer = last_type != "answer_attempt"
    return {"n_user_turns": user_turns, "n_answer_attempts": attempts, "first_answer_turn": first_answer_turn, "n_questions": questions, "ended_without_answer": ended_without_answer, "cost_usd": round(cost, 6)}


def run_one(todo: dict, run_dir: Path) -> tuple[dict, list | None]:
    from simulator_full import ConversationSimulatorFull
    from simulator_sharded import ConversationSimulatorSharded

    from convcontract.eval.simulator_contract import ConversationSimulatorContract

    sample, arm, model = todo["sample"], todo["arm"], todo["model"]
    t0 = time.time()
    trace = None
    if arm in ("full", "concat"):
        # The single-turn simulator keeps its trace local, so let it write its own log.
        sim = ConversationSimulatorFull(sample, assistant_model=model, system_model=todo["system_model"], run_concat=(arm == "concat"), temperature=todo["temperature"], dataset_fn=harness.DATASET, log_folder=str(run_dir / "harness_logs"))
        is_correct, score = sim.run(save_log=True)
        stats = {"n_user_turns": 1, "n_answer_attempts": 1, "first_answer_turn": 1, "n_questions": 0, "ended_without_answer": False, "cost_usd": None}
    else:
        if arm in ("sharded", "sharded-answer"):
            sim = ConversationSimulatorSharded(sample, assistant_model=model, system_model=todo["system_model"], user_model=todo["user_model"], assistant_temperature=todo["temperature"], dataset_fn=harness.DATASET)
            if arm == "sharded-answer":
                sim.system_message += "\n\n" + ALWAYS_ANSWER
                sim.trace[0]["content"] = sim.system_message
            is_correct, score = sim.run(save_log=False)
        else:
            policy = {"contract-clarify": "clarify", "user-only": "user-only"}.get(arm, "answer")
            sim = ConversationSimulatorContract(sample, assistant_model=model, system_model=todo["system_model"], user_model=todo["user_model"], extractor_model=todo["extractor_model"], policy=policy, assistant_temperature=todo["temperature"])
            is_correct, score = sim.run()
        trace = sim.trace
        stats = summarize_trace(trace)

    record = {
        "key": todo["key"],
        "model": model,
        "task": sample["task"],
        "task_id": sample["task_id"],
        "arm": arm,
        "run_idx": todo["run_idx"],
        "score": score,
        "is_correct": bool(is_correct),
        "seconds": round(time.time() - t0, 1),
        **stats,
    }
    return record, trace


def select_samples(tasks, limit, seed, min_shards=0, code_source="all", max_shards=None):
    with open(harness.DATASET) as f:
        data = json.load(f)
    selected = []
    for task in tasks:
        pool = sorted(
            (d for d in data if d["task"] == task and min_shards <= len(d["shards"]) <= (max_shards or 99) and (task != "code" or code_source == "all" or d.get("source") == code_source)),
            key=lambda d: d["task_id"],
        )
        if not pool:
            raise ValueError(f"No samples for task {task!r} with these filters")
        if limit and limit < len(pool):
            pool = random.Random(seed).sample(pool, limit)
        selected.extend(pool)
    return selected


# Options a preset may fill. None in argparse means "not passed"; explicit flags always win.
DEFAULTS = {
    "arms": ["concat", "sharded", "contract"],
    "models": [config.EXECUTOR_MODEL],
    "extractor_model": None,
    "tasks": ["math", "code"],
    "temperature": 1.0,
    "frequency_penalty": None,
    "min_shards": 0,
    "code_source": "all",
    "extractor_providers": None,
    "user_model": config.USER_MODEL,
    "system_model": config.SYSTEM_MODEL,
    "claude_thinking": llm.CLAUDE_THINKING,
}
PRESETS = {
    "mtosc": {
        "arms": ["sharded", "contract"],
        "models": ["openai/gpt-4.1-nano"],
        "extractor_model": "meta-llama/llama-3.3-70b-instruct",
        "tasks": ["math", "actions", "code"],
        "temperature": 0.01,
        "frequency_penalty": 1.0,
        "min_shards": 6,
        "code_source": "humaneval",
        # Fast in the pilots, support JSON mode, ~$0.20/M. Default latency routing picked a $0.71/M provider;
        # the cheapest ($0.10/M) sometimes held requests for minutes.
        "extractor_providers": ["Parasail", "AkashML"],
    },
    "claude": {
        "arms": ["sharded", "sharded-answer", "user-only", "contract"],
        "models": ["claude-haiku-5-5", "claude-sonnet-5-5"],
        "tasks": ["actions"],
        "min_shards": 6,
        "user_model": "claude-haiku-5-5",
        "system_model": "claude-haiku-5-5",
        "claude_thinking": "off",
    },
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None, help="fill unset options from a preset (see above)")
    parser.add_argument("--arms", nargs="+", default=None, choices=ARMS, help="default: concat sharded contract")
    parser.add_argument("--models", nargs="+", default=None, help=f"assistant/executor models, OpenRouter ids (default: {config.EXECUTOR_MODEL})")
    parser.add_argument("--extractor-model", default=None, help="contract extractor model (default: same as each executor model)")
    parser.add_argument("--user-model", default=None, help=f"simulated user (default: {config.USER_MODEL})")
    parser.add_argument("--system-model", default=None, help=f"harness helper that classifies replies and extracts answers (default: {config.SYSTEM_MODEL})")
    parser.add_argument("--claude-thinking", choices=["off", "adaptive"], default=None, help="thinking for every Claude call (default: off)")
    parser.add_argument("--tasks", nargs="+", default=None, help="math code database actions (default: math code; database needs the Spider DBs, see README)")
    parser.add_argument("--min-shards", type=int, default=None, help="only instructions split into at least this many pieces")
    parser.add_argument("--code-source", choices=["all", "humaneval", "lcb_easy", "lcb_medium"], default=None, help="restrict the code task to one source")
    parser.add_argument("--limit", type=int, default=None, help="instructions per task (random, fixed by --seed)")
    parser.add_argument("--n-runs", type=int, default=10, help="simulations per instruction per arm")
    parser.add_argument("--temperature", type=float, default=None, help="assistant temperature (default 1.0, as in Laban et al.)")
    parser.add_argument("--frequency-penalty", type=float, default=None, help="assistant frequency_penalty, sent on every call to the assistant models")
    parser.add_argument("--extractor-providers", nargs="+", default=None, help="OpenRouter providers to try first for the extractor model (falls back to others)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dry-run", action="store_true", help="fake LLM backend: 2 instructions per task, 1 run, no API calls")
    args = parser.parse_args(argv)
    preset = PRESETS.get(args.preset, {})
    for key, default in DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, preset.get(key, default))
    llm.CLAUDE_THINKING = args.claude_thinking
    if args.frequency_penalty is not None:
        for model in args.models:
            llm.set_model_params(model, frequency_penalty=args.frequency_penalty)
    if args.extractor_providers:
        for model in {args.extractor_model or m for m in args.models}:
            llm.set_model_params(model, provider={"order": args.extractor_providers, "allow_fallbacks": True})

    if args.dry_run:
        from convcontract.fake_llm import fake_backend

        llm.set_backend(fake_backend)
        args.limit, args.n_runs = args.limit or 2, 1
        args.run_name = args.run_name or "dry-run"
    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = config.RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    harness.setup()  # chdir into the harness; absolute paths only from here
    if "database" in args.tasks and not Path("data/spider/databases").exists():
        raise SystemExit("The database task needs the Spider databases in external/lost_in_conversation/data/spider/databases (see README).")

    samples = select_samples(args.tasks, args.limit, args.seed, args.min_shards, args.code_source)
    results_path, traces_path, errors_path = run_dir / "results.jsonl", run_dir / "traces.jsonl", run_dir / "errors.jsonl"
    done = set()
    if results_path.exists():
        done = {json.loads(line)["key"] for line in results_path.read_text().splitlines() if line.strip()}

    todos = []
    for model in args.models:
        for sample in samples:
            for arm in args.arms:
                # full/concat are single-turn: same number of runs, so percentiles are comparable.
                for run_idx in range(args.n_runs):
                    key = f"{model}|{sample['task_id']}|{arm}|{run_idx}"
                    if key in done:
                        continue
                    todos.append({"key": key, "model": model, "sample": sample, "arm": arm, "run_idx": run_idx, "temperature": args.temperature, "extractor_model": args.extractor_model or model, "user_model": args.user_model, "system_model": args.system_model})
    random.Random(args.seed).shuffle(todos)

    # One line per invocation: a resumed or extended run keeps the settings of every earlier invocation.
    with open(run_dir / "invocations.jsonl", "a") as f:
        f.write(json.dumps(vars(args) | {"n_samples": len(samples), "started": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    print(f"{len(samples)} instructions x {len(args.arms)} arms x {args.n_runs} runs x {len(args.models)} models; {len(done)} done, {len(todos)} to run -> {run_dir}")

    lock = threading.Lock()
    n_errors = 0

    def work(todo):
        nonlocal n_errors
        try:
            record, trace = run_one(todo, run_dir)
        except Exception:
            with lock, open(errors_path, "a") as f:
                n_errors += 1
                f.write(json.dumps({"key": todo["key"], "error": traceback.format_exc()}) + "\n")
            return
        with lock:
            with open(results_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            if trace is not None:
                with open(traces_path, "a") as f:
                    f.write(json.dumps({"key": record["key"], "trace": trace}) + "\n")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(tqdm(pool.map(work, todos), total=len(todos)))

    if n_errors:
        print(f"{n_errors} conversations errored; see {errors_path} (re-run the same command to retry them)")

    from convcontract.eval import analyze

    analyze.main([str(run_dir)])


if __name__ == "__main__":
    main()
