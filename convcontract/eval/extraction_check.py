"""Extraction fidelity: does the contract keep everything the user said?

Feeds each instruction's shards to the extractor in order (no simulated user, executor or grader), then a
judge model checks, shard by shard, whether its information survives in the task the executor would see.
Cheap (a few cents per prompt version), so extractor prompts can be compared before any conversation runs.

    uv run python -m convcontract.eval.extraction_check --prompts extractor_v1.txt extractor.txt --limit 10
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from convcontract import config, llm
from convcontract.eval import harness
from convcontract.eval.run import select_samples

JUDGE_PROMPT = """You check whether a task description kept everything a user said.

The user stated these things, one per numbered line:
<statements>
[[STATEMENTS]]
</statements>

The task description written from them:
<task>
[[TASK]]
</task>

For each statement, decide whether every detail in it is present in the task description with the same meaning:
entities and objects, numbers and units, qualifiers (where, which, who, when, how many) and conditions.
- Different wording with identical meaning counts as preserved.
- A broader, vaguer or differently scoped restatement counts as NOT preserved (e.g. a specific object turned into a generic one, or a detail attached to the wrong item).
- A detail missing entirely counts as NOT preserved.

Output only JSON: {"verdicts": [{"id": <statement number>, "preserved": true|false, "missing": "<what was lost or changed; empty if preserved>"}]}"""


def check_one(sample: dict, prompt_name: str, prompt_text: str, extractor_model: str, judge_model: str) -> dict:
    from tasks import get_task

    from convcontract.layer import ContractAssistant

    task = get_task(sample["task"])
    assistant = ContractAssistant(system_prompt=task.generate_system_prompt(sample), extractor_model=extractor_model, extractor_prompt=prompt_text, policy="answer")
    cost, failed_turns = 0.0, 0
    for shard in sample["shards"]:
        ops, _, c = assistant.update_contract(shard["shard"])
        cost += c
        failed_turns += bool(ops.get("extractor_failed"))

    rendered = assistant.contract.render_task()
    statements = "\n".join(f"{i}. {s['shard']}" for i, s in enumerate(sample["shards"], 1))
    verdict, response = llm.chat_json([{"role": "user", "content": JUDGE_PROMPT.replace("[[STATEMENTS]]", statements).replace("[[TASK]]", rendered)}], judge_model, temperature=0.0)
    cost += response.cost_usd
    by_id = {v.get("id"): v for v in verdict.get("verdicts", []) if isinstance(v, dict)}
    shards = []
    for i, s in enumerate(sample["shards"], 1):
        v = by_id.get(i, {"preserved": False, "missing": "(judge gave no verdict)"})
        shards.append({"shard": s["shard"], "preserved": bool(v.get("preserved")), "missing": v.get("missing", "")})
    return {
        "prompt": prompt_name, "task": sample["task"], "task_id": sample["task_id"], "n_shards": len(shards),
        "n_preserved": sum(s["preserved"] for s in shards), "extractor_failed_turns": failed_turns,
        "cost_usd": round(cost, 6), "shards": shards, "rendered": rendered,
    }


def summarize(records: list[dict]):
    prompts = list(dict.fromkeys(r["prompt"] for r in records))
    print("\n| prompt | task | instructions | shards kept | instructions fully kept | extractor fallbacks | $ |")
    print("|---|---|---|---|---|---|---|")
    for p in prompts:
        rows = [r for r in records if r["prompt"] == p]
        for task in sorted({r["task"] for r in rows}) + ["all"]:
            rs = [r for r in rows if task in ("all", r["task"])]
            kept = sum(r["n_preserved"] for r in rs) / sum(r["n_shards"] for r in rs)
            full = sum(r["n_preserved"] == r["n_shards"] for r in rs) / len(rs)
            print(f"| {p} | {task} | {len(rs)} | {100 * kept:.1f}% | {100 * full:.0f}% | {sum(r['extractor_failed_turns'] for r in rs)} | {sum(r['cost_usd'] for r in rs):.3f} |")

    if len(prompts) == 2:
        a, b = ({r["task_id"]: r["n_preserved"] == r["n_shards"] for r in records if r["prompt"] == p} for p in prompts)
        shared = set(a) & set(b)
        fixed = sum(1 for t in shared if b[t] and not a[t])
        broke = sum(1 for t in shared if a[t] and not b[t])
        print(f"\n{prompts[1]} vs {prompts[0]} on {len(shared)} shared instructions: {fixed} newly fully kept, {broke} newly lossy")

    print("\nLosses (up to 5 per prompt):")
    for p in prompts:
        losses = [(r["task_id"], s) for r in records if r["prompt"] == p for s in r["shards"] if not s["preserved"]]
        print(f"\n[{p}] {len(losses)} lost statements")
        for task_id, s in losses[:5]:
            print(f"  {task_id}: \"{s['shard']}\" -> {s['missing']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", nargs="+", default=["extractor.txt"], help="extractor prompt files in convcontract/prompts/")
    parser.add_argument("--extractor-model", default="meta-llama/llama-3.3-70b-instruct")
    parser.add_argument("--judge-model", default="openai/gpt-4.1-mini")
    parser.add_argument("--tasks", nargs="+", default=["math", "actions", "code"])
    # Default 4-5 pieces: disjoint from the comparison run (6+), so prompts aren't tuned on its test instructions.
    parser.add_argument("--min-shards", type=int, default=4)
    parser.add_argument("--max-shards", type=int, default=5)
    parser.add_argument("--code-source", default="humaneval")
    parser.add_argument("--limit", type=int, default=10, help="instructions per task")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(argv)

    prompts = {name: (config.PROMPTS_DIR / name).read_text() for name in args.prompts}
    run_dir = config.RESULTS_DIR / (args.run_name or f"extraction-{time.strftime('%Y%m%d-%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)

    harness.setup()
    samples = select_samples(args.tasks, args.limit, args.seed, args.min_shards, args.code_source, args.max_shards)
    jobs = [(s, name, text) for name, text in prompts.items() for s in samples]
    print(f"{len(samples)} instructions x {len(prompts)} prompts -> {run_dir}")

    def work(job):
        try:
            return check_one(job[0], job[1], job[2], args.extractor_model, args.judge_model)
        except Exception as e:
            return {"error": repr(e), "task_id": job[0]["task_id"], "prompt": job[1]}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(tqdm(pool.map(work, jobs), total=len(jobs)))
    records = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    with open(run_dir / "extraction.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    if errors:
        print(f"{len(errors)} errored: {errors[0]}")
    summarize(records)


if __name__ == "__main__":
    main()
