"""Summarize a run with the paper's metrics, plus the paired contract-vs-baseline comparison.

    uv run python -m convcontract.eval.analyze results/<run-name>

Per instruction, over its N runs (scores scaled to 0-100):
    P  (average)       mean score
    A90 (aptitude)     90th-percentile score: how well it does on a good run
    U  (unreliability) 90th minus 10th percentile: how far apart good and bad runs are
Each metric is averaged across instructions, as in Laban et al. A conversation with no answer attempt scores 0.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from convcontract.eval.mtosc_reference import TABLE5, TASK_LABELS

BASELINE, CEILING = "sharded", "concat"
# baseline -> arms compared against it. user-only vs contract isolates the extractor (the ablation).
COMPARISONS = {
    "sharded": ("sharded-answer", "user-only", "contract", "contract-clarify"),
    "sharded-answer": ("user-only", "contract", "contract-clarify"),
    "user-only": ("contract",),
}
ORDER = ["full", "concat", "sharded", "sharded-answer", "user-only", "contract", "contract-clarify"]


def load(run_dir: Path) -> list[dict]:
    path = run_dir / "results.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def per_instruction(records):
    """{instruction: [scores 0-100]}"""
    out = defaultdict(list)
    for r in records:
        out[r["task_id"]].append(100.0 * (r["score"] or 0.0))
    return out


def metrics(by_instr: dict) -> dict:
    scores = [np.array(v) for v in by_instr.values()]
    p90 = [np.percentile(s, 90) for s in scores]
    p10 = [np.percentile(s, 10) for s in scores]
    return {
        "P": float(np.mean([s.mean() for s in scores])),
        "A90": float(np.mean(p90)),
        "U": float(np.mean(np.subtract(p90, p10))),
        "n_instr": len(scores),
        "runs_per_instr": round(float(np.mean([len(s) for s in scores])), 1),
    }


def paired_delta(a: dict, b: dict, n_boot=2000, seed=0):
    """Mean (b - a) of per-instruction averages over shared instructions, with a bootstrap 95% CI."""
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    diffs = np.array([np.mean(b[i]) - np.mean(a[i]) for i in shared])
    rng = np.random.default_rng(seed)
    boots = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    return {"delta": float(diffs.mean()), "ci_low": float(np.percentile(boots, 2.5)), "ci_high": float(np.percentile(boots, 97.5)), "n": len(shared)}


def mtosc_comparison(model: str, groups: dict) -> list[dict]:
    """Our baseline vs MT-OSC's published baseline (the anchor), and our contract gain vs their MT-OSC gain."""
    reference = TABLE5.get(model, {})
    out = []
    for task, (their_base, their_mtosc) in reference.items():
        if (model, task, BASELINE) not in groups or (model, task, "contract") not in groups:
            continue
        base = per_instruction(groups[(model, task, BASELINE)])
        d = paired_delta(base, per_instruction(groups[(model, task, "contract")]))
        if d is None:
            continue
        out.append({
            "model": model, "task": task, "label": TASK_LABELS[task],
            "their_baseline": 100 * their_base, "our_baseline": metrics(base)["P"],
            "their_gain": 100 * (their_mtosc - their_base), "our_gain": d["delta"],
            "ci_low": d["ci_low"], "ci_high": d["ci_high"], "n": d["n"],
        })
    return out


def fmt(x, digits=1):
    return "-" if x is None else f"{x:.{digits}f}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)

    records = load(args.run_dir)
    groups = defaultdict(list)
    for r in records:
        groups[(r["model"], r["task"], r["arm"])].append(r)

    summary = {"arms": [], "comparisons": []}
    for model in sorted({k[0] for k in groups}):
        print(f"\n## {model}\n")
        print("| task | arm | P | A90 | U | instr | runs/instr | 1st answer turn | questions | ended w/o answer | $/conv |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        tasks = sorted({k[1] for k in groups if k[0] == model})
        for task in tasks:
            arms = sorted({k[2] for k in groups if k[:2] == (model, task)}, key=lambda a: ORDER.index(a) if a in ORDER else 99)
            for arm in arms:
                recs = groups[(model, task, arm)]
                m = metrics(per_instruction(recs))
                first = [r["first_answer_turn"] for r in recs if r.get("first_answer_turn")]
                costs = [r["cost_usd"] for r in recs if r.get("cost_usd") is not None]
                m |= {
                    "model": model, "task": task, "arm": arm,
                    "first_answer_turn": float(np.mean(first)) if first else None,
                    "questions": float(np.mean([r.get("n_questions", 0) for r in recs])),
                    "ended_without_answer_pct": 100.0 * float(np.mean([bool(r.get("ended_without_answer")) for r in recs])),
                    "cost_usd": float(np.mean(costs)) if costs else None,
                }
                summary["arms"].append(m)
                print(f"| {task} | {arm} | {fmt(m['P'])} | {fmt(m['A90'])} | {fmt(m['U'])} | {m['n_instr']} | {m['runs_per_instr']} | {fmt(m['first_answer_turn'])} | {fmt(m['questions'], 2)} | {fmt(m['ended_without_answer_pct'], 0)}% | {fmt(m['cost_usd'], 4)} |")

        comparisons = []
        for task in tasks:
            ceiling = per_instruction(groups[(model, task, CEILING)]) if (model, task, CEILING) in groups else None
            for baseline, arms in COMPARISONS.items():
                if (model, task, baseline) not in groups:
                    continue
                base = per_instruction(groups[(model, task, baseline)])
                for arm in arms:
                    if (model, task, arm) not in groups:
                        continue
                    d = paired_delta(base, per_instruction(groups[(model, task, arm)]))
                    if d is None:
                        continue
                    gap = None
                    if ceiling and baseline == BASELINE:
                        g = paired_delta(base, ceiling)
                        if g and g["delta"] > 0:
                            gap = 100.0 * d["delta"] / g["delta"]
                    d |= {"model": model, "task": task, "arm": arm, "baseline": baseline, "gap_closed_pct": gap}
                    comparisons.append(d)
        if comparisons:
            print(f"\nPaired comparisons (by instruction; 95% bootstrap CI; gap closed = share of the {BASELINE}->{CEILING} gap recovered)\n")
            print("| task | arm | vs | delta P | 95% CI | instr | gap closed |")
            print("|---|---|---|---|---|---|---|")
            for d in comparisons:
                gap = "-" if d["gap_closed_pct"] is None else f"{d['gap_closed_pct']:.0f}%"
                print(f"| {d['task']} | {d['arm']} | {d['baseline']} | {d['delta']:+.1f} | [{d['ci_low']:+.1f}, {d['ci_high']:+.1f}] | {d['n']} | {gap} |")
        summary["comparisons"] += comparisons

        mtosc = mtosc_comparison(model, groups)
        if mtosc:
            print(f"\nvs MT-OSC published results (Singh et al. 2026, Table 5: 6+ turn samples, {model})")
            print("Anchor: our baseline should land near theirs for the gains to be comparable. Their numbers have no CI (no per-sample data published).\n")
            print("| task | their baseline | our baseline | their MT-OSC gain | our contract gain | 95% CI | instr |")
            print("|---|---|---|---|---|---|---|")
            for m in mtosc:
                print(f"| {m['label']} | {m['their_baseline']:.0f} | {m['our_baseline']:.1f} | {m['their_gain']:+.0f} | {m['our_gain']:+.1f} | [{m['ci_low']:+.1f}, {m['ci_high']:+.1f}] | {m['n']} |")
            summary["mtosc"] = summary.get("mtosc", []) + mtosc

    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {args.run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
