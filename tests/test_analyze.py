from convcontract.eval.analyze import mtosc_comparison


def rec(task_id, arm, score, task="math", model="openai/gpt-4.1-nano"):
    return {"model": model, "task": task, "task_id": task_id, "arm": arm, "score": score}


def test_mtosc_comparison_uses_published_numbers_and_paired_gain():
    groups = {
        ("openai/gpt-4.1-nano", "math", "sharded"): [rec("a", "sharded", 0.0), rec("a", "sharded", 1.0), rec("b", "sharded", 1.0)],
        ("openai/gpt-4.1-nano", "math", "contract"): [rec("a", "contract", 1.0), rec("a", "contract", 1.0), rec("b", "contract", 1.0)],
    }
    [m] = mtosc_comparison("openai/gpt-4.1-nano", groups)
    assert (m["label"], m["their_baseline"], round(m["their_gain"])) == ("GSM8K", 70.0, 7)
    assert m["our_baseline"] == 75.0  # mean of per-instruction means: (50 + 100) / 2
    assert m["our_gain"] == 25.0 and m["n"] == 2


def test_mtosc_comparison_skips_models_without_published_results():
    groups = {("some/model", "math", "sharded"): [rec("a", "sharded", 1.0, model="some/model")], ("some/model", "math", "contract"): [rec("a", "contract", 1.0, model="some/model")]}
    assert mtosc_comparison("some/model", groups) == []
