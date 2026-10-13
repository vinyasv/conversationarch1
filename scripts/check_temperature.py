"""Check that a model actually honors `temperature` through OpenRouter.

OpenRouter lists a model as supporting temperature if any provider does, and some models accept the
parameter but ignore or clamp it. The paper's reliability results depend on sampling at T=1.0, so check
before trusting a model's numbers.

    uv run python scripts/check_temperature.py
    uv run python scripts/check_temperature.py --model meta-llama/llama-3.3-70b-instruct --n 8
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from convcontract import config, llm  # noqa: E402

PROMPT = "Invent a name for a fictional coffee shop. Reply with the name only."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=config.EXECUTOR_MODEL)
    parser.add_argument("--n", type=int, default=6)
    args = parser.parse_args()

    distinct, cost = {}, 0.0
    for temperature in (0.0, 1.0):
        outputs = []
        for _ in range(args.n):
            r = llm.chat([{"role": "user", "content": PROMPT}], args.model, temperature=temperature)
            outputs.append(r.text.strip())
            cost += r.cost_usd
        distinct[temperature] = len(set(outputs))
        print(f"T={temperature}: {distinct[temperature]}/{args.n} distinct -> {sorted(set(outputs))}")

    if distinct[1.0] > distinct[0.0]:
        verdict = "temperature has an effect"
    elif distinct[0.0] > 1:
        verdict = "T=0 is already varied (a thinking model, or temperature is ignored); treat reliability comparisons with care"
    else:
        verdict = "no measurable effect at this sample size; try a larger --n"
    print(f"\n{args.model}: {verdict}. Cost ${cost:.4f}")


if __name__ == "__main__":
    main()
