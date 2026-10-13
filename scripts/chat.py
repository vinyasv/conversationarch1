"""Chat with the contract layer and watch the contract change each turn.

    uv run python scripts/chat.py
    uv run python scripts/chat.py --policy answer --system "You write SQL for a Postgres database."

Commands: /contract (print full contract), /prompt (last executor prompt), /reset, /quit
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from convcontract import config  # noqa: E402
from convcontract.layer import POLICIES, ContractAssistant  # noqa: E402


def show_contract(assistant):
    c = assistant.contract
    print(f"  goal: {c.goal or '-'}")
    for r in c.requirements:
        mark = {"active": " ", "superseded": "~", "removed": "x"}[r.status]
        print(f"  {mark} {r.id} (turn {r.turn}): {r.text}" + (f"  -> {r.superseded_by}" if r.superseded_by else ""))
    for m in c.material:
        print(f"    {m.id} (turn {m.turn}): {len(m.content)} chars of material")
    for q in c.open_questions:
        print(f"  ? {q}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=config.EXECUTOR_MODEL)
    parser.add_argument("--extractor-model", default=None)
    parser.add_argument("--policy", default="clarify", choices=POLICIES)
    parser.add_argument("--system", default="", help="system prompt / deployment context")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    def new():
        return ContractAssistant(system_prompt=args.system, executor_model=args.model, extractor_model=args.extractor_model or args.model, executor_temperature=args.temperature, policy=args.policy)

    assistant, last_prompt, total = new(), None, 0.0
    print(f"executor={args.model} extractor={args.extractor_model or args.model} policy={args.policy}. /contract /prompt /reset /quit")
    while True:
        try:
            message = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not message:
            continue
        if message == "/quit":
            break
        if message == "/reset":
            assistant, last_prompt = new(), None
            print("(new conversation)")
            continue
        if message == "/contract":
            print(json.dumps(assistant.contract.to_dict(), indent=2))
            continue
        if message == "/prompt":
            print(last_prompt or "(no executor call yet)")
            continue

        turn = assistant.respond(message)
        total += turn.cost_usd
        last_prompt = turn.executor_prompt or last_prompt
        print(f"\n[contract after turn {assistant.contract.turn}]")
        show_contract(assistant)
        for w in turn.warnings:
            print(f"  ! {w}")
        label = "question" if turn.decision == "ask" else "answer"
        print(f"\nassistant ({label}, ${turn.cost_usd:.4f}, total ${total:.4f})>\n{turn.text}")


if __name__ == "__main__":
    main()
