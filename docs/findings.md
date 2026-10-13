# Findings

## The question

When a user gives an AI assistant a task piece by piece over several messages, accuracy drops sharply compared
with giving everything at once (Laban et al., ICLR 2026: −39% on average). Most chat systems handle this by
re-sending the whole conversation every turn, including the assistant's own earlier replies.

We tested a different design, the **conversation contract**. A helper model keeps a running list of what the
user has asked for. The main model sees only that list, never the chat or its own earlier replies, and writes a
fresh answer from it every turn.

## Results

Model under test: gpt-4.1-nano. Tasks given over 6 or more messages. 3 runs per problem. Each problem compared
one to one across approaches. The range in brackets is where the true difference likely lies (95% confidence).

### Tool use (BFCL: calling the right functions with the right arguments), 61 problems

| Approach | Correct | Ended without answering | Correct when it answered |
|---|---|---|---|
| Normal chat (sees whole conversation) | 40% | 24% | 53% |
| Normal chat + "always answer, don't ask" | 45% | 16% | 54% |
| **Contract** | **73%** | **0%** | **73%** |

- **Contract vs normal chat: +33 points** [+22, +43].
- **Contract vs normal chat told to always answer: +28 points** [+16, +40].

Telling the normal chat not to ask barely helped (+5, could be chance), and it still stalled in 16% of
conversations. The contract's gain comes mainly from **better answers** (73% vs 53–54% correct when answering),
not just from never stalling.

### Coding (HumanEval), 27 problems

Normal chat 58% → contract 64%: **+6 points** [−9, +22]. Too few problems to tell this apart from chance.

### Math word problems (GSM8K), 51 problems

Normal chat 76% → contract 75%: **no difference**. (Re-graded numerically; the harness scored three correct
answers written as `64.00` as wrong. See `method.md`.)

### Cost

About 2× per conversation ($0.004 vs $0.002), because of one extra model call per turn to update the list.

## What we learned

1. **Rebuilding the request from a clean list every turn gives much better tool-use answers** (73% vs 53–54%
   correct when answering). This is the core effect, and it survives the "always answer" control. *Why* isn't
   isolated yet. Two likely reasons: the model no longer anchors on its own earlier attempts (the original paper's
   main explanation, and we saw it happen when a previous answer was accidentally fed back to our executor), and
   the requirements arrive organized in one place instead of scattered across the chat.
2. **The model stalls when left to decide.** Normal chat ended a quarter of tool-use conversations with a
   question or refusal, and an instruction not to only partly fixed that. Making the ask-or-answer decision
   outside the model removes it.
3. **The helper that writes the list is the weak point.** When it paraphrased ("a $20 picture frame" became "a
   $20 gift"), facts were lost. Requiring the user's own wording raised facts kept from about 90% to 97%.
4. **Not every task benefits.** Math showed no gain. A likely reason: in normal chat the model works the problem out
   as each fact arrives, while the contract makes it do all the arithmetic in one go. Occasionally a fact still
   gets lost in the list, too.
5. **Reliability of the plumbing matters as much as the idea.** Grading quirks, slow or broken providers, and
   malformed helper output each distorted results until fixed (see `method.md`).

## Recommendations for agent architecture

- **For tool-calling agents, rebuild each request from a structured record of the user's requirements, and don't
  feed the model its own previous attempts.** Strongest evidence here: +28 to +33 points. (Tested as a package;
  which of the two parts matters more is the next thing to measure.)
- **Decide "ask or answer" in the orchestration layer, not inside the model.** Whether asking is right depends on
  the product; this benchmark penalizes it.
- **Keep the user's wording when recording requirements, and validate the helper's output.** Extraction errors
  are the main way this design fails.
- **Pass the previous answer back only when the user's message can't be acted on without it** ("fix line 3").
- **Don't expect gains on step-by-step calculation tasks.**

## How this compares with MT-OSC

MT-OSC (Singh et al. 2026) is another fix for the same problem: it periodically shortens the chat history. On the
same model and tool-use benchmark it reports **+8 points**; the contract gained **+33** (+28 against the stronger
baseline). This comparison is **indicative, not head-to-head**:

- We used MT-OSC's published numbers and didn't run it ourselves.
- We covered 3 of its 10 datasets and 1 of its 13 models.
- Our normal-chat baseline differs from theirs (tool use 40% vs 51%, coding 58% vs 70%, math 76% vs 70%), so the
  setups aren't identical.

On coding and math, MT-OSC reports larger gains than we measured (+12 and +7).

## Limits

- One small model. A stronger model (Gemini 3.8 Flash, in an early pilot) scored 100% even with normal chat, so
  the gains may shrink as models improve.
- 3 runs per problem; coding has only 27 problems.
- Simulated users who add information and never change their minds. Correcting earlier requirements ("actually,
  make it 5") is untested, and it's where this design should differ most from history-based approaches.

## Where the numbers come from

`results/mtosc-main/` (`results.jsonl`, `traces.jsonl`, `summary.json`). Re-print the tables with
`uv run python -m convcontract.eval.analyze results/mtosc-main`. Extraction-fidelity numbers:
`results/extraction/`. Total spend for the whole project, including pilots: $4.73.
