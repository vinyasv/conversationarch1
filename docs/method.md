# How the experiment works

## The benchmark

We use Microsoft's [Lost in Conversation](https://github.com/Microsoft/lost_in_conversation) harness (Laban et al.,
ICLR 2026), cloned unmodified into `external/lost_in_conversation/`. Each task is a normal benchmark problem split
into pieces ("shards"). A simulated user (gpt-4o-mini) reveals one piece per message, the assistant replies, and
the harness grades any reply that is an answer attempt. The conversation ends when an answer is correct or every
piece has been revealed.

Tasks used in the main run (only problems split into 6 or more pieces, as in MT-OSC's comparison table):

| Harness name | Benchmark | Problems | Graded by |
|---|---|---|---|
| `actions` | BFCL (tool/function calling) | 61 | Exact function-call match |
| `code` | HumanEval | 27 | Running unit tests |
| `math` | GSM8K | 51 | Final number |

## What is compared ("arms")

| Arm | What the model sees each turn |
|---|---|
| `sharded` | **Baseline.** The whole chat so far, including its own earlier replies |
| `sharded-answer` | The baseline plus the same "always give a final answer, don't ask" instruction our executor gets. Separates "never stalls" from the architecture |
| `user-only` | **Ablation.** The contract's executor and instructions, but it sees the user's messages word for word instead of the contract (no extractor). Like the contract, never its own earlier replies. Contract vs user-only isolates the extractor |
| `contract` | **Our design.** Only the task's instructions plus a requirements list built from the user's messages. Never its own earlier replies |
| `contract-clarify` | Our design, allowed to ask one clarifying question when the extractor reports missing information (not in the main run) |
| `full`, `concat` | Single-turn references from the paper: the original problem, or all pieces in one message (not in the main run) |

The simulated user, the check for whether a reply is an answer, answer extraction and grading are identical across arms.

## Settings (`--preset mtosc`)

Chosen to match MT-OSC's published setup so the numbers can be read side by side:

- Assistant model: `openai/gpt-4.1-nano`, temperature 0.01, frequency_penalty 1 (MT-OSC's chat settings)
- Extractor: `meta-llama/llama-3.3-70b-instruct` (MT-OSC's condenser model), routed to the Parasail or AkashML provider
- Simulated user and harness grading helper: `openai/gpt-4o-mini`, as in the original paper
- 3 runs per problem per arm

## Settings (`--preset claude`)

A second model family, to test whether the tool-use result holds beyond gpt-4.1-nano:

- Assistant models: `claude-haiku-5-5` and `claude-sonnet-5-5`, each its own extractor (removes the stronger-helper
  confound of the nano run, where Llama-3.3-70B extracted for nano)
- Simulated user and harness grading helper: `claude-haiku-5-5`. This changes the benchmark's user, so these
  results are compared within the Claude run, not with the nano run or the paper
- Thinking off for every call (`thinking: disabled` on Haiku, `between_tools` on Sonnet)
- No temperature: these models reject non-default sampling values, so runs vary and 3 runs per problem measure that
- Tool use (BFCL, 6+ shards, 61 problems); arms `sharded`, `sharded-answer`, `user-only`, `contract`
- A refused request is graded as a non-answer rather than dropped; no server-side fallback model, which would swap
  the model under test

## Metrics

Scores are per conversation, 0 or 100. For each problem we average over its runs, then average across problems.

- **Accuracy**: that average. Differences between arms are paired by problem, with a 95% bootstrap confidence interval.
- **Ended without an answer**: the conversation's last reply was a question, refusal or partial answer (scores 0).
- The analysis also reports the paper's aptitude (90th percentile) and unreliability (90th minus 10th percentile).
  With only 3 runs per problem these are coarse.

## Comparison with MT-OSC

MT-OSC (Singh et al. 2026, arXiv 2604.08782) is not reimplemented. `convcontract/eval/mtosc_reference.py` holds its
published results (Table 5), and the analysis prints its reported gain next to ours. This is **indicative only**:
we ran 3 of its 10 datasets on 1 of its 13 models; its per-problem results and exact test subset are unpublished;
and our baseline differs from its baseline by 6–12 points, so the two setups aren't identical.

## Fixes to the harness (all arms equally)

- `convcontract/eval/shim/model_openai.py`: routes the harness's model calls through OpenRouter. It sits first on
  the import path because the harness's own client asserts `OPENAI_API_KEY` at import, which crashed code grading
  (grading runs in child processes).
- `convcontract/eval/patches.py`: the math answer check rejected `1198` when the reply said `$1,198`; it now also
  accepts a match after removing commas, `$` and spaces.
- Every call's `max_tokens` is raised to at least 8192 (`CC_MIN_MAX_TOKENS`); the harness's 1000-token cap cuts
  off thinking models.
- The math grader compared strings, so `64.00` failed against `64` unless a `$` was present; it now also compares
  numbers. This fix came after the main run: re-grading its math numerically changes 3 contract conversations
  (72.5 → 74.5) and no baseline ones. `docs/findings.md` reports the re-graded number.
- The tool-call grader read a bare `true`/`false` as a string, so a correct call written JSON-style
  (`sort_by_rating=true`) failed; the task prompt lists functions in JSON and never asks for Python `True`. It now
  reads them as booleans. Found in the Claude pilot, after the main run: re-grading the main run's tool use changes
  3 contract and 1 always-answer conversations (contract 73.2 → 74.9, always-answer 44.8 → 45.4, normal chat
  unchanged; contract gain +35 [+25, +46] and +30 [+18, +42]).

## Engineering lessons from the pilots

Each is now handled in code; the pilot runs are in `results/archive/pilots/`.

- **Paraphrasing loses facts.** The extractor turned "a picture frame that costs $20" into "each gift costs $20".
  The extractor prompt now requires the user's own wording. Measured with `extraction_check` on 30 problems with
  4–5 pieces (none of them in the test set): statements kept went from about 90% to 97%.
- **Valid JSON isn't valid output.** Some providers returned `{}` or invented keys. Extractor output is validated,
  retried, and falls back to the user's message word for word.
- **Showing the model its previous answer re-creates the anchoring problem.** A wrong output format was carried
  forward. The previous answer is now included only when the user's message can't be acted on without it.
- **Providers differ.** One Llama provider held requests for minutes; the fastest one cost 7× more. Each call has a
  hard time limit, and the extractor uses fast, cheap providers explicitly.

## Reproducing

```bash
uv run python -m convcontract.eval.run --preset mtosc --n-runs 3 --run-name mtosc-main
uv run python -m convcontract.eval.run --preset mtosc --tasks actions --arms sharded-answer --n-runs 3 --run-name mtosc-main
uv run python -m convcontract.eval.analyze results/mtosc-main
uv run python -m convcontract.eval.extraction_check --prompts archive/extractor_v1.txt extractor.txt --limit 10
```

The main run costs about $3 on OpenRouter; the extraction check a few cents per prompt version.
