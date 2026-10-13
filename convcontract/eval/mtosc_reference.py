"""Published MT-OSC results, for comparing against without reimplementing MT-OSC.

Source: Singh et al., "MT-OSC: Path for LLMs that Get Lost in Multi-Turn Conversation", arXiv 2604.08782v3,
Appendix D, Table 5: accuracy on samples with 6 or more turns (their test split), raw-history baseline
("MT") vs MT-OSC with window 4. Only the tasks the harness grades the same way are kept:
GSM8K = harness "math", BFCL = "actions", HumanEval = "code" (HumanEval source only).

Their chat-model settings (Appendix D): temperature 0.01, frequency_penalty 1, top_p 1, max 10000 tokens.
"""

TASK_LABELS = {"math": "GSM8K", "actions": "BFCL", "code": "HumanEval"}

# model: {task: (baseline, mt_osc)} as accuracy in [0, 1]
TABLE5 = {
    "x-ai/grok-3": {"math": (0.82, 0.86), "actions": (0.67, 0.71), "code": (0.51, 0.62)},
    "x-ai/grok-4": {"math": (0.84, 0.86), "actions": (0.64, 0.69), "code": (0.84, 0.80)},
    "meta-llama/llama-3.1-405b-instruct": {"math": (0.82, 0.86), "actions": (0.82, 0.84), "code": (0.69, 0.74)},
    "meta-llama/llama-3.3-70b-instruct": {"math": (0.80, 0.80), "actions": (0.84, 0.87), "code": (0.80, 0.84)},
    "meta-llama/llama-4-maverick": {"math": (0.76, 0.78), "actions": (0.84, 0.90), "code": (0.83, 0.91)},
    "meta-llama/llama-4-scout": {"math": (0.84, 0.86), "actions": (0.88, 0.88), "code": (0.67, 0.70)},
    "openai/gpt-4.1": {"math": (0.92, 0.87), "actions": (0.67, 0.80), "code": (0.83, 0.90)},
    "openai/gpt-4.1-mini": {"math": (0.89, 0.83), "actions": (0.67, 0.79), "code": (0.86, 0.90)},
    "openai/gpt-4.1-nano": {"math": (0.70, 0.77), "actions": (0.51, 0.59), "code": (0.70, 0.82)},
    "openai/gpt-4o": {"math": (0.83, 0.84), "actions": (0.56, 0.69), "code": (0.61, 0.79)},
    "openai/gpt-5": {"math": (0.90, 0.92), "actions": (0.64, 0.73), "code": (0.96, 1.00)},
    "openai/gpt-oss-20b": {"math": (0.86, 0.84), "actions": (0.59, 0.61), "code": (0.69, 0.70)},
    "openai/gpt-oss-120b": {"math": (0.92, 0.94), "actions": (0.49, 0.66), "code": (0.86, 0.82)},
}
