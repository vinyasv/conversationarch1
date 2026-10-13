"""Paths and model defaults. Values come from the environment (or .env at the repo root)."""
import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS_DIR = REPO_ROOT / "external" / "lost_in_conversation"
RESULTS_DIR = REPO_ROOT / "results"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

load_dotenv(REPO_ROOT / ".env")

EXECUTOR_MODEL = os.environ.get("CC_EXECUTOR_MODEL", "openai/gpt-6-luna")
# The extractor defaults to the executor model: same model, two roles.
EXTRACTOR_MODEL = os.environ.get("CC_EXTRACTOR_MODEL") or EXECUTOR_MODEL
# Simulated user and harness grading helper; kept identical to the paper.
USER_MODEL = os.environ.get("CC_USER_MODEL", "openai/gpt-4o-mini")
SYSTEM_MODEL = os.environ.get("CC_SYSTEM_MODEL", "openai/gpt-4o-mini")

MIN_MAX_TOKENS = int(os.environ.get("CC_MIN_MAX_TOKENS", "8192"))
REASONING_EFFORT = os.environ.get("CC_REASONING_EFFORT") or None
