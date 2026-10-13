"""Load the vendored Microsoft harness (external/lost_in_conversation) with our OpenRouter shim."""
import os
import sys
from pathlib import Path

from convcontract import config

SHIM_DIR = Path(__file__).resolve().parent / "shim"
DATASET = "data/sharded_instructions_600.json"  # relative to the harness dir, as the harness expects

_ready = False


def setup():
    """Make harness modules importable. Idempotent.

    Side effect: changes the working directory to the harness, because it opens prompts/ and data/
    with relative paths. Use absolute paths for anything else after calling this.
    """
    global _ready
    if _ready:
        return
    if not (config.HARNESS_DIR / "run_simulations.py").exists():
        raise FileNotFoundError(
            f"Harness not found at {config.HARNESS_DIR}. Clone it with:\n"
            "  git clone https://github.com/Microsoft/lost_in_conversation external/lost_in_conversation"
        )
    # Order matters: shim first so `model_openai` resolves to ours, then the harness. Child processes
    # started by the harness inherit sys.path, so they get the shim too.
    for p in (str(config.HARNESS_DIR), str(SHIM_DIR), str(config.REPO_ROOT)):
        if p in sys.path:
            sys.path.remove(p)
    sys.path[:0] = [str(SHIM_DIR), str(config.HARNESS_DIR), str(config.REPO_ROOT)]
    os.chdir(config.HARNESS_DIR)

    from convcontract.eval import patches

    patches.apply()
    _ready = True
