import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# The vendored bot reads its system prompt from SYSTEM_PROMPT_PATH; point it at the
# harness's filled prompt so live evals run against known ground-truth hotel data.
os.environ.setdefault(
    "SYSTEM_PROMPT_PATH",
    str(Path(__file__).parent / "data" / "system_prompt.txt"),
)


def has_key(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())


def sut_variant() -> str:
    """Which SUT variant the live evals drive, from the SUT_VARIANT env var.

    Defaults to "baseline" (the vendored production bot — a known-failing benchmark
    on Kyrgyz/grounding). Set SUT_VARIANT=fixed for the should-pass regression run:

        SUT_VARIANT=fixed .venv/bin/python -m pytest evals/test_language.py -q
    """
    return os.environ.get("SUT_VARIANT", "baseline").strip() or "baseline"
