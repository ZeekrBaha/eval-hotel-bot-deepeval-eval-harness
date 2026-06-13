"""Load the hotel system prompt (the grounding ground-truth for judge context).

The vendored bot reads this file itself via SYSTEM_PROMPT_PATH; this helper exists
so the evals can pass the same prompt to the judge as `context`.
"""

from pathlib import Path

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "data" / "system_prompt.txt"


def load_system_prompt(path: Path | None = None) -> str:
    return (path or _PROMPT_PATH).read_text(encoding="utf-8")
