"""Load the hotel system prompt and stamp today's date (as the real bot does)."""
from pathlib import Path

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "data" / "system_prompt.txt"


def load_system_prompt(path: Path | None = None) -> str:
    return (path or _PROMPT_PATH).read_text(encoding="utf-8")


def build_system_prompt(today: str, base: str | None = None) -> str:
    body = base if base is not None else load_system_prompt()
    return f"Сегодня: {today}\n\n{body}"
