# golden/loader.py
"""Load hand-labeled golden cases from data/goldens.jsonl."""
import json
from dataclasses import dataclass
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "data" / "goldens.jsonl"


@dataclass
class Golden:
    id: str
    kind: str
    lang: str
    messages: list[dict]
    expected: dict


def load_goldens(path: Path | None = None) -> list[Golden]:
    p = path or _PATH
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out.append(Golden(id=d["id"], kind=d["kind"], lang=d["lang"],
                          messages=d["messages"], expected=d["expected"]))
    return out
