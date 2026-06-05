import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


def has_key(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())
