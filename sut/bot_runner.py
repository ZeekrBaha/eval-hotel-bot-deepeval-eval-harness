"""Driver for the vendored hotel bot (the SUT).

`BotRunner` adapts the production `handle_message` (single message + db-managed
history) to the harness's `run(messages)` interface: it resets the in-memory db,
seeds the scripted prior turns, then sends the final user message through the REAL
bot and maps the result dict to a `BotOutput`. No re-implementation of bot logic
lives here — the logic is in `sut/hotel_bot/bot.py`.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from sut.hotel_bot import bot, bot_fixed, db

# Point the vendored bot at the harness's filled system prompt (it reads
# SYSTEM_PROMPT_PATH). setdefault so an explicit env override still wins. This also
# covers non-pytest entry points (e.g. `python -m meta.judge_validation`).
os.environ.setdefault(
    "SYSTEM_PROMPT_PATH",
    str(Path(__file__).resolve().parent.parent / "data" / "system_prompt.txt"),
)


@dataclass
class BotOutput:
    reply: str
    is_booking_intent: bool
    guest_name: str | None
    check_in: str | None
    check_out: str | None
    num_guests: int | None

    @property
    def booking_complete(self) -> bool:
        return all(
            v not in (None, "")
            for v in (self.guest_name, self.check_in, self.check_out, self.num_guests)
        )


_VARIANTS = {"baseline": bot, "fixed": bot_fixed}


class BotRunner:
    def __init__(
        self, platform: str = "whatsapp", sender_id: str = "eval-user", variant: str = "baseline"
    ):
        self.platform = platform
        self.sender_id = sender_id
        # "baseline" = the vendored production bot; "fixed" = code-side language routing.
        self._bot = _VARIANTS[variant]

    def run(self, messages: list[dict]) -> BotOutput:
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("conversation must end with a user message")
        *prior, last = messages

        db.reset()
        db.set_history(self.platform, self.sender_id, prior)
        result = self._bot.handle_message(self.platform, self.sender_id, last["content"])

        return BotOutput(
            reply=result["reply"],
            is_booking_intent=result["is_booking_intent"],
            guest_name=result.get("guest_name"),
            check_in=result.get("check_in"),
            check_out=result.get("check_out"),
            num_guests=result.get("num_guests"),
        )
