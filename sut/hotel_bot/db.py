# =============================================================================
# In-memory stand-in for the production Supabase-backed db (hotel-chat-bot/core/db.py).
#
# The production bot persists conversation history, daily counters, message dedup,
# and booking alerts in Postgres via RPCs. None of that is the thing under
# evaluation — we evaluate the bot's *replies*. So this module implements the same
# function signatures the bot calls, backed by plain dicts, plus two eval-only
# helpers (reset, set_history) the harness uses to script a conversation.
# =============================================================================

MAX_HISTORY = 20

# (platform, sender_id) -> list[{"role","content"}]
_history: dict[tuple[str, str], list[dict]] = {}
# (platform, sender_id) -> int
_counts: dict[tuple[str, str], int] = {}
_seen_messages: set[str] = set()
_booking_keys: dict[tuple[str, str], str] = {}


# --- eval-only helpers (not part of the production interface) -----------------


def reset() -> None:
    """Clear all in-memory state. Call before scripting a fresh conversation."""
    _history.clear()
    _counts.clear()
    _seen_messages.clear()
    _booking_keys.clear()


def set_history(platform: str, sender_id: str, messages: list[dict]) -> None:
    """Seed prior conversation turns so the bot sees scripted context."""
    _history[(platform, sender_id)] = list(messages)


# --- production interface (in-memory implementation) --------------------------


def get_history(platform: str, sender_id: str) -> list[dict]:
    return list(_history.get((platform, sender_id), []))


def increment_daily_counter(platform: str, sender_id: str) -> int:
    key = (platform, sender_id)
    _counts[key] = _counts.get(key, 0) + 1
    return _counts[key]


def append_conversation_turn(platform: str, sender_id: str, messages: dict | list[dict]) -> None:
    messages_list = messages if isinstance(messages, list) else [messages]
    key = (platform, sender_id)
    _history.setdefault(key, [])
    _history[key].extend(messages_list)
    _history[key] = _history[key][-MAX_HISTORY:]


def is_duplicate_message(message_id: str) -> bool:
    if message_id in _seen_messages:
        return True
    _seen_messages.add(message_id)
    return False


def check_and_set_booking_alert(platform: str, sender_id: str, booking: dict) -> bool:
    key = (
        f"{booking.get('guest_name')}|{booking.get('check_in')}"
        f"|{booking.get('check_out')}|{booking.get('num_guests')}"
    )
    store_key = (platform, sender_id)
    if _booking_keys.get(store_key) == key:
        return False
    _booking_keys[store_key] = key
    return True
