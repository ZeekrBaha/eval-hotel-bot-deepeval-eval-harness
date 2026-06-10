"""Guard against silent drift between the SUT response schema and the BotOutput
dataclass the harness maps it into. The two declare the same field set in two
places (bot._RESPONSE_FORMAT and bot_runner.BotOutput); this test fails loudly if
either side gains or loses a field.
"""
from typing import Any, cast

from sut.bot_runner import BotOutput
from sut.hotel_bot import bot

# The nested literal defeats mypy's inference (values collapse to Collection[str]);
# cast once so the tests can index into it.
_RESPONSE_FORMAT = cast(dict[str, Any], bot._RESPONSE_FORMAT)


def _schema_required() -> set[str]:
    return set(_RESPONSE_FORMAT["json_schema"]["schema"]["required"])


def test_botoutput_fields_match_schema_required():
    # booking_complete is a derived @property, not a dataclass field, so it is not
    # part of __dataclass_fields__ and correctly excluded.
    output_fields = set(BotOutput.__dataclass_fields__)
    assert output_fields == _schema_required()


def test_schema_properties_match_required():
    schema = _RESPONSE_FORMAT["json_schema"]["schema"]
    assert set(schema["properties"]) == set(schema["required"])
