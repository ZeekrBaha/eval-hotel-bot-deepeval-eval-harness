from sut.llm_client import FakeLLM


def test_fake_llm_returns_queued_responses_in_order():
    llm = FakeLLM(["first", "second"])
    msgs = [{"role": "user", "content": "hi"}]
    assert llm.complete(msgs, json_schema={}) == "first"
    assert llm.complete(msgs, json_schema={}) == "second"


def test_fake_llm_raises_when_exhausted():
    llm = FakeLLM(["only"])
    llm.complete([], json_schema={})
    try:
        llm.complete([], json_schema={})
        assert False, "expected IndexError"
    except IndexError:
        pass


def test_fake_llm_records_calls():
    llm = FakeLLM(["x"])
    llm.complete([{"role": "user", "content": "ping"}], json_schema={"k": 1})
    assert llm.calls[0]["messages"][0]["content"] == "ping"
    assert llm.calls[0]["json_schema"] == {"k": 1}
