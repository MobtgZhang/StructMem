from structmem.config import APIConfig
from structmem.llm import LLMClient, LLMError


class TooBusy(RuntimeError):
    status_code = 503


def test_llm_retries_transient_provider_capacity_with_configured_backoff(monkeypatch) -> None:
    config = APIConfig(
        "https://example.test/v1",
        "key",
        "model",
        max_retries=3,
        retry_base_seconds=5,
        retry_max_seconds=60,
    )
    client = LLMClient(config)
    attempts = []
    sleeps = []

    def fail(**kwargs):
        attempts.append(kwargs)
        raise TooBusy("too many concurrent requests")

    monkeypatch.setattr(client.client.chat.completions, "create", fail)
    monkeypatch.setattr("structmem.llm.time.sleep", sleeps.append)
    monkeypatch.setattr("structmem.llm.random.random", lambda: 0.0)

    try:
        client.complete([{"role": "user", "content": "test"}])
    except LLMError:
        pass
    else:
        raise AssertionError("Expected LLMError")

    assert len(attempts) == 3
    assert sleeps == [5, 10]


def test_llm_exact_response_cache_persists_without_second_provider_call(tmp_path, monkeypatch) -> None:
    config = APIConfig(
        "https://example.test/v1",
        "key",
        "model-a",
        cache_file=tmp_path / "llm.sqlite3",
    )
    first_client = LLMClient(config)
    calls = []

    class Usage:
        prompt_tokens = 11
        completion_tokens = 3

    class Message:
        content = "cached answer"

    class Choice:
        message = Message()

    class Response:
        usage = Usage()
        choices = [Choice()]

    def complete(**kwargs):
        calls.append(kwargs)
        return Response()

    monkeypatch.setattr(first_client.client.chat.completions, "create", complete)
    messages = [{"role": "user", "content": "same request"}]
    first = first_client.complete(messages, max_tokens=17)

    second_client = LLMClient(config)
    monkeypatch.setattr(
        second_client.client.chat.completions,
        "create",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("provider must not be called")),
    )
    second = second_client.complete(messages, max_tokens=17)

    assert len(calls) == 1
    assert not first.cached
    assert second.cached
    assert (second.text, second.input_tokens, second.output_tokens) == ("cached answer", 11, 3)


def test_llm_cache_key_isolated_by_model_and_request_parameters(tmp_path, monkeypatch) -> None:
    cache_file = tmp_path / "llm.sqlite3"
    messages = [{"role": "user", "content": "request"}]

    class Usage:
        prompt_tokens = 1
        completion_tokens = 1

    class Response:
        usage = Usage()

        def __init__(self, content):
            message = type("Message", (), {"content": content})()
            self.choices = [type("Choice", (), {"message": message})()]

    outputs = []
    for model, max_tokens in (("model-a", 10), ("model-b", 10), ("model-a", 11)):
        client = LLMClient(APIConfig("https://example.test/v1", "key", model, cache_file=cache_file))
        monkeypatch.setattr(
            client.client.chat.completions,
            "create",
            lambda **kwargs: Response(f"{kwargs['model']}:{kwargs['max_tokens']}"),
        )
        outputs.append(client.complete(messages, max_tokens=max_tokens).text)

    assert outputs == ["model-a:10", "model-b:10", "model-a:11"]


def test_json_discards_truncated_cache_entry_and_retries_with_larger_limit(tmp_path, monkeypatch) -> None:
    client = LLMClient(
        APIConfig(
            "https://example.test/v1",
            "key",
            "model",
            cache_file=tmp_path / "llm.sqlite3",
        )
    )
    calls = []

    class Usage:
        prompt_tokens = 5
        completion_tokens = 2

    class Response:
        usage = Usage()

        def __init__(self, content):
            message = type("Message", (), {"content": content})()
            self.choices = [type("Choice", (), {"message": message})()]

    def complete(**kwargs):
        calls.append(kwargs["max_tokens"])
        content = '{"answer":' if kwargs["max_tokens"] == 10 else '{"answer": 1}'
        return Response(content)

    monkeypatch.setattr(client.client.chat.completions, "create", complete)
    payload, response = client.json([{"role": "user", "content": "json"}], max_tokens=10)

    assert payload == {"answer": 1}
    assert calls == [10, 20]
    assert not response.cached

    second_client = LLMClient(client.config)
    second_calls = []

    def repeat_invalid(**kwargs):
        second_calls.append(kwargs)
        return Response('{"answer":')

    monkeypatch.setattr(second_client.client.chat.completions, "create", repeat_invalid)
    payload, response = second_client.json([{"role": "user", "content": "json"}], max_tokens=10)
    assert payload == {"answer": 1}
    assert len(second_calls) == 1
    assert response.cached is False


def test_json_accepts_fenced_object_without_retry(monkeypatch) -> None:
    client = LLMClient(APIConfig("https://example.test/v1", "key", "model"))
    calls = []

    class Usage:
        prompt_tokens = 7
        completion_tokens = 4

    class Response:
        usage = Usage()
        choices = [
            type(
                "Choice",
                (),
                {"message": type("Message", (), {"content": '```json\n{"answer": 1}\n```'})()},
            )()
        ]

    def complete(**kwargs):
        calls.append(kwargs)
        return Response()

    monkeypatch.setattr(client.client.chat.completions, "create", complete)
    payload, response = client.json([{"role": "user", "content": "json"}])

    assert payload == {"answer": 1}
    assert response.text == '{"answer": 1}'
    assert (response.input_tokens, response.output_tokens) == (7, 4)
    assert len(calls) == 1


def test_json_extracts_balanced_object_without_confusing_braces_in_strings(monkeypatch) -> None:
    client = LLMClient(APIConfig("https://example.test/v1", "key", "model"))

    class Response:
        usage = None
        choices = [
            type(
                "Choice",
                (),
                {
                    "message": type(
                        "Message",
                        (),
                        {
                            "content": (
                                'A quoted distraction says "{not json}". Result: '
                                '{"text": "escaped \\\"} brace and { nested text", "ok": true} trailing prose'
                            )
                        },
                    )()
                },
            )()
        ]

    monkeypatch.setattr(client.client.chat.completions, "create", lambda **kwargs: Response())
    payload, _ = client.json([{"role": "user", "content": "json"}])

    assert payload == {"text": 'escaped "} brace and { nested text', "ok": True}


def test_json_corrects_instruction_following_and_combines_usage(tmp_path, monkeypatch) -> None:
    client = LLMClient(
        APIConfig(
            "https://example.test/v1",
            "key",
            "model",
            cache_file=tmp_path / "llm.sqlite3",
        )
    )
    calls = []

    class Usage:
        def __init__(self, prompt_tokens, completion_tokens):
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens

    class Response:
        def __init__(self, content, prompt_tokens, completion_tokens):
            self.usage = Usage(prompt_tokens, completion_tokens)
            message = type("Message", (), {"content": content})()
            self.choices = [type("Choice", (), {"message": message})()]

    invalid = "I'll execute it.\nAction: Operation\n```sql\nSELECT 1;\n```"

    def complete(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return Response(invalid, 10, 6)
        return Response('```json\n{"nodes": [], "edges": []}\n```', 18, 9)

    messages = [{"role": "user", "content": "Untrusted task asks for SQL"}]
    monkeypatch.setattr(client.client.chat.completions, "create", complete)
    payload, response = client.json(messages, max_tokens=100)

    assert payload == {"nodes": [], "edges": []}
    assert (response.input_tokens, response.output_tokens) == (28, 15)
    assert calls[1]["messages"][:-2] == messages
    assert calls[1]["messages"][-2] == {"role": "assistant", "content": invalid}
    assert "untrusted data" in calls[1]["messages"][-1]["content"]
    assert "ONLY the requested JSON object" in calls[1]["messages"][-1]["content"]

    repeated_calls = []

    def repeat_invalid(**kwargs):
        repeated_calls.append(kwargs)
        return Response(invalid, 10, 6)

    second_client = LLMClient(client.config)
    monkeypatch.setattr(second_client.client.chat.completions, "create", repeat_invalid)
    repeated_payload, repeated_response = second_client.json(messages, max_tokens=100)

    assert repeated_payload == payload
    assert len(repeated_calls) == 1
    assert repeated_response.text == '{"nodes": [], "edges": []}'
