from types import SimpleNamespace

import pytest
from openai import OpenAI

from app.config import Config
from app.llm_provider import (
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI,
    anthropic_compatible_base_url,
    anthropic_native_base_url,
    create_anthropic_camel_model,
    resolve_provider,
)
from app.utils.anthropic_chat import (
    DEFAULT_MAX_TOKENS,
    FALLBACK_BETA,
    JSON_INSTRUCTION,
    MIN_MAX_TOKENS,
    AnthropicChatClient,
)
from app.utils.llm_client import LLMClient, LLMResponseError
from app.utils.openai_chat_compat import create_chat_client, create_chat_completion


def text_message(text, stop_reason="end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
    )


class StreamRecorder:
    def __init__(self, message):
        self.message = message
        self.calls = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def get_final_message(self):
        return self.message


def fake_client(message=None, base_url=None, effort=None):
    client = AnthropicChatClient(api_key="test", base_url=base_url, effort=effort)
    recorder = StreamRecorder(message or text_message("hello"))
    client.client = SimpleNamespace(
        messages=recorder,
        beta=SimpleNamespace(messages=recorder),
    )
    return client, recorder


def complete(client, **overrides):
    params = {
        "model": "claude-haiku-5-5",
        "messages": [
            {"role": "system", "content": "first"},
            {"role": "user", "content": "question"},
            {"role": "system", "content": "second"},
        ],
    }
    params.update(overrides)
    return create_chat_completion(client, **params)


def test_request_uses_system_prompt_and_drops_temperature():
    client, recorder = fake_client()

    response = complete(client, temperature=0.3)

    call = recorder.calls[0]
    assert call["system"] == "first\n\nsecond"
    assert call["messages"] == [{"role": "user", "content": "question"}]
    assert "temperature" not in call
    assert "response_format" not in call
    assert response.choices[0].message.content == "hello"
    assert response.choices[0].finish_reason == "stop"


@pytest.mark.parametrize(
    "requested, sent",
    [
        (None, DEFAULT_MAX_TOKENS),
        (4096, MIN_MAX_TOKENS),
        (MIN_MAX_TOKENS + 1, MIN_MAX_TOKENS + 1),
    ],
)
def test_token_cap_leaves_room_for_thinking(requested, sent):
    client, recorder = fake_client()

    complete(client, max_tokens=requested)

    assert recorder.calls[0]["max_tokens"] == sent


def test_effort_is_sent_only_when_configured():
    client, recorder = fake_client(effort="medium")
    complete(client)
    assert recorder.calls[0]["output_config"] == {"effort": "medium"}

    client, recorder = fake_client()
    complete(client)
    assert "output_config" not in recorder.calls[0]


def test_json_request_adds_instruction_and_strips_fence():
    client, recorder = fake_client(text_message('```json\n{"a": 1}\n```'))

    response = complete(client, response_format={"type": "json_object"})

    assert recorder.calls[0]["system"].endswith(JSON_INSTRUCTION)
    assert response.choices[0].message.content == '{"a": 1}'


def test_refusal_fallbacks_apply_to_supported_first_party_models():
    client, recorder = fake_client()
    complete(client, model="claude-opus-5-5")
    assert recorder.calls[0]["fallbacks"] == "default"
    assert recorder.calls[0]["betas"] == [FALLBACK_BETA]

    client, recorder = fake_client()
    complete(client, model="claude-haiku-5-5")
    assert "fallbacks" not in recorder.calls[0]

    client, recorder = fake_client(base_url="https://proxy.example.com")
    complete(client, model="claude-opus-5-5")
    assert "fallbacks" not in recorder.calls[0]


@pytest.mark.parametrize(
    "stop_reason, finish_reason",
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("refusal", "content_filter"),
    ],
)
def test_stop_reasons_map_to_finish_reasons(stop_reason, finish_reason):
    client, _ = fake_client(text_message("", stop_reason))

    response = complete(client)

    assert response.choices[0].finish_reason == finish_reason


def make_llm_client(monkeypatch, message):
    monkeypatch.setattr(Config, "LLM_PROVIDER", PROVIDER_ANTHROPIC)
    llm = LLMClient(api_key="test", model="claude-opus-5-5")
    chat_client, recorder = fake_client(message)
    llm.client = chat_client
    return llm, recorder


def test_llm_client_chat_json_parses_claude_response(monkeypatch):
    llm, recorder = make_llm_client(monkeypatch, text_message('{"ok": true}'))

    result = llm.chat_json([{"role": "user", "content": "go"}])

    assert result == {"ok": True}
    assert "temperature" not in recorder.calls[0]


def test_llm_client_chat_json_reports_refusal(monkeypatch):
    llm, _ = make_llm_client(monkeypatch, text_message("", "refusal"))

    with pytest.raises(LLMResponseError) as error:
        llm.chat_json([{"role": "user", "content": "go"}])

    assert error.value.finish_reason == "content_filter"


def test_client_factory_routes_by_provider(monkeypatch):
    monkeypatch.setattr(Config, "LLM_EFFORT", "high")

    monkeypatch.setattr(Config, "LLM_PROVIDER", PROVIDER_ANTHROPIC)
    client = create_chat_client("test", "https://api.anthropic.com/v1/")
    assert isinstance(client, AnthropicChatClient)
    assert client.effort == "high"

    monkeypatch.setattr(Config, "LLM_PROVIDER", PROVIDER_OPENAI)
    assert isinstance(create_chat_client("test", "https://example.com/v1"), OpenAI)

    monkeypatch.setattr(Config, "LLM_PROVIDER", "other")
    with pytest.raises(ValueError):
        create_chat_client("test", None)


@pytest.mark.parametrize(
    "explicit, model, expected",
    [
        (None, "claude-opus-5-5", PROVIDER_ANTHROPIC),
        ("", "Claude-Opus-5-5", PROVIDER_ANTHROPIC),
        (None, "gpt-4o-mini", PROVIDER_OPENAI),
        (None, None, PROVIDER_OPENAI),
        ("openai", "claude-opus-5-5", PROVIDER_OPENAI),
        (" ANTHROPIC ", "my-proxy-model", PROVIDER_ANTHROPIC),
    ],
)
def test_resolve_provider(explicit, model, expected):
    assert resolve_provider(explicit, model) == expected


@pytest.mark.parametrize(
    "url, native, compatible",
    [
        (None, None, None),
        ("", None, None),
        ("https://api.anthropic.com", "https://api.anthropic.com", "https://api.anthropic.com/v1/"),
        ("https://api.anthropic.com/v1/", "https://api.anthropic.com", "https://api.anthropic.com/v1/"),
        ("https://api.anthropic.com/v1", "https://api.anthropic.com", "https://api.anthropic.com/v1/"),
    ],
)
def test_anthropic_urls_normalize(url, native, compatible):
    assert anthropic_native_base_url(url) == native
    assert anthropic_compatible_base_url(url) == compatible


def test_camel_model_targets_claude_without_sampling_defaults():
    model = create_anthropic_camel_model(
        api_key="test",
        base_url="https://api.anthropic.com",
        model="claude-opus-5-5",
    )

    assert str(model.model_type) == "claude-opus-5-5"
    assert model.model_config_dict == {"max_tokens": 16000}
    assert model.token_limit == 48000
    assert str(model._client.base_url) == "https://api.anthropic.com/v1/"
