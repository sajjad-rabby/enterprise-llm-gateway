"""Tests for the core models and provider adapters."""

from __future__ import annotations

import json

import pytest

from enterprise_llm_gateway.models import (
    ModelRequest,
    ModelResponse,
    Usage,
    cache_key_for_request,
)
from enterprise_llm_gateway.providers import PROVIDER_TYPES, create_provider
from enterprise_llm_gateway.providers.base import (
    ProviderError,
    build_openai_messages,
    resolve_secret,
)


# -- Usage ---------------------------------------------------------------

def test_usage_totals_are_derived():
    assert Usage(10, 20).total_tokens == 30


def test_usage_accepts_explicit_total():
    assert Usage(10, 20, total_tokens=99).total_tokens == 99


def test_usage_dict_roundtrip():
    assert Usage.from_dict(Usage(3, 4).to_dict()).total_tokens == 7


def test_usage_from_empty_dict():
    assert Usage.from_dict(None).total_tokens == 0


def test_usage_carries_stop_reason():
    assert Usage(1, 1, stop_reason="end_turn").stop_reason == "end_turn"


# -- ModelRequest --------------------------------------------------------

def test_request_requires_prompt_or_messages():
    with pytest.raises(ValueError, match="prompt or messages"):
        ModelRequest()


def test_request_accepts_messages_only():
    request = ModelRequest(messages=[{"role": "user", "content": "hi"}])
    assert request.prompt == ""
    assert request.messages[0]["content"] == "hi"


def test_cache_key_is_deterministic():
    first = cache_key_for_request(ModelRequest(prompt="hello", model="m"))
    second = cache_key_for_request(ModelRequest(prompt="hello", model="m"))
    assert first == second


def test_cache_key_differs_by_sampling_parameters():
    base = ModelRequest(prompt="hello", model="m", temperature=0.0)
    warmer = ModelRequest(prompt="hello", model="m", temperature=0.9)
    assert cache_key_for_request(base) != cache_key_for_request(warmer)


def test_cache_key_ignores_user_and_metadata():
    a = ModelRequest(prompt="hi", user="alice", metadata={"x": 1})
    b = ModelRequest(prompt="hi", user="bob", metadata={"x": 2})
    assert cache_key_for_request(a) == cache_key_for_request(b)


def test_cache_key_is_stable_across_key_order():
    first = ModelRequest(prompt="hi", messages=None)
    second = ModelRequest(prompt="hi", messages=None)
    assert len(cache_key_for_request(first)) == 64


# -- ModelResponse -------------------------------------------------------

def test_response_text_alias_matches_content():
    response = ModelResponse("answer", "m", "p", Usage(1, 1))
    assert response.text == response.content


def test_response_openai_json_shape():
    payload = ModelResponse("hi", "gpt", "openai", Usage(5, 6)).to_openai_json()
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "gpt"
    assert payload["choices"][0]["message"] == {"role": "assistant", "content": "hi"}
    assert payload["usage"]["total_tokens"] == 11


def test_response_openai_json_is_serializable():
    payload = ModelResponse("hi", "gpt", "openai", Usage(1, 1)).to_openai_json()
    assert json.loads(json.dumps(payload))["object"] == "chat.completion"


def test_response_parses_from_openai_json():
    raw = {"model": "gpt", "created": 1700000000,
           "choices": [{"message": {"role": "assistant", "content": "hey"},
                        "finish_reason": "stop"}],
           "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}}
    parsed = ModelResponse.from_openai_json(raw, provider="openai")
    assert parsed.content == "hey"
    assert parsed.provider == "openai"
    assert parsed.usage.total_tokens == 5


def test_response_parse_requires_choices():
    with pytest.raises(ValueError, match="no choices"):
        ModelResponse.from_openai_json({"model": "m"})


# -- Provider helpers ----------------------------------------------------

def test_build_messages_from_prompt():
    body = build_openai_messages(ModelRequest(prompt="hi"))
    assert body["messages"] == [{"role": "user", "content": "hi"}]


def test_build_messages_passes_through_roles():
    body = build_openai_messages(ModelRequest(messages=[
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hi"},
    ]))
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_resolve_secret_prefers_env(monkeypatch):
    monkeypatch.setenv("MY_KEY", "from-env")
    config = {"api_key_env": "MY_KEY", "api_key": "inline"}
    assert resolve_secret(config, "api_key") == "from-env"


def test_resolve_secret_falls_back_to_inline():
    assert resolve_secret({"api_key": "inline"}, "api_key") == "inline"


def test_resolve_secret_raises_when_env_missing(monkeypatch):
    monkeypatch.delenv("NOT_SET_KEY", raising=False)
    with pytest.raises(ProviderError, match="not set"):
        resolve_secret({"api_key_env": "NOT_SET_KEY"}, "api_key")


def test_resolve_secret_returns_none_when_absent():
    assert resolve_secret({}, "api_key") is None


# -- Provider registry ---------------------------------------------------

@pytest.mark.parametrize("provider_type", ["openai", "anthropic", "azure",
                                           "ollama", "openai_compat"])
def test_every_registered_type_is_constructible(provider_type):
    assert provider_type in PROVIDER_TYPES


def test_create_provider_rejects_unknown_type():
    with pytest.raises(ProviderError, match="unknown provider type"):
        create_provider("mystery", {"type": "nope"})


def test_create_provider_requires_type():
    with pytest.raises(ProviderError, match="missing required field"):
        create_provider("mystery", {})


def test_openai_provider_defaults():
    provider = create_provider("openai", {"type": "openai", "default_model": "gpt-4o-mini",
                                          "api_key": "sk-test"})
    assert provider.default_model == "gpt-4o-mini"
    assert provider.timeout == 60


def test_openai_provider_uses_known_pricing():
    provider = create_provider("openai", {"type": "openai", "default_model": "gpt-4o",
                                          "api_key": "sk-test"})
    # gpt-4o: 1000 in @ 0.0025/1k + 1000 out @ 0.01/1k = 0.0125
    assert provider.estimate_cost(1000, 1000, "gpt-4o") == pytest.approx(0.0125)


def test_openai_provider_honours_custom_prices():
    provider = create_provider("openai", {
        "type": "openai", "api_key": "sk-test", "default_model": "house-model",
        "prices": {"house-model": [0.001, 0.002]}})
    assert provider.estimate_cost(1000, 1000, "house-model") == pytest.approx(0.003)


def test_ollama_costs_nothing_by_default():
    provider = create_provider("ollama", {"type": "ollama", "default_model": "llama3"})
    assert provider.estimate_cost(1000, 1000, "llama3") == 0.0


def test_ollama_can_carry_opex_pricing():
    provider = create_provider("ollama", {"type": "ollama", "default_model": "llama3",
                                          "prices": {"llama3": [0.1, 0.2]}})
    assert provider.estimate_cost(1000, 1000, "llama3") == pytest.approx(0.3)


def test_azure_requires_endpoint_and_deployment():
    with pytest.raises(ProviderError, match="endpoint"):
        create_provider("azure", {"type": "azure", "api_key": "k"})
    with pytest.raises(ProviderError, match="deployment"):
        create_provider("azure", {"type": "azure", "endpoint": "https://x", "api_key": "k"})


def test_openai_compat_requires_base_url():
    with pytest.raises(ProviderError, match="base_url"):
        create_provider("compat", {"type": "openai_compat", "default_model": "m"})


def test_timeout_is_configurable():
    provider = create_provider("ollama", {"type": "ollama", "default_model": "m", "timeout_seconds": 15})
    assert provider.timeout == 15.0
