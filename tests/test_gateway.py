"""End-to-end tests for the Gateway call path: cache, routing, ledger."""

from __future__ import annotations

import copy

import pytest

from enterprise_llm_gateway.models import ModelRequest
from enterprise_llm_gateway.providers.base import ProviderError
from enterprise_llm_gateway.router import RoutingError


def test_completion_reaches_provider_and_returns_text(make_gateway):
    gateway = make_gateway()
    result = gateway.complete(ModelRequest(prompt="hello"))
    assert result.response.content == "ok"
    assert result.provider == "primary"
    assert result.cached is False


def test_repeat_request_is_served_from_cache(make_gateway, base_config):
    config = copy.deepcopy(base_config)
    config["cache"]["enabled"] = True
    gateway = make_gateway(config)
    request = ModelRequest(prompt="cache me")
    first = gateway.complete(request)
    second = gateway.complete(request)
    assert first.cached is False
    assert second.cached is True
    assert gateway.providers["primary"].calls == 1
    assert gateway.cache.stats()["hits"] == 1


def test_disabled_cache_always_calls_provider(make_gateway, base_config):
    config = copy.deepcopy(base_config)
    config["cache"]["enabled"] = False
    gateway = make_gateway(config)
    gateway.complete(ModelRequest(prompt="no cache"))
    gateway.complete(ModelRequest(prompt="no cache"))
    assert gateway.providers["primary"].calls == 2


def test_different_prompts_are_cached_separately(make_gateway, base_config):
    config = copy.deepcopy(base_config)
    config["cache"]["enabled"] = True
    gateway = make_gateway(config)
    gateway.complete(ModelRequest(prompt="first"))
    gateway.complete(ModelRequest(prompt="second"))
    assert gateway.providers["primary"].calls == 2


def test_failover_records_serving_provider(make_gateway, base_config):
    config = copy.deepcopy(base_config)
    config["providers"]["primary"]["type"] = "fake_boom"
    config["providers"]["primary"]["status_code"] = 503
    gateway = make_gateway(config)
    result = gateway.complete(ModelRequest(prompt="hello"))
    assert result.provider == "secondary"
    assert result.attempts == ["primary", "secondary"]


def test_exhausted_routing_raises(make_gateway, base_config):
    config = copy.deepcopy(base_config)
    for name in ("primary", "secondary"):
        config["providers"][name]["type"] = "fake_boom"
        config["providers"][name]["status_code"] = 503
    gateway = make_gateway(config)
    with pytest.raises(RoutingError):
        gateway.complete(ModelRequest(prompt="hello"))


def test_failed_request_is_written_to_ledger(make_gateway, base_config):
    config = copy.deepcopy(base_config)
    for name in ("primary", "secondary"):
        config["providers"][name]["type"] = "fake_boom"
        config["providers"][name]["status_code"] = 503
    gateway = make_gateway(config)
    with pytest.raises(RoutingError):
        gateway.complete(ModelRequest(prompt="hello"))
    summary = gateway.ledger.summary()
    assert summary["requests"] >= 1
    assert summary["errors"] >= 1


def test_ledger_records_cost_from_provider_pricing(make_gateway, base_config):
    config = copy.deepcopy(base_config)
    config["providers"]["primary"]["price_per_1k_input"] = 0.001
    config["providers"]["primary"]["price_per_1k_output"] = 0.002
    gateway = make_gateway(config)
    gateway.complete(ModelRequest(prompt="bill me"))
    # FakeProvider reports Usage(10, 20): (10*0.001 + 20*0.002)/1000
    assert gateway.ledger.summary()["total_cost_usd"] == pytest.approx(0.00005)


def test_cache_hit_does_not_incur_cost(make_gateway, base_config):
    config = copy.deepcopy(base_config)
    config["cache"]["enabled"] = True
    gateway = make_gateway(config)
    gateway.complete(ModelRequest(prompt="cached"))
    gateway.complete(ModelRequest(prompt="cached"))
    summary = gateway.ledger.summary()
    assert summary["cache_hits"] == 1
    assert summary["total_cost_usd"] == pytest.approx(0.00005)


def test_stats_reports_all_sections(make_gateway):
    gateway = make_gateway()
    gateway.complete(ModelRequest(prompt="hi"))
    stats = gateway.stats()
    assert set(stats) == {"providers", "routing", "cache", "provider_health", "ledger"}
    assert stats["routing"]["strategy"] == "priority"
    assert sorted(stats["providers"]) == ["primary", "secondary"]
    assert len(stats["provider_health"]) == 2


def test_stats_records_user_for_attribution(make_gateway):
    gateway = make_gateway()
    gateway.complete(ModelRequest(prompt="hi", user="alice"))
    records = gateway.ledger._recent
    assert records[-1].user == "alice"


def test_user_is_excluded_from_cache_key(make_gateway, base_config):
    config = copy.deepcopy(base_config)
    config["cache"]["enabled"] = True
    gateway = make_gateway(config)
    gateway.complete(ModelRequest(prompt="shared", user="alice"))
    result = gateway.complete(ModelRequest(prompt="shared", user="bob"))
    assert result.cached is True


def test_ledger_persists_when_configured(make_gateway, base_config, tmp_path):
    config = copy.deepcopy(base_config)
    config["ledger"]["path"] = str(tmp_path / "ledger.jsonl")
    gateway = make_gateway(config)
    gateway.complete(ModelRequest(prompt="persisted"))
    assert (tmp_path / "ledger.jsonl").exists()


def test_non_retryable_error_propagates(make_gateway, base_config):
    config = copy.deepcopy(base_config)
    config["providers"]["primary"]["type"] = "fake_boom"
    config["providers"]["primary"]["status_code"] = 401
    gateway = make_gateway(config)
    with pytest.raises(RoutingError, match="non-retryable"):
        gateway.complete(ModelRequest(prompt="hello"))


def test_gateway_rejects_empty_request():
    with pytest.raises(ValueError):
        ModelRequest()
