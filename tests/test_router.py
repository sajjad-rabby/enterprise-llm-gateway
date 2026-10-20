"""Tests for routing strategies, failover, and health tracking."""

from __future__ import annotations

import pytest

from enterprise_llm_gateway.models import ModelRequest
from enterprise_llm_gateway.providers.base import ProviderError
from enterprise_llm_gateway.router import (
    ProviderHealth,
    Router,
    RoutingError,
    estimate_tokens,
    is_retryable,
)
from tests.conftest import ExplodingProvider, FakeProvider


def _router(candidates=("primary", "secondary"), strategy="priority",
            max_attempts=3, **provider_kwargs):
    providers = {name: FakeProvider(name, dict(provider_kwargs)) for name in candidates}
    config = {"strategy": strategy, "candidates": list(candidates),
              "max_attempts": max_attempts}
    return Router(providers, config), providers


def test_estimate_tokens_scales_with_text():
    short = estimate_tokens(ModelRequest(prompt="a" * 40))
    long = estimate_tokens(ModelRequest(prompt="a" * 400))
    assert 0 < short < long


def test_estimate_tokens_is_at_least_one_for_tiny_prompts():
    assert estimate_tokens(ModelRequest(prompt="hi")) >= 1


def test_estimate_tokens_counts_messages():
    with_messages = ModelRequest(messages=[
        {"role": "user", "content": "x" * 200},
        {"role": "assistant", "content": "y" * 200},
    ])
    assert estimate_tokens(with_messages) > 50


@pytest.mark.parametrize("status,expected", [
    (None, True), (408, True), (429, True), (500, True), (503, True), (504, True),
    (400, False), (401, False), (403, False), (404, False), (422, False),
])
def test_is_retryable_classification(status, expected):
    assert is_retryable(ProviderError("p", "msg", status_code=status)) is expected


def test_invalid_strategy_is_rejected():
    providers = {"a": FakeProvider("a", {})}
    with pytest.raises(ValueError, match="routing.strategy"):
        Router(providers, {"strategy": "nonsense", "candidates": ["a"]})


def test_fail_threshold_read_from_config():
    providers = {"a": FakeProvider("a", {})}
    router = Router(providers, {"strategy": "priority", "candidates": ["a"],
                                "fail_threshold": 7})
    assert router.fail_threshold == 7


def test_explicit_fail_threshold_overrides_config():
    providers = {"a": FakeProvider("a", {})}
    router = Router(providers, {"strategy": "priority", "candidates": ["a"],
                                "fail_threshold": 7}, fail_threshold=2)
    assert router.fail_threshold == 2


def test_fail_threshold_defaults_to_three():
    providers = {"a": FakeProvider("a", {})}
    assert Router(providers, {"candidates": ["a"]}).fail_threshold == 3


def test_priority_strategy_preserves_configured_order():
    router, _ = _router(candidates=("a", "b", "c"), strategy="priority")
    assert router.order(ModelRequest(prompt="hi")) == ["a", "b", "c"]


def test_round_robin_rotates_start_candidate():
    router, _ = _router(candidates=("a", "b", "c"), strategy="round_robin")
    orders = [router.order(ModelRequest(prompt="hi"))[0] for _ in range(3)]
    assert orders == ["a", "b", "c"]


def test_cheapest_strategy_orders_by_price():
    router, _ = _router(candidates=("expensive", "cheap"), strategy="cheapest",
                         price_per_1k_input=1.0, price_per_1k_output=1.0)
    cheap = FakeProvider("cheap", {"price_per_1k_input": 0.0, "price_per_1k_output": 0.0})
    router.providers["cheap"] = cheap
    assert router.order(ModelRequest(prompt="hello world"))[0] == "cheap"


def test_latency_strategy_prefers_configured_order_before_data():
    router, _ = _router(candidates=("a", "b"), strategy="latency")
    assert router.order(ModelRequest(prompt="hi")) == ["a", "b"]


def test_latency_strategy_ranks_by_observed_latency():
    router, providers = _router(candidates=("slow", "fast"), strategy="latency")
    router.health("slow").record_success(900.0)
    router.health("fast").record_success(50.0)
    assert router.order(ModelRequest(prompt="hi"))[0] == "fast"


def test_complete_returns_response_from_first_candidate():
    router, providers = _router(candidates=("a", "b"), strategy="priority")
    result = router.complete(ModelRequest(prompt="hello"))
    assert result.provider == "a"
    assert result.attempts == ["a"]
    assert providers["b"].calls == 0


def test_complete_fails_over_to_next_candidate():
    providers = {
        "primary": ExplodingProvider("primary", {"status_code": 503}),
        "secondary": FakeProvider("secondary", {"text": "recovered"}),
    }
    router = Router(providers, {"strategy": "priority",
                                "candidates": ["primary", "secondary"],
                                "max_attempts": 2})
    result = router.complete(ModelRequest(prompt="hello"))
    assert result.provider == "secondary"
    assert result.attempts == ["primary", "secondary"]
    assert result.response.content == "recovered"


def test_complete_raises_when_all_candidates_fail():
    providers = {
        "primary": ExplodingProvider("primary", {"status_code": 500}),
        "secondary": ExplodingProvider("secondary", {"status_code": 500}),
    }
    router = Router(providers, {"strategy": "priority",
                                "candidates": ["primary", "secondary"],
                                "max_attempts": 2})
    with pytest.raises(RoutingError, match="all candidate providers failed"):
        router.complete(ModelRequest(prompt="hello"))


def test_non_retryable_error_fails_fast():
    providers = {
        "primary": ExplodingProvider("primary", {"status_code": 400,
                                                 "message": "bad request"}),
        "secondary": FakeProvider("secondary", {}),
    }
    router = Router(providers, {"strategy": "priority",
                                "candidates": ["primary", "secondary"],
                                "max_attempts": 2})
    with pytest.raises(RoutingError, match="non-retryable"):
        router.complete(ModelRequest(prompt="hello"))
    assert providers["secondary"].calls == 0


def test_attempt_budget_is_respected():
    providers = {name: ExplodingProvider(name, {"status_code": 500})
                 for name in ("a", "b", "c")}
    router = Router(providers, {"strategy": "priority",
                                "candidates": ["a", "b", "c"],
                                "max_attempts": 1})
    with pytest.raises(RoutingError) as excinfo:
        router.complete(ModelRequest(prompt="hello"))
    assert len(excinfo.value.attempts) == 1
    assert providers["a"].calls == 1
    assert providers["b"].calls == 0


def test_no_candidates_raises():
    router = Router({}, {"strategy": "priority", "candidates": []})
    with pytest.raises(RoutingError, match="no routing candidates"):
        router.complete(ModelRequest(prompt="hello"))


def test_health_tracks_successes_and_failures():
    router, providers = _router(candidates=("a", "b"))
    providers["a"].fail_with = ProviderError("a", "boom", status_code=503)
    router.complete(ModelRequest(prompt="hi"))
    health = router.health("a")
    assert health.failures == 1
    assert health.successes == 0
    assert health.consecutive_failures == 1


def test_health_resets_consecutive_failures_on_success():
    health = ProviderHealth("p")
    health.record_failure()
    health.record_failure()
    assert health.consecutive_failures == 2
    health.record_success(100.0)
    assert health.consecutive_failures == 0


def test_degraded_provider_is_demoted_but_retained():
    providers = {
        "bad": ExplodingProvider("bad", {"status_code": 503}),
        "good": FakeProvider("good", {}),
    }
    router = Router(providers, {"strategy": "priority",
                                "candidates": ["bad", "good"],
                                "max_attempts": 2, "fail_threshold": 1})
    # First call trips the failure threshold on "bad" while still succeeding
    # overall via failover to "good".
    result = router.complete(ModelRequest(prompt="hi"))
    assert result.provider == "good"
    # From then on the healthy provider is tried first.
    order = router.order(ModelRequest(prompt="hi"))
    assert order[0] == "good"
    assert "bad" in order
    assert order[-1] == "bad"


def test_all_health_reports_every_provider():
    router, _ = _router(candidates=("a", "b"))
    names = {entry["provider"] for entry in router.all_health()}
    assert names == {"a", "b"}


def test_health_snapshot_shape():
    router, _ = _router(candidates=("a",))
    router.health("a").record_success(10.0)
    entry = router.all_health()[0]
    assert entry["success_rate"] == 1.0
    assert entry["avg_latency_ms"] == 10.0


def test_cheapest_strategy_survives_provider_cost_error():
    class BrokenCost(FakeProvider):
        def estimate_cost(self, tokens_in, tokens_out, model=None):
            raise RuntimeError("price table unavailable")

    providers = {"broken": BrokenCost("broken", {}),
                 "healthy": FakeProvider("healthy", {})}
    router = Router(providers, {"strategy": "cheapest",
                                "candidates": ["broken", "healthy"]})
    order = router.order(ModelRequest(prompt="hello"))
    assert order == ["healthy", "broken"]
