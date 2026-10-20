"""Shared pytest fixtures and offline fakes.

No test in this suite touches the network: providers are replaced with
deterministic in-process fakes so the suite is runnable in CI without
API keys and cannot produce flaky failures from upstream rate limits.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from enterprise_llm_gateway.models import ModelRequest, ModelResponse, Usage
from enterprise_llm_gateway.providers.base import Provider, ProviderError


class FakeProvider(Provider):
    """A provider that returns a canned response or raises a canned error."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        super().__init__(name, config)
        self.calls = 0
        self.fail_with: Optional[ProviderError] = None
        self.text = config.get("text", "ok")
        self.input_price = float(config.get("price_per_1k_input", 0.001))
        self.output_price = float(config.get("price_per_1k_output", 0.002))

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        if self.fail_with is not None:
            raise self.fail_with
        return ModelResponse(
            content=self.text,
            model=request.model or self.default_model or "fake-model",
            provider=self.name,
            usage=Usage(10, 20),
        )

    def estimate_cost(self, tokens_in: int, tokens_out: int,
                      model: Optional[str] = None) -> float:
        return self._cost_from_prices(tokens_in, tokens_out,
                                      self.input_price, self.output_price)


class ExplodingProvider(FakeProvider):
    """A provider that always fails, optionally with a chosen status code."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        super().__init__(name, config)
        status = config.get("status_code")
        self.fail_with = ProviderError(
            name, config.get("message", "upstream unavailable"), status_code=status)


@pytest.fixture
def fake_provider_cls():
    """Expose :class:`FakeProvider` to tests."""
    return FakeProvider


@pytest.fixture
def exploding_provider_cls():
    """Expose :class:`ExplodingProvider` to tests."""
    return ExplodingProvider


@pytest.fixture
def base_config():
    """Return a minimal valid gateway configuration."""
    return {
        "routing": {
            "strategy": "priority",
            "candidates": ["primary", "secondary"],
            "max_attempts": 2,
        },
        "cache": {"enabled": False, "ttl_seconds": 60, "max_entries": 10},
        "ledger": {},
        "providers": {
            "primary": {"type": "openai", "default_model": "fake-model"},
            "secondary": {"type": "openai", "default_model": "fake-model"},
        },
    }


@pytest.fixture
def make_gateway(base_config):
    """Factory that builds a Gateway from a config with fake providers wired in."""
    from enterprise_llm_gateway import gateway as gateway_module

    def _build(config: Optional[Dict[str, Any]] = None):
        cfg = config if config is not None else base_config
        providers = {
            name: (
                ExplodingProvider(name, section)
                if section.get("type") == "fake_boom"
                else FakeProvider(name, section)
            )
            for name, section in cfg["providers"].items()
        }
        instance = gateway_module.Gateway.__new__(gateway_module.Gateway)
        instance.config = cfg
        instance.providers = providers

        from enterprise_llm_gateway.cache import ResponseCache
        from enterprise_llm_gateway.ledger import UsageLedger
        from enterprise_llm_gateway.router import Router

        cache_cfg = cfg.get("cache", {})
        instance.cache = ResponseCache(
            ttl_seconds=float(cache_cfg.get("ttl_seconds", 60)),
            max_entries=int(cache_cfg.get("max_entries", 10)),
            enabled=bool(cache_cfg.get("enabled", True)),
        )
        instance.ledger = UsageLedger(path=cfg.get("ledger", {}).get("path"))
        instance.router = Router(
            providers, cfg.get("routing", {}),
            fail_threshold=int(cfg.get("routing", {}).get("fail_threshold", 3)),
        )
        return instance

    return _build


@pytest.fixture
def names():
    """Convenience alias used by parametrised provider-name tests."""
    return ["primary", "secondary"]
