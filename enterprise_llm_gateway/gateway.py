"""The gateway itself: routing, cache lookup, and ledger recording in one object
without standing up an HTTP server.

1. **Cache lookup** - identical prompts reuse a stored response and never
   reach a provider, so they cost nothing.
2. **Route and execute** - the router orders candidates and fails over
   until one provider succeeds.
3. **Record** - the outcome, including failures, is written to the ledger
   so spend and reliability stay auditable.

Keeping this sequence in one place is what makes the behaviour testable
without standing up an HTTP server.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

from .cache import ResponseCache
from .config import load_config
from .ledger import LedgerRecord, UsageLedger
from .models import ModelRequest, ModelResponse, cache_key_for_request
from .providers import create_provider
from .providers.base import Provider, ProviderError
from .router import DispatchResult, Router, RoutingError, estimate_tokens


class Gateway:

    """Coordinates providers, routing, caching and usage accounting.

Args:
        config: A validated gateway configuration mapping.
        cache: Optional pre-built cache; one is created when omitted.
        ledger: Optional pre-built ledger; one is created when omitted.
    """

    def __init__(self, config: Dict[str, Any], cache: Optional[ResponseCache] = None,
                 ledger: Optional[UsageLedger] = None) -> None:
        self.config = config
        providers_cfg: Dict[str, Any] = config.get("providers", {})
        self.providers: Dict[str, Provider] = {
            name: create_provider(name, section)
            for name, section in providers_cfg.items()
        }

        cache_cfg = config.get("cache", {})
        self.cache = cache if cache is not None else ResponseCache(
            ttl_seconds=float(cache_cfg.get("ttl_seconds", 3600)),
            max_entries=int(cache_cfg.get("max_entries", 1000)),
            enabled=bool(cache_cfg.get("enabled", True)),
        )

        ledger_cfg = config.get("ledger", {})
        self.ledger = ledger if ledger is not None else UsageLedger(
            path=ledger_cfg.get("path"))

        self.router = Router(
            self.providers,
            config.get("routing", {}),
            fail_threshold=int(config.get("routing", {}).get("fail_threshold", 3)),
        )

    @classmethod
    def from_config_file(cls, path: str, env: Optional[Dict[str, str]] = None) -> "Gateway":
        """Build a gateway from a JSON configuration file."""
        return cls(load_config(path, env))

    def complete(self, request: ModelRequest) -> DispatchResult:
        """Run a request through the router and return the dispatch result.

        Returns:
            A :class:`DispatchResult` whose ``cached`` flag reports whether
            the response came from cache rather than a provider.

        Raises:
            RoutingError: When no candidate provider could serve the request.
        """
        key = cache_key_for_request(request)
        cached = self.cache.get(key)
        if cached is not None:
            # Record what the call *would* have cost so the ledger can
            # report real savings, while the cache-hit flag keeps this
            # amount out of net spend.
            self._record(request, cached, cached.provider, 0.0,
                         self._cost_of(cached, cached.provider, request),
                         cache_hit=True)
            return DispatchResult(cached, cached.provider, [cached.provider], 0.0,
                                  cached=True)

        started = time.time()
        request_id = "req-" + uuid.uuid4().hex[:16]
        try:
            result = self.router.complete(request)
        except RoutingError:
            self.ledger.record(LedgerRecord(
                timestamp=time.time(), request_id=request_id, provider="",
                model=request.model or "", user=request.user,
                prompt_tokens=estimate_tokens(request), completion_tokens=0,
                cost_usd=0.0, latency_ms=(time.time() - started) * 1000.0,
                cache_hit=False, error="routing_exhausted",
            ))
            raise

        latency_ms = (time.time() - started) * 1000.0
        self.cache.set(key, result.response)
        self._record(request, result.response, result.provider, latency_ms,
                     self._cost_of(result.response, result.provider, request),
                     cache_hit=False, request_id=request_id)
        return result

    def _cost_of(self, response: ModelResponse, provider_name: str,
                 request: ModelRequest) -> float:
        """Estimate USD cost using the serving provider's price table."""
        provider = self.providers.get(provider_name)
        if provider is None:
            return 0.0
        usage = response.usage
        try:
            return provider.estimate_cost(
                usage.prompt_tokens, usage.completion_tokens,
                response.model or request.model)
        except Exception:
            return 0.0

    def _record(self, request: ModelRequest, response: ModelResponse,
                provider_name: str, latency_ms: float, cost_usd: float,
                cache_hit: bool, request_id: Optional[str] = None) -> None:
        """Append the request and response to the ledger.

Every recorded call is

        Cache hits pass the hypothetical provider cost; the ``cache_hit``
        flag keeps it out of net spend while still making the saving
        measurable in the ledger summary.
        """
        usage = response.usage
        self.ledger.record(LedgerRecord(
            timestamp=time.time(),
            request_id=request_id or ("req-" + uuid.uuid4().hex[:16]),
            provider=provider_name,
            model=response.model or request.model or "",
            user=request.user,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
        ))

    def stats(self) -> Dict[str, Any]:
        """Return a combined snapshot of cache, routing, and spend state."""
        return {
            "providers": sorted(self.providers),
            "routing": {
                "strategy": self.router.strategy,
                "candidates": list(self.router.candidates),
                "max_attempts": self.router.max_attempts,
            },
            "cache": self.cache.stats(),
            "provider_health": self.router.all_health(),
            "ledger": self.ledger.summary(),
        }
