"""Provider selection strategies.

The router prefers to reuse healthy capacity for a

The router picks an ordered list of candidate providers for a request,
then walks that list until one succeeds or the attempt budget is spent.
Ordering is driven by a configurable strategy, and providers that start
failing are demoted so a single bad deployment cannot absorb every
request while healthy capacity sits idle.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from .models import ModelRequest, ModelResponse
from .providers.base import Provider, ProviderError

# Status codes worth retrying on a different provider. 4xx codes other
# than these indicate a malformed request, which will fail identically
# everywhere, so retrying only wastes latency and budget.
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
_STRATEGIES = ("priority", "round_robin", "cheapest", "latency")
_CHARS_PER_TOKEN = 4.0


class RoutingError(Exception):
    """Raised when every candidate provider has been exhausted."""

    def __init__(self, message: str, attempts: Optional[List[str]] = None) -> None:
        super().__init__(message)
        self.attempts = attempts or []


def estimate_tokens(request: ModelRequest) -> int:
    """Rough token estimate used for cost-aware routing.

The estimate is intentionally simple: its only job is

    Uses a four-characters-per-token heuristic, which is accurate enough
    to rank providers by price for the ``cheapest`` strategy.
    """
    text = request.prompt or ""
    for message in request.messages or []:
        text += str(message.get("content", ""))
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def is_retryable(error: ProviderError) -> bool:
    """Return True when retrying this failure on another provider may help."""
    if error.status_code is None:
        return True
    return error.status_code in _RETRYABLE_STATUS


class ProviderHealth:
    """Rolling health and latency statistics for one provider."""

    __slots__ = ("name", "calls", "successes", "failures", "consecutive_failures",
                 "latency_ewma_ms", "total_latency_ms")

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0
        self.successes = 0
        self.failures = 0
        self.consecutive_failures = 0
        self.latency_ewma_ms = 0.0
        self.total_latency_ms = 0.0

    def record_success(self, latency_ms: float) -> None:
        """Fold a successful call into the rolling statistics."""
        self.calls += 1
        self.successes += 1
        self.consecutive_failures = 0
        self._update_latency(latency_ms)

    def record_failure(self) -> None:
        """Fold a failed call into the rolling statistics."""
        self.calls += 1
        self.failures += 1
        self.consecutive_failures += 1

    def _update_latency(self, latency_ms: float, weight: float = 0.3) -> None:
        if self.latency_ewma_ms <= 0.0:
            self.latency_ewma_ms = float(latency_ms)
        else:
            self.latency_ewma_ms = (
                (1 - weight) * self.latency_ewma_ms + weight * float(latency_ms)
            )
        self.total_latency_ms += float(latency_ms)

    def to_dict(self) -> Dict[str, Any]:
        """Render the health snapshot for metrics or diagnostics."""
        return {
            "provider": self.name,
            "calls": self.calls,
            "successes": self.successes,
            "failures": self.failures,
            "consecutive_failures": self.consecutive_failures,
            "success_rate": round(self.successes / self.calls, 4) if self.calls else 0.0,
            "latency_ewma_ms": round(self.latency_ewma_ms, 3),
            "avg_latency_ms": (
                round(self.total_latency_ms / self.calls, 3) if self.calls else 0.0
            ),
        }


class DispatchResult:
    """The outcome of a successful dispatch plus its failover trail."""

    __slots__ = ("response", "provider", "attempts", "total_latency_ms", "cached")

    def __init__(self, response: ModelResponse, provider: str,
                 attempts: List[str], total_latency_ms: float,
                 cached: bool = False) -> None:
        self.response = response
        self.provider = provider
        self.attempts = attempts
        self.total_latency_ms = total_latency_ms
        self.cached = cached

    def to_dict(self) -> Dict[str, Any]:
        """Render a JSON-serializable summary of the dispatch."""
        return {
            "provider": self.provider,
            "model": self.response.model,
            "attempts": list(self.attempts),
            "attempt_count": len(self.attempts),
            "latency_ms": round(self.total_latency_ms, 3),
            "cache_hit": self.cached,
            "usage": self.response.usage.to_dict(),
        }


class Router:

    """Chooses which provider handles a request and records failures.

Args:
        providers: Instantiated providers keyed by their configuration name.
        routing_config: The ``routing`` section of the gateway configuration.
        fail_threshold: Consecutive failures after which a provider is demoted
            to the end of the candidate order until it succeeds again. When
            omitted the value is read from ``routing_config``, then defaults to 3.
    """

    def __init__(self, providers: Dict[str, Provider],
                 routing_config: Dict[str, Any],
                 fail_threshold: Optional[int] = None) -> None:
        self.providers = dict(providers)
        self.config = routing_config or {}
        self.strategy = str(self.config.get("strategy", "priority"))
        if self.strategy not in _STRATEGIES:
            raise ValueError(
                "routing.strategy must be one of: " + ", ".join(_STRATEGIES))
        self.candidates: List[str] = list(self.config.get("candidates", []))
        self.max_attempts = max(1, int(self.config.get("max_attempts", 3)))
        if fail_threshold is None:
            fail_threshold = self.config.get("fail_threshold", 3)
        self.fail_threshold = max(1, int(fail_threshold))
        self._health = {name: ProviderHealth(name) for name in self.providers}
        self._cursor = 0
        self._lock = threading.Lock()

    def health(self, name: str) -> ProviderHealth:
        """Return the health tracker for a provider, creating it if absent."""
        with self._lock:
            if name not in self._health:
                self._health[name] = ProviderHealth(name)
            return self._health[name]

    def all_health(self) -> List[Dict[str, Any]]:
        """Return health snapshots for every known provider."""
        return [tracker.to_dict() for tracker in self._health.values()]

    def order(self, request: ModelRequest) -> List[str]:
        """Return the candidate providers for a request, best first.

Falls back to the configured default when no strategy

        Demoted providers (past the failure threshold) are moved to the
        end but still retained, so a fully degraded fleet keeps working
        instead of failing closed.
        """
        base = self._base_order(request)
        healthy = [name for name in base
                   if self.health(name).consecutive_failures < self.fail_threshold]
        degraded = [name for name in base if name not in healthy]
        return healthy + degraded

    def _base_order(self, request: ModelRequest) -> List[str]:
        """Apply the configured strategy to produce a candidate order."""
        if self.strategy == "priority":
            return list(self.candidates)

        if self.strategy == "round_robin":
            if not self.candidates:
                return []
            with self._lock:
                start = self._cursor % len(self.candidates)
                self._cursor = (self._cursor + 1) % len(self.candidates)
            return self.candidates[start:] + self.candidates[:start]

        if self.strategy == "cheapest":
            tokens = estimate_tokens(request)
            expected_out = int(request.max_tokens or 256)
            priced: List[Tuple[float, int, str]] = []
            for index, name in enumerate(self.candidates):
                provider = self.providers.get(name)
                if provider is None:
                    continue
                try:
                    cost = provider.estimate_cost(tokens, expected_out, request.model)
                except Exception:
                    cost = float("inf")
                priced.append((cost, index, name))
            priced.sort()
            return [name for _, _, name in priced]

        # latency: unmeasured providers keep configured order so the first
        # calls still prefer the operator's stated preference.
        measured = [n for n in self.candidates if self.health(n).latency_ewma_ms > 0]
        unmeasured = [n for n in self.candidates if n not in measured]
        measured.sort(key=lambda name: self.health(name).latency_ewma_ms)
        return measured + unmeasured

    def complete(self, request: ModelRequest) -> DispatchResult:

        """Dispatch a request, trying candidates in order until one succeeds.

Errors
            RoutingError: When every candidate is exhausted or the failures
                encountered are not retryable.
        """
        order = self.order(request)
        if not order:
            raise RoutingError("no routing candidates configured")

        attempts: List[str] = []
        errors: List[str] = []
        budget = min(self.max_attempts, len(order))
        started = time.time()

        for name in order[:budget]:
            provider = self.providers.get(name)
            if provider is None:
                continue
            attempts.append(name)
            call_started = time.time()
            try:
                response = provider.complete(request)
            except ProviderError as exc:
                self.health(name).record_failure()
                errors.append(f"{name}: {exc.message}")
                if not is_retryable(exc):
                    raise RoutingError(
                        "non-retryable provider failure: " + errors[-1], attempts)
                continue
            elapsed_ms = (time.time() - call_started) * 1000.0
            self.health(name).record_success(elapsed_ms)
            return DispatchResult(
                response=response,
                provider=name,
                attempts=attempts,
                total_latency_ms=(time.time() - started) * 1000.0,
            )

        raise RoutingError(
            "all candidate providers failed: " + "; ".join(errors), attempts)
