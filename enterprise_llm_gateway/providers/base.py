"""Provider abstraction and shared HTTP plumbing."""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..models import ModelRequest, ModelResponse


class ProviderError(Exception):
    """Raised when a provider call fails or returns an unusable response."""

    def __init__(self, provider: str, message: str,
                 status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.message = message
        self.status_code = status_code


def resolve_secret(config: Dict[str, Any], field: str) -> Optional[str]:
    """Resolve a credential, preferring ``<field>_env`` over an inline value."""
    env_name = config.get(field + "_env")
    if env_name:
        value = os.environ.get(env_name)
        if not value:
            raise ProviderError(
                config.get("name", "provider"),
                f"environment variable '{env_name}' is not set")
        return value
    inline = config.get(field)
    if inline:
        return str(inline)
    return None


def _post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any],
               timeout: float) -> Dict[str, Any]:
    """POST JSON and return the decoded response.

HTTP and transport failures are both wrapped so callers

    Raises ProviderError for network failures, non-2xx responses, and
    undecodable bodies so callers only handle one exception type.
    """
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers=dict(headers, **{"Content-Type": "application/json"}))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read(500).decode("utf-8", errors="replace")
        except Exception:
            pass
        raise ProviderError("provider", f"HTTP {exc.code}: {detail}",
                            status_code=exc.code)
    except (urllib.error.URLError, socket.timeout, TimeoutError,
            ConnectionError) as exc:
        raise ProviderError("provider", f"connection error: {exc}")
    try:
        return json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProviderError("provider", f"invalid JSON response: {exc}")


class Provider(ABC):
    """Base class every provider adapter must implement."""

    name = "provider"

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name = name
        self.config = config

    @property
    def timeout(self) -> float:
        """Per-request timeout in seconds."""
        return float(self.config.get("timeout_seconds", 60))

    @property
    def default_model(self) -> Optional[str]:
        """The model used when a request does not name one."""
        model = self.config.get("default_model")
        return str(model) if model else None

    @abstractmethod
    def complete(self, request: ModelRequest) -> ModelResponse:
        """Execute the request and return a normalized response."""

    @abstractmethod
    def estimate_cost(self, tokens_in: int, tokens_out: int,
                      model: Optional[str] = None) -> float:
        """Estimate the USD cost of a call from configured per-1k prices."""

    def _cost_from_prices(self, tokens_in: int, tokens_out: int,
                          price_in: float, price_out: float) -> float:
        """Compute USD cost from explicit per-1k-token prices."""
        return (tokens_in * float(price_in) + tokens_out * float(price_out)) / 1000.0


def build_openai_messages(request: ModelRequest) -> Dict[str, Any]:
    """Translate a :class:`ModelRequest` into an OpenAI request body.

Sampling fields are passed through to

    Returns a mapping with a ``messages`` list so callers can add
    ``model``, ``max_tokens`` and other sampling fields directly.
    """
    if request.messages:
        messages: List[Dict[str, Any]] = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in request.messages
        ]
    else:
        messages = [{"role": "user", "content": request.prompt}]
    return {"messages": messages}
