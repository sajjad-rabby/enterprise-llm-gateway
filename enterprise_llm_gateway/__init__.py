"""A dependency-free gateway that puts one stable, OpenAI-compatible API in
front of many model providers, and adds the operational controls an
enterprise deployment actually needs: response caching, strategy-driven
routing with automatic failover, and an auditable spend ledger.

Typical use::

    from enterprise_llm_gateway import Gateway, ModelRequest

    gateway = Gateway.from_config_file("config.json")
    result = gateway.complete(ModelRequest(prompt="Summarise this incident."))
    print(result.response.content, result.provider)
"""

from __future__ import annotations

from .cache import CacheEntry, ResponseCache
from .config import ConfigError, load_config, validate_config
from .gateway import Gateway
from .ledger import LedgerRecord, UsageLedger
from .models import (
    ModelRequest,
    ModelResponse,
    Usage,
    cache_key_for_request,
)
from .providers import PROVIDER_TYPES, Provider, ProviderError, create_provider
from .router import (
    DispatchResult,
    ProviderHealth,
    Router,
    RoutingError,
    estimate_tokens,
)
from .server import GatewayHandler, GatewayServer

__version__ = "1.0.0"
__author__ = "Sajjad"

__all__ = [
    "__version__",
    "__author__",
    "CacheEntry",
    "ConfigError",
    "DispatchResult",
    "Gateway",
    "GatewayHandler",
    "GatewayServer",
    "LedgerRecord",
    "ModelRequest",
    "ModelResponse",
    "PROVIDER_TYPES",
    "Provider",
    "ProviderError",
    "ProviderHealth",
    "ResponseCache",
    "Router",
    "RoutingError",
    "Usage",
    "UsageLedger",
    "cache_key_for_request",
    "create_provider",
    "estimate_tokens",
    "load_config",
    "validate_config",
]
