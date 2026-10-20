"""Provider registry.

Importing this package does not open sockets; use ``serve`` to start an
HTTP server at startup.

Maps provider ``type`` strings from the configuration to concrete provider
classes and exposes a single :func:`create_provider` factory used by the
HTTP server at startup.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from .anthropic import AnthropicProvider
from .azure import AzureProvider
from .base import Provider, ProviderError
from .ollama import OllamaProvider
from .openai import OpenAIProvider
from .openai_compat import OpenAICompatProvider

PROVIDER_TYPES: Dict[str, Callable[[str, Dict[str, Any]], Provider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "azure": AzureProvider,
    "ollama": OllamaProvider,
    "openai_compat": OpenAICompatProvider,
}


def create_provider(name: str, config: Dict[str, Any]) -> Provider:
    """Build a provider instance from its configuration.

    Args:
        name: Provider key under ``providers`` in the configuration.
        config: The provider's configuration mapping.

    Returns:
        A concrete :class:`Provider` instance.

    Raises:
        ProviderError: If the ``type`` field is missing or not registered.
    """
    provider_type = config.get("type")
    if not isinstance(provider_type, str):
        raise ProviderError(name, "missing required field 'type'")
    factory = PROVIDER_TYPES.get(provider_type)
    if factory is None:
        raise ProviderError(name, f"unknown provider type '{provider_type}'")
    return factory(name, config)
