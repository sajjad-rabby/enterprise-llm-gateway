"""Azure OpenAI Chat Completions provider."""

from __future__ import annotations

from typing import Any, Dict
from urllib.parse import urlencode

from .base import Provider, ProviderError, build_openai_messages, resolve_secret


class AzureProvider(Provider):
    """Calls an Azure OpenAI deployment's chat completions endpoint."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        super().__init__(name, config)
        self._api_key = resolve_secret(config, "api_key")
        endpoint = config.get("endpoint")
        if not endpoint or not str(endpoint).strip():
            raise ProviderError(name, "missing required field 'endpoint'")
        self._endpoint = str(endpoint).rstrip("/")
        self._deployment = str(config.get("deployment", "")).strip()
        if not self._deployment:
            raise ProviderError(name, "missing required field 'deployment'")
        self._api_version = str(config.get("api_version", "2024-02-01"))

    def complete(self, request) -> Any:
        """Send a chat completion request to the configured deployment."""
        payload = build_openai_messages(request)
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop"] = request.stop if len(request.stop) > 1 else request.stop[0]
        query = urlencode({"api-version": self._api_version})
        url = f"{self._endpoint}/openai/deployments/{self._deployment}/chat/completions?{query}"
        headers = {"Content-Type": "application/json", "api-key": self._api_key or ""}
        from .base import _post_json

        data = _post_json(url, headers, payload, self.timeout)
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
            usage_data = data.get("usage", {})
            from ..models import ModelResponse, Usage

            usage = Usage.from_dict(usage_data) if isinstance(usage_data, dict) else Usage(0, 0)
            model = data.get("model") or self._deployment
            return ModelResponse(
                content=str(content or ""),
                model=str(model),
                provider=self.name,
                usage=usage,
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(self.name, f"unexpected response shape: {exc}") from exc

    def estimate_cost(self, tokens_in: int, tokens_out: int, model=None) -> float:
        """Estimate USD cost from configured per-1K-token prices."""
        in_price, out_price = self._prices(model)
        return self._cost_from_prices(tokens_in, tokens_out, in_price, out_price)

    def _prices(self, model=None):
        custom = self.config.get("prices")
        if isinstance(custom, dict):
            for key, value in custom.items():
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    return float(value[0]), float(value[1])
        return 0.0, 0.0
