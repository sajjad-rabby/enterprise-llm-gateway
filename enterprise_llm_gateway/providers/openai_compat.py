"""Generic OpenAI-compatible endpoint provider (vLLM, LiteLLM, LM Studio...)."""

from __future__ import annotations

from typing import Any, Dict

from .base import Provider, ProviderError, build_openai_messages, resolve_secret


class OpenAICompatProvider(Provider):
    """Calls any endpoint that mirrors the OpenAI chat completions shape."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        super().__init__(name, config)
        base_url = config.get("base_url")
        if not base_url or not str(base_url).strip():
            raise ProviderError(name, "missing required field 'base_url'")
        self._base_url = str(base_url).rstrip("/")
        self._api_key = resolve_secret(config, "api_key")

    def complete(self, request) -> Any:
        """Send a chat completion request with optional bearer auth."""
        payload = build_openai_messages(request)
        if request.model:
            payload["model"] = request.model
        elif self.default_model:
            payload["model"] = self.default_model
        else:
            raise ProviderError(self.name, "no model specified and no default_model configured")
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop"] = request.stop if len(request.stop) > 1 else request.stop[0]
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        from .base import _post_json

        data = _post_json(
            self._base_url + "/chat/completions", headers, payload, self.timeout
        )
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
            usage_data = data.get("usage", {})
            from ..models import ModelResponse, Usage

            usage = Usage.from_dict(usage_data) if isinstance(usage_data, dict) else Usage(0, 0)
            model = data.get("model") or self.default_model
            return ModelResponse(
                content=str(content or ""),
                model=str(model) if model else None,
                provider=self.name,
                usage=usage,
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(self.name, f"unexpected response shape: {exc}") from exc

    def estimate_cost(self, tokens_in: int, tokens_out: int, model=None) -> float:
        """Estimate USD cost from configured per-1K-token prices."""
        prices = self.config.get("prices")
        if isinstance(prices, dict):
            value = prices.get(model or self.default_model or "")
            if isinstance(value, (list, tuple)) and len(value) == 2:
                return self._cost_from_prices(
                    tokens_in, tokens_out, float(value[0]), float(value[1])
                )
        return 0.0
