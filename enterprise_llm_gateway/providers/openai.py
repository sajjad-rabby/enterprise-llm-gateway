"""OpenAI Chat Completions provider."""

from __future__ import annotations

from typing import Any, Dict

from .base import Provider, ProviderError, build_openai_messages, resolve_secret

DEFAULT_BASE_URL = "https://api.openai.com"

_PRICING = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-4": (0.03, 0.06),
    "gpt-3.5-turbo": (0.0005, 0.0015),
}


class OpenAIProvider(Provider):
    """Calls the OpenAI ``/v1/chat/completions`` endpoint."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        super().__init__(name, config)
        self._api_key = resolve_secret(config, "api_key")
        self._base_url = str(config.get("base_url", DEFAULT_BASE_URL)).rstrip("/")

    def complete(self, request) -> Any:
        """Send a chat completion request and return a normalized response."""
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
        data = self._post("/chat/completions", headers, payload)
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
            usage_data = data.get("usage", {})
            return self._response(content, usage_data, data.get("model"))
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(self.name, f"unexpected response shape: {exc}") from exc

    def _post(self, path: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Any:
        from .base import _post_json

        return _post_json(self._base_url + path, headers, payload, self.timeout)

    def _response(self, content: Any, usage_data: Any, model: Any):
        from ..models import ModelResponse, Usage

        usage = Usage.from_dict(usage_data) if isinstance(usage_data, dict) else Usage(0, 0)
        if model is None and self.default_model:
            model = self.default_model
        return ModelResponse(
            content=str(content or ""),
            model=str(model) if model else None,
            provider=self.name,
            usage=usage,
        )

    def estimate_cost(self, tokens_in: int, tokens_out: int, model=None) -> float:
        """Estimate USD cost from the built-in price table."""
        prices = dict(_PRICING)
        custom = self.config.get("prices")
        if isinstance(custom, dict):
            for key, value in custom.items():
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    prices[str(key)] = (float(value[0]), float(value[1]))
        in_price, out_price = prices.get(model or self.default_model or "", (0.0, 0.0))
        return self._cost_from_prices(tokens_in, tokens_out, in_price, out_price)
