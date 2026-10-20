"""Anthropic Messages API provider."""

from __future__ import annotations

from typing import Any, Dict, List

from .base import Provider, ProviderError, resolve_secret

DEFAULT_BASE_URL = "https://api.anthropic.com"

_PRICING = {
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
    "claude-3-5-haiku-20241022": (0.0008, 0.004),
    "claude-3-opus-20240229": (0.015, 0.075),
    "claude-3-haiku-20240307": (0.00025, 0.00125),
}


class AnthropicProvider(Provider):
    """Calls the Anthropic ``/v1/messages`` endpoint."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        super().__init__(name, config)
        self._api_key = resolve_secret(config, "api_key")
        self._base_url = str(config.get("base_url", DEFAULT_BASE_URL)).rstrip("/")

    def complete(self, request) -> Any:
        """Send a messages request, folding system messages into ``system``."""
        system_parts: List[str] = []
        messages: List[Dict[str, str]] = []
        if request.messages:
            for message in request.messages:
                role = str(message.get("role", "user"))
                content = str(message.get("content", ""))
                if role == "system":
                    system_parts.append(content)
                else:
                    messages.append({"role": role, "content": content})
        else:
            messages.append({"role": "user", "content": request.prompt})
        payload: Dict[str, Any] = {"messages": messages, "max_tokens": 1024}
        if request.model:
            payload["model"] = request.model
        elif self.default_model:
            payload["model"] = self.default_model
        else:
            raise ProviderError(self.name, "no model specified and no default_model configured")
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop_sequences"] = list(request.stop)
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key or "",
            "anthropic-version": "2023-06-01",
        }
        from .base import _post_json

        data = _post_json(self._base_url + "/v1/messages", headers, payload, self.timeout)
        try:
            blocks = data["content"]
            text_parts = [block.get("text", "") for block in blocks if isinstance(block, dict)]
            content = "\n".join(part for part in text_parts if part)
            usage_data = data.get("usage", {})
            usage_in = int(usage_data.get("input_tokens", 0) or 0)
            usage_out = int(usage_data.get("output_tokens", 0) or 0)
            from ..models import ModelResponse, Usage

            model = data.get("model") or self.default_model
            return ModelResponse(
                content=content,
                model=str(model) if model else None,
                provider=self.name,
                usage=Usage(usage_in, usage_out, stop_reason=data.get("stop_reason")),
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(self.name, f"unexpected response shape: {exc}") from exc

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
