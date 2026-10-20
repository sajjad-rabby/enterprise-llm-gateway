"""Ollama local chat provider."""

from __future__ import annotations

from typing import Any, Dict

from .base import Provider, ProviderError, build_openai_messages

DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider(Provider):
    """Calls a local or remote Ollama ``/api/chat`` endpoint."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        super().__init__(name, config)
        self._base_url = str(config.get("base_url", DEFAULT_BASE_URL)).rstrip("/")

    def complete(self, request) -> Any:
        """Send a chat request; Ollama is free, so cost is zero."""
        payload = build_openai_messages(request)
        if request.model:
            payload["model"] = request.model
        elif self.default_model:
            payload["model"] = self.default_model
        else:
            raise ProviderError(self.name, "no model specified and no default_model configured")
        payload["stream"] = False
        options: Dict[str, Any] = {}
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.top_p is not None:
            options["top_p"] = request.top_p
        if request.stop:
            options["stop"] = list(request.stop)
        if options:
            payload["options"] = options
        headers = {"Content-Type": "application/json"}
        from .base import _post_json

        data = _post_json(self._base_url + "/api/chat", headers, payload, self.timeout)
        try:
            message = data["message"]
            content = message.get("content", "")
            usage_in = int(data.get("prompt_eval_count", 0) or 0)
            usage_out = int(data.get("eval_count", 0) or 0)
            from ..models import ModelResponse, Usage

            model = data.get("model") or self.default_model
            return ModelResponse(
                content=str(content or ""),
                model=str(model) if model else None,
                provider=self.name,
                usage=Usage(usage_in, usage_out, stop_reason=data.get("done_reason")),
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(self.name, f"unexpected response shape: {exc}") from exc

    def estimate_cost(self, tokens_in: int, tokens_out: int, model=None) -> float:
        """Ollama models run locally and cost nothing unless prices are set."""
        prices = self.config.get("prices")
        if isinstance(prices, dict):
            value = prices.get(model or self.default_model or "")
            if isinstance(value, (list, tuple)) and len(value) == 2:
                return self._cost_from_prices(
                    tokens_in, tokens_out, float(value[0]), float(value[1])
                )
        return 0.0
