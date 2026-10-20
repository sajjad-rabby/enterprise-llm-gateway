"""Core data types shared by every module.

The package is designed to run on a

These types are deliberately dependency-free so the gateway can run on a
bare Python 3.9+ interpreter with no third-party packages installed.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any, Dict, List, Optional


class Usage:
    """Token accounting for a single model response."""

    __slots__ = ("prompt_tokens", "completion_tokens", "total_tokens", "stop_reason")

    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0,
                 total_tokens: Optional[int] = None,
                 stop_reason: Optional[str] = None) -> None:
        self.prompt_tokens = int(prompt_tokens)
        self.completion_tokens = int(completion_tokens)
        if total_tokens is None:
            total_tokens = self.prompt_tokens + self.completion_tokens
        self.total_tokens = int(total_tokens)
        self.stop_reason = stop_reason

    def to_dict(self) -> Dict[str, int]:
        """Return the OpenAI-shaped usage object."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Usage":
        """Build a :class:`Usage` from an OpenAI-shaped usage object."""
        data = data or {}
        return cls(
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            total_tokens=int(data.get("total_tokens", 0)) or None,
        )


class ModelRequest:
    """A single completion request.

Requests are fingerprinted for cache lookup, so

    The gateway accepts either a raw ``prompt`` or a full ``messages``
    list. Metadata fields such as ``metadata`` and ``user`` travel with
    the request but are deliberately excluded from the cache key so that
    identical completions for different callers share one cache entry.
    """

    __slots__ = ("prompt", "messages", "model", "temperature", "max_tokens",
                 "stop", "top_p", "metadata", "user")

    def __init__(self, prompt: str = "", messages: Optional[List[Dict[str, Any]]] = None,
                 model: Optional[str] = None, temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 stop: Optional[List[str]] = None, top_p: Optional[float] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 user: Optional[str] = None) -> None:
        if not prompt and not messages:
            raise ValueError("request must include a prompt or messages")
        self.prompt = prompt
        self.messages = messages
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop
        self.top_p = top_p
        self.metadata = metadata or {}
        self.user = user

    def cache_key_payload(self) -> Dict[str, Any]:
        """Return the canonical subset of fields used for cache keys."""
        return {
            "prompt": self.prompt,
            "messages": self.messages,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": self.stop,
            "top_p": self.top_p,
        }


def cache_key_for_request(request: ModelRequest) -> str:
    """Compute the stable sha256 cache key for a request."""
    canonical = json.dumps(request.cache_key_payload(), sort_keys=True,
                           separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ModelResponse:
    """A normalized completion returned to the caller as OpenAI JSON."""

    def __init__(self, content: str, model: Optional[str], provider: str,
                 usage: Usage, finish_reason: str = "stop",
                 created_at: Optional[float] = None) -> None:
        self.content = str(content or "")
        self.model = str(model) if model else ""
        self.provider = provider
        self.usage = usage
        self.finish_reason = finish_reason or "stop"
        self.created_at = created_at if created_at is not None else time.time()

    @property
    def text(self) -> str:
        """Alias for :attr:`content` kept for readability at call sites."""
        return self.content

    def to_openai_json(self) -> Dict[str, Any]:
        """Render the response as an OpenAI ``chat.completion`` object."""
        return {
            "id": "gw-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(self.created_at),
            "model": self.model,
            "provider": self.provider,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": self.content},
                "finish_reason": self.finish_reason,
            }],
            "usage": self.usage.to_dict(),
        }

    @classmethod
    def from_openai_json(cls, data: Dict[str, Any], provider: str = "") -> "ModelResponse":
        """Parse an OpenAI-shaped completion into a :class:`ModelResponse`."""
        choices = data.get("choices")
        if not choices:
            raise ValueError("response contains no choices")
        choice = choices[0]
        message = choice.get("message") or {}
        return cls(
            content=message.get("content", ""),
            model=str(data.get("model", "")),
            provider=provider,
            usage=Usage.from_dict(data.get("usage")),
            finish_reason=str(choice.get("finish_reason", "stop")),
            created_at=float(data.get("created", time.time())),
        )
