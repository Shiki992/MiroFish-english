"""
Native Claude client behind the response shape the rest of the backend reads.

Callers read ``response.choices[0].message.content`` and ``.finish_reason``, so
this adapter returns that shape from the Anthropic Messages API. Sampling
parameters are not sent because current Claude models reject them.
"""

from __future__ import annotations

import logging
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from anthropic import Anthropic

from ..llm_provider import anthropic_native_base_url


logger = logging.getLogger(__name__)

# Requests stream, so long outputs do not hit HTTP timeouts.
DEFAULT_MAX_TOKENS = 64000
# Callers set caps for visible output, but thinking also counts toward
# max_tokens on Claude. A low cap would truncate the answer.
MIN_MAX_TOKENS = 16000

# Claude API models that accept server-side refusal fallbacks.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = {
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
}

JSON_INSTRUCTION = (
    "Respond with a single valid JSON object and nothing else. "
    "Do not wrap it in Markdown or add any text before or after it."
)

FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "refusal": "content_filter",
}


def _split_system_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Move system messages to Claude's top-level system prompt."""
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    conversation = [m for m in messages if m["role"] != "system"]
    return "\n\n".join(system_parts), conversation


def _strip_json_fence(text: str) -> str:
    cleaned = re.sub(r"^\s*```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    return re.sub(r"\n?```\s*$", "", cleaned).strip()


class AnthropicChatClient:
    """Sends chat requests through the native Anthropic SDK."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        effort: Optional[str] = None,
    ):
        native_url = anthropic_native_base_url(base_url)
        self.client = Anthropic(api_key=api_key, base_url=native_url)
        self.effort = effort
        self._first_party = (
            native_url is None
            or urlparse(native_url).hostname == "api.anthropic.com"
        )

    def create_chat_completion(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Any:
        json_requested = response_format is not None
        system, conversation = _split_system_messages(messages)
        if json_requested:
            system = f"{system}\n\n{JSON_INSTRUCTION}" if system else JSON_INSTRUCTION

        params: Dict[str, Any] = {
            "model": model,
            "max_tokens": (
                DEFAULT_MAX_TOKENS
                if max_tokens is None
                else max(max_tokens, MIN_MAX_TOKENS)
            ),
            "messages": conversation,
        }
        if system:
            params["system"] = system
        if self.effort:
            params["output_config"] = {"effort": self.effort}

        if self._first_party and model in FALLBACK_MODELS:
            stream_context = self.client.beta.messages.stream(
                betas=[FALLBACK_BETA],
                fallbacks="default",
                **params,
            )
        else:
            stream_context = self.client.messages.stream(**params)

        with stream_context as stream:
            message = stream.get_final_message()

        if message.stop_reason == "refusal":
            logger.warning("Claude declined the request (model=%s)", model)

        text = "".join(
            block.text for block in message.content if block.type == "text"
        )
        if json_requested:
            text = _strip_json_fence(text)

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=FINISH_REASONS.get(
                        message.stop_reason, message.stop_reason
                    ),
                    message=SimpleNamespace(content=text),
                )
            ]
        )
