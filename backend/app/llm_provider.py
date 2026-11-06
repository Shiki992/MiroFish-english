"""
LLM provider selection helpers.

Kept free of provider SDK imports at module level so that configuration and the
simulation scripts can use it without constructing any client.
"""

from __future__ import annotations

from typing import Any, Optional

PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
SUPPORTED_PROVIDERS = {PROVIDER_OPENAI, PROVIDER_ANTHROPIC}
SUPPORTED_EFFORTS = {"low", "medium", "high", "xhigh", "max"}

# Thinking counts toward max_tokens on Claude, so simulation agents need more
# room than their short visible actions suggest.
ANTHROPIC_SIMULATION_MAX_TOKENS = 16000
# Context kept per agent request. One observation can exceed 18,000 tokens.
ANTHROPIC_SIMULATION_CONTEXT_TOKENS = 48000


def resolve_provider(explicit: Optional[str], model: Optional[str]) -> str:
    """Return the configured provider, inferring it from a claude-* model name.

    An explicit value always wins, so a proxy that serves claude-* names over
    the OpenAI format can set LLM_PROVIDER=openai.
    """
    if explicit and explicit.strip():
        return explicit.strip().lower()
    if (model or "").strip().lower().startswith("claude"):
        return PROVIDER_ANTHROPIC
    return PROVIDER_OPENAI


def anthropic_native_base_url(base_url: Optional[str]) -> Optional[str]:
    """Return the URL for the Anthropic SDK, which appends /v1 itself."""
    if not base_url or not base_url.strip():
        return None
    url = base_url.strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return url or None


def anthropic_compatible_base_url(base_url: Optional[str]) -> Optional[str]:
    """Return the URL for CAMEL's Anthropic backend, which needs the /v1/ path."""
    native = anthropic_native_base_url(base_url)
    return f"{native}/v1/" if native else None


def create_anthropic_camel_model(
    *,
    api_key: Optional[str],
    base_url: Optional[str],
    model: str,
) -> Any:
    """Create the CAMEL backend that OASIS simulation agents use for Claude.

    CAMEL reaches Claude through Anthropic's OpenAI-compatible endpoint, which
    ignores effort settings, so LLM_EFFORT does not apply to these agents.
    """
    from camel.models import AnthropicModel

    class SimulationClaudeModel(AnthropicModel):
        # CAMEL otherwise uses max_tokens, the output cap, as the context limit.
        @property
        def token_limit(self) -> int:
            return ANTHROPIC_SIMULATION_CONTEXT_TOKENS

    return SimulationClaudeModel(
        model_type=model,
        # An explicit config keeps CAMEL from sending its unset defaults.
        model_config_dict={"max_tokens": ANTHROPIC_SIMULATION_MAX_TOKENS},
        api_key=api_key,
        url=anthropic_compatible_base_url(base_url),
    )
