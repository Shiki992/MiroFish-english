"""
Memory limits for long simulation runs.

OASIS keeps every message an agent has seen and embeds all posts in one batch
with gradient tracking on. Both grow with the run and with text length, which
can exhaust the container's memory partway through a simulation.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

RECSYS_BATCH_SIZE = 16
DEFAULT_AGENT_MEMORY_MESSAGES = 20


def agent_memory_limit() -> int:
    """Return how many recent messages each agent keeps. 0 means no limit."""
    value = os.environ.get("SIMULATION_AGENT_MEMORY_MESSAGES", "")
    try:
        return max(int(value), 0)
    except ValueError:
        return DEFAULT_AGENT_MEMORY_MESSAGES


def _trim_records(records: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """Keep system records and recent messages, starting at a user message.

    A kept history must not begin with an assistant turn or a tool result whose
    tool call was removed, because the model API rejects that.
    """
    system = [r for r in records if r.get("role_at_backend") == "system"]
    conversation = [r for r in records if r.get("role_at_backend") != "system"]
    if len(conversation) <= limit:
        return records

    recent = conversation[-limit:]
    while recent and recent[0].get("role_at_backend") != "user":
        recent = recent[1:]
    return system + recent


def trim_agent_memories(agent_graph: Any, limit: int | None = None) -> None:
    """Drop older messages from each agent's in-memory history."""
    limit = agent_memory_limit() if limit is None else limit
    if limit <= 0:
        return

    for _, agent in agent_graph.get_agents():
        storage = agent.memory._chat_history_block.storage
        records = storage.load()
        trimmed = _trim_records(records, limit)
        if len(trimmed) != len(records):
            storage.clear()
            storage.save(trimmed)


def install_recsys_memory_fix() -> None:
    """Make OASIS's Twitter recommender embed posts in small, gradient-free batches."""
    import torch
    from oasis.social_platform import process_recsys_posts, recsys

    def generate_post_vector(model, tokenizer, texts, batch_size):
        outputs = []
        with torch.no_grad():
            for start in range(0, len(texts), RECSYS_BATCH_SIZE):
                batch = texts[start:start + RECSYS_BATCH_SIZE]
                outputs.append(
                    process_recsys_posts.process_batch(model, tokenizer, batch).cpu()
                )
        return torch.cat(outputs, dim=0)

    recsys.generate_post_vector = generate_post_vector
