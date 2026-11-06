from types import SimpleNamespace

import pytest
import torch

from app.simulation_runtime import (
    DEFAULT_AGENT_MEMORY_MESSAGES,
    RECSYS_BATCH_SIZE,
    _trim_records,
    agent_memory_limit,
    install_recsys_memory_fix,
    trim_agent_memories,
)


def record(role, text=""):
    return {"role_at_backend": role, "message": text}


def make_conversation(rounds):
    records = [record("system", "persona")]
    for index in range(rounds):
        records += [
            record("user", f"observe {index}"),
            record("assistant", f"call {index}"),
            record("function", f"result {index}"),
        ]
    return records


def test_short_history_is_unchanged():
    records = make_conversation(2)

    assert _trim_records(records, 20) is records


def test_trim_keeps_persona_and_recent_messages_from_a_user_turn():
    trimmed = _trim_records(make_conversation(10), 5)

    assert trimmed[0] == record("system", "persona")
    assert trimmed[1] == record("user", "observe 9")
    assert [r["role_at_backend"] for r in trimmed[1:]] == ["user", "assistant", "function"]


def test_trim_never_leaves_an_orphaned_tool_result():
    for limit in range(1, 12):
        conversation = _trim_records(make_conversation(10), limit)[1:]
        if conversation:
            assert conversation[0]["role_at_backend"] == "user"


class FakeStorage:
    def __init__(self, records):
        self.records = records
        self.clears = 0

    def load(self):
        return list(self.records)

    def clear(self):
        self.records = []
        self.clears += 1

    def save(self, records):
        self.records.extend(records)


def graph_with(storage):
    agent = SimpleNamespace(
        memory=SimpleNamespace(_chat_history_block=SimpleNamespace(storage=storage))
    )
    return SimpleNamespace(get_agents=lambda: [(0, agent)])


def test_trim_agent_memories_rewrites_only_long_histories():
    long_history = FakeStorage(make_conversation(10))
    trim_agent_memories(graph_with(long_history), limit=5)
    assert long_history.clears == 1
    assert len(long_history.records) == 4

    short_history = FakeStorage(make_conversation(1))
    trim_agent_memories(graph_with(short_history), limit=5)
    assert short_history.clears == 0


def test_zero_limit_disables_trimming():
    storage = FakeStorage(make_conversation(10))

    trim_agent_memories(graph_with(storage), limit=0)

    assert storage.clears == 0


@pytest.mark.parametrize(
    "value, expected",
    [("", DEFAULT_AGENT_MEMORY_MESSAGES), ("bad", DEFAULT_AGENT_MEMORY_MESSAGES), ("8", 8), ("0", 0), ("-3", 0)],
)
def test_memory_limit_setting(monkeypatch, value, expected):
    monkeypatch.setenv("SIMULATION_AGENT_MEMORY_MESSAGES", value)

    assert agent_memory_limit() == expected


def test_recsys_fix_batches_without_gradients(monkeypatch):
    from oasis.social_platform import process_recsys_posts, recsys

    monkeypatch.setattr(recsys, "generate_post_vector", recsys.generate_post_vector)
    batches = []

    def fake_process_batch(model, tokenizer, batch):
        batches.append(len(batch))
        assert not torch.is_grad_enabled()
        return torch.ones(len(batch), 4)

    monkeypatch.setattr(process_recsys_posts, "process_batch", fake_process_batch)

    install_recsys_memory_fix()
    vectors = recsys.generate_post_vector(None, None, ["text"] * 40, batch_size=1000)

    assert vectors.shape == (40, 4)
    assert max(batches) == RECSYS_BATCH_SIZE
    assert sum(batches) == 40
