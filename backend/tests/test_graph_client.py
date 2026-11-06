import json
import threading
from types import SimpleNamespace

import pytest

from app.config import Config
from app.utils import graph_client
from app.utils.graph_client import (
    BatchAddItem,
    GraphClient,
    NotFoundError,
    _BatchStore,
    _ClaudeClient,
    _EpisodeJob,
    _GraphitiRuntime,
    _page,
    _parse_time,
    build_ontology_models,
)


class FakeRuntime:
    """Runs episodes immediately, or holds them until released."""

    def __init__(self, hold=False, failures=None):
        self.hold = hold
        self.failures = failures or {}
        self.jobs = []
        self.states = {}

    def enqueue(self, job):
        self.jobs.append(job)
        self.states[job.episode_uuid] = {"processed": False, "error": None}
        if not self.hold:
            self.finish(job)

    def finish(self, job):
        error = self.failures.get(job.sequence_index)
        self.states[job.episode_uuid] = {"processed": error is None, "error": error}
        if job.batch_id is not None:
            graph_client._batches.record_result(job.batch_id, job.sequence_index, error)

    def episode_state(self, episode_uuid):
        if episode_uuid not in self.states:
            raise NotFoundError(body="missing")
        return self.states[episode_uuid]


@pytest.fixture
def batches(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(tmp_path))
    monkeypatch.setattr(graph_client, "_batches", _BatchStore())
    return GraphClient.batch


def use_runtime(monkeypatch, runtime):
    monkeypatch.setattr(graph_client, "_get_runtime", lambda: runtime)
    return runtime


def submit(batches, count=3):
    batch = batches.create(metadata={"graph_id": "graph-1"})
    items = batches.add(
        batch_id=batch.batch_id,
        items=[
            BatchAddItem(graph_id="graph-1", data=f"chunk {index}", source_description="doc")
            for index in range(count)
        ],
    )
    return batch.batch_id, items


def test_batch_starts_as_draft_and_preserves_order(batches, monkeypatch):
    runtime = use_runtime(monkeypatch, FakeRuntime(hold=True))

    batch_id, items = submit(batches)

    assert batches.get(batch_id).status == "draft"
    assert [item.sequence_index for item in items] == [0, 1, 2]
    assert all(item.episode_uuid for item in items)

    batches.process(batch_id)

    assert [job.body for job in runtime.jobs] == ["chunk 0", "chunk 1", "chunk 2"]
    assert batches.get(batch_id).status == "processing"


def test_successful_batch_reports_progress_and_items(batches, monkeypatch):
    use_runtime(monkeypatch, FakeRuntime())

    batch_id, _ = submit(batches)
    batches.process(batch_id)

    summary = batches.get(batch_id)
    assert summary.status == "succeeded"
    assert summary.progress.succeeded_items == 3
    assert summary.progress.percent_complete == 100

    page = batches.list_items(batch_id=batch_id, limit=2, cursor=None)
    assert [item.status for item in page.items] == ["succeeded", "succeeded"]
    assert page.next_cursor == 2
    assert batches.list_items(batch_id=batch_id, limit=2, cursor=2).next_cursor is None


@pytest.mark.parametrize(
    "failures, status",
    [({1: "boom"}, "partial"), ({0: "a", 1: "b", 2: "c"}, "failed")],
)
def test_failed_items_are_never_reported_as_success(batches, monkeypatch, failures, status):
    use_runtime(monkeypatch, FakeRuntime(failures=failures))

    batch_id, _ = submit(batches)
    batches.process(batch_id)

    assert batches.get(batch_id).status == status
    errors = [item.error for item in batches.list_items(batch_id=batch_id).items]
    assert failures[1 if status == "partial" else 0] in errors


def test_interrupted_batch_resumes_only_pending_items_after_restart(batches, monkeypatch):
    runtime = use_runtime(monkeypatch, FakeRuntime(hold=True))
    batch_id, _ = submit(batches)
    batches.process(batch_id)
    runtime.finish(runtime.jobs[0])

    # A new process has an empty in-memory view of active batches.
    monkeypatch.setattr(graph_client, "_batches", _BatchStore())
    restarted = use_runtime(monkeypatch, FakeRuntime())

    assert batches.get(batch_id).status == "succeeded"
    assert [job.body for job in restarted.jobs] == ["chunk 1", "chunk 2"]


def test_batch_can_be_found_by_metadata_after_a_lost_reply(batches, monkeypatch):
    use_runtime(monkeypatch, FakeRuntime())

    batch_id, _ = submit(batches)

    found = batches.list(limit=100, cursor=None).batches
    assert [batch.batch_id for batch in found] == [batch_id]
    assert found[0].metadata == {"graph_id": "graph-1"}


def test_missing_batch_and_unsafe_ids_are_not_found(batches):
    for batch_id in ("missing", "../../secret"):
        with pytest.raises(NotFoundError):
            batches.get(batch_id)


def test_episode_status_surfaces_failure_instead_of_waiting_forever(monkeypatch):
    runtime = use_runtime(monkeypatch, FakeRuntime(hold=True))

    episode = GraphClient.graph.add(graph_id="graph-1", data="activity")
    assert GraphClient.graph.episode.get(uuid_=episode.uuid_).processed is False

    runtime.states[episode.uuid_] = {"processed": True, "error": None}
    assert GraphClient.graph.episode.get(uuid_=episode.uuid_).processed is True

    runtime.states[episode.uuid_] = {"processed": False, "error": "extraction failed"}
    with pytest.raises(RuntimeError, match="extraction failed"):
        GraphClient.graph.episode.get(uuid_=episode.uuid_)


def test_add_keeps_the_activity_time_and_rejects_unsafe_graph_ids(monkeypatch):
    runtime = use_runtime(monkeypatch, FakeRuntime(hold=True))

    GraphClient.graph.add(
        graph_id="graph-1",
        data="activity",
        created_at="2026-01-02T03:04:05+00:00",
        source_description="simulation",
    )
    assert runtime.jobs[0].reference_time.isoformat() == "2026-01-02T03:04:05+00:00"
    assert runtime.jobs[0].source_description == "simulation"

    with pytest.raises(ValueError):
        GraphClient.graph.add(graph_id="../other", data="activity")


def test_ontology_is_saved_for_later_episodes(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(tmp_path))
    ontology = {
        "entity_types": [{"name": "Person", "description": "A person", "attributes": []}],
        "edge_types": [],
    }

    GraphClient.graph.set_ontology(graph_ids=["graph-1"], ontology=ontology)

    saved = tmp_path / "graphiti" / "ontologies" / "graph-1.json"
    assert json.loads(saved.read_text(encoding="utf-8")) == ontology


def test_ontology_models_have_descriptions_and_optional_fields():
    models = build_ontology_models({
        "entity_types": [{
            "name": "Person",
            "description": "A person",
            "attributes": [{"name": "role", "description": "Their role"}],
        }],
        "edge_types": [{
            "name": "KNOWS",
            "description": "Knows someone",
            "attributes": [],
            "source_targets": [
                {"source": "Person", "target": "Person"},
                {"source": "Person", "target": "Entity"},
            ],
        }],
    })

    person = models["entity_types"]["Person"]
    assert person.__doc__ == "A person"
    assert person(role=None).role is None
    assert person.model_fields["role"].description == "Their role"
    assert models["edge_type_map"] == {
        ("Person", "Person"): ["KNOWS"],
        ("Person", "Entity"): ["KNOWS"],
    }


def test_empty_ontology_lets_graphiti_use_its_defaults():
    assert build_ontology_models({}) == {
        "entity_types": None,
        "edge_types": None,
        "edge_type_map": None,
    }


def test_pages_only_offer_a_cursor_when_more_may_exist():
    items = [SimpleNamespace(uuid=f"id-{index}") for index in range(2)]

    assert _page(items, 2).headers == {graph_client.NEXT_CURSOR_HEADER: "id-1"}
    assert _page(items, 3).headers == {}
    assert _page([], 3).data == []


def test_invalid_times_fall_back_to_now_and_naive_times_become_utc():
    assert _parse_time("not a time").tzinfo is not None
    assert _parse_time("2026-01-02T03:04:05").utcoffset().total_seconds() == 0
    assert _parse_time("2026-01-02T03:04:05Z").year == 2026


def test_claude_wrapper_removes_only_the_rejected_setting():
    captured = {}

    class Messages:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return "response"

    import asyncio

    client = _ClaudeClient(SimpleNamespace(messages=Messages()))
    result = asyncio.run(
        client.messages.create(model="claude-haiku-5-5", temperature=1, max_tokens=5)
    )

    assert result == "response"
    assert captured == {"model": "claude-haiku-5-5", "max_tokens": 5}


def test_runtime_processes_each_graph_in_order_and_records_failures(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(tmp_path))
    monkeypatch.delenv("GRAPHITI_TELEMETRY_ENABLED", raising=False)
    runtime = _GraphitiRuntime()
    assert graph_client.os.environ["GRAPHITI_TELEMETRY_ENABLED"] == "false"

    order = []
    finished = threading.Event()

    async def fake_add(job):
        order.append(job.body)
        if job.body == "bad":
            raise RuntimeError("extraction failed")
        if job.body == "last":
            finished.set()

    monkeypatch.setattr(runtime, "_add_episode", fake_add)
    for index, body in enumerate(("first", "bad", "last")):
        runtime.enqueue(_EpisodeJob(
            episode_uuid=f"episode-{index}",
            graph_id="graph-1",
            body=body,
            source_description="",
            reference_time=_parse_time(None),
        ))

    assert finished.wait(5)
    assert order == ["first", "bad", "last"]
    assert runtime.episode_state("episode-0")["processed"] is True
    assert "extraction failed" in runtime.episode_state("episode-1")["error"]
    with pytest.raises(NotFoundError):
        runtime.episode_state("unknown")


def test_deleted_graph_does_not_run_its_remaining_episodes(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(tmp_path))
    runtime = _GraphitiRuntime()
    ran = []

    async def fake_add(job):
        ran.append(job.body)

    monkeypatch.setattr(runtime, "_add_episode", fake_add)
    runtime._cancelled_graphs.add("graph-1")

    async def run_directly():
        import asyncio

        queue = asyncio.Queue()
        queue.put_nowait(_EpisodeJob("episode-1", "graph-1", "text", "", _parse_time(None)))
        await runtime._drain("graph-1", queue)

    runtime.run(run_directly())

    assert ran == []
    assert "deleted" in runtime.episode_state("episode-1")["error"]
