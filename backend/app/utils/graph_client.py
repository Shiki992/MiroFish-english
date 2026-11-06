"""
Self-hosted knowledge graph client built on open-source Graphiti and FalkorDB.

Services use a small synchronous API (``client.graph``, ``client.batch``) while
Graphiti runs on one background event loop. Episodes for a graph are processed
one at a time, which Graphiti requires for correct entity resolution. Batch
state is saved to disk so document ingestion can resume after a restart.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Coroutine, Dict, List, Optional, Tuple

from ..config import Config
from .logger import get_logger

logger = get_logger("mirofish.graph_client")

NEXT_CURSOR_HEADER = "graph-next-cursor"
REQUEST_TIMEOUT_SECONDS = 180.0
RATE_LIMIT_ATTEMPTS = 5
RATE_LIMIT_BASE_DELAY_SECONDS = 5.0
GRAPH_ID_PATTERN = re.compile(r"[\w-]+")


class GraphApiError(Exception):
    """A graph operation failed with an HTTP-style status."""

    def __init__(
        self,
        status_code: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
        body: Any = None,
    ):
        super().__init__(f"Graph request failed ({status_code}): {body}")
        self.status_code = status_code
        self.headers = headers
        self.body = body


class NotFoundError(GraphApiError):
    """The requested graph, node, edge, batch, or episode does not exist."""

    def __init__(self, body: Any = None, headers: Optional[Dict[str, str]] = None):
        super().__init__(status_code=404, headers=headers, body=body)


@dataclass
class BatchAddItem:
    """One text chunk submitted to an ingestion batch."""

    graph_id: str
    data: str
    type: str = "graph_episode"
    data_type: str = "text"
    source_description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def _state_dir(*parts: str) -> Path:
    path = Path(Config.UPLOAD_FOLDER).resolve().joinpath("graphiti", *parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _validate_graph_id(graph_id: str) -> str:
    if not isinstance(graph_id, str) or not GRAPH_ID_PATTERN.fullmatch(graph_id):
        raise ValueError(f"Invalid graph ID: {graph_id!r}")
    return graph_id


def _parse_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def build_ontology_models(ontology: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a stored ontology into the Pydantic models Graphiti expects."""
    from pydantic import BaseModel, Field, create_model

    def model_for(definition: Dict[str, Any]) -> Any:
        fields = {
            attribute["name"]: (
                Optional[str],
                Field(default=None, description=attribute.get("description", "")),
            )
            for attribute in definition.get("attributes", [])
        }
        model = create_model(definition["name"], __base__=BaseModel, **fields)
        model.__doc__ = definition.get("description") or definition["name"]
        return model

    entity_types = {
        definition["name"]: model_for(definition)
        for definition in ontology.get("entity_types", [])
    }
    edge_types = {
        definition["name"]: model_for(definition)
        for definition in ontology.get("edge_types", [])
    }
    edge_type_map: Dict[Tuple[str, str], List[str]] = {}
    for definition in ontology.get("edge_types", []):
        for pair in definition.get("source_targets", []):
            key = (pair.get("source", "Entity"), pair.get("target", "Entity"))
            edge_type_map.setdefault(key, []).append(definition["name"])

    return {
        "entity_types": entity_types or None,
        "edge_types": edge_types or None,
        "edge_type_map": edge_type_map or None,
    }


def _node_view(node: Any) -> SimpleNamespace:
    return SimpleNamespace(
        uuid_=node.uuid,
        uuid=node.uuid,
        name=node.name,
        labels=list(node.labels or []),
        summary=node.summary or "",
        attributes=dict(node.attributes or {}),
        created_at=node.created_at,
    )


def _edge_view(edge: Any) -> SimpleNamespace:
    return SimpleNamespace(
        uuid_=edge.uuid,
        uuid=edge.uuid,
        name=edge.name,
        fact=edge.fact,
        fact_type=edge.name,
        source_node_uuid=edge.source_node_uuid,
        target_node_uuid=edge.target_node_uuid,
        attributes=dict(edge.attributes or {}),
        created_at=edge.created_at,
        valid_at=edge.valid_at,
        invalid_at=edge.invalid_at,
        expired_at=edge.expired_at,
        episodes=list(edge.episodes or []),
    )


class _ClaudeMessages:
    """Drops the sampling setting Graphiti sends, which current SDKs reject."""

    def __init__(self, messages: Any):
        self._messages = messages

    async def create(self, **kwargs: Any) -> Any:
        kwargs.pop("temperature", None)
        return await self._messages.create(**kwargs)


class _ClaudeClient:
    def __init__(self, client: Any):
        self.messages = _ClaudeMessages(client.messages)


def _create_llm_client() -> Any:
    from graphiti_core.llm_client.config import LLMConfig

    from ..llm_provider import PROVIDER_ANTHROPIC, anthropic_native_base_url

    model = Config.GRAPH_LLM_MODEL_NAME
    if Config.LLM_PROVIDER == PROVIDER_ANTHROPIC:
        from anthropic import AsyncAnthropic
        from graphiti_core.llm_client.anthropic_client import AnthropicClient

        return AnthropicClient(
            config=LLMConfig(model=model, small_model=model),
            client=_ClaudeClient(
                AsyncAnthropic(
                    api_key=Config.LLM_API_KEY,
                    base_url=anthropic_native_base_url(Config.LLM_BASE_URL),
                )
            ),
        )

    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

    return OpenAIGenericClient(
        config=LLMConfig(
            api_key=Config.LLM_API_KEY,
            base_url=Config.LLM_BASE_URL,
            model=model,
            small_model=model,
        )
    )


def _create_embedder() -> Any:
    from graphiti_core.embedder.client import EmbedderClient

    class LocalEmbedder(EmbedderClient):
        """Runs embeddings on this machine, with no API key or rate limit."""

        def __init__(self) -> None:
            self._model = None
            self._lock = threading.Lock()

        def _encode(self, texts: Any) -> Any:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(Config.GRAPH_EMBEDDING_MODEL)
                return self._model.encode(texts)

        async def create(self, input_data: Any) -> List[float]:
            text = input_data if isinstance(input_data, str) else input_data[0]
            return (await asyncio.to_thread(self._encode, text)).tolist()

        async def create_batch(self, input_data_list: List[str]) -> List[List[float]]:
            return (await asyncio.to_thread(self._encode, input_data_list)).tolist()

    return LocalEmbedder()


def _create_cross_encoder() -> Any:
    from graphiti_core.cross_encoder.client import CrossEncoderClient

    class UnusedCrossEncoder(CrossEncoderClient):
        """Search uses reciprocal rank fusion, so this never changes the order."""

        async def rank(self, query: str, passages: List[str]) -> List[Tuple[str, float]]:
            return [(passage, 1.0) for passage in passages]

    return UnusedCrossEncoder()


@dataclass
class _EpisodeJob:
    episode_uuid: str
    graph_id: str
    body: str
    source_description: str
    reference_time: datetime
    batch_id: Optional[str] = None
    sequence_index: Optional[int] = None


class _GraphitiRuntime:
    """Owns the event loop, one Graphiti per graph, and the ingestion queues."""

    def __init__(self) -> None:
        # Graphiti reads these when it is imported.
        os.environ["GRAPHITI_TELEMETRY_ENABLED"] = "false"
        os.environ["EMBEDDING_DIM"] = str(Config.GRAPH_EMBEDDING_DIM)
        os.environ.setdefault("SEMAPHORE_LIMIT", str(Config.GRAPH_MAX_CONCURRENCY))

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name="graphiti-runtime", daemon=True
        )
        self._thread.start()

        self._lock = threading.RLock()
        self._instances: Dict[str, Any] = {}
        self._queues: Dict[str, asyncio.Queue] = {}
        self._workers: Dict[str, asyncio.Task] = {}
        self._episodes: Dict[str, Dict[str, Any]] = {}
        self._cancelled_graphs: set = set()
        self._llm_client = None
        self._embedder = None
        self._cross_encoder = None

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coroutine: Coroutine, timeout: float = REQUEST_TIMEOUT_SECONDS) -> Any:
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(timeout)

    def _shared_clients(self) -> Tuple[Any, Any, Any]:
        with self._lock:
            if self._llm_client is None:
                self._llm_client = _create_llm_client()
                self._embedder = _create_embedder()
                self._cross_encoder = _create_cross_encoder()
            return self._llm_client, self._embedder, self._cross_encoder

    def graphiti(self, graph_id: str) -> Any:
        _validate_graph_id(graph_id)
        with self._lock:
            instance = self._instances.get(graph_id)
            if instance is not None:
                return instance

            from graphiti_core import Graphiti
            from graphiti_core.driver.falkordb_driver import FalkorDriver

            llm_client, embedder, cross_encoder = self._shared_clients()
            instance = Graphiti(
                graph_driver=FalkorDriver(
                    host=Config.FALKORDB_HOST,
                    port=Config.FALKORDB_PORT,
                    username=Config.FALKORDB_USERNAME,
                    password=Config.FALKORDB_PASSWORD,
                    database=graph_id,
                ),
                llm_client=llm_client,
                embedder=embedder,
                cross_encoder=cross_encoder,
                max_coroutines=Config.GRAPH_MAX_CONCURRENCY,
            )
            self._instances[graph_id] = instance
            return instance

    async def _list_graphs(self) -> List[str]:
        from falkordb.asyncio import FalkorDB

        client = FalkorDB(
            host=Config.FALKORDB_HOST,
            port=Config.FALKORDB_PORT,
            username=Config.FALKORDB_USERNAME,
            password=Config.FALKORDB_PASSWORD,
        )
        try:
            return list(await client.list_graphs())
        finally:
            await client.aclose()

    def list_graphs(self) -> List[str]:
        return self.run(self._list_graphs())

    def require_graph(self, graph_id: str) -> None:
        _validate_graph_id(graph_id)
        if graph_id not in self.list_graphs():
            raise NotFoundError(body=f"Graph not found: {graph_id}")

    def create_graph(self, graph_id: str) -> None:
        self.run(self.graphiti(graph_id).build_indices_and_constraints())

    async def _delete_graph(self, graph_id: str) -> None:
        driver = self.graphiti(graph_id).driver
        await driver.client.select_graph(graph_id).delete()

    def delete_graph(self, graph_id: str) -> None:
        self.require_graph(graph_id)
        with self._lock:
            self._cancelled_graphs.add(graph_id)
        self.run(self._delete_graph(graph_id))
        with self._lock:
            self._instances.pop(graph_id, None)
        (_state_dir("ontologies") / f"{graph_id}.json").unlink(missing_ok=True)

    def _load_ontology(self, graph_id: str) -> Dict[str, Any]:
        path = _state_dir("ontologies") / f"{graph_id}.json"
        if not path.exists():
            return {"entity_types": None, "edge_types": None, "edge_type_map": None}
        return build_ontology_models(json.loads(path.read_text(encoding="utf-8")))

    def enqueue(self, job: _EpisodeJob) -> None:
        with self._lock:
            self._cancelled_graphs.discard(job.graph_id)
            self._episodes[job.episode_uuid] = {"processed": False, "error": None}
        self._loop.call_soon_threadsafe(self._enqueue_on_loop, job)

    def _enqueue_on_loop(self, job: _EpisodeJob) -> None:
        queue = self._queues.setdefault(job.graph_id, asyncio.Queue())
        queue.put_nowait(job)
        worker = self._workers.get(job.graph_id)
        if worker is None or worker.done():
            self._workers[job.graph_id] = self._loop.create_task(
                self._drain(job.graph_id, queue)
            )

    async def _drain(self, graph_id: str, queue: asyncio.Queue) -> None:
        while not queue.empty():
            job = queue.get_nowait()
            error: Optional[str] = None
            if graph_id in self._cancelled_graphs:
                error = f"Graph {graph_id} was deleted before this episode ran"
            else:
                try:
                    await self._add_episode(job)
                except Exception as exception:
                    logger.error("Episode failed in graph %s: %s", graph_id, exception)
                    error = str(exception) or type(exception).__name__
            with self._lock:
                self._episodes[job.episode_uuid] = {"processed": error is None, "error": error}
            if job.batch_id is not None:
                _batches.record_result(job.batch_id, job.sequence_index, error)

    async def _add_episode(self, job: _EpisodeJob) -> None:
        from graphiti_core.llm_client.errors import RateLimitError
        from graphiti_core.nodes import EpisodeType

        ontology = await asyncio.to_thread(self._load_ontology, job.graph_id)
        for attempt in range(1, RATE_LIMIT_ATTEMPTS + 1):
            try:
                await self.graphiti(job.graph_id).add_episode(
                    name=f"episode-{job.episode_uuid}",
                    episode_body=job.body,
                    source=EpisodeType.text,
                    source_description=job.source_description or "MiroFish",
                    reference_time=job.reference_time,
                    group_id=job.graph_id,
                    **ontology,
                )
                return
            except RateLimitError:
                if attempt == RATE_LIMIT_ATTEMPTS:
                    raise
                delay = RATE_LIMIT_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                logger.warning("Rate limited; retrying episode in %.0fs", delay)
                await asyncio.sleep(delay)

    def episode_state(self, episode_uuid: str) -> Dict[str, Any]:
        with self._lock:
            state = self._episodes.get(episode_uuid)
        if state is None:
            raise NotFoundError(body=f"Episode not found: {episode_uuid}")
        return state


_runtime_lock = threading.Lock()
_runtime: Optional[_GraphitiRuntime] = None


def _get_runtime() -> _GraphitiRuntime:
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = _GraphitiRuntime()
        return _runtime


class _NodePages:
    def get_by_graph_id(
        self, graph_id: str, limit: int = 100, cursor: Optional[str] = None
    ) -> SimpleNamespace:
        from graphiti_core.nodes import EntityNode

        runtime = _get_runtime()
        runtime.require_graph(graph_id)
        nodes = runtime.run(
            EntityNode.get_by_group_ids(
                runtime.graphiti(graph_id).driver,
                [graph_id],
                limit=limit,
                uuid_cursor=cursor,
            )
        )
        return _page([_node_view(node) for node in nodes], limit)


class _EdgePages:
    def get_by_graph_id(
        self, graph_id: str, limit: int = 100, cursor: Optional[str] = None
    ) -> SimpleNamespace:
        from graphiti_core.edges import EntityEdge
        from graphiti_core.errors import GroupsEdgesNotFoundError

        runtime = _get_runtime()
        runtime.require_graph(graph_id)
        try:
            edges = runtime.run(
                EntityEdge.get_by_group_ids(
                    runtime.graphiti(graph_id).driver,
                    [graph_id],
                    limit=limit,
                    uuid_cursor=cursor,
                )
            )
        except GroupsEdgesNotFoundError:
            edges = []
        return _page([_edge_view(edge) for edge in edges], limit)


def _page(items: List[SimpleNamespace], limit: int) -> SimpleNamespace:
    headers = {}
    if len(items) >= limit and items:
        headers[NEXT_CURSOR_HEADER] = items[-1].uuid
    return SimpleNamespace(data=items, headers=headers)


class _NodeApi:
    with_raw_response = _NodePages()

    def get(self, uuid_: str, graph_id: Optional[str] = None) -> SimpleNamespace:
        from graphiti_core.errors import NodeNotFoundError
        from graphiti_core.nodes import EntityNode

        runtime = _get_runtime()
        for candidate in [graph_id] if graph_id else runtime.list_graphs():
            try:
                return _node_view(
                    runtime.run(
                        EntityNode.get_by_uuid(runtime.graphiti(candidate).driver, uuid_)
                    )
                )
            except NodeNotFoundError:
                continue
        raise NotFoundError(body=f"Node not found: {uuid_}")

    def get_edges(self, node_uuid: str, graph_id: Optional[str] = None) -> List[SimpleNamespace]:
        from graphiti_core.edges import EntityEdge

        runtime = _get_runtime()
        results: List[SimpleNamespace] = []
        for candidate in [graph_id] if graph_id else runtime.list_graphs():
            edges = runtime.run(
                EntityEdge.get_by_node_uuid(runtime.graphiti(candidate).driver, node_uuid)
            )
            results.extend(_edge_view(edge) for edge in edges)
        return results


class _EdgeApi:
    with_raw_response = _EdgePages()


class _EpisodeApi:
    def get(self, uuid_: str) -> SimpleNamespace:
        state = _get_runtime().episode_state(uuid_)
        if state["error"]:
            raise RuntimeError(f"Graph episode {uuid_} failed: {state['error']}")
        return SimpleNamespace(uuid_=uuid_, uuid=uuid_, processed=state["processed"])


class _GraphApi:
    node = _NodeApi()
    edge = _EdgeApi()
    episode = _EpisodeApi()

    def create(self, graph_id: str, name: str = "", description: str = "") -> SimpleNamespace:
        _get_runtime().create_graph(_validate_graph_id(graph_id))
        return SimpleNamespace(graph_id=graph_id, name=name, description=description)

    def get(self, graph_id: str) -> SimpleNamespace:
        _get_runtime().require_graph(graph_id)
        return SimpleNamespace(graph_id=graph_id)

    def delete(self, graph_id: str) -> None:
        _get_runtime().delete_graph(graph_id)

    def set_ontology(self, graph_ids: List[str], ontology: Dict[str, Any]) -> None:
        for graph_id in graph_ids:
            _write_json(
                _state_dir("ontologies") / f"{_validate_graph_id(graph_id)}.json",
                ontology,
            )

    def add(
        self,
        graph_id: str,
        data: str,
        type: str = "text",
        created_at: Any = None,
        source_description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SimpleNamespace:
        episode_uuid = str(uuid.uuid4())
        _get_runtime().enqueue(
            _EpisodeJob(
                episode_uuid=episode_uuid,
                graph_id=_validate_graph_id(graph_id),
                body=data,
                source_description=source_description,
                reference_time=_parse_time(created_at),
            )
        )
        return SimpleNamespace(uuid_=episode_uuid, uuid=episode_uuid, processed=False)

    def search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
        reranker: str = "rrf",
    ) -> SimpleNamespace:
        from graphiti_core.search.search_config_recipes import (
            EDGE_HYBRID_SEARCH_RRF,
            NODE_HYBRID_SEARCH_RRF,
        )

        runtime = _get_runtime()
        runtime.require_graph(graph_id)
        recipe = NODE_HYBRID_SEARCH_RRF if scope == "nodes" else EDGE_HYBRID_SEARCH_RRF
        # Recipes are shared module constants, so never change them in place.
        config = recipe.model_copy(deep=True)
        config.limit = limit
        results = runtime.run(
            runtime.graphiti(graph_id).search_(query, config=config, group_ids=[graph_id])
        )
        return SimpleNamespace(
            edges=[_edge_view(edge) for edge in results.edges],
            nodes=[_node_view(node) for node in results.nodes],
        )


class _BatchStore:
    """Saves batch state to disk so ingestion can resume after a restart."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active: set = set()

    @staticmethod
    def _path(batch_id: str) -> Path:
        if not GRAPH_ID_PATTERN.fullmatch(batch_id or ""):
            raise NotFoundError(body=f"Batch not found: {batch_id}")
        return _state_dir("batches") / f"{batch_id}.json"

    def load(self, batch_id: str) -> Dict[str, Any]:
        path = self._path(batch_id)
        if not path.exists():
            raise NotFoundError(body=f"Batch not found: {batch_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, batch: Dict[str, Any]) -> None:
        _write_json(self._path(batch["batch_id"]), batch)

    def create(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        batch = {
            "batch_id": f"batch_{uuid.uuid4().hex}",
            "metadata": metadata,
            "status": "draft",
            "items": [],
        }
        with self._lock:
            self.save(batch)
        return batch

    def add_items(self, batch_id: str, items: List[BatchAddItem]) -> List[Dict[str, Any]]:
        with self._lock:
            batch = self.load(batch_id)
            added = []
            for item in items:
                record = {
                    "sequence_index": len(batch["items"]),
                    "episode_uuid": str(uuid.uuid4()),
                    "status": "pending",
                    "error": None,
                    "graph_id": _validate_graph_id(item.graph_id),
                    "data": item.data,
                    "source_description": item.source_description,
                }
                batch["items"].append(record)
                added.append(record)
            self.save(batch)
        return added

    def process(self, batch_id: str) -> None:
        with self._lock:
            batch = self.load(batch_id)
            if batch["status"] == "draft":
                batch["status"] = "processing"
                self.save(batch)
            self._start_pending(batch)

    def _start_pending(self, batch: Dict[str, Any]) -> None:
        if batch["batch_id"] in self._active or batch["status"] != "processing":
            return
        self._active.add(batch["batch_id"])
        runtime = _get_runtime()
        for item in batch["items"]:
            if item["status"] == "pending":
                runtime.enqueue(
                    _EpisodeJob(
                        episode_uuid=item["episode_uuid"],
                        graph_id=item["graph_id"],
                        body=item["data"],
                        source_description=item["source_description"],
                        reference_time=datetime.now(timezone.utc),
                        batch_id=batch["batch_id"],
                        sequence_index=item["sequence_index"],
                    )
                )
        self._finish_if_done(batch)

    def get(self, batch_id: str) -> Dict[str, Any]:
        with self._lock:
            batch = self.load(batch_id)
            # A batch still marked processing with no local worker was
            # interrupted by a restart, so continue with its pending items.
            self._start_pending(batch)
            return self.load(batch_id)

    def record_result(self, batch_id: str, sequence_index: int, error: Optional[str]) -> None:
        with self._lock:
            batch = self.load(batch_id)
            item = batch["items"][sequence_index]
            item["status"] = "failed" if error else "succeeded"
            item["error"] = error
            self._finish_if_done(batch)
            self.save(batch)

    def _finish_if_done(self, batch: Dict[str, Any]) -> None:
        statuses = [item["status"] for item in batch["items"]]
        if "pending" in statuses:
            return
        failed = statuses.count("failed")
        if failed == 0:
            batch["status"] = "succeeded"
        else:
            batch["status"] = "failed" if failed == len(statuses) else "partial"
        self._active.discard(batch["batch_id"])
        self.save(batch)


_batches = _BatchStore()


def _batch_view(batch: Dict[str, Any]) -> SimpleNamespace:
    total = len(batch["items"])
    succeeded = sum(item["status"] == "succeeded" for item in batch["items"])
    finished = sum(item["status"] != "pending" for item in batch["items"])
    return SimpleNamespace(
        batch_id=batch["batch_id"],
        metadata=batch["metadata"],
        status=batch["status"],
        progress=SimpleNamespace(
            succeeded_items=succeeded,
            percent_complete=(finished / total * 100) if total else 0,
        ),
    )


def _item_view(item: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        sequence_index=item["sequence_index"],
        episode_uuid=item["episode_uuid"],
        source_uuid=None,
        status=item["status"],
        error=item["error"],
    )


class _BatchApi:
    def create(self, metadata: Optional[Dict[str, Any]] = None) -> SimpleNamespace:
        return _batch_view(_batches.create(metadata or {}))

    def add(self, batch_id: str, items: List[BatchAddItem]) -> List[SimpleNamespace]:
        return [_item_view(item) for item in _batches.add_items(batch_id, items)]

    def process(self, batch_id: str) -> None:
        _batches.process(batch_id)

    def get(self, batch_id: str) -> SimpleNamespace:
        return _batch_view(_batches.get(batch_id))

    def list(self, limit: int = 100, cursor: Optional[int] = None) -> SimpleNamespace:
        batches = [
            _batch_view(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(_state_dir("batches").glob("*.json"))
        ]
        return SimpleNamespace(batches=batches, next_cursor=None)

    def list_items(
        self, batch_id: str, limit: int = 100, cursor: Optional[int] = None
    ) -> SimpleNamespace:
        items = _batches.load(batch_id)["items"]
        start = cursor or 0
        end = start + limit
        return SimpleNamespace(
            items=[_item_view(item) for item in items[start:end]],
            next_cursor=end if end < len(items) else None,
        )


class GraphClient:
    """Synchronous entry point for graph storage, ingestion, and search."""

    graph = _GraphApi()
    batch = _BatchApi()


def wait_until_ready(timeout: float = 60.0) -> None:
    """Block until FalkorDB accepts connections, or raise the last error."""
    deadline = time.time() + timeout
    while True:
        try:
            _get_runtime().list_graphs()
            return
        except Exception:
            if time.time() >= deadline:
                raise
            time.sleep(2)
