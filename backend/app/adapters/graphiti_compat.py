"""Drop-in replacement for zep_cloud's Zep client backed by Graphiti + Neo4j.

This adapter provides the surface that MiroFish's services already call:

    client.graph.create(...)
    client.graph.set_ontology(...)
    client.graph.add_batch(...)
    client.graph.add(...)
    client.graph.episode.get(uuid_=...)
    client.graph.delete(graph_id=...)
    client.graph.search(...)
    client.graph.node.get(uuid_=...)
    client.graph.node.get_entity_edges(node_uuid=...)
    client.graph.node.get_by_graph_id(graph_id, limit=..., uuid_cursor=...)
    client.graph.edge.get_by_graph_id(graph_id, limit=..., uuid_cursor=...)

The Zep-side dataclasses (EpisodeData, EntityEdgeSourceTarget, InternalServerError)
and ontology base classes (EntityModel, EdgeModel, EntityText) are re-exported
from this module so callers only need to switch their import path.

Design notes
------------
- Graphiti is async; MiroFish services are sync. All coroutines are dispatched
  through the dedicated event loop in app.adapters.runtime.
- A single Graphiti instance lives at module scope. group_id corresponds to
  Zep's graph_id.
- Ontology defined via set_ontology() is cached per graph_id and replayed into
  add_episode(entity_types=..., edge_types=...) on every ingest call.
- Episode status: Graphiti add_episode is awaitable and synchronous from the
  caller's perspective. A finished call means "processed=True". We track UUIDs
  in a process-local set so episode.get(uuid_) keeps working without touching
  Neo4j again.
"""

from __future__ import annotations

import logging
import uuid as _uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Iterable

from graphiti_core import Graphiti
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.nodes import EpisodeType
from pydantic import BaseModel

from app.adapters.runtime import run_async

logger = logging.getLogger("mirofish.graphiti_compat")


# ---------------------------------------------------------------------------
# Public dataclasses & types — mirror the zep_cloud surface that MiroFish uses
# ---------------------------------------------------------------------------


class InternalServerError(Exception):
    """Compatibility shim for zep_cloud.InternalServerError."""


@dataclass
class EpisodeData:
    """Mirrors zep_cloud.EpisodeData — a single text chunk to ingest."""

    data: str
    type: str = "text"  # "text" | "message" | "json"


@dataclass
class EntityEdgeSourceTarget:
    """Mirrors zep_cloud.EntityEdgeSourceTarget — source/target entity types
    that a given edge type may connect."""

    source: str
    target: str


class EntityModel(BaseModel):
    """Pydantic base for ontology entity definitions (zep_cloud compatibility)."""


class EdgeModel(BaseModel):
    """Pydantic base for ontology edge definitions (zep_cloud compatibility)."""


# Zep used `Optional[EntityText]` as a marker type. We accept any string for it;
# at runtime it's only used as a type hint for dynamically-created Pydantic classes.
EntityText = str


# ---------------------------------------------------------------------------
# Graphiti singleton & global state
# ---------------------------------------------------------------------------

_graphiti: Graphiti | None = None
_graphiti_lock = Lock()
_ontologies: dict[str, dict[str, Any]] = {}  # graph_id -> {entities, edges}
_known_graphs: set[str] = set()
_processed_episodes: set[str] = set()


async def _build_graphiti() -> Graphiti:
    from app.config import Config

    api_key = Config.LLM_API_KEY
    base_url = Config.LLM_BASE_URL
    model = Config.LLM_MODEL_NAME
    if not api_key:
        raise InternalServerError("LLM_API_KEY is not configured; cannot init Graphiti.")
    if not Config.NEO4J_PASSWORD:
        raise InternalServerError("NEO4J_PASSWORD is not configured; cannot init Graphiti.")

    llm_config = LLMConfig(api_key=api_key, base_url=base_url, model=model)
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=api_key,
            base_url=base_url,
            embedding_model="text-embedding-3-small",
        )
    )
    client = Graphiti(
        uri=Config.NEO4J_URI,
        user=Config.NEO4J_USER,
        password=Config.NEO4J_PASSWORD,
        llm_client=OpenAIClient(config=llm_config),
        embedder=embedder,
    )
    await client.build_indices_and_constraints()
    logger.info("Graphiti client initialised against %s", Config.NEO4J_URI)
    return client


async def _get_graphiti() -> Graphiti:
    global _graphiti
    if _graphiti is None:
        _graphiti = await _build_graphiti()
    return _graphiti


# ---------------------------------------------------------------------------
# Cypher helpers (paging, node/edge lookups) — kept simple, no extra ORM
# ---------------------------------------------------------------------------


async def _run_cypher(query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    g = await _get_graphiti()
    driver = g.driver  # Graphiti exposes its Neo4j driver
    async with driver.session() as session:
        result = await session.run(query, params)
        return [dict(record) async for record in result]


def _ep_uuid(result: Any) -> str:
    ep = getattr(result, "episode", None)
    return getattr(ep, "uuid", None) or getattr(result, "uuid", "")


# ---------------------------------------------------------------------------
# Result wrappers that mimic Zep's response shapes
# ---------------------------------------------------------------------------


@dataclass
class _EpisodeStub:
    uuid_: str
    processed: bool = True

    @property
    def uuid(self) -> str:
        return self.uuid_


@dataclass
class _NodeRecord:
    uuid_: str
    name: str
    labels: list[str]
    summary: str
    attributes: dict[str, Any]
    group_id: str

    @property
    def uuid(self) -> str:
        return self.uuid_


@dataclass
class _EdgeRecord:
    uuid_: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    attributes: dict[str, Any]
    group_id: str
    valid_at: Any = None
    invalid_at: Any = None
    expired_at: Any = None

    @property
    def uuid(self) -> str:
        return self.uuid_


@dataclass
class _SearchResults:
    edges: list[_EdgeRecord]
    nodes: list[_NodeRecord]


def _node_from_record(row: dict[str, Any]) -> _NodeRecord:
    n = row.get("n", row)
    props = dict(n) if hasattr(n, "items") else n
    labels = list(getattr(n, "labels", []) or [])
    return _NodeRecord(
        uuid_=props.get("uuid", ""),
        name=props.get("name", ""),
        labels=labels or ["Entity"],
        summary=props.get("summary", ""),
        attributes=props.get("attributes", {}) or {},
        group_id=props.get("group_id", ""),
    )


def _edge_from_record(row: dict[str, Any]) -> _EdgeRecord:
    r = row.get("r", row)
    props = dict(r) if hasattr(r, "items") else r
    return _EdgeRecord(
        uuid_=props.get("uuid", ""),
        name=props.get("name", ""),
        fact=props.get("fact", ""),
        source_node_uuid=row.get("source_uuid", "") or props.get("source_uuid", ""),
        target_node_uuid=row.get("target_uuid", "") or props.get("target_uuid", ""),
        attributes=props.get("attributes", {}) or {},
        group_id=props.get("group_id", ""),
        valid_at=props.get("valid_at"),
        invalid_at=props.get("invalid_at"),
        expired_at=props.get("expired_at"),
    )


# ---------------------------------------------------------------------------
# Sub-clients matching the Zep dotted-path layout
# ---------------------------------------------------------------------------


class _NodeClient:
    def get(self, uuid_: str) -> _NodeRecord | None:
        rows = run_async(
            _run_cypher(
                "MATCH (n:Entity {uuid: $uuid}) RETURN n LIMIT 1",
                {"uuid": uuid_},
            )
        )
        return _node_from_record(rows[0]) if rows else None

    def get_entity_edges(self, node_uuid: str) -> list[_EdgeRecord]:
        rows = run_async(
            _run_cypher(
                """
                MATCH (src:Entity {uuid: $uuid})-[r:RELATES_TO]-(tgt:Entity)
                RETURN r, src.uuid AS source_uuid, tgt.uuid AS target_uuid
                """,
                {"uuid": node_uuid},
            )
        )
        return [_edge_from_record(row) for row in rows]

    def get_by_graph_id(
        self,
        graph_id: str,
        *,
        limit: int = 100,
        uuid_cursor: str | None = None,
    ) -> list[_NodeRecord]:
        cursor = uuid_cursor or ""
        rows = run_async(
            _run_cypher(
                """
                MATCH (n:Entity {group_id: $gid})
                WHERE n.uuid > $cursor
                RETURN n ORDER BY n.uuid LIMIT $limit
                """,
                {"gid": graph_id, "cursor": cursor, "limit": limit},
            )
        )
        return [_node_from_record(row) for row in rows]


class _EdgeClient:
    def get_by_graph_id(
        self,
        graph_id: str,
        *,
        limit: int = 100,
        uuid_cursor: str | None = None,
    ) -> list[_EdgeRecord]:
        cursor = uuid_cursor or ""
        rows = run_async(
            _run_cypher(
                """
                MATCH (src:Entity)-[r:RELATES_TO {group_id: $gid}]-(tgt:Entity)
                WHERE r.uuid > $cursor
                RETURN r, src.uuid AS source_uuid, tgt.uuid AS target_uuid
                ORDER BY r.uuid LIMIT $limit
                """,
                {"gid": graph_id, "cursor": cursor, "limit": limit},
            )
        )
        return [_edge_from_record(row) for row in rows]


class _EpisodeClient:
    def get(self, uuid_: str) -> _EpisodeStub:
        # add_episode is awaitable; once it returns we treat the episode as processed.
        return _EpisodeStub(uuid_=uuid_, processed=uuid_ in _processed_episodes)


class _GraphClient:
    def __init__(self) -> None:
        self.node = _NodeClient()
        self.edge = _EdgeClient()
        self.episode = _EpisodeClient()

    # -- lifecycle ------------------------------------------------------------

    def create(self, *, graph_id: str, name: str = "", description: str = "") -> dict[str, Any]:
        _known_graphs.add(graph_id)
        logger.info("graph.create graph_id=%s name=%s", graph_id, name)
        return {"graph_id": graph_id, "name": name, "description": description}

    def delete(self, *, graph_id: str) -> None:
        async def _delete():
            await _run_cypher(
                "MATCH (n {group_id: $gid}) DETACH DELETE n",
                {"gid": graph_id},
            )

        run_async(_delete())
        _known_graphs.discard(graph_id)
        _ontologies.pop(graph_id, None)
        logger.info("graph.delete graph_id=%s", graph_id)

    # -- ontology -------------------------------------------------------------

    def set_ontology(
        self,
        *,
        graph_ids: list[str],
        entities: dict[str, type[BaseModel]] | None = None,
        edges: dict[str, tuple[type[BaseModel], list[EntityEdgeSourceTarget]]] | None = None,
    ) -> None:
        for graph_id in graph_ids:
            _ontologies[graph_id] = {
                "entities": entities or {},
                "edges": edges or {},
            }
        logger.info(
            "graph.set_ontology graph_ids=%s entities=%d edges=%d",
            graph_ids,
            len(entities or {}),
            len(edges or {}),
        )

    def _ontology_for(self, graph_id: str) -> dict[str, Any]:
        return _ontologies.get(graph_id, {"entities": {}, "edges": {}})

    # -- ingestion -----------------------------------------------------------

    def add(self, *, graph_id: str, type: str = "text", data: str = "") -> _EpisodeStub:
        async def _add():
            g = await _get_graphiti()
            ontology = self._ontology_for(graph_id)
            source = _episode_source_from_type(type)
            result = await g.add_episode(
                name=f"{graph_id}-{_uuid.uuid4().hex[:8]}",
                episode_body=data,
                source=source,
                source_description=f"mirofish:{type}",
                reference_time=datetime.now(timezone.utc),
                group_id=graph_id,
                entity_types=ontology["entities"] or None,
                edge_types=ontology["edges"] or None,
            )
            uid = _ep_uuid(result)
            if uid:
                _processed_episodes.add(uid)
            return _EpisodeStub(uuid_=uid, processed=True)

        return run_async(_add())

    def add_batch(self, *, graph_id: str, episodes: Iterable[EpisodeData]) -> list[_EpisodeStub]:
        episodes_list = list(episodes)

        async def _bulk():
            g = await _get_graphiti()
            ontology = self._ontology_for(graph_id)
            results: list[_EpisodeStub] = []
            for idx, ep in enumerate(episodes_list):
                source = _episode_source_from_type(ep.type)
                result = await g.add_episode(
                    name=f"{graph_id}-batch-{_uuid.uuid4().hex[:8]}",
                    episode_body=ep.data,
                    source=source,
                    source_description=f"mirofish:{ep.type}",
                    reference_time=datetime.now(timezone.utc),
                    group_id=graph_id,
                    entity_types=ontology["entities"] or None,
                    edge_types=ontology["edges"] or None,
                )
                uid = _ep_uuid(result)
                if uid:
                    _processed_episodes.add(uid)
                results.append(_EpisodeStub(uuid_=uid, processed=True))
            return results

        return run_async(_bulk())

    # -- retrieval ----------------------------------------------------------

    def search(
        self,
        *,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
        reranker: str | None = None,
    ) -> _SearchResults:
        async def _search() -> _SearchResults:
            g = await _get_graphiti()
            edge_hits = await g.search(query=query, group_ids=[graph_id], num_results=limit)
            edges = [
                _EdgeRecord(
                    uuid_=getattr(e, "uuid", ""),
                    name=getattr(e, "name", ""),
                    fact=getattr(e, "fact", ""),
                    source_node_uuid=getattr(e, "source_node_uuid", ""),
                    target_node_uuid=getattr(e, "target_node_uuid", ""),
                    attributes=getattr(e, "attributes", {}) or {},
                    group_id=graph_id,
                    valid_at=getattr(e, "valid_at", None),
                    invalid_at=getattr(e, "invalid_at", None),
                    expired_at=getattr(e, "expired_at", None),
                )
                for e in edge_hits
            ]
            nodes: list[_NodeRecord] = []
            if scope == "nodes":
                rows = await _run_cypher(
                    """
                    CALL db.index.fulltext.queryNodes('node_name_and_summary', $q)
                    YIELD node, score
                    WHERE node.group_id = $gid
                    RETURN node AS n LIMIT $limit
                    """,
                    {"q": query, "gid": graph_id, "limit": limit},
                )
                nodes = [_node_from_record(row) for row in rows]
            return _SearchResults(edges=edges, nodes=nodes)

        return run_async(_search())


def _episode_source_from_type(type_: str) -> EpisodeType:
    type_clean = (type_ or "text").lower()
    if type_clean == "message":
        return EpisodeType.message
    if type_clean == "json":
        return EpisodeType.json
    return EpisodeType.text


# ---------------------------------------------------------------------------
# Top-level Zep-compatible client
# ---------------------------------------------------------------------------


class Zep:
    """Drop-in replacement for zep_cloud.client.Zep.

    Accepts the same constructor kwargs (api_key, base_url, ...) but ignores them
    — connection details come from Config (NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD,
    LLM_*).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Eagerly trigger init so misconfig fails fast at service-startup time
        # rather than on the first ingestion call.
        run_async(_get_graphiti())
        self.graph = _GraphClient()
