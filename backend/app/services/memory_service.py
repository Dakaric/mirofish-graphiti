"""Memory service: native Graphiti + Neo4j wrapper for MiroFish.

This module replaces the temporary `app.adapters.graphiti_compat` shim.
It exposes a small, idiomatic API for everything MiroFish needs:

    register_ontology(group_id, entities, edges_with_targets)
    add_episode(group_id, content, source_type="text", reference_time=None)
    add_episodes_bulk(group_id, episodes)
    search_edges(group_id, query, limit=10)
    search_nodes(group_id, query, limit=10)
    get_node(uuid)
    get_node_edges(node_uuid)
    get_nodes_by_group(group_id, limit=100, cursor="")
    get_edges_by_group(group_id, limit=100, cursor="")
    delete_group(group_id)

`group_id` is the Graphiti name for what MiroFish calls `graph_id`.

The module owns a single Graphiti client and a dedicated background event
loop, since Graphiti is async-native and Flask handlers are sync. All public
functions are sync and dispatch into the loop via `run_async`.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid as _uuid
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Coroutine, Iterable

from graphiti_core import Graphiti
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.nodes import EpisodeType
from pydantic import BaseModel

logger = logging.getLogger("mirofish.memory")


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EdgeTarget:
    """Source/target labels for an edge type. Identical concept to Zep's
    EntityEdgeSourceTarget; kept as a plain dataclass with no SDK coupling."""

    source: str
    target: str


@dataclass
class EpisodeInput:
    """A single ingestion unit: free text + an optional source-type hint."""

    content: str
    source_type: str = "text"  # "text" | "message" | "json"


@dataclass
class NodeRecord:
    uuid: str
    name: str
    labels: list[str]
    summary: str
    attributes: dict[str, Any]
    group_id: str
    created_at: Any | None = None


@dataclass
class EdgeRecord:
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    attributes: dict[str, Any]
    group_id: str
    valid_at: Any | None = None
    invalid_at: Any | None = None
    expired_at: Any | None = None
    created_at: Any | None = None
    episodes: list[str] | None = None
    fact_type: str | None = None


@dataclass
class SearchResults:
    edges: list[EdgeRecord]
    nodes: list[NodeRecord]


# ---------------------------------------------------------------------------
# Async runtime: one daemon thread holds the event loop for the whole process
# ---------------------------------------------------------------------------


_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_lock = threading.Lock()


def _ensure_loop() -> asyncio.AbstractEventLoop:
    global _loop, _loop_thread
    if _loop is not None:
        return _loop
    with _loop_lock:
        if _loop is not None:
            return _loop
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True, name="mirofish-memory")
        thread.start()
        _loop = loop
        _loop_thread = thread
    return _loop


def run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run_coroutine_threadsafe(coro, _ensure_loop()).result()


def fire_and_forget(coro: Coroutine[Any, Any, Any]) -> Future:
    return asyncio.run_coroutine_threadsafe(coro, _ensure_loop())


# ---------------------------------------------------------------------------
# Graphiti client lifecycle
# ---------------------------------------------------------------------------


_graphiti: Graphiti | None = None
_ontologies: dict[str, dict[str, Any]] = {}


async def _build_graphiti() -> Graphiti:
    from app.config import Config

    if not Config.LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY is not configured")
    if not Config.NEO4J_PASSWORD:
        raise RuntimeError("NEO4J_PASSWORD is not configured")

    llm_config = LLMConfig(
        api_key=Config.LLM_API_KEY,
        base_url=Config.LLM_BASE_URL,
        model=Config.LLM_MODEL_NAME,
    )
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=Config.LLM_API_KEY,
            base_url=Config.LLM_BASE_URL,
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


async def _get_client() -> Graphiti:
    global _graphiti
    if _graphiti is None:
        _graphiti = await _build_graphiti()
    return _graphiti


def warmup() -> None:
    """Eagerly initialise the Graphiti client. Call once at app startup."""
    run_async(_get_client())


# ---------------------------------------------------------------------------
# Ontology
# ---------------------------------------------------------------------------


def register_ontology(
    group_id: str,
    entities: dict[str, type[BaseModel]] | None = None,
    edges: dict[str, tuple[type[BaseModel], list[EdgeTarget]]] | None = None,
) -> None:
    """Cache ontology classes for a group_id. Graphiti receives them on every
    add_episode call; we cache them once here so callers don't have to."""

    edge_types: dict[str, type[BaseModel]] = {}
    edge_type_map: dict[tuple[str, str], list[str]] = {}
    for edge_name, value in (edges or {}).items():
        if isinstance(value, tuple) and len(value) == 2:
            edge_class, targets = value
        else:
            edge_class, targets = value, []
        edge_types[edge_name] = edge_class
        for t in targets:
            src = getattr(t, "source", "Entity")
            tgt = getattr(t, "target", "Entity")
            edge_type_map.setdefault((src, tgt), []).append(edge_name)

    _ontologies[group_id] = {
        "entities": entities or {},
        "edge_types": edge_types,
        "edge_type_map": edge_type_map,
    }
    logger.info(
        "register_ontology group_id=%s entities=%d edges=%d edge_type_map=%d",
        group_id,
        len(entities or {}),
        len(edge_types),
        len(edge_type_map),
    )


def _ontology_for(group_id: str) -> dict[str, Any]:
    return _ontologies.get(group_id, {"entities": {}, "edge_types": {}, "edge_type_map": {}})


def clear_ontology(group_id: str) -> None:
    _ontologies.pop(group_id, None)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def _episode_type(source_type: str) -> EpisodeType:
    s = (source_type or "text").lower()
    if s == "message":
        return EpisodeType.message
    if s == "json":
        return EpisodeType.json
    return EpisodeType.text


def add_episode(
    group_id: str,
    content: str,
    source_type: str = "text",
    reference_time: datetime | None = None,
    name: str | None = None,
) -> str:
    """Ingest a single episode. Returns the new episode's UUID. Blocking —
    Graphiti runs several LLM calls for entity/edge extraction."""

    async def _run() -> str:
        client = await _get_client()
        ontology = _ontology_for(group_id)
        result = await client.add_episode(
            name=name or f"{group_id}-{_uuid.uuid4().hex[:8]}",
            episode_body=content,
            source=_episode_type(source_type),
            source_description=f"mirofish:{source_type}",
            reference_time=reference_time or datetime.now(timezone.utc),
            group_id=group_id,
            entity_types=ontology["entities"] or None,
            edge_types=ontology["edge_types"] or None,
            edge_type_map=ontology["edge_type_map"] or None,
        )
        episode = getattr(result, "episode", None)
        return getattr(episode, "uuid", "") or ""

    return run_async(_run())


def add_episodes_bulk(
    group_id: str,
    episodes: Iterable[EpisodeInput],
    name_prefix: str | None = None,
) -> list[str]:
    """Ingest several episodes. Returns their UUIDs in input order. Episodes
    are processed sequentially (Graphiti's bulk path requires identical
    ontologies; we want simplicity over peak throughput at PoC scale)."""

    episodes = list(episodes)
    prefix = name_prefix or f"{group_id}-batch"

    async def _run() -> list[str]:
        client = await _get_client()
        ontology = _ontology_for(group_id)
        uuids: list[str] = []
        for ep in episodes:
            result = await client.add_episode(
                name=f"{prefix}-{_uuid.uuid4().hex[:8]}",
                episode_body=ep.content,
                source=_episode_type(ep.source_type),
                source_description=f"mirofish:{ep.source_type}",
                reference_time=datetime.now(timezone.utc),
                group_id=group_id,
                entity_types=ontology["entities"] or None,
                edge_types=ontology["edge_types"] or None,
                edge_type_map=ontology["edge_type_map"] or None,
            )
            episode = getattr(result, "episode", None)
            uuids.append(getattr(episode, "uuid", "") or "")
        return uuids

    return run_async(_run())


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def _edge_from_graphiti(raw: Any, default_group_id: str = "") -> EdgeRecord:
    return EdgeRecord(
        uuid=getattr(raw, "uuid", ""),
        name=getattr(raw, "name", ""),
        fact=getattr(raw, "fact", ""),
        source_node_uuid=getattr(raw, "source_node_uuid", ""),
        target_node_uuid=getattr(raw, "target_node_uuid", ""),
        attributes=getattr(raw, "attributes", {}) or {},
        group_id=getattr(raw, "group_id", default_group_id) or default_group_id,
        valid_at=getattr(raw, "valid_at", None),
        invalid_at=getattr(raw, "invalid_at", None),
        expired_at=getattr(raw, "expired_at", None),
        created_at=getattr(raw, "created_at", None),
        episodes=list(getattr(raw, "episodes", None) or []),
        fact_type=getattr(raw, "fact_type", None) or getattr(raw, "name", None),
    )


def search_edges(group_id: str, query: str, limit: int = 10) -> list[EdgeRecord]:
    async def _run() -> list[EdgeRecord]:
        client = await _get_client()
        hits = await client.search(query=query, group_ids=[group_id], num_results=limit)
        return [_edge_from_graphiti(h, default_group_id=group_id) for h in hits]

    return run_async(_run())


async def _run_cypher(query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    client = await _get_client()
    async with client.driver.session() as session:
        result = await session.run(query, params)
        return [dict(record) async for record in result]


def search_nodes(group_id: str, query: str, limit: int = 10) -> list[NodeRecord]:
    async def _run() -> list[NodeRecord]:
        rows = await _run_cypher(
            """
            CALL db.index.fulltext.queryNodes('node_name_and_summary', $q)
            YIELD node, score
            WHERE node.group_id = $gid
            RETURN node AS n LIMIT $limit
            """,
            {"q": query, "gid": group_id, "limit": limit},
        )
        return [_node_from_row(row, group_id) for row in rows]

    return run_async(_run())


def _node_from_row(row: dict[str, Any], default_group_id: str = "") -> NodeRecord:
    n = row.get("n", row)
    props = dict(n) if hasattr(n, "items") else n
    labels = list(getattr(n, "labels", []) or [])
    return NodeRecord(
        uuid=props.get("uuid", ""),
        name=props.get("name", ""),
        labels=labels or ["Entity"],
        summary=props.get("summary", ""),
        attributes=props.get("attributes", {}) or {},
        group_id=props.get("group_id", default_group_id),
        created_at=props.get("created_at"),
    )


def _edge_from_row(row: dict[str, Any], default_group_id: str = "") -> EdgeRecord:
    r = row.get("r", row)
    props = dict(r) if hasattr(r, "items") else r
    return EdgeRecord(
        uuid=props.get("uuid", ""),
        name=props.get("name", ""),
        fact=props.get("fact", ""),
        source_node_uuid=row.get("source_uuid", "") or props.get("source_uuid", ""),
        target_node_uuid=row.get("target_uuid", "") or props.get("target_uuid", ""),
        attributes=props.get("attributes", {}) or {},
        group_id=props.get("group_id", default_group_id),
        valid_at=props.get("valid_at"),
        invalid_at=props.get("invalid_at"),
        expired_at=props.get("expired_at"),
        created_at=props.get("created_at"),
        episodes=list(props.get("episodes") or []),
        fact_type=props.get("fact_type") or props.get("name"),
    )


def get_node(uuid: str) -> NodeRecord | None:
    async def _run() -> NodeRecord | None:
        rows = await _run_cypher(
            "MATCH (n:Entity {uuid: $uuid}) RETURN n LIMIT 1",
            {"uuid": uuid},
        )
        return _node_from_row(rows[0]) if rows else None

    return run_async(_run())


def get_node_edges(node_uuid: str) -> list[EdgeRecord]:
    async def _run() -> list[EdgeRecord]:
        rows = await _run_cypher(
            """
            MATCH (src:Entity {uuid: $uuid})-[r:RELATES_TO]-(tgt:Entity)
            RETURN r, src.uuid AS source_uuid, tgt.uuid AS target_uuid
            """,
            {"uuid": node_uuid},
        )
        return [_edge_from_row(row) for row in rows]

    return run_async(_run())


def get_nodes_by_group(
    group_id: str,
    limit: int = 100,
    cursor: str | None = None,
) -> list[NodeRecord]:
    async def _run() -> list[NodeRecord]:
        rows = await _run_cypher(
            """
            MATCH (n:Entity {group_id: $gid})
            WHERE n.uuid > $cursor
            RETURN n ORDER BY n.uuid LIMIT $limit
            """,
            {"gid": group_id, "cursor": cursor or "", "limit": limit},
        )
        return [_node_from_row(row, group_id) for row in rows]

    return run_async(_run())


def get_edges_by_group(
    group_id: str,
    limit: int = 100,
    cursor: str | None = None,
) -> list[EdgeRecord]:
    async def _run() -> list[EdgeRecord]:
        rows = await _run_cypher(
            """
            MATCH (src:Entity)-[r:RELATES_TO {group_id: $gid}]-(tgt:Entity)
            WHERE r.uuid > $cursor
            RETURN r, src.uuid AS source_uuid, tgt.uuid AS target_uuid
            ORDER BY r.uuid LIMIT $limit
            """,
            {"gid": group_id, "cursor": cursor or "", "limit": limit},
        )
        return [_edge_from_row(row, group_id) for row in rows]

    return run_async(_run())


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def delete_group(group_id: str) -> None:
    async def _run() -> None:
        await _run_cypher("MATCH (n {group_id: $gid}) DETACH DELETE n", {"gid": group_id})

    run_async(_run())
    clear_ontology(group_id)
    logger.info("delete_group group_id=%s", group_id)
