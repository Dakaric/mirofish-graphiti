"""Graph builder service.

Endpoint 2: Builds a standalone knowledge graph via the native
`memory_service` (Graphiti + Neo4j). The Zep-compat wrapper has been removed;
this module talks directly to the idiomatic memory API.
"""

import threading
import time
import uuid
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from ..models.task import TaskManager, TaskStatus
from ..utils.locale import get_locale, set_locale, t
from . import memory_service
from .memory_service import EdgeTarget, EpisodeInput
from .text_processor import TextProcessor


@dataclass
class GraphInfo:
    """Graph information"""
    graph_id: str
    node_count: int
    edge_count: int
    entity_types: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
        }


class GraphBuilderService:
    """Builds knowledge graphs by orchestrating the memory service."""

    # Names that Graphiti reserves on Entity/RELATES_TO and must not be used
    # as attribute names of user-defined ontology classes.
    RESERVED_ATTR_NAMES = {
        "uuid", "name", "group_id", "name_embedding", "summary", "created_at",
    }

    def __init__(self, api_key: Optional[str] = None):
        # api_key kept for backwards-compatible call sites; ignored.
        self._api_key = api_key
        self.task_manager = TaskManager()
        memory_service.warmup()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build_graph_async(
        self,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str = "MiroFish Graph",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 3,
    ) -> str:
        task_id = self.task_manager.create_task(
            task_type="graph_build",
            metadata={
                "graph_name": graph_name,
                "chunk_size": chunk_size,
                "text_length": len(text),
            },
        )
        current_locale = get_locale()

        thread = threading.Thread(
            target=self._build_graph_worker,
            args=(task_id, text, ontology, graph_name, chunk_size, chunk_overlap, batch_size, current_locale),
            daemon=True,
        )
        thread.start()
        return task_id

    def _build_graph_worker(
        self,
        task_id: str,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int,
        locale: str = "zh",
    ):
        set_locale(locale)
        try:
            self.task_manager.update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                progress=5,
                message=t("progress.startBuildingGraph"),
            )

            graph_id = self.create_graph(graph_name)
            self.task_manager.update_task(
                task_id,
                progress=10,
                message=t("progress.graphCreated", graphId=graph_id),
            )

            self.set_ontology(graph_id, ontology)
            self.task_manager.update_task(
                task_id,
                progress=15,
                message=t("progress.ontologySet"),
            )

            chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
            total_chunks = len(chunks)
            self.task_manager.update_task(
                task_id,
                progress=20,
                message=t("progress.textSplit", count=total_chunks),
            )

            self.add_text_batches(
                graph_id,
                chunks,
                batch_size,
                progress_callback=lambda msg, prog: self.task_manager.update_task(
                    task_id,
                    progress=20 + int(prog * 0.7),  # 20–90%
                    message=msg,
                ),
            )

            self.task_manager.update_task(
                task_id,
                progress=90,
                message=t("progress.fetchingGraphInfo"),
            )

            graph_info = self._get_graph_info(graph_id)

            self.task_manager.complete_task(task_id, {
                "graph_id": graph_id,
                "graph_info": graph_info.to_dict(),
                "chunks_processed": total_chunks,
            })

        except Exception as e:  # noqa: BLE001 — propagate as task failure
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.task_manager.fail_task(task_id, error_msg)

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------

    def create_graph(self, name: str) -> str:
        """Allocate a fresh group_id. Graphiti has no separate 'create graph'
        step — the group_id is established lazily on the first add_episode."""
        return f"mirofish_{uuid.uuid4().hex[:16]}"

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        """Translate MiroFish's JSON ontology into Pydantic classes and
        register them with the memory service."""
        warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

        def safe_attr_name(attr_name: str) -> str:
            return f"entity_{attr_name}" if attr_name.lower() in self.RESERVED_ATTR_NAMES else attr_name

        entity_types: Dict[str, type[BaseModel]] = {}
        for entity_def in ontology.get("entity_types", []):
            cls_name = entity_def["name"]
            description = entity_def.get("description", f"A {cls_name} entity.")
            attrs: Dict[str, Any] = {"__doc__": description}
            annotations: Dict[str, Any] = {}
            for attr_def in entity_def.get("attributes", []):
                attr_name = safe_attr_name(attr_def["name"])
                attrs[attr_name] = Field(description=attr_def.get("description", attr_name), default=None)
                annotations[attr_name] = Optional[str]
            attrs["__annotations__"] = annotations
            entity_class = type(cls_name, (BaseModel,), attrs)
            entity_class.__doc__ = description
            entity_types[cls_name] = entity_class

        edge_definitions: Dict[str, tuple[type[BaseModel], List[EdgeTarget]]] = {}
        for edge_def in ontology.get("edge_types", []):
            edge_name = edge_def["name"]
            description = edge_def.get("description", f"A {edge_name} relationship.")
            attrs = {"__doc__": description}
            annotations = {}
            for attr_def in edge_def.get("attributes", []):
                attr_name = safe_attr_name(attr_def["name"])
                attrs[attr_name] = Field(description=attr_def.get("description", attr_name), default=None)
                annotations[attr_name] = Optional[str]
            attrs["__annotations__"] = annotations

            class_name = "".join(word.capitalize() for word in edge_name.split("_"))
            edge_class = type(class_name, (BaseModel,), attrs)
            edge_class.__doc__ = description

            targets = [
                EdgeTarget(source=st.get("source", "Entity"), target=st.get("target", "Entity"))
                for st in edge_def.get("source_targets", [])
            ]
            if targets:
                edge_definitions[edge_name] = (edge_class, targets)

        memory_service.register_ontology(
            group_id=graph_id,
            entities=entity_types or None,
            edges=edge_definitions or None,
        )

    def add_text_batches(
        self,
        graph_id: str,
        chunks: List[str],
        batch_size: int = 3,
        progress_callback: Optional[Callable] = None,
    ) -> List[str]:
        """Ingest chunks into the graph in `batch_size` slices. Each chunk
        becomes one Graphiti episode; entity/edge extraction happens during
        the call. Returns all episode UUIDs in order."""
        episode_uuids: List[str] = []
        total_chunks = len(chunks)
        if total_chunks == 0:
            return episode_uuids

        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size

            if progress_callback:
                progress = (i + len(batch_chunks)) / total_chunks
                progress_callback(
                    t("progress.sendingBatch", current=batch_num, total=total_batches, chunks=len(batch_chunks)),
                    progress,
                )

            episodes = [EpisodeInput(content=chunk, source_type="text") for chunk in batch_chunks]
            try:
                uuids = memory_service.add_episodes_bulk(
                    group_id=graph_id,
                    episodes=episodes,
                    name_prefix=f"{graph_id}-batch-{batch_num}",
                )
                episode_uuids.extend(u for u in uuids if u)
                time.sleep(1)  # small pacing so progress updates render
            except Exception as e:  # noqa: BLE001
                if progress_callback:
                    progress_callback(t("progress.batchFailed", batch=batch_num, error=str(e)), 0)
                raise

        return episode_uuids

    # ------------------------------------------------------------------
    # Read paths
    # ------------------------------------------------------------------

    def _fetch_all_nodes(self, graph_id: str, page_size: int = 100, hard_cap: int = 2000) -> List:
        all_nodes = []
        cursor: Optional[str] = None
        while True:
            page = memory_service.get_nodes_by_group(graph_id, limit=page_size, cursor=cursor)
            if not page:
                break
            all_nodes.extend(page)
            if len(all_nodes) >= hard_cap:
                return all_nodes[:hard_cap]
            if len(page) < page_size:
                break
            cursor = page[-1].uuid
            if not cursor:
                break
        return all_nodes

    def _fetch_all_edges(self, graph_id: str, page_size: int = 100) -> List:
        all_edges = []
        cursor: Optional[str] = None
        while True:
            page = memory_service.get_edges_by_group(graph_id, limit=page_size, cursor=cursor)
            if not page:
                break
            all_edges.extend(page)
            if len(page) < page_size:
                break
            cursor = page[-1].uuid
            if not cursor:
                break
        return all_edges

    def _get_graph_info(self, graph_id: str) -> GraphInfo:
        nodes = self._fetch_all_nodes(graph_id)
        edges = self._fetch_all_edges(graph_id)
        entity_types = set()
        for node in nodes:
            for label in node.labels or []:
                if label not in ("Entity", "Node"):
                    entity_types.add(label)
        return GraphInfo(
            graph_id=graph_id,
            node_count=len(nodes),
            edge_count=len(edges),
            entity_types=sorted(entity_types),
        )

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        nodes = self._fetch_all_nodes(graph_id)
        edges = self._fetch_all_edges(graph_id)
        node_map = {node.uuid: node.name or "" for node in nodes}

        nodes_data = [
            {
                "uuid": node.uuid,
                "name": node.name,
                "labels": node.labels or [],
                "summary": node.summary or "",
                "attributes": node.attributes or {},
                "created_at": str(node.created_at) if node.created_at else None,
            }
            for node in nodes
        ]

        edges_data = []
        for edge in edges:
            edges_data.append({
                "uuid": edge.uuid,
                "name": edge.name or "",
                "fact": edge.fact or "",
                "fact_type": edge.fact_type or edge.name or "",
                "source_node_uuid": edge.source_node_uuid,
                "target_node_uuid": edge.target_node_uuid,
                "source_node_name": node_map.get(edge.source_node_uuid, ""),
                "target_node_name": node_map.get(edge.target_node_uuid, ""),
                "attributes": edge.attributes or {},
                "created_at": str(edge.created_at) if edge.created_at else None,
                "valid_at": str(edge.valid_at) if edge.valid_at else None,
                "invalid_at": str(edge.invalid_at) if edge.invalid_at else None,
                "expired_at": str(edge.expired_at) if edge.expired_at else None,
                "episodes": edge.episodes or [],
            })

        return {
            "graph_id": graph_id,
            "nodes": nodes_data,
            "edges": edges_data,
            "node_count": len(nodes_data),
            "edge_count": len(edges_data),
        }

    def delete_graph(self, graph_id: str):
        memory_service.delete_group(graph_id)
