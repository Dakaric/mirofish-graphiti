"""
Zep retrieval tools service
Wraps graph search, node reading and edge querying tools for the Report Agent.

Core retrieval tools (post-optimization):
1. InsightForge - the most powerful hybrid retrieval, auto-generates sub-questions and runs multi-dimensional searches
2. PanoramaSearch - breadth search returning the full picture, including expired content
3. QuickSearch - quick semantic lookup
"""

import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from . import memory_service

from ..config import Config
from ..utils.logger import get_logger
from ..utils.llm_client import LLMClient
from ..utils.locale import get_locale, t

logger = get_logger('mirofish.zep_tools')


def _paginate(fetcher, graph_id: str, page_size: int = 100, hard_cap: int = 2000):
    """Walk all pages of a memory_service paged fetcher (get_nodes_by_group /
    get_edges_by_group) and return the flattened list. Uses uuid-cursor paging."""
    items = []
    cursor: Optional[str] = None
    while True:
        page = fetcher(graph_id, limit=page_size, cursor=cursor)
        if not page:
            break
        items.extend(page)
        if len(items) >= hard_cap:
            return items[:hard_cap]
        if len(page) < page_size:
            break
        cursor = page[-1].uuid
        if not cursor:
            break
    return items


@dataclass
class SearchResult:
    """Search result"""
    facts: List[str]
    edges: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    query: str
    total_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": self.facts,
            "edges": self.edges,
            "nodes": self.nodes,
            "query": self.query,
            "total_count": self.total_count
        }
    
    def to_text(self) -> str:
        """Convert to text format for LLM consumption (German because the result is fed into LLM prompts)"""
        text_parts = [f"Suchanfrage: {self.query}", f"{self.total_count} relevante Treffer gefunden"]

        if self.facts:
            text_parts.append("\n### Relevante Fakten:")
            for i, fact in enumerate(self.facts, 1):
                text_parts.append(f"{i}. {fact}")

        return "\n".join(text_parts)


@dataclass
class NodeInfo:
    """Node info"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes
        }
    
    def to_text(self) -> str:
        """Convert to text format (German — destined for LLM prompts)"""
        entity_type = next((l for l in self.labels if l not in ["Entity", "Node"]), "Unbekannter Typ")
        return f"Entität: {self.name} (Typ: {entity_type})\nZusammenfassung: {self.summary}"


@dataclass
class EdgeInfo:
    """Edge info"""
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None
    # Temporal information
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "fact": self.fact,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "source_node_name": self.source_node_name,
            "target_node_name": self.target_node_name,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "expired_at": self.expired_at
        }
    
    def to_text(self, include_temporal: bool = False) -> str:
        """Convert to text format (German — destined for LLM prompts)"""
        source = self.source_node_name or self.source_node_uuid[:8]
        target = self.target_node_name or self.target_node_uuid[:8]
        base_text = f"Beziehung: {source} --[{self.name}]--> {target}\nFakt: {self.fact}"

        if include_temporal:
            valid_at = self.valid_at or "unbekannt"
            invalid_at = self.invalid_at or "fortlaufend"
            base_text += f"\nGültigkeit: {valid_at} - {invalid_at}"
            if self.expired_at:
                base_text += f" (abgelaufen: {self.expired_at})"

        return base_text

    @property
    def is_expired(self) -> bool:
        """Whether the edge has expired"""
        return self.expired_at is not None

    @property
    def is_invalid(self) -> bool:
        """Whether the edge has been invalidated"""
        return self.invalid_at is not None


@dataclass
class InsightForgeResult:
    """
    InsightForge deep retrieval result.
    Contains the results for multiple sub-questions plus an aggregated analysis.
    """
    query: str
    simulation_requirement: str
    sub_queries: List[str]

    # Per-dimension retrieval results
    semantic_facts: List[str] = field(default_factory=list)  # semantic search hits
    entity_insights: List[Dict[str, Any]] = field(default_factory=list)  # entity insights
    relationship_chains: List[str] = field(default_factory=list)  # relationship chains

    # Stats
    total_facts: int = 0
    total_entities: int = 0
    total_relationships: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "simulation_requirement": self.simulation_requirement,
            "sub_queries": self.sub_queries,
            "semantic_facts": self.semantic_facts,
            "entity_insights": self.entity_insights,
            "relationship_chains": self.relationship_chains,
            "total_facts": self.total_facts,
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships
        }
    
    def to_text(self) -> str:
        """Convert to a detailed German text format (consumed by LLM prompts)"""
        text_parts = [
            f"## Tiefenanalyse der Zukunftsprognose",
            f"Analysefrage: {self.query}",
            f"Prognoseszenario: {self.simulation_requirement}",
            f"\n### Statistik der Prognosedaten",
            f"- Relevante Prognose-Fakten: {self.total_facts}",
            f"- Beteiligte Entitäten: {self.total_entities}",
            f"- Beziehungsketten: {self.total_relationships}"
        ]

        # Sub-questions
        if self.sub_queries:
            text_parts.append(f"\n### Analysierte Teilfragen")
            for i, sq in enumerate(self.sub_queries, 1):
                text_parts.append(f"{i}. {sq}")

        # Semantic search results
        if self.semantic_facts:
            text_parts.append(f"\n### [Kernfakten] (bitte im Bericht wörtlich zitieren)")
            for i, fact in enumerate(self.semantic_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")

        # Entity insights
        if self.entity_insights:
            text_parts.append(f"\n### [Kernentitäten]")
            for entity in self.entity_insights:
                text_parts.append(f"- **{entity.get('name', 'unbekannt')}** ({entity.get('type', 'Entität')})")
                if entity.get('summary'):
                    text_parts.append(f"  Zusammenfassung: \"{entity.get('summary')}\"")
                if entity.get('related_facts'):
                    text_parts.append(f"  Verwandte Fakten: {len(entity.get('related_facts', []))}")

        # Relationship chains
        if self.relationship_chains:
            text_parts.append(f"\n### [Beziehungsketten]")
            for chain in self.relationship_chains:
                text_parts.append(f"- {chain}")

        return "\n".join(text_parts)


@dataclass
class PanoramaResult:
    """
    Panorama breadth-search result.
    Contains all related information including expired content.
    """
    query: str

    # All nodes
    all_nodes: List[NodeInfo] = field(default_factory=list)
    # All edges (including expired)
    all_edges: List[EdgeInfo] = field(default_factory=list)
    # Currently active facts
    active_facts: List[str] = field(default_factory=list)
    # Expired/invalid facts (historical record)
    historical_facts: List[str] = field(default_factory=list)

    # Stats
    total_nodes: int = 0
    total_edges: int = 0
    active_count: int = 0
    historical_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "all_nodes": [n.to_dict() for n in self.all_nodes],
            "all_edges": [e.to_dict() for e in self.all_edges],
            "active_facts": self.active_facts,
            "historical_facts": self.historical_facts,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "active_count": self.active_count,
            "historical_count": self.historical_count
        }
    
    def to_text(self) -> str:
        """Convert to text format (full output, no truncation; German for LLM consumption)"""
        text_parts = [
            f"## Ergebnisse der Breitensuche (Panoramablick auf die Zukunft)",
            f"Anfrage: {self.query}",
            f"\n### Statistik",
            f"- Knotenzahl gesamt: {self.total_nodes}",
            f"- Kantenzahl gesamt: {self.total_edges}",
            f"- Aktuell gültige Fakten: {self.active_count}",
            f"- Historische/abgelaufene Fakten: {self.historical_count}"
        ]

        # Currently active facts (full output, no truncation)
        if self.active_facts:
            text_parts.append(f"\n### [Aktuell gültige Fakten] (Originaltexte aus dem Simulationsergebnis)")
            for i, fact in enumerate(self.active_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")

        # Historical/expired facts (full output, no truncation)
        if self.historical_facts:
            text_parts.append(f"\n### [Historische/abgelaufene Fakten] (Verlauf der Entwicklung)")
            for i, fact in enumerate(self.historical_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")

        # Key entities (full output, no truncation)
        if self.all_nodes:
            text_parts.append(f"\n### [Beteiligte Entitäten]")
            for node in self.all_nodes:
                entity_type = next((l for l in node.labels if l not in ["Entity", "Node"]), "Entität")
                text_parts.append(f"- **{node.name}** ({entity_type})")

        return "\n".join(text_parts)


@dataclass
class AgentInterview:
    """Single agent interview result"""
    agent_name: str
    agent_role: str  # role type (e.g. student, professor, media)
    agent_bio: str  # bio
    question: str  # interview question
    response: str  # interview answer
    key_quotes: List[str] = field(default_factory=list)  # key quotations
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "agent_bio": self.agent_bio,
            "question": self.question,
            "response": self.response,
            "key_quotes": self.key_quotes
        }
    
    def to_text(self) -> str:
        text = f"**{self.agent_name}** ({self.agent_role})\n"
        # Render the full agent_bio without truncation
        text += f"_Bio: {self.agent_bio}_\n\n"
        text += f"**Q:** {self.question}\n\n"
        text += f"**A:** {self.response}\n"
        if self.key_quotes:
            text += "\n**Kernzitate:**\n"
            for quote in self.key_quotes:
                # Strip various quote characters
                clean_quote = quote.replace('\u201c', '').replace('\u201d', '').replace('"', '')
                clean_quote = clean_quote.replace('\u300c', '').replace('\u300d', '')
                clean_quote = clean_quote.strip()
                # Drop leading punctuation
                while clean_quote and clean_quote[0] in '，,；;：:、。！？\n\r\t ':
                    clean_quote = clean_quote[1:]
                # Filter junk lines containing question markers (questions 1-9)
                skip = False
                for d in '123456789':
                    if f'\u95ee\u9898{d}' in clean_quote:
                        skip = True
                        break
                if skip:
                    continue
                # Truncate overly long content (cut at sentence end rather than mid-word)
                if len(clean_quote) > 150:
                    dot_pos = clean_quote.find('\u3002', 80)
                    if dot_pos > 0:
                        clean_quote = clean_quote[:dot_pos + 1]
                    else:
                        clean_quote = clean_quote[:147] + "..."
                if clean_quote and len(clean_quote) >= 10:
                    text += f'> "{clean_quote}"\n'
        return text


@dataclass
class InterviewResult:
    """
    Interview result.
    Contains the responses from multiple simulated agents.
    """
    interview_topic: str  # interview topic
    interview_questions: List[str]  # list of interview questions

    # Agents selected for the interview
    selected_agents: List[Dict[str, Any]] = field(default_factory=list)
    # Per-agent answers
    interviews: List[AgentInterview] = field(default_factory=list)

    # Reasoning for the agent selection
    selection_reasoning: str = ""
    # Aggregated interview summary
    summary: str = ""

    # Stats
    total_agents: int = 0
    interviewed_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "interview_topic": self.interview_topic,
            "interview_questions": self.interview_questions,
            "selected_agents": self.selected_agents,
            "interviews": [i.to_dict() for i in self.interviews],
            "selection_reasoning": self.selection_reasoning,
            "summary": self.summary,
            "total_agents": self.total_agents,
            "interviewed_count": self.interviewed_count
        }
    
    def to_text(self) -> str:
        """Convert to detailed text (German — fed into LLM prompts and report citations)"""
        text_parts = [
            "## Ausführlicher Interviewbericht",
            f"**Thema:** {self.interview_topic}",
            f"**Befragte:** {self.interviewed_count} / {self.total_agents} Simulations-Agents",
            "\n### Begründung der Auswahl der Befragten",
            self.selection_reasoning or "(automatisch ausgewählt)",
            "\n---",
            "\n### Interview-Protokoll",
        ]

        if self.interviews:
            for i, interview in enumerate(self.interviews, 1):
                text_parts.append(f"\n#### Interview #{i}: {interview.agent_name}")
                text_parts.append(interview.to_text())
                text_parts.append("\n---")
        else:
            text_parts.append("(keine Interview-Aufzeichnung)\n\n---")

        text_parts.append("\n### Interview-Zusammenfassung und Kernaussagen")
        text_parts.append(self.summary or "(keine Zusammenfassung)")

        return "\n".join(text_parts)


class ZepToolsService:
    """
    Zep retrieval tools service.

    Core retrieval tools (post-optimization):
    1. insight_forge - deep insight retrieval (most powerful; auto-generates sub-questions, multi-dimensional search)
    2. panorama_search - breadth search (full picture, including expired content)
    3. quick_search - simple lookup (fast semantic search)
    4. interview_agents - deep interviews (interview simulated agents to gather multi-perspective viewpoints)

    Base tools:
    - search_graph - semantic graph search
    - get_all_nodes - fetch all nodes of the graph
    - get_all_edges - fetch all edges of the graph (with temporal information)
    - get_node_detail - fetch detailed info for a single node
    - get_node_edges - fetch edges related to a node
    - get_entities_by_type - fetch entities filtered by type
    - get_entity_summary - fetch a relationship summary for an entity
    """

    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0
    
    def __init__(self, api_key: Optional[str] = None, llm_client: Optional[LLMClient] = None):
        # api_key kept for backwards-compatible call sites; ignored (memory_service
        # reads connection details from Config directly).
        self._api_key = api_key
        memory_service.warmup()
        # LLM client used by InsightForge to generate sub-questions
        self._llm_client = llm_client
        logger.info(t("console.zepToolsInitialized"))

    @property
    def llm(self) -> LLMClient:
        """Lazily initialize the LLM client"""
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    def _call_with_retry(self, func, operation_name: str, max_retries: int = None):
        """API call with retry logic"""
        max_retries = max_retries or self.MAX_RETRIES
        last_exception = None
        delay = self.RETRY_DELAY
        
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        t("console.zepRetryAttempt", operation=operation_name, attempt=attempt + 1, error=str(e)[:100], delay=f"{delay:.1f}")
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(t("console.zepAllRetriesFailed", operation=operation_name, retries=max_retries, error=str(e)))
        
        raise last_exception
    
    def search_graph(
        self, 
        graph_id: str, 
        query: str, 
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """
        Semantic graph search.

        Uses hybrid search (semantic + BM25) to look up information in the graph.
        Falls back to local keyword matching when the Zep Cloud search API is unavailable.

        Args:
            graph_id: graph ID (Standalone Graph)
            query: search query
            limit: max number of results
            scope: search scope, "edges" or "nodes"

        Returns:
            SearchResult: search result
        """
        logger.info(t("console.graphSearch", graphId=graph_id, query=query[:50]))

        try:
            facts: List[str] = []
            edges_result: List[Dict[str, Any]] = []
            nodes_result: List[Dict[str, Any]] = []

            if scope in ("edges", "both"):
                edge_hits = self._call_with_retry(
                    func=lambda: memory_service.search_edges(graph_id=graph_id, query=query, limit=limit),
                    operation_name=t("console.graphSearchOp", graphId=graph_id),
                )
                for edge in edge_hits:
                    if edge.fact:
                        facts.append(edge.fact)
                    edges_result.append({
                        "uuid": edge.uuid,
                        "name": edge.name,
                        "fact": edge.fact,
                        "source_node_uuid": edge.source_node_uuid,
                        "target_node_uuid": edge.target_node_uuid,
                    })

            if scope in ("nodes", "both"):
                node_hits = self._call_with_retry(
                    func=lambda: memory_service.search_nodes(graph_id=graph_id, query=query, limit=limit),
                    operation_name=t("console.graphSearchOp", graphId=graph_id),
                )
                for node in node_hits:
                    nodes_result.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "labels": node.labels,
                        "summary": node.summary,
                    })
                    if node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")

            logger.info(t("console.searchComplete", count=len(facts)))
            return SearchResult(
                facts=facts,
                edges=edges_result,
                nodes=nodes_result,
                query=query,
                total_count=len(facts),
            )
        except Exception as e:
            logger.warning(t("console.zepSearchApiFallback", error=str(e)))
            return self._local_search(graph_id, query, limit, scope)
    
    def _local_search(
        self, 
        graph_id: str, 
        query: str, 
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """
        Local keyword matching search (fallback for the Zep Search API).

        Fetches all edges/nodes and performs keyword matching locally.

        Args:
            graph_id: graph ID
            query: search query
            limit: max number of results
            scope: search scope

        Returns:
            SearchResult: search result
        """
        logger.info(t("console.usingLocalSearch", query=query[:30]))

        facts = []
        edges_result = []
        nodes_result = []

        # Extract query keywords (simple tokenization)
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]

        def match_score(text: str) -> int:
            """Compute the match score between text and query"""
            if not text:
                return 0
            text_lower = text.lower()
            # Exact query match
            if query_lower in text_lower:
                return 100
            # Keyword match
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 10
            return score

        try:
            if scope in ["edges", "both"]:
                # Fetch all edges and match
                all_edges = self.get_all_edges(graph_id)
                scored_edges = []
                for edge in all_edges:
                    score = match_score(edge.fact) + match_score(edge.name)
                    if score > 0:
                        scored_edges.append((score, edge))

                # Sort by score
                scored_edges.sort(key=lambda x: x[0], reverse=True)
                
                for score, edge in scored_edges[:limit]:
                    if edge.fact:
                        facts.append(edge.fact)
                    edges_result.append({
                        "uuid": edge.uuid,
                        "name": edge.name,
                        "fact": edge.fact,
                        "source_node_uuid": edge.source_node_uuid,
                        "target_node_uuid": edge.target_node_uuid,
                    })
            
            if scope in ["nodes", "both"]:
                # Fetch all nodes and match
                all_nodes = self.get_all_nodes(graph_id)
                scored_nodes = []
                for node in all_nodes:
                    score = match_score(node.name) + match_score(node.summary)
                    if score > 0:
                        scored_nodes.append((score, node))
                
                scored_nodes.sort(key=lambda x: x[0], reverse=True)
                
                for score, node in scored_nodes[:limit]:
                    nodes_result.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "labels": node.labels,
                        "summary": node.summary,
                    })
                    if node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")
            
            logger.info(t("console.localSearchComplete", count=len(facts)))
            
        except Exception as e:
            logger.error(t("console.localSearchFailed", error=str(e)))
        
        return SearchResult(
            facts=facts,
            edges=edges_result,
            nodes=nodes_result,
            query=query,
            total_count=len(facts)
        )
    
    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        """
        Fetch all nodes of the graph (paginated).

        Args:
            graph_id: graph ID

        Returns:
            list of nodes
        """
        logger.info(t("console.fetchingAllNodes", graphId=graph_id))

        nodes = _paginate(memory_service.get_nodes_by_group, graph_id)

        result = []
        for node in nodes:
            result.append(NodeInfo(
                uuid=node.uuid or "",
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {},
            ))

        logger.info(t("console.fetchedNodes", count=len(result)))
        return result

    def get_all_edges(self, graph_id: str, include_temporal: bool = True) -> List[EdgeInfo]:
        """
        Fetch all edges of the graph (paginated, with temporal information).

        Args:
            graph_id: graph ID
            include_temporal: whether to include temporal information (default True)

        Returns:
            list of edges (including created_at, valid_at, invalid_at, expired_at)
        """
        logger.info(t("console.fetchingAllEdges", graphId=graph_id))

        edges = _paginate(memory_service.get_edges_by_group, graph_id)

        result = []
        for edge in edges:
            edge_info = EdgeInfo(
                uuid=edge.uuid or "",
                name=edge.name or "",
                fact=edge.fact or "",
                source_node_uuid=edge.source_node_uuid or "",
                target_node_uuid=edge.target_node_uuid or "",
            )
            if include_temporal:
                edge_info.created_at = edge.created_at
                edge_info.valid_at = edge.valid_at
                edge_info.invalid_at = edge.invalid_at
                edge_info.expired_at = edge.expired_at
            result.append(edge_info)

        logger.info(t("console.fetchedEdges", count=len(result)))
        return result
    
    def get_node_detail(self, node_uuid: str) -> Optional[NodeInfo]:
        """
        Fetch detailed information for a single node.

        Args:
            node_uuid: node UUID

        Returns:
            node info or None
        """
        logger.info(t("console.fetchingNodeDetail", uuid=node_uuid[:8]))
        
        try:
            node = self._call_with_retry(
                func=lambda: memory_service.get_node(node_uuid),
                operation_name=t("console.fetchNodeDetailOp", uuid=node_uuid[:8]),
            )

            if not node:
                return None

            return NodeInfo(
                uuid=node.uuid or "",
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {},
            )
        except Exception as e:
            logger.error(t("console.fetchNodeDetailFailed", error=str(e)))
            return None
    
    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[EdgeInfo]:
        """
        Fetch all edges related to a node.

        Loads all edges of the graph and filters those connected to the given node.

        Args:
            graph_id: graph ID
            node_uuid: node UUID

        Returns:
            list of edges
        """
        logger.info(t("console.fetchingNodeEdges", uuid=node_uuid[:8]))
        
        try:
            # Fetch all graph edges then filter
            all_edges = self.get_all_edges(graph_id)

            result = []
            for edge in all_edges:
                # Check whether the edge connects to the given node (as source or target)
                if edge.source_node_uuid == node_uuid or edge.target_node_uuid == node_uuid:
                    result.append(edge)
            
            logger.info(t("console.foundNodeEdges", count=len(result)))
            return result
            
        except Exception as e:
            logger.warning(t("console.fetchNodeEdgesFailed", error=str(e)))
            return []
    
    def get_entities_by_type(
        self, 
        graph_id: str, 
        entity_type: str
    ) -> List[NodeInfo]:
        """
        Fetch entities by type.

        Args:
            graph_id: graph ID
            entity_type: entity type (e.g. Student, PublicFigure)

        Returns:
            list of entities matching the type
        """
        logger.info(t("console.fetchingEntitiesByType", type=entity_type))
        
        all_nodes = self.get_all_nodes(graph_id)
        
        filtered = []
        for node in all_nodes:
            # Check whether labels contain the given type
            if entity_type in node.labels:
                filtered.append(node)
        
        logger.info(t("console.foundEntitiesByType", count=len(filtered), type=entity_type))
        return filtered
    
    def get_entity_summary(
        self, 
        graph_id: str, 
        entity_name: str
    ) -> Dict[str, Any]:
        """
        Fetch the relationship summary for an entity.

        Searches all information related to the entity and assembles a summary.

        Args:
            graph_id: graph ID
            entity_name: entity name

        Returns:
            entity summary
        """
        logger.info(t("console.fetchingEntitySummary", name=entity_name))

        # First search for information related to the entity
        search_result = self.search_graph(
            graph_id=graph_id,
            query=entity_name,
            limit=20
        )

        # Try to locate the entity in all nodes
        all_nodes = self.get_all_nodes(graph_id)
        entity_node = None
        for node in all_nodes:
            if node.name.lower() == entity_name.lower():
                entity_node = node
                break

        related_edges = []
        if entity_node:
            # Pass the graph_id argument
            related_edges = self.get_node_edges(graph_id, entity_node.uuid)
        
        return {
            "entity_name": entity_name,
            "entity_info": entity_node.to_dict() if entity_node else None,
            "related_facts": search_result.facts,
            "related_edges": [e.to_dict() for e in related_edges],
            "total_relations": len(related_edges)
        }
    
    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """
        Fetch graph statistics.

        Args:
            graph_id: graph ID

        Returns:
            statistics
        """
        logger.info(t("console.fetchingGraphStats", graphId=graph_id))

        nodes = self.get_all_nodes(graph_id)
        edges = self.get_all_edges(graph_id)

        # Count entity type distribution
        entity_types = {}
        for node in nodes:
            for label in node.labels:
                if label not in ["Entity", "Node"]:
                    entity_types[label] = entity_types.get(label, 0) + 1

        # Count relation type distribution
        relation_types = {}
        for edge in edges:
            relation_types[edge.name] = relation_types.get(edge.name, 0) + 1
        
        return {
            "graph_id": graph_id,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_types": entity_types,
            "relation_types": relation_types
        }
    
    def get_simulation_context(
        self, 
        graph_id: str,
        simulation_requirement: str,
        limit: int = 30
    ) -> Dict[str, Any]:
        """
        Fetch simulation-related context information.

        Searches all information related to the simulation requirement.

        Args:
            graph_id: graph ID
            simulation_requirement: simulation requirement description
            limit: per-category result limit

        Returns:
            simulation context information
        """
        logger.info(t("console.fetchingSimContext", requirement=simulation_requirement[:50]))

        # Search information related to the simulation requirement
        search_result = self.search_graph(
            graph_id=graph_id,
            query=simulation_requirement,
            limit=limit
        )

        # Fetch graph statistics
        stats = self.get_graph_statistics(graph_id)

        # Fetch all entity nodes
        all_nodes = self.get_all_nodes(graph_id)

        # Filter entities that have an actual type (skip plain Entity nodes)
        entities = []
        for node in all_nodes:
            custom_labels = [l for l in node.labels if l not in ["Entity", "Node"]]
            if custom_labels:
                entities.append({
                    "name": node.name,
                    "type": custom_labels[0],
                    "summary": node.summary
                })
        
        return {
            "simulation_requirement": simulation_requirement,
            "related_facts": search_result.facts,
            "graph_statistics": stats,
            "entities": entities[:limit],  # cap the count
            "total_entities": len(entities)
        }
    
    # ========== Core retrieval tools (post-optimization) ==========
    
    def insight_forge(
        self,
        graph_id: str,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_sub_queries: int = 5
    ) -> InsightForgeResult:
        """
        InsightForge - deep insight retrieval.

        Most powerful hybrid retrieval. Automatically decomposes the question and runs multi-dimensional search:
        1. Uses an LLM to break the question into sub-questions
        2. Runs semantic search for each sub-question
        3. Extracts related entities and fetches their details
        4. Traces relationship chains
        5. Aggregates everything into a deep insight result

        Args:
            graph_id: graph ID
            query: user question
            simulation_requirement: simulation requirement description
            report_context: report context (optional, used for more focused sub-question generation)
            max_sub_queries: maximum number of sub-questions

        Returns:
            InsightForgeResult: deep insight retrieval result
        """
        logger.info(t("console.insightForgeStart", query=query[:50]))
        
        result = InsightForgeResult(
            query=query,
            simulation_requirement=simulation_requirement,
            sub_queries=[]
        )
        
        # Step 1: Use the LLM to generate sub-questions
        sub_queries = self._generate_sub_queries(
            query=query,
            simulation_requirement=simulation_requirement,
            report_context=report_context,
            max_queries=max_sub_queries
        )
        result.sub_queries = sub_queries
        logger.info(t("console.generatedSubQueries", count=len(sub_queries)))
        
        # Step 2: Run semantic search for each sub-question
        all_facts = []
        all_edges = []
        seen_facts = set()
        
        for sub_query in sub_queries:
            search_result = self.search_graph(
                graph_id=graph_id,
                query=sub_query,
                limit=15,
                scope="edges"
            )
            
            for fact in search_result.facts:
                if fact not in seen_facts:
                    all_facts.append(fact)
                    seen_facts.add(fact)
            
            all_edges.extend(search_result.edges)
        
        # Also search using the original question
        main_search = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=20,
            scope="edges"
        )
        for fact in main_search.facts:
            if fact not in seen_facts:
                all_facts.append(fact)
                seen_facts.add(fact)
        
        result.semantic_facts = all_facts
        result.total_facts = len(all_facts)
        
        # Step 3: Extract related entity UUIDs from edges; only fetch info for those entities (not all nodes)
        entity_uuids = set()
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                if source_uuid:
                    entity_uuids.add(source_uuid)
                if target_uuid:
                    entity_uuids.add(target_uuid)
        
        # Fetch details for all related entities (no limit, full output)
        entity_insights = []
        node_map = {}  # used later for building relationship chains

        for uuid in list(entity_uuids):  # process all entities, no truncation
            if not uuid:
                continue
            try:
                # Fetch info for each related node individually
                node = self.get_node_detail(uuid)
                if node:
                    node_map[uuid] = node
                    entity_type = next((l for l in node.labels if l not in ["Entity", "Node"]), "Entity")

                    # Fetch all facts related to this entity (no truncation)
                    related_facts = [
                        f for f in all_facts
                        if node.name.lower() in f.lower()
                    ]

                    entity_insights.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "type": entity_type,
                        "summary": node.summary,
                        "related_facts": related_facts  # full output, no truncation
                    })
            except Exception as e:
                logger.debug(f"Failed to fetch node {uuid}: {e}")
                continue
        
        result.entity_insights = entity_insights
        result.total_entities = len(entity_insights)
        
        # Step 4: Build all relationship chains (no limit)
        relationship_chains = []
        for edge_data in all_edges:  # process all edges, no truncation
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                relation_name = edge_data.get('name', '')
                
                source_name = node_map.get(source_uuid, NodeInfo('', '', [], '', {})).name or source_uuid[:8]
                target_name = node_map.get(target_uuid, NodeInfo('', '', [], '', {})).name or target_uuid[:8]
                
                chain = f"{source_name} --[{relation_name}]--> {target_name}"
                if chain not in relationship_chains:
                    relationship_chains.append(chain)
        
        result.relationship_chains = relationship_chains
        result.total_relationships = len(relationship_chains)
        
        logger.info(t("console.insightForgeComplete", facts=result.total_facts, entities=result.total_entities, relationships=result.total_relationships))
        return result
    
    def _generate_sub_queries(
        self,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_queries: int = 5
    ) -> List[str]:
        """
        Generate sub-questions using the LLM.

        Decomposes a complex question into multiple sub-questions that can be retrieved independently.
        """
        system_prompt = """Sie sind ein erfahrener Experte für Fragenanalyse. Ihre Aufgabe besteht darin, eine komplexe Frage in mehrere Teilfragen zu zerlegen, die sich unabhängig voneinander in einer Simulationswelt beobachten lassen.

Anforderungen:
1. Jede Teilfrage muss konkret genug sein, sodass sich in der Simulationswelt zugehörige Agent-Handlungen oder Ereignisse finden lassen.
2. Die Teilfragen sollen unterschiedliche Dimensionen der Ursprungsfrage abdecken (z. B. Wer, Was, Warum, Wie, Wann, Wo).
3. Die Teilfragen müssen zum Simulationsszenario passen.
4. Antworten Sie im JSON-Format: {"sub_queries": ["Teilfrage 1", "Teilfrage 2", ...]}"""

        user_prompt = f"""Hintergrund der Simulationsanforderung:
{simulation_requirement}

{f"Berichtskontext: {report_context[:500]}" if report_context else ""}

Bitte zerlegen Sie die folgende Frage in {max_queries} Teilfragen:
{query}

Geben Sie eine Liste der Teilfragen im JSON-Format zurück."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )

            sub_queries = response.get("sub_queries", [])
            # Ensure list of strings
            return [str(sq) for sq in sub_queries[:max_queries]]

        except Exception as e:
            logger.warning(t("console.generateSubQueriesFailed", error=str(e)))
            # Fallback: variations based on the original question
            return [
                query,
                f"Hauptbeteiligte von: {query}",
                f"Ursachen und Auswirkungen von: {query}",
                f"Entwicklungsverlauf von: {query}"
            ][:max_queries]
    
    def panorama_search(
        self,
        graph_id: str,
        query: str,
        include_expired: bool = True,
        limit: int = 50
    ) -> PanoramaResult:
        """
        PanoramaSearch - breadth search.

        Returns the full picture including all related content and historical/expired info:
        1. Fetches all related nodes
        2. Fetches all edges (including expired/invalidated ones)
        3. Classifies the result into currently active and historical information

        Use this tool when you need a full overview of an event or to trace its evolution.

        Args:
            graph_id: graph ID
            query: search query (used for relevance ranking)
            include_expired: whether to include expired content (default True)
            limit: result limit

        Returns:
            PanoramaResult: breadth search result
        """
        logger.info(t("console.panoramaSearchStart", query=query[:50]))
        
        result = PanoramaResult(query=query)
        
        # Fetch all nodes
        all_nodes = self.get_all_nodes(graph_id)
        node_map = {n.uuid: n for n in all_nodes}
        result.all_nodes = all_nodes
        result.total_nodes = len(all_nodes)

        # Fetch all edges (with temporal info)
        all_edges = self.get_all_edges(graph_id, include_temporal=True)
        result.all_edges = all_edges
        result.total_edges = len(all_edges)

        # Classify facts
        active_facts = []
        historical_facts = []

        for edge in all_edges:
            if not edge.fact:
                continue

            # Annotate facts with entity names
            source_name = node_map.get(edge.source_node_uuid, NodeInfo('', '', [], '', {})).name or edge.source_node_uuid[:8]
            target_name = node_map.get(edge.target_node_uuid, NodeInfo('', '', [], '', {})).name or edge.target_node_uuid[:8]

            # Determine whether the fact is expired/invalidated
            is_historical = edge.is_expired or edge.is_invalid

            if is_historical:
                # Historical/expired fact, add a time marker
                valid_at = edge.valid_at or "unknown"
                invalid_at = edge.invalid_at or edge.expired_at or "unknown"
                fact_with_time = f"[{valid_at} - {invalid_at}] {edge.fact}"
                historical_facts.append(fact_with_time)
            else:
                # Currently active fact
                active_facts.append(edge.fact)

        # Relevance ranking based on the query
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]
        
        def relevance_score(fact: str) -> int:
            fact_lower = fact.lower()
            score = 0
            if query_lower in fact_lower:
                score += 100
            for kw in keywords:
                if kw in fact_lower:
                    score += 10
            return score
        
        # Sort and apply the limit
        active_facts.sort(key=relevance_score, reverse=True)
        historical_facts.sort(key=relevance_score, reverse=True)
        
        result.active_facts = active_facts[:limit]
        result.historical_facts = historical_facts[:limit] if include_expired else []
        result.active_count = len(active_facts)
        result.historical_count = len(historical_facts)
        
        logger.info(t("console.panoramaSearchComplete", active=result.active_count, historical=result.historical_count))
        return result
    
    def quick_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10
    ) -> SearchResult:
        """
        QuickSearch - simple search.

        Fast, lightweight retrieval:
        1. Calls Zep semantic search directly
        2. Returns the most relevant results
        3. Suited for simple, direct lookups

        Args:
            graph_id: graph ID
            query: search query
            limit: result limit

        Returns:
            SearchResult: search result
        """
        logger.info(t("console.quickSearchStart", query=query[:50]))

        # Delegate to the existing search_graph method
        result = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=limit,
            scope="edges"
        )
        
        logger.info(t("console.quickSearchComplete", count=result.total_count))
        return result
    
    def interview_agents(
        self,
        simulation_id: str,
        interview_requirement: str,
        simulation_requirement: str = "",
        max_agents: int = 5,
        custom_questions: List[str] = None
    ) -> InterviewResult:
        """
        InterviewAgents - deep interview.

        Calls the real OASIS interview API to interview agents running in the simulation:
        1. Loads the persona files to learn about all simulated agents
        2. Uses an LLM to analyze the interview requirement and pick the most relevant agents
        3. Uses an LLM to generate interview questions
        4. Calls /api/simulation/interview/batch for the real interview (across both platforms simultaneously)
        5. Aggregates all interview results into a single report

        IMPORTANT: This requires the simulation environment to be running (OASIS environment not shut down).

        Use cases:
        - Need viewpoints on an event from different roles
        - Need to gather multiple opinions and perspectives
        - Need real responses from simulated agents (rather than LLM stand-ins)

        Args:
            simulation_id: simulation ID (used to locate persona files and call the interview API)
            interview_requirement: interview requirement description (unstructured, e.g. "understand students' view on the event")
            simulation_requirement: simulation requirement background (optional)
            max_agents: maximum number of agents to interview
            custom_questions: custom interview questions (optional; auto-generated if not provided)

        Returns:
            InterviewResult: interview result
        """
        from .simulation_runner import SimulationRunner
        
        logger.info(t("console.interviewAgentsStart", requirement=interview_requirement[:50]))
        
        result = InterviewResult(
            interview_topic=interview_requirement,
            interview_questions=custom_questions or []
        )
        
        # Step 1: Load persona files
        profiles = self._load_agent_profiles(simulation_id)

        if not profiles:
            logger.warning(t("console.profilesNotFound", simId=simulation_id))
            result.summary = "Keine interviewbaren Agent-Persona-Dateien gefunden"
            return result

        result.total_agents = len(profiles)
        logger.info(t("console.loadedProfiles", count=len(profiles)))

        # Step 2: Use the LLM to choose which agents to interview (returns agent_id list)
        selected_agents, selected_indices, selection_reasoning = self._select_agents_for_interview(
            profiles=profiles,
            interview_requirement=interview_requirement,
            simulation_requirement=simulation_requirement,
            max_agents=max_agents
        )
        
        result.selected_agents = selected_agents
        result.selection_reasoning = selection_reasoning
        logger.info(t("console.selectedAgentsForInterview", count=len(selected_agents), indices=selected_indices))
        
        # Step 3: Generate interview questions (if none were provided)
        if not result.interview_questions:
            result.interview_questions = self._generate_interview_questions(
                interview_requirement=interview_requirement,
                simulation_requirement=simulation_requirement,
                selected_agents=selected_agents
            )
            logger.info(t("console.generatedInterviewQuestions", count=len(result.interview_questions)))
        
        # Combine questions into a single interview prompt
        combined_prompt = "\n".join([f"{i+1}. {q}" for i, q in enumerate(result.interview_questions)])

        # Add an optimization prefix to constrain the agent's reply format
        INTERVIEW_PROMPT_PREFIX = (
            "Sie nehmen gerade an einem Interview teil. Beantworten Sie die folgenden Fragen "
            "unter Berücksichtigung Ihrer Persona sowie aller bisherigen Erinnerungen und Handlungen "
            "ausschließlich in Klartext.\n"
            "Antwortvorgaben:\n"
            "1. Antworten Sie unmittelbar in natürlicher Sprache und rufen Sie keinerlei Tools auf.\n"
            "2. Geben Sie weder JSON noch Tool-Call-Formate zurück.\n"
            "3. Verwenden Sie keine Markdown-Überschriften (z. B. #, ##, ###).\n"
            "4. Beantworten Sie die Fragen einzeln nach Reihenfolge und beginnen Sie jede Antwort mit \"Frage X:\" (X = Fragenummer).\n"
            "5. Trennen Sie die Antworten zu den einzelnen Fragen durch eine Leerzeile.\n"
            "6. Antworten Sie inhaltlich substanziell, mindestens 2-3 Sätze pro Frage.\n\n"
        )
        optimized_prompt = f"{INTERVIEW_PROMPT_PREFIX}{combined_prompt}"

        # Step 4: Call the real interview API (no platform specified -> both platforms simultaneously)
        try:
            # Build the batch interview list (no platform -> both platforms)
            interviews_request = []
            for agent_idx in selected_indices:
                interviews_request.append({
                    "agent_id": agent_idx,
                    "prompt": optimized_prompt  # use the optimized prompt
                    # platform left unset so the API interviews on both twitter and reddit
                })
            
            logger.info(t("console.callingBatchInterviewApi", count=len(interviews_request)))
            
            # Call SimulationRunner.interview_agents_batch (no platform -> both platforms)
            api_result = SimulationRunner.interview_agents_batch(
                simulation_id=simulation_id,
                interviews=interviews_request,
                platform=None,  # no platform specified -> dual-platform interview
                timeout=180.0   # dual-platform needs a longer timeout
            )

            logger.info(t("console.interviewApiReturned", count=api_result.get('interviews_count', 0), success=api_result.get('success')))

            # Check whether the API call succeeded
            if not api_result.get("success", False):
                error_msg = api_result.get("error", "unknown error")
                logger.warning(t("console.interviewApiReturnedFailure", error=error_msg))
                result.summary = f"Interview-API-Aufruf fehlgeschlagen: {error_msg}. Bitte prüfen Sie den Status der OASIS-Simulationsumgebung."
                return result

            # Step 5: Parse API result and build AgentInterview objects
            # Dual-platform response shape: {"twitter_0": {...}, "reddit_0": {...}, "twitter_1": {...}, ...}
            api_data = api_result.get("result", {})
            results_dict = api_data.get("results", {}) if isinstance(api_data, dict) else {}

            for i, agent_idx in enumerate(selected_indices):
                agent = selected_agents[i]
                agent_name = agent.get("realname", agent.get("username", f"Agent_{agent_idx}"))
                agent_role = agent.get("profession", "unbekannt")
                agent_bio = agent.get("bio", "")

                # Fetch this agent's interview results from both platforms
                twitter_result = results_dict.get(f"twitter_{agent_idx}", {})
                reddit_result = results_dict.get(f"reddit_{agent_idx}", {})

                twitter_response = twitter_result.get("response", "")
                reddit_response = reddit_result.get("response", "")

                # Strip potential tool-call JSON wrappers
                twitter_response = self._clean_tool_call_response(twitter_response)
                reddit_response = self._clean_tool_call_response(reddit_response)

                # Always emit a dual-platform-labelled output (German labels for the LLM-facing report)
                twitter_text = twitter_response if twitter_response else "(keine Antwort von dieser Plattform)"
                reddit_text = reddit_response if reddit_response else "(keine Antwort von dieser Plattform)"
                response_text = f"[Antwort Twitter-Plattform]\n{twitter_text}\n\n[Antwort Reddit-Plattform]\n{reddit_text}"

                # Extract key quotations (from responses on both platforms)
                import re
                combined_responses = f"{twitter_response} {reddit_response}"

                # Clean response text: strip markers, numbering, markdown noise
                clean_text = re.sub(r'#{1,6}\s+', '', combined_responses)
                clean_text = re.sub(r'\{[^}]*tool_name[^}]*\}', '', clean_text)
                clean_text = re.sub(r'[*_`|>~\-]{2,}', '', clean_text)
                clean_text = re.sub(r'Frage\s*\d+[：:]\s*', '', clean_text)
                clean_text = re.sub(r'【[^】]+】', '', clean_text)

                # Strategy 1 (primary): extract complete sentences with substantive content
                sentences = re.split(r'[。！？]', clean_text)
                meaningful = [
                    s.strip() for s in sentences
                    if 20 <= len(s.strip()) <= 150
                    and not re.match(r'^[\s\W，,；;：:、]+', s.strip())
                    and not s.strip().startswith(('{', 'Frage'))
                ]
                meaningful.sort(key=len, reverse=True)
                key_quotes = [s + "。" for s in meaningful[:3]]

                # Strategy 2 (supplementary): long text inside properly paired Chinese quotes 「」
                if not key_quotes:
                    paired = re.findall(r'\u201c([^\u201c\u201d]{15,100})\u201d', clean_text)
                    paired += re.findall(r'\u300c([^\u300c\u300d]{15,100})\u300d', clean_text)
                    key_quotes = [q for q in paired if not re.match(r'^[，,；;：:、]', q)][:3]
                
                interview = AgentInterview(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    agent_bio=agent_bio[:1000],  # widen the bio length limit
                    question=combined_prompt,
                    response=response_text,
                    key_quotes=key_quotes[:5]
                )
                result.interviews.append(interview)
            
            result.interviewed_count = len(result.interviews)
            
        except ValueError as e:
            # Simulation environment is not running
            logger.warning(t("console.interviewApiCallFailed", error=e))
            result.summary = f"Interview fehlgeschlagen: {str(e)}. Die Simulationsumgebung wurde möglicherweise beendet. Bitte stellen Sie sicher, dass die OASIS-Umgebung läuft."
            return result
        except Exception as e:
            logger.error(t("console.interviewApiCallException", error=e))
            import traceback
            logger.error(traceback.format_exc())
            result.summary = f"Im Interview-Prozess ist ein Fehler aufgetreten: {str(e)}"
            return result

        # Step 6: Generate the interview summary
        if result.interviews:
            result.summary = self._generate_interview_summary(
                interviews=result.interviews,
                interview_requirement=interview_requirement
            )
        
        logger.info(t("console.interviewAgentsComplete", count=result.interviewed_count))
        return result
    
    @staticmethod
    def _clean_tool_call_response(response: str) -> str:
        """Strip JSON tool-call wrappers from an agent's reply and return the actual content"""
        if not response or not response.strip().startswith('{'):
            return response
        text = response.strip()
        if 'tool_name' not in text[:80]:
            return response
        import re as _re
        try:
            data = json.loads(text)
            if isinstance(data, dict) and 'arguments' in data:
                for key in ('content', 'text', 'body', 'message', 'reply'):
                    if key in data['arguments']:
                        return str(data['arguments'][key])
        except (json.JSONDecodeError, KeyError, TypeError):
            match = _re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if match:
                return match.group(1).replace('\\n', '\n').replace('\\"', '"')
        return response

    def _load_agent_profiles(self, simulation_id: str) -> List[Dict[str, Any]]:
        """Load the simulation's agent persona files"""
        import os
        import csv

        # Build the persona file path
        sim_dir = os.path.join(
            os.path.dirname(__file__),
            f'../../uploads/simulations/{simulation_id}'
        )

        profiles = []

        # Prefer the Reddit JSON format
        reddit_profile_path = os.path.join(sim_dir, "reddit_profiles.json")
        if os.path.exists(reddit_profile_path):
            try:
                with open(reddit_profile_path, 'r', encoding='utf-8') as f:
                    profiles = json.load(f)
                logger.info(t("console.loadedRedditProfiles", count=len(profiles)))
                return profiles
            except Exception as e:
                logger.warning(t("console.readRedditProfilesFailed", error=e))

        # Fall back to the Twitter CSV format
        twitter_profile_path = os.path.join(sim_dir, "twitter_profiles.csv")
        if os.path.exists(twitter_profile_path):
            try:
                with open(twitter_profile_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Normalize the CSV row to the unified format
                        profiles.append({
                            "realname": row.get("name", ""),
                            "username": row.get("username", ""),
                            "bio": row.get("description", ""),
                            "persona": row.get("user_char", ""),
                            "profession": "unknown"
                        })
                logger.info(t("console.loadedTwitterProfiles", count=len(profiles)))
                return profiles
            except Exception as e:
                logger.warning(t("console.readTwitterProfilesFailed", error=e))
        
        return profiles
    
    def _select_agents_for_interview(
        self,
        profiles: List[Dict[str, Any]],
        interview_requirement: str,
        simulation_requirement: str,
        max_agents: int
    ) -> tuple:
        """
        Use the LLM to choose which agents to interview.

        Returns:
            tuple: (selected_agents, selected_indices, reasoning)
                - selected_agents: full info of the chosen agents
                - selected_indices: indices of the chosen agents (used for the API call)
                - reasoning: selection rationale
        """

        # Build the agent summary list
        agent_summaries = []
        for i, profile in enumerate(profiles):
            summary = {
                "index": i,
                "name": profile.get("realname", profile.get("username", f"Agent_{i}")),
                "profession": profile.get("profession", "unknown"),
                "bio": profile.get("bio", "")[:200],
                "interested_topics": profile.get("interested_topics", [])
            }
            agent_summaries.append(summary)

        system_prompt = """Sie sind ein erfahrener Interview-Planer. Ihre Aufgabe besteht darin, anhand der Interview-Anforderung aus der Liste der Simulations-Agents die am besten geeigneten Befragten auszuwählen.

Auswahlkriterien:
1. Identität/Beruf des Agents passt zum Interview-Thema.
2. Der Agent verfügt vermutlich über besondere oder wertvolle Sichtweisen.
3. Wählen Sie eine vielfältige Perspektivenbreite (z. B. Befürworter, Gegner, Neutrale, Fachpersonen).
4. Bevorzugen Sie Rollen mit direktem Bezug zum Ereignis.

Antworten Sie im JSON-Format:
{
    "selected_indices": [Liste der Indizes ausgewählter Agents],
    "reasoning": "Begründung der Auswahl"
}"""

        user_prompt = f"""Interview-Anforderung:
{interview_requirement}

Simulationshintergrund:
{simulation_requirement if simulation_requirement else "nicht angegeben"}

Auswählbare Agent-Liste (insgesamt {len(agent_summaries)} Agents):
{json.dumps(agent_summaries, ensure_ascii=False, indent=2)}

Bitte wählen Sie höchstens {max_agents} am besten geeignete Agents aus und erläutern Sie Ihre Auswahl."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )

            selected_indices = response.get("selected_indices", [])[:max_agents]
            reasoning = response.get("reasoning", "Automatische Auswahl basierend auf Relevanz")

            # Build the full info for the chosen agents
            selected_agents = []
            valid_indices = []
            for idx in selected_indices:
                if 0 <= idx < len(profiles):
                    selected_agents.append(profiles[idx])
                    valid_indices.append(idx)

            return selected_agents, valid_indices, reasoning

        except Exception as e:
            logger.warning(t("console.llmSelectAgentFailed", error=e))
            # Fallback: select the first N
            selected = profiles[:max_agents]
            indices = list(range(min(max_agents, len(profiles))))
            return selected, indices, "Standardauswahl-Strategie verwendet"
    
    def _generate_interview_questions(
        self,
        interview_requirement: str,
        simulation_requirement: str,
        selected_agents: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate interview questions using the LLM"""

        agent_roles = [a.get("profession", "unknown") for a in selected_agents]

        system_prompt = """Sie sind ein erfahrener Reporter/Interviewer. Erstellen Sie auf Basis der Interview-Anforderung 3-5 vertiefende Interviewfragen.

Anforderungen an die Fragen:
1. Offene Fragen, die ausführliche Antworten anregen.
2. So formuliert, dass unterschiedliche Rollen unterschiedlich antworten können.
3. Decken mehrere Dimensionen ab (Fakten, Meinungen, Empfindungen).
4. Natürliche Sprache, wie in einem echten Interview.
5. Maximal 50 Zeichen pro Frage, prägnant.
6. Direkt formuliert, ohne Hintergrunderklärungen oder Präfixe.

Antworten Sie im JSON-Format: {"questions": ["Frage 1", "Frage 2", ...]}"""

        user_prompt = f"""Interview-Anforderung: {interview_requirement}

Simulationshintergrund: {simulation_requirement if simulation_requirement else "nicht angegeben"}

Rollen der Befragten: {', '.join(agent_roles)}

Bitte erstellen Sie 3-5 Interviewfragen."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5
            )

            return response.get("questions", [f"Wie sehen Sie {interview_requirement}?"])

        except Exception as e:
            logger.warning(t("console.generateInterviewQuestionsFailed", error=e))
            return [
                f"Welche Position vertreten Sie zu {interview_requirement}?",
                "Welche Auswirkungen hat dies auf Sie oder die Gruppe, die Sie vertreten?",
                "Wie sollte das Problem Ihrer Meinung nach gelöst oder verbessert werden?"
            ]
    
    def _generate_interview_summary(
        self,
        interviews: List[AgentInterview],
        interview_requirement: str
    ) -> str:
        """Generate the interview summary"""

        if not interviews:
            return "Es wurden keine Interviews abgeschlossen"

        # Collect all interview content
        interview_texts = []
        for interview in interviews:
            interview_texts.append(f"[{interview.agent_name} ({interview.agent_role})]\n{interview.response[:500]}")

        quote_instruction = 'Use quotation marks "" when quoting interviewees' if get_locale() != 'de' else 'Verwenden Sie deutsche Anführungszeichen „..." beim Zitieren der Befragten'
        system_prompt = f"""Sie sind ein erfahrener Nachrichtenredakteur. Erstellen Sie auf Basis der Antworten mehrerer Befragter eine Interview-Zusammenfassung.

Anforderungen an die Zusammenfassung:
1. Fassen Sie die Hauptpositionen aller Seiten zusammen.
2. Benennen Sie Übereinstimmungen und Differenzen.
3. Heben Sie aussagekräftige Zitate hervor.
4. Bleiben Sie objektiv und neutral, ohne Partei zu ergreifen.
5. Maximal 1000 Wörter.

Formatvorgaben (verpflichtend):
- Verwenden Sie Klartextabschnitte, getrennt durch Leerzeilen.
- Verwenden Sie keine Markdown-Überschriften (z. B. #, ##, ###).
- Verwenden Sie keine Trennlinien (z. B. ---, ***).
- {quote_instruction}
- **Fettschrift** zur Hervorhebung von Schlüsselbegriffen ist erlaubt; sonstige Markdown-Syntax ist nicht erlaubt."""

        user_prompt = f"""Interview-Thema: {interview_requirement}

Interview-Inhalte:
{"".join(interview_texts)}

Bitte erstellen Sie die Interview-Zusammenfassung."""

        try:
            summary = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            return summary

        except Exception as e:
            logger.warning(t("console.generateInterviewSummaryFailed", error=e))
            # Fallback: simple concatenation
            return f"Es wurden {len(interviews)} Personen interviewt, darunter: " + ", ".join([i.agent_name for i in interviews])
