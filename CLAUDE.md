# MiroFish — Self-Hosted Memory Fork

Fork von **MiroFish** mit einem Ziel: **Zep Cloud** durch einen self-hosted Knowledge-Graph aus **Graphiti + Neo4j** ersetzen.

Branch: `graphiti-migration`. Hauptarbeit liegt im neuen `backend/app/services/memory_service.py` und im erweiterten `docker-compose.yml`.

## Was hier anders ist als Upstream MiroFish

- `zep-cloud` ist komplett entfernt — kein Zep-Account, kein Cloud-Call.
- Memory-Layer ist `Graphiti 0.29` gegen lokales `Neo4j 5.26 Community`, beides als Compose-Service.
- Es gibt **genau einen** Memory-Touchpoint: `app/services/memory_service.py`. Alle Services (`graph_builder`, `zep_tools`, `zep_entity_reader`, `zep_graph_memory_updater`, `oasis_profile_generator`) gehen ausschließlich über diese Schnittstelle.
- Der vorherige Zep-Compat-Wrapper (`app/adapters/graphiti_compat.py`) ist gelöscht. Niemals wieder einbauen.
- Episode-Polling ist weg. Graphitis `add_episode` ist synchron — wenn es zurückkommt, sind Nodes/Edges in Neo4j.

## Quickstart

```bash
cp .env.example .env
# .env editieren: LLM_API_KEY, NEO4J_PASSWORD, NEO4J_AUTH
docker compose up -d
docker compose logs -f mirofish
```

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:5001`
- Neo4j Browser: `http://localhost:7475` (Login mit `NEO4J_AUTH`-Werten aus `.env`)

Ports 7475/7688 statt 7474/7687 — damit andere lokale Neo4j-Container (z.B. `dwh-dbms-neo4j`) nicht kollidieren.

## ENV-Variablen

| Variable | Pflicht | Zweck |
|---|---|---|
| `LLM_API_KEY` | ja | OpenAI-kompatibler Key für Chat **und** Embeddings (Graphiti macht beides) |
| `LLM_BASE_URL` | nein | Default `https://api.openai.com/v1` |
| `LLM_MODEL_NAME` | nein | Default `gpt-4o-mini`. **Nicht `gpt-5.x` nutzen** — Graphiti sendet `reasoning.effort='minimal'`, das lehnen Reasoning-Modelle ab |
| `OPENAI_API_KEY` | ja | Graphitis Default-Cross-Encoder liest ihn **direkt** aus dem Environment. Bei OpenAI ist das derselbe Wert wie `LLM_API_KEY` |
| `NEO4J_AUTH` | ja | Compose-Variante: `user/password` in einem String |
| `NEO4J_URI` | nein | Compose-intern `bolt://neo4j:7687`, lokal Dev `bolt://localhost:7688` |
| `NEO4J_USER` | nein | Default `neo4j` |
| `NEO4J_PASSWORD` | ja | wie in `NEO4J_AUTH`, separat als Python-Setting |

`.env` ist gitignored. Niemals einchecken.

## Memory-Service-API

`app/services/memory_service.py` — einziger Entry Point:

```python
memory_service.register_ontology(group_id, entities, edges)
memory_service.add_episode(group_id, content, source_type="text")
memory_service.add_episodes_bulk(group_id, episodes)
memory_service.search_edges(group_id, query, limit=10)
memory_service.search_nodes(group_id, query, limit=10)
memory_service.get_node(uuid)
memory_service.get_node_edges(node_uuid)
memory_service.get_nodes_by_group(group_id, limit, cursor)
memory_service.get_edges_by_group(group_id, limit, cursor)
memory_service.delete_group(group_id)
```

`group_id` ist MiroFishs `graph_id`. Format: `mirofish_<16hex>`.

Sync-Bridge zur Async-Welt von Graphiti läuft über einen dedizierten Background-Loop-Thread (in `memory_service.py` selbst). Niemand sollte direkt `asyncio.run` aufrufen — immer `run_async(coro)` aus dem Service.

## Konventionen

- **Neue Memory-Aufrufe nur über `memory_service`.** Wenn etwas fehlt, dort hinzufügen — nicht `graphiti_core` direkt aus einem Service-File importieren.
- **Synchronisationspfad respektieren.** Sync Flask-Handler → `memory_service.<func>()` → Background-Loop → Graphiti. Kein eigenes Event-Loop-Spawning.
- **Ontology pro `group_id` registrieren bevor `add_episode` läuft.** `register_ontology` cached die Pydantic-Klassen; ohne den Call laufen die Episodes ohne Schema und liefern weniger gute Extraktion.
- Reasoning-Modelle (gpt-5.x, o-Serie) **vermeiden** für Graphiti-Calls. Bug-Quelle ist `reasoning.effort='minimal'`.
- `camel-oasis 0.2.5` pinnt `neo4j==5.23.0`. Wir overriden in `pyproject.toml` unter `[tool.uv].override-dependencies` auf `neo4j>=5.23` — der Driver ist mit Server 5.26 Bolt-protocol-kompatibel.

## Stolperfallen

- **`docker compose restart` lädt `.env` nicht neu.** Bei ENV-Änderungen `docker compose up -d --force-recreate mirofish`.
- **Code-Änderungen im Container live**: `docker-compose.yml` mounted `./backend/app:/app/backend/app` als Dev-Bind. Flask reloadet automatisch. Vor Production-Deploy diesen Mount rausnehmen.
- **`uv.lock` ist frozen im Dockerfile.** Nach Änderungen an `pyproject.toml` lokal `cd backend && uv lock` ausführen und `uv.lock` mit committen.
- **Neo4j-Volumen aufräumen**: `docker compose down -v` löscht alle Graphen. Bei Schema-Migrationen oder kaputten Indexen ist das oft der schnellste Reset.

## Smoke-Tests

Nach Code-Änderungen am Memory-Pfad immer beides laufen lassen:

```bash
# 1. Graph-Build
curl -X POST http://localhost:5001/api/graph/ontology/generate \
  -F "files=@seed.txt" \
  -F "simulation_requirement=..." \
  -F "project_name=smoke"
# -> project_id zurück
curl -X POST http://localhost:5001/api/graph/build \
  -H "Content-Type: application/json" \
  -d '{"project_id":"...","graph_name":"smoke"}'
# Task pollen über /api/graph/task/<task_id>

# 2. Search via ReportAgent
curl -X POST http://localhost:5001/api/report/chat \
  -H "Content-Type: application/json" \
  -d '{"simulation_id":"...","report_id":"...","message":"Suche im Graphen ...","chat_history":[]}'
```

Direkter Neo4j-Check:

```bash
docker exec mirofish-neo4j cypher-shell -u neo4j -p $NEO4J_PASSWORD \
  "MATCH (n) WHERE n.group_id STARTS WITH 'mirofish_' RETURN labels(n)[0] AS label, count(*) AS count ORDER BY count DESC"
```

## Commits

Globale Regel aus `~/.claude/CLAUDE.md`: **keine `Co-Authored-By: Claude …`-Zeile** in Commit-Messages oder MR-Bodies. Der Autor ist `Christian Langer <cl@koempf24.de>`.

## Verbindung zu KIM

`memory_service` ist bewusst portabel geschnitten — `group_id`-basierte Isolation, keine MiroFish-spezifischen Abhängigkeiten in den öffentlichen Funktionen. Wenn das Modul später in KIM landet, sollte es ohne Anpassungen wiederverwendbar sein. Sub-Module wie `simulation_runner` oder `oasis_profile_generator` bleiben MiroFish-spezifisch und werden nicht mit übertragen.
