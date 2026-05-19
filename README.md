# MiroFish-Graphiti

Self-hosted Fork von **[MiroFish](https://github.com/666ghj/MiroFish)** — der Multi-Agent-Simulations-Engine von 666ghj/Shanda Group. Diese Variante ersetzt die Cloud-Memory-Schicht durch einen lokal laufenden Knowledge-Graph aus **[Graphiti](https://github.com/getzep/graphiti)** und **[Neo4j](https://neo4j.com/)**, sodass das gesamte System ohne externen Memory-Provider auskommt.

> Upstream: **[github.com/666ghj/MiroFish](https://github.com/666ghj/MiroFish)** — alle Credits für Konzept, Frontend und das ursprüngliche Backend-Design gehen an das MiroFish-Team. Dieses Repo ist ein abgeleiteter Fork; die fachlichen Fähigkeiten (Graph Building → Simulation → Report → Interaction) stammen aus dem Original.

## Worum geht's

MiroFish nimmt Seed-Material (News, Texte, Datenanalysen) und baut daraus eine simulierte Welt aus tausenden Agents mit eigener Persona, eigenem Gedächtnis und eigenen Beziehungen. Aus diesen Interaktionen entsteht ein Prediction-Report, der erklärt, wie sich ein Szenario plausibel entwickeln könnte. Nach der Simulation kann man mit einzelnen Agents oder dem ReportAgent weiterreden, um Ergebnisse zu vertiefen.

Der Workflow bleibt identisch zum Upstream:

1. **Graph Building** — Seeds extrahieren, GraphRAG aufbauen, Personas erzeugen
2. **Environment Setup** — Entity-Relationen, Agent-Konfiguration
3. **Simulation** — Parallele Multi-Agent-Läufe, dynamische Memory-Updates
4. **Report Generation** — ReportAgent fasst die Simulation zusammen
5. **Deep Interaction** — Chat mit Agents und ReportAgent

## Was an diesem Fork anders ist

| Bereich | Upstream MiroFish | Dieser Fork |
|---|---|---|
| Memory-Layer | Zep Cloud (`zep-cloud` SDK, API-Key) | Graphiti 0.29 + Neo4j 5.26 Community, beides lokal als Compose-Service |
| Memory-Touchpoint | mehrere direkte `zep_*`-Imports | genau einer: `backend/app/services/memory_service.py` |
| Episode-Verarbeitung | asynchron mit Polling | synchron via `add_episode` (returnt erst, wenn Nodes/Edges in Neo4j sind) |
| Report-Export | nur In-App-Anzeige | zusätzlich Markdown- und PDF-Download (WeasyPrint, server-side) |
| Externe Abhängigkeit | Zep-Account erforderlich | nur OpenAI-kompatibler LLM-Endpoint |

Der frühere Zep-Compat-Wrapper wurde komplett entfernt. Neue Memory-Aufrufe laufen ausschließlich über `memory_service` — die Schnittstelle ist bewusst portabel geschnitten, sodass das Modul auch in anderen Projekten wiederverwendbar bleibt.

## Quickstart

Voraussetzung: Docker und Docker Compose.

```bash
cp .env.example .env
# .env editieren: LLM_API_KEY, OPENAI_API_KEY, NEO4J_PASSWORD, NEO4J_AUTH
docker compose up -d
docker compose logs -f mirofish
```

| Service | URL |
|---|---|
| Frontend | `http://localhost:3000` |
| Backend | `http://localhost:5001` |
| Neo4j Browser | `http://localhost:7475` |

Die Neo4j-Ports liegen auf `7475/7688` statt der Defaults, damit parallel laufende Neo4j-Container nicht kollidieren.

## ENV-Variablen

| Variable | Pflicht | Zweck |
|---|---|---|
| `LLM_API_KEY` | ja | OpenAI-kompatibler Key für Chat und Embeddings |
| `LLM_BASE_URL` | nein | Default `https://api.openai.com/v1` |
| `LLM_MODEL_NAME` | nein | Default `gpt-4o-mini`. **Nicht `gpt-5.x` / o-Serie** — Graphiti sendet `reasoning.effort='minimal'`, das lehnen Reasoning-Modelle ab |
| `OPENAI_API_KEY` | ja | Graphitis Default-Cross-Encoder liest ihn direkt aus dem Environment |
| `NEO4J_AUTH` | ja | Compose-Variante im Format `user/password` |
| `NEO4J_URI` | nein | Default `bolt://neo4j:7687` (Compose-intern) |
| `NEO4J_USER` | nein | Default `neo4j` |
| `NEO4J_PASSWORD` | ja | wie in `NEO4J_AUTH`, separat als Python-Setting |

`.env` ist gitignored und gehört nicht ins Repo.

## Memory-Service-API

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

`group_id` entspricht MiroFishs `graph_id` und hat das Format `mirofish_<16hex>`. Die Sync-Bridge zum Async-Graphiti-Loop läuft über einen dedizierten Background-Thread in `memory_service.py` — Service-Code soll `asyncio.run` nicht selbst aufrufen.

## Stolperfallen

- `docker compose restart` lädt `.env` **nicht** neu. Bei ENV-Änderungen `docker compose up -d --force-recreate mirofish`.
- Backend-Code ist via Bind-Mount live im Container (`./backend/app:/app/backend/app`). Flask reloadet automatisch. Vor Production-Deploy diesen Mount entfernen.
- `uv.lock` ist im Dockerfile frozen. Nach `pyproject.toml`-Änderungen lokal `cd backend && uv lock` ausführen und das Lockfile committen.
- `docker compose down -v` löscht das Neo4j-Volume — bei kaputten Indexen oder Schema-Migrationen der schnellste Reset.
- Ontology muss pro `group_id` registriert sein, **bevor** `add_episode` läuft. Ohne `register_ontology` extrahiert Graphiti ohne Schema.

## Smoke-Tests

```bash
# Graph aus Seed bauen
curl -X POST http://localhost:5001/api/graph/ontology/generate \
  -F "files=@seed.txt" \
  -F "simulation_requirement=..." \
  -F "project_name=smoke"

curl -X POST http://localhost:5001/api/graph/build \
  -H "Content-Type: application/json" \
  -d '{"project_id":"...","graph_name":"smoke"}'

# Search via ReportAgent
curl -X POST http://localhost:5001/api/report/chat \
  -H "Content-Type: application/json" \
  -d '{"simulation_id":"...","report_id":"...","message":"...","chat_history":[]}'
```

Direkter Neo4j-Check:

```bash
docker exec mirofish-neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
  "MATCH (n) WHERE n.group_id STARTS WITH 'mirofish_' \
   RETURN labels(n)[0] AS label, count(*) AS count ORDER BY count DESC"
```

## Credits

- **[MiroFish](https://github.com/666ghj/MiroFish)** (Upstream) — Konzept, Frontend, Simulations-Workflow. Maintained von [@666ghj](https://github.com/666ghj) mit Unterstützung der Shanda Group.
- **[OASIS](https://github.com/camel-ai/oasis)** vom CAMEL-AI-Team — Simulation-Engine, die die Agent-Interaktionen trägt.
- **[Graphiti](https://github.com/getzep/graphiti)** von getzep — der temporale Knowledge-Graph, der hier Zep Cloud ersetzt.
- **[Neo4j](https://neo4j.com/)** Community Edition — Storage-Backend für Graphiti.

Für Fragen zum ursprünglichen MiroFish, zur Vision und zu Demos siehe das [Upstream-Repo](https://github.com/666ghj/MiroFish).

## Lizenz

Folgt der Lizenz des Upstream-Projekts — siehe `LICENSE` in beiden Repos.
