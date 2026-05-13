"""
Ontology generation service
Endpoint 1: Analyzes text content and generates entity and relationship type definitions for social simulation.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient
from ..utils.locale import get_language_instruction

logger = logging.getLogger(__name__)


def _to_pascal_case(name: str) -> str:
    """Convert any-format name to PascalCase (e.g. 'works_for' -> 'WorksFor', 'person' -> 'Person')."""
    # Split by non-alphanumeric characters
    parts = re.split(r'[^a-zA-Z0-9]+', name)
    # Then split on camelCase boundaries (e.g. 'camelCase' -> ['camel', 'Case'])
    words = []
    for part in parts:
        words.extend(re.sub(r'([a-z])([A-Z])', r'\1_\2', part).split('_'))
    # Capitalize each word and drop empties
    result = ''.join(word.capitalize() for word in words if word)
    return result if result else 'Unknown'


# System prompt for ontology generation
ONTOLOGY_SYSTEM_PROMPT = """Sie sind ein professioneller Experte für die Gestaltung von Wissensgraph-Ontologien. Ihre Aufgabe ist es, den gegebenen Textinhalt und die Simulationsanforderungen zu analysieren und Entitätstypen sowie Relationstypen zu entwerfen, die für eine **Social-Media-Meinungssimulation** geeignet sind.

**Wichtig: Sie müssen gültige JSON-Daten ausgeben, keinen anderen Inhalt.**

## Hintergrund der Kernaufgabe

Wir bauen ein **Social-Media-Meinungssimulationssystem**. In diesem System gilt:
- Jede Entität ist ein "Account" oder "Akteur", der sich in sozialen Medien äußern, interagieren und Informationen verbreiten kann
- Entitäten beeinflussen sich gegenseitig, teilen, kommentieren und reagieren aufeinander
- Wir müssen die Reaktionen aller Beteiligten in Meinungsereignissen und die Pfade der Informationsverbreitung simulieren

Daher **müssen Entitäten reale Akteure sein, die sich in sozialen Medien äußern und interagieren können**:

**Erlaubt sind**:
- Konkrete Einzelpersonen (Personen des öffentlichen Lebens, Beteiligte, Meinungsführer, Fachleute, Privatpersonen)
- Firmen, Unternehmen (inkl. ihrer offiziellen Accounts)
- Organisationen (Universitäten, Verbände, NGOs, Gewerkschaften usw.)
- Behörden, Aufsichtsorgane
- Medienorganisationen (Zeitungen, Fernsehsender, Self-Media, Webseiten)
- Social-Media-Plattformen selbst
- Vertretungen bestimmter Gruppen (z. B. Alumni-Vereinigungen, Fanclubs, Aktivistengruppen)

**Nicht erlaubt sind**:
- Abstrakte Konzepte (z. B. "öffentliche Meinung", "Stimmung", "Trend")
- Themen (z. B. "wissenschaftliche Integrität", "Bildungsreform")
- Standpunkte/Haltungen (z. B. "Befürworter", "Gegner")

## Ausgabeformat

Geben Sie JSON in folgender Struktur aus:

```json
{
    "entity_types": [
        {
            "name": "Entitätstyp-Name (englisch, PascalCase)",
            "description": "Kurzbeschreibung (englisch, max. 100 Zeichen)",
            "attributes": [
                {
                    "name": "Attributname (englisch, snake_case)",
                    "type": "text",
                    "description": "Attributbeschreibung"
                }
            ],
            "examples": ["Beispielentität 1", "Beispielentität 2"]
        }
    ],
    "edge_types": [
        {
            "name": "Relationstyp-Name (englisch, UPPER_SNAKE_CASE)",
            "description": "Kurzbeschreibung (englisch, max. 100 Zeichen)",
            "source_targets": [
                {"source": "Quell-Entitätstyp", "target": "Ziel-Entitätstyp"}
            ],
            "attributes": []
        }
    ],
    "analysis_summary": "Kurze Analyse des Textinhalts"
}
```

## Designrichtlinien (extrem wichtig!)

### 1. Entwurf der Entitätstypen — strikt einzuhalten

**Anzahl: exakt 10 Entitätstypen erforderlich**

**Hierarchische Anforderungen (muss konkrete Typen UND Fallback-Typen enthalten)**:

Ihre 10 Entitätstypen müssen folgende Hierarchie umfassen:

A. **Fallback-Typen (zwingend, an den letzten 2 Positionen der Liste)**:
   - `Person`: Fallback für jede natürliche Person. Wenn eine Person zu keinem konkreteren Personentyp passt, ordnen Sie sie hier ein.
   - `Organization`: Fallback für jede Organisation. Wenn eine Organisation zu keinem konkreteren Organisationstyp passt, ordnen Sie sie hier ein.

B. **Konkrete Typen (8 Stück, basierend auf dem Textinhalt)**:
   - Entwerfen Sie konkretere Typen für die Hauptrollen im Text
   - Beispiel: Bei akademischen Vorgängen können das `Student`, `Professor`, `University` sein
   - Beispiel: Bei wirtschaftlichen Vorgängen können das `Company`, `CEO`, `Employee` sein

**Warum Fallback-Typen nötig sind**:
- Im Text tauchen verschiedenste Personen auf, etwa "Schullehrer", "Passant", "ein Internetnutzer"
- Wenn kein spezialisierter Typ passt, sollten sie in `Person` einsortiert werden
- Analog gehören kleine Organisationen oder temporäre Gruppen in `Organization`

**Designprinzipien für konkrete Typen**:
- Identifizieren Sie häufig auftretende oder zentrale Rollentypen aus dem Text
- Jeder konkrete Typ muss klar abgegrenzt sein, Überschneidungen vermeiden
- Die description muss klar erklären, wie sich dieser Typ vom Fallback-Typ unterscheidet

### 2. Entwurf der Relationstypen

- Anzahl: 6-10
- Relationen sollen reale Verbindungen in Social-Media-Interaktionen abbilden
- Stellen Sie sicher, dass die source_targets der Relationen die definierten Entitätstypen abdecken

### 3. Attributentwurf

- 1-3 Schlüsselattribute pro Entitätstyp
- **Hinweis**: Attributnamen dürfen nicht `name`, `uuid`, `group_id`, `created_at`, `summary` lauten (das sind reservierte Systembegriffe)
- Empfohlen: `full_name`, `title`, `role`, `position`, `location`, `description` etc.

## Referenz für Entitätstypen

**Personen (konkret)**:
- Student: Studierende
- Professor: Professor/Wissenschaftler
- Journalist: Journalist
- Celebrity: Prominenter/Influencer
- Executive: Führungskraft
- Official: Regierungsbeamter
- Lawyer: Anwalt
- Doctor: Arzt

**Personen (Fallback)**:
- Person: Jede natürliche Person (verwenden, wenn kein konkreter Typ passt)

**Organisationen (konkret)**:
- University: Hochschule
- Company: Unternehmen
- GovernmentAgency: Behörde
- MediaOutlet: Medienorganisation
- Hospital: Krankenhaus
- School: Schule
- NGO: Nichtregierungsorganisation

**Organisationen (Fallback)**:
- Organization: Jede Organisation (verwenden, wenn kein konkreter Typ passt)

## Referenz für Relationstypen

- WORKS_FOR: arbeitet bei
- STUDIES_AT: studiert an
- AFFILIATED_WITH: zugehörig zu
- REPRESENTS: vertritt
- REGULATES: reguliert
- REPORTS_ON: berichtet über
- COMMENTS_ON: kommentiert
- RESPONDS_TO: reagiert auf
- SUPPORTS: unterstützt
- OPPOSES: lehnt ab
- COLLABORATES_WITH: kollaboriert mit
- COMPETES_WITH: konkurriert mit
"""


class OntologyGenerator:
    """
    Ontology generator.
    Analyzes text content and generates entity and relationship type definitions.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    def generate(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an ontology definition.

        Args:
            document_texts: List of document texts
            simulation_requirement: Simulation requirement description
            additional_context: Additional context

        Returns:
            Ontology definition (entity_types, edge_types, ...)
        """
        # Build the user message
        user_message = self._build_user_message(
            document_texts,
            simulation_requirement,
            additional_context
        )

        lang_instruction = get_language_instruction()
        system_prompt = f"{ONTOLOGY_SYSTEM_PROMPT}\n\n{lang_instruction}\nIMPORTANT: Entity type names MUST be in English PascalCase (e.g., 'PersonEntity', 'MediaOrganization'). Relationship type names MUST be in English UPPER_SNAKE_CASE (e.g., 'WORKS_FOR'). Attribute names MUST be in English snake_case. Only description fields and analysis_summary should use the specified language above."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        # Call the LLM
        result = self.llm_client.chat_json(
            messages=messages,
            temperature=0.3,
            max_tokens=4096
        )

        # Validate and post-process
        result = self._validate_and_process(result)

        return result

    # Maximum text length passed to the LLM (50k characters)
    MAX_TEXT_LENGTH_FOR_LLM = 50000

    def _build_user_message(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str]
    ) -> str:
        """Build the user message"""

        # Concatenate the texts
        combined_text = "\n\n---\n\n".join(document_texts)
        original_length = len(combined_text)

        # If the text exceeds 50k characters, truncate (only affects what is sent to the LLM, not graph building)
        if len(combined_text) > self.MAX_TEXT_LENGTH_FOR_LLM:
            combined_text = combined_text[:self.MAX_TEXT_LENGTH_FOR_LLM]
            combined_text += f"\n\n...(Originaltext umfasst {original_length} Zeichen, die ersten {self.MAX_TEXT_LENGTH_FOR_LLM} Zeichen wurden zur Ontologie-Analyse übernommen)..."

        message = f"""## Simulationsanforderung

{simulation_requirement}

## Dokumentinhalt

{combined_text}
"""

        if additional_context:
            message += f"""
## Zusätzliche Hinweise

{additional_context}
"""

        message += """
Entwerfen Sie auf Grundlage des obigen Inhalts Entitäts- und Relationstypen, die für eine Social-Media-Meinungssimulation geeignet sind.

**Verbindliche Regeln**:
1. Geben Sie genau 10 Entitätstypen aus
2. Die letzten 2 müssen Fallback-Typen sein: Person (Personen-Fallback) und Organization (Organisations-Fallback)
3. Die ersten 8 sind konkrete, am Textinhalt orientierte Typen
4. Alle Entitätstypen müssen reale, sich äußernde Akteure sein, keine abstrakten Konzepte
5. Attributnamen dürfen nicht name, uuid, group_id u. ä. reservierte Begriffe verwenden; nutzen Sie stattdessen z. B. full_name, org_name
"""

        return message

    def _validate_and_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and post-process the result"""

        # Ensure required fields exist
        if "entity_types" not in result:
            result["entity_types"] = []
        if "edge_types" not in result:
            result["edge_types"] = []
        if "analysis_summary" not in result:
            result["analysis_summary"] = ""

        # Validate entity types
        # Track original-name -> PascalCase mapping to fix edge source_targets references later
        entity_name_map = {}
        for entity in result["entity_types"]:
            # Force entity name to PascalCase (required by Zep API)
            if "name" in entity:
                original_name = entity["name"]
                entity["name"] = _to_pascal_case(original_name)
                if entity["name"] != original_name:
                    logger.warning(f"Entity type name '{original_name}' auto-converted to '{entity['name']}'")
                entity_name_map[original_name] = entity["name"]
            if "attributes" not in entity:
                entity["attributes"] = []
            if "examples" not in entity:
                entity["examples"] = []
            # Cap description length at 100 characters
            if len(entity.get("description", "")) > 100:
                entity["description"] = entity["description"][:97] + "..."

        # Validate edge types
        for edge in result["edge_types"]:
            # Force edge name to SCREAMING_SNAKE_CASE (required by Zep API)
            if "name" in edge:
                original_name = edge["name"]
                edge["name"] = original_name.upper()
                if edge["name"] != original_name:
                    logger.warning(f"Edge type name '{original_name}' auto-converted to '{edge['name']}'")
            # Fix entity references in source_targets to match the converted PascalCase
            for st in edge.get("source_targets", []):
                if st.get("source") in entity_name_map:
                    st["source"] = entity_name_map[st["source"]]
                if st.get("target") in entity_name_map:
                    st["target"] = entity_name_map[st["target"]]
            if "source_targets" not in edge:
                edge["source_targets"] = []
            if "attributes" not in edge:
                edge["attributes"] = []
            if len(edge.get("description", "")) > 100:
                edge["description"] = edge["description"][:97] + "..."

        # Zep API limits: at most 10 custom entity types and 10 custom edge types
        MAX_ENTITY_TYPES = 10
        MAX_EDGE_TYPES = 10

        # Deduplicate by name, keeping the first occurrence
        seen_names = set()
        deduped = []
        for entity in result["entity_types"]:
            name = entity.get("name", "")
            if name and name not in seen_names:
                seen_names.add(name)
                deduped.append(entity)
            elif name in seen_names:
                logger.warning(f"Duplicate entity type '{name}' removed during validation")
        result["entity_types"] = deduped

        # Fallback type definitions
        person_fallback = {
            "name": "Person",
            "description": "Any individual person not fitting other specific person types.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Full name of the person"},
                {"name": "role", "type": "text", "description": "Role or occupation"}
            ],
            "examples": ["ordinary citizen", "anonymous netizen"]
        }

        organization_fallback = {
            "name": "Organization",
            "description": "Any organization not fitting other specific organization types.",
            "attributes": [
                {"name": "org_name", "type": "text", "description": "Name of the organization"},
                {"name": "org_type", "type": "text", "description": "Type of organization"}
            ],
            "examples": ["small business", "community group"]
        }

        # Check whether fallback types already exist
        entity_names = {e["name"] for e in result["entity_types"]}
        has_person = "Person" in entity_names
        has_organization = "Organization" in entity_names

        # Fallback types that need to be added
        fallbacks_to_add = []
        if not has_person:
            fallbacks_to_add.append(person_fallback)
        if not has_organization:
            fallbacks_to_add.append(organization_fallback)

        if fallbacks_to_add:
            current_count = len(result["entity_types"])
            needed_slots = len(fallbacks_to_add)

            # If adding pushes past 10, drop existing types
            if current_count + needed_slots > MAX_ENTITY_TYPES:
                # Compute how many to remove
                to_remove = current_count + needed_slots - MAX_ENTITY_TYPES
                # Remove from the tail (preserve more important specific types at the front)
                result["entity_types"] = result["entity_types"][:-to_remove]

            # Append fallback types
            result["entity_types"].extend(fallbacks_to_add)

        # Defensive cap to ensure we never exceed limits
        if len(result["entity_types"]) > MAX_ENTITY_TYPES:
            result["entity_types"] = result["entity_types"][:MAX_ENTITY_TYPES]

        if len(result["edge_types"]) > MAX_EDGE_TYPES:
            result["edge_types"] = result["edge_types"][:MAX_EDGE_TYPES]

        return result

    def generate_python_code(self, ontology: Dict[str, Any]) -> str:
        """
        Convert the ontology definition into Python code (similar to ontology.py).

        Args:
            ontology: Ontology definition

        Returns:
            Python code string
        """
        code_lines = [
            '"""',
            'Custom entity type definitions',
            'Auto-generated by MiroFish for social opinion simulation.',
            '"""',
            '',
            'from pydantic import Field',
            'from zep_cloud.external_clients.ontology import EntityModel, EntityText, EdgeModel',
            '',
            '',
            '# ============== Entity type definitions ==============',
            '',
        ]

        # Generate entity types
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            desc = entity.get("description", f"A {name} entity.")

            code_lines.append(f'class {name}(EntityModel):')
            code_lines.append(f'    """{desc}"""')

            attrs = entity.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')

            code_lines.append('')
            code_lines.append('')

        code_lines.append('# ============== Relationship type definitions ==============')
        code_lines.append('')

        # Generate relationship types
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            # Convert to PascalCase class name
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            desc = edge.get("description", f"A {name} relationship.")

            code_lines.append(f'class {class_name}(EdgeModel):')
            code_lines.append(f'    """{desc}"""')

            attrs = edge.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')

            code_lines.append('')
            code_lines.append('')

        # Generate type registries
        code_lines.append('# ============== Type configuration ==============')
        code_lines.append('')
        code_lines.append('ENTITY_TYPES = {')
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            code_lines.append(f'    "{name}": {name},')
        code_lines.append('}')
        code_lines.append('')
        code_lines.append('EDGE_TYPES = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            code_lines.append(f'    "{name}": {class_name},')
        code_lines.append('}')
        code_lines.append('')

        # Generate edge source_targets mapping
        code_lines.append('EDGE_SOURCE_TARGETS = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            source_targets = edge.get("source_targets", [])
            if source_targets:
                st_list = ', '.join([
                    f'{{"source": "{st.get("source", "Entity")}", "target": "{st.get("target", "Entity")}"}}'
                    for st in source_targets
                ])
                code_lines.append(f'    "{name}": [{st_list}],')
        code_lines.append('}')

        return '\n'.join(code_lines)
