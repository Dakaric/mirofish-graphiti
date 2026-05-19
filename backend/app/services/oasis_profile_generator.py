"""
OASIS agent profile generator
Converts entities from a Zep graph into the agent profile format expected by the OASIS simulation platform.

Optimizations:
1. Uses Zep retrieval to enrich node information further
2. Tuned prompts that produce very detailed personas
3. Distinguishes between individual entities and abstract group entities
"""

import json
import random
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from openai import OpenAI
from . import memory_service

from ..config import Config
from ..utils.logger import get_logger
from ..utils.locale import get_language_instruction, get_locale, set_locale, t
from .zep_entity_reader import EntityNode, ZepEntityReader

logger = get_logger('mirofish.oasis_profile')


@dataclass
class OasisAgentProfile:
    """OASIS agent profile data structure"""
    # Common fields
    user_id: int
    user_name: str
    name: str
    bio: str
    persona: str

    # Optional - Reddit style
    karma: int = 1000

    # Optional - Twitter style
    friend_count: int = 100
    follower_count: int = 150
    statuses_count: int = 500

    # Additional persona information
    age: Optional[int] = None
    gender: Optional[str] = None
    mbti: Optional[str] = None
    country: Optional[str] = None
    profession: Optional[str] = None
    interested_topics: List[str] = field(default_factory=list)

    # Source entity information
    source_entity_uuid: Optional[str] = None
    source_entity_type: Optional[str] = None

    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))

    def to_reddit_format(self) -> Dict[str, Any]:
        """Convert to Reddit platform format"""
        profile = {
            "user_id": self.user_id,
            "username": self.user_name,  # OASIS library requires the field name "username" (no underscore)
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "karma": self.karma,
            "created_at": self.created_at,
        }

        # Append optional persona information when present
        if self.age:
            profile["age"] = self.age
        if self.gender:
            profile["gender"] = self.gender
        if self.mbti:
            profile["mbti"] = self.mbti
        if self.country:
            profile["country"] = self.country
        if self.profession:
            profile["profession"] = self.profession
        if self.interested_topics:
            profile["interested_topics"] = self.interested_topics
        
        return profile
    
    def to_twitter_format(self) -> Dict[str, Any]:
        """Convert to Twitter platform format"""
        profile = {
            "user_id": self.user_id,
            "username": self.user_name,  # OASIS library requires the field name "username" (no underscore)
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "friend_count": self.friend_count,
            "follower_count": self.follower_count,
            "statuses_count": self.statuses_count,
            "created_at": self.created_at,
        }

        # Append optional persona information
        if self.age:
            profile["age"] = self.age
        if self.gender:
            profile["gender"] = self.gender
        if self.mbti:
            profile["mbti"] = self.mbti
        if self.country:
            profile["country"] = self.country
        if self.profession:
            profile["profession"] = self.profession
        if self.interested_topics:
            profile["interested_topics"] = self.interested_topics
        
        return profile
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a complete dict representation"""
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "karma": self.karma,
            "friend_count": self.friend_count,
            "follower_count": self.follower_count,
            "statuses_count": self.statuses_count,
            "age": self.age,
            "gender": self.gender,
            "mbti": self.mbti,
            "country": self.country,
            "profession": self.profession,
            "interested_topics": self.interested_topics,
            "source_entity_uuid": self.source_entity_uuid,
            "source_entity_type": self.source_entity_type,
            "created_at": self.created_at,
        }


class OasisProfileGenerator:
    """
    OASIS profile generator.

    Converts entities from a Zep graph into agent profiles required by the OASIS simulation.

    Features:
    1. Calls Zep graph retrieval to obtain richer context
    2. Generates very detailed personas (basic info, career history, personality traits, social media behavior, ...)
    3. Distinguishes individual entities from group/institutional entities
    """

    # MBTI types
    MBTI_TYPES = [
        "INTJ", "INTP", "ENTJ", "ENTP",
        "INFJ", "INFP", "ENFJ", "ENFP",
        "ISTJ", "ISFJ", "ESTJ", "ESFJ",
        "ISTP", "ISFP", "ESTP", "ESFP"
    ]

    # Common countries
    COUNTRIES = [
        "China", "US", "UK", "Japan", "Germany", "France",
        "Canada", "Australia", "Brazil", "India", "South Korea"
    ]

    # Entity types representing individuals (require concrete personas)
    INDIVIDUAL_ENTITY_TYPES = [
        "student", "alumni", "professor", "person", "publicfigure",
        "expert", "faculty", "official", "journalist", "activist"
    ]

    # Entity types representing groups/institutions (require representative-account personas)
    GROUP_ENTITY_TYPES = [
        "university", "governmentagency", "organization", "ngo",
        "mediaoutlet", "company", "institution", "group", "community"
    ]
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        zep_api_key: Optional[str] = None,
        graph_id: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model_name = model_name or Config.LLM_MODEL_NAME
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY is not configured")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        # memory_service replaces the previous Zep client. zep_api_key is kept
        # in the signature for backwards-compat but unused.
        self._zep_api_key = zep_api_key
        self.graph_id = graph_id
        self.memory_available = False
        try:
            memory_service.warmup()
            self.memory_available = True
        except Exception as e:
            logger.warning(f"Failed to initialize memory service: {e}")
    
    def generate_profile_from_entity(
        self, 
        entity: EntityNode, 
        user_id: int,
        use_llm: bool = True
    ) -> OasisAgentProfile:
        """
        Generate an OASIS agent profile from a Zep entity.

        Args:
            entity: Zep entity node
            user_id: User ID (for OASIS)
            use_llm: Whether to use the LLM for a detailed persona

        Returns:
            OasisAgentProfile
        """
        entity_type = entity.get_entity_type() or "Entity"

        # Basic info
        name = entity.name
        user_name = self._generate_username(name)

        # Build context
        context = self._build_entity_context(entity)

        if use_llm:
            # Generate a detailed persona via LLM
            profile_data = self._generate_profile_with_llm(
                entity_name=name,
                entity_type=entity_type,
                entity_summary=entity.summary,
                entity_attributes=entity.attributes,
                context=context
            )
        else:
            # Generate a baseline persona using rules
            profile_data = self._generate_profile_rule_based(
                entity_name=name,
                entity_type=entity_type,
                entity_summary=entity.summary,
                entity_attributes=entity.attributes
            )
        
        return OasisAgentProfile(
            user_id=user_id,
            user_name=user_name,
            name=name,
            bio=profile_data.get("bio", f"{entity_type}: {name}"),
            persona=profile_data.get("persona", entity.summary or f"A {entity_type} named {name}."),
            karma=profile_data.get("karma", random.randint(500, 5000)),
            friend_count=profile_data.get("friend_count", random.randint(50, 500)),
            follower_count=profile_data.get("follower_count", random.randint(100, 1000)),
            statuses_count=profile_data.get("statuses_count", random.randint(100, 2000)),
            age=profile_data.get("age"),
            gender=profile_data.get("gender"),
            mbti=profile_data.get("mbti"),
            country=profile_data.get("country"),
            profession=profile_data.get("profession"),
            interested_topics=profile_data.get("interested_topics", []),
            source_entity_uuid=entity.uuid,
            source_entity_type=entity_type,
        )
    
    def _generate_username(self, name: str) -> str:
        """Generate a username"""
        # Strip special characters and lowercase
        username = name.lower().replace(" ", "_")
        username = ''.join(c for c in username if c.isalnum() or c == '_')

        # Add a random suffix to avoid collisions
        suffix = random.randint(100, 999)
        return f"{username}_{suffix}"

    def _search_zep_for_entity(self, entity: EntityNode) -> Dict[str, Any]:
        """
        Use Zep graph hybrid search to fetch rich information about an entity.

        Zep has no built-in hybrid search endpoint, so edges and nodes are searched separately and merged.
        The two searches run in parallel for efficiency.

        Args:
            entity: Entity node object

        Returns:
            Dict containing facts, node_summaries, and context
        """
        import concurrent.futures

        if not self.memory_available:
            return {"facts": [], "node_summaries": [], "context": ""}

        entity_name = entity.name

        results = {
            "facts": [],
            "node_summaries": [],
            "context": ""
        }

        # graph_id is required for searching
        if not self.graph_id:
            logger.debug("Skipping Zep retrieval: graph_id not set")
            return results

        comprehensive_query = t('progress.zepSearchQuery', name=entity_name)

        def _retry(fn, label: str):
            delay = 2.0
            for attempt in range(3):
                try:
                    return fn()
                except Exception as e:
                    if attempt < 2:
                        logger.debug(f"Memory {label} attempt {attempt + 1} failed: {str(e)[:80]}, retrying...")
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.debug(f"Memory {label} still failing after 3 attempts: {e}")
            return None

        def search_edges():
            return _retry(
                lambda: memory_service.search_edges(group_id=self.graph_id, query=comprehensive_query, limit=30),
                "edge search",
            )

        def search_nodes():
            return _retry(
                lambda: memory_service.search_nodes(group_id=self.graph_id, query=comprehensive_query, limit=20),
                "node search",
            )

        try:
            # Run edge and node searches in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                edge_future = executor.submit(search_edges)
                node_future = executor.submit(search_nodes)

                # Collect results
                edge_result = edge_future.result(timeout=30)
                node_result = node_future.result(timeout=30)

            # Process edge search results
            all_facts = set()
            if edge_result:
                for edge in edge_result:
                    if edge.fact:
                        all_facts.add(edge.fact)
            results["facts"] = list(all_facts)

            # Process node search results
            all_summaries = set()
            if node_result:
                for node in node_result:
                    if node.summary:
                        all_summaries.add(node.summary)
                    if node.name and node.name != entity_name:
                        all_summaries.add(f"Verwandte Entität: {node.name}")
            results["node_summaries"] = list(all_summaries)

            # Build the combined context
            context_parts = []
            if results["facts"]:
                context_parts.append("Faktinformationen:\n" + "\n".join(f"- {f}" for f in results["facts"][:20]))
            if results["node_summaries"]:
                context_parts.append("Verwandte Entitäten:\n" + "\n".join(f"- {s}" for s in results["node_summaries"][:10]))
            results["context"] = "\n\n".join(context_parts)

            logger.info(f"Zep hybrid retrieval done: {entity_name}, fetched {len(results['facts'])} facts, {len(results['node_summaries'])} related nodes")

        except concurrent.futures.TimeoutError:
            logger.warning(f"Zep retrieval timed out ({entity_name})")
        except Exception as e:
            logger.warning(f"Zep retrieval failed ({entity_name}): {e}")

        return results
    
    def _build_entity_context(self, entity: EntityNode) -> str:
        """
        Build the full context for an entity.

        Includes:
        1. Edge information from the entity itself (facts)
        2. Detailed info on related nodes
        3. Rich info pulled in via Zep hybrid retrieval

        The returned string is consumed by an LLM prompt, so headings are in German.
        """
        context_parts = []

        # 1. Append entity attribute information
        if entity.attributes:
            attrs = []
            for key, value in entity.attributes.items():
                if value and str(value).strip():
                    attrs.append(f"- {key}: {value}")
            if attrs:
                context_parts.append("### Entitätsattribute\n" + "\n".join(attrs))

        # 2. Append related edge information (facts/relationships)
        existing_facts = set()
        if entity.related_edges:
            relationships = []
            for edge in entity.related_edges:  # no count limit
                fact = edge.get("fact", "")
                edge_name = edge.get("edge_name", "")
                direction = edge.get("direction", "")

                if fact:
                    relationships.append(f"- {fact}")
                    existing_facts.add(fact)
                elif edge_name:
                    if direction == "outgoing":
                        relationships.append(f"- {entity.name} --[{edge_name}]--> (verwandte Entität)")
                    else:
                        relationships.append(f"- (verwandte Entität) --[{edge_name}]--> {entity.name}")

            if relationships:
                context_parts.append("### Relevante Fakten und Beziehungen\n" + "\n".join(relationships))

        # 3. Append details of related nodes
        if entity.related_nodes:
            related_info = []
            for node in entity.related_nodes:  # no count limit
                node_name = node.get("name", "")
                node_labels = node.get("labels", [])
                node_summary = node.get("summary", "")

                # Drop default labels
                custom_labels = [l for l in node_labels if l not in ["Entity", "Node"]]
                label_str = f" ({', '.join(custom_labels)})" if custom_labels else ""

                if node_summary:
                    related_info.append(f"- **{node_name}**{label_str}: {node_summary}")
                else:
                    related_info.append(f"- **{node_name}**{label_str}")

            if related_info:
                context_parts.append("### Informationen zu verwandten Entitäten\n" + "\n".join(related_info))

        # 4. Use Zep hybrid retrieval to enrich further
        zep_results = self._search_zep_for_entity(entity)

        if zep_results.get("facts"):
            # Deduplicate: skip facts we already have
            new_facts = [f for f in zep_results["facts"] if f not in existing_facts]
            if new_facts:
                context_parts.append("### Über Zep gefundene Fakten\n" + "\n".join(f"- {f}" for f in new_facts[:15]))

        if zep_results.get("node_summaries"):
            context_parts.append("### Über Zep gefundene verwandte Knoten\n" + "\n".join(f"- {s}" for s in zep_results["node_summaries"][:10]))

        return "\n\n".join(context_parts)

    def _is_individual_entity(self, entity_type: str) -> bool:
        """Return True if the entity type represents an individual"""
        return entity_type.lower() in self.INDIVIDUAL_ENTITY_TYPES

    def _is_group_entity(self, entity_type: str) -> bool:
        """Return True if the entity type represents a group/institution"""
        return entity_type.lower() in self.GROUP_ENTITY_TYPES
    
    def _generate_profile_with_llm(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> Dict[str, Any]:
        """
        Generate a very detailed persona via the LLM.

        Branches by entity type:
        - Individual entities: produce a concrete person profile
        - Group/institution entities: produce a representative account profile
        """

        is_individual = self._is_individual_entity(entity_type)

        if is_individual:
            prompt = self._build_individual_persona_prompt(
                entity_name, entity_type, entity_summary, entity_attributes, context
            )
        else:
            prompt = self._build_group_persona_prompt(
                entity_name, entity_type, entity_summary, entity_attributes, context
            )

        # Retry until success or max attempts reached
        max_attempts = 3
        last_error = None

        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt(is_individual)},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7 - (attempt * 0.1)  # Lower temperature on each retry
                    # Don't set max_tokens — let the LLM expand freely
                )

                content = response.choices[0].message.content

                # Detect truncation (finish_reason != 'stop')
                finish_reason = response.choices[0].finish_reason
                if finish_reason == 'length':
                    logger.warning(f"LLM output truncated (attempt {attempt+1}), attempting repair...")
                    content = self._fix_truncated_json(content)

                # Attempt to parse JSON
                try:
                    result = json.loads(content)

                    # Validate required fields
                    if "bio" not in result or not result["bio"]:
                        result["bio"] = entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}"
                    if "persona" not in result or not result["persona"]:
                        result["persona"] = entity_summary or f"{entity_name} ist ein {entity_type}."

                    return result

                except json.JSONDecodeError as je:
                    logger.warning(f"JSON parse failed (attempt {attempt+1}): {str(je)[:80]}")

                    # Try to repair the JSON
                    result = self._try_fix_json(content, entity_name, entity_type, entity_summary)
                    if result.get("_fixed"):
                        del result["_fixed"]
                        return result

                    last_error = je

            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt+1}): {str(e)[:80]}")
                last_error = e
                import time
                time.sleep(1 * (attempt + 1))  # Exponential backoff

        logger.warning(f"LLM persona generation failed after {max_attempts} attempts: {last_error}, falling back to rule-based generation")
        return self._generate_profile_rule_based(
            entity_name, entity_type, entity_summary, entity_attributes
        )
    
    def _fix_truncated_json(self, content: str) -> str:
        """Repair truncated JSON (output cut off by max_tokens)."""
        import re

        # If the JSON is truncated, attempt to close it
        content = content.strip()

        # Count unclosed brackets
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')

        # Check for an unclosed string
        # Heuristic: if the trailing char is not a comma/closing brace, the value may be cut off
        if content and content[-1] not in '",}]':
            # Try closing the string
            content += '"'

        # Close any remaining brackets
        content += ']' * open_brackets
        content += '}' * open_braces

        return content

    def _try_fix_json(self, content: str, entity_name: str, entity_type: str, entity_summary: str = "") -> Dict[str, Any]:
        """Attempt to repair broken JSON"""
        import re

        # 1. First try to repair truncation
        content = self._fix_truncated_json(content)

        # 2. Try extracting the JSON portion
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group()

            # 3. Sanitize newline characters inside string values
            def fix_string_newlines(match):
                s = match.group(0)
                # Replace newlines inside the string with spaces
                s = s.replace('\n', ' ').replace('\r', ' ')
                # Collapse extra whitespace
                s = re.sub(r'\s+', ' ', s)
                return s

            # Match JSON string values
            json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string_newlines, json_str)

            # 4. Try parsing
            try:
                result = json.loads(json_str)
                result["_fixed"] = True
                return result
            except json.JSONDecodeError as e:
                # 5. Try a more aggressive repair
                try:
                    # Strip all control characters
                    json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
                    # Collapse all whitespace
                    json_str = re.sub(r'\s+', ' ', json_str)
                    result = json.loads(json_str)
                    result["_fixed"] = True
                    return result
                except:
                    pass

        # 6. Try extracting partial info from the content
        bio_match = re.search(r'"bio"\s*:\s*"([^"]*)"', content)
        persona_match = re.search(r'"persona"\s*:\s*"([^"]*)', content)  # may be truncated

        bio = bio_match.group(1) if bio_match else (entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}")
        persona = persona_match.group(1) if persona_match else (entity_summary or f"{entity_name} ist ein {entity_type}.")

        # If something meaningful was extracted, mark as fixed
        if bio_match or persona_match:
            logger.info("Extracted partial info from broken JSON")
            return {
                "bio": bio,
                "persona": persona,
                "_fixed": True
            }

        # 7. Total failure — return baseline structure
        logger.warning("JSON repair failed; returning baseline structure")
        return {
            "bio": entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}",
            "persona": entity_summary or f"{entity_name} ist ein {entity_type}."
        }

    def _get_system_prompt(self, is_individual: bool) -> str:
        """Return the system prompt"""
        base_prompt = "Sie sind Experte für die Erzeugung von Social-Media-Nutzerprofilen. Erzeugen Sie detaillierte, realitätsnahe Personas für eine Meinungssimulation und bilden Sie die bereits vorhandene reale Lage so weit wie möglich ab. Geben Sie zwingend gültiges JSON zurück; alle String-Werte dürfen keine unmaskierten Zeilenumbrüche enthalten."
        return f"{base_prompt}\n\n{get_language_instruction()}"

    def _build_individual_persona_prompt(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> str:
        """Build the detailed persona prompt for an individual entity"""

        attrs_str = json.dumps(entity_attributes, ensure_ascii=False) if entity_attributes else "keine"
        context_str = context[:3000] if context else "kein zusätzlicher Kontext"

        return f"""Erzeugen Sie eine detaillierte Social-Media-Nutzer-Persona für eine Entität und bilden Sie die bereits vorhandene reale Lage so weit wie möglich ab.

Entitätsname: {entity_name}
Entitätstyp: {entity_type}
Entitätszusammenfassung: {entity_summary}
Entitätsattribute: {attrs_str}

Kontextinformationen:
{context_str}

Erzeugen Sie JSON mit folgenden Feldern:

1. bio: Social-Media-Kurzbeschreibung, 200 Zeichen
2. persona: Detaillierte Persona-Beschreibung (2000 Zeichen reiner Text), enthält:
   - Grunddaten (Alter, Beruf, Bildungsweg, Wohnort)
   - Hintergrund (wichtige Erlebnisse, Bezug zum Ereignis, soziale Beziehungen)
   - Persönlichkeitsmerkmale (MBTI-Typ, Kernpersönlichkeit, emotionale Ausdrucksweise)
   - Social-Media-Verhalten (Posting-Frequenz, Inhaltsvorlieben, Interaktionsstil, sprachliche Eigenheiten)
   - Standpunkte (Haltung zum Thema, mögliche Auslöser für Empörung/Begeisterung)
   - Besonderheiten (Lieblingsfloskeln, besondere Erlebnisse, Hobbys)
   - Persönliche Erinnerungen (zentraler Teil der Persona: Bezug der Person zum Ereignis sowie bereits gezeigte Handlungen und Reaktionen im Verlauf des Ereignisses)
3. age: Alter als Zahl (zwingend Integer)
4. gender: Geschlecht, zwingend englisch: "male" oder "female"
5. mbti: MBTI-Typ (z. B. INTJ, ENFP)
6. country: Land
7. profession: Beruf
8. interested_topics: Array mit Interessensthemen

Wichtig:
- Alle Feldwerte müssen Strings oder Zahlen sein, keine Zeilenumbrüche verwenden
- persona muss ein zusammenhängender Fließtext sein
- {get_language_instruction()} (das Feld gender muss englisch male/female lauten)
- Inhalte müssen mit den Entitätsinformationen konsistent sein
- age muss ein gültiger Integer sein, gender muss "male" oder "female" sein
"""

    def _build_group_persona_prompt(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> str:
        """Build the detailed persona prompt for a group/institutional entity"""

        attrs_str = json.dumps(entity_attributes, ensure_ascii=False) if entity_attributes else "keine"
        context_str = context[:3000] if context else "kein zusätzlicher Kontext"

        return f"""Erzeugen Sie ein detailliertes Social-Media-Account-Profil für eine Institution/Gruppen-Entität und bilden Sie die bereits vorhandene reale Lage so weit wie möglich ab.

Entitätsname: {entity_name}
Entitätstyp: {entity_type}
Entitätszusammenfassung: {entity_summary}
Entitätsattribute: {attrs_str}

Kontextinformationen:
{context_str}

Erzeugen Sie JSON mit folgenden Feldern:

1. bio: Offizielle Account-Kurzbeschreibung, 200 Zeichen, professionell und sachlich
2. persona: Detaillierte Account-Beschreibung (2000 Zeichen reiner Text), enthält:
   - Grunddaten der Institution (offizieller Name, Charakter der Institution, Gründungshintergrund, Hauptaufgaben)
   - Account-Positionierung (Account-Typ, Zielgruppe, Kernfunktion)
   - Sprachstil (sprachliche Merkmale, häufige Formulierungen, Tabuthemen)
   - Inhaltsmerkmale (Inhaltstyp, Posting-Frequenz, aktive Tageszeiten)
   - Haltung (offizielle Position zu Kernthemen, Umgang mit Kontroversen)
   - Besondere Hinweise (vertretenes Gruppenprofil, Betriebsgewohnheiten)
   - Institutionelle Erinnerungen (zentraler Teil der Persona: Bezug der Institution zum Ereignis sowie bereits gezeigte Handlungen und Reaktionen im Verlauf des Ereignisses)
3. age: fest 30 (virtuelles Alter eines Institutions-Accounts)
4. gender: fest "other" (Institutions-Accounts verwenden "other" als Nicht-Person)
5. mbti: MBTI-Typ zur Beschreibung des Account-Stils (z. B. ISTJ für streng-konservativ)
6. country: Land
7. profession: Beschreibung der institutionellen Funktion
8. interested_topics: Array der Themenfelder

Wichtig:
- Alle Feldwerte müssen Strings oder Zahlen sein, keine null-Werte
- persona muss ein zusammenhängender Fließtext ohne Zeilenumbrüche sein
- {get_language_instruction()} (Feld gender muss englisch "other" sein)
- age muss Integer 30 sein, gender muss String "other" sein
- Der Account muss seiner Identität entsprechend formulieren"""
    
    def _generate_profile_rule_based(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a baseline persona using rules"""

        # Branch the persona by entity type
        entity_type_lower = entity_type.lower()
        
        if entity_type_lower in ["student", "alumni"]:
            return {
                "bio": f"{entity_type} with interests in academics and social issues.",
                "persona": f"{entity_name} is a {entity_type.lower()} who is actively engaged in academic and social discussions. They enjoy sharing perspectives and connecting with peers.",
                "age": random.randint(18, 30),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(self.MBTI_TYPES),
                "country": random.choice(self.COUNTRIES),
                "profession": "Student",
                "interested_topics": ["Education", "Social Issues", "Technology"],
            }
        
        elif entity_type_lower in ["publicfigure", "expert", "faculty"]:
            return {
                "bio": f"Expert and thought leader in their field.",
                "persona": f"{entity_name} is a recognized {entity_type.lower()} who shares insights and opinions on important matters. They are known for their expertise and influence in public discourse.",
                "age": random.randint(35, 60),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(["ENTJ", "INTJ", "ENTP", "INTP"]),
                "country": random.choice(self.COUNTRIES),
                "profession": entity_attributes.get("occupation", "Expert"),
                "interested_topics": ["Politics", "Economics", "Culture & Society"],
            }
        
        elif entity_type_lower in ["mediaoutlet", "socialmediaplatform"]:
            return {
                "bio": f"Official account for {entity_name}. News and updates.",
                "persona": f"{entity_name} is a media entity that reports news and facilitates public discourse. The account shares timely updates and engages with the audience on current events.",
                "age": 30,  # virtual age for institutional accounts
                "gender": "other",  # institutions use "other"
                "mbti": "ISTJ",  # institutional tone: strict and conservative
                "country": "China",
                "profession": "Media",
                "interested_topics": ["General News", "Current Events", "Public Affairs"],
            }

        elif entity_type_lower in ["university", "governmentagency", "ngo", "organization"]:
            return {
                "bio": f"Official account of {entity_name}.",
                "persona": f"{entity_name} is an institutional entity that communicates official positions, announcements, and engages with stakeholders on relevant matters.",
                "age": 30,  # virtual age for institutional accounts
                "gender": "other",  # institutions use "other"
                "mbti": "ISTJ",  # institutional tone: strict and conservative
                "country": "China",
                "profession": entity_type,
                "interested_topics": ["Public Policy", "Community", "Official Announcements"],
            }

        else:
            # Default persona
            return {
                "bio": entity_summary[:150] if entity_summary else f"{entity_type}: {entity_name}",
                "persona": entity_summary or f"{entity_name} is a {entity_type.lower()} participating in social discussions.",
                "age": random.randint(25, 50),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(self.MBTI_TYPES),
                "country": random.choice(self.COUNTRIES),
                "profession": entity_type,
                "interested_topics": ["General", "Social Issues"],
            }
    
    def set_graph_id(self, graph_id: str):
        """Set the graph ID used for Zep retrieval"""
        self.graph_id = graph_id

    def generate_profiles_from_entities(
        self,
        entities: List[EntityNode],
        use_llm: bool = True,
        progress_callback: Optional[callable] = None,
        graph_id: Optional[str] = None,
        parallel_count: int = 5,
        realtime_output_path: Optional[str] = None,
        output_platform: str = "reddit"
    ) -> List[OasisAgentProfile]:
        """
        Batch-generate agent profiles from entities (parallel-capable).

        Args:
            entities: List of entities
            use_llm: Whether to use the LLM for detailed personas
            progress_callback: Progress callback (current, total, message)
            graph_id: Graph ID used for Zep retrieval to fetch richer context
            parallel_count: Parallelism, default 5
            realtime_output_path: File path to write incrementally (after each profile)
            output_platform: Output format, "reddit" or "twitter"

        Returns:
            List of agent profiles
        """
        import concurrent.futures
        from threading import Lock

        # Configure graph_id for Zep retrieval
        if graph_id:
            self.graph_id = graph_id

        total = len(entities)
        profiles = [None] * total  # preallocate to preserve order
        completed_count = [0]  # use list so it is mutable from the closure
        lock = Lock()

        # Helper for incremental writes
        def save_profiles_realtime():
            """Save the profiles generated so far to disk"""
            if not realtime_output_path:
                return

            with lock:
                # Filter out unfinished slots
                existing_profiles = [p for p in profiles if p is not None]
                if not existing_profiles:
                    return

                try:
                    if output_platform == "reddit":
                        # Reddit JSON format
                        profiles_data = [p.to_reddit_format() for p in existing_profiles]
                        with open(realtime_output_path, 'w', encoding='utf-8') as f:
                            json.dump(profiles_data, f, ensure_ascii=False, indent=2)
                    else:
                        # Twitter CSV format
                        import csv
                        profiles_data = [p.to_twitter_format() for p in existing_profiles]
                        if profiles_data:
                            fieldnames = list(profiles_data[0].keys())
                            with open(realtime_output_path, 'w', encoding='utf-8', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=fieldnames)
                                writer.writeheader()
                                writer.writerows(profiles_data)
                except Exception as e:
                    logger.warning(f"Failed to save profiles incrementally: {e}")

        # Capture locale before spawning thread pool workers
        current_locale = get_locale()

        def generate_single_profile(idx: int, entity: EntityNode) -> tuple:
            """Worker function that generates a single profile"""
            set_locale(current_locale)
            entity_type = entity.get_entity_type() or "Entity"

            try:
                profile = self.generate_profile_from_entity(
                    entity=entity,
                    user_id=idx,
                    use_llm=use_llm
                )

                # Stream the generated persona to console and logs
                self._print_generated_profile(entity.name, entity_type, profile)

                return idx, profile, None

            except Exception as e:
                logger.error(f"Failed to generate persona for entity {entity.name}: {str(e)}")
                # Build a fallback profile
                fallback_profile = OasisAgentProfile(
                    user_id=idx,
                    user_name=self._generate_username(entity.name),
                    name=entity.name,
                    bio=f"{entity_type}: {entity.name}",
                    persona=entity.summary or f"A participant in social discussions.",
                    source_entity_uuid=entity.uuid,
                    source_entity_type=entity_type,
                )
                return idx, fallback_profile, str(e)

        logger.info(f"Starting parallel persona generation for {total} agents (parallelism: {parallel_count})...")
        print(f"\n{'='*60}")
        print(f"Generating agent personas - {total} entities, parallelism: {parallel_count}")
        print(f"{'='*60}\n")

        # Use a thread pool for parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_count) as executor:
            # Submit all tasks
            future_to_entity = {
                executor.submit(generate_single_profile, idx, entity): (idx, entity)
                for idx, entity in enumerate(entities)
            }

            # Collect results
            for future in concurrent.futures.as_completed(future_to_entity):
                idx, entity = future_to_entity[future]
                entity_type = entity.get_entity_type() or "Entity"

                try:
                    result_idx, profile, error = future.result()
                    profiles[result_idx] = profile

                    with lock:
                        completed_count[0] += 1
                        current = completed_count[0]

                    # Persist incrementally
                    save_profiles_realtime()

                    if progress_callback:
                        progress_callback(
                            current,
                            total,
                            f"Done {current}/{total}: {entity.name} ({entity_type})"
                        )

                    if error:
                        logger.warning(f"[{current}/{total}] {entity.name} fell back to default persona: {error}")
                    else:
                        logger.info(f"[{current}/{total}] Persona generated: {entity.name} ({entity_type})")

                except Exception as e:
                    logger.error(f"Exception while processing entity {entity.name}: {str(e)}")
                    with lock:
                        completed_count[0] += 1
                    profiles[idx] = OasisAgentProfile(
                        user_id=idx,
                        user_name=self._generate_username(entity.name),
                        name=entity.name,
                        bio=f"{entity_type}: {entity.name}",
                        persona=entity.summary or "A participant in social discussions.",
                        source_entity_uuid=entity.uuid,
                        source_entity_type=entity_type,
                    )
                    # Persist incrementally even for fallback personas
                    save_profiles_realtime()

        print(f"\n{'='*60}")
        print(f"Persona generation done. Generated {len([p for p in profiles if p])} agents.")
        print(f"{'='*60}\n")

        return profiles

    def _print_generated_profile(self, entity_name: str, entity_type: str, profile: OasisAgentProfile):
        """Stream the generated persona to the console (full content, no truncation)"""
        separator = "-" * 70

        # Build the full output without truncation
        topics_str = ', '.join(profile.interested_topics) if profile.interested_topics else 'none'

        output_lines = [
            f"\n{separator}",
            t('progress.profileGenerated', name=entity_name, type=entity_type),
            f"{separator}",
            f"Username: {profile.user_name}",
            f"",
            f"[Bio]",
            f"{profile.bio}",
            f"",
            f"[Detailed persona]",
            f"{profile.persona}",
            f"",
            f"[Basic attributes]",
            f"Age: {profile.age} | Gender: {profile.gender} | MBTI: {profile.mbti}",
            f"Profession: {profile.profession} | Country: {profile.country}",
            f"Interested topics: {topics_str}",
            separator
        ]

        output = "\n".join(output_lines)

        # Print only to console (logger no longer dumps full content)
        print(output)
    
    def save_profiles(
        self,
        profiles: List[OasisAgentProfile],
        file_path: str,
        platform: str = "reddit"
    ):
        """
        Save profiles to file (selecting the correct format per platform).

        OASIS platform format requirements:
        - Twitter: CSV format
        - Reddit: JSON format

        Args:
            profiles: List of profiles
            file_path: File path
            platform: Platform ("reddit" or "twitter")
        """
        if platform == "twitter":
            self._save_twitter_csv(profiles, file_path)
        else:
            self._save_reddit_json(profiles, file_path)

    def _save_twitter_csv(self, profiles: List[OasisAgentProfile], file_path: str):
        """
        Save Twitter profiles as CSV (per OASIS spec).

        CSV columns required by OASIS Twitter:
        - user_id: User ID (sequential from 0, by CSV order)
        - name: Real user name
        - username: Username in the system
        - user_char: Detailed persona (injected into the LLM system prompt to steer agent behavior)
        - description: Short public bio (shown on the profile page)

        Difference between user_char and description:
        - user_char: Internal use only — LLM system prompt that drives how the agent thinks and acts
        - description: External display — visible bio for other users
        """
        import csv

        # Ensure the file extension is .csv
        if not file_path.endswith('.csv'):
            file_path = file_path.replace('.json', '.csv')

        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Write the OASIS-required header
            headers = ['user_id', 'name', 'username', 'user_char', 'description']
            writer.writerow(headers)

            # Write data rows
            for idx, profile in enumerate(profiles):
                # user_char: full persona (bio + persona), used as the LLM system prompt
                user_char = profile.bio
                if profile.persona and profile.persona != profile.bio:
                    user_char = f"{profile.bio} {profile.persona}"
                # Replace newlines with spaces for CSV
                user_char = user_char.replace('\n', ' ').replace('\r', ' ')

                # description: short bio shown externally
                description = profile.bio.replace('\n', ' ').replace('\r', ' ')

                row = [
                    idx,                    # user_id: zero-based sequential ID
                    profile.name,           # name: real name
                    profile.user_name,      # username: handle
                    user_char,              # user_char: full persona (internal LLM use)
                    description             # description: short bio (external display)
                ]
                writer.writerow(row)

        logger.info(f"Saved {len(profiles)} Twitter profiles to {file_path} (OASIS CSV format)")

    def _normalize_gender(self, gender: Optional[str]) -> str:
        """
        Normalize the gender field to the English format expected by OASIS.

        OASIS expects: male, female, other.
        """
        if not gender:
            return "other"

        gender_lower = gender.lower().strip()

        # Map known values (German + English) to the OASIS gender enum
        gender_map = {
            # German
            "maennlich": "male",
            "männlich": "male",
            "weiblich": "female",
            "divers": "other",
            "organisation": "other",
            "sonstiges": "other",
            "andere": "other",
            # English passthrough
            "male": "male",
            "female": "female",
            "other": "other",
        }

        return gender_map.get(gender_lower, "other")

    def _save_reddit_json(self, profiles: List[OasisAgentProfile], file_path: str):
        """
        Save Reddit profiles as JSON.

        Uses the same shape as to_reddit_format() to ensure OASIS reads them correctly.
        The user_id field MUST be present — it is the key OASIS uses in agent_graph.get_agent().

        Required fields:
        - user_id: User ID (int, used to match poster_agent_id in initial_posts)
        - username: Username
        - name: Display name
        - bio: Bio
        - persona: Detailed persona
        - age: Age (int)
        - gender: "male", "female", or "other"
        - mbti: MBTI type
        - country: Country
        """
        data = []
        for idx, profile in enumerate(profiles):
            # Match the shape used by to_reddit_format()
            item = {
                "user_id": profile.user_id if profile.user_id is not None else idx,  # critical: must include user_id
                "username": profile.user_name,
                "name": profile.name,
                "bio": profile.bio[:150] if profile.bio else f"{profile.name}",
                "persona": profile.persona or f"{profile.name} is a participant in social discussions.",
                "karma": profile.karma if profile.karma else 1000,
                "created_at": profile.created_at,
                # OASIS required fields — ensure defaults
                "age": profile.age if profile.age else 30,
                "gender": self._normalize_gender(profile.gender),
                "mbti": profile.mbti if profile.mbti else "ISTJ",
                "country": profile.country if profile.country else "China",
            }

            # Optional fields
            if profile.profession:
                item["profession"] = profile.profession
            if profile.interested_topics:
                item["interested_topics"] = profile.interested_topics

            data.append(item)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(profiles)} Reddit profiles to {file_path} (JSON, includes user_id)")

    # Kept as alias for backward compatibility
    def save_profiles_to_json(
        self,
        profiles: List[OasisAgentProfile],
        file_path: str,
        platform: str = "reddit"
    ):
        """[Deprecated] Use save_profiles() instead."""
        logger.warning("save_profiles_to_json is deprecated; use save_profiles instead")
        self.save_profiles(profiles, file_path, platform)

