"""
LLM-driven simulation config generator.
Uses an LLM to derive detailed simulation parameters from the simulation requirement,
document text and graph information. The pipeline is fully automated and requires no manual tuning.

Generation is split into multiple steps to avoid producing a single overlong response:
1. Time configuration
2. Event configuration
3. Agent configurations (in batches)
4. Platform configuration
"""

import json
import math
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime

from openai import OpenAI

from ..config import Config
from ..utils.logger import get_logger
from ..utils.locale import get_language_instruction, t
from .zep_entity_reader import EntityNode, ZepEntityReader

logger = get_logger('mirofish.simulation_config')

# Reference daily-rhythm config for China (Beijing time)
CHINA_TIMEZONE_CONFIG = {
    # Dead hours (almost no activity)
    "dead_hours": [0, 1, 2, 3, 4, 5],
    # Morning hours (waking up)
    "morning_hours": [6, 7, 8],
    # Work hours
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    # Peak evening hours (most active)
    "peak_hours": [19, 20, 21, 22],
    # Late night (declining activity)
    "night_hours": [23],
    # Activity multipliers
    "activity_multipliers": {
        "dead": 0.05,      # virtually nobody around in the early hours
        "morning": 0.4,    # gradually picking up
        "work": 0.7,       # moderate during work hours
        "peak": 1.5,       # evening peak
        "night": 0.5       # late-night dropoff
    }
}


@dataclass
class AgentActivityConfig:
    """Per-agent activity configuration"""
    agent_id: int
    entity_uuid: str
    entity_name: str
    entity_type: str

    # Activity level (0.0-1.0)
    activity_level: float = 0.5  # overall activity level

    # Posting frequency (expected posts per hour)
    posts_per_hour: float = 1.0
    comments_per_hour: float = 2.0

    # Active hours (24h, 0-23)
    active_hours: List[int] = field(default_factory=lambda: list(range(8, 23)))

    # Response delay (latency to react to hot events, in simulated minutes)
    response_delay_min: int = 5
    response_delay_max: int = 60

    # Sentiment bias (-1.0 to 1.0, negative to positive)
    sentiment_bias: float = 0.0

    # Stance towards the topic
    stance: str = "neutral"  # supportive, opposing, neutral, observer

    # Influence weight (probability of this agent's posts being seen by others)
    influence_weight: float = 1.0


@dataclass
class TimeSimulationConfig:
    """Time-simulation configuration (rhythm based on Chinese daily routines)"""
    # Total simulated duration in hours
    total_simulation_hours: int = 72  # default: 72 hours (3 days)

    # Simulated minutes represented by one round (default: 60min, accelerates time)
    minutes_per_round: int = 60

    # Range of agents activated per hour
    agents_per_hour_min: int = 5
    agents_per_hour_max: int = 20

    # Peak hours (evening 19-22, the most active window in China)
    peak_hours: List[int] = field(default_factory=lambda: [19, 20, 21, 22])
    peak_activity_multiplier: float = 1.5

    # Off-peak hours (early morning 0-5, virtually inactive)
    off_peak_hours: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    off_peak_activity_multiplier: float = 0.05  # extremely low activity in the early hours

    # Morning hours
    morning_hours: List[int] = field(default_factory=lambda: [6, 7, 8])
    morning_activity_multiplier: float = 0.4

    # Work hours
    work_hours: List[int] = field(default_factory=lambda: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    work_activity_multiplier: float = 0.7


@dataclass
class EventConfig:
    """Event configuration"""
    # Initial events (triggered at simulation start)
    initial_posts: List[Dict[str, Any]] = field(default_factory=list)

    # Scheduled events (fired at specific times)
    scheduled_events: List[Dict[str, Any]] = field(default_factory=list)

    # Hot topic keywords
    hot_topics: List[str] = field(default_factory=list)

    # Narrative direction for the discussion
    narrative_direction: str = ""


@dataclass
class PlatformConfig:
    """Platform-specific configuration"""
    platform: str  # twitter or reddit

    # Recommendation algorithm weights
    recency_weight: float = 0.4  # recency
    popularity_weight: float = 0.3  # popularity
    relevance_weight: float = 0.3  # relevance

    # Viral threshold (interactions before content goes viral)
    viral_threshold: int = 10

    # Echo-chamber strength (how strongly similar opinions cluster)
    echo_chamber_strength: float = 0.5


@dataclass
class SimulationParameters:
    """Full simulation parameter configuration"""
    # Basic info
    simulation_id: str
    project_id: str
    graph_id: str
    simulation_requirement: str

    # Time configuration
    time_config: TimeSimulationConfig = field(default_factory=TimeSimulationConfig)

    # Agent configuration list
    agent_configs: List[AgentActivityConfig] = field(default_factory=list)

    # Event configuration
    event_config: EventConfig = field(default_factory=EventConfig)

    # Platform configuration
    twitter_config: Optional[PlatformConfig] = None
    reddit_config: Optional[PlatformConfig] = None

    # LLM configuration
    llm_model: str = ""
    llm_base_url: str = ""

    # Generation metadata
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_reasoning: str = ""  # LLM reasoning notes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict"""
        time_dict = asdict(self.time_config)
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "time_config": time_dict,
            "agent_configs": [asdict(a) for a in self.agent_configs],
            "event_config": asdict(self.event_config),
            "twitter_config": asdict(self.twitter_config) if self.twitter_config else None,
            "reddit_config": asdict(self.reddit_config) if self.reddit_config else None,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "generated_at": self.generated_at,
            "generation_reasoning": self.generation_reasoning,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to a JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class SimulationConfigGenerator:
    """
    LLM-driven simulation config generator.

    Analyzes the simulation requirement, document content and graph entities via an LLM
    and generates the best simulation parameter configuration.

    Step-by-step strategy:
    1. Generate time and event configuration (lightweight)
    2. Generate agent configurations in batches (10-20 per batch)
    3. Generate platform configuration
    """

    # Maximum context length in characters
    MAX_CONTEXT_LENGTH = 50000
    # Number of agents per batch
    AGENTS_PER_BATCH = 15

    # Per-step context truncation lengths (characters)
    TIME_CONFIG_CONTEXT_LENGTH = 10000   # time config
    EVENT_CONFIG_CONTEXT_LENGTH = 8000   # event config
    ENTITY_SUMMARY_LENGTH = 300          # entity summary
    AGENT_SUMMARY_LENGTH = 300           # entity summary inside agent config
    ENTITIES_PER_TYPE_DISPLAY = 20       # entities shown per type

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None
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

    def generate_config(
        self,
        simulation_id: str,
        project_id: str,
        graph_id: str,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode],
        enable_twitter: bool = True,
        enable_reddit: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> SimulationParameters:
        """
        Generate the full simulation config (step by step).

        Args:
            simulation_id: Simulation ID
            project_id: Project ID
            graph_id: Graph ID
            simulation_requirement: Simulation requirement description
            document_text: Original document text
            entities: Filtered list of entities
            enable_twitter: Whether to enable Twitter
            enable_reddit: Whether to enable Reddit
            progress_callback: Progress callback (current_step, total_steps, message)

        Returns:
            SimulationParameters: Full simulation parameter set
        """
        logger.info(f"Starting LLM-driven simulation config generation: simulation_id={simulation_id}, entities={len(entities)}")

        # Calculate the total number of steps
        num_batches = math.ceil(len(entities) / self.AGENTS_PER_BATCH)
        total_steps = 3 + num_batches  # time + event + N agent batches + platform

        def report_progress(step: int, message: str):
            if progress_callback:
                progress_callback(step, total_steps, message)
            logger.info(f"[{step}/{total_steps}] {message}")

        # 1. Build the base context
        context = self._build_context(
            simulation_requirement=simulation_requirement,
            document_text=document_text,
            entities=entities
        )

        reasoning_parts = []

        # ========== Step 1: Generate time config ==========
        report_progress(1, t('progress.generatingTimeConfig'))
        num_entities = len(entities)
        time_config_result = self._generate_time_config(context, num_entities)
        time_config = self._parse_time_config(time_config_result, num_entities)
        reasoning_parts.append(f"{t('progress.timeConfigLabel')}: {time_config_result.get('reasoning', t('common.success'))}")

        # ========== Step 2: Generate event config ==========
        report_progress(2, t('progress.generatingEventConfig'))
        event_config_result = self._generate_event_config(context, simulation_requirement, entities)
        event_config = self._parse_event_config(event_config_result)
        reasoning_parts.append(f"{t('progress.eventConfigLabel')}: {event_config_result.get('reasoning', t('common.success'))}")

        # ========== Steps 3..N: Agent configs (batched) ==========
        all_agent_configs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.AGENTS_PER_BATCH
            end_idx = min(start_idx + self.AGENTS_PER_BATCH, len(entities))
            batch_entities = entities[start_idx:end_idx]

            report_progress(
                3 + batch_idx,
                t('progress.generatingAgentConfig', start=start_idx + 1, end=end_idx, total=len(entities))
            )

            batch_configs = self._generate_agent_configs_batch(
                context=context,
                entities=batch_entities,
                start_idx=start_idx,
                simulation_requirement=simulation_requirement
            )
            all_agent_configs.extend(batch_configs)

        reasoning_parts.append(t('progress.agentConfigResult', count=len(all_agent_configs)))

        # ========== Assign poster agents to initial posts ==========
        logger.info("Assigning suitable poster agents to initial posts...")
        event_config = self._assign_initial_post_agents(event_config, all_agent_configs)
        assigned_count = len([p for p in event_config.initial_posts if p.get("poster_agent_id") is not None])
        reasoning_parts.append(t('progress.postAssignResult', count=assigned_count))

        # ========== Final step: Generate platform configs ==========
        report_progress(total_steps, t('progress.generatingPlatformConfig'))
        twitter_config = None
        reddit_config = None

        if enable_twitter:
            twitter_config = PlatformConfig(
                platform="twitter",
                recency_weight=0.4,
                popularity_weight=0.3,
                relevance_weight=0.3,
                viral_threshold=10,
                echo_chamber_strength=0.5
            )

        if enable_reddit:
            reddit_config = PlatformConfig(
                platform="reddit",
                recency_weight=0.3,
                popularity_weight=0.4,
                relevance_weight=0.3,
                viral_threshold=15,
                echo_chamber_strength=0.6
            )

        # Build the final parameter set
        params = SimulationParameters(
            simulation_id=simulation_id,
            project_id=project_id,
            graph_id=graph_id,
            simulation_requirement=simulation_requirement,
            time_config=time_config,
            agent_configs=all_agent_configs,
            event_config=event_config,
            twitter_config=twitter_config,
            reddit_config=reddit_config,
            llm_model=self.model_name,
            llm_base_url=self.base_url,
            generation_reasoning=" | ".join(reasoning_parts)
        )

        logger.info(f"Simulation config generated: {len(params.agent_configs)} agent configs")

        return params

    def _build_context(
        self,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode]
    ) -> str:
        """Build the LLM context, truncated to the maximum length.

        Note: this string is fed into LLM prompts, so headings are German.
        """

        # Entity summary
        entity_summary = self._summarize_entities(entities)

        # Build the context
        context_parts = [
            f"## Simulationsanforderung\n{simulation_requirement}",
            f"\n## Entitätsinformationen ({len(entities)} Stück)\n{entity_summary}",
        ]

        current_length = sum(len(p) for p in context_parts)
        remaining_length = self.MAX_CONTEXT_LENGTH - current_length - 500  # 500-char safety margin

        if remaining_length > 0 and document_text:
            doc_text = document_text[:remaining_length]
            if len(document_text) > remaining_length:
                doc_text += "\n...(Dokument gekürzt)"
            context_parts.append(f"\n## Originaldokument\n{doc_text}")

        return "\n".join(context_parts)

    def _summarize_entities(self, entities: List[EntityNode]) -> str:
        """Build a summary of entities (German text for LLM prompt consumption)."""
        lines = []

        # Group by type
        by_type: Dict[str, List[EntityNode]] = {}
        for e in entities:
            etype = e.get_entity_type() or "Unknown"
            if etype not in by_type:
                by_type[etype] = []
            by_type[etype].append(e)

        for entity_type, type_entities in by_type.items():
            lines.append(f"\n### {entity_type} ({len(type_entities)} Stück)")
            # Use the configured display count and summary length
            display_count = self.ENTITIES_PER_TYPE_DISPLAY
            summary_len = self.ENTITY_SUMMARY_LENGTH
            for e in type_entities[:display_count]:
                summary_preview = (e.summary[:summary_len] + "...") if len(e.summary) > summary_len else e.summary
                lines.append(f"- {e.name}: {summary_preview}")
            if len(type_entities) > display_count:
                lines.append(f"  ... weitere {len(type_entities) - display_count} Entitäten")

        return "\n".join(lines)

    def _call_llm_with_retry(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """Call the LLM with retry logic and JSON repair fallback"""
        import re

        max_attempts = 3
        last_error = None

        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7 - (attempt * 0.1)  # lower temperature each retry
                    # Don't set max_tokens — let the LLM expand freely
                )

                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason

                # Detect truncation
                if finish_reason == 'length':
                    logger.warning(f"LLM output truncated (attempt {attempt+1})")
                    content = self._fix_truncated_json(content)

                # Try parsing
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse failed (attempt {attempt+1}): {str(e)[:80]}")

                    # Try to repair the JSON
                    fixed = self._try_fix_config_json(content)
                    if fixed:
                        return fixed

                    last_error = e

            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt+1}): {str(e)[:80]}")
                last_error = e
                import time
                time.sleep(2 * (attempt + 1))

        raise last_error or Exception("LLM call failed")

    def _fix_truncated_json(self, content: str) -> str:
        """Repair truncated JSON"""
        content = content.strip()

        # Count unclosed brackets
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')

        # Detect unclosed strings
        if content and content[-1] not in '",}]':
            content += '"'

        # Close any remaining brackets
        content += ']' * open_brackets
        content += '}' * open_braces

        return content

    def _try_fix_config_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Attempt to repair config JSON"""
        import re

        # First try to repair truncation
        content = self._fix_truncated_json(content)

        # Extract the JSON portion
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group()

            # Strip newlines inside string values
            def fix_string(match):
                s = match.group(0)
                s = s.replace('\n', ' ').replace('\r', ' ')
                s = re.sub(r'\s+', ' ', s)
                return s

            json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string, json_str)

            try:
                return json.loads(json_str)
            except:
                # Try stripping all control chars
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
                json_str = re.sub(r'\s+', ' ', json_str)
                try:
                    return json.loads(json_str)
                except:
                    pass

        return None

    def _generate_time_config(self, context: str, num_entities: int) -> Dict[str, Any]:
        """Generate the time configuration"""
        # Use the configured truncation length
        context_truncated = context[:self.TIME_CONFIG_CONTEXT_LENGTH]

        # Compute the maximum allowed value (90% of agents)
        max_agents_allowed = max(1, int(num_entities * 0.9))

        prompt = f"""Erzeugen Sie auf Basis der folgenden Simulationsanforderung eine Zeit-Simulationskonfiguration.

{context_truncated}

## Aufgabe
Geben Sie die Zeitkonfiguration als JSON aus.

### Grundprinzipien (nur Referenz, an konkretes Ereignis und Zielgruppe anpassen):
- Leiten Sie aus dem Simulationsszenario die Zeitzone und die Tagesroutinen der Zielgruppe ab; das folgende Beispiel bezieht sich auf UTC+8
- Zwischen 0 und 5 Uhr nahezu keine Aktivität (Aktivitätsfaktor 0,05)
- Zwischen 6 und 8 Uhr beginnender Anstieg (Aktivitätsfaktor 0,4)
- Arbeitszeit 9-18 Uhr mittlere Aktivität (Aktivitätsfaktor 0,7)
- Abends 19-22 Uhr Hauptpeak (Aktivitätsfaktor 1,5)
- Ab 23 Uhr Rückgang (Aktivitätsfaktor 0,5)
- Faustregel: nachts niedrig, morgens steigend, tagsüber mittel, abends Spitze
- **Wichtig**: die folgenden Beispielwerte sind Referenz, passen Sie die konkreten Zeitfenster nach Ereignisart und Zielgruppe an
  - Beispiel: Studierende sind oft 21-23 Uhr aktiv; Medien sind ganztägig aktiv; Behörden nur in Arbeitszeit
  - Beispiel: bei kurzfristigen Hot-Topics gibt es auch nachts Diskussionen, off_peak_hours kann verkürzt werden

### Rückgabeformat JSON (kein Markdown)

Beispiel:
{{
    "total_simulation_hours": 72,
    "minutes_per_round": 60,
    "agents_per_hour_min": 5,
    "agents_per_hour_max": 50,
    "peak_hours": [19, 20, 21, 22],
    "off_peak_hours": [0, 1, 2, 3, 4, 5],
    "morning_hours": [6, 7, 8],
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    "reasoning": "Begründung der Zeitkonfiguration für dieses Ereignis"
}}

Felderläuterung:
- total_simulation_hours (int): Gesamtdauer in Stunden, 24-168, kurze Ereignisse kürzer, Dauerthemen länger
- minutes_per_round (int): Dauer einer Runde, 30-120 Minuten, empfohlen 60
- agents_per_hour_min (int): Mindestzahl pro Stunde aktivierter Agents (Bereich: 1-{max_agents_allowed})
- agents_per_hour_max (int): Maximalzahl pro Stunde aktivierter Agents (Bereich: 1-{max_agents_allowed})
- peak_hours (int-Array): Peakstunden, gemäß Zielgruppe anpassen
- off_peak_hours (int-Array): Tiefstunden, in der Regel Nacht/frühe Morgenstunden
- morning_hours (int-Array): Morgenstunden
- work_hours (int-Array): Arbeitsstunden
- reasoning (string): Kurzbegründung der Konfiguration"""

        system_prompt = "Sie sind Experte für Social-Media-Simulationen. Geben Sie reines JSON zurück; die Zeitkonfiguration muss zu den Tagesroutinen der Zielgruppe im Simulationsszenario passen."
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}"

        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"LLM time-config generation failed: {e}, using default config")
            return self._get_default_time_config(num_entities)

    def _get_default_time_config(self, num_entities: int) -> Dict[str, Any]:
        """Return the default time configuration (Chinese daily routine)"""
        return {
            "total_simulation_hours": 72,
            "minutes_per_round": 60,  # 1 hour per round, accelerates time
            "agents_per_hour_min": max(1, num_entities // 15),
            "agents_per_hour_max": max(5, num_entities // 5),
            "peak_hours": [19, 20, 21, 22],
            "off_peak_hours": [0, 1, 2, 3, 4, 5],
            "morning_hours": [6, 7, 8],
            "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            "reasoning": "Default Chinese daily-routine config (1 hour per round)"
        }

    def _parse_time_config(self, result: Dict[str, Any], num_entities: int) -> TimeSimulationConfig:
        """Parse the time-config result and validate that agents_per_hour does not exceed total agents"""
        # Read the raw values
        agents_per_hour_min = result.get("agents_per_hour_min", max(1, num_entities // 15))
        agents_per_hour_max = result.get("agents_per_hour_max", max(5, num_entities // 5))

        # Validate and clamp: ensure they don't exceed the total agent count
        if agents_per_hour_min > num_entities:
            logger.warning(f"agents_per_hour_min ({agents_per_hour_min}) exceeds total agents ({num_entities}); clamping")
            agents_per_hour_min = max(1, num_entities // 10)

        if agents_per_hour_max > num_entities:
            logger.warning(f"agents_per_hour_max ({agents_per_hour_max}) exceeds total agents ({num_entities}); clamping")
            agents_per_hour_max = max(agents_per_hour_min + 1, num_entities // 2)

        # Ensure min < max
        if agents_per_hour_min >= agents_per_hour_max:
            agents_per_hour_min = max(1, agents_per_hour_max // 2)
            logger.warning(f"agents_per_hour_min >= max; reset to {agents_per_hour_min}")

        return TimeSimulationConfig(
            total_simulation_hours=result.get("total_simulation_hours", 72),
            minutes_per_round=result.get("minutes_per_round", 60),  # default: 1 hour per round
            agents_per_hour_min=agents_per_hour_min,
            agents_per_hour_max=agents_per_hour_max,
            peak_hours=result.get("peak_hours", [19, 20, 21, 22]),
            off_peak_hours=result.get("off_peak_hours", [0, 1, 2, 3, 4, 5]),
            off_peak_activity_multiplier=0.05,  # nearly nobody around in early hours
            morning_hours=result.get("morning_hours", [6, 7, 8]),
            morning_activity_multiplier=0.4,
            work_hours=result.get("work_hours", list(range(9, 19))),
            work_activity_multiplier=0.7,
            peak_activity_multiplier=1.5
        )

    def _generate_event_config(
        self,
        context: str,
        simulation_requirement: str,
        entities: List[EntityNode]
    ) -> Dict[str, Any]:
        """Generate the event configuration"""

        # Collect available entity types as references for the LLM
        entity_types_available = list(set(
            e.get_entity_type() or "Unknown" for e in entities
        ))

        # List representative entity names per type
        type_examples = {}
        for e in entities:
            etype = e.get_entity_type() or "Unknown"
            if etype not in type_examples:
                type_examples[etype] = []
            if len(type_examples[etype]) < 3:
                type_examples[etype].append(e.name)

        type_info = "\n".join([
            f"- {etype}: {', '.join(examples)}"
            for etype, examples in type_examples.items()
        ])

        # Use the configured truncation length
        context_truncated = context[:self.EVENT_CONFIG_CONTEXT_LENGTH]

        prompt = f"""Erzeugen Sie auf Basis der folgenden Simulationsanforderung eine Ereigniskonfiguration.

Simulationsanforderung: {simulation_requirement}

{context_truncated}

## Verfügbare Entitätstypen und Beispiele
{type_info}

## Aufgabe
Geben Sie die Ereigniskonfiguration als JSON aus:
- Schlüsselwörter zu Hot-Topics extrahieren
- Entwicklungsrichtung der öffentlichen Meinung beschreiben
- Inhalte der Initialposts entwerfen, **für jeden Post zwingend poster_type (Posttyp/Verfassertyp) angeben**

**Wichtig**: poster_type muss aus den oben genannten "verfügbaren Entitätstypen" stammen, damit Initialposts geeigneten Agents zugewiesen werden können.
Beispiel: offizielle Erklärungen sollten von Official/University stammen, Nachrichten von MediaOutlet, studentische Meinungen von Student.

JSON-Format zurückgeben (kein Markdown):
{{
    "hot_topics": ["Schlüsselwort1", "Schlüsselwort2", ...],
    "narrative_direction": "<Beschreibung der Meinungsentwicklung>",
    "initial_posts": [
        {{"content": "Inhalt des Posts", "poster_type": "Entitätstyp (zwingend aus den verfügbaren Typen)"}},
        ...
    ],
    "reasoning": "<Kurzbegründung>"
}}"""

        system_prompt = "Sie sind Experte für Meinungsanalyse. Geben Sie reines JSON zurück. poster_type muss exakt zu den verfügbaren Entitätstypen passen."
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}\nIMPORTANT: The 'poster_type' field value MUST be in English PascalCase exactly matching the available entity types. Only 'content', 'narrative_direction', 'hot_topics' and 'reasoning' fields should use the specified language."

        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"LLM event-config generation failed: {e}, using default config")
            return {
                "hot_topics": [],
                "narrative_direction": "",
                "initial_posts": [],
                "reasoning": "Using default config"
            }

    def _parse_event_config(self, result: Dict[str, Any]) -> EventConfig:
        """Parse the event-config result"""
        return EventConfig(
            initial_posts=result.get("initial_posts", []),
            scheduled_events=[],
            hot_topics=result.get("hot_topics", []),
            narrative_direction=result.get("narrative_direction", "")
        )

    def _assign_initial_post_agents(
        self,
        event_config: EventConfig,
        agent_configs: List[AgentActivityConfig]
    ) -> EventConfig:
        """
        Assign suitable poster agents to initial posts.

        Matches each post's poster_type to the most fitting agent_id.
        """
        if not event_config.initial_posts:
            return event_config

        # Build an index of agents by entity type
        agents_by_type: Dict[str, List[AgentActivityConfig]] = {}
        for agent in agent_configs:
            etype = agent.entity_type.lower()
            if etype not in agents_by_type:
                agents_by_type[etype] = []
            agents_by_type[etype].append(agent)

        # Type alias map (covers different formats the LLM may emit)
        type_aliases = {
            "official": ["official", "university", "governmentagency", "government"],
            "university": ["university", "official"],
            "mediaoutlet": ["mediaoutlet", "media"],
            "student": ["student", "person"],
            "professor": ["professor", "expert", "teacher"],
            "alumni": ["alumni", "person"],
            "organization": ["organization", "ngo", "company", "group"],
            "person": ["person", "student", "alumni"],
        }

        # Track used agent indices per type to avoid reusing the same agent
        used_indices: Dict[str, int] = {}

        updated_posts = []
        for post in event_config.initial_posts:
            poster_type = post.get("poster_type", "").lower()
            content = post.get("content", "")

            # Try to find a matching agent
            matched_agent_id = None

            # 1. Direct match
            if poster_type in agents_by_type:
                agents = agents_by_type[poster_type]
                idx = used_indices.get(poster_type, 0) % len(agents)
                matched_agent_id = agents[idx].agent_id
                used_indices[poster_type] = idx + 1
            else:
                # 2. Alias match
                for alias_key, aliases in type_aliases.items():
                    if poster_type in aliases or alias_key == poster_type:
                        for alias in aliases:
                            if alias in agents_by_type:
                                agents = agents_by_type[alias]
                                idx = used_indices.get(alias, 0) % len(agents)
                                matched_agent_id = agents[idx].agent_id
                                used_indices[alias] = idx + 1
                                break
                    if matched_agent_id is not None:
                        break

            # 3. If still unmatched, fall back to the highest-influence agent
            if matched_agent_id is None:
                logger.warning(f"No matching agent for type '{poster_type}'; using highest-influence agent")
                if agent_configs:
                    # Sort by influence and pick the top one
                    sorted_agents = sorted(agent_configs, key=lambda a: a.influence_weight, reverse=True)
                    matched_agent_id = sorted_agents[0].agent_id
                else:
                    matched_agent_id = 0

            updated_posts.append({
                "content": content,
                "poster_type": post.get("poster_type", "Unknown"),
                "poster_agent_id": matched_agent_id
            })

            logger.info(f"Initial post assigned: poster_type='{poster_type}' -> agent_id={matched_agent_id}")

        event_config.initial_posts = updated_posts
        return event_config

    def _generate_agent_configs_batch(
        self,
        context: str,
        entities: List[EntityNode],
        start_idx: int,
        simulation_requirement: str
    ) -> List[AgentActivityConfig]:
        """Generate a batch of agent configurations"""

        # Build entity info (using the configured summary length)
        entity_list = []
        summary_len = self.AGENT_SUMMARY_LENGTH
        for i, e in enumerate(entities):
            entity_list.append({
                "agent_id": start_idx + i,
                "entity_name": e.name,
                "entity_type": e.get_entity_type() or "Unknown",
                "summary": e.summary[:summary_len] if e.summary else ""
            })

        prompt = f"""Erzeugen Sie auf Basis der folgenden Informationen für jede Entität eine Konfiguration für Social-Media-Aktivität.

Simulationsanforderung: {simulation_requirement}

## Entitätsliste
```json
{json.dumps(entity_list, ensure_ascii=False, indent=2)}
```

## Aufgabe
Erzeugen Sie pro Entität eine Aktivitätskonfiguration. Beachten Sie:
- **Zeitfenster passend zur Tagesroutine der Zielgruppe**: das Folgende ist Referenz (UTC+8), an das Szenario anpassen
- **Behörden/offizielle Stellen** (University/GovernmentAgency): niedrige Aktivität (0,1-0,3), tagsüber (9-17), langsame Reaktion (60-240 Min.), hoher Einfluss (2,5-3,0)
- **Medien** (MediaOutlet): mittlere Aktivität (0,4-0,6), ganztägig aktiv (8-23), schnelle Reaktion (5-30 Min.), hoher Einfluss (2,0-2,5)
- **Privatpersonen** (Student/Person/Alumni): hohe Aktivität (0,6-0,9), schwerpunktmäßig abends (18-23), schnelle Reaktion (1-15 Min.), niedriger Einfluss (0,8-1,2)
- **Personen des öffentlichen Lebens / Fachleute**: mittlere Aktivität (0,4-0,6), mittelhoher Einfluss (1,5-2,0)

JSON-Format zurückgeben (kein Markdown):
{{
    "agent_configs": [
        {{
            "agent_id": <muss exakt mit Eingabe übereinstimmen>,
            "activity_level": <0.0-1.0>,
            "posts_per_hour": <Posting-Frequenz>,
            "comments_per_hour": <Kommentar-Frequenz>,
            "active_hours": [<Liste aktiver Stunden, an Zielgruppe orientiert>],
            "response_delay_min": <Mindest-Reaktionsdelay in Minuten>,
            "response_delay_max": <Maximal-Reaktionsdelay in Minuten>,
            "sentiment_bias": <-1.0 bis 1.0>,
            "stance": "<supportive/opposing/neutral/observer>",
            "influence_weight": <Einflussgewicht>
        }},
        ...
    ]
}}"""

        system_prompt = "Sie sind Experte für die Verhaltensanalyse in sozialen Medien. Geben Sie reines JSON zurück; die Konfiguration muss zur Tagesroutine der Zielgruppe im Simulationsszenario passen."
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}\nIMPORTANT: The 'stance' field value MUST be one of the English strings: 'supportive', 'opposing', 'neutral', 'observer'. All JSON field names and numeric values must remain unchanged. Only natural language text fields should use the specified language."

        try:
            result = self._call_llm_with_retry(prompt, system_prompt)
            llm_configs = {cfg["agent_id"]: cfg for cfg in result.get("agent_configs", [])}
        except Exception as e:
            logger.warning(f"LLM agent-config batch generation failed: {e}, falling back to rule-based generation")
            llm_configs = {}

        # Build AgentActivityConfig instances
        configs = []
        for i, entity in enumerate(entities):
            agent_id = start_idx + i
            cfg = llm_configs.get(agent_id, {})

            # Fall back to rule-based generation when the LLM has nothing
            if not cfg:
                cfg = self._generate_agent_config_by_rule(entity)

            config = AgentActivityConfig(
                agent_id=agent_id,
                entity_uuid=entity.uuid,
                entity_name=entity.name,
                entity_type=entity.get_entity_type() or "Unknown",
                activity_level=cfg.get("activity_level", 0.5),
                posts_per_hour=cfg.get("posts_per_hour", 0.5),
                comments_per_hour=cfg.get("comments_per_hour", 1.0),
                active_hours=cfg.get("active_hours", list(range(9, 23))),
                response_delay_min=cfg.get("response_delay_min", 5),
                response_delay_max=cfg.get("response_delay_max", 60),
                sentiment_bias=cfg.get("sentiment_bias", 0.0),
                stance=cfg.get("stance", "neutral"),
                influence_weight=cfg.get("influence_weight", 1.0)
            )
            configs.append(config)

        return configs

    def _generate_agent_config_by_rule(self, entity: EntityNode) -> Dict[str, Any]:
        """Rule-based per-agent configuration (Chinese daily routine)"""
        entity_type = (entity.get_entity_type() or "Unknown").lower()

        if entity_type in ["university", "governmentagency", "ngo"]:
            # Official institutions: working hours, low frequency, high influence
            return {
                "activity_level": 0.2,
                "posts_per_hour": 0.1,
                "comments_per_hour": 0.05,
                "active_hours": list(range(9, 18)),  # 9:00-17:59
                "response_delay_min": 60,
                "response_delay_max": 240,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 3.0
            }
        elif entity_type in ["mediaoutlet"]:
            # Media: all-day, medium frequency, high influence
            return {
                "activity_level": 0.5,
                "posts_per_hour": 0.8,
                "comments_per_hour": 0.3,
                "active_hours": list(range(7, 24)),  # 7:00-23:59
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "observer",
                "influence_weight": 2.5
            }
        elif entity_type in ["professor", "expert", "official"]:
            # Experts/professors: workday + evening, medium frequency
            return {
                "activity_level": 0.4,
                "posts_per_hour": 0.3,
                "comments_per_hour": 0.5,
                "active_hours": list(range(8, 22)),  # 8:00-21:59
                "response_delay_min": 15,
                "response_delay_max": 90,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 2.0
            }
        elif entity_type in ["student"]:
            # Students: mostly evenings, high frequency
            return {
                "activity_level": 0.8,
                "posts_per_hour": 0.6,
                "comments_per_hour": 1.5,
                "active_hours": [8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23],  # morning + evening
                "response_delay_min": 1,
                "response_delay_max": 15,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 0.8
            }
        elif entity_type in ["alumni"]:
            # Alumni: mostly evening
            return {
                "activity_level": 0.6,
                "posts_per_hour": 0.4,
                "comments_per_hour": 0.8,
                "active_hours": [12, 13, 19, 20, 21, 22, 23],  # midday + evening
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0
            }
        else:
            # Generic individuals: evening peak
            return {
                "activity_level": 0.7,
                "posts_per_hour": 0.5,
                "comments_per_hour": 1.2,
                "active_hours": [9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23],  # daytime + evening
                "response_delay_min": 2,
                "response_delay_max": 20,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0
            }
