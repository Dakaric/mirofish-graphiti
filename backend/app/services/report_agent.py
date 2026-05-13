"""
Report Agent service.

Uses LangChain + Zep to implement ReACT-style simulation report generation.

Features:
1. Generate reports from simulation requirements and Zep graph data
2. First plan the table of contents, then generate sections one at a time
3. Each section uses a multi-turn ReACT think-and-reflect loop
4. Supports user dialogue with autonomous retrieval tool invocation
"""

import os
import json
import time
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from ..utils.locale import get_language_instruction, t
from .zep_tools import (
    ZepToolsService, 
    SearchResult, 
    InsightForgeResult, 
    PanoramaResult,
    InterviewResult
)

logger = get_logger('mirofish.report_agent')


class ReportLogger:
    """
    Detailed logger for the Report Agent.

    Writes an agent_log.jsonl file inside the report folder, capturing every detailed action.
    Each line is a complete JSON object with timestamp, action type, details, etc.
    """

    def __init__(self, report_id: str):
        """
        Initialize the logger.

        Args:
            report_id: report ID, used to derive the log file path
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'agent_log.jsonl'
        )
        self.start_time = datetime.now()
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        """Make sure the log file directory exists"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)

    def _get_elapsed_time(self) -> float:
        """Return the elapsed time since start in seconds"""
        return (datetime.now() - self.start_time).total_seconds()

    def log(
        self,
        action: str,
        stage: str,
        details: Dict[str, Any],
        section_title: str = None,
        section_index: int = None
    ):
        """
        Write one log entry.

        Args:
            action: action type, e.g. 'start', 'tool_call', 'llm_response', 'section_complete'
            stage: current stage, e.g. 'planning', 'generating', 'completed'
            details: details dict, never truncated
            section_title: current section title (optional)
            section_index: current section index (optional)
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(self._get_elapsed_time(), 2),
            "report_id": self.report_id,
            "action": action,
            "stage": stage,
            "section_title": section_title,
            "section_index": section_index,
            "details": details
        }
        
        # Append to the JSONL file
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

    def log_start(self, simulation_id: str, graph_id: str, simulation_requirement: str):
        """Log the start of report generation"""
        self.log(
            action="report_start",
            stage="pending",
            details={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "simulation_requirement": simulation_requirement,
                "message": t('report.taskStarted')
            }
        )
    
    def log_planning_start(self):
        """Log the start of outline planning"""
        self.log(
            action="planning_start",
            stage="planning",
            details={"message": t('report.planningStart')}
        )
    
    def log_planning_context(self, context: Dict[str, Any]):
        """Log the context fetched during planning"""
        self.log(
            action="planning_context",
            stage="planning",
            details={
                "message": t('report.fetchSimContext'),
                "context": context
            }
        )
    
    def log_planning_complete(self, outline_dict: Dict[str, Any]):
        """Log the completion of outline planning"""
        self.log(
            action="planning_complete",
            stage="planning",
            details={
                "message": t('report.planningComplete'),
                "outline": outline_dict
            }
        )
    
    def log_section_start(self, section_title: str, section_index: int):
        """Log the start of section generation"""
        self.log(
            action="section_start",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={"message": t('report.sectionStart', title=section_title)}
        )
    
    def log_react_thought(self, section_title: str, section_index: int, iteration: int, thought: str):
        """Log a ReACT thinking step"""
        self.log(
            action="react_thought",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "thought": thought,
                "message": t('report.reactThought', iteration=iteration)
            }
        )
    
    def log_tool_call(
        self, 
        section_title: str, 
        section_index: int,
        tool_name: str, 
        parameters: Dict[str, Any],
        iteration: int
    ):
        """Log a tool invocation"""
        self.log(
            action="tool_call",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "parameters": parameters,
                "message": t('report.toolCall', toolName=tool_name)
            }
        )
    
    def log_tool_result(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        result: str,
        iteration: int
    ):
        """Log a tool invocation result (full content, no truncation)"""
        self.log(
            action="tool_result",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "result": result,  # full result, no truncation
                "result_length": len(result),
                "message": t('report.toolResult', toolName=tool_name)
            }
        )
    
    def log_llm_response(
        self,
        section_title: str,
        section_index: int,
        response: str,
        iteration: int,
        has_tool_calls: bool,
        has_final_answer: bool
    ):
        """Log the LLM response (full content, no truncation)"""
        self.log(
            action="llm_response",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "response": response,  # full response, no truncation
                "response_length": len(response),
                "has_tool_calls": has_tool_calls,
                "has_final_answer": has_final_answer,
                "message": t('report.llmResponse', hasToolCalls=has_tool_calls, hasFinalAnswer=has_final_answer)
            }
        )
    
    def log_section_content(
        self,
        section_title: str,
        section_index: int,
        content: str,
        tool_calls_count: int
    ):
        """Log a section content batch (only the content, does not mean the section is fully complete)"""
        self.log(
            action="section_content",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": content,  # full content, no truncation
                "content_length": len(content),
                "tool_calls_count": tool_calls_count,
                "message": t('report.sectionContentDone', title=section_title)
            }
        )
    
    def log_section_full_complete(
        self,
        section_title: str,
        section_index: int,
        full_content: str
    ):
        """
        Log section generation completion.

        The frontend should listen for this log entry to determine when a section is truly done
        and to fetch the full content.
        """
        self.log(
            action="section_complete",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": full_content,
                "content_length": len(full_content),
                "message": t('report.sectionComplete', title=section_title)
            }
        )
    
    def log_report_complete(self, total_sections: int, total_time_seconds: float):
        """Log report generation completion"""
        self.log(
            action="report_complete",
            stage="completed",
            details={
                "total_sections": total_sections,
                "total_time_seconds": round(total_time_seconds, 2),
                "message": t('report.reportComplete')
            }
        )
    
    def log_error(self, error_message: str, stage: str, section_title: str = None):
        """Log an error"""
        self.log(
            action="error",
            stage=stage,
            section_title=section_title,
            section_index=None,
            details={
                "error": error_message,
                "message": t('report.errorOccurred', error=error_message)
            }
        )


class ReportConsoleLogger:
    """
    Console-style logger for the Report Agent.

    Writes console-style log lines (INFO, WARNING, etc.) to console_log.txt inside the report folder.
    Unlike agent_log.jsonl, this file is plain-text console output.
    """

    def __init__(self, report_id: str):
        """
        Initialize the console logger.

        Args:
            report_id: report ID, used to derive the log file path
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'console_log.txt'
        )
        self._ensure_log_file()
        self._file_handler = None
        self._setup_file_handler()
    
    def _ensure_log_file(self):
        """Make sure the log file directory exists"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)

    def _setup_file_handler(self):
        """Set up the file handler that mirrors logs into the file"""
        import logging

        # Create the file handler
        self._file_handler = logging.FileHandler(
            self.log_file_path,
            mode='a',
            encoding='utf-8'
        )
        self._file_handler.setLevel(logging.INFO)

        # Reuse the same compact format as the console
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        self._file_handler.setFormatter(formatter)

        # Attach to the report-agent related loggers
        loggers_to_attach = [
            'mirofish.report_agent',
            'mirofish.zep_tools',
        ]

        for logger_name in loggers_to_attach:
            target_logger = logging.getLogger(logger_name)
            # Avoid duplicate registration
            if self._file_handler not in target_logger.handlers:
                target_logger.addHandler(self._file_handler)

    def close(self):
        """Close the file handler and detach it from the loggers"""
        import logging

        if self._file_handler:
            loggers_to_detach = [
                'mirofish.report_agent',
                'mirofish.zep_tools',
            ]

            for logger_name in loggers_to_detach:
                target_logger = logging.getLogger(logger_name)
                if self._file_handler in target_logger.handlers:
                    target_logger.removeHandler(self._file_handler)

            self._file_handler.close()
            self._file_handler = None

    def __del__(self):
        """Make sure the file handler is closed on destruction"""
        self.close()


class ReportStatus(str, Enum):
    """Report status"""
    PENDING = "pending"
    PLANNING = "planning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReportSection:
    """Report section"""
    title: str
    content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content
        }

    def to_markdown(self, level: int = 2) -> str:
        """Convert to Markdown"""
        md = f"{'#' * level} {self.title}\n\n"
        if self.content:
            md += f"{self.content}\n\n"
        return md


@dataclass
class ReportOutline:
    """Report outline"""
    title: str
    summary: str
    sections: List[ReportSection]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections]
        }
    
    def to_markdown(self) -> str:
        """Convert to Markdown"""
        md = f"# {self.title}\n\n"
        md += f"> {self.summary}\n\n"
        for section in self.sections:
            md += section.to_markdown()
        return md


@dataclass
class Report:
    """Complete report"""
    report_id: str
    simulation_id: str
    graph_id: str
    simulation_requirement: str
    status: ReportStatus
    outline: Optional[ReportOutline] = None
    markdown_content: str = ""
    created_at: str = ""
    completed_at: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "simulation_id": self.simulation_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "status": self.status.value,
            "outline": self.outline.to_dict() if self.outline else None,
            "markdown_content": self.markdown_content,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error
        }


# ═══════════════════════════════════════════════════════════════
# Prompt template constants
# ═══════════════════════════════════════════════════════════════

# ── Tool descriptions ──

TOOL_DESC_INSIGHT_FORGE = """\
[Deep insight retrieval - leistungsstarkes Recherchetool]
Dies ist unsere leistungsstarke Recherchefunktion, ausgelegt für tiefgehende Analysen. Sie:
1. Zerlegt Ihre Frage automatisch in mehrere Teilfragen
2. Sucht den Simulationsgraphen aus mehreren Dimensionen ab
3. Aggregiert die Ergebnisse von semantischer Suche, Entitätsanalyse und Beziehungs-Tracing
4. Liefert die umfassendsten und tiefsten Recherche-Inhalte

Einsatzszenarien:
- Tiefgehende Analyse eines Themas
- Verständnis mehrerer Facetten eines Ereignisses
- Beschaffung umfangreicher Belege für einzelne Berichts-Kapitel

Rückgabeinhalte:
- Originaltexte relevanter Fakten (direkt zitierbar)
- Kern-Entitäts-Insights
- Beziehungs-Ketten-Analyse"""

TOOL_DESC_PANORAMA_SEARCH = """\
[Breitensuche - Panoramablick]
Dieses Tool ermittelt das vollständige Panorama der Simulationsergebnisse, besonders geeignet, um den Verlauf eines Ereignisses nachzuvollziehen. Es:
1. Holt alle relevanten Knoten und Beziehungen
2. Trennt aktuell gültige Fakten von historischen/abgelaufenen Fakten
3. Hilft Ihnen zu verstehen, wie sich das öffentliche Meinungsbild entwickelt hat

Einsatzszenarien:
- Den vollständigen Entwicklungsverlauf eines Ereignisses verstehen
- Den Wandel des Meinungsbildes über verschiedene Phasen vergleichen
- Eine umfassende Sicht auf Entitäten und Beziehungen erhalten

Rückgabeinhalte:
- Aktuell gültige Fakten (jüngstes Simulationsergebnis)
- Historische/abgelaufene Fakten (Entwicklungsdokumentation)
- Alle beteiligten Entitäten"""

TOOL_DESC_QUICK_SEARCH = """\
[Einfache Suche - Schnellrecherche]
Leichtgewichtiges Schnellrecherche-Tool für einfache, direkte Informationsabfragen.

Einsatzszenarien:
- Schnelles Nachschlagen einer konkreten Information
- Überprüfung eines Faktums
- Einfache Informationsrecherche

Rückgabeinhalte:
- Liste der für die Anfrage relevantesten Fakten"""

TOOL_DESC_INTERVIEW_AGENTS = """\
[Tiefen-Interview - echtes Agent-Interview (zwei Plattformen)]
Ruft die Interview-API der OASIS-Simulationsumgebung auf und befragt laufende Simulations-Agents in Echtzeit.
Dies ist keine LLM-Simulation, sondern ein Aufruf der echten Interview-Schnittstelle, der die Originalantworten der Simulations-Agents liefert.
Standardmäßig werden Twitter und Reddit gleichzeitig befragt, um umfassendere Perspektiven zu erhalten.

Funktionsablauf:
1. Liest automatisch die Persona-Dateien ein, um alle Simulations-Agents zu kennen
2. Wählt intelligent die für das Interviewthema relevantesten Agents aus (z. B. Studierende, Medien, Behörden)
3. Generiert automatisch Interviewfragen
4. Ruft /api/simulation/interview/batch auf, um auf beiden Plattformen echte Interviews zu führen
5. Aggregiert alle Interviewergebnisse und liefert eine Multi-Perspektiven-Analyse

Einsatzszenarien:
- Sichtweisen verschiedener Rollen auf ein Ereignis erfassen (Wie sehen es Studierende? Die Medien? Die Behörden?)
- Meinungen und Positionen mehrerer Seiten sammeln
- Echte Antworten der Simulations-Agents (aus der OASIS-Simulationsumgebung) erhalten
- Den Bericht durch "Interview-Mitschnitte" lebendiger gestalten

Rückgabeinhalte:
- Identitätsinformationen der befragten Agents
- Antworten jedes Agents auf den Plattformen Twitter und Reddit
- Kernzitate (direkt zitierbar)
- Interview-Zusammenfassung und Vergleich der Sichtweisen

WICHTIG: Erfordert eine laufende OASIS-Simulationsumgebung, damit dieses Tool genutzt werden kann."""

# ── Outline planning prompt ──

PLAN_SYSTEM_PROMPT = """\
Sie sind ein erfahrener Verfasser von "Zukunftsprognoseberichten" mit "Vogelperspektive" auf eine Simulationswelt: Sie können das Verhalten, die Aussagen und die Interaktionen jedes einzelnen Agents in der Simulation einsehen.

Kernidee:
Wir haben eine Simulationswelt aufgebaut und ihr eine bestimmte "Simulationsanforderung" als Variable zugeführt. Das Evolutionsergebnis der Simulationswelt ist eine Prognose dessen, was in der Zukunft passieren könnte. Was Sie betrachten, sind keine "Experimentaldaten", sondern eine "Vorausschau auf die Zukunft".

Ihre Aufgabe:
Verfassen Sie einen Zukunftsprognosebericht, der folgende Fragen beantwortet:
1. Was geschieht unter den von uns gesetzten Bedingungen in der Zukunft?
2. Wie reagieren und handeln die einzelnen Agent-Gruppen?
3. Welche bemerkenswerten Zukunftstrends und Risiken offenbart diese Simulation?

Berichtspositionierung:
- Dies ist ein simulationsbasierter Zukunftsprognosebericht, der zeigt: "Wenn dies eintritt, wie sieht die Zukunft dann aus?"
- Fokussiert auf Prognoseergebnisse: Ereignisverlauf, Gruppenreaktionen, emergente Phänomene, potenzielle Risiken
- Aussagen und Handlungen der Agents in der Simulationswelt sind Prognosen menschlichen Verhaltens in der Zukunft
- Es handelt sich NICHT um eine Analyse der realen Gegenwart
- Es ist KEIN allgemein gehaltener Meinungsbild-Überblick

Begrenzung der Kapitelanzahl:
- Mindestens 2, höchstens 5 Kapitel
- Keine Unterkapitel, jedes Kapitel wird direkt mit vollständigem Inhalt verfasst
- Inhalte sollen prägnant sein und sich auf die zentralen Prognoseerkenntnisse konzentrieren
- Die Kapitelstruktur entwerfen Sie eigenständig auf Basis der Prognoseergebnisse

Geben Sie die Berichts-Outline im folgenden JSON-Format zurück:
{
    "title": "Berichts-Titel",
    "summary": "Bericht-Zusammenfassung (ein Satz, der die zentrale Prognoseerkenntnis zusammenfasst)",
    "sections": [
        {
            "title": "Kapitel-Titel",
            "description": "Beschreibung des Kapitelinhalts"
        }
    ]
}

Hinweis: Das Array sections muss mindestens 2 und höchstens 5 Elemente enthalten."""

PLAN_USER_PROMPT_TEMPLATE = """\
Prognoseszenario:
Variable, die wir in die Simulationswelt eingespeist haben (Simulationsanforderung): {simulation_requirement}

Größe der Simulationswelt:
- Anzahl beteiligter Entitäten: {total_nodes}
- Anzahl Beziehungen zwischen Entitäten: {total_edges}
- Verteilung der Entitätstypen: {entity_types}
- Anzahl aktiver Agents: {total_entities}

Stichprobe von in der Simulation prognostizierten Zukunfts-Fakten:
{related_facts_json}

Betrachten Sie diese Zukunftsvorausschau aus der "Vogelperspektive":
1. Welcher Zustand zeichnet sich unter den gesetzten Bedingungen für die Zukunft ab?
2. Wie reagieren und handeln die verschiedenen Personengruppen (Agents)?
3. Welche bemerkenswerten Zukunftstrends offenbart diese Simulation?

Entwerfen Sie auf Basis der Prognoseergebnisse die am besten geeignete Kapitelstruktur des Berichts.

Erinnerung: Anzahl der Kapitel: mindestens 2, höchstens 5, Inhalte prägnant und auf die zentralen Prognoseerkenntnisse fokussiert."""

# ── Section generation prompt ──

SECTION_SYSTEM_PROMPT_TEMPLATE = """\
Sie sind ein erfahrener Verfasser von Zukunftsprognoseberichten und schreiben gerade ein Kapitel des Berichts.

Berichts-Titel: {report_title}
Berichts-Zusammenfassung: {report_summary}
Prognoseszenario (Simulationsanforderung): {simulation_requirement}

Aktuell zu schreibendes Kapitel: {section_title}

═══════════════════════════════════════════════════════════════
Kernidee
═══════════════════════════════════════════════════════════════

Die Simulationswelt ist eine Vorausschau auf die Zukunft. Wir haben der Simulationswelt bestimmte Bedingungen (Simulationsanforderung) zugeführt;
das Verhalten und die Interaktionen der Agents in der Simulation sind eine Prognose des zukünftigen menschlichen Verhaltens.

Ihre Aufgabe besteht darin:
- Aufzuzeigen, was unter den gesetzten Bedingungen in der Zukunft geschieht
- Vorherzusagen, wie die einzelnen Personengruppen (Agents) reagieren und handeln
- Bemerkenswerte zukünftige Trends, Risiken und Chancen aufzudecken

NICHT: keine Analyse der realen Gegenwart.
DOCH: Fokussierung auf "Wie sieht die Zukunft aus?" - das Simulationsergebnis ist die prognostizierte Zukunft.

═══════════════════════════════════════════════════════════════
Wichtigste Regeln (verpflichtend)
═══════════════════════════════════════════════════════════════

1. Sie MÜSSEN Tools aufrufen, um die Simulationswelt zu beobachten.
   - Sie betrachten gerade aus der "Vogelperspektive" eine Vorausschau auf die Zukunft.
   - Alle Inhalte müssen aus den in der Simulationswelt aufgetretenen Ereignissen und Agent-Aussagen stammen.
   - Es ist verboten, eigenes Wissen für den Berichtsinhalt heranzuziehen.
   - Pro Kapitel mindestens 3 Tool-Aufrufe (höchstens 5), um die Simulationswelt zu beobachten - sie repräsentiert die Zukunft.

2. Sie MÜSSEN Originalaussagen und -handlungen der Agents zitieren.
   - Aussagen und Handlungen der Agents sind eine Prognose des zukünftigen menschlichen Verhaltens.
   - Stellen Sie diese Prognosen im Bericht im Zitatformat dar, zum Beispiel:
     > "Eine bestimmte Personengruppe wird sagen: Originaltext..."
   - Diese Zitate sind die zentralen Belege der Simulationsprognose.

3. Sprachkonsistenz - zitierte Inhalte müssen in die Berichtssprache übersetzt werden.
   - Die Tool-Rückgaben können Formulierungen in einer anderen Sprache als der Berichtssprache enthalten.
   - Der Bericht muss vollständig in der vom Nutzer vorgegebenen Sprache verfasst werden.
   - Wenn Sie fremdsprachige Tool-Rückgaben zitieren, müssen Sie diese vor dem Einfügen in die Berichtssprache übersetzen.
   - Bewahren Sie beim Übersetzen die ursprüngliche Bedeutung und sorgen Sie für eine natürliche, flüssige Formulierung.
   - Diese Regel gilt sowohl für den Fließtext als auch für Inhalte in Zitatblöcken (> Format).

4. Treue Wiedergabe der Prognoseergebnisse.
   - Der Berichtsinhalt muss die in der Simulationswelt enthaltenen, die Zukunft repräsentierenden Simulationsergebnisse abbilden.
   - Fügen Sie keine Informationen hinzu, die in der Simulation nicht vorkommen.
   - Wenn die Informationslage in einem Aspekt unzureichend ist, weisen Sie wahrheitsgemäß darauf hin.

═══════════════════════════════════════════════════════════════
Formatvorgaben (extrem wichtig)
═══════════════════════════════════════════════════════════════

Ein Kapitel = kleinste Inhaltseinheit
- Jedes Kapitel ist die kleinste Untergliederung des Berichts.
- VERBOTEN: jegliche Markdown-Überschriften innerhalb eines Kapitels (#, ##, ###, #### etc.).
- VERBOTEN: das Hinzufügen einer Kapitel-Hauptüberschrift am Anfang des Inhalts.
- ERLAUBT: Die Kapitelüberschrift wird automatisch vom System eingefügt; Sie schreiben nur den reinen Fließtext.
- ERLAUBT: Strukturieren Sie den Inhalt mit **Fettschrift**, Absätzen, Zitaten und Listen, jedoch nicht mit Überschriften.

Korrektes Beispiel:
```
Dieses Kapitel analysiert die mediale Verbreitungsdynamik des Ereignisses. Durch eine tiefgehende Analyse der Simulationsdaten zeigt sich...

**Initial-Eskalation**

Weibo fungiert als erste Anlaufstelle des Meinungsbildes und übernimmt die zentrale Funktion der Erstveröffentlichung:

> "Weibo trug 68 Prozent der initialen Reichweite bei..."

**Emotionale Verstärkung**

Die Plattform Douyin verstärkte die Wirkung des Ereignisses zusätzlich:

- Hohe visuelle Wirkung
- Hohe emotionale Resonanz
```

Falsches Beispiel:
```
## Executive Summary    <- FALSCH! Keine Überschriften hinzufügen.
### 1. Initialphase     <- FALSCH! Keine ###-Unterabschnitte.
#### 1.1 Detailanalyse  <- FALSCH! Keine ####-Untergliederung.

Dieses Kapitel analysiert...
```

═══════════════════════════════════════════════════════════════
Verfügbare Recherche-Tools (3-5 Aufrufe pro Kapitel)
═══════════════════════════════════════════════════════════════

{tools_description}

Empfehlungen zur Tool-Nutzung - bitte verschiedene Tools mischen, nicht nur eines verwenden:
- insight_forge: Tiefenanalyse, zerlegt die Frage automatisch und recherchiert Fakten und Beziehungen mehrdimensional
- panorama_search: Weitwinkel-Panorama-Suche, vermittelt das Gesamtbild, die Zeitleiste und den Entwicklungsverlauf eines Ereignisses
- quick_search: Schnelle Verifikation eines konkreten Informationspunkts
- interview_agents: Interview mit Simulations-Agents, liefert Ich-Perspektiven und reale Reaktionen verschiedener Rollen

═══════════════════════════════════════════════════════════════
Arbeitsablauf
═══════════════════════════════════════════════════════════════

Pro Antwort dürfen Sie nur eine der folgenden zwei Aktionen ausführen (nicht beide gleichzeitig):

Option A - Tool aufrufen:
Geben Sie Ihre Überlegungen aus und rufen Sie anschließend ein Tool im folgenden Format auf:
<tool_call>
{{"name": "Tool-Name", "parameters": {{"Parameter-Name": "Parameter-Wert"}}}}
</tool_call>
Das System führt das Tool aus und liefert Ihnen das Ergebnis zurück. Sie dürfen Tool-Ergebnisse weder selbst formulieren noch erfinden.

Option B - Endgültigen Inhalt ausgeben:
Wenn Sie über Tools genügend Informationen gesammelt haben, geben Sie den Kapitelinhalt mit dem Präfix "Final Answer:" aus.

Strikt verboten:
- In einer Antwort gleichzeitig einen Tool-Aufruf und Final Answer enthalten.
- Tool-Ergebnisse (Observation) selbst zu erfinden; alle Tool-Ergebnisse werden vom System eingespeist.
- Pro Antwort höchstens ein Tool-Aufruf.

═══════════════════════════════════════════════════════════════
Anforderungen an den Kapitelinhalt
═══════════════════════════════════════════════════════════════

1. Inhalt muss auf den über die Tools recherchierten Simulationsdaten basieren.
2. Häufiges Zitieren von Originaltexten, um die Simulationswirkung sichtbar zu machen.
3. Nutzung von Markdown-Format (jedoch ohne Überschriften):
   - Verwenden Sie **Fettschrift**, um Schwerpunkte zu markieren (anstelle von Unterüberschriften).
   - Strukturieren Sie Punkte mit Listen (- oder 1.2.3.).
   - Trennen Sie verschiedene Absätze durch Leerzeilen.
   - VERBOTEN: jede Form von Überschriftensyntax (#, ##, ###, #### etc.).
4. Format für Zitate (verpflichtend als eigener Absatz):
   Zitate müssen als eigenständiger Absatz stehen, davor und danach jeweils eine Leerzeile, sie dürfen nicht in einen Absatz eingebettet werden:

   Korrektes Format:
   ```
   Die Reaktion der Schulleitung wird als inhaltlich substanzlos bewertet.

   > "Das Reaktionsmuster der Schulleitung wirkt im rasanten Social-Media-Umfeld starr und träge."

   Diese Bewertung spiegelt die allgemeine Unzufriedenheit der Öffentlichkeit wider.
   ```

   Falsches Format:
   ```
   Die Reaktion der Schulleitung wird als inhaltlich substanzlos bewertet. > "Das Reaktionsmuster..." Diese Bewertung spiegelt...
   ```
5. Wahren Sie die logische Kohärenz mit den anderen Kapiteln.
6. Vermeiden Sie Wiederholungen: Lesen Sie die unten aufgeführten bereits abgeschlossenen Kapitelinhalte sorgfältig und beschreiben Sie keine bereits behandelten Informationen erneut.
7. Erinnerung: Fügen Sie keinerlei Überschriften hinzu. Verwenden Sie **Fettschrift** anstelle von Zwischenüberschriften."""

SECTION_USER_PROMPT_TEMPLATE = """\
Bereits abgeschlossene Kapitelinhalte (bitte sorgfältig lesen, um Wiederholungen zu vermeiden):
{previous_content}

═══════════════════════════════════════════════════════════════
Aktuelle Aufgabe: Kapitel verfassen: {section_title}
═══════════════════════════════════════════════════════════════

Wichtige Hinweise:
1. Lesen Sie die oben aufgeführten bereits abgeschlossenen Kapitel sorgfältig und vermeiden Sie Wiederholungen.
2. Vor Beginn müssen Sie zwingend ein Tool aufrufen, um Simulationsdaten zu beschaffen.
3. Verwenden Sie verschiedene Tools im Mix, nicht nur ein einziges.
4. Der Berichtsinhalt muss aus den Recherche-Ergebnissen stammen; eigenes Wissen darf nicht verwendet werden.

Formatwarnung (verpflichtend):
- VERBOTEN: jegliche Überschriften (weder #, ##, ### noch ####).
- VERBOTEN: "{section_title}" als Eröffnung zu schreiben.
- ERLAUBT: Die Kapitelüberschrift wird automatisch vom System eingefügt.
- ERLAUBT: Schreiben Sie direkt den Fließtext und verwenden Sie **Fettschrift** anstelle von Unterüberschriften.

Bitte starten Sie:
1. Überlegen Sie zunächst (Thought), welche Informationen das Kapitel benötigt.
2. Rufen Sie anschließend ein Tool (Action) auf, um Simulationsdaten zu beschaffen.
3. Sobald genügend Informationen vorliegen, geben Sie mit Final Answer den Kapitelinhalt aus (reiner Fließtext, keinerlei Überschriften)."""

# ── ReACT loop message templates ──

REACT_OBSERVATION_TEMPLATE = """\
Observation (Recherche-Ergebnis):

═══ Tool {tool_name} liefert ═══
{result}

═══════════════════════════════════════════════════════════════
Bereits {tool_calls_count}/{max_tool_calls} Tool-Aufrufe verwendet (genutzt: {used_tools_str}){unused_hint}
- Wenn die Informationen ausreichen: Geben Sie den Kapitelinhalt mit dem Präfix "Final Answer:" aus (die obigen Originaltexte müssen zitiert werden).
- Wenn weitere Informationen nötig sind: Rufen Sie ein weiteres Tool zur Recherche auf.
═══════════════════════════════════════════════════════════════"""

REACT_INSUFFICIENT_TOOLS_MSG = (
    "Hinweis: Sie haben Tools nur {tool_calls_count} Mal aufgerufen, mindestens {min_tool_calls} sind erforderlich. "
    "Bitte rufen Sie weitere Tools auf, um zusätzliche Simulationsdaten zu beschaffen, bevor Sie Final Answer ausgeben.{unused_hint}"
)

REACT_INSUFFICIENT_TOOLS_MSG_ALT = (
    "Sie haben Tools bislang {tool_calls_count} Mal aufgerufen, mindestens {min_tool_calls} sind erforderlich. "
    "Bitte rufen Sie ein Tool auf, um Simulationsdaten zu beschaffen.{unused_hint}"
)

REACT_TOOL_LIMIT_MSG = (
    "Die Tool-Aufruf-Obergrenze ist erreicht ({tool_calls_count}/{max_tool_calls}); weitere Tool-Aufrufe sind nicht mehr möglich. "
    'Bitte geben Sie umgehend auf Basis der bereits gesammelten Informationen mit dem Präfix "Final Answer:" den Kapitelinhalt aus.'
)

REACT_UNUSED_TOOLS_HINT = "\nHinweis: Sie haben noch nicht verwendet: {unused_list}. Es empfiehlt sich, andere Tools auszuprobieren, um Mehrperspektiven-Informationen zu erhalten."

REACT_FORCE_FINAL_MSG = "Die Tool-Aufruf-Obergrenze ist erreicht. Bitte geben Sie direkt Final Answer: aus und erstellen Sie den Kapitelinhalt."

# ── Chat prompt ──

CHAT_SYSTEM_PROMPT_TEMPLATE = """\
Sie sind ein knapper, effizienter Assistent für Simulationsprognosen.

Hintergrund:
Prognosebedingung: {simulation_requirement}

Bereits erstellter Analysebericht:
{report_content}

Regeln:
1. Beantworten Sie Fragen vorrangig auf Basis des oben aufgeführten Berichtsinhalts.
2. Antworten Sie direkt und vermeiden Sie ausschweifende Gedankengänge.
3. Greifen Sie nur dann auf zusätzliche Tool-Recherche zurück, wenn der Berichtsinhalt für die Beantwortung nicht ausreicht.
4. Antworten sollen prägnant, klar und strukturiert sein.

Verfügbare Tools (nur bei Bedarf einsetzen, höchstens 1-2 Aufrufe):
{tools_description}

Format eines Tool-Aufrufs:
<tool_call>
{{"name": "Tool-Name", "parameters": {{"Parameter-Name": "Parameter-Wert"}}}}
</tool_call>

Antwortstil:
- Knapp und direkt, keine ausschweifenden Ausführungen.
- Verwenden Sie das > Format, um Schlüsselinhalte zu zitieren.
- Geben Sie zuerst die Schlussfolgerung, anschließend die Begründung."""

CHAT_OBSERVATION_SUFFIX = "\n\nBitte beantworten Sie die Frage knapp."


# ═══════════════════════════════════════════════════════════════
# ReportAgent main class
# ═══════════════════════════════════════════════════════════════


class ReportAgent:
    """
    Report Agent - simulation report generation agent.

    Uses ReACT (Reasoning + Acting):
    1. Planning phase: analyze simulation requirements and plan the report outline
    2. Generation phase: generate sections one at a time, each section may invoke tools multiple times
    3. Reflection phase: check completeness and accuracy of the content
    """

    # Maximum tool calls per section
    MAX_TOOL_CALLS_PER_SECTION = 5
    
    # Maximum number of reflection rounds
    MAX_REFLECTION_ROUNDS = 3

    # Maximum tool calls per chat message
    MAX_TOOL_CALLS_PER_CHAT = 2
    
    def __init__(
        self, 
        graph_id: str,
        simulation_id: str,
        simulation_requirement: str,
        llm_client: Optional[LLMClient] = None,
        zep_tools: Optional[ZepToolsService] = None
    ):
        """
        Initialize the Report Agent.

        Args:
            graph_id: graph ID
            simulation_id: simulation ID
            simulation_requirement: simulation requirement description
            llm_client: LLM client (optional)
            zep_tools: Zep tools service (optional)
        """
        self.graph_id = graph_id
        self.simulation_id = simulation_id
        self.simulation_requirement = simulation_requirement

        self.llm = llm_client or LLMClient()
        self.zep_tools = zep_tools or ZepToolsService()

        # Tool definitions
        self.tools = self._define_tools()

        # Logger (initialized inside generate_report)
        self.report_logger: Optional[ReportLogger] = None
        # Console logger (initialized inside generate_report)
        self.console_logger: Optional[ReportConsoleLogger] = None
        
        logger.info(t('report.agentInitDone', graphId=graph_id, simulationId=simulation_id))
    
    def _define_tools(self) -> Dict[str, Dict[str, Any]]:
        """Define the available tools"""
        return {
            "insight_forge": {
                "name": "insight_forge",
                "description": TOOL_DESC_INSIGHT_FORGE,
                "parameters": {
                    "query": "Frage oder Thema, das Sie tiefgehend analysieren möchten",
                    "report_context": "Kontext des aktuellen Berichts-Kapitels (optional, hilft bei der Generierung präziserer Teilfragen)"
                }
            },
            "panorama_search": {
                "name": "panorama_search",
                "description": TOOL_DESC_PANORAMA_SEARCH,
                "parameters": {
                    "query": "Suchanfrage, dient der Relevanzsortierung",
                    "include_expired": "Ob abgelaufene/historische Inhalte einbezogen werden sollen (Standard True)"
                }
            },
            "quick_search": {
                "name": "quick_search",
                "description": TOOL_DESC_QUICK_SEARCH,
                "parameters": {
                    "query": "Suchanfrage-Zeichenkette",
                    "limit": "Anzahl der zurückzugebenden Ergebnisse (optional, Standard 10)"
                }
            },
            "interview_agents": {
                "name": "interview_agents",
                "description": TOOL_DESC_INTERVIEW_AGENTS,
                "parameters": {
                    "interview_topic": "Interview-Thema oder Anforderungsbeschreibung (z. B.: 'Sichtweisen der Studierenden zum Formaldehyd-Vorfall im Wohnheim erfassen')",
                    "max_agents": "Maximale Anzahl zu befragender Agents (optional, Standard 5, Maximum 10)"
                }
            }
        }
    
    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any], report_context: str = "") -> str:
        """
        Execute a tool call.

        Args:
            tool_name: tool name
            parameters: tool parameters
            report_context: report context (used by InsightForge)

        Returns:
            tool execution result (text format)
        """
        logger.info(t('report.executingTool', toolName=tool_name, params=parameters))
        
        try:
            if tool_name == "insight_forge":
                query = parameters.get("query", "")
                ctx = parameters.get("report_context", "") or report_context
                result = self.zep_tools.insight_forge(
                    graph_id=self.graph_id,
                    query=query,
                    simulation_requirement=self.simulation_requirement,
                    report_context=ctx
                )
                return result.to_text()
            
            elif tool_name == "panorama_search":
                # Breadth search - full picture
                query = parameters.get("query", "")
                include_expired = parameters.get("include_expired", True)
                if isinstance(include_expired, str):
                    include_expired = include_expired.lower() in ['true', '1', 'yes']
                result = self.zep_tools.panorama_search(
                    graph_id=self.graph_id,
                    query=query,
                    include_expired=include_expired
                )
                return result.to_text()
            
            elif tool_name == "quick_search":
                # Simple search - quick lookup
                query = parameters.get("query", "")
                limit = parameters.get("limit", 10)
                if isinstance(limit, str):
                    limit = int(limit)
                result = self.zep_tools.quick_search(
                    graph_id=self.graph_id,
                    query=query,
                    limit=limit
                )
                return result.to_text()
            
            elif tool_name == "interview_agents":
                # Deep interview - call the real OASIS interview API for actual agent responses (dual-platform)
                interview_topic = parameters.get("interview_topic", parameters.get("query", ""))
                max_agents = parameters.get("max_agents", 5)
                if isinstance(max_agents, str):
                    max_agents = int(max_agents)
                max_agents = min(max_agents, 10)
                result = self.zep_tools.interview_agents(
                    simulation_id=self.simulation_id,
                    interview_requirement=interview_topic,
                    simulation_requirement=self.simulation_requirement,
                    max_agents=max_agents
                )
                return result.to_text()
            
            # ========== Backward-compatible legacy tools (internally redirected to new tools) ==========

            elif tool_name == "search_graph":
                # Redirect to quick_search
                logger.info(t('report.redirectToQuickSearch'))
                return self._execute_tool("quick_search", parameters, report_context)
            
            elif tool_name == "get_graph_statistics":
                result = self.zep_tools.get_graph_statistics(self.graph_id)
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_entity_summary":
                entity_name = parameters.get("entity_name", "")
                result = self.zep_tools.get_entity_summary(
                    graph_id=self.graph_id,
                    entity_name=entity_name
                )
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_simulation_context":
                # Redirect to insight_forge because it is more powerful
                logger.info(t('report.redirectToInsightForge'))
                query = parameters.get("query", self.simulation_requirement)
                return self._execute_tool("insight_forge", {"query": query}, report_context)
            
            elif tool_name == "get_entities_by_type":
                entity_type = parameters.get("entity_type", "")
                nodes = self.zep_tools.get_entities_by_type(
                    graph_id=self.graph_id,
                    entity_type=entity_type
                )
                result = [n.to_dict() for n in nodes]
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            else:
                return f"Unbekanntes Tool: {tool_name}. Bitte verwenden Sie eines der folgenden Tools: insight_forge, panorama_search, quick_search"

        except Exception as e:
            logger.error(t('report.toolExecFailed', toolName=tool_name, error=str(e)))
            return f"Tool-Ausführung fehlgeschlagen: {str(e)}"

    # Set of valid tool names, used to validate bare JSON fallback parsing
    VALID_TOOL_NAMES = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls out of an LLM response.

        Supported formats (in order of priority):
        1. <tool_call>{"name": "tool_name", "parameters": {...}}</tool_call>
        2. Bare JSON (whole response or a single line is a tool-call JSON)
        """
        tool_calls = []

        # Format 1: XML style (standard format)
        xml_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        for match in re.finditer(xml_pattern, response, re.DOTALL):
            try:
                call_data = json.loads(match.group(1))
                tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        if tool_calls:
            return tool_calls

        # Format 2: fallback - LLM emits bare JSON (without <tool_call> wrapper).
        # Only attempted when format 1 did not match, to avoid false positives in body text.
        stripped = response.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            try:
                call_data = json.loads(stripped)
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
                    return tool_calls
            except json.JSONDecodeError:
                pass

        # The response may contain reasoning text plus a bare JSON; try extracting the last JSON object
        json_pattern = r'(\{"(?:name|tool)"\s*:.*?\})\s*$'
        match = re.search(json_pattern, stripped, re.DOTALL)
        if match:
            try:
                call_data = json.loads(match.group(1))
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        return tool_calls

    def _is_valid_tool_call(self, data: dict) -> bool:
        """Validate whether parsed JSON is a legal tool call"""
        # Support both {"name": ..., "parameters": ...} and {"tool": ..., "params": ...} key shapes
        tool_name = data.get("name") or data.get("tool")
        if tool_name and tool_name in self.VALID_TOOL_NAMES:
            # Normalize keys to name / parameters
            if "tool" in data:
                data["name"] = data.pop("tool")
            if "params" in data and "parameters" not in data:
                data["parameters"] = data.pop("params")
            return True
        return False

    def _get_tools_description(self) -> str:
        """Build the tool description text (German because it is part of the system prompt)"""
        desc_parts = ["Verfügbare Tools:"]
        for name, tool in self.tools.items():
            params_desc = ", ".join([f"{k}: {v}" for k, v in tool["parameters"].items()])
            desc_parts.append(f"- {name}: {tool['description']}")
            if params_desc:
                desc_parts.append(f"  Parameter: {params_desc}")
        return "\n".join(desc_parts)
    
    def plan_outline(
        self, 
        progress_callback: Optional[Callable] = None
    ) -> ReportOutline:
        """
        Plan the report outline.

        Uses the LLM to analyze the simulation requirements and plan the report TOC.

        Args:
            progress_callback: progress callback function

        Returns:
            ReportOutline: the report outline
        """
        logger.info(t('report.startPlanningOutline'))

        if progress_callback:
            progress_callback("planning", 0, t('progress.analyzingRequirements'))

        # First fetch the simulation context
        context = self.zep_tools.get_simulation_context(
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement
        )
        
        if progress_callback:
            progress_callback("planning", 30, t('progress.generatingOutline'))
        
        system_prompt = f"{PLAN_SYSTEM_PROMPT}\n\n{get_language_instruction()}"
        user_prompt = PLAN_USER_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            total_nodes=context.get('graph_statistics', {}).get('total_nodes', 0),
            total_edges=context.get('graph_statistics', {}).get('total_edges', 0),
            entity_types=list(context.get('graph_statistics', {}).get('entity_types', {}).keys()),
            total_entities=context.get('total_entities', 0),
            related_facts_json=json.dumps(context.get('related_facts', [])[:10], ensure_ascii=False, indent=2),
        )

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            if progress_callback:
                progress_callback("planning", 80, t('progress.parsingOutline'))
            
            # Parse the outline
            sections = []
            for section_data in response.get("sections", []):
                sections.append(ReportSection(
                    title=section_data.get("title", ""),
                    content=""
                ))

            outline = ReportOutline(
                title=response.get("title", "Simulationsanalyse-Bericht"),
                summary=response.get("summary", ""),
                sections=sections
            )
            
            if progress_callback:
                progress_callback("planning", 100, t('progress.outlinePlanComplete'))
            
            logger.info(t('report.outlinePlanDone', count=len(sections)))
            return outline
            
        except Exception as e:
            logger.error(t('report.outlinePlanFailed', error=str(e)))
            # Return a default outline (3 sections, used as fallback)
            return ReportOutline(
                title="Zukunftsprognose-Bericht",
                summary="Analyse zukünftiger Trends und Risiken auf Basis der Simulation",
                sections=[
                    ReportSection(title="Prognoseszenario und Kernerkenntnisse"),
                    ReportSection(title="Analyse des prognostizierten Verhaltens der Personengruppen"),
                    ReportSection(title="Trendausblick und Risikohinweise")
                ]
            )
    
    def _generate_section_react(
        self, 
        section: ReportSection,
        outline: ReportOutline,
        previous_sections: List[str],
        progress_callback: Optional[Callable] = None,
        section_index: int = 0
    ) -> str:
        """
        Generate a single section using the ReACT loop.

        ReACT loop:
        1. Thought - analyze what information is needed
        2. Action - call a tool to fetch information
        3. Observation - examine the tool result
        4. Repeat until enough info is gathered or the limit is reached
        5. Final Answer - emit the section content

        Args:
            section: section to generate
            outline: complete outline
            previous_sections: previous section contents (for coherence)
            progress_callback: progress callback
            section_index: section index (used for logging)

        Returns:
            section content (Markdown format)
        """
        logger.info(t('report.reactGenerateSection', title=section.title))

        # Log the section start
        if self.report_logger:
            self.report_logger.log_section_start(section.title, section_index)
        
        system_prompt = SECTION_SYSTEM_PROMPT_TEMPLATE.format(
            report_title=outline.title,
            report_summary=outline.summary,
            simulation_requirement=self.simulation_requirement,
            section_title=section.title,
            tools_description=self._get_tools_description(),
        )
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}"

        # Build the user prompt - inject up to 4000 chars per completed section
        if previous_sections:
            previous_parts = []
            for sec in previous_sections:
                # Cap each section at 4000 chars
                truncated = sec[:4000] + "..." if len(sec) > 4000 else sec
                previous_parts.append(truncated)
            previous_content = "\n\n---\n\n".join(previous_parts)
        else:
            previous_content = "(Dies ist das erste Kapitel)"
        
        user_prompt = SECTION_USER_PROMPT_TEMPLATE.format(
            previous_content=previous_content,
            section_title=section.title,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # ReACT loop
        tool_calls_count = 0
        max_iterations = 5  # max number of iterations
        min_tool_calls = 3  # minimum number of tool calls
        conflict_retries = 0  # consecutive conflicts where the LLM emitted both a tool call and Final Answer
        used_tools = set()  # tracks which tools have been used
        all_tools = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

        # Report context, used to generate InsightForge sub-questions (German because it lands in the LLM prompt)
        report_context = f"Kapitelüberschrift: {section.title}\nSimulationsanforderung: {self.simulation_requirement}"
        
        for iteration in range(max_iterations):
            if progress_callback:
                progress_callback(
                    "generating", 
                    int((iteration / max_iterations) * 100),
                    t('progress.deepSearchAndWrite', current=tool_calls_count, max=self.MAX_TOOL_CALLS_PER_SECTION)
                )
            
            # Call the LLM
            response = self.llm.chat(
                messages=messages,
                temperature=0.5,
                max_tokens=4096
            )

            # Check whether the LLM returned None (API error or empty body)
            if response is None:
                logger.warning(t('report.sectionIterNone', title=section.title, iteration=iteration + 1))
                # If iterations remain, append messages and retry
                if iteration < max_iterations - 1:
                    messages.append({"role": "assistant", "content": "(leere Antwort)"})
                    messages.append({"role": "user", "content": "Bitte fahren Sie mit der Inhaltserstellung fort."})
                    continue
                # Final iteration also returned None, fall through to forced closure
                break

            logger.debug(f"LLM response: {response[:200]}...")

            # Parse once and reuse the result
            tool_calls = self._parse_tool_calls(response)
            has_tool_calls = bool(tool_calls)
            has_final_answer = "Final Answer:" in response

            # ── Conflict handling: LLM emitted a tool call and Final Answer at once ──
            if has_tool_calls and has_final_answer:
                conflict_retries += 1
                logger.warning(
                    t('report.sectionConflict', title=section.title, iteration=iteration+1, conflictCount=conflict_retries)
                )

                if conflict_retries <= 2:
                    # First two attempts: discard this response and ask the LLM to reply again
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": (
                            "[Formatfehler] Sie haben in einer Antwort gleichzeitig einen Tool-Aufruf und Final Answer enthalten, was nicht erlaubt ist.\n"
                            "Pro Antwort dürfen Sie nur eine der folgenden zwei Aktionen ausführen:\n"
                            "- Ein Tool aufrufen (einen <tool_call>-Block ausgeben, ohne Final Answer).\n"
                            "- Den endgültigen Inhalt ausgeben (mit Präfix 'Final Answer:', ohne <tool_call>).\n"
                            "Bitte antworten Sie erneut und führen Sie nur eine der beiden Aktionen aus."
                        ),
                    })
                    continue
                else:
                    # Third attempt: degrade gracefully, truncate to the first tool call and force execution
                    logger.warning(
                        t('report.sectionConflictDowngrade', title=section.title, conflictCount=conflict_retries)
                    )
                    first_tool_end = response.find('</tool_call>')
                    if first_tool_end != -1:
                        response = response[:first_tool_end + len('</tool_call>')]
                        tool_calls = self._parse_tool_calls(response)
                        has_tool_calls = bool(tool_calls)
                    has_final_answer = False
                    conflict_retries = 0

            # Log the LLM response
            if self.report_logger:
                self.report_logger.log_llm_response(
                    section_title=section.title,
                    section_index=section_index,
                    response=response,
                    iteration=iteration + 1,
                    has_tool_calls=has_tool_calls,
                    has_final_answer=has_final_answer
                )

            # ── Case 1: LLM emitted Final Answer ──
            if has_final_answer:
                # Insufficient tool calls -> reject and require more tool usage
                if tool_calls_count < min_tool_calls:
                    messages.append({"role": "assistant", "content": response})
                    unused_tools = all_tools - used_tools
                    unused_hint = f" (Folgende Tools wurden noch nicht verwendet, bitte ausprobieren: {', '.join(unused_tools)})" if unused_tools else ""
                    messages.append({
                        "role": "user",
                        "content": REACT_INSUFFICIENT_TOOLS_MSG.format(
                            tool_calls_count=tool_calls_count,
                            min_tool_calls=min_tool_calls,
                            unused_hint=unused_hint,
                        ),
                    })
                    continue

                # Normal completion
                final_answer = response.split("Final Answer:")[-1].strip()
                logger.info(t('report.sectionGenDone', title=section.title, count=tool_calls_count))

                if self.report_logger:
                    self.report_logger.log_section_content(
                        section_title=section.title,
                        section_index=section_index,
                        content=final_answer,
                        tool_calls_count=tool_calls_count
                    )
                return final_answer

            # ── Case 2: LLM attempts a tool call ──
            if has_tool_calls:
                # Tool budget exhausted -> notify explicitly and require Final Answer
                if tool_calls_count >= self.MAX_TOOL_CALLS_PER_SECTION:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": REACT_TOOL_LIMIT_MSG.format(
                            tool_calls_count=tool_calls_count,
                            max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        ),
                    })
                    continue

                # Execute only the first tool call
                call = tool_calls[0]
                if len(tool_calls) > 1:
                    logger.info(t('report.multiToolOnlyFirst', total=len(tool_calls), toolName=call['name']))

                if self.report_logger:
                    self.report_logger.log_tool_call(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        parameters=call.get("parameters", {}),
                        iteration=iteration + 1
                    )

                result = self._execute_tool(
                    call["name"],
                    call.get("parameters", {}),
                    report_context=report_context
                )

                if self.report_logger:
                    self.report_logger.log_tool_result(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        result=result,
                        iteration=iteration + 1
                    )

                tool_calls_count += 1
                used_tools.add(call['name'])

                # Build the unused-tools hint
                unused_tools = all_tools - used_tools
                unused_hint = ""
                if unused_tools and tool_calls_count < self.MAX_TOOL_CALLS_PER_SECTION:
                    unused_hint = REACT_UNUSED_TOOLS_HINT.format(unused_list=", ".join(unused_tools))

                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": REACT_OBSERVATION_TEMPLATE.format(
                        tool_name=call["name"],
                        result=result,
                        tool_calls_count=tool_calls_count,
                        max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        used_tools_str=", ".join(used_tools),
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # ── Case 3: neither a tool call nor Final Answer ──
            messages.append({"role": "assistant", "content": response})

            if tool_calls_count < min_tool_calls:
                # Insufficient tool usage, recommend unused tools
                unused_tools = all_tools - used_tools
                unused_hint = f" (Folgende Tools wurden noch nicht verwendet, bitte ausprobieren: {', '.join(unused_tools)})" if unused_tools else ""

                messages.append({
                    "role": "user",
                    "content": REACT_INSUFFICIENT_TOOLS_MSG_ALT.format(
                        tool_calls_count=tool_calls_count,
                        min_tool_calls=min_tool_calls,
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # Enough tool calls already; LLM emitted content without "Final Answer:" prefix.
            # Treat the content as the final answer directly to avoid spinning idle.
            logger.info(t('report.sectionNoPrefix', title=section.title, count=tool_calls_count))
            final_answer = response.strip()

            if self.report_logger:
                self.report_logger.log_section_content(
                    section_title=section.title,
                    section_index=section_index,
                    content=final_answer,
                    tool_calls_count=tool_calls_count
                )
            return final_answer
        
        # Max iterations reached, force content generation
        logger.warning(t('report.sectionMaxIter', title=section.title))
        messages.append({"role": "user", "content": REACT_FORCE_FINAL_MSG})
        
        response = self.llm.chat(
            messages=messages,
            temperature=0.5,
            max_tokens=4096
        )

        # Check whether the LLM returned None during the forced closure step
        if response is None:
            logger.error(t('report.sectionForceFailed', title=section.title))
            final_answer = t('report.sectionGenFailedContent')
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
        else:
            final_answer = response
        
        # Log section content generation completion
        if self.report_logger:
            self.report_logger.log_section_content(
                section_title=section.title,
                section_index=section_index,
                content=final_answer,
                tool_calls_count=tool_calls_count
            )
        
        return final_answer
    
    def generate_report(
        self, 
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        report_id: Optional[str] = None
    ) -> Report:
        """
        Generate the complete report (streamed section by section).

        Each section is persisted as soon as it is generated; no need to wait for the full report.
        File layout:
        reports/{report_id}/
            meta.json       - report metadata
            outline.json    - report outline
            progress.json   - generation progress
            section_01.md   - section 1
            section_02.md   - section 2
            ...
            full_report.md  - full report

        Args:
            progress_callback: progress callback (stage, progress, message)
            report_id: report ID (optional, auto-generated if missing)

        Returns:
            Report: the complete report
        """
        import uuid

        # Auto-generate the report_id when none is provided
        if not report_id:
            report_id = f"report_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now()
        
        report = Report(
            report_id=report_id,
            simulation_id=self.simulation_id,
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement,
            status=ReportStatus.PENDING,
            created_at=datetime.now().isoformat()
        )
        
        # List of completed section titles (for progress tracking)
        completed_section_titles = []

        try:
            # Initialize: create the report folder and persist initial state
            ReportManager._ensure_report_folder(report_id)

            # Initialize the structured logger (agent_log.jsonl)
            self.report_logger = ReportLogger(report_id)
            self.report_logger.log_start(
                simulation_id=self.simulation_id,
                graph_id=self.graph_id,
                simulation_requirement=self.simulation_requirement
            )

            # Initialize the console logger (console_log.txt)
            self.console_logger = ReportConsoleLogger(report_id)
            
            ReportManager.update_progress(
                report_id, "pending", 0, t('progress.initReport'),
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            # Phase 1: plan the outline
            report.status = ReportStatus.PLANNING
            ReportManager.update_progress(
                report_id, "planning", 5, t('progress.startPlanningOutline'),
                completed_sections=[]
            )

            # Log planning start
            self.report_logger.log_planning_start()
            
            if progress_callback:
                progress_callback("planning", 0, t('progress.startPlanningOutline'))
            
            outline = self.plan_outline(
                progress_callback=lambda stage, prog, msg: 
                    progress_callback(stage, prog // 5, msg) if progress_callback else None
            )
            report.outline = outline
            
            # Log planning completion
            self.report_logger.log_planning_complete(outline.to_dict())

            # Persist the outline to a file
            ReportManager.save_outline(report_id, outline)
            ReportManager.update_progress(
                report_id, "planning", 15, t('progress.outlineDone', count=len(outline.sections)),
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            logger.info(t('report.outlineSavedToFile', reportId=report_id))
            
            # Phase 2: generate sections one at a time (saving each)
            report.status = ReportStatus.GENERATING

            total_sections = len(outline.sections)
            generated_sections = []  # keep contents for context

            for i, section in enumerate(outline.sections):
                section_num = i + 1
                base_progress = 20 + int((i / total_sections) * 70)

                # Update progress
                ReportManager.update_progress(
                    report_id, "generating", base_progress,
                    t('progress.generatingSection', title=section.title, current=section_num, total=total_sections),
                    current_section=section.title,
                    completed_sections=completed_section_titles
                )

                if progress_callback:
                    progress_callback(
                        "generating",
                        base_progress,
                        t('progress.generatingSection', title=section.title, current=section_num, total=total_sections)
                    )
                
                # Generate the main section content
                section_content = self._generate_section_react(
                    section=section,
                    outline=outline,
                    previous_sections=generated_sections,
                    progress_callback=lambda stage, prog, msg:
                        progress_callback(
                            stage, 
                            base_progress + int(prog * 0.7 / total_sections),
                            msg
                        ) if progress_callback else None,
                    section_index=section_num
                )
                
                section.content = section_content
                generated_sections.append(f"## {section.title}\n\n{section_content}")

                # Persist the section
                ReportManager.save_section(report_id, section_num, section)
                completed_section_titles.append(section.title)

                # Log section completion
                full_section_content = f"## {section.title}\n\n{section_content}"

                if self.report_logger:
                    self.report_logger.log_section_full_complete(
                        section_title=section.title,
                        section_index=section_num,
                        full_content=full_section_content.strip()
                    )

                logger.info(t('report.sectionSaved', reportId=report_id, sectionNum=f"{section_num:02d}"))
                
                # Update progress
                ReportManager.update_progress(
                    report_id, "generating",
                    base_progress + int(70 / total_sections),
                    t('progress.sectionDone', title=section.title),
                    current_section=None,
                    completed_sections=completed_section_titles
                )

            # Phase 3: assemble the full report
            if progress_callback:
                progress_callback("generating", 95, t('progress.assemblingReport'))
            
            ReportManager.update_progress(
                report_id, "generating", 95, t('progress.assemblingReport'),
                completed_sections=completed_section_titles
            )
            
            # Use ReportManager to assemble the full report
            report.markdown_content = ReportManager.assemble_full_report(report_id, outline)
            report.status = ReportStatus.COMPLETED
            report.completed_at = datetime.now().isoformat()

            # Compute total elapsed time
            total_time_seconds = (datetime.now() - start_time).total_seconds()

            # Log report completion
            if self.report_logger:
                self.report_logger.log_report_complete(
                    total_sections=total_sections,
                    total_time_seconds=total_time_seconds
                )
            
            # Persist the final report
            ReportManager.save_report(report)
            ReportManager.update_progress(
                report_id, "completed", 100, t('progress.reportComplete'),
                completed_sections=completed_section_titles
            )
            
            if progress_callback:
                progress_callback("completed", 100, t('progress.reportComplete'))
            
            logger.info(t('report.reportGenDone', reportId=report_id))
            
            # Close the console logger
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None

            return report

        except Exception as e:
            logger.error(t('report.reportGenFailed', error=str(e)))
            report.status = ReportStatus.FAILED
            report.error = str(e)

            # Log the error
            if self.report_logger:
                self.report_logger.log_error(str(e), "failed")

            # Persist the failure state
            try:
                ReportManager.save_report(report)
                ReportManager.update_progress(
                    report_id, "failed", -1, t('progress.reportFailed', error=str(e)),
                    completed_sections=completed_section_titles
                )
            except Exception:
                pass  # ignore persistence errors during failure handling

            # Close the console logger
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
    
    def chat(
        self, 
        message: str,
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Chat with the Report Agent.

        During the chat the agent can autonomously invoke retrieval tools to answer questions.

        Args:
            message: user message
            chat_history: chat history

        Returns:
            {
                "response": "agent reply",
                "tool_calls": [list of invoked tools],
                "sources": [information sources]
            }
        """
        logger.info(t('report.agentChat', message=message[:50]))
        
        chat_history = chat_history or []
        
        # Fetch the already-generated report content
        report_content = ""
        try:
            report = ReportManager.get_report_by_simulation(self.simulation_id)
            if report and report.markdown_content:
                # Cap report length to avoid an oversized context window
                report_content = report.markdown_content[:15000]
                if len(report.markdown_content) > 15000:
                    report_content += "\n\n... [Berichtsinhalt gekürzt] ..."
        except Exception as e:
            logger.warning(t('report.fetchReportFailed', error=e))

        system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            report_content=report_content if report_content else "(noch kein Bericht vorhanden)",
            tools_description=self._get_tools_description(),
        )
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}"

        # Build the message list
        messages = [{"role": "system", "content": system_prompt}]

        # Append conversation history
        for h in chat_history[-10:]:  # cap history length
            messages.append(h)
        
        # Append the user message
        messages.append({
            "role": "user",
            "content": message
        })

        # ReACT loop (simplified)
        tool_calls_made = []
        max_iterations = 2  # reduced iteration count

        for iteration in range(max_iterations):
            response = self.llm.chat(
                messages=messages,
                temperature=0.5
            )

            # Parse tool calls
            tool_calls = self._parse_tool_calls(response)

            if not tool_calls:
                # No tool calls, return the response directly
                clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', response, flags=re.DOTALL)
                clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)

                return {
                    "response": clean_response.strip(),
                    "tool_calls": tool_calls_made,
                    "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
                }

            # Execute tool calls (capped)
            tool_results = []
            for call in tool_calls[:1]:  # at most one tool per round
                if len(tool_calls_made) >= self.MAX_TOOL_CALLS_PER_CHAT:
                    break
                result = self._execute_tool(call["name"], call.get("parameters", {}))
                tool_results.append({
                    "tool": call["name"],
                    "result": result[:1500]  # cap result length
                })
                tool_calls_made.append(call)

            # Append the result to the messages (German label, fed to LLM)
            messages.append({"role": "assistant", "content": response})
            observation = "\n".join([f"[Ergebnis {r['tool']}]\n{r['result']}" for r in tool_results])
            messages.append({
                "role": "user",
                "content": observation + CHAT_OBSERVATION_SUFFIX
            })

        # Max iterations reached, fetch the final response
        final_response = self.llm.chat(
            messages=messages,
            temperature=0.5
        )

        # Clean the response
        clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', final_response, flags=re.DOTALL)
        clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
        
        return {
            "response": clean_response.strip(),
            "tool_calls": tool_calls_made,
            "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
        }


class ReportManager:
    """
    Report manager.

    Handles report persistence and retrieval.

    File layout (sections persisted individually):
    reports/
      {report_id}/
        meta.json          - report metadata and status
        outline.json       - report outline
        progress.json      - generation progress
        section_01.md      - section 1
        section_02.md      - section 2
        ...
        full_report.md     - the full report
    """

    # Report storage directory
    REPORTS_DIR = os.path.join(Config.UPLOAD_FOLDER, 'reports')

    @classmethod
    def _ensure_reports_dir(cls):
        """Make sure the reports root directory exists"""
        os.makedirs(cls.REPORTS_DIR, exist_ok=True)

    @classmethod
    def _get_report_folder(cls, report_id: str) -> str:
        """Return the report folder path"""
        return os.path.join(cls.REPORTS_DIR, report_id)

    @classmethod
    def _ensure_report_folder(cls, report_id: str) -> str:
        """Make sure the report folder exists and return its path"""
        folder = cls._get_report_folder(report_id)
        os.makedirs(folder, exist_ok=True)
        return folder

    @classmethod
    def _get_report_path(cls, report_id: str) -> str:
        """Return the report metadata file path"""
        return os.path.join(cls._get_report_folder(report_id), "meta.json")

    @classmethod
    def _get_report_markdown_path(cls, report_id: str) -> str:
        """Return the full-report Markdown file path"""
        return os.path.join(cls._get_report_folder(report_id), "full_report.md")

    @classmethod
    def _get_outline_path(cls, report_id: str) -> str:
        """Return the outline file path"""
        return os.path.join(cls._get_report_folder(report_id), "outline.json")

    @classmethod
    def _get_progress_path(cls, report_id: str) -> str:
        """Return the progress file path"""
        return os.path.join(cls._get_report_folder(report_id), "progress.json")

    @classmethod
    def _get_section_path(cls, report_id: str, section_index: int) -> str:
        """Return the section Markdown file path"""
        return os.path.join(cls._get_report_folder(report_id), f"section_{section_index:02d}.md")

    @classmethod
    def _get_agent_log_path(cls, report_id: str) -> str:
        """Return the agent log file path"""
        return os.path.join(cls._get_report_folder(report_id), "agent_log.jsonl")

    @classmethod
    def _get_console_log_path(cls, report_id: str) -> str:
        """Return the console log file path"""
        return os.path.join(cls._get_report_folder(report_id), "console_log.txt")

    @classmethod
    def get_console_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Fetch console log content.

        This is the console output of the report generation process (INFO, WARNING, etc.),
        distinct from the structured agent_log.jsonl entries.

        Args:
            report_id: report ID
            from_line: line number to start reading from (for incremental polling, 0 = start)

        Returns:
            {
                "logs": [list of log lines],
                "total_lines": total line count,
                "from_line": starting line number,
                "has_more": whether more logs are available
            }
        """
        log_path = cls._get_console_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    # Keep original log line, strip trailing newline
                    logs.append(line.rstrip('\n\r'))

        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # reached end of file
        }

    @classmethod
    def get_console_log_stream(cls, report_id: str) -> List[str]:
        """
        Fetch the full console log in a single call.

        Args:
            report_id: report ID

        Returns:
            list of log lines
        """
        result = cls.get_console_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def get_agent_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Fetch agent log content.

        Args:
            report_id: report ID
            from_line: line number to start reading from (for incremental polling, 0 = start)

        Returns:
            {
                "logs": [list of log entries],
                "total_lines": total line count,
                "from_line": starting line number,
                "has_more": whether more logs are available
            }
        """
        log_path = cls._get_agent_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    try:
                        log_entry = json.loads(line.strip())
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        # Skip lines that fail to parse
                        continue

        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # reached end of file
        }

    @classmethod
    def get_agent_log_stream(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Fetch the full agent log in a single call.

        Args:
            report_id: report ID

        Returns:
            list of log entries
        """
        result = cls.get_agent_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def save_outline(cls, report_id: str, outline: ReportOutline) -> None:
        """
        Save the report outline.

        Called immediately after the planning phase completes.
        """
        cls._ensure_report_folder(report_id)
        
        with open(cls._get_outline_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(outline.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(t('report.outlineSaved', reportId=report_id))
    
    @classmethod
    def save_section(
        cls,
        report_id: str,
        section_index: int,
        section: ReportSection
    ) -> str:
        """
        Save a single section.

        Called immediately after each section is generated to enable streaming output.

        Args:
            report_id: report ID
            section_index: section index (starts at 1)
            section: section object

        Returns:
            saved file path
        """
        cls._ensure_report_folder(report_id)

        # Build the section Markdown content - strip potential duplicate headings
        cleaned_content = cls._clean_section_content(section.content, section.title)
        md_content = f"## {section.title}\n\n"
        if cleaned_content:
            md_content += f"{cleaned_content}\n\n"

        # Persist the file
        file_suffix = f"section_{section_index:02d}.md"
        file_path = os.path.join(cls._get_report_folder(report_id), file_suffix)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logger.info(t('report.sectionFileSaved', reportId=report_id, fileSuffix=file_suffix))
        return file_path
    
    @classmethod
    def _clean_section_content(cls, content: str, section_title: str) -> str:
        """
        Clean up section content.

        1. Remove a Markdown heading line at the start that duplicates the section title.
        2. Convert all ### and lower-level headings into bold text.

        Args:
            content: raw content
            section_title: section title

        Returns:
            cleaned content
        """
        import re
        
        if not content:
            return content
        
        content = content.strip()
        lines = content.split('\n')
        cleaned_lines = []
        skip_next_empty = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Detect Markdown heading lines
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)

            if heading_match:
                level = len(heading_match.group(1))
                title_text = heading_match.group(2).strip()

                # Check whether the heading duplicates the section title (within the first 5 lines)
                if i < 5:
                    if title_text == section_title or title_text.replace(' ', '') == section_title.replace(' ', ''):
                        skip_next_empty = True
                        continue

                # Convert all heading levels (#, ##, ###, #### etc.) to bold text;
                # the section title is added by the system, so the content must not contain any headings.
                cleaned_lines.append(f"**{title_text}**")
                cleaned_lines.append("")  # add empty line
                continue

            # If the previous line was a skipped heading and the current line is empty, skip too
            if skip_next_empty and stripped == '':
                skip_next_empty = False
                continue

            skip_next_empty = False
            cleaned_lines.append(line)

        # Remove leading empty lines
        while cleaned_lines and cleaned_lines[0].strip() == '':
            cleaned_lines.pop(0)

        # Remove leading horizontal rules
        while cleaned_lines and cleaned_lines[0].strip() in ['---', '***', '___']:
            cleaned_lines.pop(0)
            # Also strip empty lines after the rule
            while cleaned_lines and cleaned_lines[0].strip() == '':
                cleaned_lines.pop(0)
        
        return '\n'.join(cleaned_lines)
    
    @classmethod
    def update_progress(
        cls, 
        report_id: str, 
        status: str, 
        progress: int, 
        message: str,
        current_section: str = None,
        completed_sections: List[str] = None
    ) -> None:
        """
        Update the report generation progress.

        The frontend can poll progress.json for real-time progress.
        """
        cls._ensure_report_folder(report_id)
        
        progress_data = {
            "status": status,
            "progress": progress,
            "message": message,
            "current_section": current_section,
            "completed_sections": completed_sections or [],
            "updated_at": datetime.now().isoformat()
        }
        
        with open(cls._get_progress_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def get_progress(cls, report_id: str) -> Optional[Dict[str, Any]]:
        """Fetch the report generation progress"""
        path = cls._get_progress_path(report_id)
        
        if not os.path.exists(path):
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @classmethod
    def get_generated_sections(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Fetch the list of generated sections.

        Returns information about every persisted section file.
        """
        folder = cls._get_report_folder(report_id)
        
        if not os.path.exists(folder):
            return []
        
        sections = []
        for filename in sorted(os.listdir(folder)):
            if filename.startswith('section_') and filename.endswith('.md'):
                file_path = os.path.join(folder, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Parse the section index from the filename
                parts = filename.replace('.md', '').split('_')
                section_index = int(parts[1])

                sections.append({
                    "filename": filename,
                    "section_index": section_index,
                    "content": content
                })

        return sections
    
    @classmethod
    def assemble_full_report(cls, report_id: str, outline: ReportOutline) -> str:
        """
        Assemble the full report.

        Builds the full report from the persisted section files and cleans up headings.
        """
        folder = cls._get_report_folder(report_id)

        # Build the report header
        md_content = f"# {outline.title}\n\n"
        md_content += f"> {outline.summary}\n\n"
        md_content += f"---\n\n"

        # Read all section files in order
        sections = cls.get_generated_sections(report_id)
        for section_info in sections:
            md_content += section_info["content"]

        # Post-process: clean up heading issues across the full report
        md_content = cls._post_process_report(md_content, outline)

        # Persist the full report
        full_path = cls._get_report_markdown_path(report_id)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(t('report.fullReportAssembled', reportId=report_id))
        return md_content
    
    @classmethod
    def _post_process_report(cls, content: str, outline: ReportOutline) -> str:
        """
        Post-process the report content.

        1. Remove duplicate headings.
        2. Keep the report main title (#) and section titles (##); demote other levels (###, ####).
        3. Clean up excess blank lines and horizontal rules.

        Args:
            content: raw report content
            outline: report outline

        Returns:
            processed content
        """
        import re

        lines = content.split('\n')
        processed_lines = []
        prev_was_heading = False

        # Collect every section title from the outline
        section_titles = set()
        for section in outline.sections:
            section_titles.add(section.title)

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Detect heading lines
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)

            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()

                # Check whether this is a duplicate heading (same title within the previous 5 lines)
                is_duplicate = False
                for j in range(max(0, len(processed_lines) - 5), len(processed_lines)):
                    prev_line = processed_lines[j].strip()
                    prev_match = re.match(r'^(#{1,6})\s+(.+)$', prev_line)
                    if prev_match:
                        prev_title = prev_match.group(2).strip()
                        if prev_title == title:
                            is_duplicate = True
                            break

                if is_duplicate:
                    # Skip the duplicate heading and the blank lines that follow
                    i += 1
                    while i < len(lines) and lines[i].strip() == '':
                        i += 1
                    continue

                # Heading level handling:
                # - # (level=1) keep only the report main title
                # - ## (level=2) keep section titles
                # - ### or deeper (level>=3) convert to bold text

                if level == 1:
                    if title == outline.title:
                        # Keep the report main title
                        processed_lines.append(line)
                        prev_was_heading = True
                    elif title in section_titles:
                        # Section title incorrectly used #, fix to ##
                        processed_lines.append(f"## {title}")
                        prev_was_heading = True
                    else:
                        # Other H1 headings become bold
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                elif level == 2:
                    if title in section_titles or title == outline.title:
                        # Keep section titles
                        processed_lines.append(line)
                        prev_was_heading = True
                    else:
                        # Non-section H2 headings become bold
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                else:
                    # ### or deeper headings become bold text
                    processed_lines.append(f"**{title}**")
                    processed_lines.append("")
                    prev_was_heading = False

                i += 1
                continue

            elif stripped == '---' and prev_was_heading:
                # Skip horizontal rules immediately following a heading
                i += 1
                continue

            elif stripped == '' and prev_was_heading:
                # Keep at most one blank line after a heading
                if processed_lines and processed_lines[-1].strip() != '':
                    processed_lines.append(line)
                prev_was_heading = False

            else:
                processed_lines.append(line)
                prev_was_heading = False

            i += 1

        # Collapse consecutive blank lines (keep at most 2)
        result_lines = []
        empty_count = 0
        for line in processed_lines:
            if line.strip() == '':
                empty_count += 1
                if empty_count <= 2:
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    @classmethod
    def save_report(cls, report: Report) -> None:
        """Save the report metadata and the full report"""
        cls._ensure_report_folder(report.report_id)

        # Persist the metadata JSON
        with open(cls._get_report_path(report.report_id), 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

        # Persist the outline
        if report.outline:
            cls.save_outline(report.report_id, report.outline)

        # Persist the full Markdown report
        if report.markdown_content:
            with open(cls._get_report_markdown_path(report.report_id), 'w', encoding='utf-8') as f:
                f.write(report.markdown_content)
        
        logger.info(t('report.reportSaved', reportId=report.report_id))
    
    @classmethod
    def get_report(cls, report_id: str) -> Optional[Report]:
        """Fetch a report"""
        path = cls._get_report_path(report_id)

        if not os.path.exists(path):
            # Backward-compat: check files stored directly under the reports directory
            old_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
            if os.path.exists(old_path):
                path = old_path
            else:
                return None

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Rebuild the Report object
        outline = None
        if data.get('outline'):
            outline_data = data['outline']
            sections = []
            for s in outline_data.get('sections', []):
                sections.append(ReportSection(
                    title=s['title'],
                    content=s.get('content', '')
                ))
            outline = ReportOutline(
                title=outline_data['title'],
                summary=outline_data['summary'],
                sections=sections
            )
        
        # If markdown_content is empty, try loading it from full_report.md
        markdown_content = data.get('markdown_content', '')
        if not markdown_content:
            full_report_path = cls._get_report_markdown_path(report_id)
            if os.path.exists(full_report_path):
                with open(full_report_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
        
        return Report(
            report_id=data['report_id'],
            simulation_id=data['simulation_id'],
            graph_id=data['graph_id'],
            simulation_requirement=data['simulation_requirement'],
            status=ReportStatus(data['status']),
            outline=outline,
            markdown_content=markdown_content,
            created_at=data.get('created_at', ''),
            completed_at=data.get('completed_at', ''),
            error=data.get('error')
        )
    
    @classmethod
    def get_report_by_simulation(cls, simulation_id: str) -> Optional[Report]:
        """Fetch a report by simulation ID"""
        cls._ensure_reports_dir()

        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # New format: folder
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report and report.simulation_id == simulation_id:
                    return report
            # Backward-compat: JSON file
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report and report.simulation_id == simulation_id:
                    return report
        
        return None
    
    @classmethod
    def list_reports(cls, simulation_id: Optional[str] = None, limit: int = 50) -> List[Report]:
        """List reports"""
        cls._ensure_reports_dir()

        reports = []
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # New format: folder
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
            # Backward-compat: JSON file
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)

        # Sort by creation time, newest first
        reports.sort(key=lambda r: r.created_at, reverse=True)
        
        return reports[:limit]
    
    @classmethod
    def delete_report(cls, report_id: str) -> bool:
        """Delete a report (the entire folder)"""
        import shutil

        folder_path = cls._get_report_folder(report_id)

        # New format: delete the whole folder
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            logger.info(t('report.reportFolderDeleted', reportId=report_id))
            return True

        # Backward-compat: delete the standalone files
        deleted = False
        old_json_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
        old_md_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.md")
        
        if os.path.exists(old_json_path):
            os.remove(old_json_path)
            deleted = True
        if os.path.exists(old_md_path):
            os.remove(old_md_path)
            deleted = True
        
        return deleted
