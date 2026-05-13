"""
Zep graph memory updater service
Updates the Zep graph in real time with agent activities from a running simulation.
"""

import os
import time
import threading
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from queue import Queue, Empty

from . import memory_service

from ..config import Config
from ..utils.logger import get_logger
from ..utils.locale import get_locale, set_locale

logger = get_logger('mirofish.zep_graph_memory_updater')


@dataclass
class AgentActivity:
    """Agent activity record"""
    platform: str           # twitter / reddit
    agent_id: int
    agent_name: str
    action_type: str        # CREATE_POST, LIKE_POST, etc.
    action_args: Dict[str, Any]
    round_num: int
    timestamp: str

    def to_episode_text(self) -> str:
        """
        Convert the activity into a text description that can be sent to Zep.

        Uses a natural-language format so Zep can extract entities and relationships from it.
        Does not add a simulation-related prefix to avoid biasing the graph update.

        Note: the produced text is consumed by an LLM (Zep entity extractor),
        so the natural-language sentences are German.
        """
        # Generate different descriptions per action type
        action_descriptions = {
            "CREATE_POST": self._describe_create_post,
            "LIKE_POST": self._describe_like_post,
            "DISLIKE_POST": self._describe_dislike_post,
            "REPOST": self._describe_repost,
            "QUOTE_POST": self._describe_quote_post,
            "FOLLOW": self._describe_follow,
            "CREATE_COMMENT": self._describe_create_comment,
            "LIKE_COMMENT": self._describe_like_comment,
            "DISLIKE_COMMENT": self._describe_dislike_comment,
            "SEARCH_POSTS": self._describe_search,
            "SEARCH_USER": self._describe_search_user,
            "MUTE": self._describe_mute,
        }

        describe_func = action_descriptions.get(self.action_type, self._describe_generic)
        description = describe_func()

        # Return "agent name: activity description" without a simulation prefix
        return f"{self.agent_name}: {description}"

    def _describe_create_post(self) -> str:
        content = self.action_args.get("content", "")
        if content:
            return f"hat einen Beitrag veröffentlicht: „{content}\""
        return "hat einen Beitrag veröffentlicht"

    def _describe_like_post(self) -> str:
        """Like a post — includes original content and author"""
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")

        if post_content and post_author:
            return f"hat den Beitrag von {post_author} positiv bewertet: „{post_content}\""
        elif post_content:
            return f"hat einen Beitrag positiv bewertet: „{post_content}\""
        elif post_author:
            return f"hat einen Beitrag von {post_author} positiv bewertet"
        return "hat einen Beitrag positiv bewertet"

    def _describe_dislike_post(self) -> str:
        """Dislike a post — includes original content and author"""
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")

        if post_content and post_author:
            return f"hat den Beitrag von {post_author} negativ bewertet: „{post_content}\""
        elif post_content:
            return f"hat einen Beitrag negativ bewertet: „{post_content}\""
        elif post_author:
            return f"hat einen Beitrag von {post_author} negativ bewertet"
        return "hat einen Beitrag negativ bewertet"

    def _describe_repost(self) -> str:
        """Repost — includes original content and author"""
        original_content = self.action_args.get("original_content", "")
        original_author = self.action_args.get("original_author_name", "")

        if original_content and original_author:
            return f"hat den Beitrag von {original_author} geteilt: „{original_content}\""
        elif original_content:
            return f"hat einen Beitrag geteilt: „{original_content}\""
        elif original_author:
            return f"hat einen Beitrag von {original_author} geteilt"
        return "hat einen Beitrag geteilt"

    def _describe_quote_post(self) -> str:
        """Quote post — includes original content, author and quoting comment"""
        original_content = self.action_args.get("original_content", "")
        original_author = self.action_args.get("original_author_name", "")
        quote_content = self.action_args.get("quote_content", "") or self.action_args.get("content", "")

        base = ""
        if original_content and original_author:
            base = f"hat den Beitrag von {original_author} zitiert: „{original_content}\""
        elif original_content:
            base = f"hat einen Beitrag zitiert: „{original_content}\""
        elif original_author:
            base = f"hat einen Beitrag von {original_author} zitiert"
        else:
            base = "hat einen Beitrag zitiert"

        if quote_content:
            base += f" und kommentiert dazu: „{quote_content}\""
        return base

    def _describe_follow(self) -> str:
        """Follow user — includes followed username"""
        target_user_name = self.action_args.get("target_user_name", "")

        if target_user_name:
            return f"folgt dem Nutzer „{target_user_name}\""
        return "folgt einem Nutzer"

    def _describe_create_comment(self) -> str:
        """Create comment — includes content and the post being commented on"""
        content = self.action_args.get("content", "")
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")

        if content:
            if post_content and post_author:
                return f"kommentiert unter dem Beitrag von {post_author} „{post_content}\": „{content}\""
            elif post_content:
                return f"kommentiert unter dem Beitrag „{post_content}\": „{content}\""
            elif post_author:
                return f"kommentiert unter einem Beitrag von {post_author}: „{content}\""
            return f"kommentiert: „{content}\""
        return "hat einen Kommentar verfasst"

    def _describe_like_comment(self) -> str:
        """Like comment — includes content and author"""
        comment_content = self.action_args.get("comment_content", "")
        comment_author = self.action_args.get("comment_author_name", "")

        if comment_content and comment_author:
            return f"hat den Kommentar von {comment_author} positiv bewertet: „{comment_content}\""
        elif comment_content:
            return f"hat einen Kommentar positiv bewertet: „{comment_content}\""
        elif comment_author:
            return f"hat einen Kommentar von {comment_author} positiv bewertet"
        return "hat einen Kommentar positiv bewertet"

    def _describe_dislike_comment(self) -> str:
        """Dislike comment — includes content and author"""
        comment_content = self.action_args.get("comment_content", "")
        comment_author = self.action_args.get("comment_author_name", "")

        if comment_content and comment_author:
            return f"hat den Kommentar von {comment_author} negativ bewertet: „{comment_content}\""
        elif comment_content:
            return f"hat einen Kommentar negativ bewertet: „{comment_content}\""
        elif comment_author:
            return f"hat einen Kommentar von {comment_author} negativ bewertet"
        return "hat einen Kommentar negativ bewertet"

    def _describe_search(self) -> str:
        """Search posts — includes the search keyword"""
        query = self.action_args.get("query", "") or self.action_args.get("keyword", "")
        return f"hat nach „{query}\" gesucht" if query else "hat eine Suche durchgeführt"

    def _describe_search_user(self) -> str:
        """Search user — includes the search keyword"""
        query = self.action_args.get("query", "") or self.action_args.get("username", "")
        return f"hat nach dem Nutzer „{query}\" gesucht" if query else "hat nach einem Nutzer gesucht"

    def _describe_mute(self) -> str:
        """Mute user — includes the muted username"""
        target_user_name = self.action_args.get("target_user_name", "")

        if target_user_name:
            return f"hat den Nutzer „{target_user_name}\" stummgeschaltet"
        return "hat einen Nutzer stummgeschaltet"

    def _describe_generic(self) -> str:
        # Fallback for unknown action types
        return f"hat die Aktion {self.action_type} ausgeführt"


class ZepGraphMemoryUpdater:
    """
    Zep graph memory updater.

    Watches the simulation's actions log file and updates the Zep graph in real time with new agent activities.
    Activities are grouped per platform and sent in batches once BATCH_SIZE entries have accumulated.

    All meaningful behaviors are forwarded to Zep, with full context information in action_args:
    - Original content of liked/disliked posts
    - Original content of reposted/quoted posts
    - Username for follows/mutes
    - Original content of liked/disliked comments
    """

    # Batch size (per-platform, before sending)
    BATCH_SIZE = 5

    # Platform display name mapping (used in console messages)
    PLATFORM_DISPLAY_NAMES = {
        'twitter': 'World 1',
        'reddit': 'World 2',
    }

    # Send interval in seconds (avoid hammering the API)
    SEND_INTERVAL = 0.5

    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds

    def __init__(self, graph_id: str, api_key: Optional[str] = None):
        """
        Initialize the updater.

        Args:
            graph_id: Zep graph ID
            api_key: Zep API key (optional, defaults to config)
        """
        self.graph_id = graph_id
        # api_key kept for backwards compat; memory_service reads connection from Config.
        self._api_key = api_key
        memory_service.warmup()

        # Activity queue
        self._activity_queue: Queue = Queue()

        # Per-platform activity buffers (each accumulates up to BATCH_SIZE before flushing)
        self._platform_buffers: Dict[str, List[AgentActivity]] = {
            'twitter': [],
            'reddit': [],
        }
        self._buffer_lock = threading.Lock()

        # Control flags
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        # Metrics
        self._total_activities = 0  # activities actually enqueued
        self._total_sent = 0        # batches successfully sent to Zep
        self._total_items_sent = 0  # activity items successfully sent to Zep
        self._failed_count = 0      # failed batch sends
        self._skipped_count = 0     # filtered/skipped activities (DO_NOTHING)

        logger.info(f"ZepGraphMemoryUpdater initialized: graph_id={graph_id}, batch_size={self.BATCH_SIZE}")

    def _get_platform_display_name(self, platform: str) -> str:
        """Return the display name for a platform"""
        return self.PLATFORM_DISPLAY_NAMES.get(platform.lower(), platform)

    def start(self):
        """Start the background worker thread"""
        if self._running:
            return

        # Capture locale before spawning background thread
        current_locale = get_locale()

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            args=(current_locale,),
            daemon=True,
            name=f"ZepMemoryUpdater-{self.graph_id[:8]}"
        )
        self._worker_thread.start()
        logger.info(f"ZepGraphMemoryUpdater started: graph_id={self.graph_id}")

    def stop(self):
        """Stop the background worker thread"""
        self._running = False

        # Send any remaining activities
        self._flush_remaining()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=10)

        logger.info(f"ZepGraphMemoryUpdater stopped: graph_id={self.graph_id}, "
                   f"total_activities={self._total_activities}, "
                   f"batches_sent={self._total_sent}, "
                   f"items_sent={self._total_items_sent}, "
                   f"failed={self._failed_count}, "
                   f"skipped={self._skipped_count}")

    def add_activity(self, activity: AgentActivity):
        """
        Add an agent activity to the queue.

        All meaningful actions are queued, including:
        - CREATE_POST
        - CREATE_COMMENT
        - QUOTE_POST
        - SEARCH_POSTS
        - SEARCH_USER
        - LIKE_POST/DISLIKE_POST
        - REPOST
        - FOLLOW
        - MUTE
        - LIKE_COMMENT/DISLIKE_COMMENT

        action_args carries full context (original post text, usernames, ...).

        Args:
            activity: Agent activity record
        """
        # Skip DO_NOTHING activities
        if activity.action_type == "DO_NOTHING":
            self._skipped_count += 1
            return

        self._activity_queue.put(activity)
        self._total_activities += 1
        logger.debug(f"Enqueued activity for Zep: {activity.agent_name} - {activity.action_type}")

    def add_activity_from_dict(self, data: Dict[str, Any], platform: str):
        """
        Add an activity from a dict.

        Args:
            data: Dict parsed from actions.jsonl
            platform: Platform name (twitter/reddit)
        """
        # Skip event-type entries
        if "event_type" in data:
            return

        activity = AgentActivity(
            platform=platform,
            agent_id=data.get("agent_id", 0),
            agent_name=data.get("agent_name", ""),
            action_type=data.get("action_type", ""),
            action_args=data.get("action_args", {}),
            round_num=data.get("round", 0),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )

        self.add_activity(activity)

    def _worker_loop(self, locale: str = 'zh'):
        """Background worker loop — sends activities to Zep in per-platform batches"""
        set_locale(locale)
        while self._running or not self._activity_queue.empty():
            try:
                # Try to dequeue an activity (1s timeout)
                try:
                    activity = self._activity_queue.get(timeout=1)

                    # Append to the matching platform buffer
                    platform = activity.platform.lower()
                    with self._buffer_lock:
                        if platform not in self._platform_buffers:
                            self._platform_buffers[platform] = []
                        self._platform_buffers[platform].append(activity)

                        # Check whether the platform reached the batch size
                        if len(self._platform_buffers[platform]) >= self.BATCH_SIZE:
                            batch = self._platform_buffers[platform][:self.BATCH_SIZE]
                            self._platform_buffers[platform] = self._platform_buffers[platform][self.BATCH_SIZE:]
                            # Release lock before sending
                            self._send_batch_activities(batch, platform)
                            # Throttle to avoid hammering Zep
                            time.sleep(self.SEND_INTERVAL)

                except Empty:
                    pass

            except Exception as e:
                logger.error(f"Worker loop exception: {e}")
                time.sleep(1)

    def _send_batch_activities(self, activities: List[AgentActivity], platform: str):
        """
        Send a batch of activities to the Zep graph (merged into a single text).

        Args:
            activities: List of agent activities
            platform: Platform name
        """
        if not activities:
            return

        # Combine multiple activities into a single newline-separated text
        episode_texts = [activity.to_episode_text() for activity in activities]
        combined_text = "\n".join(episode_texts)

        # Send with retry
        for attempt in range(self.MAX_RETRIES):
            try:
                memory_service.add_episode(
                    group_id=self.graph_id,
                    content=combined_text,
                    source_type="text",
                )

                self._total_sent += 1
                self._total_items_sent += len(activities)
                display_name = self._get_platform_display_name(platform)
                logger.info(f"Batch send OK: {len(activities)} {display_name} activities -> graph {self.graph_id}")
                logger.debug(f"Batch content preview: {combined_text[:200]}...")
                return

            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(f"Batch send to Zep failed (attempt {attempt + 1}/{self.MAX_RETRIES}): {e}")
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"Batch send to Zep failed after {self.MAX_RETRIES} retries: {e}")
                    self._failed_count += 1

    def _flush_remaining(self):
        """Send any activities still in the queue and buffers"""
        # First drain the queue into the buffers
        while not self._activity_queue.empty():
            try:
                activity = self._activity_queue.get_nowait()
                platform = activity.platform.lower()
                with self._buffer_lock:
                    if platform not in self._platform_buffers:
                        self._platform_buffers[platform] = []
                    self._platform_buffers[platform].append(activity)
            except Empty:
                break

        # Then send remaining buffer contents per platform (even if smaller than BATCH_SIZE)
        with self._buffer_lock:
            for platform, buffer in self._platform_buffers.items():
                if buffer:
                    display_name = self._get_platform_display_name(platform)
                    logger.info(f"Flushing {display_name}: {len(buffer)} pending activities")
                    self._send_batch_activities(buffer, platform)
            # Clear all buffers
            for platform in self._platform_buffers:
                self._platform_buffers[platform] = []

    def get_stats(self) -> Dict[str, Any]:
        """Return the runtime statistics"""
        with self._buffer_lock:
            buffer_sizes = {p: len(b) for p, b in self._platform_buffers.items()}

        return {
            "graph_id": self.graph_id,
            "batch_size": self.BATCH_SIZE,
            "total_activities": self._total_activities,  # total activities enqueued
            "batches_sent": self._total_sent,            # batches sent successfully
            "items_sent": self._total_items_sent,        # activity items sent successfully
            "failed_count": self._failed_count,          # failed batches
            "skipped_count": self._skipped_count,        # filtered activities (DO_NOTHING)
            "queue_size": self._activity_queue.qsize(),
            "buffer_sizes": buffer_sizes,                # per-platform buffer sizes
            "running": self._running,
        }


class ZepGraphMemoryManager:
    """
    Manages Zep graph memory updaters across multiple simulations.

    Each simulation can own its own updater instance.
    """

    _updaters: Dict[str, ZepGraphMemoryUpdater] = {}
    _lock = threading.Lock()

    @classmethod
    def create_updater(cls, simulation_id: str, graph_id: str) -> ZepGraphMemoryUpdater:
        """
        Create a graph memory updater for a simulation.

        Args:
            simulation_id: Simulation ID
            graph_id: Zep graph ID

        Returns:
            ZepGraphMemoryUpdater instance
        """
        with cls._lock:
            # If one already exists, stop it first
            if simulation_id in cls._updaters:
                cls._updaters[simulation_id].stop()

            updater = ZepGraphMemoryUpdater(graph_id)
            updater.start()
            cls._updaters[simulation_id] = updater

            logger.info(f"Created graph memory updater: simulation_id={simulation_id}, graph_id={graph_id}")
            return updater

    @classmethod
    def get_updater(cls, simulation_id: str) -> Optional[ZepGraphMemoryUpdater]:
        """Return the updater for a simulation"""
        return cls._updaters.get(simulation_id)

    @classmethod
    def stop_updater(cls, simulation_id: str):
        """Stop and remove the updater for a simulation"""
        with cls._lock:
            if simulation_id in cls._updaters:
                cls._updaters[simulation_id].stop()
                del cls._updaters[simulation_id]
                logger.info(f"Stopped graph memory updater: simulation_id={simulation_id}")

    # Flag to prevent stop_all from running twice
    _stop_all_done = False

    @classmethod
    def stop_all(cls):
        """Stop all updaters"""
        # Guard against double invocation
        if cls._stop_all_done:
            return
        cls._stop_all_done = True

        with cls._lock:
            if cls._updaters:
                for simulation_id, updater in list(cls._updaters.items()):
                    try:
                        updater.stop()
                    except Exception as e:
                        logger.error(f"Failed to stop updater: simulation_id={simulation_id}, error={e}")
                cls._updaters.clear()
            logger.info("All graph memory updaters stopped")

    @classmethod
    def get_all_stats(cls) -> Dict[str, Dict[str, Any]]:
        """Return statistics for all updaters"""
        return {
            sim_id: updater.get_stats()
            for sim_id, updater in cls._updaters.items()
        }
