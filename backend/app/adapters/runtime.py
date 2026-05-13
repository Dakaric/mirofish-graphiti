"""Async-to-sync bridge for Graphiti (which is async-native) inside a sync Flask backend.

A dedicated daemon thread owns a single event loop that lives for the entire
process. Graphiti's clients (Neo4j driver, async OpenAI client) bind to this
loop, so all coroutines must be dispatched here via run_async / fire_and_forget.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from typing import Any, Coroutine

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_start_lock = threading.Lock()


def _ensure_loop() -> asyncio.AbstractEventLoop:
    global _loop, _thread
    if _loop is not None:
        return _loop
    with _start_lock:
        if _loop is not None:
            return _loop
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=loop.run_forever,
            daemon=True,
            name="mirofish-graphiti-loop",
        )
        thread.start()
        _loop = loop
        _thread = thread
    return _loop


def run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    loop = _ensure_loop()
    future: Future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


def fire_and_forget(coro: Coroutine[Any, Any, Any]) -> Future:
    loop = _ensure_loop()
    return asyncio.run_coroutine_threadsafe(coro, loop)
