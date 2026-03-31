"""
LLM connector interface for the simple hierarchical LLM modeler.

Defines a lightweight protocol for LLM interaction that can wrap the L2P
BaseLLM or any other LLM backend. Each call uses a fresh context (no
conversation history), as required by the tree-building algorithm.

:class:`CachedLLMConnector` wraps any connector with a disk-backed cache
so that identical prompts are not re-queried across restarts.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

LOG = logging.getLogger(__name__)


@runtime_checkable
class LLMConnector(Protocol):
    """Protocol for querying an LLM with a single prompt in a fresh context.

    Implementations may wrap the L2P ``BaseLLM``, call an API directly via
    ``ollama`` / ``openai`` / ``anthropic``, or provide canned responses for
    testing.
    """

    def query(self, prompt: str) -> str:
        """Send *prompt* to the LLM and return the raw text response.

        Each call MUST use a fresh context (no conversation history from
        previous calls).
        """
        ...


class L2PConnector:
    """Wraps a vendored L2P ``BaseLLM`` instance to satisfy :class:`LLMConnector`."""

    def __init__(self, base_llm) -> None:
        self._llm = base_llm

    def query(self, prompt: str) -> str:  # noqa: D401
        return self._llm.query(prompt)


class CachedLLMConnector:
    """Disk-backed caching wrapper around any :class:`LLMConnector`.

    Responses are stored as individual JSON files in *cache_dir*, keyed by a
    SHA-256 hash of the prompt text.  On a cache hit the LLM is not called,
    saving tokens and latency.  New responses are written to disk immediately
    so that even a partial run (e.g. interrupted by Ctrl-C) preserves its
    progress.

    Usage::

        raw_llm = OPENAI(model="gemini-2.5-flash", ...)
        llm = CachedLLMConnector(raw_llm, cache_dir="outputs/llm_cache")
        tree = build_tree(llm, ...)
    """

    def __init__(
        self,
        inner: LLMConnector,
        cache_dir: str | os.PathLike = "outputs/llm_cache",
    ) -> None:
        self._inner = inner
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    # ── public helpers ────────────────────────────────────────────────

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def clear(self) -> int:
        """Delete all cached entries.  Returns the number of files removed."""
        count = 0
        for p in self._cache_dir.glob("*.json"):
            p.unlink()
            count += 1
        self._hits = self._misses = 0
        LOG.info("Cleared %d cached LLM responses from %s", count, self._cache_dir)
        return count

    # ── LLMConnector interface ────────────────────────────────────────

    def query(self, prompt: str) -> str:
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_file = self._cache_dir / f"{key}.json"

        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text("utf-8"))
                self._hits += 1
                LOG.debug("Cache hit  [%s] (%d hits so far)", key[:12], self._hits)
                return data["response"]
            except (json.JSONDecodeError, KeyError):
                LOG.warning("Corrupt cache entry %s — re-querying LLM", key[:12])

        response = self._inner.query(prompt)
        self._misses += 1
        LOG.debug("Cache miss [%s] (%d misses so far)", key[:12], self._misses)

        cache_file.write_text(
            json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False),
            "utf-8",
        )
        return response
