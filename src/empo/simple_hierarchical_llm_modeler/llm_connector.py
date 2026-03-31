"""
LLM connector interface for the simple hierarchical LLM modeler.

Defines a lightweight protocol for LLM interaction that can wrap the L2P
BaseLLM or any other LLM backend. Each call uses a fresh context (no
conversation history), as required by the tree-building algorithm.
"""

from typing import Protocol, runtime_checkable


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
