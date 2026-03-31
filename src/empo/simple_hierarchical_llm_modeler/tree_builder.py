"""
Recursive tree builder that constructs a state-action-reaction trajectory tree
by querying an LLM at each expansion step.

The tree has the following node types in each cycle:

    state ──(robot action)──> state_robotaction
    state_robotaction ──(humans reaction)──> state_humansreaction
    state_humansreaction ──(consequence/observation)──> next_state  [with probability]

A *depth* of ``n_steps`` means the root state is expanded through ``n_steps``
full cycles of (robot-action, humans-reaction, consequence).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from empo.simple_hierarchical_llm_modeler.llm_connector import LLMConnector
from empo.simple_hierarchical_llm_modeler.prompts import (
    consequences_prompt,
    empowerment_prompt,
    humans_reactions_prompt,
    robot_actions_prompt,
)

LOG = logging.getLogger(__name__)

_FALLBACK_ACTION = [{"action": "do nothing", "rationale": "fallback"}]
_FALLBACK_REACTION = [{"reaction": "do nothing", "rationale": "fallback"}]
_FALLBACK_CONSEQUENCE = [
    {
        "consequence": "nothing notable happens",
        "probability": 1.0,
        "rationale": "fallback",
    }
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TreeNode:
    """A single node in the trajectory tree.

    Attributes:
        history: Ordered list of event descriptions leading to this node.
        node_type: One of ``"state"``, ``"robotaction"``, ``"humansreaction"``.
        depth: Current depth measured in full (robot-action, humans-reaction,
            consequence) cycles completed so far.
        children: List of ``(label, probability, child_node)`` triples.
            ``probability`` is 1.0 for robot-action and humans-reaction edges
            and equals the LLM-estimated probability for consequence edges.
        empowerment_estimate: For terminal-depth state nodes, the LLM-estimated
            number of meaningfully different futures.
    """

    history: List[str]
    node_type: str  # "state", "robotaction", "humansreaction"
    depth: int = 0
    children: List[Tuple[str, float, "TreeNode"]] = field(default_factory=list)
    empowerment_estimate: Optional[float] = None


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def _parse_json_list(text: str) -> List[Dict[str, Any]]:
    """Robustly extract the first JSON list from *text*."""
    # Try to find JSON array in the text
    start = text.find("[")
    if start == -1:
        raise ValueError(f"No JSON list found in LLM response: {text!r}")
    # Find matching closing bracket
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    break
    raise ValueError(f"Could not parse JSON list from LLM response: {text!r}")


def _parse_json_object(text: str) -> Dict[str, Any]:
    """Robustly extract the first JSON object from *text*."""
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in LLM response: {text!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    break
    raise ValueError(f"Could not parse JSON object from LLM response: {text!r}")


# ---------------------------------------------------------------------------
# Tree builder
# ---------------------------------------------------------------------------


def build_tree(
    llm: LLMConnector,
    initial_state_description: str,
    n_steps: int = 2,
    n_robotactions: int = 3,
    n_humansreactions: int = 3,
    n_consequences: int = 2,
    higher_level_context: Optional[str] = None,
) -> TreeNode:
    """Build a trajectory tree by recursively querying the LLM.

    Args:
        llm: LLM connector for making queries.
        initial_state_description: Natural-language description of the
            starting situation.
        n_steps: Maximum number of full (action, reaction, consequence)
            cycles to expand.
        n_robotactions: Number of distinct robot actions per state.
        n_humansreactions: Number of distinct human reactions per robot action.
        n_consequences: Number of distinct consequences per human reaction.
        higher_level_context: Optional higher-level context text for
            hierarchical mode.

    Returns:
        The root :class:`TreeNode` of the constructed tree.
    """
    root = TreeNode(history=[], node_type="state", depth=0)
    _expand_state(
        llm=llm,
        node=root,
        initial_state=initial_state_description,
        n_steps=n_steps,
        n_robotactions=n_robotactions,
        n_humansreactions=n_humansreactions,
        n_consequences=n_consequences,
        higher_level_context=higher_level_context,
    )
    return root


def _expand_state(
    llm: LLMConnector,
    node: TreeNode,
    initial_state: str,
    n_steps: int,
    n_robotactions: int,
    n_humansreactions: int,
    n_consequences: int,
    higher_level_context: Optional[str],
) -> None:
    """Expand a *state* node by generating robot actions (or terminal estimates)."""
    assert node.node_type == "state"

    if node.depth >= n_steps:
        # Terminal depth – ask for empowerment estimate
        prompt = empowerment_prompt(initial_state, node.history, higher_level_context)
        raw = llm.query(prompt)
        try:
            data = _parse_json_object(raw)
            node.empowerment_estimate = float(data.get("estimate", 1.0))
        except (ValueError, TypeError):
            LOG.warning("Failed to parse empowerment estimate, defaulting to 1.0")
            node.empowerment_estimate = 1.0
        return

    # Ask for robot actions
    prompt = robot_actions_prompt(
        initial_state, node.history, n_robotactions, higher_level_context
    )
    raw = llm.query(prompt)
    try:
        actions = _parse_json_list(raw)
    except ValueError:
        LOG.warning("Failed to parse robot actions; creating single fallback action")
        actions = list(_FALLBACK_ACTION)

    if not actions:
        LOG.warning(
            "Parsed robot actions but received empty list; creating single fallback action"
        )
        actions = list(_FALLBACK_ACTION)

    for act in actions:
        action_desc = act.get("action", "unknown action")
        child = TreeNode(
            history=node.history + [f"Robot: {action_desc}"],
            node_type="robotaction",
            depth=node.depth,
        )
        node.children.append((action_desc, 1.0, child))
        _expand_robotaction(
            llm=llm,
            node=child,
            initial_state=initial_state,
            n_steps=n_steps,
            n_humansreactions=n_humansreactions,
            n_consequences=n_consequences,
            higher_level_context=higher_level_context,
            n_robotactions=n_robotactions,
        )


def _expand_robotaction(
    llm: LLMConnector,
    node: TreeNode,
    initial_state: str,
    n_steps: int,
    n_humansreactions: int,
    n_consequences: int,
    higher_level_context: Optional[str],
    n_robotactions: int,
) -> None:
    """Expand a *robotaction* node by generating humans' reactions."""
    assert node.node_type == "robotaction"

    prompt = humans_reactions_prompt(
        initial_state, node.history, n_humansreactions, higher_level_context
    )
    raw = llm.query(prompt)
    try:
        reactions = _parse_json_list(raw)
    except ValueError:
        LOG.warning("Failed to parse human reactions; creating single fallback")
        reactions = list(_FALLBACK_REACTION)

    if not reactions:
        LOG.warning(
            "Parsed human reactions but received empty list; creating single fallback"
        )
        reactions = list(_FALLBACK_REACTION)

    for react in reactions:
        reaction_desc = react.get("reaction", "unknown reaction")
        child = TreeNode(
            history=node.history + [f"Humans: {reaction_desc}"],
            node_type="humansreaction",
            depth=node.depth,
        )
        node.children.append((reaction_desc, 1.0, child))
        _expand_humansreaction(
            llm=llm,
            node=child,
            initial_state=initial_state,
            n_steps=n_steps,
            n_consequences=n_consequences,
            higher_level_context=higher_level_context,
            n_robotactions=n_robotactions,
            n_humansreactions=n_humansreactions,
        )


def _expand_humansreaction(
    llm: LLMConnector,
    node: TreeNode,
    initial_state: str,
    n_steps: int,
    n_consequences: int,
    higher_level_context: Optional[str],
    n_robotactions: int,
    n_humansreactions: int,
) -> None:
    """Expand a *humansreaction* node by generating probabilistic consequences."""
    assert node.node_type == "humansreaction"

    prompt = consequences_prompt(
        initial_state, node.history, n_consequences, higher_level_context
    )
    raw = llm.query(prompt)
    try:
        consequences = _parse_json_list(raw)
    except ValueError:
        LOG.warning("Failed to parse consequences; creating single fallback")
        consequences = list(_FALLBACK_CONSEQUENCE)

    if not consequences:
        LOG.warning(
            "Parsed consequences but received empty list; creating single fallback"
        )
        consequences = list(_FALLBACK_CONSEQUENCE)

    # Normalize probabilities so they sum to 1
    probs = [float(c.get("probability", 1.0)) for c in consequences]
    total = sum(probs)
    if total <= 0:
        probs = [1.0 / len(consequences)] * len(consequences)
    else:
        probs = [p / total for p in probs]

    for cons, prob in zip(consequences, probs):
        cons_desc = cons.get("consequence", "unknown consequence")
        child = TreeNode(
            history=node.history + [f"Observation: {cons_desc}"],
            node_type="state",
            depth=node.depth + 1,
        )
        node.children.append((cons_desc, prob, child))
        # Recurse into the new state node
        _expand_state(
            llm=llm,
            node=child,
            initial_state=initial_state,
            n_steps=n_steps,
            n_robotactions=n_robotactions,
            n_humansreactions=n_humansreactions,
            n_consequences=n_consequences,
            higher_level_context=higher_level_context,
        )


# ---------------------------------------------------------------------------
# Utility: count nodes
# ---------------------------------------------------------------------------


def count_nodes(root: TreeNode) -> int:
    """Return the total number of nodes in the tree rooted at *root*."""
    total = 1
    for _, _, child in root.children:
        total += count_nodes(child)
    return total


def collect_leaves(root: TreeNode) -> List[TreeNode]:
    """Return all leaf nodes (no children) in the tree."""
    if not root.children:
        return [root]
    leaves: List[TreeNode] = []
    for _, _, child in root.children:
        leaves.extend(collect_leaves(child))
    return leaves
