"""
PDDL-to-WorldModel converter.

Converts MA-PDDL domain + problem specifications into a WorldModel instance
that implements get_state(), set_state(), and transition_probabilities().

State representation:
    A frozenset of ground predicate atom tuples that are true.
    e.g. frozenset({("at", "robot", "kitchen"), ("holding", "robot", "cup")})

Action representation:
    Each agent has a set of grounded actions. The joint action space is the
    Cartesian product of per-agent grounded action sets.

Transition function:
    For each joint action profile:
    1. Check preconditions for each agent's action
    2. If preconditions fail, the action becomes a no-op
    3. Apply effects with concurrent action resolution
    4. Return list of (probability, successor_state) tuples
"""

import itertools
import logging
import re
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np

from empo.world_model import WorldModel
from empo.llm_world_model.types import ConcurrentEffect, MADomainSpec, MATaskSpec

LOG = logging.getLogger(__name__)

# Type alias for ground atoms
GroundAtom = Tuple[str, ...]
PddlState = FrozenSet[GroundAtom]


class GroundAction:
    """A fully grounded (instantiated) PDDL action."""

    __slots__ = ("name", "agent", "bindings", "precondition_atoms", "add_effects", "del_effects")

    def __init__(
        self,
        name: str,
        agent: str,
        bindings: Dict[str, str],
        precondition_atoms: List[Tuple[bool, GroundAtom]],
        add_effects: List[GroundAtom],
        del_effects: List[GroundAtom],
    ):
        self.name = name
        self.agent = agent
        self.bindings = bindings
        # List of (positive?, atom) tuples for preconditions
        self.precondition_atoms = precondition_atoms
        self.add_effects = add_effects
        self.del_effects = del_effects

    def __repr__(self) -> str:
        binding_str = ", ".join(f"{k}={v}" for k, v in self.bindings.items())
        return f"GroundAction({self.name}({binding_str}), agent={self.agent})"


def _parse_atom(expr: str) -> Optional[GroundAtom]:
    """Parse a PDDL atom string like '(at robot kitchen)' into a tuple.

    Returns None if the expression is not a valid atom.
    """
    expr = expr.strip()
    if not expr.startswith("(") or not expr.endswith(")"):
        return None
    inner = expr[1:-1].strip()
    parts = inner.split()
    if not parts:
        return None
    return tuple(parts)


def _parse_pddl_expression(expr: str) -> List[Tuple[bool, GroundAtom]]:
    """Parse a PDDL precondition/effect expression into a list of (positive?, atom).

    Handles:
    - Simple atoms: (at robot kitchen) → [(True, ("at", "robot", "kitchen"))]
    - Negated atoms: (not (at robot kitchen)) → [(False, ("at", "robot", "kitchen"))]
    - Conjunctions: (and (at robot kitchen) (not (holding robot cup))) → both
    - Nested and/not

    Returns list of (is_positive, atom_tuple) pairs.
    """
    expr = expr.strip()
    if not expr:
        return []

    results: List[Tuple[bool, GroundAtom]] = []

    # Tokenize by finding balanced parenthesized sub-expressions
    def _find_subexpressions(s: str) -> List[str]:
        """Find top-level parenthesized sub-expressions."""
        subexprs = []
        depth = 0
        start = None
        for i, ch in enumerate(s):
            if ch == "(":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and start is not None:
                    subexprs.append(s[start : i + 1])
                    start = None
        return subexprs

    subexprs = _find_subexpressions(expr)
    if not subexprs:
        return []

    for subexpr in subexprs:
        inner = subexpr[1:-1].strip()
        parts = inner.split(None, 1)
        if not parts:
            continue

        keyword = parts[0].lower()

        if keyword == "and":
            # Recurse into conjunction body
            body = parts[1] if len(parts) > 1 else ""
            results.extend(_parse_pddl_expression(body))
        elif keyword == "not":
            # Negate the inner atom
            body = parts[1] if len(parts) > 1 else ""
            inner_subs = _find_subexpressions(body)
            for isub in inner_subs:
                atom = _parse_atom(isub)
                if atom:
                    results.append((False, atom))
        else:
            # Regular atom
            atom = _parse_atom(subexpr)
            if atom:
                results.append((True, atom))

    return results


def _substitute_bindings(
    atoms: List[Tuple[bool, GroundAtom]], bindings: Dict[str, str]
) -> List[Tuple[bool, GroundAtom]]:
    """Substitute parameter bindings into atom tuples."""
    result = []
    for positive, atom in atoms:
        grounded = []
        for part in atom:
            if part.startswith("?") and part in bindings:
                grounded.append(bindings[part])
            elif part.startswith("?") and part.lstrip("?") in bindings:
                grounded.append(bindings[part.lstrip("?")])
            else:
                grounded.append(part)
        result.append((positive, tuple(grounded)))
    return result


class PddlWorldModel(WorldModel):
    """
    A WorldModel backed by MA-PDDL domain and problem specifications.

    Implements the WorldModel interface (get_state, set_state,
    transition_probabilities) using PDDL semantics.
    """

    def __init__(
        self,
        domain: MADomainSpec,
        task: MATaskSpec,
        max_steps: int = 50,
    ):
        """
        Initialize from MA-PDDL specs.

        Performs eager action grounding (instantiating parameterised actions
        with concrete objects) and builds the action space.
        """
        super().__init__()

        self._domain = domain
        self._task = task
        self._max_steps = max_steps

        # Build the set of all ground atoms that can exist
        self._all_ground_atoms = self._enumerate_ground_atoms()
        self._atom_to_index = {a: i for i, a in enumerate(self._all_ground_atoms)}
        self._num_atoms = len(self._all_ground_atoms)

        # Parse initial state
        self._initial_state = self._parse_initial_state(task.initial_state)

        # Current state
        self._current_state: PddlState = self._initial_state
        self._step_count = 0

        # Ground actions per agent
        self._agent_names = [a.name for a in domain.agents]
        self._agent_ground_actions: Dict[str, List[GroundAction]] = {}
        for agent in domain.agents:
            agent_actions = domain.agent_actions.get(agent.name, [])
            self._agent_ground_actions[agent.name] = self._ground_actions(
                agent.name, agent_actions, task.objects
            )

        # Per-agent action counts (for action space)
        # Add 1 for no-op per agent
        self._agent_action_counts = {
            name: len(actions) + 1  # +1 for no-op
            for name, actions in self._agent_ground_actions.items()
        }

        # State-specific action overrides (for prescribed-action mode)
        self._state_actions: Dict[PddlState, Dict[str, List[int]]] = {}

        # Explicit transition overrides (for incremental mode)
        self._explicit_transitions: Dict[
            Tuple[PddlState, Tuple[int, ...]], List[Tuple[float, PddlState]]
        ] = {}

        # Known states set
        self._known_states: Set[PddlState] = {self._initial_state}

        # Concurrent effect rules
        self._concurrent_effects = domain.concurrent_effects

        # Set up Gymnasium spaces
        n_agents = len(self._agent_names)
        if n_agents == 0:
            n_agents = 1
            self._agent_names = ["default"]
            self._agent_ground_actions["default"] = []
            self._agent_action_counts["default"] = 1

        max_actions = max(self._agent_action_counts.values()) if self._agent_action_counts else 1

        self.observation_space = gym.spaces.MultiBinary(self._num_atoms)
        self.action_space = gym.spaces.Discrete(max_actions)
        self.agents = list(range(n_agents))

    # --- WorldModel interface ---

    def get_state(self) -> Any:
        """Return current state as (step_count, frozenset_of_ground_atoms)."""
        return (self._step_count, self._current_state)

    def set_state(self, state: Any) -> None:
        """Restore to the given state."""
        self._step_count, self._current_state = state

    def transition_probabilities(
        self,
        state: Any,
        actions: List[int],
    ) -> Optional[List[Tuple[float, Any]]]:
        """
        Compute transition probabilities for a joint action profile.

        Steps:
        1. Decode action indices to grounded actions
        2. Check preconditions per agent (failed → no-op)
        3. Collect add/delete effects per agent
        4. Resolve concurrent effects using ConcurrentEffect rules
        5. Return [(prob, successor_state), ...]
        """
        step_count, current_atoms = state

        # Check terminal
        if step_count >= self._max_steps:
            return None

        # Check explicit transitions first
        action_tuple = tuple(actions)
        key = (current_atoms, action_tuple)
        if key in self._explicit_transitions:
            results = []
            for prob, succ_atoms in self._explicit_transitions[key]:
                results.append((prob, (step_count + 1, succ_atoms)))
            return results

        # Decode actions per agent
        agent_effects: Dict[str, Tuple[Set[GroundAtom], Set[GroundAtom]]] = {}

        for i, agent_name in enumerate(self._agent_names):
            if i >= len(actions):
                continue

            action_idx = actions[i]
            ground_actions = self._agent_ground_actions.get(agent_name, [])

            # action_idx == 0 is no-op, 1..N are actual actions
            if action_idx == 0 or action_idx > len(ground_actions):
                agent_effects[agent_name] = (set(), set())
                continue

            gaction = ground_actions[action_idx - 1]

            # Check preconditions
            if self._check_preconditions(gaction, current_atoms):
                add_set = set(gaction.add_effects)
                del_set = set(gaction.del_effects)
                agent_effects[agent_name] = (add_set, del_set)
            else:
                agent_effects[agent_name] = (set(), set())

        # Resolve concurrent effects
        successor_atoms = self._resolve_concurrent(agent_effects, current_atoms)
        successor_state = (step_count + 1, successor_atoms)

        return [(1.0, successor_state)]

    # --- Gymnasium interface ---

    @property
    def human_agent_indices(self) -> List[int]:
        """Return indices of agents whose type contains 'human' or 'person'."""
        indices = []
        for i, agent in enumerate(self._domain.agents):
            if any(
                kw in agent.agent_type.lower()
                for kw in ("human", "person", "player", "user")
            ):
                indices.append(i)
        return indices if indices else [0]

    def reset(self, seed=None, options=None):
        """Reset to initial state."""
        super().reset(seed=seed)
        self._current_state = self._initial_state
        self._step_count = 0
        obs = self._state_to_obs(self._current_state)
        return obs, {}

    def step(self, actions):
        """Execute one step using transition_probabilities()."""
        if isinstance(actions, (int, np.integer)):
            actions = [int(actions)] * len(self._agent_names)
        elif not isinstance(actions, list):
            actions = list(actions)

        state = self.get_state()
        transitions = self.transition_probabilities(state, actions)

        if transitions is None:
            obs = self._state_to_obs(self._current_state)
            return obs, 0.0, True, False, {}

        # Sample from transitions
        probs = [t[0] for t in transitions]
        idx = self.np_random.choice(len(transitions), p=probs)
        _, successor_state = transitions[idx]

        self.set_state(successor_state)

        obs = self._state_to_obs(self._current_state)
        terminated = False
        truncated = self._step_count >= self._max_steps

        return obs, 0.0, terminated, truncated, {}

    # --- Incremental model extensions ---

    def add_state(
        self,
        state_atoms: PddlState,
        agent_action_indices: Optional[Dict[str, List[int]]] = None,
    ):
        """Register a new state in the model's known state space."""
        self._known_states.add(state_atoms)
        if agent_action_indices is not None:
            self._state_actions[state_atoms] = agent_action_indices

    def add_transition(
        self,
        source_atoms: PddlState,
        action_profile: Tuple[int, ...],
        outcomes: List[Tuple[float, PddlState]],
    ):
        """Add a transition edge from source_state under action_profile."""
        key = (source_atoms, action_profile)
        self._explicit_transitions[key] = outcomes
        # Track all successor states as known
        for _, succ in outcomes:
            self._known_states.add(succ)

    def add_action_schema(self, agent_name: str, action_schema: dict):
        """Add a new action schema and re-ground."""
        if agent_name not in self._domain.agent_actions:
            self._domain.agent_actions[agent_name] = []
        self._domain.agent_actions[agent_name].append(action_schema)

        # Re-ground actions for this agent
        self._agent_ground_actions[agent_name] = self._ground_actions(
            agent_name,
            self._domain.agent_actions[agent_name],
            self._task.objects,
        )
        self._agent_action_counts[agent_name] = (
            len(self._agent_ground_actions[agent_name]) + 1
        )

    def add_predicate(self, predicate: dict):
        """Add a new predicate to the domain.

        Existing states are not affected (new predicate defaults to False).
        """
        self._domain.predicates.append(predicate)
        # Re-enumerate ground atoms
        self._all_ground_atoms = self._enumerate_ground_atoms()
        self._atom_to_index = {a: i for i, a in enumerate(self._all_ground_atoms)}
        self._num_atoms = len(self._all_ground_atoms)
        self.observation_space = gym.spaces.MultiBinary(self._num_atoms)

    def add_object(self, name: str, obj_type: str):
        """Add a new object to the task."""
        self._task.objects[name] = obj_type
        # Re-enumerate ground atoms and re-ground actions
        self._all_ground_atoms = self._enumerate_ground_atoms()
        self._atom_to_index = {a: i for i, a in enumerate(self._all_ground_atoms)}
        self._num_atoms = len(self._all_ground_atoms)
        self.observation_space = gym.spaces.MultiBinary(self._num_atoms)

        for agent_name in self._agent_names:
            agent_actions = self._domain.agent_actions.get(agent_name, [])
            self._agent_ground_actions[agent_name] = self._ground_actions(
                agent_name, agent_actions, self._task.objects
            )
            self._agent_action_counts[agent_name] = (
                len(self._agent_ground_actions[agent_name]) + 1
            )

    def set_state_actions(self, state_atoms: PddlState, agent_actions: Dict[str, List[int]]):
        """Override valid actions in a specific state."""
        self._state_actions[state_atoms] = agent_actions

    @property
    def is_frontier(self) -> bool:
        """True if this model has states with unknown transitions."""
        return bool(self.frontier_states)

    @property
    def frontier_states(self) -> List[PddlState]:
        """States with prescribed actions but unknown successor transitions."""
        frontiers = []
        for state_atoms in self._known_states:
            if state_atoms in self._state_actions:
                # Has specific actions — check if all transitions are known
                has_unknown = False
                for agent_name, action_indices in self._state_actions[state_atoms].items():
                    for aidx in action_indices:
                        # Check if transition is explicitly defined
                        # Create action profile with this action
                        for combo in self._action_profiles_with(agent_name, aidx):
                            key = (state_atoms, combo)
                            if key not in self._explicit_transitions:
                                has_unknown = True
                                break
                        if has_unknown:
                            break
                    if has_unknown:
                        break
                if has_unknown:
                    frontiers.append(state_atoms)
        return frontiers

    # --- Internal helpers ---

    def _enumerate_ground_atoms(self) -> List[GroundAtom]:
        """Enumerate all possible ground atoms from predicates and objects."""
        atoms: List[GroundAtom] = []
        objects_by_type: Dict[str, List[str]] = {}
        for name, typ in self._task.objects.items():
            objects_by_type.setdefault(typ, []).append(name)

        for pred in self._domain.predicates:
            params = pred.get("params", {})
            if not params:
                # Zero-arity predicate
                atoms.append((pred["name"],))
                continue

            # Get parameter types
            param_types = list(params.values())
            param_options = []
            for ptype in param_types:
                candidates = objects_by_type.get(ptype, [])
                if not candidates:
                    # Try parent type or all objects
                    candidates = list(self._task.objects.keys())
                param_options.append(candidates)

            # Generate all groundings
            for combo in itertools.product(*param_options):
                atoms.append((pred["name"],) + combo)

        return atoms

    def _parse_initial_state(self, initial_atoms: List[str]) -> PddlState:
        """Parse initial state strings into a frozenset of ground atoms."""
        state_atoms: Set[GroundAtom] = set()
        for atom_str in initial_atoms:
            atom = _parse_atom(atom_str)
            if atom is not None:
                # Skip function assignments (= ...)
                if atom[0] != "=":
                    state_atoms.add(atom)
        return frozenset(state_atoms)

    def _ground_actions(
        self,
        agent_name: str,
        actions: List[dict],
        objects: Dict[str, str],
    ) -> List[GroundAction]:
        """Instantiate parameterised actions with concrete objects."""
        objects_by_type: Dict[str, List[str]] = {}
        for name, typ in objects.items():
            objects_by_type.setdefault(typ, []).append(name)

        ground_actions: List[GroundAction] = []

        for action in actions:
            params = action.get("params", {})
            precond_str = action.get("preconditions", "")
            effects_str = action.get("effects", "")

            # Parse preconditions and effects templates
            precond_atoms = _parse_pddl_expression(precond_str)
            effect_atoms = _parse_pddl_expression(effects_str)

            if not params:
                # Zero-parameter action
                add_effs = [a for pos, a in effect_atoms if pos]
                del_effs = [a for pos, a in effect_atoms if not pos]
                ground_actions.append(
                    GroundAction(
                        name=action["name"],
                        agent=agent_name,
                        bindings={},
                        precondition_atoms=precond_atoms,
                        add_effects=add_effs,
                        del_effects=del_effs,
                    )
                )
                continue

            # Generate all groundings
            param_names = list(params.keys())
            param_types = list(params.values())
            param_options = []
            for ptype in param_types:
                candidates = objects_by_type.get(ptype, [])
                if not candidates:
                    candidates = list(objects.keys())
                param_options.append(candidates)

            for combo in itertools.product(*param_options):
                bindings = {}
                for pname, val in zip(param_names, combo):
                    bindings[pname] = val
                    bindings[f"?{pname}"] = val

                # Ground preconditions
                grounded_preconds = _substitute_bindings(precond_atoms, bindings)
                # Ground effects
                grounded_effects = _substitute_bindings(effect_atoms, bindings)

                add_effs = [a for pos, a in grounded_effects if pos]
                del_effs = [a for pos, a in grounded_effects if not pos]

                ground_actions.append(
                    GroundAction(
                        name=action["name"],
                        agent=agent_name,
                        bindings=dict(zip(param_names, combo)),
                        precondition_atoms=grounded_preconds,
                        add_effects=add_effs,
                        del_effects=del_effs,
                    )
                )

        return ground_actions

    def _check_preconditions(
        self, action: GroundAction, state_atoms: PddlState
    ) -> bool:
        """Check if all preconditions of a grounded action are satisfied."""
        for positive, atom in action.precondition_atoms:
            if positive and atom not in state_atoms:
                return False
            if not positive and atom in state_atoms:
                return False
        return True

    def _resolve_concurrent(
        self,
        agent_effects: Dict[str, Tuple[Set[GroundAtom], Set[GroundAtom]]],
        state_atoms: PddlState,
    ) -> PddlState:
        """Apply concurrent effect resolution.

        Default (commutative): union add sets, union delete sets,
        add wins over delete (persistence assumption).
        """
        total_add: Set[GroundAtom] = set()
        total_del: Set[GroundAtom] = set()

        for agent_name, (add_set, del_set) in agent_effects.items():
            total_add |= add_set
            total_del |= del_set

        # Apply: start from current state, remove deletes, add adds
        # Add wins over delete (persistence assumption)
        new_atoms = set(state_atoms)
        new_atoms -= total_del - total_add  # Only delete what isn't also added
        new_atoms |= total_add

        return frozenset(new_atoms)

    def _state_to_obs(self, state_atoms: PddlState) -> np.ndarray:
        """Convert a frozenset state to a binary observation vector."""
        obs = np.zeros(self._num_atoms, dtype=np.int8)
        for atom in state_atoms:
            idx = self._atom_to_index.get(atom)
            if idx is not None:
                obs[idx] = 1
        return obs

    def _action_profiles_with(
        self, agent_name: str, action_idx: int
    ) -> List[Tuple[int, ...]]:
        """Generate all joint action profiles with a specific agent action."""
        agent_idx = self._agent_names.index(agent_name)
        other_counts = [
            self._agent_action_counts[n]
            for i, n in enumerate(self._agent_names)
            if i != agent_idx
        ]
        if not other_counts:
            return [(action_idx,)]

        profiles = []
        for combo in itertools.product(*[range(c) for c in other_counts]):
            profile = list(combo)
            profile.insert(agent_idx, action_idx)
            profiles.append(tuple(profile))
        return profiles
