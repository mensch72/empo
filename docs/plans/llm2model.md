# Plan: LLM-based WorldModel Formation

**Status:** Proposed
**Date:** 2026-03-30

---

## Table of Contents

- [1. Overview](#1-overview)
  - [1.1 Motivation](#11-motivation)
  - [1.2 Scope](#12-scope)
  - [1.3 Key Constraints](#13-key-constraints)
- [2. Architecture](#2-architecture)
  - [2.1 High-Level Pipeline](#21-high-level-pipeline)
  - [2.2 Module Map](#22-module-map)
  - [2.3 Relationship to Existing EMPO Modules](#23-relationship-to-existing-empo-modules)
- [3. L2P Analysis and Reuse Strategy](#3-l2p-analysis-and-reuse-strategy)
  - [3.1 L2P Components We Keep As-Is](#31-l2p-components-we-keep-as-is)
  - [3.2 L2P Components We Extend](#32-l2p-components-we-extend)
  - [3.3 L2P Components We Skip](#33-l2p-components-we-skip)
  - [3.4 unified-planning MAPDDLWriter Reuse](#34-unified-planning-mapddlwriter-reuse)
  - [3.5 pddlgymnasium Reuse](#35-pddlgymnasium-reuse)
- [4. Implementation Steps](#4-implementation-steps)
  - [Step 0: Vendor L2P and Verify Baseline](#step-0-vendor-l2p-and-verify-baseline)
  - [Step 1: AgentBuilder](#step-1-agentbuilder)
  - [Step 2: WorldModelDomainBuilder](#step-2-worldmodeldomainbuilder)
  - [Step 3: WorldModelTaskBuilder](#step-3-worldmodeltaskbuilder)
  - [Step 4: MA-PDDL Output](#step-4-ma-pddl-output)
  - [Step 5: PDDL-to-WorldModel Converter](#step-5-pddl-to-worldmodel-converter)
  - [Step 6: Prompt Templates](#step-6-prompt-templates)
  - [Step 7: End-to-End Pipeline and WorldModelBuilder Façade](#step-7-end-to-end-pipeline-and-worldmodelbuilder-façade)
  - [Step 8: Hierarchical World Model Extension](#step-8-hierarchical-world-model-extension)
- [5. Concurrent Action Semantics](#5-concurrent-action-semantics)
  - [5.1 The Problem](#51-the-problem)
  - [5.2 Proposed Strategy](#52-proposed-strategy)
  - [5.3 Representation in MA-PDDL](#53-representation-in-ma-pddl)
- [6. Design Decisions Requiring Human Input](#6-design-decisions-requiring-human-input)
- [7. Dependency Graph](#7-dependency-graph)
- [8. Risk Analysis](#8-risk-analysis)

---

## 1. Overview

### 1.1 Motivation

EMPO computes empowerment — the aggregate power of agents — over stochastic game
forms defined by `WorldModel`. Today, every `WorldModel` is hand-coded (MultiGrid,
Transport). We want to let an LLM construct a `WorldModel` from a natural language
scene description so that empowerment can be computed over semantically meaningful,
human-interpretable state spaces without manual coding.

### 1.2 Scope

Build a `WorldModelBuilder` module that takes a natural language description of a
multi-agent scenario and produces a `WorldModel`:

| Component | Source | What We Produce |
|-----------|--------|-----------------|
| **Agents** I = {1, …, n} | Extracted from scene description | Named agents with types and capabilities |
| **States** S | Typed predicates (as in PDDL) | Shared predicate-based state space |
| **Per-agent actions** {A_i} | MA-PDDL action schemas with ownership | Parameterised actions with preconditions and effects |
| **Transition function** T | Joint effect composition + PPDDL `:probabilistic-effects` | S × A_1 × … × A_n → Δ(S) |

**Crucially, there are NO goals and NO reward functions.** We only need the dynamics.
Goals and rewards are handled externally by EMPO's `PossibleGoal` and
`PossibleGoalGenerator` abstractions.

### 1.3 Key Constraints

- L2P is vendored in `vendor/l2p/` — we can modify it freely
- LLM calls are expensive and nondeterministic — all prompts must use structured
  output (JSON/PDDL) with validation schemas and retry logic
- The output `WorldModel` must implement `get_state()`, `set_state()`, and
  `transition_probabilities()` so that EMPO's backward induction and learning-based
  algorithms can operate on it
- Multi-agent semantics require concurrent action composition, which standard PDDL
  does not support — this is the hardest design challenge

---

## 2. Architecture

### 2.1 High-Level Pipeline

```
┌─────────────────────────────────┐
│  Natural Language Scene Desc    │
│  "A household with a robot     │
│   and a human. The robot can   │
│   move between rooms..."       │
└───────────────┬─────────────────┘
                │
     ┌──────────▼──────────┐
     │   1. AgentBuilder   │  Identify agents, types, capabilities
     └──────────┬──────────┘
                │ agents: List[AgentSpec]
     ┌──────────▼──────────────────────┐
     │ 2. WorldModelDomainBuilder      │  Extract shared predicates,
     │    (per-agent action schemas,   │  per-agent actions,
     │     concurrent effects)         │  interaction rules
     └──────────┬──────────────────────┘
                │ domain: MADomainSpec
     ┌──────────▼──────────────────────┐
     │ 3. WorldModelTaskBuilder        │  Extract objects,
     │    (objects + initial state,    │  initial predicate values
     │     NO goal)                    │  (no goal state)
     └──────────┬──────────────────────┘
                │ task: MATaskSpec
     ┌──────────▼──────────────────────┐
     │ 4. MA-PDDL Writer              │  Generate MA-PDDL domain
     │    (domain + problem files,     │  and problem files
     │     without :goal)              │  (primary output format)
     └──────────┬──────────────────────┘
                │ .pddl files
     ┌──────────▼──────────────────────┐
     │ 5. PDDL-to-WorldModel          │  Convert PDDL to a
     │    Converter                    │  WorldModel instance with
     │    (gym.Env + state mgmt)       │  get_state/set_state/
     │                                 │  transition_probabilities
     └──────────┬──────────────────────┘
                │
     ┌──────────▼──────────────────────┐
     │ WorldModel (gymnasium.Env)      │  ← ready for EMPO
     │  .get_state()                   │
     │  .set_state()                   │
     │  .transition_probabilities()    │
     └────────────────────────────────-┘
```

### 2.2 Module Map

```
src/empo/llm_world_model/                  # New top-level module
├── __init__.py
├── agent_builder.py                       # Step 1: LLM → agents
├── world_model_domain_builder.py          # Step 2: LLM → MA domain
├── world_model_task_builder.py            # Step 3: LLM → objects + init
├── ma_pddl_writer.py                      # Step 4: specs → MA-PDDL files
├── pddl_world_model.py                    # Step 5: PDDL → WorldModel
├── world_model_builder.py                 # Step 7: end-to-end façade
├── hierarchical_builder.py                # Step 8: hierarchical extension
├── types.py                               # Shared type definitions
│
├── prompts/                               # Step 6: prompt templates
│   ├── identify_agents.txt
│   ├── formalize_predicates.txt
│   ├── formalize_agent_actions.txt
│   ├── identify_concurrent_effects.txt
│   ├── formalize_initial_state.txt
│   └── identify_abstraction_levels.txt
│
└── tests/                                 # or in top-level tests/
    ├── test_agent_builder.py
    ├── test_world_model_domain_builder.py
    ├── test_world_model_task_builder.py
    ├── test_ma_pddl_writer.py
    ├── test_pddl_world_model.py
    ├── test_world_model_builder.py
    └── mock_llm.py                        # Reuse/adapt from L2P
```

### 2.3 Relationship to Existing EMPO Modules

| Existing Module | Relationship |
|-----------------|-------------|
| `empo.world_model.WorldModel` | `PddlWorldModel` (Step 5) **subclasses** this |
| `empo.possible_goal.PossibleGoal` | Consumers of the produced `WorldModel` define goals over it |
| `empo.backward_induction` | Operates on the produced `WorldModel` unchanged |
| `empo.learning_based` | Operates on the produced `WorldModel` unchanged |
| `empo.hierarchical.HierarchicalWorldModel` | Step 8 produces hierarchical models by composing `PddlWorldModel` levels |
| `empo.hierarchical.LevelMapper` | Step 8 implements a `PddlLevelMapper` based on LLM-identified abstraction levels |
| `vendor/l2p/l2p/domain_builder.py` | `WorldModelDomainBuilder` extends `DomainBuilder` |
| `vendor/l2p/l2p/task_builder.py` | `WorldModelTaskBuilder` extends `TaskBuilder` |
| `vendor/l2p/l2p/llm/` | Reused directly for all LLM calls |
| `vendor/l2p/l2p/utils/` | Reused for PDDL types, parsing, validation |

---

## 3. L2P Analysis and Reuse Strategy

### 3.1 L2P Components We Keep As-Is

| Component | File | Reason |
|-----------|------|--------|
| LLM abstraction layer | `l2p/llm/base.py`, `openai.py`, `huggingface.py`, `vllm.py`, `unified.py` | Provider-agnostic LLM interface; no changes needed |
| PDDL type definitions | `l2p/utils/pddl_types.py` | `Predicate`, `Function`, `Action`, `ParameterList` are reusable; we add `AgentSpec` alongside |
| PDDL parser | `l2p/utils/pddl_parser.py` | `parse_types()`, `parse_predicates()`, `parse_action()`, `parse_objects()`, `parse_initial()` all reusable |
| PDDL syntax validator | `l2p/utils/pddl_validator.py` | `SyntaxValidator` with type/predicate/action validation; reusable for MA-PDDL with minor extensions |
| PDDL formatter | `l2p/utils/pddl_format.py` | `format_types_to_string()`, `format_expression()`, `format_actions()` reusable |
| Prompt builder | `l2p/prompt_builder.py` | Template loading and placeholder substitution |
| Feedback builder | `l2p/feedback_builder.py` | LLM/human feedback loop for iterative refinement |
| MockLLM for testing | `l2p/tests/mock_llm.py` | Canned-response testing without API calls |

### 3.2 L2P Components We Extend

| Component | What Changes | Why |
|-----------|-------------|-----|
| `DomainBuilder` | Subclass as `WorldModelDomainBuilder` adding: per-agent action extraction, concurrent effect identification, probabilistic effects support | L2P assumes single-agent PDDL; we need MA-PDDL with ownership and concurrency |
| `TaskBuilder` | Subclass as `WorldModelTaskBuilder` removing goal extraction, adding per-agent object validation | We have no goals; need to validate agent-owned objects |
| `pddl_types.py` | Add `AgentSpec`, `ConcurrentEffect`, `MADomainSpec`, `MATaskSpec` types alongside existing types | New data structures for multi-agent concepts |
| `pddl_validator.py` | Extend `SyntaxValidator` with `validate_agent_actions()`, `validate_concurrent_effects()` | Need MA-PDDL-specific validation |
| `pddl_format.py` | Add `format_ma_domain()`, `format_ma_problem()` for MA-PDDL output | L2P only generates single-agent PDDL |

### 3.3 L2P Components We Skip

| Component | Reason |
|-----------|--------|
| `l2p/utils/pddl_planner.py` (FastDownward) | We don't plan; we only need the dynamics |
| `formalize_goal_state()` in TaskBuilder | No goals in our pipeline |
| `generate_requirements()` auto-detection | We set `:requirements` explicitly (`:multi-agent`, `:probabilistic-effects`, etc.) |
| `paper_reconstructions/` | Example implementations; not needed |

### 3.4 unified-planning MAPDDLWriter Reuse

The unified-planning library's `MAPDDLWriter` handles:
- Per-agent domain/problem file generation (factored output)
- Public vs private fluent separation
- Agent-specific action scoping
- `Dot(agent, fluent)` notation for agent-specific predicates

**Decision needed:** We could either:

**(A) Use unified-planning as a dependency** for MA-PDDL I/O:
- Pros: Production-quality MA-PDDL writer, handles factored/unfactored output
- Cons: Heavy dependency (pulls in entire unified-planning framework), may not
  support `:probabilistic-effects` or goal-free problems natively

**(B) Write a custom lightweight MA-PDDL writer** in `ma_pddl_writer.py`:
- Pros: Minimal dependencies, full control over `:probabilistic-effects` and
  goal-free output, can use L2P's existing `pddl_format.py` as foundation
- Cons: More code to write and maintain

> **Recommendation:** Option (B) — custom lightweight writer. The unified-planning
> framework is designed for planning, not dynamics-only extraction. A custom writer
> that extends L2P's `pddl_format.py` is simpler and gives full control over the
> output format. We can still follow the MA-PDDL file structure from
> unified-planning as a reference for the output format.

### 3.5 pddlgymnasium Reuse

The `pddlgymnasium` library converts PDDL domains/problems into Gymnasium environments.
It supports:
- PDDL 1.2 features (STRIPS, typing, quantifiers, equality, constants)
- Some PPDDL probabilistic effects (for built-in environments)
- Single-agent environments only

**Decision needed:** We could either:

**(A) Adapt pddlgymnasium** as the PDDL-to-WorldModel converter:
- Pros: Existing PDDL-to-Gym conversion, handles observation/action spaces
- Cons: Single-agent only (would need significant extension for multi-agent),
  limited PPDDL support, no `get_state()`/`set_state()`/`transition_probabilities()`

**(B) Write a custom `PddlWorldModel`** that directly interprets the MA-PDDL specs:
- Pros: Full control, native multi-agent support, built-in state management,
  exact `transition_probabilities()` from PDDL semantics
- Cons: More code to write

> **Recommendation:** Option (B) — custom `PddlWorldModel`. The state management
> requirements (`get_state`/`set_state`/`transition_probabilities`) are EMPO-specific
> and would require deep changes to pddlgymnasium. A purpose-built converter that
> works directly with our `MADomainSpec` and `MATaskSpec` types is cleaner.
> However, pddlgymnasium's source code is a useful reference for PDDL-to-Gym
> conversion patterns (especially action grounding and predicate evaluation).

---

## 4. Implementation Steps

### Step 0: Vendor L2P and Verify Baseline

**Description:** Set up the vendored L2P with working imports and verify that the
original L2P pipeline still works with a minimal test.

**Files to create/modify:**
- `vendor/l2p/` — already cloned (Step 1 of the issue)
- `VENDOR.md` — add L2P section (already done)
- `Dockerfile` / `docker-compose.yml` — add `vendor/l2p` to PYTHONPATH
- `tests/test_l2p_baseline.py` — minimal test that L2P imports work and
  `DomainBuilder`/`TaskBuilder` can be instantiated with `MockLLM`

**Dependencies:** None

**Estimated complexity:** S (Small)

**Acceptance criteria:**
- `from l2p import DomainBuilder, TaskBuilder, OPENAI` works
- `from l2p.utils.pddl_types import Predicate, Action` works
- Baseline test passes: `DomainBuilder().formalize_types(model=MockLLM(), ...)`
  returns expected output with canned LLM response
- L2P's own tests pass: `python -m pytest vendor/l2p/tests/`

---

### Step 1: AgentBuilder

**Description:** New module that prompts the LLM to identify distinct agents in a
natural language scene description. For each agent, extracts: name, type,
capabilities (natural language), and action ownership hints.

**Files to create:**
- `src/empo/llm_world_model/agent_builder.py`
- `src/empo/llm_world_model/types.py`
- `src/empo/llm_world_model/prompts/identify_agents.txt`
- `tests/test_agent_builder.py`

**Key types (in `types.py`):**
```python
@dataclass
class AgentSpec:
    """Specification of an agent extracted from natural language."""
    name: str                        # e.g. "robot", "human"
    agent_type: str                  # e.g. "robot", "person"
    capabilities: List[str]          # NL descriptions of what the agent can do
    action_hints: List[str]          # Candidate action names (NL)

@dataclass
class ConcurrentEffect:
    """Describes how two simultaneous actions interact."""
    agent_a: str                     # Name of first agent
    action_a: str                    # Action name of first agent
    agent_b: str                     # Name of second agent
    action_b: str                    # Action name of second agent
    effect_type: str                 # "commutative" | "conflicting" | "synergistic"
    resolution: str                  # NL description of what happens
    pddl_effect: Optional[str]      # PDDL effect override (if non-commutative)

@dataclass
class MADomainSpec:
    """Multi-agent PDDL domain specification (no goals)."""
    name: str
    types: Dict[str, str]
    constants: Dict[str, str]
    predicates: List[Predicate]
    functions: List[Function]
    agents: List[AgentSpec]
    agent_actions: Dict[str, List[Action]]  # agent_name → actions
    concurrent_effects: List[ConcurrentEffect]
    requirements: List[str]

@dataclass
class MATaskSpec:
    """Multi-agent PDDL task specification (no goals)."""
    name: str
    domain_name: str
    objects: Dict[str, str]
    initial_state: List[Dict]
    agent_objects: Dict[str, List[str]]  # agent_name → owned objects
```

**`AgentBuilder` class:**
```python
class AgentBuilder:
    """Extract agent specifications from natural language scene descriptions."""

    def identify_agents(
        self,
        model: BaseLLM,
        scene_desc: str,
        prompt_template: str,
        max_retries: int = 3,
    ) -> Tuple[List[AgentSpec], str]:
        """
        Prompt the LLM to identify agents in the scene description.

        Returns:
            (agents, llm_raw_output)
        """
        ...
```

**Prompt template (`identify_agents.txt`):**

The prompt should instruct the LLM to output structured JSON:
```
Given the following scene description, identify all distinct agents
(entities that can take actions and affect the world state).

Scene description:
{scene_desc}

For each agent, provide:
- name: a short identifier (lowercase, no spaces)
- agent_type: the type/category of the agent
- capabilities: list of things the agent can do
- action_hints: list of candidate action names

Output as JSON array:
### AGENTS
```json
[
  {
    "name": "robot",
    "agent_type": "robot",
    "capabilities": ["move between rooms", "pick up objects", "place objects"],
    "action_hints": ["move", "pick_up", "place"]
  },
  ...
]
```
```

**Dependencies:** Step 0

**Estimated complexity:** S (Small)

**Acceptance criteria:**
- `AgentBuilder().identify_agents(MockLLM(...), scene_desc, template)` returns
  a list of `AgentSpec` objects
- JSON output is validated against schema
- Test covers: 2-agent scenario, error handling for malformed LLM output, retry logic

---

### Step 2: WorldModelDomainBuilder

**Description:** Extends L2P's `DomainBuilder` to support multi-agent domain
extraction. The key additions are: per-agent action schemas with ownership,
concurrent effect identification, and optional probabilistic effects.

**Files to create/modify:**
- `src/empo/llm_world_model/world_model_domain_builder.py`
- `src/empo/llm_world_model/prompts/formalize_predicates.txt`
- `src/empo/llm_world_model/prompts/formalize_agent_actions.txt`
- `src/empo/llm_world_model/prompts/identify_concurrent_effects.txt`
- `tests/test_world_model_domain_builder.py`

**`WorldModelDomainBuilder` class:**
```python
class WorldModelDomainBuilder(DomainBuilder):
    """
    Extends L2P's DomainBuilder for multi-agent world model extraction.

    Differences from base DomainBuilder:
    1. Actions are agent-owned (each action belongs to exactly one agent)
    2. Supports concurrent action effect resolution
    3. Supports :probabilistic-effects (PPDDL)
    4. No goal-related methods
    """

    def formalize_shared_predicates(
        self,
        model: BaseLLM,
        scene_desc: str,
        agents: List[AgentSpec],
        prompt_template: str,
        types: Dict[str, str] = None,
        syntax_validator: SyntaxValidator = None,
        max_retries: int = 3,
    ) -> Tuple[List[Predicate], str, Tuple[bool, str]]:
        """
        Extract typed predicates describing the shared state space.

        Unlike base formalize_predicates(), this emphasises:
        - Predicates describe the SHARED world state (not per-agent)
        - Agent-position predicates like (at ?agent - agent ?room - room)
        - Object-state predicates like (holding ?agent - agent ?obj - object)
        - Environment predicates like (door_open ?door - door)
        """
        ...

    def formalize_agent_actions(
        self,
        model: BaseLLM,
        scene_desc: str,
        agent: AgentSpec,
        prompt_template: str,
        types: Dict[str, str] = None,
        predicates: List[Predicate] = None,
        functions: List[Function] = None,
        syntax_validator: SyntaxValidator = None,
        max_retries: int = 3,
    ) -> Tuple[List[Action], List[Predicate], str, Tuple[bool, str]]:
        """
        Extract action schemas owned by a specific agent.

        Each action is called once per agent. The prompt includes the agent's
        capabilities and the shared predicates. Actions are tagged with the
        owning agent's name.

        Returns:
            (actions, new_predicates, llm_output, validation_info)
        """
        ...

    def identify_concurrent_effects(
        self,
        model: BaseLLM,
        scene_desc: str,
        agents: List[AgentSpec],
        agent_actions: Dict[str, List[Action]],
        predicates: List[Predicate],
        prompt_template: str,
        max_retries: int = 3,
    ) -> Tuple[List[ConcurrentEffect], str]:
        """
        Identify how concurrent actions from different agents interact.

        Prompts the LLM to classify all cross-agent action pairs as:
        - commutative: effects apply independently (default)
        - conflicting: e.g. both agents try to pick up same object → neither succeeds
        - synergistic: combined effect differs from sum of individual effects

        Only non-commutative pairs need explicit resolution rules.
        """
        ...

    def formalize_probabilistic_effects(
        self,
        model: BaseLLM,
        scene_desc: str,
        agent: AgentSpec,
        action: Action,
        prompt_template: str,
        max_retries: int = 3,
    ) -> Tuple[Optional[str], str]:
        """
        Optionally convert deterministic effects to probabilistic effects.

        If the scene description implies stochastic outcomes (e.g. "the robot
        sometimes drops objects"), produce PPDDL :probabilistic-effects syntax.

        Returns:
            (ppddl_effects_str_or_None, llm_output)
        """
        ...

    def build_ma_domain(
        self,
        scene_desc: str,
        agents: List[AgentSpec],
        model: BaseLLM,
        prompt_templates: Dict[str, str],
        enable_probabilistic: bool = False,
        syntax_validator: SyntaxValidator = None,
        max_retries: int = 3,
    ) -> MADomainSpec:
        """
        Orchestrate full multi-agent domain extraction.

        Pipeline:
        1. formalize_types() (inherited from DomainBuilder)
        2. formalize_shared_predicates()
        3. For each agent: formalize_agent_actions()
        4. identify_concurrent_effects()
        5. Optionally: formalize_probabilistic_effects() per action
        6. Assemble MADomainSpec
        """
        ...
```

**Key design decisions for this step:**

1. **Predicate extraction** reuses L2P's `formalize_predicates()` internally but
   wraps it with a prompt that emphasises shared state (no per-agent private state).

2. **Action extraction** calls L2P's `formalize_pddl_action()` once per agent-action
   pair (not once per action). The prompt includes the agent's name, type, and
   capabilities to scope the action appropriately.

3. **Concurrent effects** are identified in a separate LLM call after all actions
   are extracted. The LLM sees all agent-action pairs and classifies cross-agent
   interactions. This is the most novel and challenging part.

4. **Probabilistic effects** are an optional post-processing step. For each action,
   the LLM can identify stochastic outcomes and produce PPDDL syntax.

**Dependencies:** Steps 0, 1

**Estimated complexity:** L (Large)

**Acceptance criteria:**
- `WorldModelDomainBuilder` can extract types, predicates, and per-agent actions
  from a household scene description using MockLLM
- Concurrent effects are correctly classified for a test scenario with known
  conflicts (e.g. both agents picking up same object)
- Output `MADomainSpec` contains all required fields
- Tests cover: 2-agent scenario, empty concurrent effects, conflicting actions,
  probabilistic effects toggle

---

### Step 3: WorldModelTaskBuilder

**Description:** Extends L2P's `TaskBuilder` to extract objects and initial state
only (no goal state). Adds validation that all agent-owned objects are properly
instantiated.

**Files to create:**
- `src/empo/llm_world_model/world_model_task_builder.py`
- `src/empo/llm_world_model/prompts/formalize_initial_state.txt`
- `tests/test_world_model_task_builder.py`

**`WorldModelTaskBuilder` class:**
```python
class WorldModelTaskBuilder(TaskBuilder):
    """
    Extends L2P's TaskBuilder for goal-free world model extraction.

    Differences from base TaskBuilder:
    1. No goal extraction (formalize_goal_state is removed/no-op)
    2. Validates agent objects are instantiated
    3. generate_task() omits :goal section
    """

    def formalize_objects_and_initial(
        self,
        model: BaseLLM,
        scene_desc: str,
        domain: MADomainSpec,
        prompt_template: str,
        syntax_validator: SyntaxValidator = None,
        max_retries: int = 3,
    ) -> Tuple[MATaskSpec, str, Tuple[bool, str]]:
        """
        Extract objects and initial state from scene description.

        Validates:
        - Every agent in domain.agents has a corresponding object
        - All typed objects match domain types
        - Initial predicates use only declared predicates and objects
        """
        ...

    def generate_task(
        self,
        domain_name: str,
        problem_name: str,
        objects: Dict[str, str],
        initial: List[Dict],
        goal: List[Dict] = None,  # Ignored; always empty
    ) -> str:
        """
        Override base to produce PDDL problem WITHOUT :goal section.
        """
        ...
```

**Dependencies:** Steps 0, 1, 2

**Estimated complexity:** M (Medium)

**Acceptance criteria:**
- `WorldModelTaskBuilder` extracts objects and initial state from MockLLM
- Goal section is absent from generated PDDL problem
- Validation catches: missing agent objects, type mismatches, undefined predicates
- Tests cover: valid extraction, missing agent object, invalid predicate in init

---

### Step 4: MA-PDDL Output

**Description:** A lightweight MA-PDDL writer that generates domain and problem files
in MA-PDDL format (without `:goal`). Supports `:multi-agent`,
`:probabilistic-effects`, and optionally `:partial-observability` requirements.

**Files to create:**
- `src/empo/llm_world_model/ma_pddl_writer.py`
- `tests/test_ma_pddl_writer.py`

**`MAPddlWriter` class:**
```python
class MAPddlWriter:
    """
    Writes MA-PDDL domain and problem files from MADomainSpec and MATaskSpec.

    Output format follows the MA-PDDL convention used by unified-planning:
    - Per-agent domain files with agent-owned actions
    - Shared types, predicates, and constants
    - :requirements includes :multi-agent
    - Optionally includes :probabilistic-effects

    Can produce either:
    - Factored output: separate domain/problem per agent
    - Unfactored output: single domain/problem with all agents
    """

    def write_domain(self, domain: MADomainSpec, factored: bool = False) -> Union[str, Dict[str, str]]:
        """Generate MA-PDDL domain file(s)."""
        ...

    def write_problem(self, task: MATaskSpec, domain: MADomainSpec, factored: bool = False) -> Union[str, Dict[str, str]]:
        """Generate MA-PDDL problem file(s) without :goal."""
        ...

    def write_files(self, domain: MADomainSpec, task: MATaskSpec, output_dir: str, factored: bool = False) -> List[str]:
        """Write all files to disk and return paths."""
        ...
```

**Reference:** The unified-planning `MAPDDLWriter` handles factored/unfactored
output, public/private fluent separation, and agent-specific action scoping.
We replicate this structure but:
- Omit `:goal` sections
- Add `:probabilistic-effects` support
- Build on L2P's `pddl_format.py` utilities instead of unified-planning internals

**Dependencies:** Steps 2, 3

**Estimated complexity:** M (Medium)

**Acceptance criteria:**
- Generated PDDL files parse correctly (validate with L2P's `SyntaxValidator`)
- Unfactored output contains all agents' actions in one domain file
- Factored output produces per-agent domain/problem files
- No `:goal` section in any output
- `:requirements` includes `:multi-agent` and optionally `:probabilistic-effects`

---

### Step 5: PDDL-to-WorldModel Converter

**Description:** Convert MA-PDDL domain + problem specs into a `WorldModel` instance
that implements `get_state()`, `set_state()`, and `transition_probabilities()`.

This is the critical bridge between the PDDL representation and EMPO's computation
framework.

**Files to create:**
- `src/empo/llm_world_model/pddl_world_model.py`
- `tests/test_pddl_world_model.py`

**`PddlWorldModel` class:**
```python
class PddlWorldModel(WorldModel):
    """
    A WorldModel backed by MA-PDDL domain and problem specifications.

    State representation:
        A frozenset of ground predicate atoms that are true.
        e.g. frozenset({("at", "robot", "kitchen"), ("holding", "robot", "cup"), ...})

    Action representation:
        Each agent has a set of grounded actions derived from the action schemas
        and the current set of objects. The joint action space is the Cartesian
        product of per-agent action sets.

    Transition function:
        For each joint action profile:
        1. Check preconditions for each agent's action
        2. If an agent's preconditions fail, their action becomes a no-op
        3. Apply effects with concurrent action resolution:
           a. Commutative effects: apply independently
           b. Conflicting effects: apply resolution rule from ConcurrentEffect
           c. Probabilistic effects: branch into multiple outcomes with probabilities
        4. Return list of (probability, successor_state) tuples
    """

    def __init__(self, domain: MADomainSpec, task: MATaskSpec):
        """
        Initialize from MA-PDDL specs.

        Performs action grounding (instantiating parameterised actions with
        concrete objects) and builds the action space.
        """
        ...

    # --- WorldModel interface ---

    def get_state(self) -> FrozenSet[Tuple[str, ...]]:
        """Return current set of true ground atoms as a frozenset (hashable)."""
        ...

    def set_state(self, state: FrozenSet[Tuple[str, ...]]) -> None:
        """Restore the world to the given set of true ground atoms."""
        ...

    def transition_probabilities(
        self,
        state: FrozenSet[Tuple[str, ...]],
        actions: List[int],
    ) -> Optional[List[Tuple[float, FrozenSet[Tuple[str, ...]]]]]:
        """
        Compute transition probabilities for a joint action profile.

        Steps:
        1. Decode action indices to grounded action names
        2. Check preconditions per agent (failed → no-op)
        3. Collect add/delete effects per agent
        4. Resolve concurrent effects using ConcurrentEffect rules
        5. If probabilistic: enumerate outcome branches
        6. Return [(prob, successor_state), ...]
        """
        ...

    # --- Gymnasium interface ---

    @property
    def observation_space(self) -> gym.spaces.MultiBinary:
        """Each ground atom is a binary feature."""
        ...

    @property
    def action_space(self) -> gym.spaces.MultiDiscrete:
        """Per-agent action spaces as MultiDiscrete."""
        ...

    def reset(self, seed=None, options=None):
        """Reset to initial state."""
        ...

    def step(self, actions):
        """Execute one step using transition_probabilities()."""
        ...

    # --- Action grounding ---

    def _ground_actions(self, agent_name: str, actions: List[Action], objects: Dict[str, str]) -> List[GroundAction]:
        """Instantiate parameterised actions with concrete objects."""
        ...

    def _evaluate_precondition(self, precondition: str, state: FrozenSet, bindings: Dict) -> bool:
        """Evaluate a PDDL precondition against the current state."""
        ...

    def _apply_effects(self, effects: str, state: FrozenSet, bindings: Dict) -> Tuple[Set, Set]:
        """Compute add and delete sets from PDDL effects expression."""
        ...

    def _resolve_concurrent(self, agent_effects: Dict[str, Tuple[Set, Set]], state: FrozenSet) -> FrozenSet:
        """Apply concurrent effect resolution rules."""
        ...
```

**Key design challenges:**

1. **Action grounding:** Parameterised actions like `move(?agent - robot, ?from - room, ?to - room)`
   must be instantiated with all valid object combinations. This can produce a large
   action space. Consider lazy grounding (only ground applicable actions) vs eager
   grounding (ground all at init).

   > **Recommendation:** Eager grounding at init. The state spaces we target are
   > small enough (LLM-described scenes have ~10-50 objects) that eager grounding
   > is tractable and simplifies `transition_probabilities()`.

2. **Precondition evaluation:** PDDL preconditions are logical expressions with
   `and`, `or`, `not`, `forall`, `exists`. Need a simple evaluator over
   frozenset-based states. Consider using the `pddl` Python package for parsing
   or implement a lightweight recursive evaluator.

3. **Concurrent effect resolution:** When two agents act simultaneously:
   - Default (commutative): union add sets, union delete sets, add wins over delete
   - Conflicting: check `ConcurrentEffect` rules, apply resolution
   - Probabilistic: multiply outcome probabilities

4. **State hashability:** Using `frozenset` of ground atom tuples ensures hashability
   for DAG computation and backward induction.

**Dependencies:** Steps 2, 3

**Estimated complexity:** L (Large)

**Acceptance criteria:**
- `PddlWorldModel` can be instantiated from a simple 2-agent domain
- `get_state()` returns a hashable frozenset
- `set_state(get_state())` is identity
- `transition_probabilities()` returns valid probability distributions (sum to 1.0)
- Precondition failure → no-op (action has no effect)
- Concurrent conflict resolution works for a test case
- `get_dag()` (inherited from WorldModel) produces a valid DAG
- Tests cover: simple domain, concurrent actions, probabilistic effects, terminal states

---

### Step 6: Prompt Templates

**Description:** Create all prompt templates for the pipeline. Each prompt uses
structured output (JSON or PDDL code blocks) with clear section headers that
L2P's parser can handle.

**Files to create:**
- `src/empo/llm_world_model/prompts/identify_agents.txt`
- `src/empo/llm_world_model/prompts/formalize_predicates.txt`
- `src/empo/llm_world_model/prompts/formalize_agent_actions.txt`
- `src/empo/llm_world_model/prompts/identify_concurrent_effects.txt`
- `src/empo/llm_world_model/prompts/formalize_initial_state.txt`
- `src/empo/llm_world_model/prompts/identify_abstraction_levels.txt` (for Step 8)

**Design principles for prompts:**

1. **Structured output:** Every prompt requests output in a specific format
   (JSON array/object, or PDDL code blocks with `###` section headers) that
   can be deterministically parsed.

2. **Validation schemas:** Each prompt includes an example of valid output and
   specifies field types/constraints. This enables L2P's retry-on-parse-failure
   mechanism.

3. **No goals:** Prompts explicitly state "Do NOT include goals or objectives.
   We only need the dynamics of the world."

4. **Agent ownership:** Action prompts explicitly state which agent owns the action
   and scope capabilities accordingly.

5. **Concurrency awareness:** The concurrent effects prompt presents all cross-agent
   action pairs and asks the LLM to classify each as commutative, conflicting, or
   synergistic.

**Template sketches:**

**`formalize_predicates.txt`** (adapted from L2P's `formalize_predicates.txt`):
```
You are formalizing a multi-agent world model. Given the following scene
description and agent list, extract typed predicates that describe the
SHARED world state.

Scene description:
{scene_desc}

Agents:
{agents_summary}

Types:
{types}

Guidelines:
- Predicates describe the SHARED state of the world, NOT per-agent private state
- Include predicates for: agent positions, object locations, object states,
  spatial relationships, and any other relevant world properties
- Use typed parameters (e.g., ?agent - agent, ?room - room)
- Do NOT include predicates about goals, rewards, or objectives

### New Predicates
```
- (predicate_name ?param1 - type1 ?param2 - type2): 'description'
```
```

**`formalize_agent_actions.txt`**:
```
You are formalizing actions for a specific agent in a multi-agent world model.

Scene description:
{scene_desc}

Agent: {agent_name} (type: {agent_type})
Capabilities: {agent_capabilities}

Available types:
{types}

Available predicates:
{predicates}

Formalize the actions that {agent_name} can perform. Each action should have:
- Parameters (typed)
- Preconditions (PDDL logical expression)
- Effects (PDDL logical expression)

Guidelines:
- The first parameter of every action should be the agent: ?agent - {agent_type}
- Only include actions this specific agent can perform
- Effects should modify the shared world state (predicates listed above)
- Do NOT include goal-related actions

For each action, provide:

### Action Parameters
```
- ?param - type: 'description'
```

### Action Preconditions
```
(and
    (precondition1)
    (precondition2)
)
```

### Action Effects
```
(and
    (effect1)
    (not (effect2))
)
```
```

**`identify_concurrent_effects.txt`**:
```
You are analyzing concurrent action effects in a multi-agent world model.

Scene description:
{scene_desc}

Agents and their actions:
{agent_actions_summary}

When two agents act simultaneously, their effects may interact.
For each pair of actions from DIFFERENT agents, classify the interaction:

1. COMMUTATIVE: effects apply independently (most common, default)
2. CONFLICTING: effects contradict — need a resolution rule
   (e.g., both agents try to pick up the same object → neither succeeds)
3. SYNERGISTIC: combined effect differs from sum of individual effects
   (e.g., two agents pushing same object together succeeds, one alone fails)

List ONLY non-commutative pairs (conflicts and synergies).
If all pairs are commutative, output an empty list.

### CONCURRENT EFFECTS
```json
[
  {
    "agent_a": "robot",
    "action_a": "pick_up",
    "agent_b": "human",
    "action_b": "pick_up",
    "effect_type": "conflicting",
    "resolution": "If both agents try to pick up the same object simultaneously,
                   neither succeeds and the object remains in place.",
    "pddl_condition": "(same_target ?obj)",
    "pddl_effect": "(not (holding ?agent_a ?obj)) (not (holding ?agent_b ?obj))"
  }
]
```
```

**`formalize_initial_state.txt`**:
```
You are defining the initial state of a multi-agent world model.

Scene description:
{scene_desc}

Types:
{types}

Objects:
{objects}

Predicates:
{predicates}

Define the initial state by listing all predicates that are TRUE at the start.
Predicates not listed are assumed FALSE (closed-world assumption).

Do NOT include any goal state or objectives.

### OBJECTS
```
object_name - type ; description
```

### INITIAL
```
(predicate obj1 obj2) ; description
(= (function obj) value) ; numeric initialization
```
```

**Dependencies:** Steps 1–5 (templates are used by all builders, but can be
drafted in parallel with implementation)

**Estimated complexity:** M (Medium)

**Acceptance criteria:**
- All templates load correctly via L2P's `PromptBuilder`
- Placeholder substitution produces well-formed prompts
- MockLLM tests verify that expected output formats are parseable
- Templates are internally consistent (same type/predicate names across prompts)

---

### Step 7: End-to-End Pipeline and WorldModelBuilder Façade

**Description:** A single-entry-point class that orchestrates the full pipeline
from natural language to `WorldModel`.

**Files to create:**
- `src/empo/llm_world_model/world_model_builder.py`
- `src/empo/llm_world_model/__init__.py`
- `tests/test_world_model_builder.py`
- `examples/llm_world_model/build_household_world.py`

**`WorldModelBuilder` class:**
```python
class WorldModelBuilder:
    """
    End-to-end pipeline: natural language → WorldModel.

    Usage:
        builder = WorldModelBuilder(llm=OPENAI(model="gpt-4o"))
        world_model = builder.build(
            scene_desc="A household with a robot and a human...",
            enable_probabilistic=True,
        )

        # world_model is a WorldModel (gymnasium.Env) ready for EMPO
        state = world_model.get_state()
        transitions = world_model.transition_probabilities(state, [0, 1])
    """

    def __init__(
        self,
        llm: BaseLLM,
        prompt_dir: str = None,  # defaults to bundled prompts
        syntax_validator: SyntaxValidator = None,
        max_retries: int = 3,
    ):
        ...

    def build(
        self,
        scene_desc: str,
        enable_probabilistic: bool = False,
        domain_name: str = "world",
        problem_name: str = "scenario",
    ) -> PddlWorldModel:
        """
        Build a WorldModel from a natural language scene description.

        Steps:
        1. Identify agents (AgentBuilder)
        2. Extract domain (WorldModelDomainBuilder)
        3. Extract task (WorldModelTaskBuilder)
        4. Convert to WorldModel (PddlWorldModel)

        Returns:
            A PddlWorldModel instance ready for EMPO computation.
        """
        ...

    def build_and_export(
        self,
        scene_desc: str,
        output_dir: str,
        factored: bool = False,
        enable_probabilistic: bool = False,
    ) -> Tuple[PddlWorldModel, List[str]]:
        """
        Build WorldModel and also export MA-PDDL files.

        Returns:
            (world_model, list_of_pddl_file_paths)
        """
        ...
```

**Dependencies:** Steps 1–6

**Estimated complexity:** M (Medium)

**Acceptance criteria:**
- `WorldModelBuilder.build()` produces a valid `PddlWorldModel` from a canned
  scene description (using MockLLM with pre-recorded outputs)
- The produced `WorldModel` works with `get_dag()` and backward induction
- Example script `build_household_world.py` demonstrates end-to-end usage
- Integration test: build → compute empowerment (using backward induction on small
  state space)

---

### Step 8: Hierarchical World Model Extension

**Description:** Extend the pipeline to produce hierarchical world models as defined
in `src/empo/hierarchical/`. The LLM identifies abstraction levels (e.g. rooms vs
positions within rooms) and produces a `HierarchicalWorldModel` with a coarse-level
`PddlWorldModel` and a fine-level `PddlWorldModel` connected by a `PddlLevelMapper`.

**Files to create:**
- `src/empo/llm_world_model/hierarchical_builder.py`
- `src/empo/llm_world_model/prompts/identify_abstraction_levels.txt`
- `tests/test_hierarchical_builder.py`

**`HierarchicalWorldModelBuilder` class:**
```python
class HierarchicalWorldModelBuilder:
    """
    Extends WorldModelBuilder to produce hierarchical world models.

    The LLM identifies natural abstraction levels in the scene description:
    - Level 0 (coarse): e.g. rooms, zones, or areas
    - Level 1 (fine): e.g. positions within rooms, specific object locations

    Produces a HierarchicalWorldModel with:
    - levels: [coarse_world_model, fine_world_model]
    - mappers: [PddlLevelMapper connecting coarse ↔ fine]
    """

    def identify_abstraction_levels(
        self,
        model: BaseLLM,
        scene_desc: str,
        domain: MADomainSpec,
        prompt_template: str,
        max_retries: int = 3,
    ) -> Tuple[List[AbstractionLevel], str]:
        """
        Prompt the LLM to identify natural abstraction levels.

        Returns a list of abstraction levels, each with:
        - name: e.g. "room_level", "position_level"
        - predicates: which predicates belong to this level
        - state_aggregation: how fine states map to coarse states
        - macro_actions: what actions exist at the coarse level
        """
        ...

    def build_hierarchical(
        self,
        scene_desc: str,
        base_builder: WorldModelBuilder,
        model: BaseLLM,
        prompt_template: str,
    ) -> HierarchicalWorldModel:
        """
        Build a two-level hierarchical world model.

        1. Build fine-level WorldModel using base_builder
        2. Identify abstraction levels via LLM
        3. Build coarse-level WorldModel (subset of predicates, macro-actions)
        4. Create PddlLevelMapper connecting the two
        5. Return HierarchicalWorldModel([coarse, fine], [mapper])
        """
        ...
```

**`PddlLevelMapper` class:**
```python
class PddlLevelMapper(LevelMapper):
    """
    Maps between coarse and fine PDDL-based world models.

    Implements the LevelMapper interface:
    - super_state(): project fine-level ground atoms to coarse-level atoms
    - super_agent(): identity mapping (same agents at both levels)
    - is_feasible(): check if fine action is compatible with coarse macro-action
    - return_control(): check if macro-action goal condition is satisfied
    """

    def __init__(
        self,
        coarse_model: PddlWorldModel,
        fine_model: PddlWorldModel,
        state_projection: Dict[str, str],  # fine_predicate → coarse_predicate
        macro_action_goals: Dict[str, str],  # macro_action → goal condition (PDDL)
    ):
        ...
```

**Dependencies:** Steps 5, 7, and the existing `empo.hierarchical` module

**Estimated complexity:** L (Large)

**Acceptance criteria:**
- `HierarchicalWorldModelBuilder` produces a valid `HierarchicalWorldModel`
- Coarse-level model has fewer states than fine-level model
- `PddlLevelMapper.super_state()` correctly projects fine → coarse states
- `return_control()` correctly identifies when a macro-action is complete
- The hierarchical model works with `compute_hierarchical_robot_policy()`
- Test covers: 2-level hierarchy, state projection, macro-action completion

---

## 5. Concurrent Action Semantics

### 5.1 The Problem

Standard PDDL assumes sequential action execution. In EMPO, agents act
simultaneously. When two agents act in the same time step, their effects can:

1. **Compose independently** (commutative): robot moves left, human moves right →
   both movements happen
2. **Conflict**: robot picks up cup, human picks up cup → conflict (who gets it?)
3. **Interact synergistically**: robot pushes door, human pushes door → door opens
   (neither alone can open it)

L2P has no concept of concurrent actions. This is the most novel extension.

### 5.2 Proposed Strategy

**Phase 1 (conservative, deterministic):**

1. **Default: commutative.** If the LLM doesn't flag a conflict, effects apply
   independently. Add effects = union of all agents' add sets. Delete effects =
   union of all agents' delete sets. When an atom is in both add and delete sets,
   add wins (persistence assumption).

2. **Conflict detection via LLM.** The `identify_concurrent_effects` prompt asks
   the LLM to enumerate non-commutative action pairs. For each conflict, the LLM
   specifies:
   - A **condition** (PDDL expression) for when the conflict occurs
     (e.g. `(same_target ?obj)`)
   - A **resolution** (PDDL effect override)
     (e.g. `(not (holding ?a ?obj)) (not (holding ?b ?obj))`)

3. **Implementation in `PddlWorldModel.transition_probabilities()`:**
   ```
   for each agent:
       if preconditions satisfied:
           collect (add_set, delete_set) from effects
       else:
           no-op
   
   for each ConcurrentEffect rule:
       if rule.condition is satisfied by current bindings:
           replace affected agents' effects with rule.pddl_effect
   
   apply commutative composition of remaining effects
   ```

**Phase 2 (probabilistic, optional):**

4. **Stochastic conflict resolution.** Instead of deterministic resolution, conflicts
   produce probabilistic outcomes. E.g. "both agents try to pick up the same
   object" → P=0.5 agent A gets it, P=0.5 agent B gets it.

5. **PPDDL syntax:** Actions can have `:probabilistic-effects`:
   ```pddl
   (probabilistic
     0.5 (and (holding robot cup) (not (holding human cup)))
     0.5 (and (holding human cup) (not (holding robot cup))))
   ```

### 5.3 Representation in MA-PDDL

Since standard MA-PDDL doesn't have a native concurrent-effects section, we
encode concurrent effects as:

1. **Comments in the domain file** for human readability
2. **A separate JSON metadata file** (`concurrent_effects.json`) that the
   `PddlWorldModel` loader reads alongside the PDDL files
3. **As conditional effects** in the PDDL actions (where expressible):
   ```pddl
   (:action pick_up
     :agent robot
     :parameters (?obj - object)
     :precondition (and (at robot ?obj) (not (holding robot ?obj)))
     :effect (and
       (when (not (picking_up human ?obj))   ; No conflict
         (holding robot ?obj))
       (when (picking_up human ?obj)          ; Conflict → nobody gets it
         (not (holding robot ?obj)))))
   ```

> **Design decision needed:** Which encoding to use for concurrent effects.
> Option 3 (conditional effects) is most self-contained but requires the concept
> of "what the other agent is doing" to be expressible in PDDL, which is non-standard.
> Option 2 (JSON sidecar) is cleanest but means the PDDL files alone are not
> self-describing.
>
> **Recommendation:** Use Option 2 (JSON sidecar) as the primary encoding, with
> Option 1 (comments) for human readability. The `PddlWorldModel` constructor
> accepts both the PDDL specs and the `ConcurrentEffect` list.

---

## 6. Design Decisions Requiring Human Input

| # | Decision | Options | Recommendation | Impact |
|---|----------|---------|----------------|--------|
| 1 | MA-PDDL writer: use unified-planning or custom? | (A) unified-planning, (B) custom | (B) Custom lightweight writer | Step 4 |
| 2 | PDDL-to-WorldModel: adapt pddlgymnasium or custom? | (A) pddlgymnasium, (B) custom | (B) Custom `PddlWorldModel` | Step 5 |
| 3 | Action grounding strategy: eager or lazy? | (A) Eager at init, (B) Lazy per-step | (A) Eager — small state spaces | Step 5 |
| 4 | Concurrent effect encoding: conditional effects, JSON sidecar, or both? | (A) Conditional PDDL, (B) JSON sidecar, (C) Both | (B) JSON sidecar + comments | Steps 4, 5 |
| 5 | LLM provider for testing: mock-only or live API tests? | (A) MockLLM only, (B) Optional live tests with `@pytest.mark.llm` | (B) Both — mock for CI, live for development | All steps |
| 6 | Probabilistic effects: required in Phase 1 or deferred? | (A) Required, (B) Optional/deferred | (B) Optional flag in `build()` | Steps 2, 5 |
| 7 | State representation: frozenset of atoms or bit-vector? | (A) Frozenset (readable), (B) Bit-vector (fast) | (A) Frozenset for clarity, optimize later | Step 5 |
| 8 | Hierarchical levels: LLM-identified or user-specified? | (A) LLM auto-identifies, (B) User hints, (C) Both | (C) LLM suggests, user confirms | Step 8 |

---

## 7. Dependency Graph

```
Step 0: Vendor L2P ──────────────────────────────────────────┐
    │                                                         │
    ├── Step 1: AgentBuilder ─────────┐                       │
    │                                 │                       │
    ├── Step 6: Prompt Templates ─────┤ (can be done in       │
    │   (drafted in parallel)         │  parallel with 1-5)   │
    │                                 │                       │
    └── Step 2: WorldModelDomainBuilder                       │
            │                         │                       │
            ├── Step 3: WorldModelTaskBuilder                  │
            │       │                                         │
            │       └── Step 4: MA-PDDL Output                │
            │               │                                 │
            └───────────────┴── Step 5: PDDL-to-WorldModel    │
                                    │                         │
                                    └── Step 7: E2E Pipeline ─┘
                                            │
                                            └── Step 8: Hierarchical Extension
```

**Suggested PR ordering:**
1. **PR 1:** Step 0 (vendor L2P, baseline test) — can merge immediately
2. **PR 2:** Steps 1 + 6 (AgentBuilder + prompt templates) — parallel work
3. **PR 3:** Step 2 (WorldModelDomainBuilder) — core complexity
4. **PR 4:** Steps 3 + 4 (TaskBuilder + MA-PDDL writer) — smaller, dependent on PR 3
5. **PR 5:** Step 5 (PddlWorldModel) — core converter, dependent on PR 3
6. **PR 6:** Step 7 (end-to-end pipeline) — integration, dependent on PRs 2-5
7. **PR 7:** Step 8 (hierarchical) — independent extension, dependent on PR 6

---

## 8. Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| LLM produces inconsistent PDDL across prompts (type names differ between predicates and actions) | High | High | Accumulative prompting: each prompt includes all previous outputs as context; validation rejects inconsistencies |
| Concurrent effect space is combinatorially large for many agents/actions | Medium | Medium | Only flag non-commutative pairs; for n agents × m actions, worst case is O(n²m²) pairs but most are commutative |
| Action grounding produces too many ground actions for `transition_probabilities()` | Medium | High | Limit objects per type; lazy grounding fallback; warn if action space exceeds threshold |
| PDDL precondition/effect evaluation is slow for complex expressions | Low | Medium | Cache grounded preconditions; precompile to Python callables |
| LLM hallucinates predicates/types not declared in earlier steps | High | Medium | Strict validation against declared types/predicates; retry with error context |
| Hierarchical level identification is subjective/unreliable | Medium | Low | Step 8 is optional; user can override LLM suggestions |
| L2P upstream changes break vendored copy | Low | Low | Vendored copy is frozen; update manually when needed |
