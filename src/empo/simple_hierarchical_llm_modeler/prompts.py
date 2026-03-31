"""
Prompt templates used by the tree builder and hierarchical modeler.

Every function returns a *complete* prompt string ready to send to the LLM.
All prompts request JSON output so that responses can be parsed reliably.
"""

from typing import List, Optional


def _history_block(history: List[str]) -> str:
    """Format a history of events into a numbered block."""
    if not history:
        return "No events have occurred yet."
    lines = [f"  {i + 1}. {event}" for i, event in enumerate(history)]
    return "\n".join(lines)


# ── Step: robot action generation ─────────────────────────────────────────────


def robot_actions_prompt(
    initial_state: str,
    history: List[str],
    n_robotactions: int,
    higher_level_context: Optional[str] = None,
) -> str:
    ctx = ""
    if higher_level_context:
        ctx = f"Higher-level context: {higher_level_context}\n\n"
    return (
        f"{ctx}"
        f"Situation: {initial_state}\n\n"
        f"Assume the following happened so far:\n{_history_block(history)}\n\n"
        f"Please name {n_robotactions} distinct high-level action options the robot "
        f"has now that differ in their consequences on the empowerment of all humans "
        f"they have a consequence on.\n\n"
        f"Return ONLY a JSON list of exactly {n_robotactions} objects, each with keys "
        f'"action" (concise robot action description) and "rationale" (brief rationale).\n'
        f"Example: "
        f'[{{"action": "...", "rationale": "..."}}, ...]'
    )


# ── Step: humans reaction generation ──────────────────────────────────────────


def humans_reactions_prompt(
    initial_state: str,
    history: List[str],
    n_humansreactions: int,
    higher_level_context: Optional[str] = None,
) -> str:
    ctx = ""
    if higher_level_context:
        ctx = f"Higher-level context: {higher_level_context}\n\n"
    return (
        f"{ctx}"
        f"Situation: {initial_state}\n\n"
        f"Assume the following happened so far:\n{_history_block(history)}\n\n"
        f"Please name {n_humansreactions} distinct high-level things that the humans "
        f"can do that differ in their consequences on the empowerment of all humans "
        f"this has a consequence on.\n\n"
        f"Return ONLY a JSON list of exactly {n_humansreactions} objects, each with keys "
        f'"reaction" (concise humans\' reaction description) and "rationale" (brief rationale).\n'
        f"Example: "
        f'[{{"reaction": "...", "rationale": "..."}}, ...]'
    )


# ── Step: consequence generation ──────────────────────────────────────────────


def consequences_prompt(
    initial_state: str,
    history: List[str],
    n_consequences: int,
    higher_level_context: Optional[str] = None,
) -> str:
    ctx = ""
    if higher_level_context:
        ctx = f"Higher-level context: {higher_level_context}\n\n"
    return (
        f"{ctx}"
        f"Situation: {initial_state}\n\n"
        f"Assume the following happened so far:\n{_history_block(history)}\n\n"
        f"Please name {n_consequences} distinct observable consequences and their "
        f"probabilities.\n\n"
        f"A consequence is something that happens in the environment as a direct "
        f"result of the actions already taken — NOT a subsequent action by the "
        f"robot or humans. Subsequent actions will be asked about separately.\n\n"
        f"Each consequence description must state concretely what changed in the "
        f"situation — what can be directly observed (e.g. a door opened, traffic "
        f"appeared, rain started, a message was displayed, an object moved). "
        f"Do NOT describe implications for empowerment or decision-making in the "
        f"consequence description; put that analysis in the rationale instead.\n\n"
        f"The {n_consequences} consequences should differ in their implications on "
        f"the empowerment of all affected humans.\n\n"
        f"Return ONLY a JSON list of exactly {n_consequences} objects, each with keys "
        f'"consequence" (concise description of the observable state change), '
        f'"probability" (float 0-1), '
        f'and "rationale" (brief analysis of empowerment implications).  '
        f"Probabilities must sum to 1.\n"
        f"Example: "
        f'[{{"consequence": "...", "probability": 0.6, "rationale": "..."}}, ...]'
    )


# ── Step: terminal empowerment estimation ─────────────────────────────────────


def empowerment_prompt(
    initial_state: str,
    history: List[str],
    higher_level_context: Optional[str] = None,
) -> str:
    ctx = ""
    if higher_level_context:
        ctx = f"Higher-level context: {higher_level_context}\n\n"
    return (
        f"{ctx}"
        f"Situation: {initial_state}\n\n"
        f"Assume the following happened so far:\n{_history_block(history)}\n\n"
        f"We want to estimate how much effective power (empowerment) the affected "
        f"humans collectively have in this situation.\n\n"
        f"To do this, think step by step:\n"
        f"1. List the key independent choices or decisions the affected humans "
        f"can still make from this point on (e.g. where to go, what to say, "
        f"whether to cooperate, etc.).\n"
        f"2. For each choice, estimate roughly how many meaningfully different "
        f"options they have.\n"
        f"3. Multiply the numbers of options across all independent choices to "
        f"get the total number of meaningfully different futures they can "
        f"bring about together with sufficient reliability.\n\n"
        f"Return ONLY a JSON object with keys "
        f'"choices" (list of objects with "choice" and "n_options"), '
        f'"estimate" (the product of all n_options, as a positive number), '
        f'and "rationale" (brief summary).\n'
        f"Example: "
        f'{{"choices": [{{"choice": "where to go", "n_options": 3}}, '
        f'{{"choice": "whether to cooperate", "n_options": 2}}], '
        f'"estimate": 6, "rationale": "3 destinations × 2 cooperation modes"}}'
    )


# ── Hierarchical: status check ────────────────────────────────────────────────


def hierarchical_status_prompt(
    higher_level_context: str,
    higher_level_action: str,
    initial_state: str,
    history: List[str],
) -> str:
    return (
        f"Higher-level context: {higher_level_context}\n\n"
        f"Situation: {initial_state}\n\n"
        f"Assume the following happened so far:\n{_history_block(history)}\n\n"
        f"Does this state already constitute a success of the higher-level "
        f'activity ("{higher_level_action}"), or a failure in that higher-level '
        f"activity, or is that higher-level activity still in progress?\n\n"
        f'Please respond with ONLY a JSON object with key "status" whose value '
        f'is exactly one of "success", "failure", or "still in progress".\n'
        f'Example: {{"status": "still in progress"}}'
    )


# ── Hierarchical: consequence matching ────────────────────────────────────────


def match_consequence_prompt(
    higher_level_context: str,
    higher_level_action: str,
    known_consequences: List[str],
    status: str,
    initial_state: str,
    history: List[str],
) -> str:
    cons_list = "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(known_consequences))
    return (
        f"Higher-level context: {higher_level_context}\n\n"
        f"Situation: {initial_state}\n\n"
        f"Assume the following happened so far:\n{_history_block(history)}\n\n"
        f'The higher-level activity "{higher_level_action}" has resulted in '
        f'"{status}". The robot believed the activity could lead to one of the '
        f"following consequences:\n{cons_list}\n\n"
        f"Does the current {status} correspond to one of these, or is this "
        f"situation not covered by them?\n\n"
        f"Return ONLY a JSON object with keys "
        f'"match" (the 1-based index of the matching consequence, or null if none match) '
        f'and "new_consequence" (a concise description if no match, or null if there is '
        f"a match).\n"
        f'Example if matching: {{"match": 2, "new_consequence": null}}\n'
        f'Example if new: {{"match": null, "new_consequence": "..."}}'
    )
