#!/usr/bin/env python3
"""
Two-level airport taxi example using the simple hierarchical LLM modeler.

Scenario
--------
A robot taxi driver and two potential passengers stand in front of the
departure building of an airport.

* **Level 0 (coarse):** High-level decisions – e.g. "offer to drive both
  passengers together", "offer separate rides", "wait for instructions".
* **Level 1 (fine):** Detailed execution of the chosen high-level plan –
  e.g. "open trunk for luggage", "confirm destination", "start driving".

This script can run in two modes:

1. **Mock mode** (default) – uses a deterministic MockLLM so no API key is
   needed.  Useful for testing and understanding the data flow.
2. **Live mode** – uses a real LLM via the L2P OpenAI connector.  Requires
   an API key.

Usage
-----
::

    # Mock mode (no API key needed):
    PYTHONPATH=src:vendor/multigrid:vendor/l2p python examples/hierarchical_llm/airport_taxi_example.py

    # Live mode (requires OPENAI_API_KEY):
    PYTHONPATH=src:vendor/multigrid:vendor/l2p python examples/hierarchical_llm/airport_taxi_example.py --live
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap

# ---------------------------------------------------------------------------
# Mock LLM – deterministic, no network access
# ---------------------------------------------------------------------------

SCENARIO = (
    "A robot taxi driver and two potential passengers (Alice and Bob) stand "
    "in front of the departure building of an airport. Alice needs to go to "
    "the city centre and Bob needs to go to the train station, which is on "
    "the way to the city centre."
)


class AirportMockLLM:
    """Returns plausible canned responses for the airport-taxi scenario."""

    def query(self, prompt: str) -> str:
        """Return a deterministic JSON response based on prompt keywords."""
        # --- Robot actions ---
        if "action options the robot" in prompt:
            n = _extract_n(prompt)
            pool = [
                {
                    "action": "Offer to drive both passengers together, dropping Bob at the train station first",
                    "rationale": "Maximises both passengers' options by getting them moving quickly and efficiently",
                },
                {
                    "action": "Ask the passengers about their destinations before deciding",
                    "rationale": "Gathering information preserves flexibility for all parties",
                },
                {
                    "action": "Offer to drive Alice to the city centre first and come back for Bob",
                    "rationale": "Prioritises Alice but reduces Bob's options by making him wait",
                },
            ]
            return json.dumps(pool[:n])

        # --- Humans reactions ---
        if "things that the humans" in prompt:
            n = _extract_n(prompt)
            pool = [
                {
                    "reaction": "Both passengers agree and get into the taxi",
                    "rationale": "Cooperative outcome that preserves everyone's empowerment",
                },
                {
                    "reaction": "Alice agrees but Bob decides to take a different taxi",
                    "rationale": "Bob exercises his own choice, reducing coordination but preserving his autonomy",
                },
                {
                    "reaction": "Both passengers decline and decide to share a ride-share instead",
                    "rationale": "Passengers choose an alternative, reducing robot's influence but preserving their options",
                },
            ]
            return json.dumps(pool[:n])

        # --- Consequences ---
        if "consequences and their probabilities" in prompt:
            n = _extract_n(prompt)
            pool = [
                {
                    "consequence": "The taxi departs smoothly with the agreed passengers",
                    "probability": 0.7,
                    "rationale": "Most likely outcome when passengers have agreed",
                },
                {
                    "consequence": "A traffic jam blocks the route and the passengers must choose an alternative",
                    "probability": 0.3,
                    "rationale": "Traffic is common near the airport, creating new decision points",
                },
            ]
            selected = pool[:n]
            total = sum(c["probability"] for c in selected)
            for c in selected:
                c["probability"] = round(c["probability"] / total, 2)
            return json.dumps(selected)

        # --- Empowerment estimate ---
        if "meaningfully different futures" in prompt:
            return json.dumps(
                {
                    "estimate": 12,
                    "rationale": "Passengers can still choose destinations, routes, and whether to continue together",
                }
            )

        # --- Hierarchical status ---
        if "success of the higher-level" in prompt:
            return json.dumps({"status": "still in progress"})

        # --- Consequence matching ---
        if "correspond to one of these" in prompt:
            return json.dumps({"match": 1, "new_consequence": None})

        return json.dumps({"error": "unrecognised prompt"})


def _extract_n(prompt: str) -> int:
    m = re.search(r"name\s+(\d+)", prompt)
    return int(m.group(1)) if m else 2


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------


def _indent(text: str, level: int = 1) -> str:
    return textwrap.indent(text, "  " * level)


def print_tree(node, depth: int = 0) -> None:
    prefix = "  " * depth
    tag = node.node_type.upper()
    hist = " → ".join(node.history[-1:]) if node.history else "(root)"
    emp = ""
    if node.empowerment_estimate is not None:
        emp = f"  [empowerment ≈ {node.empowerment_estimate}]"
    print(f"{prefix}[{tag}] {hist}{emp}")
    for label, prob, child in node.children:
        pstr = f" (p={prob:.2f})" if node.node_type == "humansreaction" else ""
        print(f"{prefix}  ├─ {label}{pstr}")
        print_tree(child, depth + 2)


def print_world_model(model) -> None:
    print("\n=== NLWorldModel Summary ===")
    print(f"  States: {len(model.states)}")
    print(f"  Terminal states: {len(model.terminal_states)}")
    print(f"  Robot actions at root: {model.robot_action_labels()}")
    s0 = model.get_state()
    for ra_idx, ra_label in enumerate(model.robot_action_labels()):
        hr_labels = model.humans_reaction_labels(robot_action_index=ra_idx)
        for hr_idx, hr_label in enumerate(hr_labels):
            trans = model.transition_probabilities(s0, [ra_idx, hr_idx])
            if trans:
                outcomes = ", ".join(
                    f"p={p:.2f}→{model.state_description(ns)[:60]}..."
                    for p, ns in trans
                )
                print(f"  ({ra_label}, {hr_label}): {outcomes}")
    for ts in model.terminal_states:
        v = model.V_r_estimate(ts)
        print(f"  Terminal V_r({model.state_description(ts)[:50]}...) = {v:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Airport taxi hierarchical LLM example"
    )
    parser.add_argument(
        "--live", action="store_true", help="Use a real LLM (needs API key)"
    )
    args = parser.parse_args()

    if args.live:
        try:
            from l2p.llm.openai import OPENAI

            llm = OPENAI(model="gpt-4o-mini")
        except Exception as exc:
            print(f"Cannot initialise live LLM: {exc}")
            print("Falling back to mock mode.")
            llm = AirportMockLLM()
    else:
        llm = AirportMockLLM()
        print("Running in MOCK mode (deterministic, no API key needed).\n")

    # Import after path setup
    from empo.simple_hierarchical_llm_modeler import (
        build_tree,
        build_two_level_model,
        collect_leaves,
        count_nodes,
        NLWorldModel,
    )

    # ── Single-level demo ──────────────────────────────────────────────────
    print("=" * 70)
    print("STEP 1: Build a single-level trajectory tree (depth=1)")
    print("=" * 70)

    tree = build_tree(
        llm,
        SCENARIO,
        n_steps=1,
        n_robotactions=3,
        n_humansreactions=2,
        n_consequences=2,
    )
    print(
        f"\nTree has {count_nodes(tree)} nodes, {len(collect_leaves(tree))} leaves.\n"
    )
    print_tree(tree)

    model = NLWorldModel.from_tree(tree, SCENARIO)
    print_world_model(model)

    # ── Two-level hierarchical demo ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: Build a two-level hierarchical world model (lazy)")
    print("=" * 70)

    hmodel = build_two_level_model(
        llm,
        SCENARIO,
        coarse_n_steps=1,
        fine_n_steps=1,
        n_robotactions=3,
        n_humansreactions=2,
        n_consequences=2,
    )
    print(f"\nLazy hierarchical model: {hmodel.num_levels} levels")
    print(f"Fine model built yet? {hmodel.finest() is not None}")

    print("\n--- Coarse level (Level 0) ---")
    print_world_model(hmodel.coarsest())

    # Now simulate taking the first coarse action -> fine model built lazily
    coarse_labels = hmodel.coarsest().robot_action_labels()
    if coarse_labels:
        chosen_action = coarse_labels[0]
        print(f"\n--- Taking coarse action: {chosen_action} ---")
        print("(Fine model is built lazily now...)")
        fine, mapper = hmodel.get_fine_model(chosen_action)
        print(f"Fine model built? {hmodel.finest() is not None}")
        print("\n--- Fine level (Level 1) ---")
        print_world_model(fine)

    print("\nDone!")


if __name__ == "__main__":
    main()
