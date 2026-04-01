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

    # Live mode (requires GOOGLE_API_KEY):
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

DEFAULT_SCENARIO = (
    "A robot taxi driver and two potential passengers (Alice and Bob) stand "
    "in front of the departure building of an airport."
)


class AirportMockLLM:
    """Returns plausible canned responses for the airport-taxi scenario."""

    @property
    def context_length(self) -> int:
        return 1_000_000

    def query(self, prompt: str) -> str:
        """Return a deterministic JSON response based on prompt keywords."""
        # --- Robot actions ---
        if "robot can perform" in prompt or "robot can engage in" in prompt:
            n = _extract_n(prompt)
            pool = [
                {
                    "activity": "Offer to drive both passengers together, dropping Bob at the train station first",
                    "rationale": "Maximises both passengers' options by getting them moving quickly and efficiently",
                },
                {
                    "activity": "Ask the passengers about their destinations before deciding",
                    "rationale": "Gathering information preserves flexibility for all parties",
                },
                {
                    "activity": "Offer to drive Alice to the city centre first and come back for Bob",
                    "rationale": "Prioritises Alice but reduces Bob's options by making him wait",
                },
            ]
            return json.dumps(pool[:n])

        # --- Humans reactions ---
        if "things" in prompt and "affected humans" in prompt:
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
        if "ways the situation could look" in prompt:
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

        # --- Batched empowerment estimate ---
        if "Score all scenarios on the SAME SCALE" in prompt:
            # Count how many scenarios are in the prompt
            n_scenarios = prompt.count("Scenario ")
            return json.dumps([
                {
                    "choices": [{"choice": "destination", "n_options": 3},
                                {"choice": "cooperation mode", "n_options": 2}],
                    "estimate": 12 - i,
                    "rationale": f"Passengers can still choose destinations, routes, and whether to continue together (scenario {i + 1})",
                }
                for i in range(n_scenarios)
            ])

        # --- Individual empowerment estimate (fallback) ---
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
    parser.add_argument(
        "--llm", type=str, default="google/gemini-2.5-flash",
        metavar="PROVIDER/MODEL",
        help="LLM to use in live mode as provider/model "
             "(default: google/gemini-2.5-flash). "
             "Providers: google, openai, deepseek, mistral, huggingface, anthropic"
    )
    parser.add_argument(
        "--depth", type=int, default=1, help="Number of full expansion cycles (default: 1)"
    )
    parser.add_argument(
        "--situation", type=str, default=DEFAULT_SCENARIO,
        help="Initial situation description (default: airport taxi scenario)"
    )
    parser.add_argument(
        "--export", type=str, default=None, metavar="PATH",
        help="Export tree to file. Format is inferred from extension: "
             ".json (default), .yaml/.yml, or .html"
    )
    args = parser.parse_args()

    scenario = args.situation

    if args.live:
        try:
            import os
            from l2p.llm.openai import OPENAI

            # Parse provider/model from --llm argument
            if "/" in args.llm:
                provider, model = args.llm.split("/", 1)
            else:
                provider, model = "google", args.llm

            # Provider-specific configuration
            _PROVIDER_CONFIG = {
                "google": {
                    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                    "api_key_env": "GOOGLE_API_KEY",
                },
                "openai": {
                    "base_url": "https://api.openai.com/v1/",
                    "api_key_env": "OPENAI_API_KEY",
                },
                "deepseek": {
                    "base_url": "https://api.deepseek.com/v1/",
                    "api_key_env": "DEEPSEEK_API_KEY",
                },
                "mistral": {
                    "base_url": "https://api.mistral.ai/v1/",
                    "api_key_env": "MISTRAL_API_KEY",
                },
                "huggingface": {
                    "base_url": "https://api-inference.huggingface.co/v1/",
                    "api_key_env": "HF_API_KEY",
                },
                "anthropic": {
                    "api_key_env": "ANTHROPIC_API_KEY",
                },
            }

            if provider not in _PROVIDER_CONFIG:
                raise ValueError(
                    f"Unknown provider '{provider}'. "
                    f"Choose from: {', '.join(_PROVIDER_CONFIG)}"
                )

            cfg = _PROVIDER_CONFIG[provider]
            api_key = os.environ.get(cfg["api_key_env"])
            if not api_key:
                raise ValueError(
                    f"Set {cfg['api_key_env']} environment variable for {provider}"
                )

            print(f"Using LLM: {provider}/{model}")
            if provider == "anthropic":
                from empo.simple_hierarchical_llm_modeler.llm_connector import (
                    AnthropicConnector,
                )
                llm = AnthropicConnector(model=model, api_key=api_key)
            else:
                llm = OPENAI(
                    model=model,
                    provider=provider,
                    api_key=api_key,
                    base_url=cfg["base_url"],
                )
        except Exception as exc:
            print(f"Cannot initialise live LLM: {exc}")
            print("Falling back to mock mode.")
            llm = AirportMockLLM()
    else:
        llm = AirportMockLLM()
        print("Running in MOCK mode (deterministic, no API key needed).\n")

    # Import after path setup
    from empo.simple_hierarchical_llm_modeler import (
        CachedLLMConnector,
        LiveTreeRenderer,
        StatsTrackingLLM,
        build_tree,
        build_two_level_model,
        collect_leaves,
        count_nodes,
        NLWorldModel,
    )

    # Wrap live LLM with a disk cache to avoid re-querying on restarts.
    # Mock mode is deterministic and instant, so no caching needed.
    raw_llm = llm if args.live else None
    if args.live:
        cache_model = args.llm.replace("/", "_")
        llm = CachedLLMConnector(llm, cache_dir=f"outputs/llm_cache/{cache_model}")

    # Wrap with stats tracking (uses real token counts from L2P when available)
    llm = StatsTrackingLLM(llm, raw_llm=raw_llm)

    # ── Coarse-level tree ──────────────────────────────────────────────────
    if not args.live:
        print("=" * 70)
        print("Building coarse-level hierarchical world model")
        print("=" * 70)

    renderer = LiveTreeRenderer(root_label=scenario) if args.live else None

    hmodel = build_two_level_model(
        llm,
        scenario,
        coarse_n_steps=args.depth,
        fine_n_steps=args.depth,
        n_robotactions=3,
        n_humansreactions=2,
        n_consequences=2,
        coarse_time_horizon="one day",
        fine_time_horizon="one hour",
        on_update=renderer.update if renderer else None,
    )
    if renderer:
        renderer.finish()
        import time; time.sleep(2)
        renderer.close()

    coarse_tree = hmodel._coarse_tree
    if not args.live:
        print(f"\nCoarse tree: {count_nodes(coarse_tree)} nodes, "
              f"{len(collect_leaves(coarse_tree))} leaves.\n")
        print(coarse_tree.render(root_label=scenario))
        print_world_model(hmodel.coarsest())

    # ── Fine-level tree (for first coarse action) ─────────────────────────
    fine_tree = None
    coarse_labels = hmodel.coarsest().robot_action_labels()
    if coarse_labels:
        chosen_action = coarse_labels[0]
        # Pick the first human reaction for this coarse action
        chosen_reaction = None
        if coarse_tree is not None:
            for label, _, child in coarse_tree.children:
                if label == chosen_action and child.children:
                    chosen_reaction = child.children[0][0]  # first reaction label
                    break

        if not args.live:
            print(f"\n{'=' * 70}")
            print(f"Building fine-level tree for: {chosen_action}")
            if chosen_reaction:
                print(f"  Human reaction: {chosen_reaction}")
            print("=" * 70)

        # Build context lines showing the higher-level path
        context = [
            f"▸ Situation: {scenario}",
            f"▸ Coarse action: {chosen_action}",
        ]
        if chosen_reaction:
            context.append(f"▸ Human reaction: {chosen_reaction}")
        renderer2 = LiveTreeRenderer(
            root_label=scenario, context_lines=context
        ) if args.live else None

        fine, mapper = hmodel.get_fine_model(
            chosen_action,
            coarse_human_reaction_label=chosen_reaction,
            on_update=renderer2.update if renderer2 else None,
        )
        if renderer2:
            renderer2.finish()
            import time; time.sleep(2)
            renderer2.close()

        fine_tree = hmodel._fine_trees.get((chosen_action, chosen_reaction))
        if not args.live and fine_tree is not None:
            print(f"\nFine tree: {count_nodes(fine_tree)} nodes, "
                  f"{len(collect_leaves(fine_tree))} leaves.\n")
            print(fine_tree.render(root_label=scenario))
            print_world_model(fine)

    # ── Export trees if requested ──────────────────────────────────────────
    if args.export:
        ext = args.export.rsplit(".", 1)[-1].lower() if "." in args.export else "json"
        fmt = {"json": "json", "yaml": "yaml", "yml": "yaml", "html": "html"}.get(ext, "json")

        if fmt == "html" and fine_tree is not None:
            # Export both coarse and fine trees into one HTML file
            from empo.simple_hierarchical_llm_modeler.tree_builder import _trees_to_html
            sections = [
                ("Coarse-level tree", coarse_tree.to_dict(root_label=scenario)),
                (f"Fine-level tree — {chosen_action}", fine_tree.to_dict(root_label=scenario)),
            ]
            with open(args.export, "w") as f:
                f.write(_trees_to_html(sections))
        else:
            coarse_tree.export(args.export, fmt=fmt, root_label=scenario)
        print(f"Tree(s) exported to {args.export} ({fmt})")

    # Print token usage statistics
    llm.print_report()

    print("\nDone!")


if __name__ == "__main__":
    main()
