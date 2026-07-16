"""
Tests for the Tools WorldModel.

Validates:
1. Environment creation and reset
2. State get/set roundtrip
3. Action encoding/decoding
4. Transition probabilities (sum to 1, correct successor count)
5. step() overrides WorldModel.step() to return dummy obs (Discrete(1) contract)
6. perceived_state masks has_requested correctly
7. Goal classes (HoldGoal, WorkbenchGoal)
8. Goal generator and sampler
9. Heuristic policy produces valid distributions
10. Rendering produces an image
"""

import numpy as np
import pytest

# Import using PYTHONPATH-compatible paths
from empo.world_specific_helpers.tools import (
    HoldGoal,
    IdleGoal,
    ToolsGoalGenerator,
    ToolsGoalSampler,
    ToolsHeuristicPolicy,
    ToolsWorldModel,
    WorkbenchGoal,
    _action_acquire,
    _action_give_idx,
    action_name,
    create_tools_env,
    decode_action,
)
from empo.world_model import WorldModel

# ---- fixtures ----


@pytest.fixture
def small_env():
    """Small deterministic environment: 2 agents, 2 tools, 5 steps."""
    return ToolsWorldModel(
        n_agents=2,
        n_tools=2,
        max_steps=5,
        p_failure=0.0,
        seed=42,
        robot_agent_indices=[0],
    )


@pytest.fixture
def example_env():
    """Example-sized environment: 4 agents, 6 tools, 10 steps."""
    return ToolsWorldModel(
        n_agents=4,
        n_tools=6,
        max_steps=10,
        p_failure=0.1,
        seed=123,
        robot_agent_indices=[0],
    )


# ---- basic construction ----


class TestConstruction:
    def test_creates(self, small_env):
        assert small_env.n_agents == 2
        assert small_env.n_tools == 2
        assert small_env.max_steps == 5

    def test_agent_indices(self, small_env):
        assert small_env.robot_agent_indices == [0]
        assert small_env.human_agent_indices == [1]

    def test_action_space_size(self, small_env):
        # Per-agent: n_tools + len(give_targets[i])
        nap = small_env.n_actions_per_agent
        assert len(nap) == 2
        for i, na in enumerate(nap):
            assert na == small_env.n_tools + len(small_env.give_targets[i])

    def test_graph_diagonals(self, small_env):
        for i in range(small_env.n_agents):
            assert small_env.can_hear[i, i]
            assert small_env.can_reach[i, i]
            assert small_env.can_grab[i, i]

    def test_is_world_model(self, small_env):
        assert isinstance(small_env, WorldModel)

    def test_factory(self):
        env = create_tools_env(n_agents=3, n_tools=4, seed=0)
        assert env.n_agents == 3
        assert env.n_tools == 4


# ---- state management ----


class TestState:
    def test_reset(self, small_env):
        small_env.reset(seed=42)
        state = small_env.get_state()
        assert state[0] == 5  # remaining steps

    def test_get_set_roundtrip(self, small_env):
        small_env.reset(seed=42)
        s1 = small_env.get_state()
        small_env.set_state(s1)
        s2 = small_env.get_state()
        assert s1 == s2

    def test_state_is_hashable(self, small_env):
        small_env.reset(seed=42)
        s = small_env.get_state()
        d = {s: 1}
        assert d[s] == 1

    def test_tool_placement_invariant(self, small_env):
        """Each tool is on exactly one workbench (initially no holds)."""
        small_env.reset(seed=42)
        _, wb, holds, req = small_env.get_state()
        for k in range(small_env.n_tools):
            wb_count = sum(wb[i][k] for i in range(small_env.n_agents))
            hd_count = sum(holds[i][k] for i in range(small_env.n_agents))
            assert wb_count + hd_count == 1, f"Tool {k}: wb={wb_count}, hd={hd_count}"
            assert hd_count == 0  # initially no one holds anything


# ---- action encoding ----


class TestActions:
    def test_acquire(self, small_env):
        gt = small_env.give_targets[0]
        m = small_env.n_tools
        assert decode_action(0, m, gt) == ("acquire", 0)
        assert decode_action(1, m, gt) == ("acquire", 1)

    def test_give(self, small_env):
        gt = small_env.give_targets[0]
        m = small_env.n_tools
        for j_pos, j in enumerate(gt):
            assert decode_action(m + j_pos, m, gt) == ("give", j)

    def test_action_name(self, small_env):
        gt = small_env.give_targets[0]
        m = small_env.n_tools
        assert "acq" in action_name(0, m, gt)
        if len(gt) > 0:
            assert "give" in action_name(m, m, gt)

    def test_encode_helpers(self, small_env):
        assert _action_acquire(0) == 0
        assert _action_acquire(1) == 1
        m = small_env.n_tools
        assert _action_give_idx(0, m) == m
        assert _action_give_idx(1, m) == m + 1


# ---- transition probabilities ----


class TestTransitions:
    def test_terminal_returns_none(self, small_env):
        small_env.reset(seed=42)
        state = list(small_env.get_state())
        state[0] = 0  # remaining = 0
        state = tuple(state)
        assert small_env.transition_probabilities(state, [0, 0]) is None

    def test_pass_pass_deterministic(self, small_env):
        """With p_failure=0, pass/pass → single successor."""
        small_env.reset(seed=42)
        s = small_env.get_state()
        trans = small_env.transition_probabilities(s, [0, 0])
        assert trans is not None
        assert len(trans) == 1
        prob, ns = trans[0]
        assert abs(prob - 1.0) < 1e-9
        assert ns[0] == s[0] - 1  # time decremented

    def test_probabilities_sum_to_one(self, example_env):
        example_env.reset(seed=123)
        s = example_env.get_state()
        trans = example_env.transition_probabilities(s, [0, 0, 0, 0])
        assert trans is not None
        total = sum(p for p, _ in trans)
        assert abs(total - 1.0) < 1e-9

    def test_at_most_n_plus_1_successors(self, example_env):
        """Number of distinct successors ≤ n + 1."""
        example_env.reset(seed=123)
        s = example_env.get_state()
        nap = example_env.n_actions_per_agent
        for _ in range(5):
            actions = [np.random.randint(nap[i]) for i in range(4)]
            trans = example_env.transition_probabilities(s, actions)
            if trans is not None:
                assert len(trans) <= example_env.n_agents + 1

    def test_tool_invariant_maintained(self, small_env):
        """After any transition, each tool is still on exactly one wb or in one hand."""
        small_env.reset(seed=42)
        s = small_env.get_state()
        nap = small_env.n_actions_per_agent
        # try every action pair
        for a0 in range(nap[0]):
            for a1 in range(nap[1]):
                trans = small_env.transition_probabilities(s, [a0, a1])
                if trans is None:
                    continue
                for prob, ns in trans:
                    _, wb, holds, _ = ns
                    for k in range(small_env.n_tools):
                        total = sum(wb[i][k] for i in range(small_env.n_agents))
                        total += sum(holds[i][k] for i in range(small_env.n_agents))
                        assert (
                            total == 1
                        ), f"Tool {k} count={total} after actions ({a0},{a1})"


# ---- step uses transition_probabilities ----


class TestStep:
    def test_step_inherits_from_worldmodel(self, small_env):
        """step() is overridden to return dummy observations (Discrete(1))."""
        # ToolsWorldModel overrides step() for Gymnasium contract consistency
        assert type(small_env).step is not WorldModel.step

    def test_step_produces_valid_state(self, small_env):
        small_env.reset(seed=42)
        _obs, reward, terminated, truncated, info = small_env.step([0, 0])
        state = small_env.get_state()
        assert state[0] == small_env.max_steps - 1
        assert not terminated

    def test_step_reaches_terminal(self, small_env):
        small_env.reset(seed=42)
        for _ in range(small_env.max_steps):
            _obs, reward, terminated, truncated, info = small_env.step([0, 0])
        assert terminated

    def test_step_consistent_with_transition_probs(self, small_env):
        """step() outcome must be one of the transition_probabilities outcomes."""
        small_env.reset(seed=42)
        state = small_env.get_state()
        actions = [0, 0]
        trans = small_env.transition_probabilities(state, actions)
        possible_states = {s for _, s in trans}

        small_env.set_state(state)
        small_env.step(actions)
        new_state = small_env.get_state()
        assert new_state in possible_states


# ---- perceived state ----


class TestPerceivedState:
    def test_default_perceived_state_unchanged(self):
        """WorldModel base default returns state unchanged."""
        from empo.world_model import WorldModel as WM

        assert hasattr(WM, "perceived_state")

        # Use a minimal concrete subclass that does NOT override perceived_state
        class _MinimalWorldModel(WM):
            def __init__(self):
                super().__init__()
                self._state = (1, (0,))

            def get_state(self):
                return self._state

            def set_state(self, state):
                self._state = state

            def transition_probabilities(self, state, actions):
                return [(1.0, state)]

        env = _MinimalWorldModel()
        s = env.get_state()
        # Base default returns the true state unchanged for any agent index
        assert env.perceived_state(s, 0) == s
        assert env.perceived_state(s, 99) == s

    def test_tools_masks_unheard_requests(self):
        """Requests from agents that cannot be heard are masked to 0."""
        env = ToolsWorldModel(
            n_agents=3,
            n_tools=2,
            max_steps=5,
            p_failure=0.0,
            seed=0,
            # Manually set can_hear: agent 0 can hear only itself
            can_hear=np.array(
                [
                    [True, False, False],
                    [True, True, True],
                    [True, True, True],
                ]
            ),
            can_reach=np.ones((3, 3), dtype=bool),
            can_grab=np.ones((3, 3), dtype=bool),
        )
        env.reset(seed=0)
        # Manually set a request: agent 1 requests tool 0
        s = list(env.get_state())
        req = [list(row) for row in s[3]]
        req[1][0] = 1
        s[3] = tuple(tuple(r) for r in req)
        s = tuple(s)

        # Agent 0 cannot hear agent 1 → perceived request for agent 1 is all 0s
        ps = env.perceived_state(s, 0)
        assert ps[3][1] == (0, 0)

        # Agent 2 can hear agent 1 → perceived request for agent 1 is kept
        ps2 = env.perceived_state(s, 2)
        assert ps2[3][1][0] == 1

    def test_self_requests_always_visible(self):
        """An agent can always hear itself (diagonal = True)."""
        env = ToolsWorldModel(
            n_agents=2,
            n_tools=2,
            max_steps=5,
            p_failure=0.0,
            seed=0,
            can_hear=np.eye(2, dtype=bool),  # only self
            can_reach=np.ones((2, 2), dtype=bool),
            can_grab=np.ones((2, 2), dtype=bool),
        )
        env.reset(seed=0)
        s = list(env.get_state())
        req = [list(row) for row in s[3]]
        req[0][1] = 1  # agent 0 requests tool 1
        s[3] = tuple(tuple(r) for r in req)
        s = tuple(s)

        ps = env.perceived_state(s, 0)
        assert ps[3][0][1] == 1  # own request visible


# ---- goals ----


class TestGoals:
    def test_hold_goal(self, small_env):
        small_env.reset(seed=42)
        g = HoldGoal(small_env, agent_idx=0, tool_idx=0)
        state = small_env.get_state()
        # initially nobody holds anything
        assert g.is_achieved(state) == 0

    def test_hold_goal_hash_eq(self, small_env):
        g1 = HoldGoal(small_env, 0, 1)
        g2 = HoldGoal(small_env, 0, 1)
        g3 = HoldGoal(small_env, 1, 0)
        assert g1 == g2
        assert hash(g1) == hash(g2)
        assert g1 != g3

    def test_workbench_goal(self, small_env):
        small_env.reset(seed=42)
        state = small_env.get_state()
        _, wb, _, _ = state
        # find a tool that IS on agent 0's workbench
        for k in range(small_env.n_tools):
            if wb[0][k]:
                g = WorkbenchGoal(small_env, 0, k)
                assert g.is_achieved(state) == 1
                break

    def test_workbench_goal_hash_eq(self, small_env):
        g1 = WorkbenchGoal(small_env, 0, 1)
        g2 = WorkbenchGoal(small_env, 0, 1)
        assert g1 == g2 and hash(g1) == hash(g2)

    def test_goal_immutability(self, small_env):
        g = HoldGoal(small_env, 0, 0)
        with pytest.raises(AttributeError):
            g.agent_idx = 99


# ---- goal generator / sampler ----


class TestGoalGenSampler:
    def test_generator_yields_goals(self, example_env):
        gen = ToolsGoalGenerator(example_env)
        goals = list(
            gen.generate(example_env.get_state(), example_env.human_agent_indices[0])
        )
        assert len(goals) > 0
        for g, w in goals:
            assert isinstance(g, (HoldGoal, WorkbenchGoal, IdleGoal))
            assert w > 0

    def test_generator_weights_sum(self, example_env):
        example_env.reset(seed=123)
        gen = ToolsGoalGenerator(example_env)
        total = 0.0
        for agent_idx in example_env.human_agent_indices:
            for _, w in gen.generate(example_env.get_state(), agent_idx):
                total += w
        # weights are 1/total_goals, summed over all agents
        assert total > 0

    def test_sampler(self, example_env):
        example_env.reset(seed=123)
        sampler = ToolsGoalSampler(example_env)
        g, w = sampler.sample(
            example_env.get_state(), example_env.human_agent_indices[0]
        )
        assert isinstance(g, (HoldGoal, WorkbenchGoal, IdleGoal))
        assert w == 1.0


# ---- heuristic policy ----


class TestHeuristicPolicy:
    def test_returns_valid_distribution(self, example_env):
        example_env.reset(seed=123)
        gen = ToolsGoalGenerator(example_env)
        pol = ToolsHeuristicPolicy(example_env, gen, beta=5.0)
        state = example_env.get_state()
        nap = example_env.n_actions_per_agent
        for hi in example_env.human_agent_indices:
            dist = pol(state, hi)
            assert dist.shape == (nap[hi],)
            assert abs(dist.sum() - 1.0) < 1e-6
            assert (dist >= 0).all()

    def test_goal_conditioned(self, example_env):
        example_env.reset(seed=123)
        gen = ToolsGoalGenerator(example_env)
        pol = ToolsHeuristicPolicy(example_env, gen, beta=5.0)
        state = example_env.get_state()
        hi = example_env.human_agent_indices[0]
        nap = example_env.n_actions_per_agent
        goal = HoldGoal(example_env, hi, 0)
        dist = pol(state, hi, goal)
        assert dist.shape == (nap[hi],)
        assert abs(dist.sum() - 1.0) < 1e-6

    def test_sample(self, example_env):
        example_env.reset(seed=123)
        gen = ToolsGoalGenerator(example_env)
        pol = ToolsHeuristicPolicy(example_env, gen, beta=5.0)
        state = example_env.get_state()
        hi = example_env.human_agent_indices[0]
        action = pol.sample(state, hi)
        assert 0 <= action < example_env.n_actions_per_agent[hi]


# ---- rendering ----


class TestRendering:
    def test_render_produces_array(self, small_env):
        # render() returns None when render_mode is not "rgb_array"
        small_env.reset(seed=42)
        assert small_env.render() is None

    def test_render_rgb_array(self):
        env = ToolsWorldModel(
            n_agents=2,
            n_tools=2,
            max_steps=5,
            p_failure=0.0,
            seed=42,
            robot_agent_indices=[0],
            render_mode="rgb_array",
        )
        env.reset(seed=42)
        img = env.render()
        assert img is not None
        assert img.ndim == 3
        assert img.shape[2] == 3  # RGB


# ---- specific action effects ----


class TestActionEffects:
    def test_take_tool_from_own_workbench(self):
        """Agent can acquire a tool from their own workbench (reachable → take)."""
        env = ToolsWorldModel(
            n_agents=2,
            n_tools=1,
            max_steps=5,
            p_failure=0.0,
            seed=0,
            can_reach=np.ones((2, 2), dtype=bool),
            can_grab=np.ones((2, 2), dtype=bool),
            can_hear=np.ones((2, 2), dtype=bool),
        )
        env.reset(seed=0)
        # Ensure tool 0 is on agent 0's workbench
        remaining = env.max_steps
        wb = ((1,), (0,))
        hd = ((0,), (0,))
        rq = ((0,), (0,))
        env.set_state((remaining, wb, hd, rq))

        state = env.get_state()
        # Agent 0 acquires tool 0 (reachable → take), agent 1 acquires tool 0 (idle)
        trans = env.transition_probabilities(
            state, [_action_acquire(0), _action_acquire(0)]
        )
        assert trans is not None
        _, ns = trans[0]
        _, wb, holds, _ = ns
        assert holds[0][0] == 1
        assert wb[0][0] == 0

    def test_give_tool_to_neighbour(self):
        """Agent gives held tool to neighbour's workbench."""
        env = ToolsWorldModel(
            n_agents=2,
            n_tools=1,
            max_steps=5,
            p_failure=0.0,
            seed=0,
            can_reach=np.ones((2, 2), dtype=bool),
            can_grab=np.ones((2, 2), dtype=bool),
            can_hear=np.ones((2, 2), dtype=bool),
        )
        env.reset(seed=0)
        # Agent 0 holds tool 0
        remaining = env.max_steps
        wb = ((0,), (0,))
        hd = ((1,), (0,))
        rq = ((0,), (0,))
        env.set_state((remaining, wb, hd, rq))

        state = env.get_state()
        # Agent 0 gives to agent 1 (agent 1 is in give_targets[0])
        gt0 = env.give_targets[0]
        give_action = _action_give_idx(gt0.index(1), env.n_tools)
        # Agent 1 does a give (no-op since not holding anything)
        noop_1 = _action_give_idx(0, env.n_tools)
        trans = env.transition_probabilities(state, [give_action, noop_1])
        assert trans is not None
        _, ns = trans[0]
        _, wb, holds, _ = ns
        assert holds[0][0] == 0  # agent 0 no longer holds
        assert wb[1][0] == 1  # tool on agent 1's workbench

    def test_acquire_unreachable_sets_request(self):
        """Acquiring a tool not on a reachable workbench sets request flag."""
        env = ToolsWorldModel(
            n_agents=2,
            n_tools=2,
            max_steps=5,
            p_failure=0.0,
            seed=0,
            can_reach=np.eye(2, dtype=bool),  # can only reach self
            can_grab=np.ones((2, 2), dtype=bool),
            can_hear=np.ones((2, 2), dtype=bool),
        )
        env.reset(seed=0)
        # Ensure agent 0 does NOT have tool 1
        remaining = env.max_steps
        wb = ((0, 0), (1, 1))
        hd = ((0, 0), (0, 0))
        rq = ((0, 0), (0, 0))
        env.set_state((remaining, wb, hd, rq))

        state = env.get_state()
        # Agent 0 acquires tool 1 → not reachable → sets request
        trans = env.transition_probabilities(
            state, [_action_acquire(1), _action_acquire(0)]
        )
        assert trans is not None
        _, ns = trans[0]
        _, _, _, req = ns
        assert req[0][1] == 1

    def test_acquire_unreachable_cancels_previous_request(self):
        """Acquiring a new unreachable tool cancels the previous request."""
        env = ToolsWorldModel(
            n_agents=2,
            n_tools=2,
            max_steps=5,
            p_failure=0.0,
            seed=0,
            can_reach=np.eye(2, dtype=bool),  # can only reach self
            can_grab=np.ones((2, 2), dtype=bool),
            can_hear=np.ones((2, 2), dtype=bool),
        )
        env.reset(seed=0)
        remaining = env.max_steps
        wb = ((0, 0), (1, 1))
        hd = ((0, 0), (0, 0))
        rq = ((1, 0), (0, 0))  # agent 0 already requested tool 0
        env.set_state((remaining, wb, hd, rq))

        state = env.get_state()
        # Agent 0 now acquires tool 1 (unreachable → request, cancels old)
        trans = env.transition_probabilities(
            state, [_action_acquire(1), _action_acquire(0)]
        )
        assert trans is not None
        _, ns = trans[0]
        _, _, _, req = ns
        assert req[0][0] == 0  # old request cancelled
        assert req[0][1] == 1  # new request set

    def test_failure_produces_extra_successor(self):
        """With p_failure > 0 and a non-trivial action, more successors appear."""
        env = ToolsWorldModel(
            n_agents=2,
            n_tools=1,
            max_steps=5,
            p_failure=0.5,
            seed=0,
            can_reach=np.ones((2, 2), dtype=bool),
            can_grab=np.ones((2, 2), dtype=bool),
            can_hear=np.ones((2, 2), dtype=bool),
        )
        env.reset(seed=0)
        remaining = env.max_steps
        wb = ((1,), (0,))
        hd = ((0,), (0,))
        rq = ((0,), (0,))
        env.set_state((remaining, wb, hd, rq))

        state = env.get_state()
        # Agent 0 acquires tool, agent 1 acquires tool 0 (idle)
        trans = env.transition_probabilities(
            state, [_action_acquire(0), _action_acquire(0)]
        )
        assert trans is not None
        # Should have ≤ 3 successors (no-fail, agent0-fails, agent1-fails)
        assert len(trans) <= 3
        total = sum(p for p, _ in trans)
        assert abs(total - 1.0) < 1e-9

    def test_conflict_resolution_lower_index_wins(self):
        """When two agents try to take the same tool and can't grab from hands,
        the lower-index agent takes it and the higher-index agent fails."""
        env = ToolsWorldModel(
            n_agents=2,
            n_tools=1,
            max_steps=5,
            p_failure=0.0,
            seed=0,
            can_reach=np.ones((2, 2), dtype=bool),
            # can_grab only from self — cannot grab from each other's hand
            can_grab=np.eye(2, dtype=bool),
            can_hear=np.ones((2, 2), dtype=bool),
        )
        env.reset(seed=0)
        remaining = env.max_steps
        wb_t = ((1,), (0,))
        hd_t = ((0,), (0,))
        rq_t = ((0,), (0,))
        env.set_state((remaining, wb_t, hd_t, rq_t))

        state = env.get_state()
        # Both agents try to acquire tool 0
        trans = env.transition_probabilities(
            state, [_action_acquire(0), _action_acquire(0)]
        )
        assert trans is not None
        _, ns = trans[0]
        _, wb, holds, _ = ns
        # Agent 0 (lower index) picks it up first.
        # Agent 1 cannot grab from agent 0's hand (can_grab[1,0]=False),
        # and it's no longer on any workbench, so agent 1's acquire fails.
        assert holds[0][0] == 1
        assert holds[1][0] == 0

    def test_conflict_acquire_grab_blocked_same_timestep(self):
        """Even with can_grab[1,0]=True, agent 1 cannot grab a tool that
        agent 0 just claimed via 'acquire' in the same timestep."""
        env = ToolsWorldModel(
            n_agents=2,
            n_tools=1,
            max_steps=5,
            p_failure=0.0,
            seed=0,
            can_reach=np.ones((2, 2), dtype=bool),
            can_grab=np.ones((2, 2), dtype=bool),  # full grab access
            can_hear=np.ones((2, 2), dtype=bool),
        )
        env.reset(seed=0)
        remaining = env.max_steps
        wb_t = ((1,), (0,))
        hd_t = ((0,), (0,))
        rq_t = ((0,), (0,))
        env.set_state((remaining, wb_t, hd_t, rq_t))

        state = env.get_state()
        # Both try to acquire tool 0
        trans = env.transition_probabilities(
            state, [_action_acquire(0), _action_acquire(0)]
        )
        assert trans is not None
        _, ns = trans[0]
        _, wb, holds, _ = ns
        # Agent 0 claims tool 0; agent 1's acquire is blocked by the
        # claimed_tools lock even though can_grab[1,0] is True.
        assert holds[0][0] == 1
        assert holds[1][0] == 0

    def test_shape_validation_rejects_bad_can_hear(self):
        """Passing a wrongly-shaped can_hear raises ValueError."""
        with pytest.raises(ValueError, match="can_hear shape"):
            ToolsWorldModel(
                n_agents=2,
                n_tools=1,
                max_steps=3,
                p_failure=0.0,
                seed=0,
                can_hear=np.ones((3, 3), dtype=bool),
            )

    def test_perceived_state_invalid_agent_index(self):
        """perceived_state raises ValueError for out-of-range agent_index."""
        env = ToolsWorldModel(
            n_agents=2,
            n_tools=1,
            max_steps=3,
            p_failure=0.0,
            seed=0,
            can_hear=np.ones((2, 2), dtype=bool),
            can_reach=np.ones((2, 2), dtype=bool),
            can_grab=np.ones((2, 2), dtype=bool),
        )
        env.reset(seed=0)
        s = env.get_state()
        with pytest.raises(ValueError, match="agent_index"):
            env.perceived_state(s, 5)


# ---- parameter validation ----


class TestParameterValidation:
    """Tests for Waxman, grab_prob, and agent index validation in __init__."""

    def test_waxman_hear_alpha_out_of_range(self):
        with pytest.raises(ValueError, match="waxman_hear_alpha"):
            ToolsWorldModel(
                n_agents=2, n_tools=1, max_steps=3, seed=0, waxman_hear_alpha=1.5
            )

    def test_waxman_hear_alpha_negative(self):
        with pytest.raises(ValueError, match="waxman_hear_alpha"):
            ToolsWorldModel(
                n_agents=2, n_tools=1, max_steps=3, seed=0, waxman_hear_alpha=-0.1
            )

    def test_waxman_reach_alpha_out_of_range(self):
        with pytest.raises(ValueError, match="waxman_reach_alpha"):
            ToolsWorldModel(
                n_agents=2, n_tools=1, max_steps=3, seed=0, waxman_reach_alpha=2.0
            )

    def test_waxman_hear_beta_zero(self):
        with pytest.raises(ValueError, match="waxman_hear_beta"):
            ToolsWorldModel(
                n_agents=2, n_tools=1, max_steps=3, seed=0, waxman_hear_beta=0.0
            )

    def test_waxman_hear_beta_negative(self):
        with pytest.raises(ValueError, match="waxman_hear_beta"):
            ToolsWorldModel(
                n_agents=2, n_tools=1, max_steps=3, seed=0, waxman_hear_beta=-1.0
            )

    def test_waxman_reach_beta_zero(self):
        with pytest.raises(ValueError, match="waxman_reach_beta"):
            ToolsWorldModel(
                n_agents=2, n_tools=1, max_steps=3, seed=0, waxman_reach_beta=0.0
            )

    def test_grab_prob_out_of_range(self):
        with pytest.raises(ValueError, match="grab_prob"):
            ToolsWorldModel(n_agents=2, n_tools=1, max_steps=3, seed=0, grab_prob=1.5)

    def test_grab_prob_negative(self):
        with pytest.raises(ValueError, match="grab_prob"):
            ToolsWorldModel(n_agents=2, n_tools=1, max_steps=3, seed=0, grab_prob=-0.1)

    def test_robot_waxman_overrides_validated(self):
        with pytest.raises(ValueError, match="robot_waxman_hear_alpha"):
            ToolsWorldModel(
                n_agents=2, n_tools=1, max_steps=3, seed=0, robot_waxman_hear_alpha=5.0
            )
        with pytest.raises(ValueError, match="robot_waxman_reach_beta"):
            ToolsWorldModel(
                n_agents=2, n_tools=1, max_steps=3, seed=0, robot_waxman_reach_beta=-1.0
            )
        with pytest.raises(ValueError, match="robot_grab_prob"):
            ToolsWorldModel(
                n_agents=2, n_tools=1, max_steps=3, seed=0, robot_grab_prob=2.0
            )

    def test_valid_waxman_params_accepted(self):
        env = ToolsWorldModel(
            n_agents=2,
            n_tools=1,
            max_steps=3,
            seed=0,
            waxman_hear_alpha=0.5,
            waxman_hear_beta=0.3,
            waxman_reach_alpha=0.4,
            waxman_reach_beta=0.2,
            grab_prob=0.7,
        )
        assert env.n_agents == 2

    def test_robot_agent_index_out_of_range(self):
        with pytest.raises(ValueError, match="robot_agent_indices"):
            ToolsWorldModel(
                n_agents=2, n_tools=1, max_steps=3, seed=0, robot_agent_indices=[5]
            )

    def test_robot_agent_index_duplicate(self):
        with pytest.raises(ValueError, match="duplicate"):
            ToolsWorldModel(
                n_agents=3, n_tools=1, max_steps=3, seed=0, robot_agent_indices=[0, 0]
            )

    def test_robot_human_overlap(self):
        with pytest.raises(ValueError, match="disjoint"):
            ToolsWorldModel(
                n_agents=3,
                n_tools=1,
                max_steps=3,
                seed=0,
                robot_agent_indices=[0, 1],
                human_agent_indices_list=[1, 2],
            )

    def test_robot_human_incomplete_coverage(self):
        with pytest.raises(ValueError, match="cover all agents"):
            ToolsWorldModel(
                n_agents=3,
                n_tools=1,
                max_steps=3,
                seed=0,
                robot_agent_indices=[0],
                human_agent_indices_list=[1],
            )

    def test_p_failure_out_of_range(self):
        with pytest.raises(ValueError, match="p_failure"):
            ToolsWorldModel(n_agents=2, n_tools=1, max_steps=3, seed=0, p_failure=1.5)


# ---- pickle safety ----


class TestPickleSafety:
    """Verify ToolsWorldModel survives pickle round-trip."""

    def test_pickle_roundtrip(self, small_env):
        import pickle

        small_env.reset(seed=42)
        s1 = small_env.get_state()
        data = pickle.dumps(small_env)
        env2 = pickle.loads(data)
        s2 = env2.get_state()
        assert s1 == s2

    def test_pickle_preserves_graphs(self, small_env):
        import pickle

        data = pickle.dumps(small_env)
        env2 = pickle.loads(data)
        assert (small_env.can_hear == env2.can_hear).all()
        assert (small_env.can_reach == env2.can_reach).all()
        assert (small_env.can_grab == env2.can_grab).all()

    def test_pickle_preserves_roles(self, small_env):
        import pickle

        data = pickle.dumps(small_env)
        env2 = pickle.loads(data)
        assert env2.robot_agent_indices == small_env.robot_agent_indices
        assert env2.human_agent_indices == small_env.human_agent_indices
