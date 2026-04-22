import torch

from empo.learning_based.phase2.simplified_x_h import (
    compute_simplified_x_h_td_targets,
)


class CountingWorldModel:
    def __init__(self):
        self.calls = 0

    def transition_probabilities(self, state, actions):
        self.calls += 1
        if actions[0] == 0:
            return [(1.0, ("left",))]
        return [(1.0, ("right",))]


class NoTransitionWorldModel:
    def transition_probabilities(self, state, actions):
        raise AssertionError("inverse-dynamics path should not query transitions")


class HybridWorldModel:
    def __init__(self):
        self.calls = 0

    def transition_probabilities(self, state, actions):
        self.calls += 1
        if actions[0] == 0:
            return [(0.2, ("s_prime",)), (0.8, ("other",))]
        return [(1.0, ("s_prime",))]


class DummyInverseDynamics:
    def __init__(self, probs):
        self._logits = torch.log(torch.tensor([probs], dtype=torch.float32))

    def forward(self, state, next_state, world_model, human_agent_idx, device="cpu"):
        return self._logits.to(device)


def test_compute_simplified_x_h_td_targets_caches_transition_mass():
    world_model = CountingWorldModel()
    states = [("s",), ("s",), ("s",)]
    next_states = [("left",), ("right",), ("left",)]
    human_indices = [0, 0, 0]
    x_h_next_values = torch.tensor([1.0, 1.0, 1.0])
    robot_policy_per_state = {("s",): torch.tensor([1.0])}

    targets = compute_simplified_x_h_td_targets(
        states,
        next_states,
        human_indices,
        gamma_h=1.0,
        zeta=1.0,
        epsilon_h=0.0,
        num_actions=2,
        num_agents=1,
        human_agent_indices=[0],
        robot_agent_indices=[],
        x_h_next_values=x_h_next_values,
        robot_policy_per_state=robot_policy_per_state,
        action_index_to_tuple=lambda idx: (),
        other_human_probs_fn=lambda state, agent_index: [1.0],
        world_model=world_model,
        device="cpu",
    )

    assert torch.allclose(targets, torch.tensor([2.0, 2.0, 2.0]))
    assert world_model.calls == 2


def test_compute_simplified_x_h_td_targets_uses_inverse_dynamics_ratio_target():
    world_model = HybridWorldModel()
    inverse_dynamics = DummyInverseDynamics([0.3, 0.7])

    targets = compute_simplified_x_h_td_targets(
        [("s",)],
        [("s_prime",)],
        [0],
        gamma_h=0.5,
        zeta=1.0,
        epsilon_h=0.25,
        num_actions=2,
        num_agents=1,
        human_agent_indices=[0],
        robot_agent_indices=[],
        x_h_next_values=torch.tensor([2.0]),
        robot_policy_per_state={("s",): torch.tensor([1.0])},
        action_index_to_tuple=lambda idx: (),
        other_human_probs_fn=lambda state, agent_index: [0.75, 0.25],
        world_model=world_model,
        inverse_dynamics_network=inverse_dynamics,
        device="cpu",
    )

    expected_ratio = 0.75 * (0.7 / 0.25) + 0.25 * ((0.3 / 0.75 + 0.7 / 0.25) / 2.0)
    expected_target = 1.0 + 0.5 * (0.4 ** 0.0) * expected_ratio * 2.0

    assert torch.allclose(targets, torch.tensor([expected_target], dtype=torch.float32))
    assert world_model.calls == 2


def test_compute_simplified_x_h_td_targets_uses_marginal_factor_for_zeta_not_one():
    world_model = HybridWorldModel()
    inverse_dynamics = DummyInverseDynamics([0.3, 0.7])

    targets = compute_simplified_x_h_td_targets(
        [("s",)],
        [("s_prime",)],
        [0],
        gamma_h=0.5,
        zeta=2.0,
        epsilon_h=0.25,
        num_actions=2,
        num_agents=1,
        human_agent_indices=[0],
        robot_agent_indices=[],
        x_h_next_values=torch.tensor([2.0]),
        robot_policy_per_state={("s",): torch.tensor([1.0])},
        action_index_to_tuple=lambda idx: (),
        other_human_probs_fn=lambda state, agent_index: [0.75, 0.25],
        world_model=world_model,
        inverse_dynamics_network=inverse_dynamics,
        device="cpu",
    )

    expected_ratio = 0.75 * (0.7 / 0.25) + 0.25 * ((0.3 / 0.75 + 0.7 / 0.25) / 2.0)
    expected_target = 1.0 + (0.5 ** 2.0) * (0.4 ** 1.0) * (expected_ratio ** 2.0) * 2.0

    assert torch.allclose(targets, torch.tensor([expected_target], dtype=torch.float32))
    assert world_model.calls == 2