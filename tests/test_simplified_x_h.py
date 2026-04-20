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