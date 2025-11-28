import numpy as np
from itertools import product

from empo.possible_goal import PossibleGoalGenerator
from empo.human_policy_prior import TabularHumanPolicyPrior
from empo.world_model import WorldModel

def compute_human_policy_prior(world_model: WorldModel, human_agent_indices: list, possible_goal_generator: 'PossibleGoalGenerator', believed_others_policy = None, beta: float = 1, gamma: float = 1) -> dict:

    human_policy_priors = {} # these will be a mixture of system-1 and system-2 policies

#    Q_vectors = {}
    system2_policies = {} # these will be Boltzmann policies with fixed inverse temperature beta for now
    V_values = {} # these will be based on the system-2 policies for now

    num_agents = len(world_model.agents)
    num_actions = world_model.action_space.n
    actions = range(num_actions)

    if believed_others_policy is None:
        def bop (state, agent_index, action): 
            uniform_p = 1 / num_actions**(num_agents - 1)
            # each action profile for the other (!) agents gets the same probability, and the agent's own action is always put to -1 since it will be overwritten in the loop below:
            all_actions = list(range(num_actions))
            return [(uniform_p, list(action_profile)) for action_profile in product(*[
                [-1] if idx == agent_index else all_actions
                for idx in range(num_agents)])]
        believed_others_policy = bop

    # first get the dag of the world model:
    states, state_to_idx, successors, transitions = world_model.get_dag(return_probabilities=True)
    print(transitions[0][0])

    # now loop over the nodes in reverse topological order:
    for state_index in range(len(states)-1, -1, -1):
        state = states[state_index]
        vs = V_values[state_index] = {}
        if world_model.is_terminal(state):
            # in terminal states, policy and Q values are undefined, only V values need computation:
            for agent_index in human_agent_indices:
                vsi = vs[agent_index] = {}
                for possible_goal, _ in possible_goal_generator.generate(state, agent_index):
                    vsi[possible_goal] = possible_goal.is_achieved(state)
        else:
#            qs = Q_vectors[state_index] = {}
            ps = system2_policies[state] = {}
            for agent_index in human_agent_indices:
#                qsi = qs[agent_index] = {}
                psi = ps[agent_index] = {}
                vsi = vs[agent_index] = {}
                for possible_goal, _ in possible_goal_generator.generate(state, agent_index):
                    # if the goal is achieved in that state, the human will not care about future rewards and use a uniform policy:
                    if possible_goal.is_achieved(state):
                        vsi[possible_goal] = 1
#                        qsi[possible_goal] = np.ones(num_actions)
                        psi[possible_goal] = np.ones(num_actions) / num_actions
                    else:
                        # otherwise, compute the Q values as expected future V values, and the policy as a Boltzmann policy based on those Q values:
                        expected_Vs = np.zeros(num_actions)
                        for action in actions:
                            for action_profile_prob, action_profile in believed_others_policy(state, agent_index, action):
                                action_profile[agent_index] = action
                                # TODO: convert profile [a,b,c] into index a + b*num_actions + c*num_actions*num_actions ...
                                action_profile_index = 0
                                for idx, a in enumerate(action_profile):
                                    action_profile_index += a * (num_actions ** idx)
                                _, next_state_probabilities, next_state_indices = transitions[state_index][action_profile_index]
                                for i, next_state_prob in enumerate(next_state_probabilities):
                                    expected_Vs[action] += action_profile_prob * next_state_prob * V_values[next_state_indices[i]][agent_index][possible_goal]
                        q = gamma * expected_Vs
#                        qsi[possible_goal] = q
                        # Boltzmann policy:
                        p = np.exp(beta * q)
                        p /= np.sum(p)
                        psi[possible_goal] = p
                        vsi[possible_goal] = np.sum(p * q)
    
    human_policy_priors = system2_policies # TODO: mix with system-1 policies!

    return TabularHumanPolicyPrior(
        world_model=world_model, human_agent_indices=human_agent_indices, possible_goal_generator=possible_goal_generator, values=human_policy_priors
    )
