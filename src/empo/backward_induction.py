import numpy as np
from itertools import product

from empo.possible_goal import PossibleGoalGenerator
from empo.human_policy_prior import TabularHumanPolicyPrior
from empo.world_model import WorldModel

DEBUG = False #True

def compute_human_policy_prior(world_model: WorldModel, human_agent_indices: list, possible_goal_generator: 'PossibleGoalGenerator', believed_others_policy = None, beta: float = 1, gamma: float = 1) -> dict:

    human_policy_priors = {} # these will be a mixture of system-1 and system-2 policies

#    Q_vectors = {}
    system2_policies = {} # these will be Boltzmann policies with fixed inverse temperature beta for now
    # V_values will be indexed as V_values[state_index][agent_index][possible_goal]
    # Using nested lists for faster access on first two levels

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

    # Precompute powers for action profile indexing
    action_powers = num_actions ** np.arange(num_agents)

    # first get the dag of the world model:
    states, state_to_idx, successors, transitions = world_model.get_dag(return_probabilities=True)
    print(f"No. of states: {len(states)}")
    
    # Initialize V_values as nested lists for faster access
    V_values = [[{} for _ in range(num_agents)] for _ in range(len(states))]

    # now loop over the nodes in reverse topological order:
    for state_index in range(len(states)-1, -1, -1):
        if DEBUG:
            print(f"Processing state {state_index}")
        state = states[state_index]
        if world_model.is_terminal(state):
            if DEBUG:
                print(f"  Terminal state")
            # in terminal states, policy and Q values are undefined, only V values need computation:
            for agent_index in human_agent_indices:
                if DEBUG:
                    print(f"  Human agent {agent_index}")
                for possible_goal, _ in possible_goal_generator.generate(state, agent_index):
                    v = V_values[state_index][agent_index][possible_goal] = possible_goal.is_achieved(state)
                    if DEBUG:
                        print(f"    Possible goal: {possible_goal}, V = {v:.4f}")
        else:
#            qs = Q_vectors[state_index] = {}
            ps = system2_policies[state] = {}
            for agent_index in human_agent_indices:
                if DEBUG:
                    print(f"  Human agent {agent_index}")
#                qsi = qs[agent_index] = {}
                psi = ps[agent_index] = {}
                for possible_goal, _ in possible_goal_generator.generate(state, agent_index):
                    if DEBUG:
                        print(f"    Possible goal: {possible_goal}")
                    # if the goal is achieved in that state, the human will not care about future rewards and use a uniform policy:
                    if possible_goal.is_achieved(state):
                        if DEBUG:
                            print(f"      Goal achieved in this state; using uniform policy")
                        V_values[state_index][agent_index][possible_goal] = 1
#                        qsi[possible_goal] = np.ones(num_actions)
                        psi[possible_goal] = np.ones(num_actions) / num_actions
                    else:
                        # otherwise, compute the Q values as expected future V values, and the policy as a Boltzmann policy based on those Q values:
                        expected_Vs = np.zeros(num_actions)
                        for action in actions:
                            v = 0
                            for action_profile_prob, action_profile in believed_others_policy(state, agent_index, action):
                                action_profile[agent_index] = action
                                # convert profile [a,b,c] into index a + b*num_actions + c*num_actions*num_actions ...
                                # Optimized base conversion using precomputed powers
                                action_profile_index = np.dot(action_profile, action_powers)
                                _, next_state_probabilities, next_state_indices = transitions[state_index][action_profile_index]
                                # Vectorized computation using numpy
                                v_values_array = np.array([V_values[next_state_indices[i]][agent_index][possible_goal] 
                                                          for i in range(len(next_state_indices))])
                                v += action_profile_prob * np.dot(next_state_probabilities, v_values_array)
                            expected_Vs[action] = v
                        q = gamma * expected_Vs
#                        qsi[possible_goal] = q
                        # Boltzmann policy:
                        p = np.exp(beta * q)
                        p /= np.sum(p)
                        psi[possible_goal] = p
                        v = V_values[state_index][agent_index][possible_goal] = np.sum(p * q)
                        if DEBUG:
                            print(f"      Goal not achieved; V = {v:.4f}")
    
    human_policy_priors = system2_policies # TODO: mix with system-1 policies!

    return TabularHumanPolicyPrior(
        world_model=world_model, human_agent_indices=human_agent_indices, possible_goal_generator=possible_goal_generator, values=human_policy_priors
    )
