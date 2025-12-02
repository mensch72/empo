"""
Neural Network-based Stochastic Approximation for Human Policy Priors.

This package provides neural network function approximators for computing human
policy priors when the state space, number of agents, and goal space are too
large for tabular methods (backward induction).

The approach approximates the same Bellman-style computation as the tabular
method, but uses neural networks trained via stochastic gradient descent on
sampled states rather than exact computation over all states.

Main components:
    - StateEncoder: Encodes grid-based states into feature vectors (CNN-based)
    - GoalEncoder: Encodes possible goals (target positions) into feature vectors
    - AgentEncoder: Encodes agent attributes (position, direction, index)
    - QNetwork (h_Q): Maps (state, human, goal) -> Q-values for each action
    - PolicyPriorNetwork (h_phi): Maps (state, human) -> marginal policy prior
    - NeuralHumanPolicyPrior: HumanPolicyPrior implementation using neural networks
    - train_neural_policy_prior: Training function for the neural networks
    - PathDistanceCalculator: Computes path-based distances for reward shaping
    - OBJECT_TYPE_TO_CHANNEL: Mapping from object types to channel indices
    - NUM_OBJECT_TYPE_CHANNELS: Total number of object type channels
    - OVERLAPPABLE_OBJECTS, NON_OVERLAPPABLE_IMMOBILE_OBJECTS, NON_OVERLAPPABLE_MOBILE_OBJECTS:
        Object type categories for "other objects" channels
    - DEFAULT_ACTION_ENCODING, SMALL_ACTION_ENCODING: Standard action encodings

For multigrid-specific implementations, use the `multigrid` subpackage:
    >>> from empo.nn_based.multigrid import (
    ...     MultiGridStateEncoder,
    ...     MultiGridAgentEncoder,
    ...     MultiGridQNetwork,
    ...     MultiGridNeuralHumanPolicyPrior,
    ...     train_multigrid_neural_policy_prior,
    ... )

Mathematical background:
    The networks approximate:
    
    h_Q(s, h, g) ≈ Q^π(s, a, g) for Boltzmann policy π
    
    h_pi(s, h, g) = softmax(β * h_Q(s, h, g))  [goal-specific policy]
    
    h_phi(s, h) = E_g[h_pi(s, h, g)]  [marginal policy prior]

Example usage:
    >>> from empo.nn_based import NeuralHumanPolicyPrior, train_neural_policy_prior
    >>> 
    >>> # Train the neural network on sampled states
    >>> neural_prior = train_neural_policy_prior(
    ...     env=env,
    ...     human_agent_indices=[0, 1],
    ...     goal_sampler=goal_sampler,
    ...     num_episodes=1000,
    ...     beta=10.0
    ... )
    >>> 
    >>> # Use like tabular policy prior
    >>> action_dist = neural_prior(state, agent_idx=0, goal=my_goal)
    >>>
    >>> # Save trained model
    >>> neural_prior.save("my_policy.pt")
    >>>
    >>> # Load for a different environment
    >>> loaded_prior = NeuralHumanPolicyPrior.load(
    ...     "my_policy.pt", new_env, human_indices,
    ...     infeasible_actions_become=0
    ... )
"""

# Import generic world_model components from the main module
from empo.nn_based.neural_policy_prior import (
    # Constants for grid encoding
    OBJECT_TYPE_TO_CHANNEL,
    NUM_OBJECT_TYPE_CHANNELS,
    OVERLAPPABLE_OBJECTS,
    NON_OVERLAPPABLE_IMMOBILE_OBJECTS,
    NON_OVERLAPPABLE_MOBILE_OBJECTS,
    DEFAULT_ACTION_ENCODING,
    SMALL_ACTION_ENCODING,
    # Encoders (generic)
    StateEncoder,
    GoalEncoder,
    AgentEncoder,
    # Networks (generic)
    QNetwork,
    PolicyPriorNetwork,
    # Policy Prior (generic)
    NeuralHumanPolicyPrior,
    # Training (generic)
    train_neural_policy_prior,
    create_policy_prior_networks,
    # Reward Shaping
    PathDistanceCalculator,
)

# Also expose the multigrid subpackage
from empo.nn_based import multigrid

__all__ = [
    # Constants
    "OBJECT_TYPE_TO_CHANNEL",
    "NUM_OBJECT_TYPE_CHANNELS",
    "OVERLAPPABLE_OBJECTS",
    "NON_OVERLAPPABLE_IMMOBILE_OBJECTS",
    "NON_OVERLAPPABLE_MOBILE_OBJECTS",
    "DEFAULT_ACTION_ENCODING",
    "SMALL_ACTION_ENCODING",
    # Encoders
    "StateEncoder",
    "GoalEncoder", 
    "AgentEncoder",
    # Networks
    "QNetwork",
    "PolicyPriorNetwork",
    # Policy Prior
    "NeuralHumanPolicyPrior",
    # Training
    "train_neural_policy_prior",
    "create_policy_prior_networks",
    # Reward Shaping
    "PathDistanceCalculator",
    # Subpackage
    "multigrid",
]
