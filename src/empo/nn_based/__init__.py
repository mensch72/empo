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
"""

from empo.nn_based.neural_policy_prior import (
    # Encoders
    StateEncoder,
    GoalEncoder,
    AgentEncoder,
    # Networks
    QNetwork,
    PolicyPriorNetwork,
    # Policy Prior
    NeuralHumanPolicyPrior,
    # Training
    train_neural_policy_prior,
    create_policy_prior_networks,
)

__all__ = [
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
]
