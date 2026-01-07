"""
Exploration policies for Phase 2 training in multigrid environments.

This module provides exploration policies that can be used during epsilon-greedy
training for both robots and humans.

Classes:
    MultiGridRobotExplorationPolicy: Simple Markovian policy for robots
    MultiGridHumanExplorationPolicy: Simple Markovian policy for humans  
    MultiGridMultiStepExplorationPolicy: Non-Markovian multi-step policy for both
"""

from typing import Any, List, Optional, Tuple, Union

import numpy as np

from empo.human_policy_prior import HumanPolicyPrior
from empo.robot_policy import RobotPolicy


class MultiGridRobotExplorationPolicy(RobotPolicy):
    """
    Simple exploration policy for epsilon-greedy robot action selection.
    
    This policy samples actions according to fixed probabilities, but avoids
    attempting "forward" when a robot cannot move forward (blocked by wall,
    object that can't be pushed, etc.).
    
    When forward is blocked for a robot, its probability mass is redistributed 
    proportionally to the other actions for that robot.
    
    Supports multiple robots - each robot's action is sampled independently.
    
    Designed for use with SmallActions (0=still, 1=left, 2=right, 3=forward).
    
    Example usage:
        # Prefer forward (0.6), then right (0.2), with low chance for still/left
        exploration = MultiGridRobotExplorationPolicy(
            action_probs=[0.1, 0.1, 0.2, 0.6]  # still, left, right, forward
        )
        trainer = MultiGridPhase2Trainer(
            ...,
            robot_exploration_policy=exploration
        )
    """
    
    def __init__(
        self,
        action_probs: Optional[List[float]] = None,
        robot_agent_indices: Optional[List[int]] = None
    ):
        """
        Initialize the exploration policy.
        
        Args:
            action_probs: Probabilities for each action [still, left, right, forward].
                         Default: [0.1, 0.1, 0.2, 0.6] (bias toward forward/right).
            robot_agent_indices: List of robot agent indices. If None, will be 
                                 detected from world model on reset().
        """
        if action_probs is None:
            action_probs = [0.1, 0.1, 0.2, 0.6]  # still, left, right, forward
        
        if len(action_probs) != 4:
            raise ValueError(f"action_probs must have 4 elements, got {len(action_probs)}")
        
        total = sum(action_probs)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"action_probs must sum to 1.0, got {total}")
        
        self.action_probs = np.array(action_probs, dtype=np.float64)
        self._robot_agent_indices = robot_agent_indices
        self._world_model = None
    
    def reset(self, world_model: Any) -> None:
        """
        Reset the policy at episode start.
        
        Args:
            world_model: The MultiGrid environment.
        """
        self._world_model = world_model
        # Auto-detect robot indices from world model if not provided
        if self._robot_agent_indices is None and hasattr(world_model, 'get_robot_agent_indices'):
            self._robot_agent_indices = world_model.get_robot_agent_indices()
    
    @property
    def robot_agent_indices(self) -> List[int]:
        """Get the robot agent indices."""
        if self._robot_agent_indices is None:
            return [1]  # Default: single robot at index 1
        return self._robot_agent_indices
    
    def sample(self, state: Any) -> Tuple[int, ...]:
        """
        Sample exploration actions for all robots, avoiding forward when blocked.
        
        Each robot's action is sampled independently.
        
        Args:
            state: Environment state tuple from get_state().
        
        Returns:
            Tuple of action indices, one per robot: (action_robot0, action_robot1, ...)
            where each action is 0=still, 1=left, 2=right, 3=forward
        """
        actions = []
        for robot_idx in self.robot_agent_indices:
            action = self._sample_single_robot_action(robot_idx, state)
            actions.append(action)
        
        return tuple(actions)
    
    def _sample_single_robot_action(self, robot_idx: int, state: Any) -> int:
        """
        Sample an action for a single robot.
        
        Uses the environment's can_forward() method to check if forward movement
        is possible, accounting for robot capabilities (can push rocks, can enter
        magic walls).
        
        Args:
            robot_idx: Index of the robot agent
            state: Full environment state tuple from get_state()
        
        Returns:
            Action index: 0=still, 1=left, 2=right, 3=forward
        """
        if self._world_model is None:
            # No world model - can't check if forward is blocked
            return int(np.random.choice(4, p=self.action_probs))
        
        # Use the environment's can_forward method which accounts for agent capabilities
        # We use the full state passed by the caller to avoid inconsistencies
        # with the world model's internal current state.
        can_move_forward = self._world_model.can_forward(state, robot_idx)
        
        if not can_move_forward:
            # Redistribute forward probability to other actions
            probs = self.action_probs.copy()
            forward_prob = probs[3]
            probs[3] = 0.0
            
            # Distribute proportionally to remaining actions
            remaining = probs.sum()
            if remaining > 0:
                probs *= (1.0 + forward_prob / remaining)
            else:
                # All probabilities were zero except forward - use uniform
                probs = np.array([1/3, 1/3, 1/3, 0.0])
            
            return int(np.random.choice(4, p=probs))
        else:
            return int(np.random.choice(4, p=self.action_probs))


class MultiGridHumanExplorationPolicy(HumanPolicyPrior):
    """
    Smart exploration policy for human agents in multigrid environments.
    
    This policy samples actions according to fixed probabilities, but avoids
    attempting "forward" when a human cannot move forward. Unlike robot exploration,
    this policy accounts for human-specific limitations:
    
    - Humans can push blocks (all agents can)
    - Humans CANNOT push rocks (requires can_push_rocks=True)
    - Humans CANNOT enter magic walls (requires can_enter_magic_walls=True)
    
    When forward is blocked for a human, its probability mass is redistributed 
    proportionally to the other actions for that human.
    
    Designed for use with SmallActions (0=still, 1=left, 2=right, 3=forward).
    
    Example usage:
        # Prefer forward (0.6), then right (0.2), with low chance for still/left
        exploration = MultiGridHumanExplorationPolicy(
            world_model=env,
            human_agent_indices=[0],
            action_probs=[0.1, 0.1, 0.2, 0.6]  # still, left, right, forward
        )
        trainer = MultiGridPhase2Trainer(
            ...,
            human_exploration_policy=exploration
        )
    """
    
    def __init__(
        self,
        world_model: Any = None,
        human_agent_indices: List[int] = None,
        action_probs: Optional[List[float]] = None,
    ):
        """
        Initialize the exploration policy.
        
        Args:
            world_model: The environment/world model (can be set later via set_world_model).
            human_agent_indices: List of human agent indices (can be empty, not used directly).
            action_probs: Probabilities for each action [still, left, right, forward].
                         Default: [0.1, 0.1, 0.2, 0.6] (bias toward forward/right).
        """
        # Initialize parent with world_model and human_agent_indices
        # Use empty list if not provided (indices not used in this implementation)
        super().__init__(world_model, human_agent_indices or [])
        
        if action_probs is None:
            action_probs = [0.1, 0.1, 0.2, 0.6]  # still, left, right, forward
        
        if len(action_probs) != 4:
            raise ValueError(f"action_probs must have 4 elements, got {len(action_probs)}")
        
        total = sum(action_probs)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"action_probs must sum to 1.0, got {total}")
        
        self.action_probs = np.array(action_probs, dtype=np.float64)
    
    def reset(self, world_model: Any) -> None:
        """
        Reset the policy at episode start.
        
        Args:
            world_model: The environment/world model (must be a MultiGridEnv).
        """
        self.set_world_model(world_model)
    
    def __call__(
        self,
        state: Any,
        human_agent_index: int,
        possible_goal: Any = None
    ) -> np.ndarray:
        """
        Get action distribution for a human agent (required by HumanPolicyPrior).
        
        Returns the base action probabilities, adjusted if forward is blocked.
        
        Args:
            state: Current world state.
            human_agent_index: Index of the human agent.
            possible_goal: Goal (not used in this exploration policy).
        
        Returns:
            np.ndarray: Probability distribution over actions.
        """
        if self.world_model is None:
            return self.action_probs.copy()
        
        can_move_forward = self.world_model.can_forward(state, human_agent_index)
        
        if not can_move_forward:
            probs = self.action_probs.copy()
            forward_prob = probs[3]
            probs[3] = 0.0
            remaining = probs.sum()
            if remaining > 0:
                probs *= (1.0 + forward_prob / remaining)
            else:
                probs = np.array([1/3, 1/3, 1/3, 0.0])
            return probs
        else:
            return self.action_probs.copy()
    
    def sample(self, state: Any, human_agent_index: int, goal: Any = None) -> int:
        """
        Sample an exploration action for a single human, avoiding forward when blocked.
        
        The goal parameter is accepted for interface compatibility but currently
        not used (exploration is goal-independent).
        
        Uses the environment's can_forward() method to check if the human can
        move forward, accounting for human-specific limitations:
        - Cannot push rocks (only agents with can_push_rocks=True can)
        - Cannot enter magic walls (only agents with can_enter_magic_walls=True can)
        
        Args:
            state: Environment state tuple from get_state().
            human_agent_index: Index of the human agent.
            goal: The human's current goal (not used in this implementation).
        
        Returns:
            Action index: 0=still, 1=left, 2=right, 3=forward
        """
        # Get action distribution (handles forward blocking)
        probs = self(state, human_agent_index, goal)
        return int(np.random.choice(len(probs), p=probs))


class MultiGridMultiStepExplorationPolicy(RobotPolicy, HumanPolicyPrior):
    """
    Non-Markovian multi-step exploration policy for multigrid environments.
    
    This policy implements temporally-extended exploration by sampling multi-step
    action sequences and executing them stepwise. This breaks the Markovian property
    of simple exploration policies and enables more directed spatial exploration.
    
    The policy samples one of these multi-step sequence types:
    - "still": k times still (stay in place)
    - "forward": k times forward (move straight ahead)
    - "left_forward": turn left, then k times forward
    - "right_forward": turn right, then k times forward  
    - "back_forward": turn left twice (180°), then k times forward
    
    In each case, k >= 1 is drawn from a geometric distribution with configurable
    expected value (can be different for each sequence type).
    
    Sequences are only started if the required forward movement is possible:
    - "forward" requires can_forward in current direction
    - "left_forward" requires can_forward after turning left
    - "right_forward" requires can_forward after turning right
    - "back_forward" requires can_forward after turning 180°
    
    The can_forward check uses the environment's can_forward() method which
    correctly accounts for agent-specific capabilities:
    - Robots (can_push_rocks=True, can_enter_magic_walls=True) can push rocks
    - Humans (can_push_rocks=False) cannot push rocks
    
    If forward movement becomes blocked during sequence execution (e.g., another
    agent moves into the cell), the sequence is cancelled and a new one is sampled.
    
    This policy inherits from BOTH RobotPolicy and HumanPolicyPrior, providing:
    - RobotPolicy.sample(state) -> tuple of actions for all agents
    - HumanPolicyPrior.__call__(state, agent_index, goal) -> action distribution
    - HumanPolicyPrior.sample(state, agent_index, goal) -> single action
    
    Example usage for robots:
        exploration = MultiGridMultiStepExplorationPolicy(
            agent_indices=[1],  # Robot at index 1
            sequence_probs={'still': 0.1, 'forward': 0.5, 'left_forward': 0.15,
                           'right_forward': 0.15, 'back_forward': 0.1},
            expected_k=2.0
        )
        trainer = MultiGridPhase2Trainer(
            robot_exploration_policy=exploration,
            ...
        )
    
    Example usage for humans:
        exploration = MultiGridMultiStepExplorationPolicy(
            agent_indices=[0],  # Human at index 0
            sequence_probs={'still': 0.1, 'forward': 0.5, 'left_forward': 0.15,
                           'right_forward': 0.15, 'back_forward': 0.1},
            expected_k={'forward': 3.0, 'left_forward': 2.0}  # Per-type expected_k
        )
        trainer = MultiGridPhase2Trainer(
            human_exploration_policy=exploration,
            ...
        )
    """
    
    # Action constants (SmallActions)
    ACTION_STILL = 0
    ACTION_LEFT = 1
    ACTION_RIGHT = 2
    ACTION_FORWARD = 3
    
    # Sequence type constants
    SEQ_STILL = 'still'
    SEQ_FORWARD = 'forward'
    SEQ_LEFT_FORWARD = 'left_forward'
    SEQ_RIGHT_FORWARD = 'right_forward'
    SEQ_BACK_FORWARD = 'back_forward'
    
    ALL_SEQUENCE_TYPES = [SEQ_STILL, SEQ_FORWARD, SEQ_LEFT_FORWARD, SEQ_RIGHT_FORWARD, SEQ_BACK_FORWARD]
    
    def __init__(
        self,
        agent_indices: Optional[List[int]] = None,
        sequence_probs: Optional[dict] = None,
        expected_k: Union[float, dict] = 2.0,
        world_model: Any = None,
    ):
        """
        Initialize the multi-step exploration policy.
        
        Args:
            agent_indices: List of agent indices this policy applies to.
                          If None, will be auto-detected from world_model on reset().
            sequence_probs: Dictionary mapping sequence type to relative probability.
                           Keys should be from: 'still', 'forward', 'left_forward',
                           'right_forward', 'back_forward'.
                           Default: {'still': 0.05, 'forward': 0.5, 'left_forward': 0.18,
                                    'right_forward': 0.18, 'back_forward': 0.09}
                           (biases toward forward, then 90° turns, then 180° turn)
            expected_k: Expected value of k in the geometric distribution.
                       k is the number of steps in the sequence (after any turns).
                       Can be either:
                       - A single float (>= 1.0): Used for all sequence types. Default: 2.0.
                       - A dict mapping sequence type to expected_k value, e.g.:
                         {'still': 1.0, 'forward': 3.0, 'left_forward': 2.0, ...}
                         Missing keys use default of 2.0.
                       The success probability p is computed as p = 1/expected_k.
            world_model: Optional world model reference. Can be set later via reset().
        """
        # Initialize both parent classes
        # Note: RobotPolicy has no __init__, HumanPolicyPrior needs world_model and indices
        # We handle this manually since we have custom initialization logic
        
        # Process expected_k - can be float or dict
        if isinstance(expected_k, dict):
            # Validate all values
            for seq_type, k_val in expected_k.items():
                if seq_type not in self.ALL_SEQUENCE_TYPES:
                    raise ValueError(f"Unknown sequence type in expected_k: {seq_type}")
                if k_val < 1.0:
                    raise ValueError(f"expected_k['{seq_type}'] must be >= 1.0, got {k_val}")
            # Store as dict, with default for missing keys
            self._expected_k_dict = {seq: expected_k.get(seq, 2.0) for seq in self.ALL_SEQUENCE_TYPES}
            self.expected_k = None  # Indicate dict mode
        else:
            if expected_k < 1.0:
                raise ValueError(f"expected_k must be >= 1.0, got {expected_k}")
            self.expected_k = expected_k
            self._expected_k_dict = None  # Indicate single-value mode
        
        if sequence_probs is None:
            # Default: prefer forward > 90° turns > 180° turn > still
            sequence_probs = {
                self.SEQ_STILL: 0.05,
                self.SEQ_FORWARD: 0.50,
                self.SEQ_LEFT_FORWARD: 0.18,
                self.SEQ_RIGHT_FORWARD: 0.18,
                self.SEQ_BACK_FORWARD: 0.09,
            }
        
        # Validate sequence_probs keys
        for key in sequence_probs:
            if key not in self.ALL_SEQUENCE_TYPES:
                raise ValueError(f"Unknown sequence type: {key}. Valid types: {self.ALL_SEQUENCE_TYPES}")
        
        # Normalize probabilities
        total = sum(sequence_probs.values())
        if total <= 0:
            raise ValueError("sequence_probs must have at least one non-zero probability")
        
        self.sequence_probs = {k: v / total for k, v in sequence_probs.items()}
        
        self._agent_indices = agent_indices
        
        # Set up world_model and human_agent_indices for HumanPolicyPrior compatibility
        self.world_model = world_model
        # human_agent_indices is provided by property, but we store _agent_indices
        
        # Per-agent state for ongoing sequences
        # Maps agent_index -> {'remaining_actions': [...], 'current_type': str}
        self._agent_sequences: dict = {}
    
    def _get_expected_k(self, seq_type: str) -> float:
        """Get the expected_k value for a sequence type."""
        if self._expected_k_dict is not None:
            return self._expected_k_dict.get(seq_type, 2.0)
        return self.expected_k
    
    def _get_geom_p(self, seq_type: str) -> float:
        """Get the geometric distribution parameter p for a sequence type."""
        return 1.0 / self._get_expected_k(seq_type)
    
    def reset(self, world_model: Any) -> None:
        """
        Reset the policy at episode start.
        
        Clears all ongoing sequences and caches the world model reference.
        
        Args:
            world_model: The MultiGrid environment.
        """
        self.world_model = world_model
        self._agent_sequences = {}  # Clear all ongoing sequences
        
        # Auto-detect agent indices if not provided
        if self._agent_indices is None:
            if hasattr(world_model, 'get_robot_agent_indices'):
                self._agent_indices = world_model.get_robot_agent_indices()
            elif hasattr(world_model, 'get_human_agent_indices'):
                self._agent_indices = world_model.get_human_agent_indices()
    
    def set_world_model(self, world_model: Any) -> None:
        """
        Set the world model reference (HumanPolicyPrior interface).
        
        Args:
            world_model: The MultiGrid environment.
        """
        self.world_model = world_model
    
    @property
    def agent_indices(self) -> List[int]:
        """Get the agent indices this policy applies to."""
        if self._agent_indices is None:
            return [0]  # Default: single agent at index 0
        return self._agent_indices
    
    @property
    def human_agent_indices(self) -> List[int]:
        """Alias for agent_indices (HumanPolicyPrior interface)."""
        return self.agent_indices
    
    def sample(self, state: Any, human_agent_index: Optional[int] = None, 
               possible_goal: Any = None, agent_index: Optional[int] = None,
               goal: Any = None) -> Any:
        """
        Sample action(s) for the given state.
        
        This method supports two call signatures:
        1. sample(state) - Returns tuple of actions for all agents (RobotPolicy interface)
        2. sample(state, human_agent_index, possible_goal=None) - Returns single action 
           for one agent (HumanPolicyPrior interface)
        3. sample(state, agent_index=N) - Returns single action for agent N
           (legacy RobotPolicy-style interface)
        
        The possible_goal/goal parameter is accepted for interface compatibility but not used
        (exploration is goal-independent).
        
        Args:
            state: Environment state tuple from get_state().
            human_agent_index: If provided, sample for this specific agent only.
            possible_goal: Ignored (for HumanPolicyPrior compatibility).
            agent_index: Alias for human_agent_index (legacy compatibility).
            goal: Alias for possible_goal (legacy compatibility).
        
        Returns:
            If human_agent_index/agent_index is None: Tuple of action indices, one per agent.
            If human_agent_index/agent_index is provided: Single action index (int).
        """
        # Support both parameter names for backward compatibility
        idx = human_agent_index if human_agent_index is not None else agent_index
        
        if idx is not None:
            # Single-agent call (HumanPolicyPrior-style)
            return self._sample_single_agent(idx, state)
        else:
            # Multi-agent call (RobotPolicy-style)
            actions = []
            for i in self.agent_indices:
                action = self._sample_single_agent(i, state)
                actions.append(action)
            return tuple(actions)
    
    def __call__(
        self,
        state: Any,
        human_agent_index: Optional[int] = None,
        possible_goal: Any = None
    ) -> np.ndarray:
        """
        Get action distribution for agent(s).
        
        This method supports two call signatures:
        1. __call__(state) - Returns joint action probabilities over all action 
           combinations for all agents (RobotPolicy interface). Shape: (num_actions^num_agents,)
        2. __call__(state, human_agent_index, possible_goal=None) - Returns single-agent
           action probabilities (HumanPolicyPrior interface). Shape: (4,)
        
        Returns the probability distribution over actions based on the current
        sequence state. If no sequence is active, returns the probability of
        each action being the first action of a feasible sequence.
        
        Args:
            state: Current world state.
            human_agent_index: If provided, return distribution for this specific agent only.
                              If None, return joint distribution over all agents.
            possible_goal: Ignored (exploration is goal-independent).
        
        Returns:
            np.ndarray: If human_agent_index is provided, probability distribution over 
                       single-agent actions [still, left, right, forward] with shape (4,).
                       If human_agent_index is None, joint probability distribution over
                       all action combinations with shape (num_actions^num_agents,).
        """
        if human_agent_index is not None:
            # Single-agent call (HumanPolicyPrior interface)
            return self._get_single_agent_probs(state, human_agent_index)
        else:
            # Multi-agent call (RobotPolicy interface)
            # Compute joint probabilities assuming independence between agents
            num_actions = 4  # SmallActions: still, left, right, forward
            agent_probs = []
            for agent_idx in self.agent_indices:
                probs = self._get_single_agent_probs(state, agent_idx)
                agent_probs.append(probs)
            
            if len(agent_probs) == 0:
                # No agents - return uniform over single action
                return np.ones(num_actions) / num_actions
            
            if len(agent_probs) == 1:
                # Single agent - return its probabilities directly
                return agent_probs[0]
            
            # Multiple agents - compute joint probabilities via outer product
            # For 2 agents with 4 actions each: result is 4*4 = 16 probabilities
            # Joint index = a0 * 4 + a1 for agents [a0, a1]
            # This matches the action_tuple_to_index convention in RobotQNetwork
            joint_probs = agent_probs[0]
            for i in range(1, len(agent_probs)):
                # Outer product: joint_probs[i,j] = joint_probs[i] * agent_probs[j]
                joint_probs = np.outer(joint_probs, agent_probs[i]).flatten()
            
            return joint_probs
    
    def _get_single_agent_probs(self, state: Any, agent_index: int) -> np.ndarray:
        """
        Get action probabilities for a single agent.
        
        Args:
            state: Current world state.
            agent_index: Index of the agent.
        
        Returns:
            np.ndarray: Probability distribution over actions [still, left, right, forward].
        """
        # Get agent's current direction from state
        agent_dir = self._get_agent_direction(state, agent_index)
        
        # Check if agent has an ongoing sequence
        if agent_index in self._agent_sequences:
            seq_info = self._agent_sequences[agent_index]
            if seq_info['remaining_actions']:
                # Return deterministic distribution for current sequence action
                next_action = seq_info['remaining_actions'][0]
                probs = np.zeros(4, dtype=np.float64)
                probs[next_action] = 1.0
                return probs
        
        # No ongoing sequence - compute distribution based on feasible sequences
        probs = np.zeros(4, dtype=np.float64)
        
        # Build list of feasible sequence types and their first actions
        feasible_seq_types = []
        feasible_first_actions = []
        feasible_probs = []
        
        for seq_type in self.ALL_SEQUENCE_TYPES:
            if seq_type not in self.sequence_probs:
                continue
            base_prob = self.sequence_probs[seq_type]
            if base_prob <= 0:
                continue
            
            first_action = self._get_first_action_for_sequence(seq_type)
            is_feasible = self._is_sequence_feasible(seq_type, state, agent_index, agent_dir)
            
            if is_feasible:
                feasible_seq_types.append(seq_type)
                feasible_first_actions.append(first_action)
                feasible_probs.append(base_prob)
        
        if not feasible_seq_types:
            # No feasible sequences - fall back to uniform over non-forward actions
            probs[:3] = 1.0 / 3.0
            return probs
        
        # Normalize feasible probabilities
        total = sum(feasible_probs)
        for first_action, prob in zip(feasible_first_actions, feasible_probs):
            probs[first_action] += prob / total
        
        return probs
    
    def _sample_single_agent(self, agent_index: int, state: Any) -> int:
        """
        Sample an action for a single agent, continuing or starting a sequence.
        
        Args:
            agent_index: Index of the agent.
            state: Environment state.
        
        Returns:
            Action index: 0=still, 1=left, 2=right, 3=forward
        """
        # Get agent's current direction from state
        agent_dir = self._get_agent_direction(state, agent_index)
        
        # Check if agent has an ongoing sequence
        if agent_index in self._agent_sequences:
            seq_info = self._agent_sequences[agent_index]
            
            if seq_info['remaining_actions']:
                next_action = seq_info['remaining_actions'][0]
                
                # If next action is forward, check if still possible
                if next_action == self.ACTION_FORWARD:
                    if self.world_model is not None:
                        can_forward = self.world_model.can_forward(state, agent_index)
                        if not can_forward:
                            # Forward blocked - cancel sequence and sample new one
                            del self._agent_sequences[agent_index]
                            return self._start_new_sequence(agent_index, state, agent_dir)
                
                # Pop and return the action
                seq_info['remaining_actions'].pop(0)
                if not seq_info['remaining_actions']:
                    # Sequence completed
                    del self._agent_sequences[agent_index]
                
                return next_action
        
        # No ongoing sequence - start a new one
        return self._start_new_sequence(agent_index, state, agent_dir)
    
    def _start_new_sequence(self, agent_index: int, state: Any, agent_dir: int) -> int:
        """
        Start a new action sequence for an agent.
        
        Args:
            agent_index: Index of the agent.
            state: Environment state.
            agent_dir: Agent's current direction (0-3).
        
        Returns:
            First action of the new sequence.
        """
        # Build list of feasible sequence types
        feasible_types = []
        feasible_probs = []
        
        for seq_type in self.ALL_SEQUENCE_TYPES:
            if seq_type not in self.sequence_probs:
                continue
            base_prob = self.sequence_probs[seq_type]
            if base_prob <= 0:
                continue
            
            is_feasible = self._is_sequence_feasible(seq_type, state, agent_index, agent_dir)
            
            if is_feasible:
                feasible_types.append(seq_type)
                feasible_probs.append(base_prob)
        
        if not feasible_types:
            # No feasible sequences - fall back to 'still' sequence with k=1
            self._agent_sequences[agent_index] = {
                'remaining_actions': [],  # 'still' action returned immediately
                'current_type': self.SEQ_STILL
            }
            return self.ACTION_STILL
        
        # Normalize and sample sequence type
        total = sum(feasible_probs)
        normalized_probs = [p / total for p in feasible_probs]
        seq_type = np.random.choice(feasible_types, p=normalized_probs)
        
        # Sample k from geometric distribution (k >= 1) using per-sequence-type expected_k
        geom_p = self._get_geom_p(seq_type)
        k = np.random.geometric(geom_p)
        
        # Build action sequence
        actions = self._build_action_sequence(seq_type, k)
        
        # Store remaining actions (excluding first one which we return)
        first_action = actions[0]
        self._agent_sequences[agent_index] = {
            'remaining_actions': actions[1:],
            'current_type': seq_type
        }
        
        return first_action
    
    def _is_sequence_feasible(self, seq_type: str, state: Any, agent_index: int, agent_dir: int) -> bool:
        """
        Check if a sequence type is feasible given current state.
        
        Uses world_model.can_forward() which correctly accounts for agent-specific
        capabilities (robots can push rocks, humans cannot).
        
        Args:
            seq_type: Sequence type string.
            state: Environment state.
            agent_index: Index of the agent.
            agent_dir: Agent's current direction (0-3).
        
        Returns:
            True if the sequence can be started.
        """
        if seq_type == self.SEQ_STILL:
            return True  # Always feasible
        
        if self.world_model is None:
            return True  # Can't check - assume feasible
        
        # Determine the direction we'll be facing when attempting forward
        if seq_type == self.SEQ_FORWARD:
            forward_dir = agent_dir
        elif seq_type == self.SEQ_LEFT_FORWARD:
            forward_dir = (agent_dir - 1) % 4
        elif seq_type == self.SEQ_RIGHT_FORWARD:
            forward_dir = (agent_dir + 1) % 4
        elif seq_type == self.SEQ_BACK_FORWARD:
            forward_dir = (agent_dir + 2) % 4
        else:
            return False
        
        # Check if forward is possible in that direction
        # We need to temporarily consider the agent facing that direction
        return self._can_forward_in_direction(state, agent_index, forward_dir)
    
    def _can_forward_in_direction(self, state: Any, agent_index: int, direction: int) -> bool:
        """
        Check if agent can move forward if facing a given direction.
        
        This creates a modified state with the agent facing the specified direction
        and checks can_forward(). The can_forward() method in MultiGridEnv correctly
        checks the specific agent's capabilities (can_push_rocks, can_enter_magic_walls).
        
        Args:
            state: Environment state tuple.
            agent_index: Index of the agent.
            direction: Direction to check (0-3).
        
        Returns:
            True if forward movement would be possible facing that direction.
        """
        if self.world_model is None:
            return True
        
        # State format: (step_count, agent_states, mobile_objects, mutable_objects)
        step_count, agent_states, mobile_objects, mutable_objects = state
        
        # Modify agent's direction in the state
        modified_agent_states = list(agent_states)
        agent_state = list(modified_agent_states[agent_index])
        agent_state[2] = direction  # Position 2 is direction
        modified_agent_states[agent_index] = tuple(agent_state)
        
        modified_state = (step_count, tuple(modified_agent_states), mobile_objects, mutable_objects)
        
        return self.world_model.can_forward(modified_state, agent_index)
    
    def _get_first_action_for_sequence(self, seq_type: str) -> int:
        """Get the first action of a sequence type."""
        if seq_type == self.SEQ_STILL:
            return self.ACTION_STILL
        elif seq_type == self.SEQ_FORWARD:
            return self.ACTION_FORWARD
        elif seq_type == self.SEQ_LEFT_FORWARD:
            return self.ACTION_LEFT
        elif seq_type == self.SEQ_RIGHT_FORWARD:
            return self.ACTION_RIGHT
        elif seq_type == self.SEQ_BACK_FORWARD:
            return self.ACTION_LEFT  # First of two left turns
        else:
            return self.ACTION_STILL
    
    def _build_action_sequence(self, seq_type: str, k: int) -> List[int]:
        """
        Build the full action sequence for a given type and k.
        
        Args:
            seq_type: Sequence type string.
            k: Number of main actions (still or forward).
        
        Returns:
            List of action indices.
        """
        if seq_type == self.SEQ_STILL:
            return [self.ACTION_STILL] * k
        elif seq_type == self.SEQ_FORWARD:
            return [self.ACTION_FORWARD] * k
        elif seq_type == self.SEQ_LEFT_FORWARD:
            return [self.ACTION_LEFT] + [self.ACTION_FORWARD] * k
        elif seq_type == self.SEQ_RIGHT_FORWARD:
            return [self.ACTION_RIGHT] + [self.ACTION_FORWARD] * k
        elif seq_type == self.SEQ_BACK_FORWARD:
            return [self.ACTION_LEFT, self.ACTION_LEFT] + [self.ACTION_FORWARD] * k
        else:
            return [self.ACTION_STILL]
    
    def _get_agent_direction(self, state: Any, agent_index: int) -> int:
        """
        Extract agent's direction from state.
        
        Args:
            state: Environment state tuple.
            agent_index: Index of the agent.
        
        Returns:
            Direction (0-3).
        """
        # State format: (step_count, agent_states, mobile_objects, mutable_objects)
        _, agent_states, _, _ = state
        agent_state = agent_states[agent_index]
        # Agent state format: (x, y, direction, ...)
        return agent_state[2]
    
    def __getstate__(self):
        """Exclude world_model from pickling (it may contain thread locks)."""
        state = self.__dict__.copy()
        state['world_model'] = None
        state['_agent_sequences'] = {}  # Clear sequences on pickle
        return state
    
    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)
