"""
Simplified Multigrid-specific Phase 2 Trainer.

This module provides a simplified training class for Phase 2 of the EMPO framework
specialized for multigrid environments. It inherits most logic from the base trainer
and only overrides the environment-specific abstract methods.

This is a simpler alternative to trainer.py that avoids manual tensorization
and batching optimizations, letting the base trainer use encode_and_forward
consistently. This enables drop-in replacement of neural networks with lookup tables.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from gym_multigrid.multigrid import MultiGridEnv

from empo.nn_based.phase2.config import Phase2Config
from empo.nn_based.phase2.trainer import BasePhase2Trainer, Phase2Networks
from empo.nn_based.phase2.lookup import is_lookup_table_network


class SimpleMultiGridPhase2Trainer(BasePhase2Trainer):
    """
    Simplified Phase 2 trainer for multigrid environments.
    
    This trainer inherits all the training logic from BasePhase2Trainer and only
    implements the environment-specific abstract methods:
    - check_goal_achieved: Check if a human's goal is achieved
    - step_environment: Execute actions in the environment
    - reset_environment: Reset environment to initial state
    
    Unlike the full MultiGridPhase2Trainer, this version:
    - Uses encode_and_forward consistently (no manual tensorization)
    - Works as a drop-in for both neural networks and lookup tables
    - Is simpler and easier to maintain (~100 lines vs ~2000 lines)
    
    The tradeoff is less optimization for batched tensor operations, but this
    is acceptable for smaller environments or when using lookup tables.
    
    Args:
        env: MultiGridEnv instance.
        networks: Phase2Networks container.
        config: Phase2Config.
        human_agent_indices: List of human agent indices.
        robot_agent_indices: List of robot agent indices.
        human_policy_prior: HumanPolicyPrior instance with .sample(state, human_idx, goal) method.
        goal_sampler: PossibleGoalSampler instance with .sample(state, human_idx) -> (goal, weight) method.
        device: Torch device.
        verbose: Enable progress output (tqdm progress bar).
        debug: Enable verbose debug output.
        tensorboard_dir: Directory for TensorBoard logs (optional).
        world_model_factory: Optional factory for creating world models. Required for
            async training where the environment cannot be pickled.
    """
    
    def __init__(
        self,
        env: MultiGridEnv,
        networks: Phase2Networks,
        config: Phase2Config,
        human_agent_indices: List[int],
        robot_agent_indices: List[int],
        human_policy_prior: Callable,
        goal_sampler: Callable,
        device: str = 'cpu',
        verbose: bool = False,
        debug: bool = False,
        tensorboard_dir: Optional[str] = None,
        profiler: Optional[Any] = None,
        world_model_factory: Optional[Any] = None,
    ):
        self.env = env
        self.world_model_factory = world_model_factory
        super().__init__(
            networks=networks,
            config=config,
            human_agent_indices=human_agent_indices,
            robot_agent_indices=robot_agent_indices,
            human_policy_prior=human_policy_prior,
            goal_sampler=goal_sampler,
            device=device,
            verbose=verbose,
            debug=debug,
            tensorboard_dir=tensorboard_dir,
            profiler=profiler,
        )
    
    def __getstate__(self):
        """Exclude env from pickling (for async training)."""
        state = super().__getstate__()
        state['env'] = None
        return state
    
    def __setstate__(self, state):
        """Restore state after unpickling."""
        super().__setstate__(state)
    
    def _ensure_world_model(self):
        """Ensure world model is available, creating from factory if needed."""
        if self.env is None:
            if self.world_model_factory is None:
                raise RuntimeError(
                    "No world_model_factory provided. For async training, "
                    "you must pass a world_model_factory to the trainer."
                )
            self._create_env_from_factory()
    
    def _create_env_from_factory(self):
        """Create environment from factory and update dependent components."""
        self.env = self.world_model_factory.create()
        if hasattr(self.goal_sampler, 'set_world_model'):
            self.goal_sampler.set_world_model(self.env)
        if hasattr(self.human_policy_prior, 'set_world_model'):
            self.human_policy_prior.set_world_model(self.env)
    
    def get_cache_stats(self) -> Dict[str, Tuple[int, int]]:
        """
        Get cache hit/miss statistics from all encoders.
        
        For lookup table networks, returns empty dict (no encoder caches).
        
        Returns:
            Dict mapping encoder name to (hits, misses) tuple.
        """
        # Check if using lookup tables
        if is_lookup_table_network(self.networks.q_r):
            return {}
        
        # Neural networks have encoder caches
        return {
            'state': self.networks.q_r.state_encoder.get_cache_stats(),
            'goal': self.networks.v_h_e.goal_encoder.get_cache_stats(),
            'agent': self.networks.v_h_e.agent_encoder.get_cache_stats(),
        }
    
    def reset_cache_stats(self):
        """Reset cache hit/miss counters for all encoders."""
        if is_lookup_table_network(self.networks.q_r):
            return
        self.networks.q_r.state_encoder.reset_cache_stats()
        self.networks.v_h_e.goal_encoder.reset_cache_stats()
        self.networks.v_h_e.agent_encoder.reset_cache_stats()
    
    def clear_caches(self):
        """Clear encoder caches. Call after each training step to prevent memory growth."""
        if is_lookup_table_network(self.networks.q_r):
            return
        self.networks.q_r.state_encoder.clear_cache()
        self.networks.v_h_e.goal_encoder.clear_cache()
        self.networks.v_h_e.agent_encoder.clear_cache()
    
    def check_goal_achieved(self, state: Any, human_idx: int, goal: Any) -> bool:
        """Check if human's goal is achieved."""
        step_count, agent_states, mobile_objects, mutable_objects = state
        agent_state = agent_states[human_idx]
        agent_x, agent_y = int(agent_state[0]), int(agent_state[1])
        
        # Goals should have a target_pos attribute or is_achieved method
        if hasattr(goal, 'target_pos'):
            goal_pos = goal.target_pos
            return agent_x == goal_pos[0] and agent_y == goal_pos[1]
        elif hasattr(goal, 'is_achieved'):
            return goal.is_achieved(state)
        else:
            return False
    
    def step_environment(
        self,
        state: Any,
        robot_action: Tuple[int, ...],
        human_actions: List[int]
    ) -> Any:
        """Execute actions and return next state."""
        # Build action list for all agents
        actions = []
        human_idx = 0
        robot_idx = 0
        
        for agent_idx in range(len(self.env.agents)):
            if agent_idx in self.human_agent_indices:
                actions.append(human_actions[human_idx])
                human_idx += 1
            elif agent_idx in self.robot_agent_indices:
                actions.append(robot_action[robot_idx])
                robot_idx += 1
            else:
                actions.append(0)  # Idle for unknown agents
        
        # Step environment
        self.env.step(actions)
        return self.env.get_state()
    
    def reset_environment(self) -> Any:
        """Reset environment and return initial state."""
        if self.world_model_factory is not None:
            self._create_env_from_factory()
        elif self.env is None:
            raise RuntimeError(
                "No world_model_factory provided and env is None. "
                "For async training, you must pass a world_model_factory."
            )
        
        self.env.reset()
        return self.env.get_state()
