"""
Test parallel DAG computation.

Verifies that parallel and sequential DAG computation produce identical results.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'multigrid'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from gym_multigrid.multigrid import (
    MultiGridEnv, World, Grid, Agent, Wall,
    Block, Rock, MagicWall, UnsteadyGround
)


class AllObjectsTestEnv(MultiGridEnv):
    """
    Test environment with all special objects on unsteady ground.
    
    Layout (6x6):
    W  W  W  W  W  W
    W  H  .  B  R  W   (H=human, B=block, R=rock)
    W  .  .  .  .  W
    W  Rb M  .  .  W   (Rb=robot, M=magic wall facing west)
    W  .  .  .  .  W
    W  W  W  W  W  W
    
    All non-wall, non-object cells have unsteady ground underneath.
    
    - Human agent (index 0): can_enter_magic_walls=True
    - Robot agent (index 1): can_push_rocks=True
    """
    
    def __init__(self, max_steps=3):
        # Create agents with special abilities
        # Agent 0: "human" - can enter magic walls
        # Agent 1: "robot" - can push rocks
        human = Agent(World, 0, can_enter_magic_walls=True, can_push_rocks=False)
        robot = Agent(World, 1, can_enter_magic_walls=False, can_push_rocks=True)
        self.agents = [human, robot]
        self._max_steps_init = max_steps
        
        super().__init__(
            width=6,
            height=6,
            max_steps=max_steps,
            agents=self.agents,
            partial_obs=False,
            objects_set=World
        )
    
    def _get_construction_args(self):
        """No positional args needed."""
        return ()
    
    def _get_construction_kwargs(self):
        """Return only the parameters this class's __init__ accepts."""
        return {'max_steps': self._max_steps_init}
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        
        # Add walls around perimeter
        for i in range(width):
            self.grid.set(i, 0, Wall(World))
            self.grid.set(i, height-1, Wall(World))
        for j in range(height):
            self.grid.set(0, j, Wall(World))
            self.grid.set(width-1, j, Wall(World))
        
        # Place unsteady ground on all interior cells first
        # (they will be overlapped by agents and objects)
        for x in range(1, width-1):
            for y in range(1, height-1):
                self.grid.set(x, y, UnsteadyGround(World, stumble_probability=0.3))
        
        # Place objects (they will be on top of unsteady ground in terrain_grid)
        self.grid.set(3, 1, Block(World))  # Block at (3,1)
        self.grid.set(4, 1, Rock(World))   # Rock at (4,1)
        
        # Magic wall facing west (magic_side=2), so can be entered from the west
        self.grid.set(2, 3, MagicWall(World, magic_side=2, entry_probability=1.0))
        
        # Place human agent at (1,1) facing east
        self.agents[0].pos = np.array([1, 1])
        self.agents[0].dir = 0  # facing east/right
        self.agents[0].init_pos = self.agents[0].pos.copy()
        self.agents[0].init_dir = self.agents[0].dir
        # Save unsteady ground to terrain_grid before placing agent
        unsteady_at_human = self.grid.get(1, 1)
        self.terrain_grid.set(1, 1, unsteady_at_human)
        self.grid.set(1, 1, self.agents[0])
        self.agents[0].on_unsteady_ground = True
        
        # Place robot agent at (1,3) facing east
        self.agents[1].pos = np.array([1, 3])
        self.agents[1].dir = 0  # facing east/right
        self.agents[1].init_pos = self.agents[1].pos.copy()
        self.agents[1].init_dir = self.agents[1].dir
        # Save unsteady ground to terrain_grid before placing agent
        unsteady_at_robot = self.grid.get(1, 3)
        self.terrain_grid.set(1, 3, unsteady_at_robot)
        self.grid.set(1, 3, self.agents[1])
        self.agents[1].on_unsteady_ground = True


def test_parallel_dag_correctness():
    """Test that parallel and sequential DAG computation give identical results."""
    
    # Create a moderately complex environment
    env = MultiGridEnv(
        map='''
We We We We We We We
We .. .. .. Un Un We
We .. .. .. Un Un We
We .. .. Ay Un Un We
We .. .. .. Un Un We
We .. .. .. Un Un We
We We We We We We We
''',
        objects_set=World,
        orientations=['n'],
        max_steps=4,
        partial_obs=False
    )
    
    print("\n" + "="*70)
    print("Testing Parallel DAG Computation Correctness")
    print("="*70)
    
    # Compute DAG sequentially
    print("\nComputing DAG sequentially...")
    states_seq, state_to_idx_seq, successors_seq = env.get_dag(return_probabilities=False)
    print(f"  States: {len(states_seq)}")
    
    # Compute DAG in parallel
    print("\nComputing DAG in parallel...")
    states_par, state_to_idx_par, successors_par = env.get_dag_parallel(return_probabilities=False, num_workers=4)
    print(f"  States: {len(states_par)}")
    
    # Verify same number of states
    assert len(states_seq) == len(states_par), \
        f"Different number of states: seq={len(states_seq)}, par={len(states_par)}"
    
    # Verify same states (as sets, since order might differ)
    states_set_seq = set(states_seq)
    states_set_par = set(states_par)
    assert states_set_seq == states_set_par, "Different state sets"
    
    # Verify same successors (need to map through indices)
    print("\nVerifying successor structure...")
    for state in states_seq:
        idx_seq = state_to_idx_seq[state]
        idx_par = state_to_idx_par[state]
        
        # Get successor states (not indices) for comparison
        succ_states_seq = set(states_seq[s_idx] for s_idx in successors_seq[idx_seq])
        succ_states_par = set(states_par[s_idx] for s_idx in successors_par[idx_par])
        
        assert succ_states_seq == succ_states_par, \
            f"Different successors for state {state}"
    
    # Verify root state is the same
    assert states_seq[0] == states_par[0], "Different root states"
    
    # Verify topological ordering (all successors have higher indices)
    print("\nVerifying topological ordering...")
    for idx, succ_list in enumerate(successors_par):
        for succ_idx in succ_list:
            assert idx < succ_idx, f"Topological order violated: {idx} -> {succ_idx}"
    
    print("\n✓ All checks passed! Parallel and sequential DAG computation are identical.")
    print("="*70)


def test_parallel_dag_with_probabilities():
    """Test parallel DAG computation with probability tracking."""
    
    env = MultiGridEnv(
        map='''
We We We We We
We .. .. Ar We
We .. .. .. We
We We We We We
''',
        objects_set=World,
        orientations=['e'],
        max_steps=3,
        partial_obs=False
    )
    
    print("\n" + "="*70)
    print("Testing Parallel DAG with Probabilities")
    print("="*70)
    
    # Sequential with probabilities
    print("\nComputing DAG sequentially (with probabilities)...")
    states_seq, state_to_idx_seq, successors_seq, trans_seq = env.get_dag(return_probabilities=True)
    print(f"  States: {len(states_seq)}")
    
    # Parallel with probabilities
    print("\nComputing DAG in parallel (with probabilities)...")
    states_par, state_to_idx_par, successors_par, trans_par = env.get_dag_parallel(
        return_probabilities=True, num_workers=2
    )
    print(f"  States: {len(states_par)}")
    
    # Verify same states
    assert len(states_seq) == len(states_par)
    assert set(states_seq) == set(states_par)
    
    print("\nVerifying transition probabilities...")
    # For each state, verify transitions match
    for state in states_seq:
        idx_seq = state_to_idx_seq[state]
        idx_par = state_to_idx_par[state]
        
        # Get transitions as sets of (action, successor_state, prob) tuples
        trans_set_seq = set()
        for action, probs, succ_indices in trans_seq[idx_seq]:
            for prob, succ_idx in zip(probs, succ_indices):
                succ_state = states_seq[succ_idx]
                trans_set_seq.add((action, succ_state, prob))
        
        trans_set_par = set()
        for action, probs, succ_indices in trans_par[idx_par]:
            for prob, succ_idx in zip(probs, succ_indices):
                succ_state = states_par[succ_idx]
                trans_set_par.add((action, succ_state, prob))
        
        assert trans_set_seq == trans_set_par, \
            f"Different transitions for state {state}"
    
    print("\n✓ Probability tracking verified! Sequential and parallel match.")
    print("="*70)


def test_parallel_dag_all_objects():
    """
    Test parallel DAG with all special objects in one environment.
    
    Environment contains:
    - One "human" agent (can_enter_magic_walls=True)
    - One "robot" agent (can_push_rocks=True)
    - Block (pushable by any agent)
    - Rock (pushable only by robot)
    - Magic wall (can be entered only by human)
    - All floor tiles are unsteady ground (causes stumbling)
    
    Verifies that parallel and sequential DAG computation produce exactly
    the same results in terms of states, state ordering, and transitions.
    """
    print("\n" + "="*70)
    print("Testing Parallel DAG with All Special Objects")
    print("="*70)
    
    env = AllObjectsTestEnv()
    
    print("\nEnvironment configuration:")
    print("  - Human agent (can_enter_magic_walls=True) at (1,1)")
    print("  - Robot agent (can_push_rocks=True) at (1,3)")
    print("  - Block at (3,1)")
    print("  - Rock at (4,1)")
    print("  - Magic wall at (2,3)")
    print("  - All floor is unsteady ground")
    print("  - Max steps: 3")
    
    # Compute DAG sequentially with probabilities
    print("\nComputing DAG sequentially...")
    states_seq, state_to_idx_seq, successors_seq, trans_seq = env.get_dag(return_probabilities=True)
    print(f"  States: {len(states_seq)}")
    
    # Compute DAG in parallel with probabilities
    print("\nComputing DAG in parallel...")
    states_par, state_to_idx_par, successors_par, trans_par = env.get_dag_parallel(
        return_probabilities=True, num_workers=4
    )
    print(f"  States: {len(states_par)}")
    
    # ============================================
    # Verify DAG states match
    # ============================================
    print("\nVerifying DAG states...")
    
    states_set_seq = set(states_seq)
    states_set_par = set(states_par)
    
    extra_in_par = states_set_par - states_set_seq
    extra_in_seq = states_set_seq - states_set_par
    
    if extra_in_par:
        print(f"  Extra states in parallel ({len(extra_in_par)}):")
        for s in sorted(extra_in_par)[:3]:
            print(f"    {s}")
    
    if extra_in_seq:
        print(f"  Extra states in sequential ({len(extra_in_seq)}):")
        for s in sorted(extra_in_seq)[:3]:
            print(f"    {s}")
    
    assert len(states_seq) == len(states_par), \
        f"Different number of states: seq={len(states_seq)}, par={len(states_par)}"
    
    assert states_set_seq == states_set_par, \
        f"Different state sets! Extra in seq: {len(extra_in_seq)}, Extra in par: {len(extra_in_par)}"
    print("  ✓ Same states in both DAGs")
    
    # ============================================
    # Verify state ordering (topological + matching)
    # ============================================
    print("\nVerifying state ordering...")
    
    # Both should have the same root state
    assert states_seq[0] == states_par[0], \
        f"Different root states: seq={states_seq[0]}, par={states_par[0]}"
    print("  ✓ Same root state")
    
    # Both should be topologically ordered
    for label, successors, states in [("sequential", successors_seq, states_seq), 
                                       ("parallel", successors_par, states_par)]:
        for idx, succ_list in enumerate(successors):
            for succ_idx in succ_list:
                assert idx < succ_idx, \
                    f"{label}: Topological order violated: {idx} -> {succ_idx}"
    print("  ✓ Both have valid topological ordering")
    
    # States should be in exactly the same order
    for i in range(len(states_seq)):
        assert states_seq[i] == states_par[i], \
            f"States differ at index {i}: seq={states_seq[i]}, par={states_par[i]}"
    print("  ✓ States are in identical order")
    
    # ============================================
    # Verify successor structure matches
    # ============================================
    print("\nVerifying successor structure...")
    for i in range(len(states_seq)):
        succ_set_seq = set(successors_seq[i])
        succ_set_par = set(successors_par[i])
        assert succ_set_seq == succ_set_par, \
            f"Different successors at index {i}"
    print("  ✓ Identical successor structure")
    
    # ============================================
    # Verify transitions match exactly
    # ============================================
    print("\nVerifying transitions...")
    num_transitions = 0
    for i in range(len(states_seq)):
        # Convert transitions to comparable format
        def trans_to_set(trans_list, states):
            """Convert transitions to a set of (action, succ_state, prob) tuples."""
            result = set()
            for action, probs, succ_indices in trans_list:
                for prob, succ_idx in zip(probs, succ_indices):
                    succ_state = states[succ_idx]
                    result.add((action, succ_state, prob))
            return result
        
        trans_set_seq = trans_to_set(trans_seq[i], states_seq)
        trans_set_par = trans_to_set(trans_par[i], states_par)
        
        assert trans_set_seq == trans_set_par, \
            f"Different transitions at state index {i}"
        num_transitions += len(trans_set_seq)
    
    print(f"  ✓ All {num_transitions} transition entries match exactly")
    
    # ============================================
    # Final summary
    # ============================================
    print("\n" + "="*70)
    print("✓ All checks passed!")
    print(f"  - {len(states_seq)} states verified")
    print(f"  - {num_transitions} transition entries verified")
    print("  - State ordering identical")
    print("  - Topological order valid")
    print("  - All special objects handled correctly")
    print("="*70)


if __name__ == '__main__':
    test_parallel_dag_correctness()
    test_parallel_dag_with_probabilities()
    test_parallel_dag_all_objects()
    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)
