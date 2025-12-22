# Transport Environment Integration Plan

## Objective

Integrate the `ai_transport` PettingZoo environment into the EMPO framework to study how AI-controlled vehicles can maximize passenger empowerment by flexibly announcing destination regions (clusters) rather than fixed routes.

The following proposal is meant as a rough design. You can deviate from it in the details when you think it is more sensible to change some details.

## Architecture Overview

```
empo/
├── vendor/
│   ├── multigrid/          # existing
│   └── ai_transport/       # NEW: vendored transport env
└── nn_based/
    ├── minigrid/           # existing
    └── transport/          # NEW: parallel structure
        ├── __init__.py
        ├── encoder.py      # GNN encoder
        ├── agent.py        # DQN/PPO agent with action masking
        ├── training.py     # training loop
        └── config.py       # hyperparameters
```

---

## Phase 1: Vendor ai_transport

### 1.1 Add as Git Subtree

```bash
cd /path/to/empo
git subtree add --prefix=vendor/ai_transport \
    https://github.com/pik-gane/ai_transport.git main --squash
```

### 1.2 Update VENDOR.md

Add section documenting:
- Source repository: `https://github.com/pik-gane/ai_transport`
- Purpose: Multi-agent transport environment for empowerment research
- Modifications: Cluster-based destination system (see Phase 2)
- Update command: `git subtree pull --prefix=vendor/ai_transport ...`

### 1.3 Update PYTHONPATH

Ensure `vendor/ai_transport` is importable:
- Already handled by existing `PYTHONPATH` setup in Docker/compose
- Verify with: `python -c "from ai_transport import parallel_env"`

---

## Phase 2: Modify ai_transport for Cluster-Based Destinations

### 2.1 Add Network Clustering

Find k disjoint sets S_i of network nodes so that, roughly, for every pair of nodes v1, v2, there is at least one S_i so that it is plausible that when going from v1 to S_i on a plausible route, one will come close by node v2. The simplest initial approach for this is probably to simply use k means clustering on the full set of nodes, with a sufficiently large k, such as k=20.

**File:** `vendor/ai_transport/ai_transport/envs/clustering.py`

```python
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List

def cluster_network(G: nx.DiGraph, k: int = 20, method: str = 'kmeans') -> Dict:
    """
    Cluster network nodes into k regions.
    
    Returns:
        {
            'node_to_cluster': {node_id: cluster_id},
            'cluster_to_nodes': {cluster_id: [node_ids]},
            'centroids': {cluster_id: centroid_node_id}
        }
    """
    if method == 'kmeans':
        # Extract 2D coordinates
        coords = np.array([G.nodes[n].get('pos', (0, 0)) for n in G.nodes()])
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(coords)
        
        node_to_cluster = {node: int(labels[i]) for i, node in enumerate(G.nodes())}
        
        # Find centroid nodes (nearest node to cluster center)
        centroids = {}
        for c in range(k):
            cluster_nodes = [n for n, cl in node_to_cluster.items() if cl == c]
            if cluster_nodes:
                center = kmeans.cluster_centers_[c]
                distances = [np.linalg.norm(coords[i] - center) 
                           for i, n in enumerate(G.nodes()) if n in cluster_nodes]
                centroids[c] = cluster_nodes[np.argmin(distances)]
        
    elif method == 'spectral':
        # TODO: implement spectral clustering on graph structure
        raise NotImplementedError("Spectral clustering not yet implemented")
    
    cluster_to_nodes = {}
    for node, cluster in node_to_cluster.items():
        cluster_to_nodes.setdefault(cluster, []).append(node)
    
    return {
        'node_to_cluster': node_to_cluster,
        'cluster_to_nodes': cluster_to_nodes,
        'centroids': centroids
    }
```

### 2.2 Modify TransportEnv

**File:** `vendor/ai_transport/ai_transport/envs/transport_env.py`

Add to `__init__`:
```python
def __init__(self, ..., num_clusters: int = 20, clustering_method: str = 'kmeans'):
    # ... existing init ...
    
    # Add clustering
    from .clustering import cluster_network
    self.num_clusters = num_clusters
    self.cluster_info = cluster_network(self.network, k=num_clusters, method=clustering_method)
    self.node_to_cluster = self.cluster_info['node_to_cluster']
    self.cluster_centroids = self.cluster_info['centroids']
```

Modify vehicle destination handling:
- Replace node-based `vehicle_destination` with `vehicle_destination_cluster`
- Update observations to include cluster information

---

## Phase 3: Define Fixed Action Space with Masking

### 3.1 Action Space Definition

**File:** `src/empo/transport_action_space.py`

```python
from enum import IntEnum
from typing import Dict, List
import numpy as np

class TransportAction(IntEnum):
    STAY = 0
    EXIT = 1
    BOARD_0 = 2   # board first vehicle
    BOARD_1 = 3
    # ... 
    BOARD_9 = 11  # board 10th vehicle
    DEST_CLUSTER_0 = 12  # announce destination cluster 0
    # ...
    DEST_CLUSTER_19 = 31  # announce destination cluster 19
    DEPART_EDGE_0 = 32  # depart on first outgoing edge
    # ...
    DEPART_EDGE_9 = 41  # depart on 10th outgoing edge

MAX_VEHICLES_AT_NODE = 10
MAX_OUTGOING_EDGES = 10
NUM_CLUSTERS = 20
TOTAL_ACTIONS = 42

def compute_action_mask(agent_id: str, observation: Dict, env) -> np.ndarray:
    """
    Compute valid action mask for given agent and observation.
    
    Returns:
        Boolean mask of shape (TOTAL_ACTIONS,), True = valid action
    """
    mask = np.zeros(TOTAL_ACTIONS, dtype=bool)
    mask[TransportAction.STAY] = True  # always valid
    
    agent_type = 'vehicle' if agent_id.startswith('vehicle_') else 'human'
    step_type = observation['step_type']
    position = observation.get('my_position') or env.agent_positions[agent_id]
    
    if step_type == 'routing' and agent_type == 'vehicle':
        # Can announce destination clusters
        mask[TransportAction.DEST_CLUSTER_0:TransportAction.DEST_CLUSTER_19+1] = True
    
    elif step_type == 'unboarding':
        if agent_type == 'human' and observation.get('aboard') is not None:
            mask[TransportAction.EXIT] = True
    
    elif step_type == 'boarding':
        if agent_type == 'human':
            vehicles_here = observation.get('agents_here', {})
            vehicles = [a for a in vehicles_here if a.startswith('vehicle_')]
            num_vehicles = min(len(vehicles), MAX_VEHICLES_AT_NODE)
            for i in range(num_vehicles):
                mask[TransportAction.BOARD_0 + i] = True
    
    elif step_type == 'departing':
        if isinstance(position, tuple):  # on edge
            pass  # only STAY valid
        else:  # at node
            outgoing_edges = list(env.network.out_edges(position))
            num_edges = min(len(outgoing_edges), MAX_OUTGOING_EDGES)
            for i in range(num_edges):
                mask[TransportAction.DEPART_EDGE_0 + i] = True
    
    return mask

def decode_action(action_idx: int, agent_id: str, observation: Dict, env) -> Dict:
    """
    Convert action index to environment-specific action.
    
    Returns:
        Action dict compatible with ai_transport environment
    """
    if action_idx == TransportAction.STAY:
        return {'type': 'pass'}
    
    elif action_idx == TransportAction.EXIT:
        return {'type': 'unboard'}
    
    elif TransportAction.BOARD_0 <= action_idx <= TransportAction.BOARD_9:
        vehicle_idx = action_idx - TransportAction.BOARD_0
        vehicles_here = [a for a in observation.get('agents_here', {}) if a.startswith('vehicle_')]
        return {'type': 'board', 'vehicle_id': vehicles_here[vehicle_idx]}
    
    elif TransportAction.DEST_CLUSTER_0 <= action_idx <= TransportAction.DEST_CLUSTER_19:
        cluster_id = action_idx - TransportAction.DEST_CLUSTER_0
        return {'type': 'set_destination', 'cluster': cluster_id}
    
    elif TransportAction.DEPART_EDGE_0 <= action_idx <= TransportAction.DEPART_EDGE_9:
        edge_idx = action_idx - TransportAction.DEPART_EDGE_0
        position = observation.get('my_position') or env.agent_positions[agent_id]
        outgoing_edges = list(env.network.out_edges(position))
        return {'type': 'depart', 'edge': outgoing_edges[edge_idx]}
    
    raise ValueError(f"Invalid action index: {action_idx}")
```

---

## Phase 4: Create nn_based/transport

### 4.1 Directory Structure

```
nn_based/transport/
├── __init__.py
├── config.py          # Hyperparameters
├── encoder.py         # GNN encoder
├── agent.py           # DQN agent with masking
├── training.py        # Training loop
└── utils.py           # Helper functions
```

### 4.2 GNN Encoder

example code:

**File:** `nn_based/transport/encoder.py`

```python
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data

class TransportGNNEncoder(nn.Module):
    """
    Graph Neural Network encoder for transport environment.
    
    Processes:
    - Node features: [#humans, #vehicles, cluster_id, is_current_position]
    - Edge features: [length, speed, capacity, #agents_on_edge]
    - Global features: [real_time, step_type_onehot]
    """
    
    def __init__(self, 
                 node_feature_dim: int = 16,
                 edge_feature_dim: int = 8,
                 hidden_dim: int = 128,
                 num_gnn_layers: int = 3,
                 global_feature_dim: int = 8):
        super().__init__()
        
        # Feature embeddings
        self.node_embed = nn.Linear(node_feature_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_feature_dim, hidden_dim)
        
        # GNN layers (use GCN, GAT, or GIN)
        self.gnn_layers = nn.ModuleList([
            pyg_nn.GATConv(hidden_dim, hidden_dim, edge_dim=hidden_dim)
            for _ in range(num_gnn_layers)
        ])
        
        # Global context integration
        self.global_mlp = nn.Sequential(
            nn.Linear(global_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Final agent embedding
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # concat(node_embed, global_embed)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, observation_dict: dict) -> torch.Tensor:
        """
        Args:
            observation_dict: Contains 'graph' (PyG Data), 'agent_node_idx', 'global_features'
        
        Returns:
            Agent embedding of shape (hidden_dim,)
        """
        graph = observation_dict['graph']
        agent_idx = observation_dict['agent_node_idx']
        global_feat = observation_dict['global_features']
        
        # Embed node and edge features
        x = self.node_embed(graph.x)
        edge_attr = self.edge_embed(graph.edge_attr)
        
        # Apply GNN layers
        for gnn in self.gnn_layers:
            x = gnn(x, graph.edge_index, edge_attr=edge_attr)
            x = torch.relu(x)
        
        # Extract agent's node embedding
        agent_embed = x[agent_idx]
        
        # Global context
        global_embed = self.global_mlp(global_feat)
        
        # Combine
        combined = torch.cat([agent_embed, global_embed], dim=-1)
        output = self.output_mlp(combined)
        
        return output

def observation_to_graph(observation: dict, env) -> dict:
    """
    Convert ai_transport observation to PyTorch Geometric Data.
    
    Returns:
        Dict with keys: 'graph', 'agent_node_idx', 'global_features'
    """
    # TODO: Implement conversion
    # - Extract node features from observation['network_nodes'] or local info
    # - Extract edge features from observation['network_edges']
    # - Build edge_index from network structure
    # - Identify agent's current node index
    # - Extract global features (time, step_type)
    pass
```


## Integration with EMPO Framework

### 5.1 Update requirements.txt

Add:
```
torch-geometric>=2.4.0
networkx>=3.0
scikit-learn>=1.3.0
```

### 5.2 Update Documentation


---

## Notes

- **Observation format:** Choose between 'global' (full state) or 'local_plus_counts' (partial observability) based on empowerment computation needs
- **Reward design:** Currently zero rewards in env; implement empowerment-based rewards externally
- **Scalability:** Start with small networks (N=20-30 nodes) for testing; scale up gradually
- **GNN library:** Using PyTorch Geometric; alternative: DGL if preferred
- **Action masking:** Critical for stability; verify masks are correct in all step types
