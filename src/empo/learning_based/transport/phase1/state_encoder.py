"""
GNN-based state encoder for transport environment.

This module provides a Graph Neural Network encoder for processing transport
environment state. Unlike the CNN-based encoder used for multigrid, this
encoder uses message passing on the network graph structure.

The encoder can use different GNN layers (GCN, GAT, GIN) and supports both
node-based and cluster-based routing modes.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional

from empo.learning_based.phase1.state_encoder import BaseStateEncoder
from .constants import (
    NODE_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    AGENT_FEATURE_DIM,
    DEFAULT_GNN_HIDDEN_DIM,
    DEFAULT_GNN_NUM_LAYERS,
    DEFAULT_OUTPUT_FEATURE_DIM,
    MAX_CLUSTERS,
)


class TransportStateEncoder(BaseStateEncoder):
    """
    GNN-based encoder for transport environment states.
    
    This encoder processes the transport network using Graph Neural Networks,
    capturing both local node features and global network structure.
    
    Architecture:
    1. Node embedding MLP: node features -> hidden dim
    2. Edge embedding MLP: edge features -> hidden dim
    3. GNN message passing layers (configurable type)
    4. Global pooling (mean over all nodes)
    5. Global context MLP: global + agent features -> hidden dim
    6. Output MLP: concatenated features -> output dim
    
    Args:
        num_clusters: Number of clusters (0 if not using clusters)
        max_nodes: Maximum number of nodes in network (for padding)
        feature_dim: Output feature dimension
        hidden_dim: Hidden dimension for GNN and MLPs
        num_gnn_layers: Number of GNN message passing layers
        gnn_type: Type of GNN layer ('gcn', 'gat', 'gin')
        use_edge_features: Whether to use edge features in GNN
    
    Example:
        >>> encoder = TransportStateEncoder(num_clusters=10, max_nodes=100)
        >>> graph_data = observation_to_graph_data(env, query_agent_idx=0)
        >>> encoded = encoder(
        ...     graph_data['node_features'],
        ...     graph_data['edge_index'],
        ...     graph_data['edge_features'],
        ...     graph_data['global_features'],
        ...     graph_data['agent_features']
        ... )
        >>> print(encoded.shape)
        torch.Size([1, 128])
    """
    
    def __init__(
        self,
        num_clusters: int = 0,
        max_nodes: int = 100,
        feature_dim: int = DEFAULT_OUTPUT_FEATURE_DIM,
        hidden_dim: int = DEFAULT_GNN_HIDDEN_DIM,
        num_gnn_layers: int = DEFAULT_GNN_NUM_LAYERS,
        gnn_type: str = 'gcn',
        use_edge_features: bool = True
    ):
        super().__init__(feature_dim)
        self.num_clusters = num_clusters
        self.max_nodes = max_nodes
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.gnn_type = gnn_type
        self.use_edge_features = use_edge_features
        
        # Node feature embedding
        self.node_embed = nn.Sequential(
            nn.Linear(NODE_FEATURE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Edge feature embedding
        self.edge_embed = nn.Sequential(
            nn.Linear(EDGE_FEATURE_DIM, hidden_dim),
            nn.ReLU(),
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_gnn_layers):
            if gnn_type == 'gcn':
                # Simple GCN-like layer (without torch_geometric dependency)
                self.gnn_layers.append(
                    GNNLayer(hidden_dim, hidden_dim, use_edge_features=use_edge_features)
                )
            elif gnn_type == 'gat':
                self.gnn_layers.append(
                    GATLayer(hidden_dim, hidden_dim, num_heads=4, use_edge_features=use_edge_features)
                )
            elif gnn_type == 'gin':
                self.gnn_layers.append(
                    GINLayer(hidden_dim, hidden_dim)
                )
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        # Global context embedding
        self.global_embed = nn.Sequential(
            nn.Linear(GLOBAL_FEATURE_DIM + AGENT_FEATURE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Output projection
        # Combines: pooled node features + query node features + global context
        combined_dim = hidden_dim * 3
        self.output_fc = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        global_features: torch.Tensor,
        agent_features: torch.Tensor,
        query_node_idx: Optional[int] = None,
        batch_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode transport state into feature vector.
        
        Args:
            node_features: (num_nodes, NODE_FEATURE_DIM) or (batch, max_nodes, NODE_FEATURE_DIM)
            edge_index: (2, num_edges) source/target node indices
            edge_features: (num_edges, EDGE_FEATURE_DIM)
            global_features: (GLOBAL_FEATURE_DIM,) or (batch, GLOBAL_FEATURE_DIM)
            agent_features: (AGENT_FEATURE_DIM,) or (batch, AGENT_FEATURE_DIM)
            query_node_idx: Index of query agent's node (if at node)
            batch_idx: Optional batch indices for batched processing
        
        Returns:
            Encoded state tensor (batch, feature_dim)
        """
        # Handle single sample vs batch
        if node_features.dim() == 2:
            # Single sample - add batch dimension
            node_features = node_features.unsqueeze(0)
            if global_features.dim() == 1:
                global_features = global_features.unsqueeze(0)
            if agent_features.dim() == 1:
                agent_features = agent_features.unsqueeze(0)
        
        batch_size = node_features.shape[0]
        device = node_features.device
        
        # Embed node features
        x = self.node_embed(node_features)  # (batch, num_nodes, hidden_dim)
        
        # Embed edge features
        if self.use_edge_features:
            edge_attr = self.edge_embed(edge_features)  # (num_edges, hidden_dim)
        else:
            edge_attr = None
        
        # Apply GNN layers (simplified - process each sample in batch)
        # For efficiency with batched graphs, you'd use scatter operations
        for gnn in self.gnn_layers:
            x = gnn(x, edge_index, edge_attr)
            x = torch.relu(x)
        
        # Global pooling (mean over nodes)
        pooled = x.mean(dim=1)  # (batch, hidden_dim)
        
        # Query node embedding (use pooled if not at node)
        if query_node_idx is not None and query_node_idx >= 0:
            query_embed = x[:, query_node_idx, :]  # (batch, hidden_dim)
        else:
            query_embed = pooled  # Use pooled if agent not at specific node
        
        # Global context
        global_context = torch.cat([global_features, agent_features], dim=-1)
        global_embed = self.global_embed(global_context)  # (batch, hidden_dim)
        
        # Combine all features
        combined = torch.cat([pooled, query_embed, global_embed], dim=-1)
        output = self.output_fc(combined)
        
        return output
    
    def tensorize_state(
        self,
        state: Any,
        world_model: Any,
        device: str = 'cpu'
    ) -> Dict[str, torch.Tensor]:
        """
        Convert raw state to input tensors (preprocessing, NOT neural network encoding).
        
        This method extracts features from the environment and prepares them
        as tensors for the forward() pass.
        
        Args:
            state: Ignored (transport env doesn't have separate state)
            world_model: TransportEnvWrapper instance
            device: Torch device
        
        Returns:
            Dictionary with tensors ready for forward()
        """
        from .feature_extraction import observation_to_graph_data
        
        return observation_to_graph_data(world_model, device)
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration for save/load."""
        return {
            'num_clusters': self.num_clusters,
            'max_nodes': self.max_nodes,
            'feature_dim': self.feature_dim,
            'hidden_dim': self.hidden_dim,
            'num_gnn_layers': self.num_gnn_layers,
            'gnn_type': self.gnn_type,
            'use_edge_features': self.use_edge_features,
        }


class GNNLayer(nn.Module):
    """
    Simple GNN layer with message passing.
    
    This is a basic GCN-like layer that doesn't require torch_geometric.
    For production use, consider using PyTorch Geometric layers.
    """
    
    def __init__(self, in_dim: int, out_dim: int, use_edge_features: bool = True):
        super().__init__()
        self.use_edge_features = use_edge_features
        
        if use_edge_features:
            self.message_fc = nn.Linear(in_dim * 2 + in_dim, out_dim)  # node + neighbor + edge
        else:
            self.message_fc = nn.Linear(in_dim * 2, out_dim)
        
        self.update_fc = nn.Linear(out_dim, out_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, num_nodes, in_dim)
            edge_index: (2, num_edges)
            edge_attr: (num_edges, edge_dim) if use_edge_features
        
        Returns:
            Updated node features (batch, num_nodes, out_dim)
        """
        batch_size, num_nodes, in_dim = x.shape
        device = x.device
        
        # Initialize output
        out = torch.zeros(batch_size, num_nodes, self.update_fc.out_features, device=device)
        
        if edge_index.shape[1] == 0:
            return self.update_fc(torch.zeros(batch_size, num_nodes, self.message_fc.out_features, device=device))
        
        src, dst = edge_index[0], edge_index[1]
        
        # Aggregate messages for each node
        for b in range(batch_size):
            # Message: concatenate source node, destination node, and edge features
            messages = torch.zeros(num_nodes, self.message_fc.out_features, device=device)
            counts = torch.zeros(num_nodes, device=device)
            
            for i in range(edge_index.shape[1]):
                s, d = src[i].item(), dst[i].item()
                
                if self.use_edge_features and edge_attr is not None:
                    msg_input = torch.cat([x[b, s], x[b, d], edge_attr[i]], dim=-1)
                else:
                    msg_input = torch.cat([x[b, s], x[b, d]], dim=-1)
                
                msg = self.message_fc(msg_input)
                messages[d] = messages[d] + msg
                counts[d] = counts[d] + 1
            
            # Average messages
            counts = torch.clamp(counts, min=1)
            messages = messages / counts.unsqueeze(-1)
            
            out[b] = self.update_fc(messages)
        
        return out


class GATLayer(nn.Module):
    """
    Graph Attention Network layer.
    
    Simple implementation without torch_geometric.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        use_edge_features: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.use_edge_features = use_edge_features
        
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        
        if use_edge_features:
            self.edge_proj = nn.Linear(in_dim, out_dim)
        
        self.out_proj = nn.Linear(out_dim, out_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Simple attention-based message passing."""
        batch_size, num_nodes, in_dim = x.shape
        device = x.device
        
        # Project queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Simple attention aggregation
        out = torch.zeros_like(q)
        
        if edge_index.shape[1] == 0:
            return self.out_proj(out)
        
        src, dst = edge_index[0], edge_index[1]
        
        for b in range(batch_size):
            for i in range(edge_index.shape[1]):
                s, d = src[i].item(), dst[i].item()
                
                # Compute attention score
                attn = (q[b, d] * k[b, s]).sum() / (self.head_dim ** 0.5)
                attn = torch.sigmoid(attn)  # Simplified attention
                
                # Add to output
                out[b, d] = out[b, d] + attn * v[b, s]
        
        return self.out_proj(out)


class GINLayer(nn.Module):
    """
    Graph Isomorphism Network layer.
    
    Simple implementation without torch_geometric.
    """
    
    def __init__(self, in_dim: int, out_dim: int, eps: float = 0.0):
        super().__init__()
        self.eps = eps
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """GIN message passing."""
        batch_size, num_nodes, in_dim = x.shape
        device = x.device
        
        # Aggregate neighbor features
        agg = torch.zeros_like(x)
        
        if edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            
            for b in range(batch_size):
                for i in range(edge_index.shape[1]):
                    s, d = src[i].item(), dst[i].item()
                    agg[b, d] = agg[b, d] + x[b, s]
        
        # GIN update: MLP((1 + eps) * x + aggregated)
        out = self.mlp((1 + self.eps) * x + agg)
        
        return out
