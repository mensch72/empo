# Planning Document: Better Network Initialization Strategies

This document explores strategies for initializing Phase 2 neural networks to improve training stability and convergence.

## Current State

Currently, Phase 2 networks use **PyTorch default initialization**:
- `nn.Linear` weights: Kaiming uniform (fan_in mode)
- `nn.Linear` biases: Uniform(-1/√fan_in, 1/√fan_in)
- `nn.Conv2d` weights: Kaiming uniform
- `nn.Embedding` weights: Normal(0, 1)

No explicit initialization code is applied. Networks start with random weights and rely on:
1. **Staged warm-up** to break circular dependencies
2. **Gradient clipping** to prevent exploding gradients
3. **Learning rate schedules** to stabilize late training

## Motivation for Better Initialization

### Problem 1: Output Range Mismatch

Each network has known output constraints:
- **V_h^e**: Output in [0, 1] (goal achievement probability)
- **X_h**: Output in (0, 1] (aggregate power, bounded by clamp)
- **U_r**: Output in (-∞, 0) (negative power metric)
- **Q_r**: Output in (-∞, 0) (negative action-values)
- **V_r**: Output in (-∞, 0) (negative state-values)

With random initialization:
- Outputs may start far from expected ranges
- `ensure_negative()` via `-softplus()` adds nonlinearity that interacts poorly with random inputs
- Early training steps waste time moving outputs to feasible regions

### Problem 2: Gradient Flow Issues

The Phase 2 networks have deep dependency chains:
```
Q_r → π_r → V_h^e → X_h → U_r → V_r → Q_r (circular)
```

Poor initialization can cause:
- Vanishing gradients in deep MLP heads
- Exploding gradients when computing power metrics (e.g., X_h^{-ξ})
- Unstable early targets for TD learning

### Problem 3: Encoder-Head Interaction

Networks use shared state encoders with separate heads:
- Shared encoder trained with V_h^e loss
- Q_r has its own encoder (trained with Q_r loss)
- Mismatch in encoder vs. head initialization can cause imbalanced gradient magnitudes

## Proposed Initialization Strategies

### Strategy 1: Output-Aware Last Layer Initialization

Initialize the final linear layer to produce outputs in the expected range from the start.

#### For V_h^e (output ∈ [0, 1])

The final layer goes through sigmoid. Initialize to output ~0.5 (neutral):

```python
def init_v_h_e_output(layer: nn.Linear):
    """Initialize final layer for sigmoid output centered at 0.5."""
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)  # sigmoid(0) = 0.5
```

Or for slightly pessimistic initialization (V_h^e closer to 0):

```python
def init_v_h_e_output_pessimistic(layer: nn.Linear):
    """Initialize final layer for low V_h^e (conservative)."""
    nn.init.zeros_(layer.weight)
    nn.init.constant_(layer.bias, -2.0)  # sigmoid(-2) ≈ 0.12
```

#### For X_h (output ∈ (0, 1])

Similar to V_h^e since X_h uses sigmoid:

```python
def init_x_h_output(layer: nn.Linear):
    """Initialize final layer for X_h centered at 0.5."""
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
```

#### For Q_r (output < 0)

Q_r uses `-softplus(raw)` to ensure negativity. Initialize for moderate negative values:

```python
def init_q_r_output(layer: nn.Linear):
    """Initialize final layer for Q_r ≈ -1."""
    nn.init.zeros_(layer.weight)
    # -softplus(0) = -log(2) ≈ -0.69
    # -softplus(1) ≈ -1.31
    nn.init.constant_(layer.bias, 1.0)  # Q_r starts around -1.3
```

#### For U_r (output < 0)

U_r also uses negative transformation:

```python
def init_u_r_output(layer: nn.Linear):
    """Initialize final layer for U_r ≈ -1."""
    nn.init.zeros_(layer.weight)
    nn.init.constant_(layer.bias, 0.0)  # Starts at -softplus(0) ≈ -0.69
```

### Strategy 2: Orthogonal Initialization for ReLU Networks

For hidden layers with ReLU activations, orthogonal initialization with gain √2 preserves gradient norms:

```python
def init_hidden_layers_orthogonal(module: nn.Module):
    """Apply orthogonal initialization to all linear layers except the last."""
    for name, layer in module.named_modules():
        if isinstance(layer, nn.Linear):
            # Check if it's an intermediate layer (followed by ReLU)
            if 'value_head' in name or 'q_head' in name:
                if layer.out_features > 1:  # Not the final output layer
                    nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                    nn.init.zeros_(layer.bias)
```

**Why orthogonal?**
- Preserves gradient magnitude through deep networks
- Particularly effective for ReLU activations
- Proven benefits in RL (PPO, DQN implementations)

### Strategy 3: Xavier/Glorot for Sigmoid Outputs

For layers followed by sigmoid/tanh, Xavier initialization is optimal:

```python
def init_sigmoid_layers_xavier(module: nn.Module):
    """Apply Xavier initialization for sigmoid/tanh layers."""
    # For the pre-sigmoid linear layer in V_h^e and X_h
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
```

### Strategy 4: Encoder-Specific Initialization

#### CNN Encoder (Grid Encoding)

For convolutional layers in the state encoder:

```python
def init_cnn_encoder(module: nn.Module):
    """Initialize CNN with appropriate scheme for feature extraction."""
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            # Kaiming for ReLU activations
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)
```

#### Agent Identity Encoder (Embeddings)

For learned embeddings, smaller variance helps:

```python
def init_embeddings(embedding: nn.Embedding):
    """Initialize embeddings with small variance."""
    nn.init.normal_(embedding.weight, mean=0.0, std=0.1)
```

### Strategy 5: Warm-Start from Phase 1 (Transfer Learning)

If a human policy prior was trained in Phase 1, transfer relevant weights:

```python
def transfer_from_phase1(
    v_h_e_network: MultiGridHumanGoalAchievementNetwork,
    phase1_q_network: QNetwork
):
    """Transfer encoder weights from Phase 1 Q-network."""
    # Copy state encoder weights (if architectures match)
    if hasattr(phase1_q_network, 'state_encoder'):
        v_h_e_network.state_encoder.load_state_dict(
            phase1_q_network.state_encoder.state_dict(),
            strict=False  # Allow missing keys
        )
    
    # Copy goal encoder weights
    if hasattr(phase1_q_network, 'goal_encoder'):
        v_h_e_network.goal_encoder.load_state_dict(
            phase1_q_network.goal_encoder.state_dict(),
            strict=False
        )
```

**Caveats**:
- Architecture must match or be adaptable
- May bias V_h^e toward Phase 1 human policy behavior
- Requires Phase 1 checkpoints to be saved

### Strategy 6: Layer-Wise Learning Rate Scaling (LLRD)

Not initialization per se, but related: use different learning rates for different layers:

```python
def get_layer_wise_lr_groups(network, base_lr, decay_factor=0.9):
    """Create parameter groups with decaying LR for deeper layers."""
    params = []
    
    # Encoder layers: lower LR (pretrained or foundational)
    if hasattr(network, 'state_encoder'):
        params.append({
            'params': network.state_encoder.parameters(),
            'lr': base_lr * decay_factor ** 2
        })
    
    # Hidden layers: medium LR
    if hasattr(network, 'value_head'):
        hidden_params = list(network.value_head[:-1].parameters())
        params.append({
            'params': hidden_params,
            'lr': base_lr * decay_factor
        })
    
    # Output layer: full LR (needs most adaptation)
    output_params = list(network.value_head[-1].parameters())
    params.append({
        'params': output_params,
        'lr': base_lr
    })
    
    return params
```

### Strategy 7: Spectral Normalization for Stability

Apply spectral normalization to prevent Lipschitz constant explosion:

```python
from torch.nn.utils import spectral_norm

def apply_spectral_norm(module: nn.Module, layers_to_normalize: list = None):
    """Apply spectral normalization to specified linear layers."""
    for name, layer in module.named_modules():
        if isinstance(layer, nn.Linear):
            if layers_to_normalize is None or name in layers_to_normalize:
                spectral_norm(layer)
```

**Benefits**:
- Bounded Lipschitz constant → stable gradients
- Particularly useful for U_r and Q_r where inputs have high variance

**Drawbacks**:
- Slightly slower forward pass
- May limit network expressiveness

### Strategy 8: Fixup Initialization for Residual Connections

If networks use residual connections (not currently, but could be added):

```python
def init_fixup(module: nn.Module, num_layers: int):
    """Fixup initialization for residual networks without normalization."""
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.bias)
            if hasattr(layer, 'is_residual') and layer.is_residual:
                nn.init.zeros_(layer.weight)
            else:
                # Scale down by depth
                nn.init.normal_(layer.weight, std=num_layers ** (-0.5))
```

## Recommended Implementation

### Phase 1: Conservative Defaults

Add a single initialization function that applies sensible defaults:

```python
def initialize_phase2_networks(networks: Phase2Networks):
    """Apply recommended initialization to all Phase 2 networks."""
    
    # V_h^e: Xavier for sigmoid, zeros for output
    _init_value_network(networks.v_h_e, output_bias=0.0)
    
    # X_h: Same as V_h^e
    _init_value_network(networks.x_h, output_bias=0.0)
    
    # U_r: Orthogonal hidden, appropriate output init
    if networks.u_r is not None:
        _init_negative_network(networks.u_r)
    
    # Q_r: Orthogonal hidden, negative output init
    _init_negative_network(networks.q_r)
    
    # V_r: Same as Q_r
    if networks.v_r is not None:
        _init_negative_network(networks.v_r)

def _init_value_network(network, output_bias=0.0):
    """Initialize network with sigmoid output."""
    for name, module in network.named_modules():
        if isinstance(module, nn.Linear):
            if 'head' in name and module.out_features == 1:
                # Output layer
                nn.init.zeros_(module.weight)
                nn.init.constant_(module.bias, output_bias)
            else:
                # Hidden layers
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)

def _init_negative_network(network):
    """Initialize network with negative output (via -softplus)."""
    for name, module in network.named_modules():
        if isinstance(module, nn.Linear):
            if 'head' in name and module.out_features <= network.num_action_combinations:
                # Output layer
                nn.init.zeros_(module.weight)
                nn.init.constant_(module.bias, 1.0)  # -softplus(1) ≈ -1.3
            else:
                # Hidden layers
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)
```

### Phase 2: Config-Driven Selection

Add configuration options to `Phase2Config`:

```python
@dataclass
class Phase2Config:
    # ... existing fields ...
    
    # Initialization strategy
    init_strategy: str = "default"  # "default", "orthogonal", "xavier", "phase1_transfer"
    init_output_bias_v_h_e: float = 0.0  # Bias for V_h^e output layer
    init_output_bias_q_r: float = 1.0    # Bias for Q_r output layer
    init_use_spectral_norm: bool = False  # Apply spectral normalization
```

### Phase 3: Ablation Studies

Create benchmark to compare initialization strategies:

1. **Baseline**: PyTorch defaults (current)
2. **Output-aware**: Strategy 1 (zero weights, tuned biases)
3. **Orthogonal**: Strategy 2 (orthogonal hidden, output-aware final)
4. **Full**: Strategies 1 + 2 + spectral norm
5. **Transfer**: Strategy 5 (from Phase 1 checkpoint)

Metrics to compare:
- Training loss curves (convergence speed)
- Gradient norms over training
- Final policy performance
- Variance across random seeds

## Implementation Checklist

### Core Changes

1. [ ] Create `src/empo/learning_based/phase2/initialization.py`
   - Define initialization functions for each network type
   - Provide `initialize_phase2_networks()` entry point

2. [ ] Add config options to `Phase2Config`
   - `init_strategy`
   - Per-network output bias settings

3. [ ] Call initialization in `create_phase2_networks()`
   - Apply after network construction, before returning

### Testing

4. [ ] Unit tests for initialization functions
   - Verify output ranges are as expected
   - Verify gradient flow with synthetic forward/backward pass

5. [ ] Integration test comparing strategies
   - Short training runs with different initialization
   - Check for NaN/inf issues

### Documentation

6. [ ] Update `docs/API.md` with new config options

7. [ ] Add initialization strategy to TensorBoard logging
   - Log which strategy was used
   - Log initial weight statistics

## Potential Risks

1. **Over-optimization**: Heavily tuned initialization may not generalize across environments

2. **Interaction with warm-up**: Current warm-up stages assume random initialization; different init may change optimal stage durations

3. **Breaking changes**: Existing checkpoints were trained with PyTorch defaults; loading them after changing initialization code could cause issues (though checkpoints store weights directly)

4. **Debugging difficulty**: Custom initialization adds another variable when debugging training issues

## References

- Glorot & Bengio (2010): "Understanding the difficulty of training deep feedforward neural networks"
- He et al. (2015): "Delving Deep into Rectifiers" (Kaiming initialization)
- Saxe et al. (2014): "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks" (orthogonal)
- Zhang et al. (2019): "Fixup Initialization" (residual networks without normalization)
- Miyato et al. (2018): "Spectral Normalization for GANs" (applicable to RL)
