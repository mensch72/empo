"""
Random Network Distillation (RND) module for curiosity-driven exploration.

RND uses prediction error as a novelty signal: a trainable predictor network
tries to match the output of a fixed random target network. High prediction
error indicates novel/unfamiliar states, providing an exploration bonus.

The module supports multi-encoder input where different state encoders are
introduced during warmup stages. Each encoder's features are weighted by a
coefficient that ramps from 0 to 1 during its introduction stage.

Reference:
    Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2018).
    Exploration by Random Network Distillation. arXiv:1810.12894
"""

from typing import List, Optional

import torch
import torch.nn as nn


class RNDModule(nn.Module):
    """
    Random Network Distillation module for novelty detection.
    
    The target network has frozen random weights. The predictor network is
    trained to match the target's output. Prediction error indicates novelty:
    - High error → state rarely seen → novel
    - Low error → state frequently seen → familiar
    
    Supports multi-encoder input: the input can be a concatenation of features
    from multiple encoders, each multiplied by a coefficient (0-1) that ramps
    up during the warmup stage when that encoder is introduced.
    
    Args:
        input_dim: Total dimension of input features (sum of all encoder dims).
        encoder_dims: Optional list of individual encoder dimensions. If provided,
                     enables per-encoder coefficient weighting. Must sum to input_dim.
        feature_dim: Dimension of RND output features.
        hidden_dim: Dimension of hidden layers.
        normalize: Whether to use running normalization for novelty scores.
        normalization_decay: EMA decay for running mean/std (0.99 = slow adaptation).
    """
    
    def __init__(
        self,
        input_dim: int,
        encoder_dims: Optional[List[int]] = None,
        feature_dim: int = 64,
        hidden_dim: int = 256,
        normalize: bool = True,
        normalization_decay: float = 0.99,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.encoder_dims = encoder_dims
        self.feature_dim = feature_dim
        self.normalize = normalize
        self.normalization_decay = normalization_decay
        
        # Validate encoder_dims if provided
        if encoder_dims is not None:
            total = sum(encoder_dims)
            if total != input_dim:
                raise ValueError(
                    f"encoder_dims must sum to input_dim. "
                    f"Got sum={total}, input_dim={input_dim}"
                )
        
        # Target network: FROZEN RANDOM weights
        # Simple 2-layer MLP - doesn't need to be deep since outputs are arbitrary
        self.target = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        
        # Freeze target network (random weights never change)
        for param in self.target.parameters():
            param.requires_grad = False
        
        # Predictor network: TRAINABLE
        # Slightly deeper than target to have capacity to match
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        
        # Running statistics for normalization
        # Use buffers so they're saved/loaded with state_dict but not trained
        self.register_buffer('running_mean', torch.tensor(0.0))
        self.register_buffer('running_var', torch.tensor(1.0))
        self.register_buffer('update_count', torch.tensor(0))
        
        # Track last batch stats for logging (not saved to state_dict)
        self._last_batch_raw_mean = 0.0
        self._last_batch_raw_std = 1.0
    
    def apply_encoder_coefficients(
        self,
        state_features: torch.Tensor,
        encoder_coefficients: Optional[List[float]] = None
    ) -> torch.Tensor:
        """
        Apply per-encoder coefficients to weighted features.
        
        Each encoder's features are multiplied by its coefficient. This allows
        smooth introduction of new encoders during warmup stages.
        
        Args:
            state_features: Concatenated features (batch_size, input_dim).
            encoder_coefficients: List of coefficients [0, 1] for each encoder.
                                 Must have same length as encoder_dims.
                                 If None, all coefficients are 1.0.
        
        Returns:
            Weighted features (batch_size, input_dim).
        """
        if encoder_coefficients is None or self.encoder_dims is None:
            return state_features
        
        if len(encoder_coefficients) != len(self.encoder_dims):
            raise ValueError(
                f"encoder_coefficients length {len(encoder_coefficients)} != "
                f"encoder_dims length {len(self.encoder_dims)}"
            )
        
        # Build coefficient tensor matching feature dimensions
        # Each encoder's features get multiplied by its coefficient
        coef_expanded = []
        for coef, dim in zip(encoder_coefficients, self.encoder_dims):
            coef_expanded.extend([coef] * dim)
        
        coef_tensor = torch.tensor(
            coef_expanded, dtype=state_features.dtype, device=state_features.device
        )
        
        return state_features * coef_tensor
    
    def compute_novelty(
        self,
        state_features: torch.Tensor,
        update_stats: bool = True,
        encoder_coefficients: Optional[List[float]] = None
    ) -> torch.Tensor:
        """
        Compute novelty scores for given state features.
        
        Novelty = MSE between predictor and target outputs.
        Optionally normalized by running mean/std for stability.
        
        Args:
            state_features: Input features (batch_size, input_dim).
            update_stats: Whether to update running mean/std statistics.
            encoder_coefficients: Optional per-encoder coefficients for smooth
                                 warmup transitions. See apply_encoder_coefficients().
            
        Returns:
            Novelty scores (batch_size,). Higher = more novel.
        """
        # Apply per-encoder coefficients for smooth warmup transitions
        weighted_features = self.apply_encoder_coefficients(
            state_features, encoder_coefficients
        )
        
        with torch.no_grad():
            target_out = self.target(weighted_features)
        
        # Predictor output (with gradients for training)
        pred_out = self.predictor(weighted_features)
        
        # MSE per sample
        novelty = ((target_out - pred_out) ** 2).mean(dim=-1)
        
        if self.normalize:
            # Update running statistics
            if update_stats and self.training:
                batch_mean = novelty.mean().detach()
                batch_var = novelty.var().detach() if len(novelty) > 1 else torch.tensor(0.0, device=novelty.device)
                
                # Track raw batch stats for logging
                self._last_batch_raw_mean = batch_mean.item()
                self._last_batch_raw_std = (batch_var + 1e-8).sqrt().item()
                
                # EMA update
                self.update_count += 1
                if self.update_count == 1:
                    self.running_mean = batch_mean
                    self.running_var = batch_var
                else:
                    alpha = 1.0 - self.normalization_decay
                    self.running_mean = (1 - alpha) * self.running_mean + alpha * batch_mean
                    self.running_var = (1 - alpha) * self.running_var + alpha * batch_var
            
            # Normalize novelty
            std = torch.sqrt(self.running_var + 1e-8)
            novelty = (novelty - self.running_mean) / std
        
        return novelty
    
    def compute_novelty_no_grad(
        self,
        state_features: torch.Tensor,
        encoder_coefficients: Optional[List[float]] = None,
        use_raw: bool = True
    ) -> torch.Tensor:
        """
        Compute novelty scores without gradients (for action selection).
        
        Does not update running statistics.
        
        Args:
            state_features: Input features (batch_size, input_dim).
            encoder_coefficients: Optional per-encoder coefficients for smooth
                                 warmup transitions.
            use_raw: If True, return raw MSE values (better for exploration).
                    If False, return normalized values (legacy behavior).
                    Default is True for better exploration.
            
        Returns:
            Novelty scores (batch_size,). Higher = more novel.
        """
        # Apply per-encoder coefficients for smooth warmup transitions
        weighted_features = self.apply_encoder_coefficients(
            state_features, encoder_coefficients
        )
        
        with torch.no_grad():
            target_out = self.target(weighted_features)
            pred_out = self.predictor(weighted_features)
            
            # MSE per sample (this is the raw novelty - higher = more novel)
            novelty = ((target_out - pred_out) ** 2).mean(dim=-1)
            
            # For exploration, raw novelty is better because:
            # 1. It's always non-negative (no clamping needed)
            # 2. It provides better signal differentiation between states
            # 3. Normalized novelty clusters around 0, losing discrimination
            if not use_raw and self.normalize:
                std = torch.sqrt(self.running_var + 1e-8)
                novelty = (novelty - self.running_mean) / std
            
            return novelty
    
    def compute_loss(
        self,
        state_features: torch.Tensor,
        encoder_coefficients: Optional[List[float]] = None
    ) -> torch.Tensor:
        """
        Compute RND loss for training the predictor.
        
        Loss = MSE between predictor output and (detached) target output.
        This trains the predictor to recognize states it has seen before.
        
        Args:
            state_features: Input features (batch_size, input_dim).
            encoder_coefficients: Optional per-encoder coefficients for smooth
                                 warmup transitions.
            
        Returns:
            Scalar loss tensor.
        """
        # Apply per-encoder coefficients for smooth warmup transitions
        weighted_features = self.apply_encoder_coefficients(
            state_features, encoder_coefficients
        )
        
        with torch.no_grad():
            target_out = self.target(weighted_features)
        
        pred_out = self.predictor(weighted_features)
        
        # MSE per sample (for stats tracking)
        mse_per_sample = ((target_out.detach() - pred_out.detach()) ** 2).mean(dim=-1)
        
        # Track batch stats for logging (this is what we actually care about)
        if self.training:
            batch_mean = mse_per_sample.mean().item()
            batch_std = mse_per_sample.std().item() if len(mse_per_sample) > 1 else 0.0
            self._last_batch_raw_mean = batch_mean
            self._last_batch_raw_std = batch_std
        
        # MSE loss (scalar)
        loss = ((target_out - pred_out) ** 2).mean()
        
        return loss
    
    def get_statistics(self) -> dict:
        """Get running statistics for logging."""
        return {
            'rnd_running_mean': self.running_mean.item(),
            'rnd_running_std': torch.sqrt(self.running_var + 1e-8).item(),
            'rnd_update_count': self.update_count.item(),
            # Raw batch stats show actual novelty values before normalization
            'rnd_batch_raw_mean': self._last_batch_raw_mean,
            'rnd_batch_raw_std': self._last_batch_raw_std,
        }


class RNDModuleWithEncoder(nn.Module):
    """
    RND module that wraps a state encoder for end-to-end use.
    
    This variant takes raw states and uses a provided encoder to convert
    them to features before computing novelty. The encoder is NOT trained
    by RND loss - only the predictor network is trained.
    
    Args:
        encoder: State encoder network with forward(state) -> features.
        feature_dim: Dimension of RND output features.
        hidden_dim: Dimension of hidden layers.
        normalize: Whether to use running normalization.
        normalization_decay: EMA decay for running mean/std.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int = 64,
        hidden_dim: int = 256,
        normalize: bool = True,
        normalization_decay: float = 0.99,
    ):
        super().__init__()
        
        self.encoder = encoder
        
        # Get encoder output dimension
        # Assumes encoder has a feature_dim attribute
        if hasattr(encoder, 'feature_dim'):
            input_dim = encoder.feature_dim
        else:
            raise ValueError("Encoder must have feature_dim attribute")
        
        # Create base RND module
        self.rnd = RNDModule(
            input_dim=input_dim,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            normalize=normalize,
            normalization_decay=normalization_decay,
        )
    
    def compute_novelty_from_features(
        self,
        features: torch.Tensor,
        update_stats: bool = True
    ) -> torch.Tensor:
        """
        Compute novelty from pre-computed encoder features.
        
        Use this when features are already available (e.g., from shared encoder).
        
        Args:
            features: Encoder output features (batch_size, feature_dim).
            update_stats: Whether to update running statistics.
            
        Returns:
            Novelty scores (batch_size,).
        """
        return self.rnd.compute_novelty(features, update_stats=update_stats)
    
    def compute_novelty_from_features_no_grad(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute novelty from features without gradients (for action selection).
        
        Args:
            features: Encoder output features (batch_size, feature_dim).
            
        Returns:
            Novelty scores (batch_size,).
        """
        return self.rnd.compute_novelty_no_grad(features)
    
    def compute_loss(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute RND loss from pre-computed encoder features.
        
        Args:
            features: Encoder output features (batch_size, feature_dim).
            
        Returns:
            Scalar loss tensor.
        """
        return self.rnd.compute_loss(features)
    
    def get_statistics(self) -> dict:
        """Get running statistics for logging."""
        return self.rnd.get_statistics()
    
    @property
    def predictor(self) -> nn.Module:
        """Access to predictor network for optimizer."""
        return self.rnd.predictor
    
    @property
    def target(self) -> nn.Module:
        """Access to target network."""
        return self.rnd.target
