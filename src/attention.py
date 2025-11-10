"""
Attention Mechanisms for Multimodal Fusion

Implements:
1. CrossModalAttention: Attention between different modalities
2. TemporalAttention: Attention over time steps in sequences
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention: Modality A attends to Modality B.
    
    Example: Video features attend to IMU features to incorporate
    relevant motion information at each timestep.
    """
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            query_dim: Dimension of query modality features
            key_dim: Dimension of key/value modality features  
            hidden_dim: Hidden dimension for attention computation
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        # Implement multi-head attention projections
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(key_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.dropout = nn.Dropout(dropout)
        # Hint: Use nn.Linear for Q, K, V projections
        # Query from modality A, Key and Value from modality B
        
        # raise NotImplementedError("Implement cross-modal attention projections")
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for cross-modal attention.
        
        Args:
            query: (batch_size, query_dim) - features from modality A
            key: (batch_size, key_dim) - features from modality B
            value: (batch_size, key_dim) - features from modality B
            mask: Optional (batch_size,) - binary mask for valid keys
            
        Returns:
            attended_features: (batch_size, hidden_dim) - query attended by key/value
            attention_weights: (batch_size, num_heads, 1, 1) - attention scores
        """
        batch_size = query.size(0)
        # Implement multi-head attention computation
        # Steps:
        #   1. Project query, key, value to (batch, num_heads, seq_len, head_dim)
        query = self.query_proj(query).view(batch_size, self.num_heads, 1, self.head_dim)  # (B, H, 1, D/H)
        key = self.key_proj(key).view(batch_size, self.num_heads, 1, self.head_dim)  # (B, H, 1, D/H)
        value = self.value_proj(value).view(batch_size, self.num_heads, 1, self.head_dim)  # (B, H, 1, D/H)
        #   2. Compute attention scores: Q @ K^T / sqrt(head_dim)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, 1, 1)
        #   3. Apply mask if provided (set masked positions to -inf before softmax)
        # when set to -inf softmax produce NaN so we use 0 instead.
        if mask is not None:
            mask = mask.view(batch_size, 1, 1, 1)
            attn_scores = attn_scores.masked_fill(mask == 0, 0)
        #   4. Apply softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, H, 1, 1)
        attn_weights = self.dropout(attn_weights)
        #   5. Apply attention to values: attn_weights @ V
        attended = torch.matmul(attn_weights, value)  # (B, H, 1, D/H)
        #   6. Reshape and project back to hidden_dim
        attended = attended.view(batch_size, -1)  # (B, hidden_dim)
        attended = self.out_proj(attended)  # (B, hidden_dim)
        return attended, attn_weights
        # raise NotImplementedError("Implement cross-modal attention forward pass")


class TemporalAttention(nn.Module):
    """
    Temporal attention: Attend over sequence of time steps.
    
    Useful for: Variable-length sequences, weighting important timesteps
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            feature_dim: Dimension of input features at each timestep
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Implement self-attention over temporal dimension
        # Hint: Similar to CrossModalAttention but Q, K, V from same modality
        self.query_proj = nn.Linear(feature_dim, hidden_dim)
        self.key_proj = nn.Linear(feature_dim, hidden_dim)
        self.value_proj = nn.Linear(feature_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # raise NotImplementedError("Implement temporal attention")
    
    def forward(
        self,
        sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for temporal attention.
        
        Args:
            sequence: (batch_size, seq_len, feature_dim) - temporal sequence
            mask: Optional (batch_size, seq_len) - binary mask for valid timesteps
            
        Returns:
            attended_sequence: (batch_size, seq_len, hidden_dim) - attended features
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size = sequence.size(0)
        seq_len = sequence.size(1)
        # TODO: Implement temporal self-attention
        # Steps:
        #   1. Project sequence to Q, K, V
        Q = self.query_proj(sequence).view(batch_size, self.num_heads, seq_len, self.head_dim)  # (B, H, T, Hd)
        K = self.key_proj(sequence).view(batch_size, self.num_heads, seq_len, self.head_dim)   # (B, H, T, Hd)
        V = self.value_proj(sequence).view(batch_size, self.num_heads, seq_len, self.head_dim)  # (B, H, T, Hd)

        #   2. Compute self-attention over sequence length
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, T, T)

        #   3. Apply mask for variable-length sequences
        if mask is not None:
            # mask: (B, T) → (B, 1, 1, T)
            mask = mask.view(batch_size, 1, 1, seq_len)
            attn_scores = attn_scores.masked_fill(mask == 0, 0.0) 
        #   4. Return attended sequence and weights
        attn_weights = torch.softmax(attn_scores, dim=-1)# (B, H, T, T)
        attn_weights = self.dropout(attn_weights)
        attended = torch.matmul(attn_weights, V)               
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim) # (B, T, hidden_dim)
        attended = self.out_proj(attended)# (B, T, hidden_dim)

        return attended, attn_weights  
        raise NotImplementedError("Implement temporal attention forward pass")
    
    def pool_sequence(
        self,
        sequence: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool sequence to fixed-size representation using attention weights.
        
        Args:
            sequence: (batch_size, seq_len, hidden_dim)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
            
        Returns:
            pooled: (batch_size, hidden_dim) - fixed-size representation
        """
        # Implement attention-based pooling
        # Option 1: Weighted average using mean attention weights
        attnn_mean = attention_weights.mean(dim=1)  # (B, T, T)
        time_weights = attnn_mean.mean(dim=1)  # (B, T)
        timestep_importance = timestep_importance / timestep_importance.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        pooled = torch.sum(sequence * time_weights.unsqueeze(-1), dim=1)  # (B, hidden_dim)
        return pooled
        # Option 2: Learn pooling query vector (not implemented here)
        # Option 3: Take output at special [CLS] token position (not implemented here)
        
        raise NotImplementedError("Implement attention-based pooling")


class PairwiseModalityAttention(nn.Module):
    """
    Pairwise attention between all modality combinations.
    
    For M modalities, computes M*(M-1)/2 pairwise attention operations.
    Example: {video, audio, IMU} -> {video<->audio, video<->IMU, audio<->IMU}
    """
    
    def __init__(
        self,
        modality_dims: dict,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            modality_dims: Dict mapping modality names to feature dimensions
                          Example: {'video': 512, 'audio': 128, 'imu': 64}
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.num_modalities = len(self.modality_names)
        self.hidden_dim = hidden_dim
        
        # Create CrossModalAttention for each modality pair
        self.cross_attn = nn.ModuleDict()
        for i, mod_a_name in enumerate(self.modality_names):
            for j, mod_b_name in enumerate(self.modality_names):
                if i == j:
                    continue
                key = f"{mod_a_name}_to_{mod_b_name}"
                self.cross_attn[key] = CrossModalAttention(
                    query_dim=modality_dims[mod_a_name],
                    key_dim=modality_dims[mod_b_name],
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout
                )
        # Hint: Use nn.ModuleDict with keys like "video_to_audio"
        # For each pair (A, B), create attention A->B and B->A
        
        #raise NotImplementedError("Implement pairwise modality attention")
    
    def forward(
        self,
        modality_features: dict,
        modality_mask: Optional[torch.Tensor] = None
    ) -> Tuple[dict, dict]:
        """
        Apply pairwise attention between all modalities.
        
        Args:
            modality_features: Dict of {modality_name: features}
                             Each tensor: (batch_size, feature_dim)
            modality_mask: (batch_size, num_modalities) - availability mask
            
        Returns:
            attended_features: Dict of {modality_name: attended_features}
            attention_maps: Dict of {f"{mod_a}_to_{mod_b}": attention_weights}
        """
        # Implement pairwise attention
        # Steps:
        #   1. For each modality pair (A, B):
        #      - Apply attention A->B (A attends to B)
        #      - Apply attention B->A (B attends to A)
        attended_features = {m: [] for m in self.modality_names}
        attention_maps = {}
        # iterate over all modality pairs
        for mod_a_name in self.modality_names:
            for mod_b_name in self.modality_names:
                if mod_a_name == mod_b_name:
                    continue
                key = f"{mod_a_name}_to_{mod_b_name}"
                query = modality_features[mod_a_name].unsqueeze(1)  # (B, 1, Dq)
                key_t = modality_features[mod_b_name].unsqueeze(1)  # (B, 1, Dk)
                value_t = key_t
                if modality_mask is not None:
                    idx_a = self.modality_names.index(mod_a_name)
                    idx_b = self.modality_names.index(mod_b_name)

                    valid_q = modality_mask[:, idx_a].bool()  # Which samples of modality A are valid in current batch
                    valid_k = modality_mask[:, idx_b].bool()  # Which samples of modality B are valid

                    # If both modalities are invalid for the entire batch, skip computation
                    if not (valid_q & valid_k).any():
                        continue

                    # mask shape should be (B, 1): True = mask this sample
                    mask = (~valid_k).unsqueeze(1)
                else:
                    valid_q = valid_k = None
                    mask = None
                out, attn_map = self.cross_attn[key](
                    query=query,
                    key=key_t,
                    value=value_t,
                    mask=mask
                )

                out = out.squeeze(1)
                if modality_mask is not None:
                    out = out * valid_q.unsqueeze(-1)
                    attended_features[mod_a_name].append(out)
                    attention_maps[key] = attn_map
        #   2. Aggregate attended features (options: sum)
        for mod in self.modality_names:
            if attended_features[mod]:
                attended_features[mod] = torch.sum(torch.stack(attended_features[mod], dim=0), dim=0)
            else:
                attended_features[mod] = modality_features[mod]
        #   3. Handle missing modalities using mask
        #   4. Return attended features and attention maps for visualization  
        return attended_features, attention_maps


        
        raise NotImplementedError("Implement pairwise attention forward pass")


def visualize_attention(
    attention_weights: torch.Tensor,
    modality_names: list,
    save_path: str = None
) -> None:
    """
    Visualize attention weights between modalities.
    
    Args:
        attention_weights: (num_heads, num_queries, num_keys) or similar
        modality_names: List of modality names for labeling
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # TODO: Implement attention visualization
    # Create heatmap showing which modalities attend to which
    # Useful for understanding fusion behavior
    
    raise NotImplementedError("Implement attention visualization")


if __name__ == '__main__':
    # Simple test
    print("Testing attention mechanisms...")
    
    batch_size = 4
    query_dim = 512  # e.g., video features
    key_dim = 64     # e.g., IMU features
    hidden_dim = 256
    num_heads = 4
    
    # Test CrossModalAttention
    print("\nTesting CrossModalAttention...")
    try:
        attn = CrossModalAttention(query_dim, key_dim, hidden_dim, num_heads)
        
        query = torch.randn(batch_size, query_dim)
        key = torch.randn(batch_size, key_dim)
        value = torch.randn(batch_size, key_dim)
        
        attended, weights = attn(query, key, value)
        
        assert attended.shape == (batch_size, hidden_dim)
        print(f"✓ CrossModalAttention working! Output shape: {attended.shape}")
        
    except NotImplementedError:
        print("✗ CrossModalAttention not implemented yet")
    except Exception as e:
        print(f"✗ CrossModalAttention error: {e}")
    
    # Test TemporalAttention
    print("\nTesting TemporalAttention...")
    try:
        seq_len = 10
        feature_dim = 128
        
        temporal_attn = TemporalAttention(feature_dim, hidden_dim, num_heads)
        sequence = torch.randn(batch_size, seq_len, feature_dim)
        
        attended_seq, weights = temporal_attn(sequence)
        
        assert attended_seq.shape == (batch_size, seq_len, hidden_dim)
        print(f"✓ TemporalAttention working! Output shape: {attended_seq.shape}")
        
    except NotImplementedError:
        print("✗ TemporalAttention not implemented yet")
    except Exception as e:
        print(f"✗ TemporalAttention error: {e}")

