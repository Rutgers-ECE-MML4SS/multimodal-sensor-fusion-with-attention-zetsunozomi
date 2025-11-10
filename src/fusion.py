"""
Multimodal Fusion Architectures for Sensor Integration

This module implements three fusion strategies:
1. Early Fusion: Concatenate features before processing
2. Late Fusion: Independent processing, combine predictions
3. Hybrid Fusion: Cross-modal attention + learned weighting
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from attention import CrossModalAttention


class EarlyFusion(nn.Module):
    """
    Early fusion: Concatenate encoder outputs and process jointly.
    
    Pros: Joint representation learning across modalities
    Cons: Requires temporal alignment, sensitive to missing modalities
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_classes: int = 11,
        dropout: float = 0.1
    ):
        """
        Args:
            modality_dims: Dictionary mapping modality name to feature dimension
                          Example: {'video': 512, 'imu': 64}
            hidden_dim: Hidden dimension for fusion network
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.modality_dims = modality_dims
        # get total input dimension
        total_dim = sum(modality_dims.values())
        # Define fusion network as the HINT.
        self.fusion= nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        # Hint: Concatenate all modality features, pass through MLP
        # Architecture suggestion:
        #   concat_dim = sum(modality_dims.values())
        #   Linear(concat_dim, hidden_dim) -> ReLU -> Dropout
        #   Linear(hidden_dim, hidden_dim) -> ReLU -> Dropout
        #   Linear(hidden_dim, num_classes)
        #raise NotImplementedError("Implement early fusion architecture")
    
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with early fusion.
        
        Args:
            modality_features: Dict of {modality_name: features}
                             Each tensor shape: (batch_size, feature_dim)
            modality_mask: Binary mask (batch_size, num_modalities)
                          1 = available, 0 = missing
                          
        Returns:
            logits: (batch_size, num_classes)
        """
        for feat in modality_features.values():
            if feat is not None:
                batch_size = feat.shape[0]
                break
        else:
            raise ValueError("At least one modality feature must be provided")
        device = next(self.parameters()).device
        fused = []
        # Steps:
        #   1. Extract features for each modality from dict
        #   2. Handle missing modalities (use zeros or learned embeddings)
        # Determine batch size from available features
        for idx, name in enumerate(self.modality_names):
            feat = modality_features.get(name, None)
            expected_dim = self.modality_dims[name]
            # case 1: modality missing in input dict (name not found)
            if feat is None:
                feat = torch.zeros(batch_size, expected_dim, device=device)
            # case 2: modality present but masked out
            if modality_mask is not None:
                mask = modality_mask[:, idx].unsqueeze(1)  # (batch_size, 1)
                feat = feat * mask  # Zero out missing modalities
            # case 3: modality has NaNs in encoder input. It's processed in encoders.py to return zero feature for those samples.
            fused.append(feat)
        #   3. Concatenate all features
        #   4. Pass through fusion network
        fused = torch.cat(fused, dim=1)  # (batch_size, total_dim)
        logits = self.fusion(fused)  # (batch_size, num_classes)
        return logits
        raise NotImplementedError("Implement early fusion forward pass")


class LateFusion(nn.Module):
    """
    Late fusion: Independent classifiers per modality, combine predictions.
    
    Pros: Handles asynchronous sensors, modular per-modality training
    Cons: Limited cross-modal interaction, fusion only at decision level
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_classes: int = 11,
        dropout: float = 0.1
    ):
        """
        Args:
            modality_dims: Dictionary mapping modality name to feature dimension
            hidden_dim: Hidden dimension for per-modality classifiers
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.num_modalities = len(self.modality_names)
        self.classifiers = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_classes)
                )
                for name, dim in modality_dims.items()
            }
        )
        # Hint: Use nn.ModuleDict to store per-modality classifiers
        # Each classifier: Linear(modality_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, num_classes)
        
        # Learn fusion weights (how to combine predictions)
        # Option 1: Learnable weights (nn.Parameter)
        self.raw_fusion_weights = nn.Parameter(torch.zeros(self.num_modalities))
        # Option 2: Attention over predictions (not implemented here)
        # Option 3: Simple averaging (not implemented here)
        # raise NotImplementedError("Implement late fusion architecture")
    
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with late fusion.
        
        Args:
            modality_features: Dict of {modality_name: features}
            modality_mask: Binary mask for available modalities
            
        Returns:
            logits: (batch_size, num_classes) - fused predictions
            per_modality_logits: Dict of individual modality predictions
        """
        for feat in modality_features.values():
            if feat is not None:
                batch_size = feat.shape[0]
                break
        else:
            raise ValueError("At least one modality feature must be provided")
        device = next(self.parameters()).device

        # Steps:
        #   1. Get predictions from each modality classifier
        #   2. Handle missing modalities (mask out or skip)
        per_modality_logits: Dict[str, torch.Tensor] = {}
        modality_logits = []
        for idx, name in enumerate(self.modality_names):
            feat = modality_features.get(name, None)
            expected_dim = self.classifiers[name][0].in_features
            # case 1: modality missing in input dict (name not found)
            if feat is None:
                feat = torch.zeros(batch_size, expected_dim, device=device)
            # case 2: modality present but masked out
            if modality_mask is not None:
                mask = modality_mask[:, idx].unsqueeze(1)  # (batch_size, 1)
                feat = feat * mask  # Zero out missing modalities
            logits = self.classifiers[name](feat)  # (batch_size, num_classes)
            per_modality_logits[name] = logits
            modality_logits.append(logits.unsqueeze(2))  # (batch_size, num_classes, 1)
        

        #   3. Combine predictions using fusion weights
        weights = torch.softmax(self.raw_fusion_weights, dim=0)  # (num_modalities,)
        fused_logits = sum(w * logit for w, logit in zip(weights, modality_logits))
        fused_logits = fused_logits.squeeze(2)
        #   4. Return both fused and per-modality predictions
        return fused_logits, per_modality_logits
        raise NotImplementedError("Implement late fusion forward pass")


class HybridFusion(nn.Module):
    """
    Hybrid fusion: Cross-modal attention + learned fusion weights.
    
    Pros: Rich cross-modal interaction, robust to missing modalities
    Cons: More complex, higher computation cost
    
    This is the main focus of the assignment!
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_classes: int = 11,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            modality_dims: Dictionary mapping modality name to feature dimension
            hidden_dim: Hidden dimension for fusion
            num_classes: Number of output classes
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.num_modalities = len(self.modality_names)
        self.hidden_dim = hidden_dim
        
        # Project each modality to common hidden dimension
        # Hint: Use nn.ModuleDict with Linear layers per modality
        self.proj = nn.ModuleDict({
            name: nn.Linear(in_dim, hidden_dim) 
            for name, in_dim in modality_dims.items()
        })
        # Implement cross-modal attention
        # Use CrossModalAttention from attention.py
        # Each modality should attend to all other modalities
        self.cross_attn = CrossModalAttention(
            query_dim=hidden_dim,
            key_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Learn adaptive fusion weights based on modality availability
        # Hint: Small MLP that takes modality mask and outputs weights
        mid = max(32, hidden_dim // 2)
        self.weight_mlp = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, 1)
        )
        # Final classifier
        # Takes fused representation -> num_classes logits
        self.cls = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.dropout = nn.Dropout(dropout)
        # raise NotImplementedError("Implement hybrid fusion architecture")
    
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with hybrid fusion.
        
        Args:
            modality_features: Dict of {modality_name: features}
            modality_mask: Binary mask for available modalities
            return_attention: If True, return attention weights for visualization
            
        Returns:
            logits: (batch_size, num_classes)
            attention_info: Optional dict with attention weights and fusion weights
        """
        # Implement forward pass
        batch_size = next(iter(modality_features.values())).size(0)
        device = next(iter(modality_features.values())).device
        num_modalities = len(self.modality_names)
        
        # modality mask
        modality_mask = torch.ones(batch_size, num_modalities, device=device, dtype=torch.float32) if modality_mask is None else modality_mask.float().to(device)
        name2idx = {name: idx for idx, name in enumerate(self.modality_names)}

        # Steps:
        #   1. Project all modalities to common hidden dimension
        projected_feats = {}

        for name in self.modality_names:
            if name not in modality_features or modality_features[name] is None:
                modality_mask[:, name2idx[name]] = 0.0
                projected_feats[name] = None
                continue
                
            x = modality_features[name]  # (B, Dm)
            if torch.allclose(x, torch.zeros_like(x)):
                modality_mask[:, name2idx[name]] = 0.0
                projected_feats[name] = None
            projected_feats[name] = self.proj[name](x)  # (B, H)
        
        #   2. Apply cross-modal attention between modality pairs
        attended_feats = {}
        attention_dict = {} if return_attention else None
        
        # iterate over each modality as query
        for qname in self.modality_names:
            if projected_feats[qname] is None:
                attended_feats[qname] = torch.zeros(batch_size, self.hidden_dim, device=device)
                continue
            q = projected_feats[qname]  # (B, H)
            pair_attended = []
            pair_attn_vis = {}
            # iterate over other modalities as keys
            for kname in self.modality_names:
                if kname == qname or projected_feats[kname] is None:
                    continue
                key_mask = modality_mask[:, name2idx[kname]]  # (B,)
                k = projected_feats[kname]
                attended_ij, attn_w_ij = self.cross_attn(q, k, k, key_mask)  # (B,H), (B,Hh,1,1)
                pair_attended.append(attended_ij)
                if return_attention:
                    pair_attn_vis[kname] = attn_w_ij
            if len(pair_attended) > 0:
                cross_mean = torch.stack(pair_attended, dim=0).mean(dim=0)
                attended_feats[qname] = 0.5 * q + 0.5 * cross_mean
            else:
                attended_feats[qname] = q

            if return_attention:
                attention_dict[qname] = pair_attn_vis
        #   3. Compute adaptive fusion weights based on modality_mask
        fusion_weights = self.compute_adaptive_weights(attended_feats, modality_mask)  # (B, M)
        #   4. Fuse attended representations with learned weights
        feats_stack = torch.stack([attended_feats[name] for name in self.modality_names], dim=1)  # (B, M, H)
        fused = torch.sum(fusion_weights.unsqueeze(-1) * feats_stack, dim=1)  # (B, H)
        #   5. Pass through final classifier
        fused = self.dropout(fused)
        logits = self.cls(fused)
        #   6. Optionally return attention weights for visualization
        if return_attention:
            return logits, {"cross_attn": attention_dict, "fusion_weights": fusion_weights}
        else:
            return logits, None
        raise NotImplementedError("Implement hybrid fusion forward pass")
    
    def compute_adaptive_weights(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adaptive fusion weights based on modality availability.
        
        Args:
            modality_features: Dict of modality features
            modality_mask: (batch_size, num_modalities) binary mask
            
        Returns:
            weights: (batch_size, num_modalities) normalized fusion weights
        """
        # Implement adaptive weighting
        # Ideas:
        #   1. Learn weight predictor from modality features + mask
        #   2. Higher weights for more reliable/informative modalities
        feats = torch.stack(
            [modality_features[name] for name in self.modality_names],
            dim=1
        )  # (B, M, H)
        masked_feats = feats * modality_mask.unsqueeze(-1)
        logits = self.weight_mlp(masked_feats).squeeze(-1)  # (B, M)
        
        
        #   3. Ensure weights sum to 1 (softmax) and respect mask
        weights = torch.softmax(logits, dim=-1)  # (B, M)
        return weights
        raise NotImplementedError("Implement adaptive weight computation")


# Helper functions

def build_fusion_model(
    fusion_type: str,
    modality_dims: Dict[str, int],
    num_classes: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to build fusion models.
    
    Args:
        fusion_type: One of ['early', 'late', 'hybrid']
        modality_dims: Dictionary mapping modality names to dimensions
        num_classes: Number of output classes
        **kwargs: Additional arguments for fusion model
        
    Returns:
        Fusion model instance
    """
    fusion_classes = {
        'early': EarlyFusion,
        'late': LateFusion,
        'hybrid': HybridFusion,
    }
    
    if fusion_type not in fusion_classes:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    return fusion_classes[fusion_type](
        modality_dims=modality_dims,
        num_classes=num_classes,
        **kwargs
    )


if __name__ == '__main__':
    # Simple test to verify implementation
    print("Testing fusion architectures...")
    
    # Test configuration
    modality_dims = {'video': 512, 'imu': 64}
    num_classes = 11
    batch_size = 4
    
    # Create dummy features
    features = {
        'video': torch.randn(batch_size, 512),
        'imu': torch.randn(batch_size, 64)
    }
    mask = torch.tensor([[1, 1], [1, 0], [0, 1], [1, 1]])  # Different availability patterns
    
    # Test each fusion type
    for fusion_type in ['early', 'late', 'hybrid']:
        print(f"\nTesting {fusion_type} fusion...")
        try:
            model = build_fusion_model(fusion_type, modality_dims, num_classes)
            
            if fusion_type == 'late':
                logits, per_mod_logits = model(features, mask)
            else:
                logits = model(features, mask)
            
            assert logits.shape == (batch_size, num_classes), \
                f"Expected shape ({batch_size}, {num_classes}), got {logits.shape}"
            print(f"✓ {fusion_type} fusion working! Output shape: {logits.shape}")
            
        except NotImplementedError:
            print(f"✗ {fusion_type} fusion not implemented yet")
        except Exception as e:
            print(f"✗ {fusion_type} fusion error: {e}")

