"""
Modality-Specific Encoders for Sensor Fusion

Implements lightweight encoders suitable for CPU training:
1. SequenceEncoder: For time-series data (IMU, audio, motion capture)
2. FrameEncoder: For frame-based data (video features)
3. SimpleMLPEncoder: For pre-extracted features
"""

import torch
import torch.nn as nn
from typing import Optional


class SequenceEncoder(nn.Module):
    """
    Encoder for sequential/time-series sensor data.
    
    Options: 1D CNN, LSTM, GRU, or Transformer
    Output: Fixed-size embedding per sequence
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        encoder_type: str = 'lstm',
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Dimension of input features at each timestep
            hidden_dim: Hidden dimension for RNN/Transformer
            output_dim: Output embedding dimension
            num_layers: Number of encoder layers
            encoder_type: One of ['lstm', 'gru', 'cnn', 'transformer']
            dropout: Dropout probability
        """
        super().__init__()
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Choose ONE of the following architectures:
        
        if encoder_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, 
                                batch_first=True, dropout=dropout)
            self.projection = nn.Linear(hidden_dim, output_dim)

            
        elif encoder_type == 'gru':
            self.rnn = nn.GRU(
                input_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout)
            self.projection = nn.Linear(hidden_dim, output_dim)
            
        elif encoder_type == 'cnn':
            # Stack of Conv1d -> BatchNorm -> ReLU -> Pool
            # Example:
            self.conv = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
            self.projection = nn.Linear(hidden_dim, output_dim)
            # self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
            # self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            # self.pool = nn.AdaptiveAvgPool1d(1)
        elif encoder_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dropout=dropout)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.projection = nn.Linear(input_dim, output_dim)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # raise NotImplementedError(f"Implement {encoder_type} sequence encoder")
    
    def forward(
        self,
        sequence: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode variable-length sequences.
        
        Args:
            sequence: (batch_size, seq_len, input_dim) - input sequence
            lengths: Optional (batch_size,) - actual sequence lengths (for padding)
            
        Returns:
            encoding: (batch_size, output_dim) - fixed-size embedding
        """
        # examine input for NaNs. Leave them as none feature, process in fusion.py
        nan_mask = torch.isnan(sequence).any(dim=(1, 2))  # (B,)
        num_nan = nan_mask.sum().item()
        total = sequence.size(0)
        #print(f"[{self.encoder_type}] input shape={tuple(sequence.shape)}, "
        #      f"{nan_mask.sum().item()}/{total} contain NaN")
        if num_nan == total:
            return None  # all NaN
        clean_idx = (~nan_mask).nonzero(as_tuple=True)[0]
        clean_seq = sequence[clean_idx]
        if self.encoder_type in ['lstm', 'gru']:
            # Handle variable-length sequences
            if lengths is not None:
                packed_sequence = nn.utils.rnn.pack_padded_sequence(clean_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
                packed_output, _ = self.rnn(packed_sequence)
                rnn_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            else:
                rnn_output, _ = self.rnn(clean_seq)
            # Take the last hidden state for each sequence
            if lengths is not None:
                last_indices = (lengths - 1).view(-1, 1, 1).expand(-1, 1, rnn_output.size(2))
                last_hidden = rnn_output.gather(1, last_indices).squeeze(1)
            else:
                last_hidden = rnn_output[:, -1, :]
            
            encoding = self.projection(last_hidden)
        elif self.encoder_type == 'cnn':
            # CNN expects (batch_size, input_dim, seq_len)
            clean_seq = clean_seq.transpose(1, 2)  # (B, D, T)
            conv_output = self.conv(clean_seq).squeeze(-1)  # (B, hidden_dim)
            encoding = self.projection(conv_output)
        
        elif self.encoder_type == 'transformer':
            # Transformer expects (seq_len, batch_size, input_dim)
            clean_seq = clean_seq.transpose(0, 1)  # (T, B, D)
            transformer_output = self.transformer(clean_seq)  # (T, B, D)
            encoding = self.projection(transformer_output[-1])  # Take the last timestep
        else:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}")
        feat = torch.zeros(total, self.output_dim, device=sequence.device)
        feat[clean_idx] = encoding
        # if num_nan > 0:
        #    print(f"[{self.encoder_type}] has {num_nan} samples with NaN, returning zero feature for them")

        return feat
        # Handle variable-length sequences if lengths provided
        # Return fixed-size embedding via pooling or taking last hidden state
        
        # raise NotImplementedError("Implement sequence encoder forward pass")


class FrameEncoder(nn.Module):
    """
    Encoder for frame-based data (e.g., video features).
    
    Aggregates frame-level features into video-level embedding.
    """
    
    def __init__(
        self,
        frame_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        temporal_pooling: str = 'attention',
        dropout: float = 0.1
    ):
        """
        Args:
            frame_dim: Dimension of per-frame features
            hidden_dim: Hidden dimension
            output_dim: Output embedding dimension
            temporal_pooling: How to pool frames ['average', 'max', 'attention']
            dropout: Dropout probability
        """
        super().__init__()
        self.temporal_pooling = temporal_pooling

        # 1. Frame-level processing (optional MLP)
        self.frame_processor = nn.Sequential(
            nn.Linear(frame_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Temporal aggregation (pooling or attention)
        
        if temporal_pooling == 'attention':
            # Learn which frames are important
            self.attention = nn.Linear(hidden_dim, 1)
        elif temporal_pooling in ['average', 'max']:
            # Simple pooling, no learnable parameters needed
            pass
        else:
            raise ValueError(f"Unknown pooling: {temporal_pooling}")
        
        # Add projection layer
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )
        # raise NotImplementedError("Implement frame encoder")
    
    def forward(
        self,
        frames: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode sequence of frames.
        
        Args:
            frames: (batch_size, num_frames, frame_dim) - frame features
            mask: Optional (batch_size, num_frames) - valid frame mask
            
        Returns:
            encoding: (batch_size, output_dim) - video-level embedding
        """

        # 1. Process frames (optional)
        processed_frames = self.frame_processor(frames)
        # 2. Apply temporal pooling
        if self.temporal_pooling == 'attention':
            pooled = self.attention_pool(processed_frames, mask)  # (B, hidden_dim)
        elif self.temporal_pooling == 'average':
            if mask is not None:
                processed_frames = processed_frames * mask.unsqueeze(-1)  # Mask invalid frames
                pooled = processed_frames.sum(dim=1) / mask.sum(dim=1, keepdim=True)  # Weighted average
            else:
                pooled = processed_frames.mean(dim=1)  # Simple average pooling
        elif self.temporal_pooling == 'max':
            if mask is not None:
                processed_frames = processed_frames.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            pooled = processed_frames.max(dim=1).values  # Max pooling
        else:
            raise ValueError(f"Unknown pooling: {self.temporal_pooling}")
        # 3. Project to output dimension
        encoding = self.projection(pooled)
        return encoding
        # raise NotImplementedError("Implement frame encoder forward pass")
    
    def attention_pool(
        self,
        frames: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool frames using learned attention weights.
        
        Args:
            frames: (batch_size, num_frames, frame_dim)
            mask: Optional (batch_size, num_frames) - valid frames
            
        Returns:
            pooled: (batch_size, frame_dim) - attended frame features
        """
        # TODO: Implement attention pooling
        # 1. Compute attention scores for each frame
        attn_scores = self.attention(frames).squeeze(-1)  # (B, T)
        # 2. Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask.bool(), float('-inf'))  # Mask invalid frames
        # 3. Softmax to get weights
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, T)

        # 4. Weighted sum of frames
        pooled = torch.sum(frames * attn_weights.unsqueeze(-1), dim=1)  # (B, D)
        return pooled
        # raise NotImplementedError("Implement attention pooling")


class SimpleMLPEncoder(nn.Module):
    """
    Simple MLP encoder for pre-extracted features.
    
    Use this when working with pre-computed features
    (e.g., ResNet features for images, MFCC for audio).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        batch_norm: bool = True
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of hidden layers
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        # Implement MLP encoder
        # Architecture: Input -> [Linear -> BatchNorm -> ReLU -> Dropout] x num_layers -> Output
        
        layers = []
        current_dim = input_dim
        
        # Add hidden layers
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        # Add output layer
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # raise NotImplementedError("Implement MLP encoder")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode features through MLP.
        
        Args:
            features: (batch_size, input_dim) - input features
            
        Returns:
            encoding: (batch_size, output_dim) - encoded features
        """
        # Implement forward pass
        return self.encoder(features)
        
        raise NotImplementedError("Implement MLP encoder forward pass")


def build_encoder(
    modality: str,
    input_dim: int,
    output_dim: int,
    encoder_config: dict = None
) -> nn.Module:
    """
    Factory function to build appropriate encoder for each modality.
    
    Args:
        modality: Modality name ('video', 'audio', 'imu', etc.)
        input_dim: Input feature dimension
        output_dim: Output embedding dimension
        encoder_config: Optional config dict with encoder hyperparameters
        
    Returns:
        Encoder module appropriate for the modality
    """
    if encoder_config is None:
        encoder_config = {}
    
    # TODO: Implement encoder selection logic
    # Example heuristics:
    # - 'video' -> FrameEncoder
    # - 'imu', 'audio', 'mocap' -> SequenceEncoder
    # - Pre-extracted features -> SimpleMLPEncoder
    
    if modality in ['video', 'frames']:
        return FrameEncoder(
            frame_dim=input_dim,
            output_dim=output_dim,
            **encoder_config
        )
    elif modality in ['audio', 'mocap', 'accelerometer'] or modality:
        return SequenceEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            **encoder_config
        )
    else:
        # Default to MLP for unknown modalities
        return SimpleMLPEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            **encoder_config
        )


if __name__ == '__main__':
    # Test encoders
    print("Testing encoders...")
    
    batch_size = 4
    seq_len = 100
    input_dim = 64
    output_dim = 128
    
    # Test SequenceEncoder
    print("\nTesting SequenceEncoder...")
    for enc_type in ['lstm', 'gru', 'cnn']:
        try:
            encoder = SequenceEncoder(
                input_dim=input_dim,
                output_dim=output_dim,
                encoder_type=enc_type
            )
            
            sequence = torch.randn(batch_size, seq_len, input_dim)
            output = encoder(sequence)
            
            assert output.shape == (batch_size, output_dim)
            print(f"✓ {enc_type} encoder working! Output shape: {output.shape}")
            
        except NotImplementedError:
            print(f"✗ {enc_type} encoder not implemented yet")
        except Exception as e:
            print(f"✗ {enc_type} encoder error: {e}")
    
    # Test FrameEncoder
    print("\nTesting FrameEncoder...")
    try:
        num_frames = 30
        frame_dim = 512
        
        encoder = FrameEncoder(
            frame_dim=frame_dim,
            output_dim=output_dim,
            temporal_pooling='attention'
        )
        
        frames = torch.randn(batch_size, num_frames, frame_dim)
        output = encoder(frames)
        
        assert output.shape == (batch_size, output_dim)
        print(f"✓ FrameEncoder working! Output shape: {output.shape}")
        
    except NotImplementedError:
        print("✗ FrameEncoder not implemented yet")
    except Exception as e:
        print(f"✗ FrameEncoder error: {e}")
    
    # Test SimpleMLPEncoder
    print("\nTesting SimpleMLPEncoder...")
    try:
        encoder = SimpleMLPEncoder(
            input_dim=input_dim,
            output_dim=output_dim
        )
        
        features = torch.randn(batch_size, input_dim)
        output = encoder(features)
        
        assert output.shape == (batch_size, output_dim)
        print(f"✓ SimpleMLPEncoder working! Output shape: {output.shape}")
        
    except NotImplementedError:
        print("✗ SimpleMLPEncoder not implemented yet")
    except Exception as e:
        print(f"✗ SimpleMLPEncoder error: {e}")

