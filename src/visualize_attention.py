import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse

from src.fusion import HybridFusion
from src.analysis import plot_attention_weights, plot_calibration_diagram

from data import create_dataloaders

from train import MultimodalFusionModule
def collect_and_visualize(
    model,
    dataloader: DataLoader,
    modality_names,
    device: str = 'cuda',
    output_dir: str = 'analysis'
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_modalities = len(modality_names)
    
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    attention_weights_sum = torch.zeros(num_modalities, num_modalities)
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (features, mask, labels) in enumerate(dataloader):  # dataloader returns 3 values
            # Move data to device (ignore mask, let model handle feature validity)
            features = {k: v.to(device) if v is not None else None 
                      for k, v in features.items()}
            labels = labels.to(device)
            
            # Encode each modality
            encoded_features = {}
            for modality in modality_names:
                if modality in features and features[modality] is not None:
                    # Only encode valid features
                    feat = features[modality]
                    encoded_feat = model.encoders[modality](feat)
                    if encoded_feat is not None:  # Encoder might return None due to NaN
                        encoded_features[modality] = encoded_feat
            
            # Pass encoded features directly to fusion model, let it handle feature validity
            logits, attention_info = model.fusion_model(encoded_features, None, return_attention=True)
            
            probs = torch.softmax(logits, dim=-1)
            confidences, predictions = torch.max(probs, dim=-1)
            
            all_probs.append(confidences.cpu())
            all_preds.append(predictions.cpu())
            all_labels.append(labels.cpu())
            
            # Process cross-modal attention
            batch_attention = torch.zeros_like(attention_weights_sum)
            if 'cross_attn' in attention_info:
                cross_attn = attention_info['cross_attn']
                for i, qi in enumerate(modality_names):
                    if qi not in cross_attn:
                        continue
                    for j, kj in enumerate(modality_names):
                        # Skip diagonal elements (self-attention) and non-existent cross-attention
                        if i == j or kj not in cross_attn[qi]:
                            continue
                        # attention weights shape: (batch_size, num_heads, 1, 1)
                        attn = cross_attn[qi][kj]
                        
                        # Extract raw attention scores
                        # Average over batch and head dimensions
                        raw_attn = attn.mean(dim=(0, 1))  # (1, 1)
                        batch_attention[i, j] = raw_attn.mean().cpu()

            attention_weights_sum += batch_attention

            # Process fusion weights (if they exist)
            if 'fusion_weights' in attention_info:
                fusion_w = attention_info['fusion_weights']  # (batch_size, num_modalities)
                fusion_w = fusion_w.mean(dim=0)  # (num_modalities,)
                # We can optionally add fusion weights to the visualization
                # Here we print them out
                print(f"Fusion weights: {fusion_w}")
            
            attention_weights_sum += batch_attention
            num_batches += 1
            
            if batch_idx == 0:
                print(f"Successfully processed first batch, found {len(modality_names)} modalities")
    
    # Calculate average attention weights
    attention_weights = attention_weights_sum / num_batches
    
    # Normalize non-diagonal elements
    diagonal_mask = ~torch.eye(attention_weights.shape[0], dtype=torch.bool)
    non_diagonal = attention_weights[diagonal_mask]
    
    # Set diagonal elements to NaN
    attention_weights = attention_weights.numpy()
    np.fill_diagonal(attention_weights, np.nan)
    
    confidences = torch.cat(all_probs).numpy()
    predictions = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    plot_attention_weights(
        attention_weights=attention_weights,
        modality_names=modality_names,
        save_path=output_dir / 'attention_weights.png'
    )
    
    plot_calibration_diagram(
        confidences=confidences,
        predictions=predictions,
        labels=labels,
        save_path=output_dir / 'calibration.png'
    )
    
    print(f"Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='runs/a2_hybrid_pamap2/checkpoints/last-v1.ckpt',
                       help='Path to saved model checkpoint')
    parser.add_argument('--data_root', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='analysis/viz',
                       help='Directory to save visualizations')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
        # Due to PyTorch 2.6 safe loading, we implement manual loading
    print(f"Manually loading checkpoint: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    hparams = checkpoint["hyper_parameters"]
    print("Instantiating model from hyperparameters...")
    model = MultimodalFusionModule(**hparams)
    print("Loading state dict...")
    model.load_state_dict(checkpoint["state_dict"])

    print("Model loaded manually successfully.")
    model.eval()
    model.to(args.device)
    
    # Get config from model
    config = model.config
    
    # Create dataloaders
    print("Creating dataloaders...")
    _, _, test_loader = create_dataloaders(
        dataset_name=config.dataset.name,
        data_dir=config.dataset.data_dir,
        modalities=config.dataset.modalities,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers
    )
    
    collect_and_visualize(
        model=model,
        dataloader=test_loader,
        modality_names=config.dataset.modalities,
        device=args.device,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()