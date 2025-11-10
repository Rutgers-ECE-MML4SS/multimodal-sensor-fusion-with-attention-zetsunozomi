"""
Uncertainty Quantification for Multimodal Fusion

Implements methods for estimating and calibrating confidence scores:
1. MC Dropout for epistemic uncertainty
2. Calibration metrics (ECE, reliability diagrams)
3. Uncertainty-weighted fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict


class MCDropoutUncertainty(nn.Module):
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    Runs multiple forward passes with dropout enabled to estimate
    prediction uncertainty via variance.
    """
    
    def __init__(self, model: nn.Module, num_samples: int = 10):
        """
        Args:
            model: The model to estimate uncertainty for
            num_samples: Number of MC dropout samples
        """
        super().__init__()
        self.model = model
        self.num_samples = num_samples
    
    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.
        
        Returns:
            mean_logits: (batch_size, num_classes) - mean prediction
            uncertainty: (batch_size,) - prediction uncertainty (variance)
        """
        
        # Implement MC Dropout
        # Steps:
        #   1. Enable dropout in model (model.train())
        self.model.train()
        #   2. Run num_samples forward passes
        preds = []
        for _ in range(self.num_samples):
            logits = self.model(*args, **kwargs)
            preds.append(F.softmax(logits, dim=1).unsqueeze(0))  # (1, batch_size, num_classes)
        preds = torch.cat(preds, dim=0)  # (num_samples, batch_size, num_classes)

        #   3. Compute mean and variance of predictions
        mean_preds = preds.mean(dim=0)  # (batch_size, num_classes)
        uncertainty = preds.var(dim=0).mean(dim=1)  # (batch_size
        #   4. Return mean prediction and uncertainty
        return mean_preds, uncertainty
        raise NotImplementedError("Implement MC Dropout uncertainty")


class CalibrationMetrics:
    """
    Compute calibration metrics for confidence estimates.
    
    Key metrics:
    - Expected Calibration Error (ECE)
    - Maximum Calibration Error (MCE)  
    - Negative Log-Likelihood (NLL)
    """
    
    @staticmethod
    def expected_calibration_error(
        confidences: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        num_bins: int = 15
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE = Σ (|bin_accuracy - bin_confidence|) * (bin_size / total_size)
        
        Args:
            confidences: (N,) - predicted confidence scores [0, 1]
            predictions: (N,) - predicted class indices
            labels: (N,) - ground truth class indices
            num_bins: Number of bins for calibration
            
        Returns:
            ece: Expected Calibration Error (lower is better)
        """
        device = confidences.device
        ece = torch.zeros(1, device=device)
        total_samples = confidences.size(0)


        # Implement ECE calculation
        # Steps:
        #   1. Bin predictions by confidence level
        # Compute bin counts and edges using histc
        bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        #   2. For each bin, compute accuracy and average confidence
        #   3. Compute weighted difference |accuracy - confidence|
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            bin_size = in_bin.float().sum()

            if bin_size > 0:
                acc = (predictions[in_bin] == labels[in_bin]).float().mean()
                conf = confidences[in_bin].mean()
                ece += (bin_size / total_samples) * torch.abs(acc - conf)

        #   4. Return ECE
        return ece.item()
        # Hint: Use np.histogram or torch.histc to bin confidences
        
        raise NotImplementedError("Implement ECE calculation")
    
    @staticmethod
    def maximum_calibration_error(
        confidences: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        num_bins: int = 15
    ) -> float:
        """
        Compute Maximum Calibration Error (MCE).
        
        MCE = max_bin |bin_accuracy - bin_confidence|
        
        Returns:
            mce: Maximum calibration error across bins
        """
        # Implement MCE
        # Similar to ECE but take max instead of average
        device = confidences.device
        mce = torch.zeros(1, device=device)

        bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            bin_size = in_bin.float().sum()

            if bin_size > 0:
                acc = (predictions[in_bin] == labels[in_bin]).float().mean()
                conf = confidences[in_bin].mean()
                diff = torch.abs(acc - conf)
                if diff > mce:
                    mce = diff

        return mce.item()
        raise NotImplementedError("Implement MCE calculation")
    
    @staticmethod
    def negative_log_likelihood(
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """
        Compute average Negative Log-Likelihood (NLL).
        
        NLL = -log P(y_true | x)
        
        Args:
            logits: (N, num_classes) - predicted logits
            labels: (N,) - ground truth labels
            
        Returns:
            nll: Average negative log-likelihood
        """
        # Implement NLL
        import torch.nn.functional as F

        nll = F.cross_entropy(logits, labels, reduction='mean')
        return nll.item()
        # Hint: Use F.cross_entropy which computes -log(softmax(logits)[label])
        
        raise NotImplementedError("Implement NLL calculation")
    
    @staticmethod
    def reliability_diagram(
        confidences: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
        num_bins: int = 15,
        save_path: str = None
    ) -> None:
        """
        Plot reliability diagram showing calibration.
        
        X-axis: Predicted confidence
        Y-axis: Actual accuracy
        Perfect calibration: y = x (diagonal line)
        
        Args:
            confidences: (N,) - confidence scores
            predictions: (N,) - predicted classes
            labels: (N,) - ground truth
            num_bins: Number of bins
            save_path: Optional path to save plot
        """
        import matplotlib.pyplot as plt
        import numpy as np
        # Implement reliability diagram
        # Steps:
        #   1. Bin predictions by confidence
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        accuracies = np.zeros(num_bins)
        confidences_mean = np.zeros(num_bins)
        bin_counts = np.zeros(num_bins)
        #   2. Compute accuracy per bin
        for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            bin_count = np.sum(in_bin)
            bin_counts[i] = bin_count

            if bin_count > 0:
                accuracies[i] = np.mean(predictions[in_bin] == labels[in_bin])
                confidences_mean[i] = np.mean(confidences[in_bin])
        ece = np.sum(bin_counts / np.sum(bin_counts) * np.abs(accuracies - confidences_mean))
        #   3. Plot bar chart: confidence vs accuracy
        plt.figure(figsize=(6, 6))
        plt.bar(bin_lowers + (1 / (2 * num_bins)),
                accuracies, width=1/num_bins, edgecolor='black', alpha=0.7, label='Accuracy per bin')
        #   4. Add diagonal line for perfect calibration
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        #   5. Add ECE to plot
        plt.title(f'Reliability Diagram (ECE={ece:.3f})')
        plt.legend(loc='best')
        plt.grid(True)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            print("No save path provided; skipping save.")
        plt.close()
        
        # raise NotImplementedError("Implement reliability diagram")


class UncertaintyWeightedFusion(nn.Module):
    """
    Fuse modalities weighted by inverse uncertainty.
    
    Intuition: More uncertain modalities get lower weight.
    Weight_i ∝ 1 / (uncertainty_i + ε)
    """
    
    def __init__(self, epsilon: float = 1e-6):
        """
        Args:
            epsilon: Small constant to avoid division by zero
        """
        super().__init__()
        self.epsilon = epsilon
    
    def forward(
        self,
        modality_predictions: Dict[str, torch.Tensor],
        modality_uncertainties: Dict[str, torch.Tensor],
        modality_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse modality predictions weighted by inverse uncertainty.
        
        Args:
            modality_predictions: Dict of {modality: logits}
                                Each tensor: (batch_size, num_classes)
            modality_uncertainties: Dict of {modality: uncertainty}
                                   Each tensor: (batch_size,)
            modality_mask: (batch_size, num_modalities) - availability mask
            
        Returns:
            fused_logits: (batch_size, num_classes) - weighted fusion
            fusion_weights: (batch_size, num_modalities) - used weights
        """
        # Implement uncertainty-weighted fusion
        # Steps:
        #   1. Compute inverse uncertainty weights: w_i = 1/(σ_i + ε)
        uncertainties = torch.stack([modality_uncertainties[m] for m in modality_uncertainties], dim=1)  # (B, M)
        inv_uncertainties = 1.0 / (uncertainties + self.epsilon)  # (B, M)
        #   2. Normalize weights to sum to 1
        weights = inv_uncertainties * modality_mask  # (B, M)
        #   3. Apply modality mask (zero weight for missing modalities)
        weight_sum = weights.sum(dim=1, keepdim=True) + self.epsilon
        normalized_weights = weights / weight_sum  # (B, M)
        #   4. Fuse predictions: Σ w_i * pred_i
        preds = torch.stack([modality_predictions[m] for m in modality_predictions], dim=1)  # (B, M, C)
        fused_logits = torch.sum(normalized_weights.unsqueeze(-1) * preds, dim=1)  # (B, C)
        #   5. Return fused predictions and weights
        return fused_logits, normalized_weights
        # raise NotImplementedError("Implement uncertainty-weighted fusion")


class TemperatureScaling(nn.Module):
    """
    Post-hoc calibration via temperature scaling.
    
    Learns a single temperature parameter T that scales logits:
    P_calibrated = softmax(logits / T)
    
    Reference: Guo et al. "On Calibration of Modern Neural Networks", ICML 2017
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: (batch_size, num_classes) - model outputs
            
        Returns:
            scaled_logits: (batch_size, num_classes) - temperature-scaled logits
        """
        return logits / self.temperature
    
    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50
    ) -> None:
        """
        Learn optimal temperature on validation set.
        
        Args:
            logits: (N, num_classes) - validation set logits
            labels: (N,) - validation set labels
            lr: Learning rate
            max_iter: Maximum optimization iterations
        """
        # TODO: Implement temperature calibration
        # Steps:
        #   1. Initialize temperature = 1.0
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        nll_criterion = nn.CrossEntropyLoss()

        logits = logits.detach()
        labels = labels.detach()
        #   2. Optimize temperature to minimize NLL on validation set
        #   3. Use LBFGS or Adam optimizer
        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = nll_criterion(scaled_logits, labels)
            loss.backward()
            return loss
        optimizer.step(closure)
        self.temperature.data = torch.clamp(self.temperature.data, min=1e-3, max=100.0)
        
        # raise NotImplementedError("Implement temperature calibration")


class EnsembleUncertainty:
    """
    Estimate uncertainty via ensemble of models.
    
    Train multiple models with different initializations/data splits.
    Uncertainty = variance across ensemble predictions.
    """
    
    def __init__(self, models: list):
        """
        Args:
            models: List of trained models (same architecture)
        """
        self.models = models
        self.num_models = len(models)
    
    def predict_with_uncertainty(
        self,
        inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions and uncertainty from ensemble.
        
        Args:
            inputs: Model inputs
            
        Returns:
            mean_predictions: (batch_size, num_classes) - average prediction
            uncertainty: (batch_size,) - prediction variance
        """
        # TODO: Implement ensemble prediction
        # Steps:
        #   1. Get predictions from all models
        all_preds = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(inputs)  # (B, C)
                probs = torch.softmax(logits, dim=-1)  # convert to probability space
                all_preds.append(probs)
        all_preds = torch.stack(all_preds, dim=0)  # (num_models, B, C)
        #   2. Compute mean prediction
        mean_predictions = all_preds.mean(dim=0)  # (B, C)
        #   3. Compute variance as uncertainty measure
        variance = all_preds.var(dim=0)  # (B, C)
        uncertainty = variance.mean(dim=1)  # (B,)
        #   4. Return mean and uncertainty
        return mean_predictions, uncertainty
        
        # raise NotImplementedError("Implement ensemble uncertainty")


def compute_calibration_metrics(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Compute all calibration metrics on a dataset.
    
    Args:
        model: Trained model
        dataloader: Test/validation dataloader
        device: Device to run on
        
    Returns:
        metrics: Dict with ECE, MCE, NLL, accuracy
    """
    model.eval()
    all_confidences = []
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
            
            all_confidences.append(confidences.cpu())
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
    
    confidences = torch.cat(all_confidences)
    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)
    
    # : Compute and return all metrics
    ece = CalibrationMetrics.expected_calibration_error(confidences, predictions, labels)
    mce = CalibrationMetrics.maximum_calibration_error(confidences, predictions, labels)
    nll = CalibrationMetrics.negative_log_likelihood(logits, labels)
    acc = (predictions == labels).float().mean().item()
    
    metrics = {
        'ECE': ece,
        'MCE': mce,
        'NLL': nll,
        'Accuracy': acc
    }

    return metrics
    raise NotImplementedError("Implement calibration metrics computation")


if __name__ == '__main__':
    # Test calibration metrics
    print("Testing calibration metrics...")
    
    # Generate fake predictions
    num_samples = 1000
    num_classes = 10
    
    # Well-calibrated predictions
    logits = torch.randn(num_samples, num_classes)
    labels = torch.randint(0, num_classes, (num_samples,))
    probs = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(probs, dim=1)
    
    # Test ECE
    try:
        ece = CalibrationMetrics.expected_calibration_error(
            confidences, predictions, labels
        )
        print(f"✓ ECE computed: {ece:.4f}")
    except NotImplementedError:
        print("✗ ECE not implemented yet")
    
    # Test reliability diagram
    try:
        CalibrationMetrics.reliability_diagram(
            confidences.numpy(),
            predictions.numpy(),
            labels.numpy(),
            save_path='test_reliability.png'
        )
        print("✓ Reliability diagram created")
    except NotImplementedError:
        print("✗ Reliability diagram not implemented yet")

