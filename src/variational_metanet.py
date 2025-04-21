"""
Variational MetaNet implementation for precomputed features.

This module implements a Bayesian approach to task vector combination using
variational inference. It models the uncertainty in composition coefficients
by generating both mean and variance parameters of the posterior distribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from tqdm import tqdm

class VariationalMetaNet(nn.Module):
    """MetaNet with proper variational inference for task vector composition."""

    def __init__(self,
                 feature_dim,
                 task_vectors,
                 blockwise=False,
                 base_threshold=0.05,
                 beta=1.0,
                 uncertainty_reg=0.01,
                 reg_coefficient=0.001,
                 margin_weight=0.0001,
                 kl_weight=0.1,
                 num_samples=1,
                 gating_enabled=True):
        """Initialize VariationalMetaNet.

        Parameters:
        ----------
        feature_dim: int
            Dimension of the pre-computed feature vectors
        task_vectors: int or list
            Number of task vectors or list of task vectors
        blockwise: bool
            Whether to use different coefficients for each parameter block
        base_threshold: float
            Initial value for base threshold τ₀
        beta: float
            Initial value for sensitivity parameter β
        uncertainty_reg: float
            Weight for uncertainty regularization in loss function
        reg_coefficient: float
            Regularization coefficient for beta and threshold
        margin_weight: float
            Weight for margin loss
        kl_weight: float
            Weight for KL divergence in ELBO
        num_samples: int
            Number of samples to draw from the posterior during training
        gating_enabled: bool
            Whether to use the gating mechanism
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.blockwise = blockwise
        self.uncertainty_reg = uncertainty_reg
        self.reg_coefficient = reg_coefficient
        self.margin_weight = margin_weight
        self.kl_weight = kl_weight
        self.num_samples = num_samples
        self.gating_enabled = gating_enabled

        # Initialize learnable gating parameters
        self.log_base_threshold = nn.Parameter(torch.tensor([math.log(max(base_threshold, 1e-5))], dtype=torch.float))
        self.log_beta = nn.Parameter(torch.tensor([math.log(max(beta * 0.95, 1e-5))], dtype=torch.float))

        # Register buffers for monitoring
        self.register_buffer('initial_base_threshold', torch.tensor([base_threshold], dtype=torch.float))
        self.register_buffer('initial_beta', torch.tensor([beta], dtype=torch.float))

        # Handle task vectors input
        if isinstance(task_vectors, int):
            self.num_task_vectors = task_vectors
        else:
            self.task_vectors = task_vectors
            self.num_task_vectors = len(task_vectors)

        # Default block number (will be determined dynamically during forward pass)
        self.num_blocks = 96  # Uses a large default value (safe for most ViT models)

        # Initialize networks for mean and log_variance prediction
        if blockwise:
            # Create networks with larger output dimensions to handle various block counts
            self.mean_net = self._build_inference_network(feature_dim, self.num_task_vectors * self.num_blocks)
            self.logvar_net = self._build_inference_network(feature_dim, self.num_task_vectors * self.num_blocks)
        else:
            self.mean_net = self._build_inference_network(feature_dim, self.num_task_vectors)
            self.logvar_net = self._build_inference_network(feature_dim, self.num_task_vectors)

        # For feature-based transformation
        self.task_features = nn.ParameterList([
            nn.Parameter(torch.randn(feature_dim, feature_dim) * 0.01, requires_grad=True)
            for _ in range(self.num_task_vectors)
        ])

        # Initialize a projection layer to transform task vectors
        self.projection = nn.Linear(feature_dim, feature_dim, bias=False)
        nn.init.eye_(self.projection.weight)  # Initialize as identity

        # Storage for computed values during forward pass
        self.last_means = None
        self.last_logvars = None
        self.last_samples = None
        self.last_uncertainties = None
        self.last_gated_samples = None
        self.last_thresholds = None
        self.last_binary_mask = None

        # Tracking variables
        self._forward_count = 0
        self._reg_loss_count = 0
        self.training_mode = True

    def _build_inference_network(self, input_dim, output_dim):
        """Build inference network for mean or variance prediction.

        Parameters:
        ----------
        input_dim: int
            Input dimension (feature size)
        output_dim: int
            Output dimension (number of coefficients)

        Returns:
        ----------
        network: nn.Sequential
            Neural network for inference
        """
        hidden_dim = max(input_dim // 4, output_dim)

        network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Initialize with small random values
        nn.init.normal_(network[0].weight, mean=0.0, std=0.01)
        nn.init.normal_(network[0].bias, mean=0.0, std=0.01)
        nn.init.normal_(network[2].weight, mean=0.0, std=0.01)
        nn.init.normal_(network[2].bias, mean=0.0, std=0.01)

        return network

    @property
    def base_threshold(self):
        """Get the actual base threshold value (always positive)."""
        return torch.exp(self.log_base_threshold)

    @property
    def beta(self):
        """Get the actual beta value (always positive)."""
        return torch.exp(self.log_beta)

    def compute_uncertainty(self, means, logvars):
        """Compute uncertainty based on variational posterior.

        Parameters:
        ----------
        means: Tensor [batch_size, num_coeffs]
            Mean parameters of the posterior
        logvars: Tensor [batch_size, num_coeffs]
            Log variance parameters of the posterior

        Returns:
        ----------
        uncertainties: Tensor [batch_size, num_coeffs]
            Uncertainty scores for each coefficient
        """
        # Convert log variance to standard deviation
        stdevs = torch.exp(0.5 * logvars)

        # Coefficient of variation (relative uncertainty)
        rel_uncertainty = stdevs / (torch.abs(means) + 1e-8)

        # Scale to [0, 1] range
        normalized_uncertainty = torch.tanh(rel_uncertainty)

        return normalized_uncertainty

    def reparameterize(self, mean, logvar):
        """Sample from the variational posterior using reparameterization trick.

        Parameters:
        ----------
        mean: Tensor
            Mean parameters of the posterior
        logvar: Tensor
            Log variance parameters of the posterior

        Returns:
        ----------
        samples: Tensor
            Samples from the posterior distribution
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def adaptive_gating(self, samples, uncertainties):
        """Apply adaptive thresholding based on uncertainty.

        Parameters:
        ----------
        samples: Tensor
            Samples from the posterior distribution
        uncertainties: Tensor
            Uncertainty scores for each coefficient

        Returns:
        ----------
        gated_samples: Tensor
            Samples after applying adaptive gating
        thresholds: Tensor
            Computed thresholds for each coefficient
        """
        # Get base_threshold and beta from log-parameterized versions
        base_threshold = self.base_threshold
        beta_val = self.beta

        # Compute adaptive thresholds - higher uncertainty means higher threshold
        thresholds = base_threshold * (1.0 + beta_val * uncertainties)

        # Add coefficient-specific dynamic adjustment based on distribution statistics
        if self.training and hasattr(self, 'last_means') and self.last_means is not None:
            # Get batch size and structure from current samples
            batch_size, num_vectors, num_blocks = samples.shape

            # Dynamically update num_blocks based on current input
            if self.blockwise and hasattr(self, 'num_blocks'):
                self.num_blocks = num_blocks

            # Ensure dimensions match by reshaping to match threshold dimensions
            if self.blockwise:
                # Reshape means to match current shape (accounting for potential dimension mismatch)
                try:
                    # Try to reshape using the current structure
                    reshaped_means = self.last_means.reshape(-1, num_vectors, num_blocks)
                except RuntimeError:
                    # If reshape fails, use a dimension-agnostic approach
                    means_flat = self.last_means.reshape(self.last_means.size(0), -1)
                    means_per_vector = means_flat.size(1) // num_vectors
                    if means_per_vector >= num_blocks:
                        # Truncate extra dimensions
                        reshaped_means = means_flat[:, :num_vectors*num_blocks].reshape(-1, num_vectors, num_blocks)
                    else:
                        # Pad with zeros to match dimensions
                        padding = torch.zeros(means_flat.size(0), num_vectors*num_blocks - means_flat.size(1),
                                             device=means_flat.device)
                        padded_means = torch.cat([means_flat, padding], dim=1)
                        reshaped_means = padded_means.reshape(-1, num_vectors, num_blocks)

                # Calculate statistics correctly with the properly reshaped tensor
                mean_abs = torch.mean(torch.abs(reshaped_means), dim=0, keepdim=True)
                std_abs = torch.std(torch.abs(reshaped_means), dim=0, keepdim=True) + 1e-6

                # Ensure broadcasting works correctly
                if mean_abs.shape != thresholds.shape:
                    mean_abs = mean_abs.expand_as(thresholds)
                    std_abs = std_abs.expand_as(thresholds)
            else:
                # For non-blockwise, simpler reshape is sufficient
                mean_abs = torch.mean(torch.abs(self.last_means), dim=0, keepdim=True)
                std_abs = torch.std(torch.abs(self.last_means), dim=0, keepdim=True) + 1e-6

            # Apply scaled adjustment based on global statistics (with dimension check)
            if mean_abs.shape == thresholds.shape:
                thresholds = thresholds * (1.0 + 0.1 * (std_abs / (mean_abs + 1e-8)))

        # Gradually anneal thresholds during early training
        if hasattr(self, '_forward_count'):
            early_training_factor = max(1.0, 2.0 - self._forward_count / 1000.0)
            thresholds = thresholds * early_training_factor

        # Apply gating - use a smoother transition for better gradients
        if self.training:
            # Differentiable approximation during training
            sigmoid_scale = 20.0  # Steepness of the sigmoid
            gating_mask = torch.sigmoid(sigmoid_scale * (torch.abs(samples) - thresholds))
            gated_samples = samples * gating_mask

            # Store actual binary mask for statistics
            self.last_binary_mask = (torch.abs(samples) >= thresholds).float().detach()
        else:
            # Hard gating during evaluation for exact sparsity
            gating_mask = (torch.abs(samples) >= thresholds).float()
            gated_samples = samples * gating_mask

        return gated_samples, thresholds

    def forward(self, features, num_samples=None):
        """Forward pass with variational inference.

        Parameters:
        ----------
        features: Tensor [batch_size, feature_dim]
            Pre-computed feature vectors
        num_samples: int or None
            Number of samples to draw (defaults to self.num_samples)

        Returns:
        ----------
        output: Tensor [batch_size, feature_dim]
            Transformed feature vectors
        """
        self._forward_count += 1

        if num_samples is None:
            num_samples = self.num_samples if self.training else 1

        # Generate mean and log variance parameters
        means = self.mean_net(features)
        logvars = self.logvar_net(features)

        # Constrain log variance for numerical stability
        logvars = torch.clamp(logvars, min=-8.0, max=2.0)

        self.last_means = means.detach()
        self.last_logvars = logvars.detach()

        # Determine blocks dynamically from network output
        if self.blockwise:
            total_outputs = means.size(1)
            inferred_blocks = total_outputs // self.num_task_vectors
            self.num_blocks = min(inferred_blocks, self.num_blocks)  # Use smaller of inferred or existing

            # Reshape parameters using the inferred block count
            means = means[:, :self.num_task_vectors*self.num_blocks].reshape(-1, self.num_task_vectors, self.num_blocks)
            logvars = logvars[:, :self.num_task_vectors*self.num_blocks].reshape(-1, self.num_task_vectors, self.num_blocks)
        else:
            means = means.reshape(-1, self.num_task_vectors, 1)
            logvars = logvars.reshape(-1, self.num_task_vectors, 1)

        # Compute uncertainty for gating
        uncertainties = self.compute_uncertainty(means, logvars)
        self.last_uncertainties = uncertainties.detach()

        # Multiple samples if requested (MC sampling)
        batch_size = features.size(0)
        all_outputs = []

        for sample_idx in range(num_samples):
            # Sample from the variational posterior using reparameterization
            samples = self.reparameterize(means, logvars)

            # Apply adaptive gating if enabled
            if self.gating_enabled:
                gated_samples, thresholds = self.adaptive_gating(samples, uncertainties)
                self.last_gated_samples = gated_samples.detach()
                self.last_thresholds = thresholds.detach()
            else:
                # If gating is disabled, use samples directly
                gated_samples = samples
                self.last_gated_samples = samples.detach()

            # Average across blocks for blockwise mode
            if self.blockwise:
                coefficients = gated_samples.mean(dim=2)
            else:
                coefficients = gated_samples.squeeze(2)

            # Apply task vectors directly in feature space
            sample_outputs = []

            for i in range(batch_size):
                # Get coefficients for this sample
                sample_coeffs = coefficients[i]

                # Apply task vectors as feature transformations
                transformed = features[i].unsqueeze(0)

                for j, task_matrix in enumerate(self.task_features):
                    # Apply task vector with its coefficient
                    coeff = sample_coeffs[j]
                    task_effect = coeff * torch.matmul(transformed, task_matrix)
                    transformed = transformed + task_effect

                # Project back to original feature space
                transformed = self.projection(transformed)
                sample_outputs.append(transformed)

            # Combine batch outputs
            sample_output = torch.cat(sample_outputs, dim=0)
            all_outputs.append(sample_output)

        # Average predictions from multiple samples
        if num_samples > 1:
            output = torch.stack(all_outputs).mean(dim=0)
        else:
            output = all_outputs[0]

        return output

    def kl_divergence_loss(self):
        """Compute KL divergence loss for the variational posterior.

        Returns:
        ----------
        loss: Tensor
            KL divergence loss component of ELBO
        """
        if self.last_means is None or self.last_logvars is None:
            return torch.tensor(0.0, device=self.log_base_threshold.device)

        # Reshape if needed
        means = self.last_means
        logvars = self.last_logvars

        # Prior parameters (standard Gaussian with spike)
        prior_mean = torch.zeros_like(means)
        prior_logvar = torch.zeros_like(logvars)

        # Standard KL divergence between two Gaussians
        # KL(q||p) = 0.5 * (log(σ²_p/σ²_q) + (σ²_q + (μ_q - μ_p)²)/σ²_p - 1)
        kl_div = 0.5 * (
            prior_logvar - logvars +
            (torch.exp(logvars) + (means - prior_mean).pow(2)) / torch.exp(prior_logvar) -
            1.0
        )

        # If gating is enabled, account for Spike-and-Slab prior
        if self.gating_enabled and hasattr(self, 'last_binary_mask'):
            # Get proportion of non-zeroed coefficients (slab component)
            slab_proportion = self.last_binary_mask.mean()

            # Mixture weight for the spike component
            pi = 0.5  # Default 50% spike probability in prior

            # Adjust KL divergence for spike component
            # log(q(z)/p(z)) where p(z) is a mixture
            spike_penalty = -torch.log(pi + (1 - pi) * torch.exp(-kl_div) + 1e-10)

            # Apply penalty more heavily to near-threshold values
            if hasattr(self, 'last_thresholds'):
                threshold_distance = torch.abs(torch.abs(means) - self.last_thresholds.mean())
                threshold_factor = torch.exp(-5.0 * threshold_distance) + 0.5
                spike_penalty = spike_penalty * threshold_factor

            # Final KL with spike component
            kl_div = kl_div * slab_proportion + spike_penalty * (1 - slab_proportion)

        return kl_div.mean() * self.kl_weight

    def margin_loss(self):
        """Compute margin loss to encourage decisive gating.

        Returns:
        ----------
        loss: Tensor
            Margin loss to discourage values near threshold
        """
        if not self.gating_enabled or self.last_means is None or self.last_thresholds is None:
            return torch.tensor(0.0, device=self.log_base_threshold.device)

        # Calculate average threshold
        avg_threshold = self.last_thresholds.mean()

        # Define margin width (20% of threshold)
        margin_width = avg_threshold * 0.2

        # Calculate distance from threshold
        threshold_distance = torch.abs(torch.abs(self.last_means) - avg_threshold)

        # Penalty for being within margin
        in_margin = (threshold_distance < margin_width).float()

        # Higher penalty for values very close to threshold
        penalty_factor = (1.0 - threshold_distance / (margin_width + 1e-8)) * in_margin

        return (penalty_factor * self.margin_weight).mean()

    def parameter_exploration_loss(self):
        """Compute loss to encourage parameter exploration.

        Returns:
        ----------
        loss: Tensor
            Regularization loss to encourage parameter exploration
        """
        if not self.gating_enabled:
            return torch.tensor(0.0, device=self.log_base_threshold.device)

        # Get initial values
        init_beta = self.initial_beta.item()
        init_threshold = self.initial_base_threshold.item()

        # Calculate the distance from initial values
        beta_dist = torch.abs(self.beta - init_beta)
        threshold_dist = torch.abs(self.base_threshold - init_threshold)

        # Encourage parameters to move away from initialization
        beta_reg = -torch.log(beta_dist.clamp(min=1e-5)) * self.reg_coefficient
        threshold_reg = -torch.log(threshold_dist.clamp(min=1e-5)) * self.reg_coefficient

        return beta_reg + threshold_reg

    def uncertainty_regularization_loss(self):
        """Compute complete regularization loss.

        Returns:
        ----------
        loss: Tensor
            Combined regularization losses
        """
        self._reg_loss_count += 1

        # Combine all regularization components
        kl_loss = self.kl_divergence_loss()
        margin_loss = self.margin_loss()
        param_loss = self.parameter_exploration_loss()

        # Uncertainty regularization of active coefficients
        uncertainty_loss = torch.tensor(0.0, device=self.log_base_threshold.device)

        if self.gating_enabled and self.last_uncertainties is not None and self.last_gated_samples is not None:
            # Create mask for active (non-gated) coefficients
            active_mask = (self.last_gated_samples != 0).float()

            # Reshape for averaging if blockwise
            if self.blockwise:
                active_mask = active_mask.mean(dim=2)
                uncertainties = self.last_uncertainties.mean(dim=2)
            else:
                active_mask = active_mask.squeeze(2)
                uncertainties = self.last_uncertainties.squeeze(2)

            # Compute weighted uncertainty loss - only penalize non-zero coefficients
            uncertainty_loss = torch.sum(active_mask * uncertainties) * self.uncertainty_reg

        total_loss = kl_loss + margin_loss + param_loss + uncertainty_loss

        return total_loss

    def get_gating_stats(self):
        """Get statistics about the gating process for monitoring.

        Returns:
        ----------
        stats: dict
            Dictionary with gating statistics
        """
        # Handle evaluation mode differently
        if not self.training_mode or not self.gating_enabled:
            return {
                "gating_ratio": 0.0,
                "avg_threshold": self.base_threshold.item(),
                "avg_uncertainty": 0.0,
                "base_threshold": self.base_threshold.item(),
                "beta": self.beta.item(),
                "log_base_threshold": self.log_base_threshold.item(),
                "log_beta": self.log_beta.item(),
                "predictive_variance": 0.0,
                "forward_count": self._forward_count if hasattr(self, '_forward_count') else 0,
                "reg_loss_count": self._reg_loss_count if hasattr(self, '_reg_loss_count') else 0,
            }

        # Return training mode stats if available
        if self.last_gated_samples is None:
            return {
                "gating_ratio": 0.0,
                "avg_threshold": self.base_threshold.item(),
                "avg_uncertainty": 0.0,
                "base_threshold": self.base_threshold.item(),
                "beta": self.beta.item(),
                "log_base_threshold": self.log_base_threshold.item(),
                "log_beta": self.log_beta.item(),
                "predictive_variance": 0.0,
                "forward_count": self._forward_count if hasattr(self, '_forward_count') else 0,
                "reg_loss_count": self._reg_loss_count if hasattr(self, '_reg_loss_count') else 0,
            }

        # Calculate gating statistics
        total_coeffs = self.last_gated_samples.numel()
        nonzero_coeffs = (self.last_gated_samples != 0).sum().item()
        gating_ratio = nonzero_coeffs / total_coeffs if total_coeffs > 0 else 0.0

        avg_threshold = self.last_thresholds.mean().item() if self.last_thresholds is not None else 0.0
        avg_uncertainty = self.last_uncertainties.mean().item() if self.last_uncertainties is not None else 0.0

        # Calculate average predictive variance (derived from logvars)
        predictive_variance = torch.exp(self.last_logvars).mean().item() if self.last_logvars is not None else 0.0

        return {
            "gating_ratio": gating_ratio,
            "avg_threshold": avg_threshold,
            "avg_uncertainty": avg_uncertainty,
            "base_threshold": self.base_threshold.item(),
            "beta": self.beta.item(),
            "log_base_threshold": self.log_base_threshold.item(),
            "log_beta": self.log_beta.item(),
            "predictive_variance": predictive_variance,
            "forward_count": self._forward_count if hasattr(self, '_forward_count') else 0,
            "reg_loss_count": self._reg_loss_count if hasattr(self, '_reg_loss_count') else 0,
        }

    def get_posterior_stats(self, features):
        """Get detailed posterior statistics for a batch of features.

        Parameters:
        ----------
        features: Tensor [batch_size, feature_dim]
            Input features

        Returns:
        ----------
        stats: dict
            Dictionary with posterior distribution statistics
        """
        with torch.no_grad():
            # Generate mean and log variance parameters
            means = self.mean_net(features)
            logvars = self.logvar_net(features)

            # Constrain log variance for numerical stability
            logvars = torch.clamp(logvars, min=-8.0, max=2.0)

            # Convert to standard deviation for easier interpretation
            stdevs = torch.exp(0.5 * logvars)

            # Handle blockwise reshape the same way as in forward()
            if self.blockwise:
                # Infer block count from network output
                total_outputs = means.size(1)
                inferred_blocks = total_outputs // self.num_task_vectors
                used_blocks = min(inferred_blocks, self.num_blocks)

                # Reshape properly using detected dimensions
                means_shaped = means[:, :self.num_task_vectors*used_blocks].reshape(-1, self.num_task_vectors, used_blocks)
                logvars_shaped = logvars[:, :self.num_task_vectors*used_blocks].reshape(-1, self.num_task_vectors, used_blocks)
                stdevs_shaped = stdevs[:, :self.num_task_vectors*used_blocks].reshape(-1, self.num_task_vectors, used_blocks)

                # Compute uncertainty on shaped tensors
                uncertainties = self.compute_uncertainty(means_shaped, logvars_shaped)

                # Compute coefficient of variation as a relative uncertainty measure
                coeff_variation = stdevs_shaped / (torch.abs(means_shaped) + 1e-8)
            else:
                # Simple reshape for non-blockwise case
                means_shaped = means.reshape(-1, self.num_task_vectors, 1)
                logvars_shaped = logvars.reshape(-1, self.num_task_vectors, 1)
                stdevs_shaped = stdevs.reshape(-1, self.num_task_vectors, 1)

                # Compute uncertainty
                uncertainties = self.compute_uncertainty(means_shaped, logvars_shaped)

                # Compute coefficient of variation
                coeff_variation = stdevs_shaped / (torch.abs(means_shaped) + 1e-8)

            # Gating probabilities
            if self.gating_enabled:
                base_threshold = self.base_threshold
                beta_val = self.beta

                thresholds = base_threshold * (1.0 + beta_val * uncertainties)

                # Probability of coefficient being active
                active_probs = 1.0 - torch.distributions.Normal(0, 1).cdf(
                    (thresholds - torch.abs(means_shaped)) / (stdevs_shaped + 1e-8)
                )
            else:
                thresholds = torch.zeros_like(means_shaped)
                active_probs = torch.ones_like(means_shaped)

            return {
                "means": means_shaped.detach().cpu(),
                "stdevs": stdevs_shaped.detach().cpu(),
                "coeff_var": coeff_variation.detach().cpu(),
                "uncertainties": uncertainties.detach().cpu(),
                "thresholds": thresholds.detach().cpu(),
                "active_probs": active_probs.detach().cpu(),
            }

    def monte_carlo_predictions(self, features, classifier, num_samples=10):
        """Get Monte Carlo predictions by sampling from the posterior.

        Parameters:
        ----------
        features: Tensor [batch_size, feature_dim]
            Input features
        classifier: nn.Module
            Classification head
        num_samples: int
            Number of Monte Carlo samples

        Returns:
        ----------
        dict:
            Dictionary with prediction statistics
        """
        batch_size = features.shape[0]

        with torch.no_grad():
            # Get predictions for multiple samples
            all_logits = []

            for _ in range(num_samples):
                # Forward pass with a single sample
                transformed_features = self.forward(features, num_samples=1)

                # Get classifier predictions
                logits = classifier(transformed_features)
                all_logits.append(logits)

            # Stack predictions
            all_logits = torch.stack(all_logits, dim=0)  # [num_samples, batch_size, num_classes]

            # Calculate mean prediction
            mean_logits = all_logits.mean(dim=0)  # [batch_size, num_classes]
            mean_probs = F.softmax(mean_logits, dim=1)

            # Calculate predictive entropy (total uncertainty)
            predictive_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)

            # Calculate aleatoric uncertainty (expected entropy)
            all_probs = F.softmax(all_logits, dim=2)  # [num_samples, batch_size, num_classes]
            sample_entropies = -torch.sum(all_probs * torch.log(all_probs + 1e-10), dim=2)  # [num_samples, batch_size]
            aleatoric_uncertainty = sample_entropies.mean(dim=0)  # [batch_size]

            # Calculate epistemic uncertainty (mutual information)
            epistemic_uncertainty = predictive_entropy - aleatoric_uncertainty

            # Get predicted class
            _, predicted = mean_logits.max(dim=1)

            return {
                "predictions": predicted.cpu(),
                "mean_probs": mean_probs.cpu(),
                "predictive_entropy": predictive_entropy.cpu(),
                "aleatoric_uncertainty": aleatoric_uncertainty.cpu(),
                "epistemic_uncertainty": epistemic_uncertainty.cpu(),
                "all_probs": all_probs.cpu(),
            }