"""
Spike-and-Slab Variational MetaNet implementation for precomputed features.

This module implements a Bayesian approach to task vector combination using
variational inference with a Spike-and-Slab prior. It models the uncertainty
in composition coefficients using amortized inference networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from tqdm import tqdm

class SpikeAndSlabMetaNet(nn.Module):
    """MetaNet with Spike-and-Slab variational inference for task vector composition."""

    def __init__(self,
                 feature_dim,
                 task_vectors,
                 blockwise=False,
                 kl_weight=0.1,
                 num_samples=1,
                 prior_pi=0.5,
                 prior_sigma=1.0,
                 temperature=0.1):
        """Initialize SpikeAndSlabMetaNet.

        Parameters:
        ----------
        feature_dim: int
            Dimension of the pre-computed feature vectors
        task_vectors: int or list
            Number of task vectors or list of task vectors
        blockwise: bool
            Whether to use different coefficients for each parameter block
        kl_weight: float
            Weight for KL divergence in ELBO
        num_samples: int
            Number of samples to draw from the posterior during training
        prior_pi: float
            Prior probability for the slab component (sparsity control)
        prior_sigma: float
            Standard deviation for the slab component
        temperature: float
            Temperature parameter for Gumbel-Softmax sampling
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.blockwise = blockwise
        self.kl_weight = kl_weight
        self.num_samples = num_samples
        self.prior_pi = prior_pi
        self.prior_sigma = prior_sigma
        self.temperature = temperature

        # Handle task vectors input
        if isinstance(task_vectors, int):
            self.num_task_vectors = task_vectors
        else:
            self.task_vectors = task_vectors
            self.num_task_vectors = len(task_vectors)

        # Default block number (will be determined dynamically during forward pass)
        self.num_blocks = 96  # Uses a large default value (safe for most ViT models)

        # Initialize networks for mean, log_variance, and inclusion probability prediction
        if blockwise:
            # Create networks with larger output dimensions to handle various block counts
            self.mean_net = self._build_inference_network(feature_dim, self.num_task_vectors * self.num_blocks)
            self.logvar_net = self._build_inference_network(feature_dim, self.num_task_vectors * self.num_blocks)
            self.logit_net = self._build_inference_network(feature_dim, self.num_task_vectors * self.num_blocks)
        else:
            self.mean_net = self._build_inference_network(feature_dim, self.num_task_vectors)
            self.logvar_net = self._build_inference_network(feature_dim, self.num_task_vectors)
            self.logit_net = self._build_inference_network(feature_dim, self.num_task_vectors)

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
        self.last_logits = None
        self.last_samples = None
        self.last_binary_indicators = None

        # Tracking variables
        self._forward_count = 0
        self._kl_loss_count = 0
        self.training_mode = True

    def _build_inference_network(self, input_dim, output_dim):
        """Build inference network for posterior parameter prediction.

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

    def reparameterize_gaussian(self, mean, logvar):
        """Sample from Gaussian posterior using reparameterization trick.

        Parameters:
        ----------
        mean: Tensor
            Mean parameters of the posterior
        logvar: Tensor
            Log variance parameters of the posterior

        Returns:
        ----------
        samples: Tensor
            Samples from the Gaussian distribution
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def reparameterize_binary(self, logits, tau=1.0, hard=False):
        """Sample binary values using Gumbel-Softmax trick.

        Parameters:
        ----------
        logits: Tensor
            Logits for binary distribution
        tau: float
            Temperature parameter for Gumbel-Softmax
        hard: bool
            Whether to use hard sampling in forward pass

        Returns:
        ----------
        samples: Tensor
            Binary samples (or soft approximations)
        """
        # Create binary logits (for 0 and 1)
        logits_expanded = torch.stack([torch.zeros_like(logits), logits], dim=-1)

        # Apply Gumbel-Softmax
        y_soft = F.gumbel_softmax(logits_expanded, tau=tau, hard=hard, dim=-1)

        # Return probability of being 1
        return y_soft[..., 1]

    def forward(self, features, num_samples=None):
        """Forward pass with Spike-and-Slab variational inference.

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

        # Generate mean, log variance, and inclusion logit parameters
        means = self.mean_net(features)
        logvars = self.logvar_net(features)
        logits = self.logit_net(features)

        # Constrain log variance for numerical stability
        logvars = torch.clamp(logvars, min=-8.0, max=2.0)

        # Store for KL calculation
        self.last_means = means.detach()
        self.last_logvars = logvars.detach()
        self.last_logits = logits.detach()

        # Determine blocks dynamically from network output
        if self.blockwise:
            total_outputs = means.size(1)
            inferred_blocks = total_outputs // self.num_task_vectors
            self.num_blocks = min(inferred_blocks, self.num_blocks)  # Use smaller of inferred or existing

            # Reshape parameters using the inferred block count
            means = means[:, :self.num_task_vectors*self.num_blocks].reshape(-1, self.num_task_vectors, self.num_blocks)
            logvars = logvars[:, :self.num_task_vectors*self.num_blocks].reshape(-1, self.num_task_vectors, self.num_blocks)
            logits = logits[:, :self.num_task_vectors*self.num_blocks].reshape(-1, self.num_task_vectors, self.num_blocks)
        else:
            means = means.reshape(-1, self.num_task_vectors, 1)
            logvars = logvars.reshape(-1, self.num_task_vectors, 1)
            logits = logits.reshape(-1, self.num_task_vectors, 1)

        # Multiple samples if requested (MC sampling)
        batch_size = features.size(0)
        all_outputs = []
        all_binary_indicators = []

        for sample_idx in range(num_samples):
            # Sample from the Gaussian part of posterior using reparameterization
            gaussian_samples = self.reparameterize_gaussian(means, logvars)

            # Sample binary indicators using Gumbel-Softmax
            # Lower temperature during inference for more discrete samples
            current_temp = self.temperature if self.training else self.temperature * 0.1
            binary_indicators = self.reparameterize_binary(logits, tau=current_temp, hard=not self.training)

            # Store binary indicators for sparsity statistics
            all_binary_indicators.append(binary_indicators.detach())

            # Apply binary mask to create spike-and-slab samples
            samples = gaussian_samples * binary_indicators

            # Store samples for potential later use
            self.last_samples = samples.detach()
            self.last_binary_indicators = binary_indicators.detach()

            # Average across blocks for blockwise mode
            if self.blockwise:
                coefficients = samples.mean(dim=2)
            else:
                coefficients = samples.squeeze(2)

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
            self.last_binary_indicators = torch.stack(all_binary_indicators).mean(dim=0)
        else:
            output = all_outputs[0]
            self.last_binary_indicators = all_binary_indicators[0]

        return output

    def kl_divergence_loss(self):
        """Compute KL divergence loss for the Spike-and-Slab variational posterior.

        Returns:
        ----------
        loss: Tensor
            KL divergence loss component of ELBO
        """
        self._kl_loss_count += 1

        if self.last_means is None or self.last_logvars is None or self.last_logits is None:
            return torch.tensor(0.0, device=self.last_means.device if self.last_means is not None else 'cpu')

        # Reshape parameters for easier calculation
        means = self.last_means
        logvars = self.last_logvars
        logits = self.last_logits

        # Calculate posterior inclusion probability q(γ=1|x)
        q_gamma = torch.sigmoid(logits)

        # KL for the Bernoulli part (binary indicators)
        kl_bernoulli = q_gamma * torch.log(q_gamma / self.prior_pi + 1e-10) + \
                       (1 - q_gamma) * torch.log((1 - q_gamma) / (1 - self.prior_pi) + 1e-10)

        # KL for the Gaussian part (only matters when γ=1)
        # KL(q(w|γ=1)||p(w|γ=1)) = 0.5 * [log(σ²_p/σ²_q) + (σ²_q + μ²_q)/σ²_p - 1]
        kl_gaussian = 0.5 * (
            -logvars +
            (torch.exp(logvars) + means.pow(2)) / (self.prior_sigma**2) -
            1 +
            2 * math.log(self.prior_sigma)
        )

        # Total KL is the sum of Bernoulli KL and expected Gaussian KL (weighted by inclusion probability)
        kl_total = kl_bernoulli + q_gamma * kl_gaussian

        return kl_total.mean() * self.kl_weight

    def get_sparsity_stats(self):
        """Get sparsity statistics for monitoring.

        Returns:
        ----------
        stats: dict
            Dictionary with sparsity statistics
        """
        if self.last_binary_indicators is None:
            return {
                "sparsity_ratio": 0.0,
                "active_coefficient_ratio": 0.0,
                "posterior_inclusion_prob": 0.0,
                "prior_inclusion_prob": self.prior_pi,
                "prior_sigma": self.prior_sigma,
                "kl_weight": self.kl_weight,
                "temperature": self.temperature,
                "forward_count": self._forward_count if hasattr(self, '_forward_count') else 0,
                "kl_loss_count": self._kl_loss_count if hasattr(self, '_kl_loss_count') else 0,
            }

        # Calculate sparsity ratio (percentage of zeroed coefficients)
        if hasattr(self, 'last_binary_indicators'):
            active_indicators = (self.last_binary_indicators > 0.5).float()
            sparsity_ratio = 1.0 - active_indicators.mean().item()

            # Also calculate average value of continuous indicators
            posterior_inclusion_prob = self.last_binary_indicators.mean().item()
        else:
            sparsity_ratio = 0.0
            posterior_inclusion_prob = 0.0

        # For samples, calculate ratio of truly non-zero coefficients
        if hasattr(self, 'last_samples'):
            non_zero_coeffs = (torch.abs(self.last_samples) > 1e-6).float()
            active_coefficient_ratio = non_zero_coeffs.mean().item()
        else:
            active_coefficient_ratio = 0.0

        # Return comprehensive stats
        return {
            "sparsity_ratio": sparsity_ratio,
            "active_coefficient_ratio": active_coefficient_ratio,
            "posterior_inclusion_prob": posterior_inclusion_prob,
            "prior_inclusion_prob": self.prior_pi,
            "prior_sigma": self.prior_sigma,
            "kl_weight": self.kl_weight,
            "temperature": self.temperature,
            "forward_count": self._forward_count if hasattr(self, '_forward_count') else 0,
            "kl_loss_count": self._kl_loss_count if hasattr(self, '_kl_loss_count') else 0,
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
            # Generate posterior parameters
            means = self.mean_net(features)
            logvars = self.logvar_net(features)
            logits = self.logit_net(features)

            # Constrain log variance for numerical stability
            logvars = torch.clamp(logvars, min=-8.0, max=2.0)

            # Convert to standard deviation for easier interpretation
            stdevs = torch.exp(0.5 * logvars)

            # Calculate inclusion probabilities
            inclusion_probs = torch.sigmoid(logits)

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
                inclusion_probs_shaped = inclusion_probs[:, :self.num_task_vectors*used_blocks].reshape(-1, self.num_task_vectors, used_blocks)

                # Compute coefficient of variation as a relative uncertainty measure
                coeff_variation = stdevs_shaped / (torch.abs(means_shaped) + 1e-8)
            else:
                # Simple reshape for non-blockwise case
                means_shaped = means.reshape(-1, self.num_task_vectors, 1)
                logvars_shaped = logvars.reshape(-1, self.num_task_vectors, 1)
                stdevs_shaped = stdevs.reshape(-1, self.num_task_vectors, 1)
                inclusion_probs_shaped = inclusion_probs.reshape(-1, self.num_task_vectors, 1)

                # Compute coefficient of variation
                coeff_variation = stdevs_shaped / (torch.abs(means_shaped) + 1e-8)

            # Extract samples for visualization
            # Sample gaussian part
            gaussian_samples = self.reparameterize_gaussian(means_shaped, logvars_shaped)

            # Sample binary indicators (use hard=True for visualization)
            binary_indicators = self.reparameterize_binary(
                logits.reshape_as(means_shaped),
                tau=self.temperature * 0.1,
                hard=True
            )

            # Apply binary mask to create spike-and-slab samples
            samples = gaussian_samples * binary_indicators

            return {
                "means": means_shaped.detach().cpu(),
                "stdevs": stdevs_shaped.detach().cpu(),
                "coeff_var": coeff_variation.detach().cpu(),
                "inclusion_probs": inclusion_probs_shaped.detach().cpu(),
                "binary_indicators": binary_indicators.detach().cpu(),
                "samples": samples.detach().cpu(),
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
            all_binary_indicators = []

            for _ in range(num_samples):
                # Forward pass with a single sample
                transformed_features = self.forward(features, num_samples=1)

                # Store binary indicators for this sample
                if hasattr(self, 'last_binary_indicators'):
                    all_binary_indicators.append(self.last_binary_indicators)

                # Get classifier predictions
                logits = classifier(transformed_features)
                all_logits.append(logits)

            # Stack predictions
            all_logits = torch.stack(all_logits, dim=0)  # [num_samples, batch_size, num_classes]

            # Average binary indicators across samples if available
            if all_binary_indicators:
                binary_indicators = torch.stack(all_binary_indicators, dim=0)
                avg_binary_indicators = binary_indicators.mean(dim=0)
                sparsity_ratio = 1.0 - (binary_indicators > 0.5).float().mean().item()
            else:
                avg_binary_indicators = None
                sparsity_ratio = 0.0

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

            result = {
                "predictions": predicted.cpu(),
                "mean_probs": mean_probs.cpu(),
                "predictive_entropy": predictive_entropy.cpu(),
                "aleatoric_uncertainty": aleatoric_uncertainty.cpu(),
                "epistemic_uncertainty": epistemic_uncertainty.cpu(),
                "all_probs": all_probs.cpu(),
                "sparsity_ratio": sparsity_ratio,
            }

            if avg_binary_indicators is not None:
                result["binary_indicators"] = avg_binary_indicators.cpu()

            return result