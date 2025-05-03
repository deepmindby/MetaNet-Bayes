"""
Task-level Variational MetaNet implementation for precomputed features.

This module implements a Bayesian approach to task vector combination using
variational inference at the task level. Instead of amortized inference networks,
it directly learns the variational parameters for the entire task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class TaskVariationalMetaNet(nn.Module):
    """MetaNet with task-level variational inference for task vector composition."""

    def __init__(self,
                 feature_dim,
                 task_vectors,
                 blockwise=False,
                 kl_weight=0.1,
                 num_samples=1):
        """Initialize TaskVariationalMetaNet.

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
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.blockwise = blockwise
        self.kl_weight = kl_weight
        self.num_samples = num_samples

        # Handle task vectors input
        if isinstance(task_vectors, int):
            self.num_task_vectors = task_vectors
        else:
            self.task_vectors = task_vectors
            self.num_task_vectors = len(task_vectors)

        # Default block number for blockwise mode
        self.num_blocks = 96  # Uses a large default value (safe for most ViT models)

        # Initialize learnable variational parameters for the task
        if blockwise:
            # Create parameters for each task vector and each block
            self.mean_params = nn.Parameter(torch.zeros(self.num_task_vectors, self.num_blocks))
            self.logvar_params = nn.Parameter(torch.zeros(self.num_task_vectors, self.num_blocks))

            # Initialize with small values
            nn.init.normal_(self.mean_params, mean=0.0, std=0.01)
            nn.init.constant_(self.logvar_params, -3.0)  # Start with small variance
        else:
            # Create parameters for each task vector (global coefficients)
            self.mean_params = nn.Parameter(torch.zeros(self.num_task_vectors))
            self.logvar_params = nn.Parameter(torch.zeros(self.num_task_vectors))

            # Initialize with small values
            nn.init.normal_(self.mean_params, mean=0.0, std=0.01)
            nn.init.constant_(self.logvar_params, -3.0)  # Start with small variance

        # For feature-based transformation
        self.task_features = nn.ParameterList([
            nn.Parameter(torch.randn(feature_dim, feature_dim) * 0.01, requires_grad=True)
            for _ in range(self.num_task_vectors)
        ])

        # Initialize a projection layer to transform task vectors
        self.projection = nn.Linear(feature_dim, feature_dim, bias=False)
        nn.init.eye_(self.projection.weight)  # Initialize as identity

        # Storage for computed values during forward pass
        self.last_samples = None

        # Tracking variables
        self._forward_count = 0
        self._kl_loss_count = 0
        self.training_mode = True

    def reparameterize(self, mean, logvar):
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

    def forward(self, features, num_samples=None):
        """Forward pass with task-level variational inference.

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

        # Get the variational parameters
        mean = self.mean_params
        logvar = self.logvar_params

        # Constrain log variance for numerical stability
        logvar = torch.clamp(logvar, min=-8.0, max=2.0)

        # Multiple samples if requested (MC sampling)
        batch_size = features.size(0)
        all_outputs = []

        for sample_idx in range(num_samples):
            # Sample from the posterior using reparameterization
            if self.blockwise:
                # For blockwise, sample for each task vector and each block
                coefficients_samples = self.reparameterize(mean, logvar)
                # Average across blocks to get per-task-vector coefficients
                coefficients = coefficients_samples.mean(dim=1)
            else:
                # For global coefficients, sample directly
                coefficients = self.reparameterize(mean, logvar)

            # Store samples for potential later use
            self.last_samples = coefficients.detach()

            # Apply task vectors directly in feature space
            sample_outputs = []

            for i in range(batch_size):
                # Apply task vectors as feature transformations
                transformed = features[i].unsqueeze(0)

                for j, task_matrix in enumerate(self.task_features):
                    # Apply task vector with its coefficient
                    coeff = coefficients[j]
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
        self._kl_loss_count += 1

        # Get the variational parameters
        mean = self.mean_params
        logvar = self.logvar_params

        # Constrain log variance for numerical stability
        logvar = torch.clamp(logvar, min=-8.0, max=2.0)

        # KL divergence between the posterior q(z|x) and the prior p(z) = N(0, I)
        # KL(q(z|x) || p(z)) = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        return kl_div * self.kl_weight

    def get_variational_params(self):
        """Get the current variational parameters.

        Returns:
        ----------
        params: dict
            Dictionary with variational parameters
        """
        mean = self.mean_params.detach()
        logvar = self.logvar_params.detach().clamp(min=-8.0, max=2.0)
        std = torch.exp(0.5 * logvar)

        # Calculate coefficient of variation (relative uncertainty)
        coeff_var = std / (torch.abs(mean) + 1e-8)

        return {
            "mean": mean.cpu(),
            "std": std.cpu(),
            "coeff_var": coeff_var.cpu(),
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

            # Get current variational parameters
            params = self.get_variational_params()

            return {
                "predictions": predicted.cpu(),
                "mean_probs": mean_probs.cpu(),
                "predictive_entropy": predictive_entropy.cpu(),
                "aleatoric_uncertainty": aleatoric_uncertainty.cpu(),
                "epistemic_uncertainty": epistemic_uncertainty.cpu(),
                "all_probs": all_probs.cpu(),
                "variational_params": params,
            }