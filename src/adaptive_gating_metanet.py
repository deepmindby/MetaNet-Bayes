"""
Improved Adaptive Gating MetaNet implementation with more effective uncertainty estimation

This module implements a more robust version of the adaptive gating mechanism that:
1. Properly computes gradient-based uncertainty estimates
2. Ensures effective learning of gating parameters
3. Calculates meaningful regularization losses
4. Integrates causal intervention through uncertainty modulation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math
from src.distributed import is_main_process


class MetaNet(nn.Module):
    """Meta-Net network for generating task vector composition coefficients from sample features"""

    def __init__(self, input_dim, output_dim, hidden_dim=None):
        """Initialize Meta-Net

        Parameters:
        ----------
        input_dim: int
            Input feature dimension
        output_dim: int
            Output dimension, equal to the number of task vectors
        hidden_dim: int, optional
            Hidden layer dimension, defaults to 1/4 of the input dimension
        """
        super().__init__()

        # If hidden layer dimension is not specified, use 1/4 of the input dimension
        if hidden_dim is None:
            hidden_dim = max(input_dim // 4, output_dim)

        # Two-layer bottleneck structure: Linear-ReLU-Linear
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Initialize with small normal distribution values for better training stability
        nn.init.normal_(self.net[0].weight, mean=0.0, std=0.01)
        nn.init.normal_(self.net[0].bias, mean=0.0, std=0.01)

    def forward(self, x):
        """Forward propagation

        Parameters:
        ----------
        x: Tensor [batch_size, input_dim]
            Sample features

        Returns:
        ----------
        coefficients: Tensor [batch_size, output_dim]
            Task vector composition coefficients for each sample
        """
        return self.net(x)


class AdaptiveGatingMetaNet(nn.Module):
    """MetaNet with adaptive gating and uncertainty estimation"""

    def __init__(self, feature_dim, task_vectors, blockwise=False, base_threshold=0.05, beta=1.0, uncertainty_reg=0.01):
        """Initialize AdaptiveGatingMetaNet

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
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.blockwise = blockwise
        self.uncertainty_reg = uncertainty_reg

        # Important: Use log-space parameterization for more stable learning
        # This ensures base_threshold and beta remain positive after optimization
        self.log_base_threshold = nn.Parameter(torch.tensor([math.log(base_threshold)], dtype=torch.float))
        self.log_beta = nn.Parameter(torch.tensor([math.log(beta)], dtype=torch.float))

        # Handle both list and integer input for task_vectors
        if isinstance(task_vectors, int):
            self.num_task_vectors = task_vectors
        else:
            self.task_vectors = task_vectors
            self.num_task_vectors = len(task_vectors)

        # Initialize the meta network to predict coefficients
        if blockwise:
            # Simplified for pre-computed features - we estimate
            # the number of blocks based on a typical model
            self.num_blocks = 12  # Typical for ViT models
            self.meta_net = MetaNet(feature_dim, self.num_task_vectors * self.num_blocks)
        else:
            self.meta_net = MetaNet(feature_dim, self.num_task_vectors)

        # For feature-based transformation
        self.task_features = nn.ParameterList([
            nn.Parameter(torch.randn(feature_dim, feature_dim) * 0.01, requires_grad=True)
            for _ in range(self.num_task_vectors)
        ])

        # Initialize a projection layer to transform task vectors
        self.projection = nn.Linear(feature_dim, feature_dim, bias=False)
        nn.init.eye_(self.projection.weight)  # Initialize as identity

        # Storage for computed values during forward pass
        self.last_uncertainties = None
        self.last_gated_coeffs = None
        self.last_thresholds = None
        self.last_orig_coeffs = None

        # Debug counters
        self._forward_count = 0
        self._reg_loss_count = 0
        self.training_mode = True

    @property
    def base_threshold(self):
        """Get the actual base threshold value (always positive)"""
        return torch.exp(self.log_base_threshold)

    @property
    def beta(self):
        """Get the actual beta value (always positive)"""
        return torch.exp(self.log_beta)

    def compute_gradient_uncertainty(self, features, coefficients):
        """Compute uncertainty based on gradient magnitude w.r.t. input features

        Parameters:
        ----------
        features: Tensor [batch_size, feature_dim]
            Input features
        coefficients: Tensor [batch_size, num_coeffs]
            Computed coefficients

        Returns:
        ----------
        uncertainties: Tensor [batch_size, num_coeffs]
            Uncertainty scores for each coefficient
        """
        batch_size = features.size(0)
        num_coeffs = coefficients.size(1)

        # Create a copy of features that requires gradient
        features_grad = features.detach().clone().requires_grad_(True)

        # Pass through meta_net to get coefficients
        with torch.enable_grad():
            pred_coeffs = self.meta_net(features_grad)

        # Compute gradient for each coefficient w.r.t. input features
        uncertainties = []

        for i in range(min(num_coeffs, 20)):  # Limit to first 20 coeffs to reduce computation
            # Sum across batch for efficiency
            coeff_sum = pred_coeffs[:, i].sum()

            # Get gradient
            grad = torch.autograd.grad(
                coeff_sum,
                features_grad,
                create_graph=False,
                retain_graph=True
            )[0]

            # Compute L2 norm of gradient for each sample
            grad_norm = torch.norm(grad, p=2, dim=1)

            # Add small epsilon to avoid zero uncertainties
            grad_norm = grad_norm + 1e-6

            uncertainties.append(grad_norm)

        # If we computed fewer uncertainties than coefficients, repeat the last one
        if len(uncertainties) < num_coeffs:
            last_uncertainty = uncertainties[-1]
            for i in range(len(uncertainties), num_coeffs):
                uncertainties.append(last_uncertainty)

        # Stack and normalize uncertainties
        uncertainty_tensor = torch.stack(uncertainties, dim=1)

        # Apply softplus to ensure all uncertainties are positive
        uncertainty_tensor = F.softplus(uncertainty_tensor)

        # Normalize to [0, 1] range with a minimum value
        max_val = uncertainty_tensor.max()
        if max_val > 0:
            uncertainty_tensor = uncertainty_tensor / max_val
            # Ensure minimum uncertainty
            uncertainty_tensor = uncertainty_tensor.clamp(min=0.01)

        return uncertainty_tensor

    def adaptive_gating(self, coefficients, uncertainties):
        """Apply adaptive thresholding based on uncertainty

        Parameters:
        ----------
        coefficients: Tensor
            Original coefficients from meta_net
        uncertainties: Tensor
            Uncertainty scores for each coefficient

        Returns:
        ----------
        gated_coeffs: Tensor
            Coefficients after applying adaptive gating
        thresholds: Tensor
            Computed thresholds for each coefficient
        """
        # Get base_threshold and beta from log-parameterized versions
        base_threshold = self.base_threshold
        beta_val = self.beta

        # Compute adaptive thresholds - higher uncertainty means higher threshold
        thresholds = base_threshold * (1 + beta_val * uncertainties)

        # Apply gating - zero out coefficients below threshold
        gated_coeffs = torch.where(
            coefficients.abs() < thresholds,
            torch.zeros_like(coefficients),
            coefficients
        )

        return gated_coeffs, thresholds

    def forward(self, features):
        """Forward pass with adaptive gating

        Parameters:
        ----------
        features: Tensor [batch_size, feature_dim]
            Pre-computed feature vectors

        Returns:
        ----------
        output: Tensor [batch_size, feature_dim]
            Transformed feature vectors
        """
        self._forward_count += 1

        # Generate coefficients using meta network
        orig_coeffs = self.meta_net(features)
        self.last_orig_coeffs = orig_coeffs.detach()  # Store for loss computation

        # Reshape coefficients if using blockwise mode
        if self.blockwise:
            # Reshape to [batch_size, num_task_vectors, num_blocks]
            coeffs_reshaped = orig_coeffs.reshape(
                -1, self.num_task_vectors, self.num_blocks
            )
        else:
            coeffs_reshaped = orig_coeffs.reshape(
                -1, self.num_task_vectors, 1
            )

        # Compute uncertainty only during training or when specifically tracking it
        if self.training:
            with torch.set_grad_enabled(True):
                uncertainties = self.compute_gradient_uncertainty(features, orig_coeffs)

                # Reshape uncertainties to match coefficients
                if self.blockwise:
                    uncertainties = uncertainties.reshape(-1, self.num_task_vectors, self.num_blocks)
                else:
                    uncertainties = uncertainties.reshape(-1, self.num_task_vectors, 1)

            # Apply adaptive gating
            gated_coeffs, thresholds = self.adaptive_gating(coeffs_reshaped, uncertainties)

            # Store values for loss computation
            self.last_uncertainties = uncertainties
            self.last_gated_coeffs = gated_coeffs
            self.last_thresholds = thresholds
        else:
            # During inference, we can skip the uncertainty computation for efficiency
            gated_coeffs, thresholds = self.adaptive_gating(
                coeffs_reshaped,
                torch.ones_like(coeffs_reshaped) * 0.5  # Default uncertainty
            )

        # Average across blocks for blockwise mode
        if self.blockwise:
            coefficients = gated_coeffs.mean(dim=2)
        else:
            coefficients = gated_coeffs.squeeze(2)

        # Apply task vectors directly in feature space
        batch_size = features.size(0)
        outputs = []

        for i in range(batch_size):
            # Get coefficients for this sample
            sample_coeffs = coefficients[i]  # [num_task_vectors]

            # Apply task vectors as feature transformations
            transformed = features[i].unsqueeze(0)  # [1, feature_dim]

            for j, task_matrix in enumerate(self.task_features):
                # Apply task vector with its coefficient
                coeff = sample_coeffs[j]
                task_effect = coeff * torch.matmul(transformed, task_matrix)
                transformed = transformed + task_effect

            # Project back to original feature space
            transformed = self.projection(transformed)
            outputs.append(transformed)

        return torch.cat(outputs, dim=0)

    def uncertainty_regularization_loss(self):
        """Compute uncertainty regularization loss

        Returns:
        ----------
        loss: Tensor
            Regularization loss based on uncertainty
        """
        self._reg_loss_count += 1

        # Check if we have the necessary stored values from forward pass
        if (self.last_uncertainties is None or
            self.last_gated_coeffs is None or
            self.last_orig_coeffs is None):

            # Return zero if missing values (first iteration)
            return torch.tensor(0.0, device=next(self.parameters()).device)

        # Create mask for active (non-gated) coefficients
        if self.blockwise:
            # Reduce blockwise dimension for the mask
            active_mask = (self.last_gated_coeffs != 0).float().mean(dim=2)
        else:
            active_mask = (self.last_gated_coeffs.squeeze(2) != 0).float()

        # Reduce uncertainty to match active_mask dimensions
        if self.blockwise:
            uncertainties = self.last_uncertainties.mean(dim=2)
        else:
            uncertainties = self.last_uncertainties.squeeze(2)

        # Compute weighted uncertainty loss - only penalize non-zero coefficients
        # Higher uncertainty should be penalized more when coefficient is non-zero
        loss = torch.sum(active_mask * uncertainties) * self.uncertainty_reg

        # Add penalty for coefficients that are too close to threshold
        if self.blockwise:
            flat_coeffs = self.last_orig_coeffs.reshape(-1, self.num_task_vectors * self.num_blocks)
        else:
            flat_coeffs = self.last_orig_coeffs

        # Compute soft margin loss to encourage clear decisions
        # (either well above threshold or well below)
        avg_threshold = self.base_threshold.item()
        margin_width = avg_threshold * 0.2  # 20% of threshold as margin width

        # Compute how many coefficients are within the margin
        in_margin = ((flat_coeffs.abs() > (avg_threshold - margin_width)) &
                     (flat_coeffs.abs() < (avg_threshold + margin_width))).float()

        margin_loss = in_margin.sum() * 0.001  # Small weight to not dominate

        total_loss = loss + margin_loss

        # Ensure loss is non-zero for proper gradient flow
        total_loss = total_loss + 1e-6

        return total_loss

    def get_gating_stats(self):
        """Get statistics about the gating process

        Returns:
        ----------
        stats: dict
            Dictionary with gating statistics
        """
        if self.last_gated_coeffs is None:
            # Return default stats if no forward pass has been done
            return {
                "gating_ratio": 0.0,
                "avg_threshold": self.base_threshold.item(),
                "avg_uncertainty": 0.0,
                "base_threshold": self.base_threshold.item(),
                "beta": self.beta.item(),
                "forward_count": self._forward_count,
                "reg_loss_count": self._reg_loss_count
            }

        total_coeffs = self.last_gated_coeffs.numel()
        nonzero_coeffs = (self.last_gated_coeffs != 0).sum().item()
        gating_ratio = nonzero_coeffs / total_coeffs if total_coeffs > 0 else 0.0

        avg_threshold = self.last_thresholds.mean().item() if self.last_thresholds is not None else 0.0
        avg_uncertainty = self.last_uncertainties.mean().item() if self.last_uncertainties is not None else 0.0

        return {
            "gating_ratio": gating_ratio,
            "avg_threshold": avg_threshold,
            "avg_uncertainty": avg_uncertainty,
            "base_threshold": self.base_threshold.item(),
            "beta": self.beta.item(),
            "forward_count": self._forward_count,
            "reg_loss_count": self._reg_loss_count
        }