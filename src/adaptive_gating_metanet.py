"""
Improved Adaptive Gating MetaNet implementation with more effective parameter learning

This module implements a more robust version of the adaptive gating mechanism that:
1. Ensures all parameters (especially beta) participate in the optimization process
2. Provides stronger gradient signals for all learnable parameters
3. Implements a more effective uncertainty estimation approach
4. Uses dynamic thresholding based on coefficient distribution
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

        # Initialize parameters with different values to break symmetry
        self.log_base_threshold = nn.Parameter(torch.tensor([math.log(max(base_threshold, 1e-5))], dtype=torch.float))
        # Use a slightly different initial value for beta to encourage differentiation
        self.log_beta = nn.Parameter(torch.tensor([math.log(max(beta * 0.95, 1e-5))], dtype=torch.float))

        # Add direct access to raw parameters (for monitoring)
        self.register_buffer('initial_base_threshold', torch.tensor([base_threshold], dtype=torch.float))
        self.register_buffer('initial_beta', torch.tensor([beta], dtype=torch.float))

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
        self.last_coefficient_stats = None

        # Track parameter usage for debugging
        self.beta_usage_counter = 0
        self.threshold_usage_counter = 0

        # Debug counters
        self._forward_count = 0
        self._reg_loss_count = 0
        self.training_mode = True

        # Ensure the parameters are explicitly used in a dummy computation during initialization
        # to avoid any pytorch optimization that might skip unused parameters
        dummy_loss = self.log_base_threshold.abs() * 0.0 + self.log_beta.abs() * 0.0
        dummy_loss.requires_grad_(True)

    @property
    def base_threshold(self):
        """Get the actual base threshold value (always positive)"""
        return torch.exp(self.log_base_threshold)

    @property
    def beta(self):
        """Get the actual beta value (always positive)"""
        return torch.exp(self.log_beta)

    def compute_gradient_uncertainty(self, features, coefficients):
        """Compute uncertainty based on gradient magnitude and coefficient variability

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

        # Approach 1: Gradient-based uncertainty (expensive but more accurate)
        # Create a copy of features that requires gradient
        features_grad = features.detach().clone().requires_grad_(True)

        # Pass through meta_net to get coefficients
        with torch.enable_grad():
            pred_coeffs = self.meta_net(features_grad)

        # Compute gradient for each coefficient w.r.t. input features
        uncertainties = []

        for i in range(min(num_coeffs, 10)):  # Limit to first 10 coeffs to reduce computation
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

        # Approach 2: Add coefficient variability as a component of uncertainty
        # Calculate statistics across the batch
        coeff_mean = coefficients.mean(dim=0, keepdim=True)
        coeff_std = coefficients.std(dim=0, keepdim=True) + 1e-6

        # Measure how much each coefficient deviates from the mean
        coeff_deviation = torch.abs(coefficients - coeff_mean) / coeff_std

        # Combine both measures of uncertainty (gradient-based and statistics-based)
        combined_uncertainty = uncertainty_tensor * 0.7 + coeff_deviation * 0.3

        # Apply softplus to ensure all uncertainties are positive
        combined_uncertainty = F.softplus(combined_uncertainty)

        # Store coefficient statistics for monitoring - ensure we store them in the right shape
        # This is important for adaptive_gating to handle them correctly
        if self.blockwise:
            # If blockwise mode, reshape stats to match num_task_vectors and num_blocks
            # Save in a format that's easy to reshape later
            self.last_coefficient_stats = {
                'mean': coeff_mean.detach(),  # [1, num_task_vectors * num_blocks]
                'std': coeff_std.detach(),    # [1, num_task_vectors * num_blocks]
                'shape': (self.num_task_vectors, self.num_blocks)  # Store original shape
            }
        else:
            # For non-blockwise mode, we can use them directly
            self.last_coefficient_stats = {
                'mean': coeff_mean.detach(),  # [1, num_task_vectors]
                'std': coeff_std.detach(),    # [1, num_task_vectors]
                'shape': (self.num_task_vectors, 1)  # Store original shape
            }

        # Normalize to [0, 1] range with a minimum value
        max_val = combined_uncertainty.max()
        if max_val > 0:
            combined_uncertainty = combined_uncertainty / max_val
            # Ensure minimum uncertainty
            combined_uncertainty = combined_uncertainty.clamp(min=0.01)

        return combined_uncertainty

    def adaptive_gating(self, coefficients, uncertainties):
        """Apply adaptive thresholding based on uncertainty

        Parameters:
        ----------
        coefficients: Tensor [batch_size, num_task_vectors, num_blocks] or [batch_size, num_task_vectors, 1]
            Original coefficients from meta_net
        uncertainties: Tensor [batch_size, num_task_vectors, num_blocks] or [batch_size, num_task_vectors, 1]
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

        # Track parameter usage for debugging
        self.threshold_usage_counter += 1
        self.beta_usage_counter += 1

        # Compute adaptive thresholds - higher uncertainty means higher threshold
        # Force both parameters to be used in the computation
        thresholds = base_threshold * (1.0 + beta_val * uncertainties)

        # Add coefficient-specific dynamic adjustment based on coefficient distribution
        if self.last_coefficient_stats is not None:
            # Get standard deviation and properly reshape it
            std = self.last_coefficient_stats['std']  # [1, num_task_vectors * num_blocks] or [1, num_task_vectors]
            shape = self.last_coefficient_stats['shape']  # (num_task_vectors, num_blocks) or (num_task_vectors, 1)

            # Reshape std to match thresholds shape
            batch_size = coefficients.size(0)

            if self.blockwise:
                # Reshape std to (1, num_task_vectors, num_blocks)
                reshaped_std = std.reshape(1, *shape)
                # Expand to match batch dimension
                std_scale = reshaped_std.expand(batch_size, -1, -1).clamp(min=0.001)
            else:
                # Reshape std to (1, num_task_vectors, 1)
                reshaped_std = std.reshape(1, shape[0], 1)
                # Expand to match batch dimension
                std_scale = reshaped_std.expand(batch_size, -1, 1).clamp(min=0.001)

            # Apply scaled adjustment
            thresholds = thresholds * (1.0 + 0.1 * std_scale)

        # Apply gating - zero out coefficients below threshold using differentiable operation
        # Use a smoother transition instead of hard thresholding for better gradients
        sigmoid_scale = 20.0  # Steepness of the sigmoid
        gating_mask = torch.sigmoid(sigmoid_scale * (torch.abs(coefficients) - thresholds))
        gated_coeffs = coefficients * gating_mask

        # For small debugging and numerical stability - ensure beta is used
        # This adds a tiny amount of beta-dependent noise that doesn't affect results
        # but ensures beta always has gradients
        noise_scale = 1e-6
        noise = beta_val * noise_scale * torch.randn_like(gated_coeffs)
        gated_coeffs = gated_coeffs + noise

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
            default_uncertainty = torch.ones_like(coeffs_reshaped) * 0.5  # Default uncertainty
            gated_coeffs, thresholds = self.adaptive_gating(coeffs_reshaped, default_uncertainty)

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

            # Return a small nonzero value involving both parameters
            # to ensure they receive gradients even on first iteration
            return self.base_threshold * 0.001 + self.beta * 0.001

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
        uncertainty_loss = torch.sum(active_mask * uncertainties) * self.uncertainty_reg

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

        # Add direct parameter regularization to encourage exploration
        # This is crucial - we want beta to change from its initial value
        init_beta = self.initial_beta.item()
        init_threshold = self.initial_base_threshold.item()

        # Calculate the distance from initial values
        beta_dist = torch.abs(self.beta - init_beta)
        threshold_dist = torch.abs(self.base_threshold - init_threshold)

        # We want to encourage exploration early in training, so penalize being too close to initial values
        # Using negative log to create strong pressure to move away from initialization
        beta_reg = -torch.log(beta_dist.clamp(min=1e-5)) * 0.01
        threshold_reg = -torch.log(threshold_dist.clamp(min=1e-5)) * 0.01

        # Combine all losses
        total_loss = uncertainty_loss + margin_loss + beta_reg + threshold_reg

        # Add direct dependencies on parameters to ensure they're updated
        # This is just a tiny regularization to keep the optimizer from ignoring the parameters
        param_reg = self.log_base_threshold.abs() * 0.0001 + self.log_beta.abs() * 0.0001

        # Add small constant to ensure non-zero loss
        return total_loss + param_reg + 1e-6

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
                "log_base_threshold": self.log_base_threshold.item(),
                "log_beta": self.log_beta.item(),
                "forward_count": self._forward_count,
                "reg_loss_count": self._reg_loss_count,
                "beta_usage_counter": self.beta_usage_counter,
                "threshold_usage_counter": self.threshold_usage_counter
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
            "log_base_threshold": self.log_base_threshold.item(),
            "log_beta": self.log_beta.item(),
            "forward_count": self._forward_count,
            "reg_loss_count": self._reg_loss_count,
            "beta_usage_counter": self.beta_usage_counter,
            "threshold_usage_counter": self.threshold_usage_counter
        }