"""MetaNet Architecture for Pre-computed Features

This module implements a modified version of MetaNet that works with
pre-computed features directly instead of processing images through
CLIP encoders, significantly accelerating training and inference.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
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


class PrecomputedMetaNet(nn.Module):
    """MetaNet for pre-computed features that directly applies task vectors
    to pre-computed features rather than re-encoding images
    """

    def __init__(self, feature_dim, task_vectors, blockwise=False, enable_causal=False, top_k_ratio=0.1):
        """Initialize PrecomputedMetaNet

        Parameters:
        ----------
        feature_dim: int
            Dimension of the pre-computed feature vectors
        task_vectors: List of task vectors or int
            List of task vectors for composition or int specifying number of task vectors
        blockwise: bool
            Whether to use different coefficients for each parameter block
        enable_causal: bool
            Whether to enable causal intervention
        top_k_ratio: float
            Ratio of top blocks to use for intervention
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.blockwise = blockwise
        self.enable_causal = enable_causal
        self.top_k_ratio = top_k_ratio

        # Adaptive gating parameters
        self.use_gating = False
        self.gating_threshold = 0.0
        self.sampling_std = 0.01
        self.num_samples = 1
        self.inference_mode = False

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

        self.printed_selection_info = False

        # For storing uncertainty information during inference
        self.last_prediction_variance = None

    def set_inference_mode(self, mode=True, gating_params=None):
        """Set model to inference mode with adaptive gating parameters

        Parameters:
        ----------
        mode: bool
            Whether to enable inference mode
        gating_params: dict, optional
            Dictionary with gating parameters:
            - use_gating: Whether to use the gating mechanism
            - gating_threshold: Threshold for gating mechanism (tau)
            - sampling_std: Standard deviation for sampling noise
            - num_samples: Number of samples for uncertainty estimation
        """
        self.inference_mode = mode

        if gating_params is not None:
            self.use_gating = gating_params.get('use_gating', False)
            self.gating_threshold = gating_params.get('gating_threshold', 0.0)
            self.sampling_std = gating_params.get('sampling_std', 0.01)
            self.num_samples = gating_params.get('num_samples', 1)

    def apply_gating(self, coefficients):
        """Apply gating mechanism to coefficients

        Parameters:
        ----------
        coefficients: Tensor
            Task vector coefficients

        Returns:
        ----------
        gated_coefficients: Tensor
            Coefficients after applying gating
        """
        if not self.use_gating:
            return coefficients

        # Apply gating: coefficients are kept only if >= threshold
        return torch.where(
            coefficients >= self.gating_threshold,
            coefficients,
            torch.zeros_like(coefficients)
        )

    def get_uncertainty(self):
        """Get uncertainty from last multi-sample prediction

        Returns:
        ----------
        uncertainty: float or None
            Variance across multiple predictions, or None if not available
        """
        return self.last_prediction_variance

    def _standard_forward(self, features, coefficients=None):
        """Standard forward pass implementation (without sampling)

        Parameters:
        ----------
        features: Tensor [batch_size, feature_dim]
            Pre-computed feature vectors
        coefficients: Tensor, optional
            Pre-computed coefficients (if None, they will be generated)

        Returns:
        ----------
        output: Tensor [batch_size, feature_dim]
            Transformed feature vectors
        """
        # Generate coefficients if not provided
        if coefficients is None:
            if self.blockwise:
                # Reshape to [batch_size, num_task_vectors, num_blocks]
                batch_coefficients = self.meta_net(features).reshape(
                    -1, self.num_task_vectors, self.num_blocks
                )

                # For causal models during evaluation, add a small noise to coefficients
                if self.enable_causal and not self.training:
                    # Use deterministic noise based on feature hash to ensure consistency
                    feature_hash = features.sum(dim=1, keepdim=True).detach()
                    noise_scale = 0.001  # Very small scale to not affect performance significantly
                    noise = (torch.sin(feature_hash * 1000) * noise_scale).reshape(-1, 1, 1)
                    batch_coefficients = batch_coefficients * (1.0 + noise)

                # Apply gating if in inference mode
                if self.inference_mode and self.use_gating:
                    batch_coefficients = self.apply_gating(batch_coefficients)

                # Average over blocks for simplicity in the pre-computed case
                coefficients = batch_coefficients.mean(dim=2)
            else:
                coefficients = self.meta_net(features)

                # For non-blockwise causal models
                if self.enable_causal and not self.training:
                    feature_hash = features.sum(dim=1, keepdim=True).detach()
                    noise_scale = 0.001
                    noise = (torch.sin(feature_hash * 1000) * noise_scale).reshape(-1, 1)
                    coefficients = coefficients * (1.0 + noise)

                # Apply gating if in inference mode
                if self.inference_mode and self.use_gating:
                    coefficients = self.apply_gating(coefficients)

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

    def forward(self, features):
        """Forward pass using pre-computed features

        Parameters:
        ----------
        features: Tensor [batch_size, feature_dim]
            Pre-computed feature vectors

        Returns:
        ----------
        output: Tensor [batch_size, feature_dim]
            Transformed feature vectors
        """
        # Reset uncertainty measurement at the start of each forward pass
        self.last_prediction_variance = None

        # If in inference mode with multiple samples, perform multi-sample prediction
        if self.inference_mode and self.num_samples > 1:
            return self.forward_with_sampling(features)

        # Standard forward pass (training or single-sample inference)
        return self._standard_forward(features)

    def forward_with_sampling(self, features):
        """Forward pass with multiple sampling for uncertainty estimation

        Parameters:
        ----------
        features: Tensor [batch_size, feature_dim]
            Pre-computed feature vectors

        Returns:
        ----------
        output: Tensor [batch_size, feature_dim]
            Transformed feature vectors (average across samples)
        """
        all_outputs = []

        # Generate base coefficients
        if self.blockwise:
            base_coefficients = self.meta_net(features).reshape(
                -1, self.num_task_vectors, self.num_blocks
            )
            # Mean over blocks to get shape [batch_size, num_task_vectors] for easier sampling
            if self.num_samples > 1:
                mean_coefficients = base_coefficients.mean(dim=2)
        else:
            base_coefficients = self.meta_net(features)
            mean_coefficients = base_coefficients

        # Perform multiple forward passes with different coefficient samples
        for sample_idx in range(self.num_samples):
            # First sample uses the original coefficients
            if sample_idx == 0:
                if self.blockwise:
                    # For blockwise, use mean over blocks
                    sampled_coefficients = mean_coefficients
                else:
                    sampled_coefficients = base_coefficients
            else:
                # Add random noise for subsequent samples
                noise = torch.randn_like(mean_coefficients) * self.sampling_std
                sampled_coefficients = mean_coefficients + noise

            # Apply gating to the sampled coefficients
            if self.use_gating:
                sampled_coefficients = self.apply_gating(sampled_coefficients)

            # Perform forward pass with the sampled coefficients
            outputs = self._standard_forward(features, sampled_coefficients)
            all_outputs.append(outputs)

        # Stack all outputs: [num_samples, batch_size, feature_dim]
        stacked_outputs = torch.stack(all_outputs)

        # Calculate variance across samples for uncertainty estimation
        if self.num_samples > 1:
            # Compute variance across the sample dimension
            self.last_prediction_variance = torch.var(stacked_outputs, dim=0).mean().item()

        # Return the average prediction
        return torch.mean(stacked_outputs, dim=0)

    def compute_intervention_loss(self, features):
        """Compute causal intervention loss for pre-computed features

        Parameters:
        ----------
        features: Tensor [batch_size, feature_dim]
            Pre-computed feature vectors

        Returns:
        ----------
        loss: Tensor
            Intervention loss
        """
        if not self.enable_causal or not self.blockwise:
            return torch.tensor(0.0, device=features.device)

        # Generate coefficients
        batch_coefficients = self.meta_net(features).reshape(
            -1, self.num_task_vectors, self.num_blocks
        )

        # Select top-k blocks based on coefficient magnitude
        avg_coef_magnitude = batch_coefficients.abs().mean(dim=(0, 1))
        k = max(1, int(self.num_blocks * self.top_k_ratio))
        _, top_k_indices = torch.topk(avg_coef_magnitude, k)

        # Only print selection info once
        if not self.printed_selection_info and is_main_process():
            print(f"Selected {k} out of {self.num_blocks} blocks for intervention")
            self.printed_selection_info = True

        # Compute regular outputs
        regular_outputs = self.forward(features)

        # Compute intervention effects
        total_variance = 0.0

        # Use tqdm only on main process
        if is_main_process():
            progress_iter = tqdm(top_k_indices, desc="Computing intervention effects", leave=False)
        else:
            progress_iter = top_k_indices

        for j in progress_iter:
            intervention_diffs = []

            for i in range(features.size(0)):
                # Create a modified coefficient tensor with zeroed block j
                modified_coeffs = batch_coefficients[i].clone()
                modified_coeffs[:, j] = 0.0

                # Compute output with this intervention
                modified_output = self._forward_with_coeffs(features[i:i + 1], modified_coeffs)

                # Compute squared difference
                diff = torch.sum((regular_outputs[i] - modified_output[0]) ** 2)
                intervention_diffs.append(diff)

            if intervention_diffs:
                total_variance += torch.stack(intervention_diffs).mean()

        return total_variance

    def _forward_with_coeffs(self, features, coefficients):
        """Forward pass with specific coefficients

        Parameters:
        ----------
        features: Tensor [batch_size, feature_dim]
            Pre-computed feature vectors
        coefficients: Tensor [num_task_vectors, num_blocks]
            Pre-defined coefficients

        Returns:
        ----------
        output: Tensor [batch_size, feature_dim]
            Transformed feature vectors
        """
        # Average over blocks
        avg_coeffs = coefficients.mean(dim=1)

        # Apply task vectors
        transformed = features.clone()

        for j, task_matrix in enumerate(self.task_features):
            coeff = avg_coeffs[j]
            task_effect = coeff * torch.matmul(transformed, task_matrix)
            transformed = transformed + task_effect

        # Project back
        return self.projection(transformed)

    def manual_intervention(self, features, block_idx=None, zero_blocks=None):
        """Perform manual intervention by zeroing specific blocks

        Parameters:
        ----------
        features: Tensor [batch_size, feature_dim]
            Pre-computed feature vectors
        block_idx: int or list, optional
            Index or indices of blocks to zero out
        zero_blocks: list, optional
            List of block indices to zero out (alternative format)

        Returns:
        ----------
        output: Tensor [batch_size, feature_dim]
            Transformed feature vectors after intervention
        """
        # Combine block indices from both parameters
        blocks_to_zero = []
        if block_idx is not None:
            if isinstance(block_idx, (list, tuple)):
                blocks_to_zero.extend(block_idx)
            else:
                blocks_to_zero.append(block_idx)

        if zero_blocks is not None:
            blocks_to_zero.extend(zero_blocks)

        # Remove duplicates
        blocks_to_zero = list(set(blocks_to_zero))

        if not blocks_to_zero:
            return self.forward(features)

        # Generate coefficients
        if self.blockwise:
            batch_coefficients = self.meta_net(features).reshape(
                -1, self.num_task_vectors, self.num_blocks
            )

            # Zero out specified blocks
            for j in blocks_to_zero:
                if j < self.num_blocks:
                    batch_coefficients[:, :, j] = 0.0

            # Apply gating if needed
            if self.use_gating:
                batch_coefficients = self.apply_gating(batch_coefficients)

            # Perform forward pass with modified coefficients
            batch_size = features.size(0)
            outputs = []

            for i in range(batch_size):
                # Average over blocks
                sample_coeffs = batch_coefficients[i].mean(dim=1)

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
        else:
            # Non-blockwise model doesn't support block intervention
            print("Warning: Block intervention not supported for non-blockwise model")
            return self.forward(features)