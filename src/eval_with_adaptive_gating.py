"""Evaluation Script for Adaptive Gating MetaNet Models

This script evaluates Adaptive Gating MetaNet models using pre-computed features,
with enhanced support for augmented data and detailed performance reporting.
Also supports evaluation of models trained without gating mechanism or MetaNet,
as well as Atlas models with direct gating mechanism.
"""

import os
import json
import torch
import traceback
import gc
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import numpy as np
import math

from src.args import parse_arguments
from src.adaptive_gating_metanet import AdaptiveGatingMetaNet


class DirectFeatureModel(torch.nn.Module):
    """
    A direct feature model that passes features directly to classifier without MetaNet transformations.
    This implements the Atlas approach using precomputed features.
    When gating_no_metanet is enabled, it applies gating mechanism directly to features.
    """
    def __init__(self, feature_dim, gating_no_metanet=False, base_threshold=0.05, beta=1.0, uncertainty_reg=0.01):
        """Initialize DirectFeatureModel

        Parameters:
        ----------
        feature_dim: int
            Dimension of the pre-computed feature vectors
        gating_no_metanet: bool
            Whether to apply gating mechanism directly to features
        base_threshold: float
            Base threshold for gating mechanism (only used if gating_no_metanet is True)
        beta: float
            Beta parameter for uncertainty weighting (only used if gating_no_metanet is True)
        uncertainty_reg: float
            Weight for uncertainty regularization (only used if gating_no_metanet is True)
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.gating_no_metanet = gating_no_metanet

        # Identity projection (no transformation)
        self.projection = torch.nn.Identity()

        # Add a dummy parameter with requires_grad=True to ensure DDP works
        # This parameter won't affect forward computation
        self.dummy_param = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

        # Small linear layer that won't affect output (scaled by zero)
        # This ensures we have trainable parameters for DDP
        self.dummy_linear = torch.nn.Linear(feature_dim, feature_dim)
        # Initialize to near-zero weights
        torch.nn.init.zeros_(self.dummy_linear.weight)
        torch.nn.init.zeros_(self.dummy_linear.bias)

        # Adding gating mechanism if enabled
        if self.gating_no_metanet:
            # Initialize learnable gating parameters
            self.log_base_threshold = torch.nn.Parameter(
                torch.tensor([math.log(max(base_threshold, 1e-5))], dtype=torch.float)
            )
            self.log_beta = torch.nn.Parameter(
                torch.tensor([math.log(max(beta * 0.95, 1e-5))], dtype=torch.float)
            )

            # Register buffers for monitoring
            self.register_buffer('initial_base_threshold', torch.tensor([base_threshold], dtype=torch.float))
            self.register_buffer('initial_beta', torch.tensor([beta], dtype=torch.float))

            # Uncertainty related variables
            self.uncertainty_reg = uncertainty_reg
            self._forward_count = 0
            self._reg_loss_count = 0
            self.training_mode = True

            # Storage for computed values during forward pass
            self.last_uncertainties = None
            self.last_gated_features = None
            self.last_thresholds = None
            self.last_orig_features = None
            self.last_gating_mask = None

            # Simple transform to generate feature-specific uncertainty
            self.uncertainty_net = torch.nn.Sequential(
                torch.nn.Linear(feature_dim, feature_dim // 4),
                torch.nn.ReLU(),
                torch.nn.Linear(feature_dim // 4, feature_dim),
                torch.nn.Sigmoid()
            )

    @property
    def base_threshold(self):
        """Get the actual base threshold value (always positive)"""
        if self.gating_no_metanet:
            return torch.exp(self.log_base_threshold)
        return torch.tensor(0.0)

    @property
    def beta(self):
        """Get the actual beta value (always positive)"""
        if self.gating_no_metanet:
            return torch.exp(self.log_beta)
        return torch.tensor(0.0)

    def compute_uncertainty(self, features):
        """Compute uncertainty based on feature characteristics

        Parameters:
        ----------
        features: Tensor [batch_size, feature_dim]
            Input features

        Returns:
        ----------
        uncertainties: Tensor [batch_size, feature_dim]
            Uncertainty scores for each feature dimension
        """
        batch_size = features.size(0)

        # Get feature-specific uncertainty using the network
        feature_uncertainty = self.uncertainty_net(features)

        # Add batch statistics component - how much each feature varies across the batch
        batch_std = features.std(dim=0, keepdim=True).expand(batch_size, -1)
        batch_std = batch_std / (batch_std.max() + 1e-8)  # Normalize

        # Combine components
        combined_uncertainty = 0.7 * feature_uncertainty + 0.3 * batch_std

        # Add a small random component to break symmetry during training
        # In evaluation, use a fixed small value for deterministic results
        if self.training_mode:
            random_noise = torch.rand_like(combined_uncertainty) * 0.1
        else:
            random_noise = torch.ones_like(combined_uncertainty) * 0.05

        combined_uncertainty = combined_uncertainty + random_noise

        # Normalize to [0, 1] range with a minimum value
        combined_uncertainty = combined_uncertainty.clamp(min=0.01, max=1.0)

        return combined_uncertainty

    def adaptive_gating(self, features, uncertainties):
        """Apply adaptive thresholding based on uncertainty

        Parameters:
        ----------
        features: Tensor [batch_size, feature_dim]
            Original features
        uncertainties: Tensor [batch_size, feature_dim]
            Uncertainty scores for each feature dimension

        Returns:
        ----------
        gated_features: Tensor [batch_size, feature_dim]
            Features after applying adaptive gating
        thresholds: Tensor [batch_size, feature_dim]
            Computed thresholds for each feature dimension
        """
        # Get base_threshold and beta from log-parameterized versions
        base_threshold = self.base_threshold
        beta_val = self.beta

        # Compute adaptive thresholds - higher uncertainty means higher threshold
        thresholds = base_threshold * (1.0 + beta_val * uncertainties)

        # Normalize features for gating
        feature_norms = torch.norm(features, dim=1, keepdim=True)
        normalized_features = features / (feature_norms + 1e-8)
        feature_magnitudes = torch.abs(normalized_features)

        # Apply gating using a smooth transition for better gradients
        sigmoid_scale = 20.0  # Steepness of the sigmoid
        gating_mask = torch.sigmoid(sigmoid_scale * (feature_magnitudes - thresholds))
        gated_features = features * gating_mask

        # Store the actual gating mask and thresholds for statistics
        self.last_gating_mask = (feature_magnitudes >= thresholds).float().detach()

        return gated_features, thresholds

    def forward(self, features):
        """Forward pass with optional adaptive gating

        Parameters:
        ----------
        features: Tensor [batch_size, feature_dim]
            Pre-computed feature vectors

        Returns:
        ----------
        features: Tensor [batch_size, feature_dim]
            Original or gated features
        """
        if self.gating_no_metanet:
            # Apply gating to features
            self._forward_count = getattr(self, '_forward_count', 0) + 1

            # Store original features
            self.last_orig_features = features.detach()

            # Compute uncertainty
            uncertainties = self.compute_uncertainty(features)
            self.last_uncertainties = uncertainties

            # Apply adaptive gating
            gated_features, thresholds = self.adaptive_gating(features, uncertainties)
            self.last_gated_features = gated_features
            self.last_thresholds = thresholds

            # Apply the dummy computation as before (zero-scaled)
            return gated_features + self.dummy_linear(features) * 0.0
        else:
            # Original behavior - identity projection + add scaled dummy computation (scaling factor = 0)
            return self.projection(features) + self.dummy_linear(features) * 0.0

    def uncertainty_regularization_loss(self):
        """Calculate regularization loss based on uncertainty and gating

        Returns:
        ----------
        loss: Tensor
            Regularization loss
        """
        if not self.gating_no_metanet or self.uncertainty_reg < 1e-8:
            # Return a small loss based on dummy parameter to ensure it receives gradient
            return self.dummy_param.sum() * 0.0  # Multiply by 0 to not affect actual training

        self._reg_loss_count = getattr(self, '_reg_loss_count', 0) + 1

        # Check if we have the necessary stored values from forward pass
        if (self.last_uncertainties is None or
                self.last_gated_features is None or
                self.last_orig_features is None):
            return self.base_threshold * 0.001 + self.beta * 0.001

        # Create mask for active (non-gated) features
        active_mask = (self.last_gated_features != 0).float()

        # Compute weighted uncertainty loss - only penalize non-zero features
        uncertainty_loss = torch.sum(active_mask * self.last_uncertainties) * self.uncertainty_reg

        # Add parameter regularization
        init_beta = self.initial_beta.item()
        init_threshold = self.initial_base_threshold.item()

        # Calculate the distance from initial values
        beta_dist = torch.abs(self.beta - init_beta)
        threshold_dist = torch.abs(self.base_threshold - init_threshold)

        # Encourage parameters to move away from initialization
        reg_coefficient = 0.001
        beta_reg = -torch.log(beta_dist.clamp(min=1e-5)) * reg_coefficient
        threshold_reg = -torch.log(threshold_dist.clamp(min=1e-5)) * reg_coefficient

        # Combine all losses
        total_loss = uncertainty_loss + beta_reg + threshold_reg

        return total_loss

    def get_gating_stats(self):
        """Get statistics about the gating process for monitoring

        Returns:
        ----------
        stats: dict
            Dictionary with gating statistics
        """
        if not self.gating_no_metanet:
            return {}

        # Calculate gating statistics
        if self.last_gated_features is None or self.last_gating_mask is None:
            # Generate sample data for stats if none available
            batch_size = 64
            features = torch.randn(batch_size, self.feature_dim, device=self.log_base_threshold.device)

            with torch.no_grad():
                uncertainties = self.compute_uncertainty(features)
                thresholds = self.base_threshold * (1.0 + self.beta * uncertainties)

                # Normalize features for gating mask calculation
                feature_norms = torch.norm(features, dim=1, keepdim=True)
                normalized_features = features / (feature_norms + 1e-8)
                feature_magnitudes = torch.abs(normalized_features)

                gating_mask = (feature_magnitudes >= thresholds).float()
                gating_ratio = gating_mask.mean().item()
        else:
            # Use stored values from forward pass
            gating_ratio = self.last_gating_mask.mean().item()

        # Get learned parameter values
        current_base_threshold = self.base_threshold.item()
        current_beta = self.beta.item()
        current_log_base_threshold = self.log_base_threshold.item()
        current_log_beta = self.log_beta.item()

        # Get initial parameter values
        initial_base_threshold = self.initial_base_threshold.item()
        initial_beta = self.initial_beta.item()

        # Calculate change from initial values
        threshold_change = ((current_base_threshold - initial_base_threshold) / initial_base_threshold) * 100
        beta_change = ((current_beta - initial_beta) / initial_beta) * 100

        # Get average threshold if available
        avg_threshold = self.last_thresholds.mean().item() if hasattr(self, 'last_thresholds') and self.last_thresholds is not None else current_base_threshold
        avg_uncertainty = self.last_uncertainties.mean().item() if hasattr(self, 'last_uncertainties') and self.last_uncertainties is not None else 0.0

        return {
            "gating_ratio": gating_ratio,
            "avg_threshold": avg_threshold,
            "avg_uncertainty": avg_uncertainty,
            "base_threshold": current_base_threshold,
            "beta": current_beta,
            "log_base_threshold": current_log_base_threshold,
            "log_beta": current_log_beta,
            "initial_base_threshold": initial_base_threshold,
            "initial_beta": initial_beta,
            "threshold_change_percent": threshold_change,
            "beta_change_percent": beta_change,
            "forward_count": self._forward_count if hasattr(self, '_forward_count') else 0,
            "reg_loss_count": self._reg_loss_count if hasattr(self, '_reg_loss_count') else 0,
        }


class PrecomputedFeatureDataset(torch.utils.data.Dataset):
    """Dataset for precomputed features with minimal logging"""

    def __init__(self, features_path, labels_path, verbose=False):
        """Initialize with paths to features and labels"""
        super().__init__()

        # Load features and labels
        if verbose:
            print(f"Loading features from {features_path}")
        self.features = torch.load(features_path)

        if verbose:
            print(f"Loading labels from {labels_path}")
        self.labels = torch.load(labels_path)

        if len(self.features) != len(self.labels):
            raise ValueError(f"Features ({len(self.features)}) and labels ({len(self.labels)}) count mismatch")

        if verbose:
            print(f"Loaded {len(self.features)} samples with feature dim {self.features.shape[1]}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "labels": self.labels[idx],
            "index": idx
        }


class TestOnlyFeatures:
    """Dataset container class for test-only pre-computed features"""

    def __init__(self, feature_dir, batch_size=128, num_workers=4, verbose=False):
        """Initialize with directory containing pre-computed features"""
        # Verify directory exists
        if not os.path.exists(feature_dir):
            raise FileNotFoundError(f"Feature directory not found: {feature_dir}")

        if verbose:
            print(f"Looking for test features in: {feature_dir}")

        # Define standard test feature paths
        test_features_path = os.path.join(feature_dir, "test_features.pt")
        test_labels_path = os.path.join(feature_dir, "test_labels.pt")

        # If test files don't exist, try val files
        if not os.path.exists(test_features_path):
            test_features_path = os.path.join(feature_dir, "val_features.pt")
            test_labels_path = os.path.join(feature_dir, "val_labels.pt")

            if not os.path.exists(test_features_path):
                raise FileNotFoundError(f"Could not find test or val features in {feature_dir}")

        if verbose:
            print(f"Using test features: {test_features_path}")
            print(f"Using test labels: {test_labels_path}")

        # Load test features and labels
        try:
            self.test_dataset = PrecomputedFeatureDataset(
                test_features_path,
                test_labels_path,
                verbose=verbose
            )

            # Create test loader
            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=min(num_workers, 2),  # Limit workers for stability
                pin_memory=True,
                drop_last=False,
                timeout=60,  # Add timeout to prevent hangs
            )

            if verbose:
                print(f"Created dataloader with {len(self.test_dataset)} samples")
        except Exception as e:
            print(f"Error loading test features/labels: {e}")
            traceback.print_exc()
            raise

        # Load classnames if available
        classnames_path = os.path.join(feature_dir, "classnames.txt")
        if os.path.exists(classnames_path):
            with open(classnames_path, "r") as f:
                self.classnames = [line.strip() for line in f.readlines()]
            if verbose:
                print(f"Loaded {len(self.classnames)} class names from {classnames_path}")
        else:
            # Create dummy classnames if file doesn't exist
            unique_labels = torch.unique(self.test_dataset.labels)
            self.classnames = [f"class_{i}" for i in range(len(unique_labels))]
            if verbose:
                print(f"Created {len(self.classnames)} dummy class names")


def cleanup_resources(dataset):
    """Cleanup data resources to prevent memory leaks"""
    if dataset is None:
        return

    try:
        # Clear dataset references
        if hasattr(dataset, 'test_loader') and dataset.test_loader is not None:
            dataset.test_loader = None
        if hasattr(dataset, 'test_dataset') and dataset.test_dataset is not None:
            dataset.test_dataset = None
    except Exception as e:
        print(f"Warning during dataset cleanup: {e}")

    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()


def find_model_path(model_dir, dataset_name, model_name, no_gating=False, no_metanet=False, gating_no_metanet=False, debug=False):
    """Find the model path for a given dataset, supporting different model types

    Args:
        model_dir: Directory containing trained models
        dataset_name: Name of the dataset
        model_name: Name of the model (e.g., ViT-B-32)
        no_gating: Whether to look for a model trained without gating
        no_metanet: Whether to look for a model trained without MetaNet
        gating_no_metanet: Whether to look for a model trained with gating on Atlas
        debug: Whether to print debug information

    Returns:
        str: Path to the model file
    """
    # Base name handling
    base_name = dataset_name
    if dataset_name.endswith("Val"):
        base_name = dataset_name[:-3]

    val_name = f"{base_name}Val"

    # Choose suffix based on model type
    if no_metanet:
        if gating_no_metanet:
            gating_suffix = "_atlas_gating"  # Atlas with gating
        else:
            gating_suffix = "_atlas"  # Standard Atlas
    elif no_gating:
        gating_suffix = "_no_gating"  # MetaNet without gating
    else:
        gating_suffix = "_adaptive_gating"  # Standard adaptive gating

    # Model-specific directory
    model_specific_dir = os.path.join(model_dir, model_name)

    # Check if model-specific directory exists
    if os.path.exists(model_specific_dir):
        model_dir = model_specific_dir
        if debug:
            print(f"Using model-specific directory: {model_dir}")

    # Check these paths in order with various naming conventions
    possible_paths = [
        # Model-specific paths with explicit naming
        os.path.join(model_dir, val_name, f"best{gating_suffix}_model.pt"),
        os.path.join(model_dir, val_name, f"best_{gating_suffix.strip('_')}_model.pt"),
        os.path.join(model_dir, base_name, f"best{gating_suffix}_model.pt"),
        os.path.join(model_dir, base_name, f"best_{gating_suffix.strip('_')}_model.pt"),

        # Atlas with gating may have specific naming
        os.path.join(model_dir, val_name, f"best_atlas_with_gating_model.pt"),
        os.path.join(model_dir, base_name, f"best_atlas_with_gating_model.pt"),
        os.path.join(model_dir, val_name, f"atlas_gating_model.pt"),
        os.path.join(model_dir, base_name, f"atlas_gating_model.pt"),
    ]

    # Add standard paths that might be used as fallbacks
    possible_paths.extend([
        os.path.join(model_dir, val_name, f"best_precomputed_model.pt"),
        os.path.join(model_dir, base_name, f"best_precomputed_model.pt"),
        os.path.join(model_dir, val_name, f"best_model.pt"),
        os.path.join(model_dir, base_name, f"best_model.pt"),
    ])

    # Also try in top-level directories (for backward compatibility)
    if model_dir != model_specific_dir:
        top_level_paths = [
            # Top-level directories
            os.path.join(model_specific_dir, val_name, f"best{gating_suffix}_model.pt"),
            os.path.join(model_specific_dir, val_name, f"best_{gating_suffix.strip('_')}_model.pt"),
            os.path.join(model_specific_dir, base_name, f"best{gating_suffix}_model.pt"),
            os.path.join(model_specific_dir, base_name, f"best_{gating_suffix.strip('_')}_model.pt"),
            os.path.join(model_specific_dir, val_name, f"best_precomputed_model.pt"),
            os.path.join(model_specific_dir, base_name, f"best_precomputed_model.pt"),

            # Atlas with gating may have specific naming in top-level dirs too
            os.path.join(model_specific_dir, val_name, f"best_atlas_with_gating_model.pt"),
            os.path.join(model_specific_dir, base_name, f"best_atlas_with_gating_model.pt"),
        ]
        possible_paths.extend(top_level_paths)

    for path in possible_paths:
        if os.path.exists(path):
            if debug:
                print(f"Found model at: {path}")
            return path

    # If we reach here, try to find any .pt file in dataset directories
    for root, _, files in os.walk(os.path.join(model_dir, val_name)):
        for file in files:
            if file.endswith(".pt"):
                path = os.path.join(root, file)
                if debug:
                    print(f"Found model at: {path} (through directory search)")
                return path

    for root, _, files in os.walk(os.path.join(model_dir, base_name)):
        for file in files:
            if file.endswith(".pt"):
                path = os.path.join(root, file)
                if debug:
                    print(f"Found model at: {path} (through directory search)")
                return path

    if no_metanet:
        if gating_no_metanet:
            model_type = "Atlas with Gating"
        else:
            model_type = "Atlas"
    else:
        model_type = "no-gating" if no_gating else "gating"

    raise FileNotFoundError(f"Could not find {model_type} model for {dataset_name} (model: {model_name}) in {model_dir}")


def evaluate_model(model_path, dataset, device, args):
    """Evaluate model on dataset

    Args:
        model_path: Path to saved model
        dataset: Dataset with precomputed features
        device: Computation device
        args: Command line arguments

    Returns:
        dict: Evaluation results
    """
    # Load model state
    try:
        if args.debug:
            print(f"Loading model from: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        if args.debug:
            print(f"State dict keys: {list(state_dict.keys())}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        if args.debug:
            traceback.print_exc()
        raise

    # Extract model configuration from state dict if available
    if 'config' in state_dict:
        if args.debug:
            print(f"Loading model configuration from checkpoint")
        config = state_dict['config']
        feature_dim = config.get('feature_dim')
        model_name = config.get('model_name', args.model)  # Get model name from config

        # Check for no_metanet flag in config
        no_metanet = config.get('no_metanet', False)
        gating_no_metanet = config.get('gating_no_metanet', False)

        if no_metanet:
            # Atlas model (with or without gating)
            if gating_no_metanet:
                print(f"Detected Atlas model with Gating for {model_name}")

                # Extract gating parameters if available
                base_threshold = config.get('base_threshold', args.base_threshold)
                beta = config.get('beta', args.beta)
                uncertainty_reg = config.get('uncertainty_reg', args.uncertainty_reg)

                print(f"  Gating parameters: αT={base_threshold:.4f}, β={beta:.4f}")
            else:
                print(f"Detected Atlas model (no MetaNet) for {model_name}")

            if args.debug:
                print(f"Creating DirectFeatureModel with gating={gating_no_metanet}")
        else:
            # Check for MetaNet specific parameters
            num_task_vectors = config.get('num_task_vectors', args.num_task_vectors)
            blockwise = config.get('blockwise', args.blockwise_coef)
            base_threshold = config.get('base_threshold', args.base_threshold)
            beta = config.get('beta', args.beta)
            uncertainty_reg = config.get('uncertainty_reg', args.uncertainty_reg)
            use_augmentation = config.get('use_augmentation', True)
            no_gating = config.get('no_gating', args.no_gating)

            # Check if the model was trained with no-gating mode
            if no_gating or base_threshold < 1e-6:
                print(f"Detected model trained without gating mechanism for {model_name}")
                no_gating = True
            else:
                # Print configuration for model identification
                print(f"Model parameters for {model_name}: αT={base_threshold:.4f}, β={beta:.4f}")
                print(f"Using {'blockwise' if blockwise else 'global'} coefficients with {num_task_vectors} task vectors")
    else:
        # No config found, use command line arguments and analyze model
        if args.debug:
            print("No configuration found in model, using args or test features")

        # Sample features to get dimensions
        batch = next(iter(dataset.test_loader))
        if isinstance(batch, dict):
            features = batch["features"]
        else:
            features = batch[0]
        feature_dim = features.shape[1]
        model_name = args.model  # Default to model name from args

        # Use provided arguments for model type
        no_metanet = args.no_metanet
        gating_no_metanet = args.gating_no_metanet

        if not no_metanet:
            num_task_vectors = args.num_task_vectors
            blockwise = args.blockwise_coef
            base_threshold = args.base_threshold
            beta = args.beta
            uncertainty_reg = args.uncertainty_reg
            use_augmentation = True
            no_gating = args.no_gating

            # Print inferred parameters
            if args.no_gating:
                print(f"Using model without gating mechanism for {model_name}")
            else:
                print(f"Using default parameters for {model_name}: αT={base_threshold:.4f}, β={beta:.4f}")
        else:
            if gating_no_metanet:
                base_threshold = args.base_threshold
                beta = args.beta
                uncertainty_reg = args.uncertainty_reg
                print(f"Using Atlas with gating mechanism for {model_name}: αT={base_threshold:.4f}, β={beta:.4f}")
            else:
                print(f"Using Atlas implementation (direct features) for {model_name}")

    # Create the appropriate model based on detected or specified type
    try:
        if args.debug:
            if no_metanet:
                if gating_no_metanet:
                    print("Creating DirectFeatureModel instance with gating...")
                else:
                    print("Creating DirectFeatureModel instance...")
            else:
                print("Creating AdaptiveGatingMetaNet instance...")

        if no_metanet:
            # Create Atlas model with optional gating
            use_gating = gating_no_metanet or args.gating_no_metanet
            model = DirectFeatureModel(
                feature_dim=feature_dim,
                gating_no_metanet=use_gating,
                base_threshold=args.base_threshold if not 'base_threshold' in locals() else base_threshold,
                beta=args.beta if not 'beta' in locals() else beta,
                uncertainty_reg=args.uncertainty_reg if not 'uncertainty_reg' in locals() else uncertainty_reg
            )
            if args.debug:
                print(f"Created DirectFeatureModel for Atlas{' with gating' if use_gating else ''}")
        else:
            # Create MetaNet model
            model = AdaptiveGatingMetaNet(
                feature_dim=feature_dim,
                task_vectors=num_task_vectors,
                blockwise=blockwise,
                base_threshold=base_threshold,
                beta=beta,
                uncertainty_reg=uncertainty_reg
            )
            if args.debug:
                print(f"Created AdaptiveGatingMetaNet with{'out' if no_gating else ''} gating")

        # In evaluation, we should set training mode to False
        if hasattr(model, 'training_mode'):
            model.training_mode = False

        # Load model weights
        if 'meta_net' in state_dict:
            if args.debug:
                print("Loading weights from 'meta_net' key")
            model.load_state_dict(state_dict['meta_net'])
        else:
            # Try different key patterns
            key_patterns = [
                'module.meta_net.',
                'meta_net.',
                'module.metanet.',
                'metanet.',
                'model.meta_net.',
                'model.image_encoder.meta_net.',
                'dummy_param',  # For Atlas models
                'dummy_linear',  # For Atlas models
                'uncertainty_net',  # For Atlas models with gating
                'log_base_threshold',  # For Atlas models with gating
                'log_beta',  # For Atlas models with gating
            ]

            found_keys = False
            for pattern in key_patterns:
                pattern_keys = {k[len(pattern):]: v for k, v in state_dict.items() if k.startswith(pattern)}
                if pattern_keys:
                    if args.debug:
                        print(f"Found {len(pattern_keys)} keys with pattern: {pattern}")
                    try:
                        model.load_state_dict(pattern_keys)
                        found_keys = True
                        break
                    except Exception as e:
                        if args.debug:
                            print(f"Failed to load with pattern {pattern}: {e}")
                        continue

            if not found_keys:
                print("Warning: Could not find model parameters in state dict")
                try:
                    # Try loading directly
                    model.load_state_dict(state_dict)
                except Exception as e:
                    if args.debug:
                        print(f"Direct loading failed: {e}")
                        traceback.print_exc()

                    # For Atlas models, we can proceed even without parameters
                    if not no_metanet and not args.no_metanet:
                        raise ValueError("Could not load model parameters")
                    else:
                        print("Proceeding with default Atlas model (no parameters required)")
    except Exception as e:
        print(f"Error creating or loading model: {e}")
        if args.debug:
            traceback.print_exc()
        raise

    model = model.to(device)
    model.eval()

    # Get model type information for reporting
    model_info = {
        'no_metanet': no_metanet if 'no_metanet' in locals() else args.no_metanet,
        'gating_no_metanet': gating_no_metanet if 'gating_no_metanet' in locals() else args.gating_no_metanet,
        'model_name': model_name,  # Keep track of model name
    }

    if not model_info['no_metanet']:
        model_info.update({
            'no_gating': no_gating if 'no_gating' in locals() else args.no_gating,
            'base_threshold': getattr(model, 'base_threshold', torch.tensor(0.0)).item()
                if hasattr(model, 'base_threshold') else 0.0,
            'beta': getattr(model, 'beta', torch.tensor(0.0)).item()
                if hasattr(model, 'beta') else 0.0,
        })
    elif model_info['no_metanet'] and model_info['gating_no_metanet']:
        # Get gating parameters for Atlas with gating
        model_info.update({
            'base_threshold': getattr(model, 'base_threshold', torch.tensor(0.0)).item()
                if hasattr(model, 'base_threshold') else 0.0,
            'beta': getattr(model, 'beta', torch.tensor(0.0)).item()
                if hasattr(model, 'beta') else 0.0,
            'log_base_threshold': getattr(model, 'log_base_threshold', torch.tensor(0.0)).item()
                if hasattr(model, 'log_base_threshold') else 0.0,
            'log_beta': getattr(model, 'log_beta', torch.tensor(0.0)).item()
                if hasattr(model, 'log_beta') else 0.0,
        })

    # Create classifier
    try:
        if args.debug:
            print("Creating classifier...")
        num_classes = len(dataset.classnames)
        classifier = torch.nn.Linear(feature_dim, num_classes)

        # Try different key patterns for classifier
        if 'classifier' in state_dict:
            if args.debug:
                print("Loading weights from 'classifier' key")
            classifier.load_state_dict(state_dict['classifier'])
        else:
            # Try different key patterns
            key_patterns = [
                'module.classification_head.',
                'classification_head.',
                'classifier.',
                'module.classifier.',
                'model.classifier.',
                'model.classification_head.',
            ]

            found_keys = False
            for pattern in key_patterns:
                pattern_keys = {k[len(pattern):]: v for k, v in state_dict.items() if k.startswith(pattern)}
                if pattern_keys:
                    try:
                        classifier.load_state_dict(pattern_keys)
                        found_keys = True
                        break
                    except Exception as e:
                        if args.debug:
                            print(f"Failed to load classifier with pattern {pattern}: {e}")
                        continue

            if not found_keys:
                # Look for weight and bias directly
                try:
                    for key in state_dict.keys():
                        if key.endswith('weight') and state_dict[key].shape == classifier.weight.shape:
                            classifier.weight.data = state_dict[key]
                            found_keys = True
                            print(f"Found classifier weight with key: {key}")
                        if key.endswith('bias') and state_dict[key].shape == classifier.bias.shape:
                            classifier.bias.data = state_dict[key]
                            found_keys = True
                            print(f"Found classifier bias with key: {key}")
                except Exception as e:
                    if args.debug:
                        print(f"Failed to match classifier weights: {e}")

                if not found_keys:
                    print("Warning: Could not find classifier weights in the model")
    except Exception as e:
        print(f"Error creating or loading classifier: {e}")
        if args.debug:
            traceback.print_exc()
        raise

    classifier = classifier.to(device)
    classifier.eval()

    # Start evaluation
    if args.debug:
        print("Starting evaluation...")

    correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    all_preds = []
    all_labels = []
    all_confidences = []

    try:
        with torch.no_grad():
            for batch in tqdm(dataset.test_loader, desc="Evaluating", disable=not args.verbose):
                if isinstance(batch, dict):
                    features = batch["features"].to(device)
                    labels = batch["labels"].to(device)
                else:
                    features, labels = batch
                    features = features.to(device)
                    labels = labels.to(device)

                # Forward pass
                transformed_features = model(features)
                outputs = classifier(transformed_features)

                # Get predictions and confidences
                probabilities = torch.softmax(outputs, dim=1)
                confidences, predicted = torch.max(probabilities, dim=1)

                # Compute accuracy
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Update per-class metrics
                for i, label in enumerate(labels):
                    label_idx = label.item()
                    prediction = predicted[i].item()

                    per_class_total[label_idx] += 1
                    if prediction == label_idx:
                        per_class_correct[label_idx] += 1

                # Store predictions and labels for further analysis
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())

    except Exception as e:
        print(f"Error during evaluation: {e}")
        if args.debug:
            traceback.print_exc()
        raise

    # Calculate overall accuracy
    accuracy = correct / total if total > 0 else 0.0

    # Calculate per-class accuracy
    per_class_acc = {}
    per_class_report = []

    for cls_idx in range(len(dataset.classnames)):
        cls_name = dataset.classnames[cls_idx]
        if per_class_total[cls_idx] > 0:
            cls_acc = per_class_correct[cls_idx] / per_class_total[cls_idx]
            per_class_acc[cls_name] = float(cls_acc)
            per_class_report.append({
                'class_id': cls_idx,
                'class_name': cls_name,
                'accuracy': float(cls_acc),
                'correct': per_class_correct[cls_idx],
                'total': per_class_total[cls_idx]
            })

    # Calculate confidence statistics
    all_confidences = np.array(all_confidences)
    confidence_stats = {
        'mean': float(np.mean(all_confidences)),
        'median': float(np.median(all_confidences)),
        'min': float(np.min(all_confidences)),
        'max': float(np.max(all_confidences)),
        'std': float(np.std(all_confidences))
    }

    # Determine model type from parameters
    if model_info['no_metanet']:
        if model_info['gating_no_metanet']:
            model_type = "Atlas_WithGating"
        else:
            model_type = "Atlas"
    elif model_info.get('no_gating', False):
        model_type = "MetaNet_NoGating"
    else:
        model_type = "AdaptiveGating"

    if not model_info['no_metanet'] and 'blockwise' in locals() and blockwise:
        model_type += "_Blockwise"

    # Get gating stats if applicable
    gating_stats = None
    if (not model_info['no_metanet'] and not model_info.get('no_gating', False)) or \
       (model_info['no_metanet'] and model_info['gating_no_metanet']):
        if hasattr(model, 'get_gating_stats'):
            gating_stats = model.get_gating_stats()

            # Display detailed gating information for Atlas with Gating
            if model_info['no_metanet'] and model_info['gating_no_metanet'] and gating_stats:
                print("\nGating Parameters for Atlas with Gating:")
                print(f"  Base threshold (αT): {gating_stats['base_threshold']:.4f} (initialized as {gating_stats['initial_base_threshold']:.4f}, change: {gating_stats['threshold_change_percent']:.2f}%)")
                print(f"  Beta (β): {gating_stats['beta']:.4f} (initialized as {gating_stats['initial_beta']:.4f}, change: {gating_stats['beta_change_percent']:.2f}%)")
                print(f"  Gating ratio: {gating_stats['gating_ratio']*100:.2f}% of features retained")
                print(f"  Average threshold: {gating_stats['avg_threshold']:.4f}")

    # Compile complete results
    results = {
        'accuracy': accuracy,
        'num_correct': correct,
        'num_samples': total,
        'per_class_accuracy': per_class_acc,
        'per_class_report': per_class_report,
        'confidence_stats': confidence_stats,
        'config': {
            'feature_dim': feature_dim,
            'no_metanet': model_info['no_metanet'],
            'gating_no_metanet': model_info['gating_no_metanet'] if model_info['no_metanet'] else False,
            'model_name': model_info['model_name'],  # Include model name in results
        },
        'model_path': model_path,
        'model_type': model_type,
        'evaluation_timestamp': datetime.now().isoformat(),
    }

    # Add MetaNet specific information if applicable
    if not model_info['no_metanet']:
        results['config'].update({
            'num_task_vectors': num_task_vectors if 'num_task_vectors' in locals() else args.num_task_vectors,
            'blockwise': blockwise if 'blockwise' in locals() else args.blockwise_coef,
            'base_threshold': model_info['base_threshold'],
            'beta': model_info['beta'],
            'no_gating': model_info.get('no_gating', False),
        })

        # Add computed gating ratio for MetaNet models with gating
        if not model_info.get('no_gating', False) and hasattr(model, 'get_gating_stats'):
            gating_ratio = gating_stats.get('gating_ratio', 0.0) if gating_stats else 0.0
            results['computed_gating_ratio'] = gating_ratio

    # Add Atlas with Gating specific information if applicable
    elif model_info['no_metanet'] and model_info['gating_no_metanet']:
        results['config'].update({
            'base_threshold': model_info['base_threshold'],
            'beta': model_info['beta'],
            'log_base_threshold': model_info.get('log_base_threshold', 0.0),
            'log_beta': model_info.get('log_beta', 0.0),
        })

        # Add computed gating ratio and parameter change data for Atlas with gating
        if hasattr(model, 'get_gating_stats'):
            if gating_stats:
                results['computed_gating_ratio'] = gating_stats.get('gating_ratio', 0.0)

                # Add detailed parameter stats for Atlas with Gating
                results['gating_parameter_evolution'] = {
                    'initial_base_threshold': gating_stats.get('initial_base_threshold', 0.0),
                    'final_base_threshold': gating_stats.get('base_threshold', 0.0),
                    'threshold_change_percent': gating_stats.get('threshold_change_percent', 0.0),
                    'initial_beta': gating_stats.get('initial_beta', 0.0),
                    'final_beta': gating_stats.get('beta', 0.0),
                    'beta_change_percent': gating_stats.get('beta_change_percent', 0.0),
                }

    # Add gating stats if available
    if gating_stats is not None:
        if (not model_info['no_metanet'] and not model_info.get('no_gating', False)) or \
           (model_info['no_metanet'] and model_info['gating_no_metanet']):
            results['gating_stats'] = {
                'gating_ratio': gating_stats.get('gating_ratio', 0.0),
                'avg_threshold': gating_stats.get('avg_threshold', 0.0),
                'base_threshold': gating_stats.get('base_threshold', 0.0),
                'beta': gating_stats.get('beta', 0.0),
                'log_base_threshold': gating_stats.get('log_base_threshold', 0.0),
                'log_beta': gating_stats.get('log_beta', 0.0),
                'avg_uncertainty': gating_stats.get('avg_uncertainty', 0.0),
                'initial_base_threshold': gating_stats.get('initial_base_threshold', 0.0),
                'initial_beta': gating_stats.get('initial_beta', 0.0),
                'threshold_change_percent': gating_stats.get('threshold_change_percent', 0.0),
                'beta_change_percent': gating_stats.get('beta_change_percent', 0.0),
            }

    if args.debug:
        print(f"Evaluation complete. Accuracy: {accuracy * 100:.2f}%")

        if ((not model_info['no_metanet'] and not model_info.get('no_gating', False)) or \
            (model_info['no_metanet'] and model_info['gating_no_metanet'])) and gating_stats is not None:
            print(f"Gating ratio: {gating_stats.get('gating_ratio', 0.0):.4f}")

    return results


def main():
    """Main evaluation function"""
    # Parse arguments using the updated args.py
    args = parse_arguments()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model-specific save directory for results
    model_save_dir = os.path.join(args.save_dir, args.model, "evaluation_results")
    os.makedirs(model_save_dir, exist_ok=True)
    print(f"Results will be saved to: {model_save_dir}")

    # Create model-specific model directory if it exists
    model_specific_dir = os.path.join(args.model_dir, args.model)
    if os.path.exists(model_specific_dir):
        args.model_dir = model_specific_dir
        print(f"Using model-specific directory for models: {args.model_dir}")

    # Generate descriptive suffix for results
    if args.no_metanet:
        if args.gating_no_metanet:
            config_suffix = "_atlas_with_gating"
        else:
            config_suffix = "_atlas"
    elif args.no_gating:
        config_suffix = "_no_gating"
    else:
        config_suffix = "_adaptive_gating"

    if args.blockwise_coef:
        config_suffix += "_blockwise"
    if args.compare_models:
        config_suffix += "_compare"

    # Print configuration
    print(f"\n=== Evaluation Configuration ===")
    print(f"Model: {args.model}")
    if args.no_metanet:
        if args.gating_no_metanet:
            print(f"Model type: Atlas with Gating")
            print(f"Base threshold (αT): {args.base_threshold:.4f}, Beta (β): {args.beta:.4f}")
        else:
            print(f"Model type: Atlas (No MetaNet)")
    else:
        print(f"Using MetaNet: {not args.no_metanet}")
        print(f"Blockwise coefficients: {args.blockwise_coef}")
        if args.no_gating:
            print(f"Mode: MetaNet only (no gating)")
        else:
            print(f"Default αT: {args.base_threshold:.4f}, β: {args.beta:.4f}")

    print(f"Number of task vectors: {args.num_task_vectors}")
    print(f"Datasets to evaluate: {args.datasets}")
    print("=" * 30)

    # Overall results
    all_results = {}
    summary_results = []

    for dataset_name in args.datasets:
        print(f"\n=== Evaluating dataset {dataset_name} ===")
        dataset = None

        try:
            # Build feature directory path with known structure
            feature_dir = os.path.join(args.data_location, "precomputed_features", args.model, dataset_name)
            if not os.path.exists(feature_dir):
                # Try with Val suffix
                feature_dir = os.path.join(args.data_location, "precomputed_features", args.model, dataset_name + "Val")
                if not os.path.exists(feature_dir):
                    print(f"Features for {dataset_name} not found, skipping")
                    continue

            # Create dataset
            dataset = TestOnlyFeatures(
                feature_dir=feature_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                verbose=args.debug
            )

            try:
                # Find model path based on requested model type
                model_path = find_model_path(
                    args.model_dir,
                    dataset_name,
                    args.model,  # Pass model name to find_model_path
                    no_gating=args.no_gating,
                    no_metanet=args.no_metanet,
                    gating_no_metanet=args.gating_no_metanet,
                    debug=args.debug
                )
                print(f"Using model: {os.path.basename(model_path)}")
            except FileNotFoundError as e:
                print(f"Error: {e}")
                print(f"Skipping evaluation for {dataset_name}")
                continue

            # Evaluate model
            results = evaluate_model(
                model_path,
                dataset,
                device,
                args
            )

            # Print results
            print(f"Model: {results['config']['model_name']}")
            print(f"Accuracy: {results['accuracy'] * 100:.2f}% ({results['num_correct']}/{results['num_samples']})")

            # Print model-specific information
            if not results['config']['no_metanet'] and not results['config'].get('no_gating', False):
                if 'computed_gating_ratio' in results:
                    print(f"Gating parameters: αT={results['config']['base_threshold']:.4f}, "
                        f"β={results['config']['beta']:.4f}, "
                        f"gating ratio={results['computed_gating_ratio']:.4f}")
                else:
                    print(f"Gating parameters: αT={results['config']['base_threshold']:.4f}, "
                        f"β={results['config']['beta']:.4f}")
            elif results['config']['no_metanet'] and results['config'].get('gating_no_metanet', False):
                if 'computed_gating_ratio' in results:
                    print(f"Atlas with Gating - αT={results['config']['base_threshold']:.4f}, "
                        f"β={results['config']['beta']:.4f}, "
                        f"gating ratio={results['computed_gating_ratio']:.4f}")
                else:
                    print(f"Atlas with Gating - αT={results['config']['base_threshold']:.4f}, "
                        f"β={results['config']['beta']:.4f}")

                # Print parameter evolution if available
                if 'gating_parameter_evolution' in results:
                    param_evo = results['gating_parameter_evolution']
                    print(f"Parameter evolution:")
                    print(f"  αT: {param_evo['initial_base_threshold']:.4f} → {param_evo['final_base_threshold']:.4f} ({param_evo['threshold_change_percent']:.2f}%)")
                    print(f"  β: {param_evo['initial_beta']:.4f} → {param_evo['final_beta']:.4f} ({param_evo['beta_change_percent']:.2f}%)")

            elif results['config']['no_metanet']:
                print(f"Model type: Atlas (direct features)")
            else:
                print(f"Model type: MetaNet without gating")

            # Detailed output if requested
            if args.verbose:
                if 'gating_stats' in results and ((not results['config']['no_metanet'] and not results['config'].get('no_gating', False)) or
                                                 (results['config']['no_metanet'] and results['config'].get('gating_no_metanet', False))):
                    gating = results['gating_stats']
                    print(f"Gating details - ratio: {gating['gating_ratio']:.4f}, threshold: {gating['avg_threshold']:.4f}")

                    # Show parameter evolution for Atlas with Gating
                    if results['config']['no_metanet'] and results['config'].get('gating_no_metanet', False):
                        print(f"Parameter evolution details:")
                        print(f"  αT: {gating['initial_base_threshold']:.4f} → {gating['base_threshold']:.4f} ({gating['threshold_change_percent']:.2f}%)")
                        print(f"  β: {gating['initial_beta']:.4f} → {gating['beta']:.4f} ({gating['beta_change_percent']:.2f}%)")
                        print(f"  Log αT: {gating['log_base_threshold']:.4f}")
                        print(f"  Log β: {gating['log_beta']:.4f}")

                print("\nTop 5 classes by accuracy:")
                per_class_report = results['per_class_report']
                per_class_report.sort(key=lambda x: x['accuracy'], reverse=True)

                for i, cls_data in enumerate(per_class_report[:5]):
                    print(f"  {cls_data['class_name']}: {cls_data['accuracy'] * 100:.2f}% "
                          f"({cls_data['correct']}/{cls_data['total']})")

                print("\nConfidence statistics - Mean: {:.4f}, Median: {:.4f}, Std: {:.4f}".format(
                    results['confidence_stats']['mean'],
                    results['confidence_stats']['median'],
                    results['confidence_stats']['std']
                ))

            # Store results
            all_results[dataset_name] = results

            # Add to summary
            summary_entry = {
                'dataset': dataset_name,
                'model_type': results['model_type'],
                'accuracy': results['accuracy'],
                'samples': results['num_samples'],
                'no_metanet': results['config']['no_metanet'],
                'model_name': results['config']['model_name'],  # Include model name
            }

            # Add MetaNet specific information if applicable
            if not results['config']['no_metanet']:
                summary_entry.update({
                    'alpha': results['config']['base_threshold'],
                    'beta': results['config']['beta'],
                    'no_gating': results['config'].get('no_gating', False),
                })

                if 'computed_gating_ratio' in results:
                    summary_entry['gating_ratio'] = results['computed_gating_ratio']

            # Add Atlas with Gating information if applicable
            elif results['config']['no_metanet'] and results['config'].get('gating_no_metanet', False):
                # Include both original and learned parameters
                if 'gating_parameter_evolution' in results:
                    param_evo = results['gating_parameter_evolution']
                    summary_entry.update({
                        'initial_alpha': param_evo['initial_base_threshold'],
                        'alpha': param_evo['final_base_threshold'],
                        'alpha_change': param_evo['threshold_change_percent'],
                        'initial_beta': param_evo['initial_beta'],
                        'beta': param_evo['final_beta'],
                        'beta_change': param_evo['beta_change_percent'],
                        'gating_no_metanet': True,
                    })
                else:
                    summary_entry.update({
                        'alpha': results['config']['base_threshold'],
                        'beta': results['config']['beta'],
                        'gating_no_metanet': True,
                    })

                if 'computed_gating_ratio' in results:
                    summary_entry['gating_ratio'] = results['computed_gating_ratio']

            summary_results.append(summary_entry)

        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            traceback.print_exc()
        finally:
            # Clean up dataset resources
            cleanup_resources(dataset)
            torch.cuda.empty_cache()
            gc.collect()

    # Calculate average accuracy
    if summary_results:
        avg_accuracy = sum(r['accuracy'] for r in summary_results) / len(summary_results)
        all_results['average_accuracy'] = avg_accuracy
        print(f"\nAverage accuracy across all datasets: {avg_accuracy * 100:.2f}%")

    # Save all results with model name included in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        model_save_dir,
        f"evaluation_{args.model}{config_suffix}_{timestamp}.json"
    )

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"\nAll evaluation results saved to: {results_path}")

    # Print summary with improved alignment and separate sections for different model types
    print("\n" + "=" * 100)
    if args.no_metanet:
        if args.gating_no_metanet:
            model_type_str = "Atlas with Gating"
        else:
            model_type_str = "Atlas (No MetaNet)"
    elif args.no_gating:
        model_type_str = "MetaNet (No Gating)"
    else:
        model_type_str = "Adaptive Gating MetaNet"

    print(f"Summary of {args.model} - {model_type_str} Models")
    print("-" * 100)

    # Group summary results by model type
    atlas_with_gating_results = [r for r in summary_results if r.get('no_metanet', False) and r.get('gating_no_metanet', False)]
    atlas_results = [r for r in summary_results if r.get('no_metanet', False) and not r.get('gating_no_metanet', False)]
    metanet_no_gating_results = [r for r in summary_results if not r.get('no_metanet', False) and r.get('no_gating', False)]
    adaptive_gating_results = [r for r in summary_results if not r.get('no_metanet', False) and not r.get('no_gating', False)]

    # Print Atlas with Gating results
    if atlas_with_gating_results:
        print("\nAtlas with Gating Results:")
        print(f"{'Dataset':<15} | {'Accuracy':^10} | {'Model':^10} | {'Init αT':^8} | {'αT':^8} | {'Change':^8} | {'Init β':^8} | {'β':^8} | {'Change':^8} | {'Gating %':^10}")
        print("-" * 110)

        for result in sorted(atlas_with_gating_results, key=lambda x: x['dataset']):
            # Format fields with precise spacing and alignment
            dataset_field = f"{result['dataset']:<15}"
            accuracy_field = f"{result['accuracy']*100:>8.2f}%"
            model_name_field = f"{result['model_name']:<10}"

            # Handle cases where we don't have initial parameter data
            init_alpha_field = f"{result.get('initial_alpha', result.get('alpha', 0)):>7.4f}"
            alpha_field = f"{result.get('alpha', 0):>7.4f}"
            alpha_change_field = f"{result.get('alpha_change', 0):>7.2f}%"

            init_beta_field = f"{result.get('initial_beta', result.get('beta', 0)):>7.4f}"
            beta_field = f"{result.get('beta', 0):>7.4f}"
            beta_change_field = f"{result.get('beta_change', 0):>7.2f}%"

            gating_field = f"{result.get('gating_ratio', 0)*100:>8.2f}%" if 'gating_ratio' in result else "   N/A   "

            # Print with strict alignment using separators
            print(f"{dataset_field} | {accuracy_field:^10} | {model_name_field:^10} | {init_alpha_field:^8} | {alpha_field:^8} | {alpha_change_field:^8} | {init_beta_field:^8} | {beta_field:^8} | {beta_change_field:^8} | {gating_field:^10}")

    # Print Atlas results
    if atlas_results:
        print("\nAtlas (No MetaNet) Results:")
        print(f"{'Dataset':<15} | {'Accuracy':^10} | {'Model':^12} | {'Model Type':^25}")
        print("-" * 68)

        for result in sorted(atlas_results, key=lambda x: x['dataset']):
            # Format fields with precise spacing and alignment
            dataset_field = f"{result['dataset']:<15}"
            accuracy_field = f"{result['accuracy']*100:>8.2f}%"
            model_name_field = f"{result['model_name']:<12}"
            model_type_field = f"{result['model_type']:<25}"

            # Print with strict alignment using separators
            print(f"{dataset_field} | {accuracy_field:^10} | {model_name_field:^12} | {model_type_field:^25}")

    # Print MetaNet without gating results
    if metanet_no_gating_results:
        print("\nMetaNet (No Gating) Results:")
        print(f"{'Dataset':<15} | {'Accuracy':^10} | {'Model':^12} | {'Model Type':^25}")
        print("-" * 68)

        for result in sorted(metanet_no_gating_results, key=lambda x: x['dataset']):
            # Format fields with precise spacing and alignment
            dataset_field = f"{result['dataset']:<15}"
            accuracy_field = f"{result['accuracy']*100:>8.2f}%"
            model_name_field = f"{result['model_name']:<12}"
            model_type_field = f"{result['model_type']:<25}"

            # Print with strict alignment using separators
            print(f"{dataset_field} | {accuracy_field:^10} | {model_name_field:^12} | {model_type_field:^25}")

    # Print Adaptive Gating results
    if adaptive_gating_results:
        print("\nAdaptive Gating MetaNet Results:")
        print(f"{'Dataset':<15} | {'Accuracy':^10} | {'Model':^10} | {'αT':^8} | {'β':^8} | {'Gating %':^10} | {'Model Type':^25}")
        print("-" * 95)

        for result in sorted(adaptive_gating_results, key=lambda x: x['dataset']):
            # Format fields with precise spacing and alignment
            dataset_field = f"{result['dataset']:<15}"
            accuracy_field = f"{result['accuracy']*100:>8.2f}%"
            model_name_field = f"{result['model_name']:<10}"
            alpha_field = f"{result['alpha']:>7.4f}"
            beta_field = f"{result['beta']:>7.4f}"
            gating_field = f"{result.get('gating_ratio', 0)*100:>8.2f}%" if 'gating_ratio' in result else "   N/A   "
            model_type_field = f"{result['model_type']:<25}"

            # Print with strict alignment using separators
            print(f"{dataset_field} | {accuracy_field:^10} | {model_name_field:^10} | {alpha_field:^8} | {beta_field:^8} | {gating_field:^10} | {model_type_field:^25}")

    print("=" * 100)


if __name__ == "__main__":
    main()