"""Optimized Training Script Using Pre-computed Features

This script provides an optimized approach to training models using pre-computed
CLIP features with a clean and direct path handling strategy. Supports training
with MetaNet, adaptive gating, or direct feature approach (Atlas).
"""

import os
import time
import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
# Removed GradScaler and autocast imports
import gc
import traceback
from datetime import datetime
import math

from src.adaptive_gating_metanet import AdaptiveGatingMetaNet
from src.utils import cosine_lr
from src.datasets.common import maybe_dictionarize
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp
from src.visualize_coefficients import visualize_task_vector_coefficients

from src.args import parse_arguments
args = parse_arguments()


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

        # Add a small random component to break symmetry
        if self.training:
            random_noise = torch.rand_like(combined_uncertainty) * 0.1
        else:
            # In evaluation mode, use fixed small value for deterministic results
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
    """Dataset for precomputed features with augmentation support"""

    def __init__(self, features_path, labels_path, verbose=False,
                 augmentation_paths=None, use_augmentation=True):
        """
        Initialize dataset with paths to precomputed features and labels

        Args:
            features_path: Path to precomputed features tensor
            labels_path: Path to labels tensor
            verbose: Whether to print detailed logs
            augmentation_paths: List of paths to augmented feature/label pairs
            use_augmentation: Whether to use augmented versions when available
        """
        super().__init__()

        # Store augmentation settings
        self.training = True  # Default to training mode
        self.use_augmentation = use_augmentation
        self.augmentation_paths = []
        if augmentation_paths is not None:
            self.augmentation_paths = augmentation_paths

        # Load base features and labels
        try:
            self.features = torch.load(features_path)
            # if verbose:
            #     print(f"Successfully loaded features from {features_path}, shape: {self.features.shape}")
        except Exception as e:
            print(f"Error loading features from {features_path}: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to load features from {features_path}: {e}")

        try:
            self.labels = torch.load(labels_path)
            # if verbose:
            #     print(f"Successfully loaded labels from {labels_path}, shape: {self.labels.shape}")
        except Exception as e:
            print(f"Error loading labels from {labels_path}: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to load labels from {labels_path}: {e}")

        # Validate that features and labels have matching sizes
        if len(self.features) != len(self.labels):
            raise ValueError(f"Features ({len(self.features)}) and labels ({len(self.labels)}) count mismatch")

        # Load augmented versions if provided
        self.augmented_features = []
        self.augmented_labels = []

        if augmentation_paths and use_augmentation:
            for aug_idx, (aug_feat_path, aug_label_path) in enumerate(augmentation_paths):
                if os.path.exists(aug_feat_path) and os.path.exists(aug_label_path):
                    try:
                        aug_features = torch.load(aug_feat_path)
                        aug_labels = torch.load(aug_label_path)

                        # Verify shapes match original features
                        if aug_features.shape == self.features.shape and aug_labels.shape == self.labels.shape:
                            self.augmented_features.append(aug_features)
                            self.augmented_labels.append(aug_labels)
                            # if verbose:
                            #     print(f"Loaded augmented version {aug_idx + 1} from {aug_feat_path}")
                        else:
                            print(f"Warning: Augmented version {aug_idx + 1} has mismatched shape, skipping")
                    except Exception as e:
                        print(f"Error loading augmented version {aug_idx + 1}: {e}")
                        if verbose:
                            traceback.print_exc()

            # if verbose:
            #     print(f"Loaded {len(self.augmented_features)} augmented versions")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # During training, randomly choose from augmented versions if available
        if self.training and self.augmented_features and self.use_augmentation and random.random() > 0.2:
            # 80% chance to use augmented features
            aug_idx = random.randint(0, len(self.augmented_features) - 1)
            return {
                "features": self.augmented_features[aug_idx][idx],
                "labels": self.augmented_labels[aug_idx][idx],
                "index": idx,
                "augmented": True
            }
        else:
            # Use original features or when evaluating
            return {
                "features": self.features[idx],
                "labels": self.labels[idx],
                "index": idx,
                "augmented": False
            }

    def train(self, mode=True):
        """Sets the dataset in training mode (will use augmentations)"""
        self.training = mode
        return self


class PrecomputedFeatures:
    """Dataset container class for precomputed features with augmentation support"""

    def __init__(self,
                 feature_dir,
                 batch_size=128,
                 num_workers=8,
                 persistent_workers=False,
                 use_augmentation=True):
        """
        Initialize with directory containing precomputed features

        Args:
            feature_dir: Path to directory with precomputed features
            batch_size: Batch size for dataloaders
            num_workers: Number of worker threads for dataloaders
            persistent_workers: Whether to keep worker processes alive
            use_augmentation: Whether to use augmentations during training
        """
        # Verify directory exists
        if not os.path.exists(feature_dir):
            raise FileNotFoundError(f"Feature directory not found: {feature_dir}")

        # print(f"Loading features from {feature_dir}")
        # print(f"Augmentation enabled: {use_augmentation}")

        # Define file paths - direct and simple
        train_features_path = os.path.join(feature_dir, "train_features.pt")
        train_labels_path = os.path.join(feature_dir, "train_labels.pt")
        val_features_path = os.path.join(feature_dir, "val_features.pt")
        val_labels_path = os.path.join(feature_dir, "val_labels.pt")

        # Check if train files exist
        if not os.path.exists(train_features_path):
            raise FileNotFoundError(f"Train features not found at {train_features_path}")

        # Find augmented versions
        augmentation_paths = []
        aug_idx = 1

        while True:
            aug_feat_path = os.path.join(feature_dir, f"train_features_aug{aug_idx}.pt")
            aug_label_path = os.path.join(feature_dir, f"train_labels_aug{aug_idx}.pt")

            if os.path.exists(aug_feat_path) and os.path.exists(aug_label_path):
                augmentation_paths.append((aug_feat_path, aug_label_path))
                aug_idx += 1
            else:
                break

        # print(f"Found {len(augmentation_paths)} augmented versions")

        # Create train dataset with augmentation support
        self.train_dataset = PrecomputedFeatureDataset(
            train_features_path,
            train_labels_path,
            verbose=True,
            augmentation_paths=augmentation_paths,
            use_augmentation=use_augmentation
        )

        # Enable training mode for train dataset
        self.train_dataset.train(True)

        # Create train loader
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=persistent_workers and num_workers > 0,
            pin_memory=True,
            drop_last=False,
            timeout=120,  # Add timeout to prevent worker hangs
        )

        # Create test dataset (no augmentation for evaluation)
        self.test_dataset = PrecomputedFeatureDataset(
            val_features_path,
            val_labels_path,
            verbose=True,
            augmentation_paths=None,  # No augmentation for test dataset
            use_augmentation=False
        )

        # Set test dataset to evaluation mode
        self.test_dataset.train(False)

        # Create test loader
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers and num_workers > 0,
            pin_memory=True,
            drop_last=False,
            timeout=120,
        )

        # Load classnames if available
        classnames_path = os.path.join(feature_dir, "classnames.txt")
        if os.path.exists(classnames_path):
            with open(classnames_path, "r") as f:
                self.classnames = [line.strip() for line in f.readlines()]
            # print(f"Loaded {len(self.classnames)} class names from {classnames_path}")
        else:
            # Create dummy classnames if file doesn't exist
            unique_labels = torch.unique(self.train_dataset.labels)
            self.classnames = [f"class_{i}" for i in range(len(unique_labels))]
            print(f"Created {len(self.classnames)} dummy class names")


def cleanup_resources(dataset):
    """Cleanup data resources to prevent memory leaks"""
    if dataset is None:
        return

    try:
        # Clear dataset references
        if hasattr(dataset, 'test_loader') and dataset.test_loader is not None:
            dataset.test_loader = None
        if hasattr(dataset, 'train_loader') and dataset.train_loader is not None:
            dataset.train_loader = None
        if hasattr(dataset, 'test_dataset') and dataset.test_dataset is not None:
            dataset.test_dataset = None
        if hasattr(dataset, 'train_dataset') and dataset.train_dataset is not None:
            dataset.train_dataset = None
    except Exception as e:
        print(f"Warning during dataset cleanup: {e}")

    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_model(model, classifier, dataset, device):
    """Evaluate model on dataset"""
    model.eval()
    classifier.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataset.test_loader:
            batch = maybe_dictionarize(batch)
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            transformed_features = model(features)
            outputs = classifier(transformed_features)

            # Compute accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return correct / total


def plot_training_metrics(train_losses, reg_losses, gating_stats, dataset_name, plot_dir, model_name="Unknown", no_gating=False, no_metanet=False, gating_no_metanet=False):
    """Plot and save training metrics (loss curves, parameter evolution)"""
    # Create figure for loss plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot training loss
    ax1.plot(train_losses, 'r-', linewidth=1.5)
    ax1.set_ylabel('Task Loss', fontsize=14)

    # Adjust title based on mode
    if no_metanet:
        if gating_no_metanet:
            mode_str = f"{model_name} - Atlas with Gating"
        else:
            mode_str = f"{model_name} - Atlas (No MetaNet)"
    elif no_gating:
        mode_str = f"{model_name} - MetaNet Only (No Gating)"
    else:
        mode_str = f"{model_name} - Adaptive Gating MetaNet"

    ax1.set_title(f'Training Loss for {dataset_name} - {mode_str}', fontsize=16)
    ax1.grid(True, alpha=0.7)

    # Plot regularization loss
    ax2.plot(reg_losses, 'g-', linewidth=1.5)
    ax2.set_xlabel('Iterations', fontsize=14)
    ax2.set_ylabel('Regularization Loss', fontsize=14)

    if no_metanet:
        if gating_no_metanet:
            ax2.set_title(f'Regularization Loss for {dataset_name} - {model_name} Atlas with Gating', fontsize=16)
        else:
            ax2.set_title(f'Regularization Loss for {dataset_name} - {model_name} Atlas (expected to be zero)', fontsize=16)
    elif no_gating:
        ax2.set_title(f'Regularization Loss for {dataset_name} - {model_name} No Gating (expected to be zero)', fontsize=16)
    else:
        ax2.set_title(f'Uncertainty Regularization Loss for {dataset_name} - {model_name}', fontsize=16)

    ax2.grid(True, alpha=0.7)

    plt.tight_layout()

    # Add suffix to filename based on mode
    if no_metanet:
        if gating_no_metanet:
            suffix = "_atlas_with_gating"
        else:
            suffix = "_atlas"
    elif no_gating:
        suffix = "_no_gating"
    else:
        suffix = ""

    plt.savefig(os.path.join(plot_dir, f'{dataset_name}{suffix}_loss_curves.png'), dpi=300)
    plt.close()

    # Plot parameter evolution for Atlas with Gating
    if no_metanet and gating_no_metanet and gating_stats:
        # Create parameter evolution plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

        # Extract parameters over time
        iterations = list(range(len(gating_stats)))
        base_thresholds = [stats.get('base_threshold', 0) for stats in gating_stats]
        betas = [stats.get('beta', 0) for stats in gating_stats]
        gating_ratios = [stats.get('gating_ratio', 0) * 100 for stats in gating_stats]

        # Plot base threshold evolution
        ax1.plot(iterations, base_thresholds, 'b-', linewidth=2)
        ax1.set_ylabel('Base Threshold (αT)', fontsize=14)
        ax1.set_title(f'Gating Parameter Evolution for {dataset_name} - {model_name} Atlas', fontsize=16)
        ax1.grid(True, alpha=0.7)

        # Plot beta evolution
        ax2.plot(iterations, betas, 'g-', linewidth=2)
        ax2.set_ylabel('Beta (β)', fontsize=14)
        ax2.grid(True, alpha=0.7)

        # Plot gating ratio evolution
        ax3.plot(iterations, gating_ratios, 'r-', linewidth=2)
        ax3.set_xlabel('Iterations', fontsize=14)
        ax3.set_ylabel('Gating Ratio (%)', fontsize=14)
        ax3.grid(True, alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{dataset_name}_atlas_gating_parameters.png'), dpi=300)
        plt.close()


def plot_validation_metrics(val_accuracies, base_threshold_values, beta_values,
                           log_base_threshold_values, log_beta_values,
                           dataset_name, plot_dir, model_name="Unknown", no_gating=False, no_metanet=False, gating_no_metanet=False):
    """Plot and save validation metrics and parameter evolution"""
    # Create figure for accuracy plot
    fig, ax = plt.subplots(figsize=(12, 6))

    epochs_range = range(1, len(val_accuracies) + 1)
    ax.plot(epochs_range, [acc * 100 for acc in val_accuracies], 'b-o',
            linewidth=2, markersize=8)
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)

    # Adjust title based on mode
    if no_metanet:
        if gating_no_metanet:
            mode_str = f"{model_name} - Atlas with Gating"
        else:
            mode_str = f"{model_name} - Atlas (No MetaNet)"
    elif no_gating:
        mode_str = f"{model_name} - MetaNet Only (No Gating)"
    else:
        mode_str = f"{model_name} - Adaptive Gating MetaNet"

    ax.set_title(f'Validation Accuracy for {dataset_name} - {mode_str}', fontsize=16)
    ax.grid(True, alpha=0.7)

    # Add annotations for key points
    best_epoch = np.argmax(val_accuracies) + 1
    best_acc = max(val_accuracies) * 100
    ax.annotate(f'Best: {best_acc:.2f}%',
                xy=(best_epoch, best_acc),
                xytext=(best_epoch + 1, best_acc - 5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=12)

    plt.tight_layout()

    # Add suffix to filename based on mode
    if no_metanet:
        if gating_no_metanet:
            suffix = "_atlas_with_gating"
        else:
            suffix = "_atlas"
    elif no_gating:
        suffix = "_no_gating"
    else:
        suffix = ""

    plt.savefig(os.path.join(plot_dir, f'{dataset_name}{suffix}_accuracy.png'), dpi=300)
    plt.close()

    # Skip parameter evolution plot in no-metanet without gating mode or no-gating mode
    if (no_metanet and not gating_no_metanet) or no_gating:
        return

    # Create figure for parameter evolution - works for both MetaNet with gating and Atlas with gating
    if len(base_threshold_values) > 0 and len(beta_values) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot actual parameters
        ax1.plot(epochs_range, base_threshold_values, 'r-o', linewidth=2, markersize=8, label='αT')
        ax1.plot(epochs_range, beta_values, 'g-o', linewidth=2, markersize=8, label='β')
        ax1.set_ylabel('Parameter Value', fontsize=14)

        if no_metanet and gating_no_metanet:
            ax1.set_title(f'Gating Parameters Evolution for {dataset_name} - {model_name} Atlas', fontsize=16)
        else:
            ax1.set_title(f'Gating Parameters Evolution for {dataset_name} - {model_name}', fontsize=16)

        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.7)

        # Add annotations for final values
        ax1.annotate(f'Final: {base_threshold_values[-1]:.4f}',
                    xy=(len(base_threshold_values), base_threshold_values[-1]),
                    xytext=(len(base_threshold_values) - 3, base_threshold_values[-1] + 0.02),
                    arrowprops=dict(facecolor='red', shrink=0.05, width=1.5),
                    fontsize=10, color='red')

        ax1.annotate(f'Final: {beta_values[-1]:.4f}',
                    xy=(len(beta_values), beta_values[-1]),
                    xytext=(len(beta_values) - 3, beta_values[-1] - 0.05),
                    arrowprops=dict(facecolor='green', shrink=0.05, width=1.5),
                    fontsize=10, color='green')

        # Plot log parameters
        ax2.plot(epochs_range, log_base_threshold_values, 'r--o', linewidth=2, markersize=8, label='log(αT)')
        ax2.plot(epochs_range, log_beta_values, 'g--o', linewidth=2, markersize=8, label='log(β)')
        ax2.set_xlabel('Epochs', fontsize=14)
        ax2.set_ylabel('Log Parameter Value', fontsize=14)

        if no_metanet and gating_no_metanet:
            ax2.set_title(f'Log Gating Parameters Evolution for {dataset_name} - {model_name} Atlas', fontsize=16)
        else:
            ax2.set_title(f'Log Gating Parameters Evolution for {dataset_name} - {model_name}', fontsize=16)

        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.7)

        plt.tight_layout()

        if no_metanet and gating_no_metanet:
            plt.savefig(os.path.join(plot_dir, f'{dataset_name}_atlas_gating_parameter_evolution.png'), dpi=300)
        else:
            plt.savefig(os.path.join(plot_dir, f'{dataset_name}_parameter_evolution.png'), dpi=300)

        plt.close()


def train_with_adaptive_gating(rank, args):
    """Main training function with adaptive gating"""
    args.rank = rank

    # Initialize distributed setup
    setup_ddp(args.rank, args.world_size, port=args.port)

    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Apply no-gating settings if specified
    if args.no_gating and not args.no_metanet:
        # Store original values for logging purposes
        original_base_threshold = args.base_threshold
        original_beta = args.beta
        original_uncertainty_reg = args.uncertainty_reg

        # Set very small values to effectively disable gating
        args.base_threshold = 1e-9
        args.beta = 1e-9
        args.uncertainty_reg = 0.0

        if is_main_process():
            print(
                f"No-gating mode enabled: base_threshold={args.base_threshold}, beta={args.beta}, uncertainty_reg={args.uncertainty_reg}")
            print(
                f"Original values: base_threshold={original_base_threshold}, beta={original_beta}, uncertainty_reg={original_uncertainty_reg}")

    # Process specified datasets
    datasets_to_process = args.datasets if args.datasets else [
        "Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"
    ]

    # Create model-specific directory path
    model_save_dir = os.path.join(args.save_dir, args.model)

    # Ensure save directory exists
    os.makedirs(model_save_dir, exist_ok=True)
    if is_main_process():
        print(f"Using model-specific save directory: {model_save_dir}")

    # Print configuration (main process only)
    if rank == 0:
        print(f"\n=== Training Configuration ===")
        print(f"Model: {args.model}")
        print(f"Using MetaNet: {not args.no_metanet}")

        if args.no_metanet:
            if args.gating_no_metanet:
                print(f"Atlas with Gating Mode: True")
                print(f"Initial αT: {args.base_threshold:.4f}, β: {args.beta:.4f}")
                print(f"Uncertainty regularization: {args.uncertainty_reg}")
            else:
                print(f"Standard Atlas Mode: True")
        elif not args.no_gating:
            print(f"Using blockwise coefficients: {args.blockwise_coef}")
            print(f"Initial αT: {args.base_threshold:.4f}, β: {args.beta:.4f}")
            print(f"Uncertainty regularization: {args.uncertainty_reg}")
        else:
            print(f"No-gating mode: True")

        print(f"Using augmentation: {args.use_augmentation}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.lr}")
        print(f"Epochs: {args.epochs}")
        print(f"Save directory: {model_save_dir}")
        print(f"Datasets: {datasets_to_process}")
        print("=" * 30)

    for dataset_name in datasets_to_process:
        if is_main_process():
            if args.no_metanet:
                if args.gating_no_metanet:
                    print(f"=== Training on {dataset_name} with Atlas + Gating ===")
                else:
                    print(f"=== Training on {dataset_name} with Atlas (no MetaNet) ===")
            elif args.no_gating:
                print(f"=== Training on {dataset_name} with MetaNet only (no gating) ===")
            else:
                print(f"=== Training on {dataset_name} with adaptive gating ===")

        # Setup save directory for this dataset (include model name in path)
        save_dir = os.path.join(model_save_dir, dataset_name + "Val")
        if is_main_process():
            os.makedirs(save_dir, exist_ok=True)

        dataset = None

        try:
            # Define feature directory based on standard structure
            feature_dir = os.path.join(args.data_location, "precomputed_features", args.model, dataset_name + "Val")

            # Verify directory exists before proceeding
            if not os.path.exists(feature_dir):
                if is_main_process():
                    print(f"Error: Feature directory not found at {feature_dir}")
                continue

            # Load dataset with precomputed features
            dataset = PrecomputedFeatures(
                feature_dir=feature_dir,
                batch_size=args.batch_size,
                num_workers=2,  # Reduced for stability
                use_augmentation=args.use_augmentation,
            )

            # Get feature dimension from dataset
            sample_batch = next(iter(dataset.train_loader))
            sample_batch = maybe_dictionarize(sample_batch)
            feature_dim = sample_batch["features"].shape[1]
            if is_main_process():
                print(f"Feature dimension: {feature_dim}")

            if args.no_metanet:
                # Create direct feature model (Atlas approach)
                model = DirectFeatureModel(
                    feature_dim=feature_dim,
                    gating_no_metanet=args.gating_no_metanet,
                    base_threshold=args.base_threshold,
                    beta=args.beta,
                    uncertainty_reg=args.uncertainty_reg
                )
                if is_main_process():
                    if args.gating_no_metanet:
                        print(f"Created DirectFeatureModel (Atlas approach) with gating mechanism")
                        print(f"Initial gating parameters: αT={args.base_threshold:.4f}, β={args.beta:.4f}")
                    else:
                        print(f"Created DirectFeatureModel (Atlas approach)")
            else:
                # Create adaptive gating model (with or without gating)
                model = AdaptiveGatingMetaNet(
                    feature_dim=feature_dim,
                    task_vectors=args.num_task_vectors,
                    blockwise=args.blockwise_coef,
                    base_threshold=args.base_threshold,
                    beta=args.beta,
                    uncertainty_reg=args.uncertainty_reg,
                    reg_coefficient=args.reg_coefficient if hasattr(args, 'reg_coefficient') else 0.001,
                    margin_weight=args.margin_weight if hasattr(args, 'margin_weight') else 0.0001
                )
                if is_main_process():
                    model_type = "with adaptive gating" if not args.no_gating else "without gating"
                    print(f"Created AdaptiveGatingMetaNet {model_type}")

            model = model.to(rank)

            data_loader = dataset.train_loader
            num_batches = len(data_loader)

            # Set print frequency
            print_every = max(num_batches // 10, 1)

            # Distributed training setup
            ddp_loader = distribute_loader(data_loader)
            ddp_model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.rank],
                find_unused_parameters=True
            )

            # Setup classifier layer
            num_classes = len(dataset.classnames)
            classifier = torch.nn.Linear(feature_dim, num_classes).cuda()

            # Setup optimizer with parameter groups
            # Group parameters to allow different learning rates
            gating_log_params = []
            uncertainty_net_params = []
            meta_net_params = []
            other_params = []

            # Separate parameters by type - handle different model types
            if args.no_metanet:
                if args.gating_no_metanet:
                    # For Atlas with gating, create dedicated groups for gating parameters
                    for name, param in ddp_model.named_parameters():
                        if 'log_beta' in name or 'log_base_threshold' in name:
                            gating_log_params.append(param)
                        elif 'uncertainty_net' in name:
                            uncertainty_net_params.append(param)
                        else:
                            other_params.append(param)

                    # Add classifier parameters
                    other_params.extend(list(classifier.parameters()))
                else:
                    # For Atlas without gating, all parameters go to other_params
                    other_params = list(ddp_model.parameters()) + list(classifier.parameters())
            else:
                # For MetaNet variants, separate parameter types
                for name, param in ddp_model.named_parameters():
                    if 'log_beta' in name or 'log_base_threshold' in name:
                        gating_log_params.append(param)
                    elif 'meta_net' in name:
                        meta_net_params.append(param)
                    else:
                        other_params.append(param)

                # Add classifier parameters
                other_params.extend(list(classifier.parameters()))

            # Create parameter groups with different learning rates based on model type
            if args.no_metanet:
                if args.gating_no_metanet:
                    # Atlas with gating parameter groups
                    param_groups = [
                        {'params': gating_log_params, 'lr': args.lr * args.lr_multiplier,
                         'weight_decay': args.weight_decay},
                        {'params': uncertainty_net_params, 'lr': args.lr * 5.0, 'weight_decay': 0.001},
                        {'params': other_params, 'lr': args.lr, 'weight_decay': args.wd}
                    ]
                    if is_main_process():
                        print(f"Setup optimizer with special groups for Atlas with Gating")
                        print(
                            f"  Gating params: LR = {args.lr * args.lr_multiplier}, Weight Decay = {args.weight_decay}")
                        print(f"  Uncertainty Net: LR = {args.lr * 5.0}, Weight Decay = 0.001")
                        print(f"  Other params: LR = {args.lr}, Weight Decay = {args.wd}")
                else:
                    # Simple parameter group for Atlas
                    param_groups = [
                        {'params': other_params, 'lr': args.lr, 'weight_decay': args.wd}
                    ]
            elif args.no_gating:
                # For no-gating, we mainly care about meta_net parameters
                param_groups = [
                    {'params': meta_net_params, 'lr': args.lr * 3, 'weight_decay': 0.001},  # Higher LR
                    {'params': gating_log_params, 'lr': args.lr * 0.0001, 'weight_decay': 0.0},  # Very low LR
                    {'params': other_params, 'lr': args.lr, 'weight_decay': args.wd}
                ]
            else:
                param_groups = [
                    {'params': gating_log_params, 'lr': args.lr * args.lr_multiplier,
                     'weight_decay': args.weight_decay},  # Higher LR for gating
                    {'params': meta_net_params, 'lr': args.lr * 3, 'weight_decay': 0.001},  # Higher LR for meta_net
                    {'params': other_params, 'lr': args.lr, 'weight_decay': args.wd}
                ]

            optimizer = torch.optim.AdamW(param_groups)

            # Learning rate scheduler
            scheduler = cosine_lr(
                optimizer,
                args.lr,
                0,
                args.epochs * num_batches
            )

            # Loss function
            loss_fn = torch.nn.CrossEntropyLoss()

            # Remove the GradScaler initialization
            # scaler = GradScaler(enabled=True)

            # Training monitoring
            train_losses = []
            reg_losses = []
            detailed_gating_stats = []
            val_accuracies = []
            gating_stats = []
            base_threshold_values = []
            beta_values = []
            log_base_threshold_values = []
            log_beta_values = []
            best_acc = 0.0
            best_model_state = None

            # Training loop
            for epoch in range(args.epochs):
                ddp_model.train()
                classifier.train()

                epoch_loss = 0.0
                epoch_reg_loss = 0.0
                batch_count = 0

                if is_main_process():
                    print(f"\nEpoch {epoch + 1}/{args.epochs} - Training")

                for i, batch in enumerate(ddp_loader):
                    start_time = time.time()

                    try:
                        batch = maybe_dictionarize(batch)
                        features = batch["features"].to(rank)
                        labels = batch["labels"].to(rank)

                        # Remove the autocast context
                        # Forward pass
                        transformed_features = ddp_model(features)
                        logits = classifier(transformed_features)

                        # Task loss
                        task_loss = loss_fn(logits, labels)

                        # Add uncertainty regularization
                        # Will be effectively 0 for Atlas or no-gating models
                        reg_loss = ddp_model.module.uncertainty_regularization_loss()

                        total_loss = task_loss + reg_loss

                        # Backward pass - replace scaler usage with standard backward
                        total_loss.backward()

                        # Step optimizer - standard version without scaler
                        scheduler(i + epoch * num_batches)
                        optimizer.step()
                        optimizer.zero_grad()

                        # Record stats
                        task_loss_cpu = task_loss.item()
                        reg_loss_cpu = reg_loss.item()
                        batch_count += 1
                        epoch_loss += task_loss_cpu
                        epoch_reg_loss += reg_loss_cpu

                        if is_main_process():
                            train_losses.append(task_loss_cpu)
                            reg_losses.append(reg_loss_cpu)

                            # Get current gating parameters for Atlas with gating or adaptive gating MetaNet
                            if (args.no_metanet and args.gating_no_metanet) or (
                                    not args.no_metanet and not args.no_gating):
                                gating_values = {}
                                with torch.no_grad():
                                    gating_values["base_threshold"] = float(ddp_model.module.base_threshold.item())
                                    gating_values["beta"] = float(ddp_model.module.beta.item())
                                    gating_values["log_base_threshold"] = float(
                                        ddp_model.module.log_base_threshold.item())
                                    gating_values["log_beta"] = float(ddp_model.module.log_beta.item())

                                    # Get gating statistics
                                    if hasattr(ddp_model.module, 'get_gating_stats'):
                                        stats = ddp_model.module.get_gating_stats()
                                        if stats:
                                            gating_stats.append(stats)
                                            detailed_gating_stats.append(stats)
                                            # Extract gating ratio
                                            gating_values["gating_ratio"] = stats.get('gating_ratio', 0.0)

                        # Print progress (only with reduced frequency)
                        if i % print_every == 0 and is_main_process():
                            # Get current gating parameters if applicable
                            if (args.no_metanet and args.gating_no_metanet) or (
                                    not args.no_metanet and not args.no_gating):
                                current_base_threshold = gating_values["base_threshold"]
                                current_beta = gating_values["beta"]
                                gating_ratio = gating_values.get("gating_ratio", 0.0)

                                # Detailed output for gating model
                                print(f"  Batch {i:4d}/{num_batches:4d} | Task Loss: {task_loss_cpu:.4f} | "
                                      f"Reg Loss: {reg_loss_cpu:.4f} | αT: {current_base_threshold:.4f} | "
                                      f"β: {current_beta:.4f} | Gate: {gating_ratio * 100:.1f}% | "
                                      f"t: {time.time() - start_time:.2f}s")
                            else:
                                # Compact output for no-gating or Atlas models
                                print(f"  Batch {i:4d}/{num_batches:4d} | Task Loss: {task_loss_cpu:.4f} | "
                                      f"Reg Loss: {reg_loss_cpu:.4f} | "
                                      f"t: {time.time() - start_time:.2f}s")

                    except Exception as e:
                        if is_main_process():
                            print(f"  Error in batch {i}: {e}")
                            traceback.print_exc()
                        # Skip this batch but continue training
                        continue

                # Record epoch stats
                avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
                avg_epoch_reg_loss = epoch_reg_loss / batch_count if batch_count > 0 else 0

                # Record parameter values at the end of each epoch for gating models
                if is_main_process():
                    # For gating models, track parameter evolution
                    if (args.no_metanet and args.gating_no_metanet) or (not args.no_metanet and not args.no_gating):
                        current_base_threshold = float(ddp_model.module.base_threshold.item())
                        current_beta = float(ddp_model.module.beta.item())
                        current_log_base_threshold = float(ddp_model.module.log_base_threshold.item())
                        current_log_beta = float(ddp_model.module.log_beta.item())

                        # Get detailed gating statistics
                        if hasattr(ddp_model.module, 'get_gating_stats'):
                            latest_stats = ddp_model.module.get_gating_stats()
                            if latest_stats:
                                gating_ratio = latest_stats.get('gating_ratio', 0.0)
                                initial_threshold = latest_stats.get('initial_base_threshold', current_base_threshold)
                                initial_beta = latest_stats.get('initial_beta', current_beta)

                                # Calculate changes from initial values
                                threshold_change = ((current_base_threshold - initial_threshold) / max(initial_threshold, 1e-5)) * 100
                                beta_change = ((current_beta - initial_beta) / max(initial_beta, 1e-5)) * 100

                                print(f"  Gating stats: ratio={gating_ratio*100:.1f}%, αT={current_base_threshold:.4f} ({threshold_change:+.1f}%), β={current_beta:.4f} ({beta_change:+.1f}%)")

                        # Save to lists
                        base_threshold_values.append(current_base_threshold)
                        beta_values.append(current_beta)
                        log_base_threshold_values.append(current_log_base_threshold)
                        log_beta_values.append(current_log_beta)

                        # Epoch summary with gating parameters
                        print(f"  Summary: Task Loss: {avg_epoch_loss:.4f} | Reg Loss: {avg_epoch_reg_loss:.4f} | "
                              f"αT: {current_base_threshold:.4f} | β: {current_beta:.4f}")
                    else:
                        # For no-gating or Atlas models without gating, simpler summary
                        base_threshold_values.append(0.0)  # Placeholder values
                        beta_values.append(0.0)
                        log_base_threshold_values.append(0.0)
                        log_beta_values.append(0.0)
                        print(f"  Summary: Task Loss: {avg_epoch_loss:.4f} | Reg Loss: {avg_epoch_reg_loss:.4f}")

                # Evaluate on validation set
                if is_main_process():
                    print(f"Epoch {epoch+1}/{args.epochs} - Validation")

                    val_acc = evaluate_model(
                        model=ddp_model.module,
                        classifier=classifier,
                        dataset=dataset,
                        device=rank
                    )
                    val_accuracies.append(val_acc)

                    print(f"  Accuracy: {val_acc*100:.2f}% ({int(val_acc * len(dataset.test_dataset))}/{len(dataset.test_dataset)})")

                    # Save best model
                    if val_acc > best_acc:
                        best_acc = val_acc

                        # Store model configuration along with weights
                        if args.no_metanet:
                            if args.gating_no_metanet:
                                # Configuration for Atlas with Gating model
                                config = {
                                    'feature_dim': feature_dim,
                                    'no_metanet': True,
                                    'gating_no_metanet': True,
                                    'base_threshold': current_base_threshold,
                                    'beta': current_beta,
                                    'log_base_threshold': current_log_base_threshold,
                                    'log_beta': current_log_beta,
                                    'uncertainty_reg': args.uncertainty_reg,
                                    'model_name': args.model,
                                    'use_augmentation': args.use_augmentation
                                }
                            else:
                                # Configuration for Atlas model
                                config = {
                                    'feature_dim': feature_dim,
                                    'no_metanet': True,
                                    'model_name': args.model,
                                    'use_augmentation': args.use_augmentation
                                }
                        elif args.no_gating:
                            # Configuration for no-gating MetaNet
                            config = {
                                'feature_dim': feature_dim,
                                'num_task_vectors': args.num_task_vectors,
                                'blockwise': args.blockwise_coef,
                                'base_threshold': 0.0,  # Use zero values for no-gating
                                'beta': 0.0,
                                'log_base_threshold': 0.0,
                                'log_beta': 0.0,
                                'uncertainty_reg': 0.0,
                                'model_name': args.model,
                                'use_augmentation': args.use_augmentation,
                                'no_gating': True,
                                'no_metanet': False
                            }
                        else:
                            # Configuration for adaptive gating MetaNet
                            config = {
                                'feature_dim': feature_dim,
                                'num_task_vectors': args.num_task_vectors,
                                'blockwise': args.blockwise_coef,
                                'base_threshold': current_base_threshold,
                                'beta': current_beta,
                                'log_base_threshold': current_log_base_threshold,
                                'log_beta': current_log_beta,
                                'uncertainty_reg': args.uncertainty_reg,
                                'model_name': args.model,
                                'use_augmentation': args.use_augmentation,
                                'no_gating': False,
                                'no_metanet': False
                            }

                        best_model_state = {
                            'meta_net': ddp_model.module.state_dict(),
                            'classifier': classifier.state_dict(),
                            'epoch': epoch,
                            'acc': val_acc,
                            'config': config
                        }

                        # Print appropriate message based on model type
                        if args.no_metanet:
                            if args.gating_no_metanet:
                                print(f"  New best model! (Atlas with Gating, αT={current_base_threshold:.4f}, β={current_beta:.4f})")
                            else:
                                print(f"  New best model! (Atlas approach)")
                        elif args.no_gating:
                            print(f"  New best model! (MetaNet without gating)")
                        else:
                            print(f"  New best model! αT: {current_base_threshold:.4f}, β: {current_beta:.4f}")

            # Save results
            if is_main_process():
                # Save best model with appropriate filename suffix
                if args.no_metanet:
                    if args.gating_no_metanet:
                        model_type_suffix = "_atlas_with_gating"
                    else:
                        model_type_suffix = "_atlas"
                elif args.no_gating:
                    model_type_suffix = "_no_gating"
                else:
                    model_type_suffix = "_adaptive_gating"

                if best_model_state:
                    best_model_path = os.path.join(save_dir, f"best{model_type_suffix}_model.pt")
                    print(f"Saving best model to {best_model_path}")
                    torch.save(best_model_state, best_model_path)

                    # Save a copy with standard name for compatibility
                    torch.save(best_model_state, os.path.join(save_dir, "best_precomputed_model.pt"))

                # Save training history
                history = {
                    'train_losses': train_losses,
                    'reg_losses': reg_losses,
                    'val_accuracies': val_accuracies,
                    'gating_stats': detailed_gating_stats,  # Save detailed gating stats
                    'base_threshold_values': base_threshold_values,
                    'beta_values': beta_values,
                    'log_base_threshold_values': log_base_threshold_values,
                    'log_beta_values': log_beta_values,
                    'best_acc': best_acc,
                    'config': config if 'config' in locals() else {},
                    'use_augmentation': args.use_augmentation,
                    'no_gating': args.no_gating,
                    'no_metanet': args.no_metanet,
                    'gating_no_metanet': args.gating_no_metanet
                }

                # Use appropriate suffix for the history file
                history_path = os.path.join(save_dir, f"{model_type_suffix.strip('_')}_training_history.json")

                with open(history_path, 'w') as f:
                    # Convert numpy values to Python types
                    for key in history:
                        if isinstance(history[key], (list, dict)) and key not in ['gating_stats']:
                            history[key] = [float(item) if isinstance(item, (np.floating, np.integer)) else item
                                          for item in history[key]]

                    json.dump(history, f, indent=4)

                # Create plots directory within the model directory
                plot_dir = os.path.join(model_save_dir, "training_plots")
                os.makedirs(plot_dir, exist_ok=True)

                # Create plots for loss curves and validation metrics - include model name
                plot_training_metrics(train_losses, reg_losses, detailed_gating_stats, dataset_name, plot_dir,
                                    model_name=args.model,
                                    no_gating=args.no_gating,
                                    no_metanet=args.no_metanet,
                                    gating_no_metanet=args.gating_no_metanet)

                plot_validation_metrics(
                    val_accuracies, base_threshold_values, beta_values,
                    log_base_threshold_values, log_beta_values,
                    dataset_name, plot_dir,
                    model_name=args.model,
                    no_gating=args.no_gating,
                    no_metanet=args.no_metanet,
                    gating_no_metanet=args.gating_no_metanet
                )

                # Print completion message with appropriate model description
                if args.no_metanet:
                    if args.gating_no_metanet:
                        mode_str = "Atlas with Gating"
                    else:
                        mode_str = "Atlas approach (no MetaNet)"
                elif args.no_gating:
                    mode_str = "MetaNet only (no gating)"
                else:
                    mode_str = "adaptive gating"

                print(f"Training completed for {dataset_name} with {mode_str}. Best validation accuracy: {best_acc*100:.2f}%")

                # Print final parameters for gating models
                if args.no_metanet and args.gating_no_metanet:
                    print(f"Final parameters - αT: {base_threshold_values[-1]:.4f}, β: {beta_values[-1]:.4f}")

                    if len(detailed_gating_stats) > 0:
                        last_stats = detailed_gating_stats[-1]
                        initial_threshold = last_stats.get('initial_base_threshold', base_threshold_values[-1])
                        initial_beta = last_stats.get('initial_beta', beta_values[-1])
                        threshold_change = last_stats.get('threshold_change_percent', 0.0)
                        beta_change = last_stats.get('beta_change_percent', 0.0)
                        gating_ratio = last_stats.get('gating_ratio', 0.0)

                        print(f"Parameter changes from initial values:")
                        print(f"  αT: {initial_threshold:.4f} → {base_threshold_values[-1]:.4f} ({threshold_change:+.1f}%)")
                        print(f"  β: {initial_beta:.4f} → {beta_values[-1]:.4f} ({beta_change:+.1f}%)")
                        print(f"  Gating ratio (% of features retained): {gating_ratio*100:.1f}%")

                elif not args.no_metanet and not args.no_gating:
                    print(f"Final parameters - αT: {base_threshold_values[-1]:.4f}, β: {beta_values[-1]:.4f}")

        except Exception as e:
            if is_main_process():
                print(f"Error processing dataset {dataset_name}: {e}")
                traceback.print_exc()
        finally:
            # Clean up dataset resources
            cleanup_resources(dataset)
            torch.cuda.empty_cache()
            gc.collect()

    if is_main_process():
        # Generate visualizations for all datasets
        try:
            print("Generating task vector coefficient visualizations for all datasets...")

            # Create a common visualization directory for all models
            vis_dir = os.path.join(model_save_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)

            # Generate visualization for standard task vectors
            visualization_path = visualize_task_vector_coefficients(
                model_dir=model_save_dir,  # This should contain all dataset subdirectories
                output_dir=vis_dir,
                datasets=None,  # Automatically detect all datasets
                model_name=args.model,
                max_vectors_to_show=8,
                max_blocks_per_vector=12
            )

            if visualization_path:
                print(f"Visualization for standard task vectors saved to: {visualization_path}")
            else:
                print("Failed to generate visualization for standard task vectors")
        except Exception as e:
            print(f"Warning: Failed to generate visualizations: {e}")
            import traceback
            traceback.print_exc()

    # Clean up distributed environment
    cleanup_ddp()


if __name__ == "__main__":
    try:
        torch.multiprocessing.spawn(train_with_adaptive_gating, args=(args,), nprocs=args.world_size)
    except Exception as e:
        print(f"Training failed with error: {e}")
        traceback.print_exc()