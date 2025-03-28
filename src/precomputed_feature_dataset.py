"""
Dataset classes for precomputed features with support for augmented versions

This module provides dataset classes for working with precomputed CLIP features,
including support for randomly selecting from multiple augmented versions.
"""

import os
import torch
import traceback
import glob
import random
from torch.utils.data import Dataset, DataLoader


class PrecomputedFeatureDataset(Dataset):
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

        # Check if file exists before trying to load
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")

        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        # Load base features and labels
        try:
            self.features = torch.load(features_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load features from {features_path}: {e}")

        try:
            self.labels = torch.load(labels_path)
        except Exception as e:
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

                        # Verify shapes
                        if aug_features.shape == self.features.shape and aug_labels.shape == self.labels.shape:
                            self.augmented_features.append(aug_features)
                            self.augmented_labels.append(aug_labels)
                    except Exception:
                        # Continue silently on error
                        pass

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
                 preprocess=None,  # Not used but kept for API compatibility
                 location=None,  # Not used but kept for API compatibility
                 batch_size=128,
                 num_workers=8,
                 persistent_workers=False,
                 use_augmentation=True,  # Whether to use augmentations
                 max_augmentations=None):  # Maximum number of augmentations to use
        """
        Initialize with directory containing precomputed features

        Args:
            feature_dir: Path to directory with precomputed features
            preprocess: Unused, kept for API compatibility
            location: Unused, kept for API compatibility
            batch_size: Batch size for dataloaders
            num_workers: Number of worker threads for dataloaders
            persistent_workers: Whether to keep worker processes alive
            use_augmentation: Whether to use augmentations during training
            max_augmentations: Maximum number of augmentations to load (None for all)
        """
        # Verify directory exists
        if not os.path.exists(feature_dir):
            raise FileNotFoundError(f"Feature directory not found: {feature_dir}")

        # Define file paths
        train_features_path = os.path.join(feature_dir, "train_features.pt")
        train_labels_path = os.path.join(feature_dir, "train_labels.pt")
        val_features_path = os.path.join(feature_dir, "val_features.pt")
        val_labels_path = os.path.join(feature_dir, "val_labels.pt")

        # Check if train files exist
        if not os.path.exists(train_features_path):
            raise FileNotFoundError(f"Train features not found at {train_features_path}")

        # Find augmented versions without verbose output
        augmentation_paths = []
        aug_idx = 1
        aug_count = 0

        # Count available augmentation files without printing
        while True:
            aug_feat_path = os.path.join(feature_dir, f"train_features_aug{aug_idx}.pt")
            aug_label_path = os.path.join(feature_dir, f"train_labels_aug{aug_idx}.pt")

            if os.path.exists(aug_feat_path) and os.path.exists(aug_label_path):
                augmentation_paths.append((aug_feat_path, aug_label_path))
                aug_idx += 1
                aug_count += 1

                # If max_augmentations is set, limit the number of augmentations
                if max_augmentations is not None and aug_count >= max_augmentations:
                    break
            else:
                break

        # Use val features if available, otherwise fall back to train features for testing
        test_features_path = val_features_path if os.path.exists(val_features_path) else train_features_path
        test_labels_path = val_labels_path if os.path.exists(val_labels_path) else train_labels_path

        # Create datasets
        self.train_dataset = PrecomputedFeatureDataset(
            train_features_path,
            train_labels_path,
            verbose=False,
            augmentation_paths=augmentation_paths,
            use_augmentation=use_augmentation
        )

        # Enable training mode for train dataset
        self.train_dataset.train(True)

        # Create train loader with safe settings
        self.train_loader = DataLoader(
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
            test_features_path,
            test_labels_path,
            verbose=False,
            augmentation_paths=None,  # No augmentation for test dataset
            use_augmentation=False
        )

        # Set test dataset to evaluation mode
        self.test_dataset.train(False)

        # Create test loader with safe settings
        self.test_loader = DataLoader(
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
        else:
            # Create dummy classnames if file doesn't exist
            unique_labels = torch.unique(self.train_dataset.labels)
            self.classnames = [f"class_{i}" for i in range(len(unique_labels))]