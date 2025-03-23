"""
Dataset classes for precomputed features

This module provides dataset classes for working with precomputed CLIP features,
eliminating the need to run the ViT encoder during training.
"""

import os
import torch
import traceback
from torch.utils.data import Dataset, DataLoader


class PrecomputedFeatureDataset(Dataset):
    """Dataset for precomputed features"""

    def __init__(self, features_path, labels_path):
        """
        Initialize dataset with paths to precomputed features and labels

        Args:
            features_path: Path to precomputed features tensor
            labels_path: Path to labels tensor
        """
        super().__init__()

        # Check if file exists before trying to load
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")

        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        try:
            self.features = torch.load(features_path)
            print(f"Successfully loaded features from {features_path}, shape: {self.features.shape}")
        except Exception as e:
            print(f"Error loading features from {features_path}: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to load features from {features_path}: {e}")

        try:
            self.labels = torch.load(labels_path)
            print(f"Successfully loaded labels from {labels_path}, shape: {self.labels.shape}")
        except Exception as e:
            print(f"Error loading labels from {labels_path}: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to load labels from {labels_path}: {e}")

        # Validate that features and labels have matching sizes
        if len(self.features) != len(self.labels):
            raise ValueError(f"Features ({len(self.features)}) and labels ({len(self.labels)}) count mismatch")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "labels": self.labels[idx],
            "index": idx
        }


class PrecomputedFeatures:
    """Dataset container class for precomputed features"""

    def __init__(self,
                 feature_dir,
                 preprocess=None,  # Not used but kept for API compatibility
                 location=None,  # Not used but kept for API compatibility
                 batch_size=128,
                 num_workers=8,
                 persistent_workers=False):
        """
        Initialize with directory containing precomputed features

        Args:
            feature_dir: Path to directory with precomputed features
            preprocess: Unused, kept for API compatibility
            location: Unused, kept for API compatibility
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            persistent_workers: Whether to keep worker processes alive
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

        # Use val features if available, otherwise fall back to train features for testing
        test_features_path = val_features_path if os.path.exists(val_features_path) else train_features_path
        test_labels_path = val_labels_path if os.path.exists(val_labels_path) else train_labels_path

        print(f"Using test features from: {test_features_path}")

        # Create datasets
        self.train_dataset = PrecomputedFeatureDataset(
            train_features_path,
            train_labels_path
        )

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

        # Create test dataset
        self.test_dataset = PrecomputedFeatureDataset(
            test_features_path,
            test_labels_path
        )

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
            print(f"Loaded {len(self.classnames)} class names from {classnames_path}")
        else:
            # Create dummy classnames if file doesn't exist
            unique_labels = torch.unique(self.train_dataset.labels)
            self.classnames = [f"class_{i}" for i in range(len(unique_labels))]
            print(f"Created {len(self.classnames)} dummy class names")