"""
SUN397 Patch Module for Precomputed Features

This module provides a fix for SUN397 dataset loading issues by
ensuring consistent class name handling and path resolution.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import glob

class SUN397FixedFeatures:
    """Special handler for SUN397 precomputed features"""

    def __init__(self, feature_dir, preprocess=None, location=None, batch_size=128, num_workers=8):
        """Initialize with directory containing precomputed features

        Args:
            feature_dir: Path to directory with precomputed features
            preprocess: Not used, kept for API compatibility
            location: Not used, kept for API compatibility
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
        """
        print(f"Using SUN397 specialized loader from: {feature_dir}")

        # Try to find feature files with various naming patterns
        train_features_path = self._find_file(feature_dir, ["train_features.pt", "features.pt"])
        train_labels_path = self._find_file(feature_dir, ["train_labels.pt", "labels.pt"])
        val_features_path = self._find_file(feature_dir, ["val_features.pt", "test_features.pt"])
        val_labels_path = self._find_file(feature_dir, ["val_labels.pt", "test_labels.pt"])

        if not train_features_path:
            raise FileNotFoundError(f"Could not find training features in {feature_dir}")

        # Create train dataset
        self.train_dataset = self._create_dataset(train_features_path, train_labels_path)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        # Create test dataset
        test_features = val_features_path if val_features_path else train_features_path
        test_labels = val_labels_path if val_labels_path else train_labels_path
        self.test_dataset = self._create_dataset(test_features, test_labels)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        # Load or create classnames
        classnames_path = os.path.join(feature_dir, "classnames.txt")
        if os.path.exists(classnames_path):
            with open(classnames_path, "r") as f:
                self.classnames = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(self.classnames)} SUN397 class names from file")
        else:
            # Generate classnames based on original SUN397 class format
            unique_labels = torch.unique(self.train_dataset.labels).tolist()
            num_classes = len(unique_labels)

            if num_classes <= 397:  # Reasonable number for SUN397
                # Try to create proper classnames based on Sun397 format
                self.classnames = []
                for i in range(num_classes):
                    self.classnames.append(f"scene {i}")
                print(f"Created {len(self.classnames)} SUN397 scene class names")
            else:
                # Fallback to generic class names
                self.classnames = [f"class_{i}" for i in range(num_classes)]
                print(f"Created {num_classes} generic class names")

            # Save classnames for future use
            with open(classnames_path, "w") as f:
                f.write("\n".join(self.classnames))

    def _find_file(self, base_dir, possible_names):
        """Try to find a file from a list of possible names

        Args:
            base_dir: Base directory to search in
            possible_names: List of possible filenames

        Returns:
            Path to found file or None if not found
        """
        # First check in the base directory
        for name in possible_names:
            path = os.path.join(base_dir, name)
            if os.path.exists(path):
                print(f"Found file: {path}")
                return path

        # Check in subdirectories
        for subdir in ["train", "test", "val"]:
            subdir_path = os.path.join(base_dir, subdir)
            if os.path.exists(subdir_path):
                for name in possible_names:
                    path = os.path.join(subdir_path, name)
                    if os.path.exists(path):
                        print(f"Found file in subdirectory: {path}")
                        return path

        # If still not found, try to find any .pt file that might match
        pt_files = glob.glob(os.path.join(base_dir, "*.pt"))
        for file in pt_files:
            filename = os.path.basename(file)
            if any(possible_name in filename for possible_name in possible_names):
                print(f"Found potential match: {file}")
                return file

        print(f"Warning: Could not find any of {possible_names} in {base_dir}")
        return None

    def _create_dataset(self, features_path, labels_path):
        """Create a dataset from features and labels

        Args:
            features_path: Path to features file
            labels_path: Path to labels file

        Returns:
            Dataset with features and labels
        """
        if not features_path or not labels_path:
            raise FileNotFoundError(f"Missing features or labels file for SUN397")

        try:
            print(f"Loading features from: {features_path}")
            features = torch.load(features_path)
            print(f"Features shape: {features.shape}")

            print(f"Loading labels from: {labels_path}")
            labels = torch.load(labels_path)
            print(f"Labels shape: {labels.shape}")

            if len(features) != len(labels):
                raise ValueError(f"Features ({len(features)}) and labels ({len(labels)}) count mismatch")

            class SimpleDataset(Dataset):
                def __init__(self, features, labels):
                    self.features = features
                    self.labels = labels

                def __len__(self):
                    return len(self.features)

                def __getitem__(self, idx):
                    return {
                        "features": self.features[idx],
                        "labels": self.labels[idx],
                        "index": idx
                    }

            return SimpleDataset(features, labels)
        except Exception as e:
            print(f"Error loading SUN397 dataset: {e}")
            import traceback
            traceback.print_exc()
            raise