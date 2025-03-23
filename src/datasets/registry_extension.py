"""Extension to Dataset Registry to Support Pre-computed Features

This module extends the existing dataset registry to handle pre-computed features.
It detects when a dataset name starts with "precomputed_" and loads the appropriate
pre-computed features instead of processing images.
"""

import os
import sys

def get_precomputed_dataset(dataset_name, model_name, location, batch_size=128, num_workers=8):
    """Get dataset with pre-computed features

    Args:
        dataset_name: Name of the dataset (without 'precomputed_' prefix)
        model_name: Name of the model used for feature extraction
        location: Root data directory
        batch_size: Batch size for dataloaders
        num_workers: Number of worker threads

    Returns:
        dataset: Dataset with pre-computed features
    """
    # Import here to avoid circular imports
    from src.datasets.precomputed_features import PrecomputedFeatures

    # Clean dataset name if it has "precomputed_" prefix
    if dataset_name.startswith("precomputed_"):
        dataset_name = dataset_name[len("precomputed_"):]

    # Model name already has correct format (e.g., ViT-B-32) - don't modify it
    model_name_for_path = model_name

    # Build feature directory path
    precomputed_dir = os.path.join(location, "precomputed_features")
    feature_dir = os.path.join(precomputed_dir, model_name_for_path, dataset_name)

    # Check if features exist
    if not os.path.exists(feature_dir):
        raise FileNotFoundError(f"Pre-computed features not found at {feature_dir}")

    # Debugging info
    print(f"Loading pre-computed features from {feature_dir}")

    # Create and return dataset
    return PrecomputedFeatures(
        feature_dir=feature_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )