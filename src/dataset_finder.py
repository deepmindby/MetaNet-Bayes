"""
Dataset finder utility

This module provides functions to flexibly locate dataset files and pre-computed features
across different directory structures and naming conventions.
"""

import os
import glob
import torch
from typing import List, Dict, Tuple, Optional, Union


def find_dataset_dir(base_dir: str, dataset_name: str, case_sensitive: bool = False) -> List[str]:
    """
    Find all directories that could potentially match the dataset name
    with case-insensitive matching if requested.

    Args:
        base_dir: Base directory to search
        dataset_name: Name of the dataset to find
        case_sensitive: Whether to use case-sensitive matching

    Returns:
        List of potential matching directories
    """
    # First try the exact name
    exact_path = os.path.join(base_dir, dataset_name)
    if os.path.exists(exact_path):
        return [exact_path]

    # If case-insensitive, try to match with any case
    matched_dirs = []
    if not case_sensitive and os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and item.lower() == dataset_name.lower():
                matched_dirs.append(item_path)

    # Also try with and without "Val" suffix
    if dataset_name.endswith("Val"):
        base_name = dataset_name[:-3]  # Remove "Val"
        base_path = os.path.join(base_dir, base_name)
        if os.path.exists(base_path):
            matched_dirs.append(base_path)

        # Also try lower case
        if not case_sensitive:
            lower_base_path = os.path.join(base_dir, base_name.lower())
            if os.path.exists(lower_base_path):
                matched_dirs.append(lower_base_path)
    else:
        # Try adding "Val" suffix
        val_path = os.path.join(base_dir, dataset_name + "Val")
        if os.path.exists(val_path):
            matched_dirs.append(val_path)

        # Also try lower case
        if not case_sensitive:
            lower_val_path = os.path.join(base_dir, dataset_name.lower() + "Val")
            if os.path.exists(lower_val_path):
                matched_dirs.append(lower_val_path)

    return matched_dirs


def find_precomputed_features(
        model_name: str,
        dataset_name: str,
        base_dirs: List[str]
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Find pre-computed feature files for a given model and dataset.
    Searches in multiple possible locations and with different naming conventions.

    Args:
        model_name: Name of the model (e.g. "ViT-B-32")
        dataset_name: Name of the dataset (e.g. "MNIST" or "MNISTVal")
        base_dirs: List of base directories to search

    Returns:
        Tuple of (train_features_path, train_labels_path, classnames_path)
        Returns None for each item if not found
    """
    # Clean up dataset name
    clean_name = dataset_name
    if dataset_name.startswith("precomputed_"):
        clean_name = dataset_name[len("precomputed_"):]

    # Possible feature directory patterns
    possible_patterns = []

    # Try with and without "Val" suffix
    dataset_variants = [clean_name]
    if clean_name.endswith("Val"):
        dataset_variants.append(clean_name[:-3])
    else:
        dataset_variants.append(clean_name + "Val")

    # Add lowercase variants
    dataset_variants.extend([v.lower() for v in dataset_variants])
    dataset_variants.extend([v.upper() for v in dataset_variants])

    # Remove duplicates while preserving order
    dataset_variants = list(dict.fromkeys(dataset_variants))

    # Generate all possible patterns
    for base_dir in base_dirs:
        for variant in dataset_variants:
            # Standard pattern: base_dir/precomputed_features/model_name/dataset
            possible_patterns.append(os.path.join(base_dir, "precomputed_features", model_name, variant))

            # Alternative patterns
            possible_patterns.append(os.path.join(base_dir, "precomputed_features", variant))
            possible_patterns.append(os.path.join(base_dir, model_name, variant))
            possible_patterns.append(os.path.join(base_dir, "features", model_name, variant))
            possible_patterns.append(os.path.join(base_dir, variant))

    # Different naming conventions for feature files
    feature_file_patterns = [
        ("train_features.pt", "train_labels.pt", "classnames.txt"),
        ("features.pt", "labels.pt", "classes.txt"),
        ("train/features.pt", "train/labels.pt", "classnames.txt"),
    ]

    # Search for files
    for dir_pattern in possible_patterns:
        if not os.path.exists(dir_pattern):
            continue

        for feat_pat, label_pat, class_pat in feature_file_patterns:
            feat_path = os.path.join(dir_pattern, feat_pat)
            label_path = os.path.join(dir_pattern, label_pat)
            class_path = os.path.join(dir_pattern, class_pat)

            if os.path.exists(feat_path) and os.path.exists(label_path):
                return feat_path, label_path, class_path if os.path.exists(class_path) else None

    # If nothing found, return None for all
    return None, None, None


def ensure_dir_exists(path: str) -> bool:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        True if the directory exists or was created, False otherwise
    """
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {path}: {e}")
        return False


def save_features_with_backup(features: torch.Tensor, labels: torch.Tensor,
                              feature_path: str, label_path: str,
                              classnames: Optional[List[str]] = None,
                              classname_path: Optional[str] = None) -> bool:
    """
    Save features and labels to disk with a backup mechanism.

    Args:
        features: Feature tensor
        labels: Label tensor
        feature_path: Path to save features
        label_path: Path to save labels
        classnames: Optional list of class names
        classname_path: Optional path to save class names

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(feature_path), exist_ok=True)

        # Save with backup approach
        # First create temporary files
        temp_feature_path = feature_path + ".tmp"
        temp_label_path = label_path + ".tmp"

        torch.save(features, temp_feature_path)
        torch.save(labels, temp_label_path)

        # Then rename to final filename
        os.replace(temp_feature_path, feature_path)
        os.replace(temp_label_path, label_path)

        # Save classnames if provided
        if classnames and classname_path:
            with open(classname_path, "w") as f:
                f.write("\n".join(classnames))

        return True
    except Exception as e:
        print(f"Error saving features: {e}")
        return False