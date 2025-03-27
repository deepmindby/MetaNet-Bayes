"""Evaluation Script for Adaptive Gating MetaNet Models with Augmentation Support

This script evaluates Adaptive Gating MetaNet models using pre-computed features,
with enhanced support for augmented data and detailed performance reporting.
"""

import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
import traceback
import gc
from collections import defaultdict
from datetime import datetime


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

        # Define possible test feature file paths with different naming conventions
        possible_test_paths = [
            # Standard names
            (os.path.join(feature_dir, "test_features.pt"), os.path.join(feature_dir, "test_labels.pt")),
            (os.path.join(feature_dir, "val_features.pt"), os.path.join(feature_dir, "val_labels.pt")),
            # Alternative names
            (os.path.join(feature_dir, "features.pt"), os.path.join(feature_dir, "labels.pt")),
            (os.path.join(feature_dir, "eval_features.pt"), os.path.join(feature_dir, "eval_labels.pt")),
            # Subdirectory structure
            (os.path.join(feature_dir, "test", "features.pt"), os.path.join(feature_dir, "test", "labels.pt")),
            (os.path.join(feature_dir, "val", "features.pt"), os.path.join(feature_dir, "val", "labels.pt")),
        ]

        # Try to find test features and labels
        test_features_path = None
        test_labels_path = None

        for feat_path, label_path in possible_test_paths:
            if os.path.exists(feat_path) and os.path.exists(label_path):
                test_features_path = feat_path
                test_labels_path = label_path
                if verbose:
                    print(f"Found test features at: {test_features_path}")
                    print(f"Found test labels at: {test_labels_path}")
                break

        if test_features_path is None:
            raise FileNotFoundError(f"Could not find test features in {feature_dir}")

        # Load test features and labels
        try:
            self.test_dataset = PrecomputedFeatureDataset(
                test_features_path,
                test_labels_path,
                verbose=verbose
            )

            # Create test loader with safer settings
            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=min(num_workers, 2),  # Limit workers
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

        # Check for augmentation information
        self.has_augmentation_info = False
        aug_info_path = os.path.join(feature_dir, "augmentation_info.json")
        if os.path.exists(aug_info_path):
            try:
                with open(aug_info_path, 'r') as f:
                    self.augmentation_info = json.load(f)
                self.has_augmentation_info = True
                if verbose:
                    print(f"Found augmentation info: {self.augmentation_info}")
            except Exception as e:
                print(f"Error loading augmentation info: {e}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate adaptive gating models with augmentation support")
    parser.add_argument("--model", type=str, default="ViT-B-32",
                        help="CLIP model used for feature extraction")
    parser.add_argument("--data-location", type=str,
                        default=os.path.join(os.getcwd(), "MetaNet-Bayes"),
                        help="Root directory for datasets")
    parser.add_argument("--feature-dir", type=str, default=None,
                        help="Explicit directory for precomputed features (overrides data-location)")
    parser.add_argument("--save-dir", type=str, default="results",
                        help="Directory to save evaluation results")
    parser.add_argument("--model-dir", type=str, default="checkpoints_adaptive_gating",
                        help="Directory with trained models")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of worker threads for data loading")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"],
                        help="Datasets to evaluate")
    parser.add_argument("--blockwise-coef", action="store_true", default=False,
                        help="Whether blockwise coefficients were used")
    parser.add_argument("--base-threshold", type=float, default=0.05,
                        help="Base threshold for gating mechanism")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Beta parameter for uncertainty weighting")
    parser.add_argument("--uncertainty-reg", type=float, default=0.01,
                        help="Uncertainty regularization weight")
    parser.add_argument("--num-task-vectors", type=int, default=8,
                        help="Number of task vectors used")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Provide detailed output")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug mode")
    parser.add_argument("--only-test", action="store_true", default=True,
                        help="Only use test data for evaluation (enabled by default)")
    # Add augmentation-specific parameters
    parser.add_argument("--use-augmentation", action="store_true", default=True,
                        help="Whether to use augmentations during training")
    parser.add_argument("--augmented-only", action="store_true", default=False,
                        help="Only evaluate models trained with augmented data")
    parser.add_argument("--compare-augmentation", action="store_true", default=False,
                        help="Compare augmented and non-augmented models")
    parser.add_argument("--max-augmentations", type=int, default=10,
                        help="Maximum number of augmentations used during training")
    # Add gating parameter exploration
    parser.add_argument("--explore-gating", action="store_true", default=False,
                        help="Explore impact of different gating parameters")
    parser.add_argument("--alpha-range", type=str, default="0.01,0.05,0.1",
                        help="Comma-separated list of alpha thresholds to explore")
    parser.add_argument("--beta-range", type=str, default="0.5,1.0,2.0",
                        help="Comma-separated list of beta values to explore")
    return parser.parse_args()


def get_test_dataset(dataset_name, model_name, location, batch_size=128, num_workers=4, debug=False):
    """Get dataset with pre-computed test features with improved path handling"""
    # Clean dataset name if it has "precomputed_" prefix
    if dataset_name.startswith("precomputed_"):
        dataset_name = dataset_name[len("precomputed_"):]

    # Handle the case where dataset ends with "Val"
    base_name = dataset_name
    if dataset_name.endswith("Val"):
        base_name = dataset_name[:-3]

    # Try with common paths - now with more path variations
    possible_paths = [
        # With model name
        os.path.join(location, "precomputed_features", model_name, dataset_name),
        os.path.join(location, "precomputed_features", model_name, base_name),
        os.path.join(location, "precomputed_features", model_name, base_name + "Val"),
        # Without model name
        os.path.join(location, "precomputed_features", dataset_name),
        os.path.join(location, "precomputed_features", base_name),
        os.path.join(location, "precomputed_features", base_name + "Val"),
        # Other common locations
        os.path.join(location, dataset_name),
        os.path.join(location, base_name),
        os.path.join(location, base_name + "Val"),
    ]

    # Special case for SUN397
    if "SUN397" in dataset_name:
        sun_paths = [
            os.path.join(location, "precomputed_features", "SUN397"),
            os.path.join(location, "SUN397"),
            os.path.join(location, "precomputed_features", model_name, "SUN397"),
            os.path.join(location, "precomputed_features", "SUN397Val"),
        ]
        possible_paths = sun_paths + possible_paths

    # If we're in MetaNet-Bayes/MetaNet-Bayes, fix the redundant path
    for i, path in enumerate(possible_paths):
        if "MetaNet-Bayes/MetaNet-Bayes" in path:
            possible_paths[i] = path.replace("MetaNet-Bayes/MetaNet-Bayes", "MetaNet-Bayes")

    if debug:
        print(f"Looking for features for {dataset_name} in {len(possible_paths)} locations")
    else:
        print(f"Searching for {dataset_name} features...")

    # Try each possible path
    for path in possible_paths:
        if os.path.exists(path):
            if debug:
                print(f"Found directory at: {path}")

            try:
                # Try to create test dataset
                return TestOnlyFeatures(
                    feature_dir=path,
                    batch_size=batch_size,
                    num_workers=min(num_workers, 2),  # Limit workers for safety
                    verbose=debug
                )
            except FileNotFoundError:
                if debug:
                    print(f"No test features found in {path}")
                continue
            except Exception as e:
                print(f"Error loading from {path}: {e}")
                if debug:
                    traceback.print_exc()
                continue

    # If we get here, try a recursive search for any path containing the dataset name
    if debug:
        print(f"Standard search failed, performing recursive search for {base_name}...")

    for root, dirs, files in os.walk(location):
        if base_name.lower() in root.lower() and any(f.endswith('.pt') for f in files):
            if debug:
                print(f"Found potential directory: {root}")

            pt_files = [f for f in files if f.endswith('.pt')]
            if debug:
                print(f"Found PT files: {pt_files}")

            # Try to create test dataset
            try:
                return TestOnlyFeatures(
                    feature_dir=root,
                    batch_size=batch_size,
                    num_workers=min(num_workers, 2),
                    verbose=debug
                )
            except Exception as e:
                if debug:
                    print(f"Error creating dataset from {root}: {e}")
                # Continue searching

    # If we get here, all paths failed
    print(f"Warning: Could not find test features for {dataset_name}")
    if debug:
        print("Attempted paths:")
        for path in possible_paths:
            exists = "exists" if os.path.exists(path) else "not found"
            print(f"  - {path} ({exists})")

    return None


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
        if hasattr(dataset, 'test_features') and dataset.test_features is not None:
            dataset.test_features = None
        if hasattr(dataset, 'test_labels') and dataset.test_labels is not None:
            dataset.test_labels = None
    except Exception as e:
        print(f"Warning during dataset cleanup: {e}")

    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()


def find_model_path(model_dir, model_name, dataset_name, debug=False, augmented_only=False):
    """Find the correct model path with better error handling"""
    # Try all possible path combinations
    possible_paths = []

    # Clean dataset name if needed
    clean_name = dataset_name
    if dataset_name.startswith("precomputed_"):
        clean_name = dataset_name[len("precomputed_"):]

    # Try with and without "Val" suffix
    for name in [clean_name, f"{clean_name}Val", clean_name.replace("Val", "")]:
        # Main path options
        path_options = [
            # Standard paths for adaptive gating with augmentation
            (model_dir, name, "best_adaptive_gating_model_augmented.pt"),
            (model_dir, model_name, name, "best_adaptive_gating_model_augmented.pt"),

            # Standard paths for adaptive gating
            (model_dir, name, "best_adaptive_gating_model.pt"),
            (model_dir, model_name, name, "best_adaptive_gating_model.pt"),

            # Alternative naming patterns
            (model_dir, name, "adaptive_gating_model_augmented.pt"),
            (model_dir, name, "adaptive_gating_model.pt"),
            (model_dir, model_name, name, "adaptive_gating_model_augmented.pt"),
            (model_dir, model_name, name, "adaptive_gating_model.pt"),

            # With explicit configuration in filename
            (model_dir, name, "best_model_adaptive_gating_augmented.pt"),
            (model_dir, name, "best_model_adaptive_gating.pt"),
            (model_dir, model_name, name, "best_model_adaptive_gating_augmented.pt"),
            (model_dir, model_name, name, "best_model_adaptive_gating.pt"),

            # Generic model paths (check contents for adaptive gating)
            (model_dir, name, "best_model.pt"),
            (model_dir, model_name, name, "best_model.pt"),
            (model_dir, name, "model.pt"),
            (model_dir, model_name, name, "model.pt"),
        ]

        for path_parts in path_options:
            path = os.path.join(*path_parts)
            possible_paths.append(path)
            if os.path.exists(path):
                # For augmented_only mode, check if model was trained with augmentation
                if augmented_only:
                    try:
                        state_dict = torch.load(path, map_location='cpu')
                        if 'config' in state_dict and state_dict['config'].get('use_augmentation', False):
                            if debug:
                                print(f"Found augmented adaptive gating model at: {path}")
                            return path
                    except Exception:
                        continue
                else:
                    if debug:
                        print(f"Found adaptive gating model at: {path}")
                    return path

    # Try recursive search if no exact match is found
    if debug:
        print(f"No exact model path found, trying recursive search for {clean_name}...")

    # Look for any file matching the pattern in the model directory
    for root, dirs, files in os.walk(model_dir):
        if clean_name.lower() in root.lower():
            for file in files:
                if file.endswith('.pt') and ('model' in file.lower() or 'best' in file.lower()):
                    model_path = os.path.join(root, file)
                    # Try to check if it's an adaptive gating model
                    try:
                        state_dict = torch.load(model_path, map_location='cpu')
                        # Check if this looks like an adaptive gating model
                        is_adaptive = False
                        if 'config' in state_dict:
                            # Check for adaptive gating parameters
                            if 'base_threshold' in state_dict['config'] and 'beta' in state_dict['config']:
                                is_adaptive = True

                        # For augmented_only, also check augmentation flag
                        if augmented_only:
                            if not (is_adaptive and state_dict.get('config', {}).get('use_augmentation', False)):
                                continue

                        if is_adaptive:
                            if debug:
                                print(f"Found adaptive gating model at: {model_path}")
                            return model_path
                    except Exception:
                        # If we can't determine, but "adaptive_gating" is in path, return it
                        if "adaptive_gating" in model_path:
                            if debug:
                                print(f"Found potential adaptive gating model: {model_path}")
                            return model_path
                        continue

    # If we still haven't found anything, try a more permissive approach
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith('.pt') and clean_name.lower() in file.lower():
                model_path = os.path.join(root, file)
                if debug:
                    print(f"Found potential model file: {model_path}")
                return model_path

    # If no paths are found, print all tried paths and raise an error
    if debug:
        print("Attempted the following model paths:")
        for path in possible_paths:
            print(f"  - {path} {'(exists)' if os.path.exists(path) else '(not found)'}")

    raise FileNotFoundError(f"Could not find adaptive gating model for {dataset_name} in {model_dir}")


def compute_gating_ratio(model, features, device):
    """Compute actual gating ratio for diagnostic purposes"""
    # Ensure model is in evaluation mode
    model.eval()

    # Sample a subset of features for efficiency
    sample_size = min(64, features.size(0))
    indices = torch.randperm(features.size(0))[:sample_size]
    sample_features = features[indices].to(device)

    with torch.no_grad():
        # Generate coefficients
        if hasattr(model, 'blockwise') and model.blockwise:
            batch_coefficients = model.meta_net(sample_features).reshape(
                -1, model.num_task_vectors, model.num_blocks)

            # Compute thresholds using actual parameters
            base_threshold = model.base_threshold
            beta_val = model.beta

            # Default uncertainty values
            uncertainties = torch.ones_like(batch_coefficients) * 0.5

            # Compute thresholds
            thresholds = base_threshold * (1.0 + beta_val * uncertainties)

            # Compute gating mask and ratio
            gating_mask = (torch.abs(batch_coefficients) >= thresholds).float()
            gating_ratio = gating_mask.mean().item()

            return gating_ratio
        else:
            return 0.0  # Not applicable for non-blockwise models


def evaluate_model(model_path, dataset, device, args):
    """Evaluate adaptive gating model on dataset"""
    # Import adaptive gating model class
    from src.adaptive_gating_metanet import AdaptiveGatingMetaNet

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
        num_task_vectors = config.get('num_task_vectors', args.num_task_vectors)
        blockwise = config.get('blockwise', args.blockwise_coef)
        base_threshold = config.get('base_threshold', args.base_threshold)
        beta = config.get('beta', args.beta)
        uncertainty_reg = config.get('uncertainty_reg', args.uncertainty_reg)
        use_augmentation = config.get('use_augmentation', False)
        max_augmentations = config.get('max_augmentations', None)

        # Always print config values - this is important for debugging
        print(f"Using model parameters: αT={base_threshold:.4f}, β={beta:.4f}")

        if use_augmentation:
            print(f"Model trained with augmentation (max versions: {max_augmentations if max_augmentations else 'all'})")

        if args.debug:
            print(f"Model config: feature_dim={feature_dim}, num_task_vectors={num_task_vectors}, "
                  f"blockwise={blockwise}, base_threshold={base_threshold}, beta={beta}, "
                  f"use_augmentation={use_augmentation}")
    else:
        # Get feature dimension from dataset if not in config
        if args.debug:
            print("No configuration found in model, using args or test features")

        # Sample features to get dimensions
        batch = next(iter(dataset.test_loader))
        if isinstance(batch, dict):
            features = batch["features"]
        else:
            features = batch[0]
        feature_dim = features.shape[1]

        # Use provided arguments
        num_task_vectors = args.num_task_vectors
        blockwise = args.blockwise_coef
        base_threshold = args.base_threshold
        beta = args.beta
        uncertainty_reg = args.uncertainty_reg
        use_augmentation = args.use_augmentation
        max_augmentations = args.max_augmentations

        print(f"WARNING: No config in model, using default parameters: αT={base_threshold:.4f}, β={beta:.4f}")

        # Try to infer augmentation from model path
        if "augmented" in model_path.lower():
            use_augmentation = True
            print(f"Detected augmentation use from model path")

        if args.debug:
            print(f"Using parameters: feature_dim={feature_dim}, num_task_vectors={num_task_vectors}, "
                  f"blockwise={blockwise}, base_threshold={base_threshold}, beta={beta}, "
                  f"use_augmentation={use_augmentation}")

    # Create model
    try:
        if args.debug:
            print("Creating AdaptiveGatingMetaNet instance...")
        model = AdaptiveGatingMetaNet(
            feature_dim=feature_dim,
            task_vectors=num_task_vectors,
            blockwise=blockwise,
            base_threshold=base_threshold,
            beta=beta,
            uncertainty_reg=uncertainty_reg
        )

        # In evaluation, we should set training mode to False
        if hasattr(model, 'training_mode'):
            model.training_mode = False

        # Load meta_net weights
        if 'meta_net' in state_dict:
            if args.debug:
                print("Loading weights from 'meta_net' key")
            model.load_state_dict(state_dict['meta_net'])
        else:
            # Try different key patterns
            key_patterns = [
                # Search for keys with these prefixes
                'module.meta_net.',
                'meta_net.',
                'module.metanet.',
                'metanet.',
                'model.meta_net.',
                'model.image_encoder.meta_net.',
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
                print("WARNING: Could not find meta_net parameters in model state dict")
                print("Trying model parameters directly...")
                try:
                    # Try loading directly
                    model.load_state_dict(state_dict)
                except Exception as e:
                    if args.debug:
                        print(f"Direct loading failed: {e}")
                        traceback.print_exc()
                    raise ValueError("Could not load model parameters")
    except Exception as e:
        print(f"Error creating or loading model: {e}")
        if args.debug:
            traceback.print_exc()
        raise

    model = model.to(device)
    model.eval()

    # Add flag to indicate model is loaded
    model.is_loaded = True

    # Verify the loaded parameters match expected values from config
    actual_base_threshold = model.base_threshold.item()
    actual_beta = model.beta.item()

    # Print actual values to verify they were loaded correctly
    print(f"Model loaded with: αT={actual_base_threshold:.4f}, β={actual_beta:.4f}")

    # Check if actual values match config/expected values within a tolerance
    threshold_diff = abs(actual_base_threshold - base_threshold)
    beta_diff = abs(actual_beta - beta)

    # Warn if significant differences found
    if threshold_diff > 1e-4 or beta_diff > 1e-4:
        print(f"WARNING: Parameter mismatch! Expected αT={base_threshold:.4f}, β={beta:.4f}")
        print(f"         but got αT={actual_base_threshold:.4f}, β={actual_beta:.4f}")

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
                    print("WARNING: Could not find classifier weights in the model")
    except Exception as e:
        print(f"Error creating or loading classifier: {e}")
        if args.debug:
            traceback.print_exc()
        raise

    classifier = classifier.to(device)
    classifier.eval()

    # Compute gating ratio from first batch of features
    first_batch = next(iter(dataset.test_loader))
    if isinstance(first_batch, dict):
        first_features = first_batch["features"].to(device)
    else:
        first_features, _ = first_batch
        first_features = first_features.to(device)

    gating_ratio = compute_gating_ratio(model, first_features, device)
    print(f"Actual gating ratio: {gating_ratio:.4f}")

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
    model_type = "AdaptiveGating"
    if blockwise:
        model_type += "_Blockwise"
    if use_augmentation:
        model_type += "_Augmented"

    # Get gating stats if available
    gating_stats = None
    if hasattr(model, 'get_gating_stats'):
        gating_stats = model.get_gating_stats()

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
            'num_task_vectors': num_task_vectors,
            'blockwise': blockwise,
            'base_threshold': actual_base_threshold,  # Use actual loaded values
            'beta': actual_beta,  # Use actual loaded values
            'uncertainty_reg': uncertainty_reg,
            'use_augmentation': use_augmentation,
            'max_augmentations': max_augmentations,
        },
        'model_path': model_path,
        'model_type': model_type,
        'evaluation_timestamp': datetime.now().isoformat(),
        'computed_gating_ratio': gating_ratio  # Add computed gating ratio
    }

    # Add gating stats if available
    if gating_stats is not None:
        results['gating_stats'] = {
            'gating_ratio': gating_stats.get('gating_ratio', 0.0),
            'avg_threshold': gating_stats.get('avg_threshold', 0.0),
            'base_threshold': gating_stats.get('base_threshold', 0.0),
            'beta': gating_stats.get('beta', 0.0)
        }

    if args.debug:
        print(f"Evaluation complete. Accuracy: {accuracy * 100:.2f}%")
        if gating_stats is not None:
            print(f"Gating ratio: {gating_stats.get('gating_ratio', 0.0):.4f}")

    return results


def main():
    """Main evaluation function"""
    args = parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Fix path issues
    save_dir = os.path.join(os.getcwd(), args.save_dir)
    if save_dir.endswith("MetaNet-Bayes/MetaNet-Bayes/results"):
        save_dir = save_dir.replace("MetaNet-Bayes/MetaNet-Bayes", "MetaNet-Bayes")

    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")

    # Fix model directory path
    model_dir = os.path.join(os.getcwd(), args.model_dir)
    if "MetaNet-Bayes/MetaNet-Bayes" in model_dir:
        model_dir = model_dir.replace("MetaNet-Bayes/MetaNet-Bayes", "MetaNet-Bayes")

    print(f"Looking for models in: {model_dir}")

    # Fix data location path
    data_location = os.path.join(os.getcwd())
    if data_location.endswith("MetaNet-Bayes"):
        data_location = data_location
    elif not "MetaNet-Bayes" in data_location:
        data_location = os.path.join(data_location, "MetaNet-Bayes")

    # Make sure we don't have redundant MetaNet-Bayes
    if "MetaNet-Bayes/MetaNet-Bayes" in data_location:
        data_location = data_location.replace("MetaNet-Bayes/MetaNet-Bayes", "MetaNet-Bayes")

    print(f"Looking for data in: {data_location}")

    # Update args with fixed paths
    args.save_dir = save_dir
    args.model_dir = model_dir
    args.data_location = data_location

    # Generate a descriptive suffix for results based on configuration
    config_suffix = "_adaptive_gating"
    if args.blockwise_coef:
        config_suffix += "_blockwise"
    if args.use_augmentation:
        config_suffix += "_augmented"
    if args.compare_augmentation:
        config_suffix += "_compare"
    if args.explore_gating:
        config_suffix += "_explore"

    # Print configuration
    print(f"\n=== Evaluation Configuration ===")
    print(f"Model: {args.model}")
    print(f"Blockwise coefficients: {args.blockwise_coef}")
    print(f"Default αT: {args.base_threshold:.4f}, β: {args.beta:.4f}")
    print(f"Uncertainty reg: {args.uncertainty_reg}")
    print(f"Using augmentation: {args.use_augmentation}")
    print(f"Augmented only: {args.augmented_only}")
    print(f"Compare augmentation: {args.compare_augmentation}")
    print(f"Explore gating: {args.explore_gating}")
    if args.explore_gating:
        print(f"Alpha range: {args.alpha_range}")
        print(f"Beta range: {args.beta_range}")
    print(f"Datasets to evaluate: {args.datasets}")
    print("=" * 30)

    # Overall results
    all_results = {}
    summary_results = []

    # If comparing augmented vs non-augmented, create separate storage
    if args.compare_augmentation:
        aug_results = {}
        non_aug_results = {}

    for dataset_name in args.datasets:
        print(f"\n=== Evaluating on {dataset_name} ===")
        dataset = None

        try:
            # Get dataset with precomputed features
            if args.feature_dir:
                # If explicit feature directory is provided, use it
                feature_dir = os.path.join(args.feature_dir, dataset_name)
                dataset = TestOnlyFeatures(
                    feature_dir=feature_dir,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    verbose=args.debug
                )
            else:
                # Otherwise search for features
                dataset = get_test_dataset(
                    dataset_name=dataset_name,
                    model_name=args.model,
                    location=args.data_location,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    debug=args.debug
                )

            if dataset is None:
                print(f"Skipping {dataset_name} - dataset loading failed")
                continue

            if args.explore_gating:
                # Parse alpha and beta ranges
                alpha_values = [float(x) for x in args.alpha_range.split(',')]
                beta_values = [float(x) for x in args.beta_range.split(',')]

                print(f"Exploring gating parameters for {dataset_name}:")
                print(f"  Alpha values: {alpha_values}")
                print(f"  Beta values: {beta_values}")

                # Find base model path
                try:
                    model_path = find_model_path(
                        args.model_dir,
                        args.model,
                        dataset_name,
                        debug=args.debug,
                        augmented_only=args.augmented_only
                    )
                    print(f"Using model: {os.path.basename(model_path)}")
                except FileNotFoundError as e:
                    print(f"ERROR: {e}")
                    print(f"Skipping exploration for {dataset_name}")
                    continue

                # Load model state dict once
                state_dict = torch.load(model_path, map_location=device)

                # Extract feature dimension
                if 'config' in state_dict and 'feature_dim' in state_dict['config']:
                    feature_dim = state_dict['config']['feature_dim']
                else:
                    batch = next(iter(dataset.test_loader))
                    if isinstance(batch, dict):
                        feature_dim = batch["features"].shape[1]
                    else:
                        feature_dim = batch[0].shape[1]

                # Extract other parameters
                if 'config' in state_dict:
                    config = state_dict['config']
                    num_task_vectors = config.get('num_task_vectors', args.num_task_vectors)
                    blockwise = config.get('blockwise', args.blockwise_coef)
                    uncertainty_reg = config.get('uncertainty_reg', args.uncertainty_reg)
                else:
                    num_task_vectors = args.num_task_vectors
                    blockwise = args.blockwise_coef
                    uncertainty_reg = args.uncertainty_reg

                # Create classifier just once
                classifier = torch.nn.Linear(feature_dim, len(dataset.classnames)).to(device)
                if 'classifier' in state_dict:
                    classifier.load_state_dict(state_dict['classifier'])

                # Import AdaptiveGatingMetaNet
                from src.adaptive_gating_metanet import AdaptiveGatingMetaNet

                # Evaluation results for each parameter configuration
                exploration_results = []

                # Explore all parameter combinations
                for alpha in alpha_values:
                    for beta in beta_values:
                        print(f"\nEvaluating with αT={alpha:.4f}, β={beta:.4f}")

                        # Create model with these parameters
                        model = AdaptiveGatingMetaNet(
                            feature_dim=feature_dim,
                            task_vectors=num_task_vectors,
                            blockwise=blockwise,
                            base_threshold=alpha,
                            beta=beta,
                            uncertainty_reg=uncertainty_reg
                        )

                        # Load model weights
                        if 'meta_net' in state_dict:
                            model.load_state_dict(state_dict['meta_net'])

                        # Set training mode to False
                        if hasattr(model, 'training_mode'):
                            model.training_mode = False

                        # Move model to device
                        model = model.to(device)
                        model.eval()

                        # Calculate gating ratio
                        first_batch = next(iter(dataset.test_loader))
                        if isinstance(first_batch, dict):
                            first_features = first_batch["features"].to(device)
                        else:
                            first_features, _ = first_batch
                            first_features = first_features.to(device)

                        # Set the parameters (override loaded values)
                        with torch.no_grad():
                            model.log_base_threshold.data = torch.tensor([np.log(alpha)], device=device)
                            model.log_beta.data = torch.tensor([np.log(beta)], device=device)

                        # Get actual values (should match what we set)
                        actual_alpha = model.base_threshold.item()
                        actual_beta = model.beta.item()
                        gating_ratio = compute_gating_ratio(model, first_features, device)

                        print(f"  Actual αT={actual_alpha:.4f}, β={actual_beta:.4f}, gating ratio: {gating_ratio:.4f}")

                        # Evaluate with these parameters
                        correct = 0
                        total = 0

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

                                # Get predictions
                                _, predicted = torch.max(outputs, dim=1)

                                # Compute accuracy
                                total += labels.size(0)
                                correct += predicted.eq(labels).sum().item()

                        # Calculate accuracy
                        accuracy = correct / total if total > 0 else 0.0
                        print(f"  Accuracy: {accuracy * 100:.2f}%")

                        # Add to results
                        exploration_results.append({
                            'alpha': alpha,
                            'beta': beta,
                            'actual_alpha': actual_alpha,
                            'actual_beta': actual_beta,
                            'gating_ratio': gating_ratio,
                            'accuracy': accuracy,
                            'correct': correct,
                            'total': total
                        })

                        # Clean up to avoid memory leaks
                        del model
                        torch.cuda.empty_cache()

                # Sort by accuracy
                exploration_results.sort(key=lambda x: x['accuracy'], reverse=True)

                # Print summary
                print(f"\nParameter exploration results for {dataset_name} (sorted by accuracy):")
                print(f"{'Alpha':<8} {'Beta':<8} {'Gating %':<10} {'Accuracy':<10}")
                print("-" * 40)
                for result in exploration_results:
                    print(f"{result['alpha']:<8.4f} {result['beta']:<8.4f} {result['gating_ratio']*100:<10.2f} {result['accuracy']*100:<10.2f}")

                # Add to overall results
                all_results[f"{dataset_name}_gating_exploration"] = exploration_results

                # Add best to summary
                best_result = exploration_results[0]
                summary_results.append({
                    'dataset': dataset_name,
                    'model_type': 'AdaptiveGating_Exploration',
                    'accuracy': best_result['accuracy'],
                    'samples': best_result['total'],
                    'alpha': best_result['alpha'],
                    'beta': best_result['beta'],
                    'gating_ratio': best_result['gating_ratio']
                })

            elif args.compare_augmentation:
                # First try to find augmented model
                try:
                    # Set augmented_only to True for this search
                    orig_augmented_only = args.augmented_only
                    args.augmented_only = True

                    aug_model_path = find_model_path(
                        args.model_dir,
                        args.model,
                        dataset_name,
                        debug=args.debug,
                        augmented_only=True
                    )

                    # Restore original setting
                    args.augmented_only = orig_augmented_only

                    print(f"Using augmented adaptive gating model from: {aug_model_path}")

                    # Evaluate augmented model
                    aug_results[dataset_name] = evaluate_model(
                        aug_model_path,
                        dataset,
                        device,
                        args
                    )

                    # Add to summary
                    summary_results.append({
                        'dataset': dataset_name,
                        'model_type': 'AdaptiveGating_Augmented',
                        'accuracy': aug_results[dataset_name]['accuracy'],
                        'samples': aug_results[dataset_name]['num_samples'],
                        'alpha': aug_results[dataset_name]['config']['base_threshold'],
                        'beta': aug_results[dataset_name]['config']['beta'],
                        'gating_ratio': aug_results[dataset_name]['computed_gating_ratio']
                    })

                    print(f"Augmented model accuracy: {aug_results[dataset_name]['accuracy'] * 100:.2f}%")
                except FileNotFoundError:
                    print(f"Could not find augmented adaptive gating model for {dataset_name}")
                    aug_results[dataset_name] = None

                # Then try to find non-augmented model
                try:
                    # Make sure augmented_only is False
                    orig_augmented_only = args.augmented_only
                    args.augmented_only = False

                    # Temporarily turn off use_augmentation to help with model search
                    orig_use_augmentation = args.use_augmentation
                    args.use_augmentation = False

                    non_aug_model_path = find_model_path(
                        args.model_dir,
                        args.model,
                        dataset_name,
                        debug=args.debug,
                        augmented_only=False
                    )

                    # Restore original settings
                    args.augmented_only = orig_augmented_only
                    args.use_augmentation = orig_use_augmentation

                    # Verify this is not an augmented model
                    state_dict = torch.load(non_aug_model_path, map_location='cpu')
                    if 'config' in state_dict and not state_dict['config'].get('use_augmentation', False):
                        print(f"Using non-augmented adaptive gating model from: {non_aug_model_path}")

                        # Evaluate non-augmented model
                        non_aug_results[dataset_name] = evaluate_model(
                            non_aug_model_path,
                            dataset,
                            device,
                            args
                        )

                        # Add to summary
                        summary_results.append({
                            'dataset': dataset_name,
                            'model_type': 'AdaptiveGating_Standard',
                            'accuracy': non_aug_results[dataset_name]['accuracy'],
                            'samples': non_aug_results[dataset_name]['num_samples'],
                            'alpha': non_aug_results[dataset_name]['config']['base_threshold'],
                            'beta': non_aug_results[dataset_name]['config']['beta'],
                            'gating_ratio': non_aug_results[dataset_name]['computed_gating_ratio']
                        })

                        print(f"Non-augmented model accuracy: {non_aug_results[dataset_name]['accuracy'] * 100:.2f}%")
                    else:
                        print(f"Found model at {non_aug_model_path} is actually augmented, skipping non-augmented evaluation")
                        non_aug_results[dataset_name] = None
                except Exception as e:
                    print(f"Could not find/evaluate non-augmented model for {dataset_name}: {e}")
                    non_aug_results[dataset_name] = None

                # If we have both results, calculate improvement
                if aug_results[dataset_name] and non_aug_results[dataset_name]:
                    aug_acc = aug_results[dataset_name]['accuracy']
                    non_aug_acc = non_aug_results[dataset_name]['accuracy']
                    improvement = (aug_acc - non_aug_acc) * 100

                    print(f"\nAugmentation Comparison for {dataset_name}:")
                    print(f"  Non-augmented: {non_aug_acc * 100:.2f}% (αT={non_aug_results[dataset_name]['config']['base_threshold']:.4f}, "
                          f"β={non_aug_results[dataset_name]['config']['beta']:.4f}, "
                          f"gating={non_aug_results[dataset_name]['computed_gating_ratio']:.4f})")
                    print(f"  Augmented:     {aug_acc * 100:.2f}% (αT={aug_results[dataset_name]['config']['base_threshold']:.4f}, "
                          f"β={aug_results[dataset_name]['config']['beta']:.4f}, "
                          f"gating={aug_results[dataset_name]['computed_gating_ratio']:.4f})")
                    print(f"  Improvement:   {improvement:.2f}% absolute ({improvement / non_aug_acc * 100:.2f}% relative)")

                    # Store combined results
                    all_results[dataset_name] = {
                        'augmented': aug_results[dataset_name],
                        'non_augmented': non_aug_results[dataset_name],
                        'improvement': improvement,
                        'relative_improvement': improvement / non_aug_acc * 100
                    }
                elif aug_results[dataset_name]:
                    all_results[dataset_name] = aug_results[dataset_name]
                elif non_aug_results[dataset_name]:
                    all_results[dataset_name] = non_aug_results[dataset_name]
            else:
                # Regular evaluation of a single model
                try:
                    # Find model path
                    model_path = find_model_path(
                        args.model_dir,
                        args.model,
                        dataset_name,
                        debug=args.debug,
                        augmented_only=args.augmented_only
                    )
                    print(f"Using model: {os.path.basename(model_path)}")
                except FileNotFoundError as e:
                    print(f"ERROR: {e}")
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
                print(f"Accuracy: {results['accuracy'] * 100:.2f}% ({results['num_correct']}/{results['num_samples']})")

                # Print gating details
                print(f"Gating parameters: αT={results['config']['base_threshold']:.4f}, "
                      f"β={results['config']['beta']:.4f}, "
                      f"gating ratio={results['computed_gating_ratio']:.4f}")

                # Print detailed results if verbose
                if args.verbose:
                    if 'gating_stats' in results:
                        gating = results['gating_stats']
                        print(f"Gating - ratio: {gating['gating_ratio']:.4f}, threshold: {gating['avg_threshold']:.4f}")

                    print("\nTop 5 classes by accuracy:")
                    per_class_report = results['per_class_report']
                    per_class_report.sort(key=lambda x: x['accuracy'], reverse=True)

                    for i, cls_data in enumerate(per_class_report[:5]):
                        print(f"  {cls_data['class_name']}: {cls_data['accuracy'] * 100:.2f}% "
                              f"({cls_data['correct']}/{cls_data['total']})")

                    print("\nConfidence stats - Mean: {:.4f}, Median: {:.4f}, Std: {:.4f}".format(
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
                    'alpha': results['config']['base_threshold'],
                    'beta': results['config']['beta'],
                    'gating_ratio': results['computed_gating_ratio']
                }

                summary_results.append(summary_entry)

        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            traceback.print_exc()
        finally:
            # Clean up dataset resources
            cleanup_resources(dataset)
            torch.cuda.empty_cache()
            gc.collect()

    # Calculate average accuracy by model type
    if summary_results:
        # Group by model type
        model_types = {}
        for result in summary_results:
            model_type = result['model_type']
            if model_type not in model_types:
                model_types[model_type] = []
            model_types[model_type].append({
                'dataset': result['dataset'],
                'accuracy': result['accuracy'],
                'alpha': result.get('alpha'),
                'beta': result.get('beta'),
                'gating_ratio': result.get('gating_ratio')
            })

        # Calculate averages and add to results
        for model_type, results in model_types.items():
            avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
            avg_gating = sum(r.get('gating_ratio', 0) for r in results if r.get('gating_ratio')) / sum(1 for r in results if r.get('gating_ratio'))

            all_results[f'average_{model_type}'] = {
                'accuracy': avg_accuracy,
                'gating_ratio': avg_gating,
                'results': results
            }

            print(f"\nAverage for {model_type} models: {avg_accuracy * 100:.2f}% (gating: {avg_gating:.4f})")

        # Overall average
        all_accuracies = [r['accuracy'] for r in summary_results]
        overall_avg = sum(all_accuracies) / len(all_accuracies)
        all_results['average_accuracy'] = overall_avg
        print(f"\nOverall average accuracy: {overall_avg * 100:.2f}%")

    # Save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        args.save_dir,
        f"evaluation{config_suffix}_{args.model}_{timestamp}.json"
    )

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"\nAll evaluation results saved to {results_path}")

    # Print summary table with fixed alignment
    print("\n=== Summary ===")
    if args.compare_augmentation:
        print(f"{'Dataset':<15} {'Standard':<10} {'Augmented':<10} {'Improvement':<12} {'αT (S)':<8} {'αT (A)':<8} {'Gating (S)':<10} {'Gating (A)':<10}")
        print("-" * 90)

        # Find all datasets with both augmented and non-augmented results
        compared_datasets = []
        for dataset in args.datasets:
            if dataset in aug_results and aug_results[dataset] and dataset in non_aug_results and non_aug_results[dataset]:
                compared_datasets.append(dataset)

        for dataset in sorted(compared_datasets):
            std_acc = non_aug_results[dataset]['accuracy'] * 100
            aug_acc = aug_results[dataset]['accuracy'] * 100
            improvement = aug_acc - std_acc
            std_alpha = non_aug_results[dataset]['config']['base_threshold']
            aug_alpha = aug_results[dataset]['config']['base_threshold']
            std_gating = non_aug_results[dataset]['computed_gating_ratio']
            aug_gating = aug_results[dataset]['computed_gating_ratio']

            print(f"{dataset:<15} {std_acc:>6.2f}%  {aug_acc:>6.2f}%  {improvement:>+10.2f}% {std_alpha:>7.4f} {aug_alpha:>7.4f} {std_gating*100:>9.2f}% {aug_gating*100:>9.2f}%")
    elif args.explore_gating:
        print(f"{'Dataset':<15} {'Best Alpha':<12} {'Best Beta':<12} {'Gating %':<10} {'Accuracy':<10}")
        print("-" * 65)

        for dataset in sorted(args.datasets):
            key = f"{dataset}_gating_exploration"
            if key in all_results and all_results[key]:
                best = all_results[key][0]  # First is the best (sorted)
                print(f"{dataset:<15} {best['alpha']:<12.4f} {best['beta']:<12.4f} {best['gating_ratio']*100:<9.2f}% {best['accuracy']*100:<9.2f}%")
    else:
        print(f"{'Dataset':<15} {'Accuracy':<10} {'αT':<8} {'β':<8} {'Gating %':<10} {'Model Type':<20}")
        print("-" * 75)

        for result in sorted(summary_results, key=lambda x: x['dataset']):
            print(
                f"{result['dataset']:<15} {result['accuracy'] * 100:>6.2f}%  "
                f"{result.get('alpha', 0):>7.4f} {result.get('beta', 0):>7.4f} "
                f"{result.get('gating_ratio', 0)*100:>9.2f}% {result['model_type']:<20}")


if __name__ == "__main__":
    main()