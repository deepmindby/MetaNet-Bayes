"""Evaluation Script Using Pre-computed Features (Test Set Only)

This script evaluates MetaNet models using only pre-computed test set features,
eliminating the need for training data during evaluation.
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
        """Initialize with paths to features and labels

        Args:
            features_path: Path to features tensor
            labels_path: Path to labels tensor
            verbose: Whether to print detailed logs
        """
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
        """Initialize with directory containing pre-computed features

        Args:
            feature_dir: Path to directory with pre-computed features
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            verbose: Whether to print detailed logs
        """
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


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate models with pre-computed features (test-only)")
    parser.add_argument("--model", type=str, default="ViT-B-32",
                        help="CLIP model used for feature extraction")
    parser.add_argument("--data-location", type=str,
                        default=os.path.join(os.getcwd(), "MetaNet-Bayes"),
                        help="Root directory for datasets")
    parser.add_argument("--feature-dir", type=str, default=None,
                        help="Explicit directory for precomputed features (overrides data-location)")
    parser.add_argument("--save-dir", type=str, default="results",
                        help="Directory to save evaluation results")
    parser.add_argument("--model-dir", type=str, default="checkpoints_precomputed",
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
    parser.add_argument("--causal_intervention", action="store_true", default=False,
                        help="Whether causal intervention was used")
    parser.add_argument("--top_k_ratio", type=float, default=0.1,
                        help="Ratio of parameter blocks used for intervention")
    parser.add_argument("--num-task-vectors", type=int, default=8,
                        help="Number of task vectors used")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Provide detailed output")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug mode")
    parser.add_argument("--only-test", action="store_true", default=True,
                        help="Only use test data for evaluation (enabled by default)")
    # Add gating mechanism parameters
    parser.add_argument("--use_gating", action="store_true", default=False,
                        help="Whether to use adaptive gating mechanism")
    parser.add_argument("--gating_threshold", type=float, default=0.0,
                        help="Threshold for gating mechanism (α_T)")
    parser.add_argument("--sampling_std", type=float, default=0.01,
                        help="Standard deviation for sampling noise")
    parser.add_argument("--num_samples", type=int, default=2, choices=[1, 2, 3],
                        help="Number of samples for uncertainty estimation (1-3)")
    return parser.parse_args()


def get_test_dataset(dataset_name, model_name, location, batch_size=128, num_workers=4, debug=False):
    """Get dataset with pre-computed test features with improved path handling

    Args:
        dataset_name: Name of the dataset
        model_name: Name of the model used for feature extraction
        location: Root data directory
        batch_size: Batch size for dataloaders
        num_workers: Number of worker threads
        debug: Whether to print debug information

    Returns:
        dataset: Dataset with pre-computed features
    """
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
    print(f"WARNING: Could not find test features for {dataset_name} in any expected location")
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


def find_model_path(model_dir, model_name, dataset_name, debug=False):
    """Find the correct model path with better error handling

    Args:
        model_dir: Base directory for models
        model_name: Name of the model architecture
        dataset_name: Name of the dataset
        debug: Whether to print debug information

    Returns:
        str: Path to the model file
    """
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
            # Standard paths
            (model_dir, name, "best_precomputed_model.pt"),
            (model_dir, model_name, name, "best_precomputed_model.pt"),
            # With explicit configuration in filename
            (model_dir, name, "best_precomputed_model_blockwise.pt"),
            (model_dir, name, "best_precomputed_model_causal.pt"),
            (model_dir, name, "best_precomputed_model_blockwise_causal.pt"),
            (model_dir, model_name, name, "best_precomputed_model_blockwise.pt"),
            (model_dir, model_name, name, "best_precomputed_model_causal.pt"),
            (model_dir, model_name, name, "best_precomputed_model_blockwise_causal.pt"),
            # With model type directories
            (f"{model_dir}-causal", model_name, name, "best_precomputed_model.pt"),
            (f"{model_dir}-meta", model_name, name, "best_precomputed_model.pt"),
            # Alternative file names
            (model_dir, name, "best_model.pt"),
            (model_dir, name, "model.pt"),
            (model_dir, model_name, name, "best_model.pt"),
            (model_dir, model_name, name, "model.pt"),
            # Check in named subdirectories
            (model_dir, name, "checkpoints", "best_model.pt"),
            (model_dir, name, "weights", "best_model.pt"),
            (model_dir, model_name, name, "checkpoints", "best_model.pt"),
            (model_dir, model_name, name, "weights", "best_model.pt"),
            # Base directory with dataset name
            (model_dir, f"{model_name}_{name}.pt"),
            (model_dir, f"model_{name}.pt"),
            (model_dir, f"best_{name}.pt"),
        ]

        for path_parts in path_options:
            path = os.path.join(*path_parts)
            possible_paths.append(path)
            if os.path.exists(path):
                if debug:
                    print(f"Found model at: {path}")
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
                    if debug:
                        print(f"Found potential model file: {model_path}")
                    return model_path

    # If no paths are found, print all tried paths and raise an error
    if debug:
        print("Attempted the following model paths:")
        for path in possible_paths:
            print(f"  - {path} {'(exists)' if os.path.exists(path) else '(not found)'}")

    raise FileNotFoundError(f"Could not find model for {dataset_name} in {model_dir}")


def evaluate_model(model_path, dataset, device, debug=False):
    """Evaluate model on dataset

    Args:
        model_path: path to saved model
        dataset: evaluation dataset
        device: computation device
        debug: whether to print debug information

    Returns:
        dict: evaluation metrics
    """
    # Import required modules
    from src.metanet_precomputed import PrecomputedMetaNet

    # Load model state
    try:
        if debug:
            print(f"Loading model from: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        if debug:
            print(f"State dict keys: {list(state_dict.keys())}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        traceback.print_exc()
        raise

    # Extract model configuration from state dict if available
    if 'config' in state_dict:
        if debug:
            print(f"Loading model configuration from checkpoint")
        config = state_dict['config']
        feature_dim = config.get('feature_dim')
        num_task_vectors = config.get('num_task_vectors', 8)
        blockwise = config.get('blockwise', False)
        enable_causal = config.get('enable_causal', False)
        top_k_ratio = config.get('top_k_ratio', 0.1)
        if debug:
            print(f"Model configuration: feature_dim={feature_dim}, "
                f"num_task_vectors={num_task_vectors}, blockwise={blockwise}, "
                f"enable_causal={enable_causal}, top_k_ratio={top_k_ratio}")
    else:
        # Get feature dimension from dataset if not in config
        if debug:
            print("No configuration found in model, using test features or arguments")

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
        enable_causal = args.causal_intervention
        top_k_ratio = args.top_k_ratio

        # Determine enable_causal from model path if not specified
        if "causal" in model_path.lower():
            enable_causal = True
            if debug:
                print(f"Detected causal model from path: {model_path}")

        if debug:
            print(f"Using parameters: feature_dim={feature_dim}, "
                f"num_task_vectors={num_task_vectors}, blockwise={blockwise}, "
                f"enable_causal={enable_causal}, top_k_ratio={top_k_ratio}")

    # Add clear logging about causal status
    if enable_causal:
        print(f"\n*** CAUSAL INTERVENTION MODEL (top-k ratio: {top_k_ratio}) ***")
    else:
        print(f"\n*** STANDARD MODEL (NO CAUSAL INTERVENTION) ***")

    # Create model
    try:
        if debug:
            print("Creating model instance...")
        model = PrecomputedMetaNet(
            feature_dim=feature_dim,
            task_vectors=num_task_vectors,
            blockwise=blockwise,
            enable_causal=enable_causal,
            top_k_ratio=top_k_ratio
        )

        # Find meta_net weights
        if 'meta_net' in state_dict:
            if debug:
                print("Loading weights from 'meta_net' key")
            model.load_state_dict(state_dict['meta_net'])
        else:
            # Try different key patterns
            key_patterns = [
                # Search for keys with these prefixes
                'module.image_encoder.meta_net.',
                'meta_net.',
                'module.meta_net.',
                'metanet.',
                'module.metanet.',
                'model.meta_net.',
                'model.metanet.',
                # For whole model state dicts, we might need these
                'model.image_encoder.meta_net.',
                'module.model.image_encoder.meta_net.',
            ]

            found_keys = False
            for pattern in key_patterns:
                pattern_keys = {k[len(pattern):]: v for k, v in state_dict.items() if k.startswith(pattern)}
                if pattern_keys:
                    if debug:
                        print(f"Found {len(pattern_keys)} keys with pattern: {pattern}")
                    try:
                        model.load_state_dict(pattern_keys)
                        found_keys = True
                        break
                    except Exception as e:
                        if debug:
                            print(f"Failed to load with pattern {pattern}: {e}")
                        continue

            if not found_keys:
                # Try loading directly if no meta_net keys found
                try:
                    if debug:
                        print("Attempting direct state dict loading")
                    model.load_state_dict(state_dict)
                except Exception as e:
                    if debug:
                        print(f"Direct loading failed: {e}")
                    print("WARNING: Could not find meta_net parameters in model state dict")
                    # Last resort - if the model has been saved with minimal parameters
                    # Try to match keys by name regardless of prefix
                    target_keys = set(dict(model.named_parameters()).keys())
                    matched_dict = {}
                    for target_key in target_keys:
                        short_key = target_key.split('.')[-1]  # e.g., 'weight', 'bias'
                        for state_key, value in state_dict.items():
                            if state_key.endswith(short_key) and value.shape == dict(model.named_parameters())[target_key].shape:
                                matched_dict[target_key] = value
                                break

                    if len(matched_dict) == len(target_keys):
                        print(f"Attempting parameter matching by shape...")
                        model.load_state_dict(matched_dict)
                    else:
                        raise ValueError("Could not load meta_net parameters from state dict")
    except Exception as e:
        print(f"Error creating or loading model: {e}")
        traceback.print_exc()
        raise

    model = model.to(device)
    model.eval()

    # Set inference mode with gating parameters
    if hasattr(model, 'set_inference_mode'):
        gating_params = {
            'use_gating': args.use_gating,
            'gating_threshold': args.gating_threshold,
            'sampling_std': args.sampling_std,
            'num_samples': args.num_samples
        }
        if debug:
            print(f"Setting inference mode with gating parameters: {gating_params}")
        model.set_inference_mode(True, gating_params)
    else:
        if args.use_gating and debug:
            print("Warning: Model does not support gating mechanism")

    # Create classifier
    try:
        if debug:
            print("Creating classifier...")
        num_classes = len(dataset.classnames)
        classifier = torch.nn.Linear(feature_dim, num_classes)

        # Try different key patterns for classifier
        if 'classifier' in state_dict:
            if debug:
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
                'module.model.classification_head.',
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
                        if debug:
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
                    if debug:
                        print(f"Failed to match classifier weights: {e}")

                if not found_keys:
                    print("WARNING: Could not find classifier weights in the model")
    except Exception as e:
        print(f"Error creating or loading classifier: {e}")
        traceback.print_exc()
        raise

    classifier = classifier.to(device)
    classifier.eval()

    # Start evaluation
    if debug:
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
            for batch in tqdm(dataset.test_loader, desc="Evaluating"):
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

    # Determine model type from path
    model_type = "STANDARD"
    if "blockwise" in model_path.lower() or blockwise:
        model_type = "BLOCKWISE"
    if "causal" in model_path.lower() or enable_causal:
        model_type = "CAUSAL"
    if ("blockwise" in model_path.lower() or blockwise) and ("causal" in model_path.lower() or enable_causal):
        model_type = "BLOCKWISE_CAUSAL"

    # Add gating information to model type if used
    if args.use_gating:
        model_type += "_GATED"

    # Get uncertainty information if available
    uncertainty = None
    if hasattr(model, 'get_uncertainty') and args.num_samples > 1:
        uncertainty = model.get_uncertainty()
        if uncertainty is not None and (debug or args.verbose):
            print(f"Prediction uncertainty: {uncertainty:.6f}")

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
            'enable_causal': enable_causal,
            'top_k_ratio': top_k_ratio,
            'use_gating': args.use_gating,
            'gating_threshold': args.gating_threshold,
            'sampling_std': args.sampling_std,
            'num_samples': args.num_samples
        },
        'model_path': model_path,
        'model_type': model_type,
        'evaluation_timestamp': datetime.now().isoformat()
    }

    # Add uncertainty information if available
    if uncertainty is not None:
        results['uncertainty'] = float(uncertainty)

    if debug:
        print(f"Evaluation complete. Accuracy: {accuracy * 100:.2f}%")
        if uncertainty is not None:
            print(f"Prediction uncertainty: {uncertainty:.6f}")

    return results


def main():
    """Main evaluation function"""
    global args
    args = parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Fix path issues - remove redundant MetaNet-Bayes in path
    save_dir = os.path.join(os.getcwd(), "results")
    if save_dir.endswith("MetaNet-Bayes/MetaNet-Bayes/results"):
        save_dir = save_dir.replace("MetaNet-Bayes/MetaNet-Bayes", "MetaNet-Bayes")

    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")

    # Fix model directory path - remove redundant MetaNet-Bayes
    model_dir = os.path.join(os.getcwd(), args.model_dir)
    if "MetaNet-Bayes/MetaNet-Bayes" in model_dir:
        model_dir = model_dir.replace("MetaNet-Bayes/MetaNet-Bayes", "MetaNet-Bayes")

    print(f"Looking for models in: {model_dir}")

    # Fix data location path
    data_location = os.path.join(os.getcwd())
    if data_location.endswith("MetaNet-Bayes"):
        # We're already in the MetaNet-Bayes directory
        data_location = data_location
    elif not "MetaNet-Bayes" in data_location:
        # Add MetaNet-Bayes if needed
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
    config_suffix = "_test_only"
    if args.blockwise_coef:
        config_suffix += "_blockwise"
    if args.causal_intervention:
        config_suffix += "_causal"
    if args.use_gating:
        config_suffix += "_gated"
        if args.num_samples > 1:
            config_suffix += f"_samples{args.num_samples}"

    # Print configuration
    print(f"\n=== Evaluation Configuration ===")
    print(f"Model: {args.model}")
    print(f"Using blockwise coefficients: {args.blockwise_coef}")
    print(f"Using causal intervention: {args.causal_intervention}")
    print(f"Top-k ratio: {args.top_k_ratio}")
    print(f"Number of task vectors: {args.num_task_vectors}")
    if args.use_gating:
        print(f"Gating enabled with threshold: {args.gating_threshold}")
        print(f"Using {args.num_samples} samples with noise std: {args.sampling_std}")
    print(f"Model directory: {args.model_dir}")
    print(f"Data location: {args.data_location}")
    print(f"Output directory: {args.save_dir}")
    print(f"Datasets to evaluate: {args.datasets}")
    print(f"Test-only mode: {args.only_test}")
    print("=" * 30)

    # Overall results
    all_results = {}
    summary_results = []

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
                print(f"Skipping {dataset_name} due to dataset loading failure")
                continue

            try:
                # Find model path
                model_path = find_model_path(
                    args.model_dir,
                    args.model,
                    dataset_name,
                    debug=args.debug
                )
                print(f"Using model from: {model_path}")
            except FileNotFoundError as e:
                print(f"ERROR: {e}")
                print(f"Skipping evaluation for {dataset_name}")
                continue

            # Evaluate model
            results = evaluate_model(
                model_path,
                dataset,
                device,
                debug=args.debug
            )

            # Print results
            print(f"Accuracy: {results['accuracy'] * 100:.2f}%")
            print(f"Number of samples: {results['num_samples']}")
            print(f"Model type: {results['model_type']}")

            # Print uncertainty if available
            if 'uncertainty' in results:
                print(f"Prediction uncertainty: {results['uncertainty']:.6f}")

            # Print detailed results if verbose
            if args.verbose:
                print("\nPer-class accuracy:")
                per_class_report = results['per_class_report']
                per_class_report.sort(key=lambda x: x['accuracy'], reverse=True)

                for cls_data in per_class_report:
                    print(f"  {cls_data['class_name']}: {cls_data['accuracy'] * 100:.2f}% "
                          f"({cls_data['correct']}/{cls_data['total']})")

                print("\nConfidence statistics:")
                conf_stats = results['confidence_stats']
                print(f"  Mean: {conf_stats['mean']:.4f}")
                print(f"  Median: {conf_stats['median']:.4f}")
                print(f"  Min: {conf_stats['min']:.4f}")
                print(f"  Max: {conf_stats['max']:.4f}")
                print(f"  Std: {conf_stats['std']:.4f}")

            # Store results
            all_results[dataset_name] = results

            # Add to summary
            summary_entry = {
                'dataset': dataset_name,
                'accuracy': results['accuracy'],
                'samples': results['num_samples'],
                'model_path': model_path,
                'model_type': results['model_type']
            }

            # Add uncertainty if available
            if 'uncertainty' in results:
                summary_entry['uncertainty'] = results['uncertainty']

            summary_results.append(summary_entry)

        except Exception as e:
            print(f"Error evaluating model for {dataset_name}: {e}")
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

        # Calculate average uncertainty if available
        if any('uncertainty' in r for r in summary_results):
            uncertainties = [r['uncertainty'] for r in summary_results if 'uncertainty' in r]
            if uncertainties:
                avg_uncertainty = sum(uncertainties) / len(uncertainties)
                all_results['average_uncertainty'] = avg_uncertainty
                print(f"Average uncertainty across all datasets: {avg_uncertainty:.6f}")

    # Save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        args.save_dir,
        f"evaluation_{args.model}{config_suffix}_{timestamp}.json"
    )

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"\nAll evaluation results saved to {results_path}")

    # Print summary
    print("\n=== Summary ===")
    if args.use_gating and args.num_samples > 1:
        print(f"{'Dataset':<15} {'Accuracy':<10} {'Uncertainty':<12} {'Model Type':<20}")
        print("-" * 60)
    else:
        print(f"{'Dataset':<15} {'Accuracy':<10} {'Model Type':<20}")
        print("-" * 45)

    for result in sorted(summary_results, key=lambda x: x['dataset']):
        if 'uncertainty' in result and args.use_gating and args.num_samples > 1:
            print(f"{result['dataset']:<15} {result['accuracy'] * 100:.2f}% {result['uncertainty']:<12.6f} {result['model_type']:<20}")
        else:
            print(f"{result['dataset']:<15} {result['accuracy'] * 100:.2f}% {result['model_type']:<20}")


if __name__ == "__main__":
    main()