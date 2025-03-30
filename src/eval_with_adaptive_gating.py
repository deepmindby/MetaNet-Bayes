"""Evaluation Script for Adaptive Gating MetaNet Models

This script evaluates Adaptive Gating MetaNet models using pre-computed features,
with enhanced support for augmented data and detailed performance reporting.
Also supports evaluation of models trained without gating mechanism or MetaNet.
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

from src.args import parse_arguments
from src.adaptive_gating_metanet import AdaptiveGatingMetaNet


class DirectFeatureModel(torch.nn.Module):
    """
    A direct feature model that passes features directly to classifier without MetaNet transformations.
    This implements the Atlas approach using precomputed features.
    """
    def __init__(self, feature_dim):
        """Initialize DirectFeatureModel

        Parameters:
        ----------
        feature_dim: int
            Dimension of the pre-computed feature vectors
        """
        super().__init__()
        self.feature_dim = feature_dim
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

    def forward(self, features):
        """Forward pass simply passes features through

        Parameters:
        ----------
        features: Tensor [batch_size, feature_dim]
            Pre-computed feature vectors

        Returns:
        ----------
        features: Tensor [batch_size, feature_dim]
            Same features, unchanged (dummy computation has no effect)
        """
        # Apply the identity projection + add scaled dummy computation (scaling factor = 0)
        return self.projection(features) + self.dummy_linear(features) * 0.0

    def uncertainty_regularization_loss(self):
        """Dummy method to match AdaptiveGatingMetaNet interface

        Returns:
        ----------
        loss: Tensor
            Zero tensor for compatibility
        """
        # Return a small loss based on dummy parameter to ensure it receives gradient
        return self.dummy_param.sum() * 0.0  # Multiply by 0 to not affect actual training

    def get_gating_stats(self):
        """Dummy method to match AdaptiveGatingMetaNet interface

        Returns:
        ----------
        stats: dict
            Empty dictionary for compatibility
        """
        return {}


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


def find_model_path(model_dir, dataset_name, no_gating=False, no_metanet=False, debug=False):
    """Find the model path for a given dataset, supporting different model types

    Args:
        model_dir: Directory containing trained models
        dataset_name: Name of the dataset
        no_gating: Whether to look for a model trained without gating
        no_metanet: Whether to look for a model trained without MetaNet
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
        gating_suffix = "_atlas"
    elif no_gating:
        gating_suffix = "_no_gating"
    else:
        gating_suffix = "_adaptive_gating"

    # Check these paths in order
    possible_paths = [
        os.path.join(model_dir, val_name, f"best{gating_suffix}_model.pt"),
        os.path.join(model_dir, val_name, f"best_{gating_suffix.strip('_')}_model.pt"),
        os.path.join(model_dir, base_name, f"best{gating_suffix}_model.pt"),
        os.path.join(model_dir, base_name, f"best_{gating_suffix.strip('_')}_model.pt"),
    ]

    # Add standard paths that might be used as fallbacks
    possible_paths.extend([
        os.path.join(model_dir, val_name, f"best_precomputed_model.pt"),
        os.path.join(model_dir, base_name, f"best_precomputed_model.pt"),
        os.path.join(model_dir, val_name, f"best_model.pt"),
        os.path.join(model_dir, base_name, f"best_model.pt"),
    ])

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

    model_type = "Atlas" if no_metanet else "no-gating" if no_gating else "gating"
    raise FileNotFoundError(f"Could not find {model_type} model for {dataset_name} in {model_dir}")


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

        # Check for no_metanet flag in config
        no_metanet = config.get('no_metanet', False)

        if no_metanet:
            # Atlas model
            print(f"Detected Atlas model (no MetaNet)")
            if args.debug:
                print(f"Creating DirectFeatureModel")
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
                print(f"Detected model trained without gating mechanism")
                no_gating = True
            else:
                # Print configuration for model identification
                print(f"Model parameters: αT={base_threshold:.4f}, β={beta:.4f}")
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

        # Use provided arguments for model type
        no_metanet = args.no_metanet

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
                print(f"Using model without gating mechanism")
            else:
                print(f"Using default parameters: αT={base_threshold:.4f}, β={beta:.4f}")
        else:
            print(f"Using Atlas implementation (direct features)")

    # Create the appropriate model based on detected or specified type
    try:
        if args.debug:
            if no_metanet:
                print("Creating DirectFeatureModel instance...")
            else:
                print("Creating AdaptiveGatingMetaNet instance...")

        if no_metanet or args.no_metanet:
            # Create Atlas model
            model = DirectFeatureModel(feature_dim=feature_dim)
            if args.debug:
                print(f"Created DirectFeatureModel for Atlas")
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
    }

    if not model_info['no_metanet']:
        model_info.update({
            'no_gating': no_gating if 'no_gating' in locals() else args.no_gating,
            'base_threshold': getattr(model, 'base_threshold', torch.tensor(0.0)).item()
                if hasattr(model, 'base_threshold') else 0.0,
            'beta': getattr(model, 'beta', torch.tensor(0.0)).item()
                if hasattr(model, 'beta') else 0.0,
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
        model_type = "Atlas"
    elif model_info['no_gating']:
        model_type = "MetaNet_NoGating"
    else:
        model_type = "AdaptiveGating"

    if not model_info['no_metanet'] and 'blockwise' in locals() and blockwise:
        model_type += "_Blockwise"

    # Get gating stats if applicable
    gating_stats = None
    if not model_info['no_metanet'] and not model_info['no_gating'] and hasattr(model, 'get_gating_stats'):
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
            'no_metanet': model_info['no_metanet'],
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
            'no_gating': model_info['no_gating'],
        })

        # Add computed gating ratio for MetaNet models
        if not model_info['no_gating'] and hasattr(model, 'get_gating_stats'):
            gating_ratio = gating_stats.get('gating_ratio', 0.0) if gating_stats else 0.0
            results['computed_gating_ratio'] = gating_ratio

    # Add gating stats if available
    if gating_stats is not None and not model_info['no_metanet'] and not model_info['no_gating']:
        results['gating_stats'] = {
            'gating_ratio': gating_stats.get('gating_ratio', 0.0),
            'avg_threshold': gating_stats.get('avg_threshold', 0.0),
            'base_threshold': gating_stats.get('base_threshold', 0.0),
            'beta': gating_stats.get('beta', 0.0)
        }

    if args.debug:
        print(f"Evaluation complete. Accuracy: {accuracy * 100:.2f}%")

        if not model_info['no_metanet'] and not model_info['no_gating'] and gating_stats is not None:
            print(f"Gating ratio: {gating_stats.get('gating_ratio', 0.0):.4f}")

    return results


def main():
    """Main evaluation function"""
    # Parse arguments using the updated args.py
    args = parse_arguments()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup directories with simplified logic
    save_dir = os.path.join(args.save_dir, "evaluation_results")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")

    # Generate descriptive suffix for results
    if args.no_metanet:
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
                    no_gating=args.no_gating,
                    no_metanet=args.no_metanet,
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
            elif results['config']['no_metanet']:
                print(f"Model type: Atlas (direct features)")
            else:
                print(f"Model type: MetaNet without gating")

            # Detailed output if requested
            if args.verbose:
                if 'gating_stats' in results and not results['config']['no_metanet'] and not results['config'].get('no_gating', False):
                    gating = results['gating_stats']
                    print(f"Gating details - ratio: {gating['gating_ratio']:.4f}, threshold: {gating['avg_threshold']:.4f}")

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

    # Save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        save_dir,
        f"evaluation_{args.model}{config_suffix}_{timestamp}.json"
    )

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"\nAll evaluation results saved to: {results_path}")

    # Print summary with improved alignment and separate sections for different model types
    print("\n" + "=" * 80)
    if args.no_metanet:
        model_type_str = "Atlas (No MetaNet)"
    elif args.no_gating:
        model_type_str = "MetaNet (No Gating)"
    else:
        model_type_str = "Adaptive Gating MetaNet"

    print(f"Summary of {model_type_str} Models")
    print("-" * 80)

    # Group summary results by model type
    atlas_results = [r for r in summary_results if r.get('no_metanet', False)]
    metanet_no_gating_results = [r for r in summary_results if not r.get('no_metanet', False) and r.get('no_gating', False)]
    adaptive_gating_results = [r for r in summary_results if not r.get('no_metanet', False) and not r.get('no_gating', False)]

    # Print Atlas results
    if atlas_results:
        print("\nAtlas (No MetaNet) Results:")
        print(f"{'Dataset':<15} | {'Accuracy':^10} | {'Model Type':^25}")
        print("-" * 58)

        for result in sorted(atlas_results, key=lambda x: x['dataset']):
            # Format fields with precise spacing and alignment
            dataset_field = f"{result['dataset']:<15}"
            accuracy_field = f"{result['accuracy']*100:>8.2f}%"
            model_type_field = f"{result['model_type']:<25}"

            # Print with strict alignment using separators
            print(f"{dataset_field} | {accuracy_field:^10} | {model_type_field:^25}")

    # Print MetaNet without gating results
    if metanet_no_gating_results:
        print("\nMetaNet (No Gating) Results:")
        print(f"{'Dataset':<15} | {'Accuracy':^10} | {'Model Type':^25}")
        print("-" * 58)

        for result in sorted(metanet_no_gating_results, key=lambda x: x['dataset']):
            # Format fields with precise spacing and alignment
            dataset_field = f"{result['dataset']:<15}"
            accuracy_field = f"{result['accuracy']*100:>8.2f}%"
            model_type_field = f"{result['model_type']:<25}"

            # Print with strict alignment using separators
            print(f"{dataset_field} | {accuracy_field:^10} | {model_type_field:^25}")

    # Print Adaptive Gating results
    if adaptive_gating_results:
        print("\nAdaptive Gating MetaNet Results:")
        print(f"{'Dataset':<15} | {'Accuracy':^10} | {'αT':^8} | {'β':^8} | {'Gating %':^10} | {'Model Type':^25}")
        print("-" * 85)

        for result in sorted(adaptive_gating_results, key=lambda x: x['dataset']):
            # Format fields with precise spacing and alignment
            dataset_field = f"{result['dataset']:<15}"
            accuracy_field = f"{result['accuracy']*100:>8.2f}%"
            alpha_field = f"{result['alpha']:>7.4f}"
            beta_field = f"{result['beta']:>7.4f}"
            gating_field = f"{result.get('gating_ratio', 0)*100:>8.2f}%" if 'gating_ratio' in result else "   N/A   "
            model_type_field = f"{result['model_type']:<25}"

            # Print with strict alignment using separators
            print(f"{dataset_field} | {accuracy_field:^10} | {alpha_field:^8} | {beta_field:^8} | {gating_field:^10} | {model_type_field:^25}")

    print("=" * 80)


if __name__ == "__main__":
    main()