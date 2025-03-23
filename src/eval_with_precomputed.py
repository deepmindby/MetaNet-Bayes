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


class TestOnlyFeatures:
    """Dataset container class for test-only pre-computed features"""

    def __init__(self, feature_dir, batch_size=128, num_workers=4):
        """Initialize with directory containing pre-computed features

        Args:
            feature_dir: Path to directory with pre-computed features
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
        """
        # Verify directory exists
        if not os.path.exists(feature_dir):
            raise FileNotFoundError(f"Feature directory not found: {feature_dir}")

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
                print(f"Found test features at: {test_features_path}")
                print(f"Found test labels at: {test_labels_path}")
                break

        if test_features_path is None:
            raise FileNotFoundError(f"Could not find test features in {feature_dir}")

        # Load test features and labels
        try:
            self.test_features = torch.load(test_features_path)
            print(f"Successfully loaded test features, shape: {self.test_features.shape}")

            self.test_labels = torch.load(test_labels_path)
            print(f"Successfully loaded test labels, shape: {self.test_labels.shape}")

            # Validate that features and labels have matching sizes
            if len(self.test_features) != len(self.test_labels):
                raise ValueError(f"Features ({len(self.test_features)}) and labels ({len(self.test_labels)}) count mismatch")
        except Exception as e:
            print(f"Error loading test features/labels: {e}")
            traceback.print_exc()
            raise

        # Create a simple dataset for the test data
        from torch.utils.data import Dataset, DataLoader

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

        self.test_dataset = SimpleDataset(self.test_features, self.test_labels)

        # Create test loader
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        # Load classnames if available
        classnames_path = os.path.join(feature_dir, "classnames.txt")
        if os.path.exists(classnames_path):
            with open(classnames_path, "r") as f:
                self.classnames = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(self.classnames)} class names from {classnames_path}")
        else:
            # Create dummy classnames if file doesn't exist
            unique_labels = torch.unique(self.test_labels)
            self.classnames = [f"class_{i}" for i in range(len(unique_labels))]
            print(f"Created {len(self.classnames)} dummy class names")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate models with pre-computed features (test-only)")
    parser.add_argument("--model", type=str, default="ViT-B-32",
                        help="CLIP model used for feature extraction")
    parser.add_argument("--data-location", type=str,
                        default=os.path.expanduser("/home/haichao/zby/MetaNet-Bayes"),
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
                        default=["MNIST", "SUN397", "SVHN", "EuroSAT", "GTSRB", "DTD", "Cars"],
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
    return parser.parse_args()


def get_test_dataset(dataset_name, model_name, location, batch_size=128, num_workers=4, debug=False):
    """Get dataset with pre-computed test features only

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
    try:
        # Build possible feature directory paths
        possible_dirs = []

        # Clean dataset name if it has "precomputed_" prefix
        clean_name = dataset_name
        if dataset_name.startswith("precomputed_"):
            clean_name = dataset_name[len("precomputed_"):]

        # Try with and without "Val" suffix
        for name in [clean_name, f"{clean_name}Val", clean_name.replace("Val", "")]:
            # Try different directory structures
            possible_dirs.extend([
                # Standard structure: model/dataset
                os.path.join(location, "precomputed_features", model_name, name),
                # Flat structure directly in precomputed_features
                os.path.join(location, "precomputed_features", name),
                # Alternative structures
                os.path.join(location, model_name, name),
                os.path.join(location, "features", model_name, name),
                os.path.join(location, "features", name),
                # Test directory directly
                os.path.join(location, name, "test"),
            ])

        # Try each directory
        for feature_dir in possible_dirs:
            if debug:
                print(f"Checking directory: {feature_dir}")

            if os.path.exists(feature_dir):
                if debug:
                    print(f"Found directory at: {feature_dir}")
                try:
                    return TestOnlyFeatures(
                        feature_dir=feature_dir,
                        batch_size=batch_size,
                        num_workers=num_workers
                    )
                except FileNotFoundError:
                    if debug:
                        print(f"No test features found in {feature_dir}")
                    continue
                except Exception as e:
                    print(f"Error loading from {feature_dir}: {e}")
                    if debug:
                        traceback.print_exc()
                    continue

        print(f"WARNING: Could not find test features for {dataset_name} in any expected location")
        for dir in possible_dirs:
            print(f"  - Tried: {dir}")
        return None

    except Exception as e:
        print(f"Error searching for {dataset_name} test features: {e}")
        if debug:
            traceback.print_exc()
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
            (model_dir, model_name, name, "best_precomputed_model.pt"),
            (model_dir, name, "best_precomputed_model.pt"),
            # With model type directories
            (f"{model_dir}-causal", model_name, name, "best_precomputed_model.pt"),
            (f"{model_dir}-meta", model_name, name, "best_precomputed_model.pt"),
            # Alternative file names
            (model_dir, model_name, name, "best_model.pt"),
            (model_dir, model_name, name, "model.pt"),
            (model_dir, name, "best_model.pt"),
            (model_dir, name, "model.pt"),
            # Check in named subdirectories
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
        if debug:
            print(f"Using parameters: feature_dim={feature_dim}, "
                f"num_task_vectors={num_task_vectors}, blockwise={blockwise}, "
                f"enable_causal={enable_causal}, top_k_ratio={top_k_ratio}")

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
            'top_k_ratio': top_k_ratio
        },
        'model_path': model_path,
        'evaluation_timestamp': datetime.now().isoformat()
    }

    if debug:
        print(f"Evaluation complete. Accuracy: {accuracy * 100:.2f}%")

    return results


def main():
    """Main evaluation function"""
    global args
    args = parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create results directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Generate a descriptive suffix for results based on configuration
    config_suffix = "_test_only"
    if args.blockwise_coef:
        config_suffix += "_blockwise"
    if args.causal_intervention:
        config_suffix += "_causal"

    # Print configuration
    print(f"\n=== Evaluation Configuration ===")
    print(f"Model: {args.model}")
    print(f"Using blockwise coefficients: {args.blockwise_coef}")
    print(f"Using causal intervention: {args.causal_intervention}")
    print(f"Top-k ratio: {args.top_k_ratio}")
    print(f"Number of task vectors: {args.num_task_vectors}")
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
                    num_workers=args.num_workers
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
            summary_results.append({
                'dataset': dataset_name,
                'accuracy': results['accuracy'],
                'samples': results['num_samples'],
                'model_path': model_path,
            })

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
    print(f"{'Dataset':<15} {'Accuracy':<10}")
    print("-" * 25)

    for result in sorted(summary_results, key=lambda x: x['dataset']):
        print(f"{result['dataset']:<15} {result['accuracy'] * 100:.2f}%")


if __name__ == "__main__":
    main()