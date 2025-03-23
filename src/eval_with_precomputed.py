"""Evaluation Script Using Pre-computed Features

This script evaluates MetaNet models using pre-computed CLIP features,
which significantly accelerates evaluation by avoiding the forward pass
through the CLIP image encoder.
"""

import os
import json
import torch
import argparse
import numpy as np
import traceback
from tqdm import tqdm
import gc
from src.metanet_precomputed import PrecomputedMetaNet


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate models with pre-computed features")
    parser.add_argument("--model", type=str, default="ViT-B-32",
                        help="CLIP model used for feature extraction")
    parser.add_argument("--data-location", type=str,
                        default=os.path.expanduser("/home/haichao/zby/MetaNet-Bayes"),
                        help="Root directory for datasets")
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
    # Optional parameters - will be loaded from model if not provided
    parser.add_argument("--blockwise-coef", action="store_true", default=False,
                        help="Whether blockwise coefficients were used")
    parser.add_argument("--causal_intervention", action="store_true", default=False,
                        help="Whether causal intervention was used")
    parser.add_argument("--top_k_ratio", type=float, default=0.1,
                        help="Ratio of parameter blocks used for intervention")
    parser.add_argument("--num-task-vectors", type=int, default=8,
                        help="Number of task vectors used")
    return parser.parse_args()


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

    # Build feature directory path - use direct path to precomputed_features
    feature_dir = os.path.join(location, "precomputed_features", model_name, dataset_name)

    # Check if features exist
    if not os.path.exists(feature_dir):
        raise FileNotFoundError(f"Pre-computed features not found at {feature_dir}")

    # Create and return dataset with limited workers to avoid resource issues
    safe_num_workers = min(4, num_workers)
    return PrecomputedFeatures(
        feature_dir=feature_dir,
        batch_size=batch_size,
        num_workers=safe_num_workers,
        persistent_workers=False
    )


def cleanup_data_resources(dataset):
    """Cleanup data resources to prevent memory leaks"""
    if dataset is None:
        return

    try:
        # Explicitly close DataLoader iterators
        for loader_name in ['train_loader', 'test_loader']:
            if hasattr(dataset, loader_name):
                loader = getattr(dataset, loader_name)
                # Remove iterator to force cleanup
                if hasattr(loader, '_iterator'):
                    loader._iterator = None

        # Clear dataset references
        dataset.train_loader = None
        dataset.test_loader = None
        dataset.train_dataset = None
        dataset.test_dataset = None
    except Exception as e:
        print(f"Warning during dataset cleanup: {e}")

    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_model(model_path, dataset, device):
    """Evaluate model on dataset

    Args:
        model_path: path to saved model
        dataset: evaluation dataset
        device: computation device

    Returns:
        dict: evaluation metrics
    """
    # Load model state
    try:
        state_dict = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        traceback.print_exc()
        raise

    # Extract model configuration from state dict if available
    if 'config' in state_dict:
        print(f"Loading model configuration from checkpoint")
        config = state_dict['config']
        feature_dim = config.get('feature_dim')
        num_task_vectors = config.get('num_task_vectors', 8)
        blockwise = config.get('blockwise', False)
        enable_causal = config.get('enable_causal', False)
        top_k_ratio = config.get('top_k_ratio', 0.1)
        print(f"Model configuration: feature_dim={feature_dim}, "
              f"num_task_vectors={num_task_vectors}, blockwise={blockwise}, "
              f"enable_causal={enable_causal}, top_k_ratio={top_k_ratio}")
    else:
        # Get feature dimension from dataset if not in config
        print("No configuration found in model, using command line arguments or defaults")
        sample_batch = next(iter(dataset.test_loader))
        if isinstance(sample_batch, dict):
            feature_dim = sample_batch["features"].shape[1]
        else:
            feature_dim = sample_batch[0].shape[1]

        # Use provided or default parameters
        num_task_vectors = args.num_task_vectors
        blockwise = args.blockwise_coef
        enable_causal = args.causal_intervention
        top_k_ratio = args.top_k_ratio
        print(f"Using parameters: feature_dim={feature_dim}, "
              f"num_task_vectors={num_task_vectors}, blockwise={blockwise}, "
              f"enable_causal={enable_causal}, top_k_ratio={top_k_ratio}")

    # Create model with appropriate configuration
    model = PrecomputedMetaNet(
        feature_dim=feature_dim,
        task_vectors=num_task_vectors,
        blockwise=blockwise,
        enable_causal=enable_causal,
        top_k_ratio=top_k_ratio
    )

    # Load model weights
    if 'meta_net' in state_dict:
        model.load_state_dict(state_dict['meta_net'])
    else:
        # Backward compatibility for old models
        model_state_keys = [k for k in state_dict.keys() if k.startswith('meta_net.')]
        if model_state_keys:
            # Extract meta_net parameters
            meta_net_state = {k[9:]: state_dict[k] for k in model_state_keys}
            model.load_state_dict(meta_net_state)
        else:
            raise ValueError("Could not find meta_net parameters in model state dict")

    model = model.to(device)

    # Create classifier
    num_classes = len(dataset.classnames)
    classifier = torch.nn.Linear(feature_dim, num_classes)

    # Load classifier weights
    if 'classifier' in state_dict:
        classifier.load_state_dict(state_dict['classifier'])
    else:
        # Backward compatibility for old models
        classifier_state_keys = [k for k in state_dict.keys() if k.startswith('classifier.')]
        if classifier_state_keys:
            # Extract classifier parameters
            classifier_state = {k[11:]: state_dict[k] for k in classifier_state_keys}
            classifier.load_state_dict(classifier_state)
        else:
            raise ValueError("Could not find classifier parameters in model state dict")

    classifier = classifier.to(device)

    # Evaluation
    model.eval()
    classifier.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

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

            # Compute accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = correct / total

    # Calculate per-class accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    per_class_acc = {}

    for cls_idx in range(len(dataset.classnames)):
        cls_mask = (all_labels == cls_idx)
        if np.sum(cls_mask) > 0:
            cls_acc = np.mean(all_preds[cls_mask] == cls_idx)
            per_class_acc[dataset.classnames[cls_idx]] = float(cls_acc)

    return {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_acc,
        'num_samples': total,
        'config': {
            'feature_dim': feature_dim,
            'num_task_vectors': num_task_vectors,
            'blockwise': blockwise,
            'enable_causal': enable_causal,
            'top_k_ratio': top_k_ratio
        }
    }


def main():
    """Main evaluation function"""
    args = parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create results directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Generate a descriptive suffix for results based on configuration
    config_suffix = "_standard"
    if args.blockwise_coef:
        config_suffix += "_blockwise"
    if args.causal_intervention:
        config_suffix += "_causal"

    # Overall results
    all_results = {}

    for dataset_name in args.datasets:
        print(f"\n=== Evaluating on {dataset_name} ===")
        dataset = None

        try:
            # Get dataset with precomputed features
            dataset = get_precomputed_dataset(
                dataset_name=dataset_name,
                model_name=args.model,
                location=args.data_location,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )

            # Find model path - check if Val suffix is needed
            base_model_path = os.path.join(
                args.model_dir,
                args.model,
                f"{dataset_name}Val",
                "best_precomputed_model.pt"
            )

            # If the path with Val doesn't exist, try without Val
            if not os.path.exists(base_model_path):
                alt_model_path = os.path.join(
                    args.model_dir,
                    args.model,
                    dataset_name,
                    "best_precomputed_model.pt"
                )
                if os.path.exists(alt_model_path):
                    model_path = alt_model_path
                else:
                    print(f"Model not found at {base_model_path} or {alt_model_path}, skipping.")
                    continue
            else:
                model_path = base_model_path

            print(f"Using model from: {model_path}")

            # Evaluate model
            results = evaluate_model(model_path, dataset, device)

            # Print results
            print(f"Accuracy: {results['accuracy'] * 100:.2f}%")
            print(f"Number of samples: {results['num_samples']}")

            # Print model configuration
            config = results['config']
            print(f"Model configuration:")
            print(f"  Feature dimension: {config['feature_dim']}")
            print(f"  Number of task vectors: {config['num_task_vectors']}")
            print(f"  Blockwise coefficients: {config['blockwise']}")
            print(f"  Causal intervention: {config['enable_causal']}")
            print(f"  Top-k ratio: {config['top_k_ratio']}")

            # Store results
            all_results[dataset_name] = results

        except Exception as e:
            print(f"Error evaluating model for {dataset_name}: {e}")
            traceback.print_exc()
        finally:
            # Clean up dataset resources
            cleanup_data_resources(dataset)
            torch.cuda.empty_cache()
            gc.collect()

    # Save all results
    results_path = os.path.join(args.save_dir, f"evaluation_{args.model}{config_suffix}.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"\nAll evaluation results saved to {results_path}")

    # Print summary
    print("\n=== Summary ===")
    print(f"{'Dataset':<15} {'Accuracy':<10}")
    print("-" * 25)

    for dataset_name, results in all_results.items():
        print(f"{dataset_name:<15} {results['accuracy'] * 100:.2f}%")


if __name__ == "__main__":
    main()