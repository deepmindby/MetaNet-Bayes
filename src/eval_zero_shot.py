"""Zero-shot Evaluation Script Using Pre-computed Features

This script evaluates the zero-shot performance of CLIP models using
pre-computed features without any fine-tuning.
"""

import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
import open_clip
from collections import defaultdict
from datetime import datetime
import gc
import traceback


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


def get_test_dataset(dataset_name, model_name, location, batch_size=128, num_workers=4, debug=False):
    """Get dataset with pre-computed test features"""
    # Handle the case where dataset ends with "Val"
    base_name = dataset_name
    if dataset_name.endswith("Val"):
        base_name = dataset_name[:-3]

    # Try model-specific paths first
    feature_dir = os.path.join(location, "precomputed_features", model_name, dataset_name)

    # Try alternative paths if needed
    if not os.path.exists(feature_dir):
        feature_dir = os.path.join(location, "precomputed_features", model_name, base_name)

    if not os.path.exists(feature_dir):
        feature_dir = os.path.join(location, "precomputed_features", model_name, base_name + "Val")

    # Fall back to generic paths if model-specific ones don't exist
    if not os.path.exists(feature_dir):
        feature_dir = os.path.join(location, "precomputed_features", dataset_name)

    if not os.path.exists(feature_dir):
        feature_dir = os.path.join(location, "precomputed_features", base_name)

    if not os.path.exists(feature_dir):
        feature_dir = os.path.join(location, "precomputed_features", base_name + "Val")

    if not os.path.exists(feature_dir):
        raise FileNotFoundError(f"Features for {dataset_name} with model {model_name} not found in {location}")

    # Create dataset
    return TestOnlyFeatures(
        feature_dir=feature_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        verbose=debug
    )


def evaluate_zero_shot(dataset, device, args):
    """Evaluate using zero-shot approach"""
    # Load CLIP model (just text encoder - we'll use precomputed image features)
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained='openai')
    model = model.to(device)
    model.eval()

    # Get text features for class names
    classnames = dataset.classnames

    # Prepare text inputs by formatting class names
    if args.prompt_ensemble:
        # Use a prompt ensemble for better zero-shot performance
        templates = [
            "a photo of a {}.",
            "a picture of a {}.",
            "an image of a {}.",
            "a {} in the image.",
            "a photo of the {}.",
            "a close-up photo of a {}.",
            "a rendition of a {}.",
            "a nice {} in the scene."
        ]

        text_features_list = []
        for template in templates:
            prompts = [template.format(name) for name in classnames]
            text_tokens = open_clip.tokenize(prompts).to(device)

            with torch.no_grad():
                template_text_features = model.encode_text(text_tokens)
                template_text_features /= template_text_features.norm(dim=-1, keepdim=True)
                text_features_list.append(template_text_features)

        # Average the text features across all templates
        text_features = torch.stack(text_features_list).mean(dim=0)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    else:
        # Use a single template
        template = "a photo of a {}."
        prompts = [template.format(name) for name in classnames]
        text_tokens = open_clip.tokenize(prompts).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

    # Evaluate
    correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    with torch.no_grad():
        for batch in tqdm(dataset.test_loader, desc="Evaluating"):
            # Get features and labels
            if isinstance(batch, dict):
                features = batch["features"].to(device)
                labels = batch["labels"].to(device)
            else:
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)

            # Normalize image features
            features /= features.norm(dim=-1, keepdim=True)

            # Compute similarity
            similarity = features @ text_features.T

            # Get predictions
            _, predicted = torch.max(similarity, dim=1)

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

    # Calculate overall accuracy
    accuracy = correct / total if total > 0 else 0.0

    # Calculate per-class accuracy
    per_class_acc = {}
    for cls_idx in range(len(dataset.classnames)):
        cls_name = dataset.classnames[cls_idx]
        if per_class_total[cls_idx] > 0:
            cls_acc = per_class_correct[cls_idx] / per_class_total[cls_idx]
            per_class_acc[cls_name] = float(cls_acc)

    results = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'per_class_accuracy': per_class_acc,
        'model_name': args.model  # Include model name for reference
    }

    return results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Zero-shot evaluation using pre-computed features")
    parser.add_argument("--model", type=str, default="ViT-B-32",
                        help="CLIP model to use (e.g. ViT-B-32)")
    parser.add_argument("--data-location", type=str,
                        default=os.path.expanduser("~/MetaNet-Bayes"),
                        help="Root directory for datasets")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of worker threads for data loading")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"],
                        help="Datasets to evaluate")
    parser.add_argument("--prompt-ensemble", action="store_true", default=False,
                        help="Use multiple prompts and average results")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug mode")
    parser.add_argument("--save-dir", type=str, default="zero_shot_results",
                        help="Directory to save evaluation results")
    return parser.parse_args()


def main():
    """Main evaluation function"""
    args = parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model-specific save directory
    model_save_dir = os.path.join(args.save_dir, args.model)
    os.makedirs(model_save_dir, exist_ok=True)
    print(f"Results will be saved to: {model_save_dir}")

    # Print configuration
    print(f"\n=== Zero-shot Evaluation Configuration ===")
    print(f"Model: {args.model}")
    print(f"Datasets to evaluate: {args.datasets}")
    print(f"Prompt ensemble: {args.prompt_ensemble}")
    print("=" * 50)

    # Overall results
    all_results = {}
    summary_results = []

    for dataset_name in args.datasets:
        print(f"Evaluating dataset: {dataset_name}")
        dataset = None

        try:
            # Get dataset with precomputed features
            dataset = get_test_dataset(
                dataset_name=dataset_name,
                model_name=args.model,
                location=args.data_location,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                debug=args.debug
            )

            # Evaluate using zero-shot approach
            results = evaluate_zero_shot(dataset, device, args)

            # Print results
            print(f"Zero-shot accuracy: {results['accuracy'] * 100:.2f}% ({results['correct']}/{results['total']})")

            # Add to results
            all_results[dataset_name] = results
            summary_results.append({
                'dataset': dataset_name,
                'accuracy': results['accuracy'],
                'correct': results['correct'],
                'total': results['total'],
                'model': args.model  # Store model name
            })

        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            if args.debug:
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

        # Save results with model name in filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"zero_shot_results_{args.model}_{timestamp}.json"
        results_path = os.path.join(model_save_dir, results_filename)

        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=4)

        print(f"\nResults saved to: {results_path}")

        # Print summary table
        print("\n=== Zero-shot Results Summary ===")
        print(f"{'Dataset':<15} | {'Accuracy':<10} | {'Model':<12}")
        print("-" * 43)

        for result in sorted(summary_results, key=lambda x: x['dataset']):
            print(f"{result['dataset']:<15} | {result['accuracy'] * 100:>8.2f}% | {result['model']:<12}")

        print("-" * 43)
        print(f"{'Average':<15} | {avg_accuracy * 100:>8.2f}% | {args.model:<12}")
        print("=" * 50)


if __name__ == "__main__":
    main()