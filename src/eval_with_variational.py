"""
Evaluation script for Spike-and-Slab Variational MetaNet models.

This script provides evaluation capabilities for Spike-and-Slab MetaNet models,
focusing on uncertainty metrics, sparsity analysis, and reliability diagrams.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
import gc
import traceback
import math
from sklearn.calibration import calibration_curve

from src.variational_metanet import SpikeAndSlabMetaNet
from src.args import parse_arguments

# Parse arguments
args = parse_arguments()

# Add variational-specific arguments
args.num_eval_samples = 20  # Number of MC samples for evaluation
args.save_uncertainty = True  # Whether to save reliability diagram
args.detailed_analysis = True  # Enable detailed analysis
args.temperature = 0.1  # Temperature for Gumbel-Softmax during evaluation


def convert_to_python_types(obj):
    """Convert numpy or torch types to native Python types.

    Args:
        obj: Object that may contain numpy or torch types

    Returns:
        Object with numpy/torch types converted to Python native types
    """
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        if obj.size == 1:  # Single element array
            return convert_to_python_types(obj.item())
        return [convert_to_python_types(x) for x in obj.tolist()]
    elif isinstance(obj, torch.Tensor):
        if obj.numel() == 1:  # Single element tensor
            return float(obj.item())
        return convert_to_python_types(obj.detach().cpu().numpy())
    elif isinstance(obj, list):
        return [convert_to_python_types(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_python_types(x) for x in obj)
    elif isinstance(obj, dict):
        return {convert_to_python_types(k): convert_to_python_types(v) for k, v in obj.items()}
    else:
        return obj


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


def find_model_path(model_dir, dataset_name, model_name, debug=False):
    """Find the Spike-and-Slab variational model path for a given dataset

    Args:
        model_dir: Directory containing trained models
        dataset_name: Name of the dataset
        model_name: Name of the model (e.g., ViT-B-32)
        debug: Whether to print debug information

    Returns:
        str: Path to the model file
    """
    # Base name handling
    base_name = dataset_name
    if dataset_name.endswith("Val"):
        base_name = dataset_name[:-3]

    val_name = f"{base_name}Val"

    # Model-specific directory
    model_specific_dir = os.path.join(model_dir, model_name)

    # Check if model-specific directory exists
    if os.path.exists(model_specific_dir):
        model_dir = model_specific_dir
        if debug:
            print(f"Using model-specific directory: {model_dir}")

    # Check these paths in order with various naming conventions
    possible_paths = [
        # Primary Spike-and-Slab model paths
        os.path.join(model_dir, val_name, "best_spike_and_slab_model.pt"),
        os.path.join(model_dir, base_name, "best_spike_and_slab_model.pt"),

        # Secondary variational model paths
        os.path.join(model_dir, val_name, "best_variational_model.pt"),
        os.path.join(model_dir, base_name, "best_variational_model.pt"),

        # Fallback paths
        os.path.join(model_dir, val_name, "best_precomputed_model.pt"),
        os.path.join(model_dir, base_name, "best_precomputed_model.pt"),
        os.path.join(model_dir, val_name, "best_model.pt"),
        os.path.join(model_dir, base_name, "best_model.pt"),
    ]

    # Also try in top-level directories (for backward compatibility)
    if model_dir != model_specific_dir:
        top_level_paths = [
            os.path.join(model_specific_dir, val_name, "best_spike_and_slab_model.pt"),
            os.path.join(model_specific_dir, base_name, "best_spike_and_slab_model.pt"),
            os.path.join(model_specific_dir, val_name, "best_variational_model.pt"),
            os.path.join(model_specific_dir, base_name, "best_variational_model.pt"),
            os.path.join(model_specific_dir, val_name, "best_precomputed_model.pt"),
            os.path.join(model_specific_dir, base_name, "best_precomputed_model.pt"),
        ]
        possible_paths.extend(top_level_paths)

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

    raise FileNotFoundError(
        f"Could not find Spike-and-Slab variational model for {dataset_name} (model: {model_name}) in {model_dir}")


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Calculate Expected Calibration Error (ECE) for multi-class problems.

    Parameters:
    ----------
    y_true: np.ndarray or torch.Tensor
        Ground truth labels
    y_prob: np.ndarray or torch.Tensor
        Predicted probabilities
    n_bins: int
        Number of bins for ECE calculation

    Returns:
    ----------
    ece: float
        Expected Calibration Error
    bin_accs: list
        Accuracy for each bin
    bin_confs: list
        Confidence for each bin
    bin_counts: list
        Sample counts for each bin
    """
    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.detach().cpu().numpy()

    # For multi-class, get the confidence of the predicted class
    if y_prob.ndim > 1 and y_prob.shape[1] > 1:
        confidence = np.max(y_prob, axis=1)
        pred_labels = np.argmax(y_prob, axis=1)

        # Convert one-hot labels to class indices if needed
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
    else:
        # For binary classification
        confidence = y_prob.squeeze()
        pred_labels = (confidence >= 0.5).astype(np.int32)

    # Check if predictions are correct
    accuracies = (pred_labels == y_true)

    # Create bins and digitize confidences
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidence, bin_boundaries) - 1

    # Ensure bin_indices are valid
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Initialize arrays to store bin statistics
    bin_accs = []
    bin_confs = []
    bin_counts = []

    # Compute bin statistics
    for i in range(n_bins):
        mask = (bin_indices == i)
        if np.any(mask):
            bin_accs.append(float(np.mean(accuracies[mask])))
            bin_confs.append(float(np.mean(confidence[mask])))
            bin_counts.append(int(np.sum(mask)))
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            bin_counts.append(0)

    # Calculate ECE
    total_samples = sum(bin_counts)
    if total_samples > 0:
        ece = sum([(count / total_samples) * abs(acc - conf)
                   for count, acc, conf in zip(bin_counts, bin_accs, bin_confs)])
    else:
        ece = 0.0

    return float(ece), bin_accs, bin_confs, bin_counts


def plot_reliability_diagram(bin_accs, bin_confs, bin_counts, dataset_name, ece, save_path, sparsity_ratio=None):
    """Plot reliability diagram showing calibration.

    Args:
        bin_accs: List of accuracies for each bin
        bin_confs: List of confidences for each bin
        bin_counts: List of sample counts for each bin
        dataset_name: Name of the dataset
        ece: Expected Calibration Error
        save_path: Path to save the figure
        sparsity_ratio: Optional sparsity ratio to include in the title
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate normalized bin counts
    total_samples = sum(bin_counts)
    if total_samples > 0:
        bin_counts_normalized = [count / total_samples for count in bin_counts]
    else:
        bin_counts_normalized = [0.0] * len(bin_counts)

    # Number of bins
    n_bins = len(bin_accs)

    # Plot histogram of sample distribution
    ax.bar(range(n_bins), bin_counts_normalized, width=0.8, alpha=0.3, color='b', label='Samples')

    # Plot accuracy and confidence
    ax.plot([0, n_bins-1], [0, 1], 'k--', label='Perfect calibration')
    ax.plot(range(n_bins), bin_accs, 'ro-', label='Accuracy')
    ax.plot(range(n_bins), bin_confs, 'bs-', label='Confidence')

    # Set labels and ticks
    ax.set_xlim([-0.5, n_bins-0.5])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Confidence', fontsize=14)
    ax.set_ylabel('Accuracy / Fraction of samples', fontsize=14)

    # Add sparsity to title if provided
    if sparsity_ratio is not None:
        ax.set_title(f"Reliability Diagram - {dataset_name}\nECE: {ece:.4f}, Sparsity: {sparsity_ratio*100:.1f}%", fontsize=16)
    else:
        ax.set_title(f"Reliability Diagram - {dataset_name} (ECE: {ece:.4f})", fontsize=16)

    ax.set_xticks(range(n_bins))
    ax.set_xticklabels([f"{b:.1f}" for b in np.linspace(0.05, 0.95, n_bins)])
    ax.legend(loc='lower right')

    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_sparsity_patterns(binary_indicators, dataset_name, save_path, sparsity_ratio=None):
    """Plot sparsity patterns from binary indicators.

    Args:
        binary_indicators: Tensor of binary indicators [n_samples, n_task_vectors, (n_blocks)]
        dataset_name: Name of the dataset
        save_path: Path to save the figure
        sparsity_ratio: Overall sparsity ratio for the title
    """
    # Take a representative subset
    if binary_indicators.dim() == 3:
        # For blockwise case, average across blocks
        binary_indicators = binary_indicators.mean(dim=2)

    # Take up to 20 samples
    if binary_indicators.shape[0] > 20:
        binary_indicators = binary_indicators[:20]

    # Convert to numpy for plotting
    indicators = binary_indicators.cpu().numpy()

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap of indicators
    im = ax.imshow(indicators, aspect='auto', cmap='Blues', interpolation='nearest')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Inclusion Probability', fontsize=12)

    # Set labels
    ax.set_xlabel('Task Vector', fontsize=14)
    ax.set_ylabel('Sample Index', fontsize=14)

    # Add sparsity to title if provided
    if sparsity_ratio is not None:
        ax.set_title(f"Coefficient Sparsity Patterns - {dataset_name}\nOverall Sparsity: {sparsity_ratio*100:.1f}%", fontsize=16)
    else:
        ax.set_title(f"Coefficient Sparsity Patterns - {dataset_name}", fontsize=16)

    # Set tick labels
    ax.set_xticks(range(indicators.shape[1]))
    ax.set_xticklabels([f"TV {i+1}" for i in range(indicators.shape[1])])

    ax.set_yticks(range(indicators.shape[0]))
    ax.set_yticklabels([f"Sample {i+1}" for i in range(indicators.shape[0])])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_uncertainty_vs_accuracy(epistemic_uncertainties, correct_predictions, dataset_name, save_path):
    """Plot relationship between epistemic uncertainty and prediction correctness.

    Args:
        epistemic_uncertainties: Array of epistemic uncertainties
        correct_predictions: Boolean array of prediction correctness
        dataset_name: Name of the dataset
        save_path: Path to save the figure
    """
    # Convert to numpy if tensors
    if isinstance(epistemic_uncertainties, torch.Tensor):
        epistemic_uncertainties = epistemic_uncertainties.detach().cpu().numpy()
    if isinstance(correct_predictions, torch.Tensor):
        correct_predictions = correct_predictions.detach().cpu().numpy()

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot histograms for correct and incorrect predictions
    bins = np.linspace(0, max(epistemic_uncertainties) * 1.1, 30)

    correct_mask = correct_predictions == 1
    incorrect_mask = correct_predictions == 0

    ax.hist(epistemic_uncertainties[correct_mask], bins=bins, alpha=0.5,
            color='green', label='Correct Predictions', density=True)
    ax.hist(epistemic_uncertainties[incorrect_mask], bins=bins, alpha=0.5,
            color='red', label='Incorrect Predictions', density=True)

    # Add vertical lines for mean values
    if np.any(correct_mask):
        mean_correct = np.mean(epistemic_uncertainties[correct_mask])
        ax.axvline(x=mean_correct, color='green', linestyle='--',
                   label=f'Mean Correct: {mean_correct:.4f}')

    if np.any(incorrect_mask):
        mean_incorrect = np.mean(epistemic_uncertainties[incorrect_mask])
        ax.axvline(x=mean_incorrect, color='red', linestyle='--',
                   label=f'Mean Incorrect: {mean_incorrect:.4f}')

    # Set labels
    ax.set_xlabel('Epistemic Uncertainty', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title(f'Epistemic Uncertainty vs. Prediction Correctness - {dataset_name}', fontsize=16)
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def evaluate_model(model_path, dataset, device, args):
    """Evaluate Spike-and-Slab variational model on dataset with Monte Carlo sampling"""
    # Extract dataset name from path for visualization titles
    try:
        dataset_name = os.path.basename(os.path.dirname(model_path))
        if dataset_name.endswith("Val"):
            dataset_name = dataset_name[:-3]
    except:
        dataset_name = "Unknown"  # Fallback name

    # Create analysis directory
    analysis_dir = os.path.join(os.path.dirname(model_path), "uncertainty_analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    # Load model state
    try:
        if args.debug:
            print(f"Loading model from: {model_path}")
        state_dict = torch.load(model_path, map_location="cpu")
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
        model_name = config.get('model_name', args.model)  # Get model name from config

        # Get Spike-and-Slab specific parameters
        num_task_vectors = config.get('num_task_vectors', args.num_task_vectors)
        blockwise = config.get('blockwise', args.blockwise_coef)
        kl_weight = config.get('kl_weight', args.kl_weight)
        prior_pi = config.get('prior_pi', 0.5)
        prior_sigma = config.get('prior_sigma', 1.0)
        temperature = config.get('temperature', args.temperature)
    else:
        # No config found, use command line arguments
        if args.debug:
            print("No configuration found in model, using args")

        # Sample features to get dimensions
        batch = next(iter(dataset.test_loader))
        if isinstance(batch, dict):
            features = batch["features"]
        else:
            features = batch[0]
        feature_dim = features.shape[1]
        model_name = args.model

        # Use provided arguments
        num_task_vectors = args.num_task_vectors
        blockwise = args.blockwise_coef
        kl_weight = args.kl_weight
        prior_pi = 0.5  # Default prior inclusion probability
        prior_sigma = 1.0  # Default prior sigma
        temperature = args.temperature

    # Create the Spike-and-Slab variational model
    try:
        if args.debug:
            print("Creating SpikeAndSlabMetaNet instance...")

        model = SpikeAndSlabMetaNet(
            feature_dim=feature_dim,
            task_vectors=num_task_vectors,
            blockwise=blockwise,
            kl_weight=kl_weight,
            num_samples=args.num_eval_samples,
            prior_pi=prior_pi,
            prior_sigma=prior_sigma,
            temperature=temperature
        )

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
                'module.',
                'model.',
                'module.metanet.',
                'metanet.',
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
                    raise ValueError("Could not load model parameters")
    except Exception as e:
        print(f"Error creating or loading model: {e}")
        if args.debug:
            traceback.print_exc()
        raise

    model = model.to(device)
    model.eval()

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
                print("Warning: Could not find classifier weights in the model")
    except Exception as e:
        print(f"Error creating or loading classifier: {e}")
        if args.debug:
            traceback.print_exc()
        raise

    classifier = classifier.to(device)
    classifier.eval()

    # Model configuration for reporting
    model_config = {
        'model_name': model_name,
        'feature_dim': feature_dim,
        'num_task_vectors': num_task_vectors,
        'blockwise': blockwise,
        'kl_weight': kl_weight,
        'prior_pi': prior_pi,
        'prior_sigma': prior_sigma,
        'temperature': temperature,
    }

    # Prepare for evaluation
    if args.debug:
        print("Starting evaluation with Monte Carlo sampling...")

    all_correct = 0
    all_total = 0
    all_predictions = []
    all_labels = []
    all_probs = []
    all_binary_indicators = []
    all_epistemic_uncertainties = []
    all_sparsity_ratios = []
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    # Evaluate with Monte Carlo sampling
    with torch.no_grad():
        for batch in dataset.test_loader:
            if isinstance(batch, dict):
                features = batch["features"].to(device)
                labels = batch["labels"].to(device)
            else:
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)

            # Get Monte Carlo predictions
            prediction_stats = model.monte_carlo_predictions(
                features, classifier, num_samples=args.num_eval_samples
            )

            # Get statistics
            predictions = prediction_stats["predictions"]
            mean_probs = prediction_stats["mean_probs"]
            epistemic_uncertainties = prediction_stats["epistemic_uncertainty"]
            sparsity_ratio = prediction_stats.get("sparsity_ratio", 0.0)

            # Get binary indicators if available
            if "binary_indicators" in prediction_stats:
                batch_indicators = prediction_stats["binary_indicators"]
                all_binary_indicators.append(batch_indicators)

            # Update metrics
            batch_size = labels.size(0)
            all_total += batch_size
            batch_correct = (predictions == labels.cpu()).sum().item()
            all_correct += batch_correct

            # Per-class accuracy
            for i, label in enumerate(labels):
                label_idx = label.item()
                prediction = predictions[i].item()

                class_total[label_idx] += 1
                if prediction == label_idx:
                    class_correct[label_idx] += 1

            # Save for later analysis
            all_predictions.append(predictions)
            all_labels.append(labels.cpu())
            all_probs.append(mean_probs)
            all_epistemic_uncertainties.append(epistemic_uncertainties)
            all_sparsity_ratios.append(sparsity_ratio)

    # Calculate overall accuracy
    accuracy = all_correct / all_total if all_total > 0 else 0.0

    # Combine prediction data
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)
    all_epistemic_uncertainties = torch.cat(all_epistemic_uncertainties)

    # Get average sparsity ratio
    avg_sparsity_ratio = float(np.mean(all_sparsity_ratios))

    # Combine binary indicators if available
    if all_binary_indicators:
        all_binary_indicators = torch.cat(all_binary_indicators, dim=0)

    # Calculate per-class accuracy
    per_class_acc = {}
    for cls_idx in range(len(dataset.classnames)):
        cls_name = dataset.classnames[cls_idx]
        if class_total[cls_idx] > 0:
            cls_acc = class_correct[cls_idx] / class_total[cls_idx]
            per_class_acc[cls_name] = float(cls_acc)

    # Calculate ECE and generate reliability diagram
    ece, bin_accs, bin_confs, bin_counts = expected_calibration_error(
        all_labels.cpu().numpy(),
        all_probs.cpu().numpy(),
        n_bins=10
    )

    # Create reliability diagram
    if args.save_uncertainty:
        try:
            reliability_path = os.path.join(analysis_dir, f"{dataset_name}_reliability_diagram.png")
            plot_reliability_diagram(
                bin_accs, bin_confs, bin_counts,
                dataset_name, ece, reliability_path,
                sparsity_ratio=avg_sparsity_ratio
            )
            print(f"Saved reliability diagram to {reliability_path}")

            # Create sparsity pattern visualization if available
            if len(all_binary_indicators) > 0:
                sparsity_path = os.path.join(analysis_dir, f"{dataset_name}_sparsity_patterns.png")
                plot_sparsity_patterns(
                    all_binary_indicators[:20],  # Use first 20 samples
                    dataset_name,
                    sparsity_path,
                    sparsity_ratio=avg_sparsity_ratio
                )
                print(f"Saved sparsity pattern visualization to {sparsity_path}")

            # Create uncertainty vs. accuracy plot
            correct_predictions = (all_predictions == all_labels).int().cpu().numpy()
            uncertainty_path = os.path.join(analysis_dir, f"{dataset_name}_uncertainty_vs_accuracy.png")
            plot_uncertainty_vs_accuracy(
                all_epistemic_uncertainties.cpu().numpy(),
                correct_predictions,
                dataset_name,
                uncertainty_path
            )
            print(f"Saved uncertainty vs. accuracy plot to {uncertainty_path}")

            # If detailed analysis is enabled, create additional visualizations
            if args.detailed_analysis:
                # Run posterior analysis on a small batch for visualization
                sample_batch = next(iter(dataset.test_loader))
                if isinstance(sample_batch, dict):
                    sample_features = sample_batch["features"].to(device)[:10]  # Use 10 samples
                else:
                    sample_features = sample_batch[0].to(device)[:10]

                # Get posterior statistics
                posterior_stats = model.get_posterior_stats(sample_features)

                # Plot inclusion probabilities
                if "inclusion_probs" in posterior_stats:
                    inclusion_probs = posterior_stats["inclusion_probs"]

                    plt.figure(figsize=(12, 6))
                    plt.imshow(inclusion_probs.cpu().numpy().reshape(len(inclusion_probs), -1),
                              aspect='auto', cmap='viridis')
                    plt.colorbar(label="Inclusion Probability")
                    plt.xlabel("Coefficient Index")
                    plt.ylabel("Sample Index")
                    plt.title(f"Coefficient Inclusion Probabilities - {dataset_name}")

                    inclusion_path = os.path.join(analysis_dir, f"{dataset_name}_inclusion_probabilities.png")
                    plt.savefig(inclusion_path, dpi=300)
                    plt.close()
                    print(f"Saved inclusion probabilities visualization to {inclusion_path}")

                # Plot samples
                if "samples" in posterior_stats:
                    samples = posterior_stats["samples"]

                    plt.figure(figsize=(12, 6))
                    plt.imshow(samples.cpu().numpy().reshape(len(samples), -1),
                              aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
                    plt.colorbar(label="Coefficient Value")
                    plt.xlabel("Coefficient Index")
                    plt.ylabel("Sample Index")
                    plt.title(f"Sparse Coefficient Samples - {dataset_name}")

                    samples_path = os.path.join(analysis_dir, f"{dataset_name}_coefficient_samples.png")
                    plt.savefig(samples_path, dpi=300)
                    plt.close()
                    print(f"Saved coefficient samples visualization to {samples_path}")
        except Exception as e:
            print(f"Error generating visualizations: {e}")
            if args.debug:
                traceback.print_exc()

    # Get sparsity statistics if available
    if hasattr(model, 'get_sparsity_stats'):
        try:
            sparsity_stats = model.get_sparsity_stats()
        except Exception as e:
            print(f"Error getting sparsity stats: {e}")
            sparsity_stats = {}
    else:
        sparsity_stats = {}

    # Calculate uncertainty metrics
    uncertainty_metrics = {
        "ece": float(ece),
        "avg_epistemic_uncertainty": float(all_epistemic_uncertainties.mean().item()),
    }

    # Calculate correlation between uncertainty and correctness
    correct_mask = (all_predictions == all_labels).bool()
    incorrect_mask = ~correct_mask

    if torch.any(correct_mask) and torch.any(incorrect_mask):
        correct_uncertainty = all_epistemic_uncertainties[correct_mask].mean().item()
        incorrect_uncertainty = all_epistemic_uncertainties[incorrect_mask].mean().item()
        uncertainty_metrics.update({
            "correct_uncertainty": float(correct_uncertainty),
            "incorrect_uncertainty": float(incorrect_uncertainty),
            "uncertainty_ratio": float(incorrect_uncertainty / (correct_uncertainty + 1e-8)),
        })

    # Calculate AUROC for uncertainty as a detector of errors
    try:
        from sklearn.metrics import roc_auc_score

        # Higher uncertainty should predict incorrect classifications
        auroc = roc_auc_score(incorrect_mask.cpu().numpy(), all_epistemic_uncertainties.cpu().numpy())
        uncertainty_metrics["uncertainty_auroc"] = float(auroc)
    except Exception as e:
        print(f"Warning: Could not calculate AUROC: {e}")
        uncertainty_metrics["uncertainty_auroc"] = 0.0

    # Combine all metrics
    metrics = {
        "accuracy": accuracy,
        "num_correct": all_correct,
        "num_samples": all_total,
        "per_class_accuracy": per_class_acc,
        "reliability": {
            "ece": ece,
            "bin_accuracies": bin_accs,
            "bin_confidences": bin_confs,
            "bin_counts": bin_counts
        },
        "uncertainty_metrics": uncertainty_metrics,
        "sparsity_metrics": {
            "avg_sparsity_ratio": avg_sparsity_ratio,
            **sparsity_stats
        },
        "config": model_config,
        "model_path": str(model_path),
        "model_type": "spike_and_slab_variational",
        "evaluation_timestamp": datetime.now().isoformat(),
    }

    # Convert to Python types for JSON serialization
    metrics = convert_to_python_types(metrics)

    # Save detailed metrics to file
    metrics_path = os.path.join(analysis_dir, f"{dataset_name}_evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved detailed metrics to {metrics_path}")

    if args.debug:
        print(f"Evaluation complete. Accuracy: {accuracy * 100:.2f}%, ECE: {ece:.4f}, Sparsity: {avg_sparsity_ratio*100:.1f}%")

    return metrics


def main():
    """Main evaluation function"""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model-specific save directory for results
    model_save_dir = os.path.join(args.save_dir, args.model, "evaluation_results_spike_and_slab")
    os.makedirs(model_save_dir, exist_ok=True)
    print(f"Results will be saved to: {model_save_dir}")

    # Create model-specific model directory if it exists
    model_specific_dir = os.path.join(args.model_dir, args.model)
    if os.path.exists(model_specific_dir):
        args.model_dir = model_specific_dir
        print(f"Using model-specific directory for models: {args.model_dir}")

    # Print configuration
    print(f"\n=== Spike-and-Slab Variational Evaluation Configuration ===")
    print(f"Model: {args.model}")
    print(f"Using blockwise coefficients: {args.blockwise_coef}")
    print(f"Monte Carlo samples: {args.num_eval_samples}")
    print(f"Save uncertainty analysis: {args.save_uncertainty}")
    print(f"Detailed analysis: {args.detailed_analysis}")
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
                feature_dir = os.path.join(args.data_location, "precomputed_features", args.model,
                                           dataset_name + "Val")
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
                # Find model path
                model_path = find_model_path(
                    args.model_dir,
                    dataset_name,
                    args.model,
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
            print(f"Model: {results['config']['model_name']}")
            print(f"Accuracy: {results['accuracy'] * 100:.2f}% ({results['num_correct']}/{results['num_samples']})")
            print(f"ECE: {results['reliability']['ece']:.4f}")

            # Print sparsity metrics
            sparsity_metrics = results.get('sparsity_metrics', {})
            if sparsity_metrics:
                print(f"Sparsity ratio: {sparsity_metrics.get('avg_sparsity_ratio', 0)*100:.1f}%")
                print(f"Posterior inclusion probability: {sparsity_metrics.get('posterior_inclusion_prob', 0):.3f}")
                print(f"Prior inclusion probability: {sparsity_metrics.get('prior_inclusion_prob', 0):.3f}")

            # Print uncertainty metrics
            uncertainty_metrics = results.get('uncertainty_metrics', {})
            if uncertainty_metrics:
                print(f"Average epistemic uncertainty: {uncertainty_metrics.get('avg_epistemic_uncertainty', 0):.4f}")

                if 'correct_uncertainty' in uncertainty_metrics and 'incorrect_uncertainty' in uncertainty_metrics:
                    print(f"Uncertainty - Correct: {uncertainty_metrics['correct_uncertainty']:.4f}, "
                          f"Incorrect: {uncertainty_metrics['incorrect_uncertainty']:.4f}, "
                          f"Ratio: {uncertainty_metrics.get('uncertainty_ratio', 0):.2f}")

                if 'uncertainty_auroc' in uncertainty_metrics:
                    print(f"Uncertainty AUROC: {uncertainty_metrics['uncertainty_auroc']:.4f}")

            # Add to results
            all_results[dataset_name] = results

            # Add to summary
            summary_entry = {
                'dataset': dataset_name,
                'accuracy': float(results['accuracy']),
                'ece': float(results['reliability']['ece']),
                'sparsity_ratio': float(sparsity_metrics.get('avg_sparsity_ratio', 0)),
                'epistemic_uncertainty': float(uncertainty_metrics.get('avg_epistemic_uncertainty', 0)),
                'uncertainty_auroc': float(uncertainty_metrics.get('uncertainty_auroc', 0)),
                'model': str(results['config']['model_name'])
            }

            summary_results.append(summary_entry)

        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            if args.debug:
                traceback.print_exc()
        finally:
            # Clean up dataset resources
            cleanup_resources(dataset)
            torch.cuda.empty_cache()
            gc.collect()

    # Calculate average metrics
    if summary_results:
        avg_accuracy = sum(r['accuracy'] for r in summary_results) / len(summary_results)
        avg_ece = sum(r['ece'] for r in summary_results) / len(summary_results)
        avg_sparsity = sum(r['sparsity_ratio'] for r in summary_results) / len(summary_results)
        avg_uncertainty = sum(r['epistemic_uncertainty'] for r in summary_results) / len(summary_results)
        avg_auroc = sum(r['uncertainty_auroc'] for r in summary_results) / len(summary_results)

        all_results['average_metrics'] = {
            'accuracy': float(avg_accuracy),
            'ece': float(avg_ece),
            'sparsity_ratio': float(avg_sparsity),
            'epistemic_uncertainty': float(avg_uncertainty),
            'uncertainty_auroc': float(avg_auroc)
        }

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(
            model_save_dir,
            f"spike_and_slab_results_{args.model}_{timestamp}.json"
        )

        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=4)

        print(f"\nAll evaluation results saved to: {results_path}")

        # Print summary table
        print("\n=== Spike-and-Slab Variational Evaluation Summary ===")
        print(f"{'Dataset':<15} | {'Accuracy':<10} | {'ECE':<8} | {'Sparsity':<10} | {'Unc. AUROC':<10}")
        print("-" * 65)

        for result in sorted(summary_results, key=lambda x: x['dataset']):
            print(f"{result['dataset']:<15} | {result['accuracy'] * 100:>8.2f}% | {result['ece']:>6.4f} | "
                  f"{result['sparsity_ratio'] * 100:>8.1f}% | {result['uncertainty_auroc']:>9.4f}")

        print("-" * 65)
        print(f"{'Average':<15} | {avg_accuracy * 100:>8.2f}% | {avg_ece:>6.4f} | "
              f"{avg_sparsity * 100:>8.1f}% | {avg_auroc:>9.4f}")
        print("=" * 65)


if __name__ == "__main__":
    main()