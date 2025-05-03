"""
Training script for Task-level Variational MetaNet using pre-computed features.

This script provides a training implementation that models the posterior
distribution through variational inference at the task level, learning
a single set of variational parameters for the entire dataset.
"""

import os
import time
import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
import gc
import traceback
from datetime import datetime
import math

from src.task_variational import TaskVariationalMetaNet
from src.utils import cosine_lr
from src.datasets.common import maybe_dictionarize
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp

from src.args import parse_arguments

args = parse_arguments()

# Add task variational-specific arguments
args.kl_weight = args.kl_weight  # Weight for KL divergence in ELBO
args.num_samples = 5  # Number of samples to draw from posterior during training


class PrecomputedFeatureDataset(torch.utils.data.Dataset):
    """Dataset for precomputed features with augmentation support"""

    def __init__(self, features_path, labels_path, verbose=False,
                 augmentation_paths=None, use_augmentation=True):
        """
        Initialize dataset with paths to precomputed features and labels

        Args:
            features_path: Path to precomputed features tensor
            labels_path: Path to labels tensor
            verbose: Whether to print detailed logs
            augmentation_paths: List of paths to augmented feature/label pairs
            use_augmentation: Whether to use augmented versions when available
        """
        super().__init__()

        # Store augmentation settings
        self.training = True  # Default to training mode
        self.use_augmentation = use_augmentation
        self.augmentation_paths = []
        if augmentation_paths is not None:
            self.augmentation_paths = augmentation_paths

        # Load base features and labels
        try:
            self.features = torch.load(features_path)
            if verbose:
                print(f"Successfully loaded features from {features_path}, shape: {self.features.shape}")
        except Exception as e:
            print(f"Error loading features from {features_path}: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to load features from {features_path}: {e}")

        try:
            self.labels = torch.load(labels_path)
            if verbose:
                print(f"Successfully loaded labels from {labels_path}, shape: {self.labels.shape}")
        except Exception as e:
            print(f"Error loading labels from {labels_path}: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to load labels from {labels_path}: {e}")

        # Validate that features and labels have matching sizes
        if len(self.features) != len(self.labels):
            raise ValueError(f"Features ({len(self.features)}) and labels ({len(self.labels)}) count mismatch")

        # Load augmented versions if provided
        self.augmented_features = []
        self.augmented_labels = []

        if augmentation_paths and use_augmentation:
            for aug_idx, (aug_feat_path, aug_label_path) in enumerate(augmentation_paths):
                if os.path.exists(aug_feat_path) and os.path.exists(aug_label_path):
                    try:
                        aug_features = torch.load(aug_feat_path)
                        aug_labels = torch.load(aug_label_path)

                        # Verify shapes match original features
                        if aug_features.shape == self.features.shape and aug_labels.shape == self.labels.shape:
                            self.augmented_features.append(aug_features)
                            self.augmented_labels.append(aug_labels)
                            if verbose:
                                print(f"Loaded augmented version {aug_idx + 1} from {aug_feat_path}")
                        else:
                            print(f"Warning: Augmented version {aug_idx + 1} has mismatched shape, skipping")
                    except Exception as e:
                        print(f"Error loading augmented version {aug_idx + 1}: {e}")
                        if verbose:
                            traceback.print_exc()

            if verbose and self.augmented_features:
                print(f"Loaded {len(self.augmented_features)} augmented versions")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # During training, randomly choose from augmented versions if available
        if self.training and self.augmented_features and self.use_augmentation and random.random() > 0.2:
            # 80% chance to use augmented features
            aug_idx = random.randint(0, len(self.augmented_features) - 1)
            return {
                "features": self.augmented_features[aug_idx][idx],
                "labels": self.augmented_labels[aug_idx][idx],
                "index": idx,
                "augmented": True
            }
        else:
            # Use original features or when evaluating
            return {
                "features": self.features[idx],
                "labels": self.labels[idx],
                "index": idx,
                "augmented": False
            }

    def train(self, mode=True):
        """Sets the dataset in training mode (will use augmentations)"""
        self.training = mode
        return self


class PrecomputedFeatures:
    """Dataset container class for precomputed features with augmentation support"""

    def __init__(self,
                 feature_dir,
                 batch_size=128,
                 num_workers=8,
                 persistent_workers=False,
                 use_augmentation=True):
        """
        Initialize with directory containing precomputed features

        Args:
            feature_dir: Path to directory with precomputed features
            batch_size: Batch size for dataloaders
            num_workers: Number of worker threads for dataloaders
            persistent_workers: Whether to keep worker processes alive
            use_augmentation: Whether to use augmentations during training
        """
        # Verify directory exists
        if not os.path.exists(feature_dir):
            raise FileNotFoundError(f"Feature directory not found: {feature_dir}")

        print(f"Loading features from {feature_dir}")
        print(f"Augmentation enabled: {use_augmentation}")

        # Define file paths
        train_features_path = os.path.join(feature_dir, "train_features.pt")
        train_labels_path = os.path.join(feature_dir, "train_labels.pt")
        val_features_path = os.path.join(feature_dir, "val_features.pt")
        val_labels_path = os.path.join(feature_dir, "val_labels.pt")

        # Check if train files exist
        if not os.path.exists(train_features_path):
            raise FileNotFoundError(f"Train features not found at {train_features_path}")

        # Find augmented versions without verbose output
        augmentation_paths = []
        aug_idx = 1

        # Count available augmentation files
        while True:
            aug_feat_path = os.path.join(feature_dir, f"train_features_aug{aug_idx}.pt")
            aug_label_path = os.path.join(feature_dir, f"train_labels_aug{aug_idx}.pt")

            if os.path.exists(aug_feat_path) and os.path.exists(aug_label_path):
                augmentation_paths.append((aug_feat_path, aug_label_path))
                aug_idx += 1
            else:
                break

        if len(augmentation_paths) > 0:
            print(f"Found {len(augmentation_paths)} augmented versions")

        # Create train dataset with augmentation support
        print("Loading training dataset...")
        self.train_dataset = PrecomputedFeatureDataset(
            train_features_path,
            train_labels_path,
            verbose=True,
            augmentation_paths=augmentation_paths,
            use_augmentation=use_augmentation
        )

        # Enable training mode for train dataset
        self.train_dataset.train(True)

        # Create train loader
        print("Creating training dataloader...")
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=persistent_workers and num_workers > 0,
            pin_memory=True,
            drop_last=False,
            timeout=120,  # Add timeout to prevent worker hangs
        )

        # Create test dataset (no augmentation for evaluation)
        print("Loading validation dataset...")
        self.test_dataset = PrecomputedFeatureDataset(
            val_features_path,
            val_labels_path,
            verbose=True,
            augmentation_paths=None,  # No augmentation for test dataset
            use_augmentation=False
        )

        # Set test dataset to evaluation mode
        self.test_dataset.train(False)

        # Create test loader
        print("Creating validation dataloader...")
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers and num_workers > 0,
            pin_memory=True,
            drop_last=False,
            timeout=120,
        )

        # Load classnames if available
        classnames_path = os.path.join(feature_dir, "classnames.txt")
        if os.path.exists(classnames_path):
            with open(classnames_path, "r") as f:
                self.classnames = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(self.classnames)} class names from {classnames_path}")
        else:
            # Create dummy classnames if file doesn't exist
            unique_labels = torch.unique(self.train_dataset.labels)
            self.classnames = [f"class_{i}" for i in range(len(unique_labels))]
            print(f"Created {len(self.classnames)} dummy class names")


def cleanup_resources(dataset):
    """Cleanup data resources to prevent memory leaks"""
    if dataset is None:
        return

    try:
        # Clear dataset references
        if hasattr(dataset, 'test_loader') and dataset.test_loader is not None:
            dataset.test_loader = None
        if hasattr(dataset, 'train_loader') and dataset.train_loader is not None:
            dataset.train_loader = None
        if hasattr(dataset, 'test_dataset') and dataset.test_dataset is not None:
            dataset.test_dataset = None
        if hasattr(dataset, 'train_dataset') and dataset.train_dataset is not None:
            dataset.train_dataset = None
    except Exception as e:
        print(f"Warning during dataset cleanup: {e}")

    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_model(model, classifier, dataset, device, num_eval_samples=10):
    """Evaluate model on dataset with Monte Carlo sampling

    Parameters:
    ----------
    model: TaskVariationalMetaNet
        The model to evaluate
    classifier: nn.Module
        Classification head
    dataset: PrecomputedFeatures
        Dataset container
    device: torch.device
        Device to use for evaluation
    num_eval_samples: int
        Number of Monte Carlo samples to draw for evaluation

    Returns:
    ----------
    dict:
        Dictionary with evaluation results
    """
    model.eval()
    classifier.eval()

    # Store metrics
    total_samples = 0
    total_correct = 0
    all_predictions = []
    all_labels = []
    all_uncertainties = []

    print("Starting evaluation...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataset.test_loader):
            if batch_idx % 10 == 0:
                print(f"  Evaluating batch {batch_idx}/{len(dataset.test_loader)}")

            batch = maybe_dictionarize(batch)
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            # Get Monte Carlo predictions
            prediction_stats = model.monte_carlo_predictions(
                features, classifier, num_samples=num_eval_samples
            )

            # Get epistemic uncertainty for analysis
            epistemic_uncertainty = prediction_stats["epistemic_uncertainty"]
            predictions = prediction_stats["predictions"]

            # Compute accuracy
            batch_size = labels.size(0)
            total_samples += batch_size
            batch_correct = (predictions == labels.cpu()).sum().item()
            total_correct += batch_correct

            # Store for later analysis
            all_predictions.append(predictions)
            all_labels.append(labels.cpu())
            all_uncertainties.append(epistemic_uncertainty)

    # Compute accuracy
    accuracy = total_correct / total_samples

    # Combine prediction tensors
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    all_uncertainties = torch.cat(all_uncertainties)

    # Get current variational parameters
    variational_params = model.get_variational_params()

    # Calculate additional metrics
    metrics = {
        "accuracy": accuracy,
        "num_correct": total_correct,
        "num_samples": total_samples,
        "avg_epistemic_uncertainty": all_uncertainties.mean().item(),
        "variational_params": {
            "mean": variational_params["mean"].tolist(),
            "std": variational_params["std"].tolist(),
            "coeff_var": variational_params["coeff_var"].tolist(),
        }
    }

    print(f"Evaluation complete. Accuracy: {accuracy * 100:.2f}% ({total_correct}/{total_samples})")

    return metrics


def train_with_task_variational(rank, args):
    """Main training function with task-level variational inference"""
    args.rank = rank

    # Initialize distributed setup
    try:
        print(f"Initializing distributed training on rank {rank}...")
        setup_ddp(args.rank, args.world_size, port=args.port)
        print(f"Distributed setup complete on rank {rank}")
    except Exception as e:
        print(f"Error in distributed setup on rank {rank}: {e}")
        traceback.print_exc()
        return

    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Process specified datasets
    datasets_to_process = args.datasets if args.datasets else [
        "Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"
    ]

    # Create model-specific directory path
    model_save_dir = os.path.join(args.save_dir, args.model)

    # Ensure save directory exists
    os.makedirs(model_save_dir, exist_ok=True)
    if is_main_process():
        print(f"Using model-specific save directory: {model_save_dir}")

    # Print configuration (main process only)
    if rank == 0:
        print(f"\n=== Training Configuration ===")
        print(f"Model: {args.model}")
        print(f"Using Task-level Variational MetaNet")
        print(f"Using blockwise coefficients: {args.blockwise_coef}")
        print(f"KL weight: {args.kl_weight:.4f}")
        print(f"MC Samples during training: {args.num_samples}")
        print(f"Using augmentation: {args.use_augmentation}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.lr}")
        print(f"Epochs: {args.epochs}")
        print(f"Save directory: {model_save_dir}")
        print(f"Datasets: {datasets_to_process}")
        print("=" * 30)

    for dataset_name in datasets_to_process:
        if is_main_process():
            print(f"\n{'='*40}")
            print(f"=== Training on {dataset_name} with Task-level Variational MetaNet ===")
            print(f"{'='*40}\n")

        # Setup save directory for this dataset (include model name in path)
        save_dir = os.path.join(model_save_dir, dataset_name + "Val")
        if is_main_process():
            os.makedirs(save_dir, exist_ok=True)

        dataset = None

        try:
            # Define feature directory based on standard structure
            feature_dir = os.path.join(args.data_location, "precomputed_features", args.model, dataset_name + "Val")

            # Verify directory exists before proceeding
            if not os.path.exists(feature_dir):
                if is_main_process():
                    print(f"Error: Feature directory not found at {feature_dir}")
                continue

            # Load dataset with precomputed features
            if is_main_process():
                print(f"Loading dataset from {feature_dir}...")

            try:
                dataset = PrecomputedFeatures(
                    feature_dir=feature_dir,
                    batch_size=args.batch_size,
                    num_workers=2,  # Reduced for stability
                    use_augmentation=args.use_augmentation,
                )
                if is_main_process():
                    print(f"Successfully loaded dataset with {len(dataset.train_dataset)} training samples and {len(dataset.test_dataset)} validation samples")
            except Exception as e:
                if is_main_process():
                    print(f"Error loading dataset: {e}")
                    traceback.print_exc()
                continue

            # Get feature dimension from dataset
            sample_batch = next(iter(dataset.train_loader))
            sample_batch = maybe_dictionarize(sample_batch)
            feature_dim = sample_batch["features"].shape[1]
            if is_main_process():
                print(f"Feature dimension: {feature_dim}")

            # Create task-level variational metanet model
            if is_main_process():
                print("Creating TaskVariationalMetaNet model...")

            try:
                model = TaskVariationalMetaNet(
                    feature_dim=feature_dim,
                    task_vectors=args.num_task_vectors,
                    blockwise=args.blockwise_coef,
                    kl_weight=args.kl_weight,
                    num_samples=args.num_samples
                )
                if is_main_process():
                    print(f"Created TaskVariationalMetaNet with {'blockwise' if args.blockwise_coef else 'global'} coefficients")
                    print(f"Using {args.num_samples} MC samples during training")
            except Exception as e:
                if is_main_process():
                    print(f"Error creating model: {e}")
                    traceback.print_exc()
                continue

            model = model.to(rank)

            data_loader = dataset.train_loader
            num_batches = len(data_loader)

            # Set print frequency
            print_every = max(num_batches // 10, 1)

            # Distributed training setup
            if is_main_process():
                print("Setting up distributed data loader...")

            try:
                ddp_loader = distribute_loader(data_loader)
                if is_main_process():
                    print(f"Created distributed data loader with {len(ddp_loader)} batches")
            except Exception as e:
                if is_main_process():
                    print(f"Error setting up distributed data loader: {e}")
                    traceback.print_exc()
                continue

            if is_main_process():
                print("Creating distributed model...")

            try:
                ddp_model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[args.rank],
                    find_unused_parameters=False
                )
                if is_main_process():
                    print("Successfully created distributed model")
            except Exception as e:
                if is_main_process():
                    print(f"Error creating distributed model: {e}")
                    traceback.print_exc()
                continue

            # Setup classifier layer
            if is_main_process():
                print("Creating classifier layer...")

            try:
                num_classes = len(dataset.classnames)
                classifier = torch.nn.Linear(feature_dim, num_classes).cuda()
                if is_main_process():
                    print(f"Created classifier with {num_classes} output classes")
            except Exception as e:
                if is_main_process():
                    print(f"Error creating classifier: {e}")
                    traceback.print_exc()
                continue

            # Setup optimizer with parameter groups
            if is_main_process():
                print("Setting up optimizer...")

            # For task-level VI, we need special treatment for variational parameters
            var_params = []
            other_params = []

            # Separate variational parameters from others
            for name, param in ddp_model.named_parameters():
                if 'mean_params' in name or 'logvar_params' in name:
                    var_params.append(param)
                else:
                    other_params.append(param)

            # Add classifier parameters
            other_params.extend(list(classifier.parameters()))

            # Create parameter groups with different learning rates
            param_groups = [
                {'params': var_params, 'lr': args.lr * 2.0, 'weight_decay': 0.0001},  # Higher LR for variational params
                {'params': other_params, 'lr': args.lr, 'weight_decay': args.wd}
            ]

            optimizer = torch.optim.AdamW(param_groups)
            if is_main_process():
                print(f"Created optimizer with {len(var_params)} variational parameters and {len(other_params)} other parameters")

            # Learning rate scheduler
            scheduler = cosine_lr(
                optimizer,
                args.lr,
                0,
                args.epochs * num_batches
            )
            if is_main_process():
                print(f"Created cosine learning rate scheduler with {args.epochs} epochs")

            # Loss function
            loss_fn = torch.nn.CrossEntropyLoss()

            # Initialize GradScaler for mixed precision training
            scaler = GradScaler(enabled=True)

            # Training monitoring
            train_losses = []
            kl_losses = []
            val_accuracies = []
            variational_params_history = []
            best_acc = 0.0
            best_model_state = None

            # Training loop
            for epoch in range(args.epochs):
                ddp_model.train()
                classifier.train()

                epoch_loss = 0.0
                epoch_kl_loss = 0.0
                batch_count = 0

                if is_main_process():
                    print(f"\nEpoch {epoch + 1}/{args.epochs} - Training")

                epoch_start_time = time.time()

                for i, batch in enumerate(ddp_loader):
                    batch_start_time = time.time()

                    try:
                        batch = maybe_dictionarize(batch)
                        features = batch["features"].to(rank)
                        labels = batch["labels"].to(rank)

                        # Use autocast for mixed precision training
                        with autocast(device_type='cuda', enabled=True):
                            # Forward pass with multiple MC samples during training
                            transformed_features = ddp_model(features, num_samples=args.num_samples)
                            logits = classifier(transformed_features)

                            # Task loss (negative log-likelihood)
                            task_loss = loss_fn(logits, labels)

                            # Variational KL divergence
                            kl_loss = ddp_model.module.kl_divergence_loss()

                            # Total loss for ELBO optimization
                            total_loss = task_loss + kl_loss

                        # Backward pass with gradient scaling
                        scaler.scale(total_loss).backward()

                        # Step optimizer
                        scheduler(i + epoch * num_batches)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                        # Record stats
                        task_loss_cpu = task_loss.item()
                        kl_loss_cpu = kl_loss.item()
                        batch_count += 1
                        epoch_loss += task_loss_cpu
                        epoch_kl_loss += kl_loss_cpu

                        batch_time = time.time() - batch_start_time

                        if is_main_process():
                            train_losses.append(task_loss_cpu)
                            kl_losses.append(kl_loss_cpu)

                        # Print progress (with reduced frequency)
                        if i % print_every == 0 and is_main_process():
                            # Simple output
                            print(f"  Batch {i:4d}/{num_batches:4d} | Task: {task_loss_cpu:.4f} | "
                                  f"KL: {kl_loss_cpu:.4f} | "
                                  f"t: {batch_time:.2f}s")

                            # Print current learning rates
                            lrs = [param_group['lr'] for param_group in optimizer.param_groups]
                            print(f"  Learning rates: {lrs}")

                    except Exception as e:
                        if is_main_process():
                            print(f"  Error in batch {i}: {e}")
                            traceback.print_exc()
                        # Skip this batch but continue training
                        continue

                # Record epoch stats
                avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
                avg_epoch_kl_loss = epoch_kl_loss / batch_count if batch_count > 0 else 0
                epoch_time = time.time() - epoch_start_time

                # Print epoch summary
                if is_main_process():
                    print(f"  Epoch {epoch + 1} completed in {epoch_time:.2f}s")
                    print(f"  Summary: Task: {avg_epoch_loss:.4f} | KL: {avg_epoch_kl_loss:.4f}")

                    # Get current variational parameters
                    if is_main_process() and hasattr(ddp_model.module, 'get_variational_params'):
                        var_params = ddp_model.module.get_variational_params()

                        # For global coefficients, show all values
                        if not args.blockwise_coef:
                            means = var_params["mean"].tolist()
                            stds = var_params["std"].tolist()

                            # Print top 3 coefficients by absolute value
                            sorted_indices = np.argsort([-abs(m) for m in means])
                            print("  Top 3 coefficients:")
                            for idx in sorted_indices[:3]:
                                print(f"    Task Vector {idx}: mean={means[idx]:.4f}, std={stds[idx]:.4f}")
                        else:
                            # For blockwise, just show stats
                            means = var_params["mean"]
                            stds = var_params["std"]
                            print(f"  Mean coefficients range: [{means.min().item():.4f}, {means.max().item():.4f}]")
                            print(f"  Std coefficients range: [{stds.min().item():.4f}, {stds.max().item():.4f}]")

                        # Save for history
                        variational_params_history.append({
                            "epoch": epoch,
                            "mean": var_params["mean"].tolist(),
                            "std": var_params["std"].tolist(),
                            "coeff_var": var_params["coeff_var"].tolist()
                        })

                # Evaluate on validation set
                if is_main_process():
                    print(f"\nEpoch {epoch + 1}/{args.epochs} - Validation")

                    # Evaluate with Monte Carlo sampling
                    eval_start_time = time.time()
                    eval_results = evaluate_model(
                        model=ddp_model.module,
                        classifier=classifier,
                        dataset=dataset,
                        device=rank,
                        num_eval_samples=10  # More samples during evaluation
                    )
                    eval_time = time.time() - eval_start_time

                    val_acc = eval_results["accuracy"]
                    val_accuracies.append(val_acc)

                    # Print evaluation results
                    print(f"  Evaluation completed in {eval_time:.2f}s")
                    print(f"  Accuracy: {val_acc * 100:.2f}% ({eval_results['num_correct']}/{eval_results['num_samples']})")
                    print(f"  Avg epistemic uncertainty: {eval_results.get('avg_epistemic_uncertainty', 0.0):.4f}")

                    # Save best model
                    if val_acc > best_acc:
                        best_acc = val_acc

                        # Store model configuration along with weights
                        config = {
                            'feature_dim': feature_dim,
                            'num_task_vectors': args.num_task_vectors,
                            'blockwise': args.blockwise_coef,
                            'kl_weight': args.kl_weight,
                            'model_name': args.model,
                            'use_augmentation': args.use_augmentation,
                            'task_level_variational': True  # Mark as task-level variational model
                        }

                        best_model_state = {
                            'meta_net': ddp_model.module.state_dict(),
                            'classifier': classifier.state_dict(),
                            'epoch': epoch,
                            'acc': val_acc,
                            'config': config
                        }

                        print(f"  New best model! Accuracy: {best_acc * 100:.2f}%")

            # Save results
            if is_main_process():
                # Save best model
                best_model_path = os.path.join(save_dir, "best_task_variational_model.pt")
                print(f"Saving best model to {best_model_path}")
                torch.save(best_model_state, best_model_path)

                # Save a copy with standard name for compatibility
                torch.save(best_model_state, os.path.join(save_dir, "best_precomputed_model.pt"))

                # Save training history
                history = {
                    'train_losses': train_losses,
                    'kl_losses': kl_losses,
                    'val_accuracies': val_accuracies,
                    'variational_params_history': variational_params_history,
                    'best_acc': float(best_acc),
                    'config': config if 'config' in locals() else {},
                    'use_augmentation': args.use_augmentation,
                }

                def convert_numpy_types(obj):
                    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
                        return float(obj)
                    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
                        return int(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, list):
                        return [convert_numpy_types(item) for item in obj]
                    elif isinstance(obj, dict):
                        return {key: convert_numpy_types(value) for key, value in obj.items()}
                    else:
                        return obj

                history = convert_numpy_types(history)

                history_path = os.path.join(save_dir, "task_variational_training_history.json")
                print(f"Saving training history to {history_path}")

                with open(history_path, 'w') as f:
                    # Convert numpy values to Python types for JSON serialization
                    json.dump(history, f, indent=4)

                # Create plots directory within the model directory
                plot_dir = os.path.join(model_save_dir, "training_plots")
                os.makedirs(plot_dir, exist_ok=True)

                print(f"Training completed for {dataset_name} with Task-level Variational MetaNet. Best accuracy: {best_acc * 100:.2f}%")

        except Exception as e:
            if is_main_process():
                print(f"Error processing dataset {dataset_name}: {e}")
                traceback.print_exc()
        finally:
            # Clean up dataset resources
            if is_main_process():
                print(f"Cleaning up resources for {dataset_name}...")
            cleanup_resources(dataset)
            torch.cuda.empty_cache()
            gc.collect()
            if is_main_process():
                print(f"Finished processing {dataset_name}")

    # Clean up distributed environment
    if is_main_process():
        print("Training complete. Cleaning up distributed environment...")
    cleanup_ddp()


if __name__ == "__main__":
    try:
        print("Starting task-level variational training...")
        torch.multiprocessing.spawn(train_with_task_variational, args=(args,), nprocs=args.world_size)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed with error: {e}")
        traceback.print_exc()