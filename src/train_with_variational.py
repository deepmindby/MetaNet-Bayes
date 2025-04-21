"""
Training script for Variational MetaNet using pre-computed features.

This script provides a training implementation that properly models
the posterior distribution through variational inference.
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

from src.variational_metanet import VariationalMetaNet
from src.utils import cosine_lr
from src.datasets.common import maybe_dictionarize
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp
from src.utils_variational import (
    get_uncertainty_metrics,
    visualize_posterior_distribution,
    save_uncertainty_analysis
)

from src.args import parse_arguments

args = parse_arguments()

# Add variational-specific arguments
args.kl_weight = 0.1  # Weight for KL divergence in ELBO
args.num_samples = 5  # Number of samples to draw from posterior during training
args.visualize_posterior = False  # Whether to visualize posterior distributions
args.variational_gating = True  # Whether to use variational gating mechanism


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
        except Exception as e:
            print(f"Error loading features from {features_path}: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to load features from {features_path}: {e}")

        try:
            self.labels = torch.load(labels_path)
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
                        else:
                            print(f"Warning: Augmented version {aug_idx + 1} has mismatched shape, skipping")
                    except Exception as e:
                        print(f"Error loading augmented version {aug_idx + 1}: {e}")
                        if verbose:
                            traceback.print_exc()

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

        # Define file paths - direct and simple
        train_features_path = os.path.join(feature_dir, "train_features.pt")
        train_labels_path = os.path.join(feature_dir, "train_labels.pt")
        val_features_path = os.path.join(feature_dir, "val_features.pt")
        val_labels_path = os.path.join(feature_dir, "val_labels.pt")

        # Check if train files exist
        if not os.path.exists(train_features_path):
            raise FileNotFoundError(f"Train features not found at {train_features_path}")

        # Find augmented versions
        augmentation_paths = []
        aug_idx = 1

        while True:
            aug_feat_path = os.path.join(feature_dir, f"train_features_aug{aug_idx}.pt")
            aug_label_path = os.path.join(feature_dir, f"train_labels_aug{aug_idx}.pt")

            if os.path.exists(aug_feat_path) and os.path.exists(aug_label_path):
                augmentation_paths.append((aug_feat_path, aug_label_path))
                aug_idx += 1
            else:
                break

        # Create train dataset with augmentation support
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


def evaluate_model(model, classifier, dataset, device, num_eval_samples=10, save_uncertainty=False, save_dir=None):
    """Evaluate model on dataset with Monte Carlo sampling

    Parameters:
    ----------
    model: VariationalMetaNet
        The model to evaluate
    classifier: nn.Module
        Classification head
    dataset: PrecomputedFeatures
        Dataset container
    device: torch.device
        Device to use for evaluation
    num_eval_samples: int
        Number of Monte Carlo samples to draw for evaluation
    save_uncertainty: bool
        Whether to save uncertainty analysis plots
    save_dir: str
        Directory to save uncertainty plots

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

    # Get proper classnames
    classnames = dataset.classnames if hasattr(dataset, 'classnames') else None

    with torch.no_grad():
        for batch in dataset.test_loader:
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
            total_correct += (predictions == labels.cpu()).sum().item()

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

    # Calculate additional metrics for uncertainty analysis
    metrics = {
        "accuracy": accuracy,
        "num_correct": total_correct,
        "num_samples": total_samples,
    }

    # Get gating statistics if available
    if hasattr(model, 'get_gating_stats'):
        gating_stats = model.get_gating_stats()
        metrics.update(gating_stats)

    # Save uncertainty analysis if requested
    if save_uncertainty and save_dir is not None:
        # Rerun a small batch for visualization
        sample_batch = None
        for batch in dataset.test_loader:
            sample_batch = batch
            break

        if sample_batch is not None:
            sample_batch = maybe_dictionarize(sample_batch)
            sample_features = sample_batch["features"].to(device)
            sample_labels = sample_batch["labels"].to(device)

            # Get posterior statistics
            posterior_stats = model.get_posterior_stats(sample_features[:20])

            # Visualize posterior
            figs = visualize_posterior_distribution(
                posterior_stats,
                num_display=5,
                num_task_vectors=model.num_task_vectors,
                blockwise=model.blockwise
            )

            # Save figures
            os.makedirs(save_dir, exist_ok=True)
            for i, fig in enumerate(figs):
                fig.savefig(os.path.join(save_dir, f"posterior_sample_{i}.png"), dpi=300)
                plt.close(fig)

            # Get uncertainty metrics
            prediction_stats = model.monte_carlo_predictions(
                sample_features, classifier, num_samples=num_eval_samples
            )

            uncertainty_metrics = save_uncertainty_analysis(
                prediction_stats,
                sample_labels,
                save_dir=save_dir,
                prefix="uncertainty"
            )

            # Add uncertainty metrics to results
            metrics.update({f"uncertainty_{k}": v for k, v in uncertainty_metrics.items()})

    return metrics


def train_with_variational(rank, args):
    """Main training function with variational inference"""
    args.rank = rank

    # Initialize distributed setup
    setup_ddp(args.rank, args.world_size, port=args.port)

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
        print(f"Using Variational MetaNet: True")
        print(f"Using blockwise coefficients: {args.blockwise_coef}")
        print(f"Using gating mechanism: {args.variational_gating}")
        print(f"Base threshold (αT): {args.base_threshold:.4f}")
        print(f"Beta (β): {args.beta:.4f}")
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
            print(f"=== Training on {dataset_name} with Variational MetaNet ===")

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
            dataset = PrecomputedFeatures(
                feature_dir=feature_dir,
                batch_size=args.batch_size,
                num_workers=2,  # Reduced for stability
                use_augmentation=args.use_augmentation,
            )

            # Get feature dimension from dataset
            sample_batch = next(iter(dataset.train_loader))
            sample_batch = maybe_dictionarize(sample_batch)
            feature_dim = sample_batch["features"].shape[1]
            if is_main_process():
                print(f"Feature dimension: {feature_dim}")

            # Create variational metanet model
            model = VariationalMetaNet(
                feature_dim=feature_dim,
                task_vectors=args.num_task_vectors,
                blockwise=args.blockwise_coef,
                base_threshold=args.base_threshold,
                beta=args.beta,
                uncertainty_reg=args.uncertainty_reg,
                reg_coefficient=args.reg_coefficient if hasattr(args, 'reg_coefficient') else 0.001,
                margin_weight=args.margin_weight if hasattr(args, 'margin_weight') else 0.0001,
                kl_weight=args.kl_weight,
                num_samples=args.num_samples,
                gating_enabled=args.variational_gating
            )
            if is_main_process():
                print(
                    f"Created VariationalMetaNet with {'blockwise' if args.blockwise_coef else 'global'} coefficients")
                print(f"Gating enabled: {args.variational_gating}")
                print(f"Using {args.num_samples} MC samples during training")

            model = model.to(rank)

            data_loader = dataset.train_loader
            num_batches = len(data_loader)

            # Set print frequency
            print_every = max(num_batches // 10, 1)

            # Distributed training setup
            ddp_loader = distribute_loader(data_loader)
            ddp_model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.rank],
                find_unused_parameters=True
            )

            # Setup classifier layer
            num_classes = len(dataset.classnames)
            classifier = torch.nn.Linear(feature_dim, num_classes).cuda()

            # Setup optimizer with parameter groups
            # Group parameters to allow different learning rates
            mean_net_params = []
            logvar_net_params = []
            gating_log_params = []
            other_params = []

            # Separate parameters by type
            for name, param in ddp_model.named_parameters():
                if 'mean_net' in name:
                    mean_net_params.append(param)
                elif 'logvar_net' in name:
                    logvar_net_params.append(param)
                elif 'log_beta' in name or 'log_base_threshold' in name:
                    gating_log_params.append(param)
                else:
                    other_params.append(param)

            # Add classifier parameters
            other_params.extend(list(classifier.parameters()))

            # Create parameter groups with different learning rates
            param_groups = [
                {'params': mean_net_params, 'lr': args.lr * 2.0, 'weight_decay': 0.001},  # Higher LR for mean
                {'params': logvar_net_params, 'lr': args.lr * 1.0, 'weight_decay': 0.0005},  # Lower for variance
                {'params': gating_log_params, 'lr': args.lr * args.lr_multiplier, 'weight_decay': args.weight_decay},
                {'params': other_params, 'lr': args.lr, 'weight_decay': args.wd}
            ]

            optimizer = torch.optim.AdamW(param_groups)

            # Learning rate scheduler
            scheduler = cosine_lr(
                optimizer,
                args.lr,
                0,
                args.epochs * num_batches
            )

            # Loss function
            loss_fn = torch.nn.CrossEntropyLoss()

            # Initialize GradScaler for mixed precision training
            scaler = GradScaler(enabled=True)

            # Training monitoring
            train_losses = []
            kl_losses = []
            reg_losses = []
            val_accuracies = []
            gating_stats = []
            uncertainty_metrics = []
            predictive_variances = []
            base_threshold_values = []
            beta_values = []
            log_base_threshold_values = []
            log_beta_values = []
            best_acc = 0.0
            best_model_state = None

            # Training loop
            for epoch in range(args.epochs):
                ddp_model.train()
                classifier.train()

                epoch_loss = 0.0
                epoch_kl_loss = 0.0
                epoch_reg_loss = 0.0
                batch_count = 0

                if is_main_process():
                    print(f"\nEpoch {epoch + 1}/{args.epochs} - Training")

                for i, batch in enumerate(ddp_loader):
                    start_time = time.time()

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

                            # Variational regularization losses (KL + uncertainty)
                            kl_loss = ddp_model.module.kl_divergence_loss()
                            reg_loss = ddp_model.module.uncertainty_regularization_loss()

                            # Total loss for ELBO optimization
                            total_loss = task_loss + kl_loss + reg_loss

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
                        reg_loss_cpu = reg_loss.item()
                        batch_count += 1
                        epoch_loss += task_loss_cpu
                        epoch_kl_loss += kl_loss_cpu
                        epoch_reg_loss += reg_loss_cpu

                        if is_main_process():
                            train_losses.append(task_loss_cpu)
                            kl_losses.append(kl_loss_cpu)
                            reg_losses.append(reg_loss_cpu)

                            # Get current gating and uncertainty parameters
                            stats = ddp_model.module.get_gating_stats()
                            if stats:
                                gating_stats.append(stats)

                                # Extract key parameters for tracking
                                base_threshold_values.append(stats["base_threshold"])
                                beta_values.append(stats["beta"])
                                log_base_threshold_values.append(stats["log_base_threshold"])
                                log_beta_values.append(stats["log_beta"])
                                predictive_variances.append(stats["predictive_variance"])

                        # Print progress (with reduced frequency)
                        if i % print_every == 0 and is_main_process():
                            # Get current parameters
                            if is_main_process() and stats:
                                current_base_threshold = stats["base_threshold"]
                                current_beta = stats["beta"]
                                gating_ratio = stats.get("gating_ratio", 0.0)
                                pred_var = stats.get("predictive_variance", 0.0)

                                # Detailed output
                                print(f"  Batch {i:4d}/{num_batches:4d} | Task: {task_loss_cpu:.4f} | "
                                      f"KL: {kl_loss_cpu:.4f} | Reg: {reg_loss_cpu:.4f} | "
                                      f"αT: {current_base_threshold:.4f} | β: {current_beta:.4f} | "
                                      f"Gate: {gating_ratio * 100:.1f}% | Var: {pred_var:.4f} | "
                                      f"t: {time.time() - start_time:.2f}s")
                            else:
                                # Simple output
                                print(f"  Batch {i:4d}/{num_batches:4d} | Task: {task_loss_cpu:.4f} | "
                                      f"KL: {kl_loss_cpu:.4f} | Reg: {reg_loss_cpu:.4f} | "
                                      f"t: {time.time() - start_time:.2f}s")

                    except Exception as e:
                        if is_main_process():
                            print(f"  Error in batch {i}: {e}")
                            traceback.print_exc()
                        # Skip this batch but continue training
                        continue

                # Record epoch stats
                avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
                avg_epoch_kl_loss = epoch_kl_loss / batch_count if batch_count > 0 else 0
                avg_epoch_reg_loss = epoch_reg_loss / batch_count if batch_count > 0 else 0

                # Print epoch summary
                if is_main_process():
                    if gating_stats:
                        latest_stats = gating_stats[-1]
                        print(f"  Summary: Task: {avg_epoch_loss:.4f} | KL: {avg_epoch_kl_loss:.4f} | "
                              f"Reg: {avg_epoch_reg_loss:.4f} | αT: {latest_stats['base_threshold']:.4f} | "
                              f"β: {latest_stats['beta']:.4f} | Gate: {latest_stats['gating_ratio'] * 100:.1f}% | "
                              f"Var: {latest_stats['predictive_variance']:.4f}")
                    else:
                        print(f"  Summary: Task: {avg_epoch_loss:.4f} | KL: {avg_epoch_kl_loss:.4f} | "
                              f"Reg: {avg_epoch_reg_loss:.4f}")

                # Evaluate on validation set
                if is_main_process():
                    print(f"Epoch {epoch + 1}/{args.epochs} - Validation")

                    # Create visualization directory for this epoch
                    if args.visualize_posterior:
                        vis_dir = os.path.join(save_dir, f"posterior_epoch_{epoch + 1}")
                        os.makedirs(vis_dir, exist_ok=True)
                    else:
                        vis_dir = None

                    # Evaluate with Monte Carlo sampling and uncertainty analysis
                    eval_results = evaluate_model(
                        model=ddp_model.module,
                        classifier=classifier,
                        dataset=dataset,
                        device=rank,
                        num_eval_samples=10,  # More samples during evaluation
                        save_uncertainty=args.visualize_posterior and (epoch % 5 == 0),  # Visualize every 5 epochs
                        save_dir=vis_dir
                    )

                    val_acc = eval_results["accuracy"]
                    val_accuracies.append(val_acc)
                    uncertainty_metrics.append(eval_results)

                    # Print evaluation results
                    print(
                        f"  Accuracy: {val_acc * 100:.2f}% ({eval_results['num_correct']}/{eval_results['num_samples']})")

                    # Print uncertainty metrics if available
                    if "uncertainty_ece" in eval_results:
                        print(f"  ECE: {eval_results['uncertainty_ece']:.4f} | "
                              f"Entropy: {eval_results['uncertainty_avg_predictive_entropy']:.4f}")

                        if "uncertainty_correct_epistemic_uncertainty" in eval_results:
                            print(f"  Correct vs. Incorrect Uncertainty: "
                                  f"{eval_results['uncertainty_correct_epistemic_uncertainty']:.4f} vs. "
                                  f"{eval_results['uncertainty_incorrect_epistemic_uncertainty']:.4f}")

                    # Save best model
                    if val_acc > best_acc:
                        best_acc = val_acc

                        # Store model configuration along with weights
                        config = {
                            'feature_dim': feature_dim,
                            'num_task_vectors': args.num_task_vectors,
                            'blockwise': args.blockwise_coef,
                            'base_threshold': eval_results.get("base_threshold", args.base_threshold),
                            'beta': eval_results.get("beta", args.beta),
                            'log_base_threshold': eval_results.get("log_base_threshold", 0.0),
                            'log_beta': eval_results.get("log_beta", 0.0),
                            'uncertainty_reg': args.uncertainty_reg,
                            'kl_weight': args.kl_weight,
                            'model_name': args.model,
                            'use_augmentation': args.use_augmentation,
                            'gating_enabled': args.variational_gating,
                            'num_samples': args.num_samples,
                            'variational': True  # Mark as variational model
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
                best_model_path = os.path.join(save_dir, "best_variational_model.pt")
                print(f"Saving best model to {best_model_path}")
                torch.save(best_model_state, best_model_path)

                # Save a copy with standard name for compatibility
                torch.save(best_model_state, os.path.join(save_dir, "best_precomputed_model.pt"))

                # Save training history
                history = {
                    'train_losses': train_losses,
                    'kl_losses': kl_losses,
                    'reg_losses': reg_losses,
                    'val_accuracies': val_accuracies,
                    'gating_stats': [
                        {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in stats.items()}
                        for stats in gating_stats
                    ],
                    'uncertainty_metrics': [
                        {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()}
                        for metrics in uncertainty_metrics
                    ],
                    'predictive_variances': predictive_variances,
                    'base_threshold_values': base_threshold_values,
                    'beta_values': beta_values,
                    'log_base_threshold_values': log_base_threshold_values,
                    'log_beta_values': log_beta_values,
                    'best_acc': float(best_acc),
                    'config': config if 'config' in locals() else {},
                    'use_augmentation': args.use_augmentation,
                    'gating_enabled': args.variational_gating,
                    'num_samples': args.num_samples,
                    'kl_weight': args.kl_weight
                }

                history_path = os.path.join(save_dir, "variational_training_history.json")

                with open(history_path, 'w') as f:
                    # Convert numpy values to Python types for JSON serialization
                    json.dump(history, f, indent=4)

                # Create plots directory within the model directory
                plot_dir = os.path.join(model_save_dir, "training_plots")
                os.makedirs(plot_dir, exist_ok=True)

                # Create plots for training progress
                # -- Loss curves
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

                # Task loss
                ax1.plot(train_losses, 'r-', linewidth=1.5)
                ax1.set_ylabel('Task Loss', fontsize=14)
                ax1.set_title(f'Training Losses for {dataset_name} - Variational MetaNet', fontsize=16)
                ax1.grid(True, alpha=0.7)

                # KL loss
                ax2.plot(kl_losses, 'b-', linewidth=1.5)
                ax2.set_ylabel('KL Loss', fontsize=14)
                ax2.grid(True, alpha=0.7)

                # Regularization loss
                ax3.plot(reg_losses, 'g-', linewidth=1.5)
                ax3.set_xlabel('Iterations', fontsize=14)
                ax3.set_ylabel('Regularization Loss', fontsize=14)
                ax3.grid(True, alpha=0.7)

                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f'{dataset_name}_variational_loss_curves.png'), dpi=300)
                plt.close()

                # -- Validation accuracy
                fig, ax = plt.subplots(figsize=(12, 6))

                epochs_range = range(1, len(val_accuracies) + 1)
                ax.plot(epochs_range, [acc * 100 for acc in val_accuracies], 'b-o',
                        linewidth=2, markersize=8)
                ax.set_xlabel('Epochs', fontsize=14)
                ax.set_ylabel('Accuracy (%)', fontsize=14)
                ax.set_title(f'Validation Accuracy for {dataset_name} - Variational MetaNet', fontsize=16)
                ax.grid(True, alpha=0.7)

                # Add annotation for best accuracy
                best_epoch = np.argmax(val_accuracies) + 1
                best_val_acc = max(val_accuracies) * 100
                ax.annotate(f'Best: {best_val_acc:.2f}%',
                            xy=(best_epoch, best_val_acc),
                            xytext=(best_epoch + 1, best_val_acc - 5),
                            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                            fontsize=12)

                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f'{dataset_name}_variational_accuracy.png'), dpi=300)
                plt.close()

                # -- Uncertainty visualization
                if predictive_variances:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(predictive_variances, 'k-', linewidth=1.5)
                    ax.set_xlabel('Iterations', fontsize=14)
                    ax.set_ylabel('Average Predictive Variance', fontsize=14)
                    ax.set_title(f'Posterior Variance Evolution for {dataset_name}', fontsize=16)
                    ax.grid(True, alpha=0.7)
                    plt.tight_layout()
                    plt.savefig(os.path.join(plot_dir, f'{dataset_name}_predictive_variance.png'), dpi=300)
                    plt.close()

                # -- Parameter evolution (if gating enabled)
                if args.variational_gating and base_threshold_values and beta_values:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

                    # Plot actual parameters
                    ax1.plot(epochs_range, base_threshold_values[:len(epochs_range)], 'r-o',
                             linewidth=2, markersize=8, label='αT')
                    ax1.plot(epochs_range, beta_values[:len(epochs_range)], 'g-o',
                             linewidth=2, markersize=8, label='β')
                    ax1.set_ylabel('Parameter Value', fontsize=14)
                    ax1.set_title(f'Gating Parameters Evolution for {dataset_name} - Variational MetaNet', fontsize=16)
                    ax1.legend(fontsize=12)
                    ax1.grid(True, alpha=0.7)

                    # Add annotations for final values
                    ax1.annotate(f'Final: {base_threshold_values[len(epochs_range) - 1]:.4f}',
                                 xy=(len(epochs_range), base_threshold_values[len(epochs_range) - 1]),
                                 xytext=(len(epochs_range) - 3, base_threshold_values[len(epochs_range) - 1] + 0.02),
                                 arrowprops=dict(facecolor='red', shrink=0.05, width=1.5),
                                 fontsize=10, color='red')

                    ax1.annotate(f'Final: {beta_values[len(epochs_range) - 1]:.4f}',
                                 xy=(len(epochs_range), beta_values[len(epochs_range) - 1]),
                                 xytext=(len(epochs_range) - 3, beta_values[len(epochs_range) - 1] - 0.05),
                                 arrowprops=dict(facecolor='green', shrink=0.05, width=1.5),
                                 fontsize=10, color='green')

                    # Plot log parameters
                    ax2.plot(epochs_range, log_base_threshold_values[:len(epochs_range)], 'r--o',
                             linewidth=2, markersize=8, label='log(αT)')
                    ax2.plot(epochs_range, log_beta_values[:len(epochs_range)], 'g--o',
                             linewidth=2, markersize=8, label='log(β)')
                    ax2.set_xlabel('Epochs', fontsize=14)
                    ax2.set_ylabel('Log Parameter Value', fontsize=14)
                    ax2.set_title(f'Log Gating Parameters Evolution for {dataset_name}', fontsize=16)
                    ax2.legend(fontsize=12)
                    ax2.grid(True, alpha=0.7)

                    plt.tight_layout()
                    plt.savefig(os.path.join(plot_dir, f'{dataset_name}_variational_parameter_evolution.png'), dpi=300)
                    plt.close()

                print(
                    f"Training completed for {dataset_name} with Variational MetaNet. Best accuracy: {best_acc * 100:.2f}%")

        except Exception as e:
            if is_main_process():
                print(f"Error processing dataset {dataset_name}: {e}")
                traceback.print_exc()
        finally:
            # Clean up dataset resources
            cleanup_resources(dataset)
            torch.cuda.empty_cache()
            gc.collect()

    cleanup_ddp()


if __name__ == "__main__":
    try:
        torch.multiprocessing.spawn(train_with_variational, args=(args,), nprocs=args.world_size)
    except Exception as e:
        print(f"Training failed with error: {e}")
        traceback.print_exc()