"""Training Script Using Adaptive Gating MetaNet with Pre-computed Features

This script trains MetaNet models with adaptive gating using pre-computed CLIP features.
"""

import os
import time
import json
import torch
import random
import socket
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler
import gc
import sys
import datetime
import traceback

from src.adaptive_gating_metanet import AdaptiveGatingMetaNet
from src.utils import cosine_lr
from src.datasets.common import maybe_dictionarize
from src.distributed import cleanup_ddp, is_main_process
from src.train_with_precomputed import get_precomputed_dataset


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


def distribute_loader(loader):
    """Distribute data loader across processes"""
    from src.distributed import distribute_loader as dist_loader
    try:
        return dist_loader(loader)
    except Exception as e:
        print(f"Error distributing loader: {e}")
        # Fallback to the original loader if distribution fails
        return loader


def setup_ddp_robust(rank, world_size, port=None):
    """More robust distributed training initialization"""
    # Use random port to avoid conflicts
    if port is None:
        port = random.randint(29500, 65000)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # Add timeout and retry logic
    max_retries = 5
    for retry in range(max_retries):
        try:
            torch.distributed.init_process_group(
                "nccl",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(minutes=1)
            )
            torch.cuda.set_device(rank)
            torch.distributed.barrier()
            print(f"Process {rank}: DDP initialized successfully on port {port}")
            return True
        except Exception as e:
            print(f"Process {rank}: DDP initialization error (attempt {retry+1}/{max_retries}): {e}")
            # Try a different port
            port = port + retry + 1
            os.environ["MASTER_PORT"] = str(port)
            time.sleep(1)

    print(f"Process {rank}: Failed to initialize distributed setup")
    return False


def plot_training_metrics(train_losses, reg_losses, dataset_name, plot_dir):
    """Plot and save training metrics (loss curves) separately

    Args:
        train_losses: List of task loss values
        reg_losses: List of regularization loss values
        dataset_name: Name of the dataset
        plot_dir: Directory to save plots
    """
    # Create figure for loss plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot training loss
    ax1.plot(train_losses, 'r-', linewidth=1.5)
    ax1.set_ylabel('Task Loss', fontsize=14)
    ax1.set_title(f'Training Loss for {dataset_name}', fontsize=16)
    ax1.grid(True, alpha=0.7)

    # Plot regularization loss
    ax2.plot(reg_losses, 'g-', linewidth=1.5)
    ax2.set_xlabel('Iterations', fontsize=14)
    ax2.set_ylabel('Regularization Loss', fontsize=14)
    ax2.set_title(f'Uncertainty Regularization Loss for {dataset_name}', fontsize=16)
    ax2.grid(True, alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{dataset_name}_loss_curves.png'), dpi=300)
    plt.close()


def plot_validation_metrics(val_accuracies, base_threshold_values, beta_values,
                           log_base_threshold_values, log_beta_values,
                           dataset_name, plot_dir):
    """Plot and save validation metrics and parameter evolution separately

    Args:
        val_accuracies: List of validation accuracy values
        base_threshold_values: List of base threshold parameter values
        beta_values: List of beta parameter values
        log_base_threshold_values: List of log base threshold parameter values
        log_beta_values: List of log beta parameter values
        dataset_name: Name of the dataset
        plot_dir: Directory to save plots
    """
    # Create figure for accuracy plot
    fig, ax = plt.subplots(figsize=(12, 6))

    epochs_range = range(1, len(val_accuracies) + 1)
    ax.plot(epochs_range, [acc * 100 for acc in val_accuracies], 'b-o',
            linewidth=2, markersize=8)
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title(f'Validation Accuracy for {dataset_name}', fontsize=16)
    ax.grid(True, alpha=0.7)

    # Add annotations for key points
    best_epoch = np.argmax(val_accuracies) + 1
    best_acc = max(val_accuracies) * 100
    ax.annotate(f'Best: {best_acc:.2f}%',
                xy=(best_epoch, best_acc),
                xytext=(best_epoch + 1, best_acc - 5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{dataset_name}_accuracy.png'), dpi=300)
    plt.close()

    # Create figure for parameter evolution
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot actual parameters
    ax1.plot(epochs_range, base_threshold_values, 'r-o', linewidth=2, markersize=8, label='αT')
    ax1.plot(epochs_range, beta_values, 'g-o', linewidth=2, markersize=8, label='β')
    ax1.set_ylabel('Parameter Value', fontsize=14)
    ax1.set_title(f'Gating Parameters Evolution for {dataset_name}', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.7)

    # Add annotations for final values
    ax1.annotate(f'Final: {base_threshold_values[-1]:.4f}',
                xy=(len(base_threshold_values), base_threshold_values[-1]),
                xytext=(len(base_threshold_values) - 3, base_threshold_values[-1] + 0.02),
                arrowprops=dict(facecolor='red', shrink=0.05, width=1.5),
                fontsize=10, color='red')

    ax1.annotate(f'Final: {beta_values[-1]:.4f}',
                xy=(len(beta_values), beta_values[-1]),
                xytext=(len(beta_values) - 3, beta_values[-1] - 0.05),
                arrowprops=dict(facecolor='green', shrink=0.05, width=1.5),
                fontsize=10, color='green')

    # Plot log parameters
    ax2.plot(epochs_range, log_base_threshold_values, 'r--o', linewidth=2, markersize=8, label='log(αT)')
    ax2.plot(epochs_range, log_beta_values, 'g--o', linewidth=2, markersize=8, label='log(β)')
    ax2.set_xlabel('Epochs', fontsize=14)
    ax2.set_ylabel('Log Parameter Value', fontsize=14)
    ax2.set_title(f'Log Gating Parameters Evolution for {dataset_name}', fontsize=16)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{dataset_name}_parameter_evolution.png'), dpi=300)
    plt.close()


def plot_gradients(grad_magnitudes, dataset_name, plot_dir):
    """Plot and save gradient magnitudes

    Args:
        grad_magnitudes: List of dictionaries containing gradient information
        dataset_name: Name of the dataset
        plot_dir: Directory to save plots
    """
    if not grad_magnitudes:
        return

    plt.figure(figsize=(12, 6))
    iterations = range(len(grad_magnitudes))

    # Extract and plot log base threshold gradients
    log_base_grads = [g.get('log_base', 0) for g in grad_magnitudes]
    plt.plot(iterations, log_base_grads, 'r-', linewidth=1.5, label='log(αT) Gradient')

    # Extract and plot log beta gradients
    log_beta_grads = [g.get('log_beta', 0) for g in grad_magnitudes]
    plt.plot(iterations, log_beta_grads, 'g-', linewidth=1.5, label='log(β) Gradient')

    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Gradient Magnitude', fontsize=14)
    plt.title(f'Parameter Gradients for {dataset_name}', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.7)

    # Set y-axis limits to focus on gradient changes
    y_min = min(min(log_base_grads), min(log_beta_grads))
    y_max = max(max(log_base_grads), max(log_beta_grads))
    y_range = y_max - y_min
    plt.ylim([y_min - 0.1 * y_range, y_max + 0.1 * y_range])

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{dataset_name}_gradient_magnitudes.png'), dpi=300)
    plt.close()


def parse_args():
    """Parse command line arguments"""
    from src.args import parse_arguments
    args = parse_arguments()

    # Set default for new parameters if not specified
    if not hasattr(args, 'base_threshold') or args.base_threshold is None:
        args.base_threshold = 0.05
    if not hasattr(args, 'beta') or args.beta is None:
        args.beta = 1.0
    if not hasattr(args, 'uncertainty_reg') or args.uncertainty_reg is None:
        args.uncertainty_reg = 0.01

    return args


def main(rank, args):
    """Main training function"""
    args.rank = rank

    # Initialize distributed setup with robust method
    if not setup_ddp_robust(args.rank, args.world_size, args.port):
        print(f"Process {rank}: Failed to initialize distributed setup. Exiting.")
        return

    # Process datasets
    if hasattr(args, 'datasets') and args.datasets:
        datasets_to_process = args.datasets
    else:
        datasets_to_process = ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"]
        # datasets_to_process = ["Cars"]

    # Fix save directory path
    if not hasattr(args, 'save') or args.save is None:
        current_dir = os.getcwd()
        if current_dir.endswith('MetaNet-Bayes'):
            args.save = os.path.join(current_dir, "checkpoints_adaptive_gating")
        else:
            args.save = os.path.join(current_dir, "MetaNet-Bayes", "checkpoints_adaptive_gating")
        if rank == 0:
            print(f"Using save directory: {args.save}")

    # Create save directory
    if is_main_process():
        os.makedirs(args.save, exist_ok=True)

    # Fix data location path
    if not hasattr(args, 'data_location') or args.data_location is None:
        current_dir = os.getcwd()
        if current_dir.endswith('MetaNet-Bayes'):
            args.data_location = current_dir
        else:
            args.data_location = os.path.join(current_dir, "MetaNet-Bayes")
        if "MetaNet-Bayes/MetaNet-Bayes" in args.data_location:
            args.data_location = args.data_location.replace("MetaNet-Bayes/MetaNet-Bayes", "MetaNet-Bayes")

    # Print configuration
    if rank == 0:
        print(f"\n=== Training Configuration ===")
        print(f"Model: {args.model}")
        print(f"Blockwise coefficients: {args.blockwise_coef}")
        print(f"Initial αT: {args.base_threshold:.4f}, β: {args.beta:.4f}")
        print(f"Uncertainty regularization: {args.uncertainty_reg}")
        print(f"Causal intervention: {args.causal_intervention if hasattr(args, 'causal_intervention') else False}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.lr}")
        print(f"Save directory: {args.save}")
        print(f"Data location: {args.data_location}")
        print("=" * 30)

    for dataset_name in datasets_to_process:
        target_dataset = f"precomputed_{dataset_name}Val"  # Use precomputed features
        if is_main_process():
            print(f"=== Training on {dataset_name} with adaptive gating ===")

        # Setup save directory for this dataset
        save_dir = os.path.join(args.save, dataset_name + "Val")
        if is_main_process():
            os.makedirs(save_dir, exist_ok=True)

        dataset = None

        try:
            # Get dataset with precomputed features
            dataset = get_precomputed_dataset(
                dataset_name=target_dataset,
                model_name=args.model,
                location=args.data_location,
                batch_size=args.batch_size,
                num_workers=2,  # Reduced for stability
                verbose=True
            )

            # Sample feature to get dimensions
            try:
                sample_batch = next(iter(dataset.train_loader))
                sample_batch = maybe_dictionarize(sample_batch)
                feature_dim = sample_batch["features"].shape[1]
                if is_main_process():
                    print(f"Feature dimension: {feature_dim}")
            except Exception as e:
                if is_main_process():
                    print(f"Error getting feature dimension: {e}")
                    traceback.print_exc()
                raise

            # Create model
            try:
                model = AdaptiveGatingMetaNet(
                    feature_dim=feature_dim,
                    task_vectors=args.num_task_vectors if hasattr(args, 'num_task_vectors') else 8,
                    blockwise=args.blockwise_coef if hasattr(args, 'blockwise_coef') else False,
                    base_threshold=args.base_threshold,
                    beta=args.beta,
                    uncertainty_reg=args.uncertainty_reg
                )

                # Handle causal intervention compatibility
                if hasattr(args, 'causal_intervention') and args.causal_intervention:
                    if is_main_process():
                        print("Note: Both causal intervention and adaptive gating are enabled")
                        print("Adaptive gating will be used as the primary intervention mechanism")

                    # Add causal_intervention attribute if needed by other parts of code
                    model.causal_intervention = True

                model = model.cuda()
            except Exception as e:
                if is_main_process():
                    print(f"Error creating model: {e}")
                    traceback.print_exc()
                raise

            data_loader = dataset.train_loader
            num_batches = len(data_loader)

            # Set print frequency
            print_every = max(num_batches // 10, 1)

            # Single GPU training instead of DDP to avoid issues with unused parameters
            if args.world_size <= 1:
                # Single GPU training setup
                print("Using single GPU training")
                ddp_model = model
                ddp_loader = data_loader
            else:
                # Distributed training setup
                try:
                    ddp_loader = distribute_loader(data_loader)
                    ddp_model = torch.nn.parallel.DistributedDataParallel(
                        model,
                        device_ids=[args.rank],
                        find_unused_parameters=False  # Changed to False to fix warning
                    )
                except Exception as e:
                    if is_main_process():
                        print(f"Error in distributed setup, falling back to single GPU: {e}")
                        ddp_model = model
                        ddp_loader = data_loader

            # Setup classifier layer
            num_classes = len(dataset.classnames)
            classifier = torch.nn.Linear(feature_dim, num_classes).cuda()

            # Setup optimizer with different parameter groups
            try:
                # Group parameters to allow different learning rates
                gating_log_params = []
                meta_net_params = []
                other_params = []

                # Separate parameters by type
                for name, param in ddp_model.named_parameters():
                    if 'log_beta' in name:
                        gating_log_params.append(param)
                    elif 'log_base_threshold' in name:
                        gating_log_params.append(param)
                    elif 'meta_net' in name:
                        meta_net_params.append(param)
                    else:
                        other_params.append(param)

                # Add classifier parameters
                other_params.extend(list(classifier.parameters()))

                # Create parameter groups with different learning rates
                param_groups = [
                    {'params': gating_log_params, 'lr': args.lr * 100, 'weight_decay': 0.0001},  # Very high LR for gating params
                    {'params': meta_net_params, 'lr': args.lr * 2, 'weight_decay': 0.001},      # Higher LR for meta_net
                    {'params': other_params, 'lr': args.lr, 'weight_decay': args.wd if hasattr(args, 'wd') else 0.01}
                ]

                optimizer = torch.optim.AdamW(param_groups)

                # Learning rate scheduler
                scheduler = cosine_lr(
                    optimizer,
                    args.lr if hasattr(args, 'lr') else 1e-3,
                    0,
                    args.epochs * num_batches if hasattr(args, 'epochs') else 10 * num_batches
                )

                # Loss function
                loss_fn = torch.nn.CrossEntropyLoss()

                # Mixed precision training
                scaler = GradScaler()
            except Exception as e:
                if is_main_process():
                    print(f"Error setting up optimizer: {e}")
                    traceback.print_exc()
                raise

            # Training monitoring
            train_losses = []
            reg_losses = []
            val_accuracies = []
            gating_stats = []
            base_threshold_values = []  # Track base threshold values
            beta_values = []  # Track beta values
            log_base_threshold_values = []  # Track log base threshold values
            log_beta_values = []  # Track log beta values
            grad_magnitudes = []  # Track gradient magnitudes
            best_acc = 0.0
            best_model_state = None

            # Training loop
            num_epochs = args.epochs if hasattr(args, 'epochs') else 10
            for epoch in range(num_epochs):
                ddp_model.train()
                classifier.train()

                epoch_loss = 0.0
                epoch_reg_loss = 0.0
                batch_count = 0

                if is_main_process():
                    print(f"\nEpoch {epoch+1}/{num_epochs} - Training")

                for i, batch in enumerate(ddp_loader):
                    start_time = time.time()

                    try:
                        batch = maybe_dictionarize(batch)
                        features = batch["features"].to(rank)
                        labels = batch["labels"].to(rank)

                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            # Forward pass
                            transformed_features = ddp_model(features)
                            logits = classifier(transformed_features)

                            # Task loss
                            task_loss = loss_fn(logits, labels)

                            # Add uncertainty regularization - use the current model instance
                            if isinstance(ddp_model, torch.nn.parallel.DistributedDataParallel):
                                reg_loss = ddp_model.module.uncertainty_regularization_loss()
                            else:
                                reg_loss = ddp_model.uncertainty_regularization_loss()

                            total_loss = task_loss + reg_loss

                        # Backward pass
                        scaler.scale(total_loss).backward()

                        # Check log parameters' gradients (only for periodic logging)
                        if is_main_process() and i % print_every == 0:
                            # Get original model reference
                            if isinstance(ddp_model, torch.nn.parallel.DistributedDataParallel):
                                model_ref = ddp_model.module
                            else:
                                model_ref = ddp_model

                            # Get gradients - use item() or None check
                            log_base_grad = model_ref.log_base_threshold.grad
                            log_beta_grad = model_ref.log_beta.grad

                            # Save gradient information for later plotting
                            grad_magnitudes.append({
                                'log_base': log_base_grad.item() if log_base_grad is not None else 0.0,
                                'log_beta': log_beta_grad.item() if log_beta_grad is not None else 0.0
                            })

                        # Step optimizer
                        scheduler(i + epoch * num_batches)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                        # Record stats
                        task_loss_cpu = task_loss.item()
                        reg_loss_cpu = reg_loss.item()
                        batch_count += 1
                        epoch_loss += task_loss_cpu
                        epoch_reg_loss += reg_loss_cpu

                        if is_main_process():
                            train_losses.append(task_loss_cpu)
                            reg_losses.append(reg_loss_cpu)

                        # Print progress (only with reduced frequency)
                        if i % print_every == 0 and is_main_process():
                            # Get original model reference
                            if isinstance(ddp_model, torch.nn.parallel.DistributedDataParallel):
                                model_ref = ddp_model.module
                            else:
                                model_ref = ddp_model

                            # Get current gating parameters
                            current_base_threshold = float(model_ref.base_threshold.item())
                            current_beta = float(model_ref.beta.item())

                            # Get gating statistics if available
                            gating_ratio = 0.0
                            if hasattr(model_ref, 'get_gating_stats'):
                                stats = model_ref.get_gating_stats()
                                if stats:
                                    gating_stats.append(stats)
                                    gating_ratio = stats.get('gating_ratio', 0.0)

                            # Compact, single-line progress output with shortened elapsed time
                            t_elapsed = time.time() - start_time
                            print(f"  Batch {i:4d}/{num_batches:4d} | Loss: {task_loss_cpu:.4f} | Reg: {reg_loss_cpu:.4f} | αT: {current_base_threshold:.4f} | β: {current_beta:.4f} | Gate: {gating_ratio:.3f} | t: {t_elapsed:.2f}s")

                    except Exception as e:
                        if is_main_process():
                            print(f"  Error in batch {i}: {e}")
                        # Skip this batch but continue training
                        continue

                # Record epoch stats
                avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
                avg_epoch_reg_loss = epoch_reg_loss / batch_count if batch_count > 0 else 0

                # Record parameter values at the end of each epoch
                if is_main_process():
                    # Get original model reference
                    if isinstance(ddp_model, torch.nn.parallel.DistributedDataParallel):
                        model_ref = ddp_model.module
                    else:
                        model_ref = ddp_model

                    # Get current parameters
                    current_base_threshold = float(model_ref.base_threshold.item())
                    current_beta = float(model_ref.beta.item())
                    current_log_base_threshold = float(model_ref.log_base_threshold.item())
                    current_log_beta = float(model_ref.log_beta.item())

                    # Save to lists
                    base_threshold_values.append(current_base_threshold)
                    beta_values.append(current_beta)
                    log_base_threshold_values.append(current_log_base_threshold)
                    log_beta_values.append(current_log_beta)

                    # Epoch summary
                    print(f"  Summary: Task Loss: {avg_epoch_loss:.4f} | Reg Loss: {avg_epoch_reg_loss:.4f} | αT: {current_base_threshold:.4f} | β: {current_beta:.4f}")

                # Evaluate on validation set
                if is_main_process():
                    print(f"Epoch {epoch+1}/{num_epochs} - Validation")

                    # Create a combined model for evaluation
                    class CombinedModel(torch.nn.Module):
                        def __init__(self, feature_model, classifier):
                            super().__init__()
                            self.feature_model = feature_model
                            self.classifier = classifier

                        def forward(self, x):
                            return self.classifier(self.feature_model(x))

                    # Evaluate using appropriate model
                    if isinstance(ddp_model, torch.nn.parallel.DistributedDataParallel):
                        eval_model = CombinedModel(ddp_model.module, classifier)
                    else:
                        eval_model = CombinedModel(ddp_model, classifier)

                    eval_model.eval()

                    # Validation loop
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for batch in dataset.test_loader:
                            batch = maybe_dictionarize(batch)
                            features = batch["features"].cuda()
                            labels = batch["labels"].cuda()

                            # Forward pass
                            outputs = eval_model(features)
                            _, predicted = outputs.max(1)

                            total += labels.size(0)
                            correct += predicted.eq(labels).sum().item()

                    val_acc = correct / total
                    val_accuracies.append(val_acc)

                    print(f"  Accuracy: {val_acc*100:.2f}% ({correct}/{total})")

                    # Save best model
                    if val_acc > best_acc:
                        best_acc = val_acc

                        # Store model configuration along with weights
                        config = {
                            'feature_dim': feature_dim,
                            'num_task_vectors': args.num_task_vectors if hasattr(args, 'num_task_vectors') else 8,
                            'blockwise': args.blockwise_coef if hasattr(args, 'blockwise_coef') else False,
                            'base_threshold': current_base_threshold,
                            'beta': current_beta,
                            'log_base_threshold': current_log_base_threshold,
                            'log_beta': current_log_beta,
                            'uncertainty_reg': args.uncertainty_reg,
                            'model_name': args.model,
                            'causal_intervention': args.causal_intervention if hasattr(args, 'causal_intervention') else False
                        }

                        # Get appropriate state dict
                        if isinstance(ddp_model, torch.nn.parallel.DistributedDataParallel):
                            model_state_dict = ddp_model.module.state_dict()
                        else:
                            model_state_dict = ddp_model.state_dict()

                        best_model_state = {
                            'meta_net': model_state_dict,
                            'classifier': classifier.state_dict(),
                            'epoch': epoch,
                            'acc': val_acc,
                            'config': config
                        }

                        print(f"  New best model! αT: {current_base_threshold:.4f}, β: {current_beta:.4f}")

            # Save results
            if is_main_process():
                # Save best model
                if best_model_state:
                    best_model_path = os.path.join(save_dir, "best_adaptive_gating_model.pt")
                    print(f"Saving best model to {best_model_path}")
                    torch.save(best_model_state, best_model_path)

                # Save training history
                history = {
                    'train_losses': train_losses,
                    'reg_losses': reg_losses,
                    'val_accuracies': val_accuracies,
                    'gating_stats': gating_stats,
                    'base_threshold_values': base_threshold_values,
                    'beta_values': beta_values,
                    'log_base_threshold_values': log_base_threshold_values,
                    'log_beta_values': log_beta_values,
                    'grad_magnitudes': grad_magnitudes,
                    'best_acc': best_acc,
                    'config': config if 'config' in locals() else {}
                }

                with open(os.path.join(save_dir, "adaptive_gating_training_history.json"), 'w') as f:
                    # Convert numpy values to Python types
                    for key in history:
                        if isinstance(history[key], (list, dict)) and key not in ['gating_stats', 'grad_magnitudes']:
                            history[key] = [float(item) if isinstance(item, (np.floating, np.integer)) else item
                                           for item in history[key]]

                    json.dump(history, f, indent=4)

                # Create plots directory
                plot_dir = os.path.join(args.save, "adaptive_gating_plots")
                os.makedirs(plot_dir, exist_ok=True)

                # Create separate plots for loss curves
                plot_training_metrics(train_losses, reg_losses, dataset_name, plot_dir)

                # Create separate plots for validation accuracy and parameter evolution
                plot_validation_metrics(
                    val_accuracies, base_threshold_values, beta_values,
                    log_base_threshold_values, log_beta_values,
                    dataset_name, plot_dir
                )

                # Create plot for gradient magnitudes
                plot_gradients(grad_magnitudes, dataset_name, plot_dir)

                print(f"Training completed for {dataset_name}. Best validation accuracy: {best_acc*100:.2f}%")
                print(f"Final parameters - αT: {base_threshold_values[-1]:.4f}, β: {beta_values[-1]:.4f}")

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
    # Parse arguments
    args = parse_args()

    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Set default values for new parameters
    if not hasattr(args, 'num_task_vectors'):
        args.num_task_vectors = 8

    if not hasattr(args, 'lr') or not args.lr:
        args.lr = 5e-3

    # Launch training with better error handling
    try:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size)
    except Exception as e:
        print(f"Training failed with error: {e}")
        traceback.print_exc()
        # Force cleanup
        for i in range(torch.cuda.device_count()):
            if i < args.world_size:
                try:
                    torch.cuda.set_device(i)
                    torch.cuda.empty_cache()
                except:
                    pass