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
        print(f"Using blockwise coefficients: {args.blockwise_coef}")
        print(f"Base threshold: {args.base_threshold}")
        print(f"Beta: {args.beta}")
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

            # Distributed training setup
            try:
                ddp_loader = distribute_loader(data_loader)
                ddp_model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[args.rank],
                    find_unused_parameters=True  # Needed because some parameters might not be used in every forward pass
                )

                # Setup classifier layer
                num_classes = len(dataset.classnames)
                classifier = torch.nn.Linear(feature_dim, num_classes).cuda()
            except Exception as e:
                if is_main_process():
                    print(f"Error in distributed setup: {e}")
                    traceback.print_exc()
                raise

            # Setup optimizer
            try:
                # Group parameters to allow different learning rates
                gating_params = []
                other_params = []

                for name, param in ddp_model.named_parameters():
                    if 'base_threshold' in name or 'beta' in name:
                        gating_params.append(param)
                    else:
                        other_params.append(param)

                # Add classifier parameters to other_params
                other_params.extend(list(classifier.parameters()))

                # Create parameter groups with different learning rates
                param_groups = [
                    {'params': gating_params, 'lr': args.lr * 10},  # if args.lr < 0.001 else args.lr},
                    {'params': other_params, 'lr': args.lr}
                ]

                optimizer = torch.optim.AdamW(
                    param_groups,
                    weight_decay=args.wd if hasattr(args, 'wd') else 0.01
                )

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

                            # Add uncertainty regularization
                            reg_loss = model.uncertainty_regularization_loss()
                            total_loss = task_loss + reg_loss

                        # Backward pass
                        scaler.scale(total_loss).backward()

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

                        # Print progress
                        if i % print_every == 0 and is_main_process():
                            print(f"Epoch {epoch+1}/{num_epochs}, Batch {i}/{num_batches}, "
                                f"Loss: {task_loss_cpu:.6f}, Reg Loss: {reg_loss_cpu:.6f}, "
                                f"Time: {time.time() - start_time:.3f}s")

                            # Get gating statistics if available
                            if hasattr(ddp_model.module, 'get_gating_stats'):
                                stats = ddp_model.module.get_gating_stats()
                                if stats:
                                    gating_stats.append(stats)
                                    print(f"Gating ratio: {stats['gating_ratio']:.3f}, "
                                          f"Avg threshold: {stats['avg_threshold']:.4f}, "
                                          f"Base threshold: {stats['base_threshold']:.4f}, "
                                          f"Beta: {stats['beta']:.4f}")

                    except Exception as e:
                        if is_main_process():
                            print(f"Error in training batch: {e}")
                            traceback.print_exc()
                        # Skip this batch but continue training
                        continue

                # Record epoch stats
                avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
                avg_epoch_reg_loss = epoch_reg_loss / batch_count if batch_count > 0 else 0

                if is_main_process():
                    print(f"Epoch {epoch+1} average - Task Loss: {avg_epoch_loss:.6f}, "
                          f"Reg Loss: {avg_epoch_reg_loss:.6f}")

                # Evaluate on validation set
                if is_main_process():
                    # Create a combined model for evaluation
                    class CombinedModel(torch.nn.Module):
                        def __init__(self, feature_model, classifier):
                            super().__init__()
                            self.feature_model = feature_model
                            self.classifier = classifier

                        def forward(self, x):
                            return self.classifier(self.feature_model(x))

                    # Evaluate without DDP wrapper
                    eval_model = CombinedModel(ddp_model.module, classifier)
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

                    print(f"Epoch {epoch+1}/{num_epochs}, "
                          f"Avg Loss: {avg_epoch_loss:.6f}, "
                          f"Val Acc: {val_acc*100:.2f}%")

                    # Save best model
                    if val_acc > best_acc:
                        best_acc = val_acc

                        # Store model configuration along with weights
                        config = {
                            'feature_dim': feature_dim,
                            'num_task_vectors': args.num_task_vectors if hasattr(args, 'num_task_vectors') else 8,
                            'blockwise': args.blockwise_coef if hasattr(args, 'blockwise_coef') else False,
                            'base_threshold': float(ddp_model.module.base_threshold.item()),
                            'beta': float(ddp_model.module.beta.item()),
                            'uncertainty_reg': args.uncertainty_reg,
                            'model_name': args.model,
                            'causal_intervention': args.causal_intervention if hasattr(args, 'causal_intervention') else False
                        }

                        best_model_state = {
                            'meta_net': ddp_model.module.state_dict(),
                            'classifier': classifier.state_dict(),
                            'epoch': epoch,
                            'acc': val_acc,
                            'config': config
                        }

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
                    'best_acc': best_acc,
                    'config': config if 'config' in locals() else {}
                }

                with open(os.path.join(save_dir, "adaptive_gating_training_history.json"), 'w') as f:
                    # Convert numpy values to Python types
                    for key in history:
                        if isinstance(history[key], (list, dict)) and key != 'gating_stats':
                            history[key] = [float(item) if isinstance(item, (np.floating, np.integer)) else item
                                           for item in history[key]]

                    json.dump(history, f, indent=4)

                # Create plots directory
                plot_dir = os.path.join(args.save, "adaptive_gating_plots")
                os.makedirs(plot_dir, exist_ok=True)

                # Plot training curves
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

                # Plot task loss
                ax1.plot(train_losses, 'r-')
                ax1.set_ylabel('Task Loss')
                ax1.set_title(f'Training Loss for {dataset_name}')
                ax1.grid(True)

                # Plot regularization loss
                ax2.plot(reg_losses, 'g-')
                ax2.set_ylabel('Regularization Loss')
                ax2.set_title(f'Uncertainty Regularization Loss for {dataset_name}')
                ax2.grid(True)

                # Plot accuracy
                epochs = range(1, len(val_accuracies) + 1)
                ax3.plot(epochs, [acc * 100 for acc in val_accuracies], 'b-o')
                ax3.set_xlabel('Epochs')
                ax3.set_ylabel('Accuracy (%)')
                ax3.set_title(f'Validation Accuracy for {dataset_name}')
                ax3.grid(True)

                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f'{dataset_name}_adaptive_gating_training_curves.png'))
                plt.close()

                print(f"Training completed for {dataset_name}. Best validation accuracy: {best_acc*100:.2f}%")

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