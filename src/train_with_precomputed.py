"""Training Script Using Pre-computed Features with Augmentation Support

This script trains MetaNet models using pre-computed CLIP features,
which significantly accelerates training by avoiding the forward pass
through the CLIP image encoder. It supports using augmented feature versions.
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

from src.metanet_precomputed import PrecomputedMetaNet
from src.utils import cosine_lr
from src.datasets.common import maybe_dictionarize
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp


def find_free_port():
    """Find a free port on localhost"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def setup_ddp_robust(rank, world_size, port=None, max_retries=5):
    """Setup distributed training with robust port handling"""
    # Generate random port if not specified
    if port is None:
        port = random.randint(40000, 65000)

    for retry in range(max_retries):
        try:
            # Use different port for each retry
            current_port = port + retry

            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = str(current_port)

            torch.distributed.init_process_group(
                "nccl",
                rank=rank,
                world_size=world_size,
            )

            torch.cuda.set_device(rank)
            torch.distributed.barrier()

            if rank == 0:
                print(f"Successfully initialized distributed setup with {world_size} processes")
            return True

        except Exception as e:
            print(f"Process {rank}: Failed on port {current_port}: {e}")
            # Wait before retrying
            time.sleep(random.uniform(1, 3))

    print(f"Process {rank}: Failed to initialize distributed setup after {max_retries} attempts")
    return False


def lp_reg(x, p=None, gamma=0.5) -> torch.Tensor:
    """L1 or L2 regularization term"""
    return 0 if p is None else gamma * torch.norm(x, p=p, dim=0).mean()


def plot_training_curves(train_losses, val_accuracies, dataset_name, save_dir, config=None):
    """Plot and save training curves

    Args:
        train_losses: List of loss values
        val_accuracies: List of validation accuracy values
        dataset_name: Name of the dataset
        save_dir: Directory to save the plot
        config: Optional configuration dictionary with model settings
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # Plot losses
    ax1.plot(train_losses, 'r-')
    ax1.set_ylabel('Loss')

    # Add configuration to title if available
    if config:
        model_type = "Standard"
        if config.get('blockwise', False):
            model_type = "Blockwise"
        if config.get('causal', False):
            model_type = f"Causal ({config.get('top_k_ratio', 0.1)})"
        title = f'{model_type} Model Training Loss for {dataset_name}'
    else:
        title = f'Training Loss for {dataset_name}'

    ax1.set_title(title)
    ax1.grid(True)

    # Plot accuracies
    epochs = list(range(1, len(val_accuracies) + 1))
    ax2.plot(epochs, [acc * 100 for acc in val_accuracies], 'b-o')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'Validation Accuracy for {dataset_name}')
    ax2.grid(True)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)

    # Add configuration to filename
    filename_suffix = ""
    if config:
        if config.get('blockwise', False):
            filename_suffix += "_blockwise"
        if config.get('causal', False):
            filename_suffix += "_causal"
        if config.get('use_augmentation', False):
            filename_suffix += "_augmented"

    plt.savefig(os.path.join(save_dir, f'{dataset_name}{filename_suffix}_training_curves.png'))
    plt.close()


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


def get_precomputed_dataset(dataset_name, model_name, location, batch_size=128, num_workers=2,
                           use_augmentation=True, max_augmentations=None, verbose=False):
    """Get dataset with pre-computed features with more robust path handling

    Args:
        dataset_name: Name of the dataset
        model_name: Name of the model used for feature extraction
        location: Root data directory
        batch_size: Batch size for dataloaders
        num_workers: Number of worker threads
        use_augmentation: Whether to use augmentations during training
        max_augmentations: Maximum number of augmentations to use (None for all)
        verbose: Whether to print detailed logs

    Returns:
        dataset: Dataset with pre-computed features
    """
    # Import required modules
    from src.precomputed_feature_dataset import PrecomputedFeatures

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

    if verbose:
        print(f"Looking for features for {dataset_name} in {len(possible_paths)} locations")

    # Try each possible path
    for path in possible_paths:
        if os.path.exists(path):
            if verbose:
                print(f"Found directory at: {path}")

            try:
                # Create our dataset with support for augmentation
                return PrecomputedFeatures(
                    feature_dir=path,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    use_augmentation=use_augmentation,
                    max_augmentations=max_augmentations
                )
            except Exception as e:
                print(f"Error creating dataset from {path}: {e}")
                continue

    # If we get here, try a recursive search for any path containing the dataset name
    if verbose:
        print(f"Standard search failed, performing recursive search for {base_name}...")

    for root, dirs, files in os.walk(location):
        if base_name.lower() in root.lower() and any(f.endswith('.pt') for f in files):
            if verbose:
                print(f"Found potential directory: {root}")

            try:
                return PrecomputedFeatures(
                    feature_dir=root,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    use_augmentation=use_augmentation,
                    max_augmentations=max_augmentations
                )
            except Exception as e:
                if verbose:
                    print(f"Error creating dataset from {root}: {e}")
                # Continue searching

    # If we get here, all paths failed
    raise FileNotFoundError(f"Pre-computed features not found for {dataset_name}")


def evaluate_model(model, dataset, device):
    """Evaluate model on dataset"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataset.test_loader:
            batch = maybe_dictionarize(batch)
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(features)

            # Compute accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return correct / total


def main(rank, args):
    """Main training function with augmentation support"""
    args.rank = rank

    # Initialize distributed setup with robust port handling
    if not setup_ddp_robust(args.rank, args.world_size, args.port):
        print(f"Process {rank}: Failed to initialize distributed setup. Exiting.")
        return

    # Process all datasets specified or use defaults
    if hasattr(args, 'datasets') and args.datasets:
        datasets_to_process = args.datasets
    else:
        datasets_to_process = ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"]

    # Fix path logic to avoid duplicating MetaNet-Bayes
    if not hasattr(args, 'save') or args.save is None:
        # Check if we're already in MetaNet-Bayes directory
        current_dir = os.getcwd()
        if current_dir.endswith('MetaNet-Bayes'):
            args.save = os.path.join(current_dir, "checkpoints_precomputed")
        else:
            args.save = os.path.join(current_dir, "MetaNet-Bayes", "checkpoints_precomputed")
        if rank == 0:
            print(f"Using save directory: {args.save}")

    # Create save directory
    if is_main_process():
        os.makedirs(args.save, exist_ok=True)

    # Set default feature directory
    if not hasattr(args, 'precomputed_dir') or args.precomputed_dir is None:
        # Check if we're already in MetaNet-Bayes directory
        current_dir = os.getcwd()
        if current_dir.endswith('MetaNet-Bayes'):
            args.precomputed_dir = os.path.join(current_dir, "precomputed_features")
        else:
            args.precomputed_dir = os.path.join(current_dir, "MetaNet-Bayes", "precomputed_features")

    # Print configuration only from main process
    if rank == 0:
        print(f"\n=== Training Configuration ===")
        print(f"Model: {args.model}")
        print(f"Using blockwise coefficients: {args.blockwise_coef}")
        print(f"Using causal intervention: {args.causal_intervention}")
        if args.causal_intervention:
            print(f"Top-k ratio: {args.top_k_ratio}")
            print(f"Variance penalty coefficient: {args.var_penalty_coef if hasattr(args, 'var_penalty_coef') else 0.01}")
        print(f"Number of task vectors: {args.num_task_vectors}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.lr}")
        print(f"Using augmentation: {args.use_augmentation}")
        if args.use_augmentation:
            print(f"Max augmentations: {args.max_augmentations if hasattr(args, 'max_augmentations') else 'All'}")
        print(f"Save directory: {args.save}")
        print("=" * 30)

    for dataset_name in datasets_to_process:
        target_dataset = f"precomputed_{dataset_name}Val"  # Use precomputed features
        if is_main_process():
            print(f"=== Training on {dataset_name} with pre-computed features ===")

        # Setup save directory for this dataset
        save_dir = os.path.join(args.save, dataset_name + "Val")
        if is_main_process():
            os.makedirs(save_dir, exist_ok=True)

        dataset = None

        try:
            # Apply dataset-specific augmentation settings
            use_augmentation = args.use_augmentation
            max_augmentations = args.max_augmentations if hasattr(args, 'max_augmentations') else None

            # Special case for SUN397 due to potential corrupted images issues
            if dataset_name == "SUN397" and hasattr(args, 'sun397_no_augmentation') and args.sun397_no_augmentation:
                if is_main_process():
                    print(f"Disabling augmentation for SUN397 dataset due to sun397_no_augmentation flag")
                use_augmentation = False

            # Get precomputed features with augmentation support
            dataset = get_precomputed_dataset(
                dataset_name=target_dataset,
                model_name=args.model,
                location=args.data_location if hasattr(args, 'data_location') else ".",
                batch_size=args.batch_size,
                num_workers=2,  # Reduced for stability
                use_augmentation=use_augmentation,
                max_augmentations=max_augmentations,
                verbose=True  # Enable verbose mode to debug path issues
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
                raise

            # Create model
            try:
                model = PrecomputedMetaNet(
                    feature_dim=feature_dim,
                    task_vectors=args.num_task_vectors,  # Number of task vectors to simulate
                    blockwise=args.blockwise_coef if hasattr(args, 'blockwise_coef') else False,
                    enable_causal=args.causal_intervention if hasattr(args, 'causal_intervention') else False,
                    top_k_ratio=args.top_k_ratio if hasattr(args, 'top_k_ratio') else 0.1
                )
                model = model.to(device)
            except Exception as e:
                if is_main_process():
                    print(f"Error creating model: {e}")
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
                    find_unused_parameters=False
                )

                # Setup classifier layer
                num_classes = len(dataset.classnames)
                classifier = torch.nn.Linear(feature_dim, num_classes).cuda()
            except Exception as e:
                if is_main_process():
                    print(f"Error in distributed setup: {e}")
                raise

            # Setup optimizer
            try:
                params = list(ddp_model.parameters()) + list(classifier.parameters())
                optimizer = torch.optim.AdamW(
                    params,
                    lr=args.lr if hasattr(args, 'lr') else 1e-3,
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
                raise

            # Training monitoring
            train_losses = []
            epoch_losses = []
            var_losses = []
            val_accuracies = []
            best_acc = 0.0
            best_model_state = None

            # Training loop
            num_epochs = args.epochs if hasattr(args, 'epochs') else 10
            for epoch in range(num_epochs):
                ddp_model.train()
                classifier.train()

                epoch_loss = 0.0
                epoch_var_loss = 0.0
                batch_count = 0
                var_loss_count = 0

                for i, batch in enumerate(ddp_loader):
                    start_time = time.time()

                    try:
                        batch = maybe_dictionarize(batch)
                        features = batch["features"].to(rank)
                        labels = batch["labels"].to(rank)

                        # Track augmentation usage if available
                        augmented = batch.get("augmented", False)
                        if i % print_every == 0 and is_main_process() and "augmented" in batch:
                            aug_count = sum(1 for item in batch["augmented"] if item)
                            print(f"Batch {i}: Using {aug_count}/{len(batch['augmented'])} augmented samples")

                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            # Forward pass
                            transformed_features = ddp_model(features)
                            logits = classifier(transformed_features)

                            # Task loss
                            task_loss = loss_fn(logits, labels)

                            # Add variance loss if causal intervention is enabled
                            if hasattr(args, 'causal_intervention') and args.causal_intervention:
                                var_loss = ddp_model.module.compute_intervention_loss(features)
                                var_penalty = args.var_penalty_coef if hasattr(args, 'var_penalty_coef') else 0.01
                                total_loss = task_loss + var_penalty * var_loss

                                # Record variance loss
                                var_loss_cpu = var_loss.item()
                                if var_loss_cpu > 0:
                                    epoch_var_loss += var_loss_cpu
                                    var_loss_count += 1
                                    if is_main_process():
                                        var_losses.append(var_loss_cpu)
                            else:
                                var_loss = torch.tensor(0.0, device=features.device)
                                var_loss_cpu = 0.0
                                total_loss = task_loss

                        # Backward pass
                        scaler.scale(total_loss).backward()

                        # Step optimizer
                        scheduler(i + epoch * num_batches)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                        # Record stats
                        task_loss_cpu = task_loss.item()
                        batch_count += 1
                        epoch_loss += task_loss_cpu
                        if is_main_process():
                            train_losses.append(task_loss_cpu)

                        # Print progress (less frequently)
                        if i % print_every == 0 and is_main_process():
                            var_str = f", Var Loss: {var_loss_cpu:.6f}" if hasattr(args, 'causal_intervention') and args.causal_intervention else ""
                            print(f"Epoch {epoch+1}/{num_epochs}, Batch {i}/{num_batches}, "
                                f"Loss: {task_loss_cpu:.6f}{var_str}, "
                                f"Time: {time.time() - start_time:.3f}s")

                    except Exception as e:
                        if is_main_process():
                            print(f"Error in training batch: {e}")
                        # Skip this batch but continue training
                        continue

                # Record epoch stats
                avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
                epoch_losses.append(avg_epoch_loss)

                # Calculate average variance loss for the epoch
                avg_epoch_var_loss = 0.0
                if var_loss_count > 0:
                    avg_epoch_var_loss = epoch_var_loss / var_loss_count

                if is_main_process():
                    print(f"Epoch {epoch+1} average - Task Loss: {avg_epoch_loss:.6f}, "
                          f"Var Loss: {avg_epoch_var_loss:.6f}")

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

                    eval_model = CombinedModel(ddp_model.module, classifier)

                    val_acc = evaluate_model(
                        model=eval_model,
                        dataset=dataset,
                        device=rank
                    )
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
                            'num_task_vectors': args.num_task_vectors,
                            'blockwise': args.blockwise_coef if hasattr(args, 'blockwise_coef') else False,
                            'enable_causal': args.causal_intervention if hasattr(args, 'causal_intervention') else False,
                            'top_k_ratio': args.top_k_ratio if hasattr(args, 'top_k_ratio') else 0.1,
                            'model_name': args.model,
                            'finetuning_mode': args.finetuning_mode if hasattr(args, 'finetuning_mode') else 'standard',
                            'use_augmentation': args.use_augmentation,
                            'max_augmentations': args.max_augmentations if hasattr(args, 'max_augmentations') else None
                        }

                        best_model_state = {
                            'meta_net': ddp_model.module.state_dict(),
                            'classifier': classifier.state_dict(),
                            'epoch': epoch,
                            'acc': val_acc,
                            'config': config  # Add configuration information
                        }

            # Save results
            if is_main_process():
                # Save best model with clear model type in filename
                model_type = ""
                if args.blockwise_coef:
                    model_type += "_blockwise"
                if args.causal_intervention:
                    model_type += "_causal"
                if args.use_augmentation:
                    model_type += "_augmented"

                model_filename = f"best_precomputed_model{model_type}.pt"

                if best_model_state:
                    best_model_path = os.path.join(save_dir, model_filename)
                    print(f"Saving best model to {best_model_path}")
                    torch.save(best_model_state, best_model_path)

                    # Save a copy with standard name for compatibility
                    torch.save(best_model_state, os.path.join(save_dir, "best_precomputed_model.pt"))

                # Save training history
                history = {
                    'train_losses': train_losses,
                    'epoch_losses': epoch_losses,
                    'val_accuracies': val_accuracies,
                    'best_acc': best_acc,
                    'var_losses': var_losses if var_losses else [],
                    'config': config if 'config' in locals() else {},
                    'use_augmentation': args.use_augmentation,
                    'max_augmentations': args.max_augmentations if hasattr(args, 'max_augmentations') else None
                }
                with open(os.path.join(save_dir, f"precomputed_training_history{model_type}.json"), 'w') as f:
                    json.dump(history, f, indent=4)

                # Plot training curves
                plot_dir = os.path.join(args.save, "precomputed_plots")
                plot_config = {
                    'blockwise': args.blockwise_coef,
                    'causal': args.causal_intervention,
                    'top_k_ratio': args.top_k_ratio if hasattr(args, 'top_k_ratio') else 0.1,
                    'use_augmentation': args.use_augmentation
                }
                plot_training_curves(epoch_losses, val_accuracies, dataset_name, plot_dir, plot_config)

                print(f"Training completed for {dataset_name}. Best validation accuracy: {best_acc*100:.2f}%")

        except Exception as e:
            if is_main_process():
                print(f"Error processing dataset {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
        finally:
            # Clean up dataset resources
            cleanup_resources(dataset)
            torch.cuda.empty_cache()
            gc.collect()

    cleanup_ddp()


if __name__ == "__main__":
    # Parse arguments
    from src.args import parse_arguments
    args = parse_arguments()

    # Set default save directory if not provided
    if not hasattr(args, 'save') or args.save is None:
        # Check if we're already in MetaNet-Bayes directory
        current_dir = os.getcwd()
        if current_dir.endswith('MetaNet-Bayes'):
            args.save = os.path.join(current_dir, "checkpoints_precomputed")
        else:
            args.save = os.path.join(current_dir, "MetaNet-Bayes", "checkpoints_precomputed")
        print(f"Using save directory: {args.save}")

    # Fix data location to proper path
    current_dir = os.getcwd()
    if hasattr(args, 'data_location') and args.data_location:
        # Remove redundant MetaNet-Bayes if present
        if "MetaNet-Bayes/MetaNet-Bayes" in args.data_location:
            args.data_location = args.data_location.replace("MetaNet-Bayes/MetaNet-Bayes", "MetaNet-Bayes")
    else:
        # Set default data location
        if current_dir.endswith('MetaNet-Bayes'):
            args.data_location = current_dir
        else:
            args.data_location = os.path.join(current_dir, "MetaNet-Bayes")

    # Set feature directory path
    if current_dir.endswith('MetaNet-Bayes'):
        args.precomputed_dir = os.path.join(current_dir, "precomputed_features")
    else:
        args.precomputed_dir = os.path.join(current_dir, "MetaNet-Bayes", "precomputed_features")

    # Augmentation settings
    if not hasattr(args, 'use_augmentation'):
        args.use_augmentation = True  # Default to using augmentation
    if not hasattr(args, 'max_augmentations'):
        args.max_augmentations = None  # Use all available augmentations

    # Special case for SUN397 - you can set a flag to disable augmentation just for SUN397
    args.sun397_no_augmentation = False  # Change to True if needed

    args.num_task_vectors = 8  # Default number of task vectors to simulate

    # Set default training parameters if not specified
    if not hasattr(args, 'epochs') or not args.epochs:
        args.epochs = 20
    if not hasattr(args, 'batch_size') or not args.batch_size:
        args.batch_size = 128  # Can use larger batch size with precomputed features
    if not hasattr(args, 'lr') or not args.lr:
        args.lr = 5e-3
    if not hasattr(args, 'wd') or not args.wd:
        args.wd = 0.01
    if not hasattr(args, 'var_penalty_coef'):
        args.var_penalty_coef = 0.01

    # Set random port if not specified to avoid conflicts
    if not hasattr(args, 'port') or not args.port:
        args.port = random.randint(40000, 65000)

    # Set exception handling for worker processes
    try:
        import torch.multiprocessing as mp
        mp.set_sharing_strategy('file_system')  # Use file_system strategy for more reliable sharing
    except:
        pass

    # Launch training with better error handling
    try:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size)
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        # Force cleanup in case of failure
        for i in range(torch.cuda.device_count()):
            if i < args.world_size:
                try:
                    torch.cuda.set_device(i)
                    torch.cuda.empty_cache()
                except:
                    pass