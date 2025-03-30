"""
Batch Feature Extraction with Progressive Timeout, Memory Management and Data Augmentation

This script provides a robust approach to feature extraction by:
1. Implementing batch-level timeouts
2. Adding detailed progress monitoring
3. Aggressively managing memory
4. Supporting GPU device selection with automatic fallback
5. Supporting multiple augmented versions of features
6. Supporting linear model feature extraction
"""

import os
import sys
import torch
import argparse
import traceback
import time
import psutil
import gc
import signal
from datetime import datetime
from tqdm import tqdm
from PIL import Image, ImageFile

# Tell PIL to be more lenient with corrupted files
ImageFile.LOAD_TRUNCATED_IMAGES = True

from src.modeling import ImageEncoder
from src.datasets.registry import get_dataset

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

# Global variables
current_batch = 0
processing_start_time = None
last_activity_time = None
STALL_THRESHOLD = 120  # Consider process stalled if no activity for 2 minutes
enable_verbose = False

def log_info(message):
    """Print logs with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def log_debug(message):
    """Print debug logs only if verbose mode is enabled"""
    if enable_verbose:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] DEBUG: {message}", flush=True)

def memory_cleanup(force_gc=False):
    """Aggressive memory cleanup"""
    torch.cuda.empty_cache()
    if force_gc:
        gc.collect()

    # Log memory stats if verbose
    if enable_verbose:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        log_debug(f"Memory usage: {memory_info.rss / (1024 ** 2):.1f} MB")

        if torch.cuda.is_available():
            log_debug(f"GPU memory: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB allocated, "
                      f"{torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB reserved")

def extract_features_safely(model, loader, features_path, labels_path, batch_timeout=30, max_retries=2):
    """
    Extract features from a data loader with enhanced safety measures

    Args:
        model: CLIP image encoder model
        loader: Data loader
        features_path: Path to save features
        labels_path: Path to save labels
        batch_timeout: Timeout in seconds for processing each batch
        max_retries: Maximum number of retries for failed batches
    """
    global current_batch, processing_start_time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_features = []
    all_labels = []

    # Check for partial results to resume from
    partial_path = features_path + ".partial"
    if os.path.exists(partial_path):
        try:
            log_info(f"Found partial results at {partial_path}, attempting to load...")
            partial_features = torch.load(partial_path)
            log_info(f"Loaded {len(partial_features)} partial features, will resume extraction")
            all_features = [partial_features]

            # Try to load partial labels too
            partial_labels_path = labels_path + ".partial"
            if os.path.exists(partial_labels_path):
                partial_labels = torch.load(partial_labels_path)
                all_labels = [partial_labels]
                log_info(f"Loaded {len(partial_labels)} partial labels")
        except Exception as e:
            log_info(f"Error loading partial results: {e}")
            # Continue without partial results

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, desc="Extracting features")):
                current_batch = batch_idx

                # If we're resuming and already have partial results, skip appropriate number of batches
                if all_features and batch_idx * loader.batch_size < len(all_features[0]):
                    if batch_idx % 50 == 0:
                        log_debug(f"Skipping batch {batch_idx} (already processed)")
                    continue

                log_debug(f"Processing batch {batch_idx}/{len(loader)}")

                batch_start = time.time()
                retry_count = 0
                success = False

                while not success and retry_count <= max_retries:
                    try:
                        # Handle different batch formats
                        if isinstance(batch, dict):
                            images = batch["images"].to(device)
                            labels = batch["labels"]
                        else:
                            images, labels = batch
                            images = images.to(device)

                        # Process in smaller chunks if batch is large
                        chunk_size = 8 if retry_count > 0 else 16  # Reduce chunk size on retry
                        batch_features = []

                        for i in range(0, images.size(0), chunk_size):
                            chunk = images[i:i + chunk_size]
                            chunk_features = model(chunk)
                            batch_features.append(chunk_features.cpu())

                            # Force sync to detect OOM errors early
                            torch.cuda.synchronize()

                            # Release memory immediately
                            torch.cuda.empty_cache()

                        # Combine chunks
                        batch_features = torch.cat(batch_features, dim=0)

                        # Add to collection
                        all_features.append(batch_features)
                        all_labels.append(labels)

                        success = True

                    except RuntimeError as e:
                        # Check if it's an OOM error
                        if "out of memory" in str(e).lower():
                            retry_count += 1
                            log_info(f"OOM error on batch {batch_idx}, retry {retry_count}/{max_retries}")

                            # Aggressive cleanup
                            memory_cleanup(True)

                            if retry_count <= max_retries:
                                # Wait before retrying
                                time.sleep(2)
                                continue

                        # If not OOM or out of retries, re-raise
                        raise

                # Break if we couldn't process this batch after retries
                if not success:
                    log_info(f"Failed to process batch {batch_idx} after {max_retries} retries, stopping extraction")
                    break

                # Log the batch processing time
                batch_time = time.time() - batch_start
                log_debug(f"Batch {batch_idx} processed in {batch_time:.2f}s")

                # Aggressive cleanup every 5 batches
                if batch_idx % 5 == 0:
                    memory_cleanup(force_gc=True)
                    log_debug("Performed aggressive memory cleanup")

                # Save intermediate results every 25 batches
                if batch_idx > 0 and (batch_idx % 25 == 0 or batch_idx == len(loader) - 1):
                    log_info(f"Saving intermediate results at batch {batch_idx}")
                    try:
                        interim_features = torch.cat(all_features, dim=0)
                        interim_labels = torch.cat(all_labels, dim=0) if all_labels[0] is not None else None

                        torch.save(interim_features, features_path + ".partial")
                        if interim_labels is not None:
                            torch.save(interim_labels, labels_path + ".partial")

                        log_info(f"Saved {len(interim_features)} intermediate samples")
                    except Exception as e:
                        log_info(f"Error saving intermediate results: {e}")

        # Concatenate all features and labels
        log_info("Processing complete, concatenating results...")
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0) if all_labels[0] is not None else None

        # Save final results
        log_info(f"Saving {len(all_features)} features to {features_path}")
        torch.save(all_features, features_path)

        if all_labels is not None:
            log_info(f"Saving {len(all_labels)} labels to {labels_path}")
            torch.save(all_labels, labels_path)

        # Clean up partial files if successful
        if os.path.exists(features_path + ".partial"):
            os.remove(features_path + ".partial")
        if os.path.exists(labels_path + ".partial"):
            os.remove(labels_path + ".partial")

        log_info(f"Feature extraction complete. Shape: {all_features.shape}")
        return True

    except Exception as e:
        log_info(f"ERROR in feature extraction: {e}")
        traceback.print_exc()

        # Try to save whatever we have collected so far
        try:
            if all_features and len(all_features) > 0:
                log_info("Attempting to save partial results...")
                interim_features = torch.cat(all_features, dim=0)
                interim_labels = torch.cat(all_labels, dim=0) if all_labels and all_labels[0] is not None else None

                rescue_features_path = features_path + ".rescue"
                rescue_labels_path = labels_path + ".rescue"

                torch.save(interim_features, rescue_features_path)
                if interim_labels is not None:
                    torch.save(interim_labels, rescue_labels_path)

                log_info(f"Saved {len(interim_features)} rescue samples to {rescue_features_path}")
        except Exception as save_error:
            log_info(f"Failed to save partial results: {save_error}")

        return False
    finally:
        # Final cleanup
        memory_cleanup(force_gc=True)
        log_info("Feature extraction function complete")

def extract_and_save_features(model, dataset_name, save_dir, data_location, batch_size, batch_timeout=60, gpu_id=0, num_augmentations=0, start_augmentation=1):
    """Process a single dataset with robust error handling and optional data augmentation

    Args:
        model: CLIP image encoder model
        dataset_name: Name of the dataset
        save_dir: Directory to save features
        data_location: Root directory for datasets
        batch_size: Batch size for feature extraction
        batch_timeout: Timeout in seconds for each batch
        gpu_id: GPU ID to use
        num_augmentations: Number of augmented versions to create (default: 0)
        start_augmentation: Starting index for augmentation (default: 1)
    """
    global processing_start_time

    log_info(f"=== Processing {dataset_name} ===")
    start_time = time.time()

    # Set GPU device if available, with safety checks
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if gpu_id >= num_gpus:
            log_info(f"Warning: Requested GPU {gpu_id} but only {num_gpus} GPUs available. Using GPU 0 instead.")
            gpu_id = 0
        torch.cuda.set_device(gpu_id)
        log_info(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        log_info("Warning: CUDA not available, using CPU instead")

    try:
        # For large datasets like SUN397, reduce batch size to prevent memory issues
        if dataset_name == "SUN397" or dataset_name == "SUN397Val":
            log_info(f"Processing {dataset_name} with reduced batch size")
            batch_size = max(16, batch_size // 2)
            log_info(f"Using batch size {batch_size} for {dataset_name}")

        # Use validation preprocessing (no random augmentations) for the standard features
        val_preprocess = model.val_preprocess

        # Get train preprocessing for augmentation (includes random transforms)
        train_preprocess = model.train_preprocess

        # Get datasets for different preprocessing methods
        log_info(f"Loading dataset {dataset_name}Val with validation preprocessing...")
        try:
            train_val_dataset = get_dataset(
                dataset_name + "Val",
                val_preprocess,
                location=data_location,
                batch_size=batch_size,
                num_workers=2,  # Reduced worker count for stability
            )
        except Exception as e:
            log_info(f"Error loading {dataset_name}Val dataset: {e}")
            log_info(f"Will try alternative loading approaches")
            train_val_dataset = None

        # If we couldn't load the Val variant, try the base dataset
        if train_val_dataset is None:
            try:
                log_info(f"Loading dataset {dataset_name} as fallback...")
                train_val_dataset = get_dataset(
                    dataset_name,
                    val_preprocess,
                    location=data_location,
                    batch_size=batch_size,
                    num_workers=2,
                )
            except Exception as e:
                log_info(f"Error loading {dataset_name} dataset as fallback: {e}")
                raise RuntimeError(f"Could not load dataset {dataset_name} in any variant")

        log_info(f"Loading dataset {dataset_name} with validation preprocessing...")
        try:
            test_dataset = get_dataset(
                dataset_name,
                val_preprocess,
                location=data_location,
                batch_size=batch_size,
                num_workers=2,  # Reduced worker count for stability
            )
        except Exception as e:
            log_info(f"Error loading test dataset {dataset_name}: {e}")
            log_info(f"Using train dataset as test dataset")
            test_dataset = train_val_dataset

        # Create save directories
        save_dir_train_val = os.path.join(save_dir, dataset_name + "Val")
        save_dir_test = os.path.join(save_dir, dataset_name)
        os.makedirs(save_dir_train_val, exist_ok=True)
        os.makedirs(save_dir_test, exist_ok=True)

        # Save classnames
        if hasattr(train_val_dataset, 'classnames'):
            with open(os.path.join(save_dir_train_val, "classnames.txt"), "w") as f:
                f.write("\n".join(train_val_dataset.classnames))
            with open(os.path.join(save_dir_test, "classnames.txt"), "w") as f:
                f.write("\n".join(test_dataset.classnames))

        # Check if we need to extract standard features
        standard_features_path = os.path.join(save_dir_train_val, "train_features.pt")
        standard_features_exist = os.path.exists(standard_features_path)

        # Calculate timeout based on dataset size - do this outside the conditional block
        # so it's available for both standard features and augmentations
        train_size = len(train_val_dataset.train_loader.dataset)
        log_info(f"Training set size: {train_size}")
        adjusted_timeout = min(batch_timeout * (train_size / 5000), 300)  # Max 5 minutes per batch
        log_info(f"Using batch timeout of {adjusted_timeout:.1f}s for extraction")

        if not standard_features_exist or num_augmentations == 0:
            if not standard_features_exist:
                # First extract standard features with validation preprocessing
                log_info("Extracting standard training features...")
                extract_features_safely(
                    model,
                    train_val_dataset.train_loader,
                    os.path.join(save_dir_train_val, "train_features.pt"),
                    os.path.join(save_dir_train_val, "train_labels.pt"),
                    batch_timeout=adjusted_timeout,
                    max_retries=3  # More retries for robustness
                )
                memory_cleanup(force_gc=True)
                time.sleep(2)  # Wait a moment before next extraction

                log_info("Extracting validation features...")
                extract_features_safely(
                    model,
                    train_val_dataset.test_loader,
                    os.path.join(save_dir_train_val, "val_features.pt"),
                    os.path.join(save_dir_train_val, "val_labels.pt"),
                    batch_timeout=adjusted_timeout,
                    max_retries=3
                )
                memory_cleanup(force_gc=True)
                time.sleep(2)  # Wait a moment before next extraction

                log_info("Extracting test features...")
                extract_features_safely(
                    model,
                    test_dataset.test_loader,
                    os.path.join(save_dir_test, "test_features.pt"),
                    os.path.join(save_dir_test, "test_labels.pt"),
                    batch_timeout=adjusted_timeout,
                    max_retries=3
                )
                memory_cleanup(force_gc=True)
                time.sleep(2)  # Wait a moment before augmentations

        # Now extract augmented features if requested
        if num_augmentations > 0:
            end_augmentation = start_augmentation + num_augmentations
            log_info(f"Generating augmented feature versions from {start_augmentation} to {end_augmentation-1}...")

            for aug_idx in range(start_augmentation, end_augmentation):
                aug_features_path = os.path.join(save_dir_train_val, f"train_features_aug{aug_idx}.pt")

                # Skip if this augmentation already exists
                if os.path.exists(aug_features_path):
                    log_info(f"Augmented version {aug_idx} already exists, skipping")
                    continue

                log_info(f"Creating augmented version {aug_idx}")

                # Load dataset with train_preprocess for random augmentations
                log_info(f"Loading dataset {dataset_name}Val with training preprocessing for augmentation {aug_idx}...")
                try:
                    aug_train_val_dataset = get_dataset(
                        dataset_name + "Val",
                        train_preprocess,
                        location=data_location,
                        batch_size=batch_size,
                        num_workers=2,
                    )

                    # Extract augmented training features
                    log_info(f"Extracting augmented training features (version {aug_idx})...")
                    aug_labels_path = os.path.join(save_dir_train_val, f"train_labels_aug{aug_idx}.pt")

                    extract_features_safely(
                        model,
                        aug_train_val_dataset.train_loader,
                        aug_features_path,
                        aug_labels_path,
                        batch_timeout=adjusted_timeout,  # 使用前面定义的调整后的超时时间
                        max_retries=3  # More retries for augmentation
                    )
                    memory_cleanup(force_gc=True)

                    # Wait between augmentations
                    time.sleep(5)

                    log_info(f"Completed augmented version {aug_idx}")
                except Exception as aug_error:
                    log_info(f"Error processing augmentation {aug_idx}: {aug_error}")
                    traceback.print_exc()
                    # Continue with next augmentation
                    continue

        log_info(f"Completed processing {dataset_name}")
        return True

    except Exception as e:
        log_info(f"ERROR processing dataset {dataset_name}: {e}")
        traceback.print_exc()
        return False

    finally:
        # Final cleanup
        memory_cleanup(force_gc=True)

        elapsed = time.time() - start_time
        log_info(f"Total processing time for {dataset_name}: {elapsed:.1f}s")
        log_info(f"GPU resources released after processing {dataset_name}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Batch feature extraction")
    parser.add_argument("--model", type=str, default="ViT-B-32",
                        help="CLIP model to use (e.g. ViT-B-32)")
    parser.add_argument("--save-dir", type=str, default="precomputed_features",
                        help="Directory to save features")
    parser.add_argument("--data-location", type=str,
                        default=os.path.expanduser("~/data"),
                        help="Root directory for datasets")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for feature extraction")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset to process")
    parser.add_argument("--batch-timeout", type=int, default=60,
                        help="Timeout in seconds for each batch")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="GPU ID to use")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--openclip-cachedir", type=str,
                        default=os.path.expanduser("~/openclip-cachedir/open_clip"),
                        help="OpenCLIP cache directory")
    parser.add_argument("--num-augmentations", type=int, default=0,
                        help="Number of augmented versions to create (0 for none)")
    parser.add_argument("--start-augmentation", type=int, default=1,
                        help="Starting index for augmentation (default: 1)")
    parser.add_argument("--linear", action="store_true",
                        help="Use linearized model for feature extraction")
    return parser.parse_args()

def main():
    global enable_verbose

    args = parse_args()
    enable_verbose = args.verbose

    # Validate GPU availability
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        log_info(f"Found {num_gpus} GPUs available")
        if args.gpu_id >= num_gpus:
            log_info(f"Warning: Requested GPU {gpu_id} but only {num_gpus} GPUs available. Using GPU 0 instead.")
            args.gpu_id = 0
    else:
        log_info("Warning: CUDA not available, using CPU instead")
        args.gpu_id = -1  # Use CPU

    # Create directory with linear subdirectory if using linear model
    if args.linear:
        model_save_dir = os.path.join(args.save_dir, args.model, "linear")
        log_info(f"Features will be saved to: {model_save_dir} (linearized model)")
    else:
        model_save_dir = os.path.join(args.save_dir, args.model)
        log_info(f"Features will be saved to: {model_save_dir}")

    os.makedirs(model_save_dir, exist_ok=True)

    # Log execution information
    with open(os.path.join(model_save_dir, "model_info.txt"), "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] Processing dataset: {args.dataset}\n")
        if args.linear:
            f.write(f"[{timestamp}] Using linearized model\n")
        if args.num_augmentations > 0:
            f.write(f"[{timestamp}] Creating {args.num_augmentations} augmented versions\n")

    # Initialize model
    log_info(f"Initializing {args.model} model...")
    model_args = type('Args', (), {
        "model": args.model,
        "openclip_cachedir": args.openclip_cachedir,
        "cache_dir": None
    })

    if args.linear:
        # Import linearized model class and create linearized model
        from src.linearized import LinearizedImageEncoder
        log_info(f"Using linearized version of {args.model}")
        image_encoder = LinearizedImageEncoder(model_args)
    else:
        # Standard model initialization
        from src.modeling import ImageEncoder
        image_encoder = ImageEncoder(model_args)

    # Process dataset with robust error handling
    success = extract_and_save_features(
        model=image_encoder,
        dataset_name=args.dataset,
        save_dir=model_save_dir,
        data_location=args.data_location,
        batch_size=args.batch_size,
        batch_timeout=args.batch_timeout,
        gpu_id=args.gpu_id,
        num_augmentations=args.num_augmentations,
        start_augmentation=args.start_augmentation
    )

    if success:
        log_info(f"Successfully processed {args.dataset}")
        return 0
    else:
        log_info(f"Failed to process {args.dataset}")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log_info("Process interrupted")
        sys.exit(1)
    except Exception as e:
        log_info(f"Unhandled exception: {e}")
        traceback.print_exc()
        sys.exit(1)