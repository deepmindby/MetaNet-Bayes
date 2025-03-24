"""
SUN397 Feature Extraction Script

A specialized script for extracting features from the SUN397 dataset,
using a more memory-efficient approach with incremental processing.
"""

import os
import sys
import torch
import argparse
import traceback
from datetime import datetime
from tqdm import tqdm
from src.modeling import ImageEncoder
from src.datasets.sun397_fix import SUN397Simple, SUN397ValSimple


def log_info(message):
    """Print logs with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def memory_cleanup():
    """Clean up memory"""
    torch.cuda.empty_cache()
    import gc
    gc.collect()


def extract_features_safely(model, loader, features_path, labels_path):
    """Extract features from a data loader with safety measures"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_features = []
    all_labels = []

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, desc="Extracting features")):
                # Handle batch
                if isinstance(batch, dict):
                    images = batch["images"].to(device)
                    labels = batch["labels"]
                else:
                    images, labels = batch
                    images = images.to(device)

                # Process in smaller chunks
                features = []
                chunk_size = 16  # Process in smaller chunks to reduce memory pressure

                for i in range(0, images.size(0), chunk_size):
                    chunk = images[i:i + chunk_size]
                    features.append(model(chunk).cpu())
                    torch.cuda.empty_cache()  # Release memory immediately

                # Combine features
                features = torch.cat(features, dim=0)

                # Add to collection
                all_features.append(features)
                all_labels.append(labels)

                # Print progress every 10 batches
                if batch_idx % 10 == 0:
                    log_info(f"Processed {batch_idx}/{len(loader)} batches")

                # Save intermediate results every 50 batches
                if batch_idx > 0 and batch_idx % 50 == 0:
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

        log_info(f"Feature extraction complete. Shape: {all_features.shape}")
        return True

    except Exception as e:
        log_info(f"ERROR in feature extraction: {e}")
        traceback.print_exc()

        # Try to save partial results
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
        memory_cleanup()


def extract_sun397_features(model, save_dir, data_location, batch_size):
    """Extract features for SUN397 dataset"""
    log_info("=== Processing SUN397 dataset ===")

    # Use validation preprocessing (no random augmentations)
    preprocess = model.val_preprocess

    try:
        # Load datasets
        log_info("Loading SUN397Val dataset...")
        train_val_dataset = SUN397ValSimple(
            preprocess,
            location=data_location,
            batch_size=batch_size,
            num_workers=2  # Reduced for stability
        )

        log_info("Loading SUN397 dataset...")
        test_dataset = SUN397Simple(
            preprocess,
            location=data_location,
            batch_size=batch_size,
            num_workers=2
        )

        # Create save directories
        save_dir_train_val = os.path.join(save_dir, "SUN397Val")
        save_dir_test = os.path.join(save_dir, "SUN397")
        os.makedirs(save_dir_train_val, exist_ok=True)
        os.makedirs(save_dir_test, exist_ok=True)

        # Save classnames
        if hasattr(train_val_dataset, 'classnames'):
            with open(os.path.join(save_dir_train_val, "classnames.txt"), "w") as f:
                f.write("\n".join(train_val_dataset.classnames))
            with open(os.path.join(save_dir_test, "classnames.txt"), "w") as f:
                f.write("\n".join(test_dataset.classnames))

        # Extract training features
        log_info("Extracting training features...")
        extract_features_safely(
            model,
            train_val_dataset.train_loader,
            os.path.join(save_dir_train_val, "train_features.pt"),
            os.path.join(save_dir_train_val, "train_labels.pt")
        )
        memory_cleanup()

        # Extract validation features
        log_info("Extracting validation features...")
        extract_features_safely(
            model,
            train_val_dataset.test_loader,
            os.path.join(save_dir_train_val, "val_features.pt"),
            os.path.join(save_dir_train_val, "val_labels.pt")
        )
        memory_cleanup()

        # Extract test features
        log_info("Extracting test features...")
        extract_features_safely(
            model,
            test_dataset.test_loader,
            os.path.join(save_dir_test, "test_features.pt"),
            os.path.join(save_dir_test, "test_labels.pt")
        )

        log_info("Completed processing SUN397")
        return True

    except Exception as e:
        log_info(f"ERROR processing SUN397 dataset: {e}")
        traceback.print_exc()
        return False

    finally:
        # Final cleanup
        memory_cleanup()
        log_info("GPU resources released after processing SUN397")


def parse_args():
    parser = argparse.ArgumentParser(description="SUN397 Feature Extraction")
    parser.add_argument("--model", type=str, default="ViT-B-32",
                        help="CLIP model to use (e.g. ViT-B-32)")
    parser.add_argument("--save-dir", type=str, default="precomputed_features",
                        help="Directory to save features")
    parser.add_argument("--data-location", type=str,
                        default=os.path.expanduser("~/data"),
                        help="Root directory for datasets")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for feature extraction")
    parser.add_argument("--openclip-cachedir", type=str,
                        default=os.path.expanduser("~/openclip-cachedir/open_clip"),
                        help="OpenCLIP cache directory")
    return parser.parse_args()


def main():
    args = parse_args()

    # Create directory
    model_save_dir = os.path.join(args.save_dir, args.model)
    log_info(f"Features will be saved to: {model_save_dir}")
    os.makedirs(model_save_dir, exist_ok=True)

    # Log execution information
    with open(os.path.join(model_save_dir, "model_info.txt"), "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] Processing dataset: SUN397\n")

    # Initialize model
    log_info(f"Initializing {args.model} model...")
    model_args = type('Args', (), {
        "model": args.model,
        "openclip_cachedir": args.openclip_cachedir,
        "cache_dir": None
    })

    image_encoder = ImageEncoder(model_args)

    # Process dataset
    success = extract_sun397_features(
        model=image_encoder,
        save_dir=model_save_dir,
        data_location=args.data_location,
        batch_size=args.batch_size
    )

    if success:
        log_info("Successfully processed SUN397")
        return 0
    else:
        log_info("Failed to process SUN397")
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