import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from collections import OrderedDict
import argparse


def visualize_task_vector_coefficients(
        model_dir,
        output_dir=None,
        datasets=None,
        model_name='ViT-B-32',
        figsize=(14, 10),
        dpi=300,
        max_vectors_to_show=8,
        max_blocks_per_vector=12
):
    """
    Visualize the learned coefficients for task vectors across different datasets.
    Creates a visualization similar to the atlas paper.

    Args:
        model_dir (str): Directory containing model checkpoints
        output_dir (str, optional): Directory to save visualizations
        datasets (list, optional): List of datasets to include
        model_name (str): Model name used (e.g., 'ViT-B-32')
        figsize (tuple): Figure size (width, height)
        dpi (int): DPI for saved figure
        max_vectors_to_show (int): Maximum number of task vectors to display
        max_blocks_per_vector (int): Maximum blocks per task vector to display
    """
    if output_dir is None:
        output_dir = os.path.join(model_dir, "visualizations")

    os.makedirs(output_dir, exist_ok=True)

    # If datasets not specified, try to find all available datasets
    if datasets is None:
        datasets = []
        for item in os.listdir(model_dir):
            if os.path.isdir(os.path.join(model_dir, item)) and "Val" in item:
                datasets.append(item.replace("Val", ""))
    else:
        # Clean dataset names by removing "Val" suffix if present
        datasets = [d.replace("Val", "") for d in datasets]

    # Ensure unique datasets and sort them
    unique_datasets = list(OrderedDict.fromkeys(datasets))
    datasets = sorted(unique_datasets)

    print(f"Found {len(datasets)} datasets: {datasets}")

    # Collect coefficients across datasets
    dataset_coefficients = {}

    for dataset in datasets:
        # Try to find model for this dataset (with 'Val' suffix)
        dataset_dir = os.path.join(model_dir, dataset + "Val")
        if not os.path.exists(dataset_dir):
            print(f"Warning: Directory not found for dataset {dataset}Val, skipping.")
            continue

        # Try different possible model file paths
        model_types = ["_adaptive_gating", "_blockwise", "", "_no_gating", "_atlas", "_atlas_with_gating"]
        model_path = None

        for suffix in model_types:
            candidate_path = os.path.join(dataset_dir, f"best{suffix}_model.pt")
            if os.path.exists(candidate_path):
                model_path = candidate_path
                break

        if model_path is None:
            # Try looking for best_precomputed_model.pt
            model_path = os.path.join(dataset_dir, "best_precomputed_model.pt")
            if not os.path.exists(model_path):
                print(f"Warning: No model found for dataset {dataset}, skipping.")
                continue

        print(f"Loading model from {model_path}")

        # Load model state
        try:
            state_dict = torch.load(model_path, map_location='cpu')

            # Extract meta_net state_dict
            meta_net_state = None

            # Check for 'meta_net' key
            if 'meta_net' in state_dict:
                if isinstance(state_dict['meta_net'], OrderedDict):
                    meta_net_state = state_dict['meta_net']
                elif hasattr(state_dict['meta_net'], 'state_dict'):
                    meta_net_state = state_dict['meta_net'].state_dict()
                else:
                    print(f"Warning: meta_net has unexpected type for {dataset}")

            # If not found directly, look for it in module structure
            if not meta_net_state:
                for key in state_dict:
                    if 'meta_net' in key and isinstance(state_dict[key], OrderedDict):
                        meta_net_state = state_dict[key]
                        break

            if not meta_net_state:
                print(f"Warning: Could not find meta_net state for {dataset}, looking for weight keys directly")
                meta_net_state = state_dict  # Try using the whole state dict

            # Look for the weights of the last layer of MetaNet
            coeff_layer_key = None
            for key in meta_net_state.keys():
                if key.endswith('net.2.weight') or '.meta_net.2.weight' in key:
                    coeff_layer_key = key
                    break

            if coeff_layer_key is None:
                print(f"Warning: Could not find coefficient layer for {dataset}")
                continue

            # Extract coefficients
            coeffs = meta_net_state[coeff_layer_key].detach().cpu().numpy()

            print(f"Dataset {dataset}: Coefficient shape = {coeffs.shape}")

            # Reshape for consistent format
            if len(coeffs.shape) == 3:  # [num_blocks, num_task_vectors, feature_dim] or similar
                # Compute the mean across the feature dimension
                dataset_coefficients[dataset] = coeffs.mean(axis=2).T  # Transpose to get [num_task_vectors, num_blocks]
            elif len(coeffs.shape) == 2:
                num_tv = min(coeffs.shape[0], max_vectors_to_show)
                # Reshape if needed for visualization
                if coeffs.shape[0] <= max_vectors_to_show:
                    # Already in right format, just compute mean across feature dimension
                    dataset_coefficients[dataset] = coeffs.mean(axis=1).reshape(-1, 1)  # [num_task_vectors, 1]
                else:
                    # Try to reshape into task vectors and blocks
                    blocks_per_vec = max_blocks_per_vector
                    vectors = min(max_vectors_to_show, coeffs.shape[0] // blocks_per_vec)
                    reshaped = coeffs[:vectors * blocks_per_vec].reshape(vectors, blocks_per_vec, -1)
                    dataset_coefficients[dataset] = reshaped.mean(axis=2)  # [vectors, blocks]
            else:
                print(f"Warning: Unexpected coefficient shape for {dataset}: {coeffs.shape}")
                continue

        except Exception as e:
            print(f"Error loading model for {dataset}: {e}")
            continue

    if not dataset_coefficients:
        print("No coefficient data found. Make sure the models contain MetaNet components.")
        return None

    # Prepare a uniform visualization format
    # Determine max dimensions across all datasets
    max_vectors = 0
    max_blocks = 0

    for ds, coeffs in dataset_coefficients.items():
        max_vectors = max(max_vectors, coeffs.shape[0])
        max_blocks = max(max_blocks, coeffs.shape[1])

    # Limit dimensions for better visualization
    max_vectors = min(max_vectors, max_vectors_to_show)
    max_blocks = min(max_blocks, max_blocks_per_vector)

    # Create a padded version of coefficient matrix for all datasets
    padded_coefficients = {}
    for ds, coeffs in dataset_coefficients.items():
        # Trim or pad the coefficients to match the max dimensions
        padded = np.zeros((max_vectors, max_blocks))

        # Copy available values
        v_dim = min(coeffs.shape[0], max_vectors)
        b_dim = min(coeffs.shape[1], max_blocks)
        padded[:v_dim, :b_dim] = coeffs[:v_dim, :b_dim]

        padded_coefficients[ds] = padded

    # Create coefficient matrix for heatmap [rows=vectors*blocks, cols=datasets]
    num_datasets = len(padded_coefficients)
    coeff_matrix = np.zeros((max_vectors * max_blocks, num_datasets))

    # Fill the coefficient matrix with values
    for col_idx, (ds, coeffs) in enumerate(padded_coefficients.items()):
        # Flatten the 2D coefficients into column
        coeff_matrix[:, col_idx] = coeffs.flatten()

    # Create row labels
    row_labels = []
    for v_idx in range(max_vectors):
        for b_idx in range(max_blocks):
            if max_blocks > 1:
                row_labels.append(f"TV_{v_idx + 1}_Block_{b_idx + 1}")
            else:
                row_labels.append(f"Task_Vector_{v_idx + 1}")

    # Create the visualization
    plt.figure(figsize=figsize)

    # Choose a better colormap and set limits for better contrast
    vmax = np.max(np.abs(coeff_matrix)) * 0.8
    vmin = -vmax

    ax = sns.heatmap(
        coeff_matrix,
        cmap='coolwarm',
        center=0,
        vmin=vmin,
        vmax=vmax,
        yticklabels=row_labels,
        xticklabels=datasets,
        cbar_kws={'label': 'Coefficient Value'}
    )

    # Improve label readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=8)

    # Add grid for better readability
    ax.grid(False)

    # Set title
    # plt.title(f"Learned coefficients on standard task vectors ({model_name})", fontsize=14, pad=20)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    fig_name = f"task_vector_coefficients_{model_name}.png"
    fig_path = os.path.join(output_dir, fig_name)
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    print(f"Visualization saved to {fig_path}")

    return fig_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize task vector coefficients")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing model checkpoints")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save visualizations")
    parser.add_argument("--datasets", type=str, nargs="+", default=None, help="List of datasets to include")
    parser.add_argument("--model-name", type=str, default="ViT-B-32", help="Model name")
    parser.add_argument("--figsize-width", type=int, default=20, help="Figure width")
    parser.add_argument("--figsize-height", type=int, default=80, help="Figure height")
    parser.add_argument("--dpi", type=int, default=800, help="DPI for saved figure")
    parser.add_argument("--max-vectors", type=int, default=8, help="Maximum number of task vectors to show")
    parser.add_argument("--max-blocks", type=int, default=12, help="Maximum blocks per vector to show")

    args = parser.parse_args()

    visualize_task_vector_coefficients(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        datasets=args.datasets,
        model_name=args.model_name,
        figsize=(args.figsize_width, args.figsize_height),
        dpi=args.dpi,
        max_vectors_to_show=args.max_vectors,
        max_blocks_per_vector=args.max_blocks
    )