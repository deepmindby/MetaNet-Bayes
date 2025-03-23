"""
Dataset utilities

Fred Zhang <frederic.zhang@adelaide.edu.au>
Paul Albert <paul.albert@adelaide.edu.au>
Australian Institute for Machine Learning

Modified from the codebase by Ilharco et al.,
at https://github.com/mlfoundations/task_vectors
"""

import os
import sys
import inspect
import torch
import copy
import itertools

from torch.utils.data.dataset import Subset, random_split

from src.datasets.cars import Cars
from src.datasets.dtd import DTD
from src.datasets.eurosat import EuroSAT, EuroSATVal
from src.datasets.gtsrb import GTSRB
from src.datasets.mnist import MNIST
from src.datasets.resisc45 import RESISC45
from src.datasets.svhn import SVHN
from src.datasets.sun397 import SUN397

# Add SUN397 fix for precomputed features
try:
    from src.datasets.sun397_fix import SUN397FixedFeatures
except ImportError:
    print("SUN397FixedFeatures not available")


registry = {
    name: obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)
}


class GenericDataset(object):
    def __init__(self):
        self.train_dataset = None
        self.train_loader = None
        self.test_dataset = None
        self.test_loader = None
        self.classnames = None


def split_train_into_train_val(dataset, new_dataset_class_name, batch_size, num_workers, val_fraction, max_val_samples=None, seed=0):
    assert val_fraction > 0. and val_fraction < 1.
    total_size = len(dataset.train_dataset)
    val_size = int(total_size * val_fraction)
    if max_val_samples is not None:
        val_size = min(val_size, max_val_samples)
    train_size = total_size - val_size

    assert val_size > 0
    assert train_size > 0

    lengths = [train_size, val_size]

    trainset, valset = random_split(
        dataset.train_dataset,
        lengths,
        generator=torch.Generator().manual_seed(seed)
    )

    new_dataset = None

    new_dataset_class = type(new_dataset_class_name, (GenericDataset, ), {})
    new_dataset = new_dataset_class()

    new_dataset.train_dataset = trainset
    new_dataset.train_loader = torch.utils.data.DataLoader(
        new_dataset.train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    new_dataset.test_dataset = valset
    new_dataset.test_loader = torch.utils.data.DataLoader(
        new_dataset.test_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    new_dataset.classnames = copy.copy(dataset.classnames)

    return new_dataset


def find_data_with_index(dataset, cls_idx):
    """Find the indices of the data corresponding to the designated class
    NOTE: This is very slow and is not recommended.

    Parameters:
    -----------
    dataset: Dataset
        A PyTorch dataset with the basic methods.
    cls_idx: int
        Index of the designated class.

    Returns:
    --------
    indices: List
        A list of indices.
    """
    indices = []
    for i, (_, target) in enumerate(dataset):
        if target == cls_idx:
            indices.append(i)
    return indices

def extract_class_data(dataset, cls_idx, batch_size, num_workers):
    """Isolate one or more designated classes from a dataset

    Parameters:
    -----------
    dataset: GenericDataset
        A dataset container, with the following attributes
            self.train_dataset, self.train_loader,
            self.test_dataset, self.test_loader, self.classnames.
    cls_idx: List[int]
        Indices of the classes to extract data for.

    Returns:
    --------
    subset: GenericDataset
        A subset containing data for the selected classes.
    """
    if cls_idx is None or len(cls_idx) == 0:
        return dataset
    if max(cls_idx) >= len(dataset.classnames):
        raise ValueError(f"Class index {cls_idx} exceeds the total class number.")
    else:
        classnames = [dataset.classnames[i] for i in cls_idx]
    subset = GenericDataset()
    subset.classnames = classnames

    train_split = dataset.class_splits['train']
    train_subset = list(itertools.chain.from_iterable(
        [train_split[str(i)] for i in cls_idx]
    ))
    test_split = dataset.class_splits['test']
    test_subset = list(itertools.chain.from_iterable(
        [test_split[str(i)] for i in cls_idx]
    ))
    subset.train_dataset = Subset(dataset.train_dataset, train_subset)
    subset.train_loader = torch.utils.data.DataLoader(
        subset.train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers
    )
    subset.test_dataset = Subset(dataset.test_dataset, test_subset)
    subset.test_loader = torch.utils.data.DataLoader(
        subset.test_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    return subset

def get_dataset(dataset_name, preprocess, location, batch_size=128, num_workers=16, val_fraction=0.1, max_val_samples=5000):
    # Handle SUN397 dataset specially for precomputed features
    if dataset_name.startswith('precomputed_') and 'SUN397' in dataset_name:
        try:
            from src.datasets.sun397_fix import SUN397FixedFeatures
            # Extract the actual dataset name
            base_dataset_name = dataset_name[len('precomputed_'):]

            # Try various paths for SUN397
            possible_sun397_paths = [
                os.path.join(location, "precomputed_features", "ViT-B-32", "SUN397"),
                os.path.join(location, "precomputed_features", "ViT-B-32", "SUN397Val"),
                os.path.join(location, "precomputed_features", "SUN397"),
                os.path.join(location, "SUN397"),
                os.path.join(location, "features", "SUN397"),
            ]

            # Try each path
            for path in possible_sun397_paths:
                print(f"Trying SUN397 path: {path}")
                if os.path.exists(path):
                    try:
                        return SUN397FixedFeatures(
                            feature_dir=path,
                            preprocess=preprocess,
                            location=location,
                            batch_size=batch_size,
                            num_workers=num_workers
                        )
                    except Exception as e:
                        print(f"Error with SUN397 path {path}: {e}")
                        continue

            # If no path works, try a recursive search
            print("Searching recursively for SUN397 features...")
            for root, dirs, files in os.walk(location):
                if "SUN397" in root and any(f.endswith(".pt") for f in files):
                    print(f"Found potential SUN397 directory: {root}")
                    try:
                        return SUN397FixedFeatures(
                            feature_dir=root,
                            preprocess=preprocess,
                            location=location,
                            batch_size=batch_size,
                            num_workers=num_workers
                        )
                    except Exception as e:
                        print(f"Error with SUN397 path {root}: {e}")
                        continue
        except ImportError:
            print("SUN397FixedFeatures not available, falling back to standard methods")
        except Exception as e:
            print(f"Error loading SUN397 with FixedFeatures: {e}")
            import traceback
            traceback.print_exc()

    # Handle datasets in the format <CLSIDX_DATASETNAME>
    # Check if requesting precomputed features for other datasets
    if dataset_name.startswith('precomputed_'):
        # Extract the actual dataset name
        base_dataset_name = dataset_name[len('precomputed_'):]

        # Try various path structures
        possible_paths = [
            os.path.join(location, 'precomputed_features', "ViT-B-32", base_dataset_name),
            os.path.join(location, 'precomputed_features', base_dataset_name),
            os.path.join(location, base_dataset_name),
            os.path.join(location, 'features', base_dataset_name)
        ]

        for feature_dir in possible_paths:
            if os.path.exists(feature_dir):
                from src.datasets.precomputed_features import PrecomputedFeatures
                print(f"Found precomputed features at: {feature_dir}")
                return PrecomputedFeatures(
                    feature_dir=feature_dir,
                    preprocess=preprocess,
                    location=location,
                    batch_size=batch_size,
                    num_workers=num_workers
                )

        raise ValueError(f"Precomputed features not found for {base_dataset_name}")

    cls_idx = None
    if '_' in dataset_name:
        cls_idx, dataset_name = dataset_name.split('_')
        cls_idx = [int(i) for i in cls_idx]

    if dataset_name.endswith('Val'):
        # Handle val splits
        if dataset_name in registry:
            dataset_class = registry[dataset_name]
        else:
            base_dataset_name = dataset_name.split('Val')[0]
            base_dataset = get_dataset(base_dataset_name, preprocess, location, batch_size, num_workers)
            base_dataset = extract_class_data(base_dataset, cls_idx, batch_size, num_workers)
            dataset = split_train_into_train_val(
                base_dataset, dataset_name, batch_size, num_workers, val_fraction, max_val_samples)
            return dataset
    else:
        assert dataset_name in registry, f'Unsupported dataset: {dataset_name}. Supported datasets: {list(registry.keys())}'
        dataset_class = registry[dataset_name]
    dataset = dataset_class(
        preprocess, location=location, batch_size=batch_size, num_workers=num_workers
    )
    dataset = extract_class_data(dataset, cls_idx, batch_size, num_workers)
    return dataset