"""
SUN397 dataset specialized loader with optimizations for feature extraction
and handling of corrupted image files.

This module provides a streamlined version of the SUN397 dataset loader
that avoids expensive class_splits computation and is optimized for
memory efficiency and stability.
"""

import os
import torch
import torchvision.datasets as datasets
from PIL import Image, ImageFile
import warnings
from torch.utils.data import Dataset

# Tell PIL to be more lenient with corrupted files
ImageFile.LOAD_TRUNCATED_IMAGES = True

class RobustImageFolder(datasets.ImageFolder):
    """A more robust ImageFolder class that can handle corrupted images"""

    def __getitem__(self, index):
        """Override to add error handling for corrupted images"""
        path, target = self.samples[index]

        # Try to load the image, with robust error handling
        for _ in range(3):  # Try up to 3 times
            try:
                sample = self.loader(path)
                if self.transform is not None:
                    sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                return sample, target
            except Exception as e:
                # If loading fails, create a blank image as fallback
                warnings.warn(f"Error loading image {path}: {e}. Using blank image.")
                # Create a 224x224 blank image (standard size for CLIP)
                sample = Image.new('RGB', (224, 224), color='black')
                if self.transform is not None:
                    sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                break  # Return the blank image

        return sample, target


class SUN397Simple:
    """Simplified SUN397 loader without expensive class_splits computation"""

    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=4):
        # Try multiple possible directories
        possible_train_dirs = [
            os.path.join(location, 'SUN397_splits', 'train'),
            os.path.join(location, 'SUN397', 'train'),
            os.path.join(location, 'train')
        ]

        possible_test_dirs = [
            os.path.join(location, 'SUN397_splits', 'test'),
            os.path.join(location, 'SUN397', 'test'),
            os.path.join(location, 'test')
        ]

        # Find valid paths
        traindir = None
        for path in possible_train_dirs:
            if os.path.exists(path):
                traindir = path
                break

        testdir = None
        for path in possible_test_dirs:
            if os.path.exists(path):
                testdir = path
                break

        if not traindir:
            raise FileNotFoundError(f"Cannot find SUN397 train directory. Tried: {possible_train_dirs}")
        if not testdir:
            raise FileNotFoundError(f"Cannot find SUN397 test directory. Tried: {possible_test_dirs}")

        print(f"Loading SUN397 train data from: {traindir}")
        print(f"Loading SUN397 test data from: {testdir}")

        # Use robust image folder implementation
        self.train_dataset = RobustImageFolder(
            traindir, transform=preprocess)

        # Use smaller batch_size and num_workers with more robust DataLoader settings
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=min(num_workers, 4),
            persistent_workers=False,
            pin_memory=True,
            timeout=120,  # Increased timeout
            prefetch_factor=2,  # Reduce prefetching
            drop_last=False
        )

        self.test_dataset = RobustImageFolder(testdir, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=min(num_workers, 4),
            persistent_workers=False,
            pin_memory=True,
            timeout=120,
            prefetch_factor=2,
            drop_last=False
        )

        # Get class names
        idx_to_class = dict((v, k)
                            for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace(
            '_', ' ') for i in range(len(idx_to_class))]

        # Provide an empty class_splits to avoid errors
        self.class_splits = {'train': {}, 'test': {}}


class SUN397ValSimple(SUN397Simple):
    """Simplified SUN397Val loader"""

    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=4):
        super().__init__(preprocess, location, batch_size, num_workers)

        # Try to find validation directory
        possible_val_dirs = [
            os.path.join(location, 'SUN397_splits', 'val'),
            os.path.join(location, 'SUN397', 'val'),
            os.path.join(location, 'val')
        ]

        valdir = None
        for path in possible_val_dirs:
            if os.path.exists(path):
                valdir = path
                break

        if valdir:
            print(f"Loading SUN397Val validation data from: {valdir}")
            self.test_dataset = RobustImageFolder(valdir, transform=preprocess)
            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                num_workers=min(num_workers, 4),
                persistent_workers=False,
                pin_memory=True,
                timeout=120,
                prefetch_factor=2,
                drop_last=False
            )
        else:
            print("Warning: No validation directory found for SUN397Val, using test data")