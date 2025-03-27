import os
import torch
import torchvision.datasets as datasets
import glob
import json
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

# Tell PIL to be more lenient with corrupted files
ImageFile.LOAD_TRUNCATED_IMAGES = True


class SUN397:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16):
        # Data loading code
        # 减小默认batch_size，避免内存问题
        batch_size = min(batch_size, 64)

        try:
            traindir = os.path.join(location, 'SUN397_splits', 'train')
            testdir = os.path.join(location, 'SUN397_splits', 'test')

            # 确保文件夹存在
            if not os.path.exists(traindir):
                raise FileNotFoundError(f"Training directory not found: {traindir}")
            if not os.path.exists(testdir):
                raise FileNotFoundError(f"Test directory not found: {testdir}")

            # Use a more robust ImageFolder implementation
            self.train_dataset = RobustImageFolder(traindir, transform=preprocess)

            # 使用更小的num_workers，避免内存问题
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=min(num_workers, 4),
                pin_memory=True,
                timeout=120,
                prefetch_factor=2
            )

            self.test_dataset = RobustImageFolder(testdir, transform=preprocess)
            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                num_workers=min(num_workers, 4),
                pin_memory=True,
                timeout=120,
                prefetch_factor=2
            )

            idx_to_class = dict((v, k)
                                for k, v in self.train_dataset.class_to_idx.items())
            self.classnames = [idx_to_class[i].replace(
                '_', ' ') for i in range(len(idx_to_class))]

            # 创建静态的class_splits属性
            self.class_splits = self.load_class_splits(location)

        except Exception as e:
            print(f"Error initializing SUN397 dataset: {e}")
            raise

    def split_class_data(self, train):
        """Find indices of the data corresponding to each class

        Parameters:
        -----------
        train: bool
            If True, find indices from the training set. Otherwise, find indices
            from the test set.

        Returns:
        --------
        indices: dict
            A dictionary with class index as the key, and the list of data indices
            as the value.
        """
        indices = {}
        dataset = self.train_dataset if train else self.test_dataset
        for i, (_, t) in enumerate(dataset):
            t_key = str(t.item()) if isinstance(t, torch.Tensor) else str(t)
            if t_key not in indices:
                indices[t_key] = [i, ]
            else:
                indices[t_key].append(i)
        return indices

    def load_class_splits(self, location):
        """Load the list of data indices for each class"""

        root_dir = os.path.join(location, 'SUN397_splits')
        if not os.path.exists(root_dir):
            root_dir = os.path.join(location, 'SUN397')

        cache_path = os.path.join(root_dir, 'class_splits.json')
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    class_splits = json.load(f)
                return class_splits
            except Exception as e:
                print(f"Error loading class splits from {cache_path}: {e}")
                # Continue to regenerate

        print(
            f"Class splits for {self.__class__.__name__} not found or invalid."
            "\nGenerating and caching class splits..."
        )
        class_splits = {
            'train': self.split_class_data(True),
            'test': self.split_class_data(False),
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        try:
            with open(cache_path, 'w') as f:
                json.dump(class_splits, f)
        except Exception as e:
            print(f"Error saving class splits to {cache_path}: {e}")

        return class_splits


class SUN397Val(SUN397):
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16):
        super().__init__(preprocess, location, batch_size, num_workers)

        # Try to load validation set if it exists
        try:
            valdir = os.path.join(location, 'SUN397_splits', 'val')
            if os.path.exists(valdir):
                self.test_dataset = RobustImageFolder(valdir, transform=preprocess)
                self.test_loader = torch.utils.data.DataLoader(
                    self.test_dataset,
                    batch_size=batch_size,
                    num_workers=min(num_workers, 4),
                    pin_memory=True,
                    timeout=120,
                    prefetch_factor=2
                )
        except Exception as e:
            print(f"Error loading validation set: {e}")
            print("Using test set as validation")
            # Fall back to using the test set as validation


class RobustImageFolder(datasets.ImageFolder):
    """More robust ImageFolder implementation that handles corrupted images"""

    def __getitem__(self, index):
        """Override to add error handling for corrupted images"""
        path, target = self.samples[index]

        # Try to load the image, with error handling
        for _ in range(3):  # Try up to 3 times
            try:
                sample = self.loader(path)
                if self.transform is not None:
                    sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                return sample, target
            except Exception as e:
                # If loading fails, try a different approach
                try:
                    # Try opening with PIL directly
                    sample = Image.open(path).convert('RGB')
                    if self.transform is not None:
                        sample = self.transform(sample)
                    if self.target_transform is not None:
                        target = self.target_transform(target)
                    return sample, target
                except:
                    # If all else fails, create a blank image
                    sample = Image.new('RGB', (224, 224), color='black')
                    if self.transform is not None:
                        sample = self.transform(sample)
                    if self.target_transform is not None:
                        target = self.target_transform(target)
                    return sample, target