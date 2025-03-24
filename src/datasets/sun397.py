import os
import torch
import torchvision.datasets as datasets
import glob
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class SUN397Dataset(Dataset):
    """
    Custom SUN397 dataset that handles the specific directory structure and provides
    better error handling for problematic image files.
    """

    def __init__(self, root, transform=None, recursive=True):
        self.root = root
        self.transform = transform
        self.samples = []
        self.targets = []
        self.class_to_idx = {}
        self.recursive = recursive

        # Find all class directories (sorting them for consistent indexing)
        class_dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

        if not class_dirs:
            raise ValueError(f"No class directories found in {root}")

        print(f"Found {len(class_dirs)} class directories in {root}")

        # Create class to index mapping
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(class_dirs)}

        # For each class directory, find all image files
        for class_name in class_dirs:
            class_dir = os.path.join(root, class_name)
            class_idx = self.class_to_idx[class_name]

            # Get all image files in this class directory
            image_files = self._find_images_in_dir(class_dir)

            if not image_files:
                print(f"Warning: No image files found in class directory {class_name}")
                # Continue anyway, as we want to load as many classes as possible
                continue

            # Add each image file to samples list
            for img_path in image_files:
                self.samples.append((img_path, class_idx))
                self.targets.append(class_idx)

        if len(self.samples) == 0:
            raise ValueError(f"No valid images found in {root}")

        print(f"Loaded {len(self.samples)} images from {len(class_dirs)} classes")

    def _find_images_in_dir(self, directory):
        """
        Find all image files in a directory and its subdirectories

        Args:
            directory: Directory to search

        Returns:
            list: List of image file paths
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']
        image_files = []

        # First try direct files in this directory
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(directory, f"*{ext.upper()}")))

        # If recursive flag is set and no images found, try subdirectories
        if self.recursive and not image_files:
            subdirs = [os.path.join(directory, d) for d in os.listdir(directory)
                       if os.path.isdir(os.path.join(directory, d))]

            for subdir in subdirs:
                # Check each subdirectory for images
                for ext in image_extensions:
                    subdir_images = glob.glob(os.path.join(subdir, f"*{ext}"))
                    subdir_images.extend(glob.glob(os.path.join(subdir, f"*{ext.upper()}")))
                    image_files.extend(subdir_images)

                # Optionally go even deeper if needed
                for subsubdir in [os.path.join(subdir, d) for d in os.listdir(subdir)
                                  if os.path.isdir(os.path.join(subdir, d))]:
                    for ext in image_extensions:
                        subsubdir_images = glob.glob(os.path.join(subsubdir, f"*{ext}"))
                        subsubdir_images.extend(glob.glob(os.path.join(subsubdir, f"*{ext.upper()}")))
                        image_files.extend(subsubdir_images)

        return image_files

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset with error handling
        """
        img_path, target = self.samples[idx]

        # Load the image
        try:
            with open(img_path, 'rb') as f:
                img = Image.open(f).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)

            return img, target
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a default image and the target
            if idx > 0:
                return self.__getitem__(idx - 1)  # Try the previous image
            else:
                # Create a small blank image as a last resort
                img = Image.new('RGB', (224, 224), color='black')
                if self.transform is not None:
                    img = self.transform(img)
                return img, target


class SUN397:
    """
    SUN397 dataset class with enhanced error handling and debugging
    """

    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16):
        # Data loading paths
        traindir = os.path.join(location, 'SUN397_splits', 'train')
        testdir = os.path.join(location, 'SUN397_splits', 'test')

        # Verify directories exist
        if not os.path.exists(traindir):
            raise FileNotFoundError(f"Training directory not found: {traindir}")
        if not os.path.exists(testdir):
            raise FileNotFoundError(f"Test directory not found: {testdir}")

        print(f"Loading SUN397 dataset from {location}")

        # Try custom dataset implementation first
        try:
            self.train_dataset = SUN397Dataset(traindir, transform=preprocess, recursive=True)
            print(f"Successfully loaded SUN397 training data with custom loader")
        except Exception as e:
            print(f"Custom loader failed: {e}, trying ImageFolder")
            # Use ImageFolder with recursive flag
            self.train_dataset = datasets.ImageFolder(traindir, transform=preprocess)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # Load test dataset
        try:
            self.test_dataset = SUN397Dataset(testdir, transform=preprocess, recursive=True)
            print(f"Successfully loaded SUN397 test data with custom loader")
        except Exception as e:
            print(f"Custom loader failed: {e}, trying ImageFolder")
            self.test_dataset = datasets.ImageFolder(testdir, transform=preprocess)

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )

        # Generate classnames from the class_to_idx mapping
        idx_to_class = dict((v, k) for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace('_', ' ') for i in range(len(idx_to_class))]

        # Create class_splits attribute for validation
        self.class_splits = self.load_class_splits(location)

    def split_class_data(self, train):
        """
        Find indices of the data corresponding to each class

        Args:
            train: If True, find indices from the training set, otherwise test set

        Returns:
            dict: Dictionary with class index as the key and indices as the value
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
        """
        Load or create the class splits data

        Args:
            location: Root directory for datasets

        Returns:
            dict: Dictionary with train and test class splits
        """
        # Try to use SUN397_splits directory for cache
        root_dir = os.path.join(location, 'SUN397_splits')
        if not os.path.exists(root_dir):
            # Fall back to SUN397 directory
            root_dir = os.path.join(location, 'SUN397')

        cache_path = os.path.join(root_dir, 'class_splits.json')

        if os.path.exists(cache_path):
            print(f"Loading cached class splits from {cache_path}")
            with open(cache_path, 'r') as f:
                class_splits = json.load(f)
            return class_splits
        else:
            print(f"Class splits for SUN397 not found. Generating and caching...")
            class_splits = {
                'train': self.split_class_data(True),
                'test': self.split_class_data(False),
            }

            # Make sure the directory exists
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            with open(cache_path, 'w') as f:
                json.dump(class_splits, f)
            return class_splits


class SUN397Val(SUN397):
    """
    SUN397 dataset with validation set override
    """

    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16):
        super().__init__(preprocess, location, batch_size, num_workers)

        # Override test dataset with validation set if it exists
        valdir = os.path.join(location, 'SUN397_splits', 'val')
        if os.path.exists(valdir):
            print(f"Using validation set from {valdir}")
            try:
                self.test_dataset = SUN397Dataset(valdir, transform=preprocess, recursive=True)
                print(f"Successfully loaded SUN397 validation data with custom loader")
            except Exception as e:
                print(f"Custom loader failed: {e}, trying ImageFolder")
                self.test_dataset = datasets.ImageFolder(valdir, transform=preprocess)

            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                num_workers=num_workers
            )
        else:
            print(f"No validation directory found at {valdir}, using test set as validation")