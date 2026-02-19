import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class DatasetManager:
    def __init__(self, dataset_name, root="./data", batch_size=32, num_workers=None):
        self.dataset_name = dataset_name
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        self.dataset_classes = {
            "CIFAR10": torchvision.datasets.CIFAR10,
            "CIFAR100": torchvision.datasets.CIFAR100,
            "MNIST": torchvision.datasets.MNIST,
            "FashionMNIST": torchvision.datasets.FashionMNIST,
            "Flowers102": torchvision.datasets.Flowers102,
            "Food101": torchvision.datasets.Food101
        }

    def create_transforms(self, augmentation_list, resize_val=(224, 224)):
        """
        Dynamically creates a transform pipeline.
        Args:
            augmentation_list: List of strings (e.g., ["RandomHorizontalFlip", "RandomCrop"])
            resize_val: Tuple for Resize
        """
        transform_list = []
        
        # Add dynamic augmentations
        for aug in augmentation_list:
            if hasattr(transforms, aug):
                # Standard defaults will be taken; for specific values like RandomRotation(10) for simpliciy
                transform_list.append(getattr(transforms, aug)())
            else:
                print(f"Warning: Transform {aug} not found in torchvision.transforms")

        # Fixed transforms
        transform_list.append(transforms.Resize(resize_val))
        transform_list.append(transforms.ToTensor())
        
        return transforms.Compose(transform_list)

    def load_datasets(self, transform, download=False):
        """Handles the logic for train, val, and test splits."""
        train_set, val_set, test_set = None, None, None

        # Logic for datasets using 'train' boolean (MNIST, CIFAR)
        if self.dataset_name in ["CIFAR10", "CIFAR100", "MNIST", "FashionMNIST"]:
            train_set = self.dataset_classes[self.dataset_name](
                root=self.root, train=True, download=download, transform=transform
            )
            test_set = self.dataset_classes[self.dataset_name](
                root=self.root, train=False, download=download, transform=transform
            )
            # Standard practice: Use a subset of train as val if the class doesn't provide one
            # For this example, we'll keep it simple:
            val_set = test_set 

        # Logic for datasets using 'split' string (Flowers, Food)
        elif self.dataset_name in ["Flowers102", "Food101"]:
            train_set = self.dataset_classes[self.dataset_name](
                root=self.root, split='train', download=download, transform=transform
            )
            test_set = self.dataset_classes[self.dataset_name](
                root=self.root, split='test', download=download, transform=transform
            )
            # Flowers102 has a dedicated 'val' split
            if self.dataset_name == "Flowers102":
                val_set = self.dataset_classes[self.dataset_name](
                    root=self.root, split='val', download=download, transform=transform
                )
            else:
                val_set = test_set

        return train_set, val_set, test_set

    def get_dataloaders(self, train_set, val_set, test_set, shuffle_train=True):
        """Creates and returns DataLoaders."""
        train_loader = DataLoader(train_set, batch_size=self.batch_size, 
                                  shuffle=shuffle_train, num_workers=self.num_workers)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, 
                                shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, 
                                 shuffle=False, num_workers=self.num_workers)
        
        # Attempt to get class names (might vary by dataset object attribute)
        class_names = getattr(train_set, 'classes', None)
        
        return train_loader, val_loader, test_loader, class_names

# --- Usage with Argparse Logic ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--resize", type=int, default=32)
    # User can pass: --augs RandomHorizontalFlip, RandomRotation, etc (NOTE: Some might erro)
    parser.add_argument("--augs", nargs='+', default=["RandomHorizontalFlip", "RandomCrop"]) 
    args = parser.parse_args()

    # 1. Initialize Manager
    manager = DatasetManager(dataset_name=args.dataset, batch_size=args.batch_size)

    # 2. Setup Transforms
    my_transforms = manager.create_transforms(augmentation_list=args.augs, 
                                             resize_val=(args.resize, args.resize))

    # 3. Load Data
    train_s, val_s, test_s = manager.load_datasets(transform=my_transforms, download=True)

    # 4. Get Loaders
    train_loader, val_loader, test_loader, classes = manager.get_dataloaders(train_s, val_s, test_s)

    print(f"Loaded {args.dataset} with {len(classes)} classes.")
    print(f"Train batches: {len(train_loader)}")