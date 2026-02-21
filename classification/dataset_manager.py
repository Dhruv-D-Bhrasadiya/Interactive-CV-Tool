import os
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T


def _random_split(dataset, ratio=0.8, seed=42):
    generator = torch.Generator().manual_seed(seed)
    train_size = int(ratio * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size], generator=generator)


def get_data(
    dataset_name: str = "CIFAR10",
    transform: T.Compose = T.Compose([T.ToTensor()]),
    batch_size: int = 32,
    root: str = "./data",
):

    num_workers = os.cpu_count()
    dataset_name = dataset_name.strip()

    # DATASETS WITH train=True / False
    train_flag_datasets = {
        "CIFAR10": torchvision.datasets.CIFAR10,
        "CIFAR100": torchvision.datasets.CIFAR100,
        "MNIST": torchvision.datasets.MNIST,
        "FashionMNIST": torchvision.datasets.FashionMNIST,
        "KMNIST": torchvision.datasets.KMNIST,
        "QMNIST": torchvision.datasets.QMNIST,
        "USPS": torchvision.datasets.USPS,
        "STL10": torchvision.datasets.STL10,
    }

    if dataset_name in train_flag_datasets:
        DatasetClass = train_flag_datasets[dataset_name]

        if dataset_name == "STL10":
            train_set = DatasetClass(root=root, split="train", download=True, transform=transform)
            test_set = DatasetClass(root=root, split="test", download=True, transform=transform)
        else:
            train_set = DatasetClass(root=root, train=True, download=True, transform=transform)
            test_set = DatasetClass(root=root, train=False, download=True, transform=transform)

        class_names = getattr(train_set, "classes", None)

    # DATASETS WITH split="train"/"test"
    elif dataset_name in [
        "SVHN", "GTSRB", "OxfordIIITPet", "Flowers102",
        "Food101", "StanfordCars", "FER2013"
    ]:
        DatasetClass = getattr(torchvision.datasets, dataset_name)

        train_set = DatasetClass(root=root, split="train", download=True, transform=transform)
        test_set = DatasetClass(root=root, split="test", download=True, transform=transform)

        class_names = getattr(train_set, "classes", None)

    # DATASETS WITH train/val/test
    elif dataset_name in ["FGVCAircraft", "PCAM"]:
        DatasetClass = getattr(torchvision.datasets, dataset_name)

        train_set = DatasetClass(root=root, split="train", download=True, transform=transform)
        test_set = DatasetClass(root=root, split="test", download=True, transform=transform)

        class_names = getattr(train_set, "classes", None)

    # DATASETS WITH ONLY ONE SPLIT → RANDOM SPLIT
    elif dataset_name in [
        "FakeData", "SEMEION", "Omniglot",
        "SUN397", "Places365", "LSUN",
        "INaturalist"
    ]:
        DatasetClass = getattr(torchvision.datasets, dataset_name)

        full_set = DatasetClass(root=root, download=True, transform=transform)
        train_set, test_set = _random_split(full_set)

        class_names = getattr(full_set, "classes", None)

    # IMAGE CAPTION DATASETS
    elif dataset_name in ["Flickr8k", "Flickr30k", "SBU"]:
        DatasetClass = getattr(torchvision.datasets, dataset_name)

        full_set = DatasetClass(root=root, download=True, transform=transform)
        train_set, test_set = _random_split(full_set)

        class_names = None  # captions, not classification

    # LFW
    elif dataset_name == "LFWPeople":
        train_set = torchvision.datasets.LFWPeople(
            root=root, split="train", download=True, transform=transform
        )
        test_set = torchvision.datasets.LFWPeople(
            root=root, split="test", download=True, transform=transform
        )
        class_names = train_set.classes

    # RENDEREDSST2
    elif dataset_name == "RenderedSST2":
        train_set = torchvision.datasets.RenderedSST2(
            root=root, split="train", download=True, transform=transform
        )
        test_set = torchvision.datasets.RenderedSST2(
            root=root, split="test", download=True, transform=transform
        )
        class_names = ["negative", "positive"]

    # IMAGENET / IMAGENETTE (REQUIRE MANUAL DOWNLOAD)
    elif dataset_name in ["ImageNet", "Imagenette"]:
        DatasetClass = getattr(torchvision.datasets, dataset_name)

        train_set = DatasetClass(root=root, split="train", transform=transform)
        test_set = DatasetClass(root=root, split="val", transform=transform)

        class_names = train_set.classes

    else:
        raise ValueError(f"{dataset_name} not supported.")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader, class_names