import os
import traceback
import shutil as sh
from dataset_manager import get_data

# List ALL datasets you want to test
DATASETS_TO_TEST = [
    "Caltech101", "Caltech256", "FakeData", "FER2013", "FGVCAircraft",
    "Flickr30k", "Flowers102", "Food101", "GTSRB", "KMNIST", "LFWPeople",
    "Omniglot", "OxfordIIITPet", "PCAM","StanfordCars", "STL10", "SUN397"
]

# too large to download:- "ImageNet", "Imagenette", "Places365", "LSUN", "INaturalist", "Flickr8k", "SBU"
# correct :- "MNIST", "CIFAR10", "FashionMNIST", "SVHN", "QMNIST", "RenderedSST2", "SEMEION", "USPS"

failed_datasets = []
successful_datasets = []

print("=" * 60)
print("STARTING DATASET DOWNLOAD TEST")
print("=" * 60)

for dataset_name in DATASETS_TO_TEST:
    print(f"\nTesting: {dataset_name}")
    print("-" * 40)

    try:
        train_loader, test_loader, class_names = get_data(dataset_name)

        # Try to fetch one batch (critical check)
        train_batch = next(iter(train_loader))
        test_batch = next(iter(test_loader))

        print(f"✓ Train loader OK | Batch size: {len(train_batch[0])}")
        print(f"✓ Test loader OK  | Batch size: {len(test_batch[0])}")
        print(f"✓ Classes: {class_names if class_names else 'N/A'}")

        successful_datasets.append(dataset_name)

        sh.rmtree("./data")

    except Exception as e:
        print(f"✗ FAILED: {dataset_name}")
        print(f"Error: {e}")
        print("Traceback:")
        traceback.print_exc()

        failed_datasets.append(dataset_name)
        if os.path.exists("./data"):
            sh.rmtree("./data")

print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)

print(f"\nSuccessful ({len(successful_datasets)}):")
for ds in successful_datasets:
    print(f"  ✓ {ds}")

print(f"\nFailed ({len(failed_datasets)}):")
for ds in failed_datasets:
    print(f"  ✗ {ds}")

print("\nDONE.")