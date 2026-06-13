import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(dataset_name="cifar10", batch_size=64):

    if dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_set = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )

        test_set = datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transform
        )

        in_channels = 3

    elif dataset_name == "fashionmnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_set = datasets.FashionMNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )

        test_set = datasets.FashionMNIST(
            root="./data",
            train=False,
            download=True,
            transform=transform
        )

        in_channels = 1

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, in_channels