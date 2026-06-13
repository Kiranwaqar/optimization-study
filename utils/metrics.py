import torch


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return (preds == labels).float().mean().item()


def compute_generalization_gap(train_acc, test_acc):
    return train_acc - test_acc