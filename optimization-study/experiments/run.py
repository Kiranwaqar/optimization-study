import sys
import os

sys.path.append(os.getcwd())

import torch
import copy
import numpy as np

from models.cnn import SimpleCNN
from data.loaders import get_dataloaders
from optimizers.factory import get_optimizer
from training.trainer import Trainer
from utils.seed import set_seed
from utils.convergence import time_to_threshold
from utils.experiment_tracker import ExperimentTracker
from plots.plot_results import plot_accuracy, plot_loss



def run_single_experiment(seed, dataset, opt_name, epochs, lr):

    print("Dataset =", dataset)

    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader, in_channels = get_dataloaders(dataset)

    print("Dataloaders created")

    model = SimpleCNN(num_classes=10, in_channels=in_channels)
    optimizer = get_optimizer(opt_name, model.parameters(), lr)

    trainer = Trainer(model, optimizer, device)

    test_acc_history = []

    print("Starting training")

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        trainer.train_one_epoch(train_loader)
        _, test_acc = trainer.evaluate(test_loader)
        test_acc_history.append(test_acc)

    return test_acc_history


def run_experiment(dataset="cifar10", epochs=1, lr=0.001):

    optimizers = ["sgd", "momentum", "adam", "adamw", "rmsprop", "adagrad"]
    seeds = [42, 52, 99]

    results = {}
    tracker = ExperimentTracker()

    for opt in optimizers:

        print(f"\n OPTIMIZER: {opt}")
        

        all_runs = []

        for seed in seeds:
            acc_history = run_single_experiment(seed, dataset, opt, epochs, lr)
            all_runs.append(acc_history)

        # convert to numpy for analysis
        all_runs = np.array(all_runs)

        mean_acc = np.mean(all_runs, axis=0)
        std_acc = np.std(all_runs, axis=0)

        convergence_epoch = time_to_threshold(mean_acc, threshold=0.70)

        results[opt] = {
            "mean_acc": mean_acc.tolist(),
            "std_acc": std_acc.tolist(),
            "convergence_epoch": convergence_epoch
        }

        tracker.add_result(opt, results[opt])

        print(f"{opt} convergence epoch: {convergence_epoch}")

    tracker.save()

    # plots
    plot_accuracy({k: {"test_acc": v["mean_acc"]} for k, v in results.items()})

    return results


if __name__ == "__main__":
    run_experiment()