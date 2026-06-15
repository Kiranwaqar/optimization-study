import sys
import os
import time

sys.path.append(os.getcwd())

import torch
import numpy as np
import pandas as pd

from models.cnn import SimpleCNN
from data.loaders import get_dataloaders
from optimizers.factory import get_optimizer
from training.trainer import Trainer
from utils.seed import set_seed
from utils.convergence import time_to_threshold
from utils.experiment_tracker import ExperimentTracker
from plots.plot_results import plot_accuracy, plot_loss


def run_single_experiment(seed, dataset, opt_name, epochs):

    print(f"\nRunning Seed: {seed}")

    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader, in_channels = get_dataloaders(dataset)

    model = SimpleCNN(
        num_classes=10,
        in_channels=in_channels
    ).to(device)

    # Optimizer-specific learning rates
    lr_config = {
        "sgd": 0.01,
        "momentum": 0.01,
        "adam": 0.001,
        "adamw": 0.001,
        "rmsprop": 0.001
    }

    optimizer = get_optimizer(
        opt_name,
        model.parameters(),
        lr_config[opt_name]
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device
    )

    test_acc_history = []
    train_loss_history = []
    test_loss_history = []

    start_time = time.time()

    print("Starting training...")

    for epoch in range(epochs):

        train_loss, train_acc = trainer.train_one_epoch(
            train_loader
        )

        test_loss, test_acc = trainer.evaluate(
            test_loader
        )

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f}"
        )

    elapsed_time = time.time() - start_time

    return (
        test_acc_history,
        train_loss_history,
        test_loss_history,
        elapsed_time
    )


def run_experiment(
    dataset="cifar10",
    epochs=10
):

    optimizers = [
        "sgd",
        "momentum",
        "adam",
        "adamw",
        "rmsprop"
    ]

    seeds = [42, 52]

    results = {}

    tracker = ExperimentTracker()

    for opt in optimizers:

        print("\n" + "=" * 50)
        print(f"OPTIMIZER: {opt}")
        print("=" * 50)

        all_runs = []
        all_times = []

        all_train_losses = []
        all_test_losses = []

        for seed in seeds:

            (
                acc_history,
                train_losses,
                test_losses,
                elapsed
            ) = run_single_experiment(
                seed=seed,
                dataset=dataset,
                opt_name=opt,
                epochs=epochs
            )

            all_runs.append(acc_history)
            all_times.append(elapsed)

            all_train_losses.append(train_losses)
            all_test_losses.append(test_losses)

        all_runs = np.array(all_runs)

        mean_acc = np.mean(all_runs, axis=0)
        std_acc = np.std(all_runs, axis=0)

        mean_train_loss = np.mean(
            np.array(all_train_losses),
            axis=0
        )

        mean_test_loss = np.mean(
            np.array(all_test_losses),
            axis=0
        )

        mean_time = np.mean(all_times)

        convergence_epoch = time_to_threshold(
            mean_acc,
            threshold=0.70
        )

        results[opt] = {
            "mean_acc": mean_acc.tolist(),
            "std_acc": std_acc.tolist(),
            "convergence_epoch": convergence_epoch,
            "training_time": float(mean_time),
            "train_loss": mean_train_loss.tolist(),
            "test_loss": mean_test_loss.tolist()
        }

        tracker.add_result(
            opt,
            results[opt]
        )

        tracker.save()

        print(
            f"\n{opt.upper()} RESULTS"
        )

        print(
            f"Convergence Epoch: {convergence_epoch}"
        )

        print(
            f"Average Training Time: {mean_time:.2f} seconds"
        )

    # Create folders if needed
    os.makedirs("outputs", exist_ok=True)

    # Accuracy plot
    plot_accuracy(
        {
            k: {"test_acc": v["mean_acc"]}
            for k, v in results.items()
        }
    )

    # Loss plot
    plot_loss(results)

    # Save CSV summary
    df = pd.DataFrame(results).T
    df.to_csv("outputs/results.csv")

    return results


if __name__ == "__main__":
    run_experiment()