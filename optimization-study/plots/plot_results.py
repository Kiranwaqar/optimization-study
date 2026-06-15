import os
import matplotlib.pyplot as plt


def plot_accuracy(results):

    os.makedirs("outputs/plots", exist_ok=True)

    plt.figure(figsize=(8, 5))

    for opt, history in results.items():
        plt.plot(
            history["test_acc"],
            label=opt.upper()
        )

    plt.title("Test Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig("outputs/plots/accuracy.png")
    plt.close()


def plot_loss(results):

    os.makedirs("outputs/plots", exist_ok=True)

    plt.figure(figsize=(8, 5))

    for opt, history in results.items():
        plt.plot(
            history["test_loss"],
            label=opt.upper()
        )

    plt.title("Test Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig("outputs/plots/loss.png")
    plt.close()