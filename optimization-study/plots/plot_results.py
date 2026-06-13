import matplotlib.pyplot as plt


def plot_accuracy(results):

    plt.figure()

    for opt, history in results.items():
        plt.plot(history["test_acc"], label=opt)

    plt.title("Test Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig("outputs/plots/accuracy.png")
    plt.show()


def plot_loss(results):

    plt.figure()

    for opt, history in results.items():
        plt.plot(history["train_loss"], label=opt)

    plt.title("Training Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig("outputs/plots/loss.png")
    plt.show()