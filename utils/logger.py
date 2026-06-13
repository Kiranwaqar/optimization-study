import csv
import os


class Logger:
    def __init__(self, filename="logs/results.csv"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.filename = filename

        with open(self.filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "optimizer", "loss", "accuracy"
            ])

    def log(self, epoch, optimizer, loss, acc):
        with open(self.filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, optimizer, loss, acc])