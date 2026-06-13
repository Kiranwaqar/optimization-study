import json
import os


class ExperimentTracker:
    def __init__(self, path="outputs/results.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.data = {}

    def add_result(self, optimizer, history):
        self.data[optimizer] = history

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=4)