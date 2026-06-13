import torch
import torch.nn as nn
from utils.metrics import accuracy


class Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()

    def train_one_epoch(self, dataloader):

        self.model.train()
        total_loss = 0
        total_acc = 0

        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_acc += accuracy(outputs, labels)

        return total_loss / len(dataloader), total_acc / len(dataloader)

    def evaluate(self, dataloader):

        self.model.eval()
        total_loss = 0
        total_acc = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

                total_loss += loss.item()
                total_acc += accuracy(outputs, labels)

        return total_loss / len(dataloader), total_acc / len(dataloader)