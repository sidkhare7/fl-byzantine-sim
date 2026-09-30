"""Small CNN for CIFAR-10 and helpers for moving weights to and from numpy."""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import normalise


class SmallCNN(nn.Module):
    """Two convolutional blocks followed by two fully connected layers.

    Roughly 270k parameters. Chosen so that a full federated round over ten
    clients finishes in tens of seconds on a laptop CPU, while still reaching
    an accuracy well above chance within a few rounds.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_weights(model):
    """Return model parameters as a list of numpy arrays."""
    return [p.detach().cpu().numpy() for p in model.state_dict().values()]


def set_weights(model, arrays):
    """Load a list of numpy arrays back into the model."""
    keys = list(model.state_dict().keys())
    state_dict = OrderedDict(
        (k, torch.tensor(v)) for k, v in zip(keys, arrays)
    )
    model.load_state_dict(state_dict, strict=True)


def train_local(model, loader, epochs, lr, device="cpu"):
    """Run local SGD and return the mean training loss."""
    model.train()
    optimiser = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    total_loss, total_batches = 0.0, 0
    for _ in range(epochs):
        for images, labels in loader:
            images = normalise(images).to(device)
            labels = labels.to(device)

            optimiser.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimiser.step()

            total_loss += loss.item()
            total_batches += 1

    return total_loss / max(total_batches, 1)


def evaluate(model, loader, device="cpu"):
    """Return (loss, accuracy) on the given loader."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = normalise(images).to(device)
            labels = labels.to(device)

            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total
