"""Simple CNN: the network from the PyTorch CIFAR10 tutorial
(https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
adapted to 1 input channel (spectrogram) and 2 output classes.

The size of the first fully connected layer is computed from the input
shape, so the model works for any clip length / number of mel bands.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, input_shape=(1, 64, 188), n_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        with torch.no_grad():
            n_flat = self._features(torch.zeros(1, *input_shape)).shape[1]
        self.fc1 = nn.Linear(n_flat, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def _features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return torch.flatten(x, 1)

    def forward(self, x):
        x = self._features(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # logits
