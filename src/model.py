import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # first layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # second layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # classification layer
        self.fc1 = nn.Linear(in_features=32*5*5, out_features=128)
        self.relu3 = nn.ReLU()
        self.output = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        
        # first layer
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # second layer
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # reshaping x
        x = x.view(x.size(0), -1)

        # classification layer
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.output(x)

        return x
