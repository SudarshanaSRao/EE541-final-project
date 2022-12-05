import torch
import torch.nn as nn
import torch.nn.functional as F


# Multilayer perceptron
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(3 * 100**2, 1000)
        self.hidden2 = nn.Linear(1000, 200)
        self.output = nn.Linear(200, 29)

    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.output(x)
        return x


# Convolutional neural network
class CNN_Base(nn.Module):
    def __init__(self):
        super(CNN_Base, self).__init__()

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Fully connected layer 1
        self.fc1 = nn.Linear(32 * 50**2, 128)

        # Output layer
        self.fc2 = nn.Linear(128, 29)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Baseline convolutional neural network from paper
class CNN_Paper(nn.Module):
    def __init__(self):
        super(CNN_Paper, self).__init__()

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.2)

        # Convolutional layer 2
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.2)

        # Fully connected layer 1
        self.fc1 = nn.Linear(32 * 25**2, 128)
        self.dropout3 = nn.Dropout1d(0.2)

        # Fully connected layer 2
        self.fc2 = nn.Linear(128, 64)
        self.dropout4 = nn.Dropout1d(0.2)

        # Output layer
        self.fc3 = nn.Linear(64, 29)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = F.relu(self.fc2(x))
        x = self.dropout4(x)
        x = self.fc3(x)
        return x


# Convolutional neural network A
class CNN_A(nn.Module):
    def __init__(self):
        super(CNN_A, self).__init__()

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Convolutional layer 2
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Fully connected layer 1
        self.fc1 = nn.Linear(32 * 25**2, 128)

        # Fully connected layer 2
        self.fc2 = nn.Linear(128, 64)

        # Output layer
        self.fc3 = nn.Linear(64, 29)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Convolutional neural network B
class CNN_B(nn.Module):
    def __init__(self):
        super(CNN_B, self).__init__()

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.05)

        # Convolutional layer 2
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.05)

        # Fully connected layer 1
        self.fc1 = nn.Linear(32 * 25**2, 128)
        self.dropout3 = nn.Dropout1d(0.05)

        # Fully connected layer 2
        self.fc2 = nn.Linear(128, 64)
        self.dropout4 = nn.Dropout1d(0.05)

        # Output layer
        self.fc3 = nn.Linear(64, 29)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = F.relu(self.fc2(x))
        x = self.dropout4(x)
        x = self.fc3(x)
        return x
