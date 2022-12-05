import torch
import torch.nn as nn
import torch.nn.functional as F


# Multilayer perceptron
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        # Fully connected layer 1
        self.hidden1 = nn.Linear(3 * 100**2, 1000)
        self.dropout1 = nn.Dropout1d(0.2)

        # Fully connected layer 2
        self.hidden2 = nn.Linear(1000, 200)
        self.dropout2 = nn.Dropout1d(0.2)

        # Output layer
        self.output = nn.Linear(200, 29)

    def forward(self, x):
        # Fully connected layer 1
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # Fully connected layer 2
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Output layer
        x = self.output(x)

        return x


# Baseline convolutional neural network from literature
class CNN_Baseline(nn.Module):
    def __init__(self):
        super(CNN_Baseline, self).__init__()
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
        # Convolutional layer 1
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.dropout1(x)

        # Convolutional layer 2
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout2(x)

        # Fully connected layer 1
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)

        # Fully connected layer 2
        x = F.relu(self.fc2(x))
        x = self.dropout4(x)
        x = self.fc3(x)
        return x
        

# Convolutional neural network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional block 1
        self.conv1a = nn.Conv2d(3, 32, 3, padding=1)
        self.norm1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, 3, padding=1)
        self.norm1b = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.2)

        # Convolutional block 2
        self.conv2a = nn.Conv2d(32, 64, 3, padding=1)
        self.norm2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, 3, padding=1)
        self.norm2b = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.2)

        # Convolutional block 3
        self.conv3a = nn.Conv2d(64, 128, 3, padding=1)
        self.norm3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, 3, padding=1)
        self.norm3b = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.2)

        # Fully connected layer 1
        self.fc1 = nn.Linear(128 * 12**2, 1024)
        self.normfc1 = nn.BatchNorm1d(1024)  
        self.dropoutfc1 = nn.Dropout1d(0.2)

        # Fully connected layer 2
        self.fc2 = nn.Linear(1024, 128)
        self.normfc2 = nn.BatchNorm1d(128)
        self.dropoutfc2 = nn.Dropout1d(0.2)

        # Output layer
        self.fc3 = nn.Linear(128, 29)

    def forward(self, x):
        # Convolutional block 1
        x = F.relu(self.conv1a(x))
        x = self.norm1a(x)
        x = F.relu(self.conv1b(x))
        x = self.norm1b(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Convolutional block 2
        x = F.relu(self.conv2a(x))
        x = self.norm2a(x)
        x = F.relu(self.conv2b(x))
        x = self.norm2b(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Convolutional block 3
        x = F.relu(self.conv3a(x))
        x = self.norm3a(x)
        x = F.relu(self.conv3b(x))
        x = self.norm3b(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        # Fully connected layer 1
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.normfc1(x)
        x = self.dropoutfc1(x)

        # Fully connected layer 2
        x = F.relu(self.fc2(x))
        x = self.normfc2(x)
        x = self.dropoutfc2(x)

        # Output layer
        x = self.fc3(x)

        return x