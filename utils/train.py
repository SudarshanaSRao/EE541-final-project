import torch
import time
import datetime
import numpy as np
import torch.nn as nn
from dataclasses import dataclass


# Display live progress bar in console
class ProgressBar:

    def __init__(self, prefix, total_steps, bar_length=40):
        # Initialize
        self.prefix = prefix
        self.total_steps = total_steps
        self.bar_length = bar_length
        self.steps = 0

    def step(self):
        # Create progress bar
        self.steps += 1
        percent = self.steps / self.total_steps
        fill_length = int(percent * self.bar_length)
        empty_length = self.bar_length - fill_length
        bar = '█' * fill_length + '-' * empty_length

        # Update progress bar display
        output = '{}: |{}| {:.1f}%'.format(self.prefix, bar, percent * 100)
        print(output, end='\r')

        # Clear line
        if self.steps == self.total_steps:
            print(' ' * len(output), end='\r')


# Container for various training metrics
@dataclass
class Metrics:
    train_loss: list
    train_accuracy: list
    test_loss: list
    test_accuracy: list
    epoch_time: list


def train_model(model, train_loader, valid_loader, learning_rate, num_epochs, device, weight_decay=0, conv=False):
    # Initialize training parameters
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Initialize metrics
    train_loss_list = []
    test_loss_list = []
    train_accuracy_list = []
    test_accuracy_list = []
    epoch_time_list = []

    # Train model
    for epoch in range(num_epochs):
        # Start timer
        start_time = time.time()

        # Train and evaluate model
        train(train_loader, model, loss_func, optimizer, device, conv)
        train_loss, train_accuracy = test(train_loader, model, loss_func, device, conv, '[2/3] Training Set Evaluation')
        valid_loss, valid_accuracy = test(valid_loader, model, loss_func, device, conv, '[3/3] Validation Set Evaluation')

        # Store epoch metrics
        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        test_loss_list.append(valid_loss)
        test_accuracy_list.append(valid_accuracy)

        # Stop timer and store epoch time
        duration = time.time() - start_time
        epoch_time_list.append(duration)

        # Output progress
        print('Epoch {} | Loss = {:.4f} | Train Accuracy = {:.2f}% | Valid Accuracy = {:.2f}% | Time = {}'
            .format(epoch + 1, train_loss, train_accuracy, valid_accuracy, datetime.timedelta(seconds=int(duration))), end='\r')
        print()

    # Initialize metrics object
    metrics = Metrics(
        train_loss=train_loss_list,
        train_accuracy=train_accuracy_list,
        test_loss=test_loss_list,
        test_accuracy=test_accuracy_list,
        epoch_time=epoch_time_list
    )

    return metrics


# Perform single training pass over dataset
def train(data_loader, model, loss_func, optimizer, device, conv):
    # Initialize parameters
    num_inputs = np.array(data_loader.dataset[0][0].numpy().shape).prod()

    # Set mode to training
    model.train()

    # Initialize progress bar
    progress = ProgressBar('[1/3] Training Step', len(data_loader))

    # Iterate through batches
    for images, labels in data_loader: 
        # Transfer images and labels to GPU
        images, labels = images.to(device), labels.to(device)
        if not conv:
            images = images.view(-1, num_inputs)
        
        # Forward pass 
        outputs = model(images)
        loss = loss_func(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Optimization
        optimizer.step()

        # Update progress
        progress.step()


# Perform single test pass over dataset
def test(data_loader, model, loss_func, device, conv, name):
    # Initialize parameters
    num_inputs = np.array(data_loader.dataset[0][0].numpy().shape).prod()
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    total_loss = 0
    correct = 0

    # Set mode to evaluation
    model.eval()

    # Initialize progress bar
    progress = ProgressBar(name, len(data_loader))

    # Iterate through batches
    with torch.no_grad():
        for images, labels in data_loader:
            # Transfer images and labels to GPU
            images, labels = images.to(device), labels.to(device)
            if not conv:
                images = images.view(-1, num_inputs)

            # Forward pass
            outputs = model(images)
            loss = loss_func(outputs, labels)

            # Transfer outputs and labels to CPU
            outputs, labels = outputs.cpu(), labels.cpu()

            # Compute batch metrics
            total_loss += loss.item()
            pred = torch.max(outputs, 1)[1]
            correct += (pred == labels).sum().numpy()

            # Update progress
            progress.step()
            
    # Compute metrics for dataset
    total_loss /= num_batches
    accuracy = (correct / size) * 100

    return total_loss, accuracy