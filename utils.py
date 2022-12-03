import time
import datetime
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


def split_dataset(dataset, num_samples, train_split, batch_size, seed=0):
    '''
    Create training and test data loaders from the given dataset.

    Params:
        dataset = PyTorch Dataset instance for full dataset
        num_samples = Number of samples to use from the full dataset
        train_split = Fraction of train data in train/test split
        batch_size Minibatch size for training
        seed = Random seed for dataset shuffle and split. Set to None for no seed.

    Returns:
        train_loader = Dataloader for the training samples
        test_loader = Dataloader for the test samples
    '''

    # Perform stratified split of dataset indicies
    train_size = int((num_samples * train_split) // batch_size) * batch_size
    test_size = num_samples - train_size
    dataset_inds = list(range(len(dataset)))
    train_inds, test_inds = train_test_split(dataset_inds, train_size=train_size, 
            test_size=test_size, random_state=seed, stratify=dataset.targets)

    # Create training and test subsets
    train_set = Subset(dataset, train_inds)
    test_set = Subset(dataset, test_inds)

    # Initialize data loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# Container for various training metrics
@dataclass
class Metrics:
    train_loss: list
    train_accuracy: list
    test_loss: list
    test_accuracy: list
    epoch_times: list


def train_model(model, train_loader, valid_loader, learning_rate, num_epochs, device, weight_decay=0, conv=False):
    # Initialize training parameters
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Initialize metrics
    train_loss_list = []
    test_loss_list = []
    train_accuracy_list = []
    test_accuracy_list = []
    epoch_times = []

    # Train model
    for epoch in range(num_epochs):
        # Start timer
        start_time = time.time()

        # Train and evaluate model
        train_loss, train_accuracy = train(train_loader, model, loss_func, optimizer, device, conv)
        valid_loss, valid_accuracy = test(valid_loader, model, loss_func, device, conv)

        # Store epoch metrics
        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        test_loss_list.append(valid_loss)
        test_accuracy_list.append(valid_accuracy)

        # Stop timer and store epoch time
        duration = time.time() - start_time
        epoch_times.append(duration)

        # Output progress
        print('Epoch {} | Loss = {:.4f} | Train Accuracy = {:.2f}% | Test Accuracy = {:.2f}% | Time = {}'
            .format(epoch + 1, train_loss, train_accuracy, valid_accuracy, datetime.timedelta(seconds=int(duration))))

    # Initialize metrics object
    metrics = Metrics(
        train_loss=train_loss_list,
        train_accuracy=train_accuracy_list,
        test_loss=test_loss_list,
        test_accuracy=test_accuracy_list,
        epoch_times=epoch_times
    )

    return metrics


# Perform single training pass over dataset
def train(data_loader, model, loss_func, optimizer, device, conv):
    # Initialize parameters
    num_inputs = np.array(data_loader.dataset[0][0].numpy().shape).prod()
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    total_loss = 0
    correct = 0

    # Set mode to training
    model.train()

    # Initialize progress bar
    progress = ProgressBar('Train Progress', len(data_loader))

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


# Perform single test pass over dataset
def test(data_loader, model, loss_func, device, conv):
    # Initialize parameters
    num_inputs = np.array(data_loader.dataset[0][0].numpy().shape).prod()
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    total_loss = 0
    correct = 0

    # Set mode to evaluation
    model.eval()

    # Initialize progress bar
    progress = ProgressBar('Valid Progress', len(data_loader))

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


# Save model and training metrics to file.
# Does not save optimizer state required for further training
def save_model(filename, model, metrics):
    state = {
        'model_state_dict': model.state_dict(),
        'metrics': metrics
        }
    torch.save(state, filename)


# Save model and training metrics from file
def load_model(filename, model):
    state = torch.load(filename)
    model.load_state_dict(state['model_state_dict'])
    metrics = state['metrics']

    return metrics


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

        # Print new line upon completion
        if self.steps == self.total_steps:
            print()

            
# Convert to decibels
def db(x):
    return 20 * np.log10(x)

def plot_metrics(metrics):
    # Output final accuracy
    print('Final Train Accuracy = {:.2f}%'.format(metrics.train_accuracy[-1]))
    print('Final Test Accuracy  = {:.2f}%'.format(metrics.test_accuracy[-1]))

    # Initialize multiple plot figure
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))

    # Plot loss curves
    epochs = np.arange(1, len(metrics.train_loss) + 1)
    ax[0].plot(epochs, db(metrics.train_loss), label='Train set')
    ax[0].plot(epochs, db(metrics.test_loss), label='Test set')

    # Plot accuracy curves
    ax[1].plot(epochs, metrics.train_accuracy)
    ax[1].plot(epochs, metrics.test_accuracy)

    # Set plot titles and labels
    ax[0].set_title("Loss Curves")
    ax[0].set_xlabel("Epoch Number")
    ax[0].set_ylabel("Log-Loss (dB)")
    ax[1].set_title("Accuracy Curves")
    ax[1].set_xlabel("Epoch Number")
    ax[1].set_ylabel("Accuracy (%)")

    # Setup parent figure
    fig.suptitle('Training Results', y=1.04)
    fig.subplots_adjust(hspace=1.5, wspace=0.4)
    fig.legend(bbox_to_anchor=(0.66, -0.2), loc='lower right', ncol=4)
    plt.show()