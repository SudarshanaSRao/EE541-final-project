import torch
import random
import operator
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


def split_dataset(dataset, num_samples, dataset_split, batch_size, seed=0):
    '''
    Create training and test data loaders from the given dataset.

    Params:
        dataset = PyTorch Dataset instance for full dataset
        num_samples = Number of samples to use from the full dataset
        dataset_split = List of fractions in train/valid/test split
        batch_size Minibatch size for training
        seed = Random seed for dataset shuffle and split. Set to None for no seed

    Returns:
        train_loader = Dataloader for the training samples
        valid_loader = Dataloader for the training samples
        test_loader = Dataloader for the test samples
    '''

    # Perform stratified split of dataset indicies
    train_size = int(round(num_samples * dataset_split[0]))
    valid_size = int(round(num_samples * dataset_split[1]))
    test_size = int(round(num_samples * dataset_split[2]))
    dataset_inds = list(range(len(dataset)))
    train_valid_inds, test_inds = train_test_split(dataset_inds, train_size=train_size + valid_size, 
            test_size=test_size, random_state=seed, stratify=dataset.targets)
    train_inds, valid_inds = train_test_split(train_valid_inds, train_size=train_size, 
            test_size=valid_size, random_state=seed, stratify=operator.itemgetter(*train_valid_inds)(dataset.targets))

    # Create training and test subsets
    train_set = Subset(dataset, train_inds)
    valid_set = Subset(dataset, valid_inds)
    test_set = Subset(dataset, test_inds)

    # Initialize data loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


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


def display_dataset(dataset, seed=0):
    rows, cols = 4, 4

    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    random.seed(seed)
    inds = random.sample(range(len(dataset)), rows * cols)

    fig, axes = plt.subplots(rows, cols)
    fig.subplots_adjust(hspace=0.4, wspace = 0.2)

    for i in range(0, rows):
        for j in range(0, cols):
            axes[i, j].get_xaxis().set_visible(False)
            axes[i, j].get_yaxis().set_visible(False)
            axes[i, j].imshow(dataset[inds[cols * i + j]][0].permute(1, 2, 0))
            axes[i, j].set_title(idx_to_class[dataset[inds[cols * i + j]][1]])

# loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# def get_mean_and_std(dataloader):
#     channels_sum, channels_squared_sum, num_batches = 0, 0, 0
#     for data, _ in dataloader:
#         # Mean over batch, height and width, but not over the channels
#         channels_sum += torch.mean(data, dim=[0,2,3])
#         channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
#         num_batches += 1
    
#     mean = channels_sum / num_batches

#     # std = sqrt(E[X^2] - (E[X])^2)
#     std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

#     return mean, std

# get_mean_and_std(loader)