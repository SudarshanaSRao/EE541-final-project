
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score



            
# Convert to decibels
def db(x):
    return 20 * np.log10(x)


def plot_metrics(metrics):
    # Output final accuracy
    print('Final Train Accuracy = {:.2f}%'.format(metrics.train_accuracy[-1]))
    print('Final Valid Accuracy  = {:.2f}%'.format(metrics.test_accuracy[-1]))
    print('Average Epoch Time   = {:.2f}s'.format(np.mean(metrics.epoch_time)))

    # Initialize multiple plot figure
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))

    # Plot loss curves
    epochs = np.arange(1, len(metrics.train_loss) + 1)
    ax[0].plot(epochs, db(metrics.train_loss), label='Train set')
    ax[0].plot(epochs, db(metrics.test_loss), label='Valid set')

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
    # fig.suptitle('Training Results', y=1.04)
    fig.subplots_adjust(hspace=1.5, wspace=0.4)
    fig.legend(bbox_to_anchor=(0.66, -0.2), loc='lower right', ncol=4)
    plt.show()


@dataclass
class EvalMetrics:
    class_precision: list
    class_recall: list
    class_f1: list
    precision: float
    recall: float
    f1: float
    accuracy: float

def eval_model(model, test_loader, device, conv=False):
    # Initialize confusion matrix and metrics
    conf_m = np.zeros((29, 29))
    labels_list = []
    preds_list = []

    # Compute confusion matrix
    with torch.no_grad():
        for images, labels in test_loader:
            # Transfer images to GPU
            images = images.to(device)
            if not conv:
                images = images.view(-1, 3 * 100**2)

            # Forward pass
            outputs = model(images)

            # Transfer outputs to CPU
            outputs = outputs.cpu()

            # Compute batch metrics
            preds = torch.max(outputs, 1)[1]
            for i in range(len(images)):
                conf_m[labels[i]][preds[i]] += 1
                labels_list += list(labels.numpy().copy())
                preds_list += list(preds.numpy().copy())
                

    # Normalize confusion matrix by true class
    for i in range(0, 29):
        conf_m[i] = conf_m[i] / np.sum(conf_m[i])

    # Compute metrics
    class_precision = precision_score(labels_list, preds_list, average='macro', zero_division=0)
    class_recall = recall_score(labels_list, preds_list, average='macro')
    class_f1 = f1_score(labels_list, preds_list, average='macro')
    precision = precision_score(labels_list, preds_list, average='macro', zero_division=0)
    recall = recall_score(labels_list, preds_list, average='macro')
    f1 = f1_score(labels_list, preds_list, average='macro')
    accuracy = accuracy_score(labels_list, preds_list)

    # Initialize metrics class
    eval_metrics = EvalMetrics(
        class_precision=class_precision,
        class_recall=class_recall,
        class_f1=class_f1,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy
    )

    # Plot metrics
    print('Precision = {:.4f}'.format(precision))
    print('Recall = {:.4f}'.format(recall))
    print('F1 = {:.4f}'.format(f1))
    print('Accuracy = {:.2f}%'.format(accuracy * 100))

    # Plot heatmap
    heatmap = sns.heatmap(conf_m, linewidth=0.4, cmap='viridis')  
    # plt.title('Model Confusion Matrix Heat Map')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()

    return eval_metrics