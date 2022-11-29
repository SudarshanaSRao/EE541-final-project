import numpy as np
import matplotlib.pyplot as plt

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

def plot_metrics(train_metrics, test_metrics):
    # Output final accuracy
    print('Final Train Accuracy = {:.2f}%'.format(train_metrics[1][-1]))
    print('Final Test Accuracy  = {:.2f}%'.format(test_metrics[1][-1]))

    # Initialize multiple plot figure
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))

    # Plot loss curves
    epochs = np.arange(1, len(train_metrics[0]) + 1)
    ax[0].plot(epochs, db(train_metrics[0]), label='Train set')
    ax[0].plot(epochs, db(test_metrics[0]), label='Test set')

    # Plot accuracy curves
    ax[1].plot(epochs, train_metrics[1])
    ax[1].plot(epochs, test_metrics[1])

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