# Display live progress bar in console
class ProgressBar:

    def __init__(self, total_steps, bar_length=40):
        # Initialize
        self.steps = 0
        self.total_steps = total_steps
        self.bar_length = bar_length

    def step(self):
        # Create progress bar
        self.steps += 1
        percent = self.steps / self.total_steps
        fill_length = int(percent * self.bar_length)
        empty_length = self.bar_length - fill_length
        bar = '█' * fill_length + '-' * empty_length

        # Update progress bar display
        output = 'Progress: |{}| {:.1f}%'.format(bar, percent * 100)
        print(output, end='\r')

        # Print new line upon completion
        if self.steps == self.total_steps:
            print()