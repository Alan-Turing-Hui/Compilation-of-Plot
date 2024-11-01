import numpy as np
import matplotlib.pyplot as plt


class ConfusionMatrixPlotter:
    """
    This class is used to plot the confusion matrix.
    It contains methods to calculate the percentage and plot the matrix with additional information.
    """
    def __init__(self, labels, cm, model_name):
        """
        Initialize the class with labels, confusion matrix, and model name.

        :param labels: List of labels for the confusion matrix.
        :param cm: The confusion matrix as a 2D numpy array.
        :param model_name: The name of the model to be used in the title.
        """
        self.labels = labels
        self.cm = cm
        self.model_name = model_name

    def percentage_value(self, value, total):
        """
        Calculate the percentage value of a given value with respect to the total.

        :param value: The value for which percentage is to be calculated.
        :param total: The total value.
        :return: The calculated percentage rounded to 1 decimal place.
        """
        return np.round((value / total) * 100, 1)

    def plot(self, save_path=None):
        """
        Plot the confusion matrix.
        It shows the matrix with labels, colors, and percentage values on the diagonal cells.

        :param save_path: Optional path to save the plotted confusion matrix. If None, the plot will be shown.
        """
        plt.imshow(self.cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix for " + self.model_name)
        plt.colorbar()

        tick_marks = np.arange(len(self.labels))
        plt.xticks(tick_marks, self.labels)
        plt.yticks(tick_marks, self.labels)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')

        # Add values to each cell of the matrix
        for i in range(len(self.cm)):
            for j in range(len(self.cm[i])):
                cell_value = self.cm[i][j]
                text_to_display = str(int(cell_value))
                if i == j:
                    total_sum = np.sum(self.cm)
                    percentage = self.percentage_value(cell_value, total_sum)
                    text_to_display += '\n({}%)'.format(int(percentage))
                plt.text(j, i, text_to_display, ha="center", va="center", color="black")

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()