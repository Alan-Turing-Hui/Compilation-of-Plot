import unittest
import numpy as np
from plot.confusion_matrix_plot import ConfusionMatrixPlotter

class TestConfusionMatrixPlotter(unittest.TestCase):
    def test_plot(self):
        labels = ['class1', 'class2', 'class3']
        cm = np.array([[10, 2, 1], [3, 15, 2], [1, 1, 12]])
        model_name = 'TestModel'
        plotter = ConfusionMatrixPlotter(labels, cm, model_name)
        try:
            plotter.plot()  
        except Exception as e:
            self.fail(f"Plotting failed with exception: {e}")


if __name__ == '__main__':
    unittest.main()