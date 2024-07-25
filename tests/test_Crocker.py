import numpy as np
from teaspoon.TDA.Persistence import CROCKER, CROCKER_Stack

import unittest


class CrockerTest(unittest.TestCase):

    def test_crocker_stack_equal_crocker_plot(self):

        dgms = [
            np.array([[0.5, 0.75], [1, 1.5], [1.25, 2.5], [2, np.inf]]),
            np.array([[0.5, 1], [1, 2], [1.25, 3], [2, np.inf]])
        ]

        plot = CROCKER(dgms, maxEps=4, numStops=10, plotting=False)
        stack = CROCKER_Stack(dgms, maxEps=4, numStops=10, plotting=False)

        self.assertTrue((plot == stack[0]).all())
        self.assertTrue(plot.ndim == 2)
        self.assertTrue(stack.ndim == 3)


if __name__ == '__main__':
    unittest.main()
