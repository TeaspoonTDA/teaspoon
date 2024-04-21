"""
This is a script to run very basic functionality tests of the code.

- Liz Munch 7/21
"""

import unittest

from teaspoon.MakeData.PointCloud import Annulus
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from teaspoon.TDA.Magnitude import MagnitudeFunction
    

class TestMagnitude(unittest.TestCase):

    def test_magnitude_vector(self):
        """
        Checking that you can compute magnitude on annulus example data
        """

        print('Testing that a magnitude returns expected vectors. ')
        P = Annulus()
        D = distance_matrix(P,P)
        T,M = MagnitudeFunction(D)


        self.assertEqual(len(T), len(M), "Magnitude needs to return two vectors of the same length")


if __name__ == '__main__':
    unittest.main()
