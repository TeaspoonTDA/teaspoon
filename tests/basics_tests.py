"""
This is a script to run very basic functionality tests of the code.

- Liz Munch 7/21
"""

import unittest

from ripser import ripser
from teaspoon.TDA.Draw import drawDgm
from teaspoon.MakeData.PointCloud import Annulus



class TestAnnulus(unittest.TestCase):

    def test_dimension_1(self):
        """
        Checking that you can generate an annlus, and that it has one prominent
        bar
        """

        print('Testing that an annulus has a single long bar. ')
        numPts = 100
        seed = 0

        # Generate Torus
        t = Annulus(N=numPts,seed = seed)

        # Compute persistence diagram
        PD1 = ripser(t,2)['dgms'][1]
        Lifetimes = PD1[:,1] - PD1[:,0]
        countLongBars = (Lifetimes > 1).sum()

        self.assertEqual(countLongBars, 1, "An annulus should have one long bar")


if __name__ == '__main__':
    unittest.main()
