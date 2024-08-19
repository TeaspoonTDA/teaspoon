"""
Tests to verify that all systems run with default inputs using the legacy meta functions. 
"""

import unittest

from ripser import ripser
from teaspoon.TDA.Draw import drawDgm
from teaspoon.MakeData.PointCloud import Annulus
from teaspoon.MakeData.DynSysLib import periodic_functions as pf
from teaspoon.MakeData.DynSysLib import noise_models as nm

class TestPeriodic(unittest.TestCase):

    def test_run_all(self):
        """
        Checking that you can run all the commands with the original inputs 
        """
        
        pf.periodic_functions('sine')
        pf.periodic_functions('incommensurate_sine')

class TestNoiseModels(unittest.TestCase):

    def test_run_all(self):
        """
        Checking that you can run all the commands with the original inputs 
        """

        nm.noise_models('gaussian_noise')
        nm.noise_models('uniform_noise')
        nm.noise_models('rayleigh_noise')
        nm.noise_models('exponential_noise')

if __name__ == '__main__':
    unittest.main()
