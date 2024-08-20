"""
Tests to verify that all systems run with default inputs using the legacy meta functions. 
"""

import unittest

from ripser import ripser
from teaspoon.TDA.Draw import drawDgm
from teaspoon.MakeData.PointCloud import Annulus
from teaspoon.MakeData.DynSysLib import periodic_functions as pf
from teaspoon.MakeData.DynSysLib import noise_models as nm
from teaspoon.MakeData.DynSysLib import maps as maps
from teaspoon.MakeData.DynSysLib import autonomous_dissipative_flows as adf

class TestPeriodic(unittest.TestCase):

    def test_run_all(self):
        """
        Checking that you can run all the commands with the original inputs 
        """

        t, ts = pf.periodic_functions('sine')
        t, ts = pf.periodic_functions('incommensurate_sine')

class TestNoiseModels(unittest.TestCase):

    def test_run_all(self):
        """
        Checking that you can run all the commands with the original inputs 
        """

        t, ts = nm.noise_models('gaussian_noise')
        t, ts = nm.noise_models('uniform_noise')
        t, ts = nm.noise_models('rayleigh_noise')
        t, ts = nm.noise_models('exponential_noise')

class TestMaps(unittest.TestCase):

    def test_run_all(self):
        """
        Checking that you can run all the commands with the original inputs 
        """

        t, ts = maps.maps('gingerbread_man_map')
        t, ts = maps.maps('sine_map')
        t, ts = maps.maps('tent_map')
        t, ts = maps.maps('linear_congruential_generator_map')
        t, ts = maps.maps('rickers_population_map')
        t, ts = maps.maps('gauss_map')
        t, ts = maps.maps('cusp_map')
        t, ts = maps.maps('pinchers_map')
        t, ts = maps.maps('sine_circle_map')
        t, ts = maps.maps('logistic_map')
        t, ts = maps.maps('henon_map')
        t, ts = maps.maps('lozi_map')
        t, ts = maps.maps('delayed_logstic_map')
        t, ts = maps.maps('tinkerbell_map')
        t, ts = maps.maps('burgers_map')
        t, ts = maps.maps('holmes_cubic_map')

class TestAutoDisFlows(unittest.TestCase):

    def test_run_all(self):
        """
        Checking that you can run all the commands with the original inputs 
        """

        t, ts = adf.autonomous_dissipative_flows('lorenz')
        t, ts = adf.autonomous_dissipative_flows('rossler')
        t, ts = adf.autonomous_dissipative_flows('coupled_lorenz_rossler')
        t, ts = adf.autonomous_dissipative_flows('coupled_rossler_rossler')
        t, ts = adf.autonomous_dissipative_flows('chua')
        t, ts = adf.autonomous_dissipative_flows('double_pendulum')
if __name__ == '__main__':
    unittest.main()
