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
from teaspoon.MakeData.DynSysLib import conservative_flows as cf
from teaspoon.MakeData.DynSysLib import delayed_flows as df
from teaspoon.MakeData.DynSysLib import driven_dissipative_flows as ddf
from teaspoon.MakeData.DynSysLib import medical_data as md

class TestPeriodic(unittest.TestCase):

    def test_run_all(self):
        """
        Checking that you can run all the commands with the original inputs 
        """
        t, ts = pf.sine()
        t, ts = pf.incommensurate_sine()

class TestNoiseModels(unittest.TestCase):

    def test_run_all(self):
        """
        Checking that you can run all the commands with the original inputs 
        """

        t, ts = nm.gaussian_noise()
        t, ts = nm.uniform_noise()
        t, ts = nm.rayleigh_noise()
        t, ts = nm.exponential_noise()

class TestMaps(unittest.TestCase):

    def test_run_all(self):
        """
        Checking that you can run all the commands with the original inputs 
        """

        t, ts = maps.gingerbread_man_map()
        t, ts = maps.sine_map()
        t, ts = maps.tent_map()
        t, ts = maps.linear_congruential_generator_map()
        t, ts = maps.rickers_population_map()
        t, ts = maps.gauss_map()
        t, ts = maps.cusp_map()
        t, ts = maps.pinchers_map()
        t, ts = maps.sine_circle_map()
        t, ts = maps.logistic_map()
        t, ts = maps.henon_map()
        t, ts = maps.lozi_map()
        t, ts = maps.delayed_logstic_map()
        t, ts = maps.tinkerbell_map()
        t, ts = maps.burgers_map()
        t, ts = maps.holmes_cubic_map()

class TestAutoDisFlows(unittest.TestCase):

    def test_run_all(self):
        """
        Checking that you can run all the commands with the original inputs 
        """

        t, ts = adf.lorenz()
        t, ts = adf.rossler()
        t, ts = adf.coupled_lorenz_rossler()
        t, ts = adf.coupled_rossler_rossler()
        t, ts = adf.chua()
        t, ts = adf.double_pendulum()
        t, ts = adf.diffusionless_lorenz_attractor()
        t, ts = adf.complex_butterfly()
        t, ts = adf.chens_system()
        t, ts = adf.hadley_circulation()
        t, ts = adf.ACT_attractor()
        t, ts = adf.rabinovich_frabrikant_attractor()
        t, ts = adf.linear_feedback_rigid_body_motion_system()
        t, ts = adf.moore_spiegel_oscillator()
        t, ts = adf.thomas_cyclically_symmetric_attractor()
        t, ts = adf.halvorsens_cyclically_symmetric_attractor()
        t, ts = adf.burke_shaw_attractor()
        t, ts = adf.rucklidge_attractor()
        t, ts = adf.WINDMI()
        t, ts = adf.simplest_quadratic_chaotic_flow()
        t, ts = adf.simplest_cubic_chaotic_flow()
        t, ts = adf.simplest_piecewise_linear_chaotic_flow()
        t, ts = adf.double_scroll()

class TestConsFlows(unittest.TestCase):

    def test_run_all(self):
        """
        Checking that you can run all the commands with the original inputs 
        """

        t, ts = cf.simplest_driven_chaotic_flow()
        t, ts = cf.nose_hoover_oscillator()
        t, ts = cf.labyrinth_chaos()
        t, ts = cf.henon_heiles_system()

class DelFlows(unittest.TestCase):

    def test_run_all(self):
        """
        Checking that you can run all the commands with the original inputs 
        """

        t, ts = df.mackey_glass()

class DrivDisFlows(unittest.TestCase):

    def test_run_all(self):
        """
        Checking that you can run all the commands with the original inputs 
        """

        t, ts = ddf.base_excited_magnetic_pendulum()
        t, ts = ddf.driven_pendulum()
        t, ts = ddf.driven_van_der_pol_oscillator()
        t, ts = ddf.shaw_van_der_pol_oscillator()
        t, ts = ddf.forced_brusselator()
        t, ts = ddf.ueda_oscillator()
        t, ts = ddf.duffings_two_well_oscillator()
        t, ts = ddf.duffing_van_der_pol_oscillator()
        t, ts = ddf.rayleigh_duffing_oscillator()
        
class MedData(unittest.TestCase):

    def test_run_all(self):
        """
        Checking that you can run all the commands with the original inputs 
        """

        t, ts = md.ECG()      
        t, ts = md.EEG()      

        
if __name__ == '__main__':
    unittest.main()
