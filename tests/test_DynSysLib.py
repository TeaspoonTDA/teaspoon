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
        t, ts = adf.autonomous_dissipative_flows('diffusionless_lorenz_attractor')
        t, ts = adf.autonomous_dissipative_flows('complex_butterfly')
        t, ts = adf.autonomous_dissipative_flows('chens_system')
        t, ts = adf.autonomous_dissipative_flows('hadley_circulation')
        t, ts = adf.autonomous_dissipative_flows('ACT_attractor')
        t, ts = adf.autonomous_dissipative_flows('rabinovich_frabrikant_attractor')
        t, ts = adf.autonomous_dissipative_flows('linear_feedback_rigid_body_motion_system')
        t, ts = adf.autonomous_dissipative_flows('moore_spiegel_oscillator')
        t, ts = adf.autonomous_dissipative_flows('thomas_cyclically_symmetric_attractor')
        t, ts = adf.autonomous_dissipative_flows('halvorsens_cyclically_symmetric_attractor')
        t, ts = adf.autonomous_dissipative_flows('burke_shaw_attractor')
        t, ts = adf.autonomous_dissipative_flows('rucklidge_attractor')
        t, ts = adf.autonomous_dissipative_flows('WINDMI')
        t, ts = adf.autonomous_dissipative_flows('simplest_quadratic_chaotic_flow')
        t, ts = adf.autonomous_dissipative_flows('simplest_cubic_chaotic_flow')
        t, ts = adf.autonomous_dissipative_flows('simplest_piecewise_linear_chaotic_flow')
        t, ts = adf.autonomous_dissipative_flows('double_scroll')

class TestConsFlows(unittest.TestCase):

    def test_run_all(self):
        """
        Checking that you can run all the commands with the original inputs 
        """

        t, ts = cf.conservative_flows('simplest_driven_chaotic_flow')
        t, ts = cf.conservative_flows('nose_hoover_oscillator')
        t, ts = cf.conservative_flows('labyrinth_chaos')
        t, ts = cf.conservative_flows('henon_heiles_system')

class DelFlows(unittest.TestCase):

    def test_run_all(self):
        """
        Checking that you can run all the commands with the original inputs 
        """

        t, ts = df.delayed_flows('mackey_glass')

class DrivDisFlows(unittest.TestCase):

    def test_run_all(self):
        """
        Checking that you can run all the commands with the original inputs 
        """

        t, ts = ddf.driven_dissipative_flows('base_excited_magnetic_pendulum')
        t, ts = ddf.driven_dissipative_flows('driven_pendulum')
        t, ts = ddf.driven_dissipative_flows('driven_van_der_pol_oscillator')
        t, ts = ddf.driven_dissipative_flows('shaw_van_der_pol_oscillator')
        t, ts = ddf.driven_dissipative_flows('forced_brusselator')
        t, ts = ddf.driven_dissipative_flows('ueda_oscillator')
        t, ts = ddf.driven_dissipative_flows('duffings_two_well_oscillator')
        t, ts = ddf.driven_dissipative_flows('duffing_van_der_pol_oscillator')
        t, ts = ddf.driven_dissipative_flows('rayleigh_duffing_oscillator')
        
class MedData(unittest.TestCase):

    def test_run_all(self):
        """
        Checking that you can run all the commands with the original inputs 
        """

        t, ts = md.medical_data('ECG')      
        t, ts = md.medical_data('EEG')      

        
if __name__ == '__main__':
    unittest.main()
