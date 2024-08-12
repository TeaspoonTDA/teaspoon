"""
I'm about to really screw with the DynSysLib organization. So I'm writing these tests to make sure that the original commands still work. 
"""

import unittest

from ripser import ripser
from teaspoon.TDA.Draw import drawDgm
from teaspoon.MakeData.PointCloud import Annulus
from teaspoon.MakeData.DynSysLib import periodic_functions as pf


class TestPeriodic(unittest.TestCase):

    def test_run_all(self):
        """
        Checking that you can run all the commands with the original inputs 
        """

        pf.periodic_functions('sine')
        pf.periodic_functions('incommensurate_sine')
        # pf.periodic_functions('err')


if __name__ == '__main__':
    unittest.main()
