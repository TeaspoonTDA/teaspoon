"""
This is a script to run very basic functionality tests of the code.

- Liz Munch 7/21
"""

import unittest
import numpy as np
from teaspoon.TDA.BuZZ import PtClouds
from teaspoon.SP.tsa_tools import takens

class TestZigZag(unittest.TestCase):

    def test_compute_zig_zag(self):

        t = np.linspace(0, 6*np.pi+1, 50)

# Amplitudes of sine waves
        amps = [0.1,0.5,1,1.5,2,1.5,1,0.5,0.1]

        ts_list = []
        ptcloud_list = []
        for a in amps:
            y = a*np.sin(t) + (0.1*np.random.randint(-100,100,len(t))/100)

            # Compute sine wave and add noise uniformly distributed in [-0.1, 0.1]
            ts_list.append(y)

            # Compute time delay embedding point clouds
            ptcloud_list.append(takens(y, n=2, tau=4))

        ZZ = PtClouds(ptcloud_list, num_landmarks=25, verbose=True)
        ZZ.run_Zigzag(r=0.85)
        dim1 = list(ZZ.zz_dgms[1][0])
        dim0 = list(ZZ.zz_dgms[0][0])
        self.assertEqual(dim0, [0,9])
        self.assertEqual(dim1, [2,6.5])
        print("ZigZag Persistence Computed")

if __name__ == '__main__':
    unittest.main()
