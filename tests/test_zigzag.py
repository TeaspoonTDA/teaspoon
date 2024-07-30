import numpy as np
from teaspoon.TDA.fast_zigzag import generate_input_file

import unittest


class FZZ(unittest.TestCase):

    def test_FZZ(self):

        point_clouds = []
        t = np.linspace(0, 2*np.pi, 8)[:-1]
        point_clouds.append(np.vstack((1*np.cos(t), 1*np.sin(t))).T)
        point_clouds.append(np.vstack((21*np.cos(t), 21*np.sin(t))).T)

        inserts, deletes = generate_input_file(point_clouds, filename='output', radius=19, n_perm=25, plotting=False)

        ins = [63, 77]
        dels = [140, 154]

        self.assertEqual(inserts, ins, 'Assert equal failed') 
        self.assertEqual(deletes, dels, 'Assert equal failed') 


if __name__ == '__main__':
    unittest.main()
