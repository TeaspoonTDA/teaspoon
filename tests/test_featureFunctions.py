import unittest
from teaspoon.ML import load_datasets
from teaspoon.ML import feature_functions as FF
import numpy as np


class TestFeatureFunctions(unittest.TestCase):
    def test_F_Image(self):
        mnist = load_datasets.mnist()
        dgms = mnist['zero_dim_rtl']
        PS = 15
        var = 1
        persImg = FF.F_Image(dgms, PS, var, pers_imager=None, training=True, parallel=True)
        self.assertEqual(persImg['F_Matrix'][0].all(), np.array([4.83911125e+00, 1.12597993e-03, 2.64421959e-20, 2.15926035e+00,1.80429682e+00, 2.51962051e+01]).all())

    def test_F_Image_parallel(self):
        mnist = load_datasets.mnist()
        dgms = mnist['zero_dim_rtl']
        PS = 15
        var = 1
        persImg = FF.F_Image(dgms, PS, var, pers_imager=None, training=True, parallel=True)
        self.assertEqual(persImg['F_Matrix'][0].all(), np.array([4.83911125e+00, 1.12597993e-03, 2.64421959e-20, 2.15926035e+00,1.80429682e+00, 2.51962051e+01]).all())

    def test_kernel_distance(self):
        mnist = load_datasets.mnist()
        dgms = mnist['zero_dim_rtl'][0:2]
        sigma=.3
        dgms = FF.reshape_to_array(dgms)
        k = FF.kernel_distance(dgms, dgms, sigma)
        self.assertEqual(k.all(), np.array([[0,.47224819],[.47224819,0]]).all())

    def test_kernel_distance(self):
        mnist = load_datasets.mnist()
        dgms = mnist['zero_dim_rtl'][0:2]
        sigma=.3
        dgms = FF.reshape_to_array(dgms)
        k = FF.kernel_distance_parallel(dgms, dgms, sigma)
        self.assertEqual(k.all(), np.array([[0,.47224819],[.47224819,0]]).all())

if __name__ == '__main__':
    unittest.main()