import unittest
from teaspoon.ML import load_datasets
from teaspoon.ML import feature_functions as FF


class TestFeatureFunctions(unittest.TestCase):
    def test_F_Image(self):
        mnist = load_datasets.mnist()
        dgms = mnist['zero_dim_rtl']
        PS = 1
        var = 1
        persImg = FF.F_Image(dgms, PS, var, plot=False, D_Img=[], pers_imager=None, training=True, parallel=False)
        self.assertEqual(persImg['F_Matrix'].all(), persImg['F_Matrix'].all())

    def test_F_Image_parallel(self):
        mnist = load_datasets.mnist()
        dgms = mnist['zero_dim_rtl']
        PS = 1
        var = 1
        persImg = FF.F_Image(dgms, PS, var, plot=False, D_Img=[], pers_imager=None, training=True, parallel=True)
        self.assertEqual(persImg['F_Matrix'].all(), persImg['F_Matrix'].all())

if __name__ == '__main__':
    unittest.main()