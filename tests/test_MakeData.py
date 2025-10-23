# Tests for the MakeData module

import pandas as pd
import teaspoon.MakeData.PointCloud as gPC
import unittest

class TestDataGeneration(unittest.TestCase):
    
    # TODO: add tests for the dynsyslib

    def test_generate_point_cloud_dataframe(self):
        df = gPC.testSetManifolds(numDgms=2, numPts=50, maxDim=2, permute=False, seed=42)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 6 * 2  # 6 manifold types * numDgms
        assert 'Dgm0' in df.columns
        assert 'Dgm1' in df.columns
        assert 'Dgm2' in df.columns
        assert 'trainingLabel' in df.columns
        assert not df.isnull().values.any()

    def test_generate_normal_distribution_dataframe(self):
        df = gPC.testSetClassification(N=10, numDgms=5, permute=False, seed=42)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 5 * 2  # numDgms * 2 types
        assert 'Dgm' in df.columns
        assert 'mean' in df.columns
        assert 'sd' in df.columns
        assert 'trainingLabel' in df.columns
        assert not df.isnull().values.any()


if __name__ == '__main__':
    unittest.main()
