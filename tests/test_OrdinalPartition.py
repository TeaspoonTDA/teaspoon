# Tests done by Audun Myers as of 11/5/20 (Version 0.0.1)


# In[ ]: Persistent homology of networks

#import needed packages
import numpy as np
from teaspoon.SP.network import ordinal_partition_graph
from teaspoon.TDA.PHN import PH_network
from teaspoon.TDA.PHN import DistanceMatrix
from teaspoon.SP.network_tools import make_network
from teaspoon.parameter_selection.MsPE import MsPE_tau

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx

import unittest


class TestOrdinalPartition(unittest.TestCase):

    def test_ordinalPartition(self):


        #generate a siple sinusoidal time series
        t = np.linspace(0,30,300)
        ts = np.sin(t) + np.sin(2*t)

        #Get appropriate dimension and delay parameters for permutations
        tau = int(MsPE_tau(ts))
        n = 5

        #create adjacency matrix, this
        A = ordinal_partition_graph(ts, n, tau)

        self.assertAlmostEqual(A.min(), 0)

if __name__ == '__main__':
    unittest.main()
