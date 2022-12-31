import numpy as np
from teaspoon.TDA.PHN import PH_network, point_summaries, DistanceMatrix
from teaspoon.parameter_selection.MsPE import MsPE_tau
from teaspoon.SP.network import ordinal_partition_graph
import math

import unittest


class PHN(unittest.TestCase):

    def test_point_summaries_empty_persistence_diagram(self):
        #generate a siple sinusoidal time series
        t = np.linspace(0,30,300)
        ts = np.sin(t) + np.sin(2*t)
        
        #Get appropriate dimension and delay parameters for permutations
        tau = int(MsPE_tau(ts))
        n = 2
        
        #create adjacency matrix, this
        A = ordinal_partition_graph(ts, n, tau)
        
        #create distance matrix and calculate persistence diagram
        D = DistanceMatrix(A, method = 'unweighted')
        diagram = PH_network(D)
        
        stats = point_summaries(diagram, A)

        self.assertTrue(math.isnan(stats[0]))
        self.assertTrue(math.isnan(stats[1]))
        self.assertTrue(math.isnan(stats[2]))
        
        
        
    def test_point_summaries_non_empty_persistence_diagram(self):
        #generate a siple sinusoidal time series
        t = np.linspace(0,30,300)
        ts = np.sin(t) + np.sin(2*t)
        
        #Get appropriate dimension and delay parameters for permutations
        tau = int(MsPE_tau(ts))
        n = 5
        
        #create adjacency matrix, this
        A = ordinal_partition_graph(ts, n, tau)
        
        #create distance matrix and calculate persistence diagram
        D = DistanceMatrix(A, method = 'unweighted')
        diagram = PH_network(D)
        
        stats = point_summaries(diagram, A)

        self.assertAlmostEqual(stats[0], 0.5, delta = 0.6)
        self.assertAlmostEqual(stats[1], 0.5, delta = 0.6)
        self.assertAlmostEqual(stats[2], 0.5, delta = 0.6)




if __name__ == '__main__':
    unittest.main()
