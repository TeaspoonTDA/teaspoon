# Tests done by Audun Myers as of 11/5/20 (Version 0.0.1)


# In[ ]: Persistent homology of networks

#import needed packages
import numpy as np
from teaspoon.SP.network import ordinal_partition_graph
from teaspoon.TDA.PHN import PH_network
from teaspoon.SP.network_tools import make_network
from teaspoon.parameter_selection.MsPE import MsPE_tau

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx

import unittest


class OrdinalPartition(unittest.TestCase):

    def test_ordinalPartition(self):


        #generate a siple sinusoidal time series
        t = np.linspace(0,30,300)
        ts = np.sin(t) + np.sin(2*t)

        #Get appropriate dimension and delay parameters for permutations
        tau = int(MsPE_tau(ts))
        n = 5

        #create adjacency matrix, this
        A = ordinal_partition_graph(ts, n, tau)

        #get networkx representation of network for plotting
        G, pos = make_network(A, position_iterations = 1000, remove_deg_zero_nodes = True)

        #create distance matrix and calculate persistence diagram
        D, diagram = PH_network(A, method = 'unweighted', distance = 'shortest_path')

        self.assertAlmostEqual(A.min(), 0)



        # TextSize = 14
        # plt.figure(2)
        # plt.figure(figsize=(8,8))
        # gs = gridspec.GridSpec(4, 2)
        #
        # ax = plt.subplot(gs[0:2, 0:2]) #plot time series
        # plt.title('Time Series', size = TextSize)
        # plt.plot(ts, 'k')
        # plt.xticks(size = TextSize)
        # plt.yticks(size = TextSize)
        # plt.xlabel('$t$', size = TextSize)
        # plt.ylabel('$x(t)$', size = TextSize)
        # plt.xlim(0,len(ts))
        #
        # ax = plt.subplot(gs[2:4, 0])
        # plt.title('Network', size = TextSize)
        # nx.draw(G, pos, with_labels=False, font_weight='bold', node_color='blue',
        #         width=1, font_size = 10, node_size = 30)
        #
        # ax = plt.subplot(gs[2:4, 1])
        # plt.title('Persistence Diagram', size = TextSize)
        # MS = 3
        # top = max(diagram[1].T[1])
        # plt.plot([0,top*1.25],[0,top*1.25],'k--')
        # plt.yticks( size = TextSize)
        # plt.xticks(size = TextSize)
        # plt.xlabel('Birth', size = TextSize)
        # plt.ylabel('Death', size = TextSize)
        # plt.plot(diagram[1].T[0],diagram[1].T[1] ,'go', markersize = MS+2)
        # plt.xlim(0,top*1.25)
        # plt.ylim(0,top*1.25)
        #
        # plt.subplots_adjust(hspace= 0.8)
        # plt.subplots_adjust(wspace= 0.35)
        # plt.show()


if __name__ == '__main__':
    unittest.main()
