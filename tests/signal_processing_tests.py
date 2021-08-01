# Tests done by Audun Myers as of 11/6/20 (Version 0.0.1)


import numpy as np
import unittest
from teaspoon.SP.tsa_tools import takens
from teaspoon.SP.tsa_tools import permutation_sequence
from teaspoon.SP.tsa_tools import k_NN
import matplotlib.pyplot as plt
from teaspoon.SP.network import knn_graph
from teaspoon.SP.network import ordinal_partition_graph
from teaspoon.SP.network_tools import make_network
from teaspoon.SP.information.entropy import MsPE
from teaspoon.SP.information.entropy import PE
from ripser import ripser

from teaspoon.SP.information.entropy import PersistentEntropy

class signalProcessing(unittest.TestCase):

    def test_takens(self):
    # In[ ]: takens

        t = np.linspace(0,30,200)
        ts = np.sin(t)  #generate a simple time series

        embedded_ts = takens(ts, n = 2, tau = 10)
        self.assertEqual(embedded_ts.shape[1], 2)

    # plt.plot(embedded_ts.T[0], embedded_ts.T[1], 'k.')
    # plt. show()


    def test_permutation(self):
        t = np.linspace(0,30,200)
        ts = np.sin(t)  #generate a simple time series

        PS = permutation_sequence(ts, n = 3, tau = 12)

        # import matplotlib.pyplot as plt
        # plt.plot(t[:len(PS)], PS, 'k')
        # plt. show()
        self.assertEqual(PS.max(), 6)



    # In[ ]: knn

    def test_knn(self):
        t = np.linspace(0,15,100)
        ts = np.sin(t)  #generate a simple time series

        embedded_ts = takens(ts, n = 2, tau = 10)

        distances, indices = k_NN(embedded_ts, k=4)

        print('need a test for this one ')


    # import matplotlib.pyplot as plt
    # plt.plot(embedded_ts.T[0], embedded_ts.T[1], 'k.')
    # i = 20 #choose arbitrary index to get NN of.
    # NN = indices[i][1:] #get nearest neighbors of point with that index.
    # plt.plot(embedded_ts.T[0][NN], embedded_ts.T[1][NN], 'rs') #plot NN
    # plt.plot(embedded_ts.T[0][i], embedded_ts.T[1][i], 'bd') #plot point of interest
    # plt. show()



    # In[ ]: make graph from time series

    def test_graphs(self):

        t = np.linspace(0,30,200)
        ts = np.sin(t) + np.sin(2*t) #generate a simple time series

        A_knn = knn_graph(ts) #ordinal partition network from time series

        A_op = ordinal_partition_graph(ts) #knn network from time series

        print('Needs a test ')

    #ANOTHER EXAMPLE

    def test_graphs_2(self):
        t = np.linspace(0,30,200)
        ts = np.sin(t) + np.sin(2*t) #generate a simple time series

        A = knn_graph(ts)

        G, pos = make_network(A)

        print('Needs a test ')


        # import matplotlib.pyplot as plt
        # import networkx as nx
        # plt.figure(figsize = (8,8))
        # plt.title('Network', size = 16)
        # nx.draw(G, pos, with_labels=False, font_weight='bold', node_color='blue',
        #         width=1, font_size = 10, node_size = 30)
        # plt.show()


    # In[ ]:  PE


    def test_PE(self):
        t = np.linspace(0,100,2000)
        ts = np.sin(t)  #generate a simple time series

        h = PE(ts, n = 6, tau = 15, normalize = True)
        print('Permutation entropy: ', h)


    # In[ ]:  MsPE

    def test_MsPE(self):
        t = np.linspace(0,100,2000)
        ts = np.sin(t)  #generate a simple time series

        delays,H = MsPE(ts, n = 6, delay_end = 40, normalize = True)

        # import matplotlib.pyplot as plt
        # plt.figure(2)
        # TextSize = 17
        # plt.figure(figsize=(8,3))
        # plt.plot(delays, H, marker = '.')
        # plt.xticks(size = TextSize)
        # plt.yticks(size = TextSize)
        # plt.ylabel(r'$h(3)$', size = TextSize)
        # plt.xlabel(r'$\tau$', size = TextSize)
        # plt.show()



    # In[ ]: persistent entropy
    def test_PersistentEntropy(self):
        #generate a simple time series with noise
        t = np.linspace(0,20,200)
        ts = np.sin(t) +np.random.normal(0,0.1,len(t))

        #embed the time series into 2 dimension space using takens embedding
        embedded_ts = takens(ts, n = 2, tau = 15)

        #calculate the rips filtration persistent homology
        result = ripser(embedded_ts, maxdim=1)
        diagram = result['dgms']

        #--------------------Plot embedding and persistence diagram---------------
        # import matplotlib.pyplot as plt
        # import matplotlib.gridspec as gridspec
        # gs = gridspec.GridSpec(1, 2)
        # plt.figure(figsize = (12,5))
        # TextSize = 17
        # MS = 4
        #
        # ax = plt.subplot(gs[0, 0])
        # plt.yticks( size = TextSize)
        # plt.xticks(size = TextSize)
        # plt.xlabel(r'$x(t)$', size = TextSize)
        # plt.ylabel(r'$x(t+\tau)$', size = TextSize)
        # plt.plot(embedded_ts.T[0], embedded_ts.T[1], 'k.')
        #
        # ax = plt.subplot(gs[0, 1])
        # top = max(diagram[1].T[1])
        # plt.plot([0,top*1.25],[0,top*1.25],'k--')
        # plt.yticks( size = TextSize)
        # plt.xticks(size = TextSize)
        # plt.xlabel('Birth', size = TextSize)
        # plt.ylabel('Death', size = TextSize)
        # plt.plot(diagram[1].T[0],diagram[1].T[1] ,'go', markersize = MS+2)
        #
        # plt.show()
        #-------------------------------------------------------------------------

        #get lifetimes (L) as difference between birth (B) and death (D) times
        B, D = diagram[1].T[0], diagram[1].T[1]
        L = D - B

        h = PersistentEntropy(lifetimes = L)

        print('Persistent entropy: ', h)


if __name__ == '__main__':
    unittest.main()
