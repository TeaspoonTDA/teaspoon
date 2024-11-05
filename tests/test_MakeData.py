# Tests done by Audun Myers as of 11/6/20 (Version 0.0.1)


# In[ ]: Dynamic Systems Library

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import unittest
from teaspoon.MakeData.DynSysLib.autonomous_dissipative_flows import rossler

class TestRossler(unittest.TestCase):

    def test_Rossler(self):

        dynamic_state = 'periodic'
        t, ts = rossler(dynamic_state=dynamic_state)

        TextSize = 15
        plt.figure(figsize = (12,4))
        gs = gridspec.GridSpec(1,2)

        ax = plt.subplot(gs[0, 0])
        plt.xticks(size = TextSize)
        plt.yticks(size = TextSize)
        plt.ylabel(r'$x(t)$', size = TextSize)
        plt.xlabel(r'$t$', size = TextSize)
        plt.plot(t,ts[0], 'k')

        ax = plt.subplot(gs[0, 1])
        plt.plot(ts[0], ts[1],'k.')
        plt.plot(ts[0], ts[1],'k', alpha = 0.25)
        plt.xticks(size = TextSize)
        plt.yticks(size = TextSize)
        plt.xlabel(r'$x(t)$', size = TextSize)
        plt.ylabel(r'$y(t)$', size = TextSize)

        # plt.show()
        print('Ran Rossler in MakeData_tests')



if __name__ == '__main__':
    unittest.main()
