# Tests done by Audun Myers as of 11/6/20 (Version 0.0.1)


# In[ ]: Dynamic Systems Library

import teaspoon.MakeData.DynSysLib.DynSysLib as DSL
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec





import unittest


class TestRossler(unittest.TestCase):

    def test_Rossler(self):


        system = 'rossler'
        dynamic_state = 'periodic'
        t, ts = DSL.DynamicSystems(system, dynamic_state)

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

    def test_Rossler2(self):



        system = 'rossler'
        UserGuide = True
        L, fs, SampleSize = 1000, 20, 2000
        dynamic_state = 5
        # the length (in seconds) of the time series, the sample rate, and the sample size of the time series of the simulated system.
        parameters = [0.1, 0.2, 13.0] # these are the a, b, and c parameters from the Rossler system model.
        InitialConditions = [1.0, 0.0, 0.0] # [x_0, y_0, x_0]
        t, ts = DSL.DynamicSystems(system, dynamic_state, L, fs, SampleSize, parameters,  InitialConditions, UserGuide)

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
        print('Ran Rossler2 in MakeData_tests')



if __name__ == '__main__':
    unittest.main()
