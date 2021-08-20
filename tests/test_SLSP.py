import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from teaspoon.TDA.SLSP import Persistence0D
from teaspoon.TDA.SLSP_tools import cutoff

import unittest


class SubLevel(unittest.TestCase):

    def test_cutoff(self):
        #---Assume the additive noise distribution------------
        dist = 'Gaussian'

        #----Generate a time series with additive noise--------
        fs, T = 40, 12
        t = np.linspace(0,T,fs*T)
        A, sd = 20,1 #signal amplitude and standard deviation
        ts_0 = A*np.sin(np.pi*t) + A*np.sin(t)
        noise = np.random.normal(0,sd,len(ts_0)) #gaussian distribution additive noise
        ts = ts_0 + noise #add noise to time series

        #--------------Run sublevel set persistence--------------------------
        feature_ind_1, feature_ind_2, persistenceDgm = Persistence0D(ts)
        B = np.flip(persistenceDgm.T[0], axis = 0) #get birth times
        D = np.flip(persistenceDgm.T[1], axis = 0) #get death times
        L = D-B #get lifetimes as difference between birth and death

        I_B = np.array(feature_ind_1.astype(int)).T-1 #indices of birth times
        T_B = np.flip(t[I_B], axis = 0) #time values at birth times


        #-------------get cutoff for persistence diagram---------------------
        C, param = cutoff(L, alpha = 0.01, n = len(ts), distribution = dist) #get cutoff

        self.assertAlmostEqual(param, 1, delta = 1)
        self.assertAlmostEqual(C, 5, delta = 5)
        # print('Distribution parameter estimate: ', param)
        # print('C: ', C)



        # #-------------------------PLOTTING THE RESULTS-----------------------
        # gs = gridspec.GridSpec(2,3)
        # plt.figure(figsize=(17,5))
        # TextSize = 15
        #
        # ax = plt.subplot(gs[0, 0:2])
        # plt.plot(t,ts, 'k')
        # plt.ylabel('$x(t)$', size = TextSize)
        # plt.xticks(size = TextSize)
        # plt.yticks(size = TextSize)
        # plt.xlim(min(t),max(t))
        #
        # ax = plt.subplot(gs[1, 0:2])
        # plt.ylabel('$L$', size = TextSize)
        # plt.xlabel('$t$', size = TextSize)
        # plt.xticks(size = TextSize)
        # plt.yticks(size = TextSize)
        # plt.plot(T_B[L>C], L[L>C], 'bd', label = r'$L$ (signal)')
        # plt.plot(T_B[L<C], L[L<C], 'ko', alpha = 0.7, label = r'$L$ (noise)')
        # plt.plot([np.min(t),np.max(t)],[C, C],'k--', label = r'$C^*_\alpha$')
        # ax.fill_between([min(t), max(t)], [C, C], color = 'red', alpha = 0.15)
        # plt.ylim(0,)
        # plt.xlim(min(t),max(t))
        # plt.legend(loc = 'right', fontsize = TextSize-3, ncol = 2)
        #
        # ax = plt.subplot(gs[0:2, 2])
        # plt.ylabel('$D$', size = TextSize)
        # plt.xlabel('$B$', size = TextSize)
        # plt.xticks(size = TextSize)
        # plt.yticks(size = TextSize)
        # plt.plot(B[L>C], D[L>C], 'bd', label = r'signal')
        # plt.plot(B[L<C], D[L<C], 'ro', alpha = 0.7, label = r'noise')
        # plt.plot([min(B), max(D)], [min(B), max(D)],'k')
        # plt.plot([min(B), max(D)], [min(B)+C, max(D)+C],'k--', label = r'$C_\alpha$')
        # ax.fill_between(x = [min(B), max(D)], y1 = [min(B)+C, max(D)+C], y2 = [min(B), max(D)],
        #                                  color = 'red', alpha = 0.15)
        # plt.legend(loc = 'lower right', fontsize = TextSize-3, bbox_to_anchor = (1.02, -0.02))
        #
        # plt.subplots_adjust(wspace=0.3)
        # plt.show()



if __name__ == '__main__':
    unittest.main()
