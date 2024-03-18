import matplotlib.pyplot as plt
import numpy as np

from teaspoon.TDA.Draw import drawDgm
from teaspoon.TDA.Persistence import BettiCurve, CROCKER, CROCKER_Stack

alpha = 0.6

dgm = np.array([[0.5, 0.75], [1, 1.5], [1.25, 2.5], [2, np.inf]])
t, b = BettiCurve(dgm, maxEps=2, numStops=10, alpha=alpha)

drawDgm(dgm)
plt.plot([0, 4], [alpha, 4+alpha])

for dg_ in dgm:
    if dg_[0] < alpha or dg_[1] < alpha:
        continue
    plt.plot([dg_[0], dg_[0]], [dg_[1], dg_[0]+alpha], '--', color="silver") # x
    plt.plot([dg_[0], dg_[1]-alpha], [dg_[1], dg_[1]], '--', color="silver")
    # x

for idx in range(len(t)-1):
    start = t[idx] - alpha
    start = 0.0 if start < 0.0 else start
    end = t[idx+1] + alpha

    plt.plot(
        [0+idx*0.05, 0+idx*0.05],
             [start, end])

plt.show()

#%%
import numpy as np
from teaspoon.TDA.Persistence import CROCKER, CROCKER_Stack

dgms = [np.array([[0.5, 0.75], [1, 1.5], [1.25, 2.5], [2, np.inf]]),
        np.array([[0.5, 1], [1, 2], [1.25, 3], [2, np.inf]])]

M = CROCKER(dgms, maxEps=4, numStops=10, plotting=False)

M2 = CROCKER_Stack(dgms, maxEps=4, numStops=10, alpha=[0, 0.1, 0.25, 0.5],
                   plotting=True)
assert (M == M2[0]).all()
#%%