"""
This module provides algorithms related to computing the magnitude. See paper below for additional details 

- Miguel O'Malley, Sara Kalisnik, Nina Otter. Alpha magnitude. Journal of Pure and Applied Algebra, Volume 227, Issue 11,2023, doi: `10.1016/j.jpaa.2023.107396 <https://doi.org/10.1016/j.jpaa.2023.107396>`_.
"""

import numpy as np
from numpy.linalg import inv

"""
.. module: Magnitude
"""


def MagnitudeFunction(D, t_range=[.1, 10], t_n=100):
    """
    This function calculates the magnitude function,  ``t -> |tX|``,  of an input distance matrix assumed to be calculated from a finite metric space ``X``, on the interval defined by t_range at t_n locations.

    - Given a finite metric space ``(X,d)`` and for matrix purposes, fix an order on the points ``x_1,\\cdots,x_n``.
    - Denote the distance matrix by ``D=[d(x_i,x_j)]_{ij}=[D_{ij}]_{ij}``
    - Denote the similarity matrix ``Z=Z_X`` to have entries ``Z_{ij}=e^{-D_{ij}}``
    - We'll also be interested in the scaled version for some ``t \\in (0,\\infty)``, where ``tZ`` is the matrix for metric space ``tX`` and ``tZ_{ij}=e^{-tD_{ij}}``
    - The magnitude, in particular of ``|tX|`` is ``|tX| = sum_{i,j} ((tZ)^{-1})_{ij}`` where ``(tZ)^{-1}`` is the inverse of the matrix ``tZ``, assuming it exists. 
    - The magnitude function is ``M: t -> |tX|``

    Args:
        D (2-D array): 2-D square distance matrix
        t_range (length 2 list): 

    Returns:
        T,M [1-D arrays]: T is the list of ``t`` values, ``M`` is the list of associated values ``|tX|``.
    """

    # Get the list of locations for which we'll calculate the magintude.
    T = np.linspace(t_range[0], t_range[1], t_n)
    M = []
    for t in T:
        tZ = np.exp(-t*D)
        tZinv = inv(tZ)
        m = np.sum(tZinv)
        M.append(m)
    return T, M


# Only runs if running from this file (This will show basic example)
if __name__ == "__main__":
    from teaspoon.MakeData.PointCloud import Annulus
    from scipy.spatial import distance_matrix
    import matplotlib.pyplot as plt
    P = Annulus()
    D = distance_matrix(P, P)
    T, M = MagnitudeFunction(D)
    plt.plot(T, M)
    plt.show()
