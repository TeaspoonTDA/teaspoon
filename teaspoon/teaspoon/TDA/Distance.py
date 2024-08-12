"""
This module provides algorithms to compute pairwise distances between persistence diagrams.  The bottleneck distance and wasserstein distance are available.

"""

import numpy as np
from typing import AnyStr
import ot
from sklearn.metrics import pairwise_distances
import persim

"""
.. module: Distance
"""


def wassersteinDist(
    pts0: np.ndarray,
    pts1: np.ndarray,
    p: int = 2,
    q: int = 2,
    y_axis: AnyStr = "death",
) -> float:
    """
    Compute the Persistant p-Wasserstein distance between the diagrams pts0, pts1 using optimal transport.

    Parameters
    ----------
    pts0: array of shape (n_top_features, 2)
        The first persistence diagram
    pts1: array of shape (n_top_features, 2)
        The second persistence diagram
    y_axis: optional, default="death"
        What the y-axis of the diagram represents. Should be one of

            * ``"lifetime"``
            * ``"death"``

    p: int, optional (default=2)
        The p in the p-Wasserstein distance to compute
    q: 1, 2 or np.inf, optional (default = 2)
        The q for the internal distance between the points, L_q.
        Uses L_infty (Chebyshev) distance if q = np.inf.
        Currently not implemented for other q.
    Returns
    -------
    distance: float
        The p-Wasserstein distance between diagrams ``pts0`` and ``pts1``
    """

    # Convert the diagram back to birth death coordinates if passed as birth, lifetime
    if y_axis == "lifetime":
        pts0[:, 1] = pts0[:, 0] + pts0[:, 1]
        pts1[:, 1] = pts1[:, 0] + pts1[:, 1]
    elif y_axis == 'death':
        pass
    else:
        raise ValueError("y_axis must be 'death' or 'lifetime'")

    # Check q. Eventually want to remove the q <=2 part.
    if type(q) == int and q >= 3:
        raise ValueError(
            "q (for the internal L_q) is currently only available for 1, 2, or np.inf")
    elif q == 1:
        # Distance to diagonal in L1 distance is just the lifetime
        extra_dist0 = (pts0[:, 1] - pts0[:, 0])
        extra_dist1 = (pts1[:, 1] - pts1[:, 0])
    elif (q >= 2):
        # Distance to diagonal in Lq distance
        # Closest point to (a,b) is at (x,x) for x = a + (b-a)/2
        extra_dist0 = (pts0[:, 1] - pts0[:, 0]) * 2**(1/q - 1)
        extra_dist1 = (pts1[:, 1] - pts1[:, 0]) * 2**(1/q - 1)
    elif q == np.inf:
        extra_dist0 = (pts0[:, 1] - pts0[:, 0]) / 2
        extra_dist1 = (pts1[:, 1] - pts1[:, 0]) / 2
    else:
        raise ValueError("q must 1, 2, or np.inf")

    # Get distances between all pairs of off-diagonal points
    # When we fix this for more q options,
    if q == np.infty:
        metric = 'chebyshev'
    elif q == 1:
        metric = 'l1'
    elif q == 2:
        metric = 'l2'

    pairwise_dist = pairwise_distances(pts0, pts1, metric=metric)

    # Add a row and column corresponding to the distance to the diagonal
    all_pairs_ground_distance_a = np.hstack(
        [pairwise_dist, extra_dist0[:, np.newaxis]])
    extra_row = np.zeros(all_pairs_ground_distance_a.shape[1])
    extra_row[: pairwise_dist.shape[1]] = extra_dist1
    all_pairs_ground_distance_a = np.vstack(
        [all_pairs_ground_distance_a, extra_row])

    # Raise all distances to the pth power
    all_pairs_ground_distance_a = all_pairs_ground_distance_a ** p

    # Build vector representing the mass at each location
    # For n0 points in the first diagram and n1 in the second,
    # the total mass for each diagram is n0+n1.
    # The mass for all off diagonal points are 1, and
    # remaining weight is placed on the diagonal.
    n0 = pts0.shape[0]
    n1 = pts1.shape[0]
    a = np.ones(n0 + 1)
    a[n0] = n1
    a = a / a.sum()
    b = np.ones(n1 + 1)
    b[n1] = n0
    b = b / b.sum()

    # Get the distance according to optimal transport
    otDist = ot.emd2(a, b, all_pairs_ground_distance_a)

    # Multiply by the total mass and raise to the pth power
    out = np.power((n0 + n1) * otDist, 1.0 / p)

    return out


def bottleneckDist(
    pts0: np.ndarray,
    pts1: np.ndarray,
    matching=True,
    plot=True
):
    """
    Compute the bottleneck distance between the diagrams pts0, pts1 using the persim package: https://persim.scikit-tda.org/en/latest/index.html

    Parameters
    ----------
    pts0: array of shape (n_top_features, 2)
        The first persistence diagram
    pts1: array of shape (n_top_features, 2)
        The second persistence diagram
    matching: boolean, True returns matched array
    plot: boolean, True provides plot of matching points.  Matching must be true for plot to return.

    Returns
    -------
    distance: float
        The bottleneck distance between diagrams ``pts0`` and ``pts1``
    """
    if matching == True and plot == True:
        d, matching = persim.bottleneck(pts0, pts1, matching=matching)
        persim.bottleneck_matching(pts0, pts1, matching)
        return d, matching
    if matching == True and plot == False:
        d, matching = persim.bottleneck(pts0, pts1, matching=matching)
        return d, matching
    if matching == False and plot == True:
        raise Exception("Matching must be 'True' to enable plotting'")
    else:
        d = persim.bottleneck(pts0, pts1, matching=False)
        return d
