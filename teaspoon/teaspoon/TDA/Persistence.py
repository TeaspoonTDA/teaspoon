# @package teaspoon.TDA.Persistence
import re
import warnings
import glob
from scipy.spatial.distance import pdist, squareform
from subprocess import DEVNULL, STDOUT, call
import subprocess
import os
import numpy as np
"""
This module includes wrappers for using various fast persistence software inside of python.
All diagrams are stored as a 2xN numpy matrix.
When a code returns multiple dimensions, these are returned as a dictionary

::

    {
        0: DgmDimension0,
        1: DgmDimension1,
        ...
    }


Infinite classes are given an entry of np.inf.
During computation, all data files are saved in a hidden folder ".teaspoonData".
This folder is created if it doesn't already exist.
Files are repeatedly emptied out of it, so do not save anything in it that you might want later!

"""

"""
.. module: Persistence
"""


#-----------------------------------------------------#
#-----------------------------------------------------#
#------------------Helper code------------------------#
#-----------------------------------------------------#
#-----------------------------------------------------#


def prepareFolders():
    """
    Generates the ".teaspoonData" folder.
    Checks that necessary folder structure system exists.
    Empties out all previously saved files to avoid confusion.
    """
    # ---- Make folders for saving files

    folders = ['.teaspoonData',
               '.teaspoonData' + os.path.sep + 'input',
               '.teaspoonData'+os.path.sep + 'output']
    for location in folders:
        if not os.path.exists(location):
            os.makedirs(location)

# ----------------------------------------------
# ----------------------------------------------
# ---Simple operations on pers dgms-------------
# ----------------------------------------------
# ----------------------------------------------

def minPers(Dgm):
    """
    Finds minimum persistence for a given diagram

    :param Dgm:
        A 2D numpy array

    :returns: 
        (float) Minimum persistence for the given diagram

    """
    try:
        lifetimes = Dgm[:, 1] - Dgm[:, 0]
        return min(lifetimes)
    except:
        return 0


def maxPers(Dgm):
    """
    Finds maximum persistence for a given diagram

    :param Dgm:
        A 2D numpy array

    :returns: 
        (float) Maximum persistence for the given diagram 

    """
    try:
        lifetimes = Dgm[:, 1] - Dgm[:, 0]
        m = max(lifetimes)
        if m == np.inf:
            # Get rid of rows with death time infinity
            numRows = Dgm.shape[0]
            rowsWithoutInf = list(set(np.where(Dgm[:, 1] != np.inf)[0]))
            m = max(lifetimes[rowsWithoutInf])
        return m
    except:
        return 0


def maxBirth(Dgm):
    """
    Finds maximum birth for a given diagram

    :param Dgm:
        A 2D numpy array

    :returns: (float) Maximum birth time for the given diagram 

    """
    try:
        m = max(Dgm[:, 0])
        if m == np.inf:
            # Get rid of rows with death time infinity
            numRows = Dgm.shape[0]
            rowsWithoutInf = list(set(np.where(Dgm[:, 1] != np.inf)[0]))
            m = max(Dgm[rowsWithoutInf, 0])

        return m
    except:
        return 0


def minBirth(Dgm):
    """
    Finds minimum birth  for a given diagram

    :param Dgm:
        A 2D numpy array

    :returns: 
        (float) Minimum birth time for the given diagram 

    """
    try:
        m = min(Dgm[:, 0])
        return m
    except:
        return 0

# \brief Gets minimum persistence for a pandas.Series with diagrams as entries
#
# @param DgmSeries
#     a pandas.Series with entries Kx2 numpy arrays representing persistence diagrams.
#
# @return float


def minPersistenceSeries(DgmsSeries):
    '''
    Takes data frame DgmsDF.
    Finds minimum persistence over all diagrams in
    column with label dgm_col.
    Gets minimum persistence for a pandas.Series with diagrams as entries

    :param DgmSeries:
        A pandas.Series with entries Kx2 numpy arrays representing persistence diagrams.

    :returns: 
        (float) Minimum persistence over all diagrams

    '''
    return min(DgmsSeries.apply(minPers))


def maxPersistenceSeries(DgmsSeries):
    '''
    Takes data frame DgmsDF.
    Finds maximum persistence over all diagrams in
    column with label dgm_col.
    Gets maximum persistence for a pandas.Series with diagrams as entries

    :param DgmsSeries:
        A pandas.Series with entries Kx2 numpy arrays representing persistence diagrams.

    :returns: 
        (float) Maximum persistence over all diagrams

    '''
    return max(DgmsSeries.apply(maxPers))


def minBirthSeries(DgmsSeries):
    '''
    Takes data frame DgmsDF.
    Finds minimum persistence over all diagrams in
    column with label dgm_col.
    Gets minimum persistence for a pandas.Series with diagrams as entries

    :param DgmSeries:
        A pandas.Series with entries Kx2 numpy arrays representing persistence diagrams.

    :returns: 
        (float) Minimum birth time over all diagrams

    '''
    return min(DgmsSeries.apply(minBirth))


def maxBirthSeries(DgmsSeries):
    '''
    Takes data frame DgmsDF.
    Finds maximum persistence over all diagrams in
    column with label dgm_col.
    It gets maximum persistence for a pandas.Series with diagrams as entries.

    :param DgmSeries:
        A pandas.Series with entries Kx2 numpy arrays representing persistence diagrams.

    :returns: 
        (float) Maximum persistence over all diagrams

    '''
    return max(DgmsSeries.apply(maxBirth))


def removeInfiniteClasses(Dgm):
    '''
    Simply deletes classes that have infinite lifetimes.

    '''
    keepRows = np.isfinite(Dgm[:, 1])
    return Dgm[keepRows, :]
