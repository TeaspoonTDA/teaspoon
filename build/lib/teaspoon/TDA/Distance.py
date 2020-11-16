"""
This module provides algorithms to compute pairwise distances between persistence diagrams.

"""

import numpy as np
import os
import subprocess
# import TSAwithTDA.pyPerseus as pyPerseus
# import TSAwithTDA.SlidingWindows as SW
from .Persistence import  prepareFolders

"""
.. module: Distance
"""

#-----------------------------------------------------------------------



## Computes the distance using the geom_bottleneck and geom_matching code
# from [Hera](https://bitbucket.org/grey_narn/hera).
# '''
# .. todo::

#     This documentation needs cleaning and updating.

#     Need to build the C++ code, then add bottleneck_dist folder and wasserstein_dist to path

#     On Liz's machine, this involves adding the following lines to the ~/.bashrc
#         export PATH="/home/liz/Dropbox/Math/Code/hera/geom_matching/wasserstein/build:$PATH"
#         export PATH="/home/liz/Dropbox/Math/Code/hera/geom_bottleneck/build/example:$PATH"
#     Default values for wasserstein_dist from Hera:

#         internal_p must be in :math:`[1.0, \infty]` (to explicitly set internal_p to :math:`\infty`, supply inf).By default wasserstein degree is 1.0, relative error is 0.01, internal norm is l_infinity.

    # hera input format:

    #     ::

    #         bottleneck_dist file1 file2  [relative_error]
    #         wasserstein_dist file1 file2  [wasserstein degree] [relative error] [internal norm]
# '''

def dgmDist_Hera(D1,D2, wassDeg = 'Bottleneck', relError = None, internal_p = None):
    """
    .. note::
        Hera must be installed and in the bash path to use this function https://bitbucket.org/grey_narn/hera

    Hera input format:
            - bottleneck_dist file1 file2  [relative_error]
            - wasserstein_dist file1 file2  [wasserstein degree] [relative error] [internal norm]  where file1 and file2 represent the persistence diagrams.
    :param ndarray (D1):
            Persistence diagram --(Nx2) matrix.
    :param ndarray (D2):
            Persistence diagram --(Nx2) matrix.
    :param wassDeg: Options are:
            - 'Bottleneck' or anything containing 'bot'. Runs the bottleneck_dist command
            - np.inf: Also returns bottleneck distance with the bottleneck_dist command
            - an integer q: Computes the q-th Wasserstein distance
    :param relError:
        - An input to both bottleneck_dist and wasserstein_dist.
        - For wasserstein_dist from hera documentation:
            If two diagrams are equal, then the exact distance 0.0 is printed (the order of points in file1 and file2 need not be the same).
            Otherwise the output is an approximation of the exact distance. Precisely:
            if :math:`d_{exact}` is the true distance and d_approx is the output, then
                :math:`\\frac{| d_{exact} - d_{approx} |}{ d_{exact} } < \\mathrm{relativeError}`.
        - For bottleneck_dist from hera documentation:
            If two diagrams are equal, then the exact distance 0.0 is printed (the order of points in file1 and file2 need not be the same).
            Otherwise the output is an approximation of the exact distance.
            Precisely: if :math:`d_{exact}` is the true distance and d_approx is the output, then
                :math:`\\frac{| d_{exact} - d_{approx} |}{ d_{exact} } < \\mathrm{relativeError}.`
        - Default value in Hera is 0.01 for Wasserstein distance and 0 (exact computation) for bottleneck distance.
        - Values passed to hera must be positive.
        .. todo:: Does this mean strictly positive? What is the behavior when passing 0?
    :param internal_p:
        This is only controllable for Wasserstein distance computation.  Matched points are measured by :math:`L_p` distance.
        Default is internal_p = infinity.

    .. todo:: Check that bottleneck_dist uses infinity even though it doesn't explicitly say so in the documentation.

    :returns: A float for the distance between the diagrams

    """
    # Make and check folder system.
    # Warning: Empties the directory every time! Don't put anything in the
    #          created .teaspoonData folder that you don't want to lose.
    prepareFolders()

    inputFile1 = '.teaspoonData/input/Dgm1.txt'
    inputFile2 = '.teaspoonData/input/Dgm2.txt'

    # Save the diagrams as text files
    np.savetxt(inputFile1, D1, fmt = '%f')
    np.savetxt(inputFile2, D2, fmt = '%f')

    # Check proper inputs
    if  not isinstance(wassDeg, (int, float)) and wassDeg != np.inf:
        if type(wassDeg) == str and 'bot' not in wassDeg.lower():
            print('Options for wassDeg are:\n-"bottleneck"\n-np.inf\n-A number in [1,\infty).')
            print('You passed something funky.  Exiting...')
            return


    # Spell out the command for bottleneck distance.
    if (type(wassDeg) == str and 'bot' in wassDeg.lower()) or wassDeg == np.inf:
        cmd = 'bottleneck_dist ' + inputFile1 + ' ' + inputFile2

        # Attach the relError to the command if it's supplied
        if relError:
            if relError <0:
                print('relError must be positive...')
                print('Exiting...')
                return
            else:
                cmd += ' ' + str(relError)

        # We're ignoring internal_p if we're doing bottleneck distance.
        if internal_p:
            print('Note: You passed an option for the internal_p.  This option')
            print('is not available in bottleneck distance in hera, so it is ')
            print('being ignored...')


    # Spell out the command for wasserstein distance.
    elif wassDeg >= 1:
        cmd = 'wasserstein_dist ' + inputFile1 + ' ' + inputFile2
        cmd += ' ' + str(wassDeg)
        # If either relError of internal_p are set, add both
        # to the command.
        if relError or internal_p:
            if relError:
                cmd += ' ' + str(relError)
            else:
                cmd += ' ' + '0.01'# default value

            if internal_p:
                cmd += ' ' + str(internal_p)
            else:
                cmd += ' ' + 'inf' # default value



    # Exit if wassDeg isn't something understandable.
    else:
        print('For computing the Wasserstein distance, wassDeg must be in [1,\infty)...')
        print('Exiting...')
        return



    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell = True)
    (out,err) = proc.communicate()
    out = out.decode().split('\n')
    distance = out[-2]
    distance = float(distance)


    return distance
