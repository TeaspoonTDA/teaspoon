## @package teaspoon.TDA.Persistence
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

import numpy as np
import os
import subprocess
from subprocess import DEVNULL, STDOUT, call
from scipy.spatial.distance import pdist, squareform
import glob
import warnings
import re


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
    #---- Make folders for saving files

    folders = ['.teaspoonData',
			   '.teaspoonData'+ os.path.sep + 'input',
			   '.teaspoonData'+os.path.sep + 'output']
    for location in folders:
        if not os.path.exists(location):
            os.makedirs(location)


def readPerseusOutput(outputFileName):
    """
    Reads in the diagrams in the format that Perseus uses.
    Returns a dictionary with integer keys representing the dimension of each diagram.
    """
    outputFiles = glob.glob(outputFileName + '*')

    # Delete the file with the betti numbers
    outputFiles = [f for f in outputFiles if 'betti' not in f]

    # Make a dictionary of the persistence diagrams
    Dgms = {}

    for f in outputFiles:

        # Remove the dimension from the file name
        # This works because the file is of the form
        # outputFileName_1.txt
        dim = int ( f[len(outputFileName) + 1:-4] )

        # Read in the diagram
        # warning wrapper is there to ignore the empty file warnings
        # However, this actually suppresses all warnings, for better or worse....
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Dgm = np.loadtxt(f)

        # Add the diagram to the dictionary
        Dgms[dim] = Dgm

    return Dgms


def readRipserString(s):
    """
    Reads string from Ripser
    """

    birth_death_str = re.findall(r"(\d*[.]?\d*)", s)

    # stuff = list(filter(None, birth_death_str))
    stuff = [x for x in birth_death_str if x is not None]

    if not stuff:
#        print("empty stuff: {}". format(stuff))
        return np.nan
    else:
#        print("full stuff: {}".format(stuff))
        return float(stuff[0])


def readRipserOutput(out, drop_inf_class=True):
    """
    Reads output from Ripser
    """
    # split into persistence diagrams
    Dgms = {}

    # Find locations where the text splits the output
    breaks = [i for i, s in enumerate(out) if 'persistence' in s]

    for j in range(len(breaks)):
        # Get the dimension using regex
        dim = int(re.search('\d+', out[breaks[j]]).group(0))

        # Extract the portions of the output between
        # the places that say persistence.
        # Note the len(out)-1 for the endIndex is because
        # the last entry of out is a blank space ' '
        #		startIndex = breaks[j]
        if j+1 == len(breaks):
            endIndex = len(out)-1
        else:
            endIndex = breaks[j+1]

        Dgm = out[breaks[j]+1 : endIndex]
        print(Dgm)
        Dgm = [X.strip()[2:-1].split(',') for X in Dgm]

        # use regular expressions to extract the birth/death times
        Dgm = [[readRipserString(X) for X in row]   for row in Dgm]

        # get rid of spurious dimensions, and reshape into D x 2
        Dgm = np.squeeze(Dgm).reshape((-1, 2))

        # check that the diagram is not empty
        if Dgm.size > 0:
            # for 0-dim persistence, set birth time to zero
            if dim is 0:
                Dgm[:,0] = 0

            # if the last entry is nan it signals an infinite class,set to inf
            if np.isnan(Dgm[-1, 1]):
                Dgm[-1, 1] = np.inf

            # remove the row with infinite classes, if requested
            if drop_inf_class and np.isinf(Dgm[-1, 1]):
                Dgm = np.delete(Dgm, -1, 0)

        # add the diagram to the dictionary
        Dgms[dim] = Dgm
    return Dgms

#-----------------------------------------------------#
#-----------------------------------------------------#
#-------------------Point cloud input-----------------#
#-----------------------------------------------------#
#-----------------------------------------------------#


#------------Ripser-----------------------------------#

def VR_Ripser(P, maxDim = 1):
    """
    Computes persistence up to dimension maxDim using Uli Bauer's Ripser.

    .. note::
        Ripser needs to be installed on machine in advance https://github.com/Ripser/ripser

    :param P:
        A point cloud as an NxD numpy array.
        N is the number of points, D is the dimension of
        Euclidean space.
    :param maxDim:
        An integer representing the maximum dimension
        for computing persistent homology

    :returns:

        A dictionary Dgms where Dgms[k] is a lx2 matrix
        giving points in the k-dimensional pers diagram

    """


    # Compute pairwise distance matrix, then use the other
    # Ripser input style to do the work.
    X = squareform(pdist(P))

    Dgms = distMat_Ripser(X,maxDim = maxDim)

    return Dgms


#------------Perseus----------------------------------#
def writePointCloudFileForPerseus(P,filename,
                                stepSize = .1,
                                numSteps = 500):
    # Writes the point cloud to a file in the perseus format.

    # Notes:
    #     Vidit has options for radius scaling factor. TODO: Figure out if this should be 1 or 2

    # Parameters
    # ----------
    # P
    #     An NxD array.  Represents $N$ points in $R^D$.
    # filename
    #     location for saving the file
    # stepSize
    # numSteps
    #     Perseus requires that you decide how many steps, and how wide they are, rather than computing all possible topological changes.  So, persistence will be calculated from parameter 0 until stepSize*numSteps.
    
    # .. note::
    #     Vidit has options for radius scaling factor.
    
    """
    Writes the point cloud to a file in the perseus format.
    
    .. todo:: Figure out if this should be 1 or 2

    :param P:
        An NxD array.  Represents :math:`N` points in :math:`\mathbb{R}^{D}`
    :param filename:
        location for saving the file
    :param stepSize:
    :param numSteps:
        Perseus requires that you decide how many steps, and how wide they are, rather than computing all possible topological changes.
        So, persistence will be calculated from parameter 0 until stepSize*numSteps.

    """

    dimension = np.shape(P)[1]
    radiusScalingFactor = 1

    file = open(filename,'w')
    file.write(str(dimension) + '\n')
    secondLine = str(radiusScalingFactor) + ' '
    secondLine += str(stepSize) + ' '
    secondLine += str(numSteps) + '\n'
    file.write(secondLine)
    for i in range(np.shape(P)[0]):
        string = ' '.join(str(x) for x in P[i,:]) + ' 0 ' + '\n'
        file.write(string)


## Does brips version of perseus.
# Computes VR persitsence on points in Euclidean space.
#
# @remark
# 1) Requires choice of maxRadius, numSteps, and/or stepSize.
#    Bad choices will give junk results.
# 2) TODO: This appears to spit out radius rather than diameter
#    persistence computations.  Need to figure this out and
#    make the choice uniform across outputs.
#
#
# @param P
#     An NxD array.  Represents N points in R^D.
# @param maxRadius
# @param stepSize
# @param numSteps
#     Perseus requires that you decide how many steps, and how wide they are, rather than computing all possible topological changes.  So, persistence will be calculated from parameter 0 until
#     maxRadius = stepSize*numSteps.
#     Only 2 of the three entries should be passed.
#     If numSteps and stepSize are passed (regardless of whether maxRadius is passed), they will be used for the computation.  Otherwise, the two non-none valued entries will be used to calculate the third.
#
# @param suppressOutput
#     If true, gets rid of printed output from perseus.
#
# @return
#     A dictionary with integer keys 0,1,...,N
#    The key gives the dimension of the persistence diagram.
def VR_Perseus(P,dim = 1,
            maxRadius = 3, numSteps = 100, stepSize = None,
            suppressOutput = True):
    """
    Does brips version of perseus.
    Computes VR persitsence on points in Euclidean space.

    .. note::

        Requires choice of maxRadius, numSteps, and/or stepSize.
        Bad choices will give junk results.

    .. todo::

        This appears to spit out radius rather than diameter
        persistence computations.  Need to figure this out and
        make the choice uniform across outputs.


    :param P:
        An NxD array.  Represents N points in :math:`\mathbb{R}^{D}`.
    :param maxRadius:
    :param stepSize:
    :param numSteps:
        Perseus requires that you decide how many steps, and how wide they are, rather than computing all possible topological changes.  So, persistence will be calculated from parameter 0 until
        maxRadius = stepSize*numSteps.
        Only 2 of the three entries should be passed.
        If numSteps and stepSize are passed (regardless of whether maxRadius is passed), they will be used for the computation.  Otherwise, the two non-none valued entries will be used to calculate the third.
    :param suppressOutput:
        If true, it gets rid of printed output from perseus.

    :returns:

        A dictionary with integer keys 0,1,...,N

        The key gives the dimension of the persistence diagram.

    """

    #---- Clean up and/or create local folder system
    prepareFolders()

    inputFileName = '.teaspoonData/input/inputMatrix.txt'
    outputFileName = '.teaspoonData/output/perseusMatrix'


    # Determine stepSize given the maxRadius and the number of steps desired.
    # maxRadius = stepSize*numSteps.
    if stepSize == None:
        if numSteps == None:
            print('You need to pass at least two of the three entries:')
            print('maxRadius = stepSize*numSteps...')
        else:
            stepSize = maxRadius/float(numSteps)
    elif numSteps == None:
        if stepSize == None:
            print('You need to pass at least two of the three entries:')
            print('maxRadius = stepSize*numSteps...')
        else:
            numSteps = maxRadius/float(stepSize)

    # Write the necessary file
    print('using stepsize', stepSize)
    print('Using numSteps', numSteps)
    writePointCloudFileForPerseus(P,inputFileName, stepSize, numSteps)


    # Run perseus
    if suppressOutput:
        stdout = DEVNULL
    else:
        stdout = None

    try:
        command = 'perseus brips ' + inputFileName + ' ' + outputFileName
        call(command,
            stdout=stdout,
            stderr=STDOUT,
            shell=True)
    except:
        print('There appears to be a problem running perseus...')
        print('Do you have it properly installed?')
        print('Exiting...')
        return


    # Read in the outputs
    Dgms = readPerseusOutput(outputFileName)

    # Convert diagrams back to distance:
    #   Perseus spits out birth/death times in terms of the
    #   number of the step at which an event happened.
    #   Note step 1  is when the radius is  0

    for key in Dgms.keys():
        Dgm = Dgms[key]

        # Change any -1 death times to infinity
        infLocs = np.where(Dgm<0)
        Dgm[infLocs] = np.inf

        # Subtract 1 to deal with the the counting from 1 issue
        Dgm -= 1
        Dgm = Dgm*stepSize



        Dgms[key] = Dgm


    return Dgms


#-----------------------------------------------------#
#-----------------------------------------------------#
#---------------Distance Matrix input-----------------#
#-----------------------------------------------------#
#-----------------------------------------------------#


#-------------Ripser----------------------------------#


## \brief Computes persistence up to maxDim using Uli Bauer's [Ripser](https://github.com/Ripser/ripser).
#
# \remark Ripser needs to be installed on machine in advance. This code doesn't check for it's existence.
#
# @param distMat -
#     A distance matrix given as a NxN numpy array
# @param maxDim -
#     An integer representing the maximum dimension
#     for computing persistent homology
#
# @return
#     A dictionary Dgms where Dgms[k] is a lx2 matrix
#     giving points in the k-dimensional pers diagram
def distMat_Ripser(distMat, maxDim = 1):
    """
    Computes persistence up to maxDim using Uli Bauer's `Ripser <https://github.com/Ripser/ripser>`_.

    .. note:: Ripser needs to be installed on machine in advance. This code doesn't check for it's existence.

    :param distMat:
        A distance matrix given as a NxN numpy array
    :param maxDim:
        An integer representing the maximum dimension
        for computing persistent homology

    :returns:

        A dictionary Dgms where Dgms[k] is a lx2 matrix
        giving points in the k-dimensional pers diagram

    """
	# Check/setup the folder structure
	# Note: empties the folders in preparation
    prepareFolders()

    current_path = os.getcwd()
    base_dir = r'.teaspoonData'
    filename = r'input{0}pointCloud.txt'.format(os.path.sep)

#	inputFileName = tempfile.TemporaryFile(dir=base_dir).name



    inputFileName = os.path.join(current_path, base_dir, filename)

	#     = '.teaspoonData/input/pointCloud.txt'
	# outputFileName = '.teaspoonData/output/perseusMatrix'
	# Note: Uli's readme says that the lower-triangluar distance
	#       matrix is preferred. TODO: try writing just the point
	#       cloud to the file and let his faster code do the
	#       heavy lifting to see if it goes faster.

	# Compute pairwise distance matrix
	# X = squareform(pdist(P))

	# Open file to write the point cloud info
    with open(inputFileName,'w') as F:
		# Write lower triang matrix to file
        for i in range(np.shape(distMat)[0]):
			#Get part of row before diagonal
            L = str(list(distMat[i,:i]))[1:-1] #[1:-1] cuts off brackets
            F.write(L + '\n')

    cmd = 'ripser --dim ' + str(maxDim) + ' ' + inputFileName

    # Run ripser and pull the output from the terminal.
    proc = subprocess.Popen(cmd,stdout=subprocess.PIPE,shell = True)
    (out,err) = proc.communicate()
    out = out.decode().split('\n')

    # Parse the output
    Dgms = readRipserOutput(out)

    return Dgms

#---------------Perseus-------------------------------#

## @todo
def distMat_Perseus():
    """
    Not yet implemented
    """
    print('Sorry, not yet implemented.  Try distMat_Ripser instead!')

#-----------------------------------------------------#
#-----------------------------------------------------#
#---------------Image matrix input--------------------#
#-----------------------------------------------------#
#-----------------------------------------------------#


#---------------Perseus-------------------------------#


def writeMatrixFileForPerseus(M,filesavename):
    """
    Given 2D matrix M, write into file format read by Perseus.
    Info on format can be found at:http://people.maths.ox.ac.uk/nanda/perseus/index.html
    
    .. todo:: Set this up to work with higher-dimensional cubical complexes
    """

    Top = np.array([[2],[np.shape(M)[0]], [np.shape(M)[1]]])

    M = M.T
    M = M.reshape((np.shape(M)[0]*np.shape(M)[1],1))
    M = np.concatenate([Top,M])

    np.savetxt(filesavename, M, fmt = '%i')

def Cubical_Perseus(M, numDigits = 2, suppressOutput = True):
    """
    Computes persistence for a matrix of function values
    using Vidit Nanda's `perseus <http://people.maths.ox.ac.uk/nanda/perseus/index.html>`_.

    .. note::
        - perseus must be in the bash path
        - matrix must be 2-dimensional

    .. todo:: Update this to accept higher dimensional cubical complexes

    :param M:
        A 2D numpy array
    :param numDigits:
        Perseus only accepts positive integer valued matrices. To
        compensate, we apply the transformation 
        
            x -> x* (10**numDigits) + M.min()
            
        then calculate persistence on the resulting matrix.
        The persistence diagram birth/death times are then converted 
        back via the inverse transform.
    :param suppressOutput:
        If true, gets rid of printed output from perseus.

    :returns: A dictionary with integer keys 0,1,...,N.
        The key gives the dimension of the persistence diagram.

    """

    #---- Clean up and/or create local folder system
    prepareFolders()

    inputFileName = '.teaspoonData/input/inputMatrix.txt'
    outputFileName = '.teaspoonData/output/perseusMatrix'


    #--- Fix integer/positivity assumption

    rangeM = (M.min(),M.max())

    if rangeM[0] <= 0:
        M = (M-rangeM[0]+1)*(10**numDigits)
        # print ('M has been modified to compensate for negatives and sigfigs...')

    elif numDigits >0:
        M = M * 10**numDigits
        # print ('M has been modified to compensate for sigfigs...')

    elif numDigits <0:
        print('Number of digits must be a positive integer.  Exiting...')
        return

    # Write the matrix to a file in the way perseus understands.
    writeMatrixFileForPerseus(M,inputFileName)


    # Run perseus
    if suppressOutput:
        stdout = DEVNULL
    else:
        stdout = None

    try:
        command = 'perseus cubtop ' + inputFileName + ' ' + outputFileName
        call(command,
            stdout=stdout,
            stderr=STDOUT,
            shell=True)

    except:
        print('There appears to be a problem running perseus...')
        print('Do you have it properly installed?')
        print('Exiting...')
        return


    # Read in the diagrams
    Dgms = readPerseusOutput(outputFileName)


    # and put the sigfigs back if they were edited
    for key in Dgms.keys():
        Dgm = Dgms[key]

        # Change any -1 death times to infinity
        infLocs = np.where(Dgm<0)
        Dgm[infLocs] = np.inf

        # and put the sigfigs back if they were edited
        if rangeM[0]<=0:
            Dgms[key] = Dgm* (10**(-numDigits)) - 1 + rangeM[0]
        else:
            Dgms[key] = Dgm * (10**(-numDigits))

    return Dgms


#----------------------------------------------
#----------------------------------------------
#---Simple operations on pers dgms-------------
#----------------------------------------------
#----------------------------------------------

def minPers(Dgm):
    """
    Finds minimum persistence for a given diagram

    :param Dgm:
        A 2D numpy array

    :returns: 
        (float) Minimum persistence for the given diagram

    """
    try:
        lifetimes = Dgm[:,1] - Dgm[:,0]
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
        lifetimes = Dgm[:,1] - Dgm[:,0]
        m = max(lifetimes)
        if m == np.inf:
            # Get rid of rows with death time infinity
            numRows = Dgm.shape[0]
            rowsWithoutInf = list(set(np.where(Dgm[:,1] != np.inf)[0]))
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
        m = max(Dgm[:,0])
        if m == np.inf:
            # Get rid of rows with death time infinity
            numRows = Dgm.shape[0]
            rowsWithoutInf = list(set(np.where(Dgm[:,1] != np.inf)[0]))
            m = max(Dgm[rowsWithoutInf,0])

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
        m = min(Dgm[:,0])
        return m
    except:
        return 0

## \brief Gets minimum persistence for a pandas.Series with diagrams as entries
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
    return min ( DgmsSeries.apply(minPers))

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
    return max ( DgmsSeries.apply(maxPers))

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
    return min ( DgmsSeries.apply(minBirth))

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
    return max ( DgmsSeries.apply(maxBirth))


def removeInfiniteClasses(Dgm):
    '''
    Simply deletes classes that have infinite lifetimes.

    '''
    keepRows = np.isfinite(Dgm[:,1])
    return Dgm[keepRows,:]
