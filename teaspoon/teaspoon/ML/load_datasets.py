import pkg_resources
import pandas as pd

def mpeg7():
    """Load the persistence diagrams from the MPEG7 dataset

    """
    
    stream = pkg_resources.resource_stream(__name__, 'datasets/mpeg7.pickle')
    return pd.read_pickle(stream)

def shrec14():
    """Load the persistence diagrams from the shrec14 dataset

    """
    
    stream = pkg_resources.resource_stream(__name__, 'datasets/shrec14.pickle')
    return pd.read_pickle(stream)

def mnist():
    """Load the persistence diagrams from the training portion of the MNIST dataset from http://yann.lecun.com/exdb/mnist/

    """
    stream = pkg_resources.resource_stream(__name__, 'datasets/mnist.pickle')
    return pd.read_pickle(stream)
