import pandas as pd
import importlib_resources


def mpeg7():
    """
    Load the persistence diagrams from the MPEG7 dataset

    """

    stream = importlib_resources.files('datasets').joinpath('mpeg7.pickle')
    data = pd.read_pickle(stream)
    return(data)


def shrec14():
    """
    Load the persistence diagrams from the shrec14 dataset

    """
    stream = importlib_resources.files('datasets').joinpath('shrec14.pickle')
    data = pd.read_pickle(stream)
    return(data)


def mnist():
    """
    Load the persistence diagrams from the training portion of the MNIST dataset from http://yann.lecun.com/exdb/mnist/

    Columns available:
    zero_dim_rtl: 0-dimensional persistence diagrams computed using right to left euler transform
    zero_dim_ltr: 0-dimensional persistence diagrams computed using left to right euler transform
    zero_dim_btt: 0-dimensional persistence diagrams computed using bottom to top euler transform
    zero_dim_ttb: 0-dimensional persistence diagrams computed using top to bottom euler transform
    one_dim_rtl: 1-dimensional persistence diagrams computed using right to left euler transform
    one_dim_ltr: 1-dimensional persistence diagrams computed using left to right euler transform
    one_dim_btt: 1-dimensional persistence diagrams computed using bottom to top euler transform
    one_dim_ttb: 1-dimensional persistence diagrams computed using top to bottom euler transform

    """

    stream = importlib_resources.files('datasets').joinpath('mnist.pickle')
    data = pd.read_pickle(stream)
    return(data)
