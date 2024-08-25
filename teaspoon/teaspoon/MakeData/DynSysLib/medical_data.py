import numpy as np
import os


def medical_data(system, dynamic_state=None, L=None, fs=None,
                 SampleSize=None, parameters=None, InitialConditions=None):

    if system == 'ECG':
        t, ts = ECG()

    if system == 'EEG':
        t, ts = EEG()

    return t, ts


def EEG(SampleSize=5000, dynamic_state='normal'):
    """
    The EEG signal was taken from andrzejak et al. [1]_. Specifically, the first 5000 data points from the EEG data of a healthy patient from set A (file Z-093) was used and the first 5000 data points of a patient experiencing a seizure from set E (file S-056) was used (see figure below for case during seizure).

    .. figure:: ../../../figures/Human_Medical_Data/EEG_Data.png

    Parameters:
        SampleSize (Optional[int]): length of sample at end of entire time series
        dynamic_state (Optional[str]): Set dynamic state as either 'normal' or 'seizure'.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`

    References
    ----------
    .. [1] Ralph G Andrzejak, Klaus Lehnertz, Florian Mormann, Christoph Rieke, Peter David, and Christian E Elger. Indications of nonlinear deterministic and nite-dimensional structures in time series of brain electrical activity: Dependence on recording region and brain state. Physical Review E, 64(6):061907, 2001.

    """
    path = os.path.join(os.path.join(os.path.join(os.path.join(os.path.join(
        os.getcwd(), 'teaspoon'), 'teaspoon'), 'MakeData'), 'DynSysLib'), 'Data')

    if dynamic_state == 'normal':  # healthy
        path = os.path.join(os.path.join(path, 'EEG'), 'Z093.txt')
        ts = [np.loadtxt(path, skiprows=1)[
            0:SampleSize]]

    if dynamic_state == 'seizure':  # seizure
        path = os.path.join(os.path.join(path, 'EEG'), 'S056.txt')
        ts = [np.loadtxt(path, skiprows=1)[
            0:SampleSize]]

    fs = 173.61
    t = np.arange(len(ts[0]))/fs
    t = t[-SampleSize:]
    ts = [(ts[0])[-SampleSize:]]

    return t, ts


def ECG(dynamic_state='normal'):
    """
    The Electrocardoagram (ECG) data was taken from SciPys misc.electrocardiogram data set. This ECG data was originally provided by the MIT-BIH Arrhythmia Database [2]_. We used data points 3000 to 5500  during normal sinus rhythm and 8500 to 11000 during arrhythmia (arrhythmia case shown below in figure).

    .. figure:: ../../../figures/Human_Medical_Data/ECG_Data.png

    Parameters:
        dynamic_state (Optional[str]): Set dynamic state as either 'normal' or 'seizure'.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`

    References
    ----------
    .. [2] George B Moody and Roger G Mark. The impact of the mit-bih arrhythmia database. IEEE Engineering in Medicine and Biology Magazine, 20(3):4550, 2001.

    """

    from scipy.datasets import electrocardiogram

    if dynamic_state == 'normal':  # healthy
        ts = [electrocardiogram()[3000:5500]]

    if dynamic_state == 'seizure':  # heart arrythmia
        ts = [electrocardiogram()[8500:11000]]

    fs = 360
    ts = [(ts[0])]
    t = np.arange(len(ts[0]))/fs

    return t, ts
