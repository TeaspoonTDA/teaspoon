import numpy as np
from gudhi import CubicalComplex
import matplotlib.pyplot as plt


def array_to_dict(result):

    result_dict = {}
    for key, value in result:
        if key not in result_dict:
            result_dict[key] = []
        result_dict[key].append(list(value))

    return result_dict


def BettiCurve(Dgm, maxEps=1, numStops=10):

    vecOfThresholds = np.linspace(0.001, maxEps, numStops)
    Betti = np.zeros(np.shape(vecOfThresholds))

    for i, v in enumerate(vecOfThresholds):
        if len(Dgm) > 0:
            Betti[i] = sum(np.logical_and((Dgm[:, 0] >= v), (Dgm[:, 1] < v)))
        else:
            Betti[i] = 0

    return vecOfThresholds, Betti


def compute_cubical_persistence(p, filter=0.02):

    p = -p / np.max(p)

    sup_pers = CubicalComplex(top_dimensional_cells=p)
    sup_pers = sup_pers.persistence(min_persistence=filter)
    # sup_pers = np.array(sup_pers)

    return array_to_dict(sup_pers)


def analytical_homological_bifurcation_plot(PDFs, bifurcation_parameters, dimensions=[0, 1], filter=0.02, maxEps=1, numStops=10, plotting=True):
    '''
    Computes the homological bifurcation plot given a list of analytically generated PDFs

    :param PDFs (list):  A python list of numpy arrays of PDFs
    :param bifurcation_parameters (array): Array of bifurcation parameters against the PDFs
    :param dimensions (Optional[list]): List of homology dimensions to make plots for 
    :param filter (Optional[float]): Persistence values above which classes are kept; default: 0.02
    :param maxEps (Optional[float]): Maximum value of threshold; default: 1
    :param numStops (Optional[int]): Number of points between 0 and maxEps; default: 10
    :param plotting (Optional[bool]): Plots the homological plot for the given diagrams; default: True

    :returns: 2D homological bifurcation plot(s)

    '''

    if len(bifurcation_parameters) != len(PDFs):
        print('Bifurcation Parameters should be the same length as number of PDFs')
        return

    All_DGMS = []
    for PDF in PDFs:
        dgm = compute_cubical_persistence(PDF, filter)
        All_DGMS.append(dgm)

    All_DGMS = np.array(All_DGMS)

    plots = []
    for key in dimensions:

        DGMS = []
        for dictionary in All_DGMS:
            array_of_arrays = []
            if key in dictionary:
                array_of_arrays.append(dictionary[key])
            else:
                array_of_arrays.append([])

            array_of_arrays = np.abs(array_of_arrays[0])
            array_of_arrays[np.isinf(array_of_arrays)] = 0
            DGMS.append(array_of_arrays)

        AllBettis = []
        for Dgm in DGMS:
            Dgm = np.array(Dgm)
            t, x = BettiCurve(Dgm, maxEps, numStops)
            AllBettis.append(x)

        M = np.array(AllBettis).T

        plots.append(M)

        if plotting:
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(3, 3), dpi=300)
            ax = fig.add_subplot(111)
            cax = ax.matshow(M, origin='lower', extent=(-1,
                             1, 0, 1), aspect=2, cmap='Set1')
            fig.colorbar(cax)
            ax.set_xticks(np.round([bifurcation_parameters[0], bifurcation_parameters[int(
                len(bifurcation_parameters)/2)], bifurcation_parameters[-1]], 1))
            ax.set_yticks([0, 0.5, 1])
            ax.set_ylabel(r'$\epsilon$')
            ax.set_xlabel('Bifurcation Parameter')
            ax.set_title(fr'Plot for $H_{key}$')
            plt.show()

    return plots
