import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
import os

# Function gets filtration from gudhi for the point cloud


def get_filtration(points, radius):

    skeleton = gd.RipsComplex(points=points, max_edge_length=radius)
    Rips_simplex = skeleton.create_simplex_tree(max_dimension=2)
    rips_generator = Rips_simplex.get_filtration()
    rips_list = list(rips_generator)
    filts = np.array(rips_list, dtype=object)[:, 0]

    return filts

# Function prints each line of filtration in a file


def print_in_file(filtration, filename, type='i'):

    if type == 'i':
        with open(filename, 'a') as file:
            for element in filtration:
                file.write(f"i {' '.join(map(str, element))}\n")
    elif type == 'd':
        filtration.reverse()
        with open(filename, 'a') as file:
            for element in filtration:
                file.write(f"d {' '.join(map(str, element))}\n")

# Function renumbers filtration based on how many point clouds have already been added since gudhi writes all filtrations starting at 0


def filtration_renumbering(filt, n):
    modified = [[x + n for x in sublist] for sublist in filt]
    return modified

# Calculating pairwise distances for greedy permutation


def dpoint2pointcloud(X, i):
    ds = pairwise_distances(X, X[i, :][None, :], metric='euclidean').flatten()
    ds[i] = 0
    return ds

# Gets the greedy permutation given the point cloud and n_perm


def get_greedy_perm(X, n_perm):
    idx_perm = np.zeros(n_perm, dtype=np.int64)
    lambdas = np.zeros(n_perm)
    def dpoint2all(i): return dpoint2pointcloud(X, i)
    ds = dpoint2all(0)
    dperm2all = [ds]

    for i in range(1, n_perm):
        idx = np.argmax(ds)
        idx_perm[i] = idx
        lambdas[i - 1] = ds[idx]
        dperm2all.append(dpoint2all(idx))
        ds = np.minimum(ds, dperm2all[-1])
    return (idx_perm)

# Reads the output file from fast-zig-zag and outputs a dictionary of all classes


def read_data(file_path):
    classes = {}
    with open(file_path, 'r') as file:
        for line in file:
            class_label, x, y = map(float, line.split())
            if class_label not in classes:
                classes[class_label] = {'x': [], 'y': []}
            classes[class_label]['x'].append(x)
            classes[class_label]['y'].append(y)

    for i in classes.keys():
        classes[i]['x'] = np.array(classes[i]['x'])
        classes[i]['y'] = np.array(classes[i]['y'])

    return classes


def remove_inner(classes, lines):

    for cl in classes.keys():

        x_coords = classes[cl]['x']
        y_coords = classes[cl]['y']

        mask = np.ones(len(x_coords), dtype=bool)

        for n in range(len(lines)):

            if n == 0:
                mask &= ~((x_coords < lines[n]) & (y_coords < lines[n]))
            else:
                mask &= ~((x_coords < lines[n]) & (
                    y_coords < lines[n]) & (x_coords > lines[n-1]))

        x_filtered = classes[cl]['x'][mask]
        y_filtered = classes[cl]['y'][mask]

        classes[cl]['x'] = x_filtered
        classes[cl]['y'] = y_filtered

    return classes


def remove_axial(classes, lines_i, lines_d):

    for i in classes.keys():

        mask = (classes[i]['x'] != classes[i]['y'])
        classes[i]['x'] = classes[i]['x'][mask]
        classes[i]['y'] = classes[i]['y'][mask]

    return classes


def clean_data(classes, lines_i, lines_d):

    for i in classes.keys():

        mask = np.zeros_like(classes[i]['x'])

        for cl in range(len(lines_i)):

            if cl == 0:
                m = np.logical_and(
                    classes[i]['x'] <= lines_i[cl], classes[i]['x'] >= 0, classes[i]['y'] >= lines_i[cl])
                mask = mask + m
            else:
                m = np.logical_and(classes[i]['x'] <= lines_i[cl], classes[i]
                                   ['x'] >= lines_i[cl-1], classes[i]['y'] >= lines_i[cl])
                mask = mask + m

        mask = mask.astype(bool)
        classes[i]['x'] = classes[i]['x'][mask]
        classes[i]['y'] = classes[i]['y'][mask]

    return classes


def custom_searchsorted(arr, value):
    index = np.searchsorted(arr, value, side='right')
    if index > 0 and arr[index-1] == value:
        return index - 1
    return index


def update_coordinates(classes, lines_i, lines_d):

    for cl in classes.keys():

        x_coords = classes[cl]['x']
        y_coords = classes[cl]['y']

        x_new = np.zeros_like(x_coords)
        y_new = np.zeros_like(y_coords)

        for i in range(len(x_coords)):
            x = x_coords[i]
            y = y_coords[i]

            # Find the interval for x
            x_interval = custom_searchsorted(lines_i, x)  # - 1

            if x_interval <= 0:
                x_new[i] = 0
            elif x_interval == 1 & lines_i[x_interval-1] <= x <= lines_i[x_interval]:
                x_new[i] = x_interval - 0.5
            elif lines_i[x_interval] <= x <= lines_d[x_interval]:
                x_new[i] = x_interval
            else:
                x_new[i] = x_interval

            # Find the interval for y
            y_interval = custom_searchsorted(lines_d, y)

            if y_interval == len(lines_d)-2 & lines_i[y_interval] <= y <= lines_d[y_interval]:
                y_new[i] = y_interval + 1.0
            elif lines_d[-2] < y < lines_d[-1]:
                y_new[i] = y_interval  # + 1.0
            elif lines_i[y_interval] <= y <= lines_d[y_interval]:
                y_new[i] = y_interval + 1.0
            else:
                y_new[i] = y_interval

        classes[cl]['x'] = x_new
        classes[cl]['y'] = y_new

    return classes

# Function to plot the PD


def plot_data(classes, plotH2=False):
    ymax = 0
    markers = ['o', 'x', '.']
    for class_label, coordinates in classes.items():
        if plotH2:
            plt.scatter(coordinates['x'], coordinates['y'],
                        label=f'H{int(class_label)}', marker=markers[int(class_label)])
            if max(coordinates['y'], default=0) > ymax:
                ymax = max(coordinates['y'], default=0)
        else:
            if class_label != 2:
                plt.scatter(coordinates['x'], coordinates['y'],
                            label=f'H{int(class_label)}', marker=markers[int(class_label)])
                if max(coordinates['y'], default=0) > ymax:
                    ymax = max(coordinates['y'], default=0)

    plt.title('Fast-Zigzag Persistence Output')
    plt.xlabel('Birth')
    plt.ylabel('Death')
    plt.plot([0, ymax], [0, ymax], '--k')
    plt.legend(loc="lower right")
    plt.show()

# Generates the input file for fast-zigzag given a list of point clouds


def generate_input_file(point_clouds, filename='output', radius=10, n_perm=25, plotting=False):
    """This function generates the input file for fast-zigzag software by TDA-Jyamiti group.

    Args:
        point_clouds (list):  List of point clouds in R2.

    Other Parameters:
        filename (str): Name of file generated.
        radius (float): Radius at which the rips complex is generated.
        n_perm (int): Number of points to take from given point clouds for greedy permutation.
        plotting (bool): if True, plots the point clouds

    Returns:
        [list]: Number of lines of filtrations inserted for a point cloud.
        [list]: Number of lines of filtrations deleted for a point cloud.
    """

    if os.path.exists(filename):
        os.system(f"rm {filename}")

    if len(point_clouds) < 2:
        print('Atleast 2 point clouds should be provided')
        return

    if radius < 0:
        print('radius should be non-negative')
        return

    if plotting:
        for i in range(len(point_clouds)):
            plt.scatter(point_clouds[i][:, 0], point_clouds[i][:, 1])
        plt.show()

    if n_perm <= 0:
        print('n_perm must be a positive integer')
        return
    else:
        for num in range(0, len(point_clouds)):
            if n_perm < point_clouds[num].shape[0]:
                idx_perm = get_greedy_perm(point_clouds[num], n_perm)
                point_clouds[num] = point_clouds[num][idx_perm]

    n_points = []
    for i in range(len(point_clouds)):
        n_points.append(np.shape(point_clouds[i])[0])

    inserts = []
    deletes = []
    all = []
    total = 0

    for num in range(len(point_clouds)):

        if num == 0:
            first_elements = get_filtration(point_clouds[0], radius=radius)
            print_in_file(first_elements, filename, type='i')
            total = len(first_elements)
            inserts.append(total)
            all.append(total)

            elements = get_filtration(np.concatenate(
                (point_clouds[num], point_clouds[num+1])), radius=radius)
            elements_array = np.array([np.array(item)
                                      for item in elements], dtype=object)

            mask = np.array([np.any(item >= n_points[0])
                            for item in elements_array])
            filtered_elements = elements_array[mask]
            filtered_elements = filtered_elements.tolist()

            print_in_file(filtered_elements, filename, type='i')
            total = total + len(filtered_elements)
            inserts.append(total)
            all.append(total)

            mask = np.array([np.any(item < n_points[0])
                            for item in elements_array])
            filtered_elements = elements_array[mask]
            filtered_elements = filtered_elements.tolist()

            print_in_file(filtered_elements, filename, type='d')
            total = total + len(filtered_elements)
            deletes.append(total)
            all.append(total)

        elif num == len(point_clouds)-1:
            elements = get_filtration(point_clouds[num], radius=radius)
            elements = filtration_renumbering(elements, sum(n_points[:-1]))
            print_in_file(elements, filename, type='d')
            total = total + len(elements)
            deletes.append(total)
            all.append(total)

        else:
            elements = get_filtration(np.concatenate(
                (point_clouds[num], point_clouds[num+1])), radius=radius)
            elements = filtration_renumbering(elements, sum(n_points[:num]))
            elements_array = np.array([np.array(item)
                                      for item in elements], dtype=object)

            mask = np.array([np.any(item >= sum(n_points[:num+1]))
                            for item in elements_array])
            filtered_elements = elements_array[mask]
            filtered_elements = filtered_elements.tolist()

            print_in_file(filtered_elements, filename, type='i')
            total = total + len(filtered_elements)
            inserts.append(total)
            all.append(total)

            mask = np.array([np.any(item < sum(n_points[:num+1]))
                            for item in elements_array])
            filtered_elements = elements_array[mask]
            filtered_elements = filtered_elements.tolist()

            print_in_file(filtered_elements, filename, type='d')
            total = total + len(filtered_elements)
            deletes.append(total)
            all.append(total)

    return inserts, deletes

# Plots the output zigzag PD from the output file of fast-zigzag


def plot_output_zigzag(filename, inserts, deletes, plotH2=False, plot=True, filter=True):
    """This function takes the output file from fast-zigzag software by TDA-Jyamiti group and number of insertions/deletions output by the above function and plots the zigzag persistence diagram. Note that fast-zigzag produces closed [b,d] intervals.

    Args:
        filename (str): Name of file generated.
        inserts (list): Number of lines of filtrations inserted for a point cloud.
        deletes (list): Number of lines of filtrations deleted for a point cloud.

    Other Parameters:
        plotH2 (bool): if True, plots the H2 components in the zigzag persistence diagram which may sometimes appear in 2D point clouds
        plot (bool)L if True, plots the persistence diagram
        filter (bool): if False, returns the full persistence diagram from fast-zigzag software with no additional filtering

    Returns:
        [dict]: Dictionary of persistence points for each homology class
    """

    data = read_data(filename)
    if filter:
        data = remove_axial(data, inserts, deletes)
        data = clean_data(data, inserts, deletes)
        data = remove_inner(data, inserts)
        data = update_coordinates(data, inserts, deletes)
    if plot:
        plot_data(data, plotH2)

    return data
