import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.spatial.distance import pdist, squareform


def get_sphere(radius, density):
    ''' 
    Create a sparse hypersphere with four dimensions. The hypersphere will have a radius 'radius'.
    The magnitude of the imaginary part of 'density' defines the number of points used to create
    the hypersphere.
    '''

    origin = np.zeros((4, ))

    # define a hypercube
    hcube = np.mgrid[-radius:radius:density, -radius:radius:density, 
                     -radius:radius:density, -radius:radius:density]

    # set list of points that define a hypersphere
    hsphere = []
    step = int(density.imag)
    for x in range(step):
        for y in range(step):
            for z in range(step):
                for k in range(step):
                    v1 = np.array([hcube[0][x, y, z, k], hcube[1][x, y, z, k], hcube[2][x, y, z, k], hcube[3][x, y, z, k]])
                    if np.linalg.norm(v1 - origin) <= radius:
                        hsphere.append([v1[0], v1[1], v1[2], v1[3]])

    return hsphere

def get_normalized_metrics(metrics):
    """Apply z-score normalization to 'metrics'."""

    for idx in range(len(metrics)):
        avg = np.mean(metrics[idx])
        std = np.std(metrics[idx])
        metrics[idx] = (metrics[idx] - avg) / std

    return metrics

def approximate_distribution(hsphere, metrics, verbose):
    
    """
    Algorithm for approximating the distribution of the data with a sparse hypersphere
    ------------------

    For each data point, translate the hypersphere and add the translated points to
    the distribution set. Collisions increase a hit count for each point. This set of 
    points approximates the distribution of the real data.
    """

    distribution_points = {}
    n_data_points = len(metrics[0])

    iterate_through = tqdm(range(n_data_points)) if verbose else range(n_data_points)

    for i in iterate_through:
        data_point = [metrics[0][i], metrics[1][i], metrics[2][i], metrics[3][i]]
        transl_hsphere = np.array(hsphere) + data_point
        for sphere_point in transl_hsphere:
            t_sphere_point = tuple(sphere_point)
            if t_sphere_point in distribution_points:
                distribution_points[t_sphere_point] += 1
            else:
                distribution_points[t_sphere_point] = 1

    return distribution_points

def get_dist_point_from_window(distribution, selected_points, closest_windows, window_idx):
    
    """Returns the distribution point associated with the window with index 'window_idx'."""
    point_idx = selected_points[closest_windows.index(window_idx)]
    return distribution[point_idx]

def get_window_point(metrics, ind):
    """Returns a numpy array containing the metrics from the window point with index 'ind'."""
    return np.array([metrics[0][ind], metrics[1][ind], metrics[2][ind], metrics[3][ind]])

def euclidean_distance(v1, v2):
    """Returns the euclidean distance between points 'v1' and 'v2'."""
    return np.linalg.norm(v1-v2)

def get_window_to_swap(distribution, closest_windows, all_fnames, selected_windows, selected_fnames, selected_points, 
                       metrics, window_idx):
    """
    Iteratively find another point to replace 'window_idx'. The new window point cannot have been selected before 
    nor come from an image that already has a window point selected.
    """

    orig = get_dist_point_from_window(distribution, selected_points, closest_windows, window_idx)
    distances = []
    for target_idx in range(len(metrics[0])):
        target = get_window_point(metrics, target_idx)
        distances.append(euclidean_distance(orig, target))
    
    sort_idxs = np.argsort(distances)
    for window_idx in sort_idxs:
        if window_idx not in selected_windows and all_fnames[window_idx] not in selected_fnames:
            return window_idx

    return -1

def farthest_unselected_point(closest_windows, distances_between_windows):
    """Returns the largest distance between an unselected point and the drawn windows ('closest_windows')."""

    unselected_points = [p for p in range(len(distances_between_windows)) if p not in closest_windows]
    uns_to_win_distances = []
    for uns in unselected_points:
        distances = distances_between_windows[uns, closest_windows]
        uns_to_win_distances.append(np.min(distances))
    
    return np.max(uns_to_win_distances)
    
def draw_windows(n_iter, n_images, metrics, distribution, fnames, distances_between_windows, verbose):
    
    """
    Draw 'n_images' points from 'distribution' and select the closest windows from the drawn points. 
    A window cannot be selected twice and only a single window can be selected for each original image. 
    To do this, the windows are swapped iteratively as necessary. 'n_iter' sets of windows are drawn, 
    and the set with the smallest value of the 'farthest unselected point' metric is selected.
    """

    solution_distances = []
    solution_windows = []
    Tmetrics = np.array(metrics).T

    iterate_through = tqdm(range(n_iter)) if verbose else range(n_iter)

    for n in iterate_through:
        selected_points = np.random.choice(np.arange(0, len(distribution), 1), n_images, replace=False)

        # closest images to the drawn points
        closest_windows = []
        for orig_ind in selected_points:
            v1 = distribution[orig_ind]
            distances = np.sqrt(((v1 - Tmetrics)**2).sum(axis=1))
            closest_windows.append(np.argmin(distances))

        # iteratively swap windows from the same image
        unique_fnames = np.unique([fnames[idx] for idx in closest_windows])
        selected_fnames = list(unique_fnames.copy())
        selected_windows = closest_windows.copy()
        while len(unique_fnames) != len(closest_windows):
            windows_to_swap = []
            for fname in unique_fnames:
                windows_from_fname = [window for window in closest_windows if fnames[window] == fname]
                if len(windows_from_fname) > 1:
                    points_from_fname = [get_dist_point_from_window(distribution, selected_points, closest_windows, window) 
                                         for window in windows_from_fname]
                    distances_from_dist_point = []
                    for idx, point in enumerate(points_from_fname):
                        corr_window = windows_from_fname[idx]
                        target = get_window_point(metrics, corr_window)
                        distances_from_dist_point.append(euclidean_distance(point, target))

                    sort_idx = np.argsort(distances_from_dist_point)
                    for idx in sort_idx[1:]:
                        windows_to_swap.append(windows_from_fname[idx])

            for window in windows_to_swap:
                new_window = get_window_to_swap(distribution, closest_windows, fnames, selected_windows, 
                                                selected_fnames, selected_points, metrics, window)
                selected_windows.append(new_window)
                selected_fnames.append(fnames[new_window])
                closest_windows[closest_windows.index(window)] = new_window

            unique_fnames = np.unique([fnames[idx] for idx in closest_windows])

        solution_distances.append(farthest_unselected_point(closest_windows, distances_between_windows))
        solution_windows.append(closest_windows)

    # choose the solution in which the farthest unselected point is closest to the drawn distribution
    minind = np.argmin(solution_distances)

    return solution_windows[minind], solution_distances

def draw_sample_points(radius: int, density: int, resample_resolution: float, n_iter: int, n_images: int, 
                       metrics: ArrayLike, verbose: bool = True) -> list[int]:
    """
    Draw 'n_images' samples from a distribution of points defined by 'metric'.
    
    *** This script was originally used for drawing windows of a dataset of blood vessels, as described in [1]. 
    Therefore, data points are often mentioned as 'windows'.

    [1] da Silva, MV., Santos, NdC., Lacoste, B., & Comin, CH. A new sampling methodology for creating rich, 
    heterogeneous, subsets of samples for training image segmentation algorithms. 
    arXiv preprint arXiv:2301.04517, (2023).

    Parameters
    -----------

    radius: int
        the radius of the sphere used to fit the distribution of all windows.

    density: int
        controls the number of points inside the sphere of radius 'radius'.

    resample_resolution: float
        ratio in which the metrics of each window will be resampled. After
        z-score normalization, each metric value will be transformed as
        int(value / resampled_resolution).
    
    n_iter: int
        the number of solutions that will be drawn. This function returns the solution that
        minimizes the 'farthest unselected point' metric.

    n_images:
        the number of images to be drawn.

    verbose: bool
        if True, this function will output the progression bars and some useful
        charts.

    Returns
    -----------

    solution: list of int
        a list of the indices of the drawn windows.
    
    """

    density = complex(0, density)
    hsphere = get_sphere(radius, density)
    
    if verbose:
        fig = plt.figure(figsize=(10, 10))
        grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0.3)

        # sort the hypersphere dimension-wise
        Tsphere = np.array(hsphere).T

        grid_idx = 0
        for i in range(4):
            for j in range(4):
                grid[grid_idx].scatter(Tsphere[i], Tsphere[j])
                grid[grid_idx].set_title(f'{i} / {j}')
                grid_idx += 1

        # plot the sphere
        plt.show()
    
    Tmetrics = np.array(metrics)
    Tmetrics = get_normalized_metrics(Tmetrics)
    Tmetrics = (Tmetrics / resample_resolution).astype(np.int16)
    Tmetrics = Tmetrics.T

    if verbose:
        print('Approximating distribution...')
    
    distribution_points = approximate_distribution(hsphere, metrics, verbose)
    
    # sort the distribution metric-wise
    distribution = np.array(list(distribution_points.keys()))

    if verbose:
        # check the distribution
        fig = plt.figure(figsize=(10, 10))
        grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0.3)
        
        grid_idx = 0
        for i in range(4):
            for j in range(4):
                grid[grid_idx].scatter(distribution.T[i], distribution.T[j], alpha=1, s=1)
                grid[grid_idx].scatter(metrics[i], metrics[j], alpha=1, s=1)
                grid[grid_idx].set_title(f'{i} / {j}')
                grid_idx += 1

        plt.show()

    # distance between all windows
    distances_between_windows = squareform(pdist(Tmetrics))

    # info of all windows (original filenames and positions)
    # since we are using artificial data in this demo, we will use the
    # image index as a name
    fnames = [str(i) for i in range(len(metrics[0]))]

    if verbose:
        print('Drawing windows...')
    
    solution, solution_distances = draw_windows(n_iter, n_images, metrics, distribution, fnames, 
                                                      distances_between_windows, verbose)

    if verbose:
        plt.figure(figsize=(10, 10))
        hist, n_edges = np.histogram(solution_distances, bins=300)
        plt.bar(n_edges[1:], hist, width=n_edges[1]-n_edges[0])
        plt.title('Histogram - farthest unselected window')
        plt.show()

    if verbose:
        # plot the distribution of the selected images (with respect to each metric)
        titles = ['M1', 'M2', 'M3', 'M4']
        plt.figure(figsize=(10, 10))
        grid_idx = 1
        for idx, metric in enumerate(metrics):
            plt.subplot(2, 2, grid_idx)
            hist, n_edges = np.histogram(metric, bins=50)
            hist = hist / np.sum(hist)
            plt.bar(n_edges[1:], hist, width=n_edges[1]-n_edges[0], alpha=0.6)
            hist, n_edges = np.histogram(metrics[idx][solution], bins=50)
            hist = hist / np.sum(hist)
            plt.bar(n_edges[1:], hist, width=n_edges[1]-n_edges[0], alpha=0.6)
            plt.title(titles[idx])
            grid_idx += 1
        
        plt.show()

        plt.figure(figsize=(25, 20))
        plt.subplot(2, 3, 1)
        plt.scatter(metrics[0], metrics[1], s=1, alpha=0.3)
        plt.scatter(metrics[0][solution], metrics[1][solution], c='red', s=1)
        
        plt.subplot(2, 3, 2)
        plt.scatter(metrics[0], metrics[2], s=1, alpha=0.3)
        plt.scatter(metrics[0][solution], metrics[2][solution], c='red', s=1)

        plt.subplot(2, 3, 3)
        plt.scatter(metrics[0], metrics[3], s=1, alpha=0.3)
        plt.scatter(metrics[0][solution], metrics[3][solution], c='red', s=1)

        plt.subplot(2, 3, 4)
        plt.scatter(metrics[1], metrics[2], s=1, alpha=0.3)
        plt.scatter(metrics[1][solution], metrics[2][solution], c='red', s=1)

        plt.subplot(2, 3, 5)
        plt.scatter(metrics[1], metrics[3], s=1, alpha=0.3)
        plt.scatter(metrics[1][solution], metrics[3][solution], c='red', s=1)

        plt.subplot(2, 3, 6)
        plt.scatter(metrics[2], metrics[3], s=1, alpha=0.3)
        plt.scatter(metrics[2][solution], metrics[3][solution], c='red', s=1)

        plt.show()

    return solution
