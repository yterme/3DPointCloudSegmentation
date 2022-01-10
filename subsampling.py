#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Second script of the practical session. Subsampling of a point cloud
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.preprocessing import label_binarize

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def cloud_decimation(points, colors, labels, factor):

    decimated_points = points[::factor]
    decimated_colors = colors[::factor]
    decimated_labels = labels[::factor]

    return decimated_points, decimated_colors, decimated_labels


def grid_subsampling(points, voxel_size):

    #   Tips :
    #       > First compute voxel indices in each direction for all the points (can be negative).
    #       > Sum and count the points in each voxel (use a dictionaries with the indices as key).
    #         Remember arrays cannot be dictionary keys, but tuple can.
    #       > Divide the sum by the number of point in each cell.
    #       > Do not forget you need to return a numpy array and not a dictionary.
    #

    subsampled_points = []
    voxel_indices = np.array(points / voxel_size, dtype=int)

    voxels = {}

    for i in range(len(points)):
        t = tuple(voxel_indices[i])
        if t not in voxels:
            voxels[t] = [i]
        else:
            voxels[t].append(i)

    for v in voxels:
        voxel_points = points[voxels[v], :]

        subsampled_points.append(
            np.sum(voxel_points, axis=0) / len(voxel_points))

    return np.array(subsampled_points)

def grid_subsampling_colors(points, colors, voxel_size):

    subsampled_points = []
    subsampled_colors = []
    voxel_indices = np.array(points / voxel_size, dtype=int)

    voxels = {}

    for i in range(len(points)):
        t = tuple(voxel_indices[i])
        if t not in voxels:
            voxels[t] = [i]
        else:
            voxels[t].append(i)

    for v in voxels:
        voxel_points = points[voxels[v], :]
        voxel_colors = colors[voxels[v], :]

        subsampled_points.append(
            np.sum(voxel_points, axis=0) / len(voxel_points))
        subsampled_colors.append(
            np.sum(voxel_colors, axis=0) / len(voxel_colors))

    return np.array(subsampled_points), np.array(subsampled_colors, dtype=np.uint8)


def grid_subsampling_labels(points, colors, labels, voxel_size):
    subsampled_points = []
    subsampled_colors = []
    subsampled_labels = []
    voxel_indices = np.array(points / voxel_size, dtype=int)

    voxels = {}
    unique_labels = np.unique(labels)

    for i in range(len(points)):
        t = tuple(voxel_indices[i])
        if t not in voxels:
            voxels[t] = [i]
        else:
            voxels[t].append(i)

    for v in voxels:
        voxel_points = points[voxels[v], :]
        voxel_colors = colors[voxels[v], :]
        voxel_labels = labels[voxels[v]]

        subsampled_points.append(
            np.sum(voxel_points, axis=0) / len(voxel_points))
        subsampled_colors.append(
            np.sum(voxel_colors, axis=0) / len(voxel_colors))

        # Compute how many labels we have in each
        bins = np.sum(label_binarize(voxel_labels, unique_labels), axis=0)
        subsampled_labels.append(unique_labels[np.argmax(bins)])

    return (np.array(subsampled_points),
            np.array(subsampled_colors, dtype=np.uint8),
            np.array(subsampled_labels, dtype=np.int32))


# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#


if __name__ == '__main__':

    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']

    # Decimate the point cloud
    # ************************
    #

    # Define the decimation factor
    factor = 300

    # Decimate
    t0 = time.time()
    decimated_points, decimated_colors, decimated_labels = cloud_decimation(
        points, colors, labels, factor)
    t1 = time.time()
    print('decimation done in {:.3f} seconds'.format(t1 - t0))

    # Save
    write_ply('../data/decimated.ply', [decimated_points, decimated_colors,
                                        decimated_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    print(len(decimated_points))
    # Subsample the point cloud on a grid
    # ***********************************
    #

    # Define the size of the grid
    voxel_size = 0.2

    # Subsample
    t0 = time.time()
    subsampled_points, subsampled_colors, subsampled_labels = grid_subsampling_labels(
        points, colors, labels, voxel_size)
    t1 = time.time()
    print('Subsampling done in {:.3f} seconds'.format(t1 - t0))
    # Save
    write_ply('../data/grid_subsampled_labels.ply',
              [subsampled_points, subsampled_colors, subsampled_labels],
              ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    print('Done')
