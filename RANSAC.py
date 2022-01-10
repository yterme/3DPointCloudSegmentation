#
#
#      0===========================================================0
#      |                      TP6 Modelisation                     |
#      0===========================================================0
#
#
#------------------------------------------------------------------------------------------
#
#      First script of the practical session. Plane detection by RANSAC
#
#------------------------------------------------------------------------------------------
#
#      Xavier ROYNARD - 19/02/2018
#


#------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time



#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def compute_plane(points):
    
    v1=points[1,:]-points[0,:]
    v2=points[2,:]-points[0,:]
    
    normal=np.cross(v1,v2)
    normal/=np.linalg.norm(normal)
    distance=points[0,:].dot(normal)
    #point = np.zeros((3,1))
    #normal = np.zeros((3,1))
    
    # TODO:
    
    return distance, normal



def in_plane(points, distance, normal, threshold_in=0.1):
    
    #indexes = np.zeros(len(points), dtype=bool)
    indexes= np.abs(np.dot(points,normal)-distance)<threshold_in
    
    # TODO:
    return indexes



def RANSAC(points, NB_RANDOM_DRAWS=100, threshold_in=0.1):
    
    best_vote=0
    for i in range(NB_RANDOM_DRAWS):
        print("Draw number",i)
        idx=np.random.choice(range(len(points)),3,replace=False)
        pts_draw=points[idx,:]
        distance,normal=compute_plane(pts_draw)
        indexes_in= in_plane(points, distance, normal, threshold_in=threshold_in)
        vote=np.sum(indexes_in)
        if vote>=best_vote:
            best_distance = distance
            best_normal = normal
            best_vote= vote
            best_idx=indexes_in
    # TODO:
                
    return best_distance, best_normal, best_vote, best_idx


def recursive_RANSAC(points, NB_RANDOM_DRAWS=100, threshold_in=0.1, NB_PLANES=2):
    
    # TODO:
    remaining_inds=np.arange(0,len(points))
    #points_remaining=points.copy()
    plane_inds=np.array([], dtype=int)
    plane_labels=np.array([], dtype=int)
    for i in range(NB_PLANES):
        #if i>=1:
        #    threshold_in=0.005
        best_distance, best_normal, best_vote, best_idx=RANSAC(points[remaining_inds], \
                                        NB_RANDOM_DRAWS=NB_RANDOM_DRAWS, threshold_in=threshold_in)
        
        #indexes= in_plane(points[remaining_inds], best_distance, best_normal, threshold_in)
        good_indices=remaining_inds[best_idx]
        print(len(good_indices))
        print(best_vote)
        plane_labels=np.concatenate([plane_labels, np.array([i]*best_vote)])
        plane_inds=np.concatenate([plane_inds,good_indices])
        remaining_inds=np.setdiff1d(remaining_inds, good_indices)


    return plane_inds, remaining_inds, plane_labels


#------------------------------------------------------------------------------------------
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
    file_path = '../data/indoor_scan_sub2cm.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']
    N = len(points)
    

    # Computes the plane passing through 3 randomly chosen points
    # ************************
    #
    
    print('\n--- Question 1 and 2 ---\n')
    
    # Define parameter
    threshold_in = 0.1

    # Take randomly three points
    pts = points[np.random.randint(0, N, size=3)]
    
    # Computes the plane passing through the 3 points
    t0 = time.time()
    dist, normal = compute_plane(pts)
    t1 = time.time()
    print('plane computation done in {:.3f} seconds'.format(t1 - t0))
    
    # Find points in the plane and others
    t0 = time.time()
    points_in_plane = in_plane(points, dist, normal, threshold_in)
    t1 = time.time()
    print('plane extraction done in {:.3f} seconds'.format(t1 - t0))
    plane_inds = points_in_plane.nonzero()[0]
    remaining_inds = (1-points_in_plane).nonzero()[0]
    
    # Save extracted plane and remaining points
    write_ply('../plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    write_ply('../remaining_points.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    

    # Computes the best plane fitting the point cloud
    # ***********************************
    #
    #
    
    print('\n--- Question 3 ---\n')

    # Define parameters of RANSAC
    NB_RANDOM_DRAWS = 100
    threshold_in = 0.05

    # Find best plane by RANSAC
    t0 = time.time()
    best_distance, best_normal, best_vote, best_idx = RANSAC(points, NB_RANDOM_DRAWS, threshold_in)
    t1 = time.time()
    print('RANSAC done in {:.3f} seconds'.format(t1 - t0))
    
#    print(best_ref_pt, best_normal, "\tNb of votes:",best_vote)
    
    # Find points in the plane and others
    points_in_plane = in_plane(points, best_distance, best_normal, threshold_in)
    plane_inds = points_in_plane.nonzero()[0]
    remaining_inds = (1-points_in_plane).nonzero()[0]
    
    # Save the best extracted plane and remaining points
    write_ply('../best_plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    write_ply('../remaining_points.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    

    # Find "all planes" in the cloud
    # ***********************************
    #
    #
    
    print('\n--- Question 4 ---\n')
    
    # Define parameters of recursive_RANSAC
    NB_RANDOM_DRAWS = 100
    threshold_in = 0.05
    NB_PLANES = 2
    
    # Recursively find best plane by RANSAC
    t0 = time.time()
    plane_inds, remaining_inds, plane_labels = recursive_RANSAC(points, NB_RANDOM_DRAWS, threshold_in, NB_PLANES)
    t1 = time.time()
    print('recursive RANSAC done in {:.3f} seconds'.format(t1 - t0))
                
    # Save the best planes and remaining points
    write_ply('../best_planes.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds], plane_labels.astype(np.int32)], ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'plane_label'])
    write_ply('../remaining_points_.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    
    
    
    print('Done')
    