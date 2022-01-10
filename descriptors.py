#   Yannick Terme
#
# ---------------------------
#
#   Extraction of features
#
# ---------------------------
#
#   Inspired from TP4 - made by Hugues Thomas
#
#
#

import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree
from utils.ply import write_ply, read_ply
import time
from SC3D import SC3D
from subsampling import *

def local_PCA(points):
    n=points.shape[0]
    bar=np.mean(points,axis=0)
    centered_points=points-bar
    cov=1/n*(centered_points.T)@centered_points

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    return eigenvalues[::-1], eigenvectors[:,::-1]



#
def neighborhood_PCA_knn(query_points, cloud_points, k, tree = None, \
                         save_local = False, eig_sum_to_1=False):

    all_eigenvalues = np.zeros((query_points.shape[0], 3))
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))
    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points
    if tree == None:
        tree = KDTree(cloud_points, leaf_size=60)              
    _,inds = tree.query(query_points,k=k)
    if save_local:
        all_local_clouds=[]
    for i,_ in enumerate(query_points):
        local_cloud=cloud_points[inds[i],:]
        all_eigenvalues[i], all_eigenvectors[i]= local_PCA(local_cloud)
        if save_local:
            all_local_clouds.append(local_cloud)
        #if i% 10000==0:
        #    print(i)
    
    #all_eigenvalues = np.divide(all_eigenvalues,np.sum(all_eigenvalues, axis=1)[:,None])
    if save_local:
        return all_eigenvalues, all_eigenvectors, all_local_clouds
    else:
        return all_eigenvalues, all_eigenvectors


# Computes all the features defined in Hackel et al. 2016.
def compute_features(query_points, cloud_points, computed=False,\
                     all_eigenvalues=None, all_eigenvectors=None, all_local_clouds=None):
    epsilon=1e-20
    # Compute the features for all query points in the cloud 
    if not(computed): # if eigenvalues/eigenvectors not already computed
        all_eigenvalues, all_eigenvectors, all_local_clouds=neighborhood_PCA(query_points,\
                                                         cloud_points, radius,True)

    all_eigenvalues[all_eigenvalues<=0.]= epsilon
    
    all_plane_eigenvectors = all_eigenvectors[:,:,-1]

    
    verticality = 2./np.pi*np.arcsin(np.abs(all_plane_eigenvectors[:,-1])) #<n,ez>=n[-1] because n is normed
    linearity = 1 -  all_eigenvalues[:,1]/all_eigenvalues[:,0]
    planarity =  (all_eigenvalues[:,1]-all_eigenvalues[:,2])/all_eigenvalues[:,0]
    sphericity = all_eigenvalues[:,2]/all_eigenvalues[:,0]

    s = np.sum(all_eigenvalues, axis=1)
    omnivariance= np.prod(all_eigenvalues, axis=1)**(1/3)
    eigenentropy = -(all_eigenvalues*np.log(all_eigenvalues)).sum(axis=1)
    anisotropy =  (all_eigenvalues[:,0]-all_eigenvalues[:,2])/all_eigenvalues[:,0]
    #surface_var= all_eigenvalues[:,2]/s
    
    covariances=np.c_[verticality, linearity, planarity, sphericity,\
                     omnivariance, eigenentropy, anisotropy, all_eigenvalues]
    
    order1_axis1=np.array([np.sum(np.dot(all_local_clouds[i]-p, all_eigenvectors[i,0])) \
                  for i,p in enumerate(query_points)])
    order1_axis2=np.array([np.sum(np.dot(all_local_clouds[i]-p, all_eigenvectors[i,1])) \
                  for i,p in enumerate(query_points)])
    order2_axis1=np.array([np.sum(np.dot(all_local_clouds[i]-p, all_eigenvectors[i,0])**2) \
                  for i,p in enumerate(query_points)])
    order2_axis2=np.array([np.sum(np.dot(all_local_clouds[i]-p, all_eigenvectors[i,1])**2) \
                  for i,p in enumerate(query_points)])
    
    moments=np.c_[order1_axis1, order1_axis2, order2_axis1, order2_axis2]
    
    vertical_range= np.array([np.max(x[:,2])- np.min(x[:,2]) for x in all_local_clouds])
    height_above= np.array([np.max(x[:,2])- query_points[i,2] for i,x in enumerate(all_local_clouds)])
    height_below= np.array([query_points[i,2]- np.min(x[:,2]) for i,x in enumerate(all_local_clouds)])
    
    #horiz_dist = np.array([np.max(np.linalg.norm(query_points[i,:2] - x[:,:2]))for i,x in enumerate(all_local_clouds)])
    m=min(cloud_points[:,2])
    total_height= np.array([q[2] - m  for q in query_points])
    
    heights=np.c_[vertical_range, height_above, height_below]
    
    return np.c_[covariances, moments, heights]

#
# Computes the features of the neighbors in the direction of the eigenvectors
#
def get_neighbor_features(query_points, cloud, features, all_eigenvalues, all_eigenvectors,tree, k=10):

    n_samples, n_var = features.shape
    #all_neighbor_features = []
    #for q, vals, vects in zip(query_points, all_eigenvalues, all_eigenvectors):
    #neighbor_features=[]
    approximation = False 
    directions=[1]
    neighbors= np.array([q + u * val * vect for q, vals, vects in \
                           zip(query_points, all_eigenvalues, all_eigenvectors) \
                           for u in directions\
                           for val, vect in zip(vals, vects) ])
    if approximation: 
        ###### VERSION 1 : search in queried points (approximation) #######
        tree_approx = KDTree(query_points, leaf_size = 60)

        inds = tree_approx.query(neighbors, k=1)[1]
        neighbor_features = features[inds]
        #a,_,b = neighbor_features.shape
        neighbor_features = neighbor_features[ :, 0, :].reshape((n_samples, 6 * n_var))
        
        
    else:
        ######### VERSION 2: search in all cloud ############

        all_eigenvalues, all_eigenvectors, all_local_clouds = \
                    neighborhood_PCA_knn(neighbors, cloud, k, tree, True, eig_sum_to_1 = True)
        
        radius=2 # not used 
        neighbor_features =  compute_features(neighbors, cloud,\
                                 True, all_eigenvalues, all_eigenvectors, all_local_clouds)
        
        #neighbor_features = np.c_[np.c_[basic_features],other_features]
        n_var = neighbor_features.shape[1]
        neighbor_features = neighbor_features.reshape((n_samples, 3 * len(directions) * n_var))
    #neighbor_features = features[tree.query([q + u * val * vect for u in [-1,1]\
    #    for val, vect in zip(vals, vects)], k=1)[1][0]]
    #all_neighbor_features.append(neighbor_features)
               
         #all_neighbor_features.append(np.array(neighbor_features).ravel())
    #all_neighbor_features = np.array(all_neighbor_features)
        
    return neighbor_features

def compute_all_features(query_points, cloud, voxel_sizes, k=10):
    
    #tree = KDTree(cloud, leaf_size = 60)
    #SC3D_features = SC3D(query_points, cloud, tree, n_azimuth=1, \
    #                n_elevation=5, r_min=0.1, r_max= 2, J=4)
    cloud_sub = cloud
    for j,voxel_size in enumerate(voxel_sizes):
        print("Voxel size: {}".format(voxel_size))
        cloud_sub=grid_subsampling(cloud_sub, voxel_size)
        tree = KDTree(cloud_sub, leaf_size = 60)
        all_eigenvalues, all_eigenvectors, all_local_clouds = \
            neighborhood_PCA_knn(query_points, cloud_sub, k, tree, True, eig_sum_to_1 = False)
        all_eigenvalues_sum1 = np.divide(all_eigenvalues,np.sum(all_eigenvalues, axis=1)[:,None])

        point_features = compute_features(query_points, cloud_sub,\
              True, all_eigenvalues_sum1, all_eigenvectors, all_local_clouds)
        new_features = np.c_[point_features]
    
        #neighbor_features= get_neighbor_features(query_points, cloud_sub,\
        #                                         new_features,\
        #                                         all_eigenvalues, all_eigenvectors, tree)
    
        if j==0:
            features = np.c_[new_features]
        else:
            features = np.c_[features, new_features]
    #features = np.c_[features, SC3D_features]
    
    return features
