#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 14:30:53 2018

@author: yannickterme
"""


import numpy as np
from tqdm import tqdm

######## Compute the SC3D features of the array query_points #######
def SC3D(query_points, cloud, tree, n_azimuth=5, n_elevation=5, r_min=0.1, r_max= 1, J=2):
    # neighbors of query points
    inds = tree.query_radius(query_points, r_max)
    distances = [np.linalg.norm(q - cloud[inds[i]], axis=1) for i,q in enumerate(query_points)]
    
    # remove the points at a distance 0
    inds = np.array([i[d>0] for i,d in zip(inds,distances)])
    distances = np.array([d[d>0] for d in distances])
    
    
    # compute densities rho for all points defined as the number of neighbors in a sphere
    delta = 0.5
    inds_cloud = tree.query_radius(cloud, delta)
    densities = [len(x) for x in inds_cloud]

    # define the ranges of psi, theta, R
    azimuths = np.arange(-np.pi, np.pi, 2*np.pi/n_azimuth)
    elevations = np.arange(0, np.pi+1e-10, np.pi/n_elevation)
    R_list = np.array(np.exp([np.log(r_min)+j/(J-1)*np.log(r_max/r_min) for j in range(J)]))
    
    #initialize the histogram to 0
    total_bins = np.zeros((len(query_points), J, n_elevation, n_azimuth ))
    
    ## For each point of the neighborhood, compute the histogram
    for i_q, q in tqdm(enumerate(query_points)):
        
        idx_neighb_q = inds[i_q]
        neighbors, neighb_dist = cloud[idx_neighb_q], distances[i_q]

        neighb_bins= np.digitize(neighb_dist, R_list) 
        x_centered, y_centered = q[0] - neighbors[:,0], q[1]-neighbors[:,1]
        mask1, mask2= (y_centered<0) & (x_centered>0), (y_centered<0) & (x_centered<0)
        # Angle between vertical and vector (azimuth)
        psi = np.arctan(x_centered/y_centered)
        psi[mask1] = np.pi/2- np.arctan(x_centered[mask1]/ y_centered[mask1])
        psi[mask2] = -np.pi/2 - np.arctan(x_centered[mask2]/y_centered[mask2])        
        
        z_centered = q[2] - neighbors[:, 2]
        # Angle between horizontal plane and vector (elevation)
        theta = np.arccos(z_centered/np.linalg.norm(np.c_[x_centered,y_centered,\
                                                          z_centered], axis=1))
        # Compte bins of each angle
        inds_psi = np.digitize(psi, azimuths) - 1
        inds_theta = np.digitize(theta, elevations[:-1]) - 1
        assert((inds_psi>=0).all())
        assert((inds_theta>=0).all())
        

        
        # for each scale, iterate over neighbors and update histogram
        for i_R, R in enumerate(R_list):
            for i_n, n in enumerate(neighbors[neighb_bins==i_R]):
                total_bins[i_q, i_R, inds_theta[i_n], inds_psi[i_n]] += \
                    1/densities[idx_neighb_q[i_n]]
                    
    # this variable will be used for the normalization of the bin values
    volumes = np.diff(1-np.cos(elevations))**(1/3)  
                  
    # normalize with volume
    for i_R, R in enumerate(R_list):       
        total_bins[: , i_R , : , : ] /= R 
    for i_th in range(n_elevation):
        total_bins[: , : , i_th , : ] /= volumes[i_th]
                
                    
        
    return total_bins.reshape((len(query_points), n_azimuth*n_elevation*J))

                    
        
        
