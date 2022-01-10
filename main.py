#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:22:49 2018

@author: yannickterme
"""
#
# main
# 1. extract features
# 2. Cross validate
# 3. Train
# 4. Test



import numpy as np
from matplotlib import pyplot as plt

from functions import knn_smooth_predictions
from descriptors import compute_all_features
from extract import extract_data
from cross_val import cross_val
import time
from RANSAC import RANSAC, recursive_RANSAC

t0=time.time()

k=10
voxel_sizes=[0.025 * 2**i for i in range(9)]

paths = ['../data/Lille_street_{}.ply'.format(i_cloud) for i_cloud in range(1,4)]
#paths = ['/Users/yannickterme/Documents/MVA/NPM/projet/rueMadame_database/GT_Madame1_2.ply',
#        '/Users/yannickterme/Documents/MVA/NPM/projet/rueMadame_database/GT_Madame1_3.ply']
# initialize lists of data
clouds_query_list = [] ;features_query_list=[] ;labels_query_list=[] 
clouds_full=[]; labels_full = [] 

#ratio of subsampling
subsample = 1
# dataset is Rue-Madame
ismadame= True

for i_cloud, cloud_path in enumerate(paths):
    print("i_cloud =",i_cloud)
    
    #if i_cloud==0:
    cloud, labels, idx,query_points, query_labels = extract_data(\
                cloud_path, subsample = subsample, ismadame = ismadame, \
                sample_method = 'per_class', n_samples=1000)
    #elif i_cloud==1:
    #    cloud, labels, idx,query_points, query_labels = extract_data(\
    #            cloud_path, subsample = subsample, ismadame = ismadame, \
    #            sample_method = 'normal', n_samples= 100000)
    clouds_full.append(cloud)
    labels_full.append(labels)
    #idx_query_list.append(idx_query)
    
    clouds_query_list.append(query_points)
    labels_query_list.append(query_labels)
    
    #Compute features
    features = compute_all_features(query_points, cloud, voxel_sizes, k=k)
    features_query_list.append(features)
    
t1=time.time()
print("Features computation took {} seconds".format(int(t1-t0)))


features_query=np.r_[features_query_list]
labels_query=np.r_[labels_query_list]


depths = [4]
#depths= [10, 30]

############################
#Cross - validation
##########################
(all_mean_f1s, all_precisions, all_recalls) = cross_val(depths, clouds_query_list,\
                             features_query, labels_query, n_estimators= 50)
i_best=np.argmax(np.mean(all_mean_f1s, axis=1))
depth_best = depths[i_best]
print("Best depth: {}".format(depth_best))
print(all_mean_f1s[i_best])



############################
#Train
##########################

depth_best=4
X_train = np.concatenate(features_query_list[1:])# features_query_list[0]#
y_train =np.concatenate(labels_query_list[1:]) # labels_query_list[0]#

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=depth_best, random_state=0, n_estimators=50)
clf.fit(X_train, y_train)

############################
# Test
##########################
cloud_test = clouds_query_list[0]#np.concatenate(features_query_list)
X_test = features_query_list[0]#np.concatenate(features_query_list)
y_test = labels_query_list[0]

y_pred = clf.predict(X_test)
y_pred= knn_smooth_predictions(cloud_test, y_pred, n_iter=1, k=4)



plane_inds, remaining_inds, plane_labels =recursive_RANSAC(cloud_test, \
                            NB_RANDOM_DRAWS=1000, threshold_in=0.2, NB_PLANES=3)

for l in np.unique(plane_labels):
    inds_l = plane_inds[plane_labels==l]
    counts = np.bincount(y_pred[inds_l])
    print(counts)
    majority = np.argmax(counts)
    y_pred[inds_l] = majority
    

from sklearn.metrics import f1_score
f1_scores = f1_score(y_pred,y_test, average=None)
print("Mean F1-score: {}".format(np.mean(f1_scores)))
print(f1_scores)



# Write data
write_ply('../data/Lillestreet3_test.ply', [cloud_test, y_pred], \
                      ['x', 'y', 'z', 'label'])
