#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 17:13:36 2018

@author: yannickterme
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KDTree
from sklearn.ensemble import RandomForestClassifier
import numpy as np

#
# smoothing function, works with a kNN
#
def knn_smooth_predictions(X, y, k=10, n_iter= 3):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X, y)
    y_pred = neigh.predict(X)
    for i in range(n_iter-1):
        neigh.fit(X, y_pred)
        y_pred = neigh.predict(X)
    return(y_pred)
    
from sklearn.neighbors import kneighbors_graph


#
# Attempt of a more complicated smoothing function but did not perform well.
#
def correct_with_neighborhood(clouds_train, y_train_list, cloud_val, y_pred, k=50):
 
   neighb_classes_train=[]
   classes = np.unique(np.array(y_train_list))
   for i_cloud, cloud_train in enumerate(clouds_train):
       tree= KDTree(cloud_train, leaf_size=10)
       nn = tree.query(cloud_train, k=50)[1]
       nc = y_train_list[i][nn]
       
       nc=np.array([np.sum(nc==c, axis=1) for c in classes]).T
       neighb_classes_train.append(nc)
   neighb_classes_train = np.array(neighb_classes_train)\
       .reshape((np.sum([len(x) for x in clouds_train]), len(classes)))
   
   clf = RandomForestClassifier()
   clf.fit(neighb_classes_train, np.array(y_train_list).ravel())
   
   #
   tree= KDTree(cloud_val, leaf_size=40)
   nn_pred = nn = tree.query(cloud_val, k=k)[1]
   npred = y_pred[nn_pred]
   neighb_classes_pred=np.array([np.sum(npred==c, axis=1) for c in classes]).T
   y_pred_out = y_pred.copy()
   y_pred_pred = clf.predict(neighb_classes_pred)
   
   probas = clf.predict_proba(neighb_classes_pred)
   #maxes = np.max(probas, axis=1)
   probas_of_preds= np.array([probas[i,x-1] for i,x in enumerate(y_pred)])
   threshold=0.1
   y_pred_out[probas_of_preds<threshold] = y_pred_pred[probas_of_preds<threshold]
   
   print(np.mean(y_pred_out==y_val))
   print(np.mean(y_pred==y_val))
   
   
   return y_pred_out
