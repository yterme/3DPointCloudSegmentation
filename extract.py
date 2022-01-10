#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 15:16:37 2018

@author: yannickterme
"""
from utils.ply import write_ply, read_ply 
import numpy as np
from dicts import madame2lille, madamedict

def extract_data(cloud_path, subsample = 1, ismadame=False, sample_method='per_class',\
                  n_samples=1000):

    cloud_ply = read_ply(cloud_path)
    cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
    if ismadame:
        labels= cloud_ply['class']
        labels = np.vectorize(madamedict.get)(labels)
    else:
        labels= cloud_ply['labels']
    
    cloud = cloud[labels>0]
    labels = labels[labels>0]
    
    assert(subsample>0)
    assert(subsample<=1)
    n = len(cloud)
    idx_subsample = np.random.choice(range(n),int(subsample*n), replace= False)
    cloud = cloud[idx_subsample]
    labels = labels[idx_subsample]
    n = len(cloud)

    
    classes=np.unique(labels[labels>0]) #we leave class 0 (unknown)
    idx_query=[]
    # Sample 1000 points per class
    if sample_method == 'per_class':
        for cl in classes:
            all_idx_class = np.where(labels==cl)[0]
            if len(all_idx_class)<n_samples:
                idx_class=np.random.choice(all_idx_class, n_samples, replace = True)
            else:
                idx_class=np.random.choice(all_idx_class, n_samples,\
                        replace = False)
            idx_query=np.r_[idx_query,idx_class]
        idx_query=np.array(idx_query, dtype=int)
    else:
        if n_samples==-1: 
            idx_query = range(n)
        else:
            idx_query = np.random.choice(range(n),\
                n_samples, replace= False)
    query_points=cloud[idx_query]
    query_labels= labels[idx_query]
    return cloud, labels, idx_query, query_points, query_labels
    