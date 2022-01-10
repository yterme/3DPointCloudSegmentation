#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 16:06:39 2018

@author: yannickterme
"""
from functions import knn_smooth_predictions
from sklearn.neighbors import KDTree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, classification_report,\
                    precision_score, recall_score
import numpy as np
from utils.ply import write_ply, read_ply 


def cross_val(depths, clouds_query_list, features_query_list, labels_query_list, n_estimators=200):
    all_mean_f1s=[]
    all_recalls=[]
    all_precisions = []
    k_smooth = 4
    n_train = len(clouds_query_list) 
    for depth in depths:
        clf = RandomForestClassifier(max_depth=depth, random_state=0, n_estimators=n_estimators)
    
        f1_scores_depth = []
        recalls_depth = []
        precisions_depth = []
        for i_val in range(n_train):
            ii_train = [j for j in range(n_train)]
            del(ii_train[i_val])
            X_train = np.concatenate(features_query_list[np.array(ii_train)])
            y_train = np.concatenate(labels_query_list[np.array(ii_train)])
            X_val, y_val = features_query_list[i_val], labels_query_list[i_val]
            
            clf.fit(X_train, y_train)
            
            y_pred= clf.predict(X_val)
    
            cloud_val = clouds_query_list[i_val]
    
            #clouds_train = [clouds_query_list[j] for j in ii_train]
            #y_train_list = [abels_full[j][idx_list[j]] for j in i_val]
    
            y_pred= knn_smooth_predictions(cloud_val, y_pred, n_iter=5, k=k_smooth)
          

            f1_scores = f1_score(y_pred,y_val, average=None)
            precision = precision_score(y_pred,y_val, average=None)
            recall = recall_score(y_pred,y_val, average=None)
            
            f1_scores_depth.append(f1_scores)
            recalls_depth.append(recall)
            precisions_depth.append(precision)
        mean_f1_scores = np.mean(np.array(f1_scores_depth),axis=0)
        mean_precisions =  np.mean(np.array(precisions_depth),axis=0)
        mean_recalls =  np.mean(np.array(precisions_depth),axis=0)
    
        all_mean_f1s.append(mean_f1_scores)
        all_precisions.append(mean_precisions)
        all_recalls.append(mean_recalls)
        print("Average F1 per class:", np.mean(mean_f1_scores))
        
    return(all_mean_f1s, all_precisions, all_recalls)