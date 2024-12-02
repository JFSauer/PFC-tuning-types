#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 10:44:35 2021

@author: jonas
"""

import numpy as N
import os
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVC
import Muysers_et_al_helper_functions as olf
from sklearn.model_selection import cross_val_predict

'''
Requires output of stepwise_cell_classification as input. Paths need to be adjusted before use.
'''

N.random.seed(42)

target_dir="/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/absolute_position_decoding/using_single_neurons"
source_dir="/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/extraction_of_neurons"
target_filename="decoding_absolute_position_with_specific_types_by_mouse_SVC_model_single_neurons"

C=3
tol=0.000000001
cv=5
max_iter=1000000000
bins=N.linspace(0,2,11)
selected_mice=['44','216','219','478','481','483','485']

# Decoding function.
def run_decoding(X,y,ind):
    X_temp=N.empty((2,len(X[0])))   
    X_temp[0]=X[ind]
    X_temp[1]=X[ind]
    
    X=X_temp.T
    
    pred_local=cross_val_predict(clf,X,y,cv=kf)

    scoring=mean_squared_error(y,pred_local)
    return pred_local,scoring

score_full,score_full_rand=[],[]
score_cat,score_cat_rand=[],[]
score_same,score_same_rand=[],[]
score_side,score_side_rand=[],[]

clf=LinearSVC(C=C,max_iter=max_iter,tol=tol)
fold_iterator=KFold(n_splits=cv,shuffle=True)
kf=KFold(n_splits=cv,shuffle=True)

predicted_full,predicted_same,predicted_side,predicted_cat=[],[],[],[]
true_full=[]
pos_full=[]

y_pred_same,y_pred_side,y_pred_cat=[],[],[]

classification_all_mice=N.load("/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/cell_classification/cell_clustering_results_all_mice.npy",allow_pickle=True).item()

for n in range(len(selected_mice)):
    os.chdir(source_dir)
    print(selected_mice[n])
    
    data=N.load("%s_results_spatial_tuning.npy" %selected_mice[n],allow_pickle=True).item()
    neurons_left=data['X left']
    neurons_right=data['X right']
    
    pos_raw=data['position']
    pos=pos_raw
    for t in range(len(pos_raw)):
        if 0>pos_raw[t]>-0.5:
            pos[t]=pos[t]*-1
        if -1.5>pos[t]>-2.1:
            pos[t]=pos[t]*-1
       
    neurons=N.append(neurons_left,neurons_right,axis=1)
    
    y=N.digitize(pos,bins)
    true_full.append(y)
    
    score_full_local,score_full_rand_local=[],[]
    score_cat_local,score_cat_rand_local=[],[]
    score_same_local,score_same_rand_local=[],[]
    score_side_local,score_side_rand_local=[],[]
    
    # Get all neurons that belong to an identified functional type.
    ind_same=classification_all_mice['mouse indices']['indices same'][n]   
    ind_side=classification_all_mice['mouse indices']['indices side'][n]    
    ind_cat=classification_all_mice['mouse indices']['indices cat'][n]  
      
    for t in range(len(ind_side)):
        # Side model.       
        pred_local,scoring=run_decoding(neurons,y,ind_side[t])
        score_side.append(scoring)
    for t in range(len(ind_same)):
        # Side model.       
        pred_local,scoring=run_decoding(neurons,y,ind_same[t])
        score_same.append(scoring)
    for t in range(len(ind_cat)):
        # Side model.       
        pred_local,scoring=run_decoding(neurons,y,ind_cat[t])
        score_cat.append(scoring)
        
res={'score same':N.asarray(score_same),'score side':N.asarray(score_side),'score cat':N.asarray(score_cat),
           'score same shuff':N.asarray(score_same_rand),'score side shuff':N.asarray(score_side_rand),'score cat shuff':N.asarray(score_cat_rand),
            'position target':pos_full,'predicted same':predicted_same,'predicted side':predicted_side,'predicted cat':predicted_cat,  
           'true full':true_full,
           'C':C,'cv':cv,'max_iter':max_iter}

# Run 1-way ANOVA on data.
stat=olf.one_way_anova_general(res['score side'],res['score same'],res['score cat'])
res['1-way ANOVA']=stat


res_total=res
os.chdir(target_dir)    
N.save(target_filename,res)

