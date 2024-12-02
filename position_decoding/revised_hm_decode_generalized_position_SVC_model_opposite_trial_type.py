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

target_dir="/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/generalized_position_decoding/opposite_trial_type"
source_dir="/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/extraction_of_neurons"
target_filename="decoding_generalized_position_with_specific_types_by_mouse_SVC_model_opposite_trial_type"

cell_number=25 #Titrated to work with the set of mice listed below.
iterations=10

C=3
tol=0.000000001
cv=5
max_iter=1000000000
bins=N.linspace(0,2,11)
selected_mice=['44','216','219','478','481','483','485']

# Decoding function.
def run_decoding(neurons1,neurons2,y1,y2,ind,random=False):
    X1_temp=N.empty((len(ind),len(neurons1[ind[0]])))   
    X2_temp=N.empty((len(ind),len(neurons2[ind[0]])))   
    for i in range(len(ind)):
        X1_temp[i]=neurons1[ind[i]]
        X2_temp[i]=neurons2[ind[i]]
   
    X1=X1_temp.T
    X2=X2_temp.T
           
    clf.fit(X1,y1)
    pred_local1=clf.predict(X2)
    if random==True:
        scoring1=mean_squared_error(N.random.permutation(y2),pred_local1)
    else:
        scoring1=mean_squared_error(y2,pred_local1)
    
    clf.fit(X2,y2)
    pred_local2=clf.predict(X1)
    if random==True:
        scoring2=mean_squared_error(N.random.permutation(y1),pred_local2)
    else:
        scoring2=mean_squared_error(y1,pred_local2)
    
    scoring=N.nanmean([scoring1,scoring2])
    return pred_local2,scoring

score_cat=[]
score_same=[]
score_side=[]

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
    pos_left=data['position left']

    neurons_right=data['X right']
    pos_right=data['position right']

    pos_left=N.digitize(pos_left,bins)
    pos_right=N.digitize(pos_right,bins)

    # Get all neurons that belong to an identified functional type.
    ind_same=classification_all_mice['mouse indices']['indices same'][n]   
    ind_side=classification_all_mice['mouse indices']['indices side'][n]    
    ind_cat=classification_all_mice['mouse indices']['indices cat'][n]  
       
    score_side_local=[]
    score_same_local=[]
    score_cat_local=[]
    
    for t in range(iterations):
        # Side model.       
        ind=N.random.choice(ind_side,size=cell_number,replace=False)
        
        pred_local,scoring=run_decoding(neurons_left,neurons_right,pos_left,pos_right,ind,random=False)       
        score_side_local.append(scoring)
                
        # Same model.
        ind=N.random.choice(ind_same,size=cell_number,replace=False)
        pred_local,scoring=run_decoding(neurons_left,neurons_right,pos_left,pos_right,ind,random=False)       
        score_same_local.append(scoring)
        
        # Cat model.
        ind=N.random.choice(ind_cat,size=cell_number,replace=False)
        pred_local,scoring=run_decoding(neurons_left,neurons_right,pos_left,pos_right,ind,random=False)       
        score_cat_local.append(scoring)
        
    score_cat.append(N.nanmean(score_cat_local))
    score_same.append(N.nanmean(score_same_local))
    score_side.append(N.nanmean(score_side_local))
    

res_total={'score same':N.asarray(score_same),'score side':N.asarray(score_side),'score cat':N.asarray(score_cat),
            'position target':pos_full,'predicted same':predicted_same,'predicted side':predicted_side,'predicted cat':predicted_cat,  
           'true full':true_full,
           'C':C,'cv':cv,'max_iter':max_iter,'used neurons':cell_number,'iterations':iterations}

stat=olf.one_way_repeated_measures_anova_general_three_groups(res_total['score side'],res_total['score same'],res_total['score cat'])
res_total['1-way ANOVA']=stat
os.chdir(target_dir)    
N.save(target_filename,res_total)

