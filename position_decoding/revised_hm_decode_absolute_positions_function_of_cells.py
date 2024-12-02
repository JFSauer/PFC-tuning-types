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

target_dir="/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/absolute_position_decoding/as_function_of_neurons"
source_dir="/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/extraction_of_neurons"
target_filename="decoding_absolute_position_with_specific_types_by_mouse_SVC_model_function_of_neurons"

cell_number_list=[2,5,10,15,20] #Titrated to work with the set of mice listed below.
iterations=10

C=3
tol=0.000000001
cv=5
max_iter=1000000000
bins=N.linspace(0,2,11)
max_iter=1000000000

selected_mice=['44','216','219','478','481','483','485']

# Decoding function.
def run_decoding(X,y,ind,random=False):
    X_temp=N.empty((len(ind),len(neurons[ind[0]])))   
    for i in range(len(ind)):
        X_temp[i]=neurons[ind[i]]
  
    X=X_temp.T
    
    pred_local=cross_val_predict(clf,X,y,cv=kf)
    if random==True:
        scoring=mean_squared_error(N.random.permutation(y),pred_local)
    else:
        scoring=mean_squared_error(y,pred_local)
    return pred_local,scoring

score_cat=N.empty((len(selected_mice),len(cell_number_list))) # A mice x neurons array for saving.
score_same=N.empty_like(score_cat)
score_side=N.empty_like(score_cat)

clf=LinearSVC(C=C,max_iter=max_iter,tol=tol)
fold_iterator=KFold(n_splits=cv,shuffle=True)
kf=KFold(n_splits=cv,shuffle=True)

classification_all_mice=N.load("/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/cell_classification/cell_clustering_results_all_mice.npy",allow_pickle=True).item()

for animal in range(len(selected_mice)):
    os.chdir(source_dir)
    print(selected_mice[animal])
    
    data=N.load("%s_results_spatial_tuning.npy" %selected_mice[animal],allow_pickle=True).item()
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
    
    score_full_local=[]
    score_cat_local=[]
    score_same_local=[]
    score_side_local=[]
    
    # Get all neurons that belong to an identified functional type.
    ind_same=classification_all_mice['mouse indices']['indices same'][animal]   
    ind_side=classification_all_mice['mouse indices']['indices side'][animal]    
    ind_cat=classification_all_mice['mouse indices']['indices cat'][animal]  
       
    for n_neurons in range(len(cell_number_list)):
        cell_number=cell_number_list[n_neurons]
        
        score_cat_local=[]
        score_same_local=[]
        score_side_local=[]
        
        for t in range(iterations):
            # Side model.       
            ind=N.random.choice(ind_side,size=cell_number,replace=False)
            pred_local,scoring=run_decoding(neurons,y,ind,random=False)
            score_side_local.append(scoring)
             
            # Same model.
            ind=N.random.choice(ind_same,size=cell_number,replace=False)
            pred_local,scoring=run_decoding(neurons,y,ind,random=False)
            
            score_same_local.append(scoring)
            
            # Cat model.
            ind=N.random.choice(ind_cat,size=cell_number,replace=False)
            pred_local,scoring=run_decoding(neurons,y,ind,random=False)
            
            score_cat_local.append(scoring)       
            
        score_cat[animal][n_neurons]=N.nanmean(score_cat_local)
        score_same[animal][n_neurons]=N.nanmean(score_same_local)
        score_side[animal][n_neurons]=N.nanmean(score_side_local)

score_same_stats=N.asarray(score_same).T
score_side_stats=N.asarray(score_side).T
score_cat_stats=N.asarray(score_cat).T

p,F,anovas=[],[],[]
for n in range(len(score_same_stats)):
    stat=olf.one_way_repeated_measures_anova_general_three_groups(score_side_stats[n],score_same_stats[n],score_cat_stats[n])
    anovas.append(stat)
    p.append(stat[0]['p-unc'][0])
    F.append(stat[0]['F'][0])
    
stats={'p':p,'F':F,'ANOVAS':anovas}   

res={'score same':N.asarray(score_same),'score side':N.asarray(score_side),'score cat':N.asarray(score_cat),
     'neuron number':N.asarray(cell_number_list),'stats':stats,
     'C':C,'cv':cv,'max_iter':max_iter,'used neurons':cell_number,'iterations':iterations}


res_total=res
os.chdir(target_dir)    
N.save(target_filename,res)

