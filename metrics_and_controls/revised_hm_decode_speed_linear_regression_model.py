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
from sklearn.linear_model import LinearRegression
import Muysers_et_al_helper_functions as olf
from sklearn.model_selection import cross_val_predict
import pandas as pd
import pingouin as pg
import scipy.stats as st

N.random.seed(42)

target_dir="/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/decoding_speed"
source_dir="/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/extraction_of_neurons"
target_filename="decoding_speed_with_specific_types_by_mouse_linearRegression_model"

cell_number=25 #Titrated to work with the set of mice listed below.
iterations=10

C=3
tol=0.000000001
cv=5
ds=1
max_iter=1000000000

selected_mice=['44','216','219','478','481','483','485']

# Decoding function.
def run_decoding(neurons,y,ind,cell_number=cell_number,random=False):
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

score_full,score_full_rand=[],[]
score_cat,score_cat_rand=[],[]
score_same,score_same_rand=[],[]
score_side,score_side_rand=[],[]

clf=LinearRegression()
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
    neurons=data['X']
    y=data['speed total']
    
    
    score_full_local,score_full_rand_local=[],[]
    score_cat_local,score_cat_rand_local=[],[]
    score_same_local,score_same_rand_local=[],[]
    score_side_local,score_side_rand_local=[],[]
    
    # Get all neurons that belong to an identified functional type.
    ind_same=classification_all_mice['mouse indices']['indices same'][n]   
    ind_side=classification_all_mice['mouse indices']['indices side'][n]    
    ind_cat=classification_all_mice['mouse indices']['indices cat'][n]  
       
    for t in range(iterations):
        # Side model.       
        ind=N.random.choice(ind_side,size=cell_number,replace=False)
        pred_local,scoring=run_decoding(neurons,y,ind,cell_number=cell_number,random=False)
        
        score_side_local.append(scoring)
 
        # Same model.
        ind=N.random.choice(ind_same,size=cell_number,replace=False)
        pred_local,scoring=run_decoding(neurons,y,ind,cell_number=cell_number,random=False)
 
        score_same_local.append(scoring)
 
        # Cat model.
        ind=N.random.choice(ind_cat,size=cell_number,replace=False)
        pred_local,scoring=run_decoding(neurons,y,ind,cell_number=cell_number,random=False)
 
        score_cat_local.append(scoring)
  
        
    score_cat.append(N.nanmean(score_cat_local))
    score_same.append(N.nanmean(score_same_local))
    score_side.append(N.nanmean(score_side_local))
    
res={'score same':N.asarray(score_same),'score side':N.asarray(score_side),'score cat':N.asarray(score_cat),
            'position target':pos_full,'predicted same':predicted_same,'predicted side':predicted_side,'predicted cat':predicted_cat,  
           'true full':true_full,
           'C':C,'cv':cv,'max_iter':max_iter,'used neurons':cell_number,'iterations':iterations}

# Run 1-way ANOVA on data.
stat=olf.one_way_repeated_measures_anova_general_three_groups(res['score side'],res['score same'],res['score cat'])
res['1-way ANOVA']=stat

res_total=res
os.chdir(target_dir)    
N.save(target_filename,res)

