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
from sklearn.model_selection import cross_val_predict
import scipy.stats as st

'''
Requires output of stepwise_cell_classification as input. Paths need to be adjusted before use.
'''

N.random.seed(42)

target_dir="/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/absolute_position_decoding/random_draw"
source_dir="/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/extraction_of_neurons"
target_filename="decoding_absolute_position_with_specific_types_by_mouse_SVC_model_random_draw"

cell_number=20
iterations=50

C=3
tol=0.000000001
cv=5
max_iter=1000000000

bins=N.linspace(0,2,11)
selected_mice=['44','216','219','478','481','483','485']

# Decoding function.
def run_decoding(X,y,ind):
    X_temp=N.empty((len(ind),len(neurons[ind[0]])))   
    for i in range(len(ind)):
        X_temp[i]=neurons[ind[i]]
  
    X=X_temp.T
    
    pred_local=cross_val_predict(clf,X,y,cv=kf)

    scoring=mean_squared_error(y,pred_local)
    return pred_local,scoring

def iterate(neurons,y,ind_same,ind_side,ind_cat,iterations=iterations,cell_number=cell_number):
    score_full=[]
    prop_side,prop_same,prop_cat=[],[],[]
       
    for t in range(iterations):
        # Side model.
        ind=N.random.choice(range(len(neurons)),size=cell_number,replace=False)
        pred_local,scoring=run_decoding(neurons,y,ind)
        score_full.append(scoring)
        # Get cell numbers in the current selection.
        prop_side.append(len(list(set(ind) & set(ind_side)))/cell_number)
        prop_same.append(len(list(set(ind) & set(ind_same)))/cell_number)
        prop_cat.append(len(list(set(ind) & set(ind_cat)))/cell_number)
       
    res={'score full':score_full,'prop side':prop_side,'prop cat':prop_cat,'prop same':prop_same}
    return res


clf=LinearSVC(C=C,max_iter=max_iter,tol=tol)
fold_iterator=KFold(n_splits=cv,shuffle=True)
kf=KFold(n_splits=cv,shuffle=True)

predicted_full,predicted_same,predicted_side,predicted_cat=[],[],[],[]
true_full=[]
pos_full=[]

y_pred_same,y_pred_side,y_pred_cat=[],[],[]

classification_all_mice=N.load("/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/cell_classification/cell_clustering_results_all_mice.npy",allow_pickle=True).item()
score_full=[]
prop_side,prop_same,prop_cat=[],[],[]
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
       
    res=iterate(neurons,y,ind_same,ind_side,ind_cat,iterations=iterations,cell_number=cell_number)
    prop_same.extend(res['prop same'])
    prop_side.extend(res['prop side'])
    prop_cat.extend(res['prop cat'])
    score_full.extend(res['score full'])
    
score=N.asarray(score_full)
prop_side=N.asarray(prop_side)
prop_same=N.asarray(prop_same)
prop_cat=N.asarray(prop_cat)

r_side,p_side=st.spearmanr(score,prop_side)
r_same,p_same=st.spearmanr(score,prop_same)
r_cat,p_cat=st.spearmanr(score,prop_cat)

res_total={'score':N.asarray(score_full),'prop side':N.asarray(prop_side),'prop cat':N.asarray(prop_cat),'prop same':N.asarray(prop_same),
           'C':C,'cv':cv,'max_iter':max_iter,'used neurons':cell_number,'iterations':iterations,
           'r side':r_side,'p side':p_side,'r same':r_same,'p same':p_same,'r cat':r_cat,'p cat':p_cat}

os.chdir(target_dir)    
N.save(target_filename,res)

