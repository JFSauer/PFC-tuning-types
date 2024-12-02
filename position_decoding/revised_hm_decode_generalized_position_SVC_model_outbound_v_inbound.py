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

target_dir="/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/generalized_position_decoding/outbound_v_inbound"
source_dir="/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/extraction_of_neurons"
target_filename="decoding_generalized_position_with_specific_types_by_mouse_SVC_model_outbound_v_inbound"

cell_number=25 #Titrated to work with the set of mice listed below.
iterations=10

C=3
tol=0.000000001
cv=5
max_iter=1000000000
bins=N.linspace(0,2,11)
selected_mice=['44','216','219','478','481','483','485']

# Decoding function.
def run_decoding(X,y,ind,cell_number=cell_number,random=False):
    X_temp=N.empty((len(ind),len(y)))   
    for i in range(len(ind)):
        X_temp[i]=X[ind[i]]
  
    X_final=X_temp.T

    
    pred_local=cross_val_predict(clf,X_final,y,cv=kf)
    if random==True:
        scoring=mean_squared_error(N.random.permutation(y),pred_local)
    else:
        scoring=mean_squared_error(y,pred_local)
    return pred_local,scoring

score_cat_out=[]
score_same_out=[]
score_side_out=[]

score_cat_in=[]
score_same_in=[]
score_side_in=[]

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
    neurons1=data['X left']
    pos1=data['position left']
    neurons2=data['X right']
    pos2=data['position right']
    
    neurons=N.append(neurons1,neurons2,axis=1)
    pos=N.append(N.asarray(pos1),N.asarray(pos2))
    
    ind_out=N.ravel(N.where(N.asarray(pos)<1))
    ind_in=N.ravel(N.where(N.asarray(pos)>1))

    neurons_out=neurons[:,ind_out]
    pos_out=pos[ind_out]
    neurons_in=neurons[:,ind_in]
    pos_in=pos[ind_in]
       
    pos_in=N.digitize(pos_in,bins)
    pos_out=N.digitize(pos_out,bins)
    
    # Get all neurons that belong to an identified functional type.
    ind_same=classification_all_mice['mouse indices']['indices same'][n]   
    ind_side=classification_all_mice['mouse indices']['indices side'][n]    
    ind_cat=classification_all_mice['mouse indices']['indices cat'][n]  
       
    score_side_local_in,score_side_local_out=[],[]
    score_same_local_in,score_same_local_out=[],[]
    score_cat_local_in,score_cat_local_out=[],[]   
    
    for t in range(iterations):
        # Side model.       
        ind=N.random.choice(ind_side,size=cell_number,replace=False)
        
        pred_local,scoring=run_decoding(neurons_in,pos_in,ind,cell_number=cell_number,random=False)       
        score_side_local_in.append(scoring)
        pred_local,scoring=run_decoding(neurons_out,pos_out,ind,cell_number=cell_number,random=False)       
        score_side_local_out.append(scoring)
        
        # Same model.
        ind=N.random.choice(ind_same,size=cell_number,replace=False)
        pred_local,scoring=run_decoding(neurons_in,pos_in,ind,cell_number=cell_number,random=False)       
        score_same_local_in.append(scoring)
        pred_local,scoring=run_decoding(neurons_out,pos_out,ind,cell_number=cell_number,random=False)       
        score_same_local_out.append(scoring)
        
        # Cat model.
        ind=N.random.choice(ind_cat,size=cell_number,replace=False)
        pred_local,scoring=run_decoding(neurons_in,pos_in,ind,cell_number=cell_number,random=False)       
        score_cat_local_in.append(scoring)
        pred_local,scoring=run_decoding(neurons_out,pos_out,ind,cell_number=cell_number,random=False)       
        score_cat_local_out.append(scoring)
        
    score_cat_in.append(N.nanmean(score_cat_local_in))
    score_same_in.append(N.nanmean(score_same_local_in))
    score_side_in.append(N.nanmean(score_side_local_in))
    
    score_cat_out.append(N.nanmean(score_cat_local_out))
    score_same_out.append(N.nanmean(score_same_local_out))
    score_side_out.append(N.nanmean(score_side_local_out))
    
res_out={'score same':N.asarray(score_same_out),'score side':N.asarray(score_side_out),'score cat':N.asarray(score_cat_out)}
res_in={'score same':N.asarray(score_same_in),'score side':N.asarray(score_side_in),'score cat':N.asarray(score_cat_in)}

stats_side=olf.pairwise_test(res_out['score side'],res_in['score side'])
stats_same=olf.pairwise_test(res_out['score same'],res_in['score same'])
stats_cat=olf.pairwise_test(res_out['score cat'],res_in['score cat'])

res_total={'outward':res_out,'inward':res_in,'stats side':stats_side,'stats same':stats_same,'stats cat':stats_cat}


os.chdir(target_dir)    
N.save(target_filename,res_total)

