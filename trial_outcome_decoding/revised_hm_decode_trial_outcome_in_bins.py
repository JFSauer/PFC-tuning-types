#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 10:44:35 2021

@author: jonas
"""


import numpy as N
import os
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import Muysers_et_al_helper_functions as olf
from sklearn.model_selection import cross_val_predict

'''
Requires output of stepwise_cell_classification as input. Paths need to be adjusted before use.
'''

N.random.seed(42)

target_dir="/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/trial_type_prediction/trial_type_prediction_position_binned"
source_dir="/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/extraction_of_neurons"

cell_number=25 #Titrated to work with the set of mice listed below.
iterations=10

C=3
tol=0.000000001
cv=10
max_iter=1000000000
bins=N.linspace(0,2,21)
bin_centers=[]
for n in range(len(bins)-1):
    bin_centers.append(N.mean([bins[n+1],bins[n]]))

selected_mice=['44','216','219','478','481','483','485']

# Helper function.
def get_runs_all(pos,thres1=1.7,thres2=0.3):
    ev=[]
    # Get start points.
    for n in range(1,len(pos)-1,1):
        if pos[n-1]>thres1 and pos[n]<thres2:
            ev.append(n)
    # append end points.
    new=[]
    for n in ev:
        new.append(n-1)    
    all_points=[]
    all_points.extend(ev)
    all_points.extend(new)
    all_points.append(0)
    all_points.append(len(pos))
    all_points=N.sort(all_points)
             
    return all_points

# Decoding function.
def run_decoding(X,y,ind,cell_number=cell_number,random=False):
    X_temp=N.empty((len(ind),len(neurons[ind[0]])))   
    for i in range(len(ind)):
        X_temp[i]=neurons[ind[i]]
  
    X=X_temp.T
    
    pred_local=cross_val_predict(clf,X,y,cv=kf)
    if random==True:
        scoring=accuracy_score(N.random.permutation(y),pred_local)
    else:
        scoring=accuracy_score(y,pred_local)
    return pred_local,scoring

score_cat_total=[]
score_same_total=[]
score_side_total=[]


clf=LogisticRegression(C=C,max_iter=max_iter,tol=tol)
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
    y=N.zeros((len(neurons_left[0])))
    neurons_right=data['X right']
    y=N.append(y,N.ones((len(neurons_right[0]))))
    neurons=N.append(neurons_left,neurons_right,axis=1)
    pos=data['position']
    pos=N.abs(pos)
    true_full.append(y)
    
    score_full_local,score_full_rand_local=[],[]
    score_cat_local,score_cat_rand_local=[],[]
    score_same_local,score_same_rand_local=[],[]
    score_side_local,score_side_rand_local=[],[]
    
    # Get all neurons that belong to an identified functional type.
    ind_same=classification_all_mice['mouse indices']['indices same'][n]   
    ind_side=classification_all_mice['mouse indices']['indices side'][n]    
    ind_cat=classification_all_mice['mouse indices']['indices cat'][n]  
       
    score_cat,score_cat_rand=[],[]
    score_same,score_same_rand=[],[]
    score_side,score_side_rand=[],[]
    
    
    # Get X and y data as averages of position bins.
    labels,X_binned=[],[]
    for l in range(len(bins)-1):
        print(l)
        ind_bin=[]
        for k in range(len(pos)):
            if bins[l]<pos[k]<bins[l+1]:
                ind_bin.append(k)

        ind_bin=N.asarray(ind_bin)
        labels=y[ind_bin]
        X_binned=neurons[:,ind_bin]
        
        
        score_full_local,score_full_rand_local=[],[]
        score_cat_local,score_cat_rand_local=[],[]
        score_same_local,score_same_rand_local=[],[]
        score_side_local,score_side_rand_local=[],[]
        for t in range(iterations):
       
            # Side model.
            ind=N.random.choice(ind_side,size=cell_number,replace=False)
            X=N.empty((len(ind),len(X_binned[ind[0]])))
            
            for i in range(len(ind)):
                X[i]=X_binned[ind[i]]
    
            X=X.T
                   
            
            pred_local=cross_val_predict(clf,X,labels,cv=kf)             
            scoring=accuracy_score(labels,pred_local)
            score_side_local.append(scoring)
            if t==0:
                predicted_side.append(pred_local)
                
            # Same model.
            ind=N.random.choice(ind_same,size=cell_number,replace=False)
            X=N.empty((len(ind),len(X_binned[ind[0]])))
            
            for i in range(len(ind)):
                X[i]=X_binned[ind[i]]
    
            X=X.T
                       
            pred_local=cross_val_predict(clf,X,labels,cv=kf)             
            scoring=accuracy_score(labels,pred_local)
            score_same_local.append(scoring)
                
            # Cat model.
            ind=N.random.choice(ind_cat,size=cell_number,replace=False)
            X=N.empty((len(ind),len(X_binned[ind[0]])))
            
            for i in range(len(ind)):
                X[i]=X_binned[ind[i]]
    
            X=X.T
                   
    
            pred_local=cross_val_predict(clf,X,labels,cv=kf)             
            scoring=accuracy_score(labels,pred_local)
            score_cat_local.append(scoring)
            
        score_cat.append(N.nanmean(score_cat_local))
        score_same.append(N.nanmean(score_same_local))
        score_side.append(N.nanmean(score_side_local))

        
        
    score_cat_total.append(score_cat)
    score_same_total.append(score_same)
    score_side_total.append(score_side)
    
    
    
score_same_stats=N.asarray(score_same_total).T
score_side_stats=N.asarray(score_side_total).T
score_cat_stats=N.asarray(score_cat_total).T

p,F,anovas=[],[],[]
for n in range(len(score_same_stats)):
    stat=olf.one_way_repeated_measures_anova_general_three_groups(score_side_stats[n],score_same_stats[n],score_cat_stats[n])
    anovas.append(stat)
    p.append(stat[0]['p-unc'][0])
    F.append(stat[0]['F'][0])
    
statistics={'p':p,'F':F,'ANOVAS':anovas}    

res={'score same':N.asarray(score_same_total),'score side':N.asarray(score_side_total),'score cat':N.asarray(score_cat_total),
           'C':C,'cv':cv,'max_iter':max_iter,'used neurons':cell_number,'iterations':iterations,'statistics':statistics}

# Run 1-way ANOVA on data for each bin.
stat=olf.one_way_repeated_measures_anova_general_three_groups(res['score same'],res['score side'],res['score cat'])
res['1-way ANOVA']=stat


res_total=res
os.chdir(target_dir)    
N.save("decoding_trail_outcome_with_specific_types_by_mouse",res)

