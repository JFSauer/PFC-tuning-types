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
import pandas as pd
import pingouin as pg
import scipy.stats as st

'''
Requires output of stepwise_cell_classification as input. Paths need to be adjusted before use.
'''

N.random.seed(42)

target_dir="/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/generalized_position_decoding/activity_subsampling"
source_dir="/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/extraction_of_neurons"
target_filename="decoding_generalized_position_with_specific_types_by_mouse_SVC_model_activity_subsampled"

C=3
tol=0.000000001
cv=5
max_iter=1000000000
bins=N.linspace(0,2,11)
selected_mice=['44','216','219','478','481','483','485']

# Decoding function.
def run_decoding(X_temp,y,ind,random=False):  
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

clf=LinearSVC(C=C,max_iter=max_iter,tol=tol)
fold_iterator=KFold(n_splits=cv,shuffle=True)
kf=KFold(n_splits=cv,shuffle=True)

predicted_full,predicted_same,predicted_side,predicted_cat=[],[],[],[]
true_full=[]
pos_full=[]

y_pred_same,y_pred_side,y_pred_cat=[],[],[]

classification_all_mice=N.load("/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/cell_classification/cell_clustering_results_all_mice.npy",allow_pickle=True).item()

used_cells=[]

for n in range(len(selected_mice)):
    os.chdir(source_dir)
    print(selected_mice[n])
    
    data=N.load("%s_results_spatial_tuning.npy" %selected_mice[n],allow_pickle=True).item()
    neurons_left=data['X left']
    pos1=data['position left']
    neurons_right=data['X right']
    pos2=data['position right']
    neurons=N.append(neurons_left,neurons_right,axis=1)
    
    pos=N.append(N.asarray(pos1),N.asarray(pos2))
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
    
    # Get activity-subsampled data, removing same cells with largest_percent of activity.
    largest_percent=20
    act_same=data['mean activity'][ind_same]
    threshold=N.percentile(act_same,largest_percent)
    ind_same_subsampled=[]
    for n in range(len(data['mean activity'])):
        if n in ind_side:
            if data['mean activity'][n]<threshold:
                ind_same_subsampled.append(n)

    ind_same=N.asarray(ind_same_subsampled)
    if len(ind_side)<len(ind_cat) and len(ind_side)<len(ind_same):
        cell_number=len(ind_side)
    
    if len(ind_same)<len(ind_cat) and len(ind_same)<len(ind_side):
        cell_number=len(ind_same)
        
    if len(ind_cat)<len(ind_side) and len(ind_cat)<len(ind_same):
        cell_number=len(ind_cat)
    ind_side=N.random.choice(ind_side,size=cell_number,replace=False)
    ind_cat=N.random.choice(ind_cat,size=cell_number,replace=False)
    ind_same=N.random.choice(ind_same,size=cell_number,replace=False)
    used_cells.append(cell_number)
    # Side model.       
    pred_local,scoring=run_decoding(neurons,y,ind_side,random=False)
    score_side.append(scoring)
    
    pred_local,scoring=run_decoding(neurons,y,ind_side,random=True)
    score_side_rand.append(scoring)
 
    # Same model.
    pred_local,scoring=run_decoding(neurons,y,ind_same,random=False)    
    score_same.append(scoring)
    
    pred_local,scoring=run_decoding(neurons,y,ind_same,random=True)
    score_same_rand.append(scoring)
    
    # Cat model.
    pred_local,scoring=run_decoding(neurons,y,ind_cat,random=False)
    score_cat.append(scoring)
    
    pred_local,scoring=run_decoding(neurons,y,ind_cat,random=True)
    score_cat_rand.append(scoring)
        
        
    

res={'score same':N.asarray(score_same),'score side':N.asarray(score_side),'score cat':N.asarray(score_cat),
           'score same shuff':N.asarray(score_same_rand),'score side shuff':N.asarray(score_side_rand),'score cat shuff':N.asarray(score_cat_rand),
            'position target':pos_full,'predicted same':predicted_same,'predicted side':predicted_side,'predicted cat':predicted_cat,  
           'true full':true_full,'used neurons':N.asarray(used_cells),
           'C':C,'cv':cv,'max_iter':max_iter}

# Run 1-way ANOVA on data.
stat=olf.one_way_repeated_measures_anova_general_three_groups(res['score side'],res['score same'],res['score cat'])
res['1-way ANOVA']=stat

# Run 2-way ANOVA with vs. random and across groups comparisons.
def two_way_repeated_measures_anova_general(data1,data2):
    '''
    Data arrays given as data (mice, cells) x time points
    '''
    data=data1
    data=N.append(data1,data2,axis=0)
    data=N.ravel(data)
    condition=N.zeros((len(N.ravel(data1))))      
    condition=N.append(condition,N.ones((len(N.ravel(data1))))) 
    
    mouse=N.repeat(N.linspace(1,len(data1),len(data1)),len(data1[0]))
    mouse=N.append(mouse,mouse,axis=0)
    
    time=N.tile(N.linspace(1,len(data1[0]),len(data1[0])),len(data1)*2)

    
    df=pd.DataFrame({'mouse':mouse,
                    'time':time,
                    'condition':condition,
                    'data':data})    
    
    res = pg.rm_anova(dv='data', within=['time','condition'], subject='mouse', 
                  data=df, detailed=True)
    
    
    # Post hoc comparisons with paired t-tests and Sidak correction.
    data1=data1.T
    data2=data2.T
    t,p=[],[]
    sig=[]
    pcrit=1-(1-0.05)**(1/6)
    p_corr=[]
    for n in range(len(data1)):
        tt,pp=st.ttest_rel(data1[n],data2[n])
        t.append(tt)
        p.append(pp)
        p_corr.append(pp*0.05/pcrit)
        if pp<pcrit:
            sig.append(1)
        else:
            sig.append(0)
    

    vs_random={'p':p,'t':t,'sig':sig,'Pcrit':pcrit,'p corrected':p_corr}
    
    t,p=[],[]
    sig=[]
    p_corr=[]
    tt,pp=st.ttest_rel(data1[0],data1[1])
    t.append(tt)
    p.append(pp)
    p_corr.append(pp*0.05/pcrit)
    if pp<pcrit:
        sig.append(1)
    else:
        sig.append(0)
        
    tt,pp=st.ttest_rel(data1[0],data1[2])
    t.append(tt)
    p.append(pp)
    p_corr.append(pp*0.05/pcrit)
    if pp<pcrit:
        sig.append(1)
    else:
        sig.append(0)
    
    tt,pp=st.ttest_rel(data1[1],data1[2])
    t.append(tt)
    p.append(pp)
    p_corr.append(pp*0.05/pcrit)
    if pp<pcrit:
        sig.append(1)
    else:
        sig.append(0)
    
    groups={'p':p,'t':t,'sig':sig,'Pcrit':pcrit,'p corrected':p_corr}
    return res,vs_random,groups

real=N.empty((3,len(res['score same'])))
real[0]=res['score side']
real[1]=res['score same']
real[2]=res['score cat']
shuff=N.empty((3,len(res['score same'])))
shuff[0]=res['score side shuff']
shuff[1]=res['score same shuff']
shuff[2]=res['score cat shuff']

stat,vs_random,groups=two_way_repeated_measures_anova_general(real.T,shuff.T)
stats={'2-way RM ANOVA':stat,'vs random':vs_random,'across groups':groups}
res['2-way ANOVA']=stats

res_total=res
os.chdir(target_dir)    
N.save(target_filename,res)

