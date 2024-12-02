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
import pandas as pd
import pingouin as pg
import scipy.stats as st

'''
Requires output of stepwise_cell_classification as input. Paths need to be adjusted before use.
'''

N.random.seed(42)

target_dir="/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/trial_type_prediction/trial_type_prediction"
source_dir="/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/extraction_of_neurons"

cell_number=25 #Titrated to work with the set of mice listed below.
iterations=10

C=3
tol=0.000000001
cv=10
max_iter=1000000000

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

score_full,score_full_rand=[],[]
score_cat,score_cat_rand=[],[]
score_same,score_same_rand=[],[]
score_side,score_side_rand=[],[]

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
    
    true_full.append(y)
    
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
        
        if t==0:
            predicted_side.append(pred_local)

        score_side_local.append(scoring)
        
        pred_local,scoring=run_decoding(neurons,y,ind,cell_number=cell_number,random=True)
        score_side_rand_local.append(scoring)
 
        # Same model.
        ind=N.random.choice(ind_same,size=cell_number,replace=False)
        pred_local,scoring=run_decoding(neurons,y,ind,cell_number=cell_number,random=False)
        
        if t==0:
            predicted_same.append(pred_local)

        score_same_local.append(scoring)
        
        pred_local,scoring=run_decoding(neurons,y,ind,cell_number=cell_number,random=True)
        score_same_rand_local.append(scoring)

        
        # Cat model.
        ind=N.random.choice(ind_cat,size=cell_number,replace=False)
        pred_local,scoring=run_decoding(neurons,y,ind,cell_number=cell_number,random=False)
        
        if t==0:
            predicted_cat.append(pred_local)

        score_cat_local.append(scoring)
        
        pred_local,scoring=run_decoding(neurons,y,ind,cell_number=cell_number,random=True)
        score_cat_rand_local.append(scoring)
        
        
    score_cat.append(N.nanmean(score_cat_local))
    score_cat_rand.append(N.nanmean(score_cat_rand_local))
    score_same.append(N.nanmean(score_same_local))
    score_same_rand.append(N.nanmean(score_same_rand_local))
    score_side.append(N.nanmean(score_side_local))
    score_side_rand.append(N.nanmean(score_side_rand_local))
    

res={'score same':N.asarray(score_same),'score side':N.asarray(score_side),'score cat':N.asarray(score_cat),
           'score same shuff':N.asarray(score_same_rand),'score side shuff':N.asarray(score_side_rand),'score cat shuff':N.asarray(score_cat_rand),
            'position target':pos_full,'predicted same':predicted_same,'predicted side':predicted_side,'predicted cat':predicted_cat,  
           'true full':true_full,
           'C':C,'cv':cv,'max_iter':max_iter,'used neurons':cell_number,'iterations':iterations}

# Run 1-way ANOVA on data.
stat=olf.one_way_repeated_measures_anova_general_three_groups(res['score same'],res['score side'],res['score cat'])
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
N.save("decoding_trail_outcome_with_specific_types_by_mouse",res)

