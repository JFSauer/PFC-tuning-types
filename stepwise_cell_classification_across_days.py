#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:46:44 2024

@author: jonas
"""

from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as N
import Muysers_et_al_helper_functions as olf
import os

'''
Results of neuron_extraction_across_days are loaded by this file. The path information for data loading and target_dir need to be updated before use. 
'''

n_clusters=2
n_neighbors=1000
n_components=2

N.random.seed(42)
target_dir="/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/across_days/cell_classification"

def get_clusters(values,n_clusters=n_clusters,n_neigbors=n_neighbors,n_components=n_components):
    
    X=N.empty((2,len(values)))
    X[0]=values
    X[1]=values
    X=X.T
    #kmeans=KMeans(n_clusters=n_clusters,max_iter=100000,tol=0.0000001,init='random').fit(X)
    kmeans=SpectralClustering(n_clusters=n_clusters,n_components=n_components,n_neighbors=n_neighbors,assign_labels="discretize").fit(X)
    res=kmeans.labels_

    samples=silhouette_samples(X,res)
    individual_scores=[]
    for i in range(n_clusters):
        ith_silhouette_values=samples[res==i]
        ith_silhouette_values.sort()
        individual_scores.append(ith_silhouette_values)
        
    score=silhouette_score(X,res)
    
    return res,score,individual_scores

### Clustering step 1: find task-sequence vs. all other neurons.
eps=1
min_samples=2
peak_iterations=100

def cluster(data,use_day="1",n_clusters=n_clusters,n_neigbors=n_neighbors,n_components=n_components):
    corr=N.asarray(data['r left right %s' %use_day])
    shift=N.asarray(data['xcorr shift %s' %use_day])
    #sig_si=N.ravel(N.where(data['sig si']==1))
    # Keep track of cell index.
    total_indices=N.asarray(range(len(corr)))

    # Separate neurons along the xcorr_shift axis
    labels,score,individual_scores=get_clusters(shift)

    ind_cat_final=N.ravel(N.where(labels==1))

    # Include test for significant peak in at least one of the trajectories.
    ind_cat_final_corrected,ind_same_final_corrected,ind_side_final_corrected=[],[],[]

    for n in ind_cat_final:
        sig_peak_left=data['significant peak left %s' %use_day][n]
        sig_peak_right=data['significant peak right %s' %use_day][n]
            
        if sig_peak_left==1 and sig_peak_right==1:
            
            ind_cat_final_corrected.append(n)

    ind_cat_final=N.asarray(ind_cat_final_corrected)        

    ind_others=N.ravel(N.where(labels==0))

    corr2=corr[ind_others]
    remaining_indices=total_indices[ind_others]
    labels2,score2,individual_scores2=get_clusters(corr2)

    ind_same=N.ravel(N.where(labels2==0))
    ind_side=N.ravel(N.where(labels2==1))
    ind_same_final=remaining_indices[ind_same]
    ind_side_final=remaining_indices[ind_side]

    for n in ind_same_final:
        sig_peak_left=data['significant peak left %s' %use_day][n]
        sig_peak_right=data['significant peak right %s' %use_day][n]
            
        if sig_peak_left==1 and sig_peak_right==1:
            ind_same_final_corrected.append(n)

    for n in ind_side_final:
        sig_peak_left=data['significant peak left %s' %use_day][n]
        sig_peak_right=data['significant peak right %s' %use_day][n]
            
        if sig_peak_left==1 or sig_peak_right==1:
            ind_side_final_corrected.append(n)
        
    ind_same_final=N.asarray(ind_same_final_corrected)
    ind_side_final=N.asarray(ind_side_final_corrected)
    
    return ind_side_final, ind_same_final, ind_cat_final, labels, labels2, score, score2, individual_scores, individual_scores2

# Load the data.
data=N.load("/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/across_days/extraction_of_neurons/all_mice_results_spatial_tuning_across_days1_2_5.npy",allow_pickle=True).item()
mouse_index_array=data['classification']['mouse index array']

ind_side_final,ind_same_final,ind_cat_final,labels, labels2, score, score2, individual_scores, individual_scores2=cluster(data,use_day=1,n_clusters=n_clusters,n_neigbors=n_neighbors,n_components=n_components)

corr=N.asarray(data['r left right 1'])
shift=N.asarray(data['xcorr shift 1'])
corr2=N.asarray(data['r left right 2'])
shift2=N.asarray(data['xcorr shift 2'])

# Resort indivdual cells according to the classification scheme. Make mouse averages were applicable.

### Extract the relative indices of all categories for each mouse.
mouse_use_list=N.asarray([0,1,2,3,4,5,6,7,8,9])
def get_index(mouse_indices,target_indices,mouse_use_list):
    targets=N.zeros_like(mouse_indices)
    targets[target_indices]=1
    relative_indices=[]
    
    # Restrict to 7 mice used for decoding analysis. These are 44, 216, 218, 478, 481, 483, 485
    
    relative_indices=[]
    n_cells=[]
    for n in mouse_use_list:
        find_mouse=N.ravel(N.where(mouse_indices==n))
        relative_index=targets[find_mouse]
        relative_index=N.ravel(N.where(relative_index==1))
        relative_indices.append(relative_index)
        n_cells.append(len(relative_index))
        
    return relative_indices,n_cells

indices_cat,_=get_index(mouse_index_array,ind_cat_final,mouse_use_list)
indices_same,_=get_index(mouse_index_array,ind_same_final,mouse_use_list)
indices_side,_=get_index(mouse_index_array,ind_side_final,mouse_use_list)

indices_results={'relative mouse index':mouse_use_list,'indices cat':indices_cat,'indices side':indices_side,'indices same':indices_same}
###

### Quantification of the proportion of category neurons per mouse.
no_cells=data['no cells']
no_cells_total=data['no cells total']
_,n_same=get_index(mouse_index_array,ind_same_final,range(10))
_,n_side=get_index(mouse_index_array,ind_side_final,range(10))
_,n_cat=get_index(mouse_index_array,ind_cat_final,range(10))
prop_same,prop_side,prop_cat=[],[],[]
for n in range(len(n_same)):
    prop_same.append(n_same[n]/no_cells[n])
    prop_side.append(n_side[n]/no_cells[n])
    prop_cat.append(n_cat[n]/no_cells[n])
    
stats=olf.one_way_repeated_measures_anova_general_three_groups(prop_side,prop_same,prop_cat)
proportion_results={'# active cells':N.asarray(no_cells),'# total cells':N.asarray(no_cells_total),
                    '# same':N.asarray(n_same),'# side':N.asarray(n_side),'# cat':N.asarray(n_cat),
                    'prop same':N.asarray(prop_same),'prop side':N.asarray(prop_side),'prop cat':N.asarray(prop_cat),
                    'stats proportions':stats}

###
   
# Resort function for mouse averages. Extracts means of values for all mice that have group entries in all three categories.
def resort_by_mouse(mouse_indices,target_indices_cat,target_indices_side,target_indices_same,values):
    target_values_cat=values[target_indices_cat]
    target_values_same=values[target_indices_same]
    target_values_side=values[target_indices_side]
    
    target_indices_cat=mouse_indices[target_indices_cat]
    target_indices_same=mouse_indices[target_indices_same]
    target_indices_side=mouse_indices[target_indices_side]
    
    mouse_use_list=[]
    for n in range(15):
        if n in target_indices_cat and n in target_indices_same and n in target_indices_side:
            mouse_use_list.append(n)
        
    mean_cat,mean_side,mean_same=[],[],[]
    for n in mouse_use_list:
        local_indices=N.ravel(N.where(target_indices_cat==n))
        local_cat=target_values_cat[local_indices]
        local_indices=N.ravel(N.where(target_indices_same==n))
        local_same=target_values_same[local_indices]
        local_indices=N.ravel(N.where(target_indices_side==n))
        local_side=target_values_side[local_indices]
        
        mean_cat.append(N.nanmean(local_cat))
        mean_same.append(N.nanmean(local_same))
        mean_side.append(N.nanmean(local_side))
    
    return N.asarray(mean_side),N.asarray(mean_same),N.asarray(mean_cat) 
    
# Get consistency data.
cons_total=data['corr']
cons_side,cons_same,cons_cat=resort_by_mouse(mouse_index_array,ind_cat_final,ind_side_final,ind_same_final,cons_total)

stats=olf.one_way_repeated_measures_anova_general_three_groups(cons_side,cons_same,cons_cat)
consistency={'cat':cons_cat,'same':cons_same,'side':cons_side,'statistics':stats}

# Get cell-wise grouped data.
r_left_right=data['r left right 1']
r_left_right_classified_cat=N.asarray(r_left_right)[ind_cat_final]
r_left_right_classified_same=N.asarray(r_left_right)[ind_same_final]
r_left_right_classified_side=N.asarray(r_left_right)[ind_side_final]

xcorr_shift=data['xcorr shift 1']
xcorr_shift_classified_cat=N.asarray(xcorr_shift)[ind_cat_final]
xcorr_shift_classified_same=N.asarray(xcorr_shift)[ind_same_final]
xcorr_shift_classified_side=N.asarray(xcorr_shift)[ind_side_final]

av_hist_left1=data['av hist left 1']
av_hist_left1_cat=N.asarray(av_hist_left1[ind_cat_final])
av_hist_left1_same=N.asarray(av_hist_left1[ind_same_final])
av_hist_left1_side=N.asarray(av_hist_left1[ind_side_final])

av_hist_left2=data['av hist left 2']
av_hist_left2_cat=N.asarray(av_hist_left2[ind_cat_final])
av_hist_left2_same=N.asarray(av_hist_left2[ind_same_final])
av_hist_left2_side=N.asarray(av_hist_left2[ind_side_final])

tuning_functions={'same 1':av_hist_left1_same,'side 1':av_hist_left1_side,'cat 1':av_hist_left1_cat,
                  'same 2':av_hist_left2_same,'side 2':av_hist_left2_side,'cat 2':av_hist_left2_cat,}

# Get persistence in class params data.
rdiff_side=N.abs(corr[ind_side_final]-corr2[ind_side_final])
rdiff_same=N.abs(corr[ind_same_final]-corr2[ind_same_final])
rdiff_cat=N.abs(corr[ind_cat_final]-corr2[ind_cat_final])

rstats=olf.one_way_anova_general(rdiff_side,rdiff_same,rdiff_cat)

xdiff_side=N.abs(shift[ind_side_final]-shift2[ind_side_final])
xdiff_same=N.abs(shift[ind_same_final]-shift2[ind_same_final])
xdiff_cat=N.abs(shift[ind_cat_final]-shift2[ind_cat_final])

xstats=olf.one_way_anova_general(xdiff_side,xdiff_same,xdiff_cat)
persistence={'r diff side':rdiff_side,'r diff same':rdiff_same,'r diff cat':rdiff_cat,'r stats':rstats,
             'x diff side':xdiff_side,'x diff same':xdiff_same,'x diff cat':xdiff_cat,'x stats':xstats}

# Get activity data.
act_same=N.asarray(data['mean activity 1'][ind_same_final])
act_side=N.asarray(data['mean activity 1'][ind_side_final])
act_cat=N.asarray(data['mean activity 1'][ind_cat_final])
stats=olf.one_way_anova_general(act_side,act_same,act_cat)
act_stats={'trial':stats}

# Get activity-subsampled data, removing side cells with lowest lowest_percent of activity.
lowest_percent=35
threshold=N.percentile(act_side,lowest_percent)
ind_side_subsampled,act_side_subsampled=[],[]
for n in range(len(data['mean activity 1'])):
    if n in ind_side_final:
        if data['mean activity 1'][n]>threshold:
            ind_side_subsampled.append(n)
            act_side_subsampled.append(data['mean activity 1'][n])
act_side_subsampled=N.asarray(act_side_subsampled)
ind_side_subsampled=N.asarray(ind_side_subsampled)


stats=olf.one_way_anova_general(act_side_subsampled,act_same,act_cat)


cons_side_subsampled,cons_same_subsampled,cons_cat_subsampled=resort_by_mouse(mouse_index_array,ind_cat_final,ind_side_subsampled,ind_same_final,cons_total)
stats_subsampled=olf.one_way_repeated_measures_anova_general_three_groups(cons_side_subsampled,cons_same,cons_side)


activity_subsampled={'mean act side subsampled':act_side_subsampled,
                     'consistency side subsampled':cons_side_subsampled,'consistency same':cons_same,'consistency cat':cons_cat,
                     'statistics activity subsampled':stats,'statistics consistency subsampled':stats_subsampled}

activity={'mean act same':act_same,'mean act side':act_side,'mean act cat':act_cat,'statistics':act_stats,'subsampling':activity_subsampled}


class_cat_v_rest={'labels':labels,'total score':score,'individual scores':individual_scores}
class_same_v_side={'labels':labels2,'total score':score2,'individual scores':individual_scores2}
classification_results={'cat v rest':class_cat_v_rest,'same v side':class_same_v_side,
                        'xcorr shift':shift,'r left right':corr,
                        'ind cat':ind_cat_final,'ind same':ind_same_final,'ind side':ind_side_final}

# Get cell metrics.
peak1=data['calcium peak 1']
peak2=data['calcium peak 2']

peak1_side=peak1[ind_side_final]
peak1_same=peak1[ind_same_final]
peak1_cat=peak1[ind_cat_final]
stats_peak_size=olf.one_way_anova_general(peak1_side,peak1_same,peak1_cat)

peak_ratio=[]
for n in range(len(peak1)):
    peak_ratio.append((peak1[n]-peak2[n])/(peak1[n]+peak2[n]))
peak_ratio=N.asarray(peak_ratio)
peak_ratio_side=peak_ratio[ind_side_final]
peak_ratio_cat=peak_ratio[ind_cat_final]
peak_ratio_same=peak_ratio[ind_same_final]

stats_peak_ratio=olf.one_way_anova_general(peak_ratio_side,peak_ratio_same,peak_ratio_cat)
metrics_results={'peak ratio side':peak_ratio_side,'peak ratio same':peak_ratio_same,'peak ratio cat':peak_ratio_cat,
                 'peak 1 side':peak1_side,'peak 1 same':peak1_same,'peak 1 cat':peak1_cat,
                 'statistics peak ratio':stats_peak_ratio, 'statistics peak size':stats_peak_size}


res_total={'tuning functions left':tuning_functions,'classification':classification_results,'consistency':consistency,'mouse indices':indices_results,'proportions':proportion_results,
           'activity':activity,'metrics':metrics_results,'class persistence':persistence}
os.chdir(target_dir)
N.save('cell_clustering_results_all_mice.npy',res_total)

