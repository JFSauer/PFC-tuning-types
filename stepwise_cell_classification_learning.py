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
Results of neuron_extraction_learning are loaded by this file. The path information for data loading and target_dir need to be updated before use. 
'''

n_clusters=2
n_neighbors=1000
n_components=2

N.random.seed(42)
target_dir="/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/learning_results/cell_classification"

def get_clusters(values,n_clusters=n_clusters):
    
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

# Load the data.
data=N.load("/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/learning_results/cell_extraction/all_mice_results_spatial_tuning_learning.npy",allow_pickle=True).item()
mouse_index_array=data['classification']['mouse index array']

corr=N.asarray(data['r left right'])
shift=N.asarray(data['xcorr shift'])
# Keep track of cell index.
total_indices=N.asarray(range(len(corr)))

# Separate neurons along the xcorr_shift axis
labels,score,individual_scores=get_clusters(shift)


ind_cat_final=N.ravel(N.where(labels==1))

# Include test for significant peak in at least one of the trajectories.



ind_cat_final_corrected,ind_same_final_corrected,ind_side_final_corrected=[],[],[]

for n in ind_cat_final:
    sig_peak_left=data['significant peak left'][n]
    sig_peak_right=data['significant peak right'][n]
        
    if sig_peak_left==1 and sig_peak_right==1:
        
        ind_cat_final_corrected.append(n)

ind_cat_final=N.asarray(ind_cat_final_corrected)        


ind_others=N.ravel(N.where(labels==0))

corr2=corr[ind_others]
shift2=shift[ind_others]
remaining_indices=total_indices[ind_others]
labels2,score2,individual_scores2=get_clusters(corr2)

ind_same=N.ravel(N.where(labels2==0))
ind_side=N.ravel(N.where(labels2==1))
ind_same_final=remaining_indices[ind_same]
ind_side_final=remaining_indices[ind_side]

for n in ind_same_final:
    sig_peak_left=data['significant peak left'][n]
    sig_peak_right=data['significant peak right'][n]
        
    if sig_peak_left==1 and sig_peak_right==1:
        ind_same_final_corrected.append(n)

for n in ind_side_final:
    sig_peak_left=data['significant peak left'][n]
    sig_peak_right=data['significant peak right'][n]
        
    if sig_peak_left==1 or sig_peak_right==1:
        ind_side_final_corrected.append(n)
    
ind_same_final=N.asarray(ind_same_final_corrected)
ind_side_final=N.asarray(ind_side_final_corrected)


# Resort indivdual cells according to the classification scheme. Make mouse averages were applicable.

### Extract the relative indices of all categories for each mouse.
mouse_use_list=N.asarray([0,1,2,3,4,5,6])
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

indices_results={'relative mouse index':mouse_use_list,'absolut mouse list':[478,480,481,483,485,601,602],'indices cat':indices_cat,'indices side':indices_side,'indices same':indices_same}
###

### Quantification of the proportion of category neurons per mouse.
no_cells=data['no cells']
no_cells_total=data['no cells total']
_,n_same=get_index(mouse_index_array,ind_same_final,range(7))
_,n_side=get_index(mouse_index_array,ind_side_final,range(7))
_,n_cat=get_index(mouse_index_array,ind_cat_final,range(7))
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

### Get traces, positional data and other information for each mouse in mouse list.
total_cells,active_cells=[],[]
for n in mouse_use_list:
    total_cells.append(data['no cells total'][n])
    active_cells.append(data['no cells'][n])


###
# Grab the proportions data from the proficient group.
filename_proficient='/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/cell_classification/cell_clustering_results_all_mice.npy'
proficient=N.load(filename_proficient,allow_pickle=True).item()

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
    

# Get cell-wise grouped data.
r_left_right=data['r left right']
r_left_right_classified_cat=N.asarray(r_left_right)[ind_cat_final]
r_left_right_classified_same=N.asarray(r_left_right)[ind_same_final]
r_left_right_classified_side=N.asarray(r_left_right)[ind_side_final]

xcorr_shift=data['xcorr shift']
xcorr_shift_classified_cat=N.asarray(xcorr_shift)[ind_cat_final]
xcorr_shift_classified_same=N.asarray(xcorr_shift)[ind_same_final]
xcorr_shift_classified_side=N.asarray(xcorr_shift)[ind_side_final]

# Get activity data.
act_same=N.asarray(data['mean activity'][ind_same_final])
act_side=N.asarray(data['mean activity'][ind_side_final])
act_cat=N.asarray(data['mean activity'][ind_cat_final])
stats=olf.one_way_anova_general(act_side,act_same,act_cat)
act_stats={'trial':stats}


activity={'mean act same':act_same,'mean act side':act_side,'mean act cat':act_cat,'statistics':act_stats}

# Obtain SI data.
si_left=data['si left']
si_right=data['si right']
si_max=[]
for n in range(len(si_left)):
    if si_left[n]>si_right[n]:
        si_max.append(si_left[n])
    else:
        si_max.append(si_right[n])  
        
si_max=N.asarray(si_max)

si_max_cat=si_max[ind_cat_final]
si_max_same=si_max[ind_same_final]
si_max_side=si_max[ind_side_final]

stats=olf.one_way_anova_general(si_max_side,si_max_same,si_max_cat)
stats_compare_si_cat=olf.unpaired_test(si_max_cat,proficient['si']['si max cat'])
stats_compare_si_same=olf.unpaired_test(si_max_cat,proficient['si']['si max same'])
stats_compare_si_side=olf.unpaired_test(si_max_cat,proficient['si']['si max side'])
stats_comparison_learning_proficient={'max si cat':stats_compare_si_cat,'max si same':stats_compare_si_same,'max si side':stats_compare_si_side}


si_data={'si max':si_max,'si max cat':si_max_cat,'si max side':si_max_side,'si max same':si_max_same,
         'si max cat proficient':proficient['si']['si max cat'],'si max same proficient':proficient['si']['si max same'],'si max side proficient':proficient['si']['si max side'],
         'si stats':stats,'si stats comparison proficient':stats_comparison_learning_proficient}

# Get aSI-subsampled data, removing side cells with lowest lowest_percent of SI.
lowest_percent=50
threshold=N.percentile(si_max_side,lowest_percent)
ind_side_subsampled,si_side_subsampled=[],[]
for n in range(len(si_max)):
    if n in ind_side_final:
        if si_max[n]>threshold:
            ind_side_subsampled.append(n)
            si_side_subsampled.append(si_max[n])
si_side_subsampled=N.asarray(si_side_subsampled)
ind_side_subsampled=N.asarray(ind_side_subsampled)

include_in_comparison=N.ravel(N.where(proficient['proportions']['# active cells']>50))
proficient_prop_cat=proficient['proportions']['prop cat'][include_in_comparison]
proficient_prop_same=proficient['proportions']['prop same'][include_in_comparison]
proficient_prop_side=proficient['proportions']['prop side'][include_in_comparison]
proficient_behaviour=proficient['behaviour']['prop correct'][include_in_comparison]
learning_behaviour=data['prop correct']
behaviour_total=learning_behaviour
behaviour_total=N.append(behaviour_total,proficient_behaviour)
prop_cat_total=prop_cat
prop_cat_total=N.append(prop_cat_total,proficient_prop_cat)
prop_same_total=prop_same
prop_same_total=N.append(prop_same_total,proficient_prop_same)
prop_side_total=prop_side
prop_side_total=N.append(prop_side_total,proficient_prop_side)


behaviour_correlations={'prop same total':prop_same_total,'prop side total':prop_side_total,'prop cat total':prop_cat_total,
                        'behaviour total':behaviour_total}

# Extract the subsampled indices to quantify cell numbers.

_,n_side_subsampled=get_index(mouse_index_array,ind_side_final,range(7))
prop_side_subsampled=[]
for n in range(len(n_side)):
    prop_side_subsampled.append(n_side_subsampled[n]/no_cells[n])

stats_prop_side_subsampled=olf.unpaired_test(prop_side_subsampled,proficient_prop_side)
stats_prop_side=olf.unpaired_test(prop_side,proficient_prop_side)
stats_prop_same=olf.unpaired_test(prop_same,proficient_prop_same)
stats_prop_cat=olf.unpaired_test(prop_cat,proficient_prop_cat)




si_subsampled={'si max side subsampled':si_side_subsampled,

               'proficient prop side':proficient_prop_side,'proficient prop same':proficient_prop_same,'proficient prop cat':proficient_prop_cat,
               'learning prop side':prop_side,'learning prop same':prop_same,'learning prop cat':prop_cat, 'learning prop side subsampled':prop_side_subsampled,
               'stats prop side':stats_prop_side,'stats prop same':stats_prop_same,'stats prop cat':stats_prop_cat,
               'stats prop side subsampled':stats_prop_side_subsampled,'behaviour correlations':behaviour_correlations              
                     }

behaviour={'prop correct':data['prop correct']}


class_cat_v_rest={'labels':labels,'total score':score,'individual scores':individual_scores}
class_same_v_side={'labels':labels2,'total score':score2,'individual scores':individual_scores2}
classification_results={'cat v rest':class_cat_v_rest,'same v side':class_same_v_side,'xcorr shift':shift,'r left right':corr,
                        'ind cat':ind_cat_final,'ind same':ind_same_final,'ind side':ind_side_final}


res_total={'classification':classification_results,'mouse indices':indices_results,'proportions':proportion_results,
           'activity':activity,'behaviour':behaviour,'si':si_data,'subsampled comparison learning learned':si_subsampled}
os.chdir(target_dir)
N.save('cell_clustering_results_all_mice.npy',res_total)

