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
import scipy.stats as st

'''
Results of neuron_extraction are loaded by this file. The path information for data loading and target_dir need to be updated before use. 
'''

n_clusters=2
n_neighbors=1000
n_components=2

N.random.seed(42)
target_dir="/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/cell_classification"

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
data=N.load("/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/extraction_of_neurons/all_mice_results_spatial_tuning.npy",allow_pickle=True).item()
mouse_index_array=data['classification']['mouse index array']

corr=N.asarray(data['r left right'])
shift=N.asarray(data['xcorr shift'])
sig_si=N.ravel(N.where(data['sig si']==1))
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

'''
pl.figure()
pl.scatter(corr,shift)
pl.scatter(corr[ind_cat_final],shift[ind_cat_final],color="k")
pl.scatter(corr[ind_others],shift[ind_others],color="b")
pl.scatter(corr2[ind_same],shift2[ind_same],color="r")
pl.scatter(corr2[ind_side],shift2[ind_side],color="g")
'''



# Resort indivdual cells according to the classification scheme. Make mouse averages were applicable.

### Extract the relative indices of all categories for each mouse.
mouse_use_list=N.asarray([0,5,7,8,9,10,11])
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

indices_results={'relative mouse index':mouse_use_list,'absolut mouse list':[44,216,218,478,481,483,485],'indices cat':indices_cat,'indices side':indices_side,'indices same':indices_same}
###

### Quantification of the proportion of category neurons per mouse.
no_cells=data['no cells']
no_cells_total=data['no cells total']
_,n_same=get_index(mouse_index_array,ind_same_final,range(12))
_,n_side=get_index(mouse_index_array,ind_side_final,range(12))
_,n_cat=get_index(mouse_index_array,ind_cat_final,range(12))
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
X,X_left,X_right,position,position_left,position_right=[],[],[],[],[],[]
total_cells,active_cells=[],[]
for n in mouse_use_list:
    X.append(data['X and classes by mouse']['X'][n])
    X_left.append(data['X and classes by mouse']['X left'][n])
    X_right.append(data['X and classes by mouse']['X right'][n])
    position.append(data['X and classes by mouse']['position by mouse'][n])
    position_left.append(data['X and classes by mouse']['position left by mouse'][n])
    position_right.append(data['X and classes by mouse']['position right by mouse'][n])
    total_cells.append(data['no cells total'][n])
    active_cells.append(data['no cells'][n])

X_and_pos={'X':X,'X left':X_left,'X right':X_right,'position':position,'position left':position_left,'position right':position_right}
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
cons_total=data['consistency total']
cons_side,cons_same,cons_cat=resort_by_mouse(mouse_index_array,ind_cat_final,ind_side_final,ind_same_final,cons_total)
stats=olf.one_way_repeated_measures_anova_general_three_groups(cons_side,cons_same,cons_cat)

cons_total_fs=data['consistency fs total'] # Using first second instead.
cons_side_fs,cons_same_fs,cons_cat_fs=resort_by_mouse(mouse_index_array,ind_cat_final,ind_side_final,ind_same_final,cons_total_fs)
stats_fs=olf.one_way_repeated_measures_anova_general_three_groups(cons_side_fs,cons_same_fs,cons_cat_fs)

# Run the same analysis on a by-cell basis for significant SI neurons.
ind_side_sig_si=N.intersect1d(ind_side_final,sig_si)
ind_same_sig_si=N.intersect1d(ind_same_final,sig_si)
ind_cat_sig_si=N.intersect1d(ind_cat_final,sig_si)

cons_side_si=cons_total[ind_side_sig_si]
cons_same_si=cons_total[ind_same_sig_si]
cons_cat_si=cons_total[ind_cat_sig_si]
stats_sig_si=olf.one_way_anova_general(cons_side_si,cons_same_si,cons_cat_si)

cons_sig_si={'cons side':cons_side_si,'cons same':cons_same_si,'cons cat':cons_cat_si,'stats sig si':stats_sig_si}

consistency={'cat':cons_cat,'same':cons_same,'side':cons_side,'statistics':stats,
             'cat fs':cons_cat_fs,'same fs':cons_same_fs,'side fs':cons_side_fs,'statistics fs':stats_fs,
            
             'sig si':cons_sig_si}

# Get cell-wise grouped data.
r_left_right=data['r left right']
r_left_right_classified_cat=N.asarray(r_left_right)[ind_cat_final]
r_left_right_classified_same=N.asarray(r_left_right)[ind_same_final]
r_left_right_classified_side=N.asarray(r_left_right)[ind_side_final]

xcorr_shift=data['xcorr shift']
xcorr_shift_classified_cat=N.asarray(xcorr_shift)[ind_cat_final]
xcorr_shift_classified_same=N.asarray(xcorr_shift)[ind_same_final]
xcorr_shift_classified_side=N.asarray(xcorr_shift)[ind_side_final]

av_hist_left=data['av hist left']
av_hist_right=data['av hist right']
av_hist_left_cat=N.asarray(av_hist_left[ind_cat_final])
av_hist_left_same=N.asarray(av_hist_left[ind_same_final])
av_hist_left_side=N.asarray(av_hist_left[ind_side_final])

av_hist_left_even=data['av hist left even']
av_hist_left_even_cat=N.asarray(av_hist_left_even[ind_cat_final])
av_hist_left_even_same=N.asarray(av_hist_left_even[ind_same_final])
av_hist_left_even_side=N.asarray(av_hist_left_even[ind_side_final])

av_hist_left_odd=data['av hist left odd']
av_hist_left_odd_cat=N.asarray(av_hist_left_odd[ind_cat_final])
av_hist_left_odd_same=N.asarray(av_hist_left_odd[ind_same_final])
av_hist_left_odd_side=N.asarray(av_hist_left_odd[ind_side_final])

av_hist_left_first=data['av hist left first']
av_hist_left_first_cat=N.asarray(av_hist_left_first[ind_cat_final])
av_hist_left_first_same=N.asarray(av_hist_left_first[ind_same_final])
av_hist_left_first_side=N.asarray(av_hist_left_first[ind_side_final])

av_hist_left_second=data['av hist left second']
av_hist_left_second_cat=N.asarray(av_hist_left_second[ind_cat_final])
av_hist_left_second_same=N.asarray(av_hist_left_second[ind_same_final])
av_hist_left_second_side=N.asarray(av_hist_left_second[ind_side_final])

tuning_functions={'same':av_hist_left_same,'side':av_hist_left_side,'cat':av_hist_left_cat,
                  'same even':av_hist_left_even_same,'side even':av_hist_left_even_side,'cat even':av_hist_left_even_cat,
                  'same odd':av_hist_left_odd_same,'side odd':av_hist_left_odd_side,'cat odd':av_hist_left_odd_cat,
                  'same first':av_hist_left_first_same,'side first':av_hist_left_first_side,'cat first':av_hist_left_first_cat,
                  'same second':av_hist_left_second_same,'side second':av_hist_left_second_side,'cat second':av_hist_left_second_cat,
                  'av hist left':av_hist_left,'av hist right':av_hist_right}

# Get activity data.
act_same=N.asarray(data['mean activity'][ind_same_final])
act_side=N.asarray(data['mean activity'][ind_side_final])
act_cat=N.asarray(data['mean activity'][ind_cat_final])
stats=olf.one_way_anova_general(act_side,act_same,act_cat)
act_stats={'trial':stats}

activity_score_same=data['act score total'][ind_same_final]
activity_score_side=data['act score total'][ind_side_final]
activity_score_cat=data['act score total'][ind_cat_final]

# Get activity-subsampled data, removing side cells with lowest lowest_percent of activity.
lowest_percent=20
threshold=N.percentile(act_side,lowest_percent)
ind_side_subsampled,act_side_subsampled=[],[]
for n in range(len(data['mean activity'])):
    if n in ind_side_final:
        if data['mean activity'][n]>threshold:
            ind_side_subsampled.append(n)
            act_side_subsampled.append(data['mean activity'][n])
act_side_subsampled=N.asarray(act_side_subsampled)
ind_side_subsampled=N.asarray(ind_side_subsampled)


stats=olf.one_way_anova_general(act_side_subsampled,act_same,act_cat)


cons_side_subsampled,cons_same_subsampled,cons_cat_subsampled=resort_by_mouse(mouse_index_array,ind_cat_final,ind_side_subsampled,ind_same_final,cons_total)
stats_subsampled=olf.one_way_repeated_measures_anova_general_three_groups(cons_side_subsampled,cons_same,cons_side)
activity_subsampled={'mean act side subsampled':act_side_subsampled,
                     'consistency side subsampled':cons_side_subsampled,'consistency same':cons_same,'consistency cat':cons_cat,
                     'statistics activity subsampled':stats,'statistics consistency subsampled':stats_subsampled}

# get peak location data.
peak_location_left=data['peak location left']
peak_location_right=data['peak location right']
peak_location_left_same=peak_location_left[ind_same_final]
peak_location_right_same=peak_location_right[ind_same_final]
peak_location_left_side=peak_location_left[ind_side_final]
peak_location_right_side=peak_location_right[ind_side_final]
peak_location_left_cat=peak_location_left[ind_cat_final]
peak_location_right_cat=peak_location_right[ind_cat_final]

speed_modulation=data['speed mod r']
speed_modulation_same=speed_modulation[ind_same_final]
speed_modulation_side=speed_modulation[ind_side_final]
speed_modulation_cat=speed_modulation[ind_cat_final]

speed_r=data['speed mod r']
speed_r_side=speed_r[ind_side_final]   
speed_r_same=speed_r[ind_same_final]      
speed_r_cat=speed_r[ind_cat_final]     

speed_p=data['speed mod p']
speed_p_side=speed_p[ind_side_final]   
speed_p_same=speed_p[ind_same_final]      
speed_p_cat=speed_p[ind_cat_final]       

speed_hist=data['speed hist norm']       
speed_hist_side=speed_hist[ind_side_final]    
speed_hist_same=speed_hist[ind_same_final]    
speed_hist_cat=speed_hist[ind_cat_final]    

# Analyze peak location vs. speed r for both trajectory types.
def get_speed_peak_correlation(peak_loc,speed_r):
    ind_pos=N.ravel(N.where(speed_r>0)) # Get pos and neg modulated neurons.
    ind_neg=N.ravel(N.where(speed_r<0))

    r1,p1=st.spearmanr(peak_loc[ind_pos],speed_r[ind_pos])
    r2,p2=st.spearmanr(peak_loc[ind_neg],speed_r[ind_neg])
    temp={'r pos':r1,'p pos':p1,'r neg':r2,'p neg':p2}
    return temp

speed_peak_r_side_left=get_speed_peak_correlation(peak_location_left_side,speed_modulation_side)
speed_peak_r_side_right=get_speed_peak_correlation(peak_location_right_side,speed_modulation_side)
speed_peak_r_same_left=get_speed_peak_correlation(peak_location_left_same,speed_modulation_same)
speed_peak_r_same_right=get_speed_peak_correlation(peak_location_right_same,speed_modulation_same)
speed_peak_r_cat_left=get_speed_peak_correlation(peak_location_left_cat,speed_modulation_cat)
speed_peak_r_cat_right=get_speed_peak_correlation(peak_location_right_cat,speed_modulation_cat)



speed_total=data['speed total']
pos_total=[]
pos_left_temp=data['X and classes by mouse']['position left by mouse']
pos_right_temp=data['X and classes by mouse']['position right by mouse']
for n in range(len(speed_total)):
    temp=N.append(pos_left_temp[n],pos_right_temp[n])
    pos_total.append(temp)

binned_speed=[]

for n in range(len(speed_total)):
    b_speed,b,_=st.binned_statistic(pos_total[n],speed_total[n],bins=[0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0])
    binned_speed.append(b_speed)

speed_peak_correlation={'side left':speed_peak_r_side_left,'side right':speed_peak_r_side_right,
                        'same left':speed_peak_r_same_left,'same right':speed_peak_r_same_right,
                        'cat left':speed_peak_r_cat_left,'cat right':speed_peak_r_cat_right,
                        'speed':speed_total,'pos':pos_total,'binned speed':N.asarray(binned_speed)}    

activity={'mean act same':act_same,'mean act side':act_side,'mean act cat':act_cat,'statistics':act_stats,
          'peak location left side':peak_location_left_side,'peak location right side':peak_location_right_side,
          'peak location left same':peak_location_left_same,'peak location right same':peak_location_right_same,
          'peak location left cat':peak_location_left_cat,'peak location right cat':peak_location_right_cat,
          'speed mod same':speed_modulation_same,'speed mod side':speed_modulation_side,'speed mod cat':speed_modulation_cat,
          'speed r same':speed_r_same,'speed r side':speed_r_side,'speed r cat':speed_r_cat,
          'speed p same':speed_p_same,'speed p side':speed_p_side,'speed p cat':speed_p_cat,
          'speed hist same':speed_hist_same,'speed hist side':speed_hist_side,'speed hist cat':speed_hist_cat,
          'activity score same':activity_score_same,'activity score side':activity_score_side,'activity score cat':activity_score_cat,
          'subsampling':activity_subsampled,'speed peak correlation':speed_peak_correlation}


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

prop_cat_with_sig_si=len(N.intersect1d(ind_cat_final,sig_si))/len(ind_cat_final)
prop_same_with_sig_si=len(N.intersect1d(ind_same_final,sig_si))/len(ind_same_final)
prop_side_with_sig_si=len(N.intersect1d(ind_side_final,sig_si))/len(ind_side_final)


prop_sig_si={'prop cat with sig si':prop_cat_with_sig_si,'prop same with sig si':prop_same_with_sig_si,'prop side with sig si':prop_side_with_sig_si,
             '# cat si':len(N.intersect1d(ind_cat_final,sig_si)),'# same si':len(N.intersect1d(ind_same_final,sig_si)),'# side si':len(N.intersect1d(ind_side_final,sig_si)),
             'ind side sig si':ind_side_sig_si,'ind cat sig si':ind_cat_sig_si,'ind same sig si':ind_same_sig_si}

stats=olf.one_way_anova_general(si_max_side,si_max_same,si_max_cat)
si_data={'si max':si_max,'si max cat':si_max_cat,'si max side':si_max_side,'si max same':si_max_same,'sig si':N.asarray(sig_si),'si stats':stats,
         'proportion sig si':prop_sig_si,
         'ind side sig si':ind_side_sig_si,'ind cat sig si':ind_cat_sig_si,'ind same sig si':ind_same_sig_si}

behaviour={'prop correct':data['prop correct']}

class_cat_v_rest={'labels':labels,'total score':score,'individual scores':individual_scores}
class_same_v_side={'labels':labels2,'total score':score2,'individual scores':individual_scores2}
classification_results={'cat v rest':class_cat_v_rest,'same v side':class_same_v_side,'xcorr shift':shift,'r left right':corr,
                        'ind cat':ind_cat_final,'ind same':ind_same_final,'ind side':ind_side_final}


res_total={'tuning functions left':tuning_functions,'classification':classification_results,'consistency':consistency,'mouse indices':indices_results,
           'proportions':proportion_results,
           'data by mouse':X_and_pos,'si':si_data,'activity':activity,'behaviour':behaviour}
os.chdir(target_dir)
N.save('cell_clustering_results_all_mice.npy',res_total)

