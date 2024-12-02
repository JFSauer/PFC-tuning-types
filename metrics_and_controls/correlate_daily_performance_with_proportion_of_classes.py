#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:23:39 2022

@author: jonas
"""
import Muysers_et_al_helper_functions as olf
import h5py
import numpy as N
import os
import scipy.stats as st
import scipy.interpolate as sci
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, silhouette_samples

N.random.seed(42)

target_dir="/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/daily_behavioural_performance_vs_proportion_cell_classes"
skeleton_dir_original='/media/jonas/data3/HM_wm_data/playground/block12_analysis_final/skeletons'

Fs=20
bins=40
rand_range=range(-4000,4000,1)
peak_threshold=0.5
peak_offset=10
si_iterations=1000
roll_range=range(20,2400)

# Helper functions.
def get_single_trial_tuning_functions(traces,start1,end1,start2,end2,track,skel,conv,is_left=True,bins=bins):    
    

    lin_pos_total=[]
    lin_pos_trial=[]
    for n in range(len(start1)): # Iterate over trials.
        lin_pos_local=olf.linearize_2d_track_single_run(track,start1[n],end1[n],skel,is_left=is_left)
        lin_pos=olf.rescale_pos(lin_pos_local,0,1)
        lin_pos_local=olf.linearize_2d_track_single_run(track,start2[n],end2[n],skel,is_left=is_left)
        lin_pos_local=N.asarray(lin_pos_local)
        lin_pos.extend(olf.rescale_pos(-lin_pos_local,1,2))
        lin_pos_total.extend(lin_pos)
        lin_pos_trial.append(lin_pos)
       
    X=[]    

    for i in range(len(traces)):
        for n in range(len(start1)): # Iterate over trials.
            X.extend(traces[i][start1[n]:end1[n]])
            X.extend(traces[i][start2[n]:end2[n]])
                                           
    X=N.reshape(X,[len(traces),-1])    
    mean_act=N.nanmean(X,axis=1)   
    
    peak_location=[]
    # Get total tuning function.

    hist,a,b=st.binned_statistic(lin_pos_total,X,statistic='mean',bins=bins)
    reference_for_peak=hist # This array before normalization is used later to compare against shuffled data.
    occ,a=N.histogram(lin_pos_total,bins=bins) # Get occupancy map.
    occ=occ/N.sum(occ) # Convert to probability.
    hist_norm=hist/occ
    hist_norm=N.asarray(hist_norm)
    
    hist_norm2=N.empty_like(hist_norm)
    for n in range(len(hist_norm)):
        hist_norm2[n]=olf.rescale_pos(hist_norm[n],0,1)
        peak_location.append(N.argmax(hist_norm2[n]))
    hist=hist_norm2
    

    shuffled_for_peak=N.empty((si_iterations,len(X),len(reference_for_peak[0]))) # Dimensions iterations x cells x bins
    for iteration in range(si_iterations):
       
        lin_pos_shifted=[]
        for n in range(len(lin_pos_trial)):
            lin_pos_shifted.extend(N.roll(lin_pos_trial[n],N.random.randint(0,1000))) # Circularly shift the position information of this trial.
          
        hist_shifted,a,b=st.binned_statistic(lin_pos_shifted,X,statistic='mean',bins=bins)
        shuffled_for_peak[iteration]=hist_shifted
            
    sig_peak=[]
    for n in range(len(X)):
        is_significant=0
        local_reference=reference_for_peak[n]
        local_shuffled_for_peak=shuffled_for_peak[:,n,:].T # Gives the bins x iterations for the current neuron.
        for i in range(len(local_reference)): # Iterate over bins.
            if local_reference[i]>N.percentile(local_shuffled_for_peak[i],95):
                is_significant=1
        if is_significant==1:
            sig_peak.append(1)
        else:
            sig_peak.append(0)


    res={'av hist':hist,'peak location':N.asarray(peak_location),
         'significant peak':N.asarray(sig_peak),'mean activity':mean_act}
    return res
    
def analyse_spatial_tuning(filename,filename1,name,session):
    
    os.chdir(skeleton_dir_original)
    skel=N.load("%s_skeleton.npy" %name,allow_pickle=True).item() 
    f = h5py.File(filename, 'r')
        
    # Get the conversion factor to cm of the track.
    os.chdir(target_dir)
    conv_dict=N.load("distance_left_right_with_corrected_nreg.npy",allow_pickle=True).item()
    conv_left=conv_dict['gcamp6f%s' %name]['mean_left_session'][session[0]]
    conv_right=conv_dict['gcamp6f%s' %name]['mean_right_session'][session[0]]
    
    #%% get the correct assignments - the cells which are registered in the specified period
    f = h5py.File(filename, 'r')
    assignments = N.take((f['cellreg/assignments'][:]), session, axis=1, out=None, mode='raise')
    index_filtered = N.sum(assignments>=0, axis=1) >= n_reg
    assignments_filtered_session = N.array(assignments[index_filtered])
    
    assignments_filtered = N.full((len(assignments_filtered_session),len(f['cellreg/assignments'][1])),-1) # not really neceassary, returns an array of the shape 'all sessions in cell_reg' x 'co-registered cells in specified period'
    
    for n in range(len(session)):
        assignments_filtered[:,session[n]] = assignments_filtered_session[:,n]
    
    #numpy.insert(arr, obj(indices before which values is inserted), values, axis=None)
        
    #%% create sorted traces for specified period
    
    traces = N.zeros(assignments_filtered.shape, dtype=N.ndarray)
    for j in range(traces.shape[1]):
        for i in range(traces.shape[0]):
            traces[i,j] = f[str(j)]['activity/F'][assignments_filtered[i][j],:]
            
    traces_normed = N.zeros(assignments_filtered.shape, dtype=N.ndarray)
    for j in range(traces_normed.shape[1]):
        for i in range(traces_normed.shape[0]):
            traces_normed[i,j] = f[str(j)]['analysis/trace/Fn'][assignments_filtered[i][j],:]
            
    #%% create sorted significant transients for specified period 
    
    transients = N.zeros(assignments_filtered.shape, dtype=N.ndarray)
    for j in range(transients.shape[1]):
        for i in range(transients.shape[0]):
            transients[i,j] = f[str(j)]['analysis/transient/transients'][assignments_filtered[i][j],:]
    
    
    # Take the first session:
    n=0
   
    traces=transients[:,session[n]]
    # Get all correct left and right runs outward and inward.
    source_corr=N.asarray(f['%s/behavior/corr'%session[n]])
    source_fail=N.asarray(f['%s/behavior/fail'%session[n]])
    
    is_corr=len(source_corr[0::2])
    is_error=len(source_fail[0::2])
    prop_corr=is_corr/(is_corr+is_error)
       
    source_oleft1=olf.separate_in(N.asarray(f['%s/behavior/ocl' %session[n]]),source_corr)
    source_oright1=olf.separate_in(N.asarray(f['%s/behavior/ocr' %session[n]]),source_corr)
    source_oleft2=olf.separate_in(N.asarray(f['%s/behavior/osl' %session[n]]),source_corr)
    source_oright2=olf.separate_in(N.asarray(f['%s/behavior/osr' %session[n]]),source_corr)
    
    a,source_ileft1=olf.separate_in_dual_conditional(N.asarray(f['%s/behavior/ocl' %session[n]]),N.asarray(f['%s/behavior/isl' %session[n]]),source_corr)
    a,source_iright1=olf.separate_in_dual_conditional(N.asarray(f['%s/behavior/ocr' %session[n]]),N.asarray(f['%s/behavior/isr' %session[n]]),source_corr)
    a,source_ileft2=olf.separate_in_dual_conditional(N.asarray(f['%s/behavior/ocl' %session[n]]),N.asarray(f['%s/behavior/icl' %session[n]]),source_corr)
    a,source_iright2=olf.separate_in_dual_conditional(N.asarray(f['%s/behavior/ocr' %session[n]]),N.asarray(f['%s/behavior/icr' %session[n]]),source_corr)
       
    source_track=N.asarray(f['%s/behavior/track'%session[n]])
    
    # Case left.
    start1=source_oleft1[0::2]
    end1=source_oleft2[1::2]
    start2=source_ileft1[0::2]
    end2=source_ileft2[1::2]
    res_left=get_single_trial_tuning_functions(traces,start1,end1,start2,end2,source_track,skel,conv_left,is_left=True,bins=bins)
        
    # Case right.
    start1=source_oright1[0::2]
    end1=source_oright2[1::2]
    start2=source_iright1[0::2]
    end2=source_iright2[1::2]
    res_right=get_single_trial_tuning_functions(traces,start1,end1,start2,end2,source_track,skel,conv_right,is_left=False,bins=bins)
       
    av_hist_left=res_left['av hist']
    av_hist_right=res_right['av hist']
        
    # Keep neurons with non-zero activity and for which consistenty could be computed during during both left and right trials.
    ind=[]
    for n in range(len(av_hist_left)):
        if N.nansum(av_hist_left[n])>0 and N.nansum(av_hist_right[n])>0:
            ind.append(n)
            
    av_hist_left=av_hist_left[ind]
    av_hist_right=av_hist_right[ind]

    peak_location_left=res_left['peak location'][ind]
    peak_location_right=res_right['peak location'][ind]    
    sig_peak_left=N.asarray(res_left['significant peak'][ind])
    sig_peak_right=N.asarray(res_right['significant peak'][ind])
        
    r_left_right,xcorr_shift=[],[]
    for n in range(len(av_hist_left)):
        xcorr=N.correlate(av_hist_left[n],av_hist_right[n],mode="full")
        midline=int(len(xcorr)/2)
        xcorr_shift.append(N.abs(midline-N.argmax(xcorr))) # Distance to zero time lag in bins.
        xcorr_local,a=st.pearsonr(av_hist_left[n],av_hist_right[n])
        r_left_right.append(xcorr_local)
        
    res_total={'av hist left':av_hist_left,'av hist right':av_hist_right,
               'r left right':r_left_right,'xcorr shift':xcorr_shift,
               'peak location left':peak_location_left,'peak location right':peak_location_right,
               'significant peak left':sig_peak_left,'significant peak right':sig_peak_right,
               'used indices':ind,'prop corr':N.asarray(prop_corr)}
        
    return res_total

eps=1
min_samples=2
peak_iterations=100
n_clusters=2
n_neighbors=1000
n_components=2

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

# Resort indivdual cells according to the classification scheme. Make mouse averages were applicable.
def get_index(mouse_indices,target_indices,mouse_use_list):
    targets=N.zeros_like(mouse_indices)
    targets[target_indices]=1
    relative_indices=[]

    n_cells=[]
    for n in mouse_use_list:
        find_mouse=[]
        for i in range(len(mouse_indices)):
            if mouse_indices[i]==n:
                find_mouse.append(i)
        find_mouse=N.asarray(find_mouse)
        relative_index=targets[find_mouse]
        relative_index=N.ravel(N.where(relative_index==1))
        relative_indices.append(relative_index)
        n_cells.append(len(relative_index))
        
    return relative_indices,n_cells
# Executed part. Using the mice of Muysers et al., 2024 for which we have 12 days of recording.
filename_list=['/media/jonas/data3/HM_wm_data/playground/block12_analysis_final/44/gcamp6f44_cellreg_block1_2_curated_behavior_analysis.hdf5',

               '/media/jonas/data3/HM_wm_data/playground/block12_analysis_final/93/gcamp6f93_cellreg_block1_2_curated_behavior_analysis.hdf5',
               '/media/jonas/data3/HM_wm_data/playground/block12_analysis_final/94/gcamp6f94_cellreg_block1_2_curated_behavior_analysis.hdf5',
               '/media/jonas/data3/HM_wm_data/playground/block12_analysis_final/216/gcamp6f216_cellreg_block1_2_curated_behavior_analysis.hdf5',
               "/media/jonas/data3/HM_wm_data/playground/block12_analysis_final/218/gcamp6f218_cellreg_block1_2_curated_behavior_analysis.hdf5",
               '/media/jonas/data3/HM_wm_data/playground/block12_analysis_final/219/gcamp6f219_cellreg_block1_2_curated_behavior_analysis.hdf5']

     
session_master=[[0,1,2,4,5,6,7,10,11,12,13,14],
                [0,1,2,4,5,6,7,9,10,11,12,13],
                [0,1,2,4,5,6,7,9,10,11,12,13],
                [0,1,2,4,5,6,7,10,11,12,13,14],
                [0,1,2,4,5,6,7,10,11,12,13,14],
                [0,1,2,4,5,6,7,10,11,12,13,14]]

animal_list=["44","93","94","216","218","219"]
        
n_reg=1

behaviour_total,prop_same_total,prop_side_total,prop_cat_total=[],[],[],[]
corr_checker,shift_checker=[],[],
ind_same_checker,ind_side_checker,ind_cat_checker=[],[],[]

for current_session in range(len(session_master[0])): # Iterate over sessions.

    xcorr_shift=[]
    r_left_right=[]
    sig_peak_left,sig_peak_right=[],[]
    prop_corr=[]
    mouse_id_counter=0
    mouse_id=[]
    mouse_index_array=[]
    no_cells=[]
    peak_location_left,peak_location_right=[],[]
    
    for animal in range(len(animal_list)):
        filename=filename_list[animal]
        filename1=""
        name=animal_list[animal]
        
        session=[session_master[animal][current_session]]
        print(name)
        res=analyse_spatial_tuning(filename,filename1,name,session)
        
        if animal==0: # Account for the array concatenation difference between index 0 and the rest.
            av_hist_left=res['av hist left']
            av_hist_right=res['av hist right']
        else:
            av_hist_left=N.append(av_hist_left,res['av hist left'],axis=0)
            av_hist_right=N.append(av_hist_right,res['av hist right'],axis=0)
                    
        xcorr_shift.extend(res['xcorr shift'])
        r_left_right.extend(res['r left right'])
        
        sig_peak_left.extend(res['significant peak left'])
        sig_peak_right.extend(res['significant peak right'])
       
        no_cells.append(len(res['av hist left']))
        peak_location_left.extend(res['peak location left'])
        peak_location_right.extend(res['peak location right'])
    
        mouse_id.append(mouse_id_counter)
        id_marker=N.zeros((len(res['av hist left'])),dtype="int")
        id_marker+=mouse_id_counter # Keep track of which neuron belongs to which mouse.
        mouse_index_array.extend(id_marker)
            
        mouse_id_counter+=1
        prop_corr.append(res['prop corr'])
    
    
    # Run the 2-step classification on this recording day.
    ### Clustering step 1: find task-sequence vs. all other neurons.
        
    corr=N.asarray(r_left_right)
    shift=N.asarray(xcorr_shift)

    # Keep track of cell index.
    total_indices=N.asarray(range(len(corr)))

    # Separate neurons along the xcorr_shift axis
    
    
    labels,score,individual_scores=get_clusters(shift)

    ind_cat_final=N.ravel(N.where(labels==1))
        
    # Include test for significant peak in at least one of the trajectories.

    ind_cat_final_corrected,ind_same_final_corrected,ind_side_final_corrected=[],[],[]

    for n in ind_cat_final:                
        if sig_peak_left[n]==1 and sig_peak_right[n]==1:               
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
        if sig_peak_left[n]==1 and sig_peak_right[n]==1:  
            ind_same_final_corrected.append(n)

    for n in ind_side_final:
        if sig_peak_left[n]==1 and sig_peak_right[n]==1:  
            ind_side_final_corrected.append(n)
        
    ind_same_final=N.asarray(ind_same_final_corrected)
    ind_side_final=N.asarray(ind_side_final_corrected)


    ### Quantification of the proportion of category neurons per mouse.
    _,n_same=get_index(mouse_index_array,ind_same_final,range(6))
    _,n_side=get_index(mouse_index_array,ind_side_final,range(6))
    _,n_cat=get_index(mouse_index_array,ind_cat_final,range(6))
    prop_same,prop_side,prop_cat=[],[],[]
    
    for n in range(len(n_same)):
        prop_same.append(n_same[n]/no_cells[n])
        prop_side.append(n_side[n]/no_cells[n])
        prop_cat.append(n_cat[n]/no_cells[n])
        
    behaviour_total.extend(prop_corr)
    prop_same_total.extend(prop_same)
    prop_side_total.extend(prop_side)
    prop_cat_total.extend(prop_cat)
    
    corr_checker.append(corr)
    shift_checker.append(shift)
    ind_same_checker.append(ind_same_final)
    ind_side_checker.append(ind_side_final)
    ind_cat_checker.append(ind_cat_final)

metrics={'corr':corr_checker,'shift':shift_checker,'ind same':ind_same_checker,'ind side':ind_side_checker,'ind cat':ind_cat_checker}
res_total={'prop corr':N.asarray(behaviour_total),
           'prop side':N.asarray(prop_side_total),'prop same':N.asarray(prop_same_total),'prop cat':N.asarray(prop_cat_total),
           'metrics':metrics}
os.chdir(target_dir)
N.save("all_mice_results_proportion_correct_vs_classes.npy",res_total)