#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:23:39 2022

@author: jonas
"""
import Muysers_et_al_helper_functions as olf
import h5py
import numpy as np
import numpy as N
import os
import scipy.stats as st

'''
Analysis code to extract neurons for functional clustering. The data are accesible from https://doi.org/10.5281/zenodo.10528244. Mice 601 and 602 can be found in a separate
repository ().
For use, the skeletons have to be extracted from the archived data and placed in 'skeleton_dir' and 'skeleton_dir_original'.
Pathnames in filename_list need to be updated before use.
'''

target_dir="/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/learning_results/cell_extraction"
skeleton_dir="/media/jonas/data3/HM_wm_data/playground/learning_block/GCaMP6f_478-485/skeletons"
skeleton_dir_original='/media/jonas/data3/HM_wm_data/playground/block12_analysis_final/skeletons'
skeleton_dir_600="/media/jonas/data3/HM_wm_data/playground/learning_block/GCaMP6f_600-602/skeletons"

bins=40
rand_range=range(-4000,4000,1)
iterations=1
sig_thres=99
si_iterations=1000

def get_single_trial_tuning_functions(traces,start1,end1,start2,end2,track,skel,is_left=True,bins=bins):    
    
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
    reference_for_peak=hist
    occ,a=N.histogram(lin_pos_total,bins=bins) # Get occupancy map.
    occ=occ/N.sum(occ) # Convert to probability.
    hist=hist/occ
    hist=N.asarray(hist)
    
    hist_norm=N.empty_like(hist)
    for n in range(len(hist)):
        hist_norm[n]=olf.rescale_pos(hist[n],0,1)
        peak_location.append(N.argmax(hist_norm[n]))
    hist=hist_norm
    
    # Estimate SI.
    si=[]
    for i in range(len(hist)): # Iterate over transients.
        local=[]
        for n in range(len(hist[0])): # Iterate over the spatial tuning function.
            local.append(hist[i][n]*N.log(hist[i][n]/N.nanmean(hist[i]))*occ[n]) # Following Cholvin et al 2021 to yield bits/s
        si.append(N.nansum(local))
    
    
    # Find significant peaks.

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
    
    
   
    res={'av hist':hist,'X':X,'pos':lin_pos_total,'peak location':peak_location,
         'mean activity':mean_act,'significant peak':N.asarray(sig_peak),'si':N.asarray(si)}
    return res
    
    
def analyse_spatial_tuning(filename,name,session,skeletons='original'):
    
    if skeletons=='original':
        os.chdir(skeleton_dir_original)       
    elif skeletons=='485':
        os.chdir(skeleton_dir)
    elif skeletons=='600':
        os.chdir(skeleton_dir_600)
    skel=np.load("%s_skeleton.npy" %name,allow_pickle=True).item()    

    
    #%% get the correct assignments - the cells which are registered in the specified period
    f = h5py.File(filename, 'r')
    traces = f['0/analysis/transient/transients'][:]
    

    # Get all correct left and right runs outward and inward.
    source_corr=np.asarray(f['%s/behavior/corr'%session[0]])
    source_fail=np.asarray(f['%s/behavior/fail'%session[0]])
    source_oleft1=olf.separate_in(np.asarray(f['%s/behavior/ocl' %session[0]]),source_corr)
    source_oright1=olf.separate_in(np.asarray(f['%s/behavior/ocr' %session[0]]),source_corr)
    source_oleft2=olf.separate_in(np.asarray(f['%s/behavior/osl' %session[0]]),source_corr)
    source_oright2=olf.separate_in(np.asarray(f['%s/behavior/osr' %session[0]]),source_corr)
    
    
    a,source_ileft1=olf.separate_in_dual_conditional(np.asarray(f['%s/behavior/ocl' %session[0]]),np.asarray(f['%s/behavior/isl' %session[0]]),source_corr)
    a,source_iright1=olf.separate_in_dual_conditional(np.asarray(f['%s/behavior/ocr' %session[0]]),np.asarray(f['%s/behavior/isr' %session[0]]),source_corr)
    a,source_ileft2=olf.separate_in_dual_conditional(np.asarray(f['%s/behavior/ocl' %session[0]]),np.asarray(f['%s/behavior/icl' %session[0]]),source_corr)
    a,source_iright2=olf.separate_in_dual_conditional(np.asarray(f['%s/behavior/ocr' %session[0]]),np.asarray(f['%s/behavior/icr' %session[0]]),source_corr)
       
    source_track=np.asarray(f['%s/behavior/track'%session[0]])
    
    # Get performance of this day.
    is_corr=len(source_corr[0::2])
    is_error=len(source_fail[0::2])
    prop_corr=is_corr/(is_corr+is_error)
    
    # Case left.
    start1=source_oleft1[0::2]
    end1=source_oleft2[1::2]
    start2=source_ileft1[0::2]
    end2=source_ileft2[1::2]
    res_left=get_single_trial_tuning_functions(traces,start1,end1,start2,end2,source_track,skel,is_left=True,bins=bins)
    
    
    # Case right.
    start1=source_oright1[0::2]
    end1=source_oright2[1::2]
    start2=source_iright1[0::2]
    end2=source_iright2[1::2]
    res_right=get_single_trial_tuning_functions(traces,start1,end1,start2,end2,source_track,skel,is_left=False,bins=bins)
    
    av_hist_left=res_left['av hist']
    av_hist_right=res_right['av hist']
    sig_peak_left=res_left['significant peak']
    sig_peak_right=res_right['significant peak']
    
    # Keep neurons with non-zero activity and for which consistenty could be computed during during both left and right trials.
    ind=[]
    for n in range(len(av_hist_left)):
        if N.nansum(av_hist_left[n])>0 and N.nansum(av_hist_right[n])>0:
            ind.append(n)
            
    av_hist_left=av_hist_left[ind]
    av_hist_right=av_hist_right[ind]
    sig_peak_left=sig_peak_left[ind]
    sig_peak_right=sig_peak_right[ind]
    
    r_left_right,xcorr_shift=[],[]
    for n in range(len(av_hist_left)):
        xcorr=N.correlate(av_hist_left[n],av_hist_right[n],mode="full")
        midline=int(len(xcorr)/2)
        xcorr_shift.append(N.abs(midline-N.argmax(xcorr))) # Distance to zero time lag in bins.
        xcorr_local,a=st.pearsonr(av_hist_left[n],av_hist_right[n])
        r_left_right.append(xcorr_local)
    median_r_left_right=N.median(r_left_right)
    
    X_left=res_left['X'][ind]
    X_right=res_right['X'][ind]
    
    X=N.append(X_left,X_right,axis=1)
    pos_left=res_left['pos']
    pos_right=res_right['pos']
    pos_total=N.append(res_left['pos'],N.asarray(res_right['pos'])*-1)
    
    mean_act_left=N.asarray(res_left['mean activity'])[ind]
    mean_act_right=N.asarray(res_right['mean activity'])[ind]
    mean_act=N.empty((2,len(mean_act_left)))
    mean_act[0]=mean_act_left
    mean_act[1]=mean_act_right
    mean_act=N.mean(mean_act,axis=0)
    
    si_left=res_left['si'][ind]
    si_right=res_right['si'][ind]
    si_mean=N.empty((2,len(si_left)))
    si_mean[0]=si_left
    si_mean[1]=si_right
    si_mean=N.mean(si_mean,axis=0)
    
    res_total={'av hist left':av_hist_left,'av hist right':av_hist_right,
               'r left right':r_left_right,'median r left right':median_r_left_right,
               'xcorr shift':xcorr_shift,
               'mean activity left':mean_act_left,'mean activity right':mean_act_right,'mean activity':mean_act,
               'si left':si_left,'si right':si_right,'si mean':si_mean,
               'X':X,'X left':X_left,'X right':X_right,'position':pos_total,'position left':pos_left,'position right':pos_right,'# neurons':len(traces),
               'used indices':ind,
               'prop corr':N.asarray(prop_corr),
               'significant peak left':sig_peak_left,'significant peak right':sig_peak_right}
    os.chdir(target_dir)
    N.save("%s_results_spatial_tuning.npy" %name,res_total)
    return res_total


# Executed part.
mouse_list=['478','480','481','483','485','601','602']

hist_left,hist_right=[],[]
hist_left_first,hist_right_first=[],[]
hist_left_second,hist_right_second=[],[]
sig_left,sig_right=[],[]
sig_both,sig_none=[],[]
xcorr_shift=[]
r_left_right,median_r_left_right=[],[]
mean_act_left,mean_act_right,mean_act=[],[],[]
mean_act_mouse=[]
no_cells,no_cells_total=[],[]
prop_corr=[]
sig_peak_left=[]
sig_peak_right=[]
mouse_id_counter=0
mouse_id=[]
mouse_index_array=[]
si_left,si_right=[],[]
si_mean=[]

# 478.
filename='/media/jonas/data3/HM_wm_data/playground/learning_block/GCaMP6f_478-485/sampling_left_right/478_learning_left_right_behavior_analysis.hdf5'
session = [0]     #list of days to analyze
no_session = []                     # list of days not to be analyzed (e.g. sessions in new arena)
n_reg = 1
name="478"
print("Running ", name)
res=analyse_spatial_tuning(filename,name,session,skeletons="485")
av_hist_left=res['av hist left']
av_hist_right=res['av hist right']
xcorr_shift.extend(res['xcorr shift'])
r_left_right.extend(res['r left right'])
median_r_left_right.append(res['median r left right'])
mean_act_left.extend(res['mean activity left'])
mean_act_right.extend(res['mean activity right'])
mean_act.extend(res['mean activity'])
mean_act_mouse.append(N.nanmean(res['mean activity']))
no_cells.append(len(res['av hist left']))
prop_corr.append(res['prop corr'])
no_cells_total.append(res['# neurons'])
sig_peak_left.extend(res['significant peak left'])
sig_peak_right.extend(res['significant peak right'])
si_left.extend(res['si left'])
si_right.extend(res['si right'])
si_mean.extend(res['si mean'])

id_marker=N.zeros((len(res['xcorr shift'])))
id_marker+=mouse_id_counter # Keep track of which neuron belongs to which mouse.
mouse_index_array.extend(id_marker)
mouse_id_counter+=1

# 480.
filename='/media/jonas/data3/HM_wm_data/playground/learning_block/GCaMP6f_478-485/sampling_left_right/480_learning_left_right_behavior_analysis.hdf5'
session = [0]     #list of days to analyze
no_session = []                     # list of days not to be analyzed (e.g. sessions in new arena)
n_reg = 1
name="480"
print("Running ", name)
res=analyse_spatial_tuning(filename,name,session,skeletons="485")
av_hist_left=res['av hist left']
av_hist_right=res['av hist right']
xcorr_shift.extend(res['xcorr shift'])
r_left_right.extend(res['r left right'])
median_r_left_right.append(res['median r left right'])
mean_act_left.extend(res['mean activity left'])
mean_act_right.extend(res['mean activity right'])
mean_act.extend(res['mean activity'])
mean_act_mouse.append(N.nanmean(res['mean activity']))
no_cells.append(len(res['av hist left']))
prop_corr.append(res['prop corr'])
no_cells_total.append(res['# neurons'])
sig_peak_left.extend(res['significant peak left'])
sig_peak_right.extend(res['significant peak right'])
si_left.extend(res['si left'])
si_right.extend(res['si right'])
si_mean.extend(res['si mean'])

id_marker=N.zeros((len(res['xcorr shift'])))
id_marker+=mouse_id_counter # Keep track of which neuron belongs to which mouse.
mouse_index_array.extend(id_marker)
mouse_id_counter+=1

# 481.
filename='/media/jonas/data3/HM_wm_data/playground/learning_block/GCaMP6f_478-485/sampling_left_right/481_learning_left_right_behavior_analysis.hdf5'
session = [0]     #list of days to analyze
no_session = []                     # list of days not to be analyzed (e.g. sessions in new arena)
n_reg = 1
name="481"
print("Running ", name)
res=analyse_spatial_tuning(filename,name,session,skeletons="485")
av_hist_left=res['av hist left']
av_hist_right=res['av hist right']
xcorr_shift.extend(res['xcorr shift'])
r_left_right.extend(res['r left right'])
median_r_left_right.append(res['median r left right'])
mean_act_left.extend(res['mean activity left'])
mean_act_right.extend(res['mean activity right'])
mean_act.extend(res['mean activity'])
mean_act_mouse.append(N.nanmean(res['mean activity']))
no_cells.append(len(res['av hist left']))
prop_corr.append(res['prop corr'])
no_cells_total.append(res['# neurons'])
sig_peak_left.extend(res['significant peak left'])
sig_peak_right.extend(res['significant peak right'])
si_left.extend(res['si left'])
si_right.extend(res['si right'])
si_mean.extend(res['si mean'])

id_marker=N.zeros((len(res['xcorr shift'])))
id_marker+=mouse_id_counter # Keep track of which neuron belongs to which mouse.
mouse_index_array.extend(id_marker)
mouse_id_counter+=1

# 483.
filename='/media/jonas/data3/HM_wm_data/playground/learning_block/GCaMP6f_478-485/sampling_left_right/483_learning_left_right_behavior_analysis.hdf5'
session = [0]     #list of days to analyze
no_session = []                     # list of days not to be analyzed (e.g. sessions in new arena)
n_reg = 1
name="483"
print("Running ", name)
res=analyse_spatial_tuning(filename,name,session,skeletons="485")
av_hist_left=res['av hist left']
av_hist_right=res['av hist right']
xcorr_shift.extend(res['xcorr shift'])
r_left_right.extend(res['r left right'])
median_r_left_right.append(res['median r left right'])
mean_act_left.extend(res['mean activity left'])
mean_act_right.extend(res['mean activity right'])
mean_act.extend(res['mean activity'])
mean_act_mouse.append(N.nanmean(res['mean activity']))
no_cells.append(len(res['av hist left']))
prop_corr.append(res['prop corr'])
no_cells_total.append(res['# neurons'])
sig_peak_left.extend(res['significant peak left'])
sig_peak_right.extend(res['significant peak right'])
si_left.extend(res['si left'])
si_right.extend(res['si right'])
si_mean.extend(res['si mean'])

id_marker=N.zeros((len(res['xcorr shift'])))
id_marker+=mouse_id_counter # Keep track of which neuron belongs to which mouse.
mouse_index_array.extend(id_marker)
mouse_id_counter+=1

# 485.
filename='/media/jonas/data3/HM_wm_data/playground/learning_block/GCaMP6f_478-485/sampling_left_right/485_learning_left_right_behavior_analysis.hdf5'
session = [0]     #list of days to analyze
no_session = []                     # list of days not to be analyzed (e.g. sessions in new arena)
n_reg = 1
name="485"
print("Running ", name)
res=analyse_spatial_tuning(filename,name,session,skeletons="485")
av_hist_left=res['av hist left']
av_hist_right=res['av hist right']
xcorr_shift.extend(res['xcorr shift'])
r_left_right.extend(res['r left right'])
median_r_left_right.append(res['median r left right'])
mean_act_left.extend(res['mean activity left'])
mean_act_right.extend(res['mean activity right'])
mean_act.extend(res['mean activity'])
mean_act_mouse.append(N.nanmean(res['mean activity']))
no_cells.append(len(res['av hist left']))
prop_corr.append(res['prop corr'])
no_cells_total.append(res['# neurons'])
sig_peak_left.extend(res['significant peak left'])
sig_peak_right.extend(res['significant peak right'])
si_left.extend(res['si left'])
si_right.extend(res['si right'])
si_mean.extend(res['si mean'])

id_marker=N.zeros((len(res['xcorr shift'])))
id_marker+=mouse_id_counter # Keep track of which neuron belongs to which mouse.
mouse_index_array.extend(id_marker)
mouse_id_counter+=1


# 601.
filename='/media/jonas/data3/HM_wm_data/playground/learning_block/GCaMP6f_600-602/sampling_left_right/601_learning_left_right_behaviour_analysis.hdf5'
session = [0]     #list of days to analyze
no_session = []                     # list of days not to be analyzed (e.g. sessions in new arena)
n_reg = 1
name="601"
print("Running ", name)
res=analyse_spatial_tuning(filename,name,session,skeletons="600")
av_hist_left=res['av hist left']
av_hist_right=res['av hist right']
xcorr_shift.extend(res['xcorr shift'])
r_left_right.extend(res['r left right'])
median_r_left_right.append(res['median r left right'])
mean_act_left.extend(res['mean activity left'])
mean_act_right.extend(res['mean activity right'])
mean_act.extend(res['mean activity'])
mean_act_mouse.append(N.nanmean(res['mean activity']))
no_cells.append(len(res['av hist left']))
prop_corr.append(res['prop corr'])
no_cells_total.append(res['# neurons'])
sig_peak_left.extend(res['significant peak left'])
sig_peak_right.extend(res['significant peak right'])
si_left.extend(res['si left'])
si_right.extend(res['si right'])
si_mean.extend(res['si mean'])

id_marker=N.zeros((len(res['xcorr shift'])))
id_marker+=mouse_id_counter # Keep track of which neuron belongs to which mouse.
mouse_index_array.extend(id_marker)
mouse_id_counter+=1

# 602.
filename='/media/jonas/data3/HM_wm_data/playground/learning_block/GCaMP6f_600-602/sampling_left_right/602_learning_left_right_behaviour_analysis.hdf5'
session = [0]     #list of days to analyze
no_session = []                     # list of days not to be analyzed (e.g. sessions in new arena)
n_reg = 1
name="602"
print("Running ", name)
res=analyse_spatial_tuning(filename,name,session,skeletons="600")
av_hist_left=res['av hist left']
av_hist_right=res['av hist right']
xcorr_shift.extend(res['xcorr shift'])
r_left_right.extend(res['r left right'])
median_r_left_right.append(res['median r left right'])
mean_act_left.extend(res['mean activity left'])
mean_act_right.extend(res['mean activity right'])
mean_act.extend(res['mean activity'])
mean_act_mouse.append(N.nanmean(res['mean activity']))
no_cells.append(len(res['av hist left']))
prop_corr.append(res['prop corr'])
no_cells_total.append(res['# neurons'])
sig_peak_left.extend(res['significant peak left'])
sig_peak_right.extend(res['significant peak right'])
si_left.extend(res['si left'])
si_right.extend(res['si right'])
si_mean.extend(res['si mean'])

id_marker=N.zeros((len(res['xcorr shift'])))
id_marker+=mouse_id_counter # Keep track of which neuron belongs to which mouse.
mouse_index_array.extend(id_marker)
mouse_id_counter+=1


mouse_index_array=N.asarray(mouse_index_array)
classification={'mouse index array':mouse_index_array}

res_total={'av hist left':av_hist_left,'av hist right':av_hist_right,'hist left':hist_left,'hist right':hist_right,
           'av hist left first':N.asarray(hist_left_first),'av hist right first':N.asarray(hist_right_first),
           'av hist left second':N.asarray(hist_left_second),'av hist right second':N.asarray(hist_right_second),
           'r left right':N.asarray(r_left_right),'median r left right':N.asarray(median_r_left_right),
           'xcorr shift':N.asarray(xcorr_shift),
           'mean activity left':N.asarray(mean_act_left),'mean activity right':N.asarray(mean_act_right),'mean activity':N.asarray(mean_act),
           'mean activity mouse':N.asarray(mean_act_mouse),'no cells':no_cells,'no cells total':no_cells_total,
           'significant peak left':N.asarray(sig_peak_left),'significant peak right':N.asarray(sig_peak_right),
            'si left':N.asarray(si_left),'si right':N.asarray(si_right),'si mean':N.asarray(si_mean),
           'prop correct':N.asarray(prop_corr),
           'classification':classification,}
os.chdir(target_dir)
N.save("all_mice_results_spatial_tuning_learning.npy",res_total)