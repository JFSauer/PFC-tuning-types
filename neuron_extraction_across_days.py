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
from functools import reduce

'''
Analysis code to extract neurons for functional clustering (across days). The data are accesible from https://doi.org/10.5281/zenodo.10528244.
For use, the skeletons have to be extracted from the archived data and placed in 'skeleton_dir' and 'skeleton_dir_original'.
Pathnames in filename_list need to be updated before use.
'''

target_dir="/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/across_days/extraction_of_neurons"
skeleton_dir="/media/jonas/data3/HM_wm_data/playground/learning_block/GCaMP6f_478-485/skeletons"
skeleton_dir_original='/media/jonas/data3/HM_wm_data/playground/block12_analysis_final/skeletons'

bins=40
rand_range=range(-4000,4000,1)
iterations=1
sig_thres=99
min_number_for_remain=5
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
    
    
   
    res={'av hist':hist,'X':X,'pos':lin_pos_total,'peak location':peak_location,'mean activity':mean_act,'significant peak':N.asarray(sig_peak)}
    return res
        
def analyse_spatial_tuning(filename,name,session,temp_fix=False):
    
    if temp_fix==True:
        os.chdir(skeleton_dir_original)       
    else:
        os.chdir(skeleton_dir)
    skel=np.load("%s_skeleton.npy" %name,allow_pickle=True).item()    

    
    #%% get the correct assignments - the cells which are registered in the specified period
    f = h5py.File(filename, 'r')
    assignments = np.take((f['cellreg/assignments'][:]), session, axis=1, out=None, mode='raise')
    index_filtered = np.sum(assignments>=0, axis=1) >= n_reg
    assignments_filtered_session = np.array(assignments[index_filtered])
    
    assignments_filtered = np.full((len(assignments_filtered_session),len(f['cellreg/assignments'][1])),-1) # not really neceassary, returns an array of the shape 'all sessions in cell_reg' x 'co-registered cells in specified period'
    
    for n in range(len(session)):
        assignments_filtered[:,session[n]] = assignments_filtered_session[:,n]
    
    #numpy.insert(arr, obj(indices before which values is inserted), values, axis=None)
        
    #%% create sorted traces for specified period

            
    traces_normed = np.zeros(assignments_filtered.shape, dtype=np.ndarray)
    for j in range(traces_normed.shape[1]):
        for i in range(traces_normed.shape[0]):
            traces_normed[i,j] = f[str(j)]['analysis/trace/Fn'][assignments_filtered[i][j],:]
            
    #%% create sorted significant transients for specified period 
    
    transients = np.zeros(assignments_filtered.shape, dtype=np.ndarray)
    for j in range(transients.shape[1]):
        for i in range(transients.shape[0]):
            transients[i,j] = f[str(j)]['analysis/transient/transients'][assignments_filtered[i][j],:]
    
    
    
    # Get all neurons with non-zero activity during the trials and non-nan consistency.
    session_results={} 
    for session_index in range(len(session)):
        traces=transients[:,session[session_index]]
    
        # Get all correct left and right runs outward and inward.
        source_corr=np.asarray(f['%s/behavior/corr'%session[session_index]])
        source_oleft1=olf.separate_in(np.asarray(f['%s/behavior/ocl' %session[session_index]]),source_corr)
        source_oright1=olf.separate_in(np.asarray(f['%s/behavior/ocr' %session[session_index]]),source_corr)
        source_oleft2=olf.separate_in(np.asarray(f['%s/behavior/osl' %session[session_index]]),source_corr)
        source_oright2=olf.separate_in(np.asarray(f['%s/behavior/osr' %session[session_index]]),source_corr)
       
        a,source_ileft1=olf.separate_in_dual_conditional(np.asarray(f['%s/behavior/ocl' %session[session_index]]),np.asarray(f['%s/behavior/isl' %session[session_index]]),source_corr)
        a,source_iright1=olf.separate_in_dual_conditional(np.asarray(f['%s/behavior/ocr' %session[session_index]]),np.asarray(f['%s/behavior/isr' %session[session_index]]),source_corr)
        a,source_ileft2=olf.separate_in_dual_conditional(np.asarray(f['%s/behavior/ocl' %session[session_index]]),np.asarray(f['%s/behavior/icl' %session[session_index]]),source_corr)
        a,source_iright2=olf.separate_in_dual_conditional(np.asarray(f['%s/behavior/ocr' %session[session_index]]),np.asarray(f['%s/behavior/icr' %session[session_index]]),source_corr)
           
        source_track=np.asarray(f['%s/behavior/track'%session[session_index]])
        
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
        
            
            
        r_left_right,xcorr_shift=[],[]
        for n in range(len(av_hist_left)):
            xcorr=N.correlate(av_hist_left[n],av_hist_right[n],mode="full")
            midline=int(len(xcorr)/2)
            xcorr_shift.append(N.abs(midline-N.argmax(xcorr))) # Distance to zero time lag in bins.
            if N.isnan(N.sum(av_hist_left[n])) or N.isnan(N.sum(av_hist_right[n])):
                xcorr_local=N.nan
            else:
                xcorr_local,a=st.pearsonr(av_hist_left[n],av_hist_right[n])
            r_left_right.append(xcorr_local)
        
        
        mean_act_left=N.asarray(res_left['mean activity'])
        mean_act_right=N.asarray(res_right['mean activity'])
        
        # Identify neurons with non-zero actvity and non-nan r-l-correlation.
        ind=[]
        for n in range(len(av_hist_left)):
            if N.nansum(av_hist_left[n])>0 and N.nansum(av_hist_right[n])>0 and not N.isnan(r_left_right[n]):
                ind.append(n)
        
        temp={'indices':N.asarray(ind)}
        session_results["%i" %session_index]=temp
    
    ind_total=[]
    temp_ind={}
    for n in range(len(session_results)):
        temp_ind["%i"%n]=session_results['%i'%n]['indices']
         
    ind_total=reduce(N.intersect1d,(temp_ind['0'],temp_ind['1']))

    ind_total=N.asarray(ind_total)
    n_neurons=len(ind_total)
    n_neurons_total=len(traces)
    
    
    # Run the code again, this time restricing it to the neurons given in ind_total
    session_results={} 
    for session_index in range(len(session)):
        traces=transients[:,session[session_index]][ind_total]
        traces_measurable=N.stack(transients[:,session[session_index]]).astype(None)
        raw_peaks=N.max(traces_measurable,axis=1)
      
        # Get all correct left and right runs outward and inward.
        source_corr=np.asarray(f['%s/behavior/corr'%session[session_index]])
        source_oleft1=olf.separate_in(np.asarray(f['%s/behavior/ocl' %session[session_index]]),source_corr)
        source_oright1=olf.separate_in(np.asarray(f['%s/behavior/ocr' %session[session_index]]),source_corr)
        source_oleft2=olf.separate_in(np.asarray(f['%s/behavior/osl' %session[session_index]]),source_corr)
        source_oright2=olf.separate_in(np.asarray(f['%s/behavior/osr' %session[session_index]]),source_corr)
        
 
        a,source_ileft1=olf.separate_in_dual_conditional(np.asarray(f['%s/behavior/ocl' %session[session_index]]),np.asarray(f['%s/behavior/isl' %session[session_index]]),source_corr)
        a,source_iright1=olf.separate_in_dual_conditional(np.asarray(f['%s/behavior/ocr' %session[session_index]]),np.asarray(f['%s/behavior/isr' %session[session_index]]),source_corr)
        a,source_ileft2=olf.separate_in_dual_conditional(np.asarray(f['%s/behavior/ocl' %session[session_index]]),np.asarray(f['%s/behavior/icl' %session[session_index]]),source_corr)
        a,source_iright2=olf.separate_in_dual_conditional(np.asarray(f['%s/behavior/ocr' %session[session_index]]),np.asarray(f['%s/behavior/icr' %session[session_index]]),source_corr)
           
        source_track=np.asarray(f['%s/behavior/track'%session[session_index]])
        
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
            
            
        r_left_right,xcorr_shift=[],[]
        for n in range(len(av_hist_left)):
            xcorr=N.correlate(av_hist_left[n],av_hist_right[n],mode="full")
            midline=int(len(xcorr)/2)
            xcorr_shift.append(N.abs(midline-N.argmax(xcorr))) # Distance to zero time lag in bins.
            if N.isnan(N.sum(av_hist_left[n])) or N.isnan(N.sum(av_hist_right[n])):
                xcorr_local=N.nan
            else:
                xcorr_local,a=st.pearsonr(av_hist_left[n],av_hist_right[n])
            r_left_right.append(xcorr_local)
        median_r_left_right=N.median(r_left_right)
        
       
        X_left=res_left['X']
        X_right=res_right['X']
        X=N.append(X_left,X_right,axis=1)
        pos_total=N.append(res_left['pos'],N.asarray(res_right['pos'])*-1)
        pos_left=res_left['pos']
        pos_right=res_right['pos']
        
        mean_act_left=N.asarray(res_left['mean activity'])
        mean_act_right=N.asarray(res_right['mean activity'])
        
        # Identify neurons with non-zero actvity and non-nan r-l-correlation.
        ind=[]
        for n in range(len(av_hist_left)):
            if N.nansum(av_hist_left[n])>0 and N.nansum(av_hist_right[n])>0 and not N.isnan(r_left_right[n]):
                ind.append(n)
        
        temp={'av hist left':av_hist_left,'av hist right':av_hist_right,
                   'r left right':r_left_right,
                   'xcorr shift':xcorr_shift,
                   'mean activity left':mean_act_left,'mean activity right':mean_act_right,
                   'X':X,'position':pos_total,'X left':X_left,'X right':X_right,'position left':pos_left,'position right':pos_right,'# neurons':len(traces),'indices':ind,
                   'significant peak left':sig_peak_left,'significant peak right':sig_peak_right,'peak calcium signal':raw_peaks}
        session_results["%i" %session_index]=temp
        
    
    
    
    # Compute total correlations over days.
    corr_left,corr_right=[],[]
    left1=session_results['0']['av hist left']
    right1=session_results['0']['av hist right']

    left2=session_results['1']['av hist left']
    right2=session_results['1']['av hist right']
    for i in range(len(left1)):
        r,p=st.pearsonr(left1[i],left2[i])
        corr_left.append(r)
        r,p=st.pearsonr(right1[i],right2[i])
        corr_right.append(r)
    
    #corr_total=N.reshape(corr_total,[len(session_results)-1,-1])
    # Get preferred consistency by using the side with larger mean activity.
    corr_total=[]
    for n in range(len(mean_act_left)):
        if mean_act_left[n]>mean_act_right[n]:
            corr_total.append(corr_left[n])
        elif mean_act_left[n]<mean_act_right[n]:
            corr_total.append(corr_right[n])


    res_total={'session results':session_results,'corr':corr_total,
               'neuron number':n_neurons,'no cells':n_neurons,'no cells total':n_neurons_total}
    
    os.chdir(target_dir)
    N.save("%s_results_spatial_tuning.npy" %name,res_total)
    return res_total


# Executed part.

corr,same,side,cat=[],[],[],[]
same_remain,side_remain,cat_remain=[],[],[]
offset=0
mean_act_left1,mean_act_right1=[],[]
mean_act_left2,mean_act_right2=[],[]
corr_cat_mouse,corr_same_mouse,corr_side_mouse=[],[],[]
cat_to_same,cat_to_side,cat_to_unclassified=[],[],[]
same_to_side,same_to_cat,same_to_unclassified=[],[],[]
side_to_same,side_to_cat,side_to_unclassified=[],[],[]
no_cells,no_cells_total=[],[]
mouse_id_counter=0
mouse_id=[]
mouse_index_array=[]
xcorr_shift1,xcorr_shift2=[],[]
r_left_right1,r_left_right2=[],[]
sig_peak_left1,sig_peak_left2=[],[]
sig_peak_right1,sig_peak_right2=[],[]
calcium_peak1,calcium_peak2=[],[]

# 44.
filename='/media/jonas/data3/HM_wm_data/playground/block12_analysis_final/44/gcamp6f44_cellreg_block1_2_curated_behavior_analysis.hdf5'
session = [0,3] 
no_session=[]    #list of days to analyze                  # list of days not to be analyzed (e.g. sessions in new arena)
n_reg = 2
name="44"
print("Running ", name)
res=analyse_spatial_tuning(filename,name,session,temp_fix=True)
av_hist_left1=res['session results']['0']['av hist left']
av_hist_left2=res['session results']['1']['av hist left']
mean_act_left1=res['session results']['0']['mean activity left']
mean_act_left2=res['session results']['1']['mean activity left']
mean_act_right1=res['session results']['0']['mean activity right']
mean_act_right2=res['session results']['1']['mean activity right']
r_left_right1=res['session results']['0']['r left right']
r_left_right2=res['session results']['1']['r left right']
xcorr_shift1=res['session results']['0']['xcorr shift']
xcorr_shift2=res['session results']['1']['xcorr shift']
sig_peak_left1.extend(res['session results']['0']['significant peak left'])
sig_peak_left2.extend(res['session results']['1']['significant peak left'])
sig_peak_right1.extend(res['session results']['0']['significant peak right'])
sig_peak_right2.extend(res['session results']['1']['significant peak right'])
calcium_peak1.extend(res['session results']['0']['peak calcium signal'])
calcium_peak2.extend(res['session results']['1']['peak calcium signal'])

id_marker=N.zeros((len(res['corr'])))
id_marker+=mouse_id_counter # Keep track of which neuron belongs to which mouse.
mouse_index_array.extend(id_marker)
mouse_id_counter+=1

corr=res['corr']
no_cells_total.append(res['no cells total'])
no_cells.append(res['no cells'])
offset=offset+res['neuron number']

# 45.
filename='/media/jonas/data3/HM_wm_data/playground/block12_analysis_final/45/gcamp6f45_cellreg_block1_2_curated_behavior_analysis.hdf5'

session = [0,3] 
no_session=[]    #list of days to analyze                  # list of days not to be analyzed (e.g. sessions in new arena)
n_reg = 2
name="45"
print("Running ", name)
res=analyse_spatial_tuning(filename,name,session,temp_fix=True)

av_hist_left1=N.append(av_hist_left1,res['session results']['0']['av hist left'],axis=0)
av_hist_left2=N.append(av_hist_left2,res['session results']['1']['av hist left'],axis=0)
mean_act_left1=N.append(mean_act_left1,res['session results']['0']['mean activity left'])
mean_act_left2=N.append(mean_act_left2,res['session results']['1']['mean activity left'])
mean_act_right1=N.append(mean_act_right1,res['session results']['0']['mean activity right'])
mean_act_right2=N.append(mean_act_right2,res['session results']['1']['mean activity right'])

r_left_right1=N.append(r_left_right1,res['session results']['0']['r left right'],axis=0)
r_left_right2=N.append(r_left_right2,res['session results']['1']['r left right'],axis=0)
xcorr_shift1=N.append(xcorr_shift1,res['session results']['0']['xcorr shift'],axis=0)
xcorr_shift2=N.append(xcorr_shift2,res['session results']['1']['xcorr shift'],axis=0)
sig_peak_left1.extend(res['session results']['0']['significant peak left'])
sig_peak_left2.extend(res['session results']['1']['significant peak left'])
sig_peak_right1.extend(res['session results']['0']['significant peak right'])
sig_peak_right2.extend(res['session results']['1']['significant peak right'])
calcium_peak1.extend(res['session results']['0']['peak calcium signal'])
calcium_peak2.extend(res['session results']['1']['peak calcium signal'])

corr=N.append(corr,res['corr'])
id_marker=N.zeros((len(res['corr'])))
id_marker+=mouse_id_counter # Keep track of which neuron belongs to which mouse.
mouse_index_array.extend(id_marker)
mouse_id_counter+=1

no_cells_total.append(res['no cells total'])
no_cells.append(res['no cells'])
offset=offset+res['neuron number']

# 94.
filename='/media/jonas/data3/HM_wm_data/playground/block12_analysis_final/94/gcamp6f94_cellreg_block1_2_curated_behavior_analysis.hdf5'
session = [0,3] 
no_session=[]    #list of days to analyze                  # list of days not to be analyzed (e.g. sessions in new arena)
n_reg = 2
name="94"
print("Running ", name)
res=analyse_spatial_tuning(filename,name,session,temp_fix=True)
av_hist_left1=N.append(av_hist_left1,res['session results']['0']['av hist left'],axis=0)
av_hist_left2=N.append(av_hist_left2,res['session results']['1']['av hist left'],axis=0)
mean_act_left1=N.append(mean_act_left1,res['session results']['0']['mean activity left'])
mean_act_left2=N.append(mean_act_left2,res['session results']['1']['mean activity left'])
mean_act_right1=N.append(mean_act_right1,res['session results']['0']['mean activity right'])
mean_act_right2=N.append(mean_act_right2,res['session results']['1']['mean activity right'])
r_left_right1=N.append(r_left_right1,res['session results']['0']['r left right'],axis=0)
r_left_right2=N.append(r_left_right2,res['session results']['1']['r left right'],axis=0)
xcorr_shift1=N.append(xcorr_shift1,res['session results']['0']['xcorr shift'],axis=0)
xcorr_shift2=N.append(xcorr_shift2,res['session results']['1']['xcorr shift'],axis=0)
sig_peak_left1.extend(res['session results']['0']['significant peak left'])
sig_peak_left2.extend(res['session results']['1']['significant peak left'])
sig_peak_right1.extend(res['session results']['0']['significant peak right'])
sig_peak_right2.extend(res['session results']['1']['significant peak right'])
calcium_peak1.extend(res['session results']['0']['peak calcium signal'])
calcium_peak2.extend(res['session results']['1']['peak calcium signal'])

corr=N.append(corr,res['corr'])

id_marker=N.zeros((len(res['corr'])))
id_marker+=mouse_id_counter # Keep track of which neuron belongs to which mouse.
mouse_index_array.extend(id_marker)
mouse_id_counter+=1

no_cells_total.append(res['no cells total'])
no_cells.append(res['no cells'])
offset=offset+res['neuron number']



# 216.
filename='/media/jonas/data3/HM_wm_data/playground/block12_analysis_final/216/gcamp6f216_cellreg_block1_2_curated_behavior_analysis.hdf5'
session = [0,3] 
no_session=[]    #list of days to analyze                  # list of days not to be analyzed (e.g. sessions in new arena)
n_reg = 2
name="216"
print("Running ", name)
res=analyse_spatial_tuning(filename,name,session,temp_fix=True)
av_hist_left1=N.append(av_hist_left1,res['session results']['0']['av hist left'],axis=0)
av_hist_left2=N.append(av_hist_left2,res['session results']['1']['av hist left'],axis=0)
mean_act_left1=N.append(mean_act_left1,res['session results']['0']['mean activity left'])
mean_act_left2=N.append(mean_act_left2,res['session results']['1']['mean activity left'])
mean_act_right1=N.append(mean_act_right1,res['session results']['0']['mean activity right'])
mean_act_right2=N.append(mean_act_right2,res['session results']['1']['mean activity right'])
r_left_right1=N.append(r_left_right1,res['session results']['0']['r left right'],axis=0)
r_left_right2=N.append(r_left_right2,res['session results']['1']['r left right'],axis=0)
xcorr_shift1=N.append(xcorr_shift1,res['session results']['0']['xcorr shift'],axis=0)
xcorr_shift2=N.append(xcorr_shift2,res['session results']['1']['xcorr shift'],axis=0)
sig_peak_left1.extend(res['session results']['0']['significant peak left'])
sig_peak_left2.extend(res['session results']['1']['significant peak left'])
sig_peak_right1.extend(res['session results']['0']['significant peak right'])
sig_peak_right2.extend(res['session results']['1']['significant peak right'])
calcium_peak1.extend(res['session results']['0']['peak calcium signal'])
calcium_peak2.extend(res['session results']['1']['peak calcium signal'])

corr=N.append(corr,res['corr'])
id_marker=N.zeros((len(res['corr'])))
id_marker+=mouse_id_counter # Keep track of which neuron belongs to which mouse.
mouse_index_array.extend(id_marker)
mouse_id_counter+=1

no_cells_total.append(res['no cells total'])
no_cells.append(res['no cells'])
offset=offset+res['neuron number']

# 218.
filename="/media/jonas/data3/HM_wm_data/playground/block12_analysis_final/218/gcamp6f218_cellreg_block1_2_curated_behavior_analysis.hdf5"
session = [0,3] 
no_session=[]    #list of days to analyze                  # list of days not to be analyzed (e.g. sessions in new arena)
n_reg = 2
name="218"
print("Running ", name)
res=analyse_spatial_tuning(filename,name,session,temp_fix=True)
av_hist_left1=N.append(av_hist_left1,res['session results']['0']['av hist left'],axis=0)
av_hist_left2=N.append(av_hist_left2,res['session results']['1']['av hist left'],axis=0)
mean_act_left1=N.append(mean_act_left1,res['session results']['0']['mean activity left'])
mean_act_left2=N.append(mean_act_left2,res['session results']['1']['mean activity left'])
mean_act_right1=N.append(mean_act_right1,res['session results']['0']['mean activity right'])
mean_act_right2=N.append(mean_act_right2,res['session results']['1']['mean activity right'])
r_left_right1=N.append(r_left_right1,res['session results']['0']['r left right'],axis=0)
r_left_right2=N.append(r_left_right2,res['session results']['1']['r left right'],axis=0)
xcorr_shift1=N.append(xcorr_shift1,res['session results']['0']['xcorr shift'],axis=0)
xcorr_shift2=N.append(xcorr_shift2,res['session results']['1']['xcorr shift'],axis=0)
sig_peak_left1.extend(res['session results']['0']['significant peak left'])
sig_peak_left2.extend(res['session results']['1']['significant peak left'])
sig_peak_right1.extend(res['session results']['0']['significant peak right'])
sig_peak_right2.extend(res['session results']['1']['significant peak right'])
calcium_peak1.extend(res['session results']['0']['peak calcium signal'])
calcium_peak2.extend(res['session results']['1']['peak calcium signal'])

corr=N.append(corr,res['corr'])
id_marker=N.zeros((len(res['corr'])))
id_marker+=mouse_id_counter # Keep track of which neuron belongs to which mouse.
mouse_index_array.extend(id_marker)
mouse_id_counter+=1

no_cells_total.append(res['no cells total'])
no_cells.append(res['no cells'])
offset=offset+res['neuron number']

# 219.
filename='/media/jonas/data3/HM_wm_data/playground/block12_analysis_final/219/gcamp6f219_cellreg_block1_2_curated_behavior_analysis.hdf5'
session = [0,3] 
no_session=[]    #list of days to analyze                  # list of days not to be analyzed (e.g. sessions in new arena)
n_reg = 2
name="219"
print("Running ", name)
res=analyse_spatial_tuning(filename,name,session,temp_fix=True)
av_hist_left1=N.append(av_hist_left1,res['session results']['0']['av hist left'],axis=0)
av_hist_left2=N.append(av_hist_left2,res['session results']['1']['av hist left'],axis=0)
mean_act_left1=N.append(mean_act_left1,res['session results']['0']['mean activity left'])
mean_act_left2=N.append(mean_act_left2,res['session results']['1']['mean activity left'])
mean_act_right1=N.append(mean_act_right1,res['session results']['0']['mean activity right'])
mean_act_right2=N.append(mean_act_right2,res['session results']['1']['mean activity right'])
r_left_right1=N.append(r_left_right1,res['session results']['0']['r left right'],axis=0)
r_left_right2=N.append(r_left_right2,res['session results']['1']['r left right'],axis=0)
xcorr_shift1=N.append(xcorr_shift1,res['session results']['0']['xcorr shift'],axis=0)
xcorr_shift2=N.append(xcorr_shift2,res['session results']['1']['xcorr shift'],axis=0)
sig_peak_left1.extend(res['session results']['0']['significant peak left'])
sig_peak_left2.extend(res['session results']['1']['significant peak left'])
sig_peak_right1.extend(res['session results']['0']['significant peak right'])
sig_peak_right2.extend(res['session results']['1']['significant peak right'])
calcium_peak1.extend(res['session results']['0']['peak calcium signal'])
calcium_peak2.extend(res['session results']['1']['peak calcium signal'])

corr=N.append(corr,res['corr'])

id_marker=N.zeros((len(res['corr'])))
id_marker+=mouse_id_counter # Keep track of which neuron belongs to which mouse.
mouse_index_array.extend(id_marker)
mouse_id_counter+=1

no_cells_total.append(res['no cells total'])
no_cells.append(res['no cells'])
offset=offset+res['neuron number']

# 478.
filename='/media/jonas/data3/HM_wm_data/playground/learning_block/GCaMP6f_478-485/CellReg_478_learning_behavior_analysis_spikes.hdf5'
session = [7,10]     #list of days to analyze
no_session = []                     # list of days not to be analyzed (e.g. sessions in new arena)
n_reg = 2
name="478"
print("Running ", name)
res=analyse_spatial_tuning(filename,name,session,temp_fix=False)
av_hist_left1=N.append(av_hist_left1,res['session results']['0']['av hist left'],axis=0)
av_hist_left2=N.append(av_hist_left2,res['session results']['1']['av hist left'],axis=0)
mean_act_left1=N.append(mean_act_left1,res['session results']['0']['mean activity left'])
mean_act_left2=N.append(mean_act_left2,res['session results']['1']['mean activity left'])
mean_act_right1=N.append(mean_act_right1,res['session results']['0']['mean activity right'])
mean_act_right2=N.append(mean_act_right2,res['session results']['1']['mean activity right'])
r_left_right1=N.append(r_left_right1,res['session results']['0']['r left right'],axis=0)
r_left_right2=N.append(r_left_right2,res['session results']['1']['r left right'],axis=0)
xcorr_shift1=N.append(xcorr_shift1,res['session results']['0']['xcorr shift'],axis=0)
xcorr_shift2=N.append(xcorr_shift2,res['session results']['1']['xcorr shift'],axis=0)
sig_peak_left1.extend(res['session results']['0']['significant peak left'])
sig_peak_left2.extend(res['session results']['1']['significant peak left'])
sig_peak_right1.extend(res['session results']['0']['significant peak right'])
sig_peak_right2.extend(res['session results']['1']['significant peak right'])
calcium_peak1.extend(res['session results']['0']['peak calcium signal'])
calcium_peak2.extend(res['session results']['1']['peak calcium signal'])

corr=N.append(corr,res['corr'])
id_marker=N.zeros((len(res['corr'])))
id_marker+=mouse_id_counter # Keep track of which neuron belongs to which mouse.
mouse_index_array.extend(id_marker)
mouse_id_counter+=1

no_cells_total.append(res['no cells total'])
no_cells.append(res['no cells'])
offset=offset+res['neuron number']

# 481.
filename='/media/jonas/data3/HM_wm_data/playground/learning_block/GCaMP6f_478-485/CellReg_481_learning_behavior_analysis_spikes.hdf5'
session = [7,10]     #list of days to analyze
no_session = []                     # list of days not to be analyzed (e.g. sessions in new arena)
n_reg = 2
name="481"
print("Running ", name)
res=analyse_spatial_tuning(filename,name,session,temp_fix=False)
av_hist_left1=N.append(av_hist_left1,res['session results']['0']['av hist left'],axis=0)
av_hist_left2=N.append(av_hist_left2,res['session results']['1']['av hist left'],axis=0)
mean_act_left1=N.append(mean_act_left1,res['session results']['0']['mean activity left'])
mean_act_left2=N.append(mean_act_left2,res['session results']['1']['mean activity left'])
mean_act_right1=N.append(mean_act_right1,res['session results']['0']['mean activity right'])
mean_act_right2=N.append(mean_act_right2,res['session results']['1']['mean activity right'])
r_left_right1=N.append(r_left_right1,res['session results']['0']['r left right'],axis=0)
r_left_right2=N.append(r_left_right2,res['session results']['1']['r left right'],axis=0)
xcorr_shift1=N.append(xcorr_shift1,res['session results']['0']['xcorr shift'],axis=0)
xcorr_shift2=N.append(xcorr_shift2,res['session results']['1']['xcorr shift'],axis=0)
sig_peak_left1.extend(res['session results']['0']['significant peak left'])
sig_peak_left2.extend(res['session results']['1']['significant peak left'])
sig_peak_right1.extend(res['session results']['0']['significant peak right'])
sig_peak_right2.extend(res['session results']['1']['significant peak right'])
calcium_peak1.extend(res['session results']['0']['peak calcium signal'])
calcium_peak2.extend(res['session results']['1']['peak calcium signal'])

corr=N.append(corr,res['corr'])
id_marker=N.zeros((len(res['corr'])))
id_marker+=mouse_id_counter # Keep track of which neuron belongs to which mouse.
mouse_index_array.extend(id_marker)
mouse_id_counter+=1

no_cells_total.append(res['no cells total'])
no_cells.append(res['no cells'])
offset=offset+res['neuron number']


# 483.
filename='/media/jonas/data3/HM_wm_data/playground/learning_block/GCaMP6f_478-485/CellReg_483_learning_behavior_analysis_spikes.hdf5'
session = [7,10]     #list of days to analyze
no_session = []                     # list of days not to be analyzed (e.g. sessions in new arena)
n_reg = 2
name="483"
print("Running ", name)
res=analyse_spatial_tuning(filename,name,session,temp_fix=False)
av_hist_left1=N.append(av_hist_left1,res['session results']['0']['av hist left'],axis=0)
av_hist_left2=N.append(av_hist_left2,res['session results']['1']['av hist left'],axis=0)
mean_act_left1=N.append(mean_act_left1,res['session results']['0']['mean activity left'])
mean_act_left2=N.append(mean_act_left2,res['session results']['1']['mean activity left'])
mean_act_right1=N.append(mean_act_right1,res['session results']['0']['mean activity right'])
mean_act_right2=N.append(mean_act_right2,res['session results']['1']['mean activity right'])
r_left_right1=N.append(r_left_right1,res['session results']['0']['r left right'],axis=0)
r_left_right2=N.append(r_left_right2,res['session results']['1']['r left right'],axis=0)
xcorr_shift1=N.append(xcorr_shift1,res['session results']['0']['xcorr shift'],axis=0)
xcorr_shift2=N.append(xcorr_shift2,res['session results']['1']['xcorr shift'],axis=0)
sig_peak_left1.extend(res['session results']['0']['significant peak left'])
sig_peak_left2.extend(res['session results']['1']['significant peak left'])
sig_peak_right1.extend(res['session results']['0']['significant peak right'])
sig_peak_right2.extend(res['session results']['1']['significant peak right'])
calcium_peak1.extend(res['session results']['0']['peak calcium signal'])
calcium_peak2.extend(res['session results']['1']['peak calcium signal'])

corr=N.append(corr,res['corr'])

id_marker=N.zeros((len(res['corr'])))
id_marker+=mouse_id_counter # Keep track of which neuron belongs to which mouse.
mouse_index_array.extend(id_marker)
mouse_id_counter+=1

no_cells_total.append(res['no cells total'])
no_cells.append(res['no cells'])
offset=offset+res['neuron number']

# 485.
filename='/media/jonas/data3/HM_wm_data/playground/learning_block/GCaMP6f_478-485/CellReg_485_learning_behavior_analysis_spikes.hdf5'
session = [7,10]     #list of days to analyze
no_session = []                     # list of days not to be analyzed (e.g. sessions in new arena)
n_reg = 2
name="485"
print("Running ", name)
res=analyse_spatial_tuning(filename,name,session,temp_fix=False)
av_hist_left1=N.append(av_hist_left1,res['session results']['0']['av hist left'],axis=0)
av_hist_left2=N.append(av_hist_left2,res['session results']['1']['av hist left'],axis=0)
mean_act_left1=N.append(mean_act_left1,res['session results']['0']['mean activity left'])
mean_act_left2=N.append(mean_act_left2,res['session results']['1']['mean activity left'])
mean_act_right1=N.append(mean_act_right1,res['session results']['0']['mean activity right'])
mean_act_right2=N.append(mean_act_right2,res['session results']['1']['mean activity right'])
r_left_right1=N.append(r_left_right1,res['session results']['0']['r left right'],axis=0)
r_left_right2=N.append(r_left_right2,res['session results']['1']['r left right'],axis=0)
xcorr_shift1=N.append(xcorr_shift1,res['session results']['0']['xcorr shift'],axis=0)
xcorr_shift2=N.append(xcorr_shift2,res['session results']['1']['xcorr shift'],axis=0)
sig_peak_left1.extend(res['session results']['0']['significant peak left'])
sig_peak_left2.extend(res['session results']['1']['significant peak left'])
sig_peak_right1.extend(res['session results']['0']['significant peak right'])
sig_peak_right2.extend(res['session results']['1']['significant peak right'])
calcium_peak1.extend(res['session results']['0']['peak calcium signal'])
calcium_peak2.extend(res['session results']['1']['peak calcium signal'])

corr=N.append(corr,res['corr'])

id_marker=N.zeros((len(res['corr'])))
id_marker+=mouse_id_counter # Keep track of which neuron belongs to which mouse.
mouse_index_array.extend(id_marker)
mouse_id_counter+=1

no_cells_total.append(res['no cells total'])
no_cells.append(res['no cells'])
offset=offset+res['neuron number']

mean_act1=N.empty((2,len(mean_act_left1)))
mean_act2=N.empty_like(mean_act1)
mean_act1[0]=mean_act_left1
mean_act2[0]=mean_act_left2
mean_act1[1]=mean_act_right1
mean_act2[1]=mean_act_right2

mean_act1=N.nanmean(mean_act1,axis=0)
mean_act2=N.nanmean(mean_act2,axis=0)

mouse_index_array=N.asarray(mouse_index_array)
classification={'mouse index array':mouse_index_array}

res_total={'av hist left 1':av_hist_left1,'av hist left 2':av_hist_left2,
           'corr':corr,'r left right 1':r_left_right1,'r left right 2':r_left_right2,
           'xcorr shift 1':xcorr_shift1,'xcorr shift 2':xcorr_shift2,
           'mean activity 1':N.asarray(mean_act1),'mean activity 2':N.asarray(mean_act2),
           'mean activity left 1':N.asarray(mean_act_left1),'mean activity left 2':N.asarray(mean_act_left2),'mean activity right 1':N.asarray(mean_act_right1),'mean activity right 2':N.asarray(mean_act_right2),
           'no cells':no_cells,'no cells total':no_cells_total,
           'classification':classification,
           'significant peak left 1':N.asarray(sig_peak_left1),'significant peak left 2':N.asarray(sig_peak_left2),
           'significant peak right 1':N.asarray(sig_peak_right1),'significant peak right 2':N.asarray(sig_peak_right2),
           'calcium peak 1':N.asarray(calcium_peak1),'calcium peak 2':N.asarray(calcium_peak2)
           }
os.chdir(target_dir)
N.save("all_mice_results_spatial_tuning_across_days1_2_5.npy",res_total)