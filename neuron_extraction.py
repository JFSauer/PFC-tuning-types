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
from sklearn.cluster import KMeans as KMeans
from sklearn.metrics import silhouette_score, silhouette_samples


'''
Analysis code to extract neurons for functional clustering. The data are accesible from https://doi.org/10.5281/zenodo.10528244.
For use, the skeletons have to be extracted from the archived data and placed in 'skeleton_dir' and 'skeleton_dir_original'.
Pathnames in filename_list need to be updated before use.
'''

N.random.seed(42)

target_dir="/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/extraction_of_neurons"
skeleton_dir="/media/jonas/data3/HM_wm_data/playground/learning_block/GCaMP6f_478-485/skeletons"
skeleton_dir_original='/media/jonas/data3/HM_wm_data/playground/block12_analysis_final/skeletons'

Fs=20
bins=40
speed_bins=11
speed_max=50
speed_bin_centers=N.linspace(0,speed_max,speed_bins)
rand_range=range(-4000,4000,1)
peak_threshold=0.5
peak_offset=10
si_iterations=1000
roll_range=range(20,2400)

# Helper functions.


def measure_peak_width(data,bins=bins,threshold=peak_threshold,peak_offset=peak_offset):
    xnew=N.linspace(0,len(data),len(data)*10)       
    f=sci.UnivariateSpline(range(len(data)),data,s=0)
    intf=f(xnew)
    
    argmax=N.argmax(intf)
    if peak_offset*10<argmax<len(intf)-peak_offset*10:
        forward=intf[argmax:]
        t50_1=N.argwhere(N.diff(forward < threshold, prepend=False))[::2,0][0]
        reverse=N.flip(intf[:argmax+1])
        t50_2=N.argwhere(N.diff(reverse < threshold, prepend=False))[::2,0][0]
        t50=(t50_1+t50_2)/(bins*5)
    else:
        t50=N.nan
    return t50

def get_single_trial_tuning_functions(traces,start1,end1,start2,end2,track,skel,conv,is_left=True,bins=bins):    
    
    speed_trial=[]
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
        
        lin_pos_scaled=N.asarray(lin_pos)*conv # Convert to cm.
        speed_local=[]
        for i in range(1,len(lin_pos)):
            speed_local.append(N.abs((lin_pos_scaled[i]-lin_pos_scaled[i-1])*Fs)) # In cm/s
        speed_local.append(speed_local[-1]) # Account for the last entry point.
                
        speed_trial.extend(speed_local)


    X=[]    
    hist_total=[]
    rs_total,rs_fs_total=[],[]
    hist_first,hist_second=[],[]
    hist_odd,hist_even=[],[]
    hist_conc=[]
    hist_conc_shift=[]
    for i in range(len(traces)):
        hist_trials=N.empty((len(start1),bins))
        hist_trials_conc=[]
        hist_trials_conc_shift=[]
        for n in range(len(start1)): # Iterate over trials.
            X.extend(traces[i][start1[n]:end1[n]])
            X.extend(traces[i][start2[n]:end2[n]])
            X_local=[]
            X_local.extend(traces[i][start1[n]:end1[n]])
            X_local.extend(traces[i][start2[n]:end2[n]])
            
            hist,a,b=st.binned_statistic(lin_pos_trial[n],X_local,statistic='mean',bins=bins)
            occ,a=N.histogram(lin_pos_trial[n],bins=bins) # Get occupancy map.
            occ=occ/N.sum(occ) # Convert to probability.
            hist=hist/occ # This is the single trial tuning function.
            hist_trials[n]=hist
            
            X_local_shifted=N.roll(X_local,N.random.choice(range(20,400)))
            hist_shift,a,b=st.binned_statistic(lin_pos_trial[n],X_local_shifted,statistic='mean',bins=bins)
            hist_shift=hist_shift/occ # This is the single trial tuning function.
            
            if not N.isnan(N.mean(hist)):           
                hist_trials_conc.extend(hist) # Keep the trial only if no Nan are created.
                hist_trials_conc_shift.extend(hist_shift)
                                 
        
        hist_conc.append(hist_trials_conc)
        hist_conc_shift.append(hist_trials_conc_shift)
        
        
        hist1=N.nanmean(hist_trials[0::2,:],axis=0)
        hist2=N.nanmean(hist_trials[1::2,:],axis=0)
        rs,p=st.pearsonr(hist1,hist2)
        rs_total.append(rs)
        hist_total.append(hist_trials)      
        hist1=olf.rescale_pos(hist1,0,1) # Normalize the consistency arrays.
        hist2=olf.rescale_pos(hist2,0,1)
        hist_first.append(hist1)
        hist_second.append(hist2)
        
        hist1=N.nanmean(hist_trials[0:int(len(hist_trials)/2),:],axis=0)
        hist2=N.nanmean(hist_trials[int(len(hist_trials)/2):,:],axis=0)
        rs,p=st.pearsonr(hist1,hist2)
        rs_fs_total.append(rs)    
        hist1=olf.rescale_pos(hist1,0,1) # Normalize the consistency arrays.
        hist2=olf.rescale_pos(hist2,0,1)
        hist_odd.append(hist1)
        hist_even.append(hist2)

    X=N.reshape(X,[len(traces),-1])    
    mean_act=N.nanmean(X,axis=1)   
    
    hist_conc=N.asarray(hist_conc)
    
       
    
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
    
    # Get SI and peak width
    si=[]
    peak_width=[]
    for i in range(len(hist)): # Iterate over transients.
        peak_width.append(measure_peak_width(hist[i]))
        local=[]
        for n in range(len(hist[0])): # Iterate over the spatial tuning function.
            local.append(hist[i][n]*N.log(hist[i][n]/N.nanmean(hist[i]))*occ[n]) # Following Cholvin et al 2021 to yield bits/s
        si.append(N.nansum(local))
    
    
    speed_hist,a,b=st.binned_statistic(lin_pos_total,speed_trial,bins=bins)
    
    
    # Estimate sign. SI based on time-shifted position signals.
    si_shifted_total=N.empty((si_iterations,len(X))) 
    shuffled_for_peak=N.empty((si_iterations,len(X),len(reference_for_peak[0]))) # Dimensions iterations x cells x bins
    for iteration in range(si_iterations):
       
        lin_pos_shifted=[]
        for n in range(len(lin_pos_trial)):
            lin_pos_shifted.extend(N.roll(lin_pos_trial[n],N.random.randint(0,1000))) # Circularly shift the position information of this trial.
          
        hist_shifted,a,b=st.binned_statistic(lin_pos_shifted,X,statistic='mean',bins=bins)
        shuffled_for_peak[iteration]=hist_shifted
        si_shifted=[]
        for i in range(len(hist)): # Iterate over transients.
            local=[]
            for n in range(len(hist[0])): # Iterate over the spatial tuning function.
                local.append(hist_shifted[i][n]*N.log(hist_shifted[i][n]/N.nanmean(hist_shifted[i]))*occ[n]) # Following Cholvin et al 2021 to yield bits/s
            si_shifted.append(N.nansum(local))
        si_shifted_total[iteration]=si_shifted # This leaves an iterations x cells array for comparison
        
    sig_si=[]
    for n in range(len(si)):
        if si[n]>N.percentile(si_shifted_total.T[n],95):
            sig_si.append(1)
        else:
            sig_si.append(0)
            
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


    res={'av hist':hist,'hist':hist_total,'hist first':hist_first,'hist second':hist_second,'hist odd':hist_odd,'hist even':hist_even,
         'si':N.asarray(si),'sig si':N.asarray(sig_si),'hist conc':hist_conc,'hist conc shifted':hist_conc_shift,
         'X':X,'pos':lin_pos_total,'peak location':N.asarray(peak_location),'significant peak':N.asarray(sig_peak),'peak width':N.asarray(peak_width),'consistency':rs_total,'consistency fs':rs_fs_total,
         'mean activity':mean_act,'speed':speed_trial,'speed hist':speed_hist}
    return res

def get_X(traces,start1,end1,start2,end2,track,skel,conv,is_left=True,bins=bins):    
    
    X=[]    
    for i in range(len(traces)):
        for n in range(len(start1)): # Iterate over trials.
            X.extend(traces[i][start1[n]:end1[n]])
            X.extend(traces[i][start2[n]:end2[n]])

    X=N.reshape(X,[len(traces),-1])    
        
    res={'X':X}
    return res

def get_X_single_period(traces,start1,end1):    
    
    X=[]    
    for i in range(len(traces)):
        for n in range(len(start1)): # Iterate over trials.
            X.extend(traces[i][start1[n]:end1[n]])

    X=N.reshape(X,[len(traces),-1])    
        
    res={'X':X}
    return res


def get_single_trial_tuning_functions_simple(traces,start1,end1,start2,end2,track,skel,is_left=True,bins=bins):    
    if len(start1)>0 and len(start2)>0:
    
        # Get speed trace.
        speed=[]
        for n in range(1,len(track[0])):
            speed.append((N.sqrt(((track[1][n]-track[1][n-1])**2)+((track[0][n]-track[0][n-1])**2)))*20) # In cm/s
        
        speed_trial=[]
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
            speed_trial.extend(speed[start1[n]:end1[n]])
            speed_trial.extend(speed[start2[n]:end2[n]])
        
        
        X=[]    
        hist_total=[]
        for i in range(len(traces)):
            hist_trials=N.empty((len(start1),bins))
            for n in range(len(start1)): # Iterate over trials.
                X.extend(traces[i][start1[n]:end1[n]])
                X.extend(traces[i][start2[n]:end2[n]])
                X_local=[]
                X_local.extend(traces[i][start1[n]:end1[n]])
                X_local.extend(traces[i][start2[n]:end2[n]])
                
                hist,a,b=st.binned_statistic(lin_pos_trial[n],X_local,statistic='mean',bins=bins)
                occ,a=N.histogram(lin_pos_trial[n],bins=bins) # Get occupancy map.
                occ=occ/N.sum(occ) # Convert to probability.
                hist=hist/occ # This is the single trial tuning function.
                hist_trials[n]=hist
            
            
            hist_total.append(hist_trials)      
            
        X=N.reshape(X,[len(traces),-1])    
        mean_act=N.nanmean(X,axis=1)   
        
           
        
        peak_location=[]
        # Get total tuning function.
    
        hist,a,b=st.binned_statistic(lin_pos_total,X,statistic='mean',bins=bins)
        occ,a=N.histogram(lin_pos_total,bins=bins) # Get occupancy map.
        occ=occ/N.sum(occ) # Convert to probability.
        hist_norm=hist/occ
        hist_norm=N.asarray(hist_norm)
        
        hist_norm2=N.empty_like(hist_norm)
        for n in range(len(hist_norm)):
            hist_norm2[n]=olf.rescale_pos(hist_norm[n],0,1)
            peak_location.append(N.argmax(hist_norm2[n]))
        hist=hist_norm2
        
        # Get SI
        si=[]
        for i in range(len(hist)): # Iterate over transients.
            local=[]
            for n in range(len(hist[0])): # Iterate over the spatial tuning function.
                local.append(hist[i][n]*N.log(hist[i][n]/N.nanmean(hist[i]))*occ[n]) # Following Cholvin et al 2021 to yield bits/s
            si.append(N.nansum(local))
        
        
        
        speed_hist,a,b=st.binned_statistic(lin_pos_total,speed_trial,bins=bins)
        res={'av hist':hist,'hist':hist_total,'si':N.asarray(si),
             'X':X,'pos':lin_pos_total,'peak location':N.asarray(peak_location),'mean activity':mean_act,'speed':speed_trial,'speed hist':speed_hist,'no trials':len(start1)}
    else:
        hist=N.zeros((len(traces),bins))
        hist[:]=N.nan
        si=N.empty((len(traces)))
        si[:]=N.nan        
        res={'av hist':hist,'si':N.asarray(si),'speed':N.nan,'speed hist':N.nan,'mean activity':si,'no trials':0}
    return res


def get_mean_activity(traces,start1,end1):    
       
    X=[]
    
    for i in range(len(traces)):
        for n in range(len(start1)): # Iterate over trials.
            X.extend(traces[i][start1[n]:end1[n]])

    X=N.reshape(X,[len(traces),-1])    
    mean_act=N.nanmean(X,axis=1)   
    
    return mean_act  

def get_mean_activity_single_trials(traces,start1,end1):    
       
    act=N.empty((len(traces),len(start1))) # An empty array of shape neurons x trials.
    
    for i in range(len(traces)):
        for n in range(len(start1)): # Iterate over trials.
            act[i][n]=N.nanmean(traces[i][start1[n]:end1[n]])
    
    return N.nanmean(act,axis=1) 
    
def analyse_spatial_tuning(filename,filename1,name,session,temp_fix=False):
    

    if temp_fix==True:
        os.chdir(skeleton_dir_original)
        skel=N.load("%s_skeleton.npy" %name,allow_pickle=True).item() 
        f = h5py.File(filename, 'r')
        
    else:
        f = h5py.File(filename, 'r')
        os.chdir(skeleton_dir)
        skel=N.load("%s_skeleton.npy" %name,allow_pickle=True).item()    

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
    
    
    
    traces=transients[:,session[0]]
    traces_raw=traces_normed[:,session[0]]
    # Get all correct left and right runs outward and inward.
    source_corr=N.asarray(f['%s/behavior/corr'%session[0]])
    source_fail=N.asarray(f['%s/behavior/fail'%session[0]])
    
    is_corr=len(source_corr[0::2])
    is_error=len(source_fail[0::2])
    prop_corr=is_corr/(is_corr+is_error)
    
    
    source_oleft1=olf.separate_in(N.asarray(f['%s/behavior/ocl' %session[0]]),source_corr)
    source_oright1=olf.separate_in(N.asarray(f['%s/behavior/ocr' %session[0]]),source_corr)
    source_oleft2=olf.separate_in(N.asarray(f['%s/behavior/osl' %session[0]]),source_corr)
    source_oright2=olf.separate_in(N.asarray(f['%s/behavior/osr' %session[0]]),source_corr)
    source_sample_left=olf.separate_in(N.asarray(f['%s/behavior/sampl' %session[0]]),source_corr)
    source_sample_right=olf.separate_in(N.asarray(f['%s/behavior/sampr' %session[0]]),source_corr)
    source_rew_left=olf.separate_in(N.asarray(f['%s/behavior/rewl' %session[0]]),source_corr)
    source_rew_right=olf.separate_in(N.asarray(f['%s/behavior/rewr' %session[0]]),source_corr)
    
    a,source_ileft1=olf.separate_in_dual_conditional(N.asarray(f['%s/behavior/ocl' %session[0]]),N.asarray(f['%s/behavior/isl' %session[0]]),source_corr)
    a,source_iright1=olf.separate_in_dual_conditional(N.asarray(f['%s/behavior/ocr' %session[0]]),N.asarray(f['%s/behavior/isr' %session[0]]),source_corr)
    a,source_ileft2=olf.separate_in_dual_conditional(N.asarray(f['%s/behavior/ocl' %session[0]]),N.asarray(f['%s/behavior/icl' %session[0]]),source_corr)
    a,source_iright2=olf.separate_in_dual_conditional(N.asarray(f['%s/behavior/ocr' %session[0]]),N.asarray(f['%s/behavior/icr' %session[0]]),source_corr)
       
    source_track=N.asarray(f['%s/behavior/track'%session[0]])
    
    # Case left.
    start1=source_oleft1[0::2]
    end1=source_oleft2[1::2]
    start2=source_ileft1[0::2]
    end2=source_ileft2[1::2]
    res_left=get_single_trial_tuning_functions(traces,start1,end1,start2,end2,source_track,skel,conv_left,is_left=True,bins=bins)
    res_left_raw=get_X(traces_raw,start1,end1,start2,end2,source_track,skel,conv_left,is_left=True,bins=bins)
    res_left_sampling=get_X_single_period(traces,source_sample_left[0::2],source_sample_left[1::2]) # Get the sampling phase.
    sampling_act_left=get_mean_activity_single_trials(traces,source_sample_left[0::2],source_sample_left[1::2]) #ä Get the mean activity during left sampling.
    rew_act_left=get_mean_activity_single_trials(traces,source_rew_left[0::2],source_rew_left[1::2]) #ä Get the mean activity during left sampling.
    
    
    # Get mean activity of all cells during outward travel.
    mean_act_left_out=get_mean_activity(traces,start1,end1)   
    mean_act_left_in=get_mean_activity(traces,start2,end2)
    mean_act_left_total=get_mean_activity(traces,start1,end2)
        
    # Case right.
    start1=source_oright1[0::2]
    end1=source_oright2[1::2]
    start2=source_iright1[0::2]
    end2=source_iright2[1::2]
    res_right=get_single_trial_tuning_functions(traces,start1,end1,start2,end2,source_track,skel,conv_right,is_left=False,bins=bins)
    res_right_raw=get_X(traces_raw,start1,end1,start2,end2,source_track,skel,conv_right,is_left=False,bins=bins)
    res_right_sampling=get_X_single_period(traces,source_sample_right[0::2],source_sample_right[1::2]) # Get the sampling phase.
    sampling_act_right=get_mean_activity_single_trials(traces,source_sample_right[0::2],source_sample_right[1::2]) #ä Get the mean activity during left sampling.
    rew_act_right=get_mean_activity_single_trials(traces,source_rew_right[0::2],source_rew_right[1::2]) #ä Get the mean activity during left sampling.
    
    
    mean_act_right_out=get_mean_activity(traces,start1,end1)
    mean_act_right_in=get_mean_activity(traces,start2,end2)
    mean_act_right_total=get_mean_activity(traces,start1,end2)
    
    act_diff_score_out,act_diff_score_in,act_diff_score_total=[],[],[]
    for n in range(len(mean_act_left_out)):
        act_diff_score_out.append(N.abs((mean_act_left_out[n]-mean_act_right_out[n])/(mean_act_left_out[n]+mean_act_right_out[n])))
        act_diff_score_in.append(N.abs((mean_act_left_in[n]-mean_act_right_in[n])/(mean_act_left_in[n]+mean_act_right_in[n])))
        act_diff_score_total.append(N.abs((mean_act_left_total[n]-mean_act_right_total[n])/(mean_act_left_total[n]+mean_act_right_total[n])))
    
    # Get left/right average activity during out and in.

    mean_act_out=N.empty((2,len(mean_act_left_out)))
    mean_act_out[0]=mean_act_left_out
    mean_act_out[1]=mean_act_right_out
    mean_act_out=N.mean(mean_act_out,axis=0)
    
    mean_act_in=N.empty((2,len(mean_act_left_in)))
    mean_act_in[0]=mean_act_left_in
    mean_act_in[1]=mean_act_right_in
    mean_act_in=N.mean(mean_act_in,axis=0)
    
    av_hist_left=res_left['av hist']
    av_hist_right=res_right['av hist']
    hist_left=res_left['hist']
    hist_left_conc=res_left['hist conc']
    hist_left_conc_shifted=res_left['hist conc shifted']
    hist_right=res_right['hist']
    hist_right_conc=res_right['hist conc']
    hist_right_conc_shifted=res_right['hist conc shifted']
    hist_left_first=res_left['hist first']
    hist_right_first=res_right['hist first']
    hist_left_second=res_left['hist second']
    hist_right_second=res_right['hist second']
    
    hist_left_odd=res_left['hist odd']
    hist_right_odd=res_right['hist odd']
    hist_left_even=res_left['hist even']
    hist_right_even=res_right['hist even']
    
    # Separately run computation on error trials only.
    source_corr=N.asarray(f['%s/behavior/fail'%session[0]]) # This defines non-correct trials
    source_oleft1=olf.separate_in(N.asarray(f['%s/behavior/ocl' %session[0]]),source_corr)
    source_oright1=olf.separate_in(N.asarray(f['%s/behavior/ocr' %session[0]]),source_corr)
    source_oleft2=olf.separate_in(N.asarray(f['%s/behavior/osl' %session[0]]),source_corr)
    source_oright2=olf.separate_in(N.asarray(f['%s/behavior/osr' %session[0]]),source_corr)
    
    
    a,source_ileft1=olf.separate_in_dual_conditional(N.asarray(f['%s/behavior/ocl' %session[0]]),N.asarray(f['%s/behavior/isl' %session[0]]),source_corr)
    a,source_iright1=olf.separate_in_dual_conditional(N.asarray(f['%s/behavior/ocr' %session[0]]),N.asarray(f['%s/behavior/isr' %session[0]]),source_corr)
    a,source_ileft2=olf.separate_in_dual_conditional(N.asarray(f['%s/behavior/ocl' %session[0]]),N.asarray(f['%s/behavior/icl' %session[0]]),source_corr)
    a,source_iright2=olf.separate_in_dual_conditional(N.asarray(f['%s/behavior/ocr' %session[0]]),N.asarray(f['%s/behavior/icr' %session[0]]),source_corr)
          
    # Case left.
    start1=source_oleft1[0::2]
    end1=source_oleft2[1::2]
    start2=source_ileft1[0::2]
    end2=source_ileft2[1::2]

    res_left_fail=get_single_trial_tuning_functions_simple(traces,start1,end1,start2,end2,source_track,skel,is_left=True,bins=bins)
           
    # Case right.
    start1=source_oright1[0::2]
    end1=source_oright2[1::2]
    start2=source_iright1[0::2]
    end2=source_iright2[1::2]
    res_right_fail=get_single_trial_tuning_functions_simple(traces,start1,end1,start2,end2,source_track,skel,is_left=False,bins=bins)
    
    
    # Keep neurons with non-zero activity and for which consistenty could be computed during during both left and right trials.
    ind=[]
    for n in range(len(av_hist_left)):
        if N.nansum(av_hist_left[n])>0 and N.nansum(av_hist_right[n])>0 and not N.isnan(res_left['consistency'][n]) and not N.isnan(res_right['consistency'][n]):
            ind.append(n)
            
    av_hist_left=av_hist_left[ind]
    av_hist_right=av_hist_right[ind]
    
    av_hist_left_fail=res_left_fail['av hist'][ind]
    av_hist_right_fail=res_right_fail['av hist'][ind]

    peak_location_left=res_left['peak location'][ind]
    peak_location_right=res_right['peak location'][ind]
    peak_width_left=res_left['peak width'][ind]
    peak_width_right=res_right['peak width'][ind]
    
    act_sample_left=sampling_act_left[ind]
    act_sample_right=sampling_act_right[ind]
    act_sample_mean=N.empty((2,len(act_sample_left)))
    act_sample_mean[0]=act_sample_left
    act_sample_mean[1]=act_sample_right
    act_sample_mean=N.nanmean(act_sample_mean,axis=0)
    
    
    act_rew_left=rew_act_left[ind]
    act_rew_right=rew_act_right[ind]
    act_rew_mean=N.empty((2,len(act_rew_left)))
    act_rew_mean[0]=act_rew_left
    act_rew_mean[1]=act_rew_right
    act_rew_mean=N.nanmean(act_rew_mean,axis=0)

    hist_left_corrected,hist_right_corrected=[],[]
    hist_left_conc_corrected,hist_right_conc_corrected=[],[]
    hist_left_conc_corrected_shifted,hist_right_conc_corrected_shifted=[],[]
    hist_left_first_corrected,hist_right_first_corrected=[],[]
    hist_left_second_corrected,hist_right_second_corrected=[],[]
    hist_left_odd_corrected,hist_right_odd_corrected=[],[]
    hist_left_even_corrected,hist_right_even_corrected=[],[]
    
    act_diff_score_out_corrected,act_diff_score_in_corrected=[],[]
    act_diff_score_total_corrected=[]
    mean_act_in_corrected,mean_act_out_corrected=[],[]

    for n in range(len(hist_left)):
        if n in ind:
            hist_left_corrected.append(hist_left[n])
            hist_right_corrected.append(hist_right[n])
            
            hist_left_conc_corrected.append(hist_left_conc[n])
            hist_right_conc_corrected.append(hist_right_conc[n])
            
            hist_left_conc_corrected_shifted.append(hist_left_conc_shifted[n])
            hist_right_conc_corrected_shifted.append(hist_right_conc_shifted[n])
            
            hist_left_first_corrected.append(hist_left_first[n])
            hist_right_first_corrected.append(hist_right_first[n])            
            hist_left_second_corrected.append(hist_left_second[n])
            hist_right_second_corrected.append(hist_right_second[n])
            
            hist_left_odd_corrected.append(hist_left_odd[n])
            hist_right_odd_corrected.append(hist_right_odd[n])            
            hist_left_even_corrected.append(hist_left_even[n])
            hist_right_even_corrected.append(hist_right_even[n])
            
            act_diff_score_out_corrected.append(act_diff_score_out[n])
            act_diff_score_in_corrected.append(act_diff_score_in[n])
            act_diff_score_total_corrected.append(act_diff_score_total[n])
            
            mean_act_out_corrected.append(mean_act_out[n])
            mean_act_in_corrected.append(mean_act_in[n])
            
      
    
    sig_peak_left=N.asarray(res_left['significant peak'][ind])
    sig_peak_right=N.asarray(res_right['significant peak'][ind])
    
    # Get cells with sig si on any of the run types.
    sig_si_left_corrected=res_left['sig si'][ind]
    sig_si_right_corrected=res_right['sig si'][ind]
    sig_si_corrected=[]
    for n in range(len(sig_si_left_corrected)):
        if sig_si_left_corrected[n]==1 or sig_si_right_corrected[n]:
            sig_si_corrected.append(1)
        else:
            sig_si_corrected.append(0)
    sig_si_corrected=N.asarray(sig_si_corrected)    
    
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
    X_left_raw=res_left_raw['X'][ind]
    X_right_raw=res_right_raw['X'][ind]
    X_left_sampling=res_left_sampling['X'][ind]
    X_right_sampling=res_right_sampling['X'][ind]
    if res_left_fail['no trials']>0:
        X_left_fail=res_left_fail['X'][ind]
        pos_left_fail=res_left_fail['pos']               
    else:
        X_left_fail=N.nan
        pos_left_fail=N.nan
    if res_right_fail['no trials']>0:
        X_right_fail=res_right_fail['X'][ind]
        pos_right_fail=res_right_fail['pos']
    else:
        X_right_fail=N.nan
        pos_right_fail=N.nan
    
    X=N.append(X_left,X_right,axis=1)
    X_raw=N.append(X_left_raw,X_right_raw,axis=1)
    X_sampling=N.append(X_left_sampling,X_right_sampling,axis=1)
    pos_total=N.append(res_left['pos'],N.asarray(res_right['pos'])*-1)
    pos_left=res_left['pos']
    pos_right=res_right['pos']
    speed_total=N.append(res_left['speed'],res_right['speed'])
    speed_total_fail=N.append(res_left_fail['speed'],res_right_fail['speed'])
    
    if not N.isnan(N.sum(res_left_fail['speed hist'])) and not N.isnan(N.sum(res_right_fail['speed hist'])):
        speed_hist_behav=N.append(res_left['speed hist'],res_right['speed hist'])
        speed_hist_behav=N.mean(N.reshape(speed_hist_behav,[2,-1]),axis=0)
        
        speed_hist_behav_fail=N.append(res_left_fail['speed hist'],res_right_fail['speed hist'])
        speed_hist_behav_fail=N.mean(N.reshape(speed_hist_behav_fail,[2,-1]),axis=0)
    else:
        speed_hist_behav=N.nan
        speed_hist_behav_fail=N.nan
    
    # Get speed histogram for each cell.
    speed_hist=st.binned_statistic(speed_total,X,bins=speed_bins,range=(0,speed_max))[0]
    speed_mod_p=N.empty((len(speed_hist)))
    speed_mod_r=N.empty_like(speed_mod_p)
    for n in range(len(speed_hist)):
        r,p=st.spearmanr(range(len(speed_hist[n])),speed_hist[n],nan_policy='omit')
        if p<0.05:
            speed_mod_p[n]=1
            speed_mod_r[n]=r
        else:
            speed_mod_p[n]=0
            speed_mod_r[n]=N.nan
    
    # Get grouping of all cells in same, side and cat.
    same,side,cat=[],[],[]
    for n in range(len(r_left_right)):
        if r_left_right[n]>0.5 and xcorr_shift[n]<15:
            same.append(n)
        elif r_left_right[n]<0.5 and xcorr_shift[n]<15:
            side.append(n)
        elif 15<xcorr_shift[n]<25:
            cat.append(n)
    same=N.asarray(same)
    side=N.asarray(side)
    cat=N.asarray(cat)
    
    cons_left=N.asarray(res_left['consistency'])[ind]
    cons_right=N.asarray(res_right['consistency'])[ind]
    cons_fs_left=N.asarray(res_left['consistency fs'])[ind]
    cons_fs_right=N.asarray(res_right['consistency fs'])[ind]
    #cons_total=[]
    #cons_total.extend(cons_left)
    #cons_total.extend(cons_right)
    #cons_total=N.nanmean(N.reshape(cons_total,[2,-1]),axis=0)
        
    
    mean_act_left=N.asarray(res_left['mean activity'])[ind]
    mean_act_right=N.asarray(res_right['mean activity'])[ind]
    mean_act=N.empty((2,len(mean_act_left)))
    mean_act[0]=mean_act_left
    mean_act[1]=mean_act_right
    mean_act=N.mean(mean_act,axis=0)
    
    # Get preferred consistency by using the side with larger mean activity.
    cons_total=[]
    cons_fs_total=[]
    for n in range(len(mean_act_left)):
        if mean_act_left[n]>mean_act_right[n]:
            cons_total.append(cons_left[n])
            cons_fs_total.append(cons_fs_left[n])
        elif mean_act_left[n]<mean_act_right[n]:
            cons_total.append(cons_right[n])
            cons_fs_total.append(cons_fs_right[n])
    
    
    
    if not N.isnan(N.sum(res_left_fail['speed hist'])) and not N.isnan(N.sum(res_right_fail['speed hist'])):
        mean_act_left=N.asarray(res_left_fail['mean activity'])[ind]
        mean_act_right=N.asarray(res_right_fail['mean activity'])[ind]
        mean_act_fail=N.empty((2,len(mean_act_left)))
        mean_act_fail[0]=mean_act_left
        mean_act_fail[1]=mean_act_right
        mean_act_fail=N.mean(mean_act_fail,axis=0)
    else:
        mean_act_fail=N.nan
    
    
    si_left=res_left['si'][ind]
    si_left_fail=res_left_fail['si'][ind]
    si_right=res_right['si'][ind]
    si_right_fail=res_right_fail['si'][ind]
    si_mean=N.empty((2,len(si_left)))
    si_mean[0]=si_left
    si_mean[1]=si_right
    si_mean=N.mean(si_mean,axis=0)
    res_total={'av hist left':av_hist_left,'av hist right':av_hist_right,'hist left':hist_left_corrected,'hist right':hist_right_corrected,'hist left conc':N.asarray(hist_left_conc_corrected),'hist right conc':N.asarray(hist_right_conc_corrected),
               'hist left conc shifted':N.asarray(hist_left_conc_corrected_shifted),'hist right conc shifted':N.asarray(hist_right_conc_corrected_shifted),
               'hist left first':hist_left_first_corrected,'hist right first':hist_right_first_corrected,
               'hist left second':hist_left_second_corrected,'hist right second':hist_right_second_corrected,
               'hist left odd':hist_left_odd_corrected,'hist right odd':hist_right_odd_corrected,
               'hist left even':hist_left_even_corrected,'hist right even':hist_right_even_corrected,
               'av hist left fail':av_hist_left_fail,'av hist right fail':av_hist_right_fail,
               'r left right':r_left_right,'median r left right':median_r_left_right,
               'consistency total':cons_total,'consistency left':cons_left,'consistency right':cons_right,'consistency fs total':cons_fs_total,
               'xcorr shift':xcorr_shift,
               'mean activity left':mean_act_left,'mean activity right':mean_act_right,'mean activity':mean_act,'mean activity fail':mean_act_fail,
               'mean activity in':mean_act_in_corrected,'mean activity out':mean_act_out_corrected,
               'si left':si_left,'si right':si_right,'si mean':si_mean,'si left fail':si_left_fail,'si right fail':si_right_fail,'sig si':sig_si_corrected,
               'peak location left':peak_location_left,'peak location right':peak_location_right,
               'significant peak left':sig_peak_left,'significant peak right':sig_peak_right,
               'peak width left':peak_width_left,'peak width right':peak_width_right,
               'sampling act left':act_sample_left,'sampling act right':act_sample_right,'sampling act mean':act_sample_mean,
               'reward act left':act_rew_left,'reward act right':act_rew_right,'reward act mean':act_rew_mean,
               'X':X,'X raw':X,'position':pos_total,'X left':X_left,'X right':X_right,'X left raw':X_left_raw,'X right raw':X_right_raw,'position left':pos_left,'position right':pos_right,
               'X left fail':X_left_fail,'X right fail':X_right_fail,'position left fail':pos_left_fail,'position right fail':pos_right_fail,
               'X left sampling':X_left_sampling,'X right sampling':X_right_sampling,'X sampling':X_sampling,
               'same':same,'side':side,'cat':cat,'# neurons':len(traces),
               'used indices':ind,'speed hist':speed_hist,'speed total':speed_total,'speed total fail':speed_total_fail,'speed distribution':speed_hist_behav,'speed distribution fail':speed_hist_behav_fail,
               'speed mod p':N.asarray(speed_mod_p),'speed mod r':N.asarray(speed_mod_r),
               'act score out':act_diff_score_out_corrected,'act score in':act_diff_score_in_corrected,'act score total':act_diff_score_total_corrected,
               'prop corr':N.asarray(prop_corr)}
    os.chdir(target_dir)
    N.save("%s_results_spatial_tuning.npy" %name,res_total)
    return res_total


# Executed part.
filename_list=['/media/jonas/data3/HM_wm_data/playground/block12_analysis_final/44/gcamp6f44_cellreg_block1_2_curated_behavior_analysis.hdf5',
               '/media/jonas/data3/HM_wm_data/playground/block12_analysis_final/45/gcamp6f45_cellreg_block1_2_curated_behavior_analysis.hdf5',
               '/media/jonas/data3/HM_wm_data/playground/block12_analysis_final/47/gcamp6f47_cellreg_block1_2_curated_behavior_analysis.hdf5',
               '/media/jonas/data3/HM_wm_data/playground/block12_analysis_final/93/gcamp6f93_cellreg_block1_2_curated_behavior_analysis.hdf5',
               '/media/jonas/data3/HM_wm_data/playground/block12_analysis_final/94/gcamp6f94_cellreg_block1_2_curated_behavior_analysis.hdf5',
               '/media/jonas/data3/HM_wm_data/playground/block12_analysis_final/216/gcamp6f216_cellreg_block1_2_curated_behavior_analysis.hdf5',
               "/media/jonas/data3/HM_wm_data/playground/block12_analysis_final/218/gcamp6f218_cellreg_block1_2_curated_behavior_analysis.hdf5",
               '/media/jonas/data3/HM_wm_data/playground/block12_analysis_final/219/gcamp6f219_cellreg_block1_2_curated_behavior_analysis.hdf5',
               '/media/jonas/data3/HM_wm_data/playground/learning_block/GCaMP6f_478-485/CellReg_478_learning_behavior_analysis_spikes.hdf5',
               '/media/jonas/data3/HM_wm_data/playground/learning_block/GCaMP6f_478-485/CellReg_481_learning_behavior_analysis_spikes.hdf5',
               '/media/jonas/data3/HM_wm_data/playground/learning_block/GCaMP6f_478-485/CellReg_483_learning_behavior_analysis_spikes.hdf5',
               '/media/jonas/data3/HM_wm_data/playground/learning_block/GCaMP6f_478-485/CellReg_485_learning_behavior_analysis_spikes.hdf5']
               
animal_list=["44","45","47","93","94","216","218","219","478","481","483","485"]
session_list=[0,0,0,0,0,0,0,0,7,7,7,7]           
temp_fix_list=[True,True,True,True,True,True,True,True,False,False,False,False]

hist_left,hist_right=[],[]
hist_left_first,hist_right_first=[],[]
hist_left_second,hist_right_second=[],[]
hist_left_odd,hist_right_odd=[],[]
hist_left_even,hist_right_even=[],[]
hist_left_fail,hist_right_fail=[],[]
sig_left,sig_right=[],[]
sig_both,sig_none=[],[]
xcorr_shift=[]
r_left_right,median_r_left_right=[],[]
si_left,si_right=[],[]
si_left_fail,si_right_fail=[],[]
si_mean=[]
sig_si=[]
sig_peak_left,sig_peak_right=[],[]
props_sig_si_side_mouse,props_sig_si_same_mouse,props_sig_si_cat_mouse=[],[],[]
prop_cat,prop_side,prop_same=[],[],[]
cons_total,cons_fs_total=[],[]
cons_total_mouse,cons_fs_total_mouse=[],[]
mean_act_left,mean_act_right,mean_act=[],[],[]
no_cells,no_cells_total=[],[]
si_left_mouse,si_right_mouse,si_mean_mouse=[],[],[]
mean_act_mouse=[]
cons_cat_mouse,cons_same_mouse,cons_side_mouse=[],[],[]
cons_fs_cat_mouse,cons_fs_same_mouse,cons_fs_side_mouse=[],[],[]

si_mean_cat_mouse,si_mean_same_mouse,si_mean_side_mouse=[],[],[]
act_score_in,act_score_out,act_score_total=[],[],[]
peak_location_left,peak_location_right=[],[]
peak_width_left,peak_width_right=[],[]
speed_hist=[]
speed_hist_behav_fail,speed_dist_behav=[],[]
speed_total,speed_total_fail=[],[]
speed_hist_behav, speed_hist_behav_fail=[],[]
speed_mod_p,speed_mod_r=[],[]

mean_act_com_c,mean_act_com_f=[],[]
mean_act_out,mean_act_in=[],[]
act_sample_left,act_sample_right,act_sample_mean=[],[],[]
sample_disc=[]
act_rew_left,act_rew_right,act_rew_mean=[],[],[]
rew_disc=[]
prop_corr=[]

n_reg=1
mouse_id_counter=0
mouse_id=[]
mouse_index_array=[]

X_by_mouse,X_left_by_mouse,X_right_by_mouse=[],[],[]
position_left_by_mouse,position_right_by_mouse,position_by_mouse=[],[],[]
for animal in range(len(animal_list)):
    filename=filename_list[animal]
    filename1=""
    name=animal_list[animal]
    temp_fix=temp_fix_list[animal]
    session=[session_list[animal]]
    print(name)
    res=analyse_spatial_tuning(filename,filename1,name,session,temp_fix=temp_fix)

    if animal==0: # Account for the array concatenation difference between index 0 and the rest.
        av_hist_left=res['av hist left']
        av_hist_right=res['av hist right']
        av_hist_left_fail=res['av hist left fail']
        av_hist_right_fail=res['av hist right fail']
        speed_hist=res['speed hist']
    else:
        av_hist_left=N.append(av_hist_left,res['av hist left'],axis=0)
        av_hist_right=N.append(av_hist_right,res['av hist right'],axis=0)
        av_hist_left_fail=N.append(av_hist_left_fail,res['av hist left fail'],axis=0)
        av_hist_right_fail=N.append(av_hist_right_fail,res['av hist right fail'],axis=0)
        speed_hist=N.append(speed_hist,res['speed hist'],axis=0)
    
    for n in range(len(res['hist left'])):
        hist_left.append(res['hist left'][n])
        hist_left_first.append(res['hist left first'][n])
        hist_left_second.append(res['hist left second'][n])
        hist_left_odd.append(res['hist left odd'][n])
        hist_left_even.append(res['hist left even'][n])
    for n in range(len(res['hist right'])):    
        hist_right.append(res['hist right'][n])
        hist_right_first.append(res['hist right first'][n])
        hist_right_second.append(res['hist right second'][n])
        hist_right_odd.append(res['hist right odd'][n])
        hist_right_even.append(res['hist right even'][n])

    xcorr_shift.extend(res['xcorr shift'])
    r_left_right.extend(res['r left right'])
    median_r_left_right.append(res['median r left right'])
    prop_cat.append(len(res['cat'])/len(res['av hist left'])) # Get the fraction of neurons as a function of active cells.
    prop_side.append(len(res['side'])/len(res['av hist left']))
    prop_same.append(len(res['same'])/len(res['av hist left']))
    
    props_sig_si_side_mouse.append(N.sum(res['sig si'][res['side']])/len(res['side']))
    props_sig_si_same_mouse.append(N.sum(res['sig si'][res['same']])/len(res['same']))
    props_sig_si_cat_mouse.append(N.sum(res['sig si'][res['cat']])/len(res['cat']))
    
    cons_total.extend(res['consistency total'])
    cons_fs_total.extend(res['consistency fs total'])
    mean_act_left.extend(res['mean activity left'])
    mean_act_right.extend(res['mean activity right'])
    mean_act.extend(res['mean activity'])
    si_left.extend(res['si left'])
    si_right.extend(res['si right'])
    si_mean.extend(res['si mean'])
    si_left_mouse.append(N.nanmean(res['si left']))
    si_right_mouse.append(N.nanmean(res['si right']))
    si_mean_mouse.append(N.nanmean(res['si mean']))
    si_mean_same_mouse.append(N.nanmean(res['si mean'][res['same']]))
    sig_si.extend(res['sig si'])
    sig_peak_left.extend(res['significant peak left'])
    sig_peak_right.extend(res['significant peak right'])
    speed_mod_p.extend(res['speed mod p'])
    speed_mod_r.extend(res['speed mod r'])
    
    mean_act_mouse.append(N.nanmean(res['mean activity']))
    no_cells.append(len(res['av hist left']))
    cons_cat_mouse.append(N.nanmean(N.asarray(res['consistency total'])[N.asarray(res['cat'])]))
    cons_same_mouse.append(N.nanmean(N.asarray(res['consistency total'])[N.asarray(res['same'])]))
    cons_side_mouse.append(N.nanmean(N.asarray(res['consistency total'])[N.asarray(res['side'])]))
    cons_fs_cat_mouse.append(N.nanmean(N.asarray(res['consistency fs total'])[N.asarray(res['cat'])]))
    cons_fs_same_mouse.append(N.nanmean(N.asarray(res['consistency fs total'])[N.asarray(res['same'])]))
    cons_fs_side_mouse.append(N.nanmean(N.asarray(res['consistency fs total'])[N.asarray(res['side'])]))
    act_score_in.extend(res['act score in'])
    act_score_out.extend(res['act score out'])
    peak_location_left.extend(res['peak location left'])
    peak_location_right.extend(res['peak location right'])
    peak_width_left.extend(res['peak width left'])
    peak_width_right.extend(res['peak width right'])
    act_score_total.extend(res['act score total'])
    si_left_fail.extend(res['si left fail'])
    si_right_fail.extend(res['si right fail'])
    speed_total.append(res['speed total'])
    speed_total_fail.append(res['speed total fail'])
    mean_act_out.extend(res['mean activity out'])
    mean_act_in.extend(res['mean activity in'])
    
    if not N.isnan(N.sum(res['speed distribution fail'])):
        speed_hist_behav_fail.append(res['speed distribution fail'])
        speed_hist_behav.append(res['speed distribution'])
        mean_act_com_c.extend(res['mean activity'])
        mean_act_com_f.extend(res['mean activity fail'])
    
    act_sample_left.extend(res['sampling act left'])
    act_sample_right.extend(res['sampling act right'])
    act_sample_mean.extend(res['sampling act mean'])

    act_rew_left.extend(res['reward act left'])
    act_rew_right.extend(res['reward act right'])
    act_rew_mean.extend(res['reward act mean'])
    no_cells_total.append(res['# neurons'])
    mouse_id.append(mouse_id_counter)
    id_marker=N.zeros((len(res['av hist left'])))
    id_marker+=mouse_id_counter # Keep track of which neuron belongs to which mouse.
    mouse_index_array.extend(id_marker)
    
    
    mouse_id_counter+=1
    X_by_mouse.append(res['X'])
    X_left_by_mouse.append(res['X left'])
    X_right_by_mouse.append(res['X right'])
    position_by_mouse.append(res['position'])
    position_left_by_mouse.append(res['position left'])
    position_right_by_mouse.append(res['position right'])
    X_left_by_mouse.append(res['X sampling'])
    X_right_by_mouse.append(res['X sampling'])
    prop_corr.append(res['prop corr'])
    

cons_total_mouse={
    'cat':N.asarray(cons_cat_mouse),
    'same':N.asarray(cons_same_mouse),
    'side':N.asarray(cons_side_mouse)
    }

cons_fs_total_mouse={
    'cat':N.asarray(cons_fs_cat_mouse),
    'same':N.asarray(cons_fs_same_mouse),
    'side':N.asarray(cons_fs_side_mouse)
    }

X_classes_by_mouse={'X':X_by_mouse,'X left':X_left_by_mouse,'X right':X_right_by_mouse,
                    'position by mouse':position_by_mouse,'position left by mouse':position_left_by_mouse,'position right by mouse':position_right_by_mouse}

mean_act_comp={'mean act correct':mean_act_com_c,'mean act fail':mean_act_com_f}

props_sig_si_by_mouse={'side':N.asarray(props_sig_si_side_mouse),'same':N.asarray(props_sig_si_same_mouse),'cat':N.asarray(props_sig_si_cat_mouse)}


# Get grouping of all cells in same, side and cat.
same,side,cat=[],[],[]
for n in range(len(r_left_right)):
    if r_left_right[n]>0.5 and xcorr_shift[n]<15:
        same.append(n)
    elif r_left_right[n]<0.5 and xcorr_shift[n]<15:
        side.append(n)
    elif 15<xcorr_shift[n]<25:
        cat.append(n)
same=N.asarray(same)
side=N.asarray(side)
cat=N.asarray(cat)

# Get peak-normalized speed histogram and the r-values of significantly speed-modulated neurons of all types.
speed_hist_norm=N.empty_like(speed_hist)
for n in range(len(speed_hist)):
    speed_hist_norm[n]=olf.rescale_pos(speed_hist[n],0,1)    


sig_mod_side,sig_mod_same,sig_mod_cat=[],[],[]
for n in range(len(speed_mod_p)):
    if speed_mod_p[n]==1:
        if n in side:
            sig_mod_side.append(n)
        elif n in same:
            sig_mod_same.append(n)
        elif n in cat:
            sig_mod_cat.append(n)

sig_mod_same=N.asarray(sig_mod_same)
sig_mod_side=N.asarray(sig_mod_side)
sig_mod_cat=N.asarray(sig_mod_cat)


speed_modulated={'ind same':sig_mod_same,'ind side':sig_mod_side,'ind cat':sig_mod_cat}
# Compute sampling discrimination for all neurons.
sample_disc=[]
for n in range(len(act_sample_left)):
    if (act_sample_left[n]+act_sample_right[n])>0:
        temp=(act_sample_left[n]-act_sample_right[n])/(act_sample_left[n]+act_sample_right[n])
    else:
        temp=0
    sample_disc.append(temp)
rew_disc1=[]
for n in range(len(act_rew_left)):
    if (act_rew_left[n]+act_rew_right[n])>0:
        temp=(act_rew_left[n]-act_rew_right[n])/(act_rew_left[n]+act_rew_right[n])
    else:
        temp=0
    rew_disc.append(temp)

total_prop_same=len(same)/len(av_hist_left)
total_prop_side=len(side)/len(av_hist_left)
total_prop_cat=len(cat)/len(av_hist_left)

props_by_cell={
    'side':N.asarray(total_prop_side),
    'same':N.asarray(total_prop_same),
    'cat':N.asarray(total_prop_cat)}


# Compute clustering into three groups for all cells.
n_clusters=3
cluster=N.empty((2,len(xcorr_shift)))
cluster[0]=r_left_right
cluster[1]=xcorr_shift

X_cluster=cluster.T
kmeans=KMeans(n_clusters=n_clusters,max_iter=100000,tol=0.0000001,init='random').fit(X_cluster)
res=kmeans.labels_

samples=silhouette_samples(X_cluster,res)
silhouette_values=[]
for i in range(n_clusters):
    ith_silhouette_values=samples[res==i]
    ith_silhouette_values.sort()
    silhouette_values.append(N.mean(ith_silhouette_values))

score=silhouette_score(X_cluster,res)

# Get all the cell IDs and resort by animal.
classify_as_cat=N.ravel(N.where(res==2))
classify_as_side=N.ravel(N.where(res==0))
classify_as_same=N.ravel(N.where(res==1))

mouse_index_array=N.asarray(mouse_index_array)
index_cat=mouse_index_array[classify_as_cat]
index_same=mouse_index_array[classify_as_same]
index_side=mouse_index_array[classify_as_side]

# Resort.
def resort_by_mouse(index_array,value_array):
    values, indices, counts = N.unique(index_array, return_counts=True, return_index=True)
    subarrays=N.split(N.asarray(value_array),indices) # This groups the values in value_array by the index given in index_array.
    
    # Get the mean values.
    means=[]
    for n in range(1,len(subarrays)):
        means.append(N.nanmean(subarrays[n]))
        
    return N.asarray(means)
    
cons_classified_cat=resort_by_mouse(index_cat,cons_total)
cons_classified_same=resort_by_mouse(index_same,cons_total)
cons_classified_side=resort_by_mouse(index_side,cons_total)

r_left_right_classified_cat=N.asarray(r_left_right)[classify_as_cat]
r_left_right_classified_same=N.asarray(r_left_right)[classify_as_same]
r_left_right_classified_side=N.asarray(r_left_right)[classify_as_side]

xcorr_shift_classified_cat=N.asarray(xcorr_shift)[classify_as_cat]
xcorr_shift_classified_same=N.asarray(xcorr_shift)[classify_as_same]
xcorr_shift_classified_side=N.asarray(xcorr_shift)[classify_as_side]

cons_classified_cat=N.asarray(cons_total)[classify_as_cat]
cons_classified_same=N.asarray(cons_total)[classify_as_same]
cons_classified_side=N.asarray(cons_total)[classify_as_side]


auto_cat={''}


classification={'mouse index array':mouse_index_array}


res_total={'classification':classification,
    'av hist left':av_hist_left,'av hist right':av_hist_right,'hist left':hist_left,'hist right':hist_right,
           'av hist left first':N.asarray(hist_left_odd),'av hist right first':N.asarray(hist_right_odd),
           'av hist left second':N.asarray(hist_left_even),'av hist right second':N.asarray(hist_right_even),
           'av hist left odd':N.asarray(hist_left_first),'av hist right odd':N.asarray(hist_right_first),
           'av hist left even':N.asarray(hist_left_second),'av hist right even':N.asarray(hist_right_second),          
           'av hist left fail':av_hist_left_fail,'av hist right fail':av_hist_right_fail,
           'r left right':N.asarray(r_left_right),'median r left right':N.asarray(median_r_left_right),
           'xcorr shift':N.asarray(xcorr_shift),'si left':N.asarray(si_left),'si right':N.asarray(si_right),'si mean':N.asarray(si_mean),'si left fail':N.asarray(si_left_fail),'si right fail':N.asarray(si_right_fail),
           'sig si':N.asarray(sig_si),
           'sig both':N.asarray(sig_both),'sig none':N.asarray(sig_none),'consistency total':N.asarray(cons_total),'consistency fs total':N.asarray(cons_fs_total),
           'mean activity left':N.asarray(mean_act_left),'mean activity right':N.asarray(mean_act_right),'mean activity':N.asarray(mean_act),
           'mean activity out':N.asarray(mean_act_out),'mean activity in':N.asarray(mean_act_in),
           'same':same,'side':side,'cat':cat,'speed hist':speed_hist,'speed hist norm':speed_hist_norm,'speed hist bins':speed_bin_centers,
           'peak location left':N.asarray(peak_location_left),'peak location right':N.asarray(peak_location_right),
           'peak width left':peak_width_left,'peak width right':peak_width_right,
           'sampling act left':N.asarray(act_sample_left),'sampling act right':N.asarray(act_sample_right),'sampling act mean':N.asarray(act_sample_mean),'sampling index':N.asarray(sample_disc),
           'reward act left':N.asarray(act_rew_left),'reward act right':N.asarray(act_rew_right),'reward act mean':N.asarray(act_rew_mean),'reward index':N.asarray(rew_disc),
           'si mean same mouse':N.asarray(si_mean_same_mouse),'si mean side mouse':N.asarray(si_mean_side_mouse),'si mean cat mouse':N.asarray(si_mean_cat_mouse),
           'si left mouse':N.asarray(si_left_mouse),'si right mouse':N.asarray(si_right_mouse),'si mean mouse':N.asarray(si_mean_mouse),
           'significant peak left':N.asarray(sig_peak_left),'significant peak right':N.asarray(sig_peak_right),
           'mean activity mouse':N.asarray(mean_act_mouse),'no cells':no_cells,'no cells total':no_cells_total,'consistencies total mouse':cons_total_mouse,'consistencies fs total mouse':cons_fs_total_mouse,
           'act score out':N.asarray(act_score_out),'act score in':N.asarray(act_score_in),'act score total':N.asarray(act_score_total),
           'speed total':speed_total,'speed total fail':speed_total_fail,'mean act comparison correct fail':mean_act_comp,
           'speed distribution':N.asarray(speed_hist_behav),'speed distribution fail':speed_hist_behav_fail,'speed mod p':N.asarray(speed_mod_p),'speed mod r':N.asarray(speed_mod_r),'speed modulated':speed_modulated,
           'mouse id':N.asarray(mouse_id),'X and classes by mouse':X_classes_by_mouse,'props by cell':props_by_cell,'props sig si by mouse':props_sig_si_by_mouse,
           'prop correct':N.asarray(prop_corr)}
os.chdir(target_dir)
N.save("all_mice_results_spatial_tuning.npy",res_total)