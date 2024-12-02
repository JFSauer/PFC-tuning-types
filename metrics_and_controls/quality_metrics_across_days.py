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

N.random.seed(42)

target_dir="/media/jonas/data3/HM_wm_data/playground/spatial_tuning_types_analysis/spatial_tuning_outward_inward/using_sign_transients/revision_analysis/quality_metrics"

Fs=20

    
def get_transients(filename,name,session,n_reg):
    
    f = h5py.File(filename, 'r')
            
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
    
    traces1=N.stack(transients[:,session[0]]).astype(None)
    traces2=N.stack(transients[:,session[1]]).astype(None)
    
    
    res_total={'traces 1':traces1,'traces 2':traces2}

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



for animal in range(len(animal_list)):
    filename=filename_list[animal]
    name=animal_list[animal]
    session=[session_list[animal]]
    print(name)
    res=get_transients(filename,name,session,n_reg)

   