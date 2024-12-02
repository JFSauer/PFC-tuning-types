#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:41:24 2024

@author: jonas
"""

from scipy import spatial
import numpy as N
import matplotlib.pyplot as pl
import pandas as pd
import pingouin as pg
import scipy.stats as st

N.random.seed(42)

def linearize_2d_track_single_run(track,start,end,skel,is_left=True):
    '''

    Parameters
    ----------
    track : 2d array
        x and y coordinates.
    left1, right1, left2, right2 : list
        start and end points used. The first list is used to extract start points, the second one to obtain end points.

    Returns
    -------
    None.

    '''
    if is_left==True:
        c=skel['skeleton left'] 
        total_length=skel['length left']
        x_real=track[0][start:end]
        y_real=track[1][start:end]
        lin_pos=[]
        for n in range(len(x_real)):
            first_ind=[x_real[n],y_real[n]]
            distance,index = spatial.KDTree(c).query(first_ind)
            lin_pos.append(index/total_length)
        
    else:
        c=skel['skeleton right'] 
        total_length=skel['length right']
        x_real=track[0][start:end]
        y_real=track[1][start:end]
        lin_pos=[]
        for n in range(len(x_real)):
            first_ind=[x_real[n],y_real[n]]
            distance,index = spatial.KDTree(c).query(first_ind)
            lin_pos.append(index/total_length)
    
        
    
    return lin_pos

def rescale_pos(values, new_min = 0, new_max= 150):
    
    output = []
    old_min, old_max = min(values), max(values)

    for v in values:
        new_v = (new_max - new_min) / (old_max - old_min) * (v - old_min) + new_min
        output.append(new_v)

    return output

def separate_in(indices,not_in):
    ''' Separate 2 periods of spike-like (point) data.
    '''    
   
    not_in_range=[]
    for n in range(len(not_in[0::2])):
        not_in_range.extend(range(not_in[0::2][n],not_in[1::2][n],1))
    not_in_range=set(not_in_range)
  
    ind=[]
    for n in range(len(indices)):
        if indices[n] in not_in_range:
            ind.append(indices[n])
   
    return ind

def separate_in_dual_conditional(indices,indices2,not_in):
    ''' Separate 2 periods of spike-like (point) data.
    '''    
   
    not_in_range=[]
    for n in range(len(not_in[0::2])):
        not_in_range.extend(range(not_in[0::2][n],not_in[1::2][n],1))
    not_in_range=set(not_in_range)
  
    ind=[]
    ind2=[]
    for n in range(len(indices)):
        if indices[n] in not_in_range:
            ind.append(indices[n])
            ind2.append(indices2[n])
   
    return ind,ind2


### Potting functions.
def boxplot3_with_points_and_lines(data1,data2,data3,figsize=[3,5],whis=[5,95],s=30,showfliers=False,widths=0.5,incl_noise=True,random_range=0.1):
    whiskerprops=dict(linestyle='-',linewidth=1,color="k")
    boxprops=dict(linestyle='-',linewidth=1,color="k")
    medianprops=dict(linestyle='-',linewidth=1,color="k")
    fig,ax=pl.subplots(figsize=figsize)
    #_=pl.subplot(121)
    data=[data1,data2,data3]
    ax.boxplot(data,whis=whis,showfliers=showfliers,boxprops=boxprops,whiskerprops=whiskerprops,medianprops=medianprops,widths=widths)
    x1=[]
    x2=[]
    x3=[]
    
    for n in range(len(data1)):
        if incl_noise==True:
            x1.append(N.random.choice(N.linspace(1-random_range,1+random_range,1000)))
        else:            
            x1.append(1)
    for n in range(len(data2)):
        if incl_noise==True:
            x2.append(N.random.choice(N.linspace(2-random_range,2+random_range,1000)))
        else:
            x2.append(2)
    for n in range(len(data3)):
        if incl_noise==True:
            x3.append(N.random.choice(N.linspace(3-random_range,3+random_range,1000)))
        else:
            x3.append(3)

    for n in range(len(data1)):
        _=ax.plot([x1[n],x2[n]],[data1[n],data2[n]],"grey",alpha=0.5)  
        _=ax.plot([x2[n],x3[n]],[data2[n],data3[n]],"grey",alpha=0.5) 
    ax.scatter(x1,data1,c="k",s=s)
    ax.scatter(x2,data2,c="k",s=s)
    ax.scatter(x3,data3,c="k",s=s)
    
def boxplot2_with_points_and_lines(data1,data2,figsize=[3,5],whis=[5,95],s=30,showfliers=False,widths=0.5,incl_noise=True,random_range=0.1):
    whiskerprops=dict(linestyle='-',linewidth=1,color="k")
    boxprops=dict(linestyle='-',linewidth=1,color="k")
    medianprops=dict(linestyle='-',linewidth=1,color="k")
    fig,ax=pl.subplots(figsize=figsize)
    #_=pl.subplot(121)
    data=[data1,data2]
    ax.boxplot(data,whis=whis,showfliers=showfliers,boxprops=boxprops,whiskerprops=whiskerprops,medianprops=medianprops,widths=widths)
    x1=[]
    x2=[]
    x3=[]
    
    for n in range(len(data1)):
        if incl_noise==True:
            x1.append(N.random.choice(N.linspace(1-random_range,1+random_range,1000)))
        else:            
            x1.append(1)
    for n in range(len(data2)):
        if incl_noise==True:
            x2.append(N.random.choice(N.linspace(2-random_range,2+random_range,1000)))
        else:
            x2.append(2)
    

    for n in range(len(data1)):
        _=ax.plot([x1[n],x2[n]],[data1[n],data2[n]],"grey",alpha=0.5)  
     
    ax.scatter(x1,data1,c="k",s=s)
    ax.scatter(x2,data2,c="k",s=s)

    
def boxplot3(data1,data2,data3,figsize=[3,5],whis=[5,95],s=30,showfliers=False,widths=0.5,incl_noise=True,random_range=0.1):
    whiskerprops=dict(linestyle='-',linewidth=1,color="k")
    boxprops=dict(linestyle='-',linewidth=1,color="k")
    medianprops=dict(linestyle='-',linewidth=1,color="k")
    fig,ax=pl.subplots(figsize=figsize)
    #_=pl.subplot(121)
    data=[data1,data2,data3]
    ax.boxplot(data,whis=whis,showfliers=showfliers,boxprops=boxprops,whiskerprops=whiskerprops,medianprops=medianprops,widths=widths)
 
def boxplot2(data1,data2,figsize=[3,5],whis=[5,95],s=30,showfliers=False,widths=0.5,incl_noise=True,random_range=0.1):
    whiskerprops=dict(linestyle='-',linewidth=1,color="k")
    boxprops=dict(linestyle='-',linewidth=1,color="k")
    medianprops=dict(linestyle='-',linewidth=1,color="k")
    fig,ax=pl.subplots(figsize=figsize)
    data=[data1,data2]
    ax.boxplot(data,whis=whis,showfliers=showfliers,boxprops=boxprops,whiskerprops=whiskerprops,medianprops=medianprops,widths=widths)
      
def boxplot2_with_points(data1,data2,figsize=[3,5],whis=[5,95],s=30,showfliers=False,widths=0.5,incl_noise=True,random_range=0.1):
    whiskerprops=dict(linestyle='-',linewidth=1,color="k")
    boxprops=dict(linestyle='-',linewidth=1,color="k")
    medianprops=dict(linestyle='-',linewidth=1,color="k")
    fig,ax=pl.subplots(figsize=figsize)
    #_=pl.subplot(121)
    data=[data1,data2]
    ax.boxplot(data,whis=whis,showfliers=showfliers,boxprops=boxprops,whiskerprops=whiskerprops,medianprops=medianprops,widths=widths)
    x1=[]
    x2=[]
    x3=[]

 
    
    for n in range(len(data1)):
        if incl_noise==True:
            x1.append(N.random.choice(N.linspace(1-random_range,1+random_range,1000)))
        else:            
            x1.append(1)
    for n in range(len(data2)):
        if incl_noise==True:
            x2.append(N.random.choice(N.linspace(2-random_range,2+random_range,1000)))
        else:
            x2.append(2)

    ax.scatter(x1,data1,c="k",s=s)
    ax.scatter(x2,data2,c="k",s=s)
    
def boxplot6_with_points(data1,data2,data3,data4,data5,data6,figsize=[5,5],whis=[5,95],showfliers=False,s=30,widths=0.5,incl_noise=True,random_range=0.1):
    whiskerprops=dict(linestyle='-',linewidth=1,color="k")
    boxprops=dict(linestyle='-',linewidth=1,color="k")
    medianprops=dict(linestyle='-',linewidth=1,color="k")
    fig,ax=pl.subplots(figsize=figsize)
    #_=pl.subplot(121)
    data=[data1,data2,data3,data4,data5,data6]
    ax.boxplot(data,whis=whis,showfliers=showfliers,boxprops=boxprops,whiskerprops=whiskerprops,medianprops=medianprops,widths=widths)
    x1=[]
    x2=[]
    x3=[]
    x4=[]
    x5=[]
    x6=[]
   
    for n in range(len(data1)):
        if incl_noise==True:
            x1.append(N.random.choice(N.linspace(1-random_range,1+random_range,1000)))
        else:            
            x1.append(1)
    for n in range(len(data2)):
        if incl_noise==True:
            x2.append(N.random.choice(N.linspace(2-random_range,2+random_range,1000)))
        else:
            x2.append(2)
    for n in range(len(data3)):
        if incl_noise==True:
            x3.append(N.random.choice(N.linspace(3-random_range,3+random_range,1000)))
        else:
            x3.append(3)
    for n in range(len(data4)):
        if incl_noise==True:
            x4.append(N.random.choice(N.linspace(4-random_range,4+random_range,1000)))
        else:
            x4.append(4)
    for n in range(len(data5)):
        if incl_noise==True:
            x5.append(N.random.choice(N.linspace(5-random_range,5+random_range,1000)))
        else:
            x4.append(5)
    for n in range(len(data6)):
        if incl_noise==True:
            x6.append(N.random.choice(N.linspace(6-random_range,6+random_range,1000)))
        else:
            x4.append(6)


    ax.scatter(x1,data1,c="k",s=s)
    ax.scatter(x2,data2,c="k",s=s)
    ax.scatter(x3,data3,c="k",s=s)    
    ax.scatter(x4,data4,c="k",s=s)   
    ax.scatter(x5,data5,c="k",s=s) 
    ax.scatter(x6,data6,c="k",s=s) 
     
      
    
    
def violinplot3(data1,data2,data3,figsize=[3,5.9],points=100,log=False,showmeans=False,showmedians=True):
    data1=N.asarray(data1)
    data2=N.asarray(data2)
    data3=N.asarray(data3)
    
    
    data1=data1[~N.isnan(data1)]
    data2=data2[~N.isnan(data2)]
    data3=data3[~N.isnan(data3)]
    data=[data1,data2,data3]
    fig,ax=pl.subplots(figsize=figsize)
    ax.violinplot(data,showmeans=showmeans,showextrema=False,showmedians=showmedians,points=points)
    
    if log==True:
        ax.set_yscale('log')


def scatter_plot(x,y,figsize=[3,3],s=20,plot_mean_line=False,bins=10):
    pl.figure(figsize=figsize)
    pl.scatter(x,y,s=s)
    
    if plot_mean_line==True:
        mean_data,a,b=st.binned_statistic(x,y,bins=bins)
        bin_centers=[]
        for i in range(len(a)-1):
            bin_centers.append((a[i+1]+a[i])/2)
        _=pl.plot(bin_centers,mean_data,"r")

def plot_sorted_spatial_map_one_condition(res1,cmap='viridis',shading='gouraud',low=0,high=1,colorbar=False,rasterized=True,blank=False,sort_by=1):
    
    peak=[]
    for n in range(len(res1)):
        peak.append(N.argmax(res1[n]))
    
    peak=N.asarray(peak)
    indices=N.argsort(peak)

    
    res1=N.asarray(res1)

    
    pl.figure()
    pl.pcolormesh(res1[indices],cmap=cmap,shading=shading,rasterized=rasterized)
    pl.ylim(0,len(res1))
    pl.xlim(0,len(res1[0]))
    if colorbar==True:
        pl.colorbar()
    pl.clim(low,high)
    #pl.xlim(0,bins)
    pl.ylabel("Cells")
    if blank==True:
        pl.axis('off')
    
   
     
def plot_sorted_spatial_map_two_conditions_one_sorting(res1,res2,cmap='viridis',shading='gouraud',low=0,high=1,colorbar=False,rasterized=True,blank=False,sort_by=1):
    ind=[]
    for n in range(len(res1)):
        if not N.isnan(N.mean(res1[n])) and not N.isnan(N.mean(res2[n])):
            ind.append(n)

    ind=N.asarray(ind)
    res1_final=res1[ind]
    res2_final=res2[ind]
    
    peak=[]
    for n in range(len(res1_final)):
        if sort_by==1:
            peak.append(N.argmax(res1_final[n]))
        else:
            peak.append(N.argmax(res2_final[n]))
    peak=N.asarray(peak)
    indices=N.argsort(peak)
    
    pl.figure()
    pl.subplot(121)
    pl.pcolormesh(res1_final[indices],cmap=cmap,shading=shading,rasterized=rasterized)
    pl.ylim(0,len(res1_final))
    pl.xlim(0,len(res1_final[0]))
    pl.clim(low,high)
    #pl.xlim(0,bins)
    pl.ylabel("Cells")
    if blank==True:
        pl.axis('off')
    
    pl.subplot(122)
    pl.pcolormesh(res2_final[indices],cmap=cmap,shading=shading,rasterized=rasterized)
    pl.ylim(0,len(res2_final))
    pl.xlim(0,len(res2_final[0]))
    pl.clim(low,high)
    #pl.xlim(0,bins)
    pl.xlabel("Spatial bins")
    if blank==True:
        pl.axis('off')
    
  
    
    
### Statistics functions.

def one_way_anova_general(data1,data2,data3):
    '''
    Data arrays given as data points (1d)
    '''
    data=[]
    data.extend(data1)
    data.extend(data2)
    data.extend(data3)
    data=N.asarray(data)
    subject=N.asarray(range(1,len(data)+1,1)) # Increasing number of subject
    
    condition=N.zeros((len(N.ravel(data1))))      
    condition=N.append(condition,N.ones((len(N.ravel(data2))))) 
    condition=N.append(condition,N.full((len(N.ravel(data3))),2))

    


    
    df=pd.DataFrame({'subject':subject,
                    'condition':condition,
                    'data':data})    
    
    res = pg.anova(dv='data', between='condition',
                  data=df, detailed=True)
    
    t=df.pairwise_tukey(dv='data',between='condition')
    res_total={'anova':res,'pairwise':t}
    
    return res_total
def one_way_repeated_measures_anova_general_three_groups(array1,array2,array3):
    '''
    Data arrays given as data (mice, cells) x time points
    '''
    temp=[]
    temp.extend(array1)
    temp.extend(array2)
    temp.extend(array3)
    data1=N.reshape(temp,[3,-1]).T
   
    
    mouse=N.repeat(N.linspace(1,len(data1),len(data1)),len(data1[0]))
    
    time=N.tile(N.linspace(1,len(data1[0]),len(data1[0])),len(data1))
    data=N.ravel(data1)

    
    df=pd.DataFrame({'mouse':mouse,
                    'time':time,
                    'data':data})    
    
    res = pg.rm_anova(dv='data', within='time', subject='mouse', 
                  data=df, detailed=True)
    
    # Post hoc comparisons with paired t-tests and Sidak correction.
    data=data1.T
    t,p=[],[]
    sig=[]
    comp=[]
    pcrit=1-(1-0.05)**(1/len(data))
    p_corr=[]
    for n in range(len(data)-1):
        for i in range(n+1,len(data),1):
            local_comp=[n, i]
            comp.append(local_comp)
            tt,pp=st.ttest_rel(data[n],data[i])
            t.append(tt)
            p.append(pp)
            p_corr.append(pp*0.05/pcrit)
            if pp<pcrit:
                sig.append(1)
            else:
                sig.append(0)
    

    res_total={'p':p,'t':t,'sig':sig,'Pcrit':pcrit,'Pcorr':p_corr,'comparisons':comp}
    

    return res,res_total

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
    pcrit=1-(1-0.05)**(1/len(data1))
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
    

    res_total={'p':p,'t':t,'sig':sig,'Pcrit':pcrit,'p corrected':p_corr}


    t=df.pairwise_tukey(dv='data',between='condition').round(3)
    

    return res,res_total

def two_sample_random_permutation_test_tuning_functions(left,right,iterations=1000):
    '''

    Parameters
    ----------
    left, right : 2d arrays
        Tuning functions.

    Returns
    -------
    p: p-value

    '''
    
    # This function tests whether the tuning functions during left and right trials are similar.
    # The basic idea is to compute the average peak location, randomly permute the trial ID iteration times and again compute peak location each time.
    # If the real difference in peak location is larger than the 95th percentile, there is difference in tuning on both trajectories.
    
    length_left=len(left)
    length_right=len(right)
    
    
    
    def circ_dist(left,right,bins=40):
        peak1=N.argmax(N.nanmean(left,axis=0))
        peak2=N.argmax(N.nanmean(right,axis=0))
        i=(peak1-peak2)%bins
        j=(peak2-peak1)%bins
        return min(i,j)
    
    def correlation(left,right):
        r,p=st.pearsonr(N.nanmean(left,axis=0),N.nanmean(right,axis=0))
        return r
    
    real_distance=correlation(left,right)
    
    conc=N.concatenate((left,right),axis=0)
    
    permuted_distances=[]
    for n in range(iterations):
        new=N.random.permutation(conc)
        
        l=new[:length_left,:]
        r=new[length_left:,:]
        permuted_distances.append(correlation(l,r))
        
    if real_distance>N.percentile(permuted_distances,99):
        sig=1
    else:
        sig=0
        
    return sig,permuted_distances,real_distance

def pairwise_test(data1,data2):
    tt,norm_check1=st.shapiro(data1)
    tt,norm_check2=st.shapiro(data2)
    if norm_check1<0.05 and norm_check2<0.05:
        tt,pp=st.wilcoxon(data1,data2)
        test="Wilcoxon"
    else:
        tt,pp=st.ttest_rel(data1,data2)
        test="Paired t"
    res={'p':pp,'t':tt,'test':test}
    return res

def unpaired_test(data1,data2):
    tt,norm_check1=st.shapiro(data1)
    tt,norm_check2=st.shapiro(data2)
    if norm_check1<0.05 and norm_check2<0.05:
        if len(data1)<50 and len(data2)<50:
            tt,pp=st.mannwhitneyu(data1,data2)
            test="MWU"
        else:
            tt,pp=st.ttest_ind(data1,data2,equal_var=False)
            test="Welch's test"
    else:
        tt,pp=st.ttest_ind(data1,data2)
        test="t test"
    res={'p':pp,'t':tt,'test':test}
    return res 
    
    