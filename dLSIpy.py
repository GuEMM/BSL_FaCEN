# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 21:20:34 2020

@author: edued
"""

import numpy as np
from scipy import ndimage
from PIL import Image
from scipy.interpolate import interp1d

#function for lasca contrast calculation
def lasca2(imarray, wsize = 7):
    
    imarray = np.float64(imarray)
    
    immean = ndimage.uniform_filter(imarray, size=wsize)	
    im2mean = ndimage.uniform_filter(np.square(imarray), size=wsize)
    imcontrast = np.sqrt(im2mean / np.square(immean) - 1)
    
    return imcontrast

def dlsi_proc_decompose_2(data, wsize,shift,beta):
    ''' 
     rho - static part
     dyn - dynamic part (mixed)
     contrast - standard LASCA
     Input parameters:
        data(x, y, time) - data array with intensity (assumed double)
        wsize - window size
        shift - number of frames to jump for static and dynamic
                part calculation. 1 should work fine in most cases
        beta - This is the square of the maximal contrast (measured on teflon
               block or dead rat.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Adapted from matlab script
    
    Copyright 2006-2010 University of Fribourg
    Contact: Pavel Zakharov - pv.zakharov@gmail.com
    '''
    N, s1,s2 = data.shape
    
    data = np.float64(data)
    
    out_contrast = np.zeros([s1,s2])
    
    for i in range(N):
                            
        out_contrast += np.float64(lasca2(data[i])/N)
        
        #if i%10 ==0:
                #print('Lasca: %d\n',i)
    
    # intensity for normalization (symmetric)
    norm_m =  ndimage.uniform_filter(np.mean(data[0:-shift],axis=0), size=wsize)*ndimage.uniform_filter(np.mean(data[shift:],axis=0), size=wsize)
    
    # static part
    out_rho2 = (ndimage.uniform_filter(np.mean(data[:-shift]*data[shift:] ,axis=0),size=wsize)/norm_m -1)/beta
    
    # cutting min and max
    out_rho2[out_rho2 < 0] = 0.001
    out_rho2[out_rho2 > beta] = beta
    
    out_rho = np.sqrt(out_rho2)
    
    # dinamic part
    out_dyn = np.sqrt(ndimage.uniform_filter(np.mean((data[0:-shift] - data[shift:])**2/2,axis=0),size=wsize)/norm_m/beta)
    
    return out_rho, out_dyn, out_contrast

def get_dynamic_tc_dws(rho,Kb,T):
    '''
    Estimates the decorrelation time from the contrast using the DWS model
    
    Adapted from Pavel Zakharov matlab implementation
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %   Copyright 2006-2010 University of Fribourg
    %   Contact: Pavel Zakharov - pv.zakharov@gmail.com 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''
    x = np.arange(1e-4,10,1e-3)
    
    x = 1./x
    
    x2= x/4.
    
    #Create the dynamic contrast as a function fo correlation time  for the defined rho
    contrast_th_dws_12 = np.sqrt(((1 - rho)**2  * ((3 + 6 * np.sqrt(x) + 4 * x) * np.exp(-2*np.sqrt(x))- 3 + 2 * x) / 2) / x**2 +
                                 (2 * rho * (1 - rho) * ((3 + 6 * np.sqrt(x2) + 4 * x2) * np.exp(-2*np.sqrt(x2)) - 3 + 2*x2) / 2) / x2**2)
    
    gam = 1.5
    
    interp = interp1d(contrast_th_dws_12,x, kind='linear', fill_value='extrapolate')
     
    tc = 6*T*gam**2/interp(Kb)
       
    del interp
    
    return tc

def dlsi_proc_get_tauc(rho,dyn,contrast,exptime):
    """
    Function to convert contrast components into the correlation times 
    Extremely time consuming. 
    % Requires the function 'get_dynamic_tc_dws'.
    %
    % Input:
    % rho, dyn and contrast as obtained with 'evaluate' and 'process' function.
    %    beta - This is the square of the maximal contrast (measured on teflon
    %           block or dead rat. 
    % exptime - exposure time of the camera
    %
    % Output:
    % out_tauc - 100% correct estimation of correlation time    
    % out_tauc_wrong - traditional estimation of correlation ignoring the
    %                       static part.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    """
    import time
    
    out_tauc = np.zeros(dyn.shape)
    out_tauc_wrong = np.zeros(dyn.shape)
    
    start = time.time()
    print('Inverting')
    
    for i in range(dyn.shape[0]):
        
        for j in range(dyn.shape[1]):
            
                out_tauc[i,j] = get_dynamic_tc_dws(rho[i,j],dyn[i,j],exptime)
                out_tauc_wrong[i,j] = get_dynamic_tc_dws(0,contrast[i,j],exptime)
        
        if i%5 == 0:
            t = time.time() - start
            start = time.time()
            
            #print('Invert: %d, spent %f sec' % (i,t))
            
    return out_tauc,out_tauc_wrong

def im2col_(arr, block):
    arr = np.float64(arr)
    
    m,n = arr.shape
    
    m,n = arr.shape
    s0, s1 = arr.strides    
    
    stepsize1= block[0]
    stepsize2= block[1]
    
    nrows = m - block[0] + 1
    ncols = n - block[1] + 1
    
    shp = block[0],block[1],nrows,ncols
    
    strd = s0,s1,s0,s1
    
    out_view = np.lib.stride_tricks.as_strided(arr, shape=shp, strides=strd)

    return out_view[:,:,:,::stepsize2][:,:,::stepsize1,:].reshape(block[0]*block[1],-1)

def blockmean(data,wsize):
    '''Calculate the mean value of windows of wsize x wsize in data array and return a reduced array with the mean values'''
        
    X = im2col_(data,[wsize,wsize])

    M = np.mean(X,axis=0)

    SS = data.shape[0]//wsize,data.shape[1]//wsize

    return M.reshape(SS[0],SS[1])

def blockstd(data,wsize):
    '''Calculate the mean value of windows of wsize x wsize in data array and return a reduced array with the mean values'''
        
    X = im2col_(data,[wsize,wsize])

    M = np.std(X,axis=0)

    SS = data.shape[0]//wsize,data.shape[1]//wsize

    return M.reshape(SS[0],SS[1])

#function for lasca contrast calculation
def lasca(imarray, wsize = 7):
    
    imarray = np.float64(imarray)
    
    return blockstd(imarray,wsize)/blockmean(imarray,wsize)

def dlsi_proc_decompose(data, wsize,shift,beta):
    ''' 
     rho - static part
     dyn - dynamic part (mixed)
     contrast - standard LASCA
     Input parameters:
        data(x, y, time) - data array with intensity (assumed double)
        wsize - window size
        shift - number of frames to jump for static and dynamic
                part calculation. 1 should work fine in most cases
        beta - This is the square of the maximal contrast (measured on teflon
               block or dead rat.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Adapted from matlab script
    
    Copyright 2006-2010 University of Fribourg
    Contact: Pavel Zakharov - pv.zakharov@gmail.com
    '''
    N, s1,s2 = data.shape
    
    data = np.float64(data)
    
    out_contrast = np.zeros([s1//wsize,s2//wsize])
    
    for i in range(N):
                            
        out_contrast += lasca(data[i],wsize)/N
        
       # if i%10 ==0:
                #print('Lasca: %d\n' % i)
    
    # intensity for normalization (symmetric)
    norm_m =  blockmean(np.mean(data[0:-shift],axis=0), wsize)*blockmean(np.mean(data[shift:],axis=0), wsize)
    
    # static part
    out_rho2 = (blockmean(np.mean(data[:-shift]*data[shift:] ,axis=0), wsize)/norm_m -1)/beta
    
    # cutting min and max
    out_rho2[out_rho2 < 0] = 0.001
    out_rho2[out_rho2 > beta] = beta
    
    out_rho = np.sqrt(out_rho2)
    
    # dinamic part
    out_dyn = np.sqrt(blockmean(np.mean((data[0:-shift] - data[shift:])**2/2,axis=0),wsize)/norm_m/beta)
    
    return out_rho, out_dyn, out_contrast
