#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 21:19:57 2020

@author: simon
"""

import numpy as np
import dask.array as da
from dask.distributed import Client,LocalCluster
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import rv_continuous
from astropy.io import fits
from astropy.wcs import WCS
from tools import imf,cimf
from noisyopt import bisect,AveragedFunction
    
#%%

class mass_gen(rv_continuous):
    "mass distribution"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _pdf(self, x):
        return imf(x)    

    def _cdf(self, x):
        return cimf(x)
    
@da.as_gufunc('(),()->()', output_dtypes=(float), vectorize=True)
def calc_k(m,rho):
    """
    mass in solar mass
    rho in pc
    velocity is free (here set at 100km/s)
    """
    
    L_gamma = 0.5*(1/3)*1e36*(m)**2*(10)**(-3) # s-1
    k = L_gamma/(4*np.pi*(rho*3.086e18)**2)

    return k

@da.as_gufunc('(),()->()', output_dtypes=(float), vectorize=True)
def thrh_k(phi,tht):
    
    pos = wcs.world_to_array_index_values(phi*180/np.pi,tht*180/np.pi)
    
    return k_map[pos]
        
#%%
@da.as_gufunc('()->()', output_dtypes=(float), vectorize=True)
def simulate_starpop(f):

    max_R = 500
    max_Z = 250
    
    n_stars = int(max_R**2*np.pi*max_Z*0.14)
    
    is_astar = da.random.random(n_stars)<f
    n_astars = is_astar.sum().compute()
    
    r = da.random.random(n_astars)*max_R
    z = da.random.random(n_astars)*max_Z-max_Z/2
    phi = da.random.random(n_astars)*2*np.pi - np.pi
    
    rho = np.sqrt(r**2 + z**2)
    tht = np.pi/2 - np.arccos(z/rho)
    
    m = da.random.choice(m_sample,size=n_astars)
    
    k_cmp = calc_k(m,rho)
    k_ref = thrh_k(phi,tht)
    
    index = np.isnan(k_ref)
    
    return (k_cmp[~index]>k_ref[~index]).sum()

def oracle(f):
    
    lower = (simulate_starpop(da.ones((n_simulations,))*f) < n_sources).sum()
    
    return (lower.compute()/n_simulations)-0.95

if __name__ == '__main__':

    with warnings.catch_warnings():
    
        warnings.simplefilter("ignore")
        wcs = WCS(header=fits.getheader('K_map.fits'),naxis=2)
        k_map = fits.getdata('K_map.fits')
        k_map[k_map<1e-10] = np.nan
        n_sources = 14
        n_simulations = 10000

    mass = mass_gen(a=0.01,b=150)
    print('Sampling masses')
    m_sample = mass.rvs(size=100000)
    print('Done!')

    #client = Client()
    # client = Client(cluster)
    # print('Root-finding')
    # f = AveragedFunction(oracle)
    # r = bisect(f,1e-6,1e-4)
    f = np.tile(np.geomspace(1e-6,1e-3,20),(n_simulations,1))
    n_star = simulate_starpop(f)<14
    ratio = da.sum(n_star,axis=0)/n_simulations
    ratio = ratio.compute()

    plt.plot(np.geomspace(1e-6,1e-3,20),ratio)
