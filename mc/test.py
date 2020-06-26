#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 21:19:57 2020

@author: simon
"""

import numpy as np
import dask.array as da
import dask
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from scipy.stats import rv_continuous
from astropy.io import fits
from astropy.wcs import WCS
from tools import imf,cimf

with warnings.catch_warnings():
    
    warnings.simplefilter("ignore")
    wcs = WCS(header=fits.getheader('K_map.fits'),naxis=2)
    k_map = fits.getdata('K_map.fits')
    k_map[k_map<1e-10] = np.nan
    n_sources = 14
    n_simulations = 1000
    
#%%


class mass_gen(rv_continuous):
    "mass distribution"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _pdf(self, x):
        return imf(x)    

    def _cdf(self, x):
        return cimf(x)
    
mass = mass_gen(a=0.01,b=150)
m_sample = mass.rvs(size=10000)

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
def simulate_starpop(s,fraction = 1.5e-4):

    max_R = 500
    max_Z = 250
    
    n_stars = int(max_R**2*np.pi*max_Z*0.14)
    
    is_astar = da.random.random(n_stars)<fraction
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

lower = (simulate_starpop(da.zeros((n_simulations,))) < 14).sum()
threshold = lower.compute()/n_simulations

#%%
# fig = plt.figure(figsize=(5,4))
# axes_coords = [0, 0, 1, 1]
# ax_image = fig.add_axes(axes_coords, label="ax image")
# img = plt.imread('fermi_sky_map.png',format='float')
# ax_image.imshow(img)
# ax_image.axis('off')

# ax_aitoff = fig.add_axes(axes_coords, projection='aitoff')
# ax_aitoff.grid(True)
# ax_aitoff.patch.set_alpha(0.)
# ax_aitoff.set_xticklabels([])
# ax_aitoff.set_yticklabels([])
# ax_aitoff.scatter(phi,tht,color='white',marker='*',s=80)