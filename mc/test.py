#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 21:19:57 2020

@author: simon
"""
import warnings #, os
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
from random import choice
from dask.distributed import Client
from astropy.io import fits
from astropy.wcs import WCS
from pba import probabilistic_bisection
#%%

@da.as_gufunc('(),()->()', output_dtypes=(float), vectorize=True)
def calc_k(m,rho):
    """
    mass in solar mass
    rho in pc
    velocity is free (here set at 100km/s)
    """
    
    L_gamma = (1/3)*1e36*(m)**2*(10)**(-3)#*0.5 #s-1
    k = L_gamma/(4*np.pi*(rho*3.086e18)**2)

    return k

@da.as_gufunc('(),()->()', output_dtypes=(float), vectorize=True)
def thrh_k(glon,glat):
    
    with warnings.catch_warnings():
    
        warnings.simplefilter("ignore")
        pos = wcs.world_to_array_index_values(glon,glat)
    
    return k_map[pos]
        
#%%
@da.as_gufunc('()->()', output_dtypes=(float), vectorize=True)
def simulate_starpop(f):
    
    star_pop = da.from_zarr('bigsim.zarr')
    n_stars = star_pop.shape[1]
    #Chaque * a une probabilité f d'être une a*
    #f*1000 car la population est déjà sous échantillonée
    is_astar = (da.random.random(n_stars)<f).compute()

    #On prend seulement les étoiles retenues plus tôt   
    glon = star_pop[0,is_astar]
    glat = star_pop[1,is_astar]
    m = star_pop[2,is_astar]
    r = star_pop[3,is_astar]
    
    k_ref = thrh_k(glon,glat)
    k_cmp = calc_k(m,r)
    
    index = np.isnan(k_ref)
    
    return (k_cmp[~index]>k_ref[~index]).sum()

def oracle(f):
    
    lower = (simulate_starpop(da.ones((n_simulations,))*f) < n_sources).sum()
    
    return (lower.compute()/n_simulations)<0.95

if __name__ == '__main__':

    # client = Client()
    
    with warnings.catch_warnings():
    
        warnings.simplefilter("ignore")
        wcs = WCS(header=fits.getheader('K_map.fits'),naxis=2)
        k_map = fits.getdata('K_map.fits')
        k_map[k_map<1e-10] = np.nan
        n_sources = 14
        n_simulations = 1000

    a,b,c = probabilistic_bisection(oracle,search_interval=(0,1))
    np.save('interv.npy',a)
    np.save('res.npy',c[-1])