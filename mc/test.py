#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 21:19:57 2020

@author: simon
"""
import warnings, os
import numpy as np
import dask.array as da
from random import choice
from dask.distributed import Client,LocalCluster
from tqdm import tqdm
from scipy.stats import rv_continuous
from astropy.io import fits
from astropy.wcs import WCS
from tools import imf,cimf
    
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
def thrh_k(glon,glat):
    
    with warnings.catch_warnings():
    
        warnings.simplefilter("ignore")
        pos = wcs.world_to_array_index_values(glon,glat)
    
    return k_map[pos]
        
#%%
@da.as_gufunc('()->()', output_dtypes=(float), vectorize=True)
def simulate_starpop(f):
    
    star_pop = choice(pops)
    n_stars = star_pop.shape[1]
    #Chaque * a une probabilité f d'être une a*
    #f*1000 car la population est déjà sous échantillonée
    is_astar = da.random.random(n_stars)<f*1000

    #On prend seulement les étoiles retenues plus tôt   
    choices = da.tile(is_astar,(4,1))
    a_stars = star_pop[choices]
    
    return choices
    
    # glon = a_stars[0]
    # glat = a_stars[1]
    # m = a_stars[2]
    # rho = a_stars[3]
    
    # print(glon,glat,m,rho)
    
    # k_cmp = calc_k(m,rho)
    # k_ref = thrh_k(glon,glat)
    
    
    # index = np.isnan(k_ref)
    
    # return (k_cmp[~index]>k_ref[~index]).sum()

def oracle(f):
    
    lower = (simulate_starpop(da.ones((n_simulations,))*f) < n_sources).sum()
    
    return (lower.compute()/n_simulations)#-0.95

def load_starpops():
    
    pops = []
    for x in os.listdir('data'):
        pops.append(da.from_zarr(os.path.join('data',x)))
        
    return pops

if __name__ == '__main__':
    
    # cluster = LocalCluster()
    # client = Client(cluster)
    
    with warnings.catch_warnings():
    
        warnings.simplefilter("ignore")
        wcs = WCS(header=fits.getheader('K_map.fits'),naxis=2)
        k_map = fits.getdata('K_map.fits')
        k_map[k_map<1e-10] = np.nan
        n_sources = 14
        n_simulations = 10000

    # mass = mass_gen(a=0.01,b=150)
    # print('Sampling masses')
    # m_sample = mass.rvs(size=100000)

    # f_span = np.geomspace(1e-5,1e-3,20)
    # mean = np.zeros_like(f_span)
    # stdr = np.zeros_like(f_span)
    
    # for _,f in enumerate(tqdm(f_span)):
        
    #     res = []
    #     for i in range(10):
            
    #         res.append(oracle(f))
            
    #     mean[_] = np.mean(res)
    #     stdr[_] = np.sqrt(np.var(res))

    # import matplotlib.pyplot as plt
    # fig,ax = plt.subplots()
    # ax.plot(f,mean)
    # ax.fill_between(f,mean-stdr,mean+stdr)
    # np.save('mean.npy',mean)
    # np.save('stdr.npy',stdr)
    
    pops = load_starpops()
    x = simulate_starpop(1e-6*np.ones((1,)))