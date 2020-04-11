# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:35:11 2020

@author: simd9
"""

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import astropy.units as u

from scipy.stats import t
from astropy.io import fits
from source_analyser import Source_analyser

"""
50-100 MeV, 
100-300 MeV, 
300-1 GeV, 
1-3 GeV, 
3-10 GeV, 
10-30 GeV,
30-300 GeV
"""

bands_center = np.sqrt([50*100,100*300,300*1000,1e3*3e3,3e3*10e3,10e3*30e3,30e3*300e3])
band_error = np.vstack((np.abs(bands_center-np.array([50,100,300,1000,3000,10000,30000])),np.abs(bands_center-np.array([100,300,1000,3000,10000,30000,300000]))))

def spectrum_func(source):
    
    if source.field('SpectrumType')[0] == 'PowerLaw':
        
        E0 = source.field('Pivot_Energy')*u.MeV
        K = source.field('PL_Flux_Density')*(1/u.cm**2/u.s/u.MeV)
        gamma = source.field('PL_Index')
        
        def spectrum(E):
            
            return K*(E/E0)**(-gamma)*E**2
        
        return spectrum
    
    else :
        
        print("Implémenter les autres types de sources !!")

with fits.open('gll_psc_v21.fit') as fermi_catalog:

    i = 0    

    data = fermi_catalog[1].data
    source = data[data.field('Source_Name')=='4FGL J1325.5-4300']
    #'4FGL J1325.5-4300'
    #'4FGL J0336.0+7502'
    #'4FGL J2028.6+4110e'

    analyser = Source_analyser(source)
    analyser.plot_all()
    
    
    