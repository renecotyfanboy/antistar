# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:35:11 2020

@author: simd9
"""

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from astropy.io import fits

"""
50-100 MeV, 
100-300 MeV, 
300-1 GeV, 
1-3 GeV, 
3-10 GeV, 
10-30 GeV,3
0-300 GeV
"""

with fits.open('gll_psc_v21.fit') as fermi_catalog:

    data = fermi_catalog[1].data
    
    exclusion = {}
    exclusion['extended_source_kick'] = data.field('Extended_Source_Name').isspace()
    exclusion['class_kick'] = data.field('CLASS1').isspace() * data.field('CLASS2').isspace()
    exclusion['signifiance_kick'] = np.argmax(data.field('Sqrt_TS_Band'),axis=1) < 3
    exclusion['flag_kick'] = data.field('Flags') == 0
    
    final_indexes = np.ones(data.shape,dtype=bool)
    
    for key,value in exclusion.items():
        
        final_indexes *= value
    
    final_sources = data[final_indexes]
    
    for i in range(np.shape(final_sources.field('nuFnu_Band'))[0]):
        
        plt.plot(final_sources.field('nuFnu_Band')[i,:])
    