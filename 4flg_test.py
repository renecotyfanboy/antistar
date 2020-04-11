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
10-30 GeV,
30-300 GeV
"""

bands_center = np.sqrt([50*100,100*300,300*1000,1e3*3e3,3e3*10e3,10e3*30e3,30e3*300e3])
band_error = np.vstack((np.abs(bands_center-np.array([50,100,300,1000,3000,10000,30000])),np.abs(bands_center-np.array([100,300,1000,3000,10000,30000,300000]))))

def filter(data):
    
    def flags_criterium(flag):
        
        excluded_flags = [1,2,3,4,5,6]
        flag_str = np.binary_repr(flag,width=12)[::-1]
        
        for i in excluded_flags:
            
            i -= 1
            
            if bool(int(flag_str[i])) :
                
                return False
        
        return True
    
    exclusion = {}
    excluded_sources = {}
    
    exclusion['extended_source_kick'] = data.field('Extended_Source_Name').isspace()
    exclusion['class_kick'] = data.field('CLASS1').isspace() * data.field('CLASS2').isspace()
    exclusion['max(TS) < 1 GeV'] = np.argmax(data.field('Sqrt_TS_Band'),axis=1) < 3 
    exclusion['TS < 9 above 1 GeV'] = np.max(np.nan_to_num(data.field('Sqrt_TS_Band')[:,3:7]),axis=1) < 3
    exclusion['flag_kick'] = np.vectorize(flags_criterium)(data.field('Flags'))
    
    final_indexes = np.ones(data.shape,dtype=bool)
    
    for key,value in exclusion.items():
        
        final_indexes *= value
        excluded_sources[key] = len(data) - sum(value)
    
    return data[final_indexes], excluded_sources 

with fits.open('gll_psc_v21.fit') as fermi_catalog:

    data = fermi_catalog[1].data
    
    final_sources, excluded_sources  = filter(data)
    
    # fig,axs = plt.subplots(nrows=2,ncols=1,sharex=True)
    # axs[0].set_title('nuFnu_Band')
    # axs[1].set_title('Sqrt_TS_Band')
    
    #for i in range(np.shape(final_sources.field('nuFnu_Band'))[0]):

    i = 0
        
    # axs[0].loglog(bands_center,final_sources.field('nuFnu_Band')[i,:],label=final_sources.field('Source_Name')[i])  
    # axs[1].semilogx(bands_center,final_sources.field('Sqrt_TS_Band')[i,:])
    # axs[0].legend()
    
    plt.errorbar(bands_center,final_sources.field('nuFnu_Band')[i,:],xerr = band_error,fmt='none')
    plt.loglog()