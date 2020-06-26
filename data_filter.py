# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 14:06:03 2020

@author: simd9
"""

import numpy as np
from astropy.io import fits

def filter(data,TS_max=4):
    
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
    exclusion['TS < 9 above 1 GeV'] = np.sum(np.nan_to_num(data.field('Sqrt_TS_Band')[:,3:7])**2,axis=1) < TS_max
    exclusion['flag_kick'] = np.vectorize(flags_criterium)(data.field('Flags'))
    
    final_indexes = np.ones(data.shape,dtype=bool)
    
    for key,value in exclusion.items():
        
        final_indexes *= value
        excluded_sources[key] = len(data) - sum(value)
    
    return data[final_indexes], excluded_sources 

def get_sources(TS_max = 4, catalog='gll_psc_v23.fit'):

    with fits.open(catalog) as fermi_catalog:
    
        data4FGLDR2 = fermi_catalog[1].data    
        final_sources_4FGLDR2, excluded_sources_4FGLDR2  = filter(data4FGLDR2,TS_max=TS_max)
        
    return final_sources_4FGLDR2, excluded_sources_4FGLDR2