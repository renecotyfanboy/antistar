# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:35:11 2020

@author: simd9
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from tqdm import tqdm
from matplotlib.widgets import RadioButtons
from astropy.io import fits
from Source_4FGL import Source_4FGL
from matplotlib import rc
from mw_plot import MWSkyMap

# activate latex text rendering
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('text', usetex=True)

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
    #exclusion['TS < 9 above 1 GeV'] = np.max(np.nan_to_num(data.field('Sqrt_TS_Band')[:,3:7]),axis=1) < 3
    exclusion['TS < 9 above 1 GeV'] = np.sum(np.nan_to_num(data.field('Sqrt_TS_Band')[:,3:7])**2,axis=1) < 9
    exclusion['flag_kick'] = np.vectorize(flags_criterium)(data.field('Flags'))
    
    final_indexes = np.ones(data.shape,dtype=bool)
    
    for key,value in exclusion.items():
        
        final_indexes *= value
        excluded_sources[key] = len(data) - sum(value)
    
    return data[final_indexes], excluded_sources 

with fits.open('gll_psc_v21.fit') as fermi_catalog:

    data = fermi_catalog[1].data
    final_sources, excluded_sources  = filter(data)
    
    for _ in tqdm(range(len(final_sources))):
        
        source = Source_4FGL(final_sources[_])
        fig,ax = plt.subplots(figsize=(3,3))
        plt.loglog()
        plt.grid(True,which="both",ls="-")
        source.plot_all(ax)
        plt.title(source.name)
        plt.tight_layout()
        plt.savefig('final_sources_4FGL/'+source.name+".png", dpi=600,transparent=True)
        plt.close()