# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:35:11 2020

@author: simd9
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from matplotlib import rc

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

glon_span = []
glat_span = []

for _ in range(len(final_sources)):
    
    source = final_sources[_]
    
    if source.field('Source_Name') == "4FGL J0546.5-1100":
    
        glon_span.append(source.field('GLON'))
        glat_span.append(source.field('GLAT'))

glon_span = np.array(glon_span)
glon_span[glon_span>180] -= 360
glon_span *= -np.pi/180
glat_span = np.array(glat_span)*np.pi/180

axes_coords = [0, 0, 1, 1]

fig = plt.figure(figsize=(4,2.5))

ax_image = fig.add_axes(axes_coords, label="ax image")
img = plt.imread('fermi_sky_map.png',format='float')
ax_image.imshow(img)
ax_image.axis('off')

ax_aitoff = fig.add_axes(axes_coords, projection='aitoff')
ax_aitoff.grid(True)
ax_aitoff.patch.set_alpha(0.)
ax_aitoff.set_xticklabels([])
ax_aitoff.set_yticklabels([])
ax_aitoff.scatter(glon_span,glat_span,color='white',marker='*',s=80)

plt.title(r'7 sources with significance $< 3 \sigma$ over 1 GeV')
#plt.close()
#plt.savefig('custom_skymap.png', dpi=600,transparent=True)