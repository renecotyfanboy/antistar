# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:35:11 2020

@author: simd9
"""

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import astropy.units as u

from astropy.io import fits
from source_analyser import Source_analyser

with fits.open('gll_psc_v21.fit') as fermi_catalog:

    i = 0    

    data = fermi_catalog[1].data
    source = data[data.field('Source_Name')=='4FGL J1325.5-4300'][0]
    #'4FGL J1325.5-4300'
    #'4FGL J0336.0+7502'
    #'4FGL J2028.6+4110e'

    analyser = Source_analyser(source)
    analyser.plot_all()
    
    
    