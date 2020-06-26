# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 14:19:19 2020

@author: simd9
"""

import numpy as np

@np.vectorize
def imf(m):
    
    m0 = 0.01
    m1 = 0.08
    m2 = 0.5
    a0 = 0.3
    a1 = 1.3
    a2 = 2.3
    k0 = 7.910493859660253
    k1 = k0*(m1/m0)**(-a0)
    k2 = k1*(m2/m1)**(-a1)
        
    if m0 <= m and m <= m1 :
        return k0*(m/m0)**(-a0)
    
    elif m1 < m and m <= m2:
        return k1*(m/m1)**(-a1)
    
    elif m2 < m:
        return k2*(m/m2)**(-a2)
    
    else: 
        return float(0)
    
@np.vectorize
def cimf(m):
    
    m0 = 0.01
    m1 = 0.08
    m2 = 0.5
    a0 = 0.3
    a1 = 1.3
    a2 = 2.3
    k0 = 7.910493859660253
    k1 = k0*(m1/m0)**(-a0)
    k2 = k1*(m2/m1)**(-a1)
    c0 = k0*m0**(a0)/(1-a0)*(m1**(1-a0)-m0**(1-a0))
    c1 = k1*m1**(a1)/(1-a1)*(m2**(1-a1)-m1**(1-a1))
    c2 = k2*m2**(a2)/(1-a2)*(150**(1-a2)-m2**(1-a2))
        
    if m0 <= m <= m1 :
        return k0*m0**(a0)/(1-a0)*(m**(1-a0)-m0**(1-a0))
    
    elif m1 < m <= m2:
        return c0 + k1*m1**(a1)/(1-a1)*(m**(1-a1)-m1**(1-a1))
    
    elif m2 < m <= 150:
        return c0 + c1 + k2*m2**(a2)/(1-a2)*(m**(1-a2)-m2**(1-a2))

    elif m>150:
        return 1.
    
    elif m<0:
        return 0.