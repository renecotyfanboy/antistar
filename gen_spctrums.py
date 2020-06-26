# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:35:11 2020

@author: simd9
"""

import matplotlib.pyplot as plt
from tqdm import tqdm
from Source_4FGL import Source_4FGL
from matplotlib import rc
from data_filter import get_sources

# activate latex text rendering
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('text', usetex=True)

final_sources, excluded_sources  = get_sources()
    
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