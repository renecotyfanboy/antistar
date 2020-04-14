import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from sympy import symbols,log,sqrt,lambdify,diff
from astropy.io import fits
from scipy.stats import t as Student_t

class Source_4FGL:
        
    bands_center = np.sqrt([50*100,100*300,300*1000,1e3*3e3,3e3*10e3,10e3*30e3,30e3*300e3])
    band_error = np.vstack((np.abs(bands_center-np.array([50,100,300,1000,3000,10000,30000])),np.abs(bands_center-np.array([100,300,1000,3000,10000,30000,300000]))))
    
    spectrum = {}
    model = {}
    
    def __init__(self,source):
        
        self.source = source    
        self.name = source.field('Source_Name')
        
        #Initialize source instance
        self._init_analytical_specs()
        self._init_spectrum_specs()
        self._init_model_specs()
        
    def _init_analytical_specs(self):
        
        self.expressions = {}
        K, E, E0, gamma, alpha, beta = symbols('K E E_0 Gamma alpha beta')
        self.symbolic = {'K':K,
                         'E':E,
                         'E0':E0,
                         'gamma':gamma,
                         'alpha':alpha,
                         'beta':beta}
        
        self.expressions['PowerLaw'] = K*(E/E0)**(-gamma)
        self.expressions['LogParabola'] = K*(E/E0)**(-alpha - beta*log(E/E0))

    def _init_spectrum_specs(self):
        
        self.spectrum['Flux_Band'] = self.source.field('Flux_Band')*(u.cm**(-2)*u.s**(-1))
        self.spectrum['Unc_Flux_Band'] = self.source.field('Unc_Flux_Band')*(u.cm**(-2)*u.s**(-1))
        self.spectrum['nuFnu_Band'] = self.source.field('nuFnu_Band')*(u.erg*u.cm**(-2)*u.s**(-1))  
        self.spectrum['Unc_nuFnu_Band'] = np.vstack(np.abs(
            (self.spectrum['Unc_Flux_Band'][:,0]/self.spectrum['Flux_Band']*self.spectrum['nuFnu_Band'],
             self.spectrum['Unc_Flux_Band'][:,1]/self.spectrum['Flux_Band']*self.spectrum['nuFnu_Band']))
            )*(u.erg*u.cm**(-2)*u.s**(-1))  
        
        #Upper Limits Handling
        is_ul = np.isnan(self.spectrum['Unc_Flux_Band'][:,0])
        self.spectrum['is_ul'] = is_ul 
        self.spectrum['is_not_ul'] = np.invert(is_ul)
        
        self.spectrum['ul'] = 2*self.spectrum['Unc_nuFnu_Band'][1,is_ul] + self.spectrum['nuFnu_Band'][is_ul]
        self.spectrum['Unc_nuFnu_Band'][0,is_ul] = self.spectrum['ul']
        self.spectrum['Unc_nuFnu_Band'][1,is_ul] = np.zeros_like(self.spectrum['ul'])
        
    def _init_model_specs(self):
        
        self.model['E0'] = self.source.field('Pivot_Energy')*u.MeV
        self.model['SpectrumType'] = self.source.field('SpectrumType')
        
        if self.model['SpectrumType'] == 'PowerLaw':
            
            self.model['K'] = self.source.field('PL_Flux_Density')
            self.model['gamma'] = self.source.field('PL_Index')
            
            self.model['Unc_K'] = self.source.field('Unc_PL_Flux_Density')
            self.model['Unc_gamma'] = self.source.field('Unc_PL_Index')
            
            expr = self.expressions['PowerLaw']
            expr = expr.subs([(self.symbolic['E0'],self.model['E0']/u.MeV),
                              (self.symbolic['K'],self.model['K']),
                              (self.symbolic['gamma'],self.model['gamma'])])
            
            self._fitted_spectrum = lambdify(self.symbolic['E'], expr, 'numpy')
            
        elif self.model['SpectrumType'] == 'LogParabola':
            
            self.model['K'] = self.source.field('LP_Flux_Density')
            self.model['alpha'] = self.source.field('LP_Index')
            self.model['beta'] = self.source.field('LP_beta')
            
            self.model['Unc_K'] = self.source.field('Unc_LP_Flux_Density')
            self.model['Unc_alpha'] = self.source.field('Unc_LP_Index')
            self.model['Unc_beta'] = self.source.field('Unc_LP_beta')
        
            expr = self.expressions['LogParabola']
            expr = expr.subs([(self.symbolic['E0'],self.model['E0']/u.MeV),
                              (self.symbolic['K'],self.model['K']),
                              (self.symbolic['alpha'],self.model['alpha']),
                              (self.symbolic['beta'],self.model['beta'])])
            
            self._fitted_spectrum = lambdify(self.symbolic['E'], expr, 'numpy')
            
        else :
            
            print("Implémenter PLEC")
    
    def fitted_spectrum(self,E):
    
        return (self._fitted_spectrum(E/u.MeV)*(u.cm**(-2)*u.s**(-1)*u.MeV**(-1)))
    
    def fitted_nuFnu(self,E):
        
        return (self._fitted_spectrum(E/u.MeV)*(u.cm**(-2)*u.s**(-1)*u.MeV**(-1))*E**2
                ).to(u.erg*u.cm**(-2)*u.s**(-1))    
    
    def fitted_error_nuFnu(self,E):
        
        expr = self.expressions[self.model['SpectrumType']]*self.symbolic['E']**2
        confprob = 0.6827
        
        if self.model['SpectrumType'] == 'PowerLaw':
            
            dof = 2
            tval = Student_t.ppf(1.0 - (1.0 - confprob)/2, dof)        
            
            K_diff = diff(expr,self.symbolic['K'])
            gamma_diff = diff(expr,self.symbolic['gamma'])
            
            final_expr = sqrt((self.model['Unc_K']*K_diff)**2 
                             +(self.model['Unc_gamma']*gamma_diff)**2)
            
            final_expr = final_expr.subs([(self.symbolic['E0'],self.model['E0']/u.MeV),
                                          (self.symbolic['K'],self.model['K']),
                                          (self.symbolic['gamma'],self.model['gamma'])])
            
        elif self.model['SpectrumType'] == 'LogParabola':
        
            dof = 3
            tval = Student_t.ppf(1.0 - (1.0 - confprob)/2, dof)        
            
            K_diff = diff(expr,self.symbolic['K'])
            alpha_diff = diff(expr,self.symbolic['alpha'])
            beta_diff = diff(expr,self.symbolic['beta'])
            
            final_expr = sqrt((self.model['Unc_K']*K_diff)**2 
                             +(self.model['Unc_alpha']*alpha_diff)**2 
                             +(self.model['Unc_beta']*beta_diff)**2)
            
            final_expr = final_expr.subs([(self.symbolic['E0'],self.model['E0']/u.MeV),
                                          (self.symbolic['K'],self.model['K']),
                                          (self.symbolic['alpha'],self.model['alpha']),
                                          (self.symbolic['beta'],self.model['beta'])])
    
        f = lambdify(self.symbolic['E'], final_expr,'numpy')   

        return (tval*f(E/u.MeV)*(u.cm**(-2)*u.s**(-1)*u.MeV)).to(u.erg*u.cm**(-2)*u.s**(-1))  
    
    def plot_fitted(self,ax):
        
        E_span = np.linspace(30,300e3,100000)*u.MeV
        
        ax.loglog(E_span,
                  self.fitted_nuFnu(E_span)/(u.erg*u.cm**(-2)*u.s**(-1)),
                  color='black',linestyle='--')  
        
        ax.fill_between(E_span,
                        (self.fitted_nuFnu(E_span)+self.fitted_error_nuFnu(E_span))/(u.erg*u.cm**(-2)*u.s**(-1)),
                        (self.fitted_nuFnu(E_span)-self.fitted_error_nuFnu(E_span))/(u.erg*u.cm**(-2)*u.s**(-1)),
                        color='lightgrey')
        
    def plot_band(self,ax):
        
        ax.errorbar(self.bands_center[self.spectrum['is_not_ul']],
                    self.spectrum['nuFnu_Band'][self.spectrum['is_not_ul']]/(u.erg*u.cm**(-2)*u.s**(-1)) ,
                    xerr = self.band_error[:,self.spectrum['is_not_ul']],
                    yerr=self.spectrum['Unc_nuFnu_Band'][:,self.spectrum['is_not_ul']]/(u.erg*u.cm**(-2)*u.s**(-1)),
                    color='black',
                    capsize=2,
                    capthick=1,
                    fmt='none')
        
        ax.scatter(self.bands_center[self.spectrum['is_ul']],
                   self.spectrum['ul']/(u.erg*u.cm**(-2)*u.s**(-1)),
                   color='black',marker='v',s=40)
        
        ax.errorbar(self.bands_center[self.spectrum['is_ul']],
                    self.spectrum['ul']/(u.erg*u.cm**(-2)*u.s**(-1)),
                    xerr = self.band_error[:,self.spectrum['is_ul']],
                    yerr=self.spectrum['Unc_nuFnu_Band'][:,self.spectrum['is_ul']]/(u.erg*u.cm**(-2)*u.s**(-1)),
                    color='black',
                    capsize=2,
                    capthick=1,
                    fmt='none')
        
    def plot_all(self,ax):
        
        self.plot_fitted(ax)
        self.plot_band(ax)
        
#%% Test code

if __name__ == '__main__':

    with fits.open('gll_psc_v21.fit') as fermi_catalog:
        
        from sympy import init_printing
        init_printing(use_latex=True,forecolor="White")
        
        data = fermi_catalog[1].data
        source = Source_4FGL(data[data.field('Source_Name')=='4FGL J0336.0+7502'][0])
        
        #'4FGL J1325.5-4300'
        #'4FGL J0336.0+7502'
        #'4FGL J2028.6+4110e'
        
        fig, ax = plt.subplots()
        plt.xlim(50,200000)
        plt.ylim(0.07e-12,6e-12)
        plt.loglog()
        source.plot_fitted(ax)
        source.plot_band(ax)
    