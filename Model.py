import pickle
import json
import os
from scipy.interpolate import interp1d
import numpy as np
from . import utils
from . import Prior
class Kilonova():
    def __init__(self,model_name,model_dir = os.path.join(os.path.dirname(__file__),'models/')):
        self.model_name = model_name
        self.model_dir = model_dir
        self.load_model()
    
    def cal_lightcurve(self,param_list,times,band,dL,z=0):
        '''
        param_list: list of parameters
        times: array of time(days)
        band: string of band
        dL: luminosity distance(cm)
        z: redshift
        '''
        if self.model_type == 0:
            return self._cal_lightcurve(param_list,times,band,dL)
        elif self.model_type == 1:
            return self._cal_lightcurve_afterglowpy(param_list,times,band,dL,z)
#############################################
    
    def load_model(self):
        with open(self.model_dir+'models.json','r') as file:
            self.model_meta = json.load(file)[self.model_name]
        self.model_type = self.model_meta['model_type']
        self.param_names = self.model_meta['param_names']
        self.param_names_latex = self.model_meta['param_names_latex']
        self.bounds = self.model_meta['bounds']
        self.priors = Prior.priors_from_bound(self.bounds)
        if self.model_type == 0:
            modelfile = f'{self.model_name}.pkl'
            with open(self.model_dir + modelfile,'rb') as handle:
                self.model = pickle.load(handle)
        elif self.model_type == 1:
            try:
                import afterglowpy
            except:
                raise ImportError('afterglowpy is not installed')
            
    def _cal_lightcurve(self,param_list,times,band,dL):
        if band[-3:] == 'GHz' or band[-3:] == 'keV':
            lightcurve = np.zeros(times.shape)
            lightcurve.fill(99)
            return lightcurve
        param_mins = self.model[band]["param_mins"]
        param_maxs = self.model[band]["param_maxs"]
        mins = self.model[band]["data_mins"]
        maxs = self.model[band]["data_maxs"]
        VA = self.model[band]["VA"]
        n_coeff = self.model[band]["n_coeff"]
        tt_interp = self.model[band]["tt"]
        param_list_postprocess = np.array(param_list)
        for i in range(len(param_mins)):
            param_list_postprocess[i] = (param_list_postprocess[i] - param_mins[i]) / (param_maxs[i] - param_mins[i])
        
        model = self.model[band]["model"]            
        cAproj = model(np.atleast_2d(param_list_postprocess)).numpy().T.flatten()
        
        mag_back = np.dot(VA[:, :n_coeff], cAproj)
        mag_back = mag_back * (maxs - mins) + mins
        f = interp1d(tt_interp,mag_back,fill_value = "extrapolate")
        mag = f(times)
        lightcurve = mag - 5 + 5*np.log10(dL/utils.pc)

        return lightcurve
    
    def _cal_lightcurve_afterglowpy(self,param_list,times,band,dL,z):
        '''
        calculate light curve using afterglowpy
        param_list: list of parameters
        times: array of time(days)
        band: string of band
        dL: luminosity distance(cm)
        z: redshift

        output: array of lightcurve
        '''
        import afterglowpy as grb
        Z = {'jetType': grb.jet.Gaussian,
             'specType': 0,

             'thetaObs':0,
             'E0':10**param_list[0],
             'thetaCore':10**param_list[1],
             'n0':10**param_list[2],
             'p':param_list[3],
             'epsilon_e':10**param_list[4],
             'epsilon_B':10**param_list[5],
             'xi_N':10**param_list[6],
             'd_L':dL,
             'z':z
        }
        Z['thetaWing'] = 4 * Z['thetaCore']
        nu = np.empty(times.shape)
        nu.fill(utils.get_effective_lambda(band,wave_eff=False))
        Fnu = grb.fluxDensity(times*86400,nu,**Z)*1e-3 #Jy
        return utils.fluxdensity2mag(Fnu)
