from .Model import Kilonova
from .Prior import Prior
from .Transient import Transient
from typing import List
import numpy as np
import os
import scipy
import dynesty
from dynesty.pool import Pool as dynesty_pool
#import pymultinest

def dynesty_fit(transient:Transient,model:Kilonova,priors:List[Prior] = None,dlogz = None,processes = 1):
    print('Start fitting!')
    if priors == None:
        priors = model.priors
    n_dim = len(priors)
    data = transient.data
    dL = transient.dL
    bands = list(set(data['band']))
 
    def prior_transform(u_params):
        u_params = np.array(u_params)
        for i,prior in enumerate(priors):
            if prior.prior_type == 'uniform':
                u_params[i] = u_params[i]*(prior.max-prior.min) + prior.min
            elif prior.prior_type == 'gaussian':
                u_params[i] = scipy.stats.truncnorm.ppf(u_params[i],(prior.min-prior.mu)/prior.sigma,(prior.max-prior.mu)/prior.sigma,loc = prior.mu,scale = prior.sigma)
        return u_params
    def dlog_like(params):
        log_band = []
        params = np.array(params)
        variance = 0
        for band in bands:
            phase = data[(data['band']==band)]['phase']
            mags = model.cal_lightcurve(param_list=params,times=phase,band=band,dL=dL)
            log_i = -0.5*np.sum((data[data['band']==band]['mag']-mags)**2/(variance**2+data[data['band']==band]['mag_err']**2))
            if np.isin(99,data[data['band']==band]['mag_err']):
                upl_phase = data[(data['band']==band) & (data['mag_err']==99)]['phase']
                upl_mags = data[(data['band']==band) & (data['mag_err']==99)]['mag']
                mags = model.cal_lightcurve(param_list=params,times=upl_phase,band=band,dL=dL)
                sub = mags - upl_mags
                if np.min(sub) < 0:
                    log_i = -np.inf
            log_band.append(log_i)
        return np.sum(log_band)

    if processes > 1:
        with dynesty_pool(processes, dlog_like, prior_transform) as pool:
            dsampler = dynesty.NestedSampler(pool.loglike, pool.prior_transform,
                                    ndim=n_dim, nlive = 1500, pool = pool)
            dsampler.run_nested(dlogz = dlogz)
    
    else:
        dsampler = dynesty.NestedSampler(dlog_like, prior_transform, ndim=n_dim, nlive=1500)
        dsampler.run_nested(dlogz = dlogz)

    results = dsampler.results

    return results

# def pymultinest_fit(transient:Transient,model:Kilonova,priors:List[Prior] = None):
#     if priors == None:
#         priors = model.priors
#     ndim = len(priors)
#     data = transient.data
#     dL = transient.dL
#     bands = list(set(data['band']))
#     variance = 1

#     def prior(u_params,ndim,nparams):
#         #u_params = np.array(u_params)
#         for i,prior in enumerate(priors):
#             if prior.prior_type == 'uniform':
#                 u_params[i] = u_params[i]*(prior.max-prior.min) + prior.min
#             elif prior.prior_type == 'gaussian':
#                 u_params[i] = scipy.stats.truncnorm.ppf(u_params[i],(prior.min-prior.mu)/prior.sigma,(prior.max-prior.mu)/prior.sigma,loc = prior.mu,scale = prior.sigma)
#         return u_params

#     def loglike(cube,ndim,nparams):
#         log_band = []
#         params = []
#         for i in range(ndim):
#             params.append(cube[i])
#         params = np.array(params)
#         variance = 1
#         for band in bands:
#             phase = data[(data['band']==band)]['phase']
#             mags = model.cal_lightcurve(param_list=params,times=phase,band=band,dL=dL)
#             log_i = -0.5*np.sum((data[data['band']==band]['mag']-mags)**2/(variance**2+data[data['band']==band]['mag_err']**2))
#             if np.isin(99,data[data['band']==band]['mag_err']):
#                 upl_phase = data[(data['band']==band) & (data['mag_err']==99)]['phase']
#                 upl_mags = data[(data['band']==band) & (data['mag_err']==99)]['mag']
#                 mags = model.cal_lightcurve(param_list=params,times=upl_phase,band=band,dL=dL)
#                 sub = mags - upl_mags
#                 if np.min(sub) < 0:
#                     log_i = -np.inf
#             log_band.append(log_i)
#         return np.sum(log_band)
    
#     pymultinest.run(loglike, prior, ndim, outputfiles_basename=os.path.join(os.path.dirname(__file__),'out/'),resume = False, verbose = True)

#     a = pymultinest.Analyzer(outputfiles_basename=os.path.join(os.path.dirname(__file__),'out/'),n_params=ndim)
#     return a.get_equal_weighted_posterior()[:,:-1]