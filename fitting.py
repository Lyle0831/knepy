from .Model import Kilonova
from .Prior import Prior
from .Transient import Transient
from . import utils
from typing import List
import numpy as np
import os
import scipy
import dynesty
from dynesty.pool import Pool as dynesty_pool
import pymultinest

def dynesty_fit(transient:Transient,models:List[Kilonova],priors:List[Prior] = None,dlogz = None,processes = 1):
    print('Start fitting!')
    model_num = len(models)
    params_num = [len(model.priors) for model in models]
    if priors == None:
        priors = []
        for model in models:
            priors = priors + model.priors
    n_dim = len(priors)
    data = transient.data
    dL = transient.dL
    bands = list(set(data['band']))
 
    def prior_transform(u_params):
        '''
        Transform the unit cube to the parameter space
        '''
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
        variance = 1
        for band in bands:
            phase = data[(data['band']==band)]['phase']

            mags = []
            total = 0
            for i in range(model_num):
                model = models[i]
                mags.append(model.cal_lightcurve(param_list=params[total:total+params_num[i]],times=phase,band=band,dL=dL))
                total += params_num[i]
            mags = utils.sumab(mags)

            log_i = -0.5*np.sum((data[data['band']==band]['mag']-mags)**2/(variance**2+data[data['band']==band]['mag_err']**2))
            if np.isin(99,data[data['band']==band]['mag_err']):
                upl_phase = data[(data['band']==band) & (data['mag_err']==99)]['phase']
                upl_mags = data[(data['band']==band) & (data['mag_err']==99)]['mag']
                mags = []
                total = 0
                for i in range(model_num):
                    model = models[i]
                    mags.append(model.cal_lightcurve(param_list=params[total:total+params_num[i]],times=upl_phase,band=band,dL=dL))
                    total += params_num[i] 
                mags = utils.sumab(mags)               
                sub = mags - upl_mags
                if np.min(sub) < 0:
                    log_i = -np.inf
            log_band.append(log_i)
        return np.sum(log_band)
    
    if processes == 1:
        dsampler = dynesty.NestedSampler(dlog_like, prior_transform, ndim=n_dim, nlive=1500)
        dsampler.run_nested(dlogz = dlogz)
    else:
        with dynesty_pool(processes, dlog_like, prior_transform) as pool:
            dsampler = dynesty.NestedSampler(pool.loglike, pool.prior_transform,
                                    ndim=n_dim, nlive = 1500, pool = pool)
            dsampler.run_nested(dlogz = dlogz)

    results = dsampler.results

    return results

def pymultinest_fit(transient:Transient,models:List[Kilonova],priors:List[Prior] = None):
    print('Start fitting!')
    model_num = len(models)
    params_num = [len(model.priors) for model in models]
    if priors == None:
        priors = []
        for model in models:
            priors = priors + model.priors
    n_dim = len(priors)
    data = transient.data
    dL = transient.dL
    bands = list(set(data['band']))

    def prior_transform(u_params,ndim,nparams):
        '''
        Transform the unit cube to the parameter space
        The last two parameters are ndim and nparams, which are required by pymultinest
        '''
        for i,prior in enumerate(priors):
            if prior.prior_type == 'uniform':
                u_params[i] = u_params[i]*(prior.max-prior.min) + prior.min
            elif prior.prior_type == 'gaussian':
                u_params[i] = scipy.stats.truncnorm.ppf(u_params[i],(prior.min-prior.mu)/prior.sigma,(prior.max-prior.mu)/prior.sigma,loc = prior.mu,scale = prior.sigma)
        return u_params

    def dlog_like(cube,ndim,nparams):
        log_band = []
        params = []
        for i in range(ndim):
            params.append(cube[i])
        variance = 1
        for band in bands:
            phase = data[(data['band']==band)]['phase']

            mags = []
            total = 0
            for i in range(model_num):
                model = models[i]
                mags.append(model.cal_lightcurve(param_list=params[total:total+params_num[i]],times=phase,band=band,dL=dL))
                total += params_num[i]
            mags = utils.sumab(mags)

            log_i = -0.5*np.sum((data[data['band']==band]['mag']-mags)**2/(variance**2+data[data['band']==band]['mag_err']**2))
            if np.isin(99,data[data['band']==band]['mag_err']):
                upl_phase = data[(data['band']==band) & (data['mag_err']==99)]['phase']
                upl_mags = data[(data['band']==band) & (data['mag_err']==99)]['mag']
                mags = []
                total = 0
                for i in range(model_num):
                    model = models[i]
                    mags.append(model.cal_lightcurve(param_list=params[total:total+params_num[i]],times=upl_phase,band=band,dL=dL))
                    total += params_num[i] 
                mags = utils.sumab(mags)               
                sub = mags - upl_mags
                if np.min(sub) < 0:
                    log_i = -np.inf
            log_band.append(log_i)
        return np.sum(log_band)
    
    base_dir = os.path.join(os.path.dirname(__file__),'out/')
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    pymultinest.run(dlog_like, prior_transform, n_dim, n_live_points= 1000, evidence_tolerance= 0.1,outputfiles_basename=base_dir,resume = False, verbose = True)

    a = pymultinest.Analyzer(outputfiles_basename=base_dir,n_params=n_dim)
    return a.get_equal_weighted_posterior()[:,:-1]