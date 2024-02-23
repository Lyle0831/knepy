import astropy.units as u
import astropy.constants as c
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import sncosmo
pc = u.pc.cgs.scale
MPC_CGS = u.Mpc.cgs.scale
h_CGS = c.h.cgs.value
C_CGS = c.c.cgs.value
eV_CGS = (c.e.value * u.J).cgs.value
def redshift2dL(z,H0=67.8,Om0=0.308):
    '''
    convert redshift to luminosity distance(cm)
    z: redshift
    H0: Hubble constant(km/s/Mpc)
    Om0: matter density
    '''
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    return cosmo.luminosity_distance(z)

def get_effective_lambda(filt,wave_eff=False):
    '''
    get effective wavelength of a filter
    filt: string of filter name
    wave_eff: bool, return effective wavelength(A) or effective frequency(Hz)
    '''
    try:
        _x = sncosmo.get_bandpass(filt)
        if wave_eff:
            return _x.wave_eff
        else:
            return C_CGS/_x.wave_eff*1e8
    
    except:
        if filt[-3:] == 'GHz':
            if wave_eff:
                return C_CGS/(float(filt[:-3])*1e9)*1e8
            else:
                return float(filt[:-3])*1e9

        elif filt[-3:] == 'keV':
            if wave_eff:
                return C_CGS/(float(filt[:-3])*1e3*eV_CGS/h_CGS)*1e8
            else:
                return float(filt[:-3])*1e3*eV_CGS/h_CGS

def mag2fluxdensity(mag,mode = 'Jy'):
    #convert magnitude to flux density
    if mode == 'Jy':
        return 10**(-mag/2.5)*3631
    elif mode == 'cgs':
        #both are correct
        #return 10**(-mag/2.5)*3631*1e-23
        return 10**(-(mag+48.6)/2.5)
def fluxdensity2mag(flux,mode = 'Jy'):
    #convert flux density to magnitude
    if mode == 'Jy':
        return -2.5*np.log10(flux/3631)
    elif mode == 'cgs':
        #both are correct
        #return -2.5*np.log10(flux/3631*1e23)
        return -2.5*np.log10(flux)-48.6
def sumab(mab_list):
    _flux_all = np.zeros(len(mab_list[0]))
    for mab in mab_list:
        _flux_all += mag2fluxdensity(mab)
    return fluxdensity2mag(_flux_all)
