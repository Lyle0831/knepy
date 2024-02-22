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

def mag2fluxdensity(mag,band):
    #convert ABmag to flux density(Jy)
    bandpass = sncosmo.get_bandpass(band)
    mab = u.Magnitude(mag*u.ABmag)
    return mab.to(u.Jy,u.spectral_density(bandpass.wave_eff * u.AA)).value
