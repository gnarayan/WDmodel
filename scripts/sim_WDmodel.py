#!/usr/bin/env python
import sys
import os
import argparse
import numpy as np
import scipy.optimize as op
import scipy.integrate as scinteg
import george
import WDmodel
from astropy import units as u
from astropy.convolution import convolve, Gaussian1DKernel
from specutils.extinction import extinction, reddening
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties as FM
from matplotlib import rc
from matplotlib.mlab import rec2txt
from fit_WDmodel import read_spec

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--teff', required=True, type=float, default=40000.)
    parser.add_argument('-g', '--logg', required=True, type=float, default=7.5)
    parser.add_argument('-a', '--av', required=True, type=float, default=0.)
    parser.add_argument('-r', '--rv', required=False, type=float, default=3.1)
    parser.add_argument('--reddeningmodel', required=False, default='od94')
    parser.add_argument('-s', '--smooth', required=False, type=float, default=4.)
    parser.add_argument('--sn', required=True, type=float, default=50., help='Peak S/N')
    parser.add_argument('-n', '--errscatterratio', required=False, type=float, default=2.,\
    help='Ratio of stderr to spectrum actual spectrum variance (THERE ARE SOURCES OF ERROR THAT ARE NOT INCLUDED IN THE VARIANCE)')

    args = parser.parse_args()

    if args.teff < 17000 or args.teff > 90000:
        message = 'Teff value is out of range (17000,90000)'
        raise ValueError(message)

    if args.logg < 7.0 or args.logg > 9.5:
        message = 'logg value is out of range (7.0,9.5)'
        raise ValueError(message)


    if args.av < 0 or args.av > 5.:
        message = 'That Av value is ridiculous'
        raise ValueError(message)

    if args.rv < 2.1 or args.rv > 5.1:
        message = 'That Rv Value is ridiculous'
        raise ValueError(message)

    if args.smooth < 0:
        message = 'That Gaussian Smoothing FWHM value is ridiculous'
        raise ValueError(message)

    if args.errscatterratio < 1:
        message = 'If you have less noise (errscatterratio < 1) than the variance in your data, you computed the noise wrong'
        raise ValueError(message)

    reddeninglaws = ('od94', 'ccm89', 'gcc09', 'f99', 'fm07', 'wd01', 'd03')
    if not args.reddeningmodel in reddeninglaws:
        message = 'That reddening law is not known (%s)'%' '.join(reddeninglaws) 
        raise ValueError(message)

    return args



def sim_model(teff, logg, av=0.0, rv=3.1, sn=15., reddeninglaw='od94', norm=1., smooth=4., errscatterratio=2.):

    scaling = np.recfromtxt('error_scale.txt', names=True)
    wave = scaling.wave
    res_scale = scaling.res_scale/np.median(scaling.res_scale) # change the scaling to be dimensionless
    errorbar_scale = scaling.error_scale/np.median(scaling.error_scale) # change the scaling to be dimensionless


    # init a simple Gaussian 1D kernel to smooth the model to the resolution of the instrument
    gsig     = smooth*(0.5/(np.log(2.)**0.5))
    kernel   = Gaussian1DKernel(gsig)

    # init the model, and determine the coarse normalization to match the spectrum
    model = WDmodel.WDmodel()
    xi = model._get_xi(teff, logg, wave)
    mod = model._get_model(xi)

    # redden the model
    bluening = reddening(wave*u.Angstrom, av, r_v=3.1, model='od94')
    redmod=mod/bluening

    # smooth the model, and extract the section that overlays the model
    # since we smooth the full model, computed on the full wavelength range of the spectrum
    # and then extract the subset range that overlaps with the data
    # we avoid any edge effects with smoothing at the end of the range

    # should maybe move this to after we add the noise
    smoothedmod = convolve(redmod, kernel)

    # norm isn't a fit parameter and shouldn't matter at all, as it's simply a flux scaling
    # but we should test that it doesn't matter, and if it does, then we have issues
    smoothedmod*=norm

    err            = smoothedmod.max()/sn
    scatter        = err/errscatterratio
    signal         = smoothedmod + res_scale*scatter*np.random.randn(len(wave))

    sn_reference = 36.7
    this_noise_reference  = smoothedmod.max()/sn_reference #if this was at the reference SN this is the noise we'd get
    noise_floor_reference = this_noise_reference/1.38

        
    errorbars = np.repeat(err, len(wave))
    errorbars*=errorbar_scale
    errorbars[errorbars < noise_floor_reference] = noise_floor_reference

    out = np.rec.fromarrays([wave, signal, errorbars, smoothedmod, redmod, mod],names="wave,flux,flux_err,model_flux,model_flux_unsmoothed,model_flux_orig")
    return out
    

    




def main():
    args   = get_options()
    teff = args.teff
    logg = args.logg
    av   = args.av
    rv   = args.rv
    sn   = args.sn
    rl   = args.reddeningmodel
    smooth = args.smooth
    esr = args.errscatterratio
    out = sim_model(teff, logg, av=av, rv=rv, sn=sn, reddeninglaw=rl, norm=1., smooth=4., errscatterratio=esr)
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1,1,1)
    ax.errorbar(out.wave, out.flux, out.flux_err, marker='None', capsize=0, linestyle='-', color='grey', alpha=0.7)
    ax.plot(out.wave, out.flux, marker='None', linestyle='-', color='k',label='Output')
    ax.plot(out.wave, out.model_flux, marker='None', linestyle='-', color='red',alpha=0.5,label='Model')
    ax.plot(out.wave, out.model_flux_unsmoothed, marker='None', linestyle='-', color='orange',alpha=0.5,label='Model Unsmoothed')
    ax.plot(out.wave, out.model_flux_orig, marker='None', linestyle='-', color='blue',alpha=0.5, label='Model Unsmoothed, Unreddened')
    ax.legend(loc="upper right")
    ax.set_xlabel('Wavelength')
    ax.set_ylabel('Flux')
    plt.show(fig)
    plt.close(fig)






if __name__=='__main__':
    sys.exit(main())
