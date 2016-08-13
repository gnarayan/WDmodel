#!/usr/bin/env python
import sys
import os
import cProfile
import argparse
import numpy as np
import scipy.optimize as op
import scipy.integrate as scinteg
import george
import emcee
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
import corner
#rc('text', usetex=True)
#rc('font', family='serif')
#rc('ps', usedistiller='xpdf')
#rc('text.latex', preamble = ','.join('''\usepackage{amsmath}'''.split()))
#**************************************************************************************************************


def lnprior(theta):
    #a, tau, teff, logg, av  = np.exp(theta)
    a, tau, teff, logg, av  = theta
    if 17000. < teff < 80000. and 7.0 < logg < 9.5 and 0. <= av <= 0.5 and 1. < tau < 1000. and  0.01 < a < 10.:
        return 0.
    return -np.inf

def lnprob(theta, wave, model, data, kernel, balmer):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, wave, model, data, kernel, balmer)


def lnlike(theta, wave, model, data, kernel, balmer):
    #a, tau, teff, logg, av  = np.exp(theta)
    a, tau, teff, logg, av  = theta
    xi = model._get_xi(teff, logg, wave)
    mod = model._get_model(xi)

    # redden the model
    bluening = reddening(wave*u.Angstrom, av, r_v=3.1, model='od94')
    mod*=bluening

    # smooth the model, and extract the section that overlays the model
    # since we smooth the full model, computed on the full wavelength range of the spectrum
    # and then extract the subset range that overlaps with the data
    # we avoid any edge effects with smoothing at the end of the range
    smoothedmod = convolve(mod, kernel)
    W0, ZE, data_flux_norm = balmer
    we, smoothedfn = model._extract_from_indices(wave, smoothedmod, ZE)
    model_flux_norm = scinteg.simps(smoothedfn*we, we)
    datawave, datafn, datafnerr = data
    #gp  = george.GP(a*george.kernels.Matern32Kernel(tau))
    #gp.compute(datawave, datafnerr)
    smoothedfn*=(data_flux_norm/model_flux_norm)
    #return gp.lnlikelihood(datafn - smoothedfn)
    return -0.5*np.sum(((datafn-smoothedfn)/datafnerr)**2.)


def nll(*args):
    return -lnlike(*args)


#**************************************************************************************************************

def fit_model(objname, spec, balmer=None, av=0., rv=3.1, rvmodel='od94', smooth=4., photfile=None):

    wave = spec.wave
    flux = spec.flux
    fluxerr = spec.flux_err

    if photfile is not None:
        phot = read_phot(photfile)
        # set the likelihood functions here 
    else:
        phot = None
        # set the likelihood functions here 

    if balmer is None:
        balmer = np.arange(1, 7)
    else:
        balmer = np.array(sorted(balmer))

    nbalmer = len(balmer)
    nparam  = 5 #this is terrible

    # init a simple Gaussian 1D kernel to smooth the model to the resolution of the instrument
    gsig     = smooth*(0.5/(np.log(2.)**0.5))
    kernel   = Gaussian1DKernel(gsig)


    # init the model, and determine the coarse normalization to match the spectrum
    model = WDmodel.WDmodel()
    data = {}
     
    
    # figure out what range of data we're fitting
    # we go from slightly blue of the bluest line, to slightly red of the reddest line requested
    # we do not fit the whole range of the data, because the flux cannot be trusted as the QE rolls off
    redlinelim  = balmer.min()
    bluelinelim = balmer.max()
    _, Wred,  WIDr, DWr = model._lines[redlinelim]
    _, Wblue, WIDb, DWb = model._lines[bluelinelim]
    WA = Wblue - WIDb - DWb
    WB = Wred  + WIDr + DWr

    # extract the section of the data that covers the requested wavelength range
    # note that it's left to user to make sure they requested Balmer lines that are actually covered by the data
    W0, ZE = model._get_indices_in_range(wave, WA, WB)
    data_flux_norm = scinteg.simps(flux[ZE]*wave[ZE],wave[ZE])
    data = (wave[ZE], flux[ZE], fluxerr[ZE])

    # save the indices so we can extract the same section in the likelihood function without recomputing wastefully
    balmerwaveindex = (W0, ZE, data_flux_norm)

    p0 = np.ones(nparam).tolist()
    bounds = []

    # these are just hard-coded initial guesses
    # eventually we should make these options but the fit does wander away from them pretty quickly
    # bounds is for the scipy least squares minimizer 
    p0[0] = 1.
    p0[1] = 1.
    p0[2] = 40000.
    p0[3] = 7.5
    p0[4] = 0.1
    bounds.append((0.5,1.5))
    bounds.append((0.,1.))
    bounds.append((17000,80000))
    bounds.append((7.,9.499999))
    bounds.append((0.,0.5))
    

    # do a quick fit with minimize to get a decent starting guess
    # HACK HACK HACK - disable scipy for now, until we can figure out how to call it with the new args
    #result = op.minimize(nll, p0, args=(wave, model, data, kernel, balmerwaveindex), bounds=bounds)
    #print result

    # setup the sampler
    ndim, nwalkers = nparam, 100
    pos = [p0 + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(wave, model, data, kernel, balmerwaveindex)) 

    # do a short burn-in
    print "Burn-in"
    pos, prob, state = sampler.run_mcmc(pos, 500)
    sampler.reset()

    # production
    print "Production"
    pos = pos[np.argmax(prob)] + 1e-8 * np.random.randn(nwalkers, ndim)
    pos, prob, state = sampler.run_mcmc(pos,1000)   

    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

    # make a corner plot
    samples = sampler.chain[:,:, :].reshape((-1, ndim))
    plot_model(objname, spec, data, model, samples, kernel, balmerwaveindex, nparam)

#**************************************************************************************************************

def plot_spectrum_fit(objname, spec, data, model, samples, kernel, balmer, nparam):
    
    font  = FM(size='small')
    font2 = FM(size='x-small')
    font3 = FM(size='large')
    font4 = FM(size='medium')

    wave = spec.wave
    flux = spec.flux
    fluxerr = spec.flux_err
    WO, ZE, data_flux_norm  = balmer

    fig = plt.figure(figsize=(10,8))
    ax_spec  = fig.add_axes([0.075,0.4,0.85,0.55])
    ax_resid = fig.add_axes([0.075, 0.1,0.85, 0.25])

    ax_spec.errorbar(wave, flux, fluxerr, color='grey', alpha=0.5, capsize=0, linestyle='-', marker='None')
    if samples is not None:
        expsamples = np.ones(samples.shape)
        expsamples[:, 1] = np.exp(samples[:, 1])
        crap_a, crap_tau, teff_mcmc, logg_mcmc, av_mcmc  = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                                            zip(*np.percentile(samples, [16, 50, 84],
                                                            axis=0)))

        print teff_mcmc
        print logg_mcmc
        print av_mcmc


        _, modelflux = model.get_model(teff_mcmc[0], logg_mcmc[0], wave=wave)
        _, modelhi   = model.get_model(teff_mcmc[0]+teff_mcmc[2], logg_mcmc[0]+logg_mcmc[2], wave=wave)
        _, modello   = model.get_model(teff_mcmc[0]-teff_mcmc[1], logg_mcmc[0]-logg_mcmc[1], wave=wave)

        # redden the model
        bluening   = reddening(wave*u.Angstrom, av_mcmc[0], r_v=3.1, model='od94')
        blueninglo = reddening(wave*u.Angstrom, av_mcmc[0]+av_mcmc[2], r_v=3.1, model='od94')
        blueninghi = reddening(wave*u.Angstrom, av_mcmc[0]-av_mcmc[1], r_v=3.1, model='od94')
        modelflux*=bluening
        modelhi  *=blueninghi
        modello  *=blueninglo

        smoothed   = convolve(modelflux, kernel)
        smoothedhi = convolve(modelhi, kernel)
        smoothedlo = convolve(modello, kernel)

        model_flux_norm = scinteg.simps((smoothed*wave)[ZE], wave[ZE])
        model_flux_norm_hi = scinteg.simps((smoothedhi*wave)[ZE], wave[ZE])
        model_flux_norm_lo = scinteg.simps((smoothedlo*wave)[ZE], wave[ZE])

        smoothed *=(data_flux_norm/model_flux_norm)
        smoothedhi *=(data_flux_norm/model_flux_norm_hi)
        smoothedlo *=(data_flux_norm/model_flux_norm_lo)


        ax_spec.fill(np.concatenate([wave, wave[::-1]]), np.concatenate([smoothedhi, smoothedlo[::-1]]),\
                alpha=0.5, fc='grey', ec='None')
        ax_spec.plot(wave, smoothed, color='red', linestyle='-',marker='None')
        ax_resid.errorbar(wave, flux-smoothed, fluxerr, linestyle='-', marker=None, capsize=0, color='grey', alpha=0.5)
        ax_resid.errorbar(wave[ZE], (flux-smoothed)[ZE], fluxerr[ZE], linestyle='-',\
                    marker=None, capsize=0, color='black', alpha=0.7)

    ax_spec.errorbar(wave[ZE], flux[ZE], fluxerr[ZE], color='black', capsize=0, linestyle='-', marker='None',alpha=0.7)

    ax_resid.set_xlabel('Wavelength~(\AA)',fontproperties=font3, ha='center')
    ax_spec.set_ylabel('Normalized Flux', fontproperties=font3)
    ax_resid.set_ylabel('Fit Residual Flux', fontproperties=font3)
    return fig

#**************************************************************************************************************

def plot_model(objname, spec, data, model, samples, kernel, balmer, nparam):

    outfilename = objname.replace('.flm','.pdf')
    with PdfPages(outfilename) as pdf:
        fig =  plot_spectrum_fit(objname, spec, data, model, samples, kernel, balmer, nparam)
        pdf.savefig(fig)

        #labels = ['Nuisance Amplitude', 'Nuisance Scale', r"$T_\text{eff}$" , r"$log(g)$", r"A$_V$"]
        labels = ['Nuisance Amplitude', 'Nuisance Scale', r"Teff" , r"log(g)", r"A_V"]
        fig = corner.corner(samples, bins=41, labels=labels, show_titles=True,quantiles=(0.16,0.84),\
             use_math_text=True)
        pdf.savefig(fig)
        #endwith
    


#**************************************************************************************************************

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--specfiles', required=True, nargs='+')
    parser.add_argument('--photfiles', required=False, nargs='+')
    parser.add_argument('-a', '--av', required=True, type=float, default=0.)
    parser.add_argument('-r', '--rv', required=False, type=float, default=3.1)
    parser.add_argument('--reddeningmodel', required=False, default='od94')
    parser.add_argument('-b', '--balmerlines', nargs='+', type=int, default=range(1,7,1))
    parser.add_argument('-s', '--smooth', required=False, type=float, default=4.)
    args = parser.parse_args()
    balmer = args.balmerlines
    specfiles = args.specfiles
    photfiles = args.photfiles

    try:
        balmer = np.atleast_1d(balmer).astype('int')
        if np.any((balmer < 1) | (balmer > 6)):
            raise ValueError
    except (TypeError, ValueError), e:
        message = 'Invalid balmer line value - must be in range [1,6]'
        raise ValueError(message)

    if photfiles is not None:
        if len(specfiles) != len(photfiles):
            #TODO: This does not support multiple spectra per object, but it's a mess to include that in the likelihood
            # skip this problem for now until we come up a way of dealing with it
            message = 'If you are specifying photometry for fitting, number of files needs to match number of spectra'
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

    reddeninglaws = ('od94', 'ccm89', 'gcc09', 'f99', 'fm07', 'wd01', 'd03')
    if not args.reddeningmodel in reddeninglaws:
        message = 'That reddening law is not known (%s)'%' '.join(reddeninglaws) 
        raise ValueError(message)

    return args


#**************************************************************************************************************

def read_spec(filename):
    """
    Really quick little read spectrum from file routine
    """
    spec = np.recfromtxt(filename, names=True, dtype='float64,float64,float64')
    WDmodel.WDmodel._wave_test(spec.wave)
    return spec

#**************************************************************************************************************

def read_phot(filename):
    """
    Read photometry from file - expects to have columns mag_aper magerr_aper and pb 
    Extra columns other than these three are fine
    """
    phot = np.recfromtxt(filename, names=True)
    print rec2txt(phot)
    return phot



#**************************************************************************************************************
def main():
#    test()
    args   = get_options() 
    balmer = np.atleast_1d(args.balmerlines).astype('int')
    specfiles = args.specfiles
    photfiles = args.photfiles
    
    for i in xrange(len(specfiles)):
        if photfiles is not None:
            photfile = photfiles[i]
        else:
            photfile = None
        specfile = specfiles[i]
        spec = read_spec(specfile)
    
        fit_model(specfile, spec, balmer, av=args.av, rv=args.rv, smooth=args.smooth, photfile=photfile)    

#**************************************************************************************************************

def test(grid_name='default'):
    model = WDmodel(grid_name=grid_name)
    fig   = plt.figure(figsize=(10,5))
    ax1   = fig.add_subplot(1,1,1)
    t = np.arange(20000., 85000, 5000.)
    l = np.arange(7, 9., 0.5)
    for logg in l:
        for teff in t:
            wave, flux  = model(teff, logg, log=False)
            ax1.plot(wave, flux, 'k-')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    plt.ion()
    plt.tight_layout()


#**************************************************************************************************************



if __name__=='__main__':
    cProfile.run('main()', 'profile.dat')
    import pstats
    p = pstats.Stats('profile.dat')
    p.sort_stats('cumulative','time').print_stats(20)


