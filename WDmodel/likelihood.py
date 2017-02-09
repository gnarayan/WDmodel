import numpy as np
from astropy.convolution import convolve
#TODO: figure out a way to fix parameters, add priors

def lnprior(theta):
    teff, logg, av  = theta
    if 17000. < teff < 80000. and 7.0 < logg < 9.5 and 0. <= av <= 0.5:
        return 0.
    return -np.inf


def lnprob(theta, wave, model, kernel, balmer):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, wave, model, kernel, balmer)


def lnlike(theta, wave, model, kernel, balmerlinedata):
    teff, logg, av  = theta

    mod = model._get_red_model(teff, logg, av, wave)

    # smooth the model, and extract the section that overlays the model
    # since we smooth the full model, computed on the full wavelength range of the spectrum
    # and then extract the subset range that overlaps with the data
    # we avoid any edge effects with smoothing at the end of the range
    smoothedmod = convolve(mod, kernel)
    (line_wave, line_flux, line_fluxerr, line_number, line_cflux, line_cov, line_ind, save_ind, mu, cov) = balmerlinedata

    smoothedfn*=(line_cflux/mod_cflux)
    #return gp.lnlikelihood(datafn - smoothedfn)
    return -0.5*np.sum(((line_flux-smoothedfn)/line_fluxerr)**2.)


def nll(*args):
    return -lnlike(*args)

