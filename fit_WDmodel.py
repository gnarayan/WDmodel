#!/usr/bin/env python
import sys
import os
import cProfile
import argparse
import numpy as np
import scipy.optimize as op
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
rc('text', usetex=True)
rc('font', family='serif')
rc('ps', usedistiller='xpdf')
#**************************************************************************************************************


def lnprior(theta):
    teff, logg, av = theta
    if 20000. < teff < 75000. and 7.0 < logg < 9.5 and 0. <= av <= 0.5:
        return 0.
    return -np.inf

def lnprob(theta, wave, model, data, kernel, balmer):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, wave, model, data, kernel, balmer)


def lnlike(theta, wave, model, data, kernel, balmer):
    #teff, logg, m, c  = theta
    teff, logg, av = theta

    xi = model._get_xi(teff, logg, wave)
    mod = model._get_model(xi)
    bluening = reddening(wave*u.Angstrom, av, r_v=3.1, model='od94')
    mod*=bluening
    smoothed = convolve(mod, kernel)
    W0, ZE = balmer
    datawave, datafn, datafnerr = data
    we, fe = model._extract_from_indices(wave, smoothed, ZE)
    smoothedfn = fe #+ 10.**(m*np.log10(we) + c)
    res = (datafn - smoothedfn)
    sig = datafnerr
    chisqr = np.sum((res**2.)/(sig**2.))
    return -0.5*chisqr 

def nll(*args):
    return -lnlike(*args)


#**************************************************************************************************************

def fit_model(objname, spec, balmer=None, av=0., rv=3.1, rvmodel='od94', smooth=4., photfile=None):

    wave = spec.wave
    flux = spec.flux

    fluxvar = spec.fluxvar
    fluxerr = (fluxvar/3.)**0.5 #HACK HACK HACK 
    #Tom stacked three spectra but the variance he sent was the total, not the variance of the mean
    #in future, we'll get fluxerr = standard err of the mean (which is what we actually want)    

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
    nparam  = 3 #this is terrible

    gsig     = smooth*(0.5/(np.log(2.)**0.5))
    kernel   = Gaussian1DKernel(gsig)
    model = WDmodel.WDmodel()
    data = {}
    model._normalize_model(spec)
    print model._fluxnorm, "flux_norm"
     

    redlinelim  = balmer.min()
    bluelinelim = balmer.max()
     
    _, Wred,  WIDr, DWr = model._lines[redlinelim]
    _, Wblue, WIDb, DWb = model._lines[bluelinelim]
    WA = Wblue - WIDb - DWb
    WB = Wred  + WIDr + DWr
    W0, ZE = model._get_indices_in_range(wave, WA, WB)
    data = (wave[ZE], flux[ZE], fluxerr[ZE])
    balmerwaveindex = (W0, ZE)

    p0 = np.ones(nparam).tolist()
    bounds = []
    p0[0] = 40000.
    p0[1] = 7.5
    p0[2] = 0.1
    bounds.append((17000,80000))
    bounds.append((7.,9.499999))
    bounds.append((0.,0.5))
    #for i in range(2, nparam):
    #    bounds.append((None, None))
    

    # do a quick fit with minimize to get a decent starting guess
    result = op.minimize(nll, p0, args=(wave, model, data, kernel, balmerwaveindex), bounds=bounds)
    print result

    # setup the sampler
    ndim, nwalkers = nparam, 100
    pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(wave, model, data, kernel, balmerwaveindex)) 

    # do a short burn-in
    pos, prob, state = sampler.run_mcmc(pos, 500)
    sampler.reset()

    # production
    sampler.run_mcmc(pos, 1000)   

    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

    # make a corner plot
    samples = sampler.chain[:,:, :].reshape((-1, ndim))
    plot_model(objname, spec, data, model, samples, kernel, balmerwaveindex, nparam)

#**************************************************************************************************************

def plot_model(objname, spec, data, model, samples, kernel, balmer, nparam):

    font  = FM(size='small')
    font2 = FM(size='x-small')
    font3 = FM(size='large')
    font4 = FM(size='medium')

#    wave = spec.wave
#    flux = spec.flux

#   expsamples = np.ones(samples.shape)
#   expsamples[:, 1] = np.exp(samples[:, 1])
#   teff_mcmc, logg_mcmc  = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
#                            zip(*np.percentile(samples, [16, 50, 84],
#                                               axis=0)))
#
#   print teff_mcmc, logg_mcmc
#
#   with PdfPages(outfilename) as pdf:
#       fig = plt.figure(figsize=(8,8))
#       ax  = fig.add_subplot(1,1,1)
#
#       _, modelflux = model.get_model(teff_mcmc[0], logg_mcmc[0], wave=wave)
#       _, modelhi   = model.get_model(teff_mcmc[0]+teff_mcmc[2], logg_mcmc[0]+logg_mcmc[2], wave=wave)
#       _, modellow  = model.get_model(teff_mcmc[0]-teff_mcmc[1], logg_mcmc[0]-logg_mcmc[1], wave=wave)
#       smoothed   = convolve(modelflux, kernel)
#       smoothedhi = convolve(modelhi, kernel)
#       smoothedlo = convolve(modellow, kernel)
#
#
#       for line in data:
#           wn, fn, fnerr, wc, fc, fcerr = data[line]
#
#           offset = 0.2*line-0.2
#           W0, ZE = balmer[line]
#           we, fe = model._extract_from_indices(wave, smoothed, ZE)
#           smoothedwn, smoothedfn, norm, swc, sfc, norm2  = model._normalize_line(X, Y, we, fe, W0)
#
#           we, fe = model._extract_from_indices(wave, smoothedhi, ZE)
#           smoothedwnhi, smoothedfnhi, norm, swchi, sfchi, norm2hi  = model._normalize_line(X, Y, we, fe, W0)
#
#           we, fe = model._extract_from_indices(wave, smoothedlo, ZE)
#           smoothedwnlo, smoothedfnlo, norm, swclo, sfclo, norm2lo  = model._normalize_line(X, Y, we, fe, W0)
#
#           ax.fill(np.concatenate([smoothedwnhi, smoothedwnlo[::-1]]), np.concatenate([smoothedfnhi, smoothedfnlo[::-1]])+offset,\
#                   alpha=0.5, fc='grey', ec='None')
#           ax.errorbar(wn, fn+offset, fnerr, capsize=0, linestyle='-', lw=0.5, color='k', marker='None')
#           bluewing = (wc <= wn.min())
#           redwing  = (wc >= wn.min())
#           ax.errorbar(wc[bluewing], fc[bluewing]+offset, fcerr[bluewing], capsize=0, linestyle='-', lw=0.5, color='k', marker='None')
#           ax.errorbar(wc[redwing], fc[redwing]+offset, fcerr[redwing], capsize=0, linestyle='-', lw=0.5, color='k', marker='None')
#           ax.axhline(1.+offset,color='grey',linestyle='-.', alpha=0.3, xmin=wc.min(), xmax=wc.max())
#           ax.plot(smoothedwn, smoothedfn+offset, 'r-', alpha=0.75, marker='None')
#
#           linename = model._lines[line][0]
#           ax.annotate(r"H$_"+"\\"+linename+"$", xy=(wn[-1], offset+fn[-1]), xycoords='data', xytext= (8.,0.),\
#                    textcoords="offset points", fontproperties=font3,  ha='left', va="bottom")
#           indzero = np.abs(wn).argmin()
#           ax.annotate(str(W0)+'\AA', xy = (wn[indzero], offset+fn[indzero]),  xycoords='data',\
#                 xytext= (0.,-8.), textcoords="offset points", fontproperties=font4,  ha='center', va="top")
#
#       ax.set_xlabel('$\Delta$Wavelength~(\AA)',fontproperties=font3, ha='center')
#       ax.set_ylabel('Normalized Flux', fontproperties=font3)
#       pdf.savefig(fig)

    outfilename = objname.replace('.flm','.pdf')
    with PdfPages(outfilename) as pdf:
        labels = ["$T_eff$" , r"$log(g)$", r"A$_V$"]
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
    spec = np.recfromtxt(filename, names='wave,flux,fluxvar', dtype='float64,float64,float64')
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


