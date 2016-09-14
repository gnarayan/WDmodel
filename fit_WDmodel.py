#!/usr/bin/env python
import sys
import os
import argparse
from clint.textui import progress
import numpy as np
import scipy.optimize as op
import scipy.integrate as scinteg
import h5py
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
    teff, logg, av  = theta
    if 17000. < teff < 80000. and 7.0 < logg < 9.5 and 0. <= av <= 0.5:
        return 0.
    return -np.inf

def lnprob(theta, wave, model, data, kernel, balmer):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, wave, model, data, kernel, balmer)


def lnlike(theta, wave, model, data, kernel, balmerlinedata):
    teff, logg, av  = theta
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
    (line_wave, line_flux, line_fluxerr, line_number, line_cflux, line_cov, line_ind, save_ind, mu) = balmerlinedata

    cwave = np.delete(wave, save_ind)
    cflux = np.delete(smoothedmod, save_ind)
    mod_cflux = np.interp(line_wave, cwave, cflux)
    _, smoothedfn = model._extract_from_indices(wave, smoothedmod, line_ind)

    smoothedfn*=(line_cflux/mod_cflux)
    #return gp.lnlikelihood(datafn - smoothedfn)
    return -0.5*np.sum(((line_flux-smoothedfn)/line_fluxerr)**2.)


def nll(*args):
    return -lnlike(*args)


#**************************************************************************************************************

def fit_model(objname, spec, balmer=None, rv=3.1, rvmodel='od94', smooth=4., photfile=None,\
            nwalkers=200, nburnin=500, nprod=2000, nthreads=1, outdir=os.getcwd(), discard=5):

    wave    = spec.wave
    flux    = spec.flux
    fluxerr = spec.flux_err

    if photfile is not None:
        phot = read_phot(photfile)
        # set the likelihood functions here 
    else:
        phot = None
        # set the likelihood functions here 

    outfile = os.path.join(outdir, os.path.basename(objname.replace('.flm','.mcmc.hdf5')))
    outf = h5py.File(outfile, 'w')
    dset_spec = outf.create_group("spec")
    dset_spec.create_dataset("wave",data=wave)
    dset_spec.create_dataset("flux",data=flux)
    dset_spec.create_dataset("fluxerr",data=fluxerr)


    # init a simple Gaussian 1D kernel to smooth the model to the resolution of the instrument
    gsig     = smooth*(0.5/(np.log(2.)**0.5))
    kernel   = Gaussian1DKernel(gsig)

    # init the model, and determine the coarse normalization to match the spectrum
    model = WDmodel.WDmodel()
    data = {}
    
    #  bundle the line dnd continuum data so we don't have to extract it every step in the MCMC
    if balmer is None:
        balmer = np.arange(1, 7)
    else:
        balmer = np.array(sorted(balmer))
    nbalmer = len(balmer)
    balmerwaveindex = {}
    line_wave     = np.array([], dtype='float64', ndmin=1)
    line_flux     = np.array([], dtype='float64', ndmin=1)
    line_fluxerr  = np.array([], dtype='float64', ndmin=1)
    line_number   = np.array([], dtype='int', ndmin=1)
    line_ind      = np.array([], dtype='int', ndmin=1)
    save_ind      = np.array([], dtype='int', ndmin=1)
    for x in range(1,7):
        W0, ZE = model._get_line_indices(wave, x)
        # save the central wavelengths and spectrum specific indices for each line
        # we don't need this for the fit, but we do need this for plotting 
        x_wave, x_flux, x_fluxerr = model._extract_from_indices(wave, flux, ZE, df=fluxerr)
        if x in balmer:
            balmerwaveindex[x] = W0, ZE
            line_wave    = np.hstack((line_wave, x_wave))
            line_flux    = np.hstack((line_flux, x_flux))
            line_fluxerr = np.hstack((line_fluxerr, x_fluxerr))
            line_number  = np.hstack((line_number, np.repeat(x, len(x_wave))))
            line_ind     = np.hstack((line_ind, ZE[0]))
        save_ind     = np.hstack((save_ind, ZE[0]))

    # continuum data is just the spectrum with the Balmer lines removed
    continuumdata  = (np.delete(wave, save_ind), np.delete(flux, save_ind), np.delete(fluxerr, save_ind))


    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.errorbar(wave, flux, fluxerr,linestyle='-', marker='None', capsize=0, color='grey')
    for x in balmer:
        m = (line_number == x)
        ax.errorbar(line_wave[m], line_flux[m], line_fluxerr[m], linestyle='-', marker='None', capsize=0, color='k')
    cwave, cflux, cdflux = continuumdata
    # Eventually maybe supply an option to choose which model to use
    #gp = george.GP(kernel=george.kernels.Matern32Kernel(10))
    gp = george.GP(kernel=george.kernels.ExpSquaredKernel(10))
    gp.compute(cwave, cdflux)
    pars, result = gp.optimize(cwave, cflux, cdflux, bounds=((10,100000),))
    print pars
    print result
    mu, cov = gp.predict(cflux, wave)
    line_cflux , line_cov = gp.predict(cflux, line_wave)
    ax.errorbar(cwave, cflux, cdflux, capsize=0, linestyle='-.', marker='None',color='grey')
    ax.plot(wave, mu, color='red', linestyle='-', marker='None')
    noise = cov.diagonal()**0.5
    ax.fill_between(wave,  mu+noise, mu-noise, color='red',alpha=0.5)
    plt.show(fig)
    plt.close(fig)

    balmerlinedata = (line_wave, line_flux, line_fluxerr, line_number, line_cflux, line_cov, line_ind, save_ind, mu)
    dset_lines = outf.create_group("lines")
    dset_lines.create_dataset("line_wave",data=line_wave)
    dset_lines.create_dataset("line_flux",data=line_flux)
    dset_lines.create_dataset("line_fluxerr",data=line_fluxerr)
    dset_lines.create_dataset("line_number",data=line_number)
    dset_lines.create_dataset("line_cflux",data=line_cflux)
    dset_lines.create_dataset("line_cov",data=line_cov)
    dset_lines.create_dataset("line_ind",data=line_ind)
    dset_lines.create_dataset("save_ind",data=save_ind)

    dset_continuum = outf.create_group("continuum")
    dset_continuum.create_dataset("con_wave", data=cwave)
    dset_continuum.create_dataset("con_flux", data=cflux)
    dset_continuum.create_dataset("con_fluxerr", data=cdflux)
    dset_continuum.create_dataset("con_model", data=mu)
    dset_continuum.create_dataset("con_cov", data=cov)
    outf.close()
    
    nparam  = 3 
    p0 = np.ones(nparam).tolist()
    bounds = []
    p0[0] = 40000.
    p0[1] = 7.5
    p0[2] = 0.1
    bounds.append((17000,80000))
    bounds.append((7.,9.499999))
    bounds.append((0.,0.5))

    # do a quick fit with minimize to get a decent starting guess
    result = op.minimize(nll, p0, args=(wave, model, data, kernel, balmerlinedata), bounds=bounds)
    print result

    if nwalkers==0:
        print "nwalkers set to 0. Not running MCMC"
        return

    # setup the sampler
    pos = [result.x + 1e-3*np.random.randn(nparam) for i in range(nwalkers)]
    if nthreads > 1:
        print "Multiproc"
        sampler = emcee.EnsembleSampler(nwalkers, nparam, lnprob,\
                threads=nthreads, args=(wave, model, data, kernel, balmerlinedata)) 
    else:
        sampler = emcee.EnsembleSampler(nwalkers, nparam, lnprob, args=(wave, model, data, kernel, balmerlinedata)) 

    # do a short burn-in
    print "Burn-in"
    pos, prob, state = sampler.run_mcmc(pos, nburnin, storechain=False)
    sampler.reset()

    # setup incremental chain saving
    # TODO need to test this alongside multiprocessing
    nstep = 100
    outf = h5py.File(outfile, 'a')
    chain = outf.create_group("chain")
    dset_chain = chain.create_dataset("position",(nwalkers*nprod,3),maxshape=(None,3))

    # production
    pos = pos[np.argmax(prob)] + 1e-2 * np.random.randn(nwalkers, nparam)
    with progress.Bar(label="Production", expected_size=nprod) as bar:
        for i, result in enumerate(sampler.sample(pos, iterations=nprod)):
            position = result[0]
            dset_chain[nwalkers*i:nwalkers*(i+1),:] = position
            outf.flush()
            bar.show(i+1)
        outf.close()
    #pos, prob, state = sampler.run_mcmc(pos, nprod)   

    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

    # make a corner plot
    samples = sampler.flatchain
    samples = samples[round(discard*nwalkers*nprod/100.):]
    return data, model, samples, kernel, balmerlinedata

#**************************************************************************************************************

def plot_spectrum_fit(objname, spec, data, model, samples, kernel, balmerlinedata):
    
    font  = FM(size='small')
    font2 = FM(size='x-small')
    font3 = FM(size='large')
    font4 = FM(size='medium')

    wave = spec.wave
    flux = spec.flux
    fluxerr = spec.flux_err

    fig = plt.figure(figsize=(10,8))
    ax_spec  = fig.add_axes([0.075,0.4,0.85,0.55])
    ax_resid = fig.add_axes([0.075, 0.1,0.85, 0.25])

    ax_spec.errorbar(wave, flux, fluxerr, color='grey', alpha=0.5, capsize=0, linestyle='-', marker='None')
    if samples is not None:
        expsamples = np.ones(samples.shape)
        expsamples[:, 1] = np.exp(samples[:, 1])
        #crap_a, crap_tau, teff_mcmc, logg_mcmc, av_mcmc  = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
        #                                                    zip(*np.percentile(samples, [16, 50, 84],
        #                                                    axis=0)))
        teff_mcmc, logg_mcmc, av_mcmc  = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                                            zip(*np.percentile(samples, [16, 50, 84],
                                                            axis=0)))

        print teff_mcmc
        print logg_mcmc
        print av_mcmc

        (line_wave, line_flux, line_fluxerr, line_number, line_cflux, line_cov, line_ind, save_ind, mu) = balmerlinedata

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

        cwave = np.delete(wave, save_ind)
        smoothedc = np.interp(wave, cwave, np.delete(smoothed, save_ind))
        smoothedchi = np.interp(wave, cwave, np.delete(smoothedhi, save_ind))
        smoothedclo = np.interp(wave, cwave, np.delete(smoothedlo, save_ind))
        smoothed*=mu/smoothedc
        smoothedhi*=mu/smoothedchi
        smoothedlo*=mu/smoothedclo


        ax_spec.fill(np.concatenate([wave, wave[::-1]]), np.concatenate([smoothedhi, smoothedlo[::-1]]),\
                alpha=0.5, fc='grey', ec='None')
        ax_spec.plot(wave, smoothed, color='red', linestyle='-',marker='None')
        ax_resid.errorbar(wave, flux-smoothed, fluxerr, linestyle='-', marker=None, capsize=0, color='grey', alpha=0.5)
        for l in np.unique(line_number):
            m = (line_number == l)
            ax_resid.errorbar(line_wave[m], (line_flux[m]-smoothed[line_ind][m]), line_fluxerr[m], linestyle='-',\
                    marker=None, capsize=0, color='black', alpha=0.7)

    for l in np.unique(line_number):
        m = (line_number == l)
        ax_spec.errorbar(line_wave[m], line_flux[m], line_fluxerr[m],\
                    color='black', capsize=0, linestyle='-', marker='None',alpha=0.7)

    ax_resid.set_xlabel('Wavelength~(\AA)',fontproperties=font3, ha='center')
    ax_spec.set_ylabel('Normalized Flux', fontproperties=font3)
    ax_resid.set_ylabel('Fit Residual Flux', fontproperties=font3)
    return fig

#**************************************************************************************************************

def plot_model(objname, spec, data, model, samples, kernel, balmerlinedata, outdir=os.getcwd()):

    outfilename = os.path.join(outdir, os.path.basename(objname.replace('.flm','.pdf')))
    with PdfPages(outfilename) as pdf:
        fig =  plot_spectrum_fit(objname, spec, data, model, samples, kernel, balmerlinedata)
        pdf.savefig(fig)

        #labels = ['Nuisance Amplitude', 'Nuisance Scale', r"$T_\text{eff}$" , r"$log(g)$", r"A$_V$"]
        #labels = ['Nuisance Amplitude', 'Nuisance Scale', r"Teff" , r"log(g)", r"A_V"]
        labels = [r"Teff" , r"log(g)", r"A_V"]
        fig = corner.corner(samples, bins=41, labels=labels, show_titles=True,quantiles=(0.16,0.84),\
             use_math_text=True)
        pdf.savefig(fig)
        #endwith
    
#**************************************************************************************************************

def make_outdirs(dirname):
    print("Writing to outdir {}".format(dirname))
    if os.path.isdir(dirname):
        return

    try:
        os.makedirs(dirname)
    except OSError, e:
        message = '%s\nCould not create outdir %s for writing.'
        raise OSError(message)
    


#**************************************************************************************************************

def get_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--specfiles', required=True, nargs='+',\
            help="Specify spectra to fit. Can be used multiple times.")
    parser.add_argument('-b', '--balmerlines', nargs='+', type=int, default=range(1,7,1),\
            help="Specifiy Balmer lines to fit [1:7]")
    parser.add_argument('-o', '--outdir', required=False,\
            help="Specify a custom output directory. Default is CWD+objname/ subdir")
    parser.add_argument('--bluelimit', required=False, type=float,\
            help="Specify blue limit of spectrum - trim wavelengths lower")
    parser.add_argument('--redlimit', required=False, type=float,\
            help="Specify red limit of spectrum - trim wavelengths higher")
    parser.add_argument('-s', '--smooth', required=False, type=float, default=4.,\
            help="Specify instrumental resolution for smoothing (FWHM, angstroms)")
    parser.add_argument('-r', '--rv', required=False, type=float, default=3.1,\
            help="Specify reddening law R_V")
    parser.add_argument('--reddeningmodel', required=False, default='od94',\
            help="Specify functional form of reddening law" )
    parser.add_argument('--photfiles', required=False, nargs='+')
    parser.add_argument('--nthreads',  required=False, type=int, default=1,\
            help="Specify number of threads to use. (>1 implies multiprocessing)")
    parser.add_argument('--nwalkers',  required=False, type=int, default=200,\
            help="Specify number of walkers to use (0 disables MCMC)")
    parser.add_argument('--nburnin',  required=False, type=int, default=200,\
            help="Specify number of steps for burn-in")
    parser.add_argument('--nprod',  required=False, type=int, default=1000,\
            help="Specify number of steps for production")
    parser.add_argument('--discard',  required=False, type=float, default=5,\
            help="Specify percentage of steps to be discarded")
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

    if args.nwalkers < 0:
        message = 'Number of walkers must be greater than zero for MCMC'
        raise ValueError(message)

    if args.nthreads <= 0:
        message = 'Number of threads must be greater than zero'
        raise ValueError(message)
    
    if args.nburnin <= 0:
        message = 'Number of walkers must be greater than zero'
        raise ValueError(message)

    if args.nprod <= 0:
        message = 'Number of walkers must be greater than zero'
        raise ValueError(message)

    if not (0 <= args.discard < 100):
        message = 'Discard must be a percentage (0-100)'
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
    args   = get_options() 
    balmer = np.atleast_1d(args.balmerlines).astype('int')
    specfiles = args.specfiles
    photfiles = args.photfiles
    nwalkers  = args.nwalkers
    nburnin   = args.nburnin
    nprod     = args.nprod
    nthreads  = args.nthreads
    outdir    = args.outdir
    discard   = args.discard
    if outdir is not None:
        make_outdirs(outdir)


    for i in xrange(len(specfiles)):
        if photfiles is not None:
            photfile = photfiles[i]
        else:
            photfile = None
        specfile = specfiles[i]
        if outdir is None:
            dirname = os.path.join(os.getcwd(), os.path.basename(specfile.replace('.flm','')))
            make_outdirs(dirname)
        else:
            dirname = outdir

        spec = read_spec(specfile)

        if args.bluelimit > 0:
            bluelimit = float(args.bluelimit)
        else:
            bluelimit = spec.wave.min()

        if args.redlimit > 0:
            redlimit = float(args.redlimit)
        else:
            redlimit = spec.wave.max()
        
        mask = ((spec.wave >= bluelimit) & (spec.wave <= redlimit))
        spec = spec[mask]
        res = fit_model(specfile, spec, balmer, rv=args.rv, smooth=args.smooth, photfile=photfile,\
                nwalkers=nwalkers, nburnin=nburnin, nprod=nprod, nthreads=nthreads, outdir=dirname, discard=discard)
        data, model, samples, kernel, balmerlinedata = res
        plot_model(specfile, spec, data, model, samples, kernel, balmerlinedata, outdir=dirname)
#**************************************************************************************************************


if __name__=='__main__':
    main()
    #cProfile.run('main()', 'profile.dat')
    #import pstats
    #p = pstats.Stats('profile.dat')
    #p.sort_stats('cumulative','time').print_stats(20)


