#!/usr/bin/env python
import sys
import os
import warnings
warnings.simplefilter('once')
import argparse
from clint.textui import progress
import numpy as np
import scipy.optimize as op
import scipy.interpolate as scinterp
import scipy.integrate as scinteg
import h5py
import george
import emcee
import WDmodel
import WDmodel.io
import WDmodel.fit
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


def lnprob(theta, wave, model, kernel, balmer):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, wave, model, kernel, balmer)


def lnlike(theta, wave, model, kernel, balmerlinedata):
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
    (line_wave, line_flux, line_fluxerr, line_number, line_cflux, line_cov, line_ind, save_ind, mu, cov) = balmerlinedata

    mod_cflux = get_model_continuum(line_wave, wave, smoothedmod, save_ind, line_ind)
    _, smoothedfn = model._extract_from_indices(wave, smoothedmod, line_ind)

    smoothedfn*=(line_cflux/mod_cflux)
    #return gp.lnlikelihood(datafn - smoothedfn)
    return -0.5*np.sum(((line_flux-smoothedfn)/line_fluxerr)**2.)


def nll(*args):
    return -lnlike(*args)


#**************************************************************************************************************

def get_model_continuum(loc, wave, flux, save_ind, line_ind):

    cwave = np.delete(wave, save_ind)
    cflux = np.delete(flux, save_ind)
    coeff = np.polyfit(cwave, cflux, deg=21)
    mod_cflux = np.polyval(coeff, loc)
    #mod_cflux = np.interp(loc, cwave, cflux)
    #_, mod_cflux, _ = bspline_continuum((cwave, cflux, None), loc)
    return mod_cflux


#**************************************************************************************************************



def quick_fit_model(spec,linedata, continuumdata, save_ind,  balmer, model, kernel):
    """
    Does a quick fit of the spectrum to get an initial guess of the spectral parameters
    This isn't robust, but it's good enough for an initial guess
    The guess defines the regions of the spectrum that are actually used for each line in the full fit
    These regions are only weak functions of Teff, logg
    It beats hard-coded pre-defined regions which are only valid for some Teff, logg
    """

    nparam = 3

    wave    = spec.wave
    flux    = spec.flux
    fluxerr = spec.flux_err

    cwave, cflux, cdflux = continuumdata
    (line_wave, line_flux, line_fluxerr, line_number, line_ind) = linedata
    # do a quick Gaussian Process Fit to model the continuum of the specftrum
    gp = george.GP(kernel=george.kernels.ExpSquaredKernel(10))
    gp.compute(cwave, cdflux)
    pars, result = gp.optimize(cwave, cflux, cdflux, bounds=((10,100000),))
    mu, cov = gp.predict(cflux, wave)
    line_cflux , line_cov = gp.predict(cflux, line_wave)

    balmerlinedata = (line_wave, line_flux, line_fluxerr, line_number, line_cflux, line_cov, line_ind, save_ind, mu, cov)

    p0 = np.ones(nparam).tolist()
    bounds = []
    p0[0] = 40000.
    p0[1] = 7.5
    p0[2] = 0.1
    bounds.append((17000,80000))
    bounds.append((7.,9.499999))
    bounds.append((0.,0.5))

    # do a quick fit with minimize to get a decent starting guess
    result = op.minimize(nll, p0, args=(wave, model, kernel, balmerlinedata), bounds=bounds)
    print result
    return balmerlinedata, continuumdata, result


#**************************************************************************************************************

def fit_model(objname, spec, linedata, continuumdata, save_ind, balmer=None, rv=3.1, rvmodel='od94', smooth=4., photfile=None,\
            nwalkers=200, nburnin=500, nprod=2000, nthreads=1, outdir=os.getcwd(), redo=False):

    outfile = os.path.join(outdir, os.path.basename(objname.replace('.flm','.mcmc.hdf5')))
    if os.path.exists(outfile) and (not redo):
        print("Output file already exists. Specify --redo to clobber.")
        sys.exit(0)

    nparam = 3


    #TODO - do something with the photometry to fit Av
    if photfile is not None:
        phot = read_phot(photfile)
        # set the likelihood functions here 
    else:
        phot = None
        # set the likelihood functions here 

    wave    = spec.wave
    flux    = spec.flux
    fluxerr = spec.flux_err
    outf = h5py.File(outfile, 'w')
    dset_spec = outf.create_group("spec")
    dset_spec.create_dataset("wave",data=wave)
    dset_spec.create_dataset("flux",data=flux)
    dset_spec.create_dataset("fluxerr",data=fluxerr)

    # init a simple Gaussian 1D kernel to smooth the model to the resolution of the instrument
    gsig     = smooth/np.sqrt(8.*np.log(2.))
    kernel   = Gaussian1DKernel(gsig)

    # init the model, and determine the coarse normalization to match the spectrum
    model = WDmodel.WDmodel()
    data = {}
    
    # bundle the line dnd continuum data so we don't have to extract it every step in the MCMC
    if balmer is None:
        balmer = np.arange(1, 7)
    else:
        balmer = np.array(sorted(balmer))
    
    # do a quick, not very robust fit to create the quantities we need to store
    # get a reasonable starting position for the chains 
    # and set the wavelength thresholds for each line
    balmerlinedata, continuumdata, result = quick_fit_model(spec, linedata, continuumdata, save_ind, balmer, model, kernel)

    (line_wave, line_flux, line_fluxerr, line_number, line_cflux, line_cov, line_ind, save_ind, mu, cov) = balmerlinedata
    (cwave, cflux, cdflux) = continuumdata

    dset_lines = outf.create_group("lines")
    dset_lines.create_dataset("line_wave",data=line_wave)
    dset_lines.create_dataset("line_flux",data=line_flux)
    dset_lines.create_dataset("line_fluxerr",data=line_fluxerr)
    dset_lines.create_dataset("line_number",data=line_number)
    dset_lines.create_dataset("line_cflux",data=line_cflux)
    dset_lines.create_dataset("line_cov",data=line_cov)
    dset_lines.create_dataset("line_ind",data=line_ind)
    dset_lines.create_dataset("save_ind",data=save_ind)

    # note that we bundle mu and cov (the continuum model and error) with balmerlinedata
    # but save it with continuumdata
    # the former makes sense for fitting
    # the latter is a more logical structure 
    dset_continuum = outf.create_group("continuum")
    dset_continuum.create_dataset("con_wave", data=cwave)
    dset_continuum.create_dataset("con_flux", data=cflux)
    dset_continuum.create_dataset("con_fluxerr", data=cdflux)
    dset_continuum.create_dataset("con_model", data=mu)
    dset_continuum.create_dataset("con_cov", data=cov)
    outf.close()
    

    if nwalkers==0:
        print "nwalkers set to 0. Not running MCMC"
        return

    # setup the sampler
    pos = [result.x + 1e-1*np.random.randn(nparam) for i in range(nwalkers)]
    if nthreads > 1:
        print "Multiproc"
        sampler = emcee.EnsembleSampler(nwalkers, nparam, lnprob,\
                threads=nthreads, args=(wave, model, kernel, balmerlinedata)) 
    else:
        sampler = emcee.EnsembleSampler(nwalkers, nparam, lnprob, args=(wave, model, kernel, balmerlinedata)) 

    # do a short burn-in
    if nburnin > 0:
        print "Burn-in"
        pos, prob, state = sampler.run_mcmc(pos, nburnin, storechain=False)
        sampler.reset()
        pos = pos[np.argmax(prob)] + 1e-2 * np.random.randn(nwalkers, nparam)

    # setup incremental chain saving
    # TODO need to test this alongside multiprocessing
    nstep = 100
    outf = h5py.File(outfile, 'a')
    chain = outf.create_group("chain")
    dset_chain = chain.create_dataset("position",(nwalkers*nprod,3),maxshape=(None,3))

    # production
    with progress.Bar(label="Production", expected_size=nprod) as bar:
        for i, result in enumerate(sampler.sample(pos, iterations=nprod)):
            position = result[0]
            dset_chain[nwalkers*i:nwalkers*(i+1),:] = position
            outf.flush()
            bar.show(i+1)
    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

    samples = np.array(dset_chain)
    outf.close()
    return model, samples, kernel, balmerlinedata


#**************************************************************************************************************

def plot_spectrum_fit(objname, spec,  model, samples, kernel, balmerlinedata, bwi):
    
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

    fig2 = plt.figure(figsize=(10,5))
    ax_lines = fig2.add_subplot(1,2,1)

    fig3 = plt.figure(figsize=(10,8))
    ax_spec_cont  = fig3.add_subplot(3,1,1)
    ax_mod_cont   = fig3.add_subplot(3,1,2, sharex=ax_spec_cont)
    ax_ratio_cont = fig3.add_subplot(3,1,3, sharex=ax_spec_cont)


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


        (line_wave, line_flux, line_fluxerr, line_number, line_cflux, line_cov, line_ind, save_ind, mu, cov) = balmerlinedata

        _, modelflux = model.get_model(teff_mcmc[0], logg_mcmc[0], wave=wave, strict=False)
        _, modelhi   = model.get_model(teff_mcmc[0]+teff_mcmc[2], logg_mcmc[0]+logg_mcmc[2], wave=wave, strict=False)
        _, modello   = model.get_model(teff_mcmc[0]-teff_mcmc[1], logg_mcmc[0]-logg_mcmc[1], wave=wave, strict=False)

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
        smoothedc   = get_model_continuum(wave, wave, smoothed,   save_ind, line_ind)
        smoothedchi = get_model_continuum(wave, wave, smoothedhi, save_ind, line_ind)
        smoothedclo = get_model_continuum(wave, wave, smoothedlo, save_ind, line_ind)

        cont_ratio = mu/smoothedc
        ax_spec_cont.plot(wave, mu/np.median(mu))
        ax_mod_cont.plot(wave, smoothedc/np.median(smoothedc))
        ax_mod_cont.plot(wave, smoothed/np.median(smoothedc), ls='-', color='red', alpha=0.7)
        ax_ratio_cont.plot(wave, cont_ratio/np.median(cont_ratio))
        ax_ratio_cont.set_xlabel('Wavelength~(\AA)')
        ax_spec_cont.set_ylabel('spectrum')
        ax_mod_cont.set_ylabel('model')
        ax_ratio_cont.set_ylabel('continuum')
        ax_spec_cont.set_title('Continuum Diagnostics')
        
        smoothed*=mu/smoothedc
        smoothedhi*=mu/smoothedchi
        smoothedlo*=mu/smoothedclo
        model_lines    = smoothed[line_ind]
        model_lines_hi = smoothedhi[line_ind]
        model_lines_lo = smoothedlo[line_ind]

        ax_spec.fill(np.concatenate([wave, wave[::-1]]), np.concatenate([smoothedhi, smoothedlo[::-1]]),\
                alpha=0.5, fc='grey', ec='None')
        ax_spec.plot(wave, smoothed, color='red', linestyle='-',marker='None')
        ax_resid.errorbar(wave, flux-smoothed, fluxerr, linestyle='-', marker=None, capsize=0, color='grey', alpha=0.5)

        for l in np.unique(line_number):
            m = (line_number == l)
            try:
                W0, uZE = bwi[l]
            except KeyError, e:
                print l, e
                continue 
            ax_resid.errorbar(line_wave[m], (line_flux[m]-smoothed[line_ind][m]), line_fluxerr[m], linestyle='-',\
                    marker=None, capsize=0, color='black', alpha=0.7)
        
            ax_lines.fill_between(line_wave[m]-W0,\
                    ((model_lines_lo)/line_cflux)[m]+0.2*line_number[m],\
                    ((model_lines_hi)/line_cflux)[m]+0.2*line_number[m],\
                    facecolor='purple', alpha=0.5, interpolate=True)
            ax_lines.fill_between(line_wave[m]-W0,\
                    ((line_flux-line_fluxerr)/line_cflux)[m]+0.2*line_number[m],\
                    ((line_flux+line_fluxerr)/line_cflux)[m]+0.2*line_number[m],\
                    facecolor='grey', alpha=0.5, interpolate=True)
            ax_lines.plot(line_wave[m]-W0, line_flux[m]/line_cflux[m] + 0.2*line_number[m],color='k',ls='-',lw=2)
            ax_lines.plot(line_wave[m]-W0, model_lines[m]/line_cflux[m] + 0.2*line_number[m], color='r',ls='-',alpha=0.7)

        spacing = np.median(np.diff(wave))
        lagind = c.argmax()
        offset = -lags[lagind]*spacing
        ax_lines.set_title('Lines vs Models')
        ax_lines.set_xlabel('Delta Wavelength~(\AA)')
        ax_lines.set_ylabel('Normalized Line Flux')


    for l in np.unique(line_number):
        m = (line_number == l)
        ax_spec.errorbar(line_wave[m], line_flux[m], line_fluxerr[m],\
                    color='black', capsize=0, linestyle='-', marker='None',alpha=0.7)

    ax_resid.set_xlabel('Wavelength~(\AA)',fontproperties=font3, ha='center')
    ax_spec.set_ylabel('Normalized Flux', fontproperties=font3)
    ax_resid.set_ylabel('Fit Residual Flux', fontproperties=font3)
    return fig, fig2, fig3


#**************************************************************************************************************

def plot_continuum_fit(objname, spec, continuumdata, balmerlinedata):

    font  = FM(size='small')
    font2 = FM(size='x-small')
    font3 = FM(size='large')
    font4 = FM(size='medium')

    wave = spec.wave
    flux = spec.flux
    fluxerr = spec.flux_err
    cwave, cflux, cdflux = continuumdata
    (line_wave, line_flux, line_fluxerr, line_number, line_cflux, line_cov, line_ind, save_ind, mu, cov) = balmerlinedata

    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)
    ax1.errorbar(spec.wave, spec.flux, yerr=spec.flux_err, marker='None',\
                        linestyle='-', color='grey', alpha=0.3, capsize=0)
    ax1.plot(spec.wave, spec.flux, marker='None', linestyle='-', color='black', alpha=0.8)
    ax1.plot(cwave, cflux, marker='.', linestyle='None', color='blue', ms=1)
    ax1.plot(spec.wave, mu, marker='None', linestyle='--', color='red')

    ax2.errorbar(spec.wave, spec.flux/mu, yerr=spec.flux_err/mu, marker='None',\
            linestyle='-', color='grey', alpha=0.3, capsize=0)
    ax2.plot(spec.wave, spec.flux/mu, marker='None', linestyle='-', color='black', alpha=0.8)

    ax2.set_xlabel('Wavelength~(\AA)',fontproperties=font3, ha='center')
    ax1.set_ylabel('Flux', fontproperties=font3)
    ax2.set_ylabel('Norm Flux', fontproperties=font3)
    ax1.set_title(objname)
    plt.tight_layout()
    return fig


#**************************************************************************************************************

def plot_model(objname, spec,  model, samples, kernel, continuumdata, balmerlinedata, bwi, outdir=os.getcwd(), discard=5):

    outfilename = os.path.join(outdir, os.path.basename(objname.replace('.flm','.pdf')))
    with PdfPages(outfilename) as pdf:
        fig = plot_continuum_fit(objname, spec, continuumdata, balmerlinedata)
        pdf.savefig(fig)
        fig, fig2, fig3  =  plot_spectrum_fit(objname, spec,  model, samples, kernel, balmerlinedata, bwi)
        pdf.savefig(fig)
        pdf.savefig(fig3)
        pdf.savefig(fig2)


        #labels = ['Nuisance Amplitude', 'Nuisance Scale', r"$T_\text{eff}$" , r"$log(g)$", r"A$_V$"]
        labels = [r"Teff" , r"log(g)", r"A_V"]
        samples = samples[int(round(discard*samples.shape[0]/100)):]
        fig = corner.corner(samples, bins=41, labels=labels, show_titles=True,quantiles=(0.16,0.84),\
             use_math_text=True)
        pdf.savefig(fig)
        #endwith
    

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
    parser.add_argument('-s', '--smooth', required=False, type=float, default=None,\
            help="Specify instrumental resolution for smoothing (FWHM, angstroms)")
    parser.add_argument('-r', '--rv', required=False, type=float, default=3.1,\
            help="Specify reddening law R_V")
    parser.add_argument('--reddeningmodel', required=False, default='od94',\
            help="Specify functional form of reddening law" )
    parser.add_argument('--photfile', required=False,  default="data/WDphot_C22.dat",\
            help="Specify file containing photometry for objects")
    parser.add_argument('--blotch', required=False, action='store_true',\
            default=False, help="Blotch the spectrum to remove gaps/cosmic rays before fitting?")
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
    parser.add_argument('--redo',  required=False, action="store_true", default=False,\
            help="Clobber existing fits")
    args = parser.parse_args()
    balmer = args.balmerlines
    specfiles = args.specfiles
    photfile  = args.photfile

    try:
        balmer = np.atleast_1d(balmer).astype('int')
        if np.any((balmer < 1) | (balmer > 6)):
            raise ValueError
    except (TypeError, ValueError), e:
        message = 'Invalid balmer line value - must be in range [1,6]'
        raise ValueError(message)

    if args.rv < 2.1 or args.rv > 5.1:
        message = 'That Rv Value is ridiculous'
        raise ValueError(message)

    if args.smooth is not None:
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

def main():
    args   = get_options() 

    specfiles = args.specfiles
    photfile  = args.photfile
    nwalkers  = args.nwalkers
    nburnin   = args.nburnin
    nprod     = args.nprod
    smooth    = args.smooth
    nthreads  = args.nthreads
    outdir    = args.outdir
    discard   = args.discard
    redo      = args.redo
    blotch    = args.blotch
    bluelim   = args.bluelimit
    redlim    = args.redlimit
    balmer    = args.balmerlines

    # set the object name and create output directories
    objname, outdir = WDmodel.io.set_objname_outdir_for_specfiles(specfiles, outdir=outdir)

    # get photometry 
    phot = WDmodel.io.get_phot_for_obj(objname, photfile)

    for specfile in specfiles:

        # read spectrum
        spec = WDmodel.io.read_spec(specfile)

        # get resolution
        smooth = WDmodel.io.get_spectrum_resolution(specfile, smooth=smooth)

        # pre-process spectrum
        out = WDmodel.fit.pre_process_spectrum(spec, smooth, bluelim, redlim, balmer, blotch=blotch)
        (spec, linedata, continuumdata, save_ind, balmer, smooth, bwi) = out
    

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(spec.wave, spec.flux, 'k-')
        (line_wave, line_flux, line_fluxerr, line_number, line_ind) = linedata
        for line in np.unique(line_number):
            mask = (line_number == line)
            ax.plot(line_wave[mask], line_flux[mask], 'b-')
        cwave, cflux, cdflux = continuumdata
        ax.plot(cwave, cflux, 'r.')
        plt.ion()
        plt.show(fig)

        raw_input()
        sys.exit(-1)

        # fit the spectrum
        res = fit_model(specfile, spec, linedata, continuumdata, save_ind, balmer,\
                rv=args.rv, smooth=smooth, photfile=photfile,\
                nwalkers=nwalkers, nburnin=nburnin, nprod=nprod, nthreads=nthreads, outdir=outdir, redo=redo)
        model, samples, kernel, balmerlinedata = res

        # plot output
        plot_model(specfile, spec,  model, samples, kernel, continuumdata, balmerlinedata, bwi, outdir=outdir, discard=discard)
    return


#**************************************************************************************************************

if __name__=='__main__':
    main()
    #cProfile.run('main()', 'profile.dat')
    #import pstats
    #p = pstats.Stats('profile.dat')
    #p.sort_stats('cumulative','time').print_stats(20)


