#!/usr/bin/env python
import os
import warnings
warnings.simplefilter('once')
import argparse
import numpy as np
import WDmodel
import WDmodel.io
import WDmodel.fit
from astropy import units as u
from specutils.extinction import reddening
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties as FM
import corner
#import matplotlib.gridspec as gridspec
#from matplotlib import rc
#rc('text', usetex=True)
#rc('font', family='serif')
#rc('ps', usedistiller='xpdf')
#rc('text.latex', preamble = ','.join('''\usepackage{amsmath}'''.split()))



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
    # spectrum options
    parser.add_argument('--specfile', required=True, \
            help="Specify spectrum to fit")
    parser.add_argument('--bluelimit', required=False, type=float,\
            help="Specify blue limit of spectrum - trim wavelengths lower")
    parser.add_argument('--redlimit', required=False, type=float,\
            help="Specify red limit of spectrum - trim wavelengths higher")
    parser.add_argument('--blotch', required=False, action='store_true',\
            default=False, help="Blotch the spectrum to remove gaps/cosmic rays before fitting?")
    parser.add_argument('-f', '--fwhm', required=False, type=float, default=None,\
            help="Specify custom instrumental resolution for smoothing (FWHM, angstroms)")
    
    # output options
    parser.add_argument('-o', '--outdir', required=False,\
            help="Specify a custom output directory. Default is CWD+objname/ subdir")

    # photometry options
    parser.add_argument('--photfile', required=False,  default="data/WDphot_C22.dat",\
            help="Specify file containing photometry lookup table for objects")
    parser.add_argument('-r', '--rv', required=False, type=float, default=3.1,\
            help="Specify reddening law R_V")
    parser.add_argument('--reddeningmodel', required=False, default='od94',\
            help="Specify functional form of reddening law" )

    # fitting options
    parser.add_argument('--nwalkers',  required=False, type=int, default=200,\
            help="Specify number of walkers to use (0 disables MCMC)")
    parser.add_argument('--nburnin',  required=False, type=int, default=200,\
            help="Specify number of steps for burn-in")
    parser.add_argument('--nprod',  required=False, type=int, default=1000,\
            help="Specify number of steps for production")
    parser.add_argument('--nthreads',  required=False, type=int, default=1,\
            help="Specify number of threads to use. (>1 implies multiprocessing)")
    parser.add_argument('--discard',  required=False, type=float, default=5,\
            help="Specify percentage of steps to be discarded")
    parser.add_argument('--redo',  required=False, action="store_true", default=False,\
            help="Clobber existing fits")

    # visualization options
    parser.add_argument('-b', '--balmerlines', nargs='+', type=int, default=range(1,7,1),\
            help="Specify Balmer lines to visualize [1:7]")

    args = parser.parse_args()

    # some sanity checking for option values
    balmer = args.balmerlines
    try:
        balmer = np.atleast_1d(balmer).astype('int')
        if np.any((balmer < 1) | (balmer > 6)):
            raise ValueError
    except (TypeError, ValueError):
        message = 'Invalid balmer line value - must be in range [1,6]'
        raise ValueError(message)

    if args.rv < 2.1 or args.rv > 5.1:
        message = 'That Rv Value is ridiculous'
        raise ValueError(message)

    if args.fwhm is not None:
        if args.fwhm < 0:
            message = 'Gaussian Smoothing FWHM cannot be less that zero'
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

    specfile  = args.specfile
    bluelim   = args.bluelimit
    redlim    = args.redlimit
    blotch    = args.blotch
    fwhm      = args.fwhm

    outdir    = args.outdir

    photfile  = args.photfile
    rv        = args.rv
    rvmodel   = args.reddeningmodel
    
    nwalkers  = args.nwalkers
    nburnin   = args.nburnin
    nprod     = args.nprod
    nthreads  = args.nthreads
    discard   = args.discard
    redo      = args.redo

    balmer    = args.balmerlines

    # read spectrum
    spec = WDmodel.io.read_spec(specfile)

    # get resolution
    fwhm = WDmodel.io.get_spectrum_resolution(specfile, fwhm=fwhm)

    # pre-process spectrum
    out = WDmodel.fit.pre_process_spectrum(spec, bluelim, redlim, blotch=blotch)
    spec, cont_model, linedata, continuumdata = out

    # set the object name and create output directories
    objname, outdir = WDmodel.io.set_objname_outdir_for_specfile(specfile, outdir=outdir)

    # get photometry 
    phot = WDmodel.io.get_phot_for_obj(objname, photfile)

    # fit the spectrum
    res = WDmodel.fit.fit_model(spec, phot,\
                objname, outdir, specfile,\
                rv=rv, rvmodel=rvmodel, fwhm=fwhm,\
                nwalkers=nwalkers, nburnin=nburnin, nprod=nprod, nthreads=nthreads,\
                redo=redo)

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


