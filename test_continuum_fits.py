#!/usr/bin/env python
import sys
import os
import warnings
warnings.simplefilter('once')
import glob
import WDmodel
from fit_WDmodel import read_spec
import numpy as np
import scipy.interpolate as scinterp
import scipy.stats as scistat
import george
import statsmodels.robust
import pandas
import matplotlib.pyplot as plt
from pandas import rolling_median
from matplotlib.backends.backend_pdf import PdfPages
from mpi4py import MPI
balmer = (1, 2, 3, 4, 5, 6)



def orig_cut_lines(spec, model):
    wave    = spec.wave
    flux    = spec.flux
    fluxerr = spec.flux_err
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
    linedata = (line_wave, line_flux, line_fluxerr, line_number, line_ind)
    return linedata, continuumdata, save_ind



def bspline(cv, n=100, degree=3, periodic=False):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
        periodic: True - Curve is closed
                  False - Curve is open
    """
    # If periodic, extend the point array by count+degree+1
    cv = np.asarray(cv)
    count = len(cv)

    if periodic:
        factor, fraction = divmod(count+degree+1, count)
        cv = np.concatenate((cv,) * factor + (cv[:fraction],))
        count = len(cv)
        degree = np.clip(degree,1,degree)

    # If opened, prevent degree from exceeding count-1
    else:
        degree = np.clip(degree,1,count-1)

    # Calculate knot vector
    kv = None
    if periodic:
        kv = np.arange(0-degree,count+degree+degree-1,dtype='int')
    else:
        kv = np.array([0]*degree + range(count-degree+1) + [count-degree]*degree,dtype='int')

    # Calculate query range
    u = np.linspace(periodic,(count-degree),n)

    # Calculate result
    arange = np.arange(len(u))
    points = np.zeros((len(u),cv.shape[1]))
    for i in xrange(cv.shape[1]):
                points[arange,i] = scinterp.splev(u, (kv,cv[:,i],degree))
    return points



def gp_continuum(continuumdata, bsw, bsf, wave):
    f = scinterp.interp1d(bsw, bsf, kind='linear', fill_value='extrapolate')
    def mean_func(x):
        x = np.array(x).ravel()
        return f(x)

    cwave, cflux, cdflux = continuumdata
    kernel= np.median(cdflux)**2.*george.kernels.ExpSquaredKernel(100.)
    gp = george.GP(kernel=kernel, mean=mean_func)
    gp.compute(cwave, cdflux)
    pars, result = gp.optimize(cwave, cflux, cdflux,\
                    bounds=((None, np.log(3*np.median(cdflux)**2.)),\
                            (np.log(100.),np.log(100000)) ))
    mu, cov = gp.predict(cflux, wave)
    return wave, mu, cov



def bspline_continuum(continuumdata, wave):
    cwave, cflux, cdflux = continuumdata
    cv = zip(cwave, cflux)
    points = bspline(cv, n=len(wave))
    cbspline = np.rec.fromrecords(points, names='wave,flux')
    mu = np.interp(wave, cbspline.wave, cbspline.flux)
    return wave, mu, None



def main():
    spec_path = '/data/wdcalib/spectroscopy/'
    spec_files = list(glob.glob(os.path.join(spec_path, '*.flm')))
    model = WDmodel.WDmodel()
    figures = []
    scaling = scistat.norm.ppf(3/4.)
    window1 = 15
    window2 = 101

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    procs_num = comm.Get_size()
    nfiles = len(spec_files)
    quotient = nfiles/procs_num+1
    P = rank*quotient
    Q = (rank+1)*quotient
    if P > nfiles:                                                                                              
        P = nfiles                                                                                                
    if Q > nfiles:                                                                                              
        Q = nfiles

    for f in spec_files[P:Q]:

        # read the spectrum
        spec = read_spec(f)

        # remove any NaNs
        ind = np.where((np.isnan(spec.wave)==0) & (np.isnan(spec.flux)==0) & (np.isnan(spec.flux_err)==0))
        spec = spec[ind]

        # init some figures
        fig = plt.figure(figsize=(10,5))
        ax  = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2,1,2, sharex=ax)

        ax.plot(spec.wave, spec.flux, marker='None', linestyle='--', color='black', alpha=0.3)
        # preserve the original lines 
        linedata, continuumdata, saveind = orig_cut_lines(spec, model)

        # do some initial filtering on the spectrum
        cflux = pandas.DataFrame(spec.flux)
        med_filt = cflux.rolling(window=window2,center=True).median().fillna(method='bfill').fillna(method='ffill')
        diff = np.abs(cflux - med_filt)
        sigma = diff.rolling(window=window2, center=True).median().fillna(method='bfill').fillna(method='ffill')
        sigma/=scaling
        outlier_idx = (diff > 5.*sigma)
        mask = outlier_idx.values.ravel()
        sigma = sigma.values.ravel()

        # clip the bad outliers from the spectrum
        spec.flux[mask] = med_filt.values.ravel()[mask]

        # restore the original lines, so that they aren't clipped
        spec.flux[saveind] = linedata[1]
        linedata, continuumdata, saveind = orig_cut_lines(spec, model)

        cflux = pandas.DataFrame(continuumdata[1])
        med_filt = cflux.rolling(window=window2,center=True).median().fillna(method='bfill').fillna(method='ffill')
        diff = np.abs(cflux - med_filt)
        sigma = diff.rolling(window=window2, center=True).median().fillna(method='bfill').fillna(method='ffill')
        sigma/=scaling
        outlier_idx = (diff > 5.*sigma)
        mask = outlier_idx.values.ravel()
        sigma = sigma.values.ravel()

        # plot up the sigma clipping results
        ax2.fill_between(continuumdata[0], continuumdata[1]-sigma, continuumdata[1] + sigma, facecolor='grey', alpha=0.5)
        ax2.plot(continuumdata[0], continuumdata[1], marker='.', ms=1, linestyle='None', color='black', alpha=0.5)
        ax2.plot(continuumdata[0], continuumdata[1], marker='.', ms=1, linestyle='None', color='black', alpha=0.5)
        ax2.plot(continuumdata[0], med_filt, marker='.', ms=1, linestyle='None', color='blue', alpha=0.5)
        ax2.plot(continuumdata[0][mask], continuumdata[1][mask], marker='.', ms=2, linestyle='None',\
                color='red', alpha=0.5)
        
        # toss the outliers
        continuumdata[1][mask] = med_filt.values.ravel()[mask]
        
        # compare GP and b-spline
        bsw, bsf, _     = bspline_continuum(continuumdata, spec.wave)
        gpw, gpf, gpcov = gp_continuum(continuumdata, bsw, bsf, spec.wave)


        # plot up results
        ax.errorbar(spec.wave, spec.flux, yerr=spec.flux_err, marker='None',\
                        linestyle='-', color='grey', alpha=0.3, capsize=0)
        ax.plot(spec.wave, spec.flux, marker='None', linestyle='-', color='black', alpha=0.5)
        ax.plot(continuumdata[0], continuumdata[1], marker='.', linestyle='None', color='green')
        ax.plot(gpw, gpf, marker='None', linestyle='--', color='red')
        ax.plot(bsw, bsf, marker='None', linestyle='-.', color='blue')
    
        ax.set_ylabel('Flam')
        ax.set_xlabel('Wavelength')
        fig.suptitle(os.path.basename(f))
        plt.tight_layout()
        figures.append(fig)

    thisproc_figures = comm.gather(figures, root=0) 
    out_figures  = []
    if rank==0:
        if thisproc_figures is not None:
            if len(thisproc_figures) > 0:
                for proc in thisproc_figures:
                    out_figures += proc
    comm.Barrier()
    with PdfPages('cmod/con_mod.pdf') as pdf:
        for fig in out_figures:
            pdf.savefig(fig)
            plt.close(fig)
    return






if __name__=='__main__':
    main()
