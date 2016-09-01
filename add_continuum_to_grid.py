#!/usr/bin/env python
import sys
import os
import numpy as np
import h5py
import george
import WDmodel
import scipy.interpolate as scinterp
import scipy.signal as sig
import matplotlib.pyplot as plt
import peakutils

def main():
    in_grid  = 'TlustyGrids.hdf5'
    #out_grid = 'TlustyGrid_cont.hdf5'
    grid_name= 'default'
    indata   = h5py.File(in_grid, 'r')
    #outdata = h5py.File(out_grid, 'w')
    #for key in indata.iterkeys():
    #    indata.copy(key, outdata)
    grid = indata[grid_name]
    mod  = WDmodel.WDmodel()
    wave  = grid['wave'].value.astype('float64')
    ggrid = grid['ggrid'].value
    tgrid = grid['tgrid'].value
    flux  = grid['flux'].value.astype('float64')
    # ordering is wave logg teff
    lwave = np.log10(wave)


    # get the line indices of the model
    # clip them out from the continuum 
    line_inds = {}
    all_line_inds = np.array([],dtype='int',ndmin=1)
    for l in range(1, 7):
        W0, ZE = mod._get_line_indices(wave, l)
        line_inds[l] = (W0, ZE)
        all_line_inds = np.hstack((all_line_inds, ZE[0]))
    all_line_inds = np.unique(all_line_inds)
    cind = all_line_inds
    print len(all_line_inds), len(cind), len(wave)

    cut_wave = np.log10(np.delete(wave, cind))
    cflux = np.zeros(flux.shape,dtype='float64')

    fig = plt.figure(figsize=(10,10), tight_layout=True)
    ax  = fig.add_subplot(2,1,1)
    ax2  = fig.add_subplot(2,1,2)

    for t in xrange(len(tgrid)):
        for g in xrange(len(ggrid)):
            this_flux = flux[:,g,t]
            baseline = peakutils.baseline(np.log10(this_flux), deg=1, max_it=200, tol=1e-4)
            base2 = sig.savgol_filter(np.log10(this_flux), 2001, 3)

            cut_flux  = np.log10(np.delete(this_flux,cind))
            f = scinterp.interp1d(cut_wave, cut_flux, assume_sorted=True)
            full_flux = f(lwave)
            cflux[:,g,t] = 10.**full_flux
            ax.plot(wave, this_flux,color='black', ls='-', marker='None')
            #ax.plot(10.**cut_wave, 1+(10.**cut_flux),color='red', ls='-', marker='None',lw=3, alpha=0.7)
            ax.plot(wave, (10.**full_flux), color='grey', ls='-', marker='None')
            ax.plot(wave, 10**base2, color='red',ls='-', marker='None')
            diff = np.gradient(np.log10(this_flux), lwave)    
            ax2.plot(wave, diff, color='black')

    ax.set_xlabel('Wavelength')
    ax.set_ylabel('Flux')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.show(fig)
    plt.close(fig)



if __name__=='__main__':
    sys.exit(main())
