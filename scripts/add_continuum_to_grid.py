#!/usr/bin/env python
import sys
sys.path.append('/data/wdcalib/local/python/WDmodel/WDmodel')
import os
import numpy as np
import h5py
import george
import WDmodel
import scipy.interpolate as scinterp
import scipy.signal as sig
import matplotlib.pyplot as plt
import math as M

def filter_lines(wave, diff, diff2, lflux):
    window2 = 501
    diff_smoothed = sig.medfilt(diff, kernel_size=window2)
    diff_delta = np.abs(diff - diff_smoothed)
    diff2a = np.abs(diff2)
    mask = np.where((diff_delta < 5*np.median(diff_delta)) & (diff2a <= 5*np.median(diff2a)))[0]
    return mask

def nextpow2(i):
    buf = M.ceil(M.log(i) / M.log(2))
    return int(M.pow(2, buf))

def main():
    in_grid  = '/data/wdcalib/local/python/WDmodel/WDmodel/TlustyGrids.hdf5'
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


    fig = plt.figure(figsize=(15,10), tight_layout=True)
    ax  = fig.add_subplot(3,1,1)
    ax2  = fig.add_subplot(3,1,2, sharex=ax)
    ax3  = fig.add_subplot(3,1,3, sharex=ax)

    for t in xrange(len(tgrid)):
        off = 0
        for g in xrange(len(ggrid)):
            this_flux = flux[:,g,t]
            lflux = np.log10(this_flux)

            a_cpx = np.zeros((lflux.shape), dtype=np.complex64)
            a_cpx = sig.hilbert(this_flux)
            a_abs = abs(a_cpx)
            print a_abs.shape
            print wave.shape

            ax.plot(wave, this_flux*(10**off),color='black', ls='-', marker='None')
            ax.plot(wave, (a_abs)*(10**off), color='blue', ls='--',marker='None')
            off += 1
            break
        break

    ax.set_xlim(3700,4800)
    ax.set_xlabel('Wavelength')
    ax.set_ylabel('Flux')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.show(fig)
    plt.close(fig)



if __name__=='__main__':
    sys.exit(main())
