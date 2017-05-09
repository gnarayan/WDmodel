#!/usr/bin/env python
"""
Script illustrating that without the line information, you can shift the
temperature and av quite a lot and still get a very similar looking continuum,
and it is really the lines which set teff and logg
"""

import sys
import numpy as np
from WDmodel.WDmodel import WDmodel
from astropy import units as u
import matplotlib.pyplot as plt
import extinction

def main():
    mod = WDmodel(rvmodel='od94')
    wave = np.arange(1500., 18010., 10.)
    av   = 0.3
    _, t25 = mod.get_model(25000, 7.8, wave=wave)
    _, t80 = mod.get_model(80000, 7.8, wave=wave)
    plt.ion()
    reddening = 10.**(-0.4*extinction.odonnell94(wave, av, 3.1, unit='aa'))

    fig = plt.figure(figsize=(16,5))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    ax1.plot(wave, t25,          color='red' ,marker=None, ls='-' ,label='25k K, log g=7.8, Av=0.0')
    ax1.plot(wave, t25*reddening, color='red' ,marker=None, ls='--',label='25k K, log g=7.8, Av=0.3')
    ax1.plot(wave, t80,          color='blue',marker=None, ls='-' ,label='80k K, log g=7.8, Av=0.0')
    ax1.plot(wave, t80*reddening, color='blue',marker=None, ls='--',label='80k K, log g=7.8, Av=0.3')

    ax2.plot(wave, t25,          color='red' ,marker=None, ls='-' ,label='25k K, log g=7.8, Av=0.0')
    ax2.plot(wave, t25*reddening, color='red' ,marker=None, ls='--',label='25k K, log g=7.8, Av=0.3')
    ax2.plot(wave, t80,          color='blue',marker=None, ls='-' ,label='80k K, log g=7.8, Av=0.0')
    ax2.plot(wave, t80*reddening, color='blue',marker=None, ls='--',label='80k K, log g=7.8, Av=0.3')
    ax2.plot(wave, t25*np.mean(t80)/np.mean(t25),          color='green' ,marker=None, ls='-' ,alpha=0.6, label='25k K, log g=7.8, Av=0.0 shifted')
    ax2.plot(wave, t25*reddening*np.mean(t80)/np.mean(t25), color='green' ,marker=None, ls='--',alpha=0.6, label='25k K, log g=7.8, Av=0.3 shifted')
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(2400,3000)
    ax1.legend(frameon=False)
    ax2.legend(frameon=False)
    ax1.set_xlabel('Wavelength')
    ax2.set_xlabel('Wavelength')
    ax1.set_ylabel('Relative Flux')
    plt.savefig('WD_comparison_plot.pdf')
    plt.show(fig)


if __name__=='__main__':
    sys.exit(main())



