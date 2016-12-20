#!/usr/bin/env python
import sys
import os
import glob
import numpy as np
import WDmodel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def main():
    files = glob.glob('/data/wdcalib/spectroscopy/*.flm')
    mod = WDmodel.WDmodel()

    offset = 0.
    
    with PdfPages('err_spec_plots.pdf') as pdf:

        for f in files:
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(1,1,1)
            for line in mod._lines:
                _, H, _, _ = mod._lines[line]
                ax.axvline(H, color='red', linestyle='-.')
            out = np.recfromtxt(f, names=True)

            ax.errorbar(out.wave, out.flux, out.flux_err, marker='None', capsize=0, linestyle='-', color='grey', alpha=0.7) 
            ax.annotate(os.path.basename(f), xy=(out.wave[0], out.flux[0]),\
                    xycoords='data', xytext=(200,0), textcoords="offset points", ha='left', va='center') 
            ax.plot(out.wave, out.flux, marker='None', linestyle='-', color='k')
            ax.set_xlabel('Wavelength')                                                                                         
            ax.set_ylabel('Flux')                                                                                               
            fig.suptitle(os.path.basename(f))
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)



if __name__=='__main__':
    sys.exit(main())
