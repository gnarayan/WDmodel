#!/usr/bin/env python 
# -*- coding: utf-8 -*-
'''
Compare old and new apparent magnitudes from AC photometry
'''
import sys
import os
import astropy.table as at
from astropy.visualization import hist
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def main():
    old_file = '../20170301/AC/WDphot_C22.dat'
    new_file = 'Combined/WDphot_C22_AC_AS_combined.dat'

    old = at.Table.read(old_file, format='ascii')
    new = at.Table.read(new_file, format='ascii')
    pbs = [x for x in new.dtype.names if x.startswith('F')]
    fig1 = plt.figure(figsize=(10,15))
    fig2 = plt.figure(figsize=(10,15))
    fig3 = plt.figure(figsize=(10,15))

    cols = 'purple,blue,green,red,orange,black'
    col_pbs = dict(zip(pbs, cols.split(','))) 

    combined = at.join(old, new, keys='obj', table_names=['old','new'])
    print(combined)

    for i, pb in enumerate(pbs):
        ax1 = fig1.add_subplot(3,2,i+1)
        ax2 = fig2.add_subplot(3,2,i+1)
        ax3 = fig3.add_subplot(3,2,i+1)


        old_key = '{}_old'.format(pb)
        new_key = '{}_new'.format(pb)
        old_ekey = 'd{}_old'.format(pb)
        new_ekey = 'd{}_new'.format(pb)

        ax1.errorbar(combined[old_key], combined[new_key], xerr=combined[old_ekey], yerr=combined[new_ekey],\
                marker='o', linestyle='None', color=col_pbs[pb], label=pb)
        xmin, xmax = ax1.get_xlim()
        ax1.plot([xmin, xmax], [xmin, xmax], marker='None', linestyle='--', color='black', alpha=0.5)
        ax1.legend(frameon=False)
        ax1.set_xlabel('Old Mag')
        ax1.set_ylabel('New Mag')

        ax2.errorbar(combined[old_key], combined[new_key] - combined[old_key],\
                xerr=combined[old_ekey], yerr = combined[new_ekey],\
                marker='o', linestyle='None', color=col_pbs[pb], label=pb)
        ax2.axhline(0., marker='None', linestyle='--', color='black', alpha=0.5)
        ax2.legend(frameon=False)
        ax2.set_xlabel('Old Mag')
        ax2.set_ylabel('Delta (New - Old) Mag')

        hist(combined[new_key] - combined[old_key], histtype='stepfilled', bins='scott', facecolor=col_pbs[pb], label=pb, ax=ax3)
        ax3.axvline(0., linestyle='--', color='black', alpha=0.5)
        ax3.set_xlabel('{} New - Old'.format(pb))

    plt.tight_layout()
    with PdfPages('AC_AS_apparent_phot_comparison.pdf') as pdf:
        fig1.suptitle('Mag Mag comparison')
        pdf.attach_note('Mag Mag comparison')
        pdf.savefig(fig1)

        fig2.suptitle('Mag DeltaMag comparison')
        pdf.attach_note('Mag DeltaMag comparison')
        pdf.savefig(fig2)

        fig3.suptitle('DeltaMag histogram')
        pdf.attach_note('DeltaMag histogram')
        pdf.savefig(fig3)






if __name__=='__main__':
    sys.exit(main())


