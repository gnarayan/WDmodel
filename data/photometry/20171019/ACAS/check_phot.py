#!/usr/bin/env python
import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import astropy.stats as at
import astropy.table as ast
from numpy.lib.recfunctions import append_fields
from scipy.stats import linregress

def main():
    pattern = '*_F[0-9][0-9][0-9]W.all'
    files = glob.glob(pattern)
    tf = [fn.replace('.all','').split('_') for fn in files]
    targets, filters = zip(*tf)
    targets = sorted(set(targets))
    filters = sorted(set(filters))
    ref  = 'Msex'
    comp = ['Mila', 'Mdao']
    marker = ['o', 's']
    fig1 = plt.figure(figsize=(8, 10))
    fig2 = plt.figure(figsize=(8, 10))
    fig3 = plt.figure(figsize=(8, 10))
    for j, c in enumerate(comp):
        outfile = '{}_cut_photometry.txt'.format(c)
        if os.path.exists(outfile):
            os.remove(outfile)
        for i, pb in enumerate(filters):
            ax1 = fig1.add_subplot(3, 2, i+1)
            ax2 = fig2.add_subplot(3, 2, i+1)
            ax3 = fig3.add_subplot(3, 2, i+1)
            all_mags = None

            flag = False
            for t in targets:
                fn = '{}_{}.all'.format(t, pb)
                try:
                    mags = np.recfromtxt(fn, names=True)
                    mags = np.atleast_1d(mags)
                except ValueError as e:
                    print(fn, "format inconsistent")
                    continue
                nmags = len(mags)
                data = [np.repeat(t, nmags), np.repeat(pb, nmags), np.repeat(c, nmags)]
                names = ['objID','pb','method']
                dtypes = ['S30','S5','S4']
                mags = append_fields(mags, names, data=data, dtypes=dtypes)

                mask = (mags['d'+ref] < 0.5) & (mags['d'+c] < 0.5) & (np.abs(mags[ref] - mags[c]) < 1.)
                if len(mags[mask]) == 0:
                    print(fn, "No data", ref, c)
                    continue
                nbad = len(mags[~mask])
                if nbad > 0:
                    print(fn, nbad, "bad obs", ref, c)
                mags = mags[mask]
                if flag is False:
                    label = '{}'.format(c)
                    flag = True
                else:
                    label = None
                ax1.errorbar(mags[ref], mags[c]-mags[ref], xerr=mags['d'+ref], yerr = mags['d'+c],\
                        linestyle='None', color='C{:n}'.format(i+5*j), marker=marker[j], ms=3, alpha=1-0.6*j)

                if all_mags is None:
                    all_mags = mags
                else:
                    all_mags = np.hstack((all_mags, mags))
            #endfor targets
            mean_offset = np.average(all_mags[c]-all_mags[ref], weights = 1./(all_mags['d'+c]**2. + all_mags['d'+ref]**2.))
            label = '{} mean_offset {:.3f}'.format(c, mean_offset)
            ax1.errorbar(mags[ref], mags[c]-mags[ref], xerr=mags['d'+ref], yerr = mags['d'+c],\
                    linestyle='None', color='C{:n}'.format(i+5*j), marker=marker[j], ms=3, alpha=1-0.6*j, label=label, visible=False)

            masked_mags = at.sigma_clip(all_mags[c]-all_mags[ref],\
                    sigma=3, sigma_lower=None, sigma_upper=None, iters=5,\
                    cenfunc=np.ma.median, stdfunc=np.std, axis=None, copy=True)
            cut = masked_mags.mask

            out = ast.Table(all_mags[cut])
            outfile = '{}-{}_cut_photometry.txt'.format(c, ref)
            with open(outfile, 'a') as f:
                if i == 0:
                    out.write(f, format='ascii.fixed_width', delimiter=' ')
                else:
                    out.write(f, format='ascii.fixed_width_no_header', delimiter=' ')


            mags = all_mags[~cut]
            mean_offset = np.average(mags[c]-mags[ref], weights = 1./(mags['d'+c]**2. + mags['d'+ref]**2.))

            label = '{} mean_offset {:.3f}'.format(c, mean_offset)
            ax2.errorbar(mags[ref], mags[c]-mags[ref], xerr=mags['d'+ref], yerr = mags['d'+c],\
                    linestyle='None', color='C{:n}'.format(i+5*j), marker=marker[j], ms=3, alpha=1-0.6*j, label=label)

            slope, intercept, _, _, _ = linregress(mags[ref], mags[c]-mags[ref])
            xmin, xmax = ax2.get_xlim()
            xc = np.array([xmin, xmax])
            yc = xc*slope + intercept
            ax2.plot(xc, yc, linestyle='-', color='C{:n}'.format(i+5*j), marker='None', lw=3, alpha=0.5)


            label = '{}'.format(c)
            print(j, c, ref, 'errors', pb)
            ax3.plot(mags['d'+ref], mags['d'+c],\
                    linestyle='None', color='C{:n}'.format(i+5*j), marker=marker[j], ms=3, alpha=1-0.6*j, label=label)

            if j == 1:
                label = '{} - {}'.format('/'.join(comp), ref)
                ax1.set_xlabel('{} {}'.format(ref, pb))
                ax1.set_ylabel('{} {}'.format(label, pb))
                xmin, xmax = ax1.get_xlim()
                ax1.plot([xmin, xmax], [0., 0.], color='grey', alpha=0.5, linestyle='--', lw=2)
                ax1.legend(frameon=False)

                ax2.set_xlabel('{} {}'.format(ref, pb))
                ax2.set_ylabel('{} {}'.format(label, pb))
                xmin, xmax = ax2.get_xlim()
                ax2.plot([xmin, xmax], [0., 0.], color='grey', alpha=0.5, linestyle='--', lw=2)
                ax2.legend(frameon=False)

                ax3.set_xlabel('d{} {}'.format(ref, pb))
                ax3.set_ylabel('d{} {}'.format(label, pb))
                xmin, xmax = ax3.get_xlim()
                ax3.plot([xmin, xmax], [xmin, xmax], color='grey', alpha=0.5, linestyle='--', lw=2)
                ax3.legend(frameon=False)
        #endfor pb
        label = '{} vs {}'.format('/'.join(comp), ref)
        fig1.text(0.75, 0.25, label, ha='center', va='bottom', fontsize='medium')
        fig2.text(0.75, 0.25, label +' w/ 5-sig clip', ha='center',va='bottom', fontsize='medium')
        fig3.text(0.75, 0.25, label+' errors', ha='center', va='bottom', fontsize='medium')
    #endfor method
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig1.savefig('{}_vs_{}.pdf'.format('+'.join(comp), ref))
    fig2.savefig('{}+sigclip_vs_{}.pdf'.format('+'.join(comp), ref))
    fig3.savefig('{}_vs_{}_errors.pdf'.format('+'.join(comp), ref))


if __name__=='__main__':
    sys.exit(main())
