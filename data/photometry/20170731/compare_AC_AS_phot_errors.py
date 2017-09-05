#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple comparison of photometry, errors between AC and AS
"""
import sys
import warnings
import os
import glob
import astropy.table as at
from astropy.visualization import hist
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


def main():
    pbs = 'F275W,F336W,F475W,F625W,F775W'
    colors='purple,blue,green,red,orange'

    pbs = pbs.split(',')
    colors = colors.split(',')
    col_pbs = dict(zip(pbs, colors))

    # get a list of AC's files
    ac_dir = os.path.join('AC','single_epoch','phot_14secondary')
    ac_file_pattern = '*phot15'
    ac_full_pattern = os.path.join(ac_dir, ac_file_pattern)
    ac_files = glob.glob(ac_full_pattern)

    # setup arrays to hold:
    # AC's data
    ac_mag = []
    ac_err = []
    ac_pb  = []
    ac_jd  = []

    # AS's data
    as_mag = []
    as_err = []
    as_pb  = []

    # the difference in the data
    delta_mag     = []
    delta_ac_mag  = []
    delta_as_mag  = []
    delta_ac_err  = []
    delta_as_err  = []
    delta_pb      = []
    delta_ac_file = []
    delta_as_file = []

    # since the AC and AS records do not match exactly, keep summary of number of different observations
    record_comp = []
    nmatch = 0

    for ac_file in ac_files:
        # get the object and passband of interest
        ac_basef = os.path.basename(ac_file)
        obj, pb = ac_basef.replace('.phot15','').split('_')

        # get AS's file corresponding to the AC file
        as_file = os.path.join('AS',pb,'{}_{}.abi'.format(obj,pb))
        if not os.path.exists(as_file):
            message = 'Huh. File {} not found'.format(as_file)
            warnings.warn(message, RuntimeWarning)
            continue
        as_basef = os.path.basename(as_file)

        # read both sets of data
        ac_data = at.Table.read(ac_file, format='ascii')
        as_data = at.Table.read(as_file, format='ascii.commented_header')

        ac_npb = len(ac_data)
        as_npb = len(as_data)

        # match the AC and AS data - we cannot match on exact image since JD isn't common
        if ac_npb == as_npb:
            nmatch += 1
            delta_m = ac_data['M(15)'] - as_data['FMAG'] + 30.
            delta_ac_mag += ac_data['M(15)'].tolist()
            delta_ac_err += ac_data['dM'].tolist()
            delta_as_mag += (as_data['FMAG'] - 30.).tolist()
            delta_as_err += as_data['ERRMAG'].tolist()
            delta_pb += [pb] * ac_npb
            delta_ac_file += [ac_file] * ac_npb
            delta_as_file += [as_file] * as_npb
        else:
            # if the number of observations do not match, take the first n matching observations
            if ac_npb > as_npb:
                nt = as_npb 
                delta_m = ac_data['M(15)'][0:nt] - as_data['FMAG'] + 30.
                delta_ac_mag += ac_data['M(15)'][0:nt].tolist()
                delta_as_mag += (as_data['FMAG'] - 30.).tolist()
                delta_ac_err += ac_data['dM'][0:nt].tolist()
                delta_as_err += as_data['ERRMAG'].tolist()
            else:
                nt = ac_npb 
                delta_m = ac_data['M(15)'] - as_data['FMAG'][0:nt] + 30.
                delta_ac_mag += ac_data['M(15)'].tolist()
                delta_as_mag += (as_data['FMAG'] - 30.)[0:nt].tolist()
                delta_ac_err += ac_data['dM'].tolist()
                delta_as_err += as_data['ERRMAG'][0:nt].tolist()
            delta_pb += [pb] * nt
            delta_ac_file += [ac_file]*nt
            delta_as_file += [as_file]*nt
        delta_mag += delta_m.tolist()

        # save the number of records in both AC and AS files
        record_comp.append((ac_file, ac_npb, as_npb, as_file))

        t   = ac_data['t'].tolist()
        m1  = ac_data['M(15)'].tolist()
        dm1 = ac_data['dM'].tolist()
        p1  = [pb] * ac_npb

        m2  = (as_data['FMAG'] -30.).tolist()
        dm2 = as_data['ERRMAG'].tolist()
        p2  = [pb] * as_npb

        # save the AC data
        ac_mag += m1
        ac_err += dm1
        ac_pb  += p1

        ac_jd  += t

        # save the AS data
        as_mag += m2 
        as_err += dm2 
        as_pb  += p2 

    # save the record comparison
    record_comp = at.Table(rows=record_comp, names=('ac_file','ac_nobs','as_nobs','as_file'))
    record_comp.write('AC_AS_record_comparison.txt',format='ascii.fixed_width',overwrite=True)

    # summary message
    message = "N_Match/N_Total: {:d}/{:d} = {:.2n}%".format(nmatch, len(record_comp), nmatch*100./len(record_comp))
    print(message)

    # write out AC's data
    ac_comb = at.Table([ac_mag, ac_err, ac_pb], names=('m','dm','pb'))
    ac_comb.write('AC_records.txt', format='ascii.fixed_width', overwrite=True)

    # write out AS's data
    as_comb = at.Table([as_mag, as_err, as_pb], names=('m','dm','pb'))
    as_comb.write('AS_records.txt', format='ascii.fixed_width', overwrite=True)

    # write out the differences between the AC and AS data
    delta_comb = at.Table([delta_mag, delta_ac_mag, delta_ac_err, delta_as_mag,\
            delta_as_err, delta_pb, delta_ac_file, delta_as_file],\
            names=('deltaM','ac_m','ac_dm','as_m','as_dm','pb','ac_file','as_file'))
    delta_comb.write('AC_AS_record_differences.txt', format='ascii.fixed_width',overwrite=True)

    # create some blank canvases
    fig1  = plt.figure(figsize=(10,15))    
    fig2  = plt.figure(figsize=(10,15))    
    fig2b = plt.figure(figsize=(10,15))    
    fig3  = plt.figure(figsize=(10,15))    
    fig4  = plt.figure(figsize=(10,15))    
    fig5  = plt.figure(figsize=(10,15))    
	
    # make plot comparisons
    for i, pb in enumerate(pbs):

        ax1 = fig1.add_subplot(3,2,i+1)
        ax2 = fig2.add_subplot(3,2,i+1)
        ax2b= fig2b.add_subplot(3,2,i+1)
        ax3 = fig3.add_subplot(3,2,i+1)
        ax4 = fig4.add_subplot(3,2,i+1)
        ax5 = fig5.add_subplot(3,2,i+1)

        # split up the data by passband
        mask = (delta_comb['pb'] == pb)
        d = delta_comb[mask]

        # mag-mag plot
        ax1.errorbar(d['ac_m'], d['as_m'], xerr=d['ac_dm'], yerr=d['as_dm'],\
                marker='o', linestyle='None', color=col_pbs[pb], label=pb)
        xmin, xmax = ax1.get_xlim()
        ax1.plot([xmin, xmax], [xmin, xmax], linestyle='--', color='black', alpha=0.5)
        ax1.set_xlabel('AC mag')
        ax1.set_ylabel('AS mag')
        ax1.legend(frameon=False)

        # mag-difference plot
        ax2.errorbar(d['ac_m'], d['deltaM'], xerr=d['ac_dm'], yerr=d['as_dm'],\
                marker='o', linestyle='None', color=col_pbs[pb], label=pb)
        xmin, xmax = ax2.get_xlim()
        ax2.plot([xmin, xmax], [0., 0.], linestyle='--', color='black', alpha=0.5)
        ax2.set_ylim(-0.075, 0.075)
        ax2.set_xlabel('AC mag')
        ax2.set_ylabel('Delta mag')
        ax2.legend(frameon=True)

        # error-error plot
        ax3.plot(d['ac_dm'], d['as_dm'], marker='o', linestyle='None', color=col_pbs[pb], label=pb)
        xmin, xmax = ax3.get_xlim()
        ax3.plot([xmin, xmax], [xmin, xmax], linestyle='--', color='black', alpha=0.5)
        ax3.set_xlabel('AC err')
        ax3.set_ylabel('AS err')
        ax3.legend(frameon=False)

        # error histograms
        _, bins, patches = hist(d['ac_dm'], histtype='stepfilled', bins='knuth', ax=ax4, facecolor=col_pbs[pb])
        hist(d['as_dm'], histtype='stepfilled', bins=bins, ax=ax4, facecolor=col_pbs[pb], alpha=0.5)
        ax4.set_xlabel('{} Err'.format(pb))
        ax4.legend(frameon=False)

        # mag-err plots
        ax5.plot(d['ac_m'], d['ac_dm'], marker='o', color=col_pbs[pb], linestyle='None')
        ax5.plot(d['as_m'], d['as_dm'], marker='*', color=col_pbs[pb], alpha=0.5, linestyle='None')
        ax5.set_xlabel('{} Mag'.format(pb))
        ax5.set_ylabel('{} Err'.format(pb))
        ax5.legend(frameon=False)

        # trimmed mag differences histogram
        cut = ((d['deltaM'] > -0.075) & (d['deltaM'] < 0.075))
	hist(d['deltaM'][cut], histtype='stepfilled', bins='knuth', facecolor=col_pbs[pb], label=pb, ax=ax2b)
        ax2b.axvline(0., linestyle='--', color='black', alpha=0.5)
	ax2b.set_xlabel('{} Residual'.format(pb))

    with PdfPages('AC_AS_comparison.pdf') as pdf:
        fig1.suptitle('Mag Mag comparison')
        pdf.attach_note('Mag Mag comparison')
        pdf.savefig(fig1)

        fig2.suptitle('Mag DeltaMag comparison')
        pdf.attach_note('Mag DeltaMag comparison')
        pdf.savefig(fig2)

        fig2b.suptitle('DeltaMag histogram')
        pdf.attach_note('DeltaMag histogram')
        pdf.savefig(fig2b)

        fig3.suptitle('Err Err comparison')
        pdf.attach_note('Err Err comparison')
        pdf.savefig(fig3)

        fig4.suptitle('Err histograms')
        pdf.attach_note('Err histograms')
        pdf.savefig(fig4)

        fig5.suptitle('Mag Err comparison')
        pdf.attach_note('Mag Err comparison')
        pdf.savefig(fig5)



if __name__=='__main__':
    sys.exit(main())

