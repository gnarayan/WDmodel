#!/usr/bin/env python
"""
This script checks the zeropoints reported by A. Calamida by taking the synphot
model of each of the three primary standards, subtracting the instrumental
photometry and comparing to the zeropoints she reports. Comparison of the
reported zeropoints directly with MAST shows excellent agreements, but the
reported uncertainties are also of interest.

We also take the weighted mean of the zeropoints of the three primary
standards, compute the apparent magnitudes, the full covariance matrix of the
observations, and plots the correlation matrix. Writes out the apparent
magnitudes as a simple text file, as well as all the data as a HDF5 file.

This work supersedes dump_apparent_mags.py, which just applies A. Calamida's
reported zeropoint to the instrumental magnitudes. That approach does not let
us preserve the covariance between the primary standards.
"""
import sys
import os
import numpy as np
import astropy.table as at
import pysynphot as S
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.mlab import rec2txt
import uncertainties as u
from uncertainties import unumpy
import h5py

def main():
    # setup the passbands
    pbnames = 'F275W,F336W,F475W,F625W,F775W,F160W'
    pbnames = pbnames.split(',')
    mag_type= 'vegamag'
    map = at.Table.read('../../../WDmodel/WDmodel_pb_obsmode_map.txt', format='ascii.commented_header')
    map = dict(zip(map['pb'], map['obsmode']))

    # load the passbands
    pbs = {}
    for pb in pbnames:
        thisobsmode = map[pb]
        thispb = S.ObsBandpass(thisobsmode)
        pbs[pb] = thispb

    # setup the three primary standards
    stars   = 'gd71,gd153,g191b2b'
    stars   = stars.split(',')
    ext     = '_mod_010.fits'
    indir   = os.environ.get('PYSYN_CDBS','/data/wdcalib/synphot')
    indir   = os.path.join(indir, 'calspec')

    # setup the instrumental photometry
    a = at.Table.read('Combined/WD_cycle22_cycle20new.phot', format='ascii.commented_header')
    b = at.Table.read('Combined/ZP_all', format='ascii.commented_header')
    b['l'] *= 1000
    zppb = ['F{:.3}W'.format(str(x)) for x in b['l']]
    zp = {x:y for x, y in zip(zppb, b['ZP_10'])}
    zp.update({'d'+x:y for x, y in zip(zppb, b['eZP_10'])})

    d = a[0:]
    objects = np.array([x.lower().replace('-','') for x in d['ID']])

    # load the name mapping
    map = None
    with open('Combined/name_map.dat','r') as f:
        lines = f.readlines()
        map = {line.strip().split()[0]:line.strip().split()[1] for line in lines}

    outzp = []
    reszp = []

    zeropoint_data = {}
    zpmask = {}

    for star in stars:
        # load the spectrum of this standard
        specfile = os.path.join(indir, '{}{}'.format(star,ext))
        spec = S.FileSpectrum(specfile)

        # load the photometry of this star
        mask = np.where(objects == star)[0]

        # setup some output arrays
        this_out = [star,]
        this_res = ['res_'+star,]
        zeropoint_data[star] = []
        zpmask[star] = mask[0] # save the indices of the original array that correspond to the star

        for pb in pbnames:
            thispb = pbs[pb]

            # get the synthetic mag of this star in this passband
            ob = S.Observation(spec, thispb)
            synmag  = ob.effstim(mag_type)

            # get the instrumental mag of this star in this passband
            instmag = d[mask][pb][0]
            instmag_err = d[mask]['d'+pb+'1'][0]

            # save the zeropoint of this star in this passband
            zptmag  = synmag - instmag
            this_out.append(zptmag)
            this_out.append(instmag_err)

            imag = u.ufloat(instmag, instmag_err)
            zeropoint_data[star].append((synmag, imag, instmag_err))

            # get Annalisa's zeropoint and save the difference
            azptmag = zp[pb]
            this_res.append(zptmag - azptmag)
            this_res.append(instmag_err)
        outzp.append(this_out)
        reszp.append(this_res)

    # setup column names
    names = ['obj']
    formats = {}
    for pb in pbnames:
        names.append(pb)
        names.append('d'+pb)
        formats[pb] = '%7.5f'
        formats['d'+pb] = '%6.5f'

    # this structure holds the zeropoints derived from each of the three primary standards
    xzp = np.rec.fromrecords(outzp, names=names)

    # calculate weighted average zeropoints
    outmean = ['Weighted mean',]
    origmean = ['Annalisa ZP',]
    for pb in pbnames:
        vals = xzp[pb]
        errs = xzp['d'+pb]
        weights = 1./errs**2.
        meanzp, wsum = np.average(vals, weights=weights, returned=True)
        mean_err = (1./wsum)**0.5

        outmean.append(meanzp)
        outmean.append(mean_err)
        origmean.append(zp[pb])
        origmean.append(zp['d'+pb])

    # pretty print the data
    final = []
    # zeropoints from each star
    for rec in outzp:
        final.append(rec)
    # Annalisa's zeropoint
    final.append(origmean)
    # residuals w.r.t. Annalisa's zeropoint
    for rec in reszp:
        final.append(rec)
    # weighted average zeropoint
    final.append(outmean)

    final = at.Table(rows=final, names=names)
    print(final)

    # write the zeropoint data to a file
    final.write('zeropoint_check.txt', format='ascii.fixed_width', delimiter='  ', bookend=False, formats=formats)

    # this structure will hold the apparent magnitudes + zeropoints in each of the bands with the errors
    outdata = []

    # map the objnames in the photometry file into the spectrum names
    objnames = [objid.replace('-','',1).split('.')[0].split('+')[0].split('-')[0].replace('SDSS','sdss').lower() for objid in d['ID']]
    objnames = [map.get(objid, objid) for objid in objnames]
    objnames = np.array(objnames+['zptmag'])
    outdata.append(objnames)

    outf = h5py.File('WDphot_C22_AC_AS_combined_GNzpt.hdf5','w')
    objnames = [str(x).encode('ascii') for x in objnames]
    dt = h5py.special_dtype(vlen=str)
    outf.create_dataset('objid',data=objnames,dtype=dt)

    # propagate the zeropoint errors and write out apparent mags and covariance matrices per passband
    nobj = len(objnames)-1
    allind = np.arange(nobj)
    for i, pb in enumerate(pbnames):
        pbzptdata = [zeropoint_data[star][i] for star in stars]

        # restore the data for the three primary standards
        pbsynmag, pbinsmag, pbinsmag_err  = zip(*pbzptdata)

        # recompute the weighted average zeropoint in a way that preserves the covariance
        pbsynmag = np.array(pbsynmag)
        pbinsmag = np.array(pbinsmag)
        pbinsmag_err = np.array(pbinsmag_err)
        w = 1./pbinsmag_err**2.
        pbzpt = np.sum(w*(pbsynmag - pbinsmag))/np.sum(w)

        instmag = d[pb]
        instmag_err = d['d'+pb+'1']

        # make uncertainty arrays of the instrumental magnitudes
        instdata = unumpy.uarray(instmag, instmag_err)
        appmag = instdata + pbzpt

        # the last step breaks the covariance of the zeropoint and the three primary standards - restore them
        pbmask = [zpmask[star] for star in stars]
        pbmask = np.array(pbmask)
        appmag[pbmask] = pbinsmag + pbzpt

        # create a set of indices to separate the primary standards and the rest of the white dwarfs
        # this is for aesthetic purposes
        rest_ind = np.array(list(set(allind) - set(pbmask)))

        # save the magnitudes and errors in a struct to make a recarray
        vals = np.hstack((unumpy.nominal_values(appmag), pbzpt.n))
        errs = np.hstack((unumpy.std_devs(appmag), pbzpt.std_dev))
        outdata.append(vals)
        outdata.append(errs)

        # save the values, errors
        outf.create_dataset(pb,data=vals)
        outf.create_dataset('d'+pb, data=errs)

        # create a vector to compute the covariance matrix
        cov_vector = np.hstack((appmag, pbzpt))
        cov_mat = u.covariance_matrix(cov_vector)
        corr_mat = u.correlation_matrix(cov_vector)
        outf.create_dataset('cov_'+pb, data=cov_mat)
        outf.create_dataset('corr_'+pb, data=corr_mat)

        # create a figure of the correlation matrix
        fig = plt.figure(figsize=(14,14))
        gs1 = gridspec.GridSpec(1, 1)
        ax  = fig.add_subplot(gs1[0])

        # rearrange the entries in the correlation matrix plot for aesthetic purposes
        corr_mat = u.correlation_matrix(np.hstack((appmag[rest_ind],pbzpt, appmag[pbmask])))
        cax = ax.matshow(corr_mat)

        # plot the correlation matrix
        corr_mat_ind = np.arange(corr_mat.shape[0])
        x, y = np.meshgrid(corr_mat_ind, corr_mat_ind)
        for i, (x_val, y_val) in enumerate(zip(x.flatten(), y.flatten())):
            if x_val > y_val:
                continue
            c = '{:+.1g}'.format(corr_mat[x_val, y_val])
            if corr_mat[x_val, y_val] <= 1E-14:
                color='white'
            else:
                color='black'
            ax.text(x_val, y_val, c, va='center', ha='center', fontsize=5, color=color)

        # set the ticklabels
        ax.set_xticks(np.arange(nobj+1))
        ax.set_yticks(np.arange(nobj+1))
        objnames = np.array(objnames, dtype=np.str_)
        ticknames = objnames[rest_ind].tolist() + ['zptmag_{}'.format(pb)] + objnames[pbmask].tolist()
        ax.set_xticklabels(ticknames, rotation=90, fontsize=8)
        ax.set_yticklabels(ticknames, fontsize=8)

        # add a colorbar and title
        fig.colorbar(cax)
        fig.suptitle("{} Correlation Matrix".format(pb))

        # save the figure
        fig.savefig("figures/{}_corrmat.pdf".format(pb))
    # close the hdf5 file
    outf.close()

    # print out the apparent magnitudes
    outdata = at.Table(outdata, names=names)
    print(outdata)

    # and save the apparent mags as text
    outdata.write('WDphot_C22_AC_AS_combined_GNzpt.dat',format='ascii.fixed_width',delimiter='  ', bookend=False, formats=formats)







if __name__=='__main__':
    sys.exit(main())
