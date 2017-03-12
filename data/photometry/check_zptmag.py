#!/usr/bin/env python
"""
This script checks the zeropoints reported by A. Calamida by taking the synphot
model of each of the three primary standards, subtracing the instrumental
photometry and comparing to the zeropoints she reports. Comparison of the
reported zeropoints directly with MAST shows excellent agreements, but the
reported uncertainties are also of interest. 
"""
import sys
import os
import numpy as np
import pysynphot as S
from matplotlib.mlab import rec2txt
from uncertainties import ufloat

def main():
    # setup the passbands
    pbnames = 'F275W,F336W,F475W,F625W,F775W,F160W'
    pbnames = pbnames.split(',')
    mag_type= 'vegamag'
    map = np.recfromtxt('../../WDmodel/WDmodel_pb_obsmode_map.txt',names=True)
    map = dict(zip(map.pb, map.obsmode))

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
    photfile= 'WD_cycle22.phot'
    a = np.recfromtxt(photfile, names=True)
    ind_zp = (a.ID == 'ZP')
    zp = a[ind_zp][0]
    d = a[1:]
    objid = np.array([x.lower().replace('-','') for x in d.ID])

    outzp = []
    reszp = []
    for star in stars:
        specfile = os.path.join(indir, '{}{}'.format(star,ext))
        spec = S.FileSpectrum(specfile)

        mask = (objid == star)

        this_out = [star,]
        this_res = ['res_'+star,]
        for pb in pbnames:
            thispb = pbs[pb]

            # get the synthetic mag of this star in this passband
            ob = S.Observation(spec, thispb)
            synmag  = ob.effstim(mag_type)

            # get the instrumental mag of this star in this passband
            instmag = d[mask][pb][0]
            instmag_err = d[mask]['d'+pb][0]

            # save the zeropoint of this star in this passband
            zptmag  = synmag - instmag
            this_out.append(zptmag)
            this_out.append(instmag_err)

            # get Annalisa's zeropoint and save the difference
            azptmag = zp[pb]
            this_res.append(zptmag - azptmag)
            this_res.append(instmag_err)
        outzp.append(this_out)
        reszp.append(this_res)

    # setup column names
    names = ['obj']
    for pb in pbnames:
        names.append(pb)
        names.append('d'+pb)

    xzp = np.rec.fromrecords(outzp, names=names)

    out = []
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

    final = []
    for rec in outzp:
        final.append(rec)
    final.append(origmean)
    for rec in reszp:
        final.append(rec)
    final.append(outmean)
    final = np.rec.fromrecords(final, names=names)
    print rec2txt(final, precision=6)
    with open('zeropoint_check.txt','w') as f:
        f.write(rec2txt(final)+'\n')


if __name__=='__main__':
    sys.exit(main())
