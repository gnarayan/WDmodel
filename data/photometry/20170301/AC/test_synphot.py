#!/usr/bin/env python
"""
This script tests the consistency between pysynphot effstim, and simply doing
trapezoidal rule with computed zeropoints for the three primary standards -
synphot.  Differences are < 1E-5. trapz is an order of magnitude+ faster, and
the we don't run into pickling issues with pysynphot by using it.
"""
import sys
import os
import astropy.table as at
import numpy as np
import pysynphot as S
from matplotlib.mlab import rec2txt

def synflux(spec, pb):
    """
    Trim the spectrum to the passband, interpolate the transmission onto the
    spectrum wavelengths, and return the synthetic flux of the spectrum
    """
    ind   = np.where((spec.wave >= pb.wave.min()) & (spec.wave <= pb.wave.max()))
    transmission = np.interp(spec.wave[ind], pb.wave, pb.transmission, left=0., right=0.)
    n = np.trapz(spec.flux[ind]*spec.wave[ind]*transmission, spec.wave[ind])
    d = np.trapz(spec.wave[ind]*transmission, spec.wave[ind])
    out = n/d
    return out

def synphot(spec, pb, zp=0.):
    """
    Get the synthetic magnitude of spectrum spec through passband pb, and apply zeropoint zp
    """
    flux = synflux(spec, pb)
    m = -2.5*np.log10(flux) + zp
    return m

def chop_syn_spec_pb(spec, pb):

    # don't bother about zero padding
    mask = np.nonzero(pb.throughput)
    pwave = pb.wave[mask]
    ptput = pb.throughput[mask]

    # trim the spectrum
    ind   = np.where((spec.wave >= pwave.min()) & (spec.wave <= pwave.max()))
    swave = spec.wave[ind]
    sflux = spec.flux[ind]

    # return pb, spectrum
    outpb = np.rec.fromarrays((pwave, ptput), names='wave,transmission')
    outspec = np.rec.fromarrays((swave, sflux), names='wave,flux')
    return outpb, outspec


def main():
    # setup the three primary standards
    stars   = 'gd71,gd153,g191b2b'
    stars   = stars.split(',')
    ext     = '_mod_010.fits'
    indir   = os.environ.get('PYSYN_CDBS','/data/wdcalib/synphot')
    indir   = os.path.join(indir, 'calspec')

    # setup the magnitude system and throughput model
    mag_type= 'vegamag'
    vega    = S.Vega
    pbnames = 'F275W,F336W,F475W,F625W,F775W,F160W'
    pbnames = pbnames.split(',')
    map = at.Table.read('../../../../WDmodel/WDmodel_pb_obsmode_map.txt', format='ascii.commented_header',\
            delimiter=' ',names=('pb','obsmode'))
    map = dict(zip(map['pb'], map['obsmode']))

    mag_vega     = []
    cutpbs       = {}
    pbs          = {}

    # load the passbands
    for pb in pbnames:
        thisobsmode = map[pb]
        thispb = S.ObsBandpass(thisobsmode)
        pbs[pb] = thispb

        ob = S.Observation(vega, thispb)

        # get the pysynphot Vega magnitude (should be 0.)
        synphot_mag = ob.effstim(mag_type)
        mag_vega.append(synphot_mag)

        # cut the passband to non-zero values and interpolate onto overlapping Vega wavelengths
        cutpb, cutspec = chop_syn_spec_pb(vega, thispb)

        # set the zeropoint for the cut passband
        thiszp = synphot_mag - synphot(cutspec, cutpb)

        # save the passband and computed zeropoint
        cutpbs[pb] = (cutpb, thiszp)

    out = []
    for star in stars:
        specfile = os.path.join(indir, '{}{}'.format(star,ext))
        spec = S.FileSpectrum(specfile)
        synphot_mag = []
        simp_mag    = []
        for pb in pbnames:
            this_syn_pb  =  pbs[pb]
            this_simp_pb, this_zp = cutpbs[pb]
            this_spec = np.rec.fromarrays((spec.wave, spec.flux), names='wave,flux')

            # for each star, we get the pysynphot magnitude
            ob = S.Observation(spec, this_syn_pb)
            this_syn_mag = ob.effstim(mag_type)
            synphot_mag.append(this_syn_mag)

            # and the simple synphot magnitude, using the zeropoint computed from Vega
            this_simp_mag = synphot(this_spec, this_simp_pb, zp=this_zp)
            simp_mag.append(this_simp_mag)

        # print out the magnitudes from pysynphot, simple trapz synphot, and the residuals
        synphot_mag = np.array(synphot_mag)
        simp_mag    = np.array(simp_mag)
        res         = synphot_mag - simp_mag
        this_out = ['{}_{}'.format(star,'synphot'),] + synphot_mag.tolist()
        out.append(this_out)
        this_out = ['{}_{}'.format(star,'simple'),] + simp_mag.tolist()
        out.append(this_out)
        this_out = ['{}_{}'.format(star,'residual'),] + res.tolist()
        out.append(this_out)
    out = np.rec.fromrecords(out, names=['objphot',]+pbnames)
    print(rec2txt(out, precision=6))


if __name__=='__main__':
    sys.exit(main())
