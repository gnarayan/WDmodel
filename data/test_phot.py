#!/usr/bin/env python
import sys
import os
import numpy as np
import pysynphot as S
import WDmodel.io
from matplotlib.mlab import rec2txt

def synflux(spec, pb):
    ind   = np.where((spec.wave >= pb.wave.min()) & (spec.wave <= pb.wave.max()))
    transmission = np.interp(spec.wave[ind], pb.wave, pb.transmission, left=0., right=0.)
    n = np.trapz(spec.flux[ind]*spec.wave[ind]*transmission, spec.wave[ind])
    d = np.trapz(spec.wave[ind]*transmission, spec.wave[ind])
    out = n/d
    return out

def synphot(spec, pb, zp=0.):
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
    pbnames = 'F275W,F336W,F475W,F625W,F775W,F160W'
    pbnames = pbnames.split(',')
    pbs = WDmodel.io.get_pbmodel(pbnames)
    stars   = 'gd71,gd153,g191b2b'
    stars   = stars.split(',')
    ext     = '_mod_010.fits'
    indir   = os.environ.get('PYSYN_CDBS','/data/wdcalib/synphot')
    indir   = os.path.join(indir, 'calspec')
    mag_type= 'vegamag'
    vega    = S.Vega

    mag_vega     = []
    cutpbs       = {}
    for pb in pbnames:
        thispb = pbs[pb]
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
            ob = S.Observation(spec, this_syn_pb)
            this_syn_mag = ob.effstim(mag_type)
            synphot_mag.append(this_syn_mag)

            this_simp_mag = synphot(this_spec, this_simp_pb, zp=this_zp)
            simp_mag.append(this_simp_mag)
        print specfile

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
    print rec2txt(out, precision=5)


if __name__=='__main__':
    sys.exit(main())