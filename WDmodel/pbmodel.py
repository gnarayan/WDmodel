import numpy as np
import pysynphot as S
from . import io

def synflux(spec, ind, pb):
    """
    Compute the synthetic flux of spec through passband pb
    Accepts:
        spec: recarray spectrum (wave, flux) - assumed to be Flam
        ind : indices of spectrum that overlap with the passband pb - assumed
        to be photon transmission
        pb: the passband transmission 

    Returns the normalized flux of the spectrum through the passband
    """
    n = np.trapz(spec.flux[ind]*spec.wave[ind]*pb, spec.wave[ind])
    d = np.trapz(spec.wave[ind]*pb, spec.wave[ind])
    out = n/d
    return out


def synphot(spec, ind, pb, zp=0.):
    """
    Compute the synthetic magnitude of spec through passband pb
    Accepts:
        spec: recarray spectrum (wave, flux) - assumed to be Flam
        ind : indices of spectrum that overlap with the passband pb - assumed
        to be photon transmission
        pb: the passband transmission 
        zp: the zeropoint of the passband that puts the synthetic magnitude on
        some standard system

    Returns the normalized flux of the spectrum through the passband

    """
    flux = synflux(spec, ind, pb)
    m = -2.5*np.log10(flux) + zp
    return m


def chop_syn_spec_pb(spec, pb, model):

    # don't bother about zero padding
    mask = np.nonzero(pb.throughput)
    pwave = pb.wave[mask]
    ptput = pb.throughput[mask]

    # trim the spectrum
    _, ind = model._get_indices_in_range(spec.wave, pwave.min(), pwave.max())
    swave = spec.wave
    sflux = spec.flux

    # return pb, spectrum
    outpb = np.rec.fromarrays((pwave, ptput), names='wave,throughput')
    outspec = np.rec.fromarrays((swave, sflux), names='wave,flux')
    return outpb, outspec, ind


def get_pbmodel(pbnames, model, pbfile=None):
    """
    Accepts
        pbnames: iterable of passband names
        model: A WDmodel instance, so that we can get at the model wavelengths
        pbfile: Optional filename containing a mapping between passband name
        and obsmode string

    Parses the passband names into a synphot/pysynphot obsmode string based on
    pbfile

    pbfile must have columns with
        pb      : passband name
        obsmode : pysynphot observation mode string

    If there is no entry in pbfile for a passband, then we attempt to use the
    passband name as obsmode string as is. Loads the bandpasses corresponding
    to each obsmode.  
    
    Raises RuntimeError if a bandpass cannot be loaded.
    
    Trims the bandpass to entries with non-zero transmission
    Determines the Vegamag zeropoint for the passband (i.e. zp that gives
    Vega=0. in all passbands)
    Interpolates the bandpass onto the overlapping model wavelengths

    Returns a dictionary with pbname as key containing as a tuple:
        passband: recarray of non-zero passband transmission (wave, throughput)
        transmission: the non-zero passband transmission interpolated onto
        overlapping model wavelengths
        ind: indices of model wavelength that overlap with this passband
        zp: Vegamag zeropoint of this passband
    """

    # figure out the mapping from passband to observation mode
    if pbfile is None:
        pbfile = 'WDmodel_pb_obsmode_map.txt'
        pbfile = io.get_pkgfile(pbfile)
    pbdata = io.read_pbmap(pbfile)
    pbmap  = dict(zip(pbdata.pb, pbdata.obsmode))

    # setup the photometric system by defining the standard and corresponding magnitude system
    vega    = S.Vega
    mag_type= 'vegamag'

    out = {}

    for pb in pbnames:

        # load each passband
        obsmode = pbmap.get(pb, pb)
        try:
            bp = S.ObsBandpass(obsmode)
        except ValueError:
            message = 'Could not load passband {} from pysynphot, obsmode {}'.format(pb, obsmode)
            raise RuntimeError(message)

        # get the pysynphot Vega magnitude (should be 0. on the Vega magnitude system!)
        ob = S.Observation(vega, bp)
        synphot_mag = ob.effstim(mag_type)

        # cut the passband to non-zero values and interpolate onto overlapping Vega wavelengths
        cutpb, spec, cutind = chop_syn_spec_pb(vega, bp, model)

        # interpolate the passband onto the standard's  wavelengths
        transmission = np.interp(spec.wave[cutind], cutpb.wave, cutpb.throughput, left=0., right=0.)

        # set the zeropoint for the cut passband
        thiszp = synphot_mag - synphot(spec, cutind, transmission)

        # TODO get the model wavelengths and chop and interpolate transmission and cutind to that
        _, ind = model._get_indices_in_range(model._wave, cutpb.wave.min(), cutpb.wave.max())
        transmission = np.interp(model._wave[ind], cutpb.wave, cutpb.throughput, left=0., right=0.)

        # save everything we need for this passband
        out[pb] = (cutpb, transmission, ind, thiszp)
    return out
