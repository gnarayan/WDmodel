import numpy as np
import pysynphot as S
from . import io
from collections import OrderedDict

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


def get_model_synmags(model_spec, pbs, mu=0.):
    """
    Computes the synthetic magnitudes of SED spec through the passbands in
    dictionary pbmodel, and optionally applies an offset, mu

    Accepts
        model_spec: model recarray spectrum (wave, flux) - assumed to be Flam
        pbs: dictionary with pbname as key containing as a tuple:
            passband: recarray of non-zero passband transmission (wave,
            throughput)
            transmission: the non-zero passband transmission interpolated onto
            overlapping model wavelengths
            ind: indices of model_spec wavelength that overlap with this passband
            zp: zeropoint of this passband
            (this structure can be created by get_pbmodel)
        mu: offset to apply to the photometry (default=0)
    Returns
        recarray of synthetic magnitudes (pb, mag)
    """
    outpb  = []
    outmag = []
    for pbname, pbdata in pbs.items():
        _, transmission, ind, zp = pbdata
        pbsynmag = synphot(model_spec, ind, transmission, zp) + mu
        outmag.append(pbsynmag)
        outpb.append(pbname)
    out = np.rec.fromarrays((outpb, outmag), names='pb,mag')
    return out


def interp_passband(wave, pb, model):
    """
    Figures out the indices of the wavelength array wave, that overlap with the
    recarray passband pb (wave, throughput), interpolates the passband onto
    overlapping wavelength array, padding out of range values with zero, and
    returns the interpolated transmission, and the indices of the wavelength
    array that overlap with the passband
    """
    _, ind = model._get_indices_in_range(wave, pb.wave.min(), pb.wave.max())
    transmission = np.interp(wave[ind], pb.wave, pb.throughput, left=0., right=0.)
    return transmission, ind


def chop_syn_spec_pb(spec, model_mag, pb, model):
    """
    Trims the pysynphot bandpass pb to non-zero throughput,

    Computes the zeropoint of the passband given the SED spec, and model
    magnitude of spec in the passband

    Returns the trimmed passband and its zeropoint
    """

    # don't bother about zero padding
    mask = np.nonzero(pb.throughput)
    pwave = pb.wave[mask]
    ptput = pb.throughput[mask]
    outpb = np.rec.fromarrays((pwave, ptput), names='wave,throughput')

    # interpolate the transmission onto the overlapping wavelengths of the spectrum
    transmission, ind = interp_passband(spec.wave, outpb, model)

    # compute the zeropoint
    outzp = model_mag - synphot(spec, ind, transmission)

    return outpb, outzp


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

    out = OrderedDict()

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
        outpb, outzp = chop_syn_spec_pb(vega, synphot_mag, bp, model)

        # interpolate the passband onto the standard's  wavelengths
        transmission, ind = interp_passband(model._wave, outpb, model)

        # save everything we need for this passband
        out[pb] = (outpb, transmission, ind, outzp)

    return out
