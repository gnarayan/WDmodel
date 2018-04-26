# -*- coding: UTF-8 -*-
"""
Instrumental throughput models and calibration and synthetic photometry
routines
"""

from __future__ import absolute_import
from __future__ import unicode_literals
import numpy as np
import pysynphot as S
from . import io
from collections import OrderedDict
from six.moves import zip

def synflux(spec, ind, pb):
    """
    Compute the synthetic flux of spectrum ``spec`` through passband ``pb``

    Parameters
    ----------
    spec : :py:class:`numpy.recarray`
        The spectrum.
        Must have ``dtype=[('wave', '<f8'), ('flux', '<f8')]``
    ind : array-like
        Indices of spectrum ``spec`` that overlap with the passband ``pb``.
        Can be produced by :py:meth:`WDmodel.passband.interp_passband`
    pb : array-like
        The passband transmission.
        Must satisfy ``pb.shape == spec[ind].flux.shape``

    Returns
    -------
    flux : float
        The normalized flux of the spectrum through the passband

    Notes
    -----
        The passband is assumed to be dimensionless photon transmission
        efficiency.

        Routine is intended to be a mch faster implementation of
        :py:meth:`pysynphot.observation.Observation.effstim`, since it is called over and
        over by the samplers as a function of model parameters.

        Uses :py:func:`numpy.trapz` for interpolation.

    See Also
    --------
    :py:func:`WDmodel.passband.interp_passband`
    """
    n = np.trapz(spec.flux[ind]*spec.wave[ind]*pb, spec.wave[ind])
    d = np.trapz(spec.wave[ind]*pb, spec.wave[ind])
    out = n/d
    return out


def synphot(spec, ind, pb, zp=0.):
    """
    Compute the synthetic magnitude of spectrum ``spec`` through passband ``pb``

    Parameters
    ----------
    spec : :py:class:`numpy.recarray`
        The spectrum.
        Must have ``dtype=[('wave', '<f8'), ('flux', '<f8')]``
    ind : array-like
        Indices of spectrum ``spec`` that overlap with the passband ``pb``.
        Can be produced by :py:meth:`WDmodel.passband.interp_passband`
    pb : array-like
        The passband transmission.
        Must satisfy ``pb.shape == spec[ind].flux.shape``
    zp : float, optional
        The zeropoint to apply to the synthetic flux

    Returns
    -------
    mag : float
        The synthetic magnitude of the spectrum through the passband

    See Also
    --------
    :py:func:`WDmodel.passband.synflux`
    :py:func:`WDmodel.passband.interp_passband`
    """
    flux = synflux(spec, ind, pb)
    m = -2.5*np.log10(flux) + zp
    return m


def get_model_synmags(model_spec, pbs, mu=0.):
    """
    Computes the synthetic magnitudes of spectrum ``model_spec`` through the
    passbands ``pbs``, and optionally applies a common offset, ``mu``

    Wrapper around :py:func:`WDmodel.passband.synphot`.

    Parameters
    ----------
    spec : :py:class:`numpy.recarray`
        The spectrum.
        Must have ``dtype=[('wave', '<f8'), ('flux', '<f8')]``
    pbs : dict
        Passband dictionary containing the passbands corresponding to
        phot.pb` and generated by :py:func:`WDmodel.passband.get_pbmodel`.
    mu : float, optional
        Common achromatic photometric offset to apply to the synthetic
        magnitudes in al the passbands. Would be equal to the distance modulus
        if ``model_spec`` were normalized to return the true absolute magnitude
        of the source.

    Returns
    -------
    model_mags : None or :py:class:`numpy.recarray`
        The model magnitudes.
        Has ``dtype=[('pb', 'str'), ('mag', '<f8')]``
    """
    outpb  = []
    outmag = []
    for pbname, pbdata in pbs.items():
        _, transmission, ind, zp, _ = pbdata
        pbsynmag = synphot(model_spec, ind, transmission, zp) + mu
        outmag.append(pbsynmag)
        outpb.append(pbname)
    names=str('pb,mag')
    out = np.rec.fromarrays((outpb, outmag), names=names)
    return out


def interp_passband(wave, pb, model):
    """
    Find the indices of the wavelength array ``wave``, that overlap with the
    passband ``pb`` and interpolates the passband onto the wavelengths.

    Parameters
    ----------
    wave : array-like
        The wavelength array. Must satisfy
        :py:meth:`WDmodel.WDmodel.WDmodel._wave_test`
    pb : :py:class:`numpy.recarray`
        The passband transmission.
        Must have ``dtype=[('wave', '<f8'), ('throughput', '<f8')]``
    model : :py:class:`WDmodel.WDmodel.WDmodel` instance
        The DA White Dwarf SED model generator

    Returns
    -------
    transmission : array-like
        The transmission of the passband interpolated on to overlapping
        elements of ``wave``
    ind : array-like
        Indices of wavelength ``wave`` that overlap with the passband ``pb``.
        Produced by :py:meth:`WDmodel.WDmodel.WDmodel._get_indices_in_range`
        Satisfies ``transmission.shape == wave[ind].shape``

    Notes
    -----
        The passband ``pb`` is interpolated on to the wavelength arrray
        ``wave``. ``wave`` is typically the wavelengths of a spectrum, and have
        much better sampling than passband transmission curves. Only the
        wavelengths ``wave`` that overlap the passband are taken, and the
        passband transmission is then linearly interpolated on to these
        wavelengths. This prescription has been checked against
        :py:mod:`pysynphot` to return synthetic magnitudes that agree to be ``<
        1E-6``, while :py:func:`WDmodel.passband.synphot` is very significantly
        faster than :py:meth:`pysynphot.observation.Observation.effstim`.
    """
    _, ind = model._get_indices_in_range(wave, pb.wave.min(), pb.wave.max())
    transmission = np.interp(wave[ind], pb.wave, pb.throughput, left=0., right=0.)
    return transmission, ind


def chop_syn_spec_pb(spec, model_mag, pb, model):
    """
    Trims the pysynphot bandpass pb to non-zero throughput, computes the
    zeropoint of the passband given the SED spec, and model magnitude of spec
    in the passband

    Parameters
    ----------
    spec : :py:class:`numpy.recarray`
        The spectrum. Typically a standard which has a known ``model_mag``.
        This can be a real source such as Vega, BD+174708, or one of the three
        CALSPEC standards, or an idealized synthetic source such as AB.
        Must have ``dtype=[('wave', '<f8'), ('flux', '<f8')]``
    model_mag : float
        The apparent magnitude of the spectrum through the passband.  The
        difference between the apparent magnitude and the synthetic magnitude
        is the synthetic zeropoint.
    pb : :py:class:`numpy.recarray`
        The passband transmission.
        Must have ``dtype=[('wave', '<f8'), ('throughput', '<f8')]``
    model : :py:class:`WDmodel.WDmodel.WDmodel` instance
        The DA White Dwarf SED model generator

    Returns
    -------
    outpb : :py:class:`numpy.recarray`
        The passband transmission with zero throughput entries trimmed.
        Has ``dtype=[('wave', '<f8'), ('throughput', '<f8')]``
    outzp : float
        The synthetic zeropoint of the passband ``pb`` such that the source
        with spectrum ``spec`` will have apparent magnitude ``model_mag``
        through ``pb``. With the synthetic zeropoint computed, the synthetic
        magnitude of any source can be converted into an apparent magnitude and
        can be passed to :py:func:`WDmodel.passband.synphot`.

    See Also
    --------
    :py:func:`WDmodel.passband.interp_passband`
    :py:func:`WDmodel.passband.synphot`
    """

    # don't bother about zero padding
    mask = np.nonzero(pb.throughput)
    pwave = pb.wave[mask]
    ptput = pb.throughput[mask]
    names = str('wave,throughput')
    outpb = np.rec.fromarrays((pwave, ptput), names=names)

    # interpolate the transmission onto the overlapping wavelengths of the spectrum
    transmission, ind = interp_passband(spec.wave, outpb, model)

    # compute the zeropoint
    outzp = model_mag - synphot(spec, ind, transmission)

    return outpb, outzp


def get_pbmodel(pbnames, model, pbfile=None, mag_type=None):
    """
    Converts passband names ``pbnames`` into passband models based on the
    mapping of name to ``pysynphot`` ``obsmode`` strings in ``pbfile``.

    Parameters
    ----------
    pbnames : array-like
        List of passband names to get throughput models for Each name is
        resolved by first looking in ``pbfile`` (if provided) If an entry is
        found, that entry is treated as an ``obsmode`` for pysynphot. If the
        entry cannot be treated as an ``obsmode,`` we attempt to treat as an
        ASCII file. If neither is possible, an error is raised.
    model : :py:class:`WDmodel.WDmodel.WDmodel` instance
        The DA White Dwarf SED model generator
        All the passbands are interpolated onto the wavelengths of the SED
        model.
    pbfile : str, optional
        Filename containing mapping between ``pbnames`` and ``pysynphot``
        ``obsmode`` string, as well as the standard that has 0 magnitude in the
        system (either ''Vega'' or ''AB''). The ``obsmode`` may also be the
        fullpath to a file that is readable by ``pysynphot``
    mag_type : str, optional
        One of ''vegamag'' or ''abmag''
        Used to specify the standard that has 0 magnitude in the passband.
        If ``magsys`` is specified in ``pbfile,`` that overrides this option.
        Must be the same for all passbands listed in ``pbname`` that do not
        have ``magsys`` specified in ``pbfile``
        If ``pbnames`` require multiple ``mag_types``, concatentate the output.

    Returns
    -------
    out : dict
        Output passband model dictionary. Has passband name ``pb`` from ``pbnames`` as key.

    Raises
    ------
    RuntimeError
        If a bandpass cannot be loaded

    Notes
    -----
        Each item of ``out`` is a tuple with
            * ``pb`` : (:py:class:`numpy.recarray`)
              The passband transmission with zero throughput entries trimmed.
              Has ``dtype=[('wave', '<f8'), ('throughput', '<f8')]``
            * ``transmission`` : (array-like)
              The non-zero passband transmission interpolated onto overlapping model wavelengths
            * ``ind`` : (array-like)
              Indices of model wavelength that overlap with this passband
            * ``zp`` : (float)
              mag_type zeropoint of this passband
            * ``avgwave`` : (float)
              Passband average/reference wavelength

        ``pbfile`` must be readable by :py:func:`WDmodel.io.read_pbmap` and
        must return a :py:class:`numpy.recarray`
        with``dtype=[('pb', 'str'),('obsmode', 'str')]``

        If there is no entry in ``pbfile`` for a passband, then we attempt to
        use the passband name ``pb`` as ``obsmode`` string as is.

        Trims the bandpass to entries with non-zero transmission and determines
        the ``VEGAMAG/ABMAG`` zeropoint for the passband - i.e. ``zp`` that
        gives ``mag_Vega/AB=0.`` in all passbands.

    See Also
    --------
    :py:func:`WDmodel.io.read_pbmap`
    :py:func:`WDmodel.passband.chop_syn_spec_pb`
    """

    # figure out the mapping from passband to observation mode
    if pbfile is None:
        pbfile = 'WDmodel_pb_obsmode_map.txt'
        pbfile = io.get_pkgfile(pbfile)
    pbdata  = io.read_pbmap(pbfile)
    pbmap   = dict(list(zip(pbdata.pb, pbdata.obsmode)))
    sysmap  = dict(list(zip(pbdata.pb, pbdata.magsys)))

    # setup the photometric system by defining the standard and corresponding magnitude system
    if mag_type not in ('vegamag', 'abmag', None):
        message = 'Magnitude system must be one of abmag or vegamag'
        raise RuntimeError(message)

    # define the standards
    vega = S.Vega
    ab   = S.FlatSpectrum(3631, waveunits='angstrom', fluxunits='jy')
    ab.convert('flam')

    # defile the magnitude sysem
    if mag_type is None or mag_type == 'vegamag':
        mag_type= 'vegamag'
    else:
        mag_type = 'abmag'

    out = OrderedDict()

    for pb in pbnames:

        standard = None

        # load each passband
        obsmode = pbmap.get(pb, pb)
        magsys  = sysmap.get(pb, mag_type)

        if magsys == 'vegamag':
            standard = vega
        elif magsys == 'abmag':
            standard = ab
        else:
            message = 'Unknown standard system {} for passband {}'.format(magsys, pb)
            raise RuntimeError(message)

        # treat the passband as a obsmode string
        try:
            bp = S.ObsBandpass(obsmode)
        except ValueError:
            # if that fails, try to load the passband interpreting obsmode as a file
            message = 'Could not load pb {} as an obsmode string {}'.format(pb, obsmode)
            print(message)
            try:
                bp = S.FileBandpass(obsmode)
            except ValueError:
                message = 'Could not load passband {} from obsmode or file {}'.format(pb, obsmode)
                raise RuntimeError(message)

        avgwave = bp.avgwave()

        # get the pysynphot standard magnitude (should be 0. on the standard magnitude system!)
        ob = S.Observation(standard, bp)
        synphot_mag = ob.effstim(magsys)

        # cut the passband to non-zero values and interpolate onto overlapping standard wavelengths
        outpb, outzp = chop_syn_spec_pb(standard, synphot_mag, bp, model)

        # interpolate the passband onto the standard's  wavelengths
        transmission, ind = interp_passband(model._wave, outpb, model)

        # save everything we need for this passband
        out[pb] = (outpb, transmission, ind, outzp, avgwave)
    return out
