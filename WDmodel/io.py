# -*- coding: UTF-8 -*-
"""
I/O methods. All the submodules of the WDmodel package use this module for
almost all I/O operations.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import sys
import os
from emcee.utils import MPIPool
import argparse
import warnings
from copy import deepcopy
import numpy as np
import pkg_resources
from collections import OrderedDict
import astropy.table as at
import json
import h5py
from six.moves import range

# Declare this tuple to init the likelihood model, and to preserve order of parameters
_PARAMETER_NAMES = ("teff", "logg", "av", "rv", "dl", "fwhm", "fsig", "tau", "fw", "mu")


def get_options(args, comm):
    """
    Get command line options for the :py:mod:`WDmodel` fitter package

    Parameters
    ----------
    args : array-like
        list of the input command line arguments, typically from
        :py:data:`sys.argv`
    comm : None or :py:class:`mpi4py.mpi.MPI` instance
        Used to communicate options to all child processes if running with mpi

    Returns
    -------
    args : Namespace
        Parsed command line options
    pool : None or :py:class`emcee.utils.MPIPool`
        If running with MPI, the pool object is used to distribute the
        computations among the child process

    Raises
    ------
    ValueError
        If any input value is invalid
    """

    # create a config parser that will take a single option - param file
    conf_parser = argparse.ArgumentParser(add_help=False)

    # config options - this lets you specify a parameter configuration file,
    # set the default parameters values from it, and override them later as needed
    # if not supplied, it'll use the default parameter file included in the package
    conf_parser.add_argument("--param_file", required=False, default=None,\
            help="Specify parameter config JSON file")

    args, remaining_argv = conf_parser.parse_known_args(args)
    params = read_params(param_file=args.param_file)

    # now that we've gotten the param_file and the params (either custom, or default), create the parse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,\
                    parents=[conf_parser],\
                    description=__doc__,\
                    epilog="If running fit_WDmodel.py with MPI using mpirun, -np must be at least 2.")

    # create a couple of custom types to use with the parser
    # this type exists to make a quasi bool type instead of store_false/store_true
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    # this type exists for parameters where we can't or don't really want the
    # user to guess a default value - better to make a guess internally than
    # have a bad  starting point
    def NoneOrFloat(v):
        if v.lower() in ("none", "null", "nan"):
            return None
        else:
            return float(v)

    parser.register('type','bool',str2bool)
    parser.register('type','NoneOrFloat',NoneOrFloat)

    # multiprocessing options
    parallel = parser.add_argument_group('parallel', 'Parallel processing options')
    mproc = parallel.add_mutually_exclusive_group()
    mproc.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")
    mproc.add_argument("--mpil", dest="mpil", default=False,
                       action="store_true", help="Run with MPI and enable loadbalancing.")

    # spectrum options
    spectrum = parser.add_argument_group('spectrum', 'Spectrum options')

    spectrum.add_argument('--specfile', required=True, \
            help="Specify spectrum to fit")
    spectrum.add_argument('--spectable', required=False,  default="data/spectroscopy/spectable_resolution.dat",\
            help="Specify file containing a fwhm lookup table for specfile")
    spectrum.add_argument('--lamshift', required=False, type='NoneOrFloat', default=None,\
            help="Specify a flat wavelength shift in Angstrom to fix  slit centering errors")
    spectrum.add_argument('--vel', required=False, type=float, default=0.,\
            help="Specify a velocity shift in kmps to apply to the spectrum")
    spectrum.add_argument('--trimspec', required=False, nargs=2, default=(None,None),
                type='NoneOrFloat', metavar=("BLUELIM", "REDLIM"), help="Trim spectrum to wavelength range")
    spectrum.add_argument('--rebin',  required=False, type=int, default=1,\
            help="Rebin the spectrum by an integer factor. Output wavelengths remain uncorrelated.")
    spectrum.add_argument('--rescale',  required=False, action="store_true", default=False,\
            help="Rescale the spectrum to make the noise ~1. Changes the value, bounds, scale on dl also")
    spectrum.add_argument('--blotch', required=False, action='store_true',\
            default=False, help="Blotch the spectrum to remove gaps/cosmic rays before fitting?")

    # photometry options
    reddeninglaws = ('od94', 'ccm89', 'f99', 'custom')
    phot = parser.add_argument_group('photometry', 'Photometry options')
    phot.add_argument('--photfile', required=False,  default="data/photometry/WDphot_ILAPHv3_abmag.dat",\
            help="Specify file containing photometry lookup table for objects")
    phot.add_argument('--reddeningmodel', required=False, choices=reddeninglaws, default='f99',\
            help="Specify functional form of reddening law" )
    phot.add_argument('--phot_dispersion', required=False, type=float, default=0.003,\
            help="Specify a flat photometric dispersion error in mag to add in quadrature to the measurement errors")
    phot.add_argument('--pbfile', required=False,  default=None,\
            help="Specify file containing mapping from passband to pysynphot obsmode")
    phot.add_argument('--excludepb', nargs='+',\
            help="Specify passbands to exclude" )
    phot.add_argument('--ignorephot',  required=False, action="store_true", default=False,\
            help="Ignores missing photometry and does the fit with just the spectrum")

    # fitting options
    model = parser.add_argument_group('model',\
            'Model options. Modify using --param_file or CL. CL overrides. Caveat emptor.')
    for param in params:
        # we can't reasonably expect a user supplied guess or a static value to
        # work for some parameters. Allow None for these, and we'll determine a
        # good starting guess from the data. Note that we can actually just get
        # a starting guess for FWHM, but it's easier to use a lookup table.
        if param in ('fwhm','dl','mu'):
            dtype = 'NoneOrFloat'
        else:
            dtype = float

        model.add_argument('--{}'.format(param), required=False, type=dtype, default=params[param]['value'],\
                help="Specify param {} value".format(param))
        model.add_argument('--{}_fix'.format(param), required=False, default=params[param]['fixed'], type="bool",\
                help="Specify if param {} is fixed".format(param))
        model.add_argument('--{}_scale'.format(param), required=False, type=float, default=params[param]['scale'],\
                help="Specify param {} scale/step size".format(param))
        model.add_argument('--{}_bounds'.format(param), required=False, nargs=2, default=params[param]["bounds"],
                type=float, metavar=("LOWERLIM", "UPPERLIM"), help="Specify param {} bounds".format(param))

    # covariance model options
    covmodel = parser.add_argument_group('covariance model', 'Covariance model options')
    cov_choices=('White','Matern32','Exp','SHO')
    covmodel.add_argument('--covtype', required=False, choices=cov_choices,\
                default='Matern32', help='Specify parametric form of the covariance function to model the spectrum')
    covmodel.add_argument('--coveps', required=False, type=float, default=1e-12,\
            help="Specify accuracy of Matern32 kernel approximation")

    # MCMC config options
    mcmc = parser.add_argument_group('mcmc', 'MCMC options')
    mcmc.add_argument('--skipminuit',  required=False, action="store_true", default=False,\
            help="Skip Minuit fit - make sure to specify dl guess")
    mcmc.add_argument('--samptype', required=False, default='ensemble', choices=('ensemble', 'gibbs', 'pt'),\
            help='Specify what kind of sampler you want to use')
    mcmc.add_argument('--skipmcmc',  required=False, action="store_true", default=False,\
            help="Skip MCMC - if you skip both minuit and MCMC, simply prepares files")
    mcmc.add_argument('--ascale', required=False, type=float, default=2.0,\
            help="Specify proposal scale for MCMC")
    mcmc.add_argument('--nwalkers',  required=False, type=int, default=300,\
            help="Specify number of walkers to use (0 disables MCMC)")
    mcmc.add_argument('--ntemps', required=False, type=int, default=1,\
            help="Specify number of temperatures in ladder for parallel tempering - only available with PTSampler")
    mcmc.add_argument('--nburnin',  required=False, type=int, default=200,\
            help="Specify number of steps for burn-in")
    mcmc.add_argument('--nprod',  required=False, type=int, default=2000,\
            help="Specify number of steps for production")
    mcmc.add_argument('--everyn',  required=False, type=int, default=1,\
            help="Use only every nth point in data for computing likelihood - useful for testing.")
    mcmc.add_argument('--thin', required=False, type=int, default=1,\
            help="Save only every nth point in the chain - only works with PTSampler and Gibbs")
    mcmc.add_argument('--discard',  required=False, type=float, default=25,\
            help="Specify percentage of steps to be discarded")
    clobber = mcmc.add_mutually_exclusive_group()
    clobber.add_argument('--resume',  required=False, action="store_true", default=False,\
            help="Resume the MCMC from the last stored location")
    clobber.add_argument('--redo',  required=False, action="store_true", default=False,\
            help="Clobber existing fits")

    # visualization options
    viz = parser.add_argument_group('viz', 'Visualization options')
    viz.add_argument('-b', '--balmerlines', nargs='+', type=int, default=list(range(1,7,1)),\
            help="Specify Balmer lines to visualize [1:7]")
    viz.add_argument('--ndraws', required=False, type=int, default=21,\
            help="Specify number of draws from posterior to overplot for model")
    viz.add_argument('--savefig',  required=False, action="store_true", default=False,\
            help="Save individual plots")

    # output options
    output = parser.add_argument_group('output', 'Output options')
    output.add_argument('--outroot', required=False,
            help="Specify a custom output root directory. Directories go under outroot/objname/subdir.")
    output.add_argument('-o', '--outdir', required=False,\
            help="Specify a custom output directory. Overrides outroot.")

    args = None
    try:
        if comm.Get_rank() == 0:
            args = parser.parse_args(args=remaining_argv)
    finally:
        args = comm.bcast(args, root=0)

    if args is None:
        sys.exit(0)

    # some sanity checking for option values
    balmer = args.balmerlines
    try:
        balmer = np.atleast_1d(balmer).astype('int')
        if np.any((balmer < 1) | (balmer > 6)):
            raise ValueError
    except (TypeError, ValueError):
        message = 'Invalid balmer line value - must be in range [1,6]'
        raise ValueError(message)

    if args.rebin < 1:
        message = 'Rebin must be integer GE 1. Note that 1 does nothing. ({:g})'.format(args.rebin)
        raise ValueError(message)

    if args.phot_dispersion < 0.:
        message = 'Photometric dispersion must be GE 0. ({:g})'.format(args.phot_dispersion)
        raise ValueError(message)

    if args.coveps <= 0:
        message = 'Matern32 approximation eps must be greater than 0. ({:g})'.format(args.coveps)
        raise ValueError(message)

    if args.nwalkers <= 0:
        message = 'Number of walkers must be greater than zero for MCMC ({})'.format(args.nwalkers)
        raise ValueError(message)

    if args.nwalkers%2 != 0:
        message = 'Number of walkers must be even ({})'.format(args.nwalkers)
        raise ValueError(message)

    if args.ntemps <= 0:
        message = 'Number of temperatures must be greater than zero ({})'.format(args.ntemps)
        raise ValueError(message)

    if (args.ntemps > 1) and (args.samptype == 'ensemble'):
        message = 'Multiple temperatures only available with PTSampler or Gibbs Sampler: ({})'.format(args.ntemps)
        raise ValueError(message)

    if args.nburnin <= 0:
        message = 'Number of burnin steps must be greater than zero ({})'.format(args.nburnin)
        raise ValueError(message)

    if args.nprod <= 0:
        message = 'Number of production steps must be greater than zero ({})'.format(args.nprod)
        raise ValueError(message)

    if not (0 <= args.discard < 100):
        message = 'Discard must be a percentage (0-100) ({})'.format(args.discard)
        raise ValueError(message)

    if args.everyn < 1:
        message = 'EveryN must be integer GE 1. Note that 1 does nothing. ({:g})'.format(args.everyn)
        raise ValueError(message)

    if args.thin < 1:
        message = 'Thin must be integer GE 1. Note that 1 does nothing. ({:g})'.format(args.thin)
        raise ValueError(message)

    if args.reddeningmodel == 'custom':
        if not ((args.rv == 3.1) and (args.rv_fix == True)):
            message = 'Rv must be fixed to 3.1 for reddening model custom'
            raise ValueError(message)

    # Wait for instructions from the master process if we are running MPI
    pool = None
    if args.mpi or args.mpil:
        pool = MPIPool(loadbalance=args.mpil, debug=False)
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

    return args, pool


def copy_params(params):
    """
    Returns a deep copy of a dictionary. Necessary to ensure that dictionaries
    that nest dictionaries are properly updated.

    Parameters
    ----------
    params : dict or Object
        Any python object for which a deepcopy needs to be created.  Typically
        a parameter dictionary such as that from
        :py:func:`WDmodel.io.read_params`

    Returns
    -------
    params : Object
        A deepcopy of the object

    Notes
    -----
        Simple wrapper around :py:func:`copy.deepcopy`
    """

    return deepcopy(params)


def write_params(params, outfile):
    """
    Dumps the parameter dictionary params to a JSON file

    Parameters
    ----------
    params : dict
        A parameter dict such as that produced by
        :py:func:`WDmodel.io.read_params`
    outfile : str
        Output filename to save the parameter dict as a JSON file.

    Notes
    -----
        params is a dict the parameter names, as defined with
        :py:const:`WDmodel.io._PARAMETER_NAMES` as keys

        Each key must have a dictionary with keys:
         * ``value`` : value
         * ``fixed`` : a bool specifying if the parameter is fixed (``True``) or allowed to vary (``False``)
         * ``scale``  : a scale parameter used to set the step size in this dimension
         * ``bounds`` : An upper and lower limit on parameter values

        Any extra keys are simply written as-is JSON doesn't preserve ordering
        necessarily. This is imposed by :py:func:`WDmodel.io.read_params`

    See Also
    --------
    :py:func:`WDmodel.io.read_params`
    """
    for param in params:
        if not all (key in params[param] for key in ("value","fixed","scale", "bounds")):
            message = "Parameter {} does not have value|fixed|bounds specified in params dict".format(param)
            raise KeyError(message)

    with open(outfile, 'w') as f:
        json.dump(params, f, indent=4)


def read_params(param_file=None):
    """
    Read a JSON file that configures the default guesses and bounds for the
    parameters, as well as if they should be fixed.

    Parameters
    ----------
    param_file : str, optional
        The name of the input parameter file. If not the default file provided
        with the package, ``WDmodel_param_defaults.json``, is read.

    Returns
    -------
    params : dict
        The dictionary with the parameter ``values``, ``bounds``, ``scale`` and
        if ``fixed``. See notes for more detailed information on dictionary
        format and ``WDmodel_param_defaults.json`` for an example file for
        ``param_file``.

    Notes
    -----
        params is a dict the parameter names, as defined with
        :py:const:`WDmodel.io._PARAMETER_NAMES` as keys

        Each key must have a dictionary with keys:
         * ``value`` : value
         * ``fixed`` : a bool specifying if the parameter is fixed (``True``) or allowed to vary (``False``)
         * ``scale``  : a scale parameter used to set the step size in this dimension
         * ``bounds`` : An upper and lower limit on parameter values

        The default bounds are set by the grids available for the DA White
        Dwarf atmospheres, and by reasonable plausible ranges for the other
        parameters. Don't muck with them unless you really have good reason to.

        This routine does not do any checking of types, values or bounds. This
        is done by :py:func:`WDmodel.io.get_params_from_argparse` before the
        fit. If you setup the fit using an external code, you should check
        these values.
    """

    if param_file is None:
        param_file = 'WDmodel_param_defaults.json'
        param_file = get_pkgfile(param_file)

    with open(param_file, 'r') as f:
        params = json.load(f)

    # JSON doesn't preserve ordering at all, but I'd like to keep it consistent
    out = OrderedDict()
    for param in _PARAMETER_NAMES:
        # note that we're only checking if we have the right keys here, not if the values are reasonable
        if not param in params:
            message = "Parameter {} not found in JSON param file {}".format(param, param_file)
            raise KeyError(message)
        if not all (key in params[param] for key in ("value","fixed","scale", "bounds")):
            message = "Parameter {} does not have value|fixed|bounds specified in param file {}".format(param, param_file)
            raise KeyError(message)
        out[param] = params[param]

    return out


def get_params_from_argparse(args):
    """
    Converts an :py:class:`argparse.Namespace` into an ordered parameter
    dictionary.

    Parameters
    ----------
    args : :py:class:`argparse.Namespace`
        The parsed command-line options from :py:func:`WDmodel.io.get_options`

    Returns
    -------
    params : :py:class:`collections.OrderedDict`
        The parameter dictionary

    Raises
    ------
    RuntimeError
         If format of :py:class:`argparse.Namespace` is invalid.
         or
         If parameter is ``fixed`` but ``value`` is ``None``.
         or
         If parameter ``value`` is out of ``bounds``.

    Notes
    -----
        Assumes that the argument parser options were names
         * ``<param>_value`` : Value of the parameter (float or ``None``)
         * ``<param>_fix`` : Bool specifying if the parameter
         * ``<param>_bounds`` : tuple with lower limit and upper limit

        where <param> is one of :py:data:`WDmodel.io._PARAMETER_NAMES`

    See Also
    --------
    :py:func:`WDmodel.io.read_params`
    :py:func:`WDmodel.io.get_options`
    """

    kwargs = vars(args)
    out = OrderedDict()
    for param in _PARAMETER_NAMES:
        keys = (param, "{}_fix".format(param),"{}_scale".format(param), "{}_bounds".format(param))
        if not all (key in kwargs for key in keys):
            message = "Parameter {} does not have value|fixed|scale|bounds specified in argparse args".format(param)
            raise KeyError(message)
        out[param]={}
        value = kwargs[param]
        out[param]['value']  = value
        out[param]['fixed']  = kwargs['{}_fix'.format(param)]

        # check if are fixed to None, which would cause likelihood to always fail
        if (out[param]['fixed'] is True) and (value is None):
            message = "Parameter {} fixed but value is None - must be specified".format(param)
            raise RuntimeError(message)
        bounds = kwargs['{}_bounds'.format(param)]

        # check if value is out of bounds, which will cause fit to fail
        if value is not None:
            if value < bounds[0] or value > bounds[1]:
                message = "Parameter {} value ({}) is out of bounds ({},{})".format(param, value, bounds[0], bounds[1])
                raise RuntimeError(message)
        out[param]['scale']  = kwargs['{}_scale'.format(param)]
        out[param]['bounds'] = bounds
    return out


def read_model_grid(grid_file=None, grid_name=None):
    """
    Read the Tlusty/Hubeny grid file


    Parameters
    ----------
    grid_file : None or str
        Filename of the Tlusty model grid HDF5 file. If ``None`` reads the
        ``TlustyGrids.hdf5`` file included with the :py:mod:`WDmodel`
        package.
    grid_name : None or str
        Name of the group name in the HDF5 file to read the grid from. If
        ``None`` uses ``default``

    Returns
    -------
    grid_file : str
        Filename of the HDF5 grid file
    grid_name : str
        Name of the group within the HDF5 grid file with the grid arrays
    wave : array-like
        The wavelength array of the grid with shape ``(nwave,)``
    ggrid : array-like
        The surface gravity array of the grid with shape ``(ngrav,)``
    tgrid : array-like
        The temperature array of the grid with shape ``(ntemp,)``
    flux : array-like
        The DA white dwarf model atmosphere flux array of the grid.
        Has shape ``(nwave, ngrav, ntemp)``

    Notes
    -----
        There are no easy command line options to change this deliberately
        because changing the grid file essentially changes the entire model,
        and should not be done lightly, without careful comparison of the grids
        to quantify differences.

    See Also
    --------
    :py:class:`WDmodel.WDmodel`
    """

    if grid_file is None:
        grid_file = 'TlustyGrids.hdf5'

    # if the user specfies a file, check that it exists, and if not look inside the package directory
    if not os.path.exists(grid_file):
        grid_file = get_pkgfile(grid_file)

    if grid_name is None:
        grid_name = "default"

    with h5py.File(grid_file, 'r') as grids:
        # the original IDL SAV file Tlusty grids were annoyingly broken up by wavelength
        # this was because the authors had different wavelength spacings
        # since they didn't feel that the continuum needed much spacing anyway
        # and "wanted to save disk space"
        # and then their old IDL interpolation routine couldn't handle the variable spacing
        # so they broke up the grids
        # So really, the original IDL SAV files were annoyingly broken up by wavelength because reasons...
        # We have concatenated these "large" arrays because we don't care about disk space
        # This grid is called "default", but the originals also exist
        # and you can pass grid_name to use them if you choose to
        try:
            grid = grids[grid_name]
        except KeyError as e:
            message = '{}\nGrid {} not found in grid_file {}. Accepted values are ({})'.format(e, grid_name,\
                    grid_file, ','.join(list(grids.keys())))
            raise ValueError(message)

        wave  = grid['wave'].value.astype('float64')
        ggrid = grid['ggrid'].value.astype('float64')
        tgrid = grid['tgrid'].value.astype('float64')
        flux  = grid['flux'].value.astype('float64')

    return grid_file, grid_name, wave, ggrid, tgrid, flux


def _read_ascii(filename, **kwargs):
    """
    Read ASCII files

    Read space separated ASCII file, with column names provided on first line
    (leading ``#`` optional). ``kwargs`` are passed along to
    :py:func:`numpy.genfromtxt`. Forces any string column data to be encoded in
    ASCII, rather than Unicode.

    Parameters
    ----------
    filename : str
        Filename of the ASCII file. Column names must be provided on the first
        line.
    kwargs : dict
        Extra options, passed directly to :py:func:`numpy.genfromtxt`

    Returns
    -------
    out : :py:class:`numpy.recarray`
        Record array with the data. Field names correspond to column names in
        the file.

    See Also
    --------
    :py:func:`numpy.genfromtxt`
    """

    indata = np.recfromtxt(filename, names=True, **kwargs)
    # force bytestrings into ascii encoding and recreate the recarray
    out = []
    for name in indata.dtype.names:
        thistype =  indata[name].dtype.str.lstrip('|')
        if thistype.startswith('S'):
            out.append(np.array([str(x.decode('ascii')) for x in indata[name]]))
        else:
            out.append(indata[name])
    out = np.rec.fromarrays(out, names=indata.dtype.names)
    return out


# create some aliases
# these exist so we can flesh out full functions later
# with different formats if necessary for different sorts of data

read_phot      = _read_ascii
"""Read photometry - wraps :py:func:`_read_ascii`"""
read_spectable = _read_ascii
"""Read spectrum FWHM table - wraps :py:func:`_read_ascii`"""
read_pbmap     = _read_ascii
"""Read passband obsmode mapping table - wraps :py:func:`_read_ascii`"""
read_reddening = _read_ascii
"""Read J. Holberg's custom reddening function - wraps :py:func:`_read_ascii`"""


def get_spectrum_resolution(specfile, spectable, fwhm=None, lamshift=None):
    """
    Gets the measured FWHM from a spectrum lookup table.

    Parameters
    ----------
    specfile : str
        The spectrum filename
    spectable : str
        The spectrum FWHM lookup table filename
    fwhm : None or float, optional
        If specified, this overrides the resolution provided in the lookup
        table. If ``None`` lookups the resultion from ``spectable``.
    lamshift : None or float, optional
        If specified, this overrides the wavelength shift provided in the lookup
        table. If ``None`` lookups the wavelength shift from ``spectable``.

    Returns
    -------
    fwhm : float
        The FWHM of the spectrum file. This is typically used as an initial
        guess to the :py:mod:`WDmodel.fit` fitter routines.
    lamshift: float
        The wavelength shift to apply additively to the spectrum. This is not a
        fit parameter, and is treated as an input

    Raises
    ------
    :py:exc:`RuntimeWarning`
        If the ``spectable`` cannot be read, or the ``specfile`` name indicates
        that this is a test, or if there are no or multiple matches for
        ``specfile`` in the ``spectable``

    Notes
    -----
        If the ``specfile`` is not found, it returns a default resolution of
        ``5`` Angstroms, appropriate for the instruments used in our program.

        Note that there there's some hackish internal name fixing since T.
        Matheson's table spectrum names didn't match the spectrum filenames.
    """

    _default_resolution = 5.0
    _default_lamshift   = 0.

    try:
        spectable = read_spectable(spectable)
    except (OSError, IOError) as e:
        message = '{}\nCould not get resolution from spectable {}'.format(e, spectable)
        warnings.warn(message, RuntimeWarning)
        spectable = None

    shortfile = os.path.basename(specfile).replace('-total','')

    if fwhm is None:
        if shortfile.startswith('test'):
            message = 'Spectrum filename indicates this is a test - using default resolution'
            warnings.warn(message, RuntimeWarning)
            fwhm = _default_resolution
        elif spectable is not None:
            mask = (spectable.specname == shortfile)
            if len(spectable[mask]) != 1:
                message = 'Could not find an entry for this spectrum in the spectable file - using default resolution'
                warnings.warn(message, RuntimeWarning)
                fwhm = _default_resolution
            else:
                fwhm = spectable[mask].fwhm[0]
        else:
            fwhm = _default_resolution
    else:
        message = 'Smoothing factor specified on command line - overriding spectable file'
        warnings.warn(message, RuntimeWarning)

    if lamshift is None:
        if shortfile.startswith('test'):
            message = 'Spectrum filename indicates this is a test - using default wavelength shift'
            warnings.warn(message, RuntimeWarning)
            lamshift = _default_lamshift
        elif spectable is not None:
            mask = (spectable.specname == shortfile)
            if len(spectable[mask]) != 1:
                message = 'Could not find an entry for this spectrum in the spectable file - using wavelength shift'
                warnings.warn(message, RuntimeWarning)
                lamshift = _default_lamshift
            else:
                lamshift = spectable[mask].lamshift[0]
        else:
            lamshift = _default_lamshift

    message = 'Using smoothing instrumental FWHM {} and wavelength shift {}'.format(fwhm, lamshift)
    print(message)
    return fwhm, lamshift


def read_spec(filename, **kwargs):
    """
    Read a spectrum

    Wraps :py:func:`_read_ascii`, adding testing of the input arrays to check
    if the elements are finite, and if the errors and flux are strictly
    positive.

    Parameters
    ----------
    filename : str
        Filename of the ASCII file. Must have columns ``wave``, ``flux``,
        ``flux_err``
    kwargs : dict
        Extra options, passed directly to :py:func:`numpy.genfromtxt`

    Returns
    -------
    spec : :py:class:`numpy.recarray`
        Record array with the spectrum data.
        Has ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8')]``

    Raises
    ------
    ValueError
        If any value is not finite or if ``flux`` or ``flux_err`` have any
        values ``<= 0``

    See Also
    --------
    :py:func:`numpy.genfromtxt`
    :py:func:`_read_ascii`
    """

    spec = _read_ascii(filename, **kwargs)
    if np.any(~np.isfinite(spec.wave)) or np.any(~np.isfinite(spec.flux)) or np.any(~np.isfinite(spec.flux_err)):
        message = "Spectroscopy values and uncertainties must be finite."
        raise ValueError(message)

    if np.any(spec.flux_err <= 0.) or np.any(spec.flux <= 0.):
        message = "Spectroscopy values uncertainties must all be positive."
        raise ValueError(message)

    return spec


def get_phot_for_obj(objname, filename):
    """
    Gets the measured photometry for an object from a photometry lookup table.

    Parameters
    ----------
    objname : str
        Object name to look for photometry for
    filename : str
        The spectrum FWHM lookup table filename

    Returns
    -------
    phot : :py:class:`numpy.recarray`
        The photometry of ``objname`` with ``dtype=[('pb', 'str'), ('mag', '<f8'), ('mag_err', '<f8')]``

    Raises
    ------
    RuntimeError
        If there are no matches in the photometry lookup file or if there are
        multiple matches for an object in the photometry lookup file
    ValueError
        If the photometry or the photometry uncertainty values are not finite
        or if the photometry uncertainties are less ``<= 0``

    Notes
    -----
        The lookup file must be readable by :py:func:`read_phot`

        The column name with the object name ``objname`` expected to be ``obj``

        If column names for magnitudes are named <passband>, the column names
        for errors in magnitudes in passband must be 'd'+<passband_name>.
    """

    phot = read_phot(filename)
    mask = (phot.obj == objname)

    nmatch = len(phot[mask])
    if nmatch == 0:
        message = 'Got no matches for object {} in file {}. Did you want --ignorephot?'.format(objname, filename)
        raise RuntimeError(message)
    elif nmatch > 1:
        message = 'Got multiple matches for object {} in file {}'.format(objname, filename)
        raise RuntimeError(message)
    else:
        pass

    this_phot =  phot[mask][0]
    colnames  = this_phot.dtype.names
    pbnames   = [pb for pb in colnames[1:] if not pb.startswith('d')]

    mags = [this_phot[pb] for pb in pbnames]
    errs = [this_phot['d'+pb] for pb in pbnames]

    pbnames = np.array(pbnames)
    mags    = np.array(mags)
    errs    = np.array(errs)

    if np.any(~np.isfinite(mags)) or np.any(~np.isfinite(errs)):
        message = "Photometry values and uncertainties must be finite."
        raise ValueError(message)

    if np.any(errs <= 0.):
        message = "Photometry uncertainties must all be positive."
        raise ValueError(message)
    names=str('pb,mag,mag_err')
    out_phot = np.rec.fromarrays([pbnames, mags, errs],names=names)
    return out_phot


def make_outdirs(dirname, redo=False, resume=False):
    """
    Makes output directories

    Parameters
    ----------
    dirname : str
        The output directory name to create
    redo : bool, optional
        If ``False`` the directory will not be created if it already exists, and an error is raised
    resume : bool, optional
        If ``False`` the directory will not be created if it already exists, and an error is raised

    Returns
    -------
    None : None
        If the output directory ``dirname`` is successfully created

    Raises
    ------
    IOError
        If the output directory exists
    OSError
        If the output directory could not be created

    Notes
    -----
        If the options are parsed by :py:func:`get_options` then only one of
        ``redo`` or ``resume`` can be set, as the options are mutually
        exclusive. If ``redo`` is set, the fit is redone from scratch, while
        ``resume`` restarts the MCMC sampling from the last saved chain position.
    """

    if os.path.isdir(dirname):
        if resume or redo:
            return
        else:
            message = "Output directory {} already exists. Specify --redo to clobber.".format(dirname)
            raise IOError(message)

    try:
        os.makedirs(dirname)
    except OSError as  e:
        message = '{}\nCould not create outdir {} for writing.'.format(e,dirname)
        raise OSError(message)


def set_objname_outdir_for_specfile(specfile, outdir=None, outroot=None, redo=False, resume=False):
    """
    Sets the short human readable object name and output directory

    Parameters
    ----------
    specfile : str
        The spectrum filename
    outdir : None or str, optional
        The output directory name to create. If ``None`` this is set based on ``specfile``
    outroot : None or str, optional
        The output root directory under which to store the fits. If ``None`` the default is ``'out'``
    redo : bool, optional
        If ``False`` the directory will not be created if it already exists, and an error is raised
    resume : bool, optional
        If ``False`` the directory will not be created if it already exists, and an error is raised

    Returns
    -------
    objname : str
        The human readable object name based on the spectrum
    dirname : str
        The output directory name created if successful

    See Also
    --------
    :py:func:`make_outdirs`
    """

    if outroot is None:
        outroot = os.path.join(os.getcwd(), "out")
    basespec = os.path.basename(specfile).replace('.flm','')
    objname = basespec.split('-')[0]
    if outdir is None:
        dirname = os.path.join(outroot, objname, basespec)
    else:
        dirname = outdir
    make_outdirs(dirname, redo=redo, resume=resume)
    return objname, dirname


def get_outfile(outdir, specfile, ext, check=False, redo=False, resume=False):
    """
    Formats the output directory, spectrum filename, and an extension into an
    output filename.

    Parameters
    ----------
    outdir : str
        The output directory name for the output file
    specfile : str
        The spectrum filename
    ext : str
        The output file's extension
    check : bool, optional
        If ``True``, check if the output file exists
    redo : bool, optional
        If ``False`` and the output file already exists, an error is raised
    resume : bool, optional
        If ``False`` and the output file already exists, an error is raised

    Returns
    -------
    outfile : str
        The output filename

    Raises
    ------
    IOError
        If ``check`` is ``True``, ``redo`` and ``resume`` are ``False``, and
        ``outfile`` exists.

    Notes
    -----
        We set the output file based on the spectrum name, since we can have
        multiple spectra per object.

        If ``outdir`` is configured by :py:func:`set_objname_outdir_for_specfile` for
        ``specfile``, it'll include the object name.

    See Also
    --------
    :py:func:`set_objname_outdir_for_specfile`
    """

    outfile = os.path.join(outdir, os.path.basename(specfile.replace('.flm', ext)))
    if check:
        if os.path.exists(outfile) and not (resume or redo):
            message = "Output file {} already exists. Specify --redo to clobber.".format(outfile)
            raise IOError(message)
    return outfile


def get_pkgfile(infile):
    """
    Returns the full path to a file inside the :py:mod:`WDmodel` package

    Parameters
    ----------
    infile : str
        The name of the file to set the full package filename for

    Returns
    -------
    pkgfile : str
        The path to the file within the package.

    Raises
    ------
    IOError
        If the ``pkgfile`` could not be found inside the :py:mod:`WDmodel` package.

    Notes
    -----
        This allows the package to be installed anywhere, and the code to still
        determine the location to a file included with the package, such as the
        model grid file.
    """

    pkgfile = pkg_resources.resource_filename('WDmodel',infile)

    if not os.path.exists(pkgfile):
        message = 'Could not find package file {}'.format(pkgfile)
        raise IOError(message)
    return pkgfile


def write_fit_inputs(spec, phot, cont_model, linedata, continuumdata,\
        rvmodel, covtype, coveps, phot_dispersion, scale_factor, outfile):
    """
    Save all the inputs to the fitter to a file

    This file is enough to resume the fit with the same input, redoing the
    output, or restoring from a failure.

    Parameters
    ----------
    spec : :py:class:`numpy.recarray`
        The spectrum with ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8')]``
    phot : None or :py:class:`numpy.recarray`
        ``None`` or the photometry with ``dtype=[('pb', 'str'), ('mag', '<f8'), ('mag_err', '<f8')]``
    cont_model : :py:class:`numpy.recarray`
        The continuuum model. Must have the same structure as ``spec``.
        Produced by :py:func:`WDmodel.fit.pre_process_spectrum`.
        Used by :py:mod:`WDmodel.viz`
    linedata : :py:class:`numpy.recarray`
        The observations of the spectrum corresponding to the hydrogen Balmer
        lines. Must have ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8'), ('line_mask', 'i4')]``
        Produced by :py:func:`WDmodel.fit.pre_process_spectrum`
        Used by :py:mod:`WDmodel.viz`
    continuumdata : :py:class:`numpy.recarray`
        Data used to generate the continuum model. Must have the same structure
        as ``spec``. Produced by :py:func:`WDmodel.fit.pre_process_spectrum`
    rvmodel : ``{'ccm89','od94','f99', 'custom'}``
        Parametrization of the reddening law. Used to initialize
        :py:func:`WDmodel.WDmodel.WDmodel` instance.
    covtype : ``{'Matern32', 'SHO', 'Exp', 'White'}``
        stationary kernel type used to parametrize the covariance in
        :py:class:`WDmodel.covariance.WDmodel_CovModel`
    coveps : float
        If ``covtype`` is ``'Matern32'`` a
        :py:class:`celerite.terms.Matern32Term` is used to approximate a
        Matern32 kernel with precision `coveps`.
    phot_dispersion : float, optional
        Excess photometric dispersion to add in quadrature with the
        photometric uncertainties ``phot.mag_err`` in
        :py:class:`WDmodel.likelihood.WDmodel_Likelihood`.
    scale_factor : float
        Factor by which the flux must be scaled. Critical to getting the right
        uncertainties.
    outfile : str
        Output HDF5 filename

    Notes
    -----
        The outputs are stored in a HDF5 file with groups
         * ``spec`` - storing the spectrum and ``scale_factor``
         * ``cont_model`` - stores the continuum model
         * ``linedata`` - stores the hydrogen Balmer line data
         * ``continuumdata`` - stores the data used to generate ``cont_model``
         * ``fit_config`` - stores ``covtype``, ``coveps`` and ``rvmodel`` as attributes
         * ``phot`` - only created if ``phot`` is not ``None``, stores ``phot``, ``phot_dispersion``
    """

    outf = h5py.File(outfile, 'w')
    dset_spec = outf.create_group("spec")
    dset_spec.create_dataset("wave",data=spec.wave)
    dset_spec.create_dataset("flux",data=spec.flux)
    dset_spec.create_dataset("flux_err",data=spec.flux_err)
    dset_spec.attrs["scale_factor"] = scale_factor

    dset_cont_model = outf.create_group("cont_model")
    dset_cont_model.create_dataset("wave",data=cont_model.wave)
    dset_cont_model.create_dataset("flux",data=cont_model.flux)

    dset_linedata = outf.create_group("linedata")
    dset_linedata.create_dataset("wave",data=linedata.wave)
    dset_linedata.create_dataset("flux",data=linedata.flux)
    dset_linedata.create_dataset("flux_err",data=linedata.flux_err)
    dset_linedata.create_dataset("line_mask",data=linedata.line_mask)
    dset_linedata.create_dataset("line_ind",data=linedata.line_ind)

    dset_continuumdata = outf.create_group("continuumdata")
    dset_continuumdata.create_dataset("wave",data=continuumdata.wave)
    dset_continuumdata.create_dataset("flux",data=continuumdata.flux)
    dset_continuumdata.create_dataset("flux_err",data=continuumdata.flux_err)

    dset_fit_config = outf.create_group("fit_config")
    dset_fit_config.attrs["covtype"]=np.string_(covtype)
    dset_fit_config.attrs["coveps"]=coveps
    dset_fit_config.attrs["rvmodel"]=np.string_(rvmodel)

    if phot is not None:
        dset_phot = outf.create_group("phot")
        dt = phot.pb.dtype.str.lstrip('|').replace('U','S')
        dset_phot.create_dataset("pb", data=phot.pb.astype(np.string_), dtype=dt)
        dset_phot.create_dataset("mag",data=phot.mag)
        dset_phot.create_dataset("mag_err",data=phot.mag_err)
        dset_phot.attrs["phot_dispersion"] =phot_dispersion

    outf.close()
    message = "Wrote inputs file {}".format(outfile)
    print(message)


def read_fit_inputs(input_file):
    """
    Read the fit input HDF5 file produced by :py:func:`write_fit_inputs` and
    return :py:class:`numpy.recarray` instances with the data.

    Parameters
    ----------
    input_file : str
        The HDF5 fit inputs filename

    Returns
    -------
    spec : :py:class:`numpy.recarray`
        The spectrum with ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8')]``
    cont_model : :py:class:`numpy.recarray`
        The continuuum model. Has the same structure as ``spec``.
    linedata : :py:class:`numpy.recarray`
        The observations of the spectrum corresponding to the hydrogen Balmer
        lines. Has ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8'), ('line_mask', 'i4')]``
    continuumdata : :py:class:`numpy.recarray`
        Data used to generate the continuum model. Has the same structure as
        ``spec``.
    phot : None or :py:class:`numpy.recarray`
        ``None`` or the photometry with ``dtype=[('pb', 'str'), ('mag', '<f8'), ('mag_err', '<f8')]``
    fit_config : dict
        Dictionary with various keys needed to configure the fitter
         * ``rvmodel`` : ``{'ccm89','od94','f99', 'custom'}`` - Parametrization of the reddening law.
         * ``covtype`` : ``{'Matern32', 'SHO', 'Exp', 'White'}``- kernel type used to parametrize the covariance
         * ``coveps`` : float - Matern32 kernel precision
         * ``phot_dispersion`` : float - Excess dispersion to add in quadrature with photometric uncertainties
         * ``scale_factor`` : float - Flux scale factor

    Raises
    ------
    IOError
        If all the fit inputs could not be restored from the HDF5 ``input_file``
    RuntimeWarning
        If the ``input_file`` includes a ``phot`` group, but the data cannot be loaded.

    See Also
    --------
    :py:func:`write_fit_inputs`
    """

    d = h5py.File(input_file, mode='r')

    try:
        spec_wave = d['spec']['wave'].value
        spec_flux = d['spec']['flux'].value
        spec_ferr = d['spec']['flux_err'].value
        scale_factor = d['spec'].attrs['scale_factor']
        names=str('wave,flux,flux_err')
        spec = np.rec.fromarrays([spec_wave, spec_flux, spec_ferr], names=names)

        cmod_wave = d['cont_model']['wave'].value
        cmod_flux = d['cont_model']['flux'].value
        names=str('wave,flux')
        cont_model = np.rec.fromarrays([cmod_wave, cmod_flux], names=names)

        line_wave = d['linedata']['wave'].value
        line_flux = d['linedata']['flux'].value
        line_ferr = d['linedata']['flux_err'].value
        line_mask = d['linedata']['line_mask'].value
        line_ind  = d['linedata']['line_ind'].value
        names=str('wave,flux,flux_err,line_mask,line_ind')
        linedata = np.rec.fromarrays([line_wave, line_flux, line_ferr,line_mask,line_ind], names=names)

        cont_wave = d['continuumdata']['wave'].value
        cont_flux = d['continuumdata']['flux'].value
        cont_ferr = d['continuumdata']['flux_err'].value
        names=str('wave,flux,flux_err')
        continuumdata = np.rec.fromarrays([cont_wave, cont_flux, cont_ferr], names=names)

        fit_config = {}
        fit_config['covtype'] = d['fit_config'].attrs['covtype'].decode('ascii')
        fit_config['coveps'] = d['fit_config'].attrs['coveps']
        fit_config['rvmodel'] = d['fit_config'].attrs['rvmodel'].decode('ascii')
        fit_config['scale_factor'] = scale_factor

    except Exception as e:
        message = '{}\nCould not load all arrays from input file {}'.format(e, input_file)
        raise IOError(message)

    phot = None
    fit_config['phot_dispersion'] = 0.001
    if 'phot' in list(d.keys()):
        try:
            pb  = d['phot']['pb'].value
            pb  = np.array([str(x.decode('ascii')) for x in pb])
            mag = d['phot']['mag'].value
            mag_err = d['phot']['mag_err'].value
            names=str('pb,mag,mag_err')
            phot = np.rec.fromarrays([pb, mag, mag_err], names=names)
            phot_dispersion = d['phot'].attrs['phot_dispersion']
            fit_config['phot_dispersion'] = phot_dispersion
        except KeyError as e:
            message = '{}\nFailed to restore photometry from input file {} though group exists'.format(e, input_file)
            warnings.warn(message, RuntimeWarning)
            phot = None
    return spec, cont_model, linedata, continuumdata, phot, fit_config


def read_mcmc(input_file):
    """
    Read the saved HDF5 Markov chain file and return samples, sample log
    probabilities and chain parameters

    Parameters
    ----------
    input_file : str
        The HDF5 Markov chain filename

    Returns
    -------
    samples : array-like
        The model parameter sample chain
    samples_lnprob : array-like
        The log posterior corresponding to each of the ``samples``
    chain_params : dict
        The chain parameter dictionary
         * ``param_names`` : list - list of model parameter names
         * ``samptype`` : ``{'ensemble','pt','gibbs'}`` - the sampler to use
         * ``ntemps`` : int - the number of chain temperatures
         * ``nwalkers`` : int - the number of Goodman & Ware walkers
         * ``nprod`` : int - the number of production steps of the chain
         * ``ndim`` : int - the number of model parameters in the chain
         * ``thin`` : int - the chain thinning if any
         * ``everyn`` : int - the sparse of spectrum sampling step size
         * ``ascale`` : float - the proposal scale for the sampler

    Raises
    ------
    IOError
        If a key in the ``fit_config`` output is missing
    """

    d = h5py.File(input_file, mode='r')

    try:
        chain_params   = {}
        samples        = d['chain']['position'].value
        samples_lnprob = d['chain']['lnprob'].value
        param_names                 = d['chain']['names'].value
        param_names                 = np.array([str(x.decode('ascii')) for x in param_names])
        chain_params['param_names'] = param_names
        chain_params['samptype']    = d['chain'].attrs['samptype']
        chain_params['ntemps']      = d['chain'].attrs['ntemps']
        chain_params['nwalkers']    = d['chain'].attrs['nwalkers']
        chain_params['nprod']       = d['chain'].attrs['nprod']
        chain_params['ndim']        = d['chain'].attrs['nparam']
        chain_params['thin']        = d['chain'].attrs['thin']
        chain_params['everyn']      = d['chain'].attrs['everyn']
        chain_params['ascale']      = d['chain'].attrs['ascale']

    except KeyError as e:
        message = '{}\nCould not load all arrays from input file {}'.format(e, input_file)
        raise IOError(message)

    return samples, samples_lnprob, chain_params


def write_spectrum_model(spec, model_spec, outfile):
    """
    Write the spectrum and the model spectrum and residuals to an output file.

    Parameters
    ----------
    spec : :py:class:`numpy.recarray`
        The spectrum with ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8')]``
    model_spec : :py:class:`numpy.recarray`
        The model spectrum.
        Has ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('norm_flux', '<f8'), ('flux_err', '<f8')]``
    outfile : str
        Output space-separated text filename

    Notes
    -----
        The data is saved to a space-separated ASCII text file with 8 decimal
        places of precision.

        The order of the columns is
         * ``wave`` : array-like - the spectrum wavelength
         * ``flux`` : array-like - the observed flux
         * ``flux_err`` : array-like - the observed flux uncertainty
         * ``norm_flux`` : array-like - the model flux without the Gaussian process covariance model
         * ``model_flux`` : array-like - the model flux
         * ``model_flux_err`` : array-like - the model flux uncertainty
         * ``res_flux`` : array-like - the flux residual
    """

    out = (spec.wave, spec.flux, spec.flux_err,\
            model_spec.norm_flux, model_spec.flux, model_spec.flux_err,\
            spec.flux-model_spec.flux)
    names=str('wave,flux,flux_err,norm_flux,model_flux,model_flux_err,res_flux').split(',')
    out = at.Table(out, names=names)
    for name in names:
        out[name].format='%.8f'
    out.write(outfile, format='ascii.fixed_width', delimiter=' ', overwrite=True)
    message = "Wrote spec model file {}".format(outfile)
    print(message)


def write_phot_model(phot, model_mags, outfile):
    """
    Write the photometry, model photometry and residuals to an output file.

    Parameters
    ----------
    phot : None or :py:class:`numpy.recarray`
        ``None`` or the photometry with ``dtype=[('pb', 'str'), ('mag', '<f8'), ('mag_err', '<f8')]``
    model_mags : None or :py:class:`numpy.recarray`
        The model magnitudes.
        Has ``dtype=[('pb', 'str'), ('mag', '<f8')]``
    outfile : str
        Output space-separated text filename

    Notes
    -----
        The data is saved to a space-separated ASCII text file with 6 decimal
        places of precision.

        The order of the columns is
         * ``pb`` : array-like - the observation's passband
         * ``mag`` : array-like - the observed magnitude
         * ``mag_err`` : array-like - the observed magnitude uncertainty
         * ``model_mag`` : array-like - the model magnitude
         * ``res_mag`` : array-like - the magnitude residual
    """

    out = (phot.pb, phot.mag, phot.mag_err, model_mags.mag, phot.mag-model_mags.mag)
    names=str('pb,mag,mag_err,model_mag,res_mag').split(',')
    out = at.Table(out, names=names)
    for name in names[1:]:
        out[name].format='%.6f'
    out.write(outfile, format='ascii.fixed_width', delimiter=' ', overwrite=True)
    message= "Wrote phot model file {}".format(outfile)
    print(message)


def read_full_model(input_file):
    """
    Read the full SED model from an output file.

    Parameters
    ----------
    input_file : str
        Input HDF5 SED model filename

    Returns
    -------
    spec : :py:class:`numpy.recarray`
        Record array with the model SED.
        Has ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8')]``

    Raises
    ------
    KeyError
        If any of ``wave``, ``flux`` or ``flux_err`` is not found in the file
    ValueError
        If any value is not finite or if ``flux`` or ``flux_err`` have any
        values ``<= 0``
    """
    with h5py.File(input_file, 'r') as indata:
        try:
            inspec = indata['model']
            wave = inspec['wave'].value
            flux = inspec['flux'].value
            flux_err = inspec['flux_err'].value
        except KeyError as e:
            message = '{}\nCould not load SED model from file {}'.format(e, input_file)
            raise KeyError(message)

    spec = np.rec.fromarrays([wave, flux, flux_err], names='wave,flux,flux_err')

    if np.any(~np.isfinite(spec.wave)) or np.any(~np.isfinite(spec.flux)) or np.any(~np.isfinite(spec.flux_err)):
        message = "Spectroscopy values and uncertainties must be finite."
        raise ValueError(message)

    if np.any(spec.flux_err <= 0.) or np.any(spec.flux <= 0.):
        message = "Spectroscopy values uncertainties must all be positive."
        raise ValueError(message)

    return spec


def write_full_model(full_model, outfile):
    """
    Write the full SED model to an output file.

    Parameters
    ----------
    full_model : :py:class:`numpy.recarray`
        The SED model with ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8')]``
    outfile : str
        Output HDF5 SED model filename

    Notes
    -----
        The output is written into a group ``model`` with datasets
         * ``wave`` : array-like - the SED model wavelength
         * ``flux`` : array-like - the SED model flux
         * ``flux_err`` : array-like - the SED model flux uncertainty
    """

    outf = h5py.File(outfile, 'w')
    dset_model = outf.create_group("model")
    dset_model.create_dataset("wave",data=full_model.wave)
    dset_model.create_dataset("flux",data=full_model.flux)
    dset_model.create_dataset("flux_err",data=full_model.flux_err)
    outf.close()
    message = "Wrote full model file {}".format(outfile)
    print(message)
