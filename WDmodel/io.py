import os
import warnings
from copy import deepcopy
import numpy as np
import pkg_resources
from collections import OrderedDict
from matplotlib.mlab import rec2txt
import json
import h5py

# Declare this tuple to init the likelihood model, and to preserve order of parameters
_PARAMETER_NAMES = ("teff", "logg", "av", "rv", "dl", "fwhm", "sigf", "tau", "mu")


def copy_params(params):
    """
    Returns a deep copy of a dictionary
    """
    return deepcopy(params)


def write_params(params, outfile):
    """
    Dumps the parameter dictionary params to a JSON file

    params is a dict the parameter names, as defined with _PARAMETER_NAMES as keys
    Each key must have a dictionary with keys
        "default" : default value - make sure this is a floating point
        "fixed"   : a bool specifying if the parameter is fixed (true) or allowed to vary (false)
        "scale"   : a scale parameter used to set the step size in this dimension
        "bounds"  : An upper and lower limit on parameter values. Use null for None.
    Any extra keys are simply written as-is

    Note that JSON doesn't preserve ordering necessarily - this is enforced by read_params
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
    parameters, as well as if they should be fixed.  The JSON keys are the
    parameter names, as defined in _PARAMETER_NAMES.
    Each key must have a dictionary with keys
        "default" : default value - make sure this is a floating point
        "fixed"   : a bool specifying if the parameter is fixed (true) or allowed to vary (false)
        "scale"   : a scale parameter used to set the step size in this dimension
        "bounds"  : An upper and lower limit on parameter values. Use null for None.
    And extra keys are simply loaded as-is

    Note that the default bounds are set by the grids available for the DA
    White Dwarf atmospheres, and by reasonable plausible ranges for the other
    parameters. Don't muck with them unless you really have good reason to.

    Returns the dictionary with the parameter defaults.
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
    Converts the argparse args Namespace back into an ordered parameter
    dictionary. Assumes that the argument parser options were names
        param_value  = Value of the parameter (float or None)
        param_fix    = Bool specifying if the parameter
        param_bounds = tuple with lower limit and upper limit

    Accepts Namespace from argparse
    Returns OrderedDict of parameter keywords
    """
    kwargs = vars(args)
    out = OrderedDict()
    for param in _PARAMETER_NAMES:
        keys = (param, "{}_fix".format(param),"{}_scale".format(param), "{}_bounds".format(param))
        if not all (key in kwargs for key in keys):
            message = "Parameter {} does not have value|fixed|scale|bounds specified in argparse args".format(param)
            raise KeyError(message)
        out[param]={}
        out[param]['value']  = kwargs[param]
        out[param]['fixed']  = kwargs['{}_fix'.format(param)]
        if (out[param]['fixed'] is True) and (out[param]['value'] is None):
            message = "Parameter {} fixed but value is None - must be specified".format(param)
            raise RuntimeError(message)
        out[param]['scale']  = kwargs['{}_scale'.format(param)]
        out[param]['bounds'] = kwargs['{}_bounds'.format(param)]

    return out


def read_model_grid(grid_file=None, grid_name=None):
    """
    Read the Tlusty/Hubeny grid file (via Jay Holberg)
    NLTE grid is from an older version of Tlusty (200 vs 202 current)
    J. Holberg is working on updating the models
    """
    if grid_file is None:
        grid_file = 'TlustyGrids.hdf5'
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
        except KeyError,e:
            message = '%s\nGrid %s not found in grid_file %s. Accepted values are (%s)'%(e, grid_name, grid_file,\
                    ','.join(grids.keys()))
            raise ValueError(message)

        wave  = grid['wave'].value.astype('float64')
        ggrid = grid['ggrid'].value
        tgrid = grid['tgrid'].value
        flux  = grid['flux'].value.astype('float64')

    return grid_file, grid_name, wave, ggrid, tgrid, flux


def _read_ascii(filename, **kwargs):
    """
    Read space separated ascii file, with column names provided on first line (# optional)
    kwargs are passed along to genfromtxt
    """
    return np.recfromtxt(filename, names=True, **kwargs)


# create some aliases
# these exist so we can flesh out full functions later
# with different formats if necessary for different sorts of data
read_phot      = _read_ascii
read_spectable = _read_ascii
read_pbmap     = _read_ascii


def get_spectrum_resolution(specfile, spectable, fwhm=None):
    """
    Accepts a spectrum filename, and reads a lookup table to get the resolution
    of the spectrum spectable must contain at least two column names, specname
    and fwhm

    Note that there there's some hackish internal name fixing since T.
    Matheson's table specnames didn't match the spectrum filenames
    """
    _default_resolution = 5.0
    if fwhm is None:
        try:
            spectable = read_spectable(spectable)
        except (OSError, IOError) e:
            message = '{}\nCould not get resolution from spectable {}'.format(e, spectable)
            warnings.warn(message, RuntimeWarning)
            spectable = None
        shortfile = os.path.basename(specfile).replace('-total','')
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
    print('Using smoothing instrumental FWHM {}'.format(fwhm))
    return fwhm


def read_spec(filename, **kwargs):
    """
    Read a space separated spectrum filei, filename, with column names
    wave flux flux_err
    column names are expected on the first line
    lines beginning with # are ignored
    Removes any NaN entries (any column)
    """
    spec = _read_ascii(filename, **kwargs)
    ind = np.where((np.isnan(spec.wave)==0) & (np.isnan(spec.flux)==0) & (np.isnan(spec.flux_err)==0))
    spec = spec[ind]
    return spec


def get_phot_for_obj(objname, filename):
    """
    Accepts an object name, objname, and a lookup table filename

    The file is expected to be ascii with the first line defining column names.
    Columns names must be names of passbands (parseable by synphot) for
    magnitudes. Column names for errors in magnitudes in passband must be
    'd'+passband_name. The first column is expected to be obj for objname.
    There must be only one line per objname
    Lines beginning  with # are ignored

    Returns the photometry for objname, obj formatted as a recarray with pb, mag, mag_err

    """
    phot = read_phot(filename)
    mask = (phot.obj == objname)

    nmatch = len(phot[mask])
    if nmatch == 0:
        message = 'Got no matches for object %s in file %s. Did you want --ignorephot?'%(objname, filename)
        raise RuntimeError(message)
    elif nmatch > 1:
        message = 'Got multiple matches for object %s in file %s'%(objname, filename)
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

    out_phot = np.rec.fromarrays([pbnames, mags, errs],names='pb,mag,mag_err')
    return out_phot


def make_outdirs(dirname):
    """
    Checks if output directory exists, else creates it
    """
    print("Writing to outdir {}".format(dirname))
    if os.path.isdir(dirname):
        return

    try:
        os.makedirs(dirname)
    except OSError, e:
        message = '%s\nCould not create outdir %s for writing.'%(e,dirname)
        raise OSError(message)


def set_objname_outdir_for_specfile(specfile, outdir=None):
    """
    Accepts a spectrum filename (and optionally a preset output directory), and determines the objname
    Raises a warning if the spectra have different object names
    If output directory isn't provided, creates an output directory based on object name
    Else uses provided output directory
    Returns objname and output dirname, if directories were successfully created/exist.
    """
    basespec = os.path.basename(specfile).replace('.flm','')
    objname = basespec.split('-')[0]
    if outdir is None:
        dirname = os.path.join(os.getcwd(), "out", objname, basespec)
    else:
        dirname = outdir
    make_outdirs(dirname)
    return objname, dirname


def get_outfile(outdir, specfile, ext, check=False, redo=False):
    """
    Returns the full path to a file given outdir, specfile
    Replaces .flm at the end of specfile with extension ext (i.e. you need to include the period)
    We set the output file based on the spectrum name, since we can have multiple spectra per object
    If outdir is configured by get_objname_outdir_for_specfile, it'll take care of the objname
    """
    outfile = os.path.join(outdir, os.path.basename(specfile.replace('.flm', ext)))
    if check:
        if os.path.exists(outfile) and (not redo):
            message = "Output file %s already exists. Specify --redo to clobber."%outfile
            raise IOError(message)
    return outfile


def get_pkgfile(infile):
    """
    Returns the full path to file inside the WDmodel package
    """
    pkgfile = pkg_resources.resource_filename('WDmodel',infile)

    if not os.path.exists(pkgfile):
        message = 'Could not find package file {}'.format(pkgfile)
        raise IOError(message)
    return pkgfile


def write_fit_inputs(spec, phot, cont_model, linedata, continuumdata, outfile):
    """
    Save the spectrum, photometry (raw fit inputs) as well as a
    pseudo-continuum model and line data (visualization only inputs) to a file.

    This is intended to be called after WDmodel.fit.pre_process_spectrum() and
    WDmodel.io.get_phot_for_obj()

    This file is enough to redo the fit with the same input and
    different settings or redo the output for without redoing the fit.

    Alternatively, this file together with the HDF5 file written by
    WDmodel.fit.fit_model() with the sample chain is enough to regenerate the
    plots and output without redoing the fit.

    Accepts a recarray spectrum (spec), recarray photometry (phot), a recarray
    continuum model (cont_model), recarray Balmer line data (linedata) and
    recarray continuum line data (continuumdata) as well as outfile to
    control where the output is written.
    """

    outf = h5py.File(outfile, 'w')
    dset_spec = outf.create_group("spec")
    dset_spec.create_dataset("wave",data=spec.wave)
    dset_spec.create_dataset("flux",data=spec.flux)
    dset_spec.create_dataset("flux_err",data=spec.flux_err)

    dset_cont_model = outf.create_group("cont_model")
    dset_cont_model.create_dataset("wave",data=cont_model.wave)
    dset_cont_model.create_dataset("flux",data=cont_model.flux)

    dset_linedata = outf.create_group("linedata")
    dset_linedata.create_dataset("wave",data=linedata.wave)
    dset_linedata.create_dataset("flux",data=linedata.flux)
    dset_linedata.create_dataset("flux_err",data=linedata.flux_err)
    dset_linedata.create_dataset("line_mask",data=linedata.line_mask)

    dset_continuumdata = outf.create_group("continuumdata")
    dset_continuumdata.create_dataset("wave",data=continuumdata.wave)
    dset_continuumdata.create_dataset("flux",data=continuumdata.flux)
    dset_continuumdata.create_dataset("flux_err",data=continuumdata.flux_err)

    if phot is not None:
        dset_phot = outf.create_group("phot")
        dt = phot.pb.dtype.str.lstrip('|')
        dset_phot.create_dataset("pb", data=phot.pb, dtype=dt)
        dset_phot.create_dataset("mag",data=phot.mag)
        dset_phot.create_dataset("mag_err",data=phot.mag_err)

    outf.close()


def read_fit_inputs(input_file):
    """
    Read the saved HDF5 input_file and return recarrays of the contents
    input files are expected to contain at least 4 groups, with the 5th optional
    The groups, and the datasets they must have are
        spec
            wave, flux, flux_err
        cont_model
            wave, flux
        linedata
            wave, flux, flux_err, line_mask
        continuumdata
            wave, flux, flux_err
        [phot]
            pb, mag, mag_err

    Returns a tuple of recarrays
        spec, cont_model, linedata, continuumdata, phot[=None if absent]
    """
    d = h5py.File(input_file, mode='r')

    try:
        spec_wave = d['spec']['wave'].value
        spec_flux = d['spec']['flux'].value
        spec_ferr = d['spec']['flux_err'].value
        spec = np.rec.fromarrays([spec_wave, spec_flux, spec_ferr], names='wave,flux,flux_err')

        cmod_wave = d['cont_model']['wave'].value
        cmod_flux = d['cont_model']['flux'].value
        cont_model = np.rec.fromarrays([cmod_wave, cmod_flux], names='wave,flux')

        line_wave = d['linedata']['wave'].value
        line_flux = d['linedata']['flux'].value
        line_ferr = d['linedata']['flux_err'].value
        line_mask = d['linedata']['line_mask'].value
        linedata = np.rec.fromarrays([line_wave, line_flux, line_ferr,line_mask], names='wave,flux,flux_err,line_mask')

        cont_wave = d['continuumdata']['wave'].value
        cont_flux = d['continuumdata']['flux'].value
        cont_ferr = d['continuumdata']['flux_err'].value
        continuumdata = np.rec.fromarrays([cont_wave, cont_flux, cont_ferr], names='wave,flux,flux_err')
    except KeyError as e:
        message = '{}\nCould not load all arrays from input file {}'.format(e, input_file)
        raise IOError(message)

    phot = None
    if 'phot' in d.keys():
        try:
            pb  = d['phot']['pb'].value
            mag = d['phot']['mag'].value
            mag_err = d['phot']['mag_err'].value
            phot = np.rec.fromarrays([pb, mag, mag_err], names='pb,mag,mag_err')
        except KeyError as e:
            message = '{}\nFailed to restore photometry from input file {} though group exists'.format(e, input_file)
            warnings.warn(message, RuntimeWarning)
            phot = None
    return spec, cont_model, linedata, continuumdata, phot


def read_mcmc(input_file):
    """
    Read the saved HDF5 chain_file and return samples, sample probabilities and param names

    Returns a tuple of arrays
        param_names, samples, samples_lnprob
    """
    d = h5py.File(input_file, mode='r')

    try:
        samples = d['chain']['position'].value
        samples_lnprob = d['chain']['lnprob'].value
        param_names = d['chain']['names'].value
    except KeyError as e:
        message = '{}\nCould not load all arrays from input file {}'.format(e, input_file)
        raise IOError(message)

    return param_names, samples, samples_lnprob


def write_spectrum_model(spec, model_spec, outfile):
    """
    Write the spectrum and the model spectrum and residuals to outfile
    Accepts
        spec: recarray spectrum (wave, flux, flux_err)
        model_spec: recarray spectrum (wave, flux)
        outfile: output filename
    """
    out = (spec.wave, spec.flux, spec.flux_err, model_spec.flux, spec.flux-model_spec.flux)
    out = np.rec.fromarrays(out, names='wave,flux,flux_err,model_flux,res_flux')
    with open(outfile, 'w') as f:
        f.write(rec2txt(out, precision=8)+'\n')
    print "Wrote spec model file {}".format(outfile)


def write_phot_model(phot, model_mags, outfile):
    """
    Write the photometry, model photometry and residuals to outfile
    Accepts
        phot: recarray photometry (pb, mag, mag_err)
        model_mags: recarray model photometry (pb, mag)
        outfile: output filename
    """
    out = (phot.pb, phot.mag, phot.mag_err, model_mags.mag, phot.mag-model_mags.mag)
    out = np.rec.fromarrays(out, names='pb,mag,mag_err,model_mag,res_mag')
    with open(outfile, 'w') as f:
        f.write(rec2txt(out, precision=6)+'\n')
    print "Wrote phot model file {}".format(outfile)


def write_full_model(full_model, mu, outfile):
    """
    Write the full SED model to outfile
    Accepts
        full_model: recarray SED model (wave, flux)
        mu: Model normalization from photometry
        outfile: output filename
    """
    full_model.flux*=(10**(-0.4*mu))
    outf = h5py.File(outfile, 'w')
    dset_model = outf.create_group("model")
    dset_model.create_dataset("wave",data=full_model.wave)
    dset_model.create_dataset("flux",data=full_model.flux)
    outf.close()
    print "Wrote full model file {}".format(outfile)
