import os
import warnings
import numpy as np
import pkg_resources
import h5py


def read_model_grid(grid_file=None, grid_name=None):
    """
    Read the Tlusty/Hubeny grid file (via Jay Holberg)
    NLTE grid is from an older version of Tlusty (200 vs 202 current)
    J. Holberg is working on updating the models
    """
    if grid_file is None:
        grid_file = pkg_resources.resource_filename('WDmodel','TlustyGrids.hdf5')

    if not os.path.exists(grid_file):
        message = 'Could not find grid file %s'%grid_file
        raise IOError(message)

    if grid_name is None:
        grid_name = "default"

    with h5py.File(grid_file, 'r') as grids:
        # the original IDL SAV file Tlusty grids were annoyingly broken up by wavelength
        # this was because the authors had different wavelength spacings
        # since they didn't feel that the contiuum needed much spacing anyway
        # and "wanted to save disk space"
        # and then their old IDL interpolation routine couldn't handle the variable spacing
        # so they broke up the grids
        # So really, the original IDL SAV files were annoyingly broken up by wavelength because old dudes
        # We have concatenated these "large" arrays because we don't care about disk space
        # This grid is called "default", but the orignals also exist 
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


def get_spectrum_resolution(specfile, smooth=None):
    """
    Accepts a spectrum filename, and reads a lookup table to get the resolution of the spectrum
    """
    _default_resolution = 8.0
    if smooth is None:                                                                                                  
        spectable = read_spectable('data/spectable_resolution.dat')                                                     
        shortfile = os.path.basename(specfile).replace('-total','')                                                     
        if shortfile.startswith('test'):                                                                                
            message = 'Spectrum filename indicates this is a test - using default resolution'                       
            warnings.warn(message, RuntimeWarning)                                                                      
            smooth = _default_resolution
        else:                                                                                                           
            mask = (spectable.specname == shortfile)                                                                    
            if len(spectable[mask]) != 1:                                                                               
                message = 'Could not find an entry for this spectrum in the spectable file - using default resolution'
                warnings.warn(message, RuntimeWarning)                                                                  
                smooth = _default_resolution
            else:                                                                                                       
                smooth = spectable[mask].fwhm                                                                           
    else:                                                                                                               
        message = 'Smoothing factor specified on command line - overriding spectable file'                               
        warnings.warn(message, RuntimeWarning)                                                                          
    print('Using smoothing factor %.2f'%smooth) 
    return smooth


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

    Returns the photometry for objname, obj 

    """
    phot = read_phot(filename)
    mask = (phot.obj == objname)
    if len(phot[mask]) != 1:
        message = 'Got no or multiple matches for object %s in file %s'%(objname, filename)
        raise RuntimeError(message)
    return phot[mask]


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
    objname = os.path.basename(specfile).replace('.flm','').split('-')[0]
    if outdir is None:
        dirname = os.path.join(os.getcwd(), "out", objname)
    else:
        dirname = outdir
    make_outdirs(dirname)
    return objname, dirname