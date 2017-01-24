import os
import numpy as np
import pkg_resources
import h5py

def read_model_grid(grid_file=None, grid_name=None):
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
            message = 'Grid %s not found in grid_file %s. Accepted values are (%s)'%(grid_name, grid_file,\
                    ','.join(grids.keys()))
            raise ValueError(message)
    
        wave  = grid['wave'].value.astype('float64')
        ggrid = grid['ggrid'].value
        tgrid = grid['tgrid'].value
        flux  = grid['flux'].value.astype('float64')
    
    return grid_file, grid_name, wave, ggrid, tgrid, flux


def read_spec(filename):
    """
    Read spectrum from file routine
    Col names assumed to be on the first line (wave, flux, flux_err)
    Types set to float64
    """
    spec = np.recfromtxt(filename, names=True, dtype='float64,float64,float64')
    return spec


def read_phot(filename):
    """
    Read photometry from file - expects to have columns mag_aper magerr_aper and pb 
    Extra columns other than these three are fine
    """
    phot = np.recfromtxt(filename, names=True)
    return phot


def read_spectable(filename):
    """
    Read spectrum resolution from a file to set instrumental smoothing
    """
    spectable = np.recfromtxt(filename, names=True)
    return spectable


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
        message = '%s\nCould not create outdir %s for writing.'
        raise OSError(message)


def set_objname_outdir_for_specfiles(specfiles, outdir=None):
    """
    Accepts a list of specfiles (and optionally a preset output directory), and determines the objname
    Raises a warning if the spectra have different object names
    If output directory isn't provided, creates an output directory based on object name
    Else uses provided output directory
    Returns objname and output dirname, if directories were sucessfully created/exist.
    """
    obj = []
    for i, specfile in enumerate(specfiles):
        objname = os.path.basename(specfile).split('-')[0]
        obj.append(objname)
    obj = set(obj)
    if len(obj) > 1:
        message = "Objects are inconsistently named. Are you sure these are spectra of the same object?"
        warnings.warn(message, RuntimeWarning)
    objname = obj.pop()

    if outdir is None:
        dirname = os.path.join(os.getcwd(), "out", objname)
    else:
        dirname = outdir
    make_outdirs(dirname)
    return objname, dirname
