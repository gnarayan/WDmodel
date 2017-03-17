#!/usr/bin/env python
import sys
import argparse
import warnings
warnings.simplefilter('once')
import glob
import numpy as np
from matplotlib.mlab import rec2txt
import WDmodel.io


def get_options(args=None):
    """
    Get command line options for WDmodel files to print results for
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)

    parser.add_argument('--specfiles', nargs='+', required=False, \
            help="Specify spectrum to search for")
    parser.add_argument('-o', '--outdir', required=False,\
            help="Specify a custom output directory. Default is CWD+objname/ subdir")
    args = parser.parse_args(args=args)
    return args


def main(inargs=None):

    if inargs is None:
        inargs = sys.argv[1:]

    args = get_options(inargs)
    if args.specfiles is None:
        specfiles = glob.glob('data/spectroscopy/*flm')
    else:
        specfiles = args.specfiles

    out = []
    colnames = []
    colbool = False

    for specfile in specfiles:
        objname, outdir = WDmodel.io.set_objname_outdir_for_specfile(specfile, outdir=args.outdir)
        outfile = WDmodel.io.get_outfile(outdir, specfile, '_result.json')
        try:
            params = WDmodel.io.read_params(outfile)
        except IOError, e:
            message = 'Could not get results for {}({}) from outfile {}'.format(objname, specfile, outfile)
            warnings.warn(message)
            params = None
            continue
        this_out = []
        this_out.append(objname)
        this_out.append(specfile)
        if not colbool:
            colnames.append('obj')
            colnames.append('specfile')
        for param in params:
            this_out.append(params[param]['value'])
            this_out.append(params[param]['errors_pm'][0])
            this_out.append(params[param]['errors_pm'][1])
            if not colbool:
                colnames.append(param)
                colnames.append('errhi_{}'.format(param))
                colnames.append('errlo_{}'.format(param))
        colbool=True
        out.append(this_out)
    out = np.rec.fromrecords(out, names=colnames)
    out.sort()
    precision = [None, None] + [2,2,2]*6 + [5,5,5] + [2,2,2] + [4,4,4]
    print rec2txt(out, precision=precision )
    








if __name__=='__main__':
    main(sys.argv[1:])
