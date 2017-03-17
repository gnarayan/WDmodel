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
    pbs = 'F275W,F336W,F475W,F625W,F775W,F160W'
    pbs = pbs.split(',')

    for specfile in specfiles:
        objname, outdir = WDmodel.io.set_objname_outdir_for_specfile(specfile, outdir=args.outdir)
        outfile = WDmodel.io.get_outfile(outdir, specfile, '_phot_model.dat')
        try:
            phot = WDmodel.io.read_phot(outfile)
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

        for pb in pbs:
            if not colbool:
                colnames.append('{}'.format(pb))
                colnames.append('d{}'.format(pb))
                colnames.append('m{}'.format(pb))
                colnames.append('r{}'.format(pb))
            m = np.where(phot.pb == pb)[0]
            if len(phot.pb[m]) == 0:
                this_out.append(np.nan)
                this_out.append(np.nan)
                this_out.append(np.nan)
                this_out.append(np.nan)
            else:
                this_out.append(phot.mag[m][0])
                this_out.append(phot.mag_err[m][0])
                this_out.append(phot.model_mag[m][0])
                this_out.append(phot.res_mag[m][0])
        colbool=True
        out.append(this_out)
    out = np.rec.fromrecords(out, names=colnames)
    out.sort()
    precision = [None, None] + [4,4,4,4]*len(pbs)
    with open('residual_table.dat', 'w') as f:
        f.write(rec2txt(out, precision=precision )+'\n')
    








if __name__=='__main__':
    main(sys.argv[1:])
