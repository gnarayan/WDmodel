#!/usr/bin/env python

from __future__ import print_function

import sys
from fit_WDmodel import get_options
import numpy as np
import WDmodel
import WDmodel.fit
import WDmodel.pbmodel
import WDmodel.viz
import WDmodel.io


def main():
    args = get_options(sys.argv[1:])

    specfile  = args.specfile
    outdir    = args.outdir
    rvmodel   = args.reddeningmodel
    nwalkers  = args.nwalkers
    nburnin   = args.nburnin
    nprod     = args.nprod
    discard   = args.discard
    balmer    = args.balmerlines
    ndraws    = args.ndraws
    savefig   = args.savefig
    excludepb = args.excludepb

    # set the object name and create output directories
    objname, outdir = WDmodel.io.set_objname_outdir_for_specfile(specfile, outdir=outdir)

    # restore data
    input_file = WDmodel.io.get_outfile(outdir, specfile, '_inputs.hdf5')
    res = WDmodel.io.read_fit_inputs(input_file)
    spec, cont_model, linedata, continuumdata, phot = res


    # init model
    model = WDmodel.WDmodel()

    # exclude passbands that we want excluded
    pbnames = []
    if phot is not None:
        pbnames = np.unique(phot.pb)
        if excludepb is not None:
            pbnames = list(set(pbnames) - set(excludepb))

        # filter the photometry recarray to use only the passbands we want
        useind = [x for x, pb in enumerate(phot.pb) if pb in pbnames]
        useind = np.array(useind)
        phot = phot.take(useind)

        # set the pbnames from the trimmed photometry recarray to preserve order
        pbnames = list(phot.pb)

    # if we cut out out all the passbands, force mu to be fixed
    if len(pbnames) == 0:
        params['mu']['value'] = 0.
        params['mu']['fixed'] = True
        phot = None

    # get the throughput model
    pbs = WDmodel.pbmodel.get_pbmodel(pbnames, model)

    # restore params
    param_file = WDmodel.io.get_outfile(outdir, specfile, '_result.json')
    mcmc_params = WDmodel.io.read_params(param_file)

    # restore samples and prob
    chain_file = WDmodel.io.get_outfile(outdir, specfile, '_mcmc.hdf5')
    param_names, samples, samples_lnprob = WDmodel.io.read_mcmc(chain_file)

    # parse chain 
    result = WDmodel.fit.get_fit_params_from_samples(param_names, samples, samples_lnprob, mcmc_params,\
                    nwalkers=nwalkers, nprod=nprod, discard=discard)
    mcmc_params, in_samp, in_lnprob = result

    # plot the MCMC output
    WDmodel.viz.plot_mcmc_model(spec, phot, linedata,\
                objname, outdir, specfile,\
                model, cont_model, pbs,\
                mcmc_params, param_names, in_samp, in_lnprob,\
                rvmodel=rvmodel, balmer=balmer, ndraws=ndraws, savefig=savefig)

    return

if __name__ =='__main__':
    main()




