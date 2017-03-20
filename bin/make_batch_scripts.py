#!/usr/bin/env python
import sys
import os
import glob
import WDmodel.io

def addnewlines(l):
    return map(lambda x:x+'\n', l)

def main():
    lines = ["#!/bin/bash",
            "#SBATCH -n 32",
            "#SBATCH -N 1",
            "#SBATCH --contiguous",
            "#SBATCH -t 6-00:00",
            "#SBATCH -p general",
            "#SBATCH --mem-per-cpu=1000",
            "#SBATCH --mail-type=ALL",
            "#SBATCH --mail-user=gnarayan.work+odyssey@gmail.com"]

    phottable = WDmodel.io.read_phot('data/photometry/WDphot_C22.dat')

    specfiles = glob.glob('data/spectroscopy/*-total*flm')
    usedfiles = []

    for obj in phottable.obj:
        filepattern = "{}-*-total*.flm".format(obj)
        specpattern = os.path.join('data','spectroscopy',filepattern)
        specfiles = glob.glob(specpattern)

        for specfile in specfiles:
            print specfile
            usedfiles.append(specfile)
            outlines = []
            objname, outdir = WDmodel.io.set_objname_outdir_for_specfile(specfile)
            stdout_file = WDmodel.io.get_outfile(outdir, specfile, '.stdout')
            stderr_file = WDmodel.io.get_outfile(outdir, specfile, '.stderr')
            script_file = WDmodel.io.get_outfile("scripts", specfile,'.sh')
            outlines.append("#SBATCH -o {}".format(stdout_file))
            outlines.append("#SBATCH -e {}".format(stderr_file))
            outlines.append("mpirun -np 32 ./fit_WDmodel.py mpil --specfile {} --trimspec 3700 5200 --covtype ExpSquared --photfile data/photometry/WDphot_C22.dat --tau=5000 --tau_fix=True --tau_bounds 1000 10000 --discard=10 --phot_dispersion 0.001".format(specfile))
            out = addnewlines((lines+outlines))
            with open(script_file,'w') as f:
                f.writelines(out)

    leftfiles = set(specfiles) - set(usedfiles)
    for specfile in leftfiles:
        print specfile
        outlines = []
        objname, outdir = WDmodel.io.set_objname_outdir_for_specfile(specfile)
        stdout_file = WDmodel.io.get_outfile(outdir, specfile, '.stdout')
        stderr_file = WDmodel.io.get_outfile(outdir, specfile, '.stderr')
        script_file = WDmodel.io.get_outfile("scripts", specfile,'.sh')
        outlines.append("#SBATCH -o {}".format(stdout_file))
        outlines.append("#SBATCH -e {}".format(stderr_file))
        outlines.append("mpirun -np 32 ./fit_WDmodel.py mpil --specfile {} --trimspec 3700 5200 --covtype ExpSquared --ignorephot --tau=5000 --tau_fix=True --tau_bounds 1000 10000 --discard=10".format(specfile))
        out = addnewlines((lines+outlines))
        with open(script_file,'w') as f:
            f.writelines(out)

        

        






if __name__=='__main__':
    main()
