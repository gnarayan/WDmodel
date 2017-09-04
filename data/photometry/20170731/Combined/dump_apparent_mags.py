#!/usr/bin/env python
"""
This script converts the instrumental magnitudes in the
WD_cycle22_cycle20new.phot from A.  Calamida to apparent magnitudes, mapping
out the differences in the names between photometry and spectroscopy using
name_map.dat, and applying the zeropoints in ZP_all
"""

import sys
import numpy as np
from matplotlib.mlab import rec2txt
from uncertainties import ufloat

def main():
    a = np.recfromtxt('WD_cycle22_cycle20new.phot', names=True)
    pb = 'F275W,F336W,F475W,F625W,F775W,F160W'
    pb = pb.split(',')

    b = np.recfromtxt('ZP_all', names='l,ZP_10_2017,ZP_10,eZP_10,ZP_inf,eZP_inf')
    b.l *= 1000
    zppb = ['F{:.3}W'.format(str(x)) for x in b.l]

    zp = {x:y for x, y in zip(zppb, b.ZP_10)}
    zp.update({'d'+x:y for x, y in zip(zppb, b.eZP_10)})

    map = None
    with open('name_map.dat','r') as f:
        lines = f.readlines()
        map = {line.strip().split()[0]:line.strip().split()[1] for line in lines}

    d = a[1:]
    out = []
    for i, x in enumerate(d):
        this_out = []
        objid = str(x.ID)
        objid = objid.replace('-','',1).split('.')[0].split('+')[0].split('-')[0].replace('SDSS','sdss').lower()
        objid =  map.get(objid, objid)
        this_out.append(objid)
        for p in pb:
            inst_mag = x[p]
            if p == 'F160W':
                dinst_mag = x['d'+p]
            else:
                dinst_mag = x['d'+p+'1']

            z = zp[p]
            dz = zp['d'+p]
            inst = ufloat(inst_mag, dinst_mag)
            zero = ufloat(z, dz)
            mag = inst + zero
            this_out.append(mag.n)
            this_out.append(mag.s)
        out.append(this_out)
    names = ['obj']
    for p in pb:
        names.append(p)
        names.append('d'+p)
    out = np.rec.fromrecords(out, names=names)
    with open('WDphot_C22_AC_AS_combined.dat','w') as f:
        f.write(rec2txt(out, precision=5)+'\n')






if __name__=='__main__':
    sys.exit(main())
