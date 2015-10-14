from __future__ import print_function
import argparse
import numpy as np
import os
import sys

from ResolvedStellarPops import fileio
from eep.critical_point import Eep


def prepare_makemod(prefixs=None, tracks_dir=None, sub=None):
    ext = '.PMS'
    ext = '.DAT'
    if sub is not None:
        prefixs = os.listdir(sub)
        tracks_dir = sub
        ext = '.dat'

    zsun = 0.02
    #allzs = [p.split('Z')[1].split('_')[0] for p in prefixs]
    allzs = [p.split('Z')[1].split('Y')[0] for p in prefixs]
    zs = np.unique(np.array(allzs, dtype=float))
    prefixs = np.array(prefixs)[np.argsort(allzs)]

    # limits of metallicity grid
    modelIZmin = np.int(np.ceil(np.log10(np.min(zs) / zsun) * 10))
    modelIZmax = np.int(np.floor(np.log10(np.max(zs) / zsun) * 10))

    # metallicities
    zs_str = ','.join(np.array(zs, dtype=str))
    # max number of characters needed for directory names
    FNZ = np.max([len(p) for p in prefixs]) + 1.

    # directories
    prefix_str = '\"%s\"' % '\",\"'.join(prefixs)

    all_masses = np.array([])
    # masses N x len(zs) array
    # guess for limits of mbol, log_te grid
    mod_l0 = 1.0
    mod_l1 = 1.0
    mod_t0 = 4.0
    mod_t1 = 4.0
    for p in prefixs:
        this_dir = os.path.join(tracks_dir, p)
        track_names = fileio.get_files(this_dir, '*{}'.format(ext))
        masses = np.array([os.path.split(t)[1].split('M')[1].split('.{}'.format(ext[1]))[0]
                           for t in track_names if not 'hb' in t.lower() and not 'add' in t.lower()], dtype=float)
        all_masses = np.append(all_masses, masses)
        for t in track_names:
            data = np.genfromtxt(t, names=['logte', 'mbol'], usecols=(2,3))
            mod_t0 = np.min([mod_t0, np.min(data['logte'])])
            mod_t1 = np.max([mod_t1, np.min(data['logte'])])
            mod_l0 = np.min([mod_l0, np.min(data['mbol'])])
            mod_l1 = np.max([mod_l1, np.min(data['mbol'])])
        # find a common low and high mass at all Z.
        umasses = np.sort(np.unique(all_masses))
        min_mass = umasses[0]
        max_mass = umasses[-1]
        imin = 0
        imax = -1
        while len(np.nonzero(all_masses == min_mass)[0]) != len(zs):
            imin += 1
            min_mass = umasses[imin]
    
        while len(np.nonzero(all_masses == max_mass)[0]) != len(zs):
            imax -= 1
            max_mass = umasses[imax]
    
        if imax == -1 and imin == 0:
            masses = umasses
        elif imax == -1:
            masses = umasses[imin - 1::]
        elif imin ==0:
            masses = umasses[imin: imax + 1]
        else:
            masses = umasses[imin - 1: imax + 1]
    
        masses_str = ','.join(map(str, masses))
    
        eep = Eep()
        mdict = {'npt_low': eep.nlow,
                 'npt_hb': eep.nhb,
                 'npt_tr': eep.ntot - eep.nms - eep.nhb,
                 'npt_ms': eep.nms,
                 'masses_str': masses_str,
                 'prefix_str': prefix_str,
                 'FNZ': FNZ,
                 'zs_str': zs_str,
                 'modelIZmax': modelIZmax,
                 'modelIZmin': modelIZmin,
                 'zsun': zsun,
                 'mod_l0': mod_l0,
                 'mod_l1': mod_l1,
                 'mod_t0': mod_t0,
                 'mod_t1': mod_t1}
    
        if sub is None:
            fname = 'makemod_%s_%s.txt' % (tracks_dir.split('/')[-2], p.split('_Z')[0])
        else:
            fname = 'makemod_%s.txt' % (sub)
        with open(fname, 'w') as out:
            out.write(makemod_fmt() % mdict)
            out.write('\nmay need to adjust for rounding error:\n')
            out.write(''.join(('mod_l0: %.4f \n' % mod_l0,
                               'mod_l1: %.4f \n' % mod_l1,
                               'mod_t0: %.4f \n' % mod_t0,
                               'mod_t1: %.4f \n' % mod_t1)))

def makemod_fmt():
    return """
const double Zsun = %(zsun).2f;
const double Z[] = {%(zs_str)s};
static const int NZ = sizeof(Z)/sizeof(double);
const char FNZ[NZ][%(FNZ)i] = {%(prefix_str)s};

const double M[] = {%(masses_str)s};
static const int NM = sizeof(M)/sizeof(double);

// limits of age and metallicity coverage
static const int modelIZmin = %(modelIZmin)i;
static const int modelIZmax = %(modelIZmax)i;

// number of values along isochrone (in addition to logTe and Mbol)
static const int NHRD=3;

// range of Mbol and logTeff
static const double MOD_L0 = %(mod_l0).2f;
static const double MOD_LF = %(mod_l1).2f;
static const double MOD_T0 = %(mod_t0).2f;
static const double MOD_TF = %(mod_t1).2f;

static const int ML0 = 9; // number of mass loss steps
//static const int ML0 = 0; // number of mass loss steps
static const double ACC = 3.0; // CMD subsampling

// Number of points along tracks
static const int NPT_LOW = %(npt_low)i; // low-mass tracks points
static const int NPT_MS = %(npt_ms)i; // MS tracks points
static const int NPT_TR = %(npt_tr)i; // transition MS->HB points
static const int NPT_HB = %(npt_hb)i; // HB points

--------------------------
cd ..; make PARSEC; cd PARSEC; ./makemod -sub=????
Move current data into a safe place
Remember there are two instances of filename formats hard coded, after
that the value for mass is found by a character offset.
"""

def main(argv):
    parser = argparse.ArgumentParser(description="Prepare header for makemod.cpp")
    
    parser.add_argument('sub', type=str,
                        help='subdirectory with match track dirs (not mods)')

    args = parser.parse_args(argv)
    
    prepare_makemod(sub=args.sub)


if __name__ == "__main__":
    main(sys.argv[1:])    
