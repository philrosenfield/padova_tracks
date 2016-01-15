import argparse
import sys
import numpy as np
from StringIO import StringIO


def parse_plist(plist):
    """Parse dbert files grep'd by EEP with M= formatting cut out"""
    # genfromtext will not allow "." in column name
    ptcriname = plist[0].strip().split()[-1].replace('.', '')
    col_keys = ['age', 'logl', 'logte', 'idx', ptcriname, 'ptcriname']
    dtype = [(c, '<f8') for c in col_keys]
    dtype[-1] = (col_keys[-1], '|S15')
    pd = np.genfromtxt(StringIO(''.join(plist)), dtype=dtype)
    return pd[ptcriname], ptcriname


def format_ptcriHB(filename, nptcri=5):
    """ reformat ptcri*.dat.HB (from dbert/) file to a ptcri*.dat file (from data/) """
    outfile = filename.replace('.HB', '').replace('ptcri_', 'ptcri_hb_')
    with open(filename, 'r') as inp:
        lines = inp.readlines()

    # grep EEPs
    lines = [l for l in lines if 'He' in l]

    # split by EEP
    plists = [lines[i::nptcri] for i in range(nptcri)]

    ptcrinames = []
    for plist in plists:
        # first EEP has different formatting:
        # e.g., after EEP: M= 0.49500000 npt=771  kind=1.0000000
        if 'M' in plist[0]:
            metad = [l.split('M')[1] for l in plist]
            # position 1 because "=" would be poisiton 0
            mass = np.array([m.strip().split()[1] for m in metad], dtype=float)
            kind = np.array([float(m.strip().split('=')[-1]) for m in metad], dtype=int)
            data = np.column_stack((np.arange(len(mass)) + 1, mass, kind))
            plist = [l.split('M')[0]+'\n' for l in plist]

        pdata, ptcriname = parse_plist(plist)
        data = np.column_stack((data, pdata))
        ptcrinames.append(ptcriname)

    header = 'i mass kind' + ' '.join(ptcrinames)
    fmt = '%i %f %i ' + ' '.join('%i' * len(ptcrinames))

    np.savetxt(outfile, data, header=header, fmt=fmt)
    print('wrote {}'.format(outfile))


def main(argv):
    parser = argparse.ArgumentParser(description="convert a dbert HB file to a data HB file")

    parser.add_argument('filename', type=str, nargs='*', help='ptcri*.dat.HB files to work on')

    args = parser.parse_args(argv)

    [format_ptcriHB(f) for f in args.filename]

if __name__ == "__main__":
    main(sys.argv[1:])
