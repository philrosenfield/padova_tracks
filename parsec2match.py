"""
Interpolate PARSEC tracks for use in MATCH.

This code calls padova_tracks:
1) Redefines some equivalent evolutionary points (EEPs) from PARSEC
2) Interpolates the tracks so they all have the same number of points between
   defined EEPs.
"""

import argparse
import os
import sys

from .fileio import load_input, tfm_indict
from .match import TracksForMatch
from .utils import add_version_info


def parsec2match(infile, loud=False):
    """Do an entire set and make the plots."""
    if loud:
        print('setting prefixs')
    indict = load_parsec2match_inp(infile)

    if indict['debug']:
        loud = True

    prefixs = indict['prefixs']

    for prefix in prefixs:
        if loud:
            print('Current mix: {}'.format(prefix))
        indict['prefix'] = prefix

        if loud:
            print('loading Tracks')
        tfm = TracksForMatch(**indict)

        if loud:
            print('defining eeps')
        define_eeps(tfm)

        # do the match interpolation (produce match output files)
        if indict['do_interpolation']:
            if loud:
                print('doing match interpolation')
            indict['flag_dict'] = tfm.match_interpolation()

    return indict


def load_parsec2match_inp(infile):
    """
    Find which prefixes (Z, Y mixes) to run.

    based on inputs.prefix or inputs.prefixs.
    """
    # Prefixs = track sub directory names
    indict = load_input(infile, default_dict=tfm_indict())
    prefs = indict['prefixs']

    assert prefs is not None, 'prefixs (track subdirectories) not set'

    # tracks location
    tracks_dir = indict['tracks_dir']

    if prefs == 'all':
        # find all dirs in tracks dir skip .DS_Store and crap
        prefixs = [d for d in os.listdir(tracks_dir)
                   if os.path.isdir(os.path.join(tracks_dir, d)) and
                   not d.startswith('.')]  # FU .DS_Store
    else:
        # some subset listed in the input file (seperated by comma)
        prefixs = prefs
        if isinstance(prefixs, str):
            prefixs = [prefs]

    assert type(prefixs) == list, 'prefixs must be a list'
    indict['prefixs'] = prefixs
    return indict


def define_eeps(tfm, hb=False):
    """Add the ptcris to the tracks."""
    [tfm.define_eep_stages(track) for track in tfm.tracks]
    # line = ''

    # define the eeps
    # line = ' '.join([tfm.define_eep_stages(track) for track in tfm.tracks])

    # save_ptcri(line, loc=tfm.tracks_dir.replace('tracks', 'data'),
    #            prefix=tfm.prefix)
    return tfm


def main(argv):
    """Main caller for parsec2match."""
    parser = argparse.ArgumentParser(description="parsec2match")

    parser.add_argument('-v', '--loud', action='store_true',
                        help='verbose')

    parser.add_argument('infile', type=str,
                        help='input file')

    parser.add_argument('-i', '--input', action='store_true',
                        help='print an input file')

    args = parser.parse_args(argv)

    if args.input:
        indict = tfm_indict()
        # if you are lazy and don't edit inputs/tracks4match.json
        for k, v in indict.items():
            if k.endswith('dir'):
                indict[k] = os.getcwd()
        indict['prefixs'] = [l for l in os.listdir('.') if os.path.isdir(l)]

        lines = ''.join(['{0:s}   {1!s}\n'.format(k, v)
                         for k, v in indict.items()])

        if not os.path.isfile(args.infile):
            with open(args.infile, 'w') as inp:
                inp.write(lines)
        else:
            print(lines)
        return
    indict = parsec2match(args.infile, loud=args.loud)

    fname = add_version_info(args.infile)
    os.system('mv {} {}'.format(fname,
                                os.path.join(indict['tracks_dir'], 'logs')))


if __name__ == '__main__':
    main(sys.argv[1:])
