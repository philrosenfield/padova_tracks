'''
Interpolate PARSEC tracks for use in MATCH.
This code calls padova_tracks:
1) Redefines some equivalent evolutionary points (EEPs) from PARSEC
2) Interpolates the tracks so they all have the same number of points between
   defined EEPs.
'''
from __future__ import print_function, division
import argparse
import numpy as np
import os
import sys

from .fileio import load_input, tfm_indict
from .match import TracksForMatch
from .prepare_makemod import prepare_makemod
from .utils import add_version_info

import logging
logger = logging.getLogger()


def parsec2match(infile, loud=False):
    '''do an entire set and make the plots'''
    if loud:
        print('setting prefixs')
    indict = load_parsec2match_inp(infile)

    if indict['debug']:
        import pdb
        pdb.set_trace()

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
        define_eeps(tfm, hb=False, diag_plot=indict['diag_plot'])
        if indict['both']:
            define_eeps(tfm, hb=True, diag_plot=indict['diag_plot'])

        # do the match interpolation (produce match output files)
        if indict['do_interpolation']:
            if loud:
                print('doing match interpolation')
            indict['flag_dict'] = tfm.match_interpolation(hb=False)
            if indict['both']:
                indict['flag_dict_hb'] = tfm.match_interpolation(hb=True)

            # check the match interpolation
            # if loud:
            #    print('checking interpolation')
            # CheckMatchTracks(indict)

    print('DONE')
    return prefixs


def load_parsec2match_inp(infile):
    '''
    find which prefixes (Z, Y mixes) to run based on inputs.prefix or
    inputs.prefixs.
    '''
    # Prefixs = track sub directory names
    indict = load_input(infile, default_dict=tfm_indict())
    prefs = indict['prefixs']

    if prefs is None:
        print('prefix or prefixs not set')
        sys.exit(2)

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


def define_eeps(tfm, hb=False, save_p2m=True, diag_plot=False):
    '''add the ptcris to the tracks'''
    # assign eeps track.iptcri and track.sptcri

    if hb:
        track_str = 'hbtracks'
        defined = tfm.eep.eep_list_hb
        filename = 'define_eeps_hb_%s.log'
    else:
        track_str = 'tracks'
        defined = tfm.eep.eep_list
        filename = 'define_eeps_%s.log'

    # define the eeps
    tracks = [tfm.define_eep_stages(track)
              for track in tfm.__getattribute__(track_str)]

    p2m_file = tfm.save_ptcri(tracks)

    if diag_plot:
        tfm.diag_plots(tracks, hb=inps.hb, pat_kw={'ptcri': p2m_file},
                       extra='p2m', plot_dir=tfm.plot_dir)

    # write log file
    info_file = os.path.join(inputs.log_dir, filename % tfm.prefix.lower())
    with open(info_file, 'w') as out:
        for t in tracks:
            if t.flag is not None:
                out.write('# %s: %s\n' % (t.name, t.flag))
                continue
            out.write('# %.3f\n' % t.mass)
            if t.flag is not None:
                out.write(t.flag)
            else:
                for ptc in defined:
                    try:
                        out.write('%s: %s\n' % (ptc, t.info[ptc]))
                    except:
                        out.write('%s: %s\n' % (ptc, 'Copied from Sandro'))
    return tfm


def call_prepare_makemod(inputs):
    prefixs = get_track_subdirectories(inputs, harsh=False)
    prepare_makemod(prefixs, inputs.tracks_dir)


def main(argv):
    parser = argparse.ArgumentParser(description="parsec2match")

    parser.add_argument('-v', '--loud', action='store_true',
                        help='verbose')

    parser.add_argument('infile', type=str,
                        help='input file')

    args = parser.parse_args(argv)

    indict = parsec2match(args.infile, loud=args.loud)

    if indict['prepare_makemod']:
        call_prepare_makemod(indict)

    fname = add_version_info(args.infile)
    os.system('mv {} {}'.format(fname,
                                os.path.join(indict['tracks_dir'], 'logs')))


if __name__ == '__main__':
    main(sys.argv[1:])
