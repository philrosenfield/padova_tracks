'''
Interpolate PARSEC tracks for use in MATCH.
This code calls padova_tracks:
1) Redefines some equivalent evolutionary points (EEPs) from PARSEC
2) Interpolates the tracks so they all have the same number of points between
   defined EEPs.
'''
from __future__ import print_function
from copy import deepcopy
import numpy as np
import os
import sys

from ResolvedStellarPops import fileio
from eep.define_eep import DefineEeps
from eep.critical_point import critical_point, Eep
from match import TracksForMatch
from match import CheckMatchTracks
from prepare_makemod import prepare_makemod

import logging
logger = logging.getLogger()

__all__ = ['initialize_inputs']


def add_version_info(input_file):
    """Copy the input file and add the git hash and time the run started."""
    from time import localtime, strftime

    # create info file with time of run
    now = strftime("%Y-%m-%d %H:%M:%S", localtime())
    fname = fileio.replace_ext(input_file, '.info')
    with open(fname, 'w') as out:
        out.write('parsec2match run started %s \n' % now)
        out.write('ResolvedStellarPops git hash: ')

    # the best way to get the git hash?
    here = os.getcwd()
    rsp_home = os.path.split(os.path.split(fileio.__file__)[0])[0]
    os.chdir(rsp_home)
    os.system('git rev-parse --short HEAD >> %s' % os.path.join(here, fname))
    os.chdir(here)

    # add the input file
    os.system('cat %s >> %s' % (input_file, fname))
    return

def parsec2match(input_obj, loud=False):
    '''do an entire set and make the plots'''
    if loud:
        print('setting prefixs')
    prefixs = set_prefixs(input_obj)

    for prefix in prefixs:
        print('Current mix: %s' % prefix)
        inps = set_outdirs(input_obj, prefix)

        if loud:
            print('loading ptcri')

        inps = load_ptcri(inps)

        if loud:
            print('loading Tracks')
        tfm = TracksForMatch(inps)

        if not inps.from_p2m:
            # find the parsec2match eeps for these tracks.
            #if not inps.hb:
            #    ptcri_file = load_ptcri(inps, find=True, from_p2m=True)

            if not inps.overwrite_ptcri and os.path.isfile(ptcri_file):
                print('not overwriting %s' % ptcri_file)
            else:
                if loud:
                    print('defining eeps')
                tfm = define_eeps(tfm, inps)

            if inps.diag_plot and not inps.do_interpolation:
                # make diagnostic plots using new ptcri file
                inps.ptcri_file = None
                inps.from_p2m = True
                inps = load_ptcri(inps)
                if loud:
                    print('making parsec diag plots')

                if inps.hb:
                    pat_kw = {'ptcri': inps.ptcri_hb}
                else:
                    pat_kw = {'ptcri': inps.ptcri}
                    #tfm.diag_plots(xcols=xcols, pat_kw=pat_kw)

                tfm.diag_plots(tfm.hbtracks, hb=inps.hb, pat_kw=pat_kw,
                               extra='parsec', plot_dir=inps.plot_dir)

        # do the match interpolation (produce match output files)
        if inps.do_interpolation:
            # force reading of my eeps
            inps.ptcri_file = None
            inps.from_p2m = True
            inps = load_ptcri(inps)

            if loud:
                print('doing match interpolation')
            inps.flag_dict = tfm.match_interpolation(inps)

            # check the match interpolation
            if loud:
                print('checking interpolation')
            CheckMatchTracks(inps)

    print('DONE')
    return prefixs


def set_prefixs(inputs, harsh=True):
    '''
    find which prefixes (Z, Y mixes) to run based on inputs.prefix or
    inputs.prefixs.
    '''
    # tracks location
    tracks_dir = inputs.tracks_dir
    if inputs.prefixs == 'all':
        # find all dirs in tracks dir skip .DS_Store and crap
        prefixs = [d for d in os.listdir(tracks_dir)
                   if os.path.isdir(os.path.join(tracks_dir, d))
                   and not d.startswith('.')]
    elif inputs.prefixs is not None:
        # some subset listed in the input file (seperated by comma)
        prefixs = inputs.prefixs
    else:
        if inputs.prefix is None:
            print('prefix or prefixs not set')
            sys.exit(2)
        # just do one
        prefixs = [inputs.prefix]

    if harsh:
        del inputs.prefixs
    assert type(prefixs) == list, 'prefixs must be a list'
    return prefixs


def load_ptcri(inputs, find=False, from_p2m=False):
    '''
    load the ptcri file, either sandro's or mine
    if find is True, just return the file name, otherwise, return inputs with
    ptcri and ptcri_file attributes set.

    if from_p2m is True, force find/load my ptcri file regardless
    of inputs.from_p2m value.
    '''

    # find the ptcri file
    ptcri_file = None
    ptcri_file_hb = None
    sandro = True
    search_term = 'pt*'
    if inputs.from_p2m or from_p2m:
        sandro = False
        search_term = 'p2m*'

    search_term += '%sY*dat' % inputs.prefix.split('Y')[0]
    if inputs.ptcri_file is not None:
        ptcri_file = inputs.ptcri_file
    else:
        ptcri_files = fileio.get_files(inputs.ptcrifile_loc, search_term)
        if inputs.hb:
            ptcri_file_hb, = [p for p in ptcri_files if 'hb' in p]
        try:
            ptcri_file, = [p for p in ptcri_files if not 'hb' in p]
        except:
            pass
    #assert os.path.isfile(ptcri_file), 'ptcri file not found.'
    if find:
        return ptcri_file, ptcri_file_hb
    else:
        if ptcri_file_hb is not None:
            inputs.ptcri_file_hb = ptcri_file_hb
            inputs.ptcri_hb = critical_point(inputs.ptcri_file_hb,
                                             sandro=sandro)
            inputs.ptcri_hb.name = 'ptcri_hb_%s.dat' % inputs.prefix
        if ptcri_file is not None:
            inputs.ptcri_file = ptcri_file
            inputs.ptcri = critical_point(inputs.ptcri_file, sandro=sandro)
            inputs.ptcri.base = inputs.ptcrifile_loc
            inputs.ptcri.name = 'ptcri_%s.dat' % inputs.prefix
        return inputs


def define_eeps(tfm, inputs):
    '''add the ptcris to the tracks'''
    # assign eeps track.iptcri and track.sptcri
    de = DefineEeps()
    crit_kw = {'plot_dir': inputs.plot_dir,
               'diag_plot': inputs.track_diag_plot,
               'debug': inputs.debug,
               'hb': inputs.hb}

    if inputs.hb:
        track_str = 'hbtracks'
        defined = Eep().eep_list_hb
        filename = 'define_eeps_hb_%s.log'
        ptcri = inputs.ptcri_hb
    else:
        track_str = 'tracks'
        #defined = inputs.ptcri.please_define
        defined = Eep().eep_list
        filename = 'define_eeps_%s.log'
        ptcri = inputs.ptcri
    # load critical points calls de.define_eep
    tracks = [de.load_critical_points(track, ptcri=ptcri, **crit_kw)
              for track in tfm.__getattribute__(track_str)]

    # write log file
    info_file = os.path.join(inputs.log_dir, filename % inputs.prefix.lower())
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
                    t.info[ptc] = ''
                [out.write('%s: %s\n' % (ptc, t.info[ptc])) for ptc in defined]
    if not inputs.from_p2m:
        inputs.ptcri.save_ptcri(tracks, hb=inputs.hb)

    tfm.__setattr__(track_str, tracks)
    return tfm


def set_outdirs(input_obj, prefix):
    '''
    set up and ensure the directories for output and plotting
    diagnostic plots: tracks_dir/diag_plots/prefix
    match output: tracks_dir/match/prefix
    define_eep and match_interp logs: tracks_dir/logs/prefix
    Parameters
    ----------
    input_obj : rsp.fileio.InputParameters object
        must have attrs: tracks_dir, prefix, plot_dir, outfile_dir
        the final two can be 'default'

    Returns
    -------
    new_inputs : A copy of input_obj with plot_dir, log_dir, outfile_dir set
    '''
    new_inputs = deepcopy(input_obj)
    new_inputs.prefix = prefix

    if input_obj.plot_dir == 'default':
        new_inputs.plot_dir = os.path.join(input_obj.tracks_dir,
                                           'diag_plots',
                                           new_inputs.prefix)

    if input_obj.outfile_dir == 'default':
        new_inputs.outfile_dir = os.path.join(input_obj.tracks_dir,
                                              'match',
                                              new_inputs.prefix)
        fileio.ensure_dir(new_inputs.outfile_dir)
        new_inputs.log_dir = os.path.join(input_obj.tracks_dir, 'logs')

    for d in [new_inputs.plot_dir, new_inputs.outfile_dir, new_inputs.log_dir]:
        fileio.ensure_dir(d)

    return new_inputs


def initialize_inputs():
    '''
    Load default inputs, the eep lists, and number of equally spaced points
    between eeps. Input file will overwrite these. These should be all possible
    input options.
    '''
    input_dict = {'track_search_term': '*F7_*PMS',
                  'hbtrack_search_term':'*F7_*PMS.HB',
                  'from_p2m': False,
                  'masses': None,
                  'do_interpolation': True,
                  'debug': False,
                  'hb': True,
                  'prefix': None,
                  'prefixs': None,
                  'tracks_dir': os.getcwd(),
                  'ptcrifile_loc': None,
                  'ptcri_file': None,
                  'plot_dir': None,
                  'outfile_dir': None,
                  'diag_plot': False,
                  'match': False,
                  'agb': False,
                  'overwrite_ptcri': True,
                  'overwrite_match': True,
                  'prepare_makemod': False,
                  'track_diag_plot': False,
                  'hb_age_offset_fraction': 0.,
                  'log_dir': os.getcwd(),
                  'both': False}
    return input_dict


def call_prepare_makemod(inputs):
    prefixs = set_prefixs(inputs, harsh=False)
    prepare_makemod(prefixs, inputs.tracks_dir)

if __name__ == '__main__':
    inp_obj = fileio.InputFile(sys.argv[1], default_dict=initialize_inputs())
    add_version_info(sys.argv[1])
    loud = False
    if len(sys.argv) > 2:
        loud = True

    if inp_obj.debug:
        import pdb
        pdb.set_trace()

    prefixs = inp_obj.prefixs

    if inp_obj.hb:
        parsec2match(inp_obj, loud=loud)
        inp_obj.hb = False
        inp_obj.prefixs = prefixs

    parsec2match(inp_obj, loud=loud)

    if inp_obj.prepare_makemod:
        inp_obj.prefixs = prefixs
        call_prepare_makemod(inp_obj)
