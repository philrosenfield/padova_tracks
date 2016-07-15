'''
a container of track
'''
from __future__ import print_function
import argparse
import sys
import os
import numpy as np
import pandas as pd
import scipy

from ..fileio import get_files, get_dirs
from ..utils import sort_dict, filename_data

from .track import Track, AGBTrack
from .track_diag import TrackDiag
from ..eep.critical_point import critical_point, Eep

import logging
logger = logging.getLogger()

max_mass = 1000.
td = TrackDiag()
eep = Eep()


class TrackSet(object):
    """A class to load multiple Track instances"""
    def __init__(self, inputs=None, prefix='', match=False):
        if inputs is not None:
            self.hb = inputs.hb
            self.initialize_tracks(inputs)

        if len(prefix) > 0:
            # assume we're in a set directory
            tracks_dir = os.getcwd()
            self.tracks_base = os.path.join(tracks_dir, prefix)
            # look for data directory, should be at same level as tracks
            self.prefix = prefix
            if not match:

                data_dir = os.path.join(os.path.split(tracks_dir)[0], 'data')
                if not os.path.isdir(self.tracks_base) or not os.path.isdir(data_dir):
                    # can't guess directory structure ... give up now!
                    print('can not guess dir structure')
                    self.prefix = ''
                    return

                ptcris = get_files(data_dir, 'p2m*{}*'.format(prefix))

                if len(ptcris) == 0:
                    print('no ptcris found')
                    # parsec2match not run?
                    self.prefix = ''
                    return

                ptcri_file, = [p for p in ptcris if not 'hb' in p]
                hbptcri_file, = [p for p in ptcris if 'hb' in p]
                track_search_term='*F7_*PMS'
                hbtrack_search_term='*F7_*PMS.HB'
                ignore = None
            else:
                track_search_term = '*dat'
                hbtrack_search_term = '*HB.dat'
                ptcri_file = None
                hbptcri_file = None

            self.hb = False
            self.find_tracks(track_search_term=track_search_term,
                             ptcri_file=ptcri_file, match=match, ignore='HB')
            self.hb = True
            self.find_tracks(track_search_term=hbtrack_search_term,
                             ptcri_file=hbptcri_file, match=match)

        if 'prefix' in self.__dict__.keys():
            self.parse_prefix()
        return

    def parse_prefix(self):
        self.prefix_dict = filename_data(self.prefix, skip=0)
        return

    def initialize_tracks(self, inputs):
        self.prefix = inputs.prefix

        if not inputs.match:
            self.tracks_base = os.path.join(inputs.tracks_dir, self.prefix)
        else:
            self.tracks_base = inputs.outfile_dir
            inputs.track_search_term = \
                                inputs.track_search_term.replace('PMS', '')
            inputs.track_search_term += '.dat'

        if self.hb:
            self.find_tracks(track_search_term=inputs.hbtrack_search_term,
                             masses=inputs.hbmasses, match=inputs.match)
        else:
            self.hbtrack_names = []
            self.hbtracks = []
            self.hbmasses = []

        if not self.hb or inputs.both:
            self.find_tracks(track_search_term=inputs.track_search_term,
                             masses=inputs.masses, match=inputs.match)


    def find_masses(self, track_search_term, ignore='ALFO0'):
        track_names = get_files(self.tracks_base, track_search_term)
        if ignore is not None:
            track_names = [t for t in track_names if not ignore in t]
        mstr = '_M'

        # mass array
        mass = np.array(['.'.join(os.path.split(t)[1].split(mstr)[1].split('.')[:2])
                         for t in track_names], dtype=float)

        # inds of the masses to use and the correct order
        cut_mass, = np.nonzero(mass <= max_mass)
        morder = np.argsort(mass[cut_mass])

        # reorder by mass
        track_names = np.array(track_names)[cut_mass][morder]
        mass = mass[cut_mass][morder]

        assert len(track_names) != 0, \
            'No tracks found: %s/%s' % (self.tracks_base, track_search_term)

        assert len(mass) != 0, \
            'No tracks found: %s/%s' % (self.tracks_base, track_search_term)

        return track_names, mass

    def find_tracks(self, track_search_term='*F7_*PMS', masses=None,
                    match=False, agb=False, ptcri_file=None,
                    ignore='ALFO0'):
        '''
        loads tracks or hb tracks and their masses as attributes
        can load subset if masses (list, float, or string) is set.
        If masses is string, it must be have format like:
        '%f < 40' and it will use masses that are less 40.
        '''

        track_names, mass = self.find_masses(track_search_term, ignore=ignore)

        # only do a subset of masses
        if masses is not None:
            if type(masses) == float:
                inds = [masses]
            elif type(masses) == str:
                inds = [i for i in range(len(mass)) if eval(masses % mass[i])]
            if type(masses) == list:
                inds = np.array([], dtype=np.int)
                for m in masses:
                    try:
                        inds = np.append(inds, list(mass).index(m))
                    except ValueError:
                        # this mass is missing
                        pass
        else:
            inds = np.argsort(mass)

        track_str = 'track'
        mass_str = 'masses'

        if self.hb:
            track_str = 'hb%s' % track_str
            mass_str = 'hb%s' % mass_str

        tattr = '%ss' % track_str
        self.__setattr__('%s_names' % track_str, track_names[inds])
        trks = [Track(t, match=match, ptcri_file=ptcri_file)
                for t in track_names[inds]]
        self.__setattr__(tattr, trks)
        self.__setattr__('%s' % mass_str, \
            np.array([t.mass for t in self.__getattribute__(tattr)
                      if t.flag is None], dtype=np.float))
        return

    def eep_file(self, outfile=None):
        if outfile is None:
            outfile = '{}_eeptrack.dat'.format(self.prefix.replace('/',''))
        header = True
        wstr = 'w'
        wrote = 'wrote to'
        if os.path.isfile(outfile):
            wstr = 'a'
            header = False
            wrote = 'appended to'
        data = pd.DataFrame()

        for track in np.concatenate([self.tracks, self.hbtracks]):
            offset = 0
            if not 'iptcri' in track.__dict__.keys():
                print('no iptcri M={} {}/{} '.format(track.mass, track.base, track.name))
                if track.mass > 0.6:
                    print(track.flag)
                    #import pdb; pdb.set_trace()
                continue
            inds, = np.nonzero(track.iptcri)
            iptcri = track.iptcri[inds]
            if track.hb:
                # start counting iptcri for HB after RGB_TIP.
                offset = eep.eep_list.index(eep.eep_list_hb[0])
            df = pd.DataFrame(track.data[iptcri])

            df['iptcri'] = inds + offset
            df['hb'] = track.hb * 1
            for k, v in self.prefix_dict.items():
                df[k] = v
            data = data.append(df, ignore_index=True)

        data.to_csv(outfile, mode=wstr, index=False, sep=' ', header=header)
        print('{} {}'.format(wrote, outfile))

    def all_inds_of_eep(self, eep_name, sandro=True, hb=False):
        '''
        get all the ind for all tracks of some eep name, for example
        want ms_to of the track set? set eep_name = point_c if sandro==True.
        '''
        inds = []
        for track in self.tracks:
            ptcri_attr = self.select_ptcri(('z%g' % track.Z).replace('0.',''))
            ptcri = self.__getattribute__(ptcri_attr)
            eep_ind = ptcri.get_ptcri_name(eep_name, sandro=sandro, hb=hb)
            if sandro:
                if len(track.sptcri) <= eep_ind:
                    data_ind = -1
                else:
                    data_ind = track.sptcri[eep_ind]
            else:
                data_ind = track.iptcri[eep_ind]
            inds.append(data_ind)
        return inds

    def select_ptcri(self, criteria):
        """
        find the ptcri attribute from a list of already loaded ptcri
        attributes
        """
        criteria = criteria.lower()  # incase ya fergot
        pind, = [i for i, p in enumerate(self.ptcris) if criteria in p]
        ptcri_attr = self.ptcris[pind]
        return ptcri_attr

    def _load_ptcri(self, ptcri_loc, sandro=True, hb=False, search_extra=''):
        '''load ptcri file for each track in trackset'''

        def keyfmt(p):
            return os.path.split(p)[1].replace('0.', '').replace('.dat', '').lower()
        if sandro:
            search_term = 'pt'
        else:
            search_term = 'p2m'
        if hb:
            search_term += '_hb'

        new_keys = []
        mets = np.unique([t.Z for t in self.tracks])
        pt_search =  '%s*%s*' % (search_term, search_extra)
        ptcri_files = get_files(ptcri_loc, pt_search)
        if not hb:
            ptcri_files = [p for p in ptcri_files if not 'hb' in p]

        for p in ptcri_files:
            ptcri = critical_point(p, sandro=sandro, hb=False)
            new_key = keyfmt(p)
            self.__setattr__(new_key, ptcri)
            new_keys.append(new_key)

        self.__setattr__('ptcris', np.unique(new_keys).tolist())
        for i, track in enumerate(self.tracks):
            ptcri_name, = [p for p in ptcri_files if os.path.split(track.base)[1] in p]
            ptcri = self.__getattribute__(keyfmt(ptcri_name))
            self.tracks[i] = ptcri.load_eeps(track, sandro=sandro)
        return self.tracks

    def relationships(self, eep_name, xattr, yattr, sandro=True, xfunc=None,
                      yfunc=None, ptcri_loc=None, ptcri_search_extra='',
                      hb=False):
        """
        eg get the MSTO as a function of age for all the tracks
        eep_name = 'POINT_C' or 'MSTO' (Sandro True/False)
        ptcri_search_extra: OV0.5
        xattr = 'MASS'
        yattr = 'AGE'
        """
        if not hasattr(self, 'ptcri') and not hasattr(self, 'ptcris'):
            self._load_ptcri(ptcri_loc, sandro=sandro, hb=hb,
                             search_extra=ptcri_search_extra)
        ieeps = self.all_inds_of_eep(eep_name, sandro=sandro, hb=hb)
        xdata, ydata = zip(*[(self.tracks[i].data[xattr][ieeps[i]],
                              self.tracks[i].data[yattr][ieeps[i]])
                             for i in range(len(self.tracks))
                             if ieeps[i] != -1])
        isort = np.argsort(xdata)
        xdata = np.array(xdata)[isort]
        ydata = np.array(ydata)[isort]

        if xfunc is not None:
            xdata = eval('%s(xdata)' % xfunc)

        if yfunc is not None:
            ydata = eval('%s(ydata)' % yfunc)

        f = scipy.interpolate.interp1d(xdata, ydata, bounds_error=False)
        self.__setattr__('_'.join((xattr, yattr, 'interp')).lower(), f)
        return f

    def load_characteristics(self):
        attrs = ['Z', 'Y', 'ALFOV']
        for attr in attrs:
            self.__setattr__('%ss' % attr,
                             np.array([t.__getattribute__(attr)
                                       for t in self.tracks], dtype=np.float))

    def check_header(ts):
        [t.check_header_arg() for t in ts.tracks]

        check_list = ['AGELIMIT', 'ENV_OV', 'ALFOV', 'ISHELL']

        oneline = ' '.join(header)
        if 'RESTART' in oneline:
            print('Track restarted')

    def track_summary(self, full=True):
        if hasattr(self, 'tracks') and hasattr(self, 'ptcris'):
            ptcri_name = self.__getattribute__('ptcris')[0]
            ptcri = self.__getattribute__(ptcri_name)
            if full:
                eep_name, _ = sort_dict(ptcri.key_dict)
                fmt = ' & '.join(eep_name) + ' \\\\ \n'
                for t in self.tracks:
                    fmt += ' & '.join('{:.3g}'.format(i)
                                      for i in t.data[age][t.iptcri[t.iptcri>0]])
                    fmt += ' \\\\ \n'
            return fmt
        else:
            pass


class TrackMix(object):
    """A class to hold multiple TrackSet intances"""
    def __init__(self, inputs=None):
        if inputs is None:
            self.prefixs = []
            self.track_sets = []
        else:
            self.prefixs = inputs.prefixs
            #del inputs.prefixs
            self.load_track_sets(inputs)

    def load_track_sets(self, inputs):
        self.track_sets = np.array([])
        for prefix in self.prefixs:
            inputs.hb = True
            inputs.prefix = prefix
            if inputs.hb:
                track_set = TrackSet(inputs=inputs)
                inputs.hb = False
            track_set = np.concatenate((track_set, TrackSet(inputs=inputs)))
            track_set = self.sort_by_mass(track_set)
            self.track_sets = np.append(self.track_sets, track_set)


def big_eep_file(prefix_search_term='OV', outfile=None, match=False):

    if outfile is None:
        outfile = 'all_eeps.csv'

    prefixs = get_dirs(os.getcwd(), prefix_search_term)

    for prefix in prefixs:
        prefix = os.path.split(prefix)[1]
        ts = TrackSet(prefix=prefix, match=match)
        ts.eep_file(outfile=outfile)
    return

def main(argv):
    """
    Main function for track_set.py write eep rows to a file
    """
    parser = argparse.ArgumentParser(description="Plot or reformat calcsfh -ssp output")

    parser.add_argument('-v', '--pdb', action='store_true',
                        help='invoke pdb')

    parser.add_argument('-a', '--all', action='store_true',
                        help='make an EEP file for all track dirs in this directory')

    parser.add_argument('-m', '--match', action='store_true',
                        help='these are tracks for match')

    parser.add_argument('-o', '--outfile', type=str, default=None,
                        help='file name to write/append to')

    parser.add_argument('-s', '--search', type=str, default='OV',
                        help='Prefix search term')

    parser.add_argument('-p', '--prefix', type=str,
                        help='if not -a prefix (directory name that holds tracks) must be in cwd')

    args = parser.parse_args(argv)

    if args.pdb:
        import pdb; pdb.set_trace()

    if args.all:
        big_eep_file(prefix_search_term=args.search, outfile=args.outfile, match=args.match)
    else:
        ts = TrackSet(prefix=args.prefix, match=args.match)
        ts.eep_file(outfile=args.outfile)


if __name__ == "__main__":
    main(sys.argv[1:])
