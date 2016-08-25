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

from ..fileio import get_files, get_dirs, ts_indict
from ..utils import sort_dict, filename_data

from .track import Track
from ..eep.critical_point import CriticalPoint, Eep, find_ptcri


max_mass = 1000.
eep = Eep()


class TrackSet(object):
    """A class to load multiple Track class instances"""
    def __init__(self, **kwargs):
        default_dict = ts_indict()
        default_dict.update(kwargs)
        [self.__setattr__(k, v) for k, v in default_dict.items()]
        if self.prefix is not None:
            # assume we're in a set directory
            tracks_dir = self.tracks_dir or os.getcwd()
            self.tracks_base = os.path.join(tracks_dir, self.prefix)

            if not self.match:
                self.ptcrifile_loc = self.ptcrifile_loc or \
                    os.path.join(os.path.split(tracks_dir)[0], 'data')

                if not os.path.isdir(self.tracks_base) or \
                   not os.path.isdir(self.ptcrifile_loc):
                    # can't guess directory structure ... give up now!
                    print('can not guess dir structure')
                    self.prefix = ''
                    return

                ptcris = find_ptcri(self.prefix, from_p2m=self.from_p2m,
                                    ptcrifile_loc=self.ptcrifile_loc)
                if len(ptcris) == 0:
                    print('no ptcris found')
                    # parsec2match not run?
                    self.prefix = None
                    return
                else:
                    self.ptcri_file, self.hbptcri_file = ptcris

                self.track_search_term = self.track_search_term or '*F7_*PMS'
                self.hbtrack_search_term = self.hbtrack_search_term or \
                    track_search_term + '.HB'
                ignore = None
            else:
                self.track_search_term = '*dat'
                self.hbtrack_search_term = '*HB.dat'
                self.ptcri_file = None
                self.hbptcri_file = None

            self.find_tracks(ignore='HB')
            self.find_tracks(hb=True)

        if self.prefix is not None:
            self.parse_prefix()
        return

    def parse_prefix(self):
        self.prefix_dict = filename_data(self.prefix, skip=0)
        return

    def find_masses(self, track_search_term, ignore='ALFO0'):
        track_names = get_files(self.tracks_base, track_search_term)
        if ignore is not None:
            track_names = [t for t in track_names if ignore not in t]
        mstr = '_M'

        # mass array
        mass = np.array(['.'.join(os.path.split(t)[1]
                                  .split(mstr)[1].split('.')[:2])
                         for t in track_names], dtype=float)

        # inds of the masses to use and the correct order
        cut_mass, = np.nonzero(mass <= max_mass)
        morder = np.argsort(mass[cut_mass])

        # reorder by mass
        track_names = np.array(track_names)[cut_mass][morder]
        mass = mass[cut_mass][morder]

        err = 'No tracks found: {0:s}'.format(os.path.join(self.tracks_base,
                                                           track_search_term))
        assert len(track_names) != 0, err
        assert len(mass) != 0, err

        return track_names, mass

    def find_tracks(self, ignore='ALFO0', hb=False):
        '''
        loads tracks or hb tracks and their masses as attributes
        can load subset if masses (list, float, or string) is set.
        If masses is string, it must be have format like:
        '%f < 40' and it will use masses that are less 40.
        '''

        track_search_term = self.track_search_term
        if hb:
            hbf = 'hb{0:s}'
            track_search_term = self.hbtrack_search_term
            if 'hb' not in track_search_term.lower():
                print('warning, hb attribute assigned without hb in filename.')

        track_names, mass = self.find_masses(track_search_term, ignore=ignore)

        # only do a subset of masses
        masses = self.masses
        if masses is not None:
            if type(masses) == float:
                inds = [masses]
            elif type(masses) == str:
                inds = [i for i in range(len(mass)) if eval(masses % mass[i])]
            if type(masses) == list or type(masses) == np.ndarray:
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
        ptcri_file = self.ptcri_file
        maxmass_str = 'maxmass'
        if hb:
            track_str = hbf.format(track_str)
            mass_str = hbf.format(mass_str)
            ptcri_file = self.hbptcri_file
            maxmass_str = hbf.format(maxmass_str)

        tattr = '{0:s}s'.format(track_str)
        self.__setattr__('{0:s}_names'.format(track_str), track_names[inds])
        trks = [Track(t, match=self.match) for t in track_names[inds]]
        self.__setattr__(tattr, trks)
        self.__setattr__('%s' % mass_str,
                         np.array([t.mass for t in self.__getattribute__(tattr)
                                  if t.flag is None], dtype=np.float))
        self.__setattr__('%s' % maxmass_str,
                         np.max(self.__getattribute__(mass_str)))
        return

    def eep_file(self, outfile=None):
        if outfile is None:
            outfile = '{}_eeptrack.dat'.format(self.prefix.replace('/', ''))
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
            if 'iptcri' not in track.__dict__.keys():
                print('no iptcri M={} {}/{} '.format(track.mass, track.base,
                                                     track.name))
                if track.mass > 0.6:
                    print(track.flag)
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
            ptcri_attr = self.select_ptcri(('z%g' % track.Z).replace('0.', ''))
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
            kf = os.path.split(p)[1].replace('0.', '').replace('.dat', '')
            return kf.lower()

        if sandro:
            search_term = 'pt'
        else:
            search_term = 'p2m'
        if hb:
            search_term += '_hb'

        new_keys = []
        mets = np.unique([t.Z for t in self.tracks])
        pt_search = '%s*%s*' % (search_term, search_extra)
        ptcri_files = get_files(ptcri_loc, pt_search)
        if not hb:
            ptcri_files = [p for p in ptcri_files if 'hb' not in p]

        for p in ptcri_files:
            ptcri = CriticalPoint(p, sandro=sandro, hb=False)
            new_key = keyfmt(p)
            self.__setattr__(new_key, ptcri)
            new_keys.append(new_key)

        self.__setattr__('ptcris', np.unique(new_keys).tolist())
        for i, track in enumerate(self.tracks):
            ptcri_name, = [p for p in ptcri_files
                           if os.path.split(track.base)[1] in p]
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
                    inds = [t.iptcri[t.iptcri > 0]]
                    fmt += ' & '.join('{:.3g}'.format(i)
                                      for i in t.data[age][inds])
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
            # del inputs.prefixs
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
    parser = argparse.ArgumentParser(description="Cull EEP rows from files")

    parser.add_argument('-v', '--pdb', action='store_true',
                        help='invoke pdb')

    parser.add_argument('-a', '--all', action='store_true',
                        help='make an EEP file for all track dirs in this cwd')

    parser.add_argument('-m', '--match', action='store_true',
                        help='these are tracks for match')

    parser.add_argument('-o', '--outfile', type=str, default=None,
                        help='file name to write/append to')

    parser.add_argument('-s', '--search', type=str, default='OV',
                        help='Prefix search term')

    parser.add_argument('-p', '--prefix', type=str,
                        help='if not -a, prefix must be in cwd')

    args = parser.parse_args(argv)

    if args.pdb:
        import pdb
        pdb.set_trace()

    if args.all:
        big_eep_file(prefix_search_term=args.search, outfile=args.outfile,
                     match=args.match)
    else:
        ts = TrackSet(prefix=args.prefix, match=args.match)
        ts.eep_file(outfile=args.outfile)


if __name__ == "__main__":
    main(sys.argv[1:])
