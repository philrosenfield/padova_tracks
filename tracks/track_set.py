'''
a container of track
'''

import argparse
import sys
import os
import numpy as np
import pandas as pd
import scipy

from ..fileio import get_files, get_dirs, ts_indict
from ..utils import sort_dict, filename_data

from .track import Track
from ..eep.critical_point import CriticalPoint, Eep


max_mass = 1000.
min_mass = 0.10
eep = Eep()


class TrackSet(object):
    """A class to load multiple Track class instances"""
    def __init__(self, **kwargs):
        default_dict = ts_indict()
        default_dict.update(kwargs)
        [self.__setattr__(k, v) for k, v in list(default_dict.items())]
        if self.prefix is not None:
            # assume we're in a set directory
            tracks_dir = self.tracks_dir or os.getcwd()
            self.tracks_base = os.path.join(tracks_dir, self.prefix)
            self.find_tracks()
            filename_data(self.prefix, skip=0)
        return

    def find_tracks(self):
        """
        load all files in tracks.base as Track instances
        also add attributes masses and hbmaxmass to self.
        """
        track_names = get_files(self.tracks_base, '*.*')

        mstr = '_M'
        # mass array
        mass_ = np.array(['.'.join(os.path.split(t)[1]
                                   .split(mstr)[1].split('.')[:2])
                          for t in track_names], dtype=float)

        cut_mass, = np.nonzero((mass_ <= max_mass) & (mass_ >= min_mass))
        morder = np.argsort(mass_[cut_mass])

        # reorder by mass
        track_names = np.array(track_names)[cut_mass][morder]
        mass_ = mass_[cut_mass][morder]

        trks_ = [Track(t, match=self.match) for t in track_names]
        trks = [t for t in trks_ if t.flag is None]
        masses = np.unique([t.mass for t in trks])

        # here is where one would code a mass cut for HB only...
        hbts = [t for t in trks if t.hb]
        hbmaxmass = np.max([t.mass for t in hbts])

        self.tracks = trks
        self.masses = masses
        self.hbmaxmass = hbmaxmass

        err = 'No tracks found: {0:s}'.format(self.tracks_base)
        assert len(self.tracks) != 0, err
        assert len(self.masses) != 0, err
        return

    def eep_file(self, outfile=None):
        """Save track_set EEPs to file, must load ptcri first"""
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

        for track in self.tracks:
            offset = 0
            if 'iptcri' not in list(track.__dict__.keys()):
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
            for k, v in list(self.prefix_dict.items()):
                df[k] = v
            data = data.append(df, ignore_index=True)

        data.to_csv(outfile, mode=wstr, index=False, sep=' ', header=header)
        print('{} {}'.format(wrote, outfile))

    def all_inds_of_eep(self, eep_name, hb=False):
        '''
        get all the ind for all tracks of some eep name, for example
        want ms_to of the track set? set eep_name = MS_TO
        '''
        inds = []
        for track in self.tracks:
            ptcri_attr = self.select_ptcri(('z%g' % track.Z).replace('0.', ''))
            ptcri = self.__getattribute__(ptcri_attr)
            eep_ind = ptcri.get_ptcri_name(eep_name, hb=hb)
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

    def _load_ptcri(self, ptcri_loc, hb=False, search_extra=''):
        '''load ptcri file for each track in trackset'''
        print('this is probably broken...')

        def keyfmt(p):
            kf = os.path.split(p)[1].replace('0.', '').replace('.dat', '')
            return kf.lower()

        search_term = 'p2m'
        if hb:
            search_term += '_hb'

        new_keys = []
        mets = np.unique([t.Z for t in self.tracks])
        pt_search = '{0:s}*{1:s}*'.format(search_term, search_extra)
        ptcri_files = get_files(ptcri_loc, pt_search)
        if not hb:
            ptcri_files = [p for p in ptcri_files if 'hb' not in p]

        for p in ptcri_files:
            new_key = keyfmt(p)
            self.__setattr__(new_key, ptcri)
            new_keys.append(new_key)

        self.__setattr__('ptcris', np.unique(new_keys).tolist())
        for i, track in enumerate(self.tracks):
            ptcri_name, = [p for p in ptcri_files
                           if os.path.split(track.base)[1] in p]
            ptcri = self.__getattribute__(keyfmt(ptcri_name))
            self.tracks[i] = ptcri.load_critical_points(track)
        return self.tracks

    def relationships(self, eep_name, xattr, yattr, xfunc=None,
                      yfunc=None, ptcri_loc=None, ptcri_search_extra='',
                      hb=False):
        """
        eg get the MSTO as a function of age for all the tracks
        eep_name = 'MS_TO'
        ptcri_search_extra: OV0.5
        xattr = 'MASS'
        yattr = 'AGE'
        """
        if not hasattr(self, 'ptcri') and not hasattr(self, 'ptcris'):
            self._load_ptcri(ptcri_loc, hb=hb,
                             search_extra=ptcri_search_extra)
        ieeps = self.all_inds_of_eep(eep_name, hb=hb)
        xdata, ydata = list(zip(*[(self.tracks[i].data[xattr][ieeps[i]],
                              self.tracks[i].data[yattr][ieeps[i]])
                             for i in range(len(self.tracks))
                             if ieeps[i] != -1]))
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
