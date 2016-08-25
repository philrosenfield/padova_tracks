from __future__ import print_function
import os
import numpy as np
import sys

from ..config import logL, mass, age, MODE
from ..utils import sort_dict, get_zy
from ..fileio import get_files, load_eepdefs


def find_ptcri(prefix, from_p2m=False, ptcrifile_loc=os.getcwd()):
    search_term = 'pt*'
    if from_p2m:
        search_term = 'p2m*'

    search_term += '{0:s}Y*dat'.format(prefix.split('Y')[0])
    ptcris = get_files(ptcrifile_loc, search_term)
    try:
        ptcri_file, = [p for p in ptcris if 'hb' not in p]
        hbptcri_file, = [p for p in ptcris if 'hb' in p]
        retv = [ptcri_file, hbptcri_file]
    except:
        retv = []
    return retv


class Eep(object):
    '''
    a simple class to hold eep data. Gets added as an attribute to
    critical_point class.
    The lengths are then used in match.py
    '''
    def __init__(self):
        '''hard coded default eep_list and lengths'''
        eep_list, eep_lengths = load_eepdefs()

        ihb = eep_list.index('HE_BEG')
        itp = eep_list.index('TPAGB_BEG')
        eep_list_hb = np.copy(eep_list[ihb:])
        eep_lengths_hb = np.copy(eep_lengths[ihb:])

        self.eep_list = eep_list
        self.nticks = eep_lengths
        self.eep_list_hb = eep_list_hb
        self.nticks_hb = eep_lengths_hb

        # usefull to check match compatibility
        self.ntot = np.sum(eep_lengths)
        self.nok = self.ntot - np.sum(eep_lengths[:itp])
        ims = eep_list.index('MS_TO')
        trans = ihb - 1
        self.nlow = np.sum(eep_lengths[:ims])
        self.nhb = np.sum(eep_lengths_hb) - self.nok
        self.nms = np.sum(eep_lengths[:trans])
        self.trans = eep_lengths[trans]


class CriticalPoint(object):
    '''
    class to hold ptcri data from Sandro's ptcri file and input eep_obj
    which tells which critical points of Sandro's to ignore and which new
    ones to define. Definitions of new eeps are in the Track class.
    '''
    def __init__(self, filename=None, debug=False):
        self.debug = debug
        self.hb = False
        if filename is not None:
            if 'hb' in filename:
                self.hb = True
            self.base, self.name = os.path.split(filename)
            self.load_ptcri(filename)
            self.Z, self.Y = get_zy(filename)
        else:
            self.load_eep()

    def load_iptcri(self, track):
        try:
            iptcri = self.data_dict['M{:.3f}'.format(track.mass)]
        except KeyError:
            print('M={0.4f} not found in {1:s}'
                  .format((track.mass, os.path.join(self.base, self.name))))
            track.flag = 'no ptcri mass'
            return
        if self.sandro:
            track.sptcri = iptcri
        else:
            track.iptcri = iptcri
        return track

    def load_critical_points(self, tracks):
        '''
        Load all EEPs. First the ones defined by Sandro, then call
        define_eep_stages

        Parameters
        ----------
        track : object
            rsp.padova_tracks.Track object

        ptcri : flexible
            rsp.padova_tracks.critical_point object or filename of ptcri
            file

        plot_dir : str
            path to diagnostic plot directory

        diag_plot : bool [True]
            make diagnostic plots

        debug : bool [False]
            debug errors with pbd

        Returns
        -------
        Adds the following attributes:
        ptcri : the critical_point object
        track.iptcri : the critical point index rel to track.data
        track.mptcri : the model number of the critical point
        '''
        if not isinstance(tracks, list):
            tracks = [tracks]
        for track in tracks:
            if track.flag is not None:
                return track

            errfmt = '{}s do not match between track and ptcri file {} != {}'
            if hasattr(self, 'Z'):
                assert self.Z == track.Z, errfmt.format('Z', self.Z, track.Z)

            if hasattr(self, 'Y'):
                assert np.round(self.Y, 2) == np.round(track.Y, 2), \
                    errfmt.format('Y', self.Y, track.Y)

            self.load_iptcri(track)

            if len(self.please_define) > 0:

                # Initialize iptcri
                track.iptcri = np.zeros(len(self.eep_list), dtype=int)

                # Get the values that we won't be replacing.
                pinds = np.array([i for i, a in enumerate(self.eep_list)
                                  if a in self.sandro_eeps])

                sinds = \
                    np.array([i for i, a in enumerate(self.sandro_eeps)
                              if a in self.eep_list])

                # they may all be replaced!
                if len(pinds) > 0 and len(sinds) > 0:
                    track.iptcri[pinds] = track.sptcri[sinds] - 2

                # but if the track did not actually make it to that EEP, no -2!
                track.iptcri[track.iptcri < 0] = 0

                # and if sandro cut the track before it reached this point,
                # no index error!
                track.iptcri[track.iptcri > len(track.data[MODE])] = 0
        return tracks

    def load_eep(self):
        self.eep = Eep()
        if self.hb:
            self.eep_list = self.eep.eep_list_hb
        else:
            self.eep_list = self.eep.eep_list

        self.pdict = dict(zip(self.eep_list, range(len(self.eep_list))))
        self.please_define = self.eep_list

    def load_ptcri(self, filename):
        '''
        Read the ptcri*dat file.
        Initialize Eep
        Flag the missing eeps in the ptcri file.
        '''
        self.sandro = True
        if 'p2m' in filename:
            begin = 0
            self.sandro = False

        with open(filename, 'r') as f:
            lines = f.readlines()

        # the lines have the path name, and the path has F7.
        if self.sandro and not self.hb:
            begin, = [i for i in range(len(lines))
                      if lines[i].startswith('#') and 'F7' in lines[i]]
        else:
            begin = -1

        if self.sandro and not self.hb:
            try:
                self.fnames = [l.strip().split('../F7/')[1]
                               for l in lines[(begin+2):]]
            except IndexError:
                # last two lines of Sandro's files have a different format
                self.fnames = [l.strip().split('../F7/')[1]
                               for l in lines[(begin+2):-2]]

        # the final column is a filename.
        all_keys = lines[begin + 1].replace('#', '').strip().split()
        col_keys = all_keys[3:-1]
        # ptcri file has filename as col #19 so skip the last column
        usecols = range(0, len(all_keys) - 1)
        if self.hb and 'p2m' not in filename:
            col_keys = all_keys[3:]
            usecols = range(0, len(all_keys))
        endcol = 'FIN'
        for key in ['C_BUR', 'HeLST', 'TPAGB']:
            try:
                col_keys[col_keys.index(key)] = endcol
            except ValueError:
                pass
        # invalid_raise will skip the last rows that Sandro uses to fake the
        # youngest MS ages (600Msun).
        data = np.genfromtxt(filename, usecols=usecols, skip_header=begin + 2,
                             invalid_raise=False)
        self.data = data
        self.masses = data[:, 1]

        data_dict = {}
        for i, _ in enumerate(data):
            str_mass = 'M{0:.3f}'.format(self.masses[i])
            data_dict[str_mass] = data[i][3:].astype(int)

        self.data_dict = data_dict

        self.load_eep()

        if self.sandro:
            from ..preprocessing.debug_sandro import check_ptcri
            # loading sandro's eeps means they will be used for match
            self.sandro_eeps = col_keys
            self.sdict = dict(zip(col_keys, range(len(col_keys))))
            self.please_define = [c for c in self.eep_list
                                  if c not in col_keys]

            [check_ptcri(self, self.masses[i], data[i][3:].astype(int))
             for i in range(len(data))]

            # low mass defined by stars that do not evolove past MSTO==6
            self.low_mass = \
                np.max(np.array([k.replace('M', '') for
                                 k, v in self.data_dict.items()
                                 if len(v[v > 0]) <= 6], dtype=float))

    def save_ptcri(self, tracks, filename=None):
        '''save parsec2match EEPs in similar format as sandro's'''
        import operator

        if filename is None:
            filename = os.path.join(self.base, 'p2m_{0:s}'.format(self.name))
            if self.hb:
                filename = filename.replace('p2m', 'p2m_hb')

        # sort the dictionary by values (which are ints)
        sorted_keys, _ = zip(*sorted(self.pdict.items(),
                                     key=operator.itemgetter(1)))

        cols = ' '.join(list(sorted_keys))
        header = '# EEPs defined by sandro, basti, mist, and phil \n'
        header += '# i mass kind_track {0:s} F7name \n'.format(cols)
        linefmt = '{0:d} {1:.3f} 0.0 {2:s} {3:s} \n'
        with open(filename, 'w') as f:
            f.write(header)
            for i, track in enumerate(tracks):
                if track.flag is not None:
                    print('save_ptcri skipping {0:s}: {1:s}'
                          .format(track.name, track.flag))
                    continue
                f7name = os.path.join(track.base, track.name)
                ptcri_str = ' '.join(['{0:d}'.format(p) for p in track.iptcri])
                f.write(linefmt.format(i+1, track.mass, ptcri_str, f7name))
        return filename
