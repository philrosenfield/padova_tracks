from __future__ import print_function
import os
import numpy as np
import sys

from ..config import logL, mass, age, MODE
from ..utils import sort_dict, get_zy
from ..fileio import get_files, load_eepdefs


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

        # useful to check match compatibility
        self.ntot = np.sum(eep_lengths)
        self.nok = self.ntot - np.sum(eep_lengths[:itp])
        ims = eep_list.index('MS_TO')
        trans = ihb - 1
        self.nlow = np.sum(eep_lengths[:ims])
        self.nhb = np.sum(eep_lengths_hb) - self.nok
        self.nms = np.sum(eep_lengths[:trans])
        self.trans = eep_lengths[trans]

        self.pdict = dict(zip(self.eep_list, range(len(self.eep_list))))
        self.pdict_hb = dict(zip(self.eep_list_hb,
                                 range(len(self.eep_list_hb))))


class CriticalPoint(object):
    '''class to hold EEP data'''
    def __init__(self, filename=None, debug=False):
        self.hb = False
        if filename is not None:
            if 'hb' in filename:
                self.hb = True
            self.base, self.name = os.path.split(filename)
            self.load_ptcri(filename)
            self.Z, self.Y = get_zy(filename)

    def load_iptcri(self, track):
        try:
            iptcri = self.data_dict['M{:.3f}'.format(track.mass)]
        except KeyError:
            print('M={0.4f} not found in {1:s}'
                  .format((track.mass, os.path.join(self.base, self.name))))
            track.flag = 'no ptcri mass'
            return
        track.iptcri = iptcri
        return track

    def load_critical_points(self, tracks):
        '''
        Call load_iptcri

        Parameters
        ----------
        tracks : object or list of objects
            padova_tracks.Track object

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
        return tracks

    def load_ptcri(self, filename):
        '''
        Read the ptcri*dat file.
        Initialize Eep
        Flag the missing eeps in the ptcri file.
        '''
        with open(filename, 'r') as f:
            lines = f.readlines()

        # the lines have the path name, and the path has F7.
        # the final column is a filename.
        all_keys = lines[1].replace('#', '').strip().split()
        col_keys = all_keys[3:-1]
        # ptcri file has filename as col #19 so skip the last column
        usecols = range(0, len(all_keys) - 1)
        data = np.genfromtxt(filename, usecols=usecols, skip_header=2,
                             invalid_raise=False)
        self.data = data
        self.masses = data[:, 1]

        data_dict = {}
        for i, _ in enumerate(data):
            str_mass = 'M{0:.3f}'.format(self.masses[i])
            data_dict[str_mass] = data[i][3:].astype(int)

        self.data_dict = data_dict
