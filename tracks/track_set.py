'''
a container of track
'''
from __future__ import print_function
import os
import numpy as np
import scipy

from ..fileio import get_files
from ..utils import sort_dict

from .track import Track, AGBTrack
from .track_diag import TrackDiag
from ..eep.critical_point import critical_point

import logging
logger = logging.getLogger()

max_mass = 1000.
td = TrackDiag()


class TrackSet(object):
    """A class to load multiple Track instances"""
    def __init__(self, inputs=None):
        if inputs is None:
            self.prefix = ''
        else:
            self.hb = inputs.hb
            self.initialize_tracks(inputs)

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
                    match=False, agb=False):
        '''
        loads tracks or hb tracks and their masses as attributes
        can load subset if masses (list, float, or string) is set.
        If masses is string, it must be have format like:
        '%f < 40' and it will use masses that are less 40.
        '''

        track_names, mass = self.find_masses(track_search_term)

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
        trks = []
        for t in track_names[inds]:
            if 'AGB' in t.upper():
                trk = AGBTrack(t)
            else:
                trk = Track(t, match=match)
            trks.append(trk)
        self.__setattr__(tattr, trks)
        self.__setattr__('%s' % mass_str, \
            np.array([t.mass for t in self.__getattribute__(tattr)
                      if t.flag is None], dtype=np.float))
        return

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
