""" Modules for QA/QC on match tracks"""
import numpy as np
from .config import mass, logT, age, logL
from .eep import critical_point
from .tracks import TrackSet, Track, TrackDiag


class CheckMatchTracks(critical_point.Eep, TrackSet, TrackDiag):
    """Class to check the output from TracksForMatch."""
    def __init__(self, inputs):
        TrackDiag.__init__(self)
        critical_point.Eep.__init__(self)
        inputs.match = True
        if inputs.hb:
            inputs.hbtrack_search_term += '.dat'
            inputs.hbtrack_search_term = \
                inputs.hbtrack_search_term.replace('PMS', '')
        TrackSet.__init__(self, inputs=inputs)
        self.flag_dict = inputs.flag_dict
        if not inputs.hb:
            tracks = self.tracks
        else:
            tracks = self.hbtracks

        self.check_tracks(tracks)

    def check_tracks(self, tracks):
        """
        Check the tracks for identical and non-monotontically increasing ages

        Results go into self.match_info dictionary whose keys are set by
        M%.3f % track.mass and values filled with a list of strings with the
        information.

        If the track has already been flagged (track.flag), no test occurs.

        Parameters
        ----------
        tracks: list of padova_tracks.Track objects
        """
        self.match_info = {}
        for t in tracks:
            key = 'M%.3f' % t.mass

            if key not in self.flag_dict.keys():
                print('check_tracks: No %s in flag dict, skipping.' % key)
                continue

            if self.flag_dict[key] is not None:
                print('check_tracks: skipping %s: %s' % (t.mass, t.flag))
                continue

            test = np.diff(t.data[age]) > 0
            if False in test:
                # age where does age decrease
                bads, = np.nonzero(np.diff(t.data[age]) < 0)
                edges = np.cumsum(self.nticks)
                if len(bads) != 0:
                    if key not in self.match_info:
                        self.match_info[key] = []
                        match_info = self.match_info[key]
                    match_info.append('Age not monotonicly increasing near')
                    nears = np.concatenate([np.nonzero(j - edges < 0)[0]
                                            for j in bads])
                    bad_inds = np.unique(nears)
                    match_info.append([np.array(self.eep_list)[bad_inds],
                                       t.data[age][bads]])
                    self.flag_dict['M%.3f' % t.mass] = 'age decreases on track'
                # identical values of age
                bads1, = np.nonzero(np.diff(t.data[age]) == 0)
                if len(bads1) != 0:
                    if key not in self.match_info:
                        self.match_info[key] = []
                        match_info = self.match_info[key]
                    match_info.append(['%i identical age values' % len(bads1)])
                    nears = np.concatenate([np.nonzero(j - edges < 0)[0]
                                            for j in bads1])
                    bad_inds = np.unique(nears)
                    match_info.append(['near',
                                       np.array(self.eep_list)[bad_inds]])
                    match_info.append(['log ages:', t.data[age][bads1]])
                    match_info.append(['inds:', bads1])
