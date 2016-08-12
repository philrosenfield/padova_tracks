"""Interpolate tracks for match and check the interpolations"""
from __future__ import print_function
import logging
import numpy as np
import os
import pdb

from scipy.interpolate import splev
from scipy.interpolate import interp1d

from . import fileio
from .config import mass, logT, age, logL
from .eep import critical_point
from .eep.define_eep import DefineEeps
from .interpolate.interpolate import Interpolator
from .tracks import TrackSet, Track, TrackPlots

logger = logging.getLogger()


class TracksForMatch(TrackSet, DefineEeps, TrackPlots, Interpolator):
    """
    This class is for interpolating tracks for use in MATCH. DefineEeps is made
    for one track at a time, TracksForMatch takes a track set as input.
    """
    def __init__(self, *args, **kwargs):
        TrackPlots.__init__(self)
        TrackSet.__init__(self, **kwargs)
        DefineEeps.__init__(self, self.ptcri_file)
        self.debug = kwargs.get('debug', False)
        [self.load_critical_points(track) for track in self.tracks]
        if hasattr(self, 'hbtracks'):
            [self.load_critical_points(track) for track in self.hbtracks]

        default_outdir = \
            os.path.join(self.tracks_dir, 'match', self.prefix)
        default_plotdir = \
            os.path.join(self.tracks_dir, 'diag_plots', self.prefix)
        self.outfile_dir = self.outfile_dir or default_outdir
        self.plot_dir = self.plot_dir or default_plotdir
        self.log_dir = self.log_dir or os.path.join(self.tracks_dir, 'logs')

    def match_interpolation(self, hb=False):
        """
        Call the MATCH interpolator, make diagnostic plots

        This function writes two file types:
        match_interp logfile: Any error collections from define_eep or here.
        lines_check file: Summary of the interpolated files lengths.

        Parameters
        ----------
        inputs: fileio.InputParameters object
            requires either None or initialization:
            inputs.overwrite_match
            inputs.outfile_dir
            inputs.plot_dir
            inputs.diag_plot

        Returns
        -------
        flag_dict : dictionary identifying problematic tracks
        """
        # to pass the flags to another class
        flag_dict = {}
        info_dict = {}

        if not hb:
            tracks = self.tracks
            filename = 'match_interp_%s.log'
        else:
            tracks = self.hbtracks
            filename = 'match_interp_hb_%s.log'

        for track in tracks:
            flag_dict['M%.3f' % track.mass] = track.flag

            if track.flag is not None:
                print('skipping track M=%.3f because of flag: %s' %
                      (track.mass, track.flag))
                info_dict['M%.3f' % track.mass] = track.flag
                continue

            # interpolate tracks for match
            outfile = \
                os.path.join(self.outfile_dir,
                             'match_%s.dat' % os.path.splitext(track.name)[0])

            if not self.overwrite_match and os.path.isfile(outfile):
                print('not overwriting %s' % outfile)
                continue
            match_track = self.prepare_track(track, outfile, hb=hb)

            info_dict['M%.3f' % track.mass] = track.info

            if self.track_diag_plot:
                # make diagnostic plots
                for xcol in [logT, age]:
                    plot_dir = os.path.join(self.plot_dir, xcol.lower())
                    if not os.path.isdir(plot_dir):
                        os.makedirs(plot_dir)
                    self.check_ptcris(track, plot_dir=plot_dir, xcol=xcol,
                                      match_track=match_track)

            self.mtracks.append(match_track)

        if self.diag_plot:
            dp_kw = {'hb': hb, 'plot_dir': self.plot_dir,
                     'pat_kw': {'ptcri': self},
                     'match_tracks': self.mtracks}
            if hb:
                self.diag_plots([t for t in self.mtracks if t.hb], **dp_kw)
            else:
                self.diag_plots([t for t in self.mtracks if not t.hb], **dp_kw)
        logfile = os.path.join(self.log_dir, filename % self.prefix.lower())
        self.write_log(logfile, info_dict)
        return self.check_tracks(tracks, flag_dict)

    def write_log(self, logfile, info_dict):
        """write interpolation dictionary to file"""
        def sortbyval(d):
            """sortes keys and values of dict by values"""
            keys, vals = zip(*d.items())
            mkeys = np.array([k.replace('M', '') for k in d.keys()],
                             dtype=float)
            ikeys = np.argsort(mkeys)
            skeys = np.array(keys)[ikeys]
            svals = np.array(vals)[ikeys]
            return skeys, svals

        def sortbyeep(d, eep):
            keys, vals = zip(*d.items())
            all_inds = np.arange(len(keys))
            inds = np.argsort([eep.eep_list.index(k) for k in keys
                               if k in eep.eep_list])
            not_eep = [i for i in all_inds if i not in inds]
            if len(not_eep) > 0:
                inds = np.concatenate([inds, not_eep])
            skeys = np.array(keys)[inds]
            svals = np.array(vals)[inds]
            return skeys, svals

        eep = self.eep
        with open(logfile, 'w') as out:
            # sort by mass
            mass_, info = sortbyval(info_dict)
            for m, d in zip(mass_, info):
                out.write('# %s\n' % m)
                try:
                    # sort by EEP
                    keys, vals = sortbyeep(d, eep)
                except AttributeError:
                    out.write('%s\n' % d)
                    continue
                for k, v in zip(keys, vals):
                    out.write('%s: %s\n' % (k, v))
        return

    def prepare_track(self, track, outfile, hb=False,
                      hb_age_offset_fraction=0.):
        """
        Do MATCH interpolation, save files

        This function writes a logAge Mass logTe Mbol logg C/O file.

        Parameters
        ----------
        track : padova_tracks.Track object
        ptcri : padova_tracks.critical_point object
        outfile : str
            MATCH interpolated filename to write to
        hb : bool default is False
            specification of a Horizontal Branch track
        hb_age_offset_fraction : float default is 0.0
            to artificially increase the age of the HB:
            log_hb_age = log_hb_age * (1. + hb_age_offset_fraction)

        Returns
        -------
        adds MATCH interpolated data to track object as to_write attribute
        """
        header = 'logAge Mass logTe Mbol logg C/O'

        if hb:
            nticks = self.eep.nticks_hb
        else:
            nticks = self.eep.nticks

        logTe = np.array([])
        logL = np.array([])
        logAge = np.array([])
        Mass = np.array([])
        co = np.array([])

        # track = ptcri.load_eeps(track)
        if track.flag is not None:
            return
        nptcri = len(track.iptcri)

        for i in range(nptcri-1):
            if track.iptcri[i+1] == 0:
                # The end of the track
                break
            this_eep = self.get_ptcri_name(i)
            next_eep = self.get_ptcri_name(i+1)

            ithis_eep = track.iptcri[i]
            inext_eep = track.iptcri[i+1]

            mess = '%.3f %s=%i %s=%i' % (track.mass, this_eep, ithis_eep,
                                         next_eep, inext_eep)
            track.info[mess] = ''
            if ithis_eep == -1:
                track.info[mess] += 'Interpolation failed. eep == -1 '
                continue

            inds = np.arange(ithis_eep, inext_eep + 1)

            if len(inds) < 1:
                track.info[mess] += \
                    'Interpolation failed. %i inds between eeps. ' % len(inds)
                continue

            lagenew, lnew, tenew, massnew = \
                self.interpolate_along_track(track, inds, nticks[i], mess=mess)

            conew = np.zeros(len(massnew))
            if track.agb:
                cocheck = track.data['C/O'][inds]
                cocheck[np.isnan(cocheck)] = 0.
                if np.sum(cocheck) > 0:
                    ntps = len(np.nonzero(cocheck)[0])
                    if ntps == 1:
                        iloc, = np.nonzero(cocheck)
                        loc = iloc / (len(cocheck) - 1)
                        conew[int(nticks[i] * loc) - 1] = cocheck[iloc]
                    elif len(np.unique(cocheck)) == 1:
                        conew += np.unique(cocheck)
                    else:
                        # import pdb; pdb.set_trace()
                        fco = interp1d(np.log10(track.data[age][inds]),
                                       cocheck, bounds_error=0)
                        conew = fco(lagenew)

            if hb is True:
                lagenew = lagenew * (1. + hb_age_offset_fraction)

            if type(lagenew) is int:
                pdb.set_trace()

            if not len(lagenew) == len(lnew) == len(tenew):
                pdb.set_trace()

            logTe = np.append(logTe, tenew)
            logL = np.append(logL, lnew)
            logAge = np.append(logAge, lagenew)
            Mass = np.append(Mass, massnew)
            co = np.append(co, conew)

        Mbol = 4.77 - 2.5 * logL
        logg = -10.616 + np.log10(Mass) + 4.0 * logTe - logL
        # mass_arr = np.repeat(track.mass, len(logL))

        # inds, = np.nonzero(track.data[age] > 0.2)
        # umass = np.unique(track.data[mass][inds])
        eep = critical_point.Eep()
        if len(logL) not in [eep.nms, eep.nhb, eep.nlow, eep.ntot,
                             eep.nhb + eep.nok, eep.ntot + eep.nok]:
            print('array size is wrong')
            if self.debug:
                pdb.set_trace()

        to_write = np.column_stack([logAge, Mass, logTe, Mbol, logg, co])
        np.savetxt(outfile, to_write, header=header, fmt='%.8f')
        return Track(outfile, track_data=to_write, match=True)

    def check_tracks(self, tracks, flag_dict):
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

            if key not in flag_dict.keys():
                print('check_tracks: No %s in flag dict, skipping.' % key)
                continue

            if flag_dict[key] is not None:
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
                    flag_dict['M%.3f' % t.mass] = 'age decreases on track'
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
