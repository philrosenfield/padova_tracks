"""Interpolate tracks for match and check the interpolations"""
from __future__ import print_function
import numpy as np
import os

from interpolate import Interpolator
import fileio
from eep import critical_point, DefineEeps
from tracks import TrackSet, Track, TrackDiag
from scipy.interpolate import splev
from scipy.interpolate import interp1d
import pdb
import logging

logger = logging.getLogger()

class CheckMatchTracks(critical_point.Eep, TrackSet, TrackDiag):
    """a simple check of the output from TracksForMatch."""
    def __init__(self, inputs):
        TrackDiag.__init__(self)
        critical_point.Eep.__init__(self)
        inputs.match = True
        if inputs.hb:
            inputs.hbtrack_search_term += '.dat'
            inputs.hbtrack_search_term = inputs.hbtrack_search_term.replace('PMS', '')
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

            if not key in self.flag_dict.keys():
                print('check_tracks: No %s in flag dict, skipping.' % key)
                continue

            if self.flag_dict[key] is not None:
                print('check_tracks: skipping %s: %s' % (t.mass, t.flag))
                continue

            test = np.diff(t.data.logAge) > 0
            if False in test:
                # age where does age decrease
                bads, = np.nonzero(np.diff(t.data.logAge) < 0)
                edges = np.cumsum(self.nticks)
                if len(bads) != 0:
                    if not key in self.match_info:
                        self.match_info[key] = []
                        match_info = self.match_info[key]
                    match_info.append('Age not monotonicly increasing near')
                    nears = np.concatenate([np.nonzero(j - edges < 0)[0]
                                            for j in bads])
                    bad_inds = np.unique(nears)
                    match_info.append([np.array(self.eep_list)[bad_inds],
                                       t.data.logAge[bads]])
                    self.flag_dict['M%.3f' % t.mass] = 'age decreases on track'
                # identical values of age
                bads1, = np.nonzero(np.diff(t.data.logAge) == 0)
                if len(bads1) != 0:
                    if not key in self.match_info:
                        self.match_info[key] = []
                        match_info = self.match_info[key]
                    match_info.append(['%i identical age values' % len(bads1)])
                    nears = np.concatenate([np.nonzero(j - edges < 0)[0]
                                            for j in bads1])
                    bad_inds = np.unique(nears)
                    match_info.append(['near',
                                       np.array(self.eep_list)[bad_inds]])
                    match_info.append(['log ages:', t.data.logAge[bads1]])
                    match_info.append(['inds:', bads1])


class TracksForMatch(TrackSet, DefineEeps, TrackDiag, Interpolator):
    """
    This class is for interpolating tracks for use in MATCH. DefineEeps is made
    for one track at a time, TracksForMatch takes a track set as input.
    """
    def __init__(self, inputs):
        TrackDiag.__init__(self)
        TrackSet.__init__(self, inputs=inputs)
        DefineEeps.__init__(self)
        Interpolator.__init__(self)
        self.debug = inputs.debug

    def match_interpolation(self, inputs):
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

        if not inputs.hb:
            tracks = self.tracks
            filename = 'match_interp_%s.log'
        else:
            tracks = self.hbtracks
            filename = 'match_interp_hb_%s.log'

        if inputs.outfile_dir is None or inputs.outfile_dir == 'default':
            inputs.outfile_dir = inputs.tracks_base

        self.mtracks = []
        for track in tracks:
            flag_dict['M%.3f' % track.mass] = track.flag

            if track.flag is not None:
                print('skipping track M=%.3f because of flag: %s' %
                      (track.mass, track.flag))
                info_dict['M%.3f' % track.mass] = track.flag
                continue

            # interpolate tracks for match
            outfile = os.path.join(inputs.outfile_dir, \
                        'match_%s.dat' % track.name.replace('.PMS', ''))

            if not inputs.overwrite_match and os.path.isfile(outfile):
                print('not overwriting %s' % outfile)
                continue
            match_track = self.prepare_track(track, inputs.ptcri, outfile,
                                             hb=inputs.hb,
                                             hb_age_offset_fraction=inputs.hb_age_offset_fraction)

            info_dict['M%.3f' % track.mass] = track.info

            if inputs.track_diag_plot:
                # make diagnostic plots
                for xcol in ['LOG_TE', 'AGE']:
                    plot_dir = os.path.join(inputs.plot_dir, xcol.lower())
                    if not os.path.isdir(plot_dir):
                        os.makedirs(plot_dir)
                    self.check_ptcris(track, inputs.ptcri, plot_dir=plot_dir,
                                      xcol=xcol, hb=inputs.hb,
                                      match_track=match_track)

            self.mtracks.append(match_track)

        if inputs.diag_plot:
            dp_kw = {'hb': inputs.hb, 'plot_dir': inputs.plot_dir,
                     'pat_kw': {'ptcri': inputs.ptcri}, 'match_tracks': True}
            if inputs.hb:
                self.diag_plots(self.hbtracks, **dp_kw)
            else:
                self.diag_plots(self.tracks, **dp_kw)
        logfile = os.path.join(inputs.log_dir, filename % self.prefix.lower())
        self.write_log(logfile, info_dict)
        return flag_dict

    def write_log(self, logfile, info_dict):
        """write interpolation dictionary to file"""
        def sortbyval(d):
            """sortes keys and values of dict by values"""
            keys, vals = zip(*d.items())
            mkeys = np.array([k.replace('M', '') for k in d.keys()], dtype=float)
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

        eep = critical_point.Eep()
        with open(logfile, 'w') as out:
            # sort by mass
            mass, info = sortbyval(info_dict)
            for m, d in zip(mass, info):
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

    def prepare_track(self, track, ptcri, outfile, hb=False,
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
        header = '# logAge Mass logTe Mbol logg C/O \n'

        if hb:
            nticks = ptcri.eep.nticks_hb
        else:
            nticks = ptcri.eep.nticks

        logTe = np.array([])
        logL = np.array([])
        logAge = np.array([])
        Mass = np.array([])

        ptcri_kw = {'sandro': False, 'hb': hb}
        track = ptcri.load_eeps(track, sandro=False)
        if track.flag is not None:
            return
        nptcri = len(track.iptcri)

        for i in range(nptcri-1):
            if track.iptcri[i+1] == 0:
                # The end of the track
                break
            this_eep = ptcri.get_ptcri_name(i, **ptcri_kw)
            next_eep = ptcri.get_ptcri_name(i+1, **ptcri_kw)

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

        Mbol = 4.77 - 2.5 * logL
        logg = -10.616 + np.log10(track.mass) + 4.0 * logTe - logL
        # CO place holder!
        CO = np.zeros(len(logL))
        mass_arr = np.repeat(track.mass, len(logL))

        inds, = np.nonzero(track.data['AGE'] > 0.2)
        umass = np.unique(track.data['MASS'][inds])
        #if len(umass) > 1 or umass[0] != mass_arr[0]:
        #    logger.warning('mass array is a copy of the track.mass is that kosher?')
        #    pdb.set_trace()
        eep = critical_point.Eep()
        if len(logL) not in [eep.nms, eep.nhb, eep.nlow, eep.ntot]:
            print('array size is wrong')
            if self.debug:
                pdb.set_trace()

        to_write = np.column_stack((logAge, mass_arr, logTe, Mbol, logg, CO))
        fileio.savetxt(outfile, to_write, header=header, fmt='%.8f',
                       overwrite=True)
        return Track(outfile, track_data=to_write, match=True)

    def interpolate_along_track(self, track, inds, nticks, mess=None):
        """
        interpolate along a segment of a track in one of three ways:

        1) using DefineEeps._interpolate using log Age and no smoothing
            check for age increasing and no increase in log L or Log Te.
            The interpolation will be spline with level 3 unless the length
            of the tracks slice is less than 3, in which case it will be
            linear (see DefineEeps._interpolate doc)

        2) If the track segment is made of only duplicate values in either
            Log L or Log Te, interpolate with interp1d using the
            non-duplicate values and Log Age.

        3) If 1 produced non-monotonic increase in Age, interpolate with
            interp1d as a function of age independently for both Log Te and
            Log L.

        Parameters
        ----------
        track: padova_tracks.Track object

        inds: np.array
            slice of track

        nticks: int
            the number of points interpolated arrays will have

        mess: string
            error message key from track.info

        Returns
        -------
        arrays of interpolated values for Log Age, Log L, Log Te

        Note: a little rewrite could merge a bit of this into _interpolate
        """
        def call_interp1d(track, inds, nticks, mess=None):
            """
            Call interp1d for each dimension  individually. If LOG_L or
            LOG_TE doesn't change, will return a constant array.

            Probably could/should go in ._interpolate
            Parameters
            ----------
            track: padova_tracks.Track object

            inds: np.array
                slice of track

            nticks: int
                the number of points interpolated arrays will have

            mess: string
                error message key from track.info

            Returns
            -------
            arrays of interpolated values for Log Age, Log L, Log Te
            """
            msg = ' Match interpolation by interp1d'
            logl = track.data.LOG_L[inds]
            logte = track.data.LOG_TE[inds]
            lage = np.log10(track.data.AGE[inds])
            mass = track.data.MASS[inds]
            lagenew = np.linspace(lage[0], lage[-1], nticks)
            if np.sum(np.abs(np.diff(mass))) > 0.01:
                msg += ' with mass'
            else:
                massnew = np.repeat(track.data.MASS[inds][0], len(lagenew))
            if len(np.nonzero(np.diff(logl))[0]) == 0:
                # all LOG_Ls are the same
                lnew = np.zeros(len(lagenew)) + logl[0]
                fage_te = interp1d(lage, logte, bounds_error=0)
                tenew = fage_te(lagenew)
                msg += ', with a single value for LOG_L'
            elif len(np.nonzero(np.diff(logte))[0]) == 0:
                # all LOG_TEs are the same
                tenew = np.zeros(len(lagenew)) + logte[0]
                fage_l = interp1d(lage, logl, bounds_error=0)
                lnew = fage_l(lagenew)
                msg += ', with a single value for LOG_TE'
            else:
                fage_l = interp1d(lage, track.data.LOG_L[inds],
                                  bounds_error=0)
                lnew = fage_l(lagenew)
                fage_te = interp1d(lage, track.data.LOG_TE[inds],
                                   bounds_error=0)
                tenew = fage_te(lagenew)
                fage_m = interp1d(lage, track.data.MASS[inds],
                                  bounds_error=0)
                massnew = fage_m(lagenew)
            track.info[mess] += msg

            return lagenew, lnew, tenew, massnew

        if len(inds) < 1:
            track.info[mess] = 'not enough points for interpolation'
            return -1, -1, -1, -1

        # need to check if mass loss is important enough to include
        # in interopolation
        mass = track.data.MASS[inds]
        zcol = None
        if np.sum(np.abs(np.diff(mass))) > 0.01:
            frac_mloss = len(np.unique(mass))/float(len(mass))
            if frac_mloss > 0.75:
                zcol = 'MASS'
            else:
                import pdb; pdb.set_trace()

        # parafunc = np.log10 means np.log10(AGE)
        tckp, _, non_dupes = self._interpolate(track, inds, s=0,
                                               parafunc='np.log10',
                                               zcol=zcol)
        arb_arr = np.linspace(0, 1, nticks)

        if type(non_dupes) is int:
            # if one variable doesn't change, call interp1d
            lagenew, tenew, lnew, massnew = call_interp1d(track, inds, nticks,
                                                 mess=mess)
        else:
            if len(non_dupes) <= 3:
                # linear interpolation is automatic in self._interpolate
                track.info[mess] += ' linear interpolation'
            # normal intepolation
            if zcol is None:
                lagenew, tenew, lnew = splev(arb_arr, tckp)
            else:
                track.info[mess] += ' interpolation with mass'
                lagenew, tenew, lnew, massnew = splev(arb_arr, tckp)

            # if non-monotonic increase in age, call interp1d
            all_positive = np.diff(lagenew) > 0
            if False in all_positive:
                # probably too many nticks
                lagenew, lnew, tenew, massnew = call_interp1d(track, inds, nticks,
                                                     mess=mess)
        if zcol is None:
            massnew = np.repeat(track.data.MASS[inds][0], len(lagenew))
        return lagenew, lnew, tenew, massnew
