"""Interpolate tracks for match and check the interpolations"""
from __future__ import print_function
import numpy as np
import os

from scipy.interpolate import interp1d

from . import fileio
from .config import mass, logT, age, logL
from .eep.define_eep import DefineEeps
from .interpolate.interpolate import interpolate_along_track
from .tracks.track_set import TrackSet
from .tracks.track import Track
from .graphics import diagnostics as diag


class TracksForMatch(TrackSet, DefineEeps):
    """
    This class is for interpolating tracks for use in MATCH. DefineEeps is made
    for one track at a time, TracksForMatch takes a track set as input.
    """
    def __init__(self, *args, **kwargs):
        TrackSet.__init__(self, **kwargs)
        DefineEeps.__init__(self, self.ptcri_file)
        self.debug = kwargs.get('debug', False)
        self.load_critical_points(self.tracks)
        if hasattr(self, 'hbtracks'):
            self.load_critical_points(self.hbtracks)
        self.set_directories()

    def set_directories(self):
        default_outdir = \
            os.path.join(self.tracks_dir, 'match', self.prefix)
        default_plotdir = \
            os.path.join(self.tracks_dir, 'diag_plots', self.prefix)
        self.outfile_dir = self.outfile_dir or default_outdir
        self.plot_dir = self.plot_dir or default_plotdir
        self.log_dir = self.log_dir or os.path.join(self.tracks_dir, 'logs')
        self.logfmt = 'match_interp_{0:s}.log'
        self.intpfmt = 'match_{0:s}.dat'  # track.name here
        if hasattr(self, 'hbtracks'):
            self.hblogfmt = 'match_interp_hb_{0:s}.log'
        for d in [self.outfile_dir, self.plot_dir, self.log_dir]:
            fileio.ensure_dir(d)

    def match_interpolation(self, hb=False):
        """
        Call the MATCH interpolator, make diagnostic plots

        This function writes two file types:
        match_interp logfile: Any error collections from define_eep or here.
        lines_check file: Summary of the interpolated files lengths.
        """
        # to pass the flags to another class
        flag_dict = {}
        info_dict = {}
        self.mtracks = []

        tracks = self.tracks
        filename = self.logfmt
        if hb:
            tracks = self.hbtracks
            filename = self.hblogfmt

        tpagb_plotdir = os.path.join(self.plot_dir, 'tpagb')
        fileio.ensure_dir(tpagb_plotdir)
        tpagb_kw = {'diag': True, 'outdir': tpagb_plotdir}

        for track in tracks:
            mkey = 'M{0:.3f}'.format(track.mass)
            flag_dict[mkey] = track.flag

            if track.flag is not None:
                print('skipping track M={0:.3f} because of flag: {1:s}'
                      .format(track.mass, track.flag))
                info_dict[mkey] = track.flag
                continue

            # interpolate tracks for match
            mfn = self.intpfmt.format(os.path.splitext(track.name)[0])
            outfile = os.path.join(self.outfile_dir, mfn)

            if not self.overwrite_match and os.path.isfile(outfile):
                print('not overwriting {0:s}'.format(outfile))
                continue
            match_track = self.process_track(track, outfile, hb=hb,
                                             tpagb_kw=tpagb_kw)

            info_dict[mkey] = track.info

            if self.track_diag_plot:
                # make diagnostic plots
                for xcol in [logT, age]:
                    plot_dir = os.path.join(self.plot_dir, xcol.lower())
                    fileio.ensure_dir(plot_dir)
                    diag.check_ptcris(track, plot_dir=plot_dir, xcol=xcol,
                                      match_track=match_track)

            self.mtracks.append(match_track)

        if self.diag_plot:
            dp_kw = {'hb': hb, 'plot_dir': self.plot_dir,
                     'pat_kw': {'ptcri': self},
                     'match_tracks': self.mtracks}
            diag.diag_plots(tracks, **dp_kw)

        logfile = os.path.join(self.log_dir,
                               filename.format(self.prefix.lower()))
        self.write_log(logfile, info_dict)
        return self.check_tracks(tracks, flag_dict)

    def write_log(self, logfile, info_dict):
        """write interpolation dictionary to file"""
        def sortbyval(d):
            """sortes keys and values of dict by mass values"""
            keys, vals = zip(*d.items())
            mkeys = np.array([k.replace('M', '') for k in d.keys()],
                             dtype=float)
            ikeys = np.argsort(mkeys)
            skeys = np.array(keys)[ikeys]
            svals = np.array(vals)[ikeys]
            return skeys, svals

        def sortbyeep(d, eep):
            """sorts the eep names by eep.eep_list index order"""
            keys, vals = zip(*d.items())
            all_inds = np.arange(len(keys))
            # eep.eep_list has hb eeps as well as non-hb eeps.
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
                out.write('# {0:s}\n'.format(m))
                try:
                    # sort by EEP
                    keys, vals = sortbyeep(d, eep)
                except AttributeError:
                    out.write('{0:s}\n'.format(d))
                    continue
                for k, v in zip(keys, vals):
                    out.write('{0:s}: {1:s}\n'.format(k, v))
        return

    def process_track(self, track, outfile, hb=False, tpagb_kw=None):
        """
        Do MATCH interpolation, save files

        This function writes a logAge Mass logTe Mbol logg C/O file.

        Parameters
        ----------
        track : padova_tracks.Track object
        outfile : str
            MATCH interpolated filename to write to
        hb : bool default is False
            specification of a Horizontal Branch track

        Returns
        -------
        adds MATCH interpolated data to track object as to_write attribute
        """
        tpagb_kw = tpagb_kw or {}
        header = 'logAge Mass logTe Mbol logg C/O'
        msg = '{:.3f} {:s}={:d} {:s}={:d}'
        nticks = self.eep.nticks
        if hb:
            nticks = self.eep.nticks_hb

        pdict = self.pdict

        logTe = np.array([])
        logl = np.array([])
        logAge = np.array([])
        Mass = np.array([])
        co = np.array([])

        if track.flag is not None:
            return
        nptcri = len(track.iptcri)

        for i in range(nptcri-1):
            if track.iptcri[i+1] == 0:
                # The end of the track
                break
            this_eep = \
                list(pdict.keys())[list(pdict.values()).index(i)]
            next_eep = \
                list(pdict.keys())[list(pdict.values()).index(i+1)]

            ithis_eep = track.iptcri[i]
            inext_eep = track.iptcri[i+1]

            mess = msg.format(track.mass, this_eep, ithis_eep, next_eep,
                              inext_eep)
            track.info[mess] = ''

            inds = np.arange(ithis_eep, inext_eep + 1)

            if len(inds) < 1:
                track.info[mess] += \
                    'Interp failed: {0:d} inds between eeps.'.format(len(inds))
                continue

            lagenew, lnew, tenew, massnew = \
                interpolate_along_track(track, inds, nticks[i], mess=mess,
                                        tpagb_kw=tpagb_kw)

            conew = np.zeros(len(massnew))
            # should we care about the C/O interpolation?
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
                        fco = interp1d(np.log10(track.data[age][inds]),
                                       cocheck, bounds_error=0)
                        conew = fco(lagenew)

            if type(lagenew) is int:
                import pdb
                pdb.set_trace()

            if not len(lagenew) == len(lnew) == len(tenew):
                import pdb
                pdb.set_trace()

            logTe = np.append(logTe, tenew)
            logl = np.append(logl, lnew)
            logAge = np.append(logAge, lagenew)
            Mass = np.append(Mass, massnew)
            co = np.append(co, conew)

            if self.debug:
                print(mess, track.info[mess])

        Mbol = 4.77 - 2.5 * logl
        logg = -10.616 + np.log10(Mass) + 4.0 * logTe - logl

        eep = self.load_eep()
        if len(logl) not in [self.eep.nms, self.eep.nhb, self.eep.nlow,
                             self.eep.ntot, self.eep.nhb + self.eep.nok,
                             self.eep.ntot - self.eep.nok]:
            print('array size is wrong')
            if self.debug:
                import pdb
                pdb.set_trace()

        to_write = np.column_stack([logAge, Mass, logTe, Mbol, logg, co])
        np.savetxt(outfile, to_write, header=header, fmt='%.8f')
        return Track(outfile, track_data=to_write, match=True)

    def check_tracks(self, tracks, flag_dict):
        """
        Check the tracks for identical and non-monotontically increasing ages

        Results go into self.match_info dictionary whose keys are set by
        M{:.3f}.format(track.mass) and values filled with a list of strings
        with the information.

        If the track has already been flagged (track.flag), no test occurs.

        Parameters
        ----------
        tracks: list of padova_tracks.Track objects
        """
        self.match_info = {}
        for t in tracks:
            key = 'M{0:.3f}'.format(t.mass)

            if key not in flag_dict.keys():
                print('check_tracks: No {:s} in flag dict, skipping.'
                      .format(key))
                continue

            if flag_dict[key] is not None:
                print('check_tracks: skipping {0:s}: {1:s}'.format(t.mass,
                                                                   t.flag))
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
                    flag_dict[key] = 'age decreases on track'
                # identical values of age
                bads1, = np.nonzero(np.diff(t.data[age]) == 0)
                if len(bads1) != 0:
                    if key not in self.match_info:
                        self.match_info[key] = []
                        match_info = self.match_info[key]
                    match_info.append(['{0:d} identical age values'
                                       .format(len(bads1))])
                    nears = np.concatenate([np.nonzero(j - edges < 0)[0]
                                            for j in bads1])
                    bad_inds = np.unique(nears)
                    match_info.append(['near',
                                       np.array(self.eep_list)[bad_inds]])
                    match_info.append(['log ages:', t.data[age][bads1]])
                    match_info.append(['inds:', bads1])
        return
