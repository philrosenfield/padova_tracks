"""Interpolate tracks for match and check the interpolations"""

import numpy as np
import os

from scipy.interpolate import interp1d

from . import fileio
from .config import mass, logT, age, logL
from .eep.define_eep import DefineEeps
from .eep.critical_point import Eep
from .interpolate.interpolate import interpolate_along_track
from .tracks.track_set import TrackSet
from .tracks.track import Track
from .graphics.graphics import match_parsec, plot_tracks


class TracksForMatch(TrackSet, DefineEeps):
    """
    This class is for interpolating tracks for use in MATCH. DefineEeps is made
    for one track at a time, TracksForMatch takes a track set as input.
    """
    def __init__(self, *args, **kwargs):
        TrackSet.__init__(self, **kwargs)
        DefineEeps.__init__(self)
        self.debug = kwargs.get('debug', False)
        for track in self.tracks:
            track.iptcri = np.zeros(len(self.eep_list), dtype=int)
        if hasattr(self, 'hbtracks'):
            for track in self.hbtracks:
                track.iptcri = np.zeros(len(self.eep_list), dtype=int)
        self.set_directories()

    def set_directories(self):
        """define output directory structure and filename formats"""

        self.outfile_dir = self.outfile_dir or \
            os.path.join(self.tracks_dir, 'match', self.prefix)

        self.plot_dir = self.plot_dir or \
            os.path.join(self.tracks_dir, 'diag_plots', self.prefix)

        self.log_dir = self.log_dir or \
            os.path.join(self.tracks_dir, 'logs')

        [fileio.ensure_dir(d) for d in
         [self.outfile_dir, self.plot_dir, self.log_dir]]

        self.intpfmt = 'match_{0:s}.dat'  # track.name here
        self.logfmt = 'match_interp_{0:s}.log'

        if hasattr(self, 'hbtracks'):
            self.hblogfmt = 'match_interp_hb_{0:s}.log'

    def match_interpolation(self):
        """
        Call the MATCH interpolator, make diagnostic plots

        This function writes two file types:
        match_interp logfile: Any error collections from define_eep or here.
        """
        # to pass the flags to another class
        flag_dict = {}
        info_dict = {}
        self.mtracks = []

        filename = self.logfmt

        tpagb_plotdir = os.path.join(self.plot_dir, 'tpagb')
        fileio.ensure_dir(tpagb_plotdir)
        tpagb_kw = {'diag': self.track_diag_plot, 'outdir': tpagb_plotdir}

        for track in self.tracks:
            mkey = 'M{0:.3f}'.format(track.mass)
            flag_dict[mkey] = track.flag

            if track.flag is not None:
                print('skipping track M={0:.3f} because of flag: {1:s}'
                      .format(track.mass, track.flag))
                info_dict[mkey] = track.flag
                continue

            # interpolate tracks for match
            mfn = self.intpfmt.format(track.name)
            outfile = os.path.join(self.outfile_dir, mfn)

            if not self.overwrite_match and os.path.isfile(outfile):
                print('not overwriting {0:s}'.format(outfile))
                continue
            match_track = self.process_track(track, outfile, tpagb_kw=tpagb_kw)

            info_dict[mkey] = track.info

            if self.track_diag_plot:
                # make diagnostic plots
                for xcol in [logT, age]:
                    plot_dir = os.path.join(self.plot_dir, xcol.lower())
                    fileio.ensure_dir(plot_dir)
                    match_parsec(track, plot_dir=plot_dir, xcol=xcol,
                                 match_track=match_track, save=True,
                                 title=True)

            self.mtracks.append(match_track)

        if self.diag_plot:
            dp_kw = {'plot_dir': self.plot_dir,
                     'match_tracks': self.mtracks}
            plot_tracks(self.tracks, hb=False, **dp_kw)
            plot_tracks(self.tracks, hb=True, **dp_kw)

        logfile = os.path.join(self.log_dir,
                               filename.format(self.prefix.lower()))
        write_log(logfile, info_dict)
        return self.check_tracks(flag_dict)

    def process_track(self, track, outfile, tpagb_kw=None):
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
        pdict = track.pdict

        nticks = self.nticks
        if track.hb:
            nticks = self.nticks_hb

        logte = np.array([])
        logl = np.array([])
        logage = np.array([])
        mass_ = np.array([])
        co = np.array([])

        if track.flag is not None:
            return
        nptcri = len(track.iptcri)
        # print(track.mass, track.Z, track.hb,
        #       np.diff(track.iptcri[track.iptcri > 0]))
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

            inds = np.arange(ithis_eep, inext_eep)

            if len(inds) <= 1:
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
                        if np.sum(np.isnan(conew)) > 0:
                            binds = np.nonzero(np.isnan(conew))[0]
                            if len(lagenew) - 1 in binds:
                                conew[np.isnan(conew)] = \
                                    conew[np.nonzero(np.isnan(conew))[0]-1]
                            if len(lagenew) - 2 in binds:
                                conew[np.isnan(conew)] = \
                                    conew[np.nonzero(np.isnan(conew))[0]-2]
                            if 0 in binds:
                                conew[0] = conew[1]
                            if np.sum(np.isnan(conew)) > 0:
                                import pdb
                                pdb.set_trace()
            if type(lagenew) is int:
                import pdb
                pdb.set_trace()

            if not len(lagenew) == len(lnew) == len(tenew):
                import pdb
                pdb.set_trace()

            if np.sum(np.isnan(massnew)) > 0:
                import pdb
                pdb.set_trace()
                massnew = np.zeros(len(massnew)) + track.mass

            logte = np.append(logte, tenew)
            logl = np.append(logl, lnew)
            logage = np.append(logage, lagenew)
            mass_ = np.append(mass_, massnew)
            co = np.append(co, conew)

            if self.debug:
                print(mess, track.info[mess])

        mbol = 4.77 - 2.5 * logl
        logg = -10.616 + np.log10(mass_) + 4.0 * logte - logl

        if len(logl) not in [self.nms, self.nhb, self.nlow, self.ntot,
                             self.nhb + self.nok, self.ntot - self.nok]:
            print("Wrong match interp'ed track size: {2:d} M={0:.3f} Z={1:g}"
                  .format(track.mass, track.Z, len(logl)))
            # if self.debug:
            import pdb
            pdb.set_trace()

        for i, arr in enumerate([logage, mass_, logte, mbol, logg, co]):
            if np.sum(np.isnan(arr)) != 0:
                print('nans found in {} {}'.format(outfile, header.split()[i]))
                import pdb
                pdb.set_trace()

        to_write = np.column_stack([logage, mass_, logte, mbol, logg, co])
        np.savetxt(outfile, to_write, header=header, fmt='%.10f')
        return Track(outfile, track_data=to_write, match=True,
                     debug=self.debug)

    def check_tracks(self, flag_dict):
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
        def print_bad(err, ibad):
            edges = np.cumsum(self.nticks) - 1
            for i in ibad:
                near = np.argmin(np.abs(i - edges))
                print(i, near)
                err += '{:s}\n'.format(np.array(self.eep_list)[near])
                err += '{:f} {:d}\n'.format(t.data[age][i-1], i-1)
                err += '{:f} {:d}\n'.format(t.data[age][i], i)
                err += '{:f} {:d}\n'.format(t.data[age][i+1], i+1)
            return err
        self.match_info = {}
        for t in self.mtracks:
            err = ''
            key = 'M{0:.3f}'.format(t.mass)
            test = np.diff(t.data[age]) > 0
            if False in test:
                flag_dict[key] = 'age not monotonicly increasing on track'
                # age where does age decrease
                negs, = np.nonzero(np.diff(t.data[age]) < 0)
                iden, = np.nonzero(np.diff(t.data[age]) == 0)
                if len(negs) != 0:
                    err = print_bad('Age decreasing increasing near ', negs)
                # identical values of age
                if len(iden) != 0:
                    err = print_bad('{0:d} identical age value(s) near '
                                    .format(len(iden)), iden)
            if len(err) > 0:
                print(t.mass, t.Z, 'HB?:', t.hb)
                print(err)
                if key not in self.match_info:
                    self.match_info[key] = err
        return flag_dict


def write_log(logfile, info_dict):
    """write interpolation dictionary to file"""
    def sortbyval(d):
        """sortes keys and values of dict by mass values"""
        keys, vals = list(zip(*list(d.items())))
        mkeys = np.array([k.replace('M', '') for k in list(d.keys())],
                         dtype=float)
        ikeys = np.argsort(mkeys)
        skeys = np.array(keys)[ikeys]
        svals = np.array(vals)[ikeys]
        return skeys, svals

    def sortbyeep(d, eep):
        """sorts the eep names by eep.eep_list index order"""
        keys, vals = list(zip(*list(d.items())))
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

    eep = Eep()
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
