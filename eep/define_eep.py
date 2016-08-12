from __future__ import print_function, division
import logging
import os
import pdb
from scipy.signal import argrelextrema

import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import splev, splprep

from .critical_point import CriticalPoint, Eep

from .. import utils
from ..config import *
from ..interpolate.interpolate import Interpolator
from ..tracks.track_plots import TrackPlots

td = TrackPlots()
logger = logging.getLogger()


def check_for_monotonic_increase(de, track):
    flag = 'eeps not monotonically increasing'
    if track.flag is None:
        defined = track.iptcri[track.iptcri > 0]
        negatives = np.nonzero(np.diff(defined) <= 0)[0]
        if len(negatives) > 0:
            track.flag = flag
            print(track.flag)
            print(np.array(de.eep_list)[negatives+1])
            if de.debug:
                ax = debug_eep(track)
                td.annotate_plot(track, ax, logT, logL)
                pdb.set_trace()
    return track


def debug_eep(track, inds=None, ax=None):
    if inds is None:
        inds = track.iptcri[track.iptcri > 0]
    if ax is None:
        plt.ion()
    ax = td.hrd(track, ax=ax)
    ax = td.hrd(track, inds=track.sptcri, ax=ax, plt_kw={'label': 'sandro'})
    ax = td.hrd(track, inds=inds, ax=ax)
    td.annotate_plot(track, ax, logT, logL)

    plt.legend()
    return ax


def ibetween_ptcri(iptcri, pdict, ptcri1, ptcri2):
    try:
        inds = np.arange(iptcri[pdict[ptcri1]], iptcri[pdict[ptcri2]])
    except ValueError:
        inds = []
    return inds


class DefineEeps(Interpolator, CriticalPoint):
    '''
    Define the stages if not simply using Sandro's defaults.

    * denotes stages defined here, otherwise, taken from Sandro's defaults.
    0 PMS_BEG*     Beginning of Pre Main Sequence
                     Replaced by first model older than age = 0.2 if Sandro's
                     PMS_BEG has age <= 0.2.
    1 PMS_MIN      Minimum of Pre Main Sequence
    2 PMS_END      End of Pre-Main Sequence
    3 MS_BEG*      Starting of the central H-burning phase
                     Replaced by Log L min after PMS_END for M > high_mass.
    4 MS_TMIN*     Calculated one of four possible ways:
                     1) First Minimum in Teff
                     2) For no minimum Teff (M>50, usually >100),
                        halfway in age between MS_BEG and MS_TO
                     3) Xc=0.30 for low-mass stars (via BaSTi)
                     4) For very low mass stars that never reach Xc=0.30
                        half the age of the universe ~= 13.7/2 Gyr
    5 MS_TO*       Maximum in Teff along the Main Sequence - Turn Off (BaSTi)
                     For very low mass stars that never reach the MSTO:
                     the age of the universe ~= 13.7 Gyr
    10 RG_TIP      Tip of the RGB from Sandro is defined in 3 ways:
                     1) If the last track model still has a YCEN val > 0.1
                        the TRGB is either the min te or the last model, which
                        ever comes first. (low masses)
                     2) If there is no YCEN left in the core at the last track
                        model, TRGB is the min TE where YCEN > 1-Z-0.1.
                     3) If there is still XCEN in the core (very low mass),
                        TRGB is the final track model point.
    11 HE_BEG*     Start quiescent central He-burning phase
                     LY Min after the TRGB, a dips as the star contracts,
                     and then ramps up.
    12 YCEN_0.550* Central abundance of He equal to 0.550
    13 YCEN_0.500* Central abundance of He equal to 0.500
    14 YCEN_0.400* Central abundance of He equal to 0.400
    15 YCEN_0.200* Central abundance of He equal to 0.200
    16 YCEN_0.100* Central abundance of He equal to 0.100
    17 YCEN_0.005* Central abundance of He equal to 0.005
    18 YCEN_0.000* Central abundance of He equal to 0.000

    19 AGB_LY1*     Helium (shell) fusion first overpowers hydrogen (shell)
                    fusion
    20 AGB_LY2*     Hydrogen wins again (before TPAGB).

    21 TPAGB        Starting of the central C-burning phase
                        or beginning of TPAGB.

    Not yet implemented, no TPAGB tracks decided:
    x When the energy produced by the CNO cycle is larger than that
    provided by the He burning during the AGB (Lcno > L3alpha)
    x The maximum luminosity before the first Thermal Pulse
    x The AGB termination
    '''
    def __init__(self, filename):
        Interpolator.__init__(self)
        CriticalPoint.__init__(self, filename)

    def check_for_monotonic_increase(self, *args, **kwargs):
        return check_for_monotonic_increase(self, *args, **kwargs)

    def define_eep_stages(self, track, plot_dir=None, diag_plot=True,
                          debug=False):
        """
        Define all eeps (add as track.iptcri)

        Parameters
        ----------
        track : object
            padova_track.tracks.Track object
        plot_dir : str
            loction to put plots (if diag_plot)
        diag_plot : bool
            make diagnostic plots
        """
        self.debug = self.debug or debug

        # initalize to zero
        [self.add_eep(track, cp, 0) for cp in self.please_define]

        # TP-AGB tracks
        if track.agb:
            self.add_tpagb_eeps(track)

        # ZAHB tracks
        if track.hb:
            return self.hb_eeps(track, diag_plot=diag_plot, plot_dir=plot_dir)

        # Make sure track.age > 0.2
        self.pms_beg(track)
        # Make sure xcen isn't too large at MSBEG
        self.ms_beg_eep(track)

        # Low mass tracks
        if len(track.sptcri[track.sptcri > 0]) <= 6:
            return self.low_mass_eeps(track)

        # XCEN = 0.3, <1e-8
        self.add_ms_eeps(track)

        # high mass
        if track.mass >= high_mass:
            return self.high_mass_eeps(track)
        # Intermediate mass tracks
        else:
            return self.int_mass_eeps(track)

    def int_mass_eeps(self, track):
        nsandro_pts = len(np.nonzero(track.sptcri != 0)[0])

        ihe_beg = 0
        self.add_eep(track, 'HE_BEG', ihe_beg, message='Initializing')
        cens = self.add_cen_eeps(track)

        if cens[0] != 0:
            self.add_quiesscent_he_eep(track, 'YCEN_0.550')
            ihe_beg = track.iptcri[self.pdict['HE_BEG']]

        if ihe_beg == 0 or nsandro_pts <= 10:
            # should now make sure all other eeps are 0.
            [self.add_eep(track, cp, 0, message='no He EEPs')
             for cp in self.please_define[5:]]
        else:
            self.add_agb_eeps(track)

        return self.check_for_monotonic_increase(track)

    def low_mass_eeps(self, track):
        # no MSTO according to Sandro
        print('{} is low mass.'.format(track.mass))
        [self.add_eep(track, cp, 0, message='No MS_TO')
         for cp in self.please_define]
        ims_beg = track.iptcri[self.pdict['MS_BEG']]
        ims_to, age_ = self.add_eep_with_age(track, 'MS_TO', max_age)
        ims_tmin, _ = self.add_eep_with_age(track, 'MS_TMIN', (age_ / 2.))
        # it's possible that MS_BEG occurs after max_age / 2
        # if that's the case, take the average age between ms_beg and ms_to
        if ims_tmin <= ims_beg:
            age_ = (track.data[age][ims_to] + track.data[age][ims_beg]) / 2
            ims_tmin, _ = self.add_eep_with_age(track, 'MS_TMIN', age_)

        return self.check_for_monotonic_increase(track)

    def ms_beg_eep(self, track, xcen_=0.6):
        inds, = np.nonzero((track.data['LX'] > 0.999) &
                           (track.data[xcen] > track.data[xcen][0] - 0.0015))
        msg = 'MIST definition'
        self.add_eep(track, 'MS_BEG', inds[0], message=msg)
        return

    def hb_eeps(self, track, diag_plot=True, plot_dir=None):
        '''Call the HB EEP functions.'''
        self.add_hb_beg(track)
        self.add_cen_eeps(track)
        self.add_agb_eeps(track, diag_plot=diag_plot, plot_dir=plot_dir)
        return self.check_for_monotonic_increase(track)

    def pms_beg(self, track):
        '''check age of PMS_BEG'''
        if track.data[age][track.sptcri[0]] <= 0.2 or \
           track.data[age][track.iptcri[0]] <= 0.2:
            self.add_eep(track, 'PMS_BEG',
                         np.nonzero(np.round(track.data[age], 1) > 0.2)[0][0],
                         message='overwritten with age > 0.2')
        return

    def add_tpagb_eeps(self, track, neeps=3):
        '''
        three points for each thermal pulse
        phi_tp = 0.2
        max L
        quessent point (phi_tp max)
        '''
        # when attaching, sometimes PARSEC does NTP=1...
        step = track.data['step']
        # (this is a hack to make np.nanargmin if it exited)
        tp_start = step.tolist().index(np.nanmin(step))
        try:
            tpage = track.data[age].iloc[tp_start:] - \
                    track.data[age].iloc[tp_start]
            ages = np.linspace(0, tpage.iloc[-1], neeps)
        except:
            tpage = track.data[age][tp_start:] - \
                    track.data[age][tp_start]
            ages = np.linspace(0, tpage[-1], neeps)

        itps = tp_start + np.array([np.argmin(np.abs(tpage - a))
                                    for a in ages])
        self.add_eep(track, 'TPAGB', itps[0], message='TPAGB start')

        for i, itp in enumerate(itps[1:]):
            self.add_eep(track, 'TPAGB{}'.format(i + 1), itp,
                         message='TPAGB age {}'.format(i + 1))
        return

    def high_mass_eeps(self, track):
        ms_to = track.iptcri[self.pdict['MS_TO']]

        ind = np.argmin(np.abs(track.data[ycen][ms_to:] -
                               (track.data[ycen][ms_to] - 0.01)))
        # Almost Dottor2016, I give a 3 step offset because PARSEC does
        # not have MIST resolution.
        inds = np.arange(ms_to + 3, ms_to + ind + 1)
        rg_tip = inds[np.min([np.argmax(track.data[logL][inds]),
                              np.argmin(track.data[logT][inds])])]
        msg = 'Max L or Min T before YCEN = YCEN,MSTO - 0.01 (Dotter2016)'
        self.add_eep(track, 'RG_TIP', rg_tip, message=msg)

        cens = self.add_cen_eeps(track, istart=rg_tip)
        he_beg = self.add_quiesscent_he_eep(track, cens[0], start=rg_tip)

        self.add_agb_eeps(track)

        fin = len(track.data[logL]) - 1
        self.add_eep(track, 'TPAGB', fin, message='Last track value')
        if cens[-1] >= fin:
            track.flag = 'final track point < final ycen M=%.3f' % track.mass

        return self.check_for_monotonic_increase(track)

    def add_cen_eeps(self, track, tol=0.01, istart=None):
        '''
        Add YCEN_%.3f eeps, if YCEN=fraction not found to tol, will add 0 as
        the iptrcri, equivalent to not found.
        '''
        please_define = self.please_define
        if track.hb:
            # HB starts at the beginning
            istart = 0

        if istart is None:
            istart = track.iptcri[self.pdict['RG_TIP']]

        inds = np.arange(istart, len(track.data[ycen]))

        # use defined central values
        # e.g., YCEN_0.500
        cens = [i for i in please_define if i.startswith('YCEN')]
        cens = [float(cen.split('_')[-1]) for cen in cens]
        icens = []
        for cen in cens:
            ind, dif = utils.closest_match(cen, track.data[ycen][inds])
            icen = inds[ind]
            # some tolerance for a good match.
            if dif > tol:
                icen = 0
            self.add_eep(track, 'YCEN_%.3f' % cen, icen,
                         message='YCEN == %.6f' % track.data[ycen][icen])
            # for monotonic increase, even if there is another flare up in
            # He burning, this limits the matching indices to begin at this
            # new eep index.
            if icen > 0:
                inds = np.arange(icen + 1, len(track.data[ycen]))
            icens.append(icen)
        return icens

    def add_hb_beg(self, track):
        '''
        this is just the first line of the track with age > 0.2 yr.
        it could be placed in the load_track method. However, because
        it is not a physically meaningful eep, I'm keeping it here.
        '''
        eep_name = 'HE_BEG'
        hb_beg = track.sptcri[0]
        msg = 'Sandros He1'
        if hb_beg == 0:
            ainds, = np.nonzero(track.data[age] > 0.2)
            hb_beg = ainds[0]
            msg = 'first point with age > 0.2'
        self.add_eep(track, eep_name, hb_beg,
                     message=msg)
        return hb_beg

    def add_agb_eeps(self, track, diag_plot=False, plot_dir=None,
                     basti=False):
        '''
        This is for HB tracks... not sure if it will work for tpagb.

        These EEPS will be when 1) helium (shell) fusion first overpowers
        hydrogen (shell) fusion and 2) when hydrogen wins again (before TPAGB).
        For low-mass HB (<0.485) the hydrogen fusion is VERY low (no atm!),
        and never surpasses helium, this is still a to be done!!
        '''
        def no_agb(track):
            msg = 'no eagb '
            msg1 = msg + 'linspace between YCEN_0.000 and final track point'
            msg2 = msg + 'linspace between YCEN_0.000 and final track point'
            agb_ly1, agb_ly2 = np.round(np.linspace(track.iend_cheb,
                                        track.iptcri[-1], 4))[1:3]
            return agb_ly1, agb_ly2, msg1, msg2

        def ala_basti(track):
            ly = track.data['LY']
            lx = track.data['LX']

            # ex_inds, = np.nonzero(track.data[ycen] == 0.00)
            ex_inds, = np.nonzero(track.data[ycen] < 1e-6)

            diff_L = np.abs(ly[ex_inds] - lx[ex_inds])
            peak_dict = utils.find_peaks(diff_L)

            # if there are thermal pulses in PARSEC (not COLIBRI)
            # we should not use them.
            mins = peak_dict['minima_locations']
            if len(mins) == 0:
                # final track point
                self.add_eep(track, 'TPAGB', track.sptcri[-1],
                             message='Final track point')
                self.add_eep(track, 'TPAGB1', 0, message='No TP')
                self.add_eep(track, 'TPAGB2', 0, message='No TP')
            else:
                # first min.
                import pdb
                pdb.set_trace()
                self.add_eep(track, 'TPAGB', ex_inds[mins[0]],
                             message='Before PARSECs TP')
                self.add_eep(track, 'TPAGB1', 0, message='No TP')
                self.add_eep(track, 'TPAGB2', 0, message='No TP')

        def ala_mist(track):
            try:
                ind = np.argmin(np.abs(track.data[ycen] - 1e-6))
                try:
                    tpagb = np.nonzero(track.data['QH1'][ind:] -
                                       track.data['QHE2'][ind:] < 0.1)[0][0]
                    tpagb += ind
                    self.add_eep(track, 'TPAGB', tpagb,
                                 message='Mdiff between H and He shell < 0.1')
                except ValueError:
                    # No QH1 QHE2 in track i.e., the no shell data.
                    # The online version of Sandro's tracks do not have this.
                    ala_basti(track)
            except:
                ala_basti(track)

        end_cheb = np.argmin(np.abs(track.data[ycen] - 1e-4))
        self.add_eep(track, 'END_CHEB', end_cheb, message='YCEN=1e-4')

        if basti:
            ala_basti(track)
        else:
            ala_mist(track)

        return

    def add_ms_eeps(self, track):
        '''
        Adds  MS_TMIN and MS_TO.

        MS_TMIN found in this function is either:
        a) np.argmin(logT) between Sandro's MS_BEG and POINT_C
        b) XCEN=0.3 if low mass (low mass value is hard coded and global)
        c) the log te min on the MS found by the second derivative of
            d^2 log_te / d model^2 where model is just the inds between
            MS_BEG and POINT_C (that is, model as in model number from the
            tracks)
        d) zero

        MS_TO is either:
        a) the max log te on the MS found by
           i) by the peak log Te; or
           ii) subtracting off a linear fit in log_te vs model number
            (see c. above)
        b) zero

        if there is an error, either MS_TO or MS_TMIN will -1

        '''
        xcen_mstmin = 0.3
        xcen_msto = 1e-8

        ms_to = np.nonzero(track.data[xcen] < xcen_msto)[0][0]
        msg = 'XCEN=={0:g}'.format(xcen_msto)
        self.add_eep(track, 'MS_TO', ms_to, message=msg)

        ms_tmin = np.argmin(np.abs(track.data[xcen][:ms_to] - xcen_mstmin))
        msg = 'XCEN=={0:.1f}'.format(xcen_mstmin)
        self.add_eep(track, 'MS_TMIN', ms_tmin, message=msg)

        return ms_tmin, ms_to

    def add_eep_with_age(self, track, eep_name, age_, tol=0.1):
        iage = np.argmin(np.abs(track.data[age] - age_))
        age_diff = np.min(np.abs(track.data[age] - age_))
        msg = 'By age = %g and is %g' % (age_, track.data[age][iage])

        self.add_eep(track, eep_name, iage, message=msg)
        return iage, track.data[age][iage]

    def add_eep(self, track, eep_name, ind, message='no info',
                loud=False):
        '''
        Will add or replace the index of Track.data to track.iptcri
        '''
        track.iptcri[self.pdict[eep_name]] = ind
        track.__setattr__('i{:s}'.format(eep_name.lower()
                          .replace('.', '')), ind)
        track.info['%s' % eep_name] = message
        if loud:
            print(track.mass, eep_name, ind, message)

    def load_critical_points(self, track):
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
        if track.flag is not None:
            return track

        errfmt = '{}s do not match between track and ptcri file {} != {}'
        if hasattr(self, 'Z'):
            assert self.Z == track.Z, errfmt.format('Z', self.Z, track.Z)

        if hasattr(self, 'Y'):
            assert np.round(self.Y, 2) == np.round(track.Y, 2), \
                errfmt.format('Y', self.Y, track.Y)

        track.load_iptcri(self)
        """
        if len(track.sptcri) != len(np.nonzero(mptcri)[0]):
            if len(np.nonzero(mptcri)[0]) - len(track.sptcri) == 1:
                # Sandro truncated the track after making the ptcri file
                track.sptcri = np.append(track.sptcri,
                                         len(track.data[logL]) - 1)
                mptcri[-1] = len(track.data[logL]) + 1
            else:
                track.flag = 'ptcri file does not match track not enough MODEs'
                ind, = np.nonzero(self.masses == track.mass)
                self.fix_ptcri(self.fnames[ind])
        """
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
        return track

    def add_quiesscent_he_eep(self, track, ycen1, start='RG_TIP',
                              mist=True, diag=False):
        """
        Add HEB_BEG eep.

        He fusion starts after the RGB, but where? It was tempting to simply
        choose the min L on the HeB, but that could come after 1/2 the YCEN
        was burned for massive stars. I decided to find a place after the RGB
        where there was a bump in YCEN, a little spurt before it started
        burning He at a more consistent rate.

        The above method was not stable for all Z. I've instead moved to
        where there is a min after the TRGB in LY, that is it dips as the
        star contracts, and then ramps up.

        Parameters
        ----------
        track : object
            rsp.padova_tracks.Track object

        ycen1 : str
            end EEP to look for beginning of He burning

        start : str ['RG_TIP']
            start EEP to look for beginning of He burning

        Returns
        -------
        he_beg : int
            track.data index of HE_BEG
        """

        def ala_mist(track):
            """ T min while Ycen > Ycen_TRGB - 0.03 """
            msg = 'Tmin while YCEN > YCEN at RGB_TIP - 0.03'
            itrgb = track.iptcri[self.pdict['RG_TIP']]
            ycen_ = track.data[ycen][itrgb] - 0.03
            inds, = np.nonzero(track.data[ycen][itrgb:] > ycen_)
            ihebeg = np.argmin(track.data[logT][inds]) + itrgb
            if ihebeg == itrgb:
                return ala_phil(track)
            else:
                return ihebeg, msg

        def ala_phil(track):
            msg = 'Min LY after RG_TIP'
            if type(ycen1) != str:
                inds = np.arange(start, ycen1)
            else:
                inds = ibetween_ptcri(track.iptcri, self.pdict, start, ycen1)
            he_min = np.argmin(track.data['LY'][inds])

            if len(inds) == 0:
                msg = 'No HE_BEG M=%.4f Z=%.4f' % (track.mass, track.Z)
                return 0, msg

            # Sometimes there is a huge peak in LY before the min, find it...
            npts = inds[-1] - inds[0] + 1
            subset = npts // 3
            he_max = np.argmax(track.data['LY'][inds[:subset]])

            # Peak isn't as important as the ratio between the start and end
            rat = track.data['LY'][inds[he_max]] / track.data['LY'][inds[0]]

            # If the min is at the point next to the RG_TIP, or
            # the ratio is huge, get the min after the peak.
            amin = 0
            if he_min == 0 or rat > 10:
                amin = np.argmin(track.data['LY'][inds[he_max + 1:]])

            he_min = he_max + 1 + amin

            return inds[he_min], msg

        eep_name = 'HE_BEG'

        if mist:
            he_beg, msg = ala_mist(track)
        else:
            he_beg, msg = ala_phil(track)

        self.add_eep(track, eep_name, he_beg, message=msg)
        if self.debug:
            if diag:
                plt.ion()
                fig, ax = plt.subplots()
                t = track
                ax.plot(t.data[logT], t.data[logL])
                ax.plot(t.data[logT][he_beg], t.data[logL][he_beg], 'o',
                        label='mist')
                he_beg0 = ala_phil(track)
                ax.plot(t.data[logT][he_beg0], t.data[logL][he_beg0], 'o',
                        label='phil')
                plt.legend()
                ax.set_title('M {} Z {}'.format(track.mass, track.Z))
                plt.show()
                plt.draw()
                import pdb
                pdb.set_trace()
        return he_beg
