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
                if track.mass >= high_mass:
                    de.fix_ptcri(track=track)
                else:
                    ax = debug_eep(track)
                    td.annotate_plot(track, ax, logT, logL)
                    pdb.set_trace()


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
    6 SG_MAXL*     Maximum in logL for high-mass or Xc=0.0 for low-mass stars
                     (BaSTi)
    7 RG_MINL*     Minimum in logL for high-mass or Base of the RGB for
                     low-mass stars (BaSTi)

    8 RG_BMP1      The maximum luminosity during the RGB Bump
    9 RG_BMP2      The minimum luminosity during the RGB Bump
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

        if len(track.sptcri) <= 4 or track.data[xcen][track.sptcri[4]] < .6:
            # no MSTO in Sandro's track?
            print('WTF?')
            import pdb; pdb.set_trace()
            try:
                track.info['MS_BEG'] = \
                    'Sandro\'s MS_BEG but could be wrong XCEN < 0.6, %.f' % \
                    track.data[xcen][track.sptcri[4]]
            except:
                track.info['MS_BEG'] = 'Incomplete track?'
                print('incomplete track?!', track.mass)

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

        #self.add_sg_rg_eeps(track)
        #self.add_rg_minl_eep(track)
        self.check_for_monotonic_increase(track)

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
        imsbeg = track.sptcri[self.sdict['MS_BEG']]
        xcen_san = track.data[xcen][imsbeg]
        if xcen_san < xcen_:
            msg = 'overwrote Sandro for closer match to XCEN=%g' % xcen_
            imsbeg = np.argmin(np.abs(xcen_ - track.data[xcen]))
            self.add_eep(track, 'MS_BEG', imsbeg, message=msg)

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
        if track.data[age][track.sptcri[0]] <= 0.2:
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
        """
        def add_msbeg(track, start, end):
            # I think Sandro's PMS_END is a little early on the low Z massive
            # tracks...
            if track.Z < 0.004:
                pf_kw = {'get_max': True,
                         'more_than_one': 'last',
                         'parametric_interp': False}
                inds = ibetween_ptcri(track.sptcri, self.sdict, 'PMS_MIN',
                                      'NEAR_ZAM')
                pms_end = self.peak_finder(track, logL, inds, **pf_kw)
                msg = 'PMS_END fixed by max L between PMS_MIN and MS_BEG'
                self.add_eep(track, 'PMS_END', pms_end, message=msg)

            pf_kw = {'get_max': False, 'more_than_one': 'min of min'}

            msg = 'Min logL between %s and %s' % (start, end)
            inds = ibetween_ptcri(track.iptcri, self.pdict, start, end)
            ms_beg = self.peak_finder(track, logL, inds, **pf_kw)

            if ms_beg == -1:
                msg += ' without parametric'
                pf_kw['parametric_interp'] = False
                ms_beg = self.peak_finder(track, logL, inds, **pf_kw)

            self.add_eep(track, 'MS_BEG', ms_beg, message=msg)
            return ms_beg

        def add_mstmin(track, start, end, message=None):
            pf_kw = {'get_max': False,
                     'more_than_one': 'min of min',
                     'parametric_interp': False}
            inds = ibetween_ptcri(track.iptcri, self.pdict, start, end)
            ms_tmin = self.peak_finder(track, logT, inds, **pf_kw)

            self.add_eep(track, 'MS_TMIN', ms_tmin, message=message)
            return ms_tmin

        ms_to = np.nonzero(track.data[xcen] < 1e-8)[0][0]
        self.add_eep(track, 'MS_TO', ms_to, message='XCEN==0')

        if track.Z > 0.01:
            # find tmin before msbeg
            pms_end = track.sptcri[2]
            msg = 'Min logT between %s, %s' % ('PMS_END', 'MS_TO')
            ms_tmin = add_mstmin(track, pms_end, ms_to, message=msg)
            ms_beg = add_msbeg(track, 'PMS_END', 'MS_TMIN')
        else:
            # find msbeg before tmin
            ms_beg = add_msbeg(track, 'PMS_END', 'MS_TO')
            msg = 'Min logT between %s, %s' % ('MS_BEG', 'MS_TO')
            ms_tmin = add_mstmin(track, ms_beg, ms_to, message=msg)

        if ms_tmin == -1:
            msg = 'halfway in age between MS_BEG and MS_TO'
            halfway = (track.data[age][ms_to] + track.data[age][ms_beg])/2.
            ms_tmin = np.argmin(np.abs(track.data[age] - halfway))
            self.add_eep(track, 'MS_TMIN', ms_tmin, message=msg)
        """
        fin = len(track.data[logL]) - 1
        ms_to = track.iptcri[self.pdict['MS_TO']]
        cens = self.add_cen_eeps(track, istart=ms_to)
        # in some high mass tracks, YCEN drops very fast so
        # YCEN=0.005's closest match will be at 0.000, in that case
        # cens[-1] will occur at fin even though the track is complete.
        # this is a little reset to push that YCEN=0.000 between
        # YCEN_0.005 and fin
        # if track.data[ycen][cens[-2]] == 0.0:
        #    cens[-1] = (cens[-2] + fin) / 2
        #    self.add_eep(track, 'YCEN_0.000', cens[-1],
        #                 message='Reset between YCEN=0.005 and final point')
        heb_beg = self.add_quiesscent_he_eep(track, cens[0], start=ms_to)
        self.add_eep(track, 'TPAGB', fin, message='Last track value')

        ind = np.argmin(np.abs(track.data[ycen][ms_to:] -
                               (track.data[ycen][ms_to] - 0.01)))
        inds = np.arange(ms_to, ms_to + ind + 1)
        rg_tip = inds[np.min([np.argmax(track.data[logL][inds]),
                              np.argmin(track.data[logT][inds])])]
        # between ms_to and heb_beg eeps that are meaningless at high mass:
        _, rg_tip, _ = map(int, np.round(np.linspace(ms_to, heb_beg, 3)))
        msg = 'Max L or Min T before YCEN = YCEN,MSTO - 0.01 (Dotter2016)'
        #import pdb; pdb.set_trace()

        self.add_eep(track, 'RG_TIP', rg_tip, message=msg)
        self.add_agb_eeps(track)
        # there is a switch for high mass tracks in the add_cen_eeps and
        # add_quiesscent_he_eep functions. If the mass is higher than
        # high_mass the functions use MS_TO as the initial EEP for peak_finder.

        if cens[-1] >= fin:
            track.flag = 'final track point < final ycen M=%.3f' % track.mass

        # _ = np.concatenate(([ms_beg, ms_tmin, ms_to, heb_beg], cens, [fin]))
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

        def delta_eeps(track, before, after, col=logL):
            xdata = track.data[col]
            return np.abs(np.diff((xdata[before], xdata[after])))

        xcen_mstmin = 0.3
        xcen_msto = 1e-8

        ms_to = np.nonzero(track.data[xcen] < xcen_msto)[0][0]
        msg = 'XCEN=={0:g}'.format(xcen_msto)
        self.add_eep(track, 'MS_TO', ms_to, message=msg)

        ms_tmin = np.argmin(np.abs(track.data[xcen][:ms_to] - xcen_mstmin))
        msg = 'XCEN=={0:.1f}'.format(xcen_mstmin)
        self.add_eep(track, 'MS_TMIN', ms_tmin, message=msg)
        """
        inds = ibetween_ptcri(track.sptcri, self.sdict, 'MS_BEG', 'POINT_C')
        # transition mass
        #if track.mass <= 0.9 and track.mass >= 0.7 and len(inds) == 0:
        #    ind1 = track.sptcri[self.sdict['MS_BEG']]
        #    ind2 = len(track.data[logL]) - 1
        #    inds = np.arange(ind1, ind2)

        if len(inds) == 0:
            ms_tmin = 0
            msg = 'No points between MS_BEG and POINT_C'
        else:
            xdata = track.data[logT][inds]
            #ms_tmin = inds[np.argmin(xdata)]
            #delta_te = delta_eeps(track, ms_tmin, inds[0])
            #msg = 'Min logT'
            #if delta_te < .1:  # value to use interp instead
            #    # find the te min by interpolation.
            #    ms_tmin = utils.second_derivative(xdata, inds)
            #    msg = 'Min logT by interpolation'
            #xcen_tmin = track.data[xcen][ms_tmin]
            xcen_ = 0.3
            # BaSTi uses XCEN == 0.3, could put this as a keyword
            dte = np.abs(track.data[xcen][inds] - xcen_)
            ms_tmin = inds[np.argmin(dte)]
            msg = 'XCEN==%.1f' % xcen_
            print(msg)
        self.add_eep(track, 'MS_TMIN', ms_tmin, message=msg)

        if ms_tmin == 0:
            ms_to = 0
            msg = 'no MS_TMIN'
        else:
            eep2 = 'RG_BASE'
            inds = ibetween_ptcri(track.sptcri, self.sdict, 'MS_BEG', eep2)

            if inds[0] < ms_tmin:
                inds = np.arange(ms_tmin, inds[-1] + 1)
                eep2 = 'MS_TMIN'

            if len(inds) == 0:
                eep2 = 'final track point'
                inds = np.arange(ms_tmin, len(track.data[logT]) - 1)

            ms_to = np.nonzero(track.data[xcen] < 1e-8)[0][0]
            #ms_to = inds[np.argmax(track.data[logT][inds])]
            #msg = 'Max logT between MS_TMIN and %s' % eep2
            msg = 'xcen < 1e-8'
            #delta_te = delta_eeps(track, ms_to, ms_tmin)

            if delta_te < 0.01:
                # first try with parametric interpolation
                pf_kw = {'get_max': True,
                         'more_than_one': 'max of max',
                         'parametric_interp': False}
                inds = ibetween_ptcri(track.iptcri, self.pdict, 'MS_TMIN',
                                      'RG_BMP1')
                ms_to = self.peak_finder(track, logT, inds, **pf_kw)
                msg = 'Max logT between MS_TMIN and RG_BMP1'
                delta_te = delta_eeps(track, ms_to, ms_tmin)
                # if the points are too cluse try with less linear fit
                if ms_to == -1 or delta_te < 0.01:
                    pf_kw['less_linear_fit'] = True
                    ms_to = self.peak_finder(track, logT, inds, **pf_kw)
                    msg = 'Max logT between MS_TMIN and RG_BMP1 - linear fit'
                if ms_to == -1 or delta_te < 0.01:
                    pf_kw['less_linear_fit'] = False
                    pf_kw['more_than_one'] = 'last'
                    ms_to = self.peak_finder(track, logT, inds, **pf_kw)
                    msg = 'last logT Max between MS_TMIN and RG_BMP1'
            if ms_to == -1:
                # do the same as for tmin... take second deriviative
                xdata = track.data[logL][inds]
                ms_to = utils.second_derivative(xdata, inds)
                msg = 'Min logL between MS_TMIN and RG_BMP1 by interpolation'
                if ms_to == -1:
                    ms_to = inds[np.nonzero(track.data[xcen][inds] == 0)[0][0]]
                    msg = 'XCEN==0 as last resort'
            if ms_to == -1:
                # tried four ways!?!!
                msg = 'No MS_TO found after four methods'
                ms_to = 0



        self.add_eep(track, 'MS_TO', ms_to, message=msg)

        if ms_to == 0:
            if track.mass > self.low_mass:
                print('MS_TO and MS_TMIN by age limits %.4f' % track.mass)
            ms_beg = track.iptcri[self.pdict['MS_BEG']]
            fin = len(track.data)
            half_way = ms_beg + (fin - ms_beg) / 2
            self.add_eep(track, 'MS_TMIN', half_way,
                         message='Half way from MS_BEG to Fin')
            # self.add_eep_with_age(track, 'MS_TMIN', (13.7e9/2.))
            self.add_eep_with_age(track, 'MS_TO', max_age)
        """
        return ms_tmin, ms_to

    def add_sg_rg_eeps(self, track):
        '''
        Add SG_MAXL and RG_MINL eeps.

        SG_MAXL and RG_MINL are based on the morphology on the HRD.
        The morphology of the HRD in this region changes drastically across
        the mass and metallicity parameter space.

        This function calls add_sg_maxl_eep and add_rg_minl_eep in different
        ways to pull out the correct values of each.
        '''
        def check_minl(imin_l, msto, track):
            return [imin_l - msto < 50,  # msto is 50 models away
                    imin_l <= 0,  # peak_finder failed
                    # poor agreement with Sandro's rg_base
                    (np.abs(imin_l - track.sptcri[7]) > 100),
                    # still H burning on rgb.
                    np.round(track.data[xcen][imin_l], 4) > 0]

        eep2 = 'RG_BMP1'
        msto = track.iptcri[self.pdict['MS_TO']]

        more_than_one = 'last'
        if track.mass == 1.85:
            pdb.set_trace()
        # first shot, uses the last logL min after the MS_TO before RG_BMP1
        imin_l = self.add_rg_minl_eep(track, eep2=eep2,
                                      more_than_one=more_than_one)

        more_than_ones = ['last', 'first', 'min of min']
        i = 0
        while True in check_minl(imin_l, msto, track):
            if i == 3:
                break
            # try using the min of the logL mins between MS_TO and RG_BMP1
            imin_l = self.add_rg_minl_eep(track, ibad=imin_l,
                                          more_than_one=more_than_ones[i])
            i += 1
        """
        i = 0
        while True in check_minl(imin_l, msto, track):
            if i == 3:
                break
            # Try to do SG_MAXL first, though typically the issues
            # in RG_MAXL are on the RG_BMP1 side.
            imax_l = self.add_sg_maxl_eep(track, eep2=eep2)
            imin_l = self.add_rg_minl_eep(track, eep1='SG_MAXL', eep2=eep2,
                                          more_than_one=more_than_ones[i],
                                          ibad=imin_l)
            i += 1
        """
        if True in check_minl(imin_l, msto, track):
            # give up and take Sandro's value
            imin_l = track.sptcri[7]
            msg = 'Set RG_MINL to Sandro\'s RG_BASE'
            self.add_eep(track, 'RG_MINL', imin_l, message=msg)
        """
        imax_l = self.add_sg_maxl_eep(track)
        # high mass, low z, have hard to find base of rg, but easier to find
        # sg_maxl. This flips the order of finding. Also an issue is if
        # the L min after the MS_TO is much easier to find than the RG Base.
        if imax_l <= 0:
            imax_l = self.add_sg_maxl_eep(track, eep2=eep2)
            # imin_l = self.add_rg_minl_eep(track, eep1='SG_MAXL', eep2=eep2)

        if imax_l == -1:
            self.check_for_monotonic_increase(track)

        # if np.round(track.data[xcen][imin_l], 4) > 0 or imin_l < 0:
        """
        return

    def add_rg_minl_eep(self, track, eep1='MS_TO', eep2='RG_BMP1',
                        more_than_one='last', ibad=None):
        '''
        Add RG_MINL eep.

        The MIN L before the RGB for high mass or the base of the
        RGB for low mass.

        Parameters
        ----------
        track : object
            track object
        eep1 : string ['MS_TO']
            earlier EEP sent to self.peak_finder
        eep2 : string ['RG_BMP1']
            later EEP sent to self.peak_finder
        more_than_one : string
            extrema value sent to self.peak_finder

        Returns
        -------
        min_l : int
            track.data index of RG_MINL eep
        '''
        # find min_l with parametric interp and using the more_than_one min.
        pf_kw = {'more_than_one': more_than_one}
        inds = ibetween_ptcri(track.iptcri, self.pdict, eep1, eep2)
        if ibad is not None:
            inds = np.arange(ibad + 1, inds[-1] + 1)
        min_l = self.peak_finder(track, logL, inds,  **pf_kw)
        msg = '%s Min logL between %s and %s' % (more_than_one, eep1, eep2)
        msg += ' with parametric interp'
        if min_l <= 0:
            # try without parametric interp and with less linear fit
            pf_kw.update({'parametric_interp': False, 'less_linear_fit': True})
            msg = msg.replace('parametric interp', 'less linear fit')
            min_l = self.peak_finder(track, logL, inds, **pf_kw)

        if min_l <= 0:
            # try without parametric interp and without less linear fit
            pf_kw.update({'less_linear_fit': False})
            msg = msg.replace('less linear fit', '')
            min_l = self.peak_finder(track, logL, inds, **pf_kw)

        self.add_eep(track, 'RG_MINL', min_l, message=msg)
        return min_l

    def add_sg_maxl_eep(self, track, eep2='RG_MINL'):
        '''
        Add SG_MAXL eep between MS_TO and eep2

        If mass > inte_mass will take SG_MAXL to be the mean age
        between MSTO and RG_MINL

        Will first try peak finder between MS_TO and eep2 and take the
        max of max (if eep2 is RG_MINL) or last max.

        If SG_MAXL is within 10 inds of RG_MINL, will try peak finder
        with less linear fit, and if that's still a problem, will hack
        a simpler less linear fit.

        Parameters
        ----------
        track : object
            track object
        eep2 : string ['RG_MINL']
            later EEP sent to self.peak_finder

        Returns
        -------
        max_l : int
            track.data index of SG_MAXL eep
        '''
        if track.mass > inte_mass:
            irg_minl = self.pdict['RG_MINL']
            imsto = self.pdict['MS_TO']
            rg_minl = track.iptcri[irg_minl]
            msto = track.iptcri[imsto]
            mean_age = np.mean([track.data[age][msto],
                                track.data[age][rg_minl]])
            max_l = np.argmin(np.abs(track.data[age] - mean_age))
            msg = 'Set SG_MAXL to be mean age between MS_TO and RG_MINL'
            self.add_eep(track, 'SG_MAXL', max_l, message=msg)
            # self.check_for_monotonic_increase(track)
            return max_l

        extreme = 'max of max'
        if eep2 != 'RG_MINL':
            extreme = 'last'

        pf_kw = {'get_max': True, 'more_than_one': extreme,
                 'parametric_interp': False, 'less_linear_fit': False}

        inds = ibetween_ptcri(track.iptcri, self.pdict, 'MS_TO', eep2)
        max_l = self.peak_finder(track, logL, inds, **pf_kw)
        msg = '%s logL between MS_TO and %s' % (extreme, eep2)

        rg_minl = track.iptcri[self.pdict['RG_MINL']]

        if max_l == -1 or np.abs(rg_minl - max_l) < 10:
            msg = '%s logL MS_TO to %s less_linear_fit' % (extreme, eep2)
            pf_kw['less_linear_fit'] = True
            max_l = self.peak_finder(track, logL, inds, **pf_kw)

        if np.abs(rg_minl - max_l) < 10:
            import pdb
            pdb.set_trace()

        msto = track.iptcri[self.pdict['MS_TO']]
        if max_l == msto:
            print('SG_MAXL is at MS_TO!')
            print('XCEN at MS_TO (%i): %.3f' % (msto, track.data[xcen][msto]))
            max_l = -1
            msg = 'SG_MAXL is at MS_TO'
        self.add_eep(track, 'SG_MAXL', max_l, message=msg)
        return max_l

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
            subset = npts / 3
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
