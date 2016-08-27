from __future__ import print_function, division
import os
import pdb
from scipy.signal import argrelextrema

import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import splev, splprep

from .critical_point import CriticalPoint, Eep

from .. import utils
from ..config import *
from ..graphics.graphics import annotate_plot, hrd


def check_for_monotonic_increase(de, track):
    """check if age is increasing monotonically"""
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
                annotate_plot(track, ax, logT, logL)
                pdb.set_trace()
    return track


def debug_eep(track, inds=None, ax=None):
    if inds is None:
        inds = track.iptcri[track.iptcri > 0]
    if ax is None:
        plt.ion()
    ax = hrd(track, ax=ax)
    ax = hrd(track, inds=track.sptcri, ax=ax, plt_kw={'label': 'sandro'})
    ax = hrd(track, inds=inds, ax=ax)
    annotate_plot(track, ax, logT, logL)

    plt.legend()
    return ax


class DefineEeps(CriticalPoint):
    '''
    Define the stages if not simply using Sandro's defaults.

    0 PMS_BEG       Bressan 2012 or:
                       First model older than age = 0.2
    1 MS_BEG        Dotter 2016 or:
    2 MS_TMIN       Dotter 2016 or:
                       For very low mass stars that never reach the MSTO:
                       the age of the universe ~= 13.7 Gyr
    3 MS_TO         Dotter 2016 or:
                       For very low mass stars that never reach the MSTO:
                       the age of the universe ~= 13.7 Gyr
    4 RG_TIP       Bressan 2012:
                     1) If the last track model still has a YCEN val > 0.1
                        the TRGB is either the min te or the last model, which
                        ever comes first. (low masses)
                     2) If there is no YCEN left in the core at the last track
                        model, TRGB is the min TE where YCEN > 1-Z-0.1.
                     3) If there is still XCEN in the core (very low mass),
                        TRGB is the final track model point.
    5 HE_BEG       Dotter 2016 or:
                     LY Min after the TRGB, a dips as the star contracts,
                     and then ramps up.
    6 END_CHEB     Dotter 2016
    7 TPAGB_BEG    Marigo 2015
    '''
    def __init__(self, filename):
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

        # TP-AGB tracks
        fin = len(track.data[logL]) - 1
        finmsg = 'Last track value'
        if track.agb:
            self.add_tpagb_beg(track)
            self.add_eep(track, 'FIN', fin, message=finmsg)
        else:
            self.add_eep(track, 'TPAGB_BEG', fin, message=finmsg)
            self.add_eep(track, 'FIN', 0, message='TPAGB_BEG is FIN')

        # ZAHB tracks don't have MS.
        if track.hb:
            self.physical_age(track, 'HE_BEG')
            self.add_end_cheb(track)
            return self.check_for_monotonic_increase(track)

        # Make sure track.age > 0.2
        self.physical_age(track, 'PMS_BEG')
        self.add_ms_beg(track)

        # Low mass tracks MS set by age
        if len(track.sptcri[track.sptcri > 0]) <= 6:
            return self.low_mass_eeps(track)

        ms_tmin, ms_to = self.add_ms_eeps(track)
        end_cheb = self.add_end_cheb(track)
        rg_tip = self.add_rg_tip(track)
        he_beg = self.add_he_beg(track)

        if he_beg == 0:
            ihe_beg = self.pdict['HE_BEG']
            irest = [self.eep.eep_list[i] for i in
                     np.arange(ihe_beg, len(self.eep.eep_list))]
            [self.add_eep(track, i, 0, message=track.info['HE_BEG'],
                          loud=False)
             for i in irest]
        return self.check_for_monotonic_increase(track)

    def low_mass_eeps(self, track):
        """low mass eeps, no MSTO according to Sandro"""
        # print('{} is low mass.'.format(track.mass))
        [self.add_eep(track, cp, 0, message='No MS_TO', loud=False)
         for cp in self.please_define]
        ims_beg = track.iptcri[self.pdict['MS_BEG']]
        ims_to = self.add_eep_by_age(track, 'MS_TO', max_age)
        age_ = track.data[age][ims_to]
        ims_tmin = self.add_eep_by_age(track, 'MS_TMIN', (age_ / 2.))
        # it's possible that MS_BEG occurs after max_age / 2
        # if that's the case, take the average age between ms_beg and ms_to
        if ims_tmin <= ims_beg:
            age_ = (track.data[age][ims_to] + track.data[age][ims_beg]) / 2
            ims_tmin = self.add_eep_by_age(track, 'MS_TMIN', age_)

        return self.check_for_monotonic_increase(track)

    def add_ms_beg(self, track):
        """Add MS_BEG following as closely as possible to Dotter 2016"""
        msg = 'MIST definition'
        lx = track.data['LX']
        # LX is fraction of Ltot L_gravity is part of Ltot, so if contracting,
        # LX can be > 1. I'm ok with a contracting star starting the MS.
        if np.max(lx) > 1:
            lx /= np.max(lx)
        xcen_evo = track.data[xcen][0] - 0.0015
        inds, = np.nonzero((lx > 0.999) & (track.data[xcen] > xcen_evo))

        if len(inds) == 0:
            pms_beg = track.iptcri[self.pdict['PMS_BEG']]
            if track.mass <= self.low_mass:
                # Tc \propto (mu mH / k) (G M / R)
                tc = (10 ** track.data.LOG_R) / track.data[mass]
                ms_beg = pms_beg + np.argmin(tc[pms_beg:])
                msg += ' with min Tc'
            else:
                # LX > 0.999 may be too high.
                inds, = np.nonzero(track.data[xcen][pms_beg:] > xcen_evo)
                inds += pms_beg
                ms_beg = inds[np.argmax(track.data['LX'][inds])]
                msg += ' with max LX criterion LX={}' \
                    .format(track.data['LX'][ms_beg])
        else:
            ms_beg = inds[0]

        self.add_eep(track, 'MS_BEG', ms_beg, message=msg)
        return ms_beg

    def add_ms_eeps(self, track):
        '''Add MS_TMIN and MS_TO.'''
        xcen_mstmin = 0.3
        xcen_msto = 1e-8

        ms_to = np.nonzero(track.data[xcen] < xcen_msto)[0][0]
        msg = 'XCEN=={0:g}'.format(xcen_msto)
        self.add_eep(track, 'MS_TO', ms_to, message=msg)

        ms_tmin = np.argmin(np.abs(track.data[xcen][:ms_to] - xcen_mstmin))
        msg = 'XCEN=={0:.1f}'.format(xcen_mstmin)
        self.add_eep(track, 'MS_TMIN', ms_tmin, message=msg)

        return ms_tmin, ms_to

    def physical_age(self, track, eep_name):
        '''First line of the track with age > 0.2 yr.'''
        sidx = track.sptcri[0]
        pidx = track.iptcri[0]
        msg = "Sandro's"

        if track.data[age][sidx] <= 0.2 or track.data[age][pidx] <= 0.2:
            pidx = np.nonzero(np.round(track.data[age], 1) > 0.2)[0][0]
            msg = 'overwritten with age > 0.2'

        self.add_eep(track, eep_name, pidx, message=msg)
        return pidx

    def add_rg_tip(self, track):
        ms_to = track.iptcri[self.pdict['MS_TO']]
        ycen_ = track.data[ycen][ms_to] - 0.01
        inds = ms_to + np.nonzero(track.data[ycen][ms_to:] >= ycen_)[0]
        # Almost Dottor 2016, I give a 3 step offset because PARSEC does
        # not have MIST resolution.
        # inds = np.arange(ms_to + 3, ms_to + ind + 1)
        ilmax = np.argmax(track.data[logL][inds])
        itmin = np.argmin(track.data[logT][inds])
        if itmin == 0:
            rg_tip = inds[ilmax]
            msg = 'Max L'
        else:
            rg_tip = inds[np.min([ilmax, itmin])]
            msg = 'Max L or Min T'
        msg += ' before YCEN = YCEN_MSTO - 0.01 (Dotter 2016)'
        self.add_eep(track, 'RG_TIP', rg_tip, message=msg)
        srg_tip = track.iptcri[self.pdict['RG_TIP']]
        if srg_tip != rg_tip:
            print(srg_tip, rg_tip, track.mass, track.Z)

    def add_he_beg(self, track):
        """
        Add HEB_BEG eep at a point on the track where He is fusing at a
        consistent rate. Defined by Dotter 2016
        Parameters
        ----------
        track : object
            padova_tracks.Track object

        Returns
        -------
        he_beg : int
            track.data index of HE_BEG
        """
        #if track.mass == 1.65:
        #    import pdb; pdb.set_trace()

        if not track.hb and (track.mass <= self.hbmaxmass):
            msg = 'No HE_BEG M={0:.4f} Z={1:.4f}'.format(track.mass, track.Z)
            hebeg = 0
        else:
            msg = 'Tmin while YCEN > YCEN at RGB_TIP - 0.03'
            itrgb = track.iptcri[self.pdict['RG_TIP']]
            ycen_ = track.data[ycen][itrgb] - 0.03
            inds, = np.nonzero(track.data[ycen][itrgb:] > ycen_) + itrgb
            tc = (10 ** track.data.LOG_R) / track.data[mass]
            hebeg = inds[np.argmin(tc[inds])]
            if hebeg == itrgb:
                hebeg += 3
                msg += ' +3 step offset'

        self.add_eep(track, 'HE_BEG', hebeg, message=msg)
        return hebeg

    def add_end_cheb(self, track):
        """Add end core He burning defined as YCEN = 1e-4 by Dotter 2016"""
        tpagb_start = -1

        if track.agb:
            tpagb_start = track.iptcri[self.pdict['TPAGB_BEG']]

        end_cheb = np.argmin(np.abs(track.data[ycen][:tpagb_start] - 1e-4))
        self.add_eep(track, 'END_CHEB', end_cheb, message='YCEN=1e-4')
        return end_cheb

    def add_tpagb_beg(self, track):
        '''Add the beginning of the a colibri track.'''
        # step only is non nan during the TP-AGB phase.
        step = track.data['step']
        # (this is a hack to make np.nanargmin if it exited)
        tp_start = step.tolist().index(np.nanmin(step))
        self.add_eep(track, 'TPAGB_BEG', tp_start, message='TPAGB start')
        return tp_start

    def add_eep_by_age(self, track, eep_name, age_):
        """Add an EEP closest to age_"""
        iage = np.argmin(np.abs(track.data[age] - age_))
        msg = 'By age={0:g}, is {1:g}'.format(age_, track.data[age][iage])

        self.add_eep(track, eep_name, iage, message=msg)
        return iage

    def add_eep(self, track, eep_name, ind, message=None, loud=False):
        '''Add or replace track.iptcri value based on self.pdict[eep_name]'''
        message = message or ''

        track.iptcri[self.pdict[eep_name]] = ind
        track.__setattr__('i{:s}'.format(eep_name.lower()), ind)

        if len(message) > 0:
            track.info['{0:s}'.format(eep_name)] = message

        if loud:
            print(track.mass, eep_name, ind, message)
        return
