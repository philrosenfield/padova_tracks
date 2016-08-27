import os

import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import splev, splprep, interp1d

from .. import utils
from ..config import logT, logL, mass, age, MODE, EXT


def interpolate_(track, inds, xcol=logT, ycol=logL, paracol=age,
                 parametric=True, zcol=None, k=3, s=0., tol=1e-6,
                 linear=False):
    """
    Call scipy.optimize.splprep. Will also rid the array
    of duplicate values.

    if parametric is True use age with logT and logL
       if linear is False use log10 Age

    Parameters
    ----------
    track: object
        padova_tracks.Track object

    inds : list
        segment of track to do the interpolation

    k, s: float, int [3, 0]
        spline level and smoothing see scipy.optimize.splprep

    parametric : bool [True]
        do parametric interpolation

    xcol, ycol, paracol, zcol : str, str, str, str
        xaxis column name, xaxis column name, column for parametric
        (probably logT, logL, age, MASS)

    Returns
    -------
    tckp : array
        an input to scipy.optimize.splev

    step_size : float
        recommended stepsize to use for interpolated array

    non_dupes : list
        non duplicated indices

    NOTE : The dimensionality of tckp will change if
           using parametric_interp
    """
    zdata = None
    if zcol is not None:
        zdata = track.data[zcol][inds]

    if len(inds) <= 3:
        non_dupes = np.arange(len(inds))
        k = 1
    else:
        non_dupes = utils.remove_dupes(track.data[xcol][inds],
                                       track.data[ycol][inds],
                                       track.data[paracol][inds],
                                       inds4=zdata, tol=tol)

        if len(non_dupes) <= 3:
            k = 1

    if len(non_dupes) <= 1:
        return -1, -1

    xdata = track.data[xcol][inds][non_dupes]
    ydata = track.data[ycol][inds][non_dupes]

    if zcol is not None:
        zdata = zdata[non_dupes]

    arr = [xdata, ydata]
    if parametric:
        paradata = track.data[paracol][inds][non_dupes]
        if not linear:
            paradata = np.log10(paradata)
        arr = [paradata, xdata, ydata]

    if zcol is not None:
        arr.append(zdata)

    ((tckp, u), fp, ier, msg) = splprep(arr, s=s, k=k, full_output=1)
    if ier > 0:
        print(fp, ier, msg)
    return tckp, non_dupes


def interpolate_along_track(track, inds, nticks, zcol=None, mess=None,
                            zmsg=None, tpagb_kw=None):
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
    """
    zmsg = zmsg or ''
    tpagb_kw = tpagb_kw or {}
    linear = False

    if len(inds) < 1:
        track.info[mess] = 'not enough points for interpolation'
        return -1, -1, -1, -1

    # need to check if mass loss is important enough to include
    # in interopolation
    mass_ = track.data[mass][inds]
    if np.sum(np.abs(np.diff(mass_))) > 0.01:
        frac_mloss = len(np.unique(mass))/float(len(mass))
        if frac_mloss >= 0.25:
            zcol = mass
        else:
            print('interpolate_along_track: {0:s} frac mloss, mi, mf: ',
                  '{1:g} {2:g} {3:g}'.format(mess, frac_mloss,
                                             mass_[0], mass_[-1]),
                  'Hopefully it\'s not important: not interpolating mass.')

    if track.agb and 'TPAGB' in mess.split('=')[0]:
        return interpolate_tpagb(track, inds, nticks, mess=mess, zcol=zcol,
                                 zmsg=zmsg, **tpagb_kw)

    agediff = np.diff([np.log10(track.data[age][inds[0]]),
                       np.log10(track.data[age][inds[-1]])])
    if agediff < 1e-3:
        linear = True

    if linear:
        track.info[mess] += ' linear interp in age'

    tckp, non_dupes = interpolate_(track, inds, linear=linear, zcol=zcol)
    arb_arr = np.linspace(0, 1, nticks)

    if isinstance(non_dupes, int):
        # if one variable doesn't change, call interp1d
        lagenew, tenew, lnew, massnew = \
            call_interp1d(track, inds, nticks, mess=mess, linear=linear)
    else:
        if len(non_dupes) <= 3:
            # linear interpolation was automatic in interpolate_
            track.info[mess] += ' linear interpolation'
        # normal intepolation
        if zcol is None:
            lagenew, tenew, lnew = splev(arb_arr, tckp)
        else:
            track.info[mess] += ' interpolation with {}'.format(zcol)
            lagenew, tenew, lnew, massnew = splev(arb_arr, tckp)

        # if non-monotonic increase in age, call interp1d
        all_positive = np.diff(lagenew) > 0
        if False in all_positive:
            # probably too many nticks
            track.info[mess] += \
                ' non-monotonic increase in age. Linear interpolation'
            lagenew, lnew, tenew, massnew = \
                call_interp1d(track, inds, nticks, mess=mess,
                              linear=linear)

    if zcol is None:
        # np.array or pd.DataFrame
        try:
            massnew = np.repeat(mass_[0], len(lagenew))
        except:
            massnew = np.repeat(mass_.iloc[0], len(lagenew))

    if linear:
        lagenew = np.log10(lagenew)
    return lagenew, lnew, tenew, massnew


def call_interp1d(track, inds, nticks, mess=None, linear=False):
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
    def intp_with_constant(constarr, x, y, xnew):
        """interpolation with one variable repeated as a constant"""
        cnew = np.zeros(len(xnew)) + constarr[0]
        f = interp1d(x, y, bounds_error=0)
        return cnew, f(xnew)

    msg = ' Match interpolation by interp1d'
    cfmt = ', with a single value for {0:s}'
    # shorthands
    mass_ = track.data[mass][inds]
    logl = track.data[logL][inds]
    logte = track.data[logT][inds]
    lage = track.data[age][inds]
    if not linear:
        lage = np.log10(track.data[age][inds])

    # np.array or pd.DataFrame
    try:
        lagenew = np.linspace(lage[0], lage[-1], nticks)
    except:
        lagenew = np.linspace(lage.iloc[0], lage.iloc[-1], nticks)

    # arbitrary threshold to care about mass loss on the track.
    threshold = 0.01
    if np.sum(np.abs(np.diff(mass_))) > threshold:
        msg += ' with {0:s}'.format(mass)
        fage_m = interp1d(lage, mass_, bounds_error=0)
        massnew = fage_m(lagenew)
    else:
        # np.array or pd.DataFrame
        try:
            massnew = np.repeat(mass_.iloc[0], len(lagenew))
        except:
            massnew = np.repeat(mass_[0], len(lagenew))

    if len(np.nonzero(np.diff(logl))[0]) == 0:
        # all logls are the same
        lnew, tenew = intp_with_constant(logl, lage, logte, lagenew)
        msg += cfmt.format(logL)
    elif len(np.nonzero(np.diff(logte))[0]) == 0:
        # all logtes are the same
        tenew, lnew = intp_with_constant(logte, lage, logl, lagenew)
        msg += cfmt.format(logT)
    else:
        # do them each independently
        fage_l = interp1d(lage, logl, bounds_error=0)
        lnew = fage_l(lagenew)

        fage_te = interp1d(lage, logte, bounds_error=0)
        tenew = fage_te(lagenew)

    track.info[mess] += msg
    return lagenew, lnew, tenew, massnew


def interpolate_tpagb(track, inds, nticks, mess=None, zcol=None,
                      zmsg='', outdir=None, diag=False):
    """
    Statistically, np.choice(track.data.age[inds], nticks) is enough to
    populate the TP-AGB tracks for MATCH.

    However, I want to make sure the hard work of the TP-AGB modellers
    doesn't get erased, so the method here is a bit pedantic...

    Each TP gets interpolated seperately and then appended back together.
    """
    def diag_plot(track, inds):
        """make a 3 row plot of log L, Te, Mass vs age. (C/O?)'"""
        fig, axs = plt.subplots(nrows=3, sharex=True)
        xdata = track.data[age][inds]
        for i, col in enumerate([logL, logT, mass]):
            axs[i].plot(xdata, track.data[col][inds])
            axs[i].set_ylabel(col)
            axs[i].set_xlabel(age)
        fig1, ax1 = plt.subplots()
        ax1.plot(track.data[logT][inds], track.data[logL][inds])
        ax1.invert_xaxis()
        return fig, axs, fig1, ax1

    # interpolated arrays
    lagenews = np.array([])
    lnews = np.array([])
    tenews = np.array([])
    massnews = np.array([])

    # Divide up nticks between each TP in a way that maintains morphology.
    # get_tps sets up indices between each TP
    track.get_tps()
    age_ = track.data[age]
    tpage_ = age_[inds]
    # Approximate time step
    dt = (tpage_[-1] - tpage_[0]) / nticks
    # About how many points to have to set up equal time steps per TP
    dtp = [(age_[track.tps[j][-1]] - age_[track.tps[j][0]]) / dt
           for j in range(len(track.tps))]

    # need to be int...
    dtpr = np.round(dtp, 0)
    # Each TP should have at least 3 sampled points.
    dtpr[dtpr <= 3] = 3

    # Rounding errors amy make sum(dtpr) != nticks.
    if np.sum(dtpr) != nticks:
        off = nticks - sum(dtpr)
        if off < 0:
            # too many interpolation points but don't take away from 3.
            idx, = np.nonzero(dtpr[dtpr > 3])
            ishift = np.argmax(dtpr[idx])
            shift = dtpr[ishift]
            frac = np.abs(1 - (shift + off) / shift)
            # don't decrease the number of points by more than 10%
            if frac > 0.1:
                # divy them up
                idx = np.argsort(dtpr)[::-1]
                dtpr[idx[:int(np.abs(off))]] -= 1
            else:
                # take them from the most
                dtpr[ishift] += off
        else:
            # too few interpolation points
            ishift = np.argmin(dtpr)
            shift = dtpr[ishift]
            # eh, you can always add a couple more.
            dtpr[ishift] += off

    subticks = np.array(dtpr, dtype=int)

    if np.sum(subticks) != nticks:
        # ok, something went wrong... maybe forcing a min value of N
        # interpolation points and having too few nticks and too many TPs?
        import pdb
        pdb.set_trace()

    if diag:
        fig, axs, fig1, ax1 = diag_plot(track, inds)

    # floor: sometimes huge drop in L with interpolation
    # this resets the lowest interpolated values to the input min L value.
    # (Not implemented for Teff or ceilings, and this is aethsetic)
    floor = False
    for i, itp in enumerate(track.tps):
        arb_arr = np.linspace(0, 1, subticks[i])
        # interpolate over 6 or more, this is the bulk of the TP but not
        # exteremly fast expansion (status=3 or 5).
        # That just blows up interpolators into tiny pieces.
        iintp, = np.nonzero(track.data.status[itp] >= 6)
        k = 3
        if len(iintp) < 3:
            k = 1
        xdata = track.data[age][itp[iintp]]
        tckp, u = splprep([xdata, track.data[logL][itp[iintp]]], s=0, k=k)
        lagenew, lnew = splev(arb_arr, tckp)

        tckp, u = splprep([xdata, track.data[logT][itp[iintp]]], s=0, k=k)
        _, tenew = splev(arb_arr, tckp)

        tckp, u = splprep([xdata, track.data[mass][itp[iintp]]], s=0, k=k)
        _, massnew = splev(arb_arr, tckp)

        if floor:
            idip, = np.nonzero(track.data.status[itp] == 6)
            lfloor = itp[idip[np.argmin(track.data[logL][idip])]]
            lnew[lnew < lfloor] = lfloor

        lagenews = np.append(lagenews, lagenew)
        tenews = np.append(tenews, tenew)
        lnews = np.append(lnews, lnew)
        massnews = np.append(massnews, massnew)

        if diag:
            # for each TP (so we have different colors):
            axs[0].plot(lagenew, lnew)
            axs[1].plot(lagenew, tenew)
            axs[2].plot(lagenew, massnew)
            ax1.plot(tenew, lnew)

    if diag:
        outfile = '{}_tpagb_diag.png'.format(track.name)
        if outdir is not None:
            outfile = os.path.join(outdir, outfile)
        fig.savefig(outfile)
        fig1.savefig(outfile.replace('.png', '_hrd.png'))
        plt.close('all')
    return lagenews, lnews, tenews, massnews
