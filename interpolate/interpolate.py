import numpy as np
from scipy.interpolate import splev, splprep

from .. import utils
from ..config import *


def debug_interp(xnew, ynew):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(xnew, ynew)
    return ax


class Interpolator(object):
    def __init__(self):
        pass

    def _interpolate(self, track, inds, k=3, s=0., min_step=1e-3,
                     parametric=True, xfunc=None, yfunc=None,
                     parafunc=None, xcol=logT, ycol=logL,
                     paracol=age, zcol=None):
        """
        Call scipy.optimize.splprep. Will also rid the array
        of duplicate values.

        if parametric_interp is True use age with logT and logL
           if linear is also False use log10 Age

        Parameters
        ----------
        track: object
            rsp.padodv_tracks.Track object

        inds : list
            segment of track to do the interpolation

        k, s: float, int [3, 0]
            spline level and smoothing see scipy.optimize.splprep

        min_step : float
            minimum stepsize for resulting interpolated array

        parametric : bool [True]
            do parametric interpolation

        xfunc, yfunc : string
            wih eval, function to operate on the xdata,ydata
            eval('%s(data)' % func

        parafunc : string
            wih eval, function to operate on the parametric data
            eval('%s(paradata)' % parafunc

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
        if zcol is not None:
            zdata = track.data[zcol][inds]
        else:
            zdata = None
        just_two = False
        if not parametric:
            just_two = True

        if len(inds) <= 3:
            # print('fewer than 3 indices passed, linear interpolation')
            non_dupes = np.arange(len(inds))
            k = 1
        else:
            non_dupes = self.remove_dupes(track.data[xcol][inds],
                                          track.data[ycol][inds],
                                          track.data[paracol][inds],
                                          inds4=zdata,
                                          just_two=just_two)

            if len(non_dupes) <= 3:
                # print('fewer than 3 non_dupes, linear interpolation')
                k = 1
        if len(non_dupes) <= 1:
            return -1, -1, -1
        xdata = track.data[xcol][inds][non_dupes]
        ydata = track.data[ycol][inds][non_dupes]
        if zcol is not None:
            zdata = zdata[non_dupes]

        if xfunc is not None:
            xdata = eval('%s(xdata)' % xfunc)
        if yfunc is not None:
            ydata = eval('%s(ydata)' % yfunc)

        if parametric:
            paradata = track.data[paracol][inds][non_dupes]
            if parafunc is not None:
                paradata = eval('%s(paradata)' % parafunc)
            arr = [paradata, xdata, ydata]
            if zcol is not None:
                arr = [paradata, xdata, ydata, zdata]
        else:
            arr = [xdata, ydata]
            if zcol is not None:
                arr = [xdata, ydata, zdata]
        ((tckp, u), fp, ier, msg) = splprep(arr, s=s, k=k, full_output=1)
        if ier > 0:
            print(fp, ier, msg)
        ave_data_step = np.round(np.mean(np.abs(np.diff(xdata))), 4)
        step_size = np.max([ave_data_step, min_step])

        return tckp, step_size, non_dupes

    def remove_dupes(self, inds1, inds2, inds3, just_two=False,
                     inds4=None):
        """
        Remove duplicates so as to not brake the interpolator.

        Parameters
        ----------
        inds1, inds2, inds3 : list or np.array()
            to find unique values, must be same length
        just_two : Bool [False]
            do not include inds3

        Returns
        -------
        non_dupes : list
            indices of input arrays that are not duplicates
        """
        def unique_seq(seq, tol=1e-6):
            '''
            Not exactly unique, but only points that are farther
            apart than some tol
            '''
            return np.nonzero(np.abs(np.diff(seq)) >= tol)[0]

        un_ind1 = unique_seq(inds1)
        un_ind2 = unique_seq(inds2)
        if not just_two:
            un_ind3 = unique_seq(inds3)

        if inds4 is not None:
            un_ind4 = unique_seq(inds4)
            non_dupes = list(set(un_ind1) & set(un_ind2) &
                             set(un_ind3) & set(un_ind4))
        elif just_two:
            non_dupes = list(set(un_ind1) & set(un_ind2))
        else:
            non_dupes = list(set(un_ind1) & set(un_ind2) & set(un_ind3))

        # print(len(non_dupes))
        return non_dupes

    def peak_finder(self, track, col, inds, get_max=False,
                    more_than_one='max of max', mess_err=None, ind_tol=3,
                    dif_tol=0.01, less_linear_fit=False,
                    parametric_interp=True, steepest=False):
        '''
        finds some peaks! Usually interpolates and calls a basic diff finder,
        though some higher order derivs of the interpolation are sometimes
        used.
        '''
        if len(inds) < ind_tol:
            # sometimes there are not enough inds to interpolate
            print('Peak finder M%.3f: less than %i points=%i. Skipping.'
                  % (track.mass, ind_tol, len(inds)))
            import pdb; pdb.set_trace()
            return 0

        # use age, so logl(age), logte(age) for parametric interpolation
        tckp, step_size, non_dupes = \
            self._interpolate(track, inds, parametric=parametric_interp)

        if step_size == -1:
            # sometimes there are not enough inds to interpolate
            return 0

        arb_arr = np.arange(0, 1, step_size)
        if parametric_interp:
            agenew, xnew, ynew = splev(arb_arr, tckp)
            dagenew, dxnew, dynew = splev(arb_arr, tckp, der=1)
            intp_col = ynew
            nintp_col = xnew
            dydx = dynew / dxnew
            if col == logT:
                intp_col = xnew
                nintp_col = ynew
                dydx = dxnew / dxnew
        else:
            # interpolate logl, logte.
            xnew, ynew = splev(arb_arr, tckp)
            intp_col = ynew
            nintp_col = xnew
            if col == logT:
                intp_col = xnew
                nintp_col = ynew

        # find the peaks!
        if less_linear_fit:
            if track.mass < 5.:
                axnew = xnew
                # calculate slope using polyfit
                m, b = np.polyfit(nintp_col, intp_col, 1)
            else:
                axnew = np.arange(nintp_col.size)
                m = (intp_col[-1] - intp_col[0]) / (axnew[-1] - axnew[0])
                b = axnew[0]
            # subtract linear fit, find peaks
            aynew = intp_col - (m * axnew + b)
            peak_dict = utils.find_peaks(aynew)
        else:
            peak_dict = utils.find_peaks(intp_col)

        if get_max:
            mstr = 'max'
        else:
            mstr = 'min'

        if peak_dict['%sima_number' % mstr] > 0:
            iextr = peak_dict['%sima_locations' % mstr]
            if more_than_one == 'max of max':
                almost_ind = iextr[np.argmax(intp_col[iextr])]
            elif more_than_one == 'min of min':
                if parametric_interp:
                    almost_ind = np.argmax(dydx)
                else:
                    almost_ind = iextr[np.argmin(intp_col[iextr])]
            elif more_than_one == 'last':
                almost_ind = iextr[-1]
            elif more_than_one == 'first':
                almost_ind = iextr[0]
        else:
            # no maxs found.
            if not steepest:
                func = np.__getattribute__('arg%s' % mstr)
                # if mess_err is not None:
                #    print(mess_err)
                almost_ind = func(intp_col)
                if less_linear_fit:
                    almost_ind = intp_col[func(aynew)]
            else:
                _, d2xnew, d2ynew = splev(arb_arr, tckp, der=2)
                btnextrema = np.arange(*np.sort([np.argmax(d2ynew/d2xnew),
                                                 np.argmin(d2ynew/d2xnew)]))
                almost_ind = btnextrema[np.argmin(intp_col[btnextrema])]


        if parametric_interp:
            # closest point in interpolation to data
            ind, dif = \
                utils.closest_match2d(almost_ind,
                                      track.data[col][inds][non_dupes],
                                      np.log10(track.data[age][inds][non_dupes]),
                                      intp_col, agenew)
        else:
            # closest point in interpolation to data
            ind, dif = \
                utils.closest_match2d(almost_ind,
                                      track.data[logT][inds][non_dupes],
                                      track.data[logL][inds][non_dupes],
                                      xnew, ynew)

        if ind == -1:
            # didn't find anything.
            return ind

        if dif > dif_tol:
            # closest match was too far away from orig.
            # if diff_err is not None:
            #     print(diff_err)
            # else:
            #     print('bad match %s-%s M=%.3f' % (eep1, eep2, track.mass))
            return -1
        return inds[non_dupes][ind]

    def interpolate_along_track(self, track, inds, nticks, zcol=None,
                                mess=None, zmsg=''):
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
            logl = track.data[logL][inds]
            logte = track.data[logT][inds]
            lage = np.log10(track.data[age][inds])
            mass_ = track.data[mass][inds]
            try:
                lagenew = np.linspace(lage[0], lage[-1], nticks)
            except:
                lagenew = np.linspace(lage.iloc[0], lage.iloc[-1], nticks)
            if np.sum(np.abs(np.diff(mass_))) > 0.01:
                msg += ' with mass'
            else:
                try:
                    massnew = np.repeat(mass_.iloc[0], len(lagenew))
                except:
                    massnew = np.repeat(mass_[0], len(lagenew))
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
                fage_l = interp1d(lage, track.data[logL][inds],
                                  bounds_error=0)
                lnew = fage_l(lagenew)
                fage_te = interp1d(lage, track.data[logT][inds],
                                   bounds_error=0)
                tenew = fage_te(lagenew)
                fage_m = interp1d(lage, track.data[mass][inds],
                                  bounds_error=0)
                massnew = fage_m(lagenew)
            track.info[mess] += msg

            return lagenew, lnew, tenew, massnew

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
                print('interpolate_along_track: {} frac mloss, mi, mf: ',
                      '{} {} {}'.format(mess, frac_mloss, mass_[0], mass_[-1]))

        # parafunc = np.log10 means np.log10(AGE)
        tckp, _, non_dupes = self._interpolate(track, inds, s=0,
                                               parafunc='np.log10',
                                               zcol=zcol)
        arb_arr = np.linspace(0, 1, nticks)

        if type(non_dupes) is int:
            # if one variable doesn't change, call interp1d
            lagenew, tenew, lnew, massnew = \
                call_interp1d(track, inds, nticks, mess=mess)
        else:
            if len(non_dupes) <= 3:
                # linear interpolation is automatic in self._interpolate
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
                    call_interp1d(track, inds, nticks, mess=mess)
        if zcol is None:
            try:
                massnew = np.repeat(mass_[0], len(lagenew))
            except:
                massnew = np.repeat(mass_.iloc[0], len(lagenew))
        return lagenew, lnew, tenew, massnew
