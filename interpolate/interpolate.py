import numpy as np
from scipy.interpolate import splev, splprep
from ... import utils


class Interpolator(object):
    def __init__(self):
        pass

    def _interpolate(self, track, inds, k=3, s=0., min_step=1e-4,
                     parametric=True, xfunc=None, yfunc=None,
                     parafunc=None, xcol='LOG_TE', ycol='LOG_L',
                     paracol='AGE'):
        """
        Call scipy.optimize.splprep. Will also rid the array
        of duplicate values.

        if parametric_interp is True use AGE with LOG_TE and LOG_L
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

        xcol, ycol, paracol : str, str, str
            xaxis column name, xaxis column name, column for parametric
            (probably LOG_TE, LOG_L, AGE)

        Returns
        -------
        tckp : array
            an input to scipy.optimize.splev

        step_size : float
            recommended stepsize to use for interpolated array

        non_dupes : list
            non duplicated indices

        NOTE : The dimensionality of tckp will change if using parametric_interp
        """
        just_two = False
        if not parametric:
            just_two = True

        if len(inds) <= 3:
            #print('fewer than 3 indices passed, linear interpolation')
            non_dupes = np.arange(len(inds))
            k = 1
        else:
            non_dupes = self.remove_dupes(track.data[xcol][inds],
                                          track.data[ycol][inds],
                                          track.data[paracol][inds],
                                          just_two=just_two)

            if len(non_dupes) <= 3:
                #print('fewer than 3 non_dupes, linear interpolation')
                k = 1
        if len(non_dupes) <= 1:
            return -1, -1, -1
        xdata = track.data[xcol][inds][non_dupes]
        ydata = track.data[ycol][inds][non_dupes]

        if xfunc is not None:
            xdata = eval('%s(xdata)' % xfunc)
        if yfunc is not None:
            ydata = eval('%s(ydata)' % yfunc)

        if parametric:
            paradata = track.data[paracol][inds][non_dupes]
            if parafunc is not None:
                paradata = eval('%s(paradata)' % parafunc)
            arr = [paradata, xdata, ydata]
        else:
            arr = [xdata, ydata]

        ((tckp, u), fp, ier, msg) = splprep(arr, s=s, k=k, full_output=1)
        if ier > 0:
            print(fp, ier, msg)
        ave_data_step = np.round(np.mean(np.abs(np.diff(xdata))), 4)
        step_size = np.max([ave_data_step, min_step])

        return tckp, step_size, non_dupes

    def remove_dupes(self, inds1, inds2, inds3, just_two=False):
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
            non_dupes = list(set(un_ind1) & set(un_ind2) & set(un_ind3))
        else:
            non_dupes = list(set(un_ind1) & set(un_ind2))
        #print(len(non_dupes))
        return non_dupes


    def peak_finder(self, track, col, eep1, eep2, get_max=False, sandro=True,
                    more_than_one='max of max', mess_err=None, ind_tol=3,
                    dif_tol=0.01, less_linear_fit=False,
                    parametric_interp=True):
        '''
        finds some peaks! Usually interpolates and calls a basic diff finder,
        though some higher order derivs of the interpolation are sometimes used.
        '''
        # slice the array
        # either take inds or the EEP names
        if type(eep1) != str:
            inds = np.arange(eep1, eep2)
        else:
            inds = self.ptcri.inds_between_ptcris(track, eep1, eep2,
                                                  sandro=sandro)

        if len(inds) < ind_tol:
            # sometimes there are not enough inds to interpolate
            print('Peak finder %s-%s M%.3f: less than %i points = %i. Skipping.'
                  % (eep1, eep2, track.mass, ind_tol, len(inds)))
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
            dxnew, dynew, dagenew = splev(arb_arr, tckp, der=1)
            intp_col = ynew
            nintp_col = xnew
            dydx = dxnew / dynew
            if col == 'LOG_TE':
                intp_col = xnew
                nintp_col = ynew
                dydx = dynew / dxnew
        else:
            # interpolate logl, logte.
            xnew, ynew = splev(arb_arr, tckp)
            intp_col = ynew
            nintp_col = xnew
            if col == 'LOG_TE':
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
            intp_col = intp_col - (m * axnew + b)

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
            if mess_err is not None:
                print(mess_err)
            return -1

        if parametric_interp:
            # closest point in interpolation to data
            ind, dif = \
                utils.closest_match2d(almost_ind,
                                      track.data[col][inds][non_dupes],
                                      np.log10(track.data.AGE[inds][non_dupes]),
                                      intp_col, agenew)
        else:
            # closest point in interpolation to data
            ind, dif = \
                utils.closest_match2d(almost_ind,
                                      track.data.LOG_TE[inds][non_dupes],
                                      track.data.LOG_L[inds][non_dupes],
                                      xnew, ynew)

        if ind == -1:
            # didn't find anything.
            return ind

        if dif > dif_tol:
            # closest match was too far away from orig.
            #if diff_err is not None:
            #    print(diff_err)
            #else:
            #    print('bad match %s-%s M=%.3f' % (eep1, eep2, track.mass))
            return -1
        return inds[non_dupes][ind]
