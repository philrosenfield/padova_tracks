""" General untilities used throughout the package """
from __future__ import print_function, division
from scipy.interpolate import splev, splprep
import numpy as np

__all__ = ['closest_match', 'closest_match2d', 'extrap1d', 'find_peaks',
           'is_numeric', 'min_dist2d', 'second_derivative', 'sort_dict',
           'minmax', 'extrema', 'replace_', 'remove_dupes']


def remove_dupes(inds1, inds2, inds3=None, inds4=None, tol=1e-6):
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

    un_ind1 = unique_seq(inds1, tol=tol)
    un_ind2 = unique_seq(inds2, tol=tol)
    non_dupes = list(set(un_ind1) & set(un_ind2))

    if inds3 is not None:
        un_ind3 = unique_seq(inds3, tol=tol)
        non_dupes = list(set(un_ind1) & set(un_ind2) & set(un_ind3))

    if inds4 is not None:
        un_ind4 = unique_seq(inds4, tol=tol)
        non_dupes = list(set(un_ind1) & set(un_ind2) &
                         set(un_ind3) & set(un_ind4))
    return non_dupes


def add_ptcris(track, between_ptcris, sandro=False):
    '''return track.[s or i ]ptcri indices between between_ptcris'''
    if sandro:
        iptcri = track.sptcri
    else:
        iptcri = track.iptcri
    pinds = iptcri[between_ptcris[0]: between_ptcris[1] + 1]
    return pinds


def column_to_data(track, xcol, ycol, xdata=None, ydata=None, norm=None):
    '''
    convert a string column name to data

    returns xdata, ydata

    norm: 'xy', 'x', 'y' for which or both axis to normalize
    can also pass xdata, ydata to normalize or if its a cmd (Mag2mag only)
    '''
    norm = norm or ''
    if ydata is None:
        ydata = track.data[ycol]

    if xdata is None:
        xdata = track.data[xcol]

    if 'x' in norm:
        xdata /= np.max(xdata)

    if 'y' in norm:
        ydata /= np.max(ydata)

    return xdata, ydata


def maxmin(arr, inds=None):
    '''
    return the max and min of a column in self.data, inds to slice.
    '''
    if inds is not None:
        arr = arr[inds]
    return (np.max(arr), np.min(arr))


def second_derivative(xdata, inds, gt=False, s=0):
    '''
    The second derivative of d^2 xdata / d inds^2

    why inds for interpolation, not log l?
    if not using something like model number instead of log l,
    the tmin will get hidden by data with t < tmin but different
    log l. This is only a problem for very low Z.
    If I find the arg min of teff to be very close to MS_BEG it
    probably means the MS_BEG is at a lower Teff than Tmin.
    '''
    tckp, _ = splprep([inds, xdata], s=s, k=3)
    arb_arr = np.arange(0, 1, 1e-2)
    xnew, ynew = splev(arb_arr, tckp)
    # second derivative, bitches.
    ddxnew, ddynew = splev(arb_arr, tckp, der=2)
    ddyddx = ddynew / ddxnew
    # not just argmin, but must be actual min...
    try:
        if gt:
            aind = [a for a in np.argsort(ddyddx) if ddyddx[a-1] < 0][0]
        else:
            aind = [a for a in np.argsort(ddyddx) if ddyddx[a-1] > 0][0]
    except IndexError:
        return -1
    tmin_ind, _ = closest_match2d(aind, inds, xdata, xnew, ynew)
    return inds[tmin_ind]


def add_version_info(input_file):
    """Copy the input file and add the git hash and time the run started."""
    import os
    from time import localtime, strftime
    from .fileio import replace_ext

    # create info file with time of run
    now = strftime("%Y-%m-%d %H:%M:%S", localtime())
    fname = replace_ext(input_file, '.info')
    with open(fname, 'w') as out:
        out.write('parsec2match run started %s \n' % now)
        out.write('padova_tracks git hash: ')

    # the best way to get the git hash?
    here = os.getcwd()
    home, _ = os.path.split(__file__)

    os.chdir(home)
    os.system('git rev-parse --short HEAD >> "%s"' % os.path.join(here, fname))
    os.chdir(here)

    # add the input file
    os.system('cat %s >> %s' % (input_file, fname))
    return fname


def filename_data(fname, ext='.dat', skip=2, delimiter='_', exclude='imf'):
    """
    return a dictionary of key and values from a filename.
    E.g, ssp_imf4.85_bf0.3_dav0.0.fdat
    returns bf: 0.3, dav: 0.0
    NB: imf is excluded because it's already included in the file.

    Parameters
    ----------
    fname : str
        filename

    ext : str
        extension (sub string to remove from the tail)

    delimiter : str
        how the keyvals are separated '_' in example above

    skip : int
        skip n items (skip=1 skips ssp in the above example)

    exclude : str
        do not include this key/value in the file (default: 'imf')

    Returns
    -------
    dict of key and values from filename
    """
    import re
    keyvals = fname.replace(ext, '').split(delimiter)[skip:]
    d = {}
    for keyval in keyvals:
        kv = re.findall(r'\d+|[a-z]+', keyval, re.IGNORECASE)
        neg = ''
        if '-' in keyval:
            neg = '-'
        if kv[0].lower() == exclude.lower():
            continue
        try:
            d[kv[0]] = float(neg + '.'.join(kv[1:]))
        except ValueError:
            # print e
            # print(sys.exc_info()[1])
            pass
    return d


def get_zy(string):
    Z, Ymore = string.replace('_', '').split('Z')[1].split('Y')
    Ymore = '.'.join(Ymore.split('.')[:-1])
    Y = ''
    for y in Ymore:
        if y == '.' or y.isdigit():
            Y += y
        else:
            break
    return float(Z), float(Y)


def replace_(s, rdict):
    for k, v in rdict.items():
        s = s.replace(k, v)
    return s


def compfunc(func, arr1, arr2):
    return func([func(arr1), func(arr2)])


def extrema(arr1, arr2):
    return compfunc(np.min, arr1, arr2), compfunc(np.max, arr1, arr2)


def sort_dict(dic):
    ''' zip(*sorted(dictionary.items(), key=lambda(k,v):(v,k))) '''
    from collections import OrderedDict
    return OrderedDict(sorted(dic.items()))


def extrap1d(x, y, xout_arr):
    '''
    linear extapolation from interp1d class with a way around bounds_error.
    Adapted from:
    http://stackoverflow.com/questions/2745329/how-to-make-scipy-interpolate-give-an-extrapolated-result-beyond-the-input-range

    Parameters
    ----------
    x, y : arrays
        values to interpolate

    xout_arr : array
        x array to extrapolate to

    Returns
    -------
    f, yo : interpolator class and extrapolated y array
    '''
    from scipy.interpolate import interp1d
    # Interpolator class
    f = interp1d(x, y)
    xo = xout_arr
    # Boolean indexing approach
    # Generate an empty output array for "y" values
    yo = np.empty_like(xo)

    # Values lower than the minimum "x" are extrapolated at the same time
    low = xo < f.x[0]
    yo[low] = f.y[0] + (xo[low] - f.x[0]) * (f.y[1] - f.y[0]) \
        / (f.x[1] - f.x[0])

    # Values higher than the maximum "x" are extrapolated at same time
    high = xo > f.x[-1]
    yo[high] = f.y[-1] + (xo[high] - f.x[-1]) * (f.y[-1] - f.y[-2]) \
        / (f.x[-1] - f.x[-2])

    # Values inside the interpolation range are interpolated directly
    inside = np.logical_and(xo >= f.x[0], xo <= f.x[-1])
    yo[inside] = f(xo[inside])
    return f, yo


def find_peaks(arr):
    '''
    find maxs and mins of an array
    from
    http://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
    Parameters
    ----------
    arr : array
        input array to find maxs and mins

    Returns
    -------
    turning_points : dict
        keys:
        maxima_number: int, how many maxima in arr
        minima_number: int, how many minima in arr
        maxima_locations: list, indicies of maxima
        minima_locations: list, indicies of minima
    '''
    def cmp(a, b):
        return (a > b) - (a < b)

    gradients = np.diff(arr)
    # print gradients

    maxima_num = 0
    minima_num = 0
    max_locations = []
    min_locations = []
    count = 0
    for i in gradients[:-1]:
        count += 1

        if ((cmp(i, 0) > 0) & (cmp(gradients[count], 0) < 0) &
           (i != gradients[count])):
            maxima_num += 1
            max_locations.append(count)

        if ((cmp(i, 0) < 0) & (cmp(gradients[count], 0) > 0) &
           (i != gradients[count])):
            minima_num += 1
            min_locations.append(count)

    turning_points = {'maxima_number': maxima_num,
                      'minima_number': minima_num,
                      'maxima_locations': max_locations,
                      'minima_locations': min_locations}

    return turning_points


def min_dist2d(xpoint, ypoint, xarr, yarr):
    '''
    index and distance of point in [xarr, yarr] nearest to [xpoint, ypoint]

    Parameters
    ----------
    xpoint, ypoint : floats

    xarr, yarr : arrays

    Returns
    -------
    ind, dist : int, float
        index of xarr, arr and distance
    '''
    dist = np.sqrt((xarr - xpoint) ** 2 + (yarr - ypoint) ** 2)
    return np.argmin(dist), np.min(dist)


def closest_match2d(ind, x1, y1, x2, y2, normed=False):
    '''
    find closest point between of arrays x2[ind], y2[ind] and x1, y1.
    by minimizing the radius of a circle.
    '''
    x1n = 1.
    x2n = 1.
    y1n = 1.
    y2n = 1.
    if normed is True:
        x1n = x1 / np.max(x1)
        x2n = x2 / np.max(x2)
        y1n = y1 / np.max(y1)
        y2n = y2 / np.max(y2)

    dist = np.sqrt((x1 / x1n - x2[ind] / x2n) ** 2 +
                   (y1 / y1n - y2[ind] / y2n) ** 2)
    return np.argmin(dist), np.min(dist)


def closest_match(num, arr):
    '''index and difference of closet point of arr to num'''
    index = -1
    arr = np.nan_to_num(arr)
    difference = np.abs(num - arr[0])
    for i in range(len(arr)):
        if difference > np.abs(num - arr[i]):
            difference = np.abs(num - arr[i])
            index = i
    return index, difference


def is_numeric(lit):
    """
    value of numeric: literal, string, int, float, hex, binary
    From http://rosettacode.org/wiki/Determine_if_a_string_is_numeric#Python
    """
    # Empty String
    if len(lit) <= 0:
        return lit
    # Handle '0'
    if lit == '0':
        return 0
    # Hex/Binary
    if len(lit) > 1:  # sometimes just '-' means no data...
        litneg = lit[1:] if lit[0] == '-' else lit
        if litneg[0] == '0':
            if litneg[1] in 'xX':
                return int(lit, 16)
            elif litneg[1] in 'bB':
                return int(lit, 2)
            else:
                try:
                    return int(lit, 8)
                except ValueError:
                    pass
    # Int/Float/Complex
    try:
        return int(lit)
    except ValueError:
        pass
    try:
        return float(lit)
    except ValueError:
        pass
    try:
        return complex(lit)
    except ValueError:
        pass
    return lit
