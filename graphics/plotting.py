from __future__ import print_function
import numpy as np
import pylab as plt
from scipy import sparse
from matplotlib.mlab import griddata
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrow
from matplotlib.colors import LinearSegmentedColormap

from ..tools import parse_mag_tab

__all__ = ['colorify', 'colorplot_by_stage', 'crazy_histogram2d',
           'discrete_colors', 'latex_float', 'make_hess',
           'plot_cmd_redding_vector', 'plot_hess', 'plot_hlines', 'plot_numbs',
           'scatter_contour', 'setup_multiplot', 'setup_plot_by_stage',
           'stitch_cmap']


def colorify(data, vmin=None, vmax=None, cmap=plt.cm.Spectral):
    """ Associate a color map to a quantity vector

    Parameters
    ----------
    data: sequence
        values to index

    vmin: float, optional
        minimal value to index

    vmax: float, optional
        maximal value to index

    cmap: colormap instance
        colormap to use

    Returns
    -------
    colors: sequence
        color sequence corresponding to data

    scalarMap: colormap
        generated map
    """
    import matplotlib.colors as colors

    _vmin = vmin or min(data)
    _vmax = vmax or max(data)
    cNorm = colors.normalize(vmin=_vmin, vmax=_vmax)

    scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=cmap)
    colors = map(scalarMap.to_rgba, data)
    return colors, scalarMap


def scatter_contour(x, y, levels=10, bins=40, threshold=50, log_counts=False,
                    histogram2d_args={}, plot_args={}, contour_args={},
                    ax=None, xerr=None, yerr=None):
    """Scatter plot with contour over dense regions (adapted from astroML)

    Parameters
    ----------
    x, y : arrays
        x and y data for the contour plot

    levels : integer or array (optional, default=10)
        number of contour levels, or array of contour levels

    threshold : float (default=100)
        number of points per 2D bin at which to begin drawing contours

    log_counts :boolean (optional)
        if True, contour levels are the base-10 logarithm of bin counts.

    histogram2d_args : dict
        keyword arguments passed to crazy_histogram2d
        see doc string of crazy_histogram2d for more information

    plot_args : dict
        keyword arguments passed to pylab.plot
        see doc string of pylab.plot for more information

    contourf_args : dict
        keyword arguments passed to pylab.contourf
        see doc string of pylab.contourf for more information

    xerr, yerr : arrays
        x and y errors passed to pylab.errorbar

    ax : pylab.Axes instance
        the axes on which to plot.  If not specified, the current
        axes will be used
    """
    if ax is None:
        fig, ax = plt.subplots()

    # H, xbins, ybins = np.histogram2d(x, y, **histogram2d_args)
    # extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]
    H, extent, (xbin, ybins)  = crazy_histogram2d(x, y, bins=bins,
                                                  **histogram2d_args)

    if log_counts:
        H = np.log10(1 + H)
        threshold = np.log10(1 + threshold)

    levels = np.asarray(levels)

    if levels.size == 1:
        levels = np.linspace(threshold, H.max(), levels)

    i_min = np.argmin(levels)

    # draw a zero-width line: this gives us the outer polygon to
    # reduce the number of points we draw
    # somewhat hackish... we could probably get the same info from
    # the filled contour below.
    outline = ax.contour(H.T, levels[i_min:i_min + 1],
                         linewidths=0, extent=extent)
    try:
        outer_poly = outline.allsegs[0][0]

        ax.contourf(H.T, levels, extent=extent, **contour_args)
        X = np.hstack([x[:, None], y[:, None]])

        try:
            # this works in newer matplotlib versions
            from matplotlib.path import Path
            points_inside = Path(outer_poly).contains_points(X)
        except:
            # this works in older matplotlib versions
            import matplotlib.nxutils as nx
            points_inside = nx.points_inside_poly(X, outer_poly)

        Xplot = X[~points_inside]

        ax.scatter(Xplot[:, 0], Xplot[:, 1], **plot_args)
        if xerr is not None:
            ax.errorbar(Xplot[:, 0], Xplot[:, 1], fmt=None,
                        xerr=xerr[~points_inside], yerr=yerr[~points_inside],
                        capsize=0, ecolor='gray')
    except IndexError:
        ax.scatter(x, y, **plot_args)
        if xerr is not None:
            ax.errorbar(x, y, fmt=None, xerr=xerr, yerr=yerr, capsize=0,
                        ecolor='gray')
    return ax


def latex_float(f, precision=0.2, delimiter=r'\times'):
    """ Convert a float value into a pretty printable latex format
    makes 1.3123e-11 transformed into $1.31 x 10 ^ {-11}$

    Parameters
    ----------
    f: float
        value to convert

    precision: float, optional (default: 0.2)
        the precision will be used as the formatting: {precision}g
        default=0.2g

    delimiter: str, optional (default=r'\times')
        delimiter between the value and the exponent part
    """
    float_str = ("{0:" + str(precision) + "g}").format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return (r"{0}" + delimiter + "10^{{{1}}}").format(base, int(exponent))
    else:
        return float_str


def crazy_histogram2d(x, y, bins=10, weights=None, reduce_w=None, NULL=None,
                      reinterp=None, reverse_indices=False, xrange=None,
                      yrange=None):
    """
    Compute the sparse bi-dimensional histogram of two data samples where *x*,
    and *y* are 1-D sequences of the same length. If *weights* is None
    (default), this is a histogram of the number of occurences of the
    observations at (x[i], y[i]).

    If *weights* is specified, it specifies values at the coordinate (x[i],
    y[i]). These values are accumulated for each bin and then reduced according
    to *reduce_w* function, which defaults to numpy's sum function (np.sum).
    (If *weights* is specified, it must also be a 1-D sequence of the same
    length as *x* and *y*.)

    Parameters
    ----------
    x: ndarray[ndim=1]
        first data sample coordinates

    y: ndarray[ndim=1]
        second data sample coordinates

    bins: int or [int, int], optional
        the bin specification
        `int`       : the number of bins for the two dimensions (nx=ny=bins)
        `[int, int]`: the number of bins in each dimension (nx, ny = bins)

    weights: ndarray[ndim=1], optional
        values *w_i* weighing each sample *(x_i, y_i)*, they will be
        accumulated and reduced (using reduced_w) per bin

    reduce_w: callable, optional (default=np.sum)
        function that will reduce the *weights* values accumulated per bin
        defaults to numpy's sum function (np.sum)

    NULL: value type, optional
        filling missing data value

    reinterp: str in [None, 'nn', linear'], optional
        if set, reinterpolation is made using mlab.griddata to fill missing
        data within the convex polygone that encloses the data

    reverse_indices: bool, option
        also return the bins of each x, y point as a 2d array

    Returns
    -------
    B: ndarray[ndim=2]
        bi-dimensional histogram

    extent: tuple(4)
        (xmin, xmax, ymin, ymax) entension of the histogram

    steps: tuple(2)
        (dx, dy) bin size in x and y direction

    """
    # define the bins (do anything you want here but needs edges and sizes of the 2d bins)
    try:
        nx, ny = bins
    except TypeError:
        nx = ny = bins

    #values you want to be reported
    if weights is None:
        weights = np.ones(x.size)

    if reduce_w is None:
        reduce_w = np.sum
    else:
        if not hasattr(reduce_w, '__call__'):
            raise TypeError('reduce function is not callable')

    # culling nans
    finite_inds = (np.isfinite(x) & np.isfinite(y) & np.isfinite(weights))
    _x = np.asarray(x)[finite_inds]
    _y = np.asarray(y)[finite_inds]
    _w = np.asarray(weights)[finite_inds]

    if not (len(_x) == len(_y)) & (len(_y) == len(_w)):
        raise ValueError('Shape mismatch between x, y, and weights: {}, {}, {}'.format(_x.shape, _y.shape, _w.shape))
    if xrange is None:
        xmin, xmax = _x.min(), _x.max()
    else:
        xmin, xmax = xrange
    if yrange is None:
        ymin, ymax = _y.min(), _y.max()
    else:
        ymin, ymax = yrange
    inds, = np.nonzero((_x > xmin) & (_x < xmax) & (_y > ymin) & (_y < ymax))
    _x = _x[inds]
    _y = _y[inds]
    _w = _w[inds]

    dx = (xmax - xmin) / (nx - 1.0)
    dy = (ymax - ymin) / (ny - 1.0)

    # Basically, this is just doing what np.digitize does with one less copy
    xyi = np.vstack((_x, _y)).T
    xyi -= [xmin, ymin]
    xyi /= [dx, dy]
    xyi = np.floor(xyi, xyi).T

    #xyi contains the bins of each point as a 2d array [(xi,yi)]

    d = {}
    for e, k in enumerate(xyi.T):
        key = (k[0], k[1])

        if key in d:
            d[key].append(_w[e])
        else:
            d[key] = [_w[e]]

    _xyi = np.array(d.keys()).T
    _w   = np.array([ reduce_w(v) for v in d.values() ])

    # exploit a sparse coo_matrix to build the 2D histogram...
    _grid = sparse.coo_matrix((_w, _xyi), shape=(nx, ny))

    if reinterp is None:
        #convert sparse to array with filled value
        ## grid.toarray() does not account for filled value
        ## sparse.coo.coo_todense() does actually add the values to the existing ones, i.e. not what we want -> brute force
        if NULL is None:
            B = _grid.toarray()
        else:  # Brute force only went needed
            B = np.zeros(_grid.shape, dtype=_grid.dtype)
            B.fill(NULL)
            for (x, y, v) in zip(_grid.col, _grid.row, _grid.data):
                B[y, x] = v
    else:  # reinterp
        xi = np.arange(nx, dtype=float)
        yi = np.arange(ny, dtype=float)
        B = griddata(_grid.col.astype(float), _grid.row.astype(float),
                     _grid.data, xi, yi, interp=reinterp)

    if reverse_indices:
        return B, (xmin, xmax, ymin, ymax), (dx, dy), xyi.T, inds

    return B, (xmin, xmax, ymin, ymax), (dx, dy)


def make_hess(color, mag, binsize, **kw):
    """
    Compute a hess diagram (surface-density CMD) on photometry data.

    Paramters
    ---------
    color: ndarray
        color values

    mag: ndarray
        magnitude values

    binsize: sequence
        width of bins, in magnitudes

    cbin: sequence, optional
        set the centers of the color bins

    mbin: sequence, optional
        set the centers of the magnitude bins

    cbinsize: sequence, optional
        width of bins, in magnitudes

    Returns
    -------
    Cbin: sequence
        the centers of the color bins

    Mbin: sequence
        the centers of the magnitude bins

    Hess:
        The Hess diagram array

    2009-02-08 23:01 IJC: Created, on a whim, for LMC data (of course)
    2009-02-21 15:45 IJC: Updated with cbin, mbin options
    2012 PAR: Gutted and changed it do histogram2d for faster implementation.
    2014 MF: refactored and replace np.histogram with crazy_histogram2d
    """

    defaults = {'mbin': None, 'cbin': None, 'verbose': False}
    for key in defaults:
        if key not in kw:
            kw[key] = defaults[key]

    if kw['mbin'] is None:
        mbin = np.arange(mag.min(), mag.max(), binsize)
    else:
        mbin = np.array(kw['mbin']).copy()

    if kw['cbin'] is None:
        cbinsize = kw.get('cbinsize')
        if cbinsize is None:
            cbinsize = binsize
        cbin = np.arange(color.min(), color.max(), cbinsize)
    else:
        cbin = np.array(kw['cbin']).copy()

    hess, extent, (cbin, mbin) = crazy_histogram2d(color, mag, bins=[cbin, mbin])
    return (cbin, mbin, hess.T, extent)


def plot_hess(hess, fig=None, ax=None, colorbar=False, filter1=None,
              filter2=None, imshow=True, vmin=None, vmax=None, **kwargs):
    '''
    Plots a hess diagram with imshow.

    default kwargs passed to imshow:

    default_kw = {'norm': LogNorm(vmin=None, vmax=hess[2].max())
                  'cmap': cm.gray,
                  'interpolation': 'nearest',
                  'extent': [hess[0][0], hess[0][-1],
                             hess[1][-1], hess[1][0]]}
    '''
    ax = ax or plt.gca()

    cbin, mbin, h, extent = hess
    vmax = vmax or h.max()

    defaults = {'norm': LogNorm(vmin=vmin, vmax=vmax),
                'cmap': plt.cm.gray,
                'interpolation': 'nearest',
                'extent': extent,
                'aspect': 'auto'}

    for key in defaults:
        if key not in kwargs:
            kwargs[key] = defaults[key]

    if imshow is True:
        ax.autoscale(False)
        im = ax.imshow(h, **kwargs)

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        if colorbar is True:
            plt.colorbar(im)
    else:
        im = ax.contourf(h, **kwargs)

    if filter2 is not None and filter1 is not None:
        ax.set_ylabel('$%s$' % (filter2))
        ax.set_xlabel('$%s - %s$' % (filter1, filter2))

    return ax


def plot_cmd_redding_vector(filter1, filter2, photsys, dmag=1., ax=None):
    """ Add an arrow to show the reddening vector """

    ax = ax or plt.gca()

    Afilt1 = parse_mag_tab(photsys, filter1)
    Afilt2 = parse_mag_tab(photsys, filter2)

    Rslope = Afilt2 / (Afilt1 - Afilt2)
    dcol = dmag / Rslope
    pstart = np.array([0., 0.])
    pend = pstart + np.array([dcol, dmag])
    points = np.array([pstart, pend])
    data_to_display = ax.transData.transform
    display_to_axes = ax.transAxes.inverted().transform
    ax_coords = display_to_axes(data_to_display(points))
    dy_ax_coords = ax_coords[1, 1] - ax_coords[0, 1]
    dx_ax_coords = ax_coords[1, 0] - ax_coords[0, 0]

    arr = FancyArrow(0.05, 0.95, dx_ax_coords, dy_ax_coords,
                     transform=ax.transAxes, color='black', ec="none",
                     width=.005, length_includes_head=1, head_width=0.02)
    ax.add_patch(arr)


def setup_multiplot(nplots, xlabel=None, ylabel=None, title=None, **kwargs):
    '''
    Quick ways to prepare a figure with multiple subplots

    Parameters
    ----------
    nplots: int
        number of panels

    xlabel: str, optional
        global xlabel string

    ylabel: str, optional
        global ylabel string

    title: str, optional
        figure title

    **kargs: dict
        extra arguement are forwarded to `:func: plt.subplots`

    Returns
    -------
    fig: plt.Figure instance
        figure instance

    axes: list(plt.Axes)
        list of newly created axes
    '''
    nx = np.round(np.sqrt(nplots))
    nextra = nplots - nx ** 2
    ny = nx
    if nextra > 0:
        ny += 1
    nx = int(nx)
    ny = int(ny)

    (fig, axs) = plt.subplots(nrows=nx, ncols=ny, **kwargs)

    if ylabel is not None:
        axs[0][0].annotate(ylabel, fontsize=45, xy=(0.04, 0.5),
                           xycoords='figure fraction', va='center',
                           rotation='vertical')
    if xlabel is not None:
        axs[0][0].annotate(xlabel, fontsize=45, xy=(0.5, 0.04),
                           xycoords='figure fraction', va='center')
    if title is not None:
        axs[0][0].annotate(title, fontsize=45, xy=(0.5, 1. - 0.04),
                           xycoords='figure fraction', va='center')

    return (fig, axs)


def plot_numbs(item, xpos, ypos, ax=None, ha='left', size=20, **kwargs):
    """ Plot numbers on a figure

    Parameters
    ----------
    item: value
        item to put on the plot

    xpos: float
        x coordinate

    ypos: float
        y coordinate

    ax: plt.Axes, optional
        axes to plot in

    **kwargs: dict
        extra arguments are forwarded to `:func: plt.annotate`
    """
    ax = ax or plt.gca()
    return ax.annotate(r'$%i$' % item, xy=(xpos, ypos), ha=ha, size=size, **kwargs)


def plot_hlines(y, xmin, xmax, ax=None, **kwargs):
    """
    Plot horizontal lines on multiple axes.

    Plot horizontal lines at each `y` from `xmin` to `xmax`.

    Parameters
    ----------
    y : scalar or 1D array_like
        y-indexes where to plot the lines.

    xmin, xmax : scalar or 1D array_like
        Respective beginning and end of each line. If scalars are
        provided, all lines will have same length.

    **kwargs: optional
        forwarded to `:func: plt.hlines`
    """
    if ax is None:
        ax = plt.gca()

    if hasattr(ax, '__iter__'):
        return [plot_hlines(y, xmin, xmax, ax=axk, **kwargs) for axk in ax]
    else:
        if not hasattr(y, '__iter__'):
            return ax.hlines([y], xmin, xmax, **kwargs)
        else:
            return ax.hlines(y, xmin, xmax, **kwargs)


def setup_plot_by_stage(stage, inds=None):
    """ prepare a plot by stage

    Parameters
    ----------
    stage:

    inds: indices, slice, optional
        if set, reduces stage to the selction

    Returns
    -------
    fig: plt.Figure instance
        generated figure

    axs: list(plt.Axes)
        list newly created axes
    """
    if inds is not None:
        stage = stage[inds]
    ustage = np.unique(stage)
    nplots = ustage.size + 1.
    fig, (axs) = setup_multiplot(nplots, sharex=True, sharey=True, figsize=(12, 8))
    return fig, (axs)


def colorplot_by_stage(x, y, marker, stages, cols=None, inds=None, cmap=None, ax=None, **kwargs):
    """
    Parameters
    ----------
    x: sequence
        x values to plot

    y: sequence
        y values to plot

    marker: str
        marker argument

    stages: sequence
        list of stages to plot

    cols: sequence, optional
        color values, default is made from cmap

    inds: slice, optional
        reduce to a smaller subset for the plot

    cmap: colormap, optional
        color map for the plots if no color provided

    ax: plt.Axes, optional
        axes to plot into

    **kwargs: optional
        extra arguments are forwarded to `:func: plt.plot`

    Returns
    -------
    ax: plt.Axes
        axes instance

    cols: list
        list of stage's colors
    """
    ax = ax or plt.gca()
    # inds from calc_LFIR are based on only resolved stars.
    from TrilegalUtils import get_label_stage

    if cols is None:
        cols = discrete_colors(len(np.unique(stages)), cmap=cmap)

    for i, s in enumerate(np.unique(stages)):
        ind, = np.nonzero(stages == s)
        if ind.size == 0:
            continue
        if inds is not None:
            ind = np.intersect1d(ind, inds)
        ax.plot(x[ind], y[ind], marker, color=cols[i], mew=0, label=get_label_stage(int(s)), **kwargs)
    return ax, cols


def discrete_colors(Ncolors, cmap='gist_rainbow'):
    '''
    returns list of RGBA tuples length Ncolors

    Parameters
    ----------
    Ncolors: int
        number of discrete colors

    cmap: colormap, optional (default: plt.cm.gist_rainbow)
        colormap to sample

    Returns
    -------
    values: list
        sequence of color values
    '''
    cmap = plt.cm.get_cmap(cmap)
    return [cmap(1. * i / Ncolors) for i in range(Ncolors)]


def stitch_cmap(cmap1, cmap2, stitch_frac=0.5, dfrac=0.001):
    ''' Stitch two color maps together

    Parameters
    ----------
    cmap1: cmap
        colors from 0 and stitch_frac

    cmap2: cmap
        colors from stitch_frac to 1

    stitch_frac: float
        where the junction is located

    dfrac: float
        spacing inbetween

    Returns
    -------
    cmap: cmap
        new colormap

    .. Example:
        stitch black to white to white to red:

        >>> stitched = stitch_cmap(plt.cm.Greys_r, plt.cm.Reds, \
                                   stitch_frac=0.525, dfrac=0.05)
    '''
    seg1 = cmap1._segmentdata
    seg2 = cmap2._segmentdata

    frac1 = stitch_frac - dfrac
    frac2 = stitch_frac + dfrac

    blue_array_1 = [(i0 * frac1, i1, i2) for i0, i1, i2 in seg1['blue']]
    blue_array_2 = [(i0 * (1 - frac2) + frac2, i1, i2) for i0, i1, i2 in seg2['blue']]

    red_array_1 = [(i0 * frac1, i1, i2) for i0, i1, i2 in seg1['red']]
    red_array_2 = [(i0 * (1 - frac2) + frac2, i1, i2 ) for i0, i1, i2 in seg2['red']]

    green_array_1 = [(i0 * frac1, i1, i2) for i0, i1, i2 in seg1['green']]
    green_array_2 = [(i0 * (1 - frac2) + frac2, i1, i2) for i0, i1, i2 in seg2['green']]

    new_segments = {'blue': blue_array_1 + blue_array_2,
                    'red': red_array_1 + red_array_2,
                    'green': green_array_1 + green_array_2}

    new_cmap_name = '_'.join((cmap1.name, cmap2.name))
    new_cmap = LinearSegmentedColormap(new_cmap_name, new_segments, 1024)
    return new_cmap
