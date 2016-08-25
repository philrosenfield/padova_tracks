import numpy as np
import matplotlib.pyplot as plt
from ..utils import maxmin


def points_inside_poly(points, all_verts):
    """nxutils depreciated call modified to the approved call in mpl"""
    from matplotlib.path import Path
    return Path(all_verts, close=True).contains_points(points)


def stitch_cmap(cmap1, cmap2, stitch_frac=0.5, dfrac=0.001):
    '''
    Code adapted from Dr. Adrienne Stilp
    Stitch two color maps together:
        cmap1 from 0 and stitch_frac
        and
        cmap2 from stitch_frac to 1
        with dfrac spacing inbetween

    ex: stitch black to white to white to red:
    stitched = stitch_cmap(cm.Greys_r, cm.Reds, stitch_frac=0.525, dfrac=0.05)
    '''
    from matplotlib.colors import LinearSegmentedColormap
    seg1 = cmap1._segmentdata
    seg2 = cmap2._segmentdata

    frac1 = stitch_frac - dfrac
    frac2 = stitch_frac + dfrac

    blue_array_1 = [(i0 * frac1, i1, i2) for i0, i1, i2 in seg1['blue']]
    blue_array_2 = [(i0 * (1 - frac2) + frac2, i1, i2)
                    for i0, i1, i2 in seg2['blue']]

    red_array_1 = [(i0 * frac1, i1, i2) for i0, i1, i2 in seg1['red']]
    red_array_2 = [(i0 * (1 - frac2) + frac2, i1, i2)
                   for i0, i1, i2 in seg2['red']]

    green_array_1 = [(i0 * frac1, i1, i2) for i0, i1, i2 in seg1['green']]
    green_array_2 = [(i0 * (1 - frac2) + frac2, i1, i2)
                     for i0, i1, i2 in seg2['green']]

    new_segments = {'blue': blue_array_1 + blue_array_2,
                    'red': red_array_1 + red_array_2,
                    'green': green_array_1 + green_array_2}

    new_cmap_name = '_'.join((cmap1.name, cmap2.name))
    new_cmap = LinearSegmentedColormap(new_cmap_name, new_segments, 1024)
    return new_cmap


def arrow_on_line(ax, xarr, yarr, index, plt_kw={}):
    """put (a) FancyArrow(s) at xarr[index], yarr[index]"""
    from matplotlib.patches import FancyArrow
    arrs = []
    default = {'head_width': 0.6, 'linewidth': 1, 'length_includes_head': True,
               'ec': "none"}
    default.update(plt_kw)
    index = np.atleast_1d(index)
    x = xarr[index]
    dx = xarr[index + 1] - x
    y = yarr[index]
    dy = yarr[index + 1] - y
    for i in range(len(x)):
        arr = FancyArrow(x[i], y[i], dx[i], dy[i], **default)
        ax.add_patch(arr)
        arrs.append(arr)
    return arrs


def offset_axlims(xdata, ydata,  ax, inds=None, offx=0.002, offy=0.015):
    """set ax limits (offx, offy) fraction of the (xdata max, ydata max)"""
    if inds is not None:
        xdata = xdata[inds]
        ydata = ydata[inds]

    xmax, xmin = maxmin(xdata)
    ymax, ymin = maxmin(ydata)

    offx = 0.002 * xmax
    offy = 0.015 * ymax

    if np.diff((xmin, xmax)) == 0:
        xmin -= offx
        xmax += offx

    if np.diff((ymin, ymax)) == 0:
        ymin -= offy
        ymax += offy

    ax.set_xlim(xmax + offx, xmin - offx)
    ax.set_ylim(ymin - offy, ymax + offy)
    return ax


def forceAspect(ax, aspect=1):
    """force the aspect ratio of a given axis"""
    im = ax.get_images()
    extent = im[0].get_extent()
    asp = abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect
    ax.set_aspect(asp)


def get_plot_dims(nplots):
    """calculate the number of rows and columns needed for a plot"""
    nrows = np.round(np.sqrt(nplots))
    nextra = nplots - nrows ** 2
    ncols = nrows
    if nextra > 0:
        ncols += 1
    return int(nrows), int(ncols)


def setup_multiplot(nplots, xlabel=None, ylabel=None, title=None,
                    subplots_kws={}):
    """
    Create a plt.subplots instance with nrows and ncols caclulated by total
    number of needed plots
    """
    nrows, ncols = get_plot_dims(nplots)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, **subplots_kws)
    ann_kw = {'fontsize': 45, 'xycoords': 'figure fraction', 'va': 'center'}
    offset = 0.04

    if ylabel is not None:
        axs[0][0].annotate(ylabel,  xy=(offset, 0.5), rotation='vertical',
                           **ann_kw)
    if xlabel is not None:
        axs[0][0].annotate(xlabel, xy=(0.5, offset), **ann_kw)
    if title is not None:
        axs[0][0].annotate(title, xy=(0.5, 1. - offset), **ann_kw)

    return fig, axs


def discrete_colors(ncolors, colormap='gist_rainbow'):
    """equally spaced list of RGBA tuples length Ncolors"""
    cmap = plt.cm.get_cmap(colormap)
    return [cmap(1. * i / ncolors) for i in range(ncolors)]


def load_ann_kwargs():
    """emboss text"""
    from matplotlib.patheffects import withStroke
    myeffect = withStroke(foreground="w", linewidth=3)
    ann_kwargs = dict(path_effects=[myeffect])
    return ann_kwargs
