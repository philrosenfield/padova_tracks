import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, rc, rcParams
from matplotlib.patches import FancyArrow
from matplotlib.ticker import NullFormatter, MultipleLocator
from matplotlib.path import Path

__all__ = ['arrow_on_line', 'colorplot_by_stage', 'discrete_colors',
           'forceAspect', 'histOutline', 'load_ann_kwargs',
           'load_scatter_kwargs', 'plot_lines', 'plot_numbs',
           'points_inside_poly', 'reverse_xaxis', 'reverse_yaxis',
           'scatter_colorbar', 'set_up_three_panel_plot',
           'setup_five_panel_plot', 'setup_four_panel', 'setup_multiplot',
           'stitch_cmap', 'two_panel_plot', 'two_panel_plot_vert', 'update_rc']


def update_rc():
    """ apply some updates to make better figures """
    rcParams['text.usetex'] = True
    rcParams['text.latex.unicode'] = False
    rcParams['axes.linewidth'] = 2
    rcParams['ytick.labelsize'] = 'large'
    rcParams['xtick.labelsize'] = 'large'
    rcParams['axes.edgecolor'] = 'grey'
    rc('text', usetex=True)


def points_inside_poly(points, all_verts):
    """ Proxy to the correct way with mpl """
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
    seg1 = cmap1._segmentdata
    seg2 = cmap2._segmentdata

    frac1 = stitch_frac - dfrac
    frac2 = stitch_frac + dfrac

    blue_array_1 = [(i0 * frac1, i1, i2) for i0, i1, i2 in seg1['blue']]
    blue_array_2 = [(i0 * (1 - frac2) + frac2, i1, i2) for i0, i1, i2 in seg2['blue']]

    red_array_1 = [(i0 * frac1, i1, i2) for i0, i1, i2 in seg1['red']]
    red_array_2 = [(i0 * (1 - frac2) + frac2, i1, i2 ) for i0, i1, i2 in seg2['red']]

    green_array_1 = [(i0 * frac1, i1, i2) for i0, i1, i2 in seg1['green']]
    green_array_2 = [(i0 * (1 - frac2) + frac2, i1, i2)
                     for i0, i1, i2 in seg2['green']]

    new_segments = {'blue': blue_array_1 + blue_array_2,
                    'red': red_array_1 + red_array_2,
                    'green': green_array_1 + green_array_2}

    new_cmap_name = '_'.join((cmap1.name, cmap2.name))
    new_cmap = matplotlib.colors.LinearSegmentedColormap(new_cmap_name,
                                                         new_segments,
                                                         1024)
    return new_cmap


def arrow_on_line(ax, xarr, yarr, index, plt_kw={}):
    arrs = []
    plt_kw = dict({'head_width': 0.6, 'linewidth': 1,
                   'length_includes_head': True,
                   'ec': "none"}.items() + plt_kw.items())
    index = np.atleast_1d(index)
    x = xarr[index]
    dx = xarr[index + 1] - x
    y = yarr[index]
    dy = yarr[index + 1] - y
    for i in range(len(x)):
        arr = FancyArrow(x[i], y[i], dx[i], dy[i], **plt_kw)
        ax.add_patch(arr)
        arrs.append(arr)
    return arrs


def plot_lines(axs, xrange, yval, color='black'):
    for ax in axs:
        ax.plot((xrange), (yval, yval), color=color)


def forceAspect(ax, aspect=1):
    '''
    forces the aspect ratio of a given axis
    '''
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)


def plot_numbs(ax, item, xpos, ypos, **kwargs):
    return ax.annotate(r'$%i$' % item, xy=(xpos, ypos), ha='left', size=20, **kwargs)


def setup_multiplot(nplots, xlabel=None, ylabel=None, title=None,
                    subplots_kwargs={}):
    '''
    fyi subplots args:
        nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True,
        subplot_kw=None, **fig_kw
    '''
    nx = np.round(np.sqrt(nplots))
    nextra = nplots - nx ** 2
    ny = nx
    if nextra > 0:
        ny += 1
    nx = int(nx)
    ny = int(ny)

    (fig, axs) = plt.subplots(nrows=nx, ncols=ny, **subplots_kwargs)
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


def colorplot_by_stage(ax, x, y, marker, stages, cols=None):
    # inds from calc_LFIR are based on only resolved stars.
    from ResolvedStellarPops.TrilegalUtils import get_label_stage

    if cols is None:
        cols = discrete_colors(len(np.unique(stages)))
    for i, s in enumerate(np.unique(stages)):
        ind, = np.nonzero(stages == s)
        if ind.size == 0:
            continue
        ax.plot(x[ind], y[ind], marker, color=cols[i], mew=0, label=get_label_stage(int(s)))
    return ax, cols


def discrete_colors(Ncolors, colormap='gist_rainbow'):
    '''
    returns list of RGBA tuples length Ncolors
    '''
    cmap = cm.get_cmap(colormap)
    return [cmap(1. * i / Ncolors) for i in range(Ncolors)]


def load_scatter_kwargs(color_array, cmap=cm.jet):
    kw = {'zorder': 100,
          'alpha': 1,
          'edgecolor': 'k',
          'c': color_array,
          'cmap': cmap,
          'vmin': np.min(color_array),
          'vmax': np.max(color_array)
          }

    return kw


def scatter_colorbar(x, y, color_array, markersize=20, ax=None):
    '''
    makes a scatter plot with a float array as colors.
    '''
    # c_array is an array of floats -- so this would be your fit value
    scatter_kwargs = load_scatter_kwargs(color_array)

    ax = ax or plt.gca()
    sc = ax.scatter(x, y, markersize, **scatter_kwargs)

    # lets you get the colors if you want them for something else
    # ie sc.get_facecolors()
    # you might not need to do this
    #sc.update_scalarmappable()

    # then to get the colorbar
    plt.colorbar(sc)
    return ax, sc


def reverse_yaxis(ax):
    ax.set_ylim(ax.get_ylim()[::-1])


def reverse_xaxis(ax):
    ax.set_xlim(ax.get_xlim()[::-1])


def load_ann_kwargs():
    from matplotlib.patheffects import withStroke
    myeffect = withStroke(foreground="w", linewidth=3)
    ann_kwargs = dict(path_effects=[myeffect])
    return ann_kwargs

ann_kwargs = load_ann_kwargs()


def set_up_three_panel_plot():
    plt.figure(figsize=(8, 8))

    left, width = 0.1, 0.4
    bottom, height = 0.1, 0.4
    d = 0.01
    lefter = left + width + d
    mid = bottom + height + 2 * d
    lefts = [left, lefter, left]
    bottoms = [mid, mid, bottom]
    widths = [width, width, 2 * width + d]
    heights = [height, height, height - 0.1]

    axs = [plt.axes([l, b, w, h]) for l, b, w, h in zip(lefts, bottoms, widths, heights)]
    return axs


def two_panel_plot(sizex, sizey, xlab1, xlab2, ylab, ylab2=None, ftsize=20, mag2_cut=0, mag2_max=1):
    plt.figure(2, figsize=(sizex, sizey))
    left, width = 0.1, 0.4
    bottom, height = 0.12, 0.8

    left2 = left + width + 0.065
    if ylab2 is not None:
        left2 = left + width + 0.08
    axis1 = [left, bottom, width, height]
    axis2 = [left2, bottom, width, height]

    ax1 = plt.axes(axis1)
    ax1.set_xlim( (mag2_cut, mag2_max) )  # set all axes limits here
    #ax1.set_ylim( (0.0001, 10.) )
    ax1.set_xlabel(r'%s' % xlab1, fontsize=ftsize)
    ax1.set_ylabel(r'%s' % ylab, fontsize=ftsize)

    ax2 = plt.axes(axis2)
    ax2.set_xlim( ax1.get_xlim() )
    #ax2.set_ylim( ax1.get_ylim() )
    ax2.set_xlabel(r'%s' % xlab2, fontsize=ftsize)
    if ylab2 is not None:
        ax2.set_ylabel(r'%s' % ylab2, fontsize=ftsize)
    return ax1, ax2


def two_panel_plot_vert(oney=True, ftsize=20, mag2_cut=0, mag2_max=1):
    plt.figure(2, figsize=(8, 8))
    left, width = 0.13, 0.83
    bottom, height = 0.1, 0.41
    dh = 0.03

    axis1 = [left, bottom, width, height]
    axis2 = [left, (bottom + height + dh), width, height]

    ax1 = plt.axes(axis1)
    ax1.set_xlim( (mag2_cut, mag2_max) )  # set all axes limits here
    #ax1.set_ylim( (0.0001, 10.) )
    if oney is True:
        ax1.annotate(r'$\#/ 3.6 \mu \rm{m\ Region\ Integrated\ Flux\ (Jy}^{-1}\rm{)}$', fontsize=ftsize, xy=(0.025, .5), xycoords='figure fraction', va='center', rotation='vertical')
    ax1.set_xlabel(r'$\rm{mag}$', fontsize=ftsize)
    ax2 = plt.axes(axis2)
    ax2.set_xlim( ax1.get_xlim() )
    #ax2.set_ylim( ax1.get_ylim() )
    ax2.xaxis.set_major_formatter(NullFormatter())

    return ax1, ax2


def setup_four_panel(ftsize=20):
    left, width = 0.1, 0.4
    bottom, height = 0.1, 0.4
    lefter = left + width + 0.01
    higher = bottom + height + 0.01

    # plot and fig sizes
    plt.figure(figsize=(8, 8))

    ll_axis = [left, bottom, width, height]
    lr_axis = [lefter, bottom, width, height]
    ul_axis = [left, higher, width, height]
    ur_axis = [lefter, higher, width, height]

    ax_ll = plt.axes(ll_axis)
    ax_ll.set_xlim( (-0.75, 1))  # model and data x limits here
    ax_ll.set_ylim( (24.8, 18))  # set all y limits here

    ax_lr = plt.axes(lr_axis)
    ax_lr.set_xlim( ax_ll.get_xlim() )
    ax_lr.set_ylim( ax_ll.get_ylim() )

    ax_ul = plt.axes(ul_axis)
    ax_ul.set_xlim( ax_ll.get_xlim() )
    ax_ul.set_ylim( ax_ll.get_ylim() )

    ax_ur = plt.axes(ur_axis)
    ax_ur.set_xlim( ax_ll.get_xlim() )
    ax_ur.set_ylim( ax_ll.get_ylim() )

    ax_lr.yaxis.set_major_formatter(NullFormatter())
    ax_ur.yaxis.set_major_formatter(NullFormatter())
    ax_ur.xaxis.set_major_formatter(NullFormatter())
    ax_ul.xaxis.set_major_formatter(NullFormatter())

    # titles
    #x = fig.text(0.5, 0.96, r'$\rm{%s}$' % ('Disk Field'), horizontalalignment='center', verticalalignment='top', size=20)
    ax_ur.set_title(r'$\rm{Disk\ Field}$', color='black', fontsize=ftsize)
    ax_ul.set_title(r'$\rm{Bulge\ Field}$', color='black', fontsize=ftsize)
    ax_ll.set_ylabel(r'$F336W$', fontsize=ftsize)
    ax_ll.set_xlabel(r'$F275W-F336W$', fontsize=ftsize)
    ax_ul.set_ylabel(ax_ll.get_ylabel(), fontsize=ftsize)
    ax_lr.set_xlabel(ax_ll.get_xlabel(), fontsize=ftsize)

    return ax_ll, ax_lr, ax_ul, ax_ur


def setup_five_panel_plot(ftsize=20):
    left, width = 0.06, 0.175
    bottom, height = 0.14, 0.82
    dl = 0.01
    lefts = [left + (width + dl) * float(i) for i in range(5)]

    # plot and fig sizes
    plt.figure(figsize=(15, 4))

    axiss = [[l, bottom, width, height] for l in lefts]

    axs = []
    for i in range(len(axiss)):
        axs.append( plt.axes(axiss[i]))

    for ax in axs:
        ax.set_xlim( (-0.75, 2))   # model and data x limits here
        ax.set_ylim( (24.8, 19))   # set all y limits here
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        if axs.index(ax) > 0:
            ax.yaxis.set_major_formatter(NullFormatter())
        ax.set_xlabel(r'$F275W-F336W$', fontsize=ftsize)
        if axs.index(ax) == 0:
            ax.set_ylabel(r'$F336W$', fontsize=ftsize)

    return axs


def histOutline(dataIn, *args, **kwargs):
    '''
    Then you do something like..

    Any of the np keyword arguments can be passed:

    bins, hist = histOutline(x)

    plt.plot(bins, hist)

    or plt.fill_between(bins, hist)
    '''
    (histIn, binsIn) = np.histogram(dataIn, *args, **kwargs)

    stepSize = binsIn[1] - binsIn[0]

    bins = np.zeros(len(binsIn) * 2 + 2, dtype=np.float)
    data = np.zeros(len(binsIn) * 2 + 2, dtype=np.float)
    for bb in range(len(binsIn)):
        bins[2 * bb + 1] = binsIn[bb]
        bins[2 * bb + 2] = binsIn[bb] + stepSize
        if bb < len(histIn):
            data[2 * bb + 1] = histIn[bb]
            data[2 * bb + 2] = histIn[bb]

    bins[0] = bins[1]
    bins[-1] = bins[-2]
    data[0] = 0
    data[-1] = 0

    return (bins, data)

'''
Stuff from Adrienne:
I started this email because I wanted to know if you had the answer, and I found it along the way. Now I think it's useful knowledge. Basically, it's an easy way to put a legend anywhere on a figure in figure coordinates, not axis coordinates or data coordinates. Helpful if you have one legend for multiple subplots.

Basically, you can do a transform in legend to tell matplotlib that you're specifying the coordinates in data units or in axis units, ie,

bbox_transform = ax.transAxes (0 - 1 means left to right or bottom to top of current axis)

bbox_transform = ax.transData (specify location in data coordinates for that particular axis)

bbox_transform = fig.transFigure (specify location in figure coordinates, so 0 - 1 means bottom to top or left to right of the current *figure*, not the current axis)

Now I am trying to figure out how to get it to use points from different subplots in one legend.. fun.

proxy artists:
    p_corr = matplotlib.lines.Line2D([0], [0], marker = 'o', color = 'k',
                                     linestyle = 'None')
    p_uncorr = matplotlib.lines.Line2D([0], [0], marker = 'o', color = '0.7',
                                       linestyle = 'None',
                                       markeredgecolor = '0.7')

    l = plt.legend([p_corr, p_uncorr], ["Correlated", "Uncorrelated"],
                   bbox_transform = fig.transFigure, loc = 'upper right',
                   bbox_to_anchor = (0.9, 0.9),
                   numpoints = 1,
                   title = "Log scaling",
                   borderpad = 1,
                   handletextpad = 0.3,
                   labelspacing = 0.5)#,
                   #prop = matplotlib.font_manager.FontProperties(size=20))

I've been looking for this for a while. I don't know how you do colors in plotting but I think it's different than me. This is good for scatterplots with color.

Easy to way go from array of float values of any range -> floats between 0 and 1 with some scaling so you can pass it to a colormap -> colors:


import matplotlib

norm_instance = matplotlib.colors.Normalize(vmin = np.min(float_array),
    vmax = np.max(float_array) )

normed_floats = norm_instance( float_array )

colors = matplotlib.cm.jet( normed_floats ) # or any cmap really.


and voila, colors is your usual array of [R G B alpha] values for the color or each point. I've been trying to find something like this for a long time and finally stumbled across the right google search terms.

There's also a matplotlib.colors.LogNorm if you want to normalize the colors on a log scale.

'''


'''
import matplotlib.ticker as poop
ax.yaxis.set_major_formatter( poop.FormatStrFormatter('%g'))
'''

'''
POWERLAWS

http://arxiv.org/abs/0905.0474
   fig = plt.figure(figsize = (7,3))
    gs = gridspec.GridSpec(1, 4, left = 0.15, right = 0.9,
                           bottom = 0.1, top = 0.9,
                           width_ratios = [1, 1, 1, 0.05])
    #gs.update( ) # like subplots_adjust

    # data
    ax_s1 = host_subplot(gs[0])
    ax_s2 = host_subplot(gs[1])
    ax_m2 = host_subplot(gs[2])
    cax_hidden = host_subplot(gs[3])
    cax_hidden.axis('off') # we only need its x position for later, so hide cax.

    ### plot some shit
    im_s1 = ax_s1.imshow(A)

    fig.canvas.draw()
    extents_s1 = ax_s1.figbox.extents
    extents_cax = cax_hidden.figbox.extents

    # make a new subplot axis with specified location for the colorbar.
    poop_position = [ extents_cax[0], # left
                     extents_s1[1], # bottom
                     extents_cax[2] - extents_cax[0], # width
                     extents_s1[3] - extents_s1[1] ] # height

    poop_ax = plt.axes(poop_position)

    # make the colorbar
    cb = plt.colorbar(im_s1, cax = poop_ax)#, ticks = cb_ticks)
far easier if you have only one subplot instead of 3 like my previous example


from mpl_toolkits.axes_grid1 import make_axes_locatable

def poop():
    fig = plt.figure(figsize = (6,4))
    fig.subplots_adjust(left = 0.2, right = 0.8,
                        bottom = 0.2, top = 0.8)
    ax = host_subplot(111)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad = 1)


    # ... plot something
    mappable = plt.imshow(array)

    plt.colorbar(mappable, cax = cax)
'''
