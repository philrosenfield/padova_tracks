import os
import seaborn
import matplotlib.pyplot as plt
import numpy as np

from .utils import discrete_colors

from ..config import logL, logT, mass, age
from ..eep.critical_point import Eep
from ..utils import column_to_data

seaborn.set()


def hrd(track, ax=None, inds=None, reverse=None, plt_kw=None):
    '''
    make an hrd.
    written for interactive use (usually in pdb)
    '''
    plt_kw = plt_kw or {}
    reverse = reverse or ''
    if ax is None:
        _, ax = plt.subplots()

    l, = ax.plot(track.data[logT], track.data[logL], **plt_kw)

    if inds is not None:
        if len(inds) < 30:
            ax.plot(track.data[logT][inds], track.data[logL][inds], 'o',
                    color=l.get_color(), alpha=0.4)
        else:
            ax.plot(track.data[logT][inds], track.data[logL][inds],
                    alpha=0.4, lw=3)

    if 'x' in reverse:
        ax.invert_xaxis()

    if 'y' in reverse:
        ax.invert_yaxis()

    ax.set_ylabel(r'$\rm{L} (\rm{L}_\odot)$')
    ax.set_xlabel(r'$\log\ T_{eff}$')
    return ax


def plot_tracks(tracks, xcols=[logT, age], extra=None, hb=False, mextras=None,
                split=True, plot_dir=None, match_tracks=None):
    '''
    pat_kw go to plot all tracks default:
        'eep_list': Eep.eep_list,
        'eep_lengths': Eep.nticks,
        'plot_dir': tracks.tracks_base
    xcols are the xcolumns to make individual plots
    mass_split is a list to split masses length == 3 (I'm lazy)
    extras is the filename extra associated with each mass split
       length == mass_split + 1
    '''
    extra = extra or ''
    line_pltkw, mline_pltkw, point_pltkw, mpoint_pltkw = plot_kws()
    tracks = np.array([t for t in tracks if t.flag is None])

    if match_tracks is not None:
        match_tracks = np.asarray(match_tracks)
        assert len(tracks) == len(match_tracks)

    if hasattr(tracks, 'prefix'):
        prefix = tracks.prefix
    else:
        prefix = os.path.split(tracks[0].base)[1]

    if len(extra) > 0 and '_' not in extra:
        extra = extra + '_'

    if split:
        mass_split = [0, 1, 1.4, 3, 12, 50, 1000]
        if hb:
            extra += 'hb_'
            mass_split = [0, 0.7, 0.9, 1.4, 2, 6, 1000]

        mextras = ['_lowest', '_vlow', '_low', '_inte', '_high', '_vhigh']
        masses = [t.mass for t in tracks]
        binds = np.digitize(masses, bins=mass_split)
        _, udx = np.unique(binds, return_index=True)
        tracks_pplot = np.array([np.arange(udx[i], udx[i+1])
                                 for i in range(len(udx)-1)])
    else:
        tracks_pplot = np.array([tracks])

    for i, its in enumerate(tracks_pplot):
        for j, _ in enumerate(xcols):
            fig, ax = plt.subplots(figsize=(12, 8))
            for k in its:
                ax = match_parsec(tracks[k], match_track=match_tracks[k],
                                  ax=ax, xcol=xcols[j])

            ax.set_title(r'$%s$' % prefix.replace('_', r'\ '))
            ax.invert_xaxis()
            figname = '%s%s%s_%s.png' % (extra, xcols[j], mextras[i], prefix)
            if plot_dir is not None:
                figname = os.path.join(plot_dir, figname)
            plt.savefig(figname)
            plt.close('all')
    return


def plot_kws():
    default = {'alpha': 0.3, 'marker': 'o', 'ls': ''}
    line_pltkw = {'color': 'black'}
    mline_pltkw = {'color': 'green', 'alpha': 0.3, 'lw': 4}
    point_pltkw = {'color': 'navy', **default}
    mpoint_pltkw = {'color': 'darkred', **default}
    return line_pltkw, mline_pltkw, point_pltkw, mpoint_pltkw


def match_parsec(track, plot_dir=None, xcol=logT, ycol=logL, match_track=None,
                 ax=None, title=False, save=False):
    '''plot the track, the interpolation, with eeps'''
    if track.flag is not None:
        return

    line_pltkw, mline_pltkw, point_pltkw, mpoint_pltkw = plot_kws()

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    # parsec track
    ax = plot_track(track, xcol, ycol, ax=ax, plt_kw=line_pltkw,
                    plt_point_kw=point_pltkw)

    if match_track is not None:
        # overplot the match interpolation
        ax = plot_track(track, xcol, ycol, ax=ax, plt_kw=mline_pltkw,
                        plt_point_kw=mpoint_pltkw)

    xlab = '${}$'.format(xcol.replace('_', r'\ '))
    ylab = '${}$'.format(ycol.replace('_', r'\ '))

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if title:
        title = 'M = {0:.3f} Z = {1:.4f} Y = {2:.4f}'.format(track.mass,
                                                             track.Z, track.Y)
        fig.suptitle(title)
    if save:
        extra = ''
        ax.invert_xaxis()
        if track.hb:
            extra += 'HB_'
        extra += '{:s}'.format(xcol)

        figname = '{:s}_Z{:g}_Y{:g}_M{:.3f}.png'.format(extra, track.Z,
                                                        track.Y, track.mass)
        if plot_dir is not None:
            figname = os.path.join(plot_dir, figname)

        plt.savefig(figname)
        plt.close()
    return ax


def plot_track(track, xcol, ycol, reverse=None, ax=None, inds=None,
               plt_kw=None, clean=False, yscale='linear',
               xscale='linear', add_mass=True, plt_point_kw=None):
    '''
    ainds is passed to annotate plot, and is to only plot a subset of crit
    points.

    plot helpers:
    reverse 'xy', 'x', or 'y' will flip that axis

    '''
    plt_kw = plt_kw or {}
    plt_point_kw = plt_point_kw or {}
    reverse = reverse or ''

    if isinstance(track, str):
        from ..tracks.track import Track
        track = Track(track)

    if clean and inds is None:
        # non-physical inds go away.
        inds, = np.nonzero(track.data[age] > 0.2)

    xdata = track.data[xcol]
    ydata = track.data[ycol]

    if ax is None:
        fig, ax = plt.subplots()

    if inds is not None:
        ax.plot(xdata[inds], ydata[inds], **plt_kw)
    else:
        ax.plot(xdata, ydata, **plt_kw)

        if hasattr(track, 'iptcri'):
            pinds = track.iptcri[track.iptcri > 0]
            ax.plot(xdata[pinds], ydata[pinds], **plt_point_kw)

            if add_mass:
                try:
                    ind = pinds[3]
                except:
                    ind = pinds[1]
                ax.annotate(r'${0:g}$'.format(track.mass),
                            (xdata[ind], ydata[ind]), fontsize=10)
        elif track.match:
            if xcol == age:
                xdata = 10 ** t.data[xcol]
            eep = Eep()
            nticks = eep.nticks
            if track.hb:
                nticks = eep.nticks_hb
            minds = np.insert(np.cumsum(nticks), 0, 1) - 1
            ax.plot(xdata[minds], ydata[minds], **plt_point_kw)

    if 'x' in reverse:
        ax.invert_xaxis()

    if 'y' in reverse:
        ax.invert_yaxis()

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    return ax


def annotate_plot(track, ax, xcol, ycol, ptcri_names=None, box=True, khd=False,
                  xdata=None, ydata=None, inds=None, **kwargs):
    '''annotate plot with EEPs'''
    ptcri_names = ptcri_names or []
    eep = Eep()
    eep_list = eep.eep_list

    if track.hb:
        eep_list = eep.eep_list_hb

    fc = 'navy'
    iptcri = track.iptcri

    if len(ptcri_names) == 0:
        # assume all
        ptcri_names = eep_list
    pts = [list(eep_list).index(i) for i in ptcri_names]
    if len(pts) > len(iptcri):
        pts = pts[:len(iptcri)]
    # do not plot ptcri indices == 0, these are not defined!
    pinds = iptcri[pts][iptcri[pts] > 0]
    if inds is not None:
        pinds = np.intersect1d(inds, pinds)
    labs = ['$%s$' % p.replace('_', r'\ ') for p in ptcri_names]

    xdata, ydata = column_to_data(track, xcol, ycol, xdata=xdata,
                                  ydata=ydata)
    if box:
        # label stylings
        bbox = dict(boxstyle='round, pad=0.2', fc=fc, alpha=0.2)
        arrowprops = dict(arrowstyle='->', connectionstyle='arc3, rad=0',
                          color=fc)

    for i, (lab, x, y) in enumerate(zip(labs, xdata[pinds], ydata[pinds])):
        # varies the labels placement... default is 20, 20
        if khd:
            ax.vlines(x, 0, 1, label=lab, **kwargs)
            y = 0.75
        if box:
            xytext = ((-1.) ** (i - 1.) * 20, (-1.) ** (i + 1.) * 20)
            ax.annotate(lab, xy=(x, y), xytext=xytext, fontsize=10,
                        textcoords='offset points', ha='right',
                        va='bottom', bbox=bbox, arrowprops=arrowprops)
    return ax
