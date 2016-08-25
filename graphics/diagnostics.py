import os
import seaborn
import matplotlib.pyplot as plt
import numpy as np

from ..config import logL, logT, mass, age
from ..graphics.graphics import plot_all_tracks, plot_track, annotate_plot
from ..graphics.utils import setup_multiplot, offset_axlims, discrete_colors
from ..utils import add_ptcris
from ..eep.critical_point import Eep, CriticalPoint

seaborn.set()


def diag_plots(tracks, pat_kw=None, xcols=[logT, age],
               extra='', hb=False, mass_split='default', mextras=None,
               plot_dir='.', match_tracks=None, sandro=False):
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
    flags = [t for t in tracks if t.flag is not None]
    for t in flags:
        print('diag_plots skipping M=%.3f: %s' % (t.mass, t.flag))
    tracks = [t for t in tracks if t.flag is None]
    if hasattr(tracks, 'prefix'):
        prefix = tracks.prefix
    else:
        prefix = os.path.split(tracks[0].base)[1]

    if len(extra) > 0 and '_' not in extra:
        extra = extra + '_'

    pat_kw = pat_kw or {}
    pat_kw.update({'clean': False})
    if 'ptcri' in pat_kw.keys() and type(pat_kw['ptcri']) is str:
        pat_kw['ptcri'] = CriticalPoint(pat_kw['ptcri'])

    if hb:
        extra += 'hb_'

    if mass_split == 'default':
        mass_split = [0, 1, 1.4, 3, 12, 50, 1000]
        if hb:
            mass_split = [0, 0.7, 0.9, 1.4, 2, 6, 1000]
        mextras = ['_lowest', '_vlow', '_low', '_inte', '_high', '_vhigh']
        masses = [t.mass for t in tracks]
        binds = np.digitize(masses, bins=mass_split)
        _, udx = np.unique(binds, return_index=True)
        tracks_split = [np.arange(udx[i], udx[i+1]) for i in range(len(udx)-1)]
    else:
        tracks_split = [tracks]
        mextras = ['']
    if match_tracks is not None:
        mpat_kw = {'line_pltkw': {'color': 'black', 'lw': 2, 'alpha': 0.3},
                   'point_pltkw': {'marker': '*'}}

    for i, its in enumerate(tracks_split):
        if len(its) == 0:
            continue

        for j, _ in enumerate(xcols):
            pat_kw['xcol'] = xcols[j]
            ax = plot_all_tracks(np.asarray(tracks)[its], **pat_kw)
            if match_tracks is not None:
                mpat_kw.update({'xcol': xcols[j], 'ax': ax})
                ax = plot_all_tracks(np.asarray(match_tracks)[its],
                                     **mpat_kw)

            ax.set_title(r'$%s$' % prefix.replace('_', r'\ '))

            figname = '%s%s%s_%s.png' % (extra, xcols[j], mextras[i],
                                         prefix)
            figname = os.path.join(plot_dir, figname)
            plt.savefig(figname, dpi=300)
            plt.close('all')


def check_ptcris(track, plot_dir=None, xcol=logT, ycol=logL,
                 match_track=None):
    '''
    plot of the track, the interpolation, with each eep labeled
    '''
    if track.flag is not None:
        return

    iptcri = track.iptcri
    defined, = np.nonzero(iptcri > 0)
    eep = Eep()

    if not track.hb:
        nticks = np.cumsum(eep.nticks)
        eeplist = eep.eep_list
        isplits = [0, 3, 5, len(eeplist) - 2, len(eeplist) - 1]
    else:
        nticks = np.cumsum(eep.nticks_hb)
        eeplist = eep.eep_list_hb
        splits = [0, len(eeplist) - 2, len(eeplist) - 1]

    nticks = np.insert(nticks, 0, 0)
    plots = [np.arange(isplits[i], isplits[i+1] + 1)
             for i in range(len(isplits)-1) if isplits[i+1] in defined]

    nplots = len(plots)
    line_pltkw = {'color': 'black'}
    mline_pltkw = {'color': 'green', 'alpha': 0.3, 'lw': 4}
    point_pltkw = {'color': 'navy', 'alpha': 0.3, 'marker': 'o', 'ls': ''}

    xlab = '${}$'.format(xcol.replace('_', r'\ '))
    ylab = '${}$'.format(ycol.replace('_', r'\ '))
    title = 'M = %.3f Z = %.4f Y = %.4f' % (track.mass, track.Z, track.Y)

    fig, axs = setup_multiplot(nplots, subplots_kws={'figsize': (12, 8)})
    if nplots > 1:
        axs = axs.ravel()[:nplots]
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, wspace=0.1)
    else:
        axs = [axs]

    for i, ax in enumerate(axs):
        inds = iptcri[plots[i]][iptcri[plots[i]] > 0]
        if len(inds) <= 1:
            continue
        line = np.arange(inds[0], inds[-1] + 1)
        # parsec track
        plot_track(track, xcol, ycol, ax=ax, inds=line,
                   reverse='x', plt_kw=line_pltkw)
        # parsec track eeps
        plot_track(track, xcol, ycol, ax=ax, inds=inds,
                   plt_kw=point_pltkw)

        annotate_plot(track, ax, xcol, ycol, inds=plots[i])
        if match_track is not None:
            minds = np.arange(nticks[plots[i][0]], nticks[plots[i][-1]])
            # overplot the match interpolation
            xdata = match_track.data[xcol][minds]
            if xcol == age:
                # Match uses log ages, parsec uses linear ages.
                xdata = 10 ** xdata
            ax.plot(xdata, match_track.data[ycol][minds], **mline_pltkw)
        if xcol == age:
            ax.set_xscale('log')
    try:
        [axs[-1][i].set_xlabel(xlab) for i in range(np.shape(axs)[1])]
        [axs[i][0].set_ylabel(ylab) for i in range(np.shape(axs)[0])]
    except IndexError:
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
    fig.suptitle(title)
    extra = ''
    if track.hb:
        extra += 'HB_'
    extra += '{:s}'.format(xcol)

    figname = '{:s}_Z{:g}_Y{:g}_M{:.3f}.png'.format(extra, track.Z,
                                                    track.Y, track.mass)
    if plot_dir is not None:
        figname = os.path.join(plot_dir, figname)

    plt.savefig(figname)
    plt.close()
    return
