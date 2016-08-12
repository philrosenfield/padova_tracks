import os

import matplotlib.pyplot as plt
import numpy as np

from ..config import logL, logT, mass, age
from ..graphics.graphics import plot_all_tracks, plot_track, annotate_plot
from ..graphics.utils import setup_multiplot, offset_axlims, discrete_colors
from ..utils import add_ptcris
from ..eep.critical_point import Eep, CriticalPoint


def check_eep_hrd(tracks, ptcri_loc, between_ptcris='default', sandro=True):
    '''
    a simple debugging tool.
    Load in tracks (string or Track obj)
    give the location of the ptcri file
    and choose which set of ptcris to plot.
    returns the track set and the axs (one for each Z)

    '''
    if between_ptcris == 'default':
        between_ptcris = [0, -2]
    from ..track_set import TrackSet
    if type(tracks[0]) is str:
        from ..tracks.track import Track
        tracks = [Track(t) for t in tracks]
    ts = TrackSet()
    ts.tracks = tracks
    if not hasattr(tracks[0], 'sptcri'):
        ts._load_ptcri(ptcri_loc, sandro=True)
    if not hasattr(tracks[0], 'iptcri'):
        ts._load_ptcri(ptcri_loc, sandro=False)

    zs = np.unique([t.Z for t in tracks])
    axs = [plt.subplots()[1] for i in range(len(zs))]
    [axs[list(zs).index(t.Z)].set_title(t.Z) for t in tracks]

    for t in tracks:
        ax = axs[list(zs).index(t.Z)]
        plot_track(t, logT, logL, sandro=sandro, ax=ax,
                   between_ptcris=between_ptcris, add_ptcris=True,
                   add_mass=True)

        ptcri_names = Eep().eep_list[between_ptcris[0]: between_ptcris[1] + 1]
        annotate_plot(t, ax, logT, logL, ptcri_names=ptcri_names)

    [ax.set_xlim(ax.get_xlim()[::-1]) for ax in axs]
    return ts, axs


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

    if 'ptcri' in pat_kw.keys() and type(pat_kw['ptcri']) is str:
        pat_kw['ptcri'] = CriticalPoint(pat_kw['ptcri'])

    if hb:
        extra += 'hb_'

    if mass_split == 'default':
        mass_split = [1, 1.4, 3, 12, 50]
        if hb:
            mass_split = [0.7, 0.9, 1.4, 2, 6]
        mextras = ['_lowest', '_vlow', '_low', '_inte', '_high', '_vhigh']
        tracks_split = \
            [[i for i, t in enumerate(tracks)
                if t.mass <= mass_split[0]],
             [i for i, t in enumerate(tracks)
                if t.mass >= mass_split[0] and t.mass <= mass_split[1]],
             [i for i, t in enumerate(tracks)
                if t.mass >= mass_split[1] and t.mass <= mass_split[2]],
             [i for i, t in enumerate(tracks)
                if t.mass >= mass_split[2] and t.mass <= mass_split[3]],
             [i for i, t in enumerate(tracks)
                if t.mass >= mass_split[3] and t.mass <= mass_split[4]],
             [i for i, t in enumerate(tracks) if t.mass >= mass_split[4]]]
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

    all_inds, = np.nonzero(track.data[age] > 0.2)
    iptcri = track.iptcri
    defined, = np.nonzero(iptcri > 0)
    last = len(defined) - 1

    if not track.hb:
        eeplist = Eep().eep_list
        splits = ['PMS_BEG', 'MS_BEG', 'RG_TIP', 'YCEN_0.400',
                  'YCEN_0.100', 'TPAGB', 'TPAGB2']
    else:
        eeplist = Eep().eep_list_hb
        splits = ['HE_BEG', 'YCEN_0.005', 'TPAGB', 'TPAGB2']

    isplits = [eeplist.index(i) for i in splits]
    plots = [np.arange(isplits[i], isplits[i+1] + 1)
             for i in range(len(isplits)-1)]
    nplots = len(plots)
    if last in plots[-1]:
        nplots += 1

    line_pltkw = {'color': 'black'}
    mline_pltkw = {'color': 'green', 'alpha': 0.3, 'lw': 4}
    point_pltkw = {'color': 'navy', 'alpha': 0.3, 'marker': 'o', 'ls': ''}

    fig, axs = setup_multiplot(nplots, subplots_kws={'figsize': (12, 8)})
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, wspace=0.1)

    for i, ax in enumerate(axs.ravel()[:nplots]):
        # parsec track
        plot_track(track, xcol, ycol, ax=ax, inds=all_inds,
                   reverse='x', plt_kw=line_pltkw)
        # parsec track eeps
        plot_track(track, xcol, ycol, ax=ax, inds=iptcri[defined],
                   plt_kw=point_pltkw)

        if match_track is not None:
            # overplot the match interpolation
            xdata = match_track.data[xcol]
            if xcol == age:
                # Match uses log ages, parsec uses linear ages.
                xdata = 10 ** match_track.data[xcol]
            ax.plot(xdata, match_track.data[ycol], **mline_pltkw)

        if i < nplots - 1:
            inds = iptcri[plots[i]][iptcri[plots[i]] > 0]
            if np.sum(inds) == 0:
                continue
            annotate_plot(track, ax, xcol, ycol, ptcri_names=eeplist)
            # set axes limits around these EEPs
            ax = offset_axlims(track.data[xcol], track.data[ycol], ax,
                               inds=inds)
        else:
            # entire track
            ax = offset_axlims(track.data[xcol], track.data[ycol], ax)

        if xcol == age:
            # this is hack offset_axlims moving a 0 age to a negative...
            if ax.get_xlim()[0] <= 0:
                ax.set_xlim(0.2, ax.get_xlim()[1])
            elif ax.get_xlim()[1] <= 0:
                ax.set_xlim(ax.get_xlim()[0], 0.2)
            ax.set_xscale('log')

    [axs[-1][i].set_xlabel('${}$'.format(xcol.replace('_', r'\ ')))
     for i in range(np.shape(axs)[1])]
    [axs[i][0].set_ylabel('${}$'.format(ycol.replace('_', r'\ ')))
     for i in range(np.shape(axs)[0])]

    title = 'M = %.3f Z = %.4f Y = %.4f' % (track.mass, track.Z, track.Y)
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
    return axs
