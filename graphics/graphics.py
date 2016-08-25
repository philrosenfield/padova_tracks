import matplotlib.pyplot as plt
import numpy as np

from .utils import discrete_colors

from ..config import logL, logT, mass, age
from ..eep.critical_point import Eep
from ..utils import column_to_data


def hrd(track, ax=None, inds=None, reverse=None, plt_kw={}):
    '''
    make an hrd.
    written for interactive use (usually in pdb)
    '''
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


def plot_sandro_ptcri(track, plot_dir=None, ptcri=None):
    x = logT
    y = logL
    ax = plot_track(track, x, y, reverse='x',
                    inds=np.nonzero(track.data[age] > 0.2)[0])

    ax = annotate_plot(track, ax, x, y, sandro=True, ptcri=ptcri)
    title = 'M = %.3f Z = %.4f Y = %.4f' % (track.mass, track.Z, track.Y)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel((r'$%s$' % x).replace('_', r'\_'), fontsize=20)
    ax.set_ylabel((r'$%s$' % y).replace('_', r'\_'), fontsize=20)
    figname = 'sandro_ptcri_Z%g_Y%g_M%.3f.png' % (track.Z, track.Y,
                                                  track.mass)
    if plot_dir is not None:
        figname = os.path.join(plot_dir, figname)
        plt.savefig(figname)
        plt.close()
    return ax


def plot_all_tracks(tracks, xcol=logT, ycol=logL, ax=None,
                    ptcri=None, line_pltkw={}, point_pltkw={},
                    clean=True):
    '''plot all tracks and annotate eeps'''
    default = {'alpha': 0.5}
    default.update(line_pltkw)
    line_pltkw = default.copy()

    default = {'marker': 'o', 'ls': '', 'alpha': 0.3}
    default.update(point_pltkw)
    point_pltkw = default.copy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 9))

    xlims = np.array([])
    ylims = np.array([])
    for t in tracks:
        if t.flag is not None:
            continue

        if ptcri is not None:
            try:
                inds = ptcri.data_dict['M%.3f' % t.mass]
            except KeyError:
                # mass not found, means track was skipped
                continue
            cols = discrete_colors(len(inds) + 1, colormap='spectral')
            # skip undefined eeps or you get annotations at the start of
            # each track
            inds = inds[inds > 0]
        else:
            inds = np.arange(len(t.data[xcol]))

        xdata = t.data[xcol]
        ydata = t.data[ycol]
        if t.match:
            if 'age' in xcol.lower():
                xdata = 10 ** t.data[xcol]
            eep = Eep()
            nticks = eep.nticks
            if t.hb:
                nticks = eep.nticks_hb
            inds = np.insert(np.cumsum(nticks), 0, 1) - 1
            cols = discrete_colors(len(inds) + 1, colormap='spectral')
            # only eeps that are in the track
            inds = [eep for eep in inds if eep < len(xdata)]

        if clean:
            finds = np.arange(inds[0], inds[-1])
            ax.plot(xdata[finds], ydata[finds], **line_pltkw)
        else:
            ax.plot(xdata, ydata, **line_pltkw)
        if len(inds) < len(xdata):
            for i, _ in enumerate(inds):
                x = xdata[inds[i]]
                y = ydata[inds[i]]
                try:
                    ax.plot(x, y, color=cols[i], **point_pltkw)
                except:
                    ax.plot(x, y, **point_pltkw)

                if i == 3:
                    ax.annotate('%g' % t.mass, (x, y), fontsize=8)

        xlims = np.append(xlims, (np.min(xdata), np.max(xdata)))
        ylims = np.append(ylims, (np.min(ydata), np.max(ydata)))

    ax.set_xlim(np.max(xlims), np.min(xlims))
    ax.set_ylim(np.min(ylims), np.max(ylims))
    ax.set_xlabel(r'$%s$' % xcol.replace('_', r'\! '), fontsize=20)
    ax.set_ylabel(r'$%s$' % ycol.replace('_', r'\! '), fontsize=20)
    # ax.legend(loc=0, numpoints=1, frameon=0)
    return ax


def plot_track(track, xcol, ycol, reverse='', ax=None, inds=None, plt_kw=None,
               clean=False, sandro=False, cmd=False, convert_mag_kw={},
               norm='', arrows=False, yscale='linear', xscale='linear',
               ptcri_inds=False, ptcris=False, between_ptcris=[0, -1],
               add_mass=False):
    '''
    ainds is passed to annotate plot, and is to only plot a subset of crit
    points.
    sandro = True will plot sandro's ptcris.

    plot helpers:
    reverse 'xy', 'x', or 'y' will flip that axis
    ptcri_inds bool will annotate ptcri numbers
    add_ptcris will mark plot using track.iptcri or track.sptcri

    '''
    plt_kw = plt_kw or {}
    if type(track) == str:
        from .track import Track
        track = Track(track)

    if ax is None:
        fig, ax = plt.subplots()

    if (len(plt_kw) != 0) and ('marker' in plt_kw.keys()) and \
       ('ls' not in plt_kw.keys() or 'linestyle' not in plt_kw.keys()):
        plt_kw['ls'] = ''

    if clean and inds is None:
        # non-physical inds go away.
        inds, = np.nonzero(track.data[age] > 0.2)

    xdata, ydata = column_to_data(track, xcol, ycol, norm=norm)

    if inds is not None:
        ax.plot(xdata[inds], ydata[inds], **plt_kw)
    else:
        if not hasattr(track, 'sptcri') or not hasattr(track, 'iptcri'):
            ax.plot(xdata, ydata, **plt_kw)
        else:
            if sandro:
                iptcri = track.sptcri
            else:
                iptcri = track.iptcri
            pinds = np.arange(iptcri[between_ptcris[0]],
                              iptcri[between_ptcris[1]])
            ax.plot(xdata[pinds], ydata[pinds], **plt_kw)

    if 'x' in reverse:
        ax.invert_xaxis()

    if 'y' in reverse:
        ax.invert_yaxis()

    if ptcris:
        # very simple ... use annotate for the fancy boxes
        pinds = add_ptcris(track, between_ptcris, sandro=sandro)
        ax.plot(xdata[pinds], ydata[pinds], 'o', color='k')
        if ptcri_inds:
            [ax.annotate('%i' % i, (xdata[i], ydata[i])) for i in pinds]

    if add_mass:
        ax.annotate(r'$%g$' % track.mass, (xdata[iptcri[3]], ydata[iptcri[3]]),
                    fontsize=10)

    if arrows:
        # hard coded to be 10 equally spaced points...
        inds, = np.nonzero(track.data[age] > 0.2)
        ages = np.linspace(np.min(track.data[age][inds]),
                           np.max(track.data[age][inds]), 10)
        indz, _ = zip(*[closest_match(i, track.data[age][inds])
                        for i in ages])
        # I LOVE IT arrow on line... AOL BUHSHAHAHAHAHA
        aol_kw = plt_kw.copy()
        if 'color' in aol_kw:
            aol_kw['fc'] = aol_kw['color']
            del aol_kw['color']
        indz = indz[indz > 0]
        print(track.data[logL][inds][np.array([indz])])
        arrow_on_line(ax, xdata, ydata, indz, plt_kw=plt_kw)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    return ax


def annotate_plot(track, ax, xcol, ycol, ptcri_names=[],
                  sandro=False, box=True, khd=False,
                  xdata=None, ydata=None, inds=None, **kwargs):
    '''
    if a subset of ptcri inds are used, set them in inds. If you want
    sandro's ptcri's sandro=True, will also change the face color of the
    label bounding box so you can have both on the same plot.
    '''
    eep = Eep()
    eep_list = eep.eep_list

    if track.hb:
        eep_list = eep.eep_list_hb

    if not sandro:
        fc = 'navy'
        iptcri = track.iptcri
    else:
        fc = 'darkred'
        iptcri = track.sptcri
        eep_list = ptcri.sandro_eeps

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
