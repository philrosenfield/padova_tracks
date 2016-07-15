'''
plotting and diagnostics track is always track object.
'''
from __future__ import print_function
import matplotlib.pylab as plt
from matplotlib.ticker import MaxNLocator, NullFormatter
import numpy as np
from copy import deepcopy
import os
from ..graphics.GraphicsUtils import arrow_on_line, setup_multiplot, discrete_colors
from ..eep.critical_point import Eep
from ..utils import closest_match
from ..config import *

def offset_axlims(track, xcol, ycol, ax, inds=None):
    xmax, xmin = track.maxmin(xcol, inds=inds)
    ymax, ymin = track.maxmin(ycol, inds=inds)

    if np.diff((xmin, xmax)) == 0:
        xmin -= 0.1
        xmax += 0.1

    if np.diff((ymin, ymax)) == 0:
        ymin -= 0.5
        ymax += 0.5

    offx = 0.05
    offy = 0.1
    ax.set_xlim(xmax + offx, xmin - offx)
    ax.set_ylim(ymin - offy, ymax + offy)
    return ax

def quick_hrd(track, ax=None, inds=None, reverse='x', plt_kw={}):
    '''
    make an hrd.
    written for interactive use (usually in pdb)
    '''
    plt.ion()
    if ax is None:
        plt.figure()
        ax = plt.axes()
        reverse = 'x'

    ax.plot(track.data[logT], track.data[logL], **plt_kw)

    if inds is not None:
        ax.plot(track.data[logT][inds], track.data[logL][inds], 'o')

    if 'x' in reverse:
        ax.set_xlim(ax.get_xlim()[::-1])

    ax.set_ylabel(r'$\rm{L} (\rm{L}_\odot)$')
    ax.set_xlabel(r'$\log\ T_{eff}$')
    return ax

def check_eep_hrd(tracks, ptcri_loc, between_ptcris=[0, -2], sandro=True):
    '''
    a simple debugging tool.
    Load in tracks (string or Track obj)
    give the location of the ptcri file
    and choose which set of ptcris to plot.
    returns the track set and the axs (one for each Z)

    '''
    from .track_set import TrackSet
    if type(tracks[0]) is str:
        from .track import Track
        tracks = [Track(t) for t in tracks]
    td = TrackDiag()
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
        td.annotate_plot(t, ax, logT, logL, ptcri_names=ptcri_names)

    [ax.set_xlim(ax.get_xlim()[::-1]) for ax in axs]
    return ts, axs

def column_to_data(track, xcol, ycol, xdata=None, ydata=None, cmd=False,
                   convert_mag_kw={}, norm=''):
    '''
    convert a string column name to data

    returns xdata, ydata

    norm: 'xy', 'x', 'y' for which or both axis to normalize
    can also pass xdata, ydata to normalize or if its a cmd (Mag2mag only)
    '''
    if ydata is None:
        ydata = track.data[ycol]

    if xdata is None:
        if cmd:
            if len(convert_mag_kw) != 0:
                from ResolvedStellarPops import astronomy_utils
                photsys = convert_mag_kw['photsys']
                dmod = convert_mag_kw.get('dmod', 0.)
                Av = convert_mag_kw.get('Av', 0.)
                Mag1 = track.data[xcol]
                Mag2 = track.data[ycol]
                avdmod = {'Av': Av, 'dmod': dmod}
                mag1 = astronomy_utils.Mag2mag(Mag1, xcol, photsys, **avdmod)
                mag2 = astronomy_utils.Mag2mag(Mag2, ycol, photsys, **avdmod)
                xdata = mag1 - mag2
                ydata = mag2
            else:
                xdata = track.data[xcol] - track.data[ycol]
        else:
            xdata = track.data[xcol]

    if 'x' in norm:
        xdata /= np.max(xdata)

    if 'y' in norm:
        ydata /= np.max(ydata)

    return xdata, ydata

def add_ptcris(track, between_ptcris, sandro=False):
    '''return track.[s or i ]ptcri indices between between_ptcris'''
    if sandro:
        iptcri = track.sptcri
    else:
        iptcri = track.iptcri
    pinds = iptcri[between_ptcris[0]: between_ptcris[1] + 1]
    return pinds

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

    if len(plt_kw) != 0:
        # not sure why, but every time I send marker='o' it also sets
        # linestyle = '-' ...
        if 'marker' in plt_kw.keys():
            if not 'ls' in plt_kw.keys() or not 'linestyle' in plt_kw.keys():
                plt_kw['ls'] = ''

    if clean and inds is None:
        # non-physical inds go away.
        inds, = np.nonzero(track.data[age] > 0.2)

    xdata, ydata = column_to_data(track, xcol, ycol, cmd=cmd, norm=norm,
                                  convert_mag_kw=convert_mag_kw)

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
        ax.set_xlim(ax.get_xlim()[::-1])

    if 'y' in reverse:
        ax.set_ylim(ax.get_ylim()[::-1])

    if ptcris:
        # very simple ... use annotate for the fancy boxes
        pinds = add_ptcris(track, between_ptcris, sandro=sandro)
        ax.plot(xdata[pinds], ydata[pinds], 'o', color='k')
        if ptcri_inds:
            [ax.annotate('%i' % i, (xdata[i], ydata[i])) for i in pinds]

    if add_mass:
        ax.annotate(r'$%g$' % track.mass, (xdata[iptcri[5]], ydata[iptcri[5]]),
                    fontsize=10)

    if arrows:
        # hard coded to be 10 equally spaced points...
        inds, = np.nonzero(track.data[age] > 0.2)
        ages = np.linspace(np.min(track.data[age][inds]),
                           np.max(track.data[age][inds]), 10)
        indz, _ = zip(*[closest_match(i, track.data[age][inds])
                        for i in ages])
        # I LOVE IT arrow on line... AOL BUHSHAHAHAHAHA
        aol_kw = deepcopy(plt_kw)
        if 'color' in aol_kw:
            aol_kw['fc'] = aol_kw['color']
            del aol_kw['color']
        indz = indz[indz > 0]
        print(track.data[logL][inds][np.array([indz])])
        arrow_on_line(ax, xdata, ydata, indz, plt_kw=plt_kw)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    return ax


class TrackDiag(object):
    '''a class for plotting tracks'''
    def __init__(self):
        pass

    def quick_hrd(self, *args):
        return quick_hrd(*args)

    def plot_track(self, *args, **kwargs):
        return plot_track(*args, **kwargs)

    def diag_plots(self, tracks, pat_kw=None, xcols=[logT, age],
                   extra='', hb=False, mass_split='default', mextras=None,
                   plot_dir='.', match_tracks=False, sandro=False):
        '''
        pat_kw go to plot all tracks default:
            'eep_list': self.eep_list,
            'eep_lengths': self.nticks,
            'plot_dir': self.tracks_base
        xcols are the xcolumns to make individual plots
        mass_split is a list to split masses length == 3 (I'm lazy)
        extras is the filename extra associated with each mass split
           length == mass_split + 1
        '''
        flags = [t for t in tracks if t.flag is not None]
        for t in flags:
            print('diag_plots skipping M=%.3f: %s' % (t.mass, t.flag))
        tracks = [t for t in tracks if t.flag is None]
        if hasattr(self, 'prefix'):
            prefix = self.prefix
        else:
            prefix = os.path.split(tracks[0].base)[1]

        if len(extra) > 0 and not '_' in extra:
            extra = extra + '_'

        default = {'hb': hb}
        pat_kw = pat_kw or {}
        default.update(pat_kw)
        pat_kw = default.copy()

        if 'ptcri' in pat_kw.keys():
            if type(pat_kw['ptcri']) is str:
                pat_kw['ptcri'] = critical_point(pat_kw['ptcri'], hb=hb,
                                                 sandro=sandro)
        if hb:
            extra += 'hb_'

        if mass_split == 'default':
            mass_split = [1, 1.4, 3, 12, 50]
            if hb:
                mass_split = [0.7, 0.9, 1.4, 2, 6]
            mextras = ['_lowest', '_vlow', '_low', '_inte', '_high', '_vhigh']
            tracks_split = \
                [[i for i, t in enumerate(tracks) if t.mass <= mass_split[0]],
                 [i for i, t in enumerate(tracks) if t.mass >= mass_split[0] \
                  and t.mass <= mass_split[1]],
                 [i for i, t in enumerate(tracks) if t.mass >= mass_split[1] \
                  and t.mass <= mass_split[2]],
                 [i for i, t in enumerate(tracks) if t.mass >= mass_split[2] \
                  and t.mass <= mass_split[3]],
                 [i for i, t in enumerate(tracks) if t.mass >= mass_split[3] \
                  and t.mass <= mass_split[4]],
                 [i for i, t in enumerate(tracks) if t.mass >= mass_split[4]]]
        else:
            tracks_split = [tracks]
            mextras = ['']
        if match_tracks:
            mxcols = np.copy(xcols)
            try:
                mxcols[xcols.index('AGE')] = 'logAge'
            except ValueError:
                pass
            mpat_kw = {'line_pltkw': {'color': 'black', 'lw': 2,
                                      'alpha': 0.3},
                       'point_pltkw': {'marker': '*'},
                       'hb': hb}

        for i, its in enumerate(tracks_split):
            if len(its) == 0:
                continue

            for j in range(len(xcols)):
                pat_kw['xcol'] = xcols[j]
                ax = self.plot_all_tracks(np.asarray(tracks)[its], **pat_kw)
                if match_tracks:
                    mpat_kw.update({'xcol': mxcols[j], 'ax': ax})
                    ax = self.plot_all_tracks(np.asarray(self.mtracks)[its],
                                              **mpat_kw)

                ax.set_title(r'$%s$' % prefix.replace('_', r'\ '))

                figname = '%s%s%s_%s.png' % (extra, xcols[j], mextras[i],
                                             prefix)
                figname = os.path.join(plot_dir, figname)
                plt.savefig(figname, dpi=300)
                plt.close('all')

    def plot_all_tracks(self, tracks, xcol=logT, ycol=logL, ax=None,
                        ptcri=None, hb=False, line_pltkw={}, point_pltkw={},
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
                if hb:
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
                for i in range(len(inds)):
                    x = xdata[inds[i]]
                    y = ydata[inds[i]]
                    try:
                        ax.plot(x, y, color=cols[i], **point_pltkw)
                    except:
                        ax.plot(x, y, **point_pltkw)

                    if i == 5:
                        ax.annotate('%g' % t.mass, (x, y), fontsize=8)

            xlims = np.append(xlims, (np.min(xdata), np.max(xdata)))
            ylims = np.append(ylims, (np.min(ydata), np.max(ydata)))

        ax.set_xlim(np.max(xlims), np.min(xlims))
        ax.set_ylim(np.min(ylims), np.max(ylims))
        ax.set_xlabel(r'$%s$' % xcol.replace('_', r'\! '), fontsize=20)
        ax.set_ylabel(r'$%s$' % ycol.replace('_', r'\! '), fontsize=20)
        # ax.legend(loc=0, numpoints=1, frameon=0)
        return ax

    def annotate_plot(self, track, ax, xcol, ycol, ptcri_names=[],
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
            fc = 'blue'
            iptcri = track.iptcri
        else:
            ptcri = kwargs.get('ptcri')
            assert ptcri is not None, 'Must pass Sandro\'s eeps via ptcri obj'
            fc = 'red'
            iptcri = track.sptcri
            eep_list = ptcri.sandro_eeps

        if len(ptcri_names) == 0:
            # assume all
            ptcri_names = eep_list
        pts = [list(eep_list).index(i) for i in ptcri_names]
        if pts > len(iptcri):
            #print('Warning: more ptcri names than values. Assuming they are in order!')
            pts = pts[:len(iptcri)]
        # do not plot ptcri indices == 0, these are not defined!
        pinds = iptcri[pts][iptcri[pts] > 0]
        if inds is not None:
            pinds = map(int, np.concatenate([[i for i, ind in enumerate(inds)
                                              if ind == p] for p in pinds]))
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
                            textcoords='offset points', ha='right', va='bottom',
                            bbox=bbox, arrowprops=arrowprops)
        return ax

    def check_ptcris(self, track, ptcri, hb=False, plot_dir=None,
                     sandro_plot=False, xcol=logT, ycol=logL,
                     match_track=None):
        '''
        plot of the track, the interpolation, with each eep labeled
        '''
        if track.flag is not None:
            return

        all_inds, = np.nonzero(track.data[age] > 0.2)
        iptcri = track.iptcri
        defined, = np.nonzero(iptcri > 0)
        ptcri_kw = {'sandro': False, 'hb': hb}
        last = ptcri.get_ptcri_name(len(defined) - 1, **ptcri_kw)
        if not hb:
            plots = [['PMS_BEG', 'PMS_MIN', 'PMS_END', 'MS_BEG'],
                     ['MS_BEG', 'MS_TMIN', 'MS_TO', 'SG_MAXL', 'RG_MINL'],
                     ['RG_MINL', 'RG_BMP1', 'RG_BMP2', 'RG_TIP'],
                     ['RG_TIP', 'HE_BEG', 'YCEN_0.550', 'YCEN_0.500',
                      'YCEN_0.400'],
                     ['YCEN_0.400', 'YCEN_0.200', 'YCEN_0.100'],
                     ['YCEN_0.100', 'YCEN_0.005', 'YCEN_0.000', 'TPAGB']]
        else:
            plots = [['HE_BEG', 'YCEN_0.500', 'YCEN_0.400', 'YCEN_0.200',
                      'YCEN_0.100', 'YCEN_0.005'],
                     ['YCEN_0.005', 'YCEN_0.000', 'AGB_LY1', 'AGB_LY2'],
                     ['AGB_LY2', 'TPAGB']]

        for i, plot in enumerate(plots):
            if last in plot:
                nplots = i + 2

        line_pltkw = {'color': 'black'}
        mline_pltkw = {'color': 'green', 'alpha': 0.3, 'lw': 4}
        point_pltkw = {'marker': 'o', 'ls': '', 'alpha': 0.3, 'color': 'navy'}
        fig, axs = setup_multiplot(nplots, subplots_kwargs={'figsize': (12, 8)})
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, wspace=0.1)
        for i, ax in enumerate(axs.ravel()[:nplots]):
            self.plot_track(track, xcol, ycol, ax=ax, inds=all_inds,
                            reverse='x', plt_kw=line_pltkw)

            self.plot_track(track, xcol, ycol, ax=ax, inds=iptcri[iptcri > 0],
                            plt_kw=point_pltkw)

            if match_track is not None:
                # overplot the match interpolation
                if xcol == 'AGE':
                    mxcol = 'logAge'
                    xdata = 10 ** match_track.data[mxcol]
                else:
                    xdata = match_track.data[xcol]
                ax.plot(xdata, match_track.data[ycol], **mline_pltkw)

            if i < nplots - 1:
                ainds = [ptcri.get_ptcri_name(cp, **ptcri_kw)
                         for cp in plots[i]]
                inds = iptcri[ainds][iptcri[ainds] > 0]
                if np.sum(inds) == 0:
                    continue
                self.annotate_plot(track, ax, xcol, ycol, ptcri_names=plots[i],
                                   hb=hb)
                ax = offset_axlims(track, xcol, ycol, ax, inds=inds)
            else:
                ax = offset_axlims(track, xcol, ycol, ax)

            if 'age' in xcol:
                ax.set_xscale('log')

        [axs[-1][i].set_xlabel('$%s$' % xcol.replace('_', r'\ '), fontsize=16)
         for i in range(np.shape(axs)[1])]
        [axs[i][0].set_ylabel('$%s$' % ycol.replace('_', r'\ '), fontsize=16)
         for i in range(np.shape(axs)[0])]

        title = 'M = %.3f Z = %.4f Y = %.4f' % (track.mass, track.Z, track.Y)
        fig.suptitle(title, fontsize=20)
        extra = ''
        if hb:
            extra += 'HB_'
        extra += '%s' % xcol

        figname = '%s_Z%g_Y%g_M%.3f.png' % (extra, track.Z, track.Y,
                                            track.mass)
        if plot_dir is not None:
            figname = os.path.join(plot_dir, figname)
        plt.savefig(figname)
        plt.close()

        if not hb and sandro_plot:
            self.plot_sandro_ptcri(track, plot_dir=plot_dir)
        return axs

    def plot_sandro_ptcri(self, track, plot_dir=None, ptcri=None, hb=False):
        x = logT
        y = logL
        ax = self.plot_track(track, x, y, reverse='x',
                             inds=np.nonzero(track.data[age] > 0.2)[0])

        ax = self.annotate_plot(track, ax, x, y, hb=hb,
                                sandro=True, ptcri=ptcri)
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

    def kippenhahn(self, track, col_keys=None, heb_only=True, ptcri=None,
                   four_tops=False, xscale='linear', between_ptcris=[0,-2],
                   khd_dict=None, ax=None, norm='', annotate=False,
                   legend=False, fusion=True, convection=True):
        pinds = add_ptcris(track, between_ptcris, sandro=False)
        if heb_only:
            # Core HeB:
            inds, = np.nonzero((track.data['LY'] > 0) & (track.data.QHE1 == 0))
        else:
            inds = np.arange(pinds[pinds>0][0], pinds[pinds>0][-1])

        pinds = add_ptcris(track, between_ptcris, sandro=False)
        xdata = track.data[age][inds]

        if xscale == 'linear':
            # AGE IN Myr
            xdata /= 1e6
            xlab = '$Age\ (Myr)$'
        elif 'x' in norm:
            xdata = xdata/np.max(xdata)
            xlab = '$fractional\ Age$'
        else:
            xlab = '$\log Age\ (yr)$'

        if four_tops:
            track.calc_core_mu()
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True,
                                                figsize=(8, 8))
            import matplotlib.gridspec as gridspec
            gs = gridspec.GridSpec(8, 2)
            sm_axs = [plt.subplot(gs[i, 0:]) for i in range(4)]
            ax = plt.subplot(gs[4:, 0:])
            ycols = [logT, '', 'LOG_RHc', 'LOG_Pc']
            ycolls = ['$\log T_{eff}$', '$\mu_c$', '$\\rho_c$', '$\log P_c$']

            for smax, ycol, ycoll in zip(sm_axs, ycols, ycolls):
                if len(ycol) == 0:
                    ydata = track.muc[inds]
                else:
                    ydata = track.data[ycol][inds]
                    smax.plot(xdata, ydata, lw=3, color='black', label=ycoll)

                smax.plot(xdata, ydata, lw=3, color='black', label=ycoll)
                smax.set_ylabel('$%s$' % ycoll)
                smax.set_ylim(np.min(ydata), np.max(ydata))
                smax.yaxis.set_major_locator(MaxNLocator(4))
                smax.xaxis.set_major_formatter(NullFormatter())
            axs = np.concatenate([[ax], sm_axs])
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 8))
            axs = [ax]

        ax.set_xscale(xscale)
        # discontinuities in conv...
        p1 = np.argmin((np.diff(track.data.CF1[inds])))
        p2 = np.argmax(np.diff(track.data.CF1[inds]))
        zorder = 1
        # convections
        if convection:
            ax.fill_between(xdata[:p1], track.data.CI1[inds[:p1]],
                             track.data.CF1[inds[:p1]],
                             where=track.data.CF1[inds[:p1]]>0.2, edgecolor='none',
                             color='grey', alpha=0.4, zorder=zorder)
            ax.fill_between(xdata[p2:], track.data.CI1[inds[p2:]],
                             track.data.CF1[inds[p2:]],
                             where=track.data.CF1[inds[p2:]]<0.2, edgecolor='none',
                             color='grey', alpha=0.4, zorder=zorder)
            ax.fill_between(xdata, track.data.CI2[inds], track.data.CF2[inds],
                             color='grey', alpha=0.4, edgecolor='none',)
        if fusion:
            ax.fill_between(xdata, track.data.QH1[inds], track.data.QH2[inds],
                             color='blue', label='$H$', zorder=zorder, alpha=0.4,
                             edgecolor='none',)
            ax.fill_between(xdata, track.data.QHE1[inds], track.data.QHE2[inds],
                             color='red', label='$^4He$', zorder=zorder, alpha=0.4,
                             edgecolor='none',)

        zorder = 100
        if khd_dict is None:
            khd_dict = {xc_cen: 'green',
                        xo_cen: 'purple',
                        ycen: 'darkred',
                        'LX': 'navy',
                        'LY': 'darkred',
                        'CONV':  'black'}

        # white underneath
        [ax.plot(xdata, track.data[column][inds], lw=5, color='white')
         for column in khd_dict.keys()]

        zorder += 10
        for col, color in khd_dict.items():
            ax.plot(xdata, track.data[col][inds], ls=plot_linestyles(col), lw=3,
                    color=color, label=plot_labels(col), zorder=zorder)
            zorder += 10

        ixmax = p1 + np.argmax(track.data[logT][inds[p1:]])

        if legend:
            ax.legend(frameon=False, loc=0)

        ax.set_ylim(0, 1)
        ax.set_xlabel(xlab, fontsize=18)
        ax.set_ylabel('$m/M\ or\ f/f_{tot}$', fontsize=18)
        if annotate:
            ptcri_names = Eep().eep_list[between_ptcris[0]: between_ptcris[1] + 1]
            self.annotate_plot(track, ax, '', '', xdata=xdata, ydata=xdata,
                               ptcri_names=ptcri_names, khd=True, inds=inds,
                               lw=2)
        #[a.set_xlim(xdata[pinds[0]], xdata[pinds[-1]]) for a in axs]

        #for a in axs:
        #    a.set_xlim(xdata[0], xdata[-1])
        #    ylim = a.get_ylim()
        #    [a.vlines(xdata[i], *ylim, color='grey', lw=2)
        #              for i in [p1, itmax]]
        #    a.set_ylim(ylim)

        return axs

def plot_labels(column):
    if column == xc_cen:
        lab = '$^{12}C$'
    elif column == xo_cen:
        lab = '$^{16}O$'
    elif column == 'CONV':
        lab = r'$\rm{core}$'
    elif 'CEN' in column.upper():
        lab = '$%s$' % column.upper().replace('CEN', '_c')
    elif 'L' in column and len(column) == 2:
        lab = '$%s$' % '_'.join(column)
    else:
        print('%s label format not supported' % column)
        lab = column
    return lab

def plot_linestyles(column):
    if column == xc_cen:
        ls = '-'
    elif column == xo_cen:
        ls = '-'
    elif column == 'CONV':
        ls = '-'
    elif 'CEN' in column.upper():
        ls = '-'
    elif 'L' in column and len(column) == 2:
        ls = '--'
    else:
        print('%s line_style format not supported' % column)
        ls = '-'
    return ls
