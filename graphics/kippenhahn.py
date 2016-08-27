import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from .graphics import annotate_plot

from ..config import logT, ycen, xc_cen, xo_cen, age
from ..eep.critical_point import Eep
from ..utils import add_ptcris


def kippenhahn(track, col_keys=None, heb_only=True, ptcri=None,
               four_tops=False, xscale='linear', between_ptcris=[0, -2],
               khd_dict=None, ax=None, norm=None, annotate=False,
               legend=False, fusion=True, convection=True):
    pinds = add_ptcris(track, between_ptcris)
    norm = norm or ''
    if heb_only:
        # Core HeB:
        inds, = np.nonzero((track.data['LY'] > 0) & (track.data.QHE1 == 0))
    else:
        inds = np.arange(pinds[pinds > 0][0], pinds[pinds > 0][-1])

    pinds = add_ptcris(track, between_ptcris)
    xdata = track.data[age][inds]

    if xscale == 'linear':
        # AGE IN Myr
        xdata /= 1e6
        xlab = r'$\rm{Age (Myr)}$'
    elif 'x' in norm:
        xdata = xdata/np.max(xdata)
        xlab = r'$\rm{fractional Age}$'
    else:
        xlab = r'$\log \rm{Age (yr)}$'

    if four_tops:
        track.calc_core_mu()
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True,
                                            figsize=(8, 8))
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
    # convections
    fbkw = {'edgecolor': 'none', 'alpha': 0.4, 'zorder': 1}
    conv_kw = fbkw.copy()
    conv_kw['color'] = 'grey'
    if convection:
        ax.fill_between(xdata[:p1],
                        track.data.CI1[inds[:p1]],
                        track.data.CF1[inds[:p1]],
                        where=track.data.CF1[inds[:p1]] > 0.2,
                        **conv_kw)
        ax.fill_between(xdata[p2:],
                        track.data.CI1[inds[p2:]],
                        track.data.CF1[inds[p2:]],
                        where=track.data.CF1[inds[p2:]] < 0.2,
                        **conv_kw)
        ax.fill_between(xdata,
                        track.data.CI2[inds],
                        track.data.CF2[inds],
                        **conv_kw)
    if fusion:
        ax.fill_between(xdata,
                        track.data.QH1[inds],
                        track.data.QH2[inds],
                        color='navy', label=r'$H$', **fbkw)
        ax.fill_between(xdata,
                        track.data.QHE1[inds],
                        track.data.QHE2[inds],
                        color='darkred', label=r'$^4He$', **fbkw)

    zorder = 100
    if khd_dict is None:
        khd_dict = {xc_cen: 'darkgreen',
                    xo_cen: 'purple',
                    ycen: 'orange',
                    'LX': 'navy',
                    'LY': 'darkred',
                    'CONV':  'black'}

    # white underneath
    [ax.plot(xdata, track.data[column][inds], lw=5, color='white')
     for column in khd_dict.keys()]

    zorder += 10
    for col, color in khd_dict.items():
        ax.plot(xdata, track.data[col][inds], ls=plot_linestyles(col),
                lw=3, color=color, label=plot_labels(col), zorder=zorder)
        zorder += 10

    ixmax = p1 + np.argmax(track.data[logT][inds[p1:]])

    if legend:
        ax.legend(frameon=False, loc=0)

    ax.set_ylim(0, 1)
    ax.set_xlabel(xlab, fontsize=18)
    ax.set_ylabel('$m/M\ or\ f/f_{tot}$', fontsize=18)
    if annotate:
        ptcri_names = \
            Eep().eep_list[between_ptcris[0]: between_ptcris[1] + 1]
        self.annotate_plot(track, ax, '', '', xdata=xdata, ydata=xdata,
                           ptcri_names=ptcri_names, khd=True, inds=inds,
                           lw=2)
    # [a.set_xlim(xdata[pinds[0]], xdata[pinds[-1]]) for a in axs]

    # for a in axs:
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
